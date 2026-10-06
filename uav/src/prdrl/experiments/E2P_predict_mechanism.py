"""
E2-P -- PREDICT Mechanism Audit.

Tach biet: (A) uoc luong chuyen dong (population-centroid estimator)
sai, hay (B) ban than toan tu translation z'=z+Delta khong hieu qua
ngay ca khi chuyen dong dung.

FREEZE (khong sua): operators/predict.py, local.py, diversify.py,
keep.py, budget.py, E1 detector, E2a/E2a-R. lambda_pred=1.0 xuyen
suot; sigma_predict chi doi giua 2 pass CHU DINH (0 va 0.03), KHONG
sweep.

PREDICT_GT la ORACLE CHAN DOAN, khong bao gio duoc dua vao oracle
production hay action set that (T16).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd

from prdrl.operators.local import local
from prdrl.operators.predict import predict as predict_production

from prdrl.oracle.oracle import nsga2_survival, split_by_region

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    make_region_dynamics_map, region_center, ground_truth_dynamic_label,
    distance_to_region_ps,
)


# ============================================================
# LOCKED CONFIG (Section 2, 22 -- khong tune)
# ============================================================
LAMBDA_PRED = 1.0
SIGMA_LOCAL = 0.02
BUDGET = 10

N_OBJ = 2
D = 10
POP_SIZE = 100
N_PARTITIONS = 9          # M=2 -> R=10
WARMUP_GENERATIONS = 30
GENERATIONS_PER_CHANGE = 10


def compute_time(generation, n_t=10, tau_t=10):
    return (1.0 / n_t) * np.floor(generation / tau_t)


class PredictProblem(Problem):
    def __init__(self, t, region_dynamics, xl, xu):
        super().__init__(n_var=len(xl), n_obj=N_OBJ, xl=xl, xu=xu)
        self.t = t
        self.region_dynamics = region_dynamics

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = cp_evaluate(X, self.t, self.region_dynamics)


def evolve_nsga2(X_start, t, region_dynamics, n_gen, seed, xl, xu):
    # FIX (T14 regression): mot so toan tu ben trong NSGA2/minimize()
    # co the doc np.random (global, khong seed) thay vi Generator rieng
    # -- neu goi ham nay SAU khi mot cho khac trong cung tien trinh da
    # tung np.random.seed()/tieu thu global state (vd nsga2_survival()
    # qua quality_for()), ket qua co the LECH nho (~1e-7) so voi goi
    # doc lap cung seed=X. Reset np.random global state tat dinh ngay
    # truoc minimize(), dung CUNG seed da truyen, de dam bao "cung
    # seed -> cung ket qua" xuyen suot toan bo pipeline, khong chi
    # tung ham rieng le.
    np.random.seed(int(seed) % (2**31))
    pop_size = len(X_start)
    problem = PredictProblem(t, region_dynamics, xl, xu)
    algorithm = NSGA2(pop_size=pop_size, sampling=X_start)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed,
                      verbose=False)
    return np.asarray(result.pop.get("X"), dtype=float)


# ============================================================
# Shared translation primitive -- CUNG cong thuc cho PREDICT_POP va
# PREDICT_GT, chi khac `delta` truyen vao (Section 15 fairness).
# ============================================================

def translate_offspring(X_prev, delta_full, n_offspring, xl, xu, sigma, rng):
    """delta_full: (D,) -- CUNG cong thuc voi predict.py production (z'=
    z+delta+sigma*xi tren CA D chieu). Caller chiu trach nhiem dung
    delta_full dung nghia: PREDICT_POP dung full-D population-mean
    shift (giong het production), PREDICT_GT dung [0]+Delta_GT (ground
    truth KHONG dinh nghia "target" cho x1 -- bien vi tri tu do -- nen
    x1 khong bi dich boi GT, chi bi nhieu xi giong LOCAL)."""
    D_ = X_prev.shape[1]
    offspring = []
    for _ in range(n_offspring):
        idx = rng.integers(len(X_prev))
        parent = X_prev[idx]
        z_parent = (parent - xl) / (xu - xl)
        xi = rng.normal(loc=0.0, scale=1.0, size=D_)
        z_child = np.clip(z_parent + delta_full + sigma * xi, 0.0, 1.0)
        offspring.append(xl + z_child * (xu - xl))
    return np.array(offspring)


def make_rng(base_seed, event_id, region_id, key):
    ss = np.random.SeedSequence([int(base_seed), int(event_id),
                                 int(region_id), hash(key) % (2**31)])
    return np.random.default_rng(ss)


# ============================================================
# One region, one variant, one sigma_predict setting.
# ============================================================

def run_variant(variant, X_region, X_prev2_region, delta_pop_full,
                delta_gt_full, sigma_predict, region_id, base_seed,
                event_id, xl, xu):
    """delta_pop_full/delta_gt_full: (D,) -- xem translate_offspring()."""
    D_ = len(xl)
    if variant == "KEEP":
        return np.empty((0, D_))
    if variant == "LOCAL":
        rng = make_rng(base_seed, event_id, region_id, "LOCAL")
        return local(X_region, BUDGET, xl, xu, SIGMA_LOCAL, rng)
    if variant == "PREDICT_POP":
        rng = make_rng(base_seed, event_id, region_id, "PREDICT_SHARED")
        return translate_offspring(X_region, delta_pop_full, BUDGET, xl,
                                   xu, sigma_predict, rng)
    if variant == "PREDICT_GT":
        rng = make_rng(base_seed, event_id, region_id, "PREDICT_SHARED")
        return translate_offspring(X_region, delta_gt_full, BUDGET, xl,
                                   xu, sigma_predict, rng)
    raise ValueError(variant)


def run_one(seed, n_scored_changes, sigma_predict, n_bins=None):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    n_bins = R if n_bins is None else n_bins
    region_dynamics = make_region_dynamics_map(n_bins)
    pf_star = get_true_pf(n_points=500)
    n_distance = D - 1

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))

    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    prev_regions = None      # per-region pop split at event (k-1)
    t_prev2 = None            # thoi diem cua prev_regions

    rows = []
    gen = 0
    n_total = 1 + n_scored_changes

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        F_plus = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tilde_minus, F_tilde_plus = normalize(F_minus, F_plus)
        F_tilde_minus = np.asarray(F_tilde_minus)
        region_ids = np.asarray(partition(F_tilde_minus, V))
        current_regions = split_by_region(X_pop, region_ids, R)

        if change_idx == 1:
            prev_regions = current_regions
            t_prev2 = t_current
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current   # "t-1" cho event nay (population TRUOC change nay)
        gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)

        for r in range(R):
            mask = region_ids == r
            n_prev = int(np.sum(mask))
            if n_prev == 0:
                continue
            X_region = X_pop[mask]

            X_prev2_region = prev_regions.get(r)
            n_prev2 = int(len(X_prev2_region)) if X_prev2_region is not None else 0

            labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
            region_type = labels_r[np.argmax(counts_r)]

            # region_center() can CHINH XAC bin_index (0..n_bins-1) --
            # suy ra bang cach tim bin ma da so nghiem cua vung nay roi
            # vao (majority vote tren gt_labels da du de xac dinh loai,
            # nhung can chi so bin CU THE cho region_center()). Dung
            # bin xuat hien nhieu nhat trong vung.
            from prdrl.problems.controlled_predict import bin_of
            bins_in_region = bin_of(X_region[:, 0], n_bins)
            bin_vals, bin_counts = np.unique(bins_in_region, return_counts=True)
            dominant_bin = int(bin_vals[np.argmax(bin_counts)])

            mu_star_prev2 = region_center(region_dynamics, dominant_bin,
                                          t_prev2, n_distance)
            mu_star_prev = region_center(region_dynamics, dominant_bin,
                                         t_prev, n_distance)
            mu_star_current = region_center(region_dynamics, dominant_bin,
                                            t_new, n_distance)

            Delta_GT = mu_star_prev - mu_star_prev2
            Delta_NEXT = mu_star_current - mu_star_prev

            # mu_pop_full/Delta_POP_full: CHINH XAC cong thuc production
            # predict.py (mean tren CA D chieu, ke ca x1) -- dung de
            # sinh offspring PREDICT_POP trung thuc voi production.
            # mu_pop_prev/Delta_POP (bao cao, D-1 chieu, CHI distance
            # vars): dung de so sanh cong bang voi Delta_NEXT (ground
            # truth cung chi dinh nghia tren distance vars).
            Z_region = (X_region - xl) / (xu - xl)
            mu_pop_full_prev = np.mean(Z_region, axis=0)
            mu_pop_prev = mu_pop_full_prev[1:]
            if n_prev2 > 0:
                Z_prev2 = (X_prev2_region - xl) / (xu - xl)
                mu_pop_full_prev2 = np.mean(Z_prev2, axis=0)
                mu_pop_prev2 = mu_pop_full_prev2[1:]
                Delta_POP_full = (mu_pop_full_prev - mu_pop_full_prev2) * LAMBDA_PRED
                Delta_POP = Delta_POP_full[1:]
                has_history = True
            else:
                mu_pop_prev2 = None
                Delta_POP = None
                Delta_POP_full = None
                has_history = False

            eps = 1e-12
            if has_history:
                cos_pop_next = float(
                    np.dot(Delta_POP, Delta_NEXT)
                    / (np.linalg.norm(Delta_POP) * np.linalg.norm(Delta_NEXT) + eps)
                ) if np.linalg.norm(Delta_NEXT) > 1e-9 and np.linalg.norm(Delta_POP) > 1e-9 \
                    else float("nan")
                # T13: "handle zero norm safely" -- danh dau NaN (chu
                # khong phai mot so khong lo vo nghia ~1e10 tu chia
                # cho eps) khi Delta_NEXT~0 (vd vung stationary), cung
                # logic voi cosine o tren, vi "relative error"/"norm
                # ratio" khong co y nghia khi mau so that su la 0.
                if np.linalg.norm(Delta_NEXT) > 1e-9:
                    rel_motion_error = float(
                        np.linalg.norm(Delta_POP - Delta_NEXT)
                        / (np.linalg.norm(Delta_NEXT) + eps))
                    norm_ratio = float(
                        np.linalg.norm(Delta_POP) / (np.linalg.norm(Delta_NEXT) + eps))
                else:
                    rel_motion_error = float("nan")
                    norm_ratio = float("nan")
                centroid_error_prev2 = float(
                    np.linalg.norm(mu_pop_prev2 - mu_star_prev2))
            else:
                cos_pop_next = float("nan")
                rel_motion_error = float("nan")
                norm_ratio = float("nan")
                centroid_error_prev2 = float("nan")

            centroid_error_prev = float(np.linalg.norm(mu_pop_prev - mu_star_prev))

            # Delta_GT khong dinh nghia "target" cho x1 (bien vi tri tu
            # do) -- dem 0 cho thanh phan x1 truoc khi dung de sinh
            # offspring (T3/T4: khong dung future info; x1-component=0
            # la quy uoc, khong phai thong tin tuong lai).
            Delta_GT_full = np.concatenate([[0.0], Delta_GT])

            def evaluate_fn(X_new, _t=t_new, _rd=region_dynamics):
                return cp_evaluate(X_new, _t, _rd)

            children = {}
            child_dist = {}
            for variant in ("LOCAL", "PREDICT_POP", "PREDICT_GT"):
                if variant in ("PREDICT_POP", "PREDICT_GT") and not has_history:
                    children[variant] = None
                    child_dist[variant] = float("nan")
                    continue
                ch = run_variant(variant, X_region, X_prev2_region,
                                Delta_POP_full, Delta_GT_full,
                                sigma_predict, r, seed, change_idx, xl, xu)
                children[variant] = ch
                d = distance_to_region_ps(ch, region_dynamics, dominant_bin,
                                          t_new)
                child_dist[variant] = float(np.median(d))

            # -------------------- population-level IGD (isolated trial) --------
            # FIX (T14 regression, phat hien qua test): nsga2_survival()
            # (RankAndCrowding cua pymoo) dung tie-breaking NGAU NHIEN
            # noi bo KHONG duoc seed qua tham so nao cua ham -- xac
            # minh thuc nghiem: goi 2 lan voi CUNG input cho 2 ket qua
            # KHAC nhau. Day la mot lo hong tai lap chua tung duoc phat
            # hien truoc (cac test T14-kieu cua E2a/E2a-R chi kiem tra
            # phan sinh offspring, chua bao gio di qua buoc survival).
            # Sua (trong pham vi module nay, KHONG sua oracle.py dang
            # frozen): set np.random.seed(...) tat dinh NGAY TRUOC moi
            # lan goi nsga2_survival -- da xac nhan lam cho no tat dinh.
            def quality_for(ch, variant_key):
                if ch is None or len(ch) == 0:
                    F_new = F_plus
                else:
                    F_children = evaluate_fn(ch)
                    X_pool = np.vstack([X_pop, ch])
                    F_pool = np.vstack([F_plus, F_children])
                    survival_seed = (hash((seed, change_idx, r, variant_key))
                                     % (2**31))
                    np.random.seed(survival_seed)
                    _, F_new = nsga2_survival(X_pool, F_pool,
                                              n_survive=POP_SIZE, xl=xl, xu=xu)
                return float(igd(F_new, pf_star))

            IGD_KEEP = quality_for(None, "KEEP")
            IGD_LOCAL = quality_for(children["LOCAL"], "LOCAL")
            IGD_PREDICT_POP = (quality_for(children["PREDICT_POP"], "PREDICT_POP")
                               if children["PREDICT_POP"] is not None else float("nan"))
            IGD_PREDICT_GT = (quality_for(children["PREDICT_GT"], "PREDICT_GT")
                              if children["PREDICT_GT"] is not None else float("nan"))

            rows.append({
                "seed": seed, "event": change_idx,
                "t_prev2": float(t_prev2), "t_prev": float(t_prev),
                "t_current": float(t_new),
                "region_id": int(r), "region_type": str(region_type),
                "sigma_predict": sigma_predict,
                "n_prev2": n_prev2, "n_prev": n_prev,
                "mu_star_prev2": mu_star_prev2.tolist(),
                "mu_star_prev": mu_star_prev.tolist(),
                "mu_star_current": mu_star_current.tolist(),
                "mu_pop_prev2": (mu_pop_prev2.tolist() if mu_pop_prev2 is not None else None),
                "mu_pop_prev": mu_pop_prev.tolist(),
                "Delta_GT": Delta_GT.tolist(),
                "Delta_NEXT": Delta_NEXT.tolist(),
                "Delta_POP": (Delta_POP.tolist() if Delta_POP is not None else None),
                "cosine_POP_vs_NEXT": cos_pop_next,
                "relative_motion_error": rel_motion_error,
                "norm_ratio": norm_ratio,
                "centroid_error_prev2": centroid_error_prev2,
                "centroid_error_prev": centroid_error_prev,
                "occupancy_change": abs(n_prev - n_prev2),
                "child_distance_LOCAL": child_dist["LOCAL"],
                "child_distance_PREDICT_POP": child_dist["PREDICT_POP"],
                "child_distance_PREDICT_GT": child_dist["PREDICT_GT"],
                "IGD_KEEP": IGD_KEEP, "IGD_LOCAL": IGD_LOCAL,
                "IGD_PREDICT_POP": IGD_PREDICT_POP,
                "IGD_PREDICT_GT": IGD_PREDICT_GT,
                "FE_LOCAL": BUDGET,
                "FE_PREDICT_POP": BUDGET if has_history else None,
                "FE_PREDICT_GT": BUDGET if has_history else None,
            })

        prev_regions = current_regions
        t_prev2 = t_prev
        t_current = t_new
        X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == "__main__":
    pass
