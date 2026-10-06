"""
E2a-R -- Revised Controlled Operator Oracle Validation.

Muc tieu: kiem tra xem gain cua "regional oracle" trong E2a co THAT SU
den tu cau truc Pareto-region hay khong, bang 2 sua doi protocol
chinh:

    1. RANDOM-PARTITION REGIONAL ORACLE NULL (size-preserving, giu
       dung so luong VA kich thuoc active region) -- so sanh
       Q_regional (tu partition that) voi Q_regional cua K partition
       ngau nhien CUNG kich thuoc, cho ra excess_ROG (Section 3-6).
    2. FE-MATCHED GLOBAL COMPARATOR -- global action duoc cap CUNG
       tong so extra FE nhu regional oracle da dung, tranh regional
       "thang" chi vi dung nhieu evaluation hon (Section 7-8).

FREEZE (khong sua): operators/*, budget.py, E1 detector
(partition/severity/rhi/quality_loss), controlled_regional.py,
E2a_controlled_operator_oracle.py (giu nguyen lam tham chieu/so sanh
ket qua goc), operator parameters (sigma/lambda/B).

Module nay TU VIET LAI co che oracle (khong import respond()/
generate_children() tu E2a_controlled_operator_oracle.py) vi RNG key
can them 1 chieu moi (null_rep_idx) va budget can tro thanh
per-region dict (cho FE-matched) thay vi 1 so B duy nhat -- 2 thay doi
signature nay khong the "reuse" sach ma khong sua file E2a goc. Cac
ham thuan co hoc khong lien quan protocol (nsga2_survival,
split_by_region, cac operator, equal_budget, igd, partition,
severity, normalize, reference) van duoc TAI SU DUNG nguyen ven.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.detectors.severity import severity          # chi de GHI
from prdrl.metrics.igd import igd

from prdrl.operators.keep import keep
from prdrl.operators.local import local
from prdrl.operators.predict import predict
from prdrl.operators.diversify import diversify
from prdrl.operators.budget import equal_budget

from prdrl.oracle.oracle import nsga2_survival, split_by_region

from prdrl.history.region_history import RegionHistory
from prdrl.features.forecast_state import get_predict_motion_from_history
from prdrl.core.seeding import deterministic_rng

from prdrl.problems.controlled_operator import (
    evaluate as op_evaluate, get_bounds, get_true_pf,
    make_region_dynamics_map, ground_truth_dynamic_label, SCENARIOS,
)

import prdrl.experiments.E2a_controlled_operator_oracle as e2a


# ============================================================
# LOCKED CONFIG -- GIONG HET E2a (Section 0: khong tune)
# ============================================================
ACTIONS = e2a.ACTIONS
SIGMA_LOCAL = e2a.SIGMA_LOCAL
SIGMA_PREDICT = e2a.SIGMA_PREDICT
SIGMA_DIVERSIFY = e2a.SIGMA_DIVERSIFY
LAMBDA_PREDICT = e2a.LAMBDA_PREDICT
BUDGET = e2a.BUDGET
N_OBJ = e2a.N_OBJ
D = e2a.D
POP_SIZE = e2a.POP_SIZE
N_PARTITIONS = e2a.N_PARTITIONS
WARMUP_GENERATIONS = e2a.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2a.GENERATIONS_PER_CHANGE

compute_time = e2a.compute_time
evolve_nsga2 = e2a.evolve_nsga2

REAL_SENTINEL = -1     # null_rep_idx danh cho partition THAT


# ============================================================
# RNG: (base_seed, event_id, null_rep_idx, region_id, action_id)
# (Section 16)
# ============================================================
_ACTION_CODE = {"KEEP": 0, "LOCAL": 1, "PREDICT": 2, "DIVERSIFY": 3}


def make_rng(base_seed, event_id, null_rep_idx, region_id, action):
    # null_rep_idx that (REAL_SENTINEL=-1) tach biet hoan toan voi moi
    # null_rep_idx>=0 -- offset +1 de tranh 2 gia tri am/khac dau trung
    # SeedSequence entropy giua REAL(-1) va mot null_rep gia thuyet nao
    # do (khong xay ra vi null_rep_idx>=0 luon, nhung an toan hon).
    key = int(null_rep_idx) + 1
    ss = np.random.SeedSequence([int(base_seed), int(event_id), key,
                                 int(region_id), _ACTION_CODE[action]])
    return np.random.default_rng(ss)


# ============================================================
# Size-preserving random partition (Section 3, 15) -- giu dung R_active
# va kich thuoc tung active region, region rong VAN rong.
# ============================================================

def size_preserving_random_partition(region_ids_real, R, rng):
    region_ids_real = np.asarray(region_ids_real)
    N = len(region_ids_real)
    sizes = [int(np.sum(region_ids_real == r)) for r in range(R)]
    perm = rng.permutation(N)
    fake_ids = np.empty(N, dtype=int)
    start = 0
    for r in range(R):
        fake_ids[perm[start:start + sizes[r]]] = r
        start += sizes[r]
    return fake_ids


# ============================================================
# FE-matched per-region budget allocation (Section 7, deterministic)
# ============================================================

def fe_matched_budgets(B_total, active_regions):
    active_sorted = sorted(active_regions)
    R_active = len(active_sorted)
    if R_active == 0:
        return {}
    base = B_total // R_active
    remainder = B_total % R_active
    budgets = {}
    for i, r in enumerate(active_sorted):
        budgets[r] = base + (1 if i < remainder else 0)
    return budgets


# ============================================================
# Offspring generation -- PREDICT unavailable (khong fallback), giong
# protocol E2a goc, chi them null_rep_idx vao RNG key.
# ============================================================

def generate_children(action, region_id, X_region, history, t,
                      n_offspring, xl, xu, base_seed, event_id,
                      null_rep_idx):
    Dn = len(xl)
    if len(X_region) == 0:
        return np.empty((0, Dn))
    if action == "KEEP":
        return keep(X_region, n_offspring, xl, xu, 0.0, None)

    rng = make_rng(base_seed, event_id, null_rep_idx, region_id, action)

    if action == "LOCAL":
        return local(X_region, n_offspring, xl, xu, SIGMA_LOCAL, rng)
    if action == "DIVERSIFY":
        return diversify(X_region, n_offspring, xl, xu, SIGMA_DIVERSIFY, rng)
    if action == "PREDICT":
        motion = get_predict_motion_from_history(history, region_id, t,
                                                 LAMBDA_PREDICT)
        delta_hat = motion["delta_hat"] if motion is not None else None
        return predict(X_prev=X_region, delta_hat=delta_hat,
                       n_offspring=n_offspring, xl=xl, xu=xu,
                       sigma_pred=SIGMA_PREDICT, rng=rng,
                       sigma_local=SIGMA_LOCAL)
    raise ValueError(f"Unknown action: {action}")


def respond(X_base, F_plus, region_ids, R, actions, budgets,
           history, t, evaluate_fn, xl, xu, base_seed, event_id,
           null_rep_idx):
    """budgets: dict region_id -> n_offspring (bo qua cho KEEP)."""
    X_base = np.asarray(X_base, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    region_ids = np.asarray(region_ids)
    N = len(X_base)

    children_all = []
    for r in range(R):
        mask = region_ids == r
        X_region = X_base[mask]
        if len(X_region) == 0:
            continue
        action = actions.get(r, "KEEP")
        n_offspring = 0 if action == "KEEP" else budgets.get(r, 0)
        children = generate_children(action, r, X_region, history, t,
                                     n_offspring, xl, xu, base_seed,
                                     event_id, null_rep_idx)
        if len(children) > 0:
            children_all.append(children)

    if len(children_all) == 0:
        return {"available": True, "X": X_base.copy(), "F": F_plus.copy(),
               "extra_fe": 0}

    X_children = np.vstack(children_all)
    F_children = evaluate_fn(X_children)
    extra_fe = len(X_children)

    X_pool = np.vstack([X_base, X_children])
    F_pool = np.vstack([F_plus, F_children])
    X_new, F_new = nsga2_survival(X_pool, F_pool, n_survive=N, xl=xl, xu=xu)
    return {"available": True, "X": X_new, "F": F_new, "extra_fe": extra_fe}


def operator_heterogeneity_index(preferred_actions, eps=1e-12):
    if len(preferred_actions) < 2:
        return 0.0
    counts = np.array([sum(a == action for a in preferred_actions)
                       for action in ACTIONS], dtype=float)
    p = counts / counts.sum()
    nonzero = p > 0
    entropy = -np.sum(p[nonzero] * np.log(p[nonzero] + eps))
    return float(entropy / np.log(len(ACTIONS)))


# ============================================================
# Full oracle protocol on a GIVEN region_ids assignment (real or
# random) -- isolated trials -> preferred action -> OHI -> combined
# regional Q (Section 3, 11). KHONG tinh "best global" o day (best
# global chi tinh 1 lan tren partition THAT, dung chung cho moi null
# replicate -- xem run_one_event()).
# ============================================================

def run_regional_protocol(X_base, F_plus, region_ids, R, history, t,
                          evaluate_fn, quality_fn, xl, xu, budget,
                          base_seed, event_id, null_rep_idx):
    active_regions = [r for r in range(R) if np.any(region_ids == r)]
    equal_budgets = {r: equal_budget("LOCAL", int(np.sum(region_ids == r)),
                                     budget)
                     for r in active_regions}

    keep_result = respond(X_base, F_plus, region_ids, R, {}, {},
                          history, t, evaluate_fn, xl, xu,
                          base_seed, event_id, null_rep_idx)
    Q_keep = quality_fn(keep_result["F"])

    utilities = {}
    preferred_action = {}
    for r in active_regions:
        utilities[r] = {"KEEP": 0.0}
        for action in ("LOCAL", "PREDICT", "DIVERSIFY"):
            result = respond(X_base, F_plus, region_ids, R, {r: action},
                             equal_budgets, history, t, evaluate_fn,
                             xl, xu, base_seed, event_id, null_rep_idx)
            Q = quality_fn(result["F"])
            utilities[r][action] = float(Q - Q_keep)

        best_action, best_u = "KEEP", 0.0
        for action in ACTIONS:
            u = utilities[r].get(action, float("nan"))
            if np.isnan(u):
                continue
            if u > best_u:
                best_u = u
                best_action = action
        preferred_action[r] = best_action

    OHI = operator_heterogeneity_index(
        [preferred_action[r] for r in active_regions])

    regional_actions = {r: preferred_action[r] for r in active_regions}
    regional_result = respond(X_base, F_plus, region_ids, R,
                              regional_actions, equal_budgets,
                              history, t, evaluate_fn, xl, xu,
                              base_seed, event_id, null_rep_idx)
    Q_regional = quality_fn(regional_result["F"])

    return {
        "Q_keep": float(Q_keep), "utilities": utilities,
        "preferred_action": preferred_action, "OHI": float(OHI),
        "Q_regional": float(Q_regional),
        "regional_extra_fe": int(regional_result["extra_fe"]),
        "active_regions": active_regions,
    }


def run_best_global(X_base, F_plus, region_ids, R, active_regions,
                    history, t, evaluate_fn, quality_fn, xl, xu,
                    budget, base_seed, event_id, Q_keep):
    """Best-global STANDARD (B moi active region, protocol E2a goc)."""
    equal_budgets = {r: equal_budget("LOCAL", int(np.sum(region_ids == r)),
                                     budget)
                     for r in active_regions}
    global_results = {}
    for action in ACTIONS:
        if action == "KEEP":
            global_results[action] = {"available": True, "Q": float(Q_keep),
                                      "IGD": float(-Q_keep), "extra_fe": 0}
            continue
        actions_map = {r: action for r in active_regions}
        result = respond(X_base, F_plus, region_ids, R, actions_map,
                         equal_budgets, history, t, evaluate_fn,
                         xl, xu, base_seed, event_id, REAL_SENTINEL)
        Q = quality_fn(result["F"])
        global_results[action] = {"available": True, "Q": float(Q),
                                  "IGD": float(-Q),
                                  "extra_fe": int(result["extra_fe"])}

    best_action, best_Q = None, -np.inf
    for action in ACTIONS:
        info = global_results[action]
        if info["available"] and info["Q"] > best_Q:
            best_Q, best_action = info["Q"], action
    return global_results, best_action, best_Q


def run_best_global_fe_matched(X_base, F_plus, region_ids, R,
                               active_regions, history, t,
                               evaluate_fn, quality_fn, xl, xu, B_total,
                               base_seed, event_id):
    """Section 7-8: chi so sanh 3 active action (LOCAL/PREDICT/
    DIVERSIFY) voi CUNG tong extra FE = B_total (regional oracle da
    dung). KEEP luon 0 FE, bao cao rieng, KHONG gop vao 'active FE
    comparator' (Section 8)."""
    budgets = fe_matched_budgets(B_total, active_regions)
    results = {}
    for action in ("LOCAL", "PREDICT", "DIVERSIFY"):
        actions_map = {r: action for r in active_regions}
        result = respond(X_base, F_plus, region_ids, R, actions_map,
                         budgets, history, t, evaluate_fn, xl, xu,
                         base_seed, event_id, REAL_SENTINEL)
        Q = quality_fn(result["F"])
        results[action] = {"available": True, "Q": float(Q),
                           "IGD": float(-Q),
                           "extra_fe": int(result["extra_fe"])}

    best_action, best_Q = None, -np.inf
    for action, info in results.items():
        if info["available"] and info["Q"] > best_Q:
            best_Q, best_action = info["Q"], action
    return results, best_action, best_Q, budgets


# ============================================================
# Ground-truth dynamic-label audit per actual reference region
# (Section 9-10)
# ============================================================

def region_dynamic_composition(X_base, region_ids, R, region_dynamics,
                               purity_threshold=0.8):
    gt_labels = ground_truth_dynamic_label(X_base, region_dynamics)
    composition = {}
    dominant_type = {}
    purity = {}
    for r in range(R):
        mask = region_ids == r
        n = int(np.sum(mask))
        if n == 0:
            composition[r] = {}
            dominant_type[r] = None
            purity[r] = float("nan")
            continue
        labels, counts = np.unique(gt_labels[mask], return_counts=True)
        frac = {lbl: float(c) / n for lbl, c in zip(labels, counts)}
        composition[r] = frac
        p = max(frac.values())
        purity[r] = p
        dominant_type[r] = (max(frac, key=frac.get) if p >= purity_threshold
                           else "MIXED")
    return composition, dominant_type, purity


# ============================================================
# Null history builder -- for each null replica k, build a separate
# RegionHistory from the SAME historical populations but under
# RANDOM group assignments (keyed deterministically by base_seed /
# event_id / k / tau).  PREDICT for null group r then uses the
# centroid motion of THAT random group at t-2 and t-1, NOT the
# motion of real Pareto-region r.
# ============================================================

def _build_null_history(history, R, base_seed, event_id, k, t, xl, xu):
    """Build RegionHistory for null replica k at event t.

    One permutation is generated per (base_seed, event_id, k) with NO tau
    in the RNG key. The SAME permutation is applied at both t-2 and t-1,
    so null group r selects consistent positional slots across time. This
    eliminates tau-specific re-grouping noise from delta_hat = mu(t-1) -
    mu(t-2), ensuring the null motion reflects actual population motion
    rather than regrouping artefacts.
    """
    # Determine population size from first available historical snapshot
    N = 0
    for tau_probe in (t - 2, t - 1):
        if tau_probe < 0:
            continue
        total = sum(
            (history.get_region(r, tau_probe).n
             if history.get_region(r, tau_probe) is not None else 0)
            for r in range(R)
        )
        if total > 0:
            N = total
            break
    # Single permutation for this replica -- shared across all historical taus
    rng_k = deterministic_rng(int(base_seed), int(event_id), int(k))
    perm_k = rng_k.permutation(N) if N > 0 else np.arange(0, dtype=int)

    h_null = RegionHistory(n_regions=R)
    if N == 0:
        return h_null
    for tau in (t - 2, t - 1):
        if tau < 0:
            continue
        # Reconstruct full population at tau from all real regions
        xs, fs, real_ids_tau = [], [], []
        for r in range(R):
            snap = history.get_region(r, tau)
            if snap is None or snap.n == 0:
                continue
            xs.append(snap.X)
            fs.append(snap.F)
            real_ids_tau.append(np.full(snap.n, r, dtype=int))
        if not xs:
            continue
        X_tau = np.vstack(xs)
        F_tau = np.vstack(fs)
        real_ids_arr = np.concatenate(real_ids_tau)
        # Apply the SAME perm_k (size-preserving w.r.t. this tau's real sizes)
        sizes_tau = [int(np.sum(real_ids_arr == r)) for r in range(R)]
        rand_ids_tau = np.empty(len(real_ids_arr), dtype=int)
        start = 0
        for r_idx in range(R):
            rand_ids_tau[perm_k[start:start + sizes_tau[r_idx]]] = r_idx
            start += sizes_tau[r_idx]
        h_null.update(t=tau, X=X_tau, F=F_tau,
                      region_ids=rand_ids_tau, xl=xl, xu=xu)
    return h_null


# ============================================================
# Fixed-rule null helpers -- random projection in decision space.
# w_k is generated once per (base_seed, event_id, k), NO tau in key.
# Individuals are sorted by their projection score and cut by cumulative
# real-region sizes, so group identity is a deterministic function of
# each individual's x-coordinates, independent of real-region ordering.
# ============================================================

def _get_projection_vector(base_seed, event_id, k, D):
    """Return the normalized projection vector w_k for fixed-rule null k."""
    rng_k = deterministic_rng(int(base_seed), int(event_id), int(k))
    w_k = rng_k.standard_normal(D)
    norm = np.linalg.norm(w_k)
    if norm < 1e-12:
        w_k = np.zeros(D)
        w_k[0] = 1.0
    else:
        w_k /= norm
    return w_k


def _fixed_rule_partition(X, region_ids_real, R, w_k, xl, xu):
    """Assign fixed-rule null groups to current population X.

    Groups match real-region sizes exactly. Assignment is a stable sort
    of projection scores w_k^T z(x), where z = (x - xl) / (xu - xl),
    cut at cumulative real-region sizes.
    """
    X = np.asarray(X, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    region_ids_real = np.asarray(region_ids_real)
    denom = xu - xl
    denom = np.where(denom < 1e-12, 1.0, denom)
    z = (X - xl) / denom
    scores = z @ w_k
    order = np.argsort(scores, kind="stable")
    sizes = [int(np.sum(region_ids_real == r)) for r in range(R)]
    null_ids = np.empty(len(X), dtype=int)
    start = 0
    for r_idx in range(R):
        null_ids[order[start:start + sizes[r_idx]]] = r_idx
        start += sizes[r_idx]
    return null_ids


def _build_fixed_rule_null_history(history, R, base_seed, event_id, k, t, xl, xu):
    """Build RegionHistory for fixed-rule null replica k.

    w_k is drawn once from deterministic_rng(base_seed, event_id, k) with
    NO tau in the key, then applied at BOTH t-2 and t-1. Each individual's
    null group assignment depends only on its decision-space coordinates
    projected onto w_k -- not on real-region labels or row order.
    """
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    D = len(xl)

    # Determine population size from first available historical snapshot
    N = 0
    for tau_probe in (t - 2, t - 1):
        if tau_probe < 0:
            continue
        total = sum(
            (history.get_region(r, tau_probe).n
             if history.get_region(r, tau_probe) is not None else 0)
            for r in range(R)
        )
        if total > 0:
            N = total
            break

    h_null = RegionHistory(n_regions=R)
    if N == 0:
        return h_null

    w_k = _get_projection_vector(base_seed, event_id, k, D)

    for tau in (t - 2, t - 1):
        if tau < 0:
            continue
        xs, fs, real_ids_tau = [], [], []
        for r in range(R):
            snap = history.get_region(r, tau)
            if snap is None or snap.n == 0:
                continue
            xs.append(snap.X)
            fs.append(snap.F)
            real_ids_tau.append(np.full(snap.n, r, dtype=int))
        if not xs:
            continue
        X_tau = np.vstack(xs)
        F_tau = np.vstack(fs)
        real_ids_arr = np.concatenate(real_ids_tau)
        rand_ids_tau = _fixed_rule_partition(X_tau, real_ids_arr, R, w_k, xl, xu)
        h_null.update(t=tau, X=X_tau, F=F_tau,
                      region_ids=rand_ids_tau, xl=xl, xu=xu)
    return h_null


# ============================================================
# ONE scored event: real protocol + best-global (standard & FE-matched)
# + K random-partition nulls (Section 3-14).
# ============================================================

def run_event(X_base, F_plus, region_ids, R, history, t,
              evaluate_fn, pf_star, xl, xu, budget, base_seed, event_id,
              K, null_rng):

    def quality_fn(F):
        return -igd(F, pf_star)

    real = run_regional_protocol(X_base, F_plus, region_ids, R,
                                 history, t, evaluate_fn,
                                 quality_fn, xl, xu, budget, base_seed,
                                 event_id, REAL_SENTINEL)

    global_std_results, best_global_std_action, best_global_std_Q = \
        run_best_global(X_base, F_plus, region_ids, R,
                        real["active_regions"], history, t,
                        evaluate_fn, quality_fn, xl, xu, budget,
                        base_seed, event_id, real["Q_keep"])

    fe_results, best_fe_action, best_fe_Q, fe_budgets = \
        run_best_global_fe_matched(X_base, F_plus, region_ids, R,
                                   real["active_regions"], history, t,
                                   evaluate_fn, quality_fn, xl, xu,
                                   real["regional_extra_fe"], base_seed,
                                   event_id)

    eps = 1e-12
    ROG_standard = ((real["Q_regional"] - best_global_std_Q)
                    / (abs(best_global_std_Q) + eps))
    Delta_IGD_standard = (-best_global_std_Q) - (-real["Q_regional"])

    if best_fe_action is not None:
        ROG_FE = (real["Q_regional"] - best_fe_Q) / (abs(best_fe_Q) + eps)
        Delta_IGD_FE = (-best_fe_Q) - (-real["Q_regional"])
    else:
        ROG_FE = float("nan")
        Delta_IGD_FE = float("nan")

    # -------------------- K random-partition nulls --------------------
    # Each null replica k uses its OWN history built from random group
    # assignments at t-2 and t-1 (via _build_null_history).  PREDICT
    # for null group r therefore uses the centroid motion of the RANDOM
    # group r's members at those times, not the motion of real
    # Pareto-region r.  This is the correct null: both real and random
    # oracles compute PREDICT from their own respective group histories,
    # making the comparison fair and contamination-free.
    rog_rand, ohi_rand, q_rand = [], [], []
    for k in range(K):
        rand_region_ids = size_preserving_random_partition(
            region_ids, R, null_rng)
        history_rand_k = _build_null_history(
            history, R, base_seed, event_id, k, t, xl, xu)
        rand_result = run_regional_protocol(
            X_base, F_plus, rand_region_ids, R, history_rand_k, t,
            evaluate_fn, quality_fn, xl, xu, budget, base_seed, event_id, k)
        rog_k = ((rand_result["Q_regional"] - best_global_std_Q)
                / (abs(best_global_std_Q) + eps))
        rog_rand.append(rog_k)
        ohi_rand.append(rand_result["OHI"])
        q_rand.append(rand_result["Q_regional"])

    rog_rand = np.array(rog_rand)
    ohi_rand = np.array(ohi_rand)

    null_rog_median = float(np.median(rog_rand))
    null_rog_mean = float(np.mean(rog_rand))
    null_rog_std = float(np.std(rog_rand))
    real_rog_percentile = float(np.mean(rog_rand <= ROG_standard))
    excess_ROG = ROG_standard - null_rog_median
    excess_Delta_IGD = Delta_IGD_standard - float(
        np.median([(-best_global_std_Q) - (-q) for q in q_rand]))

    null_ohi_median = float(np.median(ohi_rand))
    real_ohi_percentile = float(np.mean(ohi_rand <= real["OHI"]))
    excess_OHI = real["OHI"] - null_ohi_median

    return {
        "real": real,
        "global_standard": {
            "results": global_std_results,
            "best_action": best_global_std_action,
            "best_Q": best_global_std_Q,
        },
        "global_fe_matched": {
            "results": fe_results, "best_action": best_fe_action,
            "best_Q": best_fe_Q, "budgets": fe_budgets,
        },
        "ROG_standard": float(ROG_standard),
        "Delta_IGD_standard": float(Delta_IGD_standard),
        "ROG_FE": float(ROG_FE), "Delta_IGD_FE": float(Delta_IGD_FE),
        "K": K,
        "random_ROG_median": null_rog_median,
        "random_ROG_mean": null_rog_mean,
        "random_ROG_std": null_rog_std,
        "real_ROG_percentile": real_rog_percentile,
        "excess_ROG": float(excess_ROG),
        "excess_Delta_IGD": float(excess_Delta_IGD),
        "random_OHI_median": null_ohi_median,
        "real_OHI_percentile": real_ohi_percentile,
        "excess_OHI": float(excess_OHI),
    }


# ============================================================
# Full run: warm-up -> baseline trajectory -> scored events (giong
# E2a, oracle KHONG anh huong baseline -- Section 8 cua E2a, giu
# nguyen).
# ============================================================

def run_one(scenario, seed, n_scored_changes, K, n_bins=None):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    n_bins = R if n_bins is None else n_bins
    region_dynamics = make_region_dynamics_map(scenario, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))

    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    history = RegionHistory(n_regions=R)
    rows = []
    gen = 0
    n_total_changes = 1 + n_scored_changes

    for change_idx in range(1, n_total_changes + 1):
        F_minus = op_evaluate(X_pop, t_current, region_dynamics)
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        F_plus = op_evaluate(X_pop, t_new, region_dynamics)

        F_tilde_minus, F_tilde_plus = normalize(F_minus, F_plus)
        F_tilde_minus = np.asarray(F_tilde_minus)
        F_tilde_plus = np.asarray(F_tilde_plus)
        region_ids = np.asarray(partition(F_tilde_minus, V))

        s_r, c_r, _q = severity(F_tilde_minus, F_tilde_plus, region_ids,
                                R, q_prev=1.0)
        active_mask = ~np.isnan(np.asarray(s_r, dtype=float))
        R_active = int(np.sum(active_mask))
        region_sizes = [int(np.sum(region_ids == r)) for r in range(R)]

        composition, dominant_type, purity = region_dynamic_composition(
            X_pop, region_ids, R, region_dynamics)

        if change_idx == 1:
            history.update(t=0, X=X_pop, F=F_minus, region_ids=region_ids,
                           xl=xl, xu=xu)
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        def evaluate_fn(X_new, _t=t_new, _rd=region_dynamics):
            return op_evaluate(X_new, _t, _rd)

        null_rng = e2a.deterministic_rng(seed, hash(scenario) % 100000,
                                         change_idx)

        history.update(t=change_idx - 1, X=X_pop, F=F_minus,
                       region_ids=region_ids, xl=xl, xu=xu)

        result = run_event(X_pop, F_plus, region_ids, R, history, change_idx,
                           evaluate_fn, pf_star, xl, xu, BUDGET, seed,
                           change_idx, K, null_rng)

        row = {
            "scenario": scenario, "seed": seed, "event_idx": change_idx,
            "t_old": float(t_current), "t_new": float(t_new),
            "R_total": R, "R_active": R_active,
            "region_sizes": region_sizes,
            "active_fraction": R_active / R,
            "region_dynamic_composition": composition,
            "region_dynamic_type": dominant_type,
            "region_dynamic_purity": purity,
            "utilities_real": result["real"]["utilities"],
            "preferred_actions_real": result["real"]["preferred_action"],
            "OHI_real": result["real"]["OHI"],
            "Q_keep": result["real"]["Q_keep"],
            "IGD_keep": -result["real"]["Q_keep"],
            "best_global_standard_action": result["global_standard"]["best_action"],
            "best_global_standard_Q": result["global_standard"]["best_Q"],
            "best_global_standard_IGD": -result["global_standard"]["best_Q"],
            "best_global_standard_FE": (
                result["global_standard"]["results"]
                [result["global_standard"]["best_action"]]["extra_fe"]
                if result["global_standard"]["best_action"] else None),
            "best_global_FE_action": result["global_fe_matched"]["best_action"],
            "best_global_FE_Q": result["global_fe_matched"]["best_Q"],
            "best_global_FE_IGD": (-result["global_fe_matched"]["best_Q"]
                                   if result["global_fe_matched"]["best_action"]
                                   else float("nan")),
            "best_global_FE": (
                result["global_fe_matched"]["results"]
                [result["global_fe_matched"]["best_action"]]["extra_fe"]
                if result["global_fe_matched"]["best_action"] else None),
            "regional_Q": result["real"]["Q_regional"],
            "regional_IGD": -result["real"]["Q_regional"],
            "regional_FE": result["real"]["regional_extra_fe"],
            "ROG_standard": result["ROG_standard"],
            "ROG_FE": result["ROG_FE"],
            "Delta_IGD_standard": result["Delta_IGD_standard"],
            "Delta_IGD_FE": result["Delta_IGD_FE"],
            "random_null_K": K,
            "random_ROG_median": result["random_ROG_median"],
            "random_ROG_mean": result["random_ROG_mean"],
            "random_ROG_std": result["random_ROG_std"],
            "real_ROG_percentile": result["real_ROG_percentile"],
            "excess_ROG": result["excess_ROG"],
            "excess_Delta_IGD": result["excess_Delta_IGD"],
            "random_OHI_median": result["random_OHI_median"],
            "real_OHI_percentile": result["real_OHI_percentile"],
            "excess_OHI": result["excess_OHI"],
        }
        rows.append(row)

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
    if obj is None:
        return None
    return obj


if __name__ == "__main__":
    pass
