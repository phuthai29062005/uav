"""
R1-P — POP Estimator Transfer Test.

Scientific question:
    Does REGION_PROTECTED still solve the cross-region coverage-loss problem
    when children are generated using the actual population-based predictor
    instead of the GT oracle?

This experiment isolates RESPONSE ALLOCATION from PREDICTION NOISE by using:
    POP_PREDICT delta with sigma = 0  (no noise)

PRIMARY MODES compared:
    POP_UNRESTRICTED — normal NSGA-II on [OLD; POP_CHILDREN]
    POP_PROTECTED    — region-protected survival on [OLD; POP_CHILDREN]

Children are byte-identical across modes (same seed, same delta_hat, sigma=0).

OPTIONAL GT ANCHOR (included, cheap):
    GT_UNRESTRICTED — normal NSGA-II on [OLD; GT_CHILDREN]
    GT_PROTECTED    — region-protected survival on [OLD; GT_CHILDREN]

GT children are separate from POP children and used only for comparison.

CONSTRAINTS (hard):
    - nsga2_survival is NOT modified.
    - region_protected_survival is reused from R1 unchanged.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0, sigma_POP=0.
    - Production sigma_P=0.03 unchanged.
    - RegionHistory is used in read-only mode for prediction.
    - No GT information in delta_hat computation.
    - Missing POP history → skip region (record has_history=False).
    - Do NOT replace missing POP motion with GT.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.R1_region_protected_response as r1

from prdrl.history.region_history import RegionHistory
from prdrl.features.forecast_state import get_predict_motion_from_history
from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import split_by_region

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    distance_to_region_ps, bin_of,
)

# ============================================================
# Config — production locked
# ============================================================
LAMBDA_PRED  = 1.0       # production — never change here
SIGMA_POP    = 0.0       # primary: isolate allocation from noise
BUDGET       = e2p3.BUDGET          # 10
POP_SIZE     = e2p3.POP_SIZE        # 100
D            = e2p3.D               # 10
N_OBJ        = e2p3.N_OBJ           # 2
N_PARTITIONS = e2p3.N_PARTITIONS    # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

# Re-use all helpers from E2P3 and R1
compute_time            = e2p3.compute_time
evolve_nsga2            = e2p3.evolve_nsga2
make_rng                = e2p3.make_rng
make_scenario_dynamics  = e2p3.make_scenario_dynamics
median_nearest_pf_distance  = e2p3.median_nearest_pf_distance
nsga2_survival_traced       = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed      = e2p3._e2p3_common_surv_seed
generate_base_samples       = e2p3.generate_base_samples
apply_samples               = e2p3.apply_samples

compute_ranks_crowding_all    = r1.compute_ranks_crowding_all
compute_pf_coverage_per_point = r1.compute_pf_coverage_per_point
region_protected_survival     = r1.region_protected_survival


# ============================================================
# Estimator diagnostics
# ============================================================

def estimator_diagnostics(delta_pop, delta_gt):
    """Return cosine, relative error, and norm ratio between pop and gt deltas.

    Both arguments must be 1-D arrays of shape (D,) in [0,1] normalized space.
    Returns a dict. All values NaN if either delta is zero-norm.
    """
    n_pop = np.linalg.norm(delta_pop)
    n_gt  = np.linalg.norm(delta_gt)
    eps   = 1e-12

    if n_pop < eps and n_gt < eps:
        return dict(estimator_cosine=float("nan"),
                    estimator_relative_error=float("nan"),
                    estimator_norm_ratio=float("nan"),
                    norm_delta_pop=float(n_pop),
                    norm_delta_gt=float(n_gt))

    cosine = float(np.dot(delta_pop, delta_gt) / (n_pop * n_gt + eps))
    rel_err = float(np.linalg.norm(delta_pop - delta_gt) / (n_gt + eps))
    norm_ratio = float(n_pop / (n_gt + eps))
    return dict(estimator_cosine=cosine,
                estimator_relative_error=rel_err,
                estimator_norm_ratio=norm_ratio,
                norm_delta_pop=float(n_pop),
                norm_delta_gt=float(n_gt))


# ============================================================
# Per-region R1-P comparison for one scored event
# ============================================================

def run_region_r1p(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
    include_gt_anchor=True,
):
    """Compare POP_UNRESTRICTED vs POP_PROTECTED for region r at production B=10.

    Also records GT_UNRESTRICTED and GT_PROTECTED as anchors if include_gt_anchor.
    Returns a list with one row dict, or empty list if region empty or no POP history.
    """
    mask = region_ids == r
    n_r_before = int(np.sum(mask))
    if n_r_before == 0:
        return []

    X_region = X_pop[mask].copy()
    N_total_old = len(X_pop)
    R = len(V)

    # Region metadata
    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type  = str(labels_r[np.argmax(counts_r)])
    bins_in_r    = bin_of(X_region[:, 0], n_bins)
    bv, bc       = np.unique(bins_in_r, return_counts=True)
    dominant_bin = int(bv[np.argmax(bc)])

    # POP estimator delta (reads only t-2 and t-1 via RegionHistory)
    # history_t is the integer index used in update(t=...) for this event's t-1 snapshot
    motion = get_predict_motion_from_history(
        history, r, t=history_t + 1, lambda_pred=LAMBDA_PRED
    )
    if motion is None:
        # Missing POP history — skip this region
        return [{"has_history": False, "region_id": r,
                 "scenario": None, "seed": None, "event_id": change_idx}]

    delta_pop = motion["delta_hat"]   # shape (D,), normalized [0,1] space

    # GT oracle delta for diagnostics and anchor
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])

    # Estimator diagnostics
    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # Shared RNG seed for children (same across A/B to guarantee identical children)
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = generate_base_samples(BUDGET, D, rng_shared, n_r_before)

    # POP children (sigma=0)
    ch_pop = apply_samples(all_idxs, all_xis, X_region, delta_pop, SIGMA_POP, xl, xu)
    F_ch_pop = cp_evaluate(ch_pop, t_new, region_dynamics)

    # Child quality
    n_unique_pop = len(np.unique(ch_pop, axis=0))
    unique_frac_pop = float(n_unique_pop) / BUDGET

    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span = xu_a - xl_a
    span = np.where(span < 1e-12, np.ones_like(span), span)
    n_clipped_pop = 0
    for pidx, xi in zip(all_idxs, all_xis):
        z_p = (X_region[pidx] - xl_a) / span
        z_raw = z_p + delta_pop / span * span  # delta_pop already in [0,1] units
        # check in normalized space
        z_raw2 = z_p + delta_pop
        n_clipped_pop += int(np.any((z_raw2 < 0.0) | (z_raw2 > 1.0)))
    clipping_frac_pop = float(n_clipped_pop) / BUDGET

    # Normalization basis for PF/occupancy metrics (same as E2P5/R1)
    F_minus_a  = np.asarray(F_minus, dtype=float)
    F_all_norm = np.vstack([F_minus_a, F_before])
    z_m   = F_all_norm.min(axis=0)
    z_nad = F_all_norm.max(axis=0)
    norm_span = (z_nad - z_m) + 1e-12

    # Direct PF reference-region assignment
    pf_star_arr = np.asarray(pf_star, dtype=float)
    pf_norm     = (pf_star_arr - z_m) / norm_span
    pf_ref_rids = np.asarray(partition(pf_norm, V), dtype=int)

    # Occupancy before
    occ_before = np.bincount(region_ids, minlength=R)

    def _coverage_metrics(F_surv, prefix):
        """Per-PF-point coverage metrics for a given survived population."""
        igd_after  = float(igd(F_surv, pf_star))
        delta_igd  = igd_before - igd_after

        _, _, delta_d_arr, _ = compute_pf_coverage_per_point(
            F_before, F_surv, pf_star, region_ids
        )
        # IGD identity check
        identity_residual = abs((igd_after - igd_before) - float(np.mean(delta_d_arr)))

        inside_d  = pf_ref_rids == r
        outside_d = ~inside_d
        mean_di = float(np.mean(delta_d_arr[inside_d]))   if np.any(inside_d)  else float("nan")
        mean_do = float(np.mean(delta_d_arr[outside_d]))  if np.any(outside_d) else float("nan")
        fw_i = float(np.mean(delta_d_arr[inside_d]  > 1e-14)) if np.any(inside_d)  else float("nan")
        fw_o = float(np.mean(delta_d_arr[outside_d] > 1e-14)) if np.any(outside_d) else float("nan")
        frac_worse  = float(np.mean(delta_d_arr > 1e-14))
        frac_impr   = float(np.mean(delta_d_arr < -1e-14))

        F_surv_norm    = (F_surv - z_m) / norm_span
        actual_rids    = np.asarray(partition(F_surv_norm, V), dtype=int)
        actual_occ     = np.bincount(actual_rids, minlength=R)
        actual_resp_delta = int(actual_occ[r]) - int(occ_before[r])
        actual_d_other = actual_occ - occ_before
        actual_d_other[r] = 0
        n_other_lose  = int(np.sum(actual_d_other < 0))
        loss_other    = int(np.sum(np.abs(actual_d_other[actual_d_other < 0])))

        return {
            f"{prefix}_igd_after":                igd_after,
            f"{prefix}_delta_igd":                delta_igd,
            f"{prefix}_mean_delta_pf_inside":     mean_di,
            f"{prefix}_mean_delta_pf_outside":    mean_do,
            f"{prefix}_frac_worse_inside":        fw_i,
            f"{prefix}_frac_worse_outside":       fw_o,
            f"{prefix}_frac_pf_worsened":         frac_worse,
            f"{prefix}_frac_pf_improved":         frac_impr,
            f"{prefix}_actual_resp_delta":        actual_resp_delta,
            f"{prefix}_actual_n_other_losing":    n_other_lose,
            f"{prefix}_actual_total_loss_other":  loss_other,
            f"{prefix}_igd_identity_residual":    float(identity_residual),
        }

    # Common survival seed
    surv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    # ─── POP_UNRESTRICTED ────────────────────────────────────
    X_pool_p = np.vstack([X_pop, ch_pop])
    F_pool_p = np.vstack([F_before, F_ch_pop])
    X_surv_pu, F_surv_pu, surv_idx_pu = nsga2_survival_traced(
        X_pool_p, F_pool_p, POP_SIZE, xl, xu, surv_seed
    )
    surv_old_pu   = surv_idx_pu[surv_idx_pu < N_total_old]
    dropped_old_pu = set(range(N_total_old)) - set(surv_old_pu.tolist())
    n_old_dropped_pu  = len(dropped_old_pu)
    n_ch_survived_pu  = int(np.sum(surv_idx_pu >= N_total_old))
    dropped_other_pu  = sum(1 for i in dropped_old_pu if region_ids[i] != r)
    dropped_same_pu   = n_old_dropped_pu - dropped_other_pu
    metrics_pu = _coverage_metrics(F_surv_pu, "pop_unr")

    # ─── POP_PROTECTED ───────────────────────────────────────
    X_surv_pp, F_surv_pp, final_idx_pp = region_protected_survival(
        X_pop, F_before, ch_pop, F_ch_pop,
        r, region_ids, n_r_before,
        xl, xu, surv_seed,
    )
    surv_old_pp_set = set(int(i) for i in final_idx_pp if i < N_total_old)
    dropped_old_pp  = set(range(N_total_old)) - surv_old_pp_set
    n_old_dropped_pp  = len(dropped_old_pp)
    n_ch_survived_pp  = int(np.sum(final_idx_pp >= N_total_old))
    dropped_other_pp  = sum(1 for i in dropped_old_pp if region_ids[i] != r)
    dropped_same_pp   = n_old_dropped_pp - dropped_other_pp
    metrics_pp = _coverage_metrics(F_surv_pp, "pop_prot")

    # Primary paired contrast
    delta_delta_igd_pop  = metrics_pp["pop_prot_delta_igd"] - metrics_pu["pop_unr_delta_igd"]
    delta_outside_cov    = (metrics_pp["pop_prot_mean_delta_pf_outside"] or 0.0) - \
                           (metrics_pu["pop_unr_mean_delta_pf_outside"] or 0.0)
    delta_ch_surv        = n_ch_survived_pp - n_ch_survived_pu

    row = {
        "has_history": True,
        "budget_b": BUDGET,
        "region_type": region_type,
        "dominant_bin": dominant_bin,
        "n_r_before": n_r_before,
        "igd_before": igd_before,
        "fe": BUDGET,
        # Estimator diagnostics
        "delta_pop": delta_pop.tolist(),
        "delta_gt":  delta_gt_full.tolist(),
        **diag,
        # Child quality
        "n_unique_children_pop":  n_unique_pop,
        "unique_child_frac_pop":  unique_frac_pop,
        "clipping_frac_pop":      clipping_frac_pop,
        # POP_UNRESTRICTED
        "pop_unr_n_old_dropped":        n_old_dropped_pu,
        "pop_unr_n_old_dropped_same":   dropped_same_pu,
        "pop_unr_n_old_dropped_other":  dropped_other_pu,
        "pop_unr_n_ch_survived":        n_ch_survived_pu,
        "pop_unr_child_survival_rate":  float(n_ch_survived_pu) / BUDGET,
        # POP_PROTECTED
        "pop_prot_n_old_dropped":       n_old_dropped_pp,
        "pop_prot_n_old_dropped_same":  dropped_same_pp,
        "pop_prot_n_old_dropped_other": dropped_other_pp,
        "pop_prot_n_ch_survived":       n_ch_survived_pp,
        "pop_prot_child_survival_rate": float(n_ch_survived_pp) / BUDGET,
        # Paired contrasts
        "delta_delta_igd":    delta_delta_igd_pop,
        "delta_outside_cov":  delta_outside_cov,
        "delta_ch_surv":      delta_ch_surv,
    }
    row.update(metrics_pu)
    row.update(metrics_pp)

    # ─── GT ANCHOR (optional) ────────────────────────────────
    if include_gt_anchor:
        rng_shared_gt = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
        all_idxs_gt, all_xis_gt = generate_base_samples(BUDGET, D, rng_shared_gt, n_r_before)
        ch_gt    = apply_samples(all_idxs_gt, all_xis_gt, X_region, delta_gt_full, 0.0, xl, xu)
        F_ch_gt  = cp_evaluate(ch_gt, t_new, region_dynamics)

        X_pool_g = np.vstack([X_pop, ch_gt])
        F_pool_g = np.vstack([F_before, F_ch_gt])
        X_surv_gu, F_surv_gu, surv_idx_gu = nsga2_survival_traced(
            X_pool_g, F_pool_g, POP_SIZE, xl, xu, surv_seed
        )
        metrics_gu = _coverage_metrics(F_surv_gu, "gt_unr")

        X_surv_gp, F_surv_gp, final_idx_gp = region_protected_survival(
            X_pop, F_before, ch_gt, F_ch_gt,
            r, region_ids, n_r_before,
            xl, xu, surv_seed,
        )
        metrics_gp = _coverage_metrics(F_surv_gp, "gt_prot")

        row["gt_delta_delta_igd"] = metrics_gp["gt_prot_delta_igd"] - metrics_gu["gt_unr_delta_igd"]
        row.update(metrics_gu)
        row.update(metrics_gp)

    return [row]


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_r1p(seed, n_scored_changes, scenario_name, include_gt_anchor=True):
    """Run R1-P comparison for one seed. Returns list of row dicts."""
    xl, xu = get_bounds(D)
    V  = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)
    n_bins = R
    region_dynamics = make_scenario_dynamics(scenario_name, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    history = RegionHistory(n_regions=R)
    rows_out = []
    gen = 0
    n_total = 1 + n_scored_changes
    history_t = -1  # integer index for history update; incremented before each update

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        F_plus  = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm    = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))

        # Update history with current population state BEFORE responding
        history_t += 1
        history.update(t=history_t, X=X_pop, F=F_minus,
                       region_ids=region_ids, xl=xl, xu=xu)

        if change_idx == 1:
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE, seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current

        for r in range(R):
            r_rows = run_region_r1p(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
                region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
                history, history_t,
                include_gt_anchor=include_gt_anchor,
            )
            for row in r_rows:
                if row.get("has_history") is False:
                    rows_out.append({
                        "has_history": False,
                        "scenario":  scenario_name,
                        "seed":      seed,
                        "event_id":  change_idx,
                        "region_id": r,
                    })
                    continue
                row.update({
                    "scenario":   scenario_name,
                    "seed":       seed,
                    "event_id":   change_idx,
                    "change_idx": change_idx,
                    "region_id":  r,
                    "t_prev":     float(t_prev),
                    "t_new":      float(t_new),
                    "history_t":  history_t,
                })
                rows_out.append(row)

        t_current = t_new
        X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                             GENERATIONS_PER_CHANGE, seed * 10000 + change_idx, xl, xu)

    return rows_out


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
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


if __name__ == "__main__":
    import csv

    N_SEEDS  = 10
    N_SCORED = 5
    SCENARIO = "PERSISTENT"
    OUT_DIR  = Path("results/R1P_pop_transfer")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "R1P_pop_transfer.jsonl"

    all_rows = []
    n_valid = 0
    n_missing = 0
    t0 = time.time()
    for seed in range(1, N_SEEDS + 1):
        t1 = time.time()
        rows = run_one_r1p(seed, N_SCORED, SCENARIO, include_gt_anchor=True)
        all_rows.extend(rows)
        seed_valid   = sum(1 for r in rows if r.get("has_history", True) is True)
        seed_missing = sum(1 for r in rows if r.get("has_history") is False)
        n_valid   += seed_valid
        n_missing += seed_missing
        print(f"{SCENARIO} seed={seed}: {seed_valid} valid, {seed_missing} skipped "
              f"in {time.time()-t1:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | {n_valid} valid rows, {n_missing} skipped "
          f"| Wrote {len(all_rows)} rows to {jsonl_path}")
