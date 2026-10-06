"""
R1 — Region-Protected Response (Proof-of-Mechanism Baseline).

Hypothesis: by protecting OLD individuals outside the response region r from
being displaced during GT injection, we prevent outside-region PF coverage
loss and improve (or neutralize) net IGD.

MECHANISM:
  UNRESTRICTED: normal NSGA-II on full merged pool OLD + CHILD_r → POP_SIZE.
  REGION_PROTECTED:
    1. ALL old individuals outside r are protected (always survive).
    2. Global NSGA-II ranks+crowding are computed on full merged pool.
    3. Exactly n_r_before slots for the r-region are filled by selecting
       the top-ranked candidates from OLD_r ∪ CHILD_r using global rank then
       global crowding as tie-break.
    4. Final population = OLD_outside ∪ selected_r_candidates.

CONSTRAINTS (hard):
  - Production survival (nsga2_survival) is NOT modified.
  - BUDGET = 10 (production), sigma_P unchanged, predictor unchanged.
  - PREDICT_GT is diagnostic oracle only, never goes to production.
  - Do NOT change: BUDGET, sigma_P, RegionHistory, RL start.
  - R1 is a proof-of-mechanism baseline — not a production algorithm.

Primary scenario: GT sigma=0, PERSISTENT, B=10.
Secondary scenario (if primary promising): POP sigma=0.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.E2P5_coverage_crowding_audit as e2p5

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import nsga2_survival, split_by_region

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    distance_to_region_ps, bin_of,
)

# ============================================================
# Config — production locked
# ============================================================
SIGMA_GT_R1  = 0.0
BUDGET       = e2p3.BUDGET          # 10 (production, never changed)
POP_SIZE     = e2p3.POP_SIZE        # 100
D            = e2p3.D               # 10
N_OBJ        = e2p3.N_OBJ           # 2
N_PARTITIONS = e2p3.N_PARTITIONS    # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

compute_time        = e2p3.compute_time
evolve_nsga2        = e2p3.evolve_nsga2
make_rng            = e2p3.make_rng
make_scenario_dynamics = e2p3.make_scenario_dynamics

median_nearest_pf_distance = e2p3.median_nearest_pf_distance
nsga2_survival_traced      = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed     = e2p3._e2p3_common_surv_seed
generate_base_samples      = e2p3.generate_base_samples
apply_samples              = e2p3.apply_samples

compute_ranks_crowding_all   = e2p5.compute_ranks_crowding_all
compute_pf_coverage_per_point = e2p5.compute_pf_coverage_per_point

# D2-A: import canonical implementation from core.
# R1 diagnostic wrapper points to the shared helper so there is only ONE
# implementation, tested by T27 in test_d2_production_path_equivalence.py.
from prdrl.core.survival import region_protected_survival  # noqa: F401


# ============================================================
# Per-region R1 comparison for one scored event
# ============================================================

def run_region_r1(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
):
    """Compare UNRESTRICTED vs REGION_PROTECTED for region r at production B=10.

    Returns a list with one row dict (both methods measured on identical children).
    Returns empty list if region is empty.
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

    # GT oracle delta (identical to E2P5)
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # Generate B=10 children (same as production budget)
    B = BUDGET
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = generate_base_samples(B, D, rng_shared, n_r_before)
    ch_gt = apply_samples(all_idxs, all_xis, X_region, delta_gt_full, 0.0, xl, xu)
    F_ch  = cp_evaluate(ch_gt, t_new, region_dynamics)

    # Common survival seed (same as E2P5 — ensures identical to unrestricted in E2P5)
    surv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    # Normalization basis for PF/occupancy metrics
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

    def _coverage_metrics(F_surv, method_name):
        """Compute per-PF-point coverage metrics for a given survived population."""
        igd_after  = float(igd(F_surv, pf_star))
        delta_igd  = igd_before - igd_after

        # PF per-point
        _, _, delta_d_arr, rep_rids = compute_pf_coverage_per_point(
            F_before, F_surv, pf_star, region_ids
        )

        # Direct region assignment
        inside_d  = pf_ref_rids == r
        outside_d = ~inside_d
        mean_di = float(np.mean(delta_d_arr[inside_d]))  if np.any(inside_d)  else float("nan")
        mean_do = float(np.mean(delta_d_arr[outside_d])) if np.any(outside_d) else float("nan")
        fw_i = float(np.mean(delta_d_arr[inside_d]  > 1e-14)) if np.any(inside_d)  else float("nan")
        fw_o = float(np.mean(delta_d_arr[outside_d] > 1e-14)) if np.any(outside_d) else float("nan")

        # Actual occupancy
        F_surv_norm = (F_surv - z_m) / norm_span
        actual_rids_after = np.asarray(partition(F_surv_norm, V), dtype=int)
        actual_occ = np.bincount(actual_rids_after, minlength=R)
        actual_resp_delta = int(actual_occ[r]) - int(occ_before[r])
        actual_d_other = actual_occ - occ_before
        actual_d_other[r] = 0
        n_other_lose = int(np.sum(actual_d_other < 0))
        loss_other   = int(np.sum(np.abs(actual_d_other[actual_d_other < 0])))

        return {
            f"{method_name}_igd_after":         igd_after,
            f"{method_name}_delta_igd":          delta_igd,
            f"{method_name}_mean_delta_pf_inside":  mean_di,
            f"{method_name}_mean_delta_pf_outside": mean_do,
            f"{method_name}_frac_worse_inside":     fw_i,
            f"{method_name}_frac_worse_outside":    fw_o,
            f"{method_name}_actual_resp_delta":       actual_resp_delta,
            f"{method_name}_actual_n_other_losing":   n_other_lose,
            f"{method_name}_actual_total_loss_other": loss_other,
        }

    # ─── UNRESTRICTED survival ────────────────────────────────
    X_pool = np.vstack([X_pop, ch_gt])
    F_pool = np.vstack([F_before, F_ch])
    X_surv_u, F_surv_u, surv_idx_u = nsga2_survival_traced(
        X_pool, F_pool, POP_SIZE, xl, xu, surv_seed
    )
    surv_old_u = surv_idx_u[surv_idx_u < N_total_old]
    dropped_old_u = set(range(N_total_old)) - set(surv_old_u.tolist())
    n_old_dropped_u      = len(dropped_old_u)
    n_ch_survived_u      = int(np.sum(surv_idx_u >= N_total_old))
    dropped_other_u      = sum(1 for i in dropped_old_u if region_ids[i] != r)
    metrics_u = _coverage_metrics(F_surv_u, "unr")

    # ─── REGION-PROTECTED survival ────────────────────────────
    X_surv_p, F_surv_p, final_idx_p = region_protected_survival(
        X_pop, F_before,
        ch_gt, F_ch,
        r, region_ids, n_r_before,
        xl, xu, surv_seed,
    )
    # Which OLD individuals were dropped under protected?
    surv_old_p_set = set(int(i) for i in final_idx_p if i < N_total_old)
    dropped_old_p  = set(range(N_total_old)) - surv_old_p_set
    n_old_dropped_p      = len(dropped_old_p)
    n_ch_survived_p      = int(np.sum(final_idx_p >= N_total_old))
    dropped_other_p      = sum(1 for i in dropped_old_p if region_ids[i] != r)
    metrics_p = _coverage_metrics(F_surv_p, "prot")

    # Primary comparison
    delta_delta_igd = metrics_p["prot_delta_igd"] - metrics_u["unr_delta_igd"]
    igd_gain_outside = metrics_u["unr_mean_delta_pf_outside"] - metrics_p["prot_mean_delta_pf_outside"]

    row = {
        "budget_b": B,
        "region_type": region_type,
        "dominant_bin": dominant_bin,
        "n_r_before": n_r_before,
        "igd_before": igd_before,
        # Unrestricted
        "unr_n_old_dropped": n_old_dropped_u,
        "unr_n_old_dropped_other": dropped_other_u,
        "unr_n_ch_survived": n_ch_survived_u,
        # Protected
        "prot_n_old_dropped": n_old_dropped_p,
        "prot_n_old_dropped_other": dropped_other_p,
        "prot_n_ch_survived": n_ch_survived_p,
        # Primary comparison
        "delta_delta_igd": delta_delta_igd,
        "igd_gain_outside": igd_gain_outside,
        "fe": B,
    }
    row.update(metrics_u)
    row.update(metrics_p)
    return [row]


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_r1(seed, n_scored_changes, scenario_name):
    """Run R1 comparison for one seed. Returns list of row dicts."""
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

    rows_out = []
    gen = 0
    n_total = 1 + n_scored_changes

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        F_plus  = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm    = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))

        if change_idx == 1:
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE, seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current

        for r in range(R):
            r_rows = run_region_r1(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
                region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
            )
            for row in r_rows:
                row.update({
                    "scenario":   scenario_name,
                    "seed":       seed,
                    "event_id":   change_idx,
                    "change_idx": change_idx,
                    "region_id":  r,
                    "t_prev":     float(t_prev),
                    "t_new":      float(t_new),
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
    OUT_DIR  = Path("results/R1_region_protected_response")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "R1_region_protected_response.jsonl"

    all_rows = []
    t0 = time.time()
    for seed in range(1, N_SEEDS + 1):
        t1 = time.time()
        rows = run_one_r1(seed, N_SCORED, SCENARIO)
        all_rows.extend(rows)
        print(f"{SCENARIO} seed={seed}: {len(rows)} rows in {time.time()-t1:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | Wrote {len(all_rows)} rows to {jsonl_path}")
