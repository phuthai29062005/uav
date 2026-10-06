"""
E2-P4 -- HEADROOM vs POPULATION DILUTION AUDIT.

Scientific question:
  Why can oracle-quality GT children survive yet global IGD improve only slightly?

Two competing hypotheses:
  H_HEADROOM:  baseline population is already so close to PF that little IGD
               improvement is possible per event.
  H_DILUTION:  only a small fraction of the population (BUDGET/POP_SIZE = 10%)
               can be replaced per region response, limiting global IGD impact.

Sub-experiment A -- BUDGET SWEEP (fixed warmup, vary B):
  Use GT sigma=0 at each B in B_VALUES = [5, 10, 20, 50, 100].
  Nested samples: first B samples of B=100 are the same as B alone.
  All B branches share the same baseline population (no mutation).

Sub-experiment B -- HEADROOM SWEEP (fixed B=BUDGET, vary warmup):
  Use GT sigma=0. Vary pre-scored-event warmup in WARMUP_LEVELS = [0, 5, 15, 30].
  All warmup branches start from the SAME X_init (same seed).
  Different warmup → different convergence → different IGD_before.
  Same dynamic schedule (t_prev2, t_prev, t_new identical for same event index).

GT sigma=0 is a diagnostic oracle — NEVER goes to production.
Do NOT change: production BUDGET, sigma_P, NSGA-II, RegionHistory, E2a/E2a-R.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3

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

import prdrl.experiments.E2P2_predict_noise_sweep as e2p2


# ============================================================
# Config — production locked
# ============================================================
SIGMA_LOCAL  = e2p3.SIGMA_LOCAL     # 0.02
SIGMA_GT_E4  = 0.0                  # diagnostic oracle, sigma=0
BUDGET       = e2p3.BUDGET          # 10 (production, never changed)
POP_SIZE     = e2p3.POP_SIZE        # 100
D            = e2p3.D               # 10
N_OBJ        = e2p3.N_OBJ           # 2
N_PARTITIONS = e2p3.N_PARTITIONS    # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS      # 30
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE  # 10
LAMBDA_PRED  = e2p3.LAMBDA_PRED     # 1.0

B_MAX    = 100
B_VALUES = [5, 10, 20, 50, 100]
WARMUP_LEVELS = [0, 5, 15, 30]

compute_time = e2p3.compute_time
evolve_nsga2 = e2p3.evolve_nsga2
make_rng     = e2p3.make_rng
make_scenario_dynamics = e2p3.make_scenario_dynamics

# Re-use corrected helpers from E2P3
median_nearest_pf_distance = e2p3.median_nearest_pf_distance
assign_ranks               = e2p3.assign_ranks
nsga2_survival_traced      = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed     = e2p3._e2p3_common_surv_seed
generate_base_samples      = e2p3.generate_base_samples
apply_samples              = e2p3.apply_samples
make_local_children        = e2p3.make_local_children


# ============================================================
# Budget-sweep per-region helper
# ============================================================

def run_region_budget_sweep(
    r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
    prev_regions, region_dynamics, n_bins, pf_star,
    xl, xu, base_seed, change_idx,
):
    """Run GT sigma=0 at each B in B_VALUES for one region/event.

    Nested samples: all_idxs[:B] for each B.
    All B branches share same X_pop (never mutated).
    Returns list of row dicts, one per B value (+1 for LOCAL reference).
    """
    mask  = region_ids == r
    n_old = int(np.sum(mask))
    if n_old == 0:
        return []

    X_region = X_pop[mask].copy()   # copy to ensure no mutation

    # Region metadata
    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type = str(labels_r[np.argmax(counts_r)])
    bins_in_region = bin_of(X_region[:, 0], n_bins)
    bin_vals, bin_counts = np.unique(bins_in_region, return_counts=True)
    dominant_bin = int(bin_vals[np.argmax(bin_counts)])

    # True PS centers for GT oracle
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_dist)
    delta_response_true_full = np.concatenate(
        [[0.0], mu_star_current - mu_star_prev]
    )

    # POP history check (needed for has_history but GT is oracle)
    X_prev2_region = prev_regions.get(r)
    has_history = X_prev2_region is not None and len(X_prev2_region) > 0
    if not has_history:
        return []  # skip first change (no t-2 snapshot)

    # Generate B_MAX nested base samples from shared RNG
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = generate_base_samples(B_MAX, D, rng_shared, n_old)

    igd_before = float(igd(F_plus, pf_star))
    N_total_old = len(X_pop)

    # Common survival seed (same for all B values for this event/region)
    common_sv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    rows = []

    # ---------- GT budget sweep ----------
    for B in B_VALUES:
        b_idxs = all_idxs[:B]
        b_xis  = all_xis[:B]

        # Unique parent diagnostics
        n_unique_parents = len(set(b_idxs))
        unique_parent_frac = n_unique_parents / B

        ch_gt = apply_samples(b_idxs, b_xis, X_region, delta_response_true_full,
                              SIGMA_GT_E4, xl, xu)
        F_ch_gt = cp_evaluate(ch_gt, t_new, region_dynamics)

        n_ch = len(ch_gt)
        ch_unique = np.unique(ch_gt, axis=0)
        n_unique_ch = len(ch_unique)
        unique_ch_frac = n_unique_ch / n_ch if n_ch > 0 else float("nan")

        # Stage 1: PS distance
        ps_dists = distance_to_region_ps(ch_gt, region_dynamics, dominant_bin, t_new)
        ps_med   = float(np.median(ps_dists))

        # Stage 2: PF distance
        pf_med, _, _ = median_nearest_pf_distance(F_ch_gt, pf_star)

        # Stage 3: ranks in merged pool
        X_pool_full = np.vstack([X_pop, ch_gt])
        F_pool_full = np.vstack([F_plus, F_ch_gt])
        ranks_full  = assign_ranks(F_pool_full)
        child_ranks = ranks_full[N_total_old:]
        frac_rank0  = float(np.mean(child_ranks == 0))

        # Stage 4: survival with provenance
        X_surv, F_surv, surv_orig_idx = nsga2_survival_traced(
            X_pool_full, F_pool_full, POP_SIZE, xl, xu, common_sv_seed
        )
        survived_child_mask = surv_orig_idx >= N_total_old
        n_ch_survived = int(np.sum(survived_child_mask))
        surv_rate     = n_ch_survived / n_ch if n_ch > 0 else float("nan")
        pop_repl_frac = n_ch_survived / POP_SIZE

        if n_ch_survived > 0:
            surv_idx = surv_orig_idx[survived_child_mask]
            surv_pf_med, _, _ = median_nearest_pf_distance(
                F_pool_full[surv_idx], pf_star
            )
        else:
            surv_pf_med = float("nan")

        igd_after = float(igd(F_surv, pf_star))
        delta_igd = igd_before - igd_after
        rel_igd   = delta_igd / (abs(igd_before) + 1e-12)

        rows.append({
            "subexp": "budget",
            "mode": "GT",
            "budget_b": B,
            "warmup_generations": WARMUP_GENERATIONS,
            "region_type": region_type,
            "dominant_bin": dominant_bin,
            "n_region_parents": n_old,
            "n_unique_parent_indices": n_unique_parents,
            "unique_parent_fraction": unique_parent_frac,
            "n_unique_children": n_unique_ch,
            "unique_child_fraction": unique_ch_frac,
            "child_ps_dist_median": ps_med,
            "child_pf_dist_median": pf_med,
            "frac_children_rank0": frac_rank0,
            "n_children_generated": n_ch,
            "n_children_survived": n_ch_survived,
            "child_survival_rate": surv_rate,
            "population_replacement_fraction": pop_repl_frac,
            "surviving_child_pf_dist_median": surv_pf_med,
            "igd_before": igd_before,
            "igd_after": igd_after,
            "delta_igd": delta_igd,
            "relative_igd_improvement": rel_igd,
            "fe": n_ch,
        })

    # ---------- LOCAL reference at production BUDGET ----------
    ch_local = make_local_children(X_region, xl, xu, base_seed, change_idx, r)
    F_ch_local = cp_evaluate(ch_local, t_new, region_dynamics)
    n_ch_loc = len(ch_local)

    X_pool_loc = np.vstack([X_pop, ch_local])
    F_pool_loc = np.vstack([F_plus, F_ch_local])
    ranks_loc  = assign_ranks(F_pool_loc)
    frac_rank0_loc = float(np.mean(ranks_loc[N_total_old:] == 0))

    X_surv_loc, F_surv_loc, surv_idx_loc = nsga2_survival_traced(
        X_pool_loc, F_pool_loc, POP_SIZE, xl, xu, common_sv_seed
    )
    n_surv_loc = int(np.sum(surv_idx_loc >= N_total_old))
    igd_after_loc = float(igd(F_surv_loc, pf_star))
    delta_igd_loc = igd_before - igd_after_loc

    rows.append({
        "subexp": "budget",
        "mode": "LOCAL",
        "budget_b": BUDGET,
        "warmup_generations": WARMUP_GENERATIONS,
        "region_type": region_type,
        "dominant_bin": dominant_bin,
        "n_region_parents": n_old,
        "n_unique_parent_indices": None,
        "unique_parent_fraction": float("nan"),
        "n_unique_children": int(len(np.unique(ch_local, axis=0))),
        "unique_child_fraction": float(len(np.unique(ch_local, axis=0))) / n_ch_loc,
        "child_ps_dist_median": float(np.median(
            distance_to_region_ps(ch_local, region_dynamics, dominant_bin, t_new))),
        "child_pf_dist_median": float(median_nearest_pf_distance(F_ch_local, pf_star)[0]),
        "frac_children_rank0": frac_rank0_loc,
        "n_children_generated": n_ch_loc,
        "n_children_survived": n_surv_loc,
        "child_survival_rate": n_surv_loc / n_ch_loc if n_ch_loc > 0 else float("nan"),
        "population_replacement_fraction": n_surv_loc / POP_SIZE,
        "surviving_child_pf_dist_median": float("nan"),
        "igd_before": igd_before,
        "igd_after": igd_after_loc,
        "delta_igd": delta_igd_loc,
        "relative_igd_improvement": delta_igd_loc / (abs(igd_before) + 1e-12),
        "fe": n_ch_loc,
    })

    return rows


# ============================================================
# Headroom-sweep per-region helper
# ============================================================

def run_region_headroom_sweep(
    r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
    prev_regions, region_dynamics, n_bins, pf_star,
    xl, xu, base_seed, change_idx, warmup_gen,
):
    """Run GT sigma=0 at B=BUDGET for one region/event at a given warmup level.

    Returns a single row dict.
    """
    mask  = region_ids == r
    n_old = int(np.sum(mask))
    if n_old == 0:
        return None

    X_region = X_pop[mask].copy()

    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type = str(labels_r[np.argmax(counts_r)])
    bins_in_region = bin_of(X_region[:, 0], n_bins)
    bin_vals, bin_counts = np.unique(bins_in_region, return_counts=True)
    dominant_bin = int(bin_vals[np.argmax(bin_counts)])

    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_dist)
    delta_response_true_full = np.concatenate(
        [[0.0], mu_star_current - mu_star_prev]
    )

    X_prev2_region = prev_regions.get(r)
    has_history = X_prev2_region is not None and len(X_prev2_region) > 0
    if not has_history:
        return None

    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    idxs, xis  = generate_base_samples(BUDGET, D, rng_shared, n_old)

    ch_gt   = apply_samples(idxs, xis, X_region, delta_response_true_full,
                            SIGMA_GT_E4, xl, xu)
    F_ch_gt = cp_evaluate(ch_gt, t_new, region_dynamics)

    n_ch = len(ch_gt)
    igd_before = float(igd(F_plus, pf_star))

    N_total_old = len(X_pop)
    X_pool_full = np.vstack([X_pop, ch_gt])
    F_pool_full = np.vstack([F_plus, F_ch_gt])

    common_sv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)
    X_surv, F_surv, surv_orig_idx = nsga2_survival_traced(
        X_pool_full, F_pool_full, POP_SIZE, xl, xu, common_sv_seed
    )

    survived_child_mask = surv_orig_idx >= N_total_old
    n_ch_survived = int(np.sum(survived_child_mask))
    surv_rate     = n_ch_survived / n_ch if n_ch > 0 else float("nan")
    pop_repl_frac = n_ch_survived / POP_SIZE

    if n_ch_survived > 0:
        surv_pf_med, _, _ = median_nearest_pf_distance(
            F_pool_full[surv_orig_idx[survived_child_mask]], pf_star
        )
    else:
        surv_pf_med = float("nan")

    igd_after = float(igd(F_surv, pf_star))
    delta_igd = igd_before - igd_after
    rel_igd   = delta_igd / (abs(igd_before) + 1e-12)

    ps_dists = distance_to_region_ps(ch_gt, region_dynamics, dominant_bin, t_new)
    pf_med, _, _ = median_nearest_pf_distance(F_ch_gt, pf_star)

    return {
        "subexp": "headroom",
        "mode": "GT",
        "budget_b": BUDGET,
        "warmup_generations": warmup_gen,
        "region_type": region_type,
        "dominant_bin": dominant_bin,
        "n_region_parents": n_old,
        "child_ps_dist_median": float(np.median(ps_dists)),
        "child_pf_dist_median": float(pf_med),
        "n_children_generated": n_ch,
        "n_children_survived": n_ch_survived,
        "child_survival_rate": surv_rate,
        "population_replacement_fraction": pop_repl_frac,
        "surviving_child_pf_dist_median": surv_pf_med,
        "igd_before": igd_before,
        "igd_after": igd_after,
        "delta_igd": delta_igd,
        "relative_igd_improvement": rel_igd,
        "fe": n_ch,
    }


# ============================================================
# Budget-sweep run for one (seed, scenario)
# ============================================================

def run_one_budget_sweep(seed, n_scored_changes, scenario_name):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)
    n_bins = R
    region_dynamics = make_scenario_dynamics(scenario_name, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_init = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t0 = compute_time(0)
    X_pop = evolve_nsga2(X_init.copy(), t0, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    prev_regions = None
    t_prev2 = None
    gen = 0
    n_total = 1 + n_scored_changes
    rows_out = []

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, compute_time(gen - GENERATIONS_PER_CHANGE
                              if gen > 0 else 0), region_dynamics)
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        t_current_prev = compute_time(gen - GENERATIONS_PER_CHANGE)
        F_plus = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))
        cur_regions = split_by_region(X_pop, region_ids, R)

        if change_idx == 1:
            prev_regions = cur_regions
            t_prev2 = t_current_prev
            X_pop = evolve_nsga2(X_pop, t_new, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current_prev

        for r in range(R):
            r_rows = run_region_budget_sweep(
                r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
                prev_regions, region_dynamics, n_bins, pf_star,
                xl, xu, seed, change_idx,
            )
            for row in r_rows:
                row.update({
                    "scenario": scenario_name,
                    "seed": seed,
                    "event_id": change_idx,
                    "region_id": r,
                    "t_prev2": float(t_prev2),
                    "t_prev": float(t_prev),
                    "t_new": float(t_new),
                })
                rows_out.append(row)

        prev_regions = cur_regions
        t_prev2 = t_prev
        X_pop = evolve_nsga2(X_pop, t_new, region_dynamics,
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows_out


# ============================================================
# Headroom-sweep run for one (seed, scenario, warmup)
# ============================================================

def run_one_headroom_sweep(seed, n_scored_changes, scenario_name, warmup_gen):
    """Run headroom sweep for one warmup level.

    X_init is derived purely from seed; same across all warmup_gen values.
    Warmup evolution uses seed*10000+warmup_gen as evolution seed.
    Dynamic schedule (t values) is identical for all warmup_gen at same event.
    """
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)
    n_bins = R
    region_dynamics = make_scenario_dynamics(scenario_name, n_bins)
    pf_star = get_true_pf(n_points=500)

    # X_init: SAME for all warmup levels (same seed -> same X_init)
    rng = np.random.default_rng(seed)
    X_init = rng.uniform(xl, xu, size=(POP_SIZE, D))

    t0 = compute_time(0)

    if warmup_gen > 0:
        X_pop = evolve_nsga2(X_init.copy(), t0, region_dynamics,
                             warmup_gen, seed * 10000 + warmup_gen, xl, xu)
    else:
        X_pop = X_init.copy()

    prev_regions = None
    t_prev2 = None
    gen = 0
    n_total = 1 + n_scored_changes
    rows_out = []

    for change_idx in range(1, n_total + 1):
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        t_current_prev = compute_time(gen - GENERATIONS_PER_CHANGE)
        F_minus = cp_evaluate(X_pop, t_current_prev, region_dynamics)
        F_plus  = cp_evaluate(X_pop, t_new,          region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm = np.asarray(F_tm)
        region_ids  = np.asarray(partition(F_tm, V))
        cur_regions = split_by_region(X_pop, region_ids, R)

        if change_idx == 1:
            prev_regions = cur_regions
            t_prev2 = t_current_prev
            X_pop = evolve_nsga2(X_pop, t_new, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current_prev

        for r in range(R):
            row = run_region_headroom_sweep(
                r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
                prev_regions, region_dynamics, n_bins, pf_star,
                xl, xu, seed, change_idx, warmup_gen,
            )
            if row is not None:
                row.update({
                    "scenario": scenario_name,
                    "seed": seed,
                    "event_id": change_idx,
                    "region_id": r,
                    "t_prev2": float(t_prev2),
                    "t_prev": float(t_prev),
                    "t_new": float(t_new),
                })
                rows_out.append(row)

        prev_regions = cur_regions
        t_prev2 = t_prev
        X_pop = evolve_nsga2(X_pop, t_new, region_dynamics,
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

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
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: 2 seeds x 2 events")
    args = parser.parse_args()

    if args.smoke:
        N_SEEDS  = 2
        N_SCORED = 2
    else:
        N_SEEDS  = 10
        N_SCORED = 5

    SCENARIOS = ["PERSISTENT", "STATIONARY"]
    OUT_DIR = Path("results/E2P4_headroom_dilution")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Sub-experiment A: Budget sweep ----
    budget_rows = []
    t0_b = time.time()
    print("=== BUDGET SWEEP ===")
    for sc in SCENARIOS:
        for seed in range(1, N_SEEDS + 1):
            t1 = time.time()
            rows = run_one_budget_sweep(seed, N_SCORED, sc)
            budget_rows.extend(rows)
            print(f"  {sc} seed={seed}: {len(rows)} rows in {time.time()-t1:.1f}s")
    budget_path = OUT_DIR / "E2P4_budget_sweep.jsonl"
    with open(budget_path, "w") as f:
        for row in budget_rows:
            f.write(json.dumps(json_safe(row)) + "\n")
    print(f"Budget sweep: {len(budget_rows)} rows in {time.time()-t0_b:.1f}s -> {budget_path}\n")

    # ---- Sub-experiment B: Headroom sweep ----
    headroom_rows = []
    t0_h = time.time()
    print("=== HEADROOM SWEEP ===")
    for sc in SCENARIOS:
        for seed in range(1, N_SEEDS + 1):
            for W in WARMUP_LEVELS:
                t1 = time.time()
                rows = run_one_headroom_sweep(seed, N_SCORED, sc, W)
                headroom_rows.extend(rows)
                print(f"  {sc} seed={seed} W={W}: {len(rows)} rows in {time.time()-t1:.1f}s")
    headroom_path = OUT_DIR / "E2P4_headroom_sweep.jsonl"
    with open(headroom_path, "w") as f:
        for row in headroom_rows:
            f.write(json.dumps(json_safe(row)) + "\n")
    print(f"Headroom sweep: {len(headroom_rows)} rows in {time.time()-t0_h:.1f}s -> {headroom_path}\n")

    print(f"DONE. Budget: {len(budget_rows)} rows, Headroom: {len(headroom_rows)} rows")
