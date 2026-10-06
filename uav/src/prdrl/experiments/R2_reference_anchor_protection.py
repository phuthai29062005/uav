"""
R2 — Reference Anchor Soft Protection.

Scientific question:
    Can protecting only ONE representative per occupied non-response region
    preserve most of the global PF coverage benefit of R1 hard protection,
    while allowing more good children to survive and avoiding the rigid
    inside-region cost of hard protection?

THREE BRANCHES per event-region (identical children, identical FE):
    A. UNRESTRICTED    — normal NSGA-II on [OLD; CHILD_r]
    B. HARD_PROTECTED  — exact R1 region_protected_survival
    C. SOFT_ANCHOR     — R2: one anchor per occupied non-response region

R2 mechanism:
    For each occupied region q != r, protect exactly ONE OLD individual
    chosen by: best merged-pool Pareto rank → highest crowding → smallest pool_idx.
    Remaining POP_SIZE - n_anchors slots filled by global NSGA-II priority
    from all non-anchor candidates.

RUNTIME INFORMATION USED IN R2 SURVIVAL:
    - X, F (merged pool)
    - Pre-response region assignments (old_region_ids)
    - Fixed reference directions V (for region partitioning at evaluation time)
    - Pareto rank and crowding on merged pool
    - Deterministic pool_idx tie-breaker

EXPLICITLY NOT USED IN R2 SURVIVAL:
    - True PF*
    - GT motion
    - mu_star
    - True coverage responsibility
    - Future information

CONSTRAINTS (hard):
    - nsga2_survival NOT modified.
    - region_protected_survival (R1) NOT modified.
    - Production BUDGET=10, sigma_P=0.03, lambda_pred=1.0 unchanged.
    - POP predictor: same path as R1-P (RegionHistory → get_predict_motion_from_history).
    - Missing history → skip (has_history=False), same as R1-P.
    - sigma_POP=0 (isolates allocation from noise; sigma=.03 in separate R2-N).
    - R2 is experimental only — never enters production in this task.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.R1_region_protected_response as r1
import prdrl.experiments.R1P_pop_transfer as r1p

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

# Re-use helpers from existing modules
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

estimator_diagnostics = r1p.estimator_diagnostics


# ============================================================
# R2 Core: NSGA-II priority selection from precomputed ranks
# ============================================================

def _nsga2_priority_selection(ranks, crowding, candidates, n_select):
    """Select n_select individuals from candidates by NSGA-II front priority.

    Processes complete fronts first. For the final partial front, selects by
    crowding DESC then pool_idx ASC (deterministic; no random tie-breaking).
    Uses precomputed ranks and crowding — no recomputation.

    Args:
        ranks:      (N_pool,) int array of Pareto ranks (0=best)
        crowding:   (N_pool,) float array of crowding distances
        candidates: list of pool indices to select from
        n_select:   number to select

    Returns:
        list of selected pool indices (unsorted)
    """
    if n_select <= 0:
        return []
    if n_select >= len(candidates):
        return list(candidates)

    # Group candidates by front rank
    rank_groups = defaultdict(list)
    for idx in candidates:
        rank_groups[int(ranks[idx])].append(idx)

    selected = []
    for rk in sorted(rank_groups.keys()):
        members = rank_groups[rk]
        remaining_needed = n_select - len(selected)
        if remaining_needed <= 0:
            break
        if len(members) <= remaining_needed:
            # Entire front fits
            selected.extend(members)
        else:
            # Partial front: sort by crowding DESC (inf first), then pool_idx ASC
            def _sort_key(i):
                cd = crowding[i]
                cd_sort = -cd if np.isfinite(cd) else -(1e15)
                return (cd_sort, i)
            front_sorted = sorted(members, key=_sort_key)
            selected.extend(front_sorted[:remaining_needed])
            break

    return selected


def select_anchors(old_region_ids, response_region, N_old, ranks, crowding):
    """Select one anchor per occupied non-response region.

    For each occupied q != response_region, pick the OLD individual with:
        1. Lowest Pareto rank in the merged pool
        2. Highest crowding distance (tie-break)
        3. Smallest pool_idx (tie-break; deterministic)

    Args:
        old_region_ids:    (N_old,) region assignments for OLD individuals
        response_region:   int, the region r being responded to
        N_old:             number of OLD individuals (pool indices 0..N_old-1)
        ranks:             (N_pool,) Pareto ranks for the merged pool
        crowding:          (N_pool,) crowding distances for the merged pool

    Returns:
        anchor_pool_indices: list of pool indices (one per occupied non-response region)
        n_occupied_regions_before: total number of distinct occupied regions
        n_anchor_regions: number of non-response occupied regions
    """
    r = response_region

    # Group OLD indices by region
    region_to_old = defaultdict(list)
    for i in range(N_old):
        q = int(old_region_ids[i])
        region_to_old[q].append(i)

    occupied_regions = set(region_to_old.keys())
    n_occupied_regions_before = len(occupied_regions)
    non_response_regions = occupied_regions - {r}
    n_anchor_regions = len(non_response_regions)

    anchor_pool_indices = []
    for q in sorted(non_response_regions):
        old_in_q = region_to_old[q]

        def _anchor_key(i):
            cd = crowding[i]
            cd_sort = -cd if np.isfinite(cd) else -(1e15)
            return (int(ranks[i]), cd_sort, i)

        best = min(old_in_q, key=_anchor_key)
        anchor_pool_indices.append(best)

    return anchor_pool_indices, n_occupied_regions_before, n_anchor_regions


def reference_anchor_survival(
    X_pool,
    F_pool,
    old_region_ids,
    response_region,
    N_old,
    n_survive,
    xl,
    xu,
    seed=None,
):
    """R2 soft-anchor survival: protect one anchor per non-response region.

    STEP 1: Compute global Pareto rank + crowding on merged pool [OLD; CHILD_r].
    STEP 2: For each occupied q != r, select one OLD anchor
            (best rank → highest crowding → smallest pool_idx).
    STEP 3: All anchors are guaranteed survivors.
    STEP 4: Remaining n_survive - n_anchors slots filled from non-anchor
            candidates using global NSGA-II priority (complete fronts first,
            partial front sorted by crowding DESC then pool_idx ASC).
    STEP 5: Return exactly n_survive survivors.

    Runtime information used: X, F, old_region_ids, Pareto rank, crowding.
    NOT used: true PF, GT motion, oracle coverage responsibility.

    Args:
        X_pool:         (N_old + B, D) merged pool [OLD; CHILD_r]
        F_pool:         (N_old + B, M) merged objectives
        old_region_ids: (N_old,) region assignments for OLD individuals
        response_region: int, region r being responded to
        N_old:          number of OLD individuals (pool indices 0..N_old-1)
        n_survive:      target population size (POP_SIZE)
        xl, xu:         bounds (not used in selection; kept for API consistency)
        seed:           optional int (not used in current implementation;
                        tie-breaking is deterministic via pool_idx)

    Returns:
        X_surv:               (n_survive, D)
        F_surv:               (n_survive, M)
        survivor_pool_indices: (n_survive,) indices into merged pool
        anchor_pool_indices:   list of anchor pool indices
        diagnostics:           dict with anchor count metrics
    """
    # STEP 1: Global ranks + crowding
    ranks, crowding = compute_ranks_crowding_all(F_pool)

    # STEP 2: Select anchors
    anchor_pool_indices, n_occupied, n_anchor_regions = select_anchors(
        old_region_ids, response_region, N_old, ranks, crowding
    )
    anchor_set = set(anchor_pool_indices)
    n_anchors = len(anchor_pool_indices)

    # STEP 3 + 4: Fill remaining slots from non-anchor candidates
    N_pool = len(X_pool)
    n_fill = n_survive - n_anchors
    candidates = [i for i in range(N_pool) if i not in anchor_set]
    remaining = _nsga2_priority_selection(ranks, crowding, candidates, n_fill)

    # STEP 5: Combine
    final_idx = sorted(anchor_pool_indices + remaining)
    assert len(final_idx) == n_survive, (
        f"R2 survival produced {len(final_idx)} individuals, "
        f"expected {n_survive} (n_anchors={n_anchors}, n_fill={n_fill})"
    )

    final_arr   = np.array(final_idx, dtype=int)
    surv_set    = set(final_idx)
    if n_anchors > 0:
        anc_surv_rate = float(sum(1 for ap in anchor_pool_indices if ap in surv_set) / n_anchors)
    else:
        anc_surv_rate = 1.0

    diagnostics = {
        "n_occupied_regions_before":  n_occupied,
        "n_anchor_regions":           n_anchor_regions,
        "n_anchors":                  n_anchors,
        "fraction_population_locked": float(n_anchors) / n_survive,
        "anchor_survival_rate":       anc_surv_rate,
    }
    return X_pool[final_arr], F_pool[final_arr], final_arr, anchor_pool_indices, diagnostics


# ============================================================
# Per-region R2 comparison for one scored event
# ============================================================

def run_region_r2(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
):
    """Compare UNRESTRICTED / HARD_PROTECTED / SOFT_ANCHOR for region r.

    Generates ONE POP child set and passes identical X_children / F_children
    to all three branches. Only survival differs.

    Returns a list with one row dict, or skipped-row dict (has_history=False).
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

    # POP estimator delta (reads only t-2 and t-1 via RegionHistory)
    motion = get_predict_motion_from_history(
        history, r, t=history_t + 1, lambda_pred=LAMBDA_PRED
    )
    if motion is None:
        return [{"has_history": False, "region_id": r, "event_id": change_idx}]

    delta_pop = motion["delta_hat"]   # shape (D,), normalized [0,1] space

    # GT delta for diagnostics only (NOT used in any survival branch)
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])

    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # Generate B=10 children ONCE (identical across all three branches)
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = generate_base_samples(BUDGET, D, rng_shared, n_r_before)
    ch_pop    = apply_samples(all_idxs, all_xis, X_region, delta_pop, SIGMA_POP, xl, xu)
    F_ch_pop  = cp_evaluate(ch_pop, t_new, region_dynamics)

    # Child quality
    n_unique_pop  = len(np.unique(ch_pop, axis=0))
    unique_frac   = float(n_unique_pop) / BUDGET
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    n_clipped = 0
    for pidx, xi in zip(all_idxs, all_xis):
        z_p   = (X_region[pidx] - xl_a) / ((xu_a - xl_a) + 1e-12)
        z_raw = z_p + delta_pop
        n_clipped += int(np.any((z_raw < 0.0) | (z_raw > 1.0)))
    clipping_frac = float(n_clipped) / BUDGET

    # Normalization basis for occupancy/PF metrics
    F_minus_a  = np.asarray(F_minus, dtype=float)
    F_all_norm = np.vstack([F_minus_a, F_before])
    z_m   = F_all_norm.min(axis=0)
    z_nad = F_all_norm.max(axis=0)
    norm_span = (z_nad - z_m) + 1e-12

    pf_star_arr = np.asarray(pf_star, dtype=float)
    pf_norm     = (pf_star_arr - z_m) / norm_span
    pf_ref_rids = np.asarray(partition(pf_norm, V), dtype=int)

    occ_before = np.bincount(region_ids, minlength=R)

    def _coverage_metrics(F_surv, prefix):
        igd_after = float(igd(F_surv, pf_star))
        delta_igd = igd_before - igd_after

        _, _, delta_d_arr, _ = compute_pf_coverage_per_point(
            F_before, F_surv, pf_star, region_ids
        )
        identity_residual = abs((igd_after - igd_before) - float(np.mean(delta_d_arr)))

        inside_d  = pf_ref_rids == r
        outside_d = ~inside_d
        mean_di = float(np.mean(delta_d_arr[inside_d]))   if np.any(inside_d)  else float("nan")
        mean_do = float(np.mean(delta_d_arr[outside_d]))  if np.any(outside_d) else float("nan")
        fw_i = float(np.mean(delta_d_arr[inside_d]  > 1e-14)) if np.any(inside_d)  else float("nan")
        fw_o = float(np.mean(delta_d_arr[outside_d] > 1e-14)) if np.any(outside_d) else float("nan")
        frac_worse = float(np.mean(delta_d_arr > 1e-14))
        frac_impr  = float(np.mean(delta_d_arr < -1e-14))

        F_surv_norm = (F_surv - z_m) / norm_span
        actual_rids = np.asarray(partition(F_surv_norm, V), dtype=int)
        actual_occ  = np.bincount(actual_rids, minlength=R)
        actual_resp_delta = int(actual_occ[r]) - int(occ_before[r])
        actual_d_other = actual_occ - occ_before
        actual_d_other[r] = 0
        n_other_lose  = int(np.sum(actual_d_other < 0))
        loss_other    = int(np.sum(np.abs(actual_d_other[actual_d_other < 0])))

        return {
            f"{prefix}_igd_after":               igd_after,
            f"{prefix}_delta_igd":               delta_igd,
            f"{prefix}_mean_delta_pf_inside":    mean_di,
            f"{prefix}_mean_delta_pf_outside":   mean_do,
            f"{prefix}_frac_worse_inside":       fw_i,
            f"{prefix}_frac_worse_outside":      fw_o,
            f"{prefix}_frac_pf_worsened":        frac_worse,
            f"{prefix}_frac_pf_improved":        frac_impr,
            f"{prefix}_actual_resp_delta":       actual_resp_delta,
            f"{prefix}_actual_n_other_losing":   n_other_lose,
            f"{prefix}_actual_total_loss_other": loss_other,
            f"{prefix}_igd_identity_residual":   float(identity_residual),
        }

    # Deterministic survival seed (same across all three branches)
    surv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    # ─── A. UNRESTRICTED ──────────────────────────────────────
    X_pool_u = np.vstack([X_pop, ch_pop])
    F_pool_u = np.vstack([F_before, F_ch_pop])
    X_surv_u, F_surv_u, surv_idx_u = nsga2_survival_traced(
        X_pool_u, F_pool_u, POP_SIZE, xl, xu, surv_seed
    )
    surv_old_u    = surv_idx_u[surv_idx_u < N_total_old]
    dropped_old_u = set(range(N_total_old)) - set(surv_old_u.tolist())
    n_old_dropped_u     = len(dropped_old_u)
    n_ch_survived_u     = int(np.sum(surv_idx_u >= N_total_old))
    dropped_other_u     = sum(1 for i in dropped_old_u if region_ids[i] != r)
    dropped_same_u      = n_old_dropped_u - dropped_other_u
    pop_replace_frac_u  = float(n_ch_survived_u) / POP_SIZE
    metrics_u = _coverage_metrics(F_surv_u, "unr")

    # ─── B. HARD_PROTECTED ────────────────────────────────────
    X_surv_h, F_surv_h, final_idx_h = region_protected_survival(
        X_pop, F_before, ch_pop, F_ch_pop,
        r, region_ids, n_r_before,
        xl, xu, surv_seed,
    )
    surv_old_h_set = set(int(i) for i in final_idx_h if i < N_total_old)
    dropped_old_h  = set(range(N_total_old)) - surv_old_h_set
    n_old_dropped_h     = len(dropped_old_h)
    n_ch_survived_h     = int(np.sum(final_idx_h >= N_total_old))
    dropped_other_h     = sum(1 for i in dropped_old_h if region_ids[i] != r)
    dropped_same_h      = n_old_dropped_h - dropped_other_h
    pop_replace_frac_h  = float(n_ch_survived_h) / POP_SIZE
    metrics_h = _coverage_metrics(F_surv_h, "hard")

    # Verify HARD cross-region invariant (structural guarantee)
    assert dropped_other_h == 0, (
        f"HARD protection dropped {dropped_other_h} outside-region OLD — "
        "region_protected_survival invariant violated"
    )

    # ─── C. SOFT_ANCHOR ───────────────────────────────────────
    X_pool_s = np.vstack([X_pop, ch_pop])
    F_pool_s = np.vstack([F_before, F_ch_pop])

    X_surv_s, F_surv_s, final_idx_s, anchor_pool_idxs, anchor_diag = \
        reference_anchor_survival(
            X_pool_s, F_pool_s,
            old_region_ids=region_ids,
            response_region=r,
            N_old=N_total_old,
            n_survive=POP_SIZE,
            xl=xl, xu=xu,
            seed=surv_seed,
        )

    # Verify all anchors survived
    surv_set_s = set(final_idx_s.tolist())
    for ap in anchor_pool_idxs:
        assert ap in surv_set_s, f"Anchor {ap} did not survive in SOFT_ANCHOR branch"

    surv_old_s_set = set(int(i) for i in final_idx_s if i < N_total_old)
    dropped_old_s  = set(range(N_total_old)) - surv_old_s_set
    n_old_dropped_s     = len(dropped_old_s)
    n_ch_survived_s     = int(np.sum(final_idx_s >= N_total_old))
    dropped_other_s     = sum(1 for i in dropped_old_s if region_ids[i] != r)
    dropped_same_s      = n_old_dropped_s - dropped_other_s
    pop_replace_frac_s  = float(n_ch_survived_s) / POP_SIZE

    # Anchor-specific metrics
    anchor_set_pool = set(anchor_pool_idxs)
    # Non-anchor OLD individuals from non-response regions that were dropped
    n_cross_dropped_non_anchor = sum(
        1 for i in dropped_old_s
        if region_ids[i] != r and i not in anchor_set_pool
    )
    # Non-response regions that lost ANY OLD (not counting anchors)
    regions_losing_non_anchor = set()
    for i in dropped_old_s:
        if region_ids[i] != r and i not in anchor_set_pool:
            regions_losing_non_anchor.add(int(region_ids[i]))
    n_anchor_regions_losing_other = len(regions_losing_non_anchor)

    # Anchor survival rate (always 1.0 by construction)
    anchor_survival_rate = 1.0 if anchor_diag["n_anchors"] == 0 else float(
        sum(1 for ap in anchor_pool_idxs if ap in surv_set_s) / anchor_diag["n_anchors"]
    )

    metrics_s = _coverage_metrics(F_surv_s, "soft")

    # ─── Paired contrasts ─────────────────────────────────────
    ddi_soft_unr  = metrics_s["soft_delta_igd"]  - metrics_u["unr_delta_igd"]
    ddi_soft_hard = metrics_s["soft_delta_igd"]  - metrics_h["hard_delta_igd"]
    ddi_hard_unr  = metrics_h["hard_delta_igd"]  - metrics_u["unr_delta_igd"]

    row = {
        "has_history":   True,
        "budget_b":      BUDGET,
        "region_type":   region_type,
        "dominant_bin":  dominant_bin,
        "n_r_before":    n_r_before,
        "igd_before":    igd_before,
        "fe":            BUDGET,
        # Estimator diagnostics
        "delta_pop":     delta_pop.tolist(),
        "delta_gt":      delta_gt_full.tolist(),
        **diag,
        # Child quality
        "n_unique_children": n_unique_pop,
        "unique_child_frac": unique_frac,
        "clipping_frac":     clipping_frac,
        # --- UNRESTRICTED ---
        "unr_n_old_dropped":       n_old_dropped_u,
        "unr_n_old_dropped_same":  dropped_same_u,
        "unr_n_old_dropped_other": dropped_other_u,
        "unr_n_ch_survived":       n_ch_survived_u,
        "unr_child_survival_rate": float(n_ch_survived_u) / BUDGET,
        "unr_pop_replace_frac":    pop_replace_frac_u,
        # --- HARD_PROTECTED ---
        "hard_n_old_dropped":       n_old_dropped_h,
        "hard_n_old_dropped_same":  dropped_same_h,
        "hard_n_old_dropped_other": dropped_other_h,
        "hard_n_ch_survived":       n_ch_survived_h,
        "hard_child_survival_rate": float(n_ch_survived_h) / BUDGET,
        "hard_pop_replace_frac":    pop_replace_frac_h,
        # --- SOFT_ANCHOR ---
        "soft_n_old_dropped":       n_old_dropped_s,
        "soft_n_old_dropped_same":  dropped_same_s,
        "soft_n_old_dropped_other": dropped_other_s,
        "soft_n_ch_survived":       n_ch_survived_s,
        "soft_child_survival_rate": float(n_ch_survived_s) / BUDGET,
        "soft_pop_replace_frac":    pop_replace_frac_s,
        # Anchor metrics
        **anchor_diag,
        "anchor_survival_rate":                  anchor_survival_rate,
        "n_cross_region_dropped_nonanchor":       n_cross_dropped_non_anchor,
        "n_anchor_regions_still_lose_nonanchor":  n_anchor_regions_losing_other,
        # Paired contrasts
        "ddi_soft_unr":   ddi_soft_unr,
        "ddi_soft_hard":  ddi_soft_hard,
        "ddi_hard_unr":   ddi_hard_unr,
        "delta_ch_surv_soft_vs_hard": n_ch_survived_s - n_ch_survived_h,
        "delta_ch_surv_soft_vs_unr":  n_ch_survived_s - n_ch_survived_u,
    }
    row.update(metrics_u)
    row.update(metrics_h)
    row.update(metrics_s)
    return [row]


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_r2(seed, n_scored_changes, scenario_name):
    """Run R2 comparison for one seed. Returns list of row dicts."""
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

    history  = RegionHistory(n_regions=R)
    rows_out = []
    gen      = 0
    n_total  = 1 + n_scored_changes
    history_t = -1

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        F_plus  = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm    = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))

        # Update history BEFORE predicting
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
            r_rows = run_region_r2(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
                region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
                history, history_t,
            )
            for row in r_rows:
                if row.get("has_history") is False:
                    rows_out.append({
                        "has_history": False,
                        "scenario":   scenario_name,
                        "seed":       seed,
                        "event_id":   change_idx,
                        "region_id":  row["region_id"],
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
    OUT_DIR  = Path("results/R2_reference_anchor_protection")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "R2_reference_anchor_protection.jsonl"

    all_rows = []
    n_valid   = 0
    n_missing = 0
    t0 = time.time()

    for seed in range(1, N_SEEDS + 1):
        t1 = time.time()
        rows = run_one_r2(seed, N_SCORED, SCENARIO)
        all_rows.extend(rows)
        seed_valid   = sum(1 for rw in rows if rw.get("has_history", True) is True)
        seed_missing = sum(1 for rw in rows if rw.get("has_history") is False)
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
