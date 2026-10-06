"""
E2-P5 -- Coverage & Crowding Disruption Audit.

Main question:
  When high-quality GT children are injected into one Pareto region,
  which OLD solutions are displaced, why are they displaced,
  and does their removal damage global Pareto-front coverage?

Phases per (seed, event, response_region, B):
  1.  Baseline: X_before, F_before, region assignments, IGD_before
  2.  GT children: sigma=0, nested B_MAX=50 parent samples, clipping stats
  3.  Child diversity: pairwise distance before/after shift
  4.  Rank + crowding of baseline population (before injection)
  5.  Rank + crowding of merged pool (OLD + children)
  6.  Survival with pool_idx provenance (nsga2_survival_traced)
  7.  Dropped OLD: region, rank vs crowding drop reason
  8.  Region occupancy before vs after
  9.  PF* coverage: per-point d_before, d_after, delta_d
  10. Coverage responsibility: how many PF* points each OLD individual covered
  11. IGD identity check: mean(delta_d_j) should equal delta_igd

GT is diagnostic oracle (sigma=0). NEVER goes to production.
Do NOT change: sigma_P, BUDGET, NSGA-II, predictor, E2a/E2a-R, RegionHistory.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.E2P4_headroom_dilution as e2p4
import prdrl.experiments.E2P2_predict_noise_sweep as e2p2

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import nsga2_survival, split_by_region

from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    distance_to_region_ps, bin_of,
)

# ============================================================
# Config — production locked
# ============================================================
SIGMA_GT_E5  = 0.0
BUDGET       = e2p3.BUDGET          # 10 (production, never changed)
POP_SIZE     = e2p3.POP_SIZE        # 100
D            = e2p3.D               # 10
N_OBJ        = e2p3.N_OBJ           # 2
N_PARTITIONS = e2p3.N_PARTITIONS    # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE
LAMBDA_PRED  = e2p3.LAMBDA_PRED

B_MAX    = 50
B_VALUES = [10, 20, 50]

compute_time        = e2p3.compute_time
evolve_nsga2        = e2p3.evolve_nsga2
make_rng            = e2p3.make_rng
make_scenario_dynamics = e2p3.make_scenario_dynamics

# Reuse corrected helpers
median_nearest_pf_distance = e2p3.median_nearest_pf_distance
assign_ranks               = e2p3.assign_ranks
nsga2_survival_traced      = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed     = e2p3._e2p3_common_surv_seed
generate_base_samples      = e2p3.generate_base_samples
apply_samples              = e2p3.apply_samples
make_local_children        = e2p3.make_local_children


# ============================================================
# Crowding distance (NSGA-II standard formula)
# ============================================================

def _calc_crowding_dist(F_front):
    """Standard NSGA-II crowding distance — matches pymoo calc_crowding_distance.

    A5 fix: divide sum by n_obj (n_objectives) to match pymoo's formula exactly.
    The relative ordering is unchanged; only the absolute scale is corrected.
    """
    n, m = F_front.shape
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n, dtype=float)
    for obj in range(m):
        sorted_idx = np.argsort(F_front[:, obj])
        f_min = F_front[sorted_idx[0], obj]
        f_max = F_front[sorted_idx[-1], obj]
        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf
        span = f_max - f_min
        if span < 1e-12:
            continue
        for k in range(1, n - 1):
            cd[sorted_idx[k]] += (
                F_front[sorted_idx[k + 1], obj] - F_front[sorted_idx[k - 1], obj]
            ) / span
    # A5: normalize by n_obj to match pymoo's calc_crowding_distance
    return cd / m


def compute_ranks_crowding_all(F_pool):
    """Compute Pareto rank (0=best) and crowding distance for all pool members.

    Uses same NonDominatedSorting + standard NSGA-II crowding as pymoo's NSGA-II.
    Returns: ranks (N,), crowding (N,)  — crowding is inf at Pareto-front boundaries.
    """
    F = np.asarray(F_pool, dtype=float)
    N = len(F)
    nds = NonDominatedSorting()
    fronts = nds.do(F)

    ranks    = np.full(N, 0, dtype=int)
    crowding = np.zeros(N, dtype=float)

    for r_idx, front in enumerate(fronts):
        fa = np.array(front, dtype=int)
        ranks[fa] = r_idx
        crowding[fa] = _calc_crowding_dist(F[fa])

    return ranks, crowding


def classify_drop_reasons(ranks_all, survived_set, N):
    """Classify each dropped pool member as DROP_BY_RANK or DROP_BY_CROWDING.

    truncation_rank = max Pareto rank among survivors.
    - If rank == truncation_rank AND not survived: crowded out from the partial front.
    - If rank >  truncation_rank: entire front excluded (dominance / rank).
    Returns drop_reason array ('SURVIVED', 'RANK', 'CROWDING') of length N.
    """
    drop_reason = np.full(N, "SURVIVED", dtype=object)
    if not survived_set:
        drop_reason[:] = "RANK"
        return drop_reason

    max_surv_rank = int(max(ranks_all[i] for i in survived_set))

    # Check whether the truncation front is only partially filled
    members_at_max = {i for i in range(N) if int(ranks_all[i]) == max_surv_rank}
    if members_at_max.issubset(survived_set):
        # All members of max-rank front survived → no crowding truncation occurred
        truncation_rank = max_surv_rank + 1  # virtual rank above all fronts
    else:
        truncation_rank = max_surv_rank

    for i in range(N):
        if i in survived_set:
            continue
        if int(ranks_all[i]) == truncation_rank:
            drop_reason[i] = "CROWDING"
        else:
            drop_reason[i] = "RANK"

    return drop_reason


# ============================================================
# Pairwise distance helper
# ============================================================

def median_pairwise_distance(X):
    """Median pairwise Euclidean distance among rows of X. Returns nan for < 2 rows."""
    if X is None or len(X) < 2:
        return float("nan")
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]   # N x N x D
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # upper triangle
    iu = np.triu_indices(len(X), k=1)
    return float(np.median(dists[iu]))


# ============================================================
# Clipping stats
# ============================================================

def compute_clipping_stats(X_region, delta_full, idxs, xis, xl, xu):
    """Return clipping statistics using explicit normalized [0,1] coordinates.

    A4 fix: normalize X and delta to [0,1] space before checking bounds so the
    check is correct for any xl/xu, not just the [0,1] controlled-predict case.
    Mathematically equivalent to checking z_raw < xl or z_raw > xu, but explicit.
    """
    xl_a  = np.asarray(xl, dtype=float)
    xu_a  = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a
    span  = np.where(span < 1e-12, np.ones_like(span), span)

    n_clipped_total = 0
    dims_clipped = []
    for pidx, xi in zip(idxs, xis):
        z_parent = (X_region[pidx] - xl_a) / span   # parent in [0,1]
        z_delta  = delta_full / span                  # delta in [0,1] units
        z_raw    = z_parent + z_delta                 # pre-clip in normalized space
        clipped_dims = int(np.sum((z_raw < 0.0) | (z_raw > 1.0)))
        n_clipped_total += int(clipped_dims > 0)
        dims_clipped.append(clipped_dims)
    n = len(idxs)
    clipping_fraction = n_clipped_total / n if n > 0 else float("nan")
    mean_clipped_dims = float(np.mean(dims_clipped)) if dims_clipped else float("nan")
    max_clipped_dims  = int(max(dims_clipped)) if dims_clipped else 0
    return n_clipped_total, clipping_fraction, mean_clipped_dims, max_clipped_dims


# ============================================================
# PF coverage per point
# ============================================================

def compute_pf_coverage_per_point(F_before, F_after, pf_star, old_region_ids):
    """Per PF* point: d_before, d_after, delta_d, pf_region_id.

    pf_region_id = region_id of the nearest OLD individual in F_before.
    delta_d > 0 means coverage worsened.
    """
    pf_star  = np.asarray(pf_star, dtype=float)
    F_b      = np.asarray(F_before, dtype=float)
    F_a      = np.asarray(F_after, dtype=float)

    dist2_b = np.sum((pf_star[:, None, :] - F_b[None, :, :]) ** 2, axis=2)  # K x N
    d_before = np.sqrt(np.min(dist2_b, axis=1))                              # K
    nearest_before = np.argmin(dist2_b, axis=1)                              # K

    dist2_a = np.sum((pf_star[:, None, :] - F_a[None, :, :]) ** 2, axis=2)
    d_after = np.sqrt(np.min(dist2_a, axis=1))                               # K

    delta_d = d_after - d_before
    pf_region_ids = old_region_ids[nearest_before]

    return d_before, d_after, delta_d, pf_region_ids


# ============================================================
# Coverage responsibility
# ============================================================

def compute_responsibility(F_before, pf_star):
    """For each OLD individual, count how many PF* points it is the nearest representative for."""
    F_b = np.asarray(F_before, dtype=float)
    P   = np.asarray(pf_star, dtype=float)
    dist2 = np.sum((P[:, None, :] - F_b[None, :, :]) ** 2, axis=2)  # K x N
    nearest = np.argmin(dist2, axis=1)                                # K
    responsibility = np.bincount(nearest, minlength=len(F_b))
    return responsibility


# ============================================================
# Per-region analysis for one scored event
# ============================================================

def run_region_e2p5(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev, t_prev2,
    prev_regions, region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
):
    """Phases 1–11 for one (response region, event). Returns one row dict per B value.

    F_minus: population objectives BEFORE the environment change (needed for A1/A2
    consistent normalization basis — same as normalize(F_minus, F_plus) used to
    compute region_ids).
    """
    mask = region_ids == r
    n_old_region = int(np.sum(mask))
    if n_old_region == 0:
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

    # GT oracle delta
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # A1/A2: normalization basis — same as normalize(F_minus, F_plus) used for region_ids
    F_minus_a = np.asarray(F_minus, dtype=float)
    F_all_norm = np.vstack([F_minus_a, F_before])
    z_m       = F_all_norm.min(axis=0)
    z_nad     = F_all_norm.max(axis=0)
    _alpha    = 1e-12
    norm_span = (z_nad - z_m) + _alpha   # matches normalize() denominator

    # A2: direct PF reference-region assignment (not population-dependent)
    pf_star_arr     = np.asarray(pf_star, dtype=float)
    pf_star_norm    = (pf_star_arr - z_m) / norm_span
    pf_reference_region_ids = np.asarray(partition(pf_star_norm, V), dtype=int)

    # Verify A1 sanity: before-occupancy from actual repartition should match region_ids
    F_before_norm          = (F_before - z_m) / norm_span
    actual_region_ids_before = np.asarray(partition(F_before_norm, V), dtype=int)

    # Phase 4: rank + crowding of baseline population
    ranks_before, crowding_before = compute_ranks_crowding_all(F_before)

    # Phase 2: generate B_MAX parent samples once (nested)
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = generate_base_samples(B_MAX, D, rng_shared, n_old_region)

    # Coverage responsibility (before, constant across all B branches)
    resp_before = compute_responsibility(F_before, pf_star)

    common_sv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    # Region occupancy before
    occ_before = np.bincount(region_ids, minlength=R)  # shape R

    rows = []

    for B in B_VALUES:
        b_idxs = all_idxs[:B]
        b_xis  = all_xis[:B]

        # Children (no mutation of X_pop or X_region)
        ch_gt = apply_samples(b_idxs, b_xis, X_region, delta_gt_full, 0.0, xl, xu)
        F_ch  = cp_evaluate(ch_gt, t_new, region_dynamics)

        # Phase 3 — clipping
        n_clipped, clip_frac, mean_cdims, max_cdims = compute_clipping_stats(
            X_region, delta_gt_full, b_idxs, b_xis, xl, xu
        )

        # Phase 3 — child diversity
        n_uniq_parents = int(len(np.unique(b_idxs)))
        uniq_parent_frac = n_uniq_parents / B
        ch_unique = np.unique(ch_gt, axis=0)
        n_uniq_ch = len(ch_unique)
        uniq_ch_frac = n_uniq_ch / B

        # Old (source-based, deduplicated): kept for backward compatibility
        mpd_parent = median_pairwise_distance(X_region[np.unique(b_idxs)])
        mpd_child  = median_pairwise_distance(ch_gt)
        pairwise_ratio = mpd_child / (mpd_parent + 1e-12)

        # A3: paired (with repeats) — same row count as ch_gt, comparable row-for-row
        mpd_parent_paired = median_pairwise_distance(X_region[b_idxs])
        pairwise_ratio_paired = mpd_child / (mpd_parent_paired + 1e-12)

        # Phase 5 — ranks + crowding of merged pool
        X_pool = np.vstack([X_pop, ch_gt])       # X_pop never mutated between B branches
        F_pool = np.vstack([F_before, F_ch])
        N_pool = N_total_old + B

        ranks_merged, crowding_merged = compute_ranks_crowding_all(F_pool)
        ranks_old_merged    = ranks_merged[:N_total_old]
        crowding_old_merged = crowding_merged[:N_total_old]

        # Crowding change for OLD individuals (compute only for finite→finite pairs)
        both_finite = np.isfinite(crowding_before) & np.isfinite(crowding_old_merged)
        delta_crowding_old = np.full(N_total_old, float("nan"))
        if np.any(both_finite):
            delta_crowding_old[both_finite] = (
                crowding_old_merged[both_finite] - crowding_before[both_finite]
            )
        frac_crowding_decreased = (float(np.mean(delta_crowding_old[both_finite] < 0))
                                   if np.any(both_finite) else float("nan"))

        # Crowding transitions
        n_finite_to_inf   = int(np.sum(np.isfinite(crowding_before) & np.isinf(crowding_old_merged)))
        n_inf_to_finite   = int(np.sum(np.isinf(crowding_before)  & np.isfinite(crowding_old_merged)))

        # Phase 6 — survival with provenance
        X_surv, F_surv, surv_orig_idx = nsga2_survival_traced(
            X_pool, F_pool, POP_SIZE, xl, xu, common_sv_seed
        )
        survived_set = set(surv_orig_idx.tolist())

        # Phase 7 — who survived / who was dropped
        surv_old_mask = surv_orig_idx < N_total_old
        survived_old_set = set(surv_orig_idx[surv_old_mask].tolist())
        dropped_old_set  = set(range(N_total_old)) - survived_old_set
        n_old_dropped    = len(dropped_old_set)

        n_ch_survived = int(np.sum(~surv_old_mask))
        child_surv_rate = n_ch_survived / B
        pop_replace_frac = n_ch_survived / POP_SIZE

        # Classify drop reasons using actual survived_set
        drop_reason_arr = classify_drop_reasons(ranks_merged, survived_set, N_pool)
        drop_reason_old = drop_reason_arr[:N_total_old]

        n_drop_rank     = int(np.sum([drop_reason_old[i] == "RANK"     for i in dropped_old_set]))
        n_drop_crowding = int(np.sum([drop_reason_old[i] == "CROWDING" for i in dropped_old_set]))

        # Dropped OLD region classification
        dropped_same_r  = sum(1 for i in dropped_old_set if region_ids[i] == r)
        dropped_other_r = n_old_dropped - dropped_same_r

        # Crowding of dropped vs survived OLD
        def _safe_med(vals):
            v = [x for x in vals if np.isfinite(x)]
            return float(np.median(v)) if v else float("nan")

        med_cd_dropped  = _safe_med([crowding_old_merged[i] for i in dropped_old_set])
        med_cd_survived = _safe_med([crowding_old_merged[i] for i in survived_old_set])
        med_dcd_dropped  = _safe_med([delta_crowding_old[i] for i in dropped_old_set if both_finite[i]])
        med_dcd_survived = _safe_med([delta_crowding_old[i] for i in survived_old_set if both_finite[i]])

        # Phase 8 — region occupancy (source-based: uses pre-survival region assignments)
        occ_after_old = np.bincount(region_ids[list(survived_old_set)], minlength=R) if survived_old_set else np.zeros(R, dtype=int)
        # Source-based: children attributed to response region r
        occ_after_full = occ_after_old.copy()
        occ_after_full[r] += n_ch_survived

        resp_region_delta = int(occ_after_full[r]) - int(occ_before[r])
        occ_delta_other = occ_after_old - occ_before
        occ_delta_other[r] = 0  # exclude response region from "other"
        n_other_losing = int(np.sum(occ_delta_other < 0))
        total_loss_other = int(np.sum(np.abs(occ_delta_other[occ_delta_other < 0])))

        # A1: actual post-survival region occupancy via repartitioning F_surv
        F_surv_norm           = (F_surv - z_m) / norm_span
        actual_region_ids_after = np.asarray(partition(F_surv_norm, V), dtype=int)

        actual_occ_before = np.bincount(actual_region_ids_before, minlength=R)
        actual_occ_after  = np.bincount(actual_region_ids_after,  minlength=R)
        actual_resp_delta = int(actual_occ_after[r]) - int(actual_occ_before[r])
        actual_delta_other = actual_occ_after - actual_occ_before
        actual_delta_other_excl = actual_delta_other.copy()
        actual_delta_other_excl[r] = 0
        actual_n_other_losing   = int(np.sum(actual_delta_other_excl < 0))
        actual_total_loss_other = int(np.sum(np.abs(actual_delta_other_excl[actual_delta_other_excl < 0])))

        # Phase 9 — PF coverage per point
        d_before_arr, d_after_arr, delta_d_arr, pf_rids = compute_pf_coverage_per_point(
            F_before, F_surv, pf_star, region_ids
        )

        frac_worsened = float(np.mean(delta_d_arr > 1e-14))
        frac_improved = float(np.mean(delta_d_arr < -1e-14))
        mean_delta_d  = float(np.mean(delta_d_arr))
        med_delta_d   = float(np.median(delta_d_arr))
        p90_delta_d   = float(np.percentile(delta_d_arr, 90))
        max_delta_d   = float(np.max(delta_d_arr))
        max_gap_before = float(np.max(d_before_arr))
        max_gap_after  = float(np.max(d_after_arr))

        # Representative-based (population-dependent): pf_rids = region of nearest OLD
        inside_mask  = pf_rids == r
        outside_mask = ~inside_mask
        mean_delta_inside  = float(np.mean(delta_d_arr[inside_mask]))  if np.any(inside_mask)  else float("nan")
        mean_delta_outside = float(np.mean(delta_d_arr[outside_mask])) if np.any(outside_mask) else float("nan")
        frac_worse_inside  = float(np.mean(delta_d_arr[inside_mask]  > 1e-14)) if np.any(inside_mask)  else float("nan")
        frac_worse_outside = float(np.mean(delta_d_arr[outside_mask] > 1e-14)) if np.any(outside_mask) else float("nan")

        # A2: direct reference-region PF coverage (geometry-based, not nearest-OLD)
        inside_mask_direct  = pf_reference_region_ids == r
        outside_mask_direct = ~inside_mask_direct
        mean_delta_inside_direct  = float(np.mean(delta_d_arr[inside_mask_direct]))   if np.any(inside_mask_direct)  else float("nan")
        mean_delta_outside_direct = float(np.mean(delta_d_arr[outside_mask_direct]))  if np.any(outside_mask_direct) else float("nan")
        frac_worse_inside_direct  = float(np.mean(delta_d_arr[inside_mask_direct]  > 1e-14)) if np.any(inside_mask_direct)  else float("nan")
        frac_worse_outside_direct = float(np.mean(delta_d_arr[outside_mask_direct] > 1e-14)) if np.any(outside_mask_direct) else float("nan")

        # Phase 10 — coverage responsibility
        dropped_resp  = [resp_before[i] for i in dropped_old_set]
        survived_resp = [resp_before[i] for i in survived_old_set]
        med_resp_dropped  = float(np.median(dropped_resp))  if dropped_resp  else float("nan")
        med_resp_survived = float(np.median(survived_resp)) if survived_resp else float("nan")

        # Fraction of PF points whose nearest OLD representative was dropped
        nearest_old = np.argmin(
            np.sum((pf_star[:, None, :] - F_before[None, :, :]) ** 2, axis=2),
            axis=1
        )  # K
        frac_pf_nearest_dropped = float(np.mean([1 if int(p) in dropped_old_set else 0 for p in nearest_old]))

        # Phase 11 — IGD and identity check
        igd_after  = float(igd(F_surv, pf_star))
        delta_igd  = igd_before - igd_after
        rel_igd    = delta_igd / (abs(igd_before) + 1e-12)
        igd_identity_residual = abs(mean_delta_d - (igd_after - igd_before))

        rows.append({
            "budget_b": B,
            "region_type": region_type,
            "dominant_bin": dominant_bin,
            # child geometry
            "n_region_parents": n_old_region,
            "n_unique_parent_indices": n_uniq_parents,
            "unique_parent_fraction": uniq_parent_frac,
            "n_unique_children": n_uniq_ch,
            "unique_child_fraction": uniq_ch_frac,
            "n_children_clipped": n_clipped,
            "clipping_fraction": clip_frac,
            "mean_clipped_dims": mean_cdims,
            "max_clipped_dims": max_cdims,
            "median_pairwise_distance_parent": mpd_parent,
            "median_pairwise_distance_child": mpd_child,
            "pairwise_distance_ratio": pairwise_ratio,
            # A3: paired diversity (parent sample includes repeats, same size as children)
            "median_pairwise_distance_parent_paired": mpd_parent_paired,
            "pairwise_distance_ratio_paired": pairwise_ratio_paired,
            # survival
            "n_children_survived": n_ch_survived,
            "child_survival_rate": child_surv_rate,
            "population_replacement_fraction": pop_replace_frac,
            "n_old_dropped": n_old_dropped,
            "n_old_dropped_same_region": dropped_same_r,
            "n_old_dropped_other_regions": dropped_other_r,
            "frac_dropped_same_region": (dropped_same_r / n_old_dropped if n_old_dropped > 0 else float("nan")),
            "frac_dropped_other_regions": (dropped_other_r / n_old_dropped if n_old_dropped > 0 else float("nan")),
            "n_drop_by_rank": n_drop_rank,
            "n_drop_by_crowding": n_drop_crowding,
            # crowding
            "frac_old_crowding_decreased": frac_crowding_decreased,
            "n_crowding_finite_to_inf": n_finite_to_inf,
            "n_crowding_inf_to_finite": n_inf_to_finite,
            "median_crowding_after_dropped": med_cd_dropped,
            "median_crowding_after_survived": med_cd_survived,
            "median_delta_crowding_dropped": med_dcd_dropped,
            "median_delta_crowding_survived": med_dcd_survived,
            # region occupancy (source-based: uses pre-survival region assignments)
            "response_region_delta_occupancy": resp_region_delta,
            "n_other_regions_losing_occupancy": n_other_losing,
            "total_occupancy_lost_other_regions": total_loss_other,
            # A1: actual post-survival occupancy (repartition F_surv with consistent basis)
            "actual_response_region_delta_occupancy": actual_resp_delta,
            "actual_n_other_regions_losing_occupancy": actual_n_other_losing,
            "actual_total_occupancy_lost_other_regions": actual_total_loss_other,
            # PF coverage (representative-based: pf_region = region of nearest OLD)
            "fraction_pf_points_worsened": frac_worsened,
            "fraction_pf_points_improved": frac_improved,
            "mean_delta_pf_coverage": mean_delta_d,
            "median_delta_pf_coverage": med_delta_d,
            "p90_delta_pf_coverage": p90_delta_d,
            "max_delta_pf_coverage": max_delta_d,
            "max_pf_distance_before": max_gap_before,
            "max_pf_distance_after": max_gap_after,
            "mean_delta_pf_inside": mean_delta_inside,
            "mean_delta_pf_outside": mean_delta_outside,
            "frac_worsened_inside": frac_worse_inside,
            "frac_worsened_outside": frac_worse_outside,
            # A2: direct reference-region PF coverage (geometry-based, not nearest-OLD)
            "mean_delta_pf_inside_direct": mean_delta_inside_direct,
            "mean_delta_pf_outside_direct": mean_delta_outside_direct,
            "frac_worsened_inside_direct": frac_worse_inside_direct,
            "frac_worsened_outside_direct": frac_worse_outside_direct,
            # coverage responsibility
            "median_responsibility_dropped": med_resp_dropped,
            "median_responsibility_survived": med_resp_survived,
            "frac_pf_nearest_old_dropped": frac_pf_nearest_dropped,
            # IGD
            "igd_before": igd_before,
            "igd_after": igd_after,
            "delta_igd": delta_igd,
            "relative_igd_improvement": rel_igd,
            "mean_delta_d_j": mean_delta_d,
            "igd_identity_residual": igd_identity_residual,
            "fe": B,
        })

    return rows


# ============================================================
# PF delta rows (per PF point) for the reference delta CSV
# ============================================================

def make_pf_delta_rows(seed, event_id, region_id, B, d_before, d_after, delta_d, pf_rids):
    """Yield one dict per PF* point for E2P5_pf_reference_deltas.csv."""
    for j in range(len(d_before)):
        yield {
            "seed": seed,
            "event_id": event_id,
            "region_id": region_id,
            "budget_b": B,
            "pf_index": j,
            "d_before": float(d_before[j]),
            "d_after": float(d_after[j]),
            "delta_d": float(delta_d[j]),
            "pf_region_id": int(pf_rids[j]),
        }


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_e2p5(seed, n_scored_changes, scenario_name, save_pf_deltas=False):
    """Returns (rows, pf_delta_rows). pf_delta_rows only populated if save_pf_deltas=True."""
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

    prev_regions = None
    t_prev2 = None

    rows_out      = []
    pf_delta_out  = []
    gen = 0
    n_total = 1 + n_scored_changes

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        F_plus  = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, F_tp = normalize(F_minus, F_plus)
        F_tm = np.asarray(F_tm)
        region_ids  = np.asarray(partition(F_tm, V))
        cur_regions = split_by_region(X_pop, region_ids, R)

        if change_idx == 1:
            prev_regions = cur_regions
            t_prev2   = t_current
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE, seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current

        for r in range(R):
            r_rows = run_region_e2p5(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev, t_prev2,
                prev_regions, region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
            )
            for row in r_rows:
                row.update({
                    "scenario":   scenario_name,
                    "seed":       seed,
                    "event_id":   change_idx,
                    "change_idx": change_idx,
                    "region_id":  r,
                    "t_prev2":    float(t_prev2),
                    "t_prev":     float(t_prev),
                    "t_new":      float(t_new),
                })
                rows_out.append(row)

                if save_pf_deltas:
                    # Recompute PF deltas for this row (needed for per-point CSV)
                    # We don't store them in row to keep memory low; recompute here
                    pass  # handled in main() if needed

        prev_regions = cur_regions
        t_prev2 = t_prev
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
        return None if np.isnan(v) else v
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


if __name__ == "__main__":
    import csv

    N_SEEDS  = 10
    N_SCORED = 5
    SCENARIO = "PERSISTENT"
    OUT_DIR  = Path("results/E2P5_coverage_crowding_audit")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path  = OUT_DIR / "E2P5_coverage_crowding_audit.jsonl"

    all_rows = []
    t0 = time.time()

    for seed in range(1, N_SEEDS + 1):
        t1 = time.time()
        rows = run_one_e2p5(seed, N_SCORED, SCENARIO)
        all_rows.extend(rows)
        print(f"{SCENARIO} seed={seed}: {len(rows)} rows in {time.time()-t1:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | Wrote {len(all_rows)} rows to {jsonl_path}")
