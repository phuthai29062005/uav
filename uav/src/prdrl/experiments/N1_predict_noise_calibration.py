"""
N1 — Predict Noise Calibration Under Fixed HARD Protection.

Scientific question:
    Given the actual POP motion estimator, and after cross-region displacement
    is controlled by R1 HARD protection, how large can PREDICT noise be before
    it overwhelms the directional motion signal and destroys useful response?

DESIGN:
    Primary: HARD protection at each sigma in SIGMA_SWEEP.
    Optional anchors: UNRESTRICTED at sigma=0 and sigma=0.03 (essentially free).
    CRN: (parent_idxs, xis) generated ONCE per (seed, event, region).
         All sigma values share identical parents and xi vectors.

TWO-PHASE DISCOVERY / HOLDOUT:
    N1-A — Discovery:  seeds 1–10, 5 scored events, all 5 sigmas.
    N1-B — Holdout:    seeds 11–20, 5 scored events, [sigma0, candidate, 0.03].
                       Only run if a clear candidate is identified from N1-A.

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged.
    - Production predictor (predict.py) unmodified.
    - region_protected_survival unmodified.
    - nsga2_survival_traced unmodified.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - No GT information enters child construction.
    - Missing history → has_history=False, skip.
    - Do NOT implement adaptive sigma in this experiment.
    - Do NOT start RL.
"""
import json
import sys
import time
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

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    bin_of,
)

# ============================================================
# Config — production locked
# ============================================================
LAMBDA_PRED   = 1.0
SIGMA_PROD    = 0.03    # production value — never change
BUDGET        = e2p3.BUDGET       # 10
POP_SIZE      = e2p3.POP_SIZE     # 100
D             = e2p3.D             # 10
N_OBJ         = e2p3.N_OBJ        # 2
N_PARTITIONS  = e2p3.N_PARTITIONS  # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

# Fixed, pre-declared sweep — NOT tuned after seeing results
SIGMA_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.03]

# Discovery / holdout seed ranges — pre-declared, disjoint
DISCOVERY_SEEDS = list(range(1, 11))   # 1–10
HOLDOUT_SEEDS   = list(range(11, 21))  # 11–20

# Helper aliases from existing modules
compute_time             = e2p3.compute_time
evolve_nsga2             = e2p3.evolve_nsga2
make_rng                 = e2p3.make_rng
make_scenario_dynamics   = e2p3.make_scenario_dynamics
median_nearest_pf_distance   = e2p3.median_nearest_pf_distance
nsga2_survival_traced        = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed       = e2p3._e2p3_common_surv_seed
generate_base_samples        = e2p3.generate_base_samples
apply_samples                = e2p3.apply_samples

compute_ranks_crowding_all    = r1.compute_ranks_crowding_all
compute_pf_coverage_per_point = r1.compute_pf_coverage_per_point
region_protected_survival     = r1.region_protected_survival

estimator_diagnostics = r1p.estimator_diagnostics


# ============================================================
# Noise magnitude helpers
# ============================================================

def rho_noise_metric(xis, sigma, delta_pop):
    """Compute noise-to-motion ratio and noise norms.

    rho_noise = median(||sigma * xi||) / (||delta_hat|| + eps)

    Args:
        xis:       list of B xi vectors, each shape (D,)
        sigma:     noise scale
        delta_pop: predicted motion, shape (D,) in [0,1]-normalized space

    Returns dict.
    """
    xi_arr = np.asarray(xis, dtype=float)
    noise_vecs  = sigma * xi_arr              # (B, D)
    noise_norms = np.linalg.norm(noise_vecs, axis=1)  # (B,)
    delta_norm  = float(np.linalg.norm(delta_pop))
    eps = 1e-12
    med_norm = float(np.median(noise_norms))
    rho = med_norm / (delta_norm + eps)
    return {
        "median_noise_norm": med_norm,
        "mean_noise_norm":   float(np.mean(noise_norms)),
        "delta_hat_norm":    delta_norm,
        "rho_noise":         rho,
    }


# ============================================================
# Child quality helpers
# ============================================================

def child_metrics(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu,
                  F_children, pf_star):
    """Per-child and aggregate quality metrics for one (sigma, region, event).

    Returns dict of clipping, uniqueness, pf_distance, displacement stats.
    """
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    B = len(parent_idxs)
    n_clipped        = 0
    clipped_dims_list = []
    d_actual_list     = []
    cosines           = []
    norm_ratios       = []
    delta_norm = float(np.linalg.norm(delta_pop))
    eps = 1e-12

    for i, (pidx, xi) in enumerate(zip(parent_idxs, xis)):
        z_p   = (X_region[pidx] - xl_a) / span
        z_raw = z_p + delta_pop + sigma * xi
        z_c   = np.clip(z_raw, 0.0, 1.0)

        # Clipping
        clipped_dims = int(np.sum((z_raw < 0.0) | (z_raw > 1.0)))
        if clipped_dims > 0:
            n_clipped += 1
        clipped_dims_list.append(clipped_dims)

        # Actual displacement (after clipping)
        d_actual = z_c - z_p
        d_actual_list.append(d_actual)

        # Displacement cosine vs delta_hat
        d_norm = float(np.linalg.norm(d_actual))
        if delta_norm > eps and d_norm > eps:
            cos = float(np.dot(d_actual, delta_pop) / (d_norm * delta_norm))
            cosines.append(cos)
            norm_ratios.append(d_norm / delta_norm)

    clipping_frac = float(n_clipped) / B if B > 0 else float("nan")

    # Unique children
    ch = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu)
    n_unique   = len(np.unique(ch, axis=0))
    unique_frac = float(n_unique) / B

    # PF distance (median)
    pf_med, _, _ = median_nearest_pf_distance(F_children, pf_star)

    disp_cosine = float(np.median(cosines))    if cosines    else float("nan")
    disp_ratio  = float(np.median(norm_ratios)) if norm_ratios else float("nan")

    return {
        "clipping_frac":              clipping_frac,
        "mean_clipped_dims":          float(np.mean(clipped_dims_list)),
        "max_clipped_dims":           int(max(clipped_dims_list)),
        "n_unique_children":          n_unique,
        "unique_child_fraction":      unique_frac,
        "child_pf_dist":              pf_med,
        "median_displacement_cosine": disp_cosine,
        "median_displacement_norm_ratio": disp_ratio,
    }


# ============================================================
# Per-region N1 run — all sigma values
# ============================================================

def run_region_n1(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
    sigma_values=None,
    include_unr_anchors=True,
):
    """Run HARD protection at each sigma for region r. Returns list of row dicts.

    CRN: one set of (parent_idxs, xis) shared across all sigma.
    Returns one row per sigma (plus optional UNR anchor rows at sigma=0, 0.03).
    Returns [] if region is empty or [] of skipped rows if no history.
    """
    if sigma_values is None:
        sigma_values = SIGMA_SWEEP

    mask = region_ids == r
    n_r_before = int(np.sum(mask))
    if n_r_before == 0:
        return []

    X_region    = X_pop[mask].copy()
    N_total_old = len(X_pop)
    R           = len(V)

    # Region metadata
    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type  = str(labels_r[np.argmax(counts_r)])
    bins_in_r    = bin_of(X_region[:, 0], n_bins)
    bv, bc       = np.unique(bins_in_r, return_counts=True)
    dominant_bin = int(bv[np.argmax(bc)])

    # POP estimator delta
    motion = get_predict_motion_from_history(
        history, r, t=history_t + 1, lambda_pred=LAMBDA_PRED
    )
    if motion is None:
        return [{"has_history": False, "region_id": r, "event_id": change_idx}]

    delta_pop = motion["delta_hat"]  # (D,) normalized [0,1] space

    # GT delta (diagnostics only — NOT used in child generation)
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])
    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # ─── CRN: generate base samples ONCE ──────────────────────────────────
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    parent_idxs, xis = generate_base_samples(BUDGET, D, rng_shared, n_r_before)

    # Normalization basis for coverage metrics
    F_minus_a  = np.asarray(F_minus, dtype=float)
    F_all_norm = np.vstack([F_minus_a, F_before])
    z_m   = F_all_norm.min(axis=0)
    z_nad = F_all_norm.max(axis=0)
    norm_span = (z_nad - z_m) + 1e-12

    pf_star_arr = np.asarray(pf_star, dtype=float)
    pf_norm     = (pf_star_arr - z_m) / norm_span
    pf_ref_rids = np.asarray(partition(pf_norm, V), dtype=int)
    occ_before  = np.bincount(region_ids, minlength=R)

    # Shared survival seed across all sigma for same (seed, event, region)
    surv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    def _coverage(F_surv):
        igd_after  = float(igd(F_surv, pf_star))
        delta_igd  = igd_before - igd_after

        _, _, delta_d_arr, _ = compute_pf_coverage_per_point(
            F_before, F_surv, pf_star, region_ids
        )
        identity_residual = abs((igd_after - igd_before) - float(np.mean(delta_d_arr)))

        inside_d  = pf_ref_rids == r
        outside_d = ~inside_d
        mdi = float(np.mean(delta_d_arr[inside_d]))   if np.any(inside_d)  else float("nan")
        mdo = float(np.mean(delta_d_arr[outside_d]))  if np.any(outside_d) else float("nan")
        fwi = float(np.mean(delta_d_arr[inside_d]  > 1e-14)) if np.any(inside_d)  else float("nan")
        fwo = float(np.mean(delta_d_arr[outside_d] > 1e-14)) if np.any(outside_d) else float("nan")
        frac_worse = float(np.mean(delta_d_arr > 1e-14))
        frac_impr  = float(np.mean(delta_d_arr < -1e-14))

        F_s_norm   = (F_surv - z_m) / norm_span
        actual_rids = np.asarray(partition(F_s_norm, V), dtype=int)
        actual_occ  = np.bincount(actual_rids, minlength=R)
        resp_delta  = int(actual_occ[r]) - int(occ_before[r])
        diff_other  = actual_occ - occ_before; diff_other[r] = 0
        n_other_lose = int(np.sum(diff_other < 0))
        loss_other   = int(np.sum(np.abs(diff_other[diff_other < 0])))

        return {
            "igd_after":               igd_after,
            "delta_igd":               delta_igd,
            "mean_delta_pf_inside":    mdi,
            "mean_delta_pf_outside":   mdo,
            "frac_worse_inside":       fwi,
            "frac_worse_outside":      fwo,
            "frac_pf_worsened":        frac_worse,
            "frac_pf_improved":        frac_impr,
            "actual_resp_region_delta": resp_delta,
            "actual_n_other_losing":   n_other_lose,
            "actual_total_loss_other": loss_other,
            "igd_identity_residual":   float(identity_residual),
        }

    # ─── Sigma=0 clipping baseline (for excess_clipping computation) ──────
    xl_a   = np.asarray(xl, dtype=float)
    xu_a   = np.asarray(xu, dtype=float)
    span_a = xu_a - xl_a
    n_clipped_sigma0 = sum(
        1 for pidx, xi in zip(parent_idxs, xis)
        if np.any(((X_region[pidx] - xl_a) / span_a + delta_pop) < 0.0)
           or np.any(((X_region[pidx] - xl_a) / span_a + delta_pop) > 1.0)
    )
    clipping_sigma0 = float(n_clipped_sigma0) / BUDGET

    rows = []

    # ─── Main sigma sweep: HARD at each sigma ─────────────────────────────
    for sigma in sigma_values:
        ch  = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu)
        F_ch = cp_evaluate(ch, t_new, region_dynamics)

        # Child metrics
        cm = child_metrics(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu,
                           F_ch, pf_star)
        nm = rho_noise_metric(xis, sigma, delta_pop)

        excess_clip = cm["clipping_frac"] - clipping_sigma0

        # HARD survival
        X_s, F_s, idx_s = region_protected_survival(
            X_pop, F_before, ch, F_ch,
            r, region_ids, n_r_before, xl, xu, surv_seed
        )
        cov = _coverage(F_s)

        # Survival counts
        surv_old_set  = set(int(i) for i in idx_s if i < N_total_old)
        dropped_old   = set(range(N_total_old)) - surv_old_set
        n_old_drop_other = sum(1 for i in dropped_old if region_ids[i] != r)
        assert n_old_drop_other == 0, (
            f"HARD protection dropped {n_old_drop_other} outside-region OLD at sigma={sigma}")
        n_ch_surv = int(np.sum(np.asarray(idx_s) >= N_total_old))
        ch_surv_rate = float(n_ch_surv) / BUDGET

        row = {
            "has_history":             True,
            "sigma":                   sigma,
            "n_r_before":              n_r_before,
            "region_type":             region_type,
            "dominant_bin":            dominant_bin,
            "fe":                      BUDGET,
            "igd_before":              igd_before,

            # Estimator
            "delta_hat_norm":          nm["delta_hat_norm"],
            **diag,

            # Noise magnitude
            "rho_noise":               nm["rho_noise"],
            "median_noise_norm":       nm["median_noise_norm"],
            "mean_noise_norm":         nm["mean_noise_norm"],

            # Child quality
            "clipping_frac":           cm["clipping_frac"],
            "excess_clipping_vs_s0":   excess_clip,
            "mean_clipped_dims":       cm["mean_clipped_dims"],
            "max_clipped_dims":        cm["max_clipped_dims"],
            "n_unique_children":       cm["n_unique_children"],
            "unique_child_fraction":   cm["unique_child_fraction"],
            "child_pf_dist":           cm["child_pf_dist"],
            "median_displacement_cosine":       cm["median_displacement_cosine"],
            "median_displacement_norm_ratio":   cm["median_displacement_norm_ratio"],

            # HARD survival
            "n_ch_survived":           n_ch_surv,
            "child_survival_rate":     ch_surv_rate,
            "n_old_dropped_other":     n_old_drop_other,

            # Population metrics
            **cov,
        }
        rows.append(row)

    # ─── Optional UNR anchors ──────────────────────────────────────────────
    # Compute UNR at sigma=0 and sigma=0.03; attach as unr_* to matching rows.
    if include_unr_anchors:
        for anc_sigma in [0.0, SIGMA_PROD]:
            if anc_sigma not in sigma_values:
                continue
            ch_anc = apply_samples(parent_idxs, xis, X_region, delta_pop,
                                   anc_sigma, xl, xu)
            F_anc  = cp_evaluate(ch_anc, t_new, region_dynamics)
            _, F_u, _ = nsga2_survival_traced(
                np.vstack([X_pop, ch_anc]),
                np.vstack([F_before, F_anc]),
                POP_SIZE, xl, xu, surv_seed
            )
            cov_u = _coverage(F_u)
            for row in rows:
                if row["sigma"] == anc_sigma:
                    for k, v in cov_u.items():
                        row[f"unr_{k}"] = v

    return rows


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_n1(seed, n_scored_changes, scenario_name, sigma_values=None,
               split="discovery", include_unr_anchors=True):
    """Run N1 for one seed. Returns list of row dicts (one per sigma per region-event)."""
    if sigma_values is None:
        sigma_values = SIGMA_SWEEP

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

    history   = RegionHistory(n_regions=R)
    rows_out  = []
    gen       = 0
    n_total   = 1 + n_scored_changes
    history_t = -1

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        F_plus  = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, _ = normalize(F_minus, F_plus)
        F_tm    = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))

        history_t += 1
        history.update(t=history_t, X=X_pop, F=F_minus,
                       region_ids=region_ids, xl=xl, xu=xu)

        if change_idx == 1:
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current

        for r in range(R):
            r_rows = run_region_n1(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
                region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
                history, history_t,
                sigma_values=sigma_values,
                include_unr_anchors=include_unr_anchors,
            )
            for row in r_rows:
                if row.get("has_history") is False:
                    rows_out.append({
                        "has_history": False,
                        "split":      split,
                        "scenario":   scenario_name,
                        "seed":       seed,
                        "event_id":   change_idx,
                        "region_id":  row["region_id"],
                    })
                    continue
                row.update({
                    "split":      split,
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
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows_out


# ============================================================
# Candidate selection (N1-A → N1-B)
# ============================================================

def select_candidate(discovery_rows, sigma_values=None):
    """Identify holdout candidate from N1-A discovery rows.

    Pre-declared selection rule:
    1. Filter to valid HARD rows (has_history=True, sigma > 0, split='discovery').
    2. For each nonzero sigma compute (vs sigma=0):
       - median D_s = delta_igd(s) - delta_igd(0)  [per matched pair]
       - mean delta_igd at sigma s
       - median clipping_frac
       - median rho_noise
       - fraction events: delta_igd(s) > 0
    3. Candidate must have:
       - mean delta_igd(s) > mean delta_igd(0.03) - 0.00002 [materially better than 0.03]
       - median rho_noise < 1.5 [noise not dominant]
       - clipping_frac < 0.7   [most children unclipped]
       - fraction_delta_igd_positive > 0.60
    4. If multiple pass: choose smallest sigma (parsimony tie-break).
    5. If none pass: return None (sigma=0 preferred or no clear candidate).

    Returns (sigma_candidate, reasoning_dict) or (None, reasoning_dict).
    """
    if sigma_values is None:
        sigma_values = SIGMA_SWEEP

    valid = [r for r in discovery_rows
             if r.get("has_history", True) and "delta_igd" in r
             and r.get("split") == "discovery"]

    # Group rows by (event_id, region_id, seed) key and sigma
    import collections
    by_key = collections.defaultdict(dict)
    for row in valid:
        key = (row["seed"], row["event_id"], row["region_id"])
        by_key[key][row["sigma"]] = row

    # Compute paired D_s for each event-region that has both sigma=0 and sigma s
    sigma_stats = {}
    for sigma in sigma_values:
        if sigma == 0.0:
            continue
        D_s_list   = []
        pf_pen_list = []
        clipping_list = []
        rho_list   = []
        digd_list  = []
        for key, sigma_map in by_key.items():
            if 0.0 in sigma_map and sigma in sigma_map:
                d_igd_0 = sigma_map[0.0]["delta_igd"]
                d_igd_s = sigma_map[sigma]["delta_igd"]
                D_s_list.append(d_igd_s - d_igd_0)
                pf_pen_list.append(sigma_map[sigma]["child_pf_dist"] -
                                   sigma_map[0.0]["child_pf_dist"])
                clipping_list.append(sigma_map[sigma]["clipping_frac"])
                rho_list.append(sigma_map[sigma]["rho_noise"])
                digd_list.append(d_igd_s)

        if not D_s_list:
            continue

        arr = np.array
        sigma_stats[sigma] = {
            "n_pairs":              len(D_s_list),
            "mean_D_s":             float(np.mean(D_s_list)),
            "median_D_s":           float(np.median(D_s_list)),
            "mean_delta_igd":       float(np.mean(digd_list)),
            "median_delta_igd":     float(np.median(digd_list)),
            "median_clipping":      float(np.median(clipping_list)),
            "median_rho":           float(np.median(rho_list)),
            "frac_digd_positive":   float(np.mean(np.array(digd_list) > 0)),
            "median_pf_penalty":    float(np.median(pf_pen_list)),
        }

    # sigma=0.03 reference
    ref_mean_digd = sigma_stats.get(0.03, {}).get("mean_delta_igd", -999.0)

    # Apply selection criteria
    reasoning = {"sigma_stats": sigma_stats, "ref_sigma03_mean_delta_igd": ref_mean_digd}
    candidates = []
    for sigma in sorted(sigma_values):
        if sigma == 0.0 or sigma not in sigma_stats:
            continue
        s = sigma_stats[sigma]
        c1 = s["mean_delta_igd"] > ref_mean_digd - 2e-5
        c2 = s["median_rho"]     < 1.5
        c3 = s["median_clipping"] < 0.70
        c4 = s["frac_digd_positive"] > 0.60
        reasoning[f"sigma_{sigma}_c1"] = c1
        reasoning[f"sigma_{sigma}_c2"] = c2
        reasoning[f"sigma_{sigma}_c3"] = c3
        reasoning[f"sigma_{sigma}_c4"] = c4
        if c1 and c2 and c3 and c4:
            candidates.append(sigma)

    if candidates:
        # Smallest sigma (parsimony tie-break)
        chosen = min(candidates)
        reasoning["candidate"] = chosen
        reasoning["tie_break_applied"] = len(candidates) > 1
        return chosen, reasoning

    reasoning["candidate"] = None
    reasoning["reason_no_candidate"] = (
        "No nonzero sigma satisfies all four pre-declared criteria simultaneously."
    )
    return None, reasoning


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

    SCENARIO = "PERSISTENT"
    N_SCORED = 5
    OUT_DIR  = Path("results/N1_predict_noise_calibration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "N1_predict_noise_calibration.jsonl"

    all_rows  = []
    n_valid   = 0
    n_missing = 0

    # ─── N1-A: Discovery ──────────────────────────────────────────────────
    print("=== N1-A: Discovery ===")
    t0 = time.time()
    for seed in DISCOVERY_SEEDS:
        t1 = time.time()
        rows = run_one_n1(seed, N_SCORED, SCENARIO, SIGMA_SWEEP, split="discovery")
        all_rows.extend(rows)
        sv = sum(1 for r in rows if r.get("has_history", True))
        ms = sum(1 for r in rows if not r.get("has_history", True))
        n_valid   += sv
        n_missing += ms
        print(f"  seed={seed}: {sv} valid rows, {ms} skipped in {time.time()-t1:.1f}s")

    # ─── Select candidate ─────────────────────────────────────────────────
    discovery_rows = [r for r in all_rows if r.get("split") == "discovery"]
    candidate, reasoning = select_candidate(discovery_rows)
    print(f"\nN1-A candidate: sigma={candidate}")
    for sigma, s in reasoning.get("sigma_stats", {}).items():
        print(f"  sigma={sigma}: mean_digd={s['mean_delta_igd']:.6f}  "
              f"rho={s['median_rho']:.2f}  clip={s['median_clipping']:.2f}  "
              f"frac+={s['frac_digd_positive']:.2f}")

    # ─── N1-B: Holdout ────────────────────────────────────────────────────
    if candidate is not None:
        holdout_sigmas = sorted({0.0, candidate, SIGMA_PROD})
        print(f"\n=== N1-B: Holdout (sigma={holdout_sigmas}) ===")
        for seed in HOLDOUT_SEEDS:
            t1 = time.time()
            rows = run_one_n1(seed, N_SCORED, SCENARIO, holdout_sigmas,
                              split="holdout", include_unr_anchors=False)
            all_rows.extend(rows)
            sv = sum(1 for r in rows if r.get("has_history", True))
            ms = sum(1 for r in rows if not r.get("has_history", True))
            n_valid   += sv
            n_missing += ms
            print(f"  seed={seed}: {sv} valid rows, {ms} skipped in {time.time()-t1:.1f}s")
    else:
        print("\nNo holdout candidate identified. N1-B skipped.")

    # ─── Save ─────────────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | {n_valid} valid, {n_missing} skipped "
          f"| {len(all_rows)} rows → {jsonl_path}")

    # ─── Save candidate ───────────────────────────────────────────────────
    with open(OUT_DIR / "N1_candidate.json", "w") as f:
        json.dump(json_safe({"candidate": candidate, "reasoning": reasoning}), f, indent=2)
    print(f"Candidate saved.")
