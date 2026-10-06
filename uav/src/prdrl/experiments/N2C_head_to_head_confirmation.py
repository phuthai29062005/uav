"""
N2C — Frozen Head-to-Head Confirmation: A025 vs F005.

Scientific question:
    On completely fresh holdout data, is signal-scaled A025
    (sigma_r = min(0.03, 0.25 * ||delta_hat|| / sqrt(d)))
    practically better than fixed sigma=0.005?
    Or are they practically equivalent?

DESIGN:
    CONFIRMATORY experiment — no parameter sweep, no candidate selection.
    Both methods are FROZEN from N2:
        F005: sigma = 0.005 (fixed constant)
        A025: sigma_r = min(0.03, 0.25 * ||delta_hat|| / sqrt(d))

    Primary arms (determine verdict):
        F005 — frozen fixed noise
        A025 — frozen adaptive noise

    Optional reference arms (do NOT affect decision):
        F0  — sigma = 0 (noise-free baseline)
        F03 — sigma = 0.03 (production)

    All arms: HARD-protected, CRN per (seed, event, region).
    Fresh holdout seeds: 21–40 (verified disjoint from seeds 1–20 used in N1/N2).

PRACTICAL EQUIVALENCE MARGIN (pre-declared):
    DELTA_PRACTICAL = 1e-5
    (same scale used as tie-margin in N2 candidate selection)

DECISION LOGIC (pre-declared):
    CASE 1 — A025_BETTER:   mean_diff > +1e-5, CI supports positive advantage
    CASE 2 — F005_BETTER:   mean_diff < -1e-5, CI supports negative direction
    CASE 3 — PRACTICALLY_EQUIVALENT: |mean_diff| <= 1e-5, no stable stratum advantage
    CASE 4 — INCONCLUSIVE:  mean near threshold or strata strongly disagree

ARCHITECTURE DECISION:
    A025_BETTER     → USE_A025_FOR_LONGITUDINAL_VALIDATION
    F005_BETTER     → USE_F005_FOR_LONGITUDINAL_VALIDATION
    PRACTICALLY_EQ  → USE_F005_FOR_LONGITUDINAL_VALIDATION  (parsimony)
    INCONCLUSIVE    → NO_ARCHITECTURE_DECISION

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged
    - Production predict() unmodified
    - region_protected_survival unmodified
    - nsga2_survival_traced unmodified
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged
    - Do NOT change 0.005, alpha=0.25, sigma_cap=0.03
    - Do NOT overwrite N1, N2 results
    - Do NOT start RL
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

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
SIGMA_CAP     = 0.03    # cap for A025
SIGMA_PROD    = 0.03    # production, unchanged
SIGMA_F005    = 0.005   # FROZEN — do not change
ALPHA_A025    = 0.25    # FROZEN — do not change
BUDGET        = e2p3.BUDGET       # 10
POP_SIZE      = e2p3.POP_SIZE     # 100
D             = e2p3.D             # 10
N_OBJ         = e2p3.N_OBJ        # 2
N_PARTITIONS  = e2p3.N_PARTITIONS  # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

# Pre-declared practical equivalence margin
DELTA_PRACTICAL = 1e-5

# Fresh holdout seeds — verified disjoint from N1/N2 seeds (1-20)
CONFIRMATION_SEEDS = list(range(21, 41))  # 21–40

# Bootstrap settings
BOOTSTRAP_SEED     = 42
BOOTSTRAP_RESAMPLES = 10_000

# Arms — F005 and A025 are primary; F0, F03 are optional reference
ARMS = [
    {"arm": "F005", "type": "fixed",    "sigma": SIGMA_F005,  "alpha": None},
    {"arm": "A025", "type": "adaptive", "sigma": None,         "alpha": ALPHA_A025},
    {"arm": "F0",   "type": "fixed",    "sigma": 0.000,        "alpha": None},
    {"arm": "F03",  "type": "fixed",    "sigma": SIGMA_PROD,   "alpha": None},
]
PRIMARY_ARMS  = ["F005", "A025"]
REFERENCE_ARMS = ["F0", "F03"]

# Helper aliases
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
# Frozen sigma formulas
# ============================================================

def sigma_f005():
    """Frozen F005 sigma = 0.005."""
    return SIGMA_F005


def sigma_a025(delta_hat_norm):
    """Frozen A025: min(0.03, 0.25 * ||delta_hat|| / sqrt(d)).

    Zero-motion rule: if delta_hat_norm < 1e-14, returns 0.

    Returns (sigma_used, sigma_raw, cap_active).
    """
    if delta_hat_norm < 1e-14:
        return 0.0, 0.0, False
    raw = ALPHA_A025 * delta_hat_norm / np.sqrt(D)
    cap = bool(raw > SIGMA_CAP)
    return float(min(raw, SIGMA_CAP)), float(raw), cap


# ============================================================
# Noise magnitude
# ============================================================

def rho_noise_metric(xis, sigma_used, delta_pop):
    """rho_noise = median(||sigma * xi||) / (||delta_hat|| + eps)."""
    xi_arr = np.asarray(xis, dtype=float)
    noise_norms = np.linalg.norm(sigma_used * xi_arr, axis=1)
    delta_norm  = float(np.linalg.norm(delta_pop))
    eps = 1e-12
    return {
        "median_noise_norm": float(np.median(noise_norms)),
        "mean_noise_norm":   float(np.mean(noise_norms)),
        "delta_hat_norm":    delta_norm,
        "rho_noise":         float(np.median(noise_norms)) / (delta_norm + eps),
    }


# ============================================================
# Child quality
# ============================================================

def child_metrics(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu,
                  F_children, pf_star):
    """Quality metrics for one arm's children."""
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    B = len(parent_idxs)
    n_clipped         = 0
    clipped_dims_list = []
    cosines           = []
    norm_ratios       = []
    delta_norm = float(np.linalg.norm(delta_pop))
    eps = 1e-12
    children_norm = []

    for pidx, xi in zip(parent_idxs, xis):
        z_p   = (X_region[pidx] - xl_a) / span
        z_raw = z_p + delta_pop + sigma_used * xi
        z_c   = np.clip(z_raw, 0.0, 1.0)
        children_norm.append(z_c)

        clipped_dims = int(np.sum((z_raw < 0.0) | (z_raw > 1.0)))
        if clipped_dims > 0:
            n_clipped += 1
        clipped_dims_list.append(clipped_dims)

        d_actual = z_c - z_p
        d_norm = float(np.linalg.norm(d_actual))
        if delta_norm > eps and d_norm > eps:
            cosines.append(float(np.dot(d_actual, delta_pop) / (d_norm * delta_norm)))
            norm_ratios.append(d_norm / delta_norm)

    clipping_frac = float(n_clipped) / B if B > 0 else float("nan")

    ch = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu)
    n_unique    = len(np.unique(ch, axis=0))
    unique_frac = float(n_unique) / B

    pf_med, _, _ = median_nearest_pf_distance(F_children, pf_star)

    disp_cosine = float(np.median(cosines))     if cosines     else float("nan")
    disp_ratio  = float(np.median(norm_ratios)) if norm_ratios else float("nan")

    # Pairwise child diversity (no dedup)
    dists = []
    for i in range(B):
        for j in range(i + 1, B):
            dists.append(float(np.linalg.norm(children_norm[i] - children_norm[j])))
    med_pw  = float(np.median(dists)) if dists else float("nan")
    mean_pw = float(np.mean(dists))   if dists else float("nan")

    return {
        "clipping_frac":                  clipping_frac,
        "mean_clipped_dims":              float(np.mean(clipped_dims_list)),
        "n_unique_children":              n_unique,
        "unique_child_fraction":          unique_frac,
        "child_pf_dist":                  pf_med,
        "median_displacement_cosine":     disp_cosine,
        "median_displacement_norm_ratio": disp_ratio,
        "median_pairwise_child_distance": med_pw,
        "mean_pairwise_child_distance":   mean_pw,
    }


# ============================================================
# Per-region N2C run — all arms
# ============================================================

def run_region_n2c(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
    arms=None,
):
    """Run all arms under HARD protection for region r.

    CRN: one (parent_idxs, xis) shared across all arms.
    Returns one row per arm; [] if empty; no-history marker if missing history.
    """
    if arms is None:
        arms = ARMS

    mask = region_ids == r
    n_r_before = int(np.sum(mask))
    if n_r_before == 0:
        return []

    X_region    = X_pop[mask].copy()
    N_total_old = len(X_pop)
    R           = len(V)

    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type  = str(labels_r[np.argmax(counts_r)])
    bins_in_r    = bin_of(X_region[:, 0], n_bins)
    bv, bc       = np.unique(bins_in_r, return_counts=True)
    dominant_bin = int(bv[np.argmax(bc)])

    motion = get_predict_motion_from_history(
        history, r, t=history_t + 1, lambda_pred=LAMBDA_PRED
    )
    if motion is None:
        return [{"has_history": False, "region_id": r, "event_id": change_idx}]

    delta_pop      = motion["delta_hat"]
    delta_hat_norm = float(np.linalg.norm(delta_pop))

    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])
    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # CRN — one sample set for ALL arms
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    parent_idxs, xis = generate_base_samples(BUDGET, D, rng_shared, n_r_before)

    F_minus_a  = np.asarray(F_minus, dtype=float)
    F_all_norm = np.vstack([F_minus_a, F_before])
    z_m   = F_all_norm.min(axis=0)
    z_nad = F_all_norm.max(axis=0)
    norm_span = (z_nad - z_m) + 1e-12

    pf_star_arr = np.asarray(pf_star, dtype=float)
    pf_norm     = (pf_star_arr - z_m) / norm_span
    pf_ref_rids = np.asarray(partition(pf_norm, V), dtype=int)
    occ_before  = np.bincount(region_ids, minlength=R)

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
        mdi = float(np.mean(delta_d_arr[inside_d]))  if np.any(inside_d)  else float("nan")
        mdo = float(np.mean(delta_d_arr[outside_d])) if np.any(outside_d) else float("nan")
        frac_worse = float(np.mean(delta_d_arr > 1e-14))
        frac_impr  = float(np.mean(delta_d_arr < -1e-14))

        F_s_norm   = (F_surv - z_m) / norm_span
        actual_rids = np.asarray(partition(F_s_norm, V), dtype=int)
        actual_occ  = np.bincount(actual_rids, minlength=R)
        resp_delta  = int(actual_occ[r]) - int(occ_before[r])
        diff_other  = actual_occ - occ_before; diff_other[r] = 0
        n_other_lose = int(np.sum(diff_other < 0))

        return {
            "igd_after":               igd_after,
            "delta_igd":               delta_igd,
            "mean_delta_pf_inside":    mdi,
            "mean_delta_pf_outside":   mdo,
            "frac_pf_worsened":        frac_worse,
            "frac_pf_improved":        frac_impr,
            "actual_resp_region_delta": resp_delta,
            "actual_n_other_losing":   n_other_lose,
            "igd_identity_residual":   float(identity_residual),
        }

    rows = []

    for arm_def in arms:
        arm_name  = arm_def["arm"]
        arm_type  = arm_def["type"]
        arm_alpha = arm_def["alpha"]

        if arm_type == "fixed":
            sigma_used  = float(arm_def["sigma"])
            sigma_raw   = None
            cap_active  = None
        else:
            # Adaptive: A025
            sigma_used, sigma_raw, cap_active = sigma_a025(delta_hat_norm)

        ch   = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu)
        F_ch = cp_evaluate(ch, t_new, region_dynamics)

        nm = rho_noise_metric(xis, sigma_used, delta_pop)
        cm = child_metrics(parent_idxs, xis, X_region, delta_pop, sigma_used,
                           xl, xu, F_ch, pf_star)

        X_s, F_s, idx_s = region_protected_survival(
            X_pop, F_before, ch, F_ch,
            r, region_ids, n_r_before, xl, xu, surv_seed
        )
        cov = _coverage(F_s)

        surv_old_set  = set(int(i) for i in idx_s if i < N_total_old)
        dropped_old   = set(range(N_total_old)) - surv_old_set
        n_old_drop_other = sum(1 for i in dropped_old if region_ids[i] != r)
        assert n_old_drop_other == 0, (
            f"HARD protection dropped {n_old_drop_other} outside-region OLD at arm={arm_name}")
        n_ch_surv    = int(np.sum(np.asarray(idx_s) >= N_total_old))
        ch_surv_rate = float(n_ch_surv) / BUDGET

        row = {
            "has_history":            True,
            "arm":                    arm_name,
            "arm_type":               arm_type,
            "alpha":                  arm_alpha,
            "sigma_used":             sigma_used,
            "sigma_raw":              sigma_raw,
            "cap_active":             cap_active,
            "motion_norm":            delta_hat_norm,
            "n_r_before":             n_r_before,
            "region_type":            region_type,
            "dominant_bin":           dominant_bin,
            "fe":                     BUDGET,
            "igd_before":             igd_before,

            # Estimator
            "delta_hat_norm":         nm["delta_hat_norm"],
            **diag,

            # Noise
            "rho_noise":              nm["rho_noise"],
            "median_noise_norm":      nm["median_noise_norm"],
            "mean_noise_norm":        nm["mean_noise_norm"],

            # Child quality
            "clipping_frac":                  cm["clipping_frac"],
            "mean_clipped_dims":              cm["mean_clipped_dims"],
            "n_unique_children":              cm["n_unique_children"],
            "unique_child_fraction":          cm["unique_child_fraction"],
            "child_pf_dist":                  cm["child_pf_dist"],
            "median_displacement_cosine":     cm["median_displacement_cosine"],
            "median_displacement_norm_ratio": cm["median_displacement_norm_ratio"],
            "median_pairwise_child_distance": cm["median_pairwise_child_distance"],
            "mean_pairwise_child_distance":   cm["mean_pairwise_child_distance"],

            # Survival
            "n_ch_survived":          n_ch_surv,
            "child_survival_rate":    ch_surv_rate,
            "n_old_dropped_other":    n_old_drop_other,

            # Population response
            **cov,
        }
        rows.append(row)

    return rows


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_n2c(seed, n_scored_changes, scenario_name, arms=None):
    """Run N2C for one seed. Returns list of row dicts."""
    if arms is None:
        arms = ARMS

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
            r_rows = run_region_n2c(
                r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
                region_dynamics, n_bins, pf_star, V,
                xl, xu, seed, change_idx,
                history, history_t,
                arms=arms,
            )
            for row in r_rows:
                if row.get("has_history") is False:
                    rows_out.append({
                        "has_history": False,
                        "scenario":    scenario_name,
                        "seed":        seed,
                        "event_id":    change_idx,
                        "region_id":   row["region_id"],
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
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows_out


# ============================================================
# Primary paired endpoint
# ============================================================

def compute_primary_endpoint(valid_rows):
    """Compute paired A025 - F005 differences per matched (seed, event, region).

    Returns dict with full paired analysis.
    """
    import collections
    by_key = collections.defaultdict(dict)
    for r in valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    diffs = []
    f005_digds = []
    a025_digds = []

    for key, arm_map in by_key.items():
        if "F005" not in arm_map or "A025" not in arm_map:
            continue
        f005 = arm_map["F005"]["delta_igd"]
        a025 = arm_map["A025"]["delta_igd"]
        diffs.append(a025 - f005)
        f005_digds.append(f005)
        a025_digds.append(a025)

    if not diffs:
        return {"n_pairs": 0}

    arr = np.asarray(diffs)
    n   = len(arr)

    # Category counts
    n_a025_better = int(np.sum(arr > DELTA_PRACTICAL))
    n_f005_better = int(np.sum(arr < -DELTA_PRACTICAL))
    n_tied        = int(np.sum(np.abs(arr) <= DELTA_PRACTICAL))

    # IQR
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    # Paired t-test (secondary)
    t_stat, t_pval = scipy_stats.ttest_1samp(arr, 0.0)

    # Wilcoxon (secondary)
    try:
        w_stat, w_pval = scipy_stats.wilcoxon(arr)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    return {
        "n_pairs":       n,
        "diffs":         diffs,
        "mean_diff":     float(np.mean(arr)),
        "median_diff":   float(np.median(arr)),
        "std_diff":      float(np.std(arr)),
        "iqr_diff":      float(q75 - q25),
        "q25":           float(q25),
        "q75":           float(q75),
        "frac_a025_better": float(n_a025_better) / n,
        "frac_tied":        float(n_tied) / n,
        "frac_f005_better": float(n_f005_better) / n,
        "frac_a025_wins_any": float(np.sum(arr > 0)) / n,
        "frac_f005_wins_any": float(np.sum(arr < 0)) / n,
        "t_stat":        float(t_stat),
        "t_pval":        float(t_pval),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
        "f005_mean_digd": float(np.mean(f005_digds)),
        "a025_mean_digd": float(np.mean(a025_digds)),
    }


# ============================================================
# Paired bootstrap CI
# ============================================================

def bootstrap_ci(diffs, seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_RESAMPLES):
    """Paired 95% bootstrap CI for the mean difference.

    Resamples matched pairs (not arms independently).

    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    n   = len(arr)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(arr[idx])
    ci_low  = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return float(np.mean(arr)), ci_low, ci_high


# ============================================================
# Motion-magnitude stratification
# ============================================================

def motion_strata(valid_rows, n_strata=4):
    """Stratify by motion_norm quartiles.

    Returns list of dicts, one per stratum.
    """
    import collections

    # Compute quartile boundaries from matched primary pairs
    by_key = collections.defaultdict(dict)
    for r in valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    motion_norms = []
    for key, arm_map in by_key.items():
        if "F005" in arm_map and "A025" in arm_map:
            motion_norms.append(arm_map["F005"]["motion_norm"])

    if not motion_norms:
        return []

    boundaries = np.percentile(motion_norms, [25, 50, 75])
    strata_results = []

    for q_idx in range(n_strata):
        lo = 0.0 if q_idx == 0 else boundaries[q_idx - 1]
        hi = float("inf") if q_idx == n_strata - 1 else boundaries[q_idx]
        label = f"Q{q_idx + 1}"

        stratum_diffs = []
        f005_digd = []
        a025_digd = []
        f005_rho  = []
        a025_rho  = []
        f005_clip = []
        a025_clip = []
        motion_vals = []

        for key, arm_map in by_key.items():
            if "F005" not in arm_map or "A025" not in arm_map:
                continue
            mn = arm_map["F005"]["motion_norm"]
            if not (lo <= mn <= hi if q_idx < n_strata - 1 else mn > lo):
                continue
            if q_idx == 0 and mn > boundaries[0]:
                continue
            if q_idx > 0 and mn <= boundaries[q_idx - 1]:
                continue

            stratum_diffs.append(arm_map["A025"]["delta_igd"] - arm_map["F005"]["delta_igd"])
            f005_digd.append(arm_map["F005"]["delta_igd"])
            a025_digd.append(arm_map["A025"]["delta_igd"])
            f005_rho.append(arm_map["F005"]["rho_noise"])
            a025_rho.append(arm_map["A025"]["rho_noise"])
            f005_clip.append(arm_map["F005"]["clipping_frac"])
            a025_clip.append(arm_map["A025"]["clipping_frac"])
            motion_vals.append(mn)

        if not stratum_diffs:
            strata_results.append({"label": label, "n": 0})
            continue

        sa = np.asarray(stratum_diffs)
        strata_results.append({
            "label":          label,
            "n":              len(sa),
            "motion_norm_med": float(np.median(motion_vals)),
            "mean_diff":      float(np.mean(sa)),
            "f005_mean_digd": float(np.mean(f005_digd)),
            "a025_mean_digd": float(np.mean(a025_digd)),
            "frac_a025_wins": float(np.mean(sa > 0)),
            "f005_median_rho": float(np.median(f005_rho)),
            "a025_median_rho": float(np.median(a025_rho)),
            "f005_median_clip": float(np.median(f005_clip)),
            "a025_median_clip": float(np.median(a025_clip)),
        })

    return strata_results


# ============================================================
# Estimator quality stratification
# ============================================================

def estimator_strata(valid_rows, n_strata=4):
    """Stratify by displacement cosine quartiles."""
    import collections

    by_key = collections.defaultdict(dict)
    for r in valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    cosines = []
    for key, arm_map in by_key.items():
        if "F005" in arm_map:
            c = arm_map["F005"].get("median_displacement_cosine", float("nan"))
            if not np.isnan(c):
                cosines.append(c)

    if not cosines:
        return []

    boundaries = np.percentile(cosines, [25, 50, 75])
    strata_results = []

    for q_idx in range(n_strata):
        label = f"Q{q_idx + 1}_cosine"
        stratum_diffs = []
        f005_digd = []
        a025_digd = []
        f005_clip = []
        a025_clip = []
        f005_rho  = []
        a025_rho  = []
        child_pf_f005 = []
        child_pf_a025 = []

        for key, arm_map in by_key.items():
            if "F005" not in arm_map or "A025" not in arm_map:
                continue
            c = arm_map["F005"].get("median_displacement_cosine", float("nan"))
            if np.isnan(c):
                continue
            if q_idx == 0 and c > boundaries[0]:
                continue
            if q_idx > 0 and c <= boundaries[q_idx - 1]:
                continue
            if q_idx < n_strata - 1 and c > boundaries[q_idx]:
                continue

            stratum_diffs.append(arm_map["A025"]["delta_igd"] - arm_map["F005"]["delta_igd"])
            f005_digd.append(arm_map["F005"]["delta_igd"])
            a025_digd.append(arm_map["A025"]["delta_igd"])
            f005_clip.append(arm_map["F005"]["clipping_frac"])
            a025_clip.append(arm_map["A025"]["clipping_frac"])
            f005_rho.append(arm_map["F005"]["rho_noise"])
            a025_rho.append(arm_map["A025"]["rho_noise"])
            child_pf_f005.append(arm_map["F005"]["child_pf_dist"])
            child_pf_a025.append(arm_map["A025"]["child_pf_dist"])

        if not stratum_diffs:
            strata_results.append({"label": label, "n": 0})
            continue

        sa = np.asarray(stratum_diffs)
        strata_results.append({
            "label":          label,
            "n":              len(sa),
            "mean_diff":      float(np.mean(sa)),
            "f005_mean_digd": float(np.mean(f005_digd)),
            "a025_mean_digd": float(np.mean(a025_digd)),
            "frac_a025_wins": float(np.mean(sa > 0)),
            "f005_median_clip": float(np.median(f005_clip)),
            "a025_median_clip": float(np.median(a025_clip)),
            "f005_median_rho":  float(np.median(f005_rho)),
            "a025_median_rho":  float(np.median(a025_rho)),
            "f005_child_pf":   float(np.mean(child_pf_f005)),
            "a025_child_pf":   float(np.mean(child_pf_a025)),
        })

    return strata_results


# ============================================================
# Per-event and per-seed analysis
# ============================================================

def per_event_analysis(valid_rows, event_ids):
    import collections
    by_key = collections.defaultdict(dict)
    for r in valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    results = {}
    for eid in event_ids:
        diffs = []
        f005_digd = []
        a025_digd = []
        for key, arm_map in by_key.items():
            if key[1] != eid:
                continue
            if "F005" not in arm_map or "A025" not in arm_map:
                continue
            diffs.append(arm_map["A025"]["delta_igd"] - arm_map["F005"]["delta_igd"])
            f005_digd.append(arm_map["F005"]["delta_igd"])
            a025_digd.append(arm_map["A025"]["delta_igd"])
        if diffs:
            sa = np.asarray(diffs)
            results[eid] = {
                "n":              len(sa),
                "mean_diff":      float(np.mean(sa)),
                "f005_mean_digd": float(np.mean(f005_digd)),
                "a025_mean_digd": float(np.mean(a025_digd)),
                "frac_a025_wins": float(np.mean(sa > 0)),
            }
        else:
            results[eid] = {"n": 0}
    return results


def per_seed_analysis(valid_rows, seeds):
    import collections
    by_key = collections.defaultdict(dict)
    for r in valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    results = {}
    for seed in seeds:
        diffs = []
        for key, arm_map in by_key.items():
            if key[0] != seed:
                continue
            if "F005" not in arm_map or "A025" not in arm_map:
                continue
            diffs.append(arm_map["A025"]["delta_igd"] - arm_map["F005"]["delta_igd"])
        if diffs:
            sa = np.asarray(diffs)
            results[seed] = {
                "n":              len(sa),
                "mean_diff":      float(np.mean(sa)),
                "frac_a025_wins": float(np.mean(sa > 0)),
            }
        else:
            results[seed] = {"n": 0}
    return results


# ============================================================
# Decision logic
# ============================================================

def decide_head_to_head(mean_diff, ci_low, ci_high, motion_strata_result):
    """Apply pre-declared decision logic. Returns (head_to_head, reasoning)."""
    reasoning = {
        "mean_diff":      mean_diff,
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "delta_practical": DELTA_PRACTICAL,
    }

    # Check stratum stability
    stratum_diffs = [s.get("mean_diff", 0.0)
                     for s in motion_strata_result if s.get("n", 0) >= 5]
    strata_disagree = (
        len(stratum_diffs) >= 2 and
        max(stratum_diffs) > DELTA_PRACTICAL and
        min(stratum_diffs) < -DELTA_PRACTICAL
    )
    reasoning["strata_disagree"] = strata_disagree

    # Case 1: A025 clearly better
    if (mean_diff > DELTA_PRACTICAL and
            ci_low > 0 and not strata_disagree):
        reasoning["decision_case"] = 1
        return "A025_BETTER", reasoning

    # Case 2: F005 clearly better
    if (mean_diff < -DELTA_PRACTICAL and
            ci_high < 0 and not strata_disagree):
        reasoning["decision_case"] = 2
        return "F005_BETTER", reasoning

    # Case 3: Practically equivalent
    if abs(mean_diff) <= DELTA_PRACTICAL and not strata_disagree:
        reasoning["decision_case"] = 3
        return "PRACTICALLY_EQUIVALENT", reasoning

    # Case 4: Inconclusive
    reasoning["decision_case"] = 4
    return "INCONCLUSIVE", reasoning


def decide_architecture(head_to_head):
    """Map head-to-head verdict to architecture decision."""
    if head_to_head in ("A025_BETTER",):
        return "USE_A025_FOR_LONGITUDINAL_VALIDATION"
    if head_to_head in ("F005_BETTER", "PRACTICALLY_EQUIVALENT"):
        return "USE_F005_FOR_LONGITUDINAL_VALIDATION"
    return "NO_ARCHITECTURE_DECISION"


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
    if isinstance(obj, bool):
        return bool(obj)
    return obj


if __name__ == "__main__":
    import csv

    SCENARIO = "PERSISTENT"
    N_SCORED = 5
    OUT_DIR  = Path("results/N2C_head_to_head_confirmation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "N2C_head_to_head_confirmation.jsonl"

    all_rows  = []
    n_valid   = 0
    n_missing = 0

    print("=== N2C: Head-to-Head Confirmation (seeds 21-40) ===")
    print(f"Arms: {[a['arm'] for a in ARMS]}")
    print(f"Primary: F005 vs A025  |  DELTA_PRACTICAL = {DELTA_PRACTICAL}")
    t0 = time.time()

    for seed in CONFIRMATION_SEEDS:
        t1 = time.time()
        rows = run_one_n2c(seed, N_SCORED, SCENARIO, ARMS)
        all_rows.extend(rows)
        sv = sum(1 for r in rows if r.get("has_history", True) and "arm" in r)
        ms = sum(1 for r in rows if not r.get("has_history", True))
        n_valid   += sv
        n_missing += ms
        print(f"  seed={seed}: {sv} valid rows, {ms} skipped in {time.time()-t1:.1f}s")

    # ─── Primary paired analysis ──────────────────────────────────────────
    valid = [r for r in all_rows if r.get("has_history", True) and "arm" in r]

    print(f"\nTotal valid rows: {len(valid)} ({n_valid} across all seeds)")
    endpoint = compute_primary_endpoint(valid)

    if endpoint["n_pairs"] > 0:
        diffs = endpoint["diffs"]
        mean_d, ci_lo, ci_hi = bootstrap_ci(diffs)
        print(f"\nPrimary endpoint ({endpoint['n_pairs']} pairs):")
        print(f"  mean_diff    = {mean_d:.8f}")
        print(f"  bootstrap CI = [{ci_lo:.8f}, {ci_hi:.8f}]")
        print(f"  A025 clearly better frac = {endpoint['frac_a025_better']:.3f}")
        print(f"  practical tie frac       = {endpoint['frac_tied']:.3f}")
        print(f"  F005 clearly better frac = {endpoint['frac_f005_better']:.3f}")

        # Stratified analyses
        mstrata = motion_strata(valid)
        estrata = estimator_strata(valid)
        event_analysis = per_event_analysis(valid, [2, 3, 4, 5, 6])
        seed_analysis  = per_seed_analysis(valid, CONFIRMATION_SEEDS)

        head_to_head, reasoning = decide_head_to_head(mean_d, ci_lo, ci_hi, mstrata)
        arch_decision = decide_architecture(head_to_head)

        print(f"\nHEAD_TO_HEAD:         {head_to_head}")
        print(f"ARCHITECTURE_DECISION: {arch_decision}")
    else:
        head_to_head  = "INCONCLUSIVE"
        arch_decision = "NO_ARCHITECTURE_DECISION"
        mean_d = ci_lo = ci_hi = float("nan")
        mstrata = estrata = []
        event_analysis = seed_analysis = {}
        reasoning = {}

    # ─── Save JSONL ───────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | {n_valid} valid, {n_missing} skipped "
          f"| {len(all_rows)} rows → {jsonl_path}")

    # ─── Save results JSON ────────────────────────────────────────────────
    results_summary = {
        "head_to_head":         head_to_head,
        "architecture_decision": arch_decision,
        "primary_endpoint":     {k: v for k, v in endpoint.items() if k != "diffs"},
        "bootstrap_mean":       mean_d,
        "bootstrap_ci_low":     ci_lo,
        "bootstrap_ci_high":    ci_hi,
        "decision_reasoning":   reasoning,
        "motion_strata":        mstrata,
        "estimator_strata":     estrata,
        "event_analysis":       event_analysis,
        "seed_analysis":        {str(k): v for k, v in seed_analysis.items()},
    }
    with open(OUT_DIR / "N2C_results.json", "w") as f:
        json.dump(json_safe(results_summary), f, indent=2)
    print("Results JSON saved.")
