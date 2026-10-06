"""
N2 — Signal-Scaled Adaptive Predict Noise.

Scientific question:
    Does scaling noise to the predicted motion magnitude (sigma_r = min(CAP,
    alpha * ||delta_hat|| / sqrt(d))) improve region response compared to
    fixed sigma choices, including the best fixed candidate from N1?

DESIGN:
    Six arms, all HARD-protected, CRN across all arms:
        F0   — fixed sigma = 0.000
        F005 — fixed sigma = 0.005  (N1 "clipping-diversity" candidate)
        F03  — fixed sigma = 0.030  (production)
        A025 — adaptive alpha = 0.25
        A050 — adaptive alpha = 0.50
        A100 — adaptive alpha = 1.00

    Adaptive formula: sigma_r = min(SIGMA_CAP, alpha * ||delta_hat|| / sqrt(d))
    Zero-motion rule: if ||delta_hat|| < 1e-14, sigma_r = 0 (no noise floor).

    CRN: generate (parent_idxs, xis) ONCE per (seed, event, region).
         All six arms share identical parents and xi vectors.
         sigma_r must NOT depend on xi — xi is the noise direction only.

TWO-PHASE DISCOVERY / HOLDOUT:
    N2-A — Discovery:  seeds 1–10, 5 scored events, all 6 arms.
    N2-B — Holdout:    seeds 11–20, 5 scored events, [F0, candidate, F03].
                       Only run if a clear adaptive candidate is identified from N2-A.

CANDIDATE ELIGIBILITY (pre-declared, all four must pass):
    E1: mean delta_igd > 0
    E2: median rho_noise < 1.5
    E3: median clipping_frac < 0.70
    E4: mean D_A_03 > 0   (adaptive arm mean delta_igd > F03 mean delta_igd)

    Selection: highest mean_delta_igd among eligible adaptive arms.
    Tie-break: smaller alpha if mean delta_igd difference < 1e-5.

THREE VERDICTS:
    ADAPTIVE_SIGNAL_SCALING:   BETTER / COMPARABLE / WORSE   (best adaptive vs F005)
    FIXED_005_STATUS:          VIABLE / MARGINAL / NOT_VIABLE
    PREDICT_NOISE_ARCHITECTURE: ADAPTIVE_PREFERRED / FIXED_005_PREFERRED /
                                 NO_CLEAR_WINNER / NO_CANDIDATE

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged.
    - Production predictor (predict.py) unmodified.
    - region_protected_survival unmodified.
    - nsga2_survival_traced unmodified.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - No GT information enters child construction.
    - Missing history → has_history=False, skip.
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
SIGMA_CAP     = 0.03    # adaptive noise ceiling — same as production sigma_P
SIGMA_PROD    = 0.03    # production value — never change
BUDGET        = e2p3.BUDGET       # 10
POP_SIZE      = e2p3.POP_SIZE     # 100
D             = e2p3.D             # 10
N_OBJ         = e2p3.N_OBJ        # 2
N_PARTITIONS  = e2p3.N_PARTITIONS  # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

# Pre-declared alpha values — NOT tuned after seeing results
ALPHA_VALUES = [0.25, 0.50, 1.00]

# Six arms — order determines iteration and reporting order
ARMS = [
    {"arm": "F0",   "type": "fixed",    "sigma": 0.000, "alpha": None},
    {"arm": "F005", "type": "fixed",    "sigma": 0.005, "alpha": None},
    {"arm": "F03",  "type": "fixed",    "sigma": 0.030, "alpha": None},
    {"arm": "A025", "type": "adaptive", "sigma": None,  "alpha": 0.25},
    {"arm": "A050", "type": "adaptive", "sigma": None,  "alpha": 0.50},
    {"arm": "A100", "type": "adaptive", "sigma": None,  "alpha": 1.00},
]
ARM_NAMES = [a["arm"] for a in ARMS]
ADAPTIVE_ARMS = [a["arm"] for a in ARMS if a["type"] == "adaptive"]
FIXED_ARMS    = [a["arm"] for a in ARMS if a["type"] == "fixed"]

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
# Adaptive sigma formula
# ============================================================

def adaptive_sigma(alpha, delta_hat_norm, sigma_cap=SIGMA_CAP):
    """Compute adaptive sigma_r = min(sigma_cap, alpha * ||delta_hat|| / sqrt(d)).

    Zero-motion rule: if delta_hat_norm < 1e-14, return (0.0, 0.0, False).

    Returns:
        sigma_used  — the sigma applied to children
        sigma_raw   — alpha * norm / sqrt(d) before capping
        cap_active  — True if sigma_raw > sigma_cap
    """
    if delta_hat_norm < 1e-14:
        return 0.0, 0.0, False
    sigma_raw = alpha * delta_hat_norm / np.sqrt(D)
    cap_active = bool(sigma_raw > sigma_cap)
    return float(min(sigma_raw, sigma_cap)), float(sigma_raw), cap_active


# ============================================================
# Pairwise child diversity
# ============================================================

def pairwise_child_diversity(children):
    """Median and mean pairwise distance in normalized [0,1] space.

    No deduplication before computing: repeated children contribute 0-distance pairs.

    Args:
        children: (B, D) array in [0,1] space

    Returns:
        (median_pairwise, mean_pairwise)
    """
    ch = np.asarray(children, dtype=float)
    B = len(ch)
    dists = []
    for i in range(B):
        for j in range(i + 1, B):
            dists.append(float(np.linalg.norm(ch[i] - ch[j])))
    if not dists:
        return float("nan"), float("nan")
    return float(np.median(dists)), float(np.mean(dists))


# ============================================================
# Noise magnitude helpers (re-used from N1 pattern)
# ============================================================

def rho_noise_metric(xis, sigma_used, delta_pop):
    """Compute noise-to-motion ratio.

    rho_noise = median(||sigma_used * xi||) / (||delta_hat|| + eps)
    """
    xi_arr = np.asarray(xis, dtype=float)
    noise_vecs  = sigma_used * xi_arr        # (B, D)
    noise_norms = np.linalg.norm(noise_vecs, axis=1)
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

def child_metrics(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu,
                  F_children, pf_star):
    """Per-child and aggregate quality metrics for one (arm, region, event)."""
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

    for i, (pidx, xi) in enumerate(zip(parent_idxs, xis)):
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
            cos = float(np.dot(d_actual, delta_pop) / (d_norm * delta_norm))
            cosines.append(cos)
            norm_ratios.append(d_norm / delta_norm)

    clipping_frac = float(n_clipped) / B if B > 0 else float("nan")

    # Unique children count
    ch = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu)
    n_unique    = len(np.unique(ch, axis=0))
    unique_frac = float(n_unique) / B

    # PF distance (median)
    pf_med, _, _ = median_nearest_pf_distance(F_children, pf_star)

    disp_cosine = float(np.median(cosines))     if cosines     else float("nan")
    disp_ratio  = float(np.median(norm_ratios)) if norm_ratios else float("nan")

    # Pairwise child diversity (no dedup)
    med_pw, mean_pw = pairwise_child_diversity(children_norm)

    return {
        "clipping_frac":                 clipping_frac,
        "mean_clipped_dims":             float(np.mean(clipped_dims_list)),
        "max_clipped_dims":              int(max(clipped_dims_list)),
        "n_unique_children":             n_unique,
        "unique_child_fraction":         unique_frac,
        "child_pf_dist":                 pf_med,
        "median_displacement_cosine":    disp_cosine,
        "median_displacement_norm_ratio": disp_ratio,
        "median_pairwise_child_distance": med_pw,
        "mean_pairwise_child_distance":  mean_pw,
    }


# ============================================================
# Per-region N2 run — all six arms
# ============================================================

def run_region_n2(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
    arms=None,
):
    """Run all six arms under HARD protection for region r.

    CRN: one set of (parent_idxs, xis) shared across all arms.
    Returns one row per arm.
    Returns [] if region is empty; single no-history marker if missing history.
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

    delta_pop = motion["delta_hat"]  # (D,) in [0,1]-normalized space
    delta_hat_norm = float(np.linalg.norm(delta_pop))

    # GT delta (diagnostics only — NOT used in child generation)
    n_dist = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])
    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # ─── CRN: generate base samples ONCE for all six arms ─────────────────
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

    # Shared survival seed across all arms for same (seed, event, region)
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

    # ─── Main arm loop ────────────────────────────────────────────────────
    rows = []

    for arm_def in arms:
        arm_name  = arm_def["arm"]
        arm_type  = arm_def["type"]
        arm_alpha = arm_def["alpha"]

        # Compute sigma for this arm
        if arm_type == "fixed":
            sigma_used  = float(arm_def["sigma"])
            sigma_raw   = None
            cap_active  = None
        else:
            sigma_used, sigma_raw, cap_active = adaptive_sigma(
                arm_alpha, delta_hat_norm
            )

        # Generate children (all arms use same parent_idxs and xis)
        ch   = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma_used, xl, xu)
        F_ch = cp_evaluate(ch, t_new, region_dynamics)

        # Noise and child metrics
        nm = rho_noise_metric(xis, sigma_used, delta_pop)
        cm = child_metrics(parent_idxs, xis, X_region, delta_pop, sigma_used,
                           xl, xu, F_ch, pf_star)

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
            "n_r_before":             n_r_before,
            "region_type":            region_type,
            "dominant_bin":           dominant_bin,
            "fe":                     BUDGET,
            "igd_before":             igd_before,

            # Estimator
            "delta_hat_norm":         nm["delta_hat_norm"],
            **diag,

            # Noise magnitude
            "rho_noise":              nm["rho_noise"],
            "median_noise_norm":      nm["median_noise_norm"],
            "mean_noise_norm":        nm["mean_noise_norm"],

            # Child quality
            "clipping_frac":                  cm["clipping_frac"],
            "mean_clipped_dims":              cm["mean_clipped_dims"],
            "max_clipped_dims":               cm["max_clipped_dims"],
            "n_unique_children":              cm["n_unique_children"],
            "unique_child_fraction":          cm["unique_child_fraction"],
            "child_pf_dist":                  cm["child_pf_dist"],
            "median_displacement_cosine":     cm["median_displacement_cosine"],
            "median_displacement_norm_ratio": cm["median_displacement_norm_ratio"],
            "median_pairwise_child_distance": cm["median_pairwise_child_distance"],
            "mean_pairwise_child_distance":   cm["mean_pairwise_child_distance"],

            # HARD survival
            "n_ch_survived":          n_ch_surv,
            "child_survival_rate":    ch_surv_rate,
            "n_old_dropped_other":    n_old_drop_other,

            # Population metrics
            **cov,
        }
        rows.append(row)

    return rows


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_n2(seed, n_scored_changes, scenario_name, arms=None, split="discovery"):
    """Run N2 for one seed. Returns list of row dicts (one per arm per region-event)."""
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
            r_rows = run_region_n2(
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
                        "split":       split,
                        "scenario":    scenario_name,
                        "seed":        seed,
                        "event_id":    change_idx,
                        "region_id":   row["region_id"],
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
# Candidate selection (N2-A → N2-B)
# ============================================================

def select_candidate(discovery_rows, arms=None):
    """Identify adaptive holdout candidate from N2-A discovery rows.

    Pre-declared eligibility (E1-E4, all must pass for an adaptive arm):
        E1: mean delta_igd > 0
        E2: median rho_noise < 1.5
        E3: median clipping_frac < 0.70
        E4: mean D_A_03 > 0  (mean delta_igd(arm) - mean delta_igd(F03) > 0)

    Selection: highest mean delta_igd among eligible adaptive arms.
    Tie-break: smaller alpha if difference < 1e-5.

    Returns (arm_candidate, reasoning_dict) or (None, reasoning_dict).
    """
    if arms is None:
        arms = ARMS

    valid = [r for r in discovery_rows
             if r.get("has_history", True) and "delta_igd" in r
             and r.get("split") == "discovery"]

    import collections
    # Group rows by (seed, event_id, region_id) → arm → row
    by_key = collections.defaultdict(dict)
    for row in valid:
        key = (row["seed"], row["event_id"], row["region_id"])
        by_key[key][row["arm"]] = row

    # Per-arm statistics
    arm_stats = {}
    for arm_def in arms:
        arm_name = arm_def["arm"]
        digd_list    = []
        rho_list     = []
        clipping_list = []
        d_a03_list   = []

        for key, arm_map in by_key.items():
            if arm_name not in arm_map:
                continue
            row = arm_map[arm_name]
            digd_list.append(row["delta_igd"])
            rho_list.append(row["rho_noise"])
            clipping_list.append(row["clipping_frac"])
            # D_A_03 = delta_igd(arm) - delta_igd(F03)
            if "F03" in arm_map:
                d_a03_list.append(row["delta_igd"] - arm_map["F03"]["delta_igd"])

        if not digd_list:
            continue

        arm_stats[arm_name] = {
            "n_rows":           len(digd_list),
            "mean_delta_igd":   float(np.mean(digd_list)),
            "median_delta_igd": float(np.median(digd_list)),
            "median_rho":       float(np.median(rho_list)),
            "median_clipping":  float(np.median(clipping_list)),
            "frac_digd_pos":    float(np.mean(np.array(digd_list) > 0)),
            "n_d_a03_pairs":    len(d_a03_list),
            "mean_D_A_03":      float(np.mean(d_a03_list)) if d_a03_list else float("nan"),
        }

    reasoning = {"arm_stats": arm_stats}

    # Apply eligibility to adaptive arms only
    eligible = []
    for arm_def in arms:
        arm_name = arm_def["arm"]
        if arm_def["type"] != "adaptive":
            continue
        if arm_name not in arm_stats:
            continue
        s = arm_stats[arm_name]
        e1 = s["mean_delta_igd"] > 0.0
        e2 = s["median_rho"]      < 1.5
        e3 = s["median_clipping"] < 0.70
        e4 = (not np.isnan(s["mean_D_A_03"])) and (s["mean_D_A_03"] > 0.0)
        reasoning[f"{arm_name}_e1"] = e1
        reasoning[f"{arm_name}_e2"] = e2
        reasoning[f"{arm_name}_e3"] = e3
        reasoning[f"{arm_name}_e4"] = e4
        if e1 and e2 and e3 and e4:
            eligible.append(arm_name)

    if not eligible:
        reasoning["candidate"] = None
        reasoning["reason_no_candidate"] = (
            "No adaptive arm satisfies all four pre-declared eligibility criteria."
        )
        return None, reasoning

    # Selection: highest mean delta_igd; tie-break smallest alpha
    def _sort_key(arm_name):
        mean_d = arm_stats[arm_name]["mean_delta_igd"]
        alpha  = next(a["alpha"] for a in arms if a["arm"] == arm_name)
        return (-mean_d, alpha)  # negate so highest mean_d sorts first

    eligible_sorted = sorted(eligible, key=_sort_key)

    # Check for near-tie among top two
    best = eligible_sorted[0]
    if len(eligible_sorted) >= 2:
        second = eligible_sorted[1]
        diff = abs(arm_stats[best]["mean_delta_igd"] - arm_stats[second]["mean_delta_igd"])
        if diff < 1e-5:
            # Use smaller alpha as tie-break
            best_alpha   = next(a["alpha"] for a in arms if a["arm"] == best)
            second_alpha = next(a["alpha"] for a in arms if a["arm"] == second)
            if second_alpha < best_alpha:
                best = second
            reasoning["tie_break_applied"] = True
        else:
            reasoning["tie_break_applied"] = False
    else:
        reasoning["tie_break_applied"] = False

    reasoning["candidate"] = best
    reasoning["n_eligible"]  = len(eligible)
    reasoning["eligible_arms"] = eligible
    return best, reasoning


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
    OUT_DIR  = Path("results/N2_signal_scaled_noise")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "N2_signal_scaled_noise.jsonl"

    all_rows  = []
    n_valid   = 0
    n_missing = 0

    # ─── N2-A: Discovery ──────────────────────────────────────────────────
    print("=== N2-A: Discovery ===")
    t0 = time.time()
    for seed in DISCOVERY_SEEDS:
        t1 = time.time()
        rows = run_one_n2(seed, N_SCORED, SCENARIO, ARMS, split="discovery")
        all_rows.extend(rows)
        sv = sum(1 for r in rows if r.get("has_history", True) and "arm" in r)
        ms = sum(1 for r in rows if not r.get("has_history", True))
        n_valid   += sv
        n_missing += ms
        print(f"  seed={seed}: {sv} valid rows, {ms} skipped in {time.time()-t1:.1f}s")

    # ─── Select candidate ─────────────────────────────────────────────────
    discovery_rows = [r for r in all_rows if r.get("split") == "discovery"]
    candidate, reasoning = select_candidate(discovery_rows)
    print(f"\nN2-A candidate arm: {candidate}")
    for arm_name, s in reasoning.get("arm_stats", {}).items():
        print(f"  {arm_name}: mean_digd={s['mean_delta_igd']:.6f}  "
              f"rho={s['median_rho']:.2f}  clip={s['median_clipping']:.2f}  "
              f"D_A03={s['mean_D_A_03']:.6f}")

    # ─── N2-B: Holdout ────────────────────────────────────────────────────
    if candidate is not None:
        cand_def  = next(a for a in ARMS if a["arm"] == candidate)
        holdout_arm_names = {"F0", "F03", candidate}
        holdout_arms = [a for a in ARMS if a["arm"] in holdout_arm_names]
        print(f"\n=== N2-B: Holdout (arms={[a['arm'] for a in holdout_arms]}) ===")
        for seed in HOLDOUT_SEEDS:
            t1 = time.time()
            rows = run_one_n2(seed, N_SCORED, SCENARIO, holdout_arms, split="holdout")
            all_rows.extend(rows)
            sv = sum(1 for r in rows if r.get("has_history", True) and "arm" in r)
            ms = sum(1 for r in rows if not r.get("has_history", True))
            n_valid   += sv
            n_missing += ms
            print(f"  seed={seed}: {sv} valid rows, {ms} skipped in {time.time()-t1:.1f}s")
    else:
        print("\nNo holdout candidate identified. N2-B skipped.")

    # ─── Save JSONL ───────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | {n_valid} valid, {n_missing} skipped "
          f"| {len(all_rows)} rows → {jsonl_path}")

    # ─── Save candidate ───────────────────────────────────────────────────
    with open(OUT_DIR / "N2_candidate.json", "w") as f:
        json.dump(json_safe({"candidate": candidate, "reasoning": reasoning}), f, indent=2)
    print(f"Candidate saved.")
