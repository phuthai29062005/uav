"""
R1-N — Production Noise Interaction Audit.

Scientific question:
    With the actual POP estimator and production sigma_P=0.03, does R1 HARD
    protection still remove the cross-region coverage-loss mechanism, and what
    performance remains after prediction noise is reintroduced?

FOUR CONDITIONS per event-region:
    POP_UNR_SIGMA0   — unrestricted NSGA-II, sigma=0.00
    POP_HARD_SIGMA0  — R1 hard protection, sigma=0.00
    POP_UNR_SIGMA03  — unrestricted NSGA-II, sigma=0.03 (production)
    POP_HARD_SIGMA03 — R1 hard protection, sigma=0.03 (production)

CRITICAL — Common Random Numbers (CRN):
    For every (seed, event, region), (parent_idxs, xis) are generated ONCE.
    Both sigma values use the SAME parents and xis; only noise scale differs.
    This isolates the sigma effect from all other sources of variation.

    ch_sigma0  = apply_samples(parent_idxs, xis, X_region, delta_hat, 0.0,  xl, xu)
    ch_sigma03 = apply_samples(parent_idxs, xis, X_region, delta_hat, 0.03, xl, xu)

HARD protection reuses EXACT frozen region_protected_survival from R1/R1-P.
UNRESTRICTED reuses EXACT nsga2_survival_traced from E2-P3.

CAUSAL CONTRASTS:
    Allocation benefit at sigma s:  A_s    = HARD_s_delta_igd - UNR_s_delta_igd
    Noise penalty for UNR:          N_UNR  = UNR03_delta_igd  - UNR0_delta_igd
    Noise penalty for HARD:         N_HARD = HARD03_delta_igd - HARD0_delta_igd
    Interaction:                    I      = A_03 - A_0

CONSTRAINTS (hard):
    - nsga2_survival NOT modified.
    - region_protected_survival NOT modified.
    - sigma_P=0.03 NOT changed.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - Only sigma values [0.0, 0.03] tested — no sweep.
    - No GT information enters delta_hat or child generation.
    - Missing history → skip (has_history=False), same as R1-P.
    - R1-N is experimental only; no production change.
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
    distance_to_region_ps, bin_of,
)

# ============================================================
# Config — production locked
# ============================================================
LAMBDA_PRED   = 1.0          # production — never change
SIGMA_PROD    = 0.03         # production sigma_P — never change here
SIGMA_ZERO    = 0.0          # control condition
BUDGET        = e2p3.BUDGET  # 10
POP_SIZE      = e2p3.POP_SIZE  # 100
D             = e2p3.D         # 10
N_OBJ         = e2p3.N_OBJ     # 2
N_PARTITIONS  = e2p3.N_PARTITIONS  # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

SIGMA_VALUES = [SIGMA_ZERO, SIGMA_PROD]

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
# Noise magnitude diagnostics
# ============================================================

def noise_diagnostics(xis, sigma, delta_pop):
    """Compute noise magnitude metrics for a set of xi samples.

    Args:
        xis:      list of length B, each xi is shape (D,)
        sigma:    noise scale
        delta_pop: predicted delta_hat, shape (D,)

    Returns dict of noise metrics.
    """
    xi_arr    = np.asarray(xis, dtype=float)
    noise_vecs = sigma * xi_arr
    noise_norms = np.linalg.norm(noise_vecs, axis=1)
    delta_norm  = float(np.linalg.norm(delta_pop))
    eps = 1e-12
    return {
        "median_noise_norm":     float(np.median(noise_norms)),
        "mean_noise_norm":       float(np.mean(noise_norms)),
        "delta_hat_norm":        delta_norm,
        "noise_to_motion_ratio": float(np.median(noise_norms) / (delta_norm + eps)),
    }


# ============================================================
# Clipping and child-quality diagnostics
# ============================================================

def child_quality_diagnostics(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu,
                               F_children, pf_star):
    """Compute per-sigma child quality: clipping, unique fraction, PS/PF distances."""
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span = (xu_a - xl_a) + 1e-12

    n_clipped = 0
    clipped_dims_list = []
    for pidx, xi in zip(parent_idxs, xis):
        z_p   = (X_region[pidx] - xl_a) / span
        z_raw = z_p + delta_pop + sigma * xi
        n_clip_dims = int(np.sum((z_raw < 0.0) | (z_raw > 1.0)))
        n_clipped += int(n_clip_dims > 0)
        clipped_dims_list.append(n_clip_dims)

    B = len(parent_idxs)
    clipping_frac  = float(n_clipped) / B if B > 0 else float("nan")
    mean_clip_dims = float(np.mean(clipped_dims_list)) if clipped_dims_list else float("nan")
    max_clip_dims  = int(max(clipped_dims_list)) if clipped_dims_list else 0

    # Unique children
    ch = apply_samples(parent_idxs, xis, X_region, delta_pop, sigma, xl, xu)
    n_unique = len(np.unique(ch, axis=0))
    unique_frac = float(n_unique) / B

    # PF distance (child → nearest PF*)
    pf_med, _, _ = median_nearest_pf_distance(F_children, pf_star)

    # PS distance (child → PS in normalized decision space)
    from prdrl.problems.controlled_predict import distance_to_region_ps
    return {
        "clipping_frac":     clipping_frac,
        "mean_clipped_dims": mean_clip_dims,
        "max_clipped_dims":  max_clip_dims,
        "n_unique_children": n_unique,
        "unique_child_frac": unique_frac,
        "child_pf_dist":     pf_med,
    }


# ============================================================
# Per-region R1-N comparison for one scored event
# ============================================================

def run_region_r1n(
    r, X_pop, F_plus, F_minus, region_ids, t_new, t_prev,
    region_dynamics, n_bins, pf_star, V,
    xl, xu, base_seed, change_idx,
    history, history_t,
):
    """Compare four conditions (UNR/HARD × sigma0/sigma03) for region r.

    CRN: parent_idxs and xis generated ONCE; both sigma values share them.
    Both survival branches within each sigma receive identical children/FE.

    Returns one row dict, or skipped row (has_history=False), or [] if region empty.
    """
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

    # POP estimator delta — uses only t-2 and t-1 via RegionHistory
    motion = get_predict_motion_from_history(
        history, r, t=history_t + 1, lambda_pred=LAMBDA_PRED
    )
    if motion is None:
        return [{"has_history": False, "region_id": r, "event_id": change_idx}]

    delta_pop = motion["delta_hat"]   # shape (D,), normalized [0,1] space

    # GT delta for diagnostics only (NOT used in child generation or survival)
    n_dist          = D - 1
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev, n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,  n_dist)
    delta_gt_full   = np.concatenate([[0.0], mu_star_current - mu_star_prev])

    diag = estimator_diagnostics(delta_pop, delta_gt_full)

    # Baseline
    F_before   = F_plus.copy()
    igd_before = float(igd(F_before, pf_star))

    # ─── CRN: generate (parent_idxs, xis) ONCE for both sigma ───────────
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    parent_idxs, xis = generate_base_samples(BUDGET, D, rng_shared, n_r_before)

    # ─── Children for sigma=0 and sigma=0.03 ─────────────────────────────
    ch_sigma0  = apply_samples(parent_idxs, xis, X_region, delta_pop, SIGMA_ZERO, xl, xu)
    F_ch_s0    = cp_evaluate(ch_sigma0, t_new, region_dynamics)

    ch_sigma03 = apply_samples(parent_idxs, xis, X_region, delta_pop, SIGMA_PROD, xl, xu)
    F_ch_s03   = cp_evaluate(ch_sigma03, t_new, region_dynamics)

    # ─── Noise magnitude diagnostics ─────────────────────────────────────
    noise_diag = noise_diagnostics(xis, SIGMA_PROD, delta_pop)

    # ─── Child quality per sigma ──────────────────────────────────────────
    cq_s0  = child_quality_diagnostics(
        parent_idxs, xis, X_region, delta_pop, SIGMA_ZERO, xl, xu, F_ch_s0,  pf_star)
    cq_s03 = child_quality_diagnostics(
        parent_idxs, xis, X_region, delta_pop, SIGMA_PROD, xl, xu, F_ch_s03, pf_star)

    # ─── Normalization basis for coverage metrics ─────────────────────────
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

    def _survival_counts(surv_idx, is_hard):
        """Compute dropped/survived counts from survived pool indices."""
        surv_old_set   = set(int(i) for i in surv_idx if i < N_total_old)
        dropped_old    = set(range(N_total_old)) - surv_old_set
        n_old_dropped  = len(dropped_old)
        n_ch_survived  = int(np.sum(surv_idx >= N_total_old))
        dropped_other  = sum(1 for i in dropped_old if region_ids[i] != r)
        dropped_same   = n_old_dropped - dropped_other
        if is_hard:
            assert dropped_other == 0, (
                f"HARD protection dropped {dropped_other} outside-region OLD"
            )
        return {
            "n_old_dropped":       n_old_dropped,
            "n_old_dropped_same":  dropped_same,
            "n_old_dropped_other": dropped_other,
            "n_ch_survived":       n_ch_survived,
            "child_survival_rate": float(n_ch_survived) / BUDGET,
        }

    # Deterministic survival seed — same for all four branches
    surv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    # ─── UNR_SIGMA0 ──────────────────────────────────────────────────────
    X_pool_u0  = np.vstack([X_pop, ch_sigma0])
    F_pool_u0  = np.vstack([F_before, F_ch_s0])
    X_s_u0, F_s_u0, idx_u0 = nsga2_survival_traced(X_pool_u0, F_pool_u0, POP_SIZE, xl, xu, surv_seed)
    cnt_u0  = _survival_counts(idx_u0, is_hard=False)
    cov_u0  = _coverage_metrics(F_s_u0, "unr0")

    # ─── HARD_SIGMA0 ─────────────────────────────────────────────────────
    X_s_h0, F_s_h0, idx_h0 = region_protected_survival(
        X_pop, F_before, ch_sigma0, F_ch_s0,
        r, region_ids, n_r_before, xl, xu, surv_seed)
    cnt_h0  = _survival_counts(idx_h0, is_hard=True)
    cov_h0  = _coverage_metrics(F_s_h0, "hard0")

    # ─── UNR_SIGMA03 ─────────────────────────────────────────────────────
    X_pool_u03 = np.vstack([X_pop, ch_sigma03])
    F_pool_u03 = np.vstack([F_before, F_ch_s03])
    X_s_u03, F_s_u03, idx_u03 = nsga2_survival_traced(X_pool_u03, F_pool_u03, POP_SIZE, xl, xu, surv_seed)
    cnt_u03 = _survival_counts(idx_u03, is_hard=False)
    cov_u03 = _coverage_metrics(F_s_u03, "unr03")

    # ─── HARD_SIGMA03 ────────────────────────────────────────────────────
    X_s_h03, F_s_h03, idx_h03 = region_protected_survival(
        X_pop, F_before, ch_sigma03, F_ch_s03,
        r, region_ids, n_r_before, xl, xu, surv_seed)
    cnt_h03 = _survival_counts(idx_h03, is_hard=True)
    cov_h03 = _coverage_metrics(F_s_h03, "hard03")

    # ─── Causal contrasts ────────────────────────────────────────────────
    A_0   = cov_h0["hard0_delta_igd"]   - cov_u0["unr0_delta_igd"]
    A_03  = cov_h03["hard03_delta_igd"] - cov_u03["unr03_delta_igd"]
    I     = A_03 - A_0

    noise_penalty_unr  = cov_u03["unr03_delta_igd"]  - cov_u0["unr0_delta_igd"]
    noise_penalty_hard = cov_h03["hard03_delta_igd"] - cov_h0["hard0_delta_igd"]

    # Child-level noise delta (paired comparison)
    delta_child_pf_noise = cq_s03["child_pf_dist"] - cq_s0["child_pf_dist"]
    delta_child_clipping  = cq_s03["clipping_frac"] - cq_s0["clipping_frac"]

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

        # Noise magnitude
        **noise_diag,

        # Child quality — sigma=0
        **{f"s0_{k}": v for k, v in cq_s0.items()},
        # Child quality — sigma=0.03
        **{f"s03_{k}": v for k, v in cq_s03.items()},

        # Noise effects on child quality
        "delta_child_pf_noise":   delta_child_pf_noise,
        "delta_child_clipping":   delta_child_clipping,

        # Survival counts — UNR_SIGMA0
        **{f"unr0_{k}": v for k, v in cnt_u0.items()},
        # Survival counts — HARD_SIGMA0
        **{f"hard0_{k}": v for k, v in cnt_h0.items()},
        # Survival counts — UNR_SIGMA03
        **{f"unr03_{k}": v for k, v in cnt_u03.items()},
        # Survival counts — HARD_SIGMA03
        **{f"hard03_{k}": v for k, v in cnt_h03.items()},

        # Causal contrasts
        "A_0":                 A_0,
        "A_03":                A_03,
        "interaction_I":       I,
        "noise_penalty_unr":   noise_penalty_unr,
        "noise_penalty_hard":  noise_penalty_hard,

        # Hard better fractions computed at analysis time
    }
    row.update(cov_u0)
    row.update(cov_h0)
    row.update(cov_u03)
    row.update(cov_h03)
    return [row]


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_r1n(seed, n_scored_changes, scenario_name):
    """Run R1-N comparison for one seed. Returns list of row dicts."""
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
            r_rows = run_region_r1n(
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
    OUT_DIR  = Path("results/R1N_production_noise_interaction")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "R1N_production_noise_interaction.jsonl"

    all_rows = []
    n_valid  = 0
    n_missing = 0
    t0 = time.time()

    for seed in range(1, N_SEEDS + 1):
        t1 = time.time()
        rows = run_one_r1n(seed, N_SCORED, SCENARIO)
        all_rows.extend(rows)
        sv = sum(1 for rw in rows if rw.get("has_history", True))
        ms = sum(1 for rw in rows if not rw.get("has_history", True))
        n_valid   += sv
        n_missing += ms
        print(f"{SCENARIO} seed={seed}: {sv} valid, {ms} skipped "
              f"in {time.time()-t1:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | {n_valid} valid rows, {n_missing} skipped "
          f"| Wrote {len(all_rows)} rows to {jsonl_path}")
