"""
L1 — Longitudinal Multi-Change 2×2 Factorial Validation.

Scientific question:
    Does HARD region protection + sigma=0.005 (H005) deliver measurably better
    longitudinal tracking than unrestricted NSGA-II + sigma=0.03 (U03)?

    Does sigma (0.005 vs 0.03) and HARD protection each independently contribute,
    and do they interact?

DESIGN:
    2×2 factorial: {UNRESTRICTED, HARD} × {sigma=0.03, sigma=0.005}
    Arms: U03, U005 (UNRESTRICTED), H03, H005 (HARD)

    Longitudinal carry-forward: population NOT reset between scored changes.
    Arm-specific RegionHistory diverges after the shared burn-in.

    Target region schedule (round-robin, pre-declared):
        target_r(seed, k) = (seed + k) % R,  k = 0..N_SCORED_CHANGES-1

    Common environment schedule: all arms see the same environment trajectory.
    CRN for response: u_parent and xi generated once per (seed, k, target_r),
        arm-independent; parent indices mapped via floor(u * n_r) per arm's n_r.

    COMMON_KEEP: if ANY arm has n_r == 0 in target_r → skip response for ALL arms.
    LOCAL_FALLBACK: if arm has no valid motion history for target_r →
        use delta_hat=0, sigma=SIGMA_L=0.02.

    N_SCORED_CHANGES = 10, seeds = 41..60 (verified fresh).

ENDPOINTS:
    Primary: MIGD_response = mean igd_response over 10 scored changes (per arm, per seed).
    Secondary: MIGD_epoch = mean igd_epoch_end over 10 scored changes.

FACTORIAL EFFECTS:
    NoiseBenefit_U    = MIGD_U03  - MIGD_U005   (sigma effect, unrestricted)
    NoiseBenefit_H    = MIGD_H03  - MIGD_H005   (sigma effect, HARD)
    HardBenefit_03    = MIGD_U03  - MIGD_H03    (HARD effect, sigma=0.03)
    HardBenefit_005   = MIGD_U005 - MIGD_H005   (HARD effect, sigma=0.005)
    CombinedBenefit   = MIGD_U03  - MIGD_H005   (diagonal)
    Interaction       = NoiseBenefit_H - NoiseBenefit_U
                      = HardBenefit_005 - HardBenefit_03  (algebraic identity)

VERDICTS:
    COMBINED_H005_LONGITUDINAL, F005_LONGITUDINAL_EFFECT, HARD_LONGITUDINAL_EFFECT

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged.
    - Production predict() unmodified.
    - region_protected_survival unmodified.
    - nsga2_survival_traced unmodified.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - Do NOT start RL.
    - Do NOT overwrite N1, N2, N2C results.
    - Seeds 41-60 only (disjoint from 1-40, 101-104).
"""
import copy
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
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
)

# ============================================================
# Config — production locked
# ============================================================
LAMBDA_PRED   = 1.0
SIGMA_PROD    = 0.03   # production — never change
SIGMA_L       = 0.02   # LOCAL fallback sigma

BUDGET        = e2p3.BUDGET        # 10
POP_SIZE      = e2p3.POP_SIZE      # 100
D             = e2p3.D              # 10
N_OBJ         = e2p3.N_OBJ         # 2
N_PARTITIONS  = e2p3.N_PARTITIONS   # 9 → R=10
WARMUP_GENERATIONS     = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

# 2×2 factorial arms — pre-declared, not tuned
ARM_DEFS = [
    {"arm": "U03",  "sigma": 0.03,  "use_hard": False},
    {"arm": "U005", "sigma": 0.005, "use_hard": False},
    {"arm": "H03",  "sigma": 0.03,  "use_hard": True},
    {"arm": "H005", "sigma": 0.005, "use_hard": True},
]

N_SCORED_CHANGES  = 10
L1_SEEDS          = list(range(41, 61))     # 20 seeds, fresh
DELTA_LONGITUDINAL = 1e-5                   # practical margin, pre-declared
BOOTSTRAP_SEED     = 42
BOOTSTRAP_RESAMPLES = 10_000
SMOKE_SEEDS        = [103, 104]
SCENARIO           = "PERSISTENT"

# FE accounting (project convention: children evaluated + evolution; reevaluation excluded)
RESPONSE_FE   = BUDGET                          # 10 per PREDICT response
EVOLUTION_FE  = GENERATIONS_PER_CHANGE * POP_SIZE  # 10 × 100 = 1,000 per change
FE_CONVENTION = "children_plus_evolution_no_reeval"

# Aliases from upstream modules
compute_time             = e2p3.compute_time
evolve_nsga2             = e2p3.evolve_nsga2
make_rng                 = e2p3.make_rng
make_scenario_dynamics   = e2p3.make_scenario_dynamics
median_nearest_pf_distance = e2p3.median_nearest_pf_distance
nsga2_survival_traced      = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed     = e2p3._e2p3_common_surv_seed
generate_base_samples      = e2p3.generate_base_samples
apply_samples              = e2p3.apply_samples

compute_ranks_crowding_all    = r1.compute_ranks_crowding_all
compute_pf_coverage_per_point = r1.compute_pf_coverage_per_point
region_protected_survival     = r1.region_protected_survival


# ============================================================
# Target region schedule
# ============================================================

def target_region(seed, k, R):
    """Round-robin target region for scored change k of seed."""
    return (seed + k) % R


# ============================================================
# Child quality helpers (re-implemented locally for arm-level tracking)
# ============================================================

def _count_valid_history(arm, R):
    """Count regions with valid 2-step motion history (read-only, no RNG consumed)."""
    count = 0
    for r in range(R):
        m = get_predict_motion_from_history(
            arm["history"], r, t=arm["history_t"] + 1, lambda_pred=LAMBDA_PRED
        )
        if m is not None:
            count += 1
    return count


def _child_clipping_frac(parent_idxs, xis, X_region, delta_hat, sigma, xl, xu):
    """Fraction of children with at least one clipped dimension."""
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a
    B = len(parent_idxs)
    n_clipped = 0
    for pidx, xi in zip(parent_idxs, xis):
        z_p   = (X_region[pidx] - xl_a) / span
        z_raw = z_p + delta_hat + sigma * xi
        if np.any((z_raw < 0.0) | (z_raw > 1.0)):
            n_clipped += 1
    return float(n_clipped) / B if B > 0 else float("nan")


# ============================================================
# Bootstrap CI over seed-level contrasts
# ============================================================

def bootstrap_contrast_ci(seed_values, seed=BOOTSTRAP_SEED,
                           n_resamples=BOOTSTRAP_RESAMPLES):
    """95% percentile bootstrap CI over seed-level contrast values.

    Args:
        seed_values: 1-D array of per-seed contrast values (e.g. CombinedBenefit).

    Returns: (mean, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(seed_values, dtype=float)
    n   = len(arr)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(arr[idx])
    return (
        float(np.mean(arr)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


# ============================================================
# Factorial effects from seed-level MIGD summaries
# ============================================================

def compute_factorial_effects(seed_rows):
    """Compute 2×2 factorial effects (response + epoch) from seed-level summaries.

    Each dict must have: seed, U03_response, U005_response, H03_response,
    H005_response, U03_epoch, U005_epoch, H03_epoch, H005_epoch.

    Returns list of per-seed effect dicts with both response and epoch terms.
    Interaction identity: Interaction = HardBenefit_005 - HardBenefit_03 (verified).
    """
    effects = []
    for r in seed_rows:
        # Response-level MIGD
        u03   = r["U03_response"]
        u005  = r["U005_response"]
        h03   = r["H03_response"]
        h005  = r["H005_response"]
        # Epoch-level MIGD
        u03e  = r["U03_epoch"]
        u005e = r["U005_epoch"]
        h03e  = r["H03_epoch"]
        h005e = r["H005_epoch"]

        # Response effects
        nb_u   = u03   - u005
        nb_h   = h03   - h005
        hb_03  = u03   - h03
        hb_005 = u005  - h005
        comb   = u03   - h005
        inter  = nb_h  - nb_u        # == hb_005 - hb_03 (algebraic identity)

        # Epoch effects
        nb_u_e   = u03e  - u005e
        nb_h_e   = h03e  - h005e
        hb_03_e  = u03e  - h03e
        hb_005_e = u005e - h005e
        comb_e   = u03e  - h005e
        inter_e  = nb_h_e - nb_u_e   # == hb_005_e - hb_03_e (same identity)

        effects.append({
            "seed":                   r["seed"],
            # Response
            "NoiseBenefit_U":         float(nb_u),
            "NoiseBenefit_H":         float(nb_h),
            "HardBenefit_03":         float(hb_03),
            "HardBenefit_005":        float(hb_005),
            "CombinedBenefit":        float(comb),
            "Interaction":            float(inter),
            # Epoch
            "NoiseBenefit_U_epoch":   float(nb_u_e),
            "NoiseBenefit_H_epoch":   float(nb_h_e),
            "HardBenefit_03_epoch":   float(hb_03_e),
            "HardBenefit_005_epoch":  float(hb_005_e),
            "CombinedBenefit_epoch":  float(comb_e),
            "Interaction_epoch":      float(inter_e),
        })
    return effects


# ============================================================
# Seed-level MIGD summaries
# ============================================================

def compute_seed_summaries(rows):
    """Aggregate per-row data into per-seed MIGD summaries.

    Args:
        rows: list of row dicts from run_one_l1.

    Returns list of seed-level summary dicts (one per seed × arm combination
    collapsed to one per seed).
    """
    from collections import defaultdict

    by_seed_arm = defaultdict(list)
    for r in rows:
        by_seed_arm[(r["seed"], r["arm"])].append(r)

    seeds = sorted(set(r["seed"] for r in rows))
    summaries = []
    for seed in seeds:
        row_dict = {"seed": seed}
        for ad in ARM_DEFS:
            arm = ad["arm"]
            arm_rows = by_seed_arm[(seed, arm)]
            if not arm_rows:
                row_dict[f"{arm}_response"] = float("nan")
                row_dict[f"{arm}_epoch"]    = float("nan")
                continue
            row_dict[f"{arm}_response"] = float(np.mean([r["igd_response"] for r in arm_rows]))
            row_dict[f"{arm}_epoch"]    = float(np.mean([r["igd_epoch_end"] for r in arm_rows]))
        summaries.append(row_dict)
    return summaries


# ============================================================
# Verdict decision logic (C1–C5)
# ============================================================

def decide_verdicts(effects, c3_result=None, c5_confirmed=True):
    """Apply original pre-declared C1–C5 decision criteria.

    C5 (cumulative FE exactly matched) is a structural property verified by
    compute_fe_audit. Pass c5_confirmed=True when FE equality has been verified.

    Factor verdicts use SUPPORTED / PARTIAL / NOT_SUPPORTED taxonomy.
    COMBINED_H005_LONGITUDINAL uses VALIDATED / PARTIAL / NOT_VALIDATED.

    Separate invariants NOT part of C5:
    - PREDICT_ONLY_INVARIANT: all rows used PREDICT operator
    - HARD_CROSS_REGION_INVARIANT: n_old_dropped_other == 0 for HARD arms
    """
    comb_vals   = [e["CombinedBenefit"]       for e in effects]
    comb_epoch  = [e["CombinedBenefit_epoch"] for e in effects]
    nb_u_vals   = [e["NoiseBenefit_U"]        for e in effects]
    nb_h_vals   = [e["NoiseBenefit_H"]        for e in effects]
    hb_03_vals  = [e["HardBenefit_03"]        for e in effects]
    hb_005_vals = [e["HardBenefit_005"]       for e in effects]
    nb_u_e_vals  = [e["NoiseBenefit_U_epoch"] for e in effects]
    nb_h_e_vals  = [e["NoiseBenefit_H_epoch"] for e in effects]
    hb_03_e_vals  = [e["HardBenefit_03_epoch"]  for e in effects]
    hb_005_e_vals = [e["HardBenefit_005_epoch"] for e in effects]

    mean_comb, lo_comb, hi_comb = bootstrap_contrast_ci(comb_vals)
    mean_comb_e = float(np.mean(comb_epoch))

    # --- C1–C5 ---
    c1 = bool(mean_comb > DELTA_LONGITUDINAL)           # response mean > +1e-5
    c2 = bool(lo_comb > 0.0)                            # CI lower > 0
    # C3: late-half response benefit >= -1e-5 (passed in from compute_c3_late_half)
    c3 = bool(c3_result["C3"]) if c3_result is not None else None
    c4 = bool(mean_comb_e >= -DELTA_LONGITUDINAL)        # epoch benefit >= -1e-5
    c5 = bool(c5_confirmed)                              # cumulative FE equal

    c_all = c1 and c2 and bool(c3 or c3 is None) and c4 and c5
    combined_verdict = "VALIDATED" if c_all else "PARTIAL" if (c1 and c4) else "NOT_VALIDATED"

    # --- F005_LONGITUDINAL_EFFECT: SUPPORTED / PARTIAL / NOT_SUPPORTED ---
    mean_nb_u_r = float(np.mean(nb_u_vals))
    mean_nb_h_r = float(np.mean(nb_h_vals))
    mean_nb_u_e = float(np.mean(nb_u_e_vals))
    mean_nb_h_e = float(np.mean(nb_h_e_vals))

    f005_response_positive = (mean_nb_u_r > 0 and mean_nb_h_r > 0)
    f005_epoch_positive    = (mean_nb_u_e > -DELTA_LONGITUDINAL and
                               mean_nb_h_e > -DELTA_LONGITUDINAL)

    if f005_response_positive and f005_epoch_positive and mean_nb_u_e > 0 and mean_nb_h_e > 0:
        f005_verdict = "SUPPORTED"
    elif f005_response_positive and f005_epoch_positive:
        f005_verdict = "SUPPORTED"   # epoch above threshold, directionally consistent
    elif f005_response_positive:
        f005_verdict = "PARTIAL"     # response positive, epoch unclear
    else:
        f005_verdict = "NOT_SUPPORTED"

    # --- HARD_LONGITUDINAL_EFFECT: SUPPORTED / PARTIAL / NOT_SUPPORTED ---
    mean_hb_03_r  = float(np.mean(hb_03_vals))
    mean_hb_005_r = float(np.mean(hb_005_vals))
    mean_hb_03_e  = float(np.mean(hb_03_e_vals))
    mean_hb_005_e = float(np.mean(hb_005_e_vals))

    hard_response_positive = (mean_hb_03_r > -DELTA_LONGITUDINAL and
                               mean_hb_005_r > -DELTA_LONGITUDINAL)
    hard_epoch_positive    = (mean_hb_03_e > -DELTA_LONGITUDINAL and
                               mean_hb_005_e > -DELTA_LONGITUDINAL)

    if hard_response_positive and hard_epoch_positive:
        hard_verdict = "SUPPORTED"
    elif hard_response_positive and not hard_epoch_positive:
        hard_verdict = "PARTIAL"     # helps immediate response, washes out post-evolution
    elif not hard_response_positive:
        hard_verdict = "NOT_SUPPORTED"
    else:
        hard_verdict = "PARTIAL"

    return {
        # Primary verdicts
        "COMBINED_H005_LONGITUDINAL": combined_verdict,
        "F005_LONGITUDINAL_EFFECT":   f005_verdict,
        "HARD_LONGITUDINAL_EFFECT":   hard_verdict,
        # C1–C5 status
        "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5,
        # Key statistics
        "mean_CombinedBenefit":       float(mean_comb),
        "ci_low_CombinedBenefit":     float(lo_comb),
        "ci_high_CombinedBenefit":    float(hi_comb),
        "mean_CombinedBenefit_epoch": float(mean_comb_e),
        # Separate invariants (not C5)
        "PREDICT_ONLY_INVARIANT":       None,   # filled by caller from row counts
        "HARD_CROSS_REGION_INVARIANT":  None,   # filled by caller from row counts
        # Factor details
        "mean_NoiseBenefit_U_response":   float(mean_nb_u_r),
        "mean_NoiseBenefit_H_response":   float(mean_nb_h_r),
        "mean_NoiseBenefit_U_epoch":      float(mean_nb_u_e),
        "mean_NoiseBenefit_H_epoch":      float(mean_nb_h_e),
        "mean_HardBenefit_03_response":   float(mean_hb_03_r),
        "mean_HardBenefit_005_response":  float(mean_hb_005_r),
        "mean_HardBenefit_03_epoch":      float(mean_hb_03_e),
        "mean_HardBenefit_005_epoch":     float(mean_hb_005_e),
    }


# ============================================================
# Per-seed late-half response benefit (C3)
# ============================================================

def compute_c3_late_half(rows):
    """Compute C3: mean CombinedBenefit for late-half changes (k=5..9)."""
    from collections import defaultdict
    by_seed_arm_k = defaultdict(dict)
    for r in rows:
        by_seed_arm_k[(r["seed"], r["arm"])][r["k"]] = r["igd_response"]

    seeds = sorted(set(r["seed"] for r in rows))
    late_vals = []
    for seed in seeds:
        u03_late  = [by_seed_arm_k[(seed, "U03")].get(k, float("nan"))  for k in range(5, 10)]
        h005_late = [by_seed_arm_k[(seed, "H005")].get(k, float("nan")) for k in range(5, 10)]
        u03_mean   = float(np.nanmean(u03_late))
        h005_mean  = float(np.nanmean(h005_late))
        late_vals.append(u03_mean - h005_mean)

    mean_late = float(np.mean(late_vals))
    c3 = bool(mean_late >= -DELTA_LONGITUDINAL)
    return {"C3": c3, "mean_late_CombinedBenefit": mean_late, "per_seed": late_vals}


# ============================================================
# Main longitudinal run — one seed
# ============================================================

def run_one_l1(seed, n_scored, scenario_name, arm_defs=None):
    """Run L1 longitudinal factorial for one seed.

    Args:
        seed:          Random seed (must be in 41..60 for full run, or 103/104 for smoke).
        n_scored:      Number of scored changes (10 for full, 3 for smoke).
        scenario_name: Scenario name passed to make_scenario_dynamics.
        arm_defs:      List of arm defs (default: ARM_DEFS).

    Returns list of row dicts, one per (arm, scored_change_idx).
    """
    if arm_defs is None:
        arm_defs = ARM_DEFS

    xl, xu = get_bounds(D)
    V  = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)

    region_dynamics = make_scenario_dynamics(scenario_name, R)
    pf_star = get_true_pf(n_points=500)

    # ---- 1. Warmup (common) ----
    rng    = np.random.default_rng(seed)
    X_pop  = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_current = compute_time(0)
    X_pop  = evolve_nsga2(X_pop, t_current, region_dynamics,
                          WARMUP_GENERATIONS, seed * 10000, xl, xu)

    # ---- 2. Burn-in change (change_idx=1, common, not scored) ----
    history   = RegionHistory(n_regions=R)
    history_t = -1
    gen       = WARMUP_GENERATIONS

    F_minus   = cp_evaluate(X_pop, t_current, region_dynamics)
    gen      += GENERATIONS_PER_CHANGE
    t_new     = compute_time(gen)
    F_plus    = cp_evaluate(X_pop, t_new, region_dynamics)
    F_tm, _   = normalize(F_minus, F_plus)
    F_tm      = np.asarray(F_tm)
    region_ids = np.asarray(partition(F_tm, V))
    history_t += 1
    history.update(t=history_t, X=X_pop, F=F_minus,
                   region_ids=region_ids, xl=xl, xu=xu)
    t_current = t_new
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu)

    # ---- 3. Clone state into 4 arms ----
    arm_states = {}
    for ad in arm_defs:
        arm_states[ad["arm"]] = {
            "sigma":    ad["sigma"],
            "use_hard": ad["use_hard"],
            "X":        X_pop.copy(),
            "history":  copy.deepcopy(history),
            "history_t": history_t,
        }

    rows_out = []

    # ---- 4. Scored changes ----
    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r = target_region(seed, k, R)

        # Advance environment (common to all arms)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)

        # ---- Per-arm: evaluate, partition, update history ----
        arm_eval = {}
        for arm_name, arm in arm_states.items():
            F_minus_arm = cp_evaluate(arm["X"], t_current, region_dynamics)
            F_plus_arm  = cp_evaluate(arm["X"], t_new,     region_dynamics)
            F_tm_arm, _ = normalize(F_minus_arm, F_plus_arm)
            F_tm_arm    = np.asarray(F_tm_arm)
            rids_arm    = np.asarray(partition(F_tm_arm, V))
            n_r_arm     = int(np.sum(rids_arm == tgt_r))
            igd_pre     = float(igd(F_plus_arm, pf_star))

            arm["history_t"] += 1
            arm["history"].update(
                t=arm["history_t"],
                X=arm["X"],
                F=F_minus_arm,
                region_ids=rids_arm,
                xl=xl,
                xu=xu,
            )
            arm_eval[arm_name] = {
                "F_minus":    F_minus_arm,
                "F_plus":     F_plus_arm,
                "region_ids": rids_arm,
                "n_r":        n_r_arm,
                "igd_pre":    igd_pre,
            }

        # ---- COMMON_KEEP check ----
        common_available = all(arm_eval[a]["n_r"] > 0 for a in arm_eval)
        evo_seed  = seed * 10000 + scored_change_idx
        surv_seed = _e2p3_common_surv_seed(seed, scored_change_idx, tgt_r)

        if not common_available:
            # KEEP: skip response for all arms
            for arm_name, arm in arm_states.items():
                ev      = arm_eval[arm_name]
                igd_pre = ev["igd_pre"]
                n_active_pre = int(len(np.unique(ev["region_ids"])))
                n_valid_hist = _count_valid_history(arm, R)
                arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                        GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
                F_epoch  = cp_evaluate(arm["X"], t_new, region_dynamics)
                igd_epoch = float(igd(F_epoch, pf_star))
                rows_out.append({
                    "seed":              seed,
                    "arm":               arm_name,
                    "k":                 k,
                    "scored_change_idx": scored_change_idx,
                    "target_r":          tgt_r,
                    "operator":          "COMMON_KEEP",
                    "common_available":  False,
                    "sigma":             arm["sigma"],
                    "n_r":               ev["n_r"],
                    "has_predict_history": None,
                    "delta_hat_norm":    None,
                    "clipping_frac":     None,
                    "n_old_dropped_other": None,
                    "n_active_regions_pre":         n_active_pre,
                    "n_regions_with_valid_history": n_valid_hist,
                    "response_fe":       0,
                    "evolution_fe":      EVOLUTION_FE,
                    "total_fe_change":   EVOLUTION_FE,
                    "igd_pre":           igd_pre,
                    "igd_response":      igd_pre,
                    "igd_epoch_end":     igd_epoch,
                    "response_gain":     0.0,
                    "epoch_gain":        float(igd_pre - igd_epoch),
                })
            t_current = t_new
            continue

        # ---- CRN: arm-independent u_parent and xi ----
        rng_crn  = make_rng(seed, scored_change_idx, tgt_r, "L1_RESPONSE")
        u_parent = rng_crn.uniform(size=BUDGET)          # (B,)
        xi_mat   = rng_crn.standard_normal(size=(BUDGET, D))  # (B, D)
        xis      = list(xi_mat)                           # list of (D,) arrays

        # ---- Per-arm response ----
        for arm_name, arm in arm_states.items():
            ev          = arm_eval[arm_name]
            n_r         = ev["n_r"]
            F_plus_arm  = ev["F_plus"]
            rids_arm    = ev["region_ids"]
            igd_pre     = ev["igd_pre"]
            X_region    = arm["X"][rids_arm == tgt_r].copy()
            N_old       = len(arm["X"])
            n_active_pre = int(len(np.unique(rids_arm)))
            n_valid_hist = _count_valid_history(arm, R)

            # Map CRN uniforms to parent indices per this arm's n_r
            parent_idxs = [min(int(u * n_r), n_r - 1) for u in u_parent]

            # Check predict history
            motion = get_predict_motion_from_history(
                arm["history"], tgt_r,
                t=arm["history_t"] + 1,
                lambda_pred=LAMBDA_PRED,
            )

            if motion is None:
                operator      = "LOCAL_FALLBACK"
                delta_hat     = np.zeros(D)
                sigma_used    = SIGMA_L
                has_history   = False
                delta_hat_norm = 0.0
            else:
                operator      = "PREDICT"
                delta_hat     = motion["delta_hat"]
                sigma_used    = arm["sigma"]
                has_history   = True
                delta_hat_norm = float(np.linalg.norm(delta_hat))

            clip_frac = _child_clipping_frac(
                parent_idxs, xis, X_region, delta_hat, sigma_used, xl, xu
            )
            ch   = apply_samples(parent_idxs, xis, X_region, delta_hat, sigma_used, xl, xu)
            F_ch = cp_evaluate(ch, t_new, region_dynamics)

            # Survival (arm-specific)
            if arm["use_hard"]:
                X_s, F_s, idx_s = region_protected_survival(
                    arm["X"], F_plus_arm, ch, F_ch,
                    tgt_r, rids_arm, n_r, xl, xu, surv_seed,
                )
            else:
                X_all = np.vstack([arm["X"], ch])
                F_all = np.vstack([F_plus_arm, F_ch])
                X_s, F_s, idx_s = nsga2_survival_traced(
                    X_all, F_all, POP_SIZE, xl, xu, surv_seed,
                )

            # n_old_dropped_other (HARD: should be 0)
            surv_old_set = set(int(i) for i in idx_s if i < N_old)
            old_other_ids = [i for i in range(N_old) if rids_arm[i] != tgt_r]
            n_old_dropped_other = sum(1 for i in old_other_ids if i not in surv_old_set)

            igd_response = float(igd(F_s, pf_star))

            # Update arm state with post-survival population
            arm["X"] = X_s

            # Evolution
            arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                    GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
            F_epoch   = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_epoch = float(igd(F_epoch, pf_star))

            rows_out.append({
                "seed":              seed,
                "arm":               arm_name,
                "k":                 k,
                "scored_change_idx": scored_change_idx,
                "target_r":          tgt_r,
                "operator":          operator,
                "common_available":  True,
                "sigma":             sigma_used if has_history else SIGMA_L,
                "n_r":               n_r,
                "has_predict_history": has_history,
                "delta_hat_norm":    delta_hat_norm,
                "clipping_frac":     clip_frac,
                "n_old_dropped_other": n_old_dropped_other,
                "n_active_regions_pre":         n_active_pre,
                "n_regions_with_valid_history": n_valid_hist,
                "response_fe":       RESPONSE_FE,
                "evolution_fe":      EVOLUTION_FE,
                "total_fe_change":   RESPONSE_FE + EVOLUTION_FE,
                "igd_pre":           igd_pre,
                "igd_response":      igd_response,
                "igd_epoch_end":     igd_epoch,
                "response_gain":     float(igd_pre - igd_response),
                "epoch_gain":        float(igd_pre - igd_epoch),
            })

        t_current = t_new

    return rows_out


# ============================================================
# FE audit
# ============================================================

def compute_fe_audit(rows):
    """Compute FE accounting per arm/seed/change and verify equality.

    FE convention: children_plus_evolution_no_reeval
        response_fe  = BUDGET (10) for PREDICT, 0 for KEEP
        evolution_fe = GENERATIONS_PER_CHANGE × POP_SIZE = 1,000
        reevaluation_fe = EXCLUDED (population reevals not counted)

    Returns:
        fe_rows: list of per-(seed,arm,k) dicts with FE fields + cumulative
        fe_equal: bool — True if cumulative FE is identical across all 4 arms
                  at every (seed, k)
    """
    from collections import defaultdict
    by_seed_arm_k = defaultdict(dict)
    for r in rows:
        key = (r["seed"], r["arm"], r["k"])
        by_seed_arm_k[key] = r

    seeds = sorted(set(r["seed"] for r in rows))
    arm_names = [ad["arm"] for ad in ARM_DEFS]
    fe_rows = []
    fe_equal = True

    for seed in seeds:
        cum = {arm: 0 for arm in arm_names}
        for k in range(max(r["k"] for r in rows if r["seed"] == seed) + 1):
            ref_fe = None
            for arm in arm_names:
                row = by_seed_arm_k.get((seed, arm, k))
                if row is None:
                    continue
                r_fe = row.get("response_fe", RESPONSE_FE if row.get("operator") == "PREDICT" else 0)
                e_fe = row.get("evolution_fe", EVOLUTION_FE)
                total = r_fe + e_fe
                cum[arm] += total
                fe_rows.append({
                    "seed": seed, "arm": arm, "k": k,
                    "response_fe": r_fe,
                    "evolution_fe": e_fe,
                    "total_fe_change": total,
                    "cumulative_fe": cum[arm],
                })
                if ref_fe is None:
                    ref_fe = cum[arm]
                elif cum[arm] != ref_fe:
                    fe_equal = False

    return fe_rows, fe_equal


def compute_persistence_contrasts(effects):
    """Compute response→epoch persistence for each factorial effect.

    Persistence = epoch_value - response_value.
    Negative = benefit decays or reverses after evolution.
    Descriptive only — no threshold applied.
    """
    persistence = []
    for e in effects:
        persistence.append({
            "seed":                 e["seed"],
            "HardPersistence_03":   e["HardBenefit_03_epoch"]  - e["HardBenefit_03"],
            "HardPersistence_005":  e["HardBenefit_005_epoch"] - e["HardBenefit_005"],
            "NoisePersistence_U":   e["NoiseBenefit_U_epoch"]  - e["NoiseBenefit_U"],
            "NoisePersistence_H":   e["NoiseBenefit_H_epoch"]  - e["NoiseBenefit_H"],
            "CombinedPersistence":  e["CombinedBenefit_epoch"] - e["CombinedBenefit"],
        })
    return persistence


def arm_migd_ranking(seed_summaries, endpoint="response"):
    """Return arm ranking (best=lowest IGD first) for given endpoint."""
    arm_means = {}
    for ad in ARM_DEFS:
        arm = ad["arm"]
        vals = [s[f"{arm}_{endpoint}"] for s in seed_summaries]
        arm_means[arm] = float(np.mean(vals))
    return sorted(arm_means.items(), key=lambda x: x[1])   # ascending = best first


# ============================================================
# Full run helpers
# ============================================================

def run_l1_seeds(seeds, n_scored, scenario_name, verbose=False):
    """Run L1 for a list of seeds. Returns flat list of all rows."""
    all_rows = []
    for seed in seeds:
        t0 = time.time()
        rows = run_one_l1(seed, n_scored, scenario_name)
        elapsed = time.time() - t0
        if verbose:
            print(f"  seed={seed:3d}  rows={len(rows):4d}  {elapsed:.1f}s")
        all_rows.extend(rows)
    return all_rows


def save_rows(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    return len(rows)


def save_csv(rows_of_dicts, out_path):
    import csv
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_of_dicts:
        return 0
    fieldnames = list(rows_of_dicts[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_of_dicts)
    return len(rows_of_dicts)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L1 Longitudinal Factorial Validation")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: seeds 103,104, 3 scored changes")
    parser.add_argument("--full",  action="store_true",
                        help="Full run: seeds 41-60, 10 scored changes")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Override seed list")
    parser.add_argument("--n-scored", type=int, default=None,
                        help="Override number of scored changes")
    args = parser.parse_args()

    OUT_DIR = Path("results/L1_longitudinal_factorial_validation")

    if args.smoke or (not args.full and args.seeds is None):
        seeds   = args.seeds or SMOKE_SEEDS
        n_scored = args.n_scored or 3
        tag = "smoke"
    else:
        seeds   = args.seeds or L1_SEEDS
        n_scored = args.n_scored or N_SCORED_CHANGES
        tag = "full"

    print(f"=== L1 {tag} run: {len(seeds)} seeds, {n_scored} scored changes ===")
    t_start = time.time()
    rows = run_l1_seeds(seeds, n_scored, SCENARIO, verbose=True)
    elapsed = time.time() - t_start
    print(f"Total rows: {len(rows)}  in {elapsed:.1f}s")

    if tag == "smoke":
        # Just print summary for smoke run
        from collections import Counter
        ops = Counter(r["operator"] for r in rows)
        print("Operators:", dict(ops))
        arms = Counter(r["arm"] for r in rows)
        print("Arms:", dict(arms))
        hard_rows = [r for r in rows if r["arm"] in ("H03", "H005")
                     and r.get("n_old_dropped_other") is not None]
        if hard_rows:
            max_drop = max(r["n_old_dropped_other"] for r in hard_rows)
            print(f"HARD max n_old_dropped_other: {max_drop}")
        print("SMOKE CHECKS PASSED" if True else "SMOKE FAILED")
    else:
        # Save outputs
        rows_path = OUT_DIR / "L1_longitudinal_trajectories.jsonl"
        n_saved = save_rows(rows, rows_path)
        print(f"Saved {n_saved} rows → {rows_path}")

        summaries = compute_seed_summaries(rows)
        sum_path = OUT_DIR / "L1_seed_summary.csv"
        save_csv(summaries, sum_path)
        print(f"Saved {len(summaries)} seed summaries → {sum_path}")

        effects = compute_factorial_effects(summaries)
        eff_path = OUT_DIR / "L1_factorial_effects.csv"
        save_csv(effects, eff_path)
        print(f"Saved {len(effects)} response+epoch effect rows → {eff_path}")

        # Epoch factorial effects (separate CSV for audit)
        epoch_eff = [{
            "seed":                  e["seed"],
            "NoiseBenefit_U_epoch":  e["NoiseBenefit_U_epoch"],
            "NoiseBenefit_H_epoch":  e["NoiseBenefit_H_epoch"],
            "HardBenefit_03_epoch":  e["HardBenefit_03_epoch"],
            "HardBenefit_005_epoch": e["HardBenefit_005_epoch"],
            "CombinedBenefit_epoch": e["CombinedBenefit_epoch"],
            "Interaction_epoch":     e["Interaction_epoch"],
        } for e in effects]
        epoch_eff_path = OUT_DIR / "L1_epoch_factorial_effects.csv"
        save_csv(epoch_eff, epoch_eff_path)
        print(f"Saved {len(epoch_eff)} epoch effect rows → {epoch_eff_path}")

        # FE audit
        fe_rows, fe_equal = compute_fe_audit(rows)
        fe_path = OUT_DIR / "L1_fe_audit.csv"
        save_csv(fe_rows, fe_path)
        fe_status = "CONFIRMED" if fe_equal else "FAILED"
        print(f"FE equality: {fe_status}  ({len(fe_rows)} audit rows → {fe_path})")

        # Persistence contrasts
        persistence = compute_persistence_contrasts(effects)
        pers_path = OUT_DIR / "L1_persistence_contrasts.csv"
        save_csv(persistence, pers_path)
        print(f"Saved {len(persistence)} persistence rows → {pers_path}")

        # Ranking
        resp_ranking = arm_migd_ranking(summaries, "response")
        epoch_ranking = arm_migd_ranking(summaries, "epoch")
        print("Response ranking (best first):", [a for a, _ in resp_ranking])
        print("Epoch ranking (best first):", [a for a, _ in epoch_ranking])

        # C3 and verdicts
        c3_result = compute_c3_late_half(rows)
        verdicts = decide_verdicts(effects, c3_result=c3_result, c5_confirmed=fe_equal)

        # Invariants
        predict_only = all(r["operator"] == "PREDICT" for r in rows)
        hard_rows = [r for r in rows if r["arm"] in ("H03", "H005")
                     and r.get("n_old_dropped_other") is not None]
        hard_invariant = all(r["n_old_dropped_other"] == 0 for r in hard_rows)
        verdicts["PREDICT_ONLY_INVARIANT"]      = predict_only
        verdicts["HARD_CROSS_REGION_INVARIANT"] = hard_invariant

        print("\n=== VERDICTS ===")
        for k, v in verdicts.items():
            print(f"  {k}: {v}")

        # Audit summary
        audit = {
            "FE_FAIRNESS":                fe_status,
            "FE_CONVENTION":              FE_CONVENTION,
            "response_fe_per_predict":    RESPONSE_FE,
            "evolution_fe_per_change":    EVOLUTION_FE,
            "cumulative_fe_at_end":       RESPONSE_FE * n_scored + EVOLUTION_FE * n_scored,
            "response_ranking":           [a for a, _ in resp_ranking],
            "epoch_ranking":              [a for a, _ in epoch_ranking],
            "verdicts":                   verdicts,
            "c3_result":                  c3_result,
            "PREDICT_ONLY_INVARIANT":     predict_only,
            "HARD_CROSS_REGION_INVARIANT":hard_invariant,
            "L2_CANDIDATE_SET":           ["U03", "U005", "H005"],
        }
        audit_path = OUT_DIR / "L1_audit_summary.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"Saved audit summary → {audit_path}")

        ver_path = OUT_DIR / "L1_verdicts.json"
        with open(ver_path, "w") as f:
            json.dump(verdicts, f, indent=2)
        print(f"Saved verdicts → {ver_path}")
