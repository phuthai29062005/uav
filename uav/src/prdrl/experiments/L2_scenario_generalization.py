"""
L2 — Scenario Generalization of F005 and HARD Response Protection.

Scientific question:
    Do the PERSISTENT findings from L1 (lower noise F005 and HARD immediate
    protection) generalize to other dynamic scenarios already defined in the
    repository?

    A. Does lower noise (U03 → U005) generalize across scenarios?
    B. Does HARD immediate response protection (U005 → H005 post-response)
       generalize across scenarios?
    C. Does HARD post-evolution persistence generalize across scenarios?
    D. Does the combined candidate (U03 → H005) generalize across scenarios?

DESIGN:
    Scenarios: PERSISTENT (fresh replication), STATIONARY.
    Primary arms: U03 (baseline), U005 (lower noise), H005 (lower noise + HARD).
    NO H03 primary arm.
    Seeds: 61–80 (verified fresh, disjoint from all prior scientific seeds).
    N_SCORED_CHANGES = 10 per seed per scenario.

EFFECTS (per scenario, positive = candidate is better):
    NoiseBenefit_response   = MIGD_response(U03)  - MIGD_response(U005)
    NoiseBenefit_epoch      = MIGD_epoch(U03)     - MIGD_epoch(U005)
    HardResponseBenefit     = MIGD_response(U005) - MIGD_response(H005)
    HardEpochBenefit        = MIGD_epoch(U005)    - MIGD_epoch(H005)
    CombinedBenefit_response = MIGD_response(U03) - MIGD_response(H005)
    CombinedBenefit_epoch   = MIGD_epoch(U03)     - MIGD_epoch(H005)

VERDICTS:
    F005_SCENARIO_GENERALIZATION, HARD_RESPONSE_GENERALIZATION,
    HARD_EPOCH_PERSISTENCE, H005_COMBINED_GENERALIZATION.

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged.
    - Production predict() unmodified.
    - region_protected_survival unmodified.
    - nsga2_survival_traced unmodified.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - Do NOT start RL.
    - Do NOT overwrite L1 or prior results.
    - Seeds 61-80 only (disjoint from 1-60, 101-106).
    - No H03 in primary scientific rows.
    - Scenario list frozen before scientific run.
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

# L2 primary arms — 3 arms, no H03
L2_ARM_DEFS = [
    {"arm": "U03",  "sigma": 0.03,  "use_hard": False},
    {"arm": "U005", "sigma": 0.005, "use_hard": False},
    {"arm": "H005", "sigma": 0.005, "use_hard": True},
]
L2_PRIMARY_ARMS = [a["arm"] for a in L2_ARM_DEFS]

N_SCORED_CHANGES   = 10
L2_SEEDS           = list(range(61, 81))   # 20 seeds, verified fresh
DELTA_GENERALIZATION = 1e-5                # practical margin, pre-declared
BOOTSTRAP_SEED     = 42
BOOTSTRAP_RESAMPLES = 10_000
SMOKE_SEEDS        = [105, 106]
L2_SCENARIOS       = ["PERSISTENT", "STATIONARY"]   # frozen before run

# FE accounting — same convention as L1
RESPONSE_FE   = BUDGET                           # 10 per PREDICT/LOCAL response
EVOLUTION_FE  = GENERATIONS_PER_CHANGE * POP_SIZE   # 10 × 100 = 1,000 per change
FE_CONVENTION = "children_plus_evolution_no_reeval"

# Aliases
compute_time             = e2p3.compute_time
evolve_nsga2             = e2p3.evolve_nsga2
make_rng                 = e2p3.make_rng
make_scenario_dynamics   = e2p3.make_scenario_dynamics
nsga2_survival_traced    = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed   = e2p3._e2p3_common_surv_seed
generate_base_samples    = e2p3.generate_base_samples
apply_samples            = e2p3.apply_samples

region_protected_survival     = r1.region_protected_survival
compute_pf_coverage_per_point = r1.compute_pf_coverage_per_point


# ============================================================
# Target region schedule (same as L1)
# ============================================================

def target_region(seed, k, R):
    """Round-robin target region for scored change k of seed."""
    return (seed + k) % R


# ============================================================
# Diagnostics helpers (read-only, no RNG consumed)
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
    n_clipped = 0
    for p_idx, xi in zip(parent_idxs, xis):
        parent = X_region[p_idx]
        child  = np.clip(parent + delta_hat + sigma * xi, xl_a, xu_a)
        if not np.allclose(parent + delta_hat + sigma * xi, child):
            n_clipped += 1
    return float(n_clipped / max(len(parent_idxs), 1))


# ============================================================
# Bootstrap CI (same as L1)
# ============================================================

def bootstrap_contrast_ci(values, n_resamples=BOOTSTRAP_RESAMPLES,
                           rng_seed=BOOTSTRAP_SEED, ci=0.95):
    """Bootstrap 95% percentile CI for a set of per-seed effect values."""
    arr = np.asarray(values, dtype=float)
    n   = len(arr)
    rng = np.random.default_rng(rng_seed)
    means = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_resamples)]
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return float(np.mean(arr)), lo, hi


# ============================================================
# Effect classification
# ============================================================

def classify_effect(mean_val, ci_lo, ci_hi, delta=DELTA_GENERALIZATION):
    """Classify an effect into descriptive categories.

    BENEFICIAL          mean > +delta AND ci_lo > 0
    DIRECTIONAL_POSITIVE mean > +delta AND ci_lo <= 0
    PRACTICALLY_NEUTRAL  abs(mean) <= delta
    DIRECTIONAL_NEGATIVE mean < -delta AND ci_hi >= 0
    HARMFUL              mean < -delta AND ci_hi < 0
    """
    if mean_val > delta and ci_lo > 0:
        return "BENEFICIAL"
    elif mean_val > delta:
        return "DIRECTIONAL_POSITIVE"
    elif abs(mean_val) <= delta:
        return "PRACTICALLY_NEUTRAL"
    elif mean_val < -delta and ci_hi < 0:
        return "HARMFUL"
    else:
        return "DIRECTIONAL_NEGATIVE"


# ============================================================
# Main L2 runner — one seed, one scenario
# ============================================================

def run_one_l2(seed, n_scored, scenario_name, arm_defs=None):
    """Run L2 longitudinal generalization for one seed and one scenario.

    Args:
        seed:          Random seed (L2_SEEDS 61..80 for full run; SMOKE_SEEDS for smoke).
        n_scored:      Number of scored changes (10 for full, 3 for smoke).
        scenario_name: Scenario name passed to make_scenario_dynamics.
        arm_defs:      List of arm defs (default: L2_ARM_DEFS).

    Returns list of row dicts, one per (arm, scored_change_idx).
    Each row includes `scenario` field.
    """
    if arm_defs is None:
        arm_defs = L2_ARM_DEFS

    xl, xu = get_bounds(D)
    V  = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)

    region_dynamics = make_scenario_dynamics(scenario_name, R)
    pf_star = get_true_pf(n_points=500)

    # ---- 1. Warmup (common to all arms within this scenario) ----
    rng   = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
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

    # ---- 3. Clone state into arms ----
    arm_states = {}
    for ad in arm_defs:
        arm_states[ad["arm"]] = {
            "sigma":     ad["sigma"],
            "use_hard":  ad["use_hard"],
            "X":         X_pop.copy(),
            "history":   copy.deepcopy(history),
            "history_t": history_t,
        }

    rows_out = []

    # ---- 4. Scored changes ----
    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r = target_region(seed, k, R)

        gen   += GENERATIONS_PER_CHANGE
        t_new  = compute_time(gen)

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

        # ---- COMMON_KEEP check (all arms must have target occupied) ----
        common_available = all(arm_eval[a]["n_r"] > 0 for a in arm_eval)
        evo_seed  = seed * 10000 + scored_change_idx
        surv_seed = _e2p3_common_surv_seed(seed, scored_change_idx, tgt_r)

        if not common_available:
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
                    "scenario":          scenario_name,
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
        rng_crn  = make_rng(seed, scored_change_idx, tgt_r, "L2_RESPONSE")
        u_parent = rng_crn.uniform(size=BUDGET)
        xi_mat   = rng_crn.standard_normal(size=(BUDGET, D))
        xis      = list(xi_mat)

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

            parent_idxs = [min(int(u * n_r), n_r - 1) for u in u_parent]

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

            surv_old_set = set(int(i) for i in idx_s if i < N_old)
            old_other_ids = [i for i in range(N_old) if rids_arm[i] != tgt_r]
            n_old_dropped_other = sum(1 for i in old_other_ids if i not in surv_old_set)

            igd_response = float(igd(F_s, pf_star))

            arm["X"] = X_s
            arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                    GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
            F_epoch   = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_epoch = float(igd(F_epoch, pf_star))

            rows_out.append({
                "scenario":          scenario_name,
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
# Seed-level MIGD summaries (per scenario)
# ============================================================

def compute_l2_seed_summaries(rows, scenario_name, arm_defs=None):
    """Aggregate rows into per-seed MIGD summaries for a single scenario.

    Returns one dict per seed with {arm}_response and {arm}_epoch fields.
    """
    from collections import defaultdict
    if arm_defs is None:
        arm_defs = L2_ARM_DEFS

    scenario_rows = [r for r in rows if r["scenario"] == scenario_name]
    by_seed_arm = defaultdict(list)
    for r in scenario_rows:
        by_seed_arm[(r["seed"], r["arm"])].append(r)

    seeds = sorted(set(r["seed"] for r in scenario_rows))
    summaries = []
    for seed in seeds:
        d = {"scenario": scenario_name, "seed": seed}
        for ad in arm_defs:
            arm = ad["arm"]
            arm_rows = by_seed_arm[(seed, arm)]
            if not arm_rows:
                d[f"{arm}_response"] = float("nan")
                d[f"{arm}_epoch"]    = float("nan")
                d[f"n_predict_{arm}"]      = 0
                d[f"n_local_{arm}"]        = 0
                d[f"n_keep_{arm}"]         = 0
                d[f"early_response_{arm}"]  = float("nan")
                d[f"late_response_{arm}"]   = float("nan")
                d[f"early_epoch_{arm}"]     = float("nan")
                d[f"late_epoch_{arm}"]      = float("nan")
                continue
            d[f"{arm}_response"] = float(np.mean([r["igd_response"] for r in arm_rows]))
            d[f"{arm}_epoch"]    = float(np.mean([r["igd_epoch_end"] for r in arm_rows]))
            d[f"n_predict_{arm}"] = sum(1 for r in arm_rows if r.get("operator") == "PREDICT")
            d[f"n_local_{arm}"]   = sum(1 for r in arm_rows if r.get("operator") == "LOCAL_FALLBACK")
            d[f"n_keep_{arm}"]    = sum(1 for r in arm_rows if r.get("operator") == "COMMON_KEEP")
            early = [r for r in arm_rows if r["k"] < 5]
            late  = [r for r in arm_rows if r["k"] >= 5]
            d[f"early_response_{arm}"] = float(np.mean([r["igd_response"] for r in early])) if early else float("nan")
            d[f"late_response_{arm}"]  = float(np.mean([r["igd_response"] for r in late]))  if late  else float("nan")
            d[f"early_epoch_{arm}"]    = float(np.mean([r["igd_epoch_end"] for r in early])) if early else float("nan")
            d[f"late_epoch_{arm}"]     = float(np.mean([r["igd_epoch_end"] for r in late]))  if late  else float("nan")
        summaries.append(d)
    return summaries


# ============================================================
# L2 scenario effects
# ============================================================

def compute_l2_scenario_effects(summaries, scenario_name):
    """Compute per-seed L2 effects for a single scenario.

    Effects:
        NoiseBenefit_response   = MIGD_response(U03) - MIGD_response(U005)
        NoiseBenefit_epoch      = MIGD_epoch(U03)    - MIGD_epoch(U005)
        HardResponseBenefit     = MIGD_response(U005) - MIGD_response(H005)
        HardEpochBenefit        = MIGD_epoch(U005)   - MIGD_epoch(H005)
        CombinedBenefit_response = MIGD_response(U03) - MIGD_response(H005)
        CombinedBenefit_epoch   = MIGD_epoch(U03)    - MIGD_epoch(H005)
        HardPersistence         = HardEpochBenefit - HardResponseBenefit
        CombinedPersistence     = CombinedBenefit_epoch - CombinedBenefit_response

    Returns list of per-seed dicts for this scenario.
    """
    sc_sums = [s for s in summaries if s.get("scenario") == scenario_name]
    effects = []
    for s in sc_sums:
        u03r  = s["U03_response"];  u005r = s["U005_response"]; h005r = s["H005_response"]
        u03e  = s["U03_epoch"];     u005e = s["U005_epoch"];    h005e = s["H005_epoch"]

        nb_r  = u03r  - u005r   # F005 response benefit
        nb_e  = u03e  - u005e   # F005 epoch benefit
        hrb   = u005r - h005r   # HARD immediate response benefit
        heb   = u005e - h005e   # HARD epoch benefit
        cb_r  = u03r  - h005r   # Combined response benefit
        cb_e  = u03e  - h005e   # Combined epoch benefit
        hpers = heb   - hrb     # HARD persistence (negative = decays)
        cpers = cb_e  - cb_r    # Combined persistence (negative = decays)

        effects.append({
            "scenario":               scenario_name,
            "seed":                   s["seed"],
            "NoiseBenefit_response":  float(nb_r),
            "NoiseBenefit_epoch":     float(nb_e),
            "HardResponseBenefit":    float(hrb),
            "HardEpochBenefit":       float(heb),
            "CombinedBenefit_response": float(cb_r),
            "CombinedBenefit_epoch":    float(cb_e),
            "HardPersistence":        float(hpers),
            "CombinedPersistence":    float(cpers),
        })
    return effects


# ============================================================
# Scenario-level bootstrap statistics
# ============================================================

def compute_scenario_stats(effects, scenario_name, effect_keys=None):
    """Compute bootstrap statistics for all effects in a scenario.

    Returns dict of effect_key -> {mean, median, iqr, ci_lo, ci_hi, classification}.
    """
    if effect_keys is None:
        effect_keys = [
            "NoiseBenefit_response", "NoiseBenefit_epoch",
            "HardResponseBenefit",   "HardEpochBenefit",
            "CombinedBenefit_response", "CombinedBenefit_epoch",
            "HardPersistence",       "CombinedPersistence",
        ]
    sc_effects = [e for e in effects if e["scenario"] == scenario_name]
    stats = {}
    for key in effect_keys:
        vals = [e[key] for e in sc_effects]
        arr  = np.asarray(vals, dtype=float)
        mean_v, ci_lo, ci_hi = bootstrap_contrast_ci(vals)
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        stats[key] = {
            "mean":    mean_v,
            "median":  float(np.median(arr)),
            "iqr":     float(q75 - q25),
            "ci_lo":   ci_lo,
            "ci_hi":   ci_hi,
            "n_seeds": len(vals),
            "classification": classify_effect(mean_v, ci_lo, ci_hi),
        }
    return stats


# ============================================================
# Meta-analysis across scenarios (equal scenario weight)
# ============================================================

def compute_meta_effects(all_effects, scenarios, effect_keys=None):
    """Construct equal-scenario-weight seed-level meta effects.

    For each seed, meta effect = mean of that seed's effect across included scenarios.
    Then bootstrap the 20 meta-effect values.

    Returns:
        meta_seed_rows: list of per-seed dicts with meta effects
        meta_stats: dict of effect_key -> bootstrap stats
    """
    if effect_keys is None:
        effect_keys = [
            "NoiseBenefit_response", "NoiseBenefit_epoch",
            "HardResponseBenefit",   "HardEpochBenefit",
            "CombinedBenefit_response", "CombinedBenefit_epoch",
        ]

    from collections import defaultdict
    by_seed_scenario = defaultdict(dict)
    for e in all_effects:
        by_seed_scenario[e["seed"]][e["scenario"]] = e

    seeds = sorted(by_seed_scenario.keys())
    meta_rows = []
    for seed in seeds:
        by_sc = by_seed_scenario[seed]
        row = {"seed": seed}
        for key in effect_keys:
            sc_vals = [by_sc[sc][key] for sc in scenarios if sc in by_sc]
            if sc_vals:
                row[f"meta_{key}"] = float(np.mean(sc_vals))
            else:
                row[f"meta_{key}"] = float("nan")
        meta_rows.append(row)

    meta_stats = {}
    for key in effect_keys:
        vals = [r[f"meta_{key}"] for r in meta_rows if not np.isnan(r[f"meta_{key}"])]
        mean_v, ci_lo, ci_hi = bootstrap_contrast_ci(vals)
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        meta_stats[key] = {
            "mean":    mean_v,
            "median":  float(np.median(vals)),
            "iqr":     float(q75 - q25),
            "ci_lo":   ci_lo,
            "ci_hi":   ci_hi,
            "n_seeds": len(vals),
            "classification": classify_effect(mean_v, ci_lo, ci_hi),
        }

    return meta_rows, meta_stats


# ============================================================
# FE audit (L2: 3 arms — U03, U005, H005)
# ============================================================

def compute_l2_fe_audit(rows, scenario_name, arm_defs=None):
    """Verify FE equality across L2 arms for a single scenario.

    Returns (fe_rows, fe_equal). fe_equal is True when cumulative FE
    is identical across all arms at every (seed, k).
    """
    from collections import defaultdict
    if arm_defs is None:
        arm_defs = L2_ARM_DEFS
    arm_names = [a["arm"] for a in arm_defs]

    sc_rows = [r for r in rows if r["scenario"] == scenario_name]
    by_seed_arm_k = defaultdict(dict)
    for r in sc_rows:
        by_seed_arm_k[(r["seed"], r["arm"])][r["k"]] = r

    seeds = sorted(set(r["seed"] for r in sc_rows))
    fe_rows = []
    fe_equal = True

    for seed in seeds:
        cum = {arm: 0 for arm in arm_names}
        k_vals = sorted(set(r["k"] for r in sc_rows if r["seed"] == seed))
        for k in k_vals:
            ref_fe = None
            for arm in arm_names:
                row = by_seed_arm_k.get((seed, arm), {}).get(k)
                if row is None:
                    continue
                r_fe = row.get("response_fe", 0)
                e_fe = row.get("evolution_fe", EVOLUTION_FE)
                cum[arm] += r_fe + e_fe
                if ref_fe is None:
                    ref_fe = cum[arm]
                elif abs(cum[arm] - ref_fe) > 0:
                    fe_equal = False
            fe_rows.append({
                "scenario":    scenario_name,
                "seed":        seed,
                "k":           k,
                **{f"cumulative_fe_{arm}": cum[arm] for arm in arm_names},
                "fe_equal_at_k": (ref_fe is not None and
                                   all(abs(cum[a] - ref_fe) < 1 for a in arm_names
                                       if by_seed_arm_k.get((seed, a), {}).get(k) is not None)),
            })

    return fe_rows, fe_equal


# ============================================================
# HARD structural invariant check
# ============================================================

def check_hard_invariant(rows, scenario_name):
    """Verify n_old_dropped_other == 0 for all H005 PREDICT/LOCAL rows.

    Returns (n_violations, violation_rows).
    """
    sc_rows = [r for r in rows
               if r["scenario"] == scenario_name
               and r["arm"] == "H005"
               and r.get("operator") in ("PREDICT", "LOCAL_FALLBACK")
               and r.get("n_old_dropped_other") is not None]
    violations = [r for r in sc_rows if r["n_old_dropped_other"] != 0]
    return len(violations), violations


# ============================================================
# L2 verdicts
# ============================================================

def decide_l2_verdicts(scenario_stats, meta_stats, scenarios,
                        delta=DELTA_GENERALIZATION):
    """Apply L2 pre-declared verdict logic.

    Returns dict with F005_SCENARIO_GENERALIZATION, HARD_RESPONSE_GENERALIZATION,
    HARD_EPOCH_PERSISTENCE, H005_COMBINED_GENERALIZATION.
    """
    non_persistent = [s for s in scenarios if s != "PERSISTENT"]

    # ---- F005_SCENARIO_GENERALIZATION ----
    f005_harmful_any_non_pers = any(
        scenario_stats[s]["NoiseBenefit_response"]["classification"] == "HARMFUL"
        or scenario_stats[s]["NoiseBenefit_epoch"]["classification"] == "HARMFUL"
        for s in non_persistent
    )
    meta_nb_r = meta_stats["NoiseBenefit_response"]["mean"]
    meta_nb_e = meta_stats["NoiseBenefit_epoch"]["mean"]
    f005_non_pers_supported = any(
        scenario_stats[s]["NoiseBenefit_response"]["mean"] > delta
        and scenario_stats[s]["NoiseBenefit_epoch"]["mean"] > delta
        for s in non_persistent
    )

    if (not f005_harmful_any_non_pers
            and meta_nb_r > delta
            and meta_nb_e > delta
            and f005_non_pers_supported):
        f005_verdict = "SUPPORTED"
    elif (not f005_harmful_any_non_pers
          and (meta_nb_r > delta or meta_nb_e > delta)):
        f005_verdict = "PARTIAL"
    elif any(
        scenario_stats[s]["NoiseBenefit_response"]["classification"] == "HARMFUL"
        or scenario_stats[s]["NoiseBenefit_epoch"]["classification"] == "HARMFUL"
        for s in non_persistent
    ) or (meta_nb_r < -delta and meta_nb_e < -delta):
        f005_verdict = "NOT_SUPPORTED"
    else:
        f005_verdict = "PARTIAL"

    # ---- HARD_RESPONSE_GENERALIZATION ----
    hard_r_harmful = any(
        scenario_stats[s]["HardResponseBenefit"]["classification"] == "HARMFUL"
        for s in non_persistent
    )
    meta_hrb = meta_stats["HardResponseBenefit"]["mean"]
    hard_r_non_pers_supported = any(
        scenario_stats[s]["HardResponseBenefit"]["mean"] > delta
        for s in non_persistent
    )

    if (not hard_r_harmful
            and meta_hrb > delta
            and hard_r_non_pers_supported):
        hard_r_verdict = "SUPPORTED"
    elif not hard_r_harmful and meta_hrb > -delta:
        hard_r_verdict = "PARTIAL"
    elif hard_r_harmful or meta_hrb < -delta:
        hard_r_verdict = "NOT_SUPPORTED"
    else:
        hard_r_verdict = "PARTIAL"

    # ---- HARD_EPOCH_PERSISTENCE ----
    hard_e_harmful = any(
        scenario_stats[s]["HardEpochBenefit"]["classification"] == "HARMFUL"
        for s in non_persistent
    )
    meta_heb = meta_stats["HardEpochBenefit"]["mean"]
    hard_e_non_pers_supported = any(
        scenario_stats[s]["HardEpochBenefit"]["mean"] > delta
        for s in non_persistent
    )

    if (not hard_e_harmful
            and meta_heb > delta
            and hard_e_non_pers_supported):
        hard_e_verdict = "SUPPORTED"
    elif hard_e_harmful or (meta_heb < -delta
                             and meta_stats["HardEpochBenefit"]["ci_hi"] < 0):
        hard_e_verdict = "NOT_SUPPORTED"
    else:
        hard_e_verdict = "PARTIAL"

    # ---- H005_COMBINED_GENERALIZATION (G1-G6) ----
    # G1: no non-PERSISTENT scenario has HARMFUL CombinedBenefit_response
    g1 = not any(
        scenario_stats[s]["CombinedBenefit_response"]["classification"] == "HARMFUL"
        for s in non_persistent
    )
    # G2: meta combined response > +1e-5
    g2 = meta_stats["CombinedBenefit_response"]["mean"] > delta
    # G3: bootstrap meta response CI lower > 0
    g3 = meta_stats["CombinedBenefit_response"]["ci_lo"] > 0
    # G4: meta combined epoch >= -1e-5
    g4 = meta_stats["CombinedBenefit_epoch"]["mean"] >= -delta
    # G5: at least one non-PERSISTENT has combined response > +1e-5
    g5 = any(
        scenario_stats[s]["CombinedBenefit_response"]["mean"] > delta
        for s in non_persistent
    )
    # G6: FE fairness confirmed (passed externally via fe_equal parameter)
    # Will be set by caller

    if g1 and g2 and g3 and g4 and g5:
        combined_verdict = "VALIDATED_PENDING_G6"
    elif g1 and g2 and g4:
        combined_verdict = "PARTIAL"
    elif not g1 or (meta_stats["CombinedBenefit_response"]["mean"] < -delta):
        combined_verdict = "NOT_VALIDATED"
    else:
        combined_verdict = "PARTIAL"

    return {
        "F005_SCENARIO_GENERALIZATION":  f005_verdict,
        "HARD_RESPONSE_GENERALIZATION":  hard_r_verdict,
        "HARD_EPOCH_PERSISTENCE":        hard_e_verdict,
        "H005_COMBINED_GENERALIZATION":  combined_verdict,
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5,
        "meta_NoiseBenefit_response":    meta_nb_r,
        "meta_NoiseBenefit_epoch":       meta_nb_e,
        "meta_HardResponseBenefit":      meta_hrb,
        "meta_HardEpochBenefit":         meta_heb,
        "meta_CombinedBenefit_response": meta_stats["CombinedBenefit_response"]["mean"],
        "meta_CombinedBenefit_epoch":    meta_stats["CombinedBenefit_epoch"]["mean"],
        "meta_cb_response_ci_lo":        meta_stats["CombinedBenefit_response"]["ci_lo"],
    }


# ============================================================
# Full-run helpers
# ============================================================

def run_l2_seeds(seeds, n_scored, scenarios, arm_defs=None, verbose=False):
    """Run L2 for all seeds × scenarios. Returns flat list of all rows."""
    all_rows = []
    for scenario in scenarios:
        for seed in seeds:
            t0 = time.time()
            rows = run_one_l2(seed, n_scored, scenario, arm_defs=arm_defs)
            elapsed = time.time() - t0
            if verbose:
                print(f"  scenario={scenario:12s}  seed={seed:3d}  rows={len(rows):4d}  {elapsed:.1f}s")
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
# Temporal architecture hypothesis assessment
# ============================================================

def assess_temporal_architecture(scenario_stats, meta_stats, scenarios):
    """Assess: 'F005 is persistent; HARD is a transient response safeguard.'

    Returns SUPPORTED / PARTIAL / NOT_SUPPORTED.
    """
    # F005 persistent: positive at both response AND epoch across scenarios
    f005_persistent_evidence = (
        meta_stats["NoiseBenefit_response"]["mean"] > 0
        and meta_stats["NoiseBenefit_epoch"]["mean"] > 0
        and meta_stats["NoiseBenefit_epoch"]["classification"] not in ("HARMFUL", "DIRECTIONAL_NEGATIVE")
    )

    # HARD transient: positive at response, negative/neutral at epoch
    hard_transient_evidence = (
        meta_stats["HardResponseBenefit"]["mean"] > 0
        and meta_stats["HardEpochBenefit"]["mean"] < 0
    )
    hard_transient_partial = (
        meta_stats["HardResponseBenefit"]["mean"] > 0
        and meta_stats["HardEpochBenefit"]["mean"] <= meta_stats["HardResponseBenefit"]["mean"]
        and meta_stats["HardEpochBenefit"]["mean"] >= -DELTA_GENERALIZATION
    )

    if f005_persistent_evidence and hard_transient_evidence:
        return "SUPPORTED"
    elif f005_persistent_evidence and (hard_transient_evidence or hard_transient_partial):
        return "SUPPORTED"
    elif f005_persistent_evidence or hard_transient_evidence:
        return "PARTIAL"
    else:
        return "NOT_SUPPORTED"


# ============================================================
# Next stage recommendation
# ============================================================

def recommend_next_stage(verdicts):
    """Map L2 verdicts to next-stage recommendation."""
    f005 = verdicts.get("F005_SCENARIO_GENERALIZATION", "")
    comb = verdicts.get("H005_COMBINED_GENERALIZATION", "")
    hard_r = verdicts.get("HARD_RESPONSE_GENERALIZATION", "")

    if f005 == "SUPPORTED" and comb in ("VALIDATED", "VALIDATED_PENDING_G6"):
        return "D1_DESIGN_RECONCILIATION"
    elif f005 == "SUPPORTED" and hard_r == "NOT_SUPPORTED":
        return "U005_DESIGN_PATH"
    elif f005 in ("PARTIAL", "NOT_SUPPORTED") or comb == "NOT_VALIDATED":
        return "SCENARIO_FAILURE_DIAGNOSIS"
    else:
        return "D1_DESIGN_RECONCILIATION"


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L2 Scenario Generalization")
    parser.add_argument("--smoke",    action="store_true",
                        help="Smoke run: seeds 105,106, 3 scored changes")
    parser.add_argument("--full",     action="store_true",
                        help="Full run: seeds 61-80, 10 scored changes, all scenarios")
    parser.add_argument("--seeds",    type=int, nargs="+")
    parser.add_argument("--n-scored", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Override scenario list (default: PERSISTENT STATIONARY)")
    args = parser.parse_args()

    OUT_DIR = Path("results/L2_scenario_generalization")

    scenarios = args.scenarios or L2_SCENARIOS

    if args.smoke or (not args.full and args.seeds is None):
        seeds    = args.seeds or SMOKE_SEEDS
        n_scored = args.n_scored or 3
        tag = "smoke"
    else:
        seeds    = args.seeds or L2_SEEDS
        n_scored = args.n_scored or N_SCORED_CHANGES
        tag = "full"

    print(f"=== L2 {tag} run: {len(seeds)} seeds × {len(scenarios)} scenarios, "
          f"{n_scored} scored changes ===")
    print(f"    Scenarios: {scenarios}")
    print(f"    Arms: {L2_PRIMARY_ARMS}")
    t_start = time.time()
    rows = run_l2_seeds(seeds, n_scored, scenarios, verbose=True)
    elapsed = time.time() - t_start
    print(f"Total rows: {len(rows)}  in {elapsed:.1f}s")

    if tag == "smoke":
        from collections import Counter
        ops = Counter(r["operator"] for r in rows)
        print("Operators:", dict(ops))
        arms = Counter(r["arm"] for r in rows)
        print("Arms:", dict(arms))
        scs  = Counter(r["scenario"] for r in rows)
        print("Scenarios:", dict(scs))
        hard_rows = [r for r in rows if r["arm"] == "H005"
                     and r.get("n_old_dropped_other") is not None]
        if hard_rows:
            violations = [r for r in hard_rows if r["n_old_dropped_other"] != 0]
            print(f"HARD invariant violations: {len(violations)} / {len(hard_rows)}")
        # FE check for each scenario
        for sc in scenarios:
            _, fe_eq = compute_l2_fe_audit(rows, sc)
            print(f"FE equality [{sc}]: {'CONFIRMED' if fe_eq else 'FAILED'}")
        print("SMOKE CHECKS COMPLETE")

    else:
        # ---- Save raw trajectories ----
        rows_path = OUT_DIR / "L2_scenario_generalization.jsonl"
        n_saved = save_rows(rows, rows_path)
        print(f"Saved {n_saved} rows → {rows_path}")

        # ---- Per-scenario seed summaries ----
        all_summaries = []
        for sc in scenarios:
            sums = compute_l2_seed_summaries(rows, sc)
            all_summaries.extend(sums)
        sum_path = OUT_DIR / "L2_seed_summary.csv"
        save_csv(all_summaries, sum_path)
        print(f"Saved {len(all_summaries)} seed summaries → {sum_path}")

        # ---- Per-scenario effects ----
        all_effects = []
        for sc in scenarios:
            effs = compute_l2_scenario_effects(all_summaries, sc)
            all_effects.extend(effs)
        eff_path = OUT_DIR / "L2_scenario_effects.csv"
        save_csv(all_effects, eff_path)
        print(f"Saved {len(all_effects)} scenario effect rows → {eff_path}")

        # ---- Per-scenario bootstrap stats ----
        sc_stats = {}
        for sc in scenarios:
            sc_stats[sc] = compute_scenario_stats(all_effects, sc)

        # ---- Meta effects ----
        meta_rows, meta_stats = compute_meta_effects(all_effects, scenarios)
        meta_path = OUT_DIR / "L2_meta_effects.csv"
        save_csv(meta_rows, meta_path)
        print(f"Saved {len(meta_rows)} meta-effect rows → {meta_path}")

        # ---- FE audit ----
        fe_ok_all = True
        all_fe_rows = []
        for sc in scenarios:
            fe_rows, fe_eq = compute_l2_fe_audit(rows, sc)
            all_fe_rows.extend(fe_rows)
            status = "CONFIRMED" if fe_eq else "FAILED"
            print(f"FE equality [{sc}]: {status}")
            if not fe_eq:
                fe_ok_all = False
        fe_path = OUT_DIR / "L2_fe_audit.csv"
        save_csv(all_fe_rows, fe_path)
        fe_all_status = "CONFIRMED" if fe_ok_all else "FAILED"

        # ---- HARD invariant ----
        hard_ok_all = True
        for sc in scenarios:
            n_viol, _ = check_hard_invariant(rows, sc)
            if n_viol > 0:
                print(f"HARD INVARIANT VIOLATION [{sc}]: {n_viol} violations — STOP")
                hard_ok_all = False
            else:
                print(f"HARD invariant [{sc}]: OK (0 violations)")

        # ---- Verdicts ----
        verdicts = decide_l2_verdicts(sc_stats, meta_stats, scenarios)
        # Resolve G6 with actual FE fairness
        if verdicts["H005_COMBINED_GENERALIZATION"] == "VALIDATED_PENDING_G6":
            verdicts["H005_COMBINED_GENERALIZATION"] = "VALIDATED" if fe_ok_all else "PARTIAL"
        verdicts["G6"] = fe_ok_all
        verdicts["FE_FAIRNESS_ALL_SCENARIOS"] = fe_all_status
        verdicts["HARD_INVARIANT_ALL_SCENARIOS"] = "OK" if hard_ok_all else "VIOLATED"

        # Temporal architecture
        temp_arch = assess_temporal_architecture(sc_stats, meta_stats, scenarios)
        verdicts["TEMPORAL_ARCHITECTURE_HYPOTHESIS"] = temp_arch

        # Next stage
        next_stage = recommend_next_stage(verdicts)
        verdicts["NEXT_STAGE"] = next_stage

        ver_path = OUT_DIR / "L2_verdicts.json"
        with open(ver_path, "w") as f:
            json.dump({"verdicts": verdicts, "scenario_stats": sc_stats,
                       "meta_stats": meta_stats}, f, indent=2)
        print(f"Saved verdicts → {ver_path}")

        print("\n=== L2 FINAL OUTPUT ===")
        print(f"L2_SCENARIOS = {scenarios}")
        print(f"FE_FAIRNESS_ALL_SCENARIOS = {fe_all_status}")
        print(f"F005_SCENARIO_GENERALIZATION = {verdicts['F005_SCENARIO_GENERALIZATION']}")
        print(f"HARD_RESPONSE_GENERALIZATION = {verdicts['HARD_RESPONSE_GENERALIZATION']}")
        print(f"HARD_EPOCH_PERSISTENCE = {verdicts['HARD_EPOCH_PERSISTENCE']}")
        print(f"H005_COMBINED_GENERALIZATION = {verdicts['H005_COMBINED_GENERALIZATION']}")
        print(f"TEMPORAL_ARCHITECTURE_HYPOTHESIS = {temp_arch}")
        print(f"NEXT_STAGE = {next_stage}")
