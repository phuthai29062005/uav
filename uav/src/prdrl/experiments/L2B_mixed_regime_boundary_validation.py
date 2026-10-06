"""
L2-B — Mixed-Regime Boundary Validation of F005 and HARD Response Protection.

Scientific question:
    When persistent and stationary regions coexist inside the same evolving population,
    are benefits of F005 and HARD concentrated where motion exists (persistent regions),
    while remaining non-harmful to stationary regions and to cross-regime PF coverage?

Uses the existing make_region_dynamics_map (half persistent, half stationary) from
src/prdrl/problems/controlled_predict.py directly — no new scenario invented.

DESIGN:
    Scenario: MIXED (make_region_dynamics_map(R), R=10: regions 0-4 persistent, 5-9 stationary)
    Primary arms: U03 (σ=0.03), U005 (σ=0.005), H005 (σ=0.005 + HARD).
    NO H03.
    Seeds: 81–100 (verified fresh, disjoint from all scientific seeds 1–80).
    N_SCORED_CHANGES = 10 per seed.

EFFECTS (positive = candidate is better):
    Global (all 10 changes):
        NB_resp_global = MIGD_resp(U03) - MIGD_resp(U005)
        HB_resp_global = MIGD_resp(U005) - MIGD_resp(H005)
        CB_resp_global = MIGD_resp(U03)  - MIGD_resp(H005)
        (epoch analogues)
    Persistent-target changes (5 per seed):
        NB_resp_P, HB_resp_P, CB_resp_P (+ epoch)
    Stationary-target changes (5 per seed):
        NB_resp_S, HB_resp_S, CB_resp_S (+ epoch)
    Selectivity contrasts:
        NB_selectivity_response = NB_resp_P - NB_resp_S
        HB_selectivity_response = HB_resp_P - HB_resp_S
        CB_selectivity_response = CB_resp_P - CB_resp_S

VERDICTS:
    F005_MIXED_BOUNDARY, HARD_MIXED_BOUNDARY,
    H005_MIXED_GLOBAL, REGIME_BOUNDARY_HYPOTHESIS.

CONSTRAINTS:
    - Production sigma_P=0.03 unchanged.
    - Production predict() unmodified.
    - region_protected_survival unmodified.
    - nsga2_survival_traced unmodified.
    - BUDGET=10, POP_SIZE=100, lambda_pred=1.0 unchanged.
    - Do NOT start RL.
    - Do NOT overwrite L1 or L2 results.
    - Seeds 81-100 only (disjoint from 1-80, 101-108).
    - No H03 in primary scientific rows.
    - Protocol frozen before scientific run (L2B_mixed_protocol.json).
    - MIXED map: NOT modified. Import existing make_region_dynamics_map directly.
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
    make_region_dynamics_map,
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

# L2-B primary arms — 3 arms, no H03
L2B_ARM_DEFS = [
    {"arm": "U03",  "sigma": 0.03,  "use_hard": False},
    {"arm": "U005", "sigma": 0.005, "use_hard": False},
    {"arm": "H005", "sigma": 0.005, "use_hard": True},
]
L2B_PRIMARY_ARMS = [a["arm"] for a in L2B_ARM_DEFS]

N_SCORED_CHANGES    = 10
L2B_SEEDS           = list(range(81, 101))   # 20 seeds, verified fresh
DELTA_BOUNDARY      = 1e-5                   # practical margin, pre-declared
BOOTSTRAP_SEED      = 42
BOOTSTRAP_RESAMPLES = 10_000
SMOKE_SEEDS         = [107, 108]

# FE accounting — same convention as L1/L2
RESPONSE_FE   = BUDGET                           # 10 per PREDICT/LOCAL response
EVOLUTION_FE  = GENERATIONS_PER_CHANGE * POP_SIZE   # 10 × 100 = 1,000 per change
FE_CONVENTION = "children_plus_evolution_no_reeval"

# Aliases — same runtime as L1/L2
compute_time             = e2p3.compute_time
evolve_nsga2             = e2p3.evolve_nsga2
make_rng                 = e2p3.make_rng
nsga2_survival_traced    = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed   = e2p3._e2p3_common_surv_seed
generate_base_samples    = e2p3.generate_base_samples
apply_samples            = e2p3.apply_samples

# D2-A: use canonical implementation from core (same as oracle.py and R1).
from prdrl.core.survival import region_protected_survival  # noqa: F401


# ============================================================
# Mixed scenario adapter (uses authoritative existing function)
# ============================================================

def make_l2b_mixed_dynamics(R):
    """Return the MIXED dynamics map using the authoritative existing function.

    Wraps make_region_dynamics_map(R) exactly — no reimplementation.
    regions 0..R//2-1 = persistent, R//2..R-1 = stationary for even R.
    """
    return make_region_dynamics_map(R)


def get_target_regime(tgt_r, mixed_map):
    """Return regime label for target region from frozen map.

    Returns 'PERSISTENT_REGION' or 'STATIONARY_REGION'.
    Does NOT infer regime from delta_hat.
    """
    label = mixed_map[tgt_r]
    if label == "persistent":
        return "PERSISTENT_REGION"
    else:
        return "STATIONARY_REGION"


def get_source_regime(region_id, mixed_map):
    """Return regime label of a source region ('persistent' or 'stationary')."""
    return mixed_map[region_id]


# ============================================================
# Target region schedule (same as L1/L2)
# ============================================================

def target_region(seed, k, R):
    """Round-robin target region for scored change k of seed."""
    return (seed + k) % R


# ============================================================
# PF zone analysis
# ============================================================

def make_pf_zones(pf_star, mixed_map):
    """Assign each pf_star point to a PF zone based on decision-space regime.

    For the benchmark: pf_star[i] = (cos(θ_i), sin(θ_i)) where θ_i = i*π/(2*(n-1)).
    Corresponding x1 = 2*θ/π, bin = floor(R * x1).
    Returns boolean mask: True = persistent zone.
    """
    n = len(pf_star)
    R = len(mixed_map)
    # Reconstruct theta from the quarter-circle PF
    # f1 = cos(theta), so theta = arccos(f1)
    theta = np.arccos(np.clip(pf_star[:, 0], 0.0, 1.0))
    x1 = 2.0 * theta / np.pi
    bins = np.minimum(np.floor(R * x1).astype(int), R - 1)
    is_persistent = np.array([mixed_map[b] == "persistent" for b in bins])
    return is_persistent


def compute_pf_zone_coverage(F_pop, pf_star, is_persistent_zone):
    """Compute mean nearest-distance from PF star to population, by zone.

    Returns (mean_dist_persistent, mean_dist_stationary).
    """
    if len(F_pop) == 0:
        return float("nan"), float("nan")
    dists = np.min(
        np.sum((pf_star[:, None, :] - F_pop[None, :, :]) ** 2, axis=2) ** 0.5,
        axis=1,
    )  # shape: (n_pf,)
    d_pers = float(np.mean(dists[is_persistent_zone])) if is_persistent_zone.any() else float("nan")
    d_stat = float(np.mean(dists[~is_persistent_zone])) if (~is_persistent_zone).any() else float("nan")
    return d_pers, d_stat


def compute_pf_zone_gains(F_before, F_after, pf_star, is_persistent_zone):
    """Compute per-point PF gains (d_before - d_after) and aggregate by zone.

    Positive gain = coverage improved.
    Returns dict with mean_pf_gain_persistent, mean_pf_gain_stationary,
    frac_pf_worse_persistent, frac_pf_worse_stationary.
    """
    if len(F_before) == 0 or len(F_after) == 0:
        return {
            "mean_pf_gain_persistent": float("nan"),
            "mean_pf_gain_stationary": float("nan"),
            "frac_pf_worse_persistent": float("nan"),
            "frac_pf_worse_stationary": float("nan"),
        }
    d_before = np.min(
        np.sum((pf_star[:, None, :] - F_before[None, :, :]) ** 2, axis=2) ** 0.5,
        axis=1,
    )
    d_after = np.min(
        np.sum((pf_star[:, None, :] - F_after[None, :, :]) ** 2, axis=2) ** 0.5,
        axis=1,
    )
    gain = d_before - d_after  # positive = improved

    p_mask = is_persistent_zone
    s_mask = ~is_persistent_zone

    def _mean_gain(mask):
        return float(np.mean(gain[mask])) if mask.any() else float("nan")

    def _frac_worse(mask):
        if not mask.any():
            return float("nan")
        return float(np.mean(gain[mask] < 0.0))

    return {
        "mean_pf_gain_persistent": _mean_gain(p_mask),
        "mean_pf_gain_stationary": _mean_gain(s_mask),
        "frac_pf_worse_persistent": _frac_worse(p_mask),
        "frac_pf_worse_stationary": _frac_worse(s_mask),
    }


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
# Bootstrap CI (same as L2)
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

def classify_effect(mean_val, ci_lo, ci_hi, delta=DELTA_BOUNDARY):
    """Classify an effect into 5 descriptive categories."""
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
# Main L2-B runner — one seed
# ============================================================

def run_one_l2b(seed, n_scored, mixed_map, arm_defs=None):
    """Run L2-B longitudinal experiment for one seed in the MIXED scenario.

    Args:
        seed:      Random seed (L2B_SEEDS 81..100 for full run).
        n_scored:  Number of scored changes (10 for full, 3 for smoke).
        mixed_map: Frozen regime map list (from make_l2b_mixed_dynamics(R)).
        arm_defs:  List of arm defs (default: L2B_ARM_DEFS).

    Returns list of row dicts, one per (arm, scored_change_idx).
    """
    if arm_defs is None:
        arm_defs = L2B_ARM_DEFS

    xl, xu = get_bounds(D)
    V  = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)

    assert R == len(mixed_map), f"mixed_map length {len(mixed_map)} != R={R}"

    region_dynamics = mixed_map  # use frozen map directly
    pf_star = get_true_pf(n_points=500)
    is_persistent_zone = make_pf_zones(pf_star, mixed_map)

    # ---- 1. Warmup (common to all arms) ----
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
        tgt_r  = target_region(seed, k, R)
        tgt_regime = get_target_regime(tgt_r, mixed_map)

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

        # ---- COMMON_KEEP check ----
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
                    "seed": seed, "arm": arm_name, "k": k,
                    "scored_change_idx": scored_change_idx,
                    "target_r": tgt_r, "target_regime": tgt_regime,
                    "operator": "COMMON_KEEP", "common_target_available": False,
                    "sigma_used": arm["sigma"], "has_predict_history": None,
                    "delta_hat_norm": None, "clipping_frac": None,
                    "n_old_dropped_other": None,
                    "n_old_dropped_other_persistent": None,
                    "n_old_dropped_other_stationary": None,
                    "n_active_regions_pre": n_active_pre,
                    "n_regions_with_valid_history": n_valid_hist,
                    "response_fe": 0, "evolution_fe": EVOLUTION_FE,
                    "total_fe_change": EVOLUTION_FE,
                    "igd_pre": igd_pre, "igd_response": igd_pre,
                    "igd_epoch_end": igd_epoch,
                    "response_gain": 0.0,
                    "epoch_gain": float(igd_pre - igd_epoch),
                    "mean_pf_gain_persistent": None,
                    "mean_pf_gain_stationary": None,
                    "frac_pf_worse_persistent": None,
                    "frac_pf_worse_stationary": None,
                })
            t_current = t_new
            continue

        # ---- CRN: arm-independent u_parent and xi ----
        rng_crn  = make_rng(seed, scored_change_idx, tgt_r, "L2B_RESPONSE")
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

            # Count total outside-target drops
            n_old_dropped_other = sum(1 for i in old_other_ids if i not in surv_old_set)

            # Drop-origin regime accounting (use SOURCE region, not actual occupancy)
            n_drop_pers = sum(
                1 for i in old_other_ids
                if i not in surv_old_set and get_source_regime(rids_arm[i], mixed_map) == "persistent"
            )
            n_drop_stat = sum(
                1 for i in old_other_ids
                if i not in surv_old_set and get_source_regime(rids_arm[i], mixed_map) == "stationary"
            )

            igd_response = float(igd(F_s, pf_star))

            # PF zone analysis (diagnostics — no RNG consumed)
            pf_zone_info = compute_pf_zone_gains(F_plus_arm, F_s, pf_star, is_persistent_zone)

            arm["X"] = X_s
            arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                    GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
            F_epoch   = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_epoch = float(igd(F_epoch, pf_star))

            rows_out.append({
                "seed": seed, "arm": arm_name, "k": k,
                "scored_change_idx": scored_change_idx,
                "target_r": tgt_r, "target_regime": tgt_regime,
                "operator": operator,
                "common_target_available": True,
                "sigma_used": sigma_used,
                "has_predict_history": has_history,
                "delta_hat_norm": delta_hat_norm,
                "clipping_frac": clip_frac,
                "n_old_dropped_other": n_old_dropped_other,
                "n_old_dropped_other_persistent": n_drop_pers,
                "n_old_dropped_other_stationary": n_drop_stat,
                "n_active_regions_pre": n_active_pre,
                "n_regions_with_valid_history": n_valid_hist,
                "response_fe": RESPONSE_FE,
                "evolution_fe": EVOLUTION_FE,
                "total_fe_change": RESPONSE_FE + EVOLUTION_FE,
                "igd_pre": igd_pre,
                "igd_response": igd_response,
                "igd_epoch_end": igd_epoch,
                "response_gain": float(igd_pre - igd_response),
                "epoch_gain": float(igd_pre - igd_epoch),
                **pf_zone_info,
            })

        t_current = t_new

    return rows_out


# ============================================================
# FE audit
# ============================================================

def compute_l2b_fe_audit(rows, arm_defs=None):
    """Verify FE equality across L2-B arms for every (seed, k).

    Returns (fe_rows, fe_equal).
    """
    from collections import defaultdict
    if arm_defs is None:
        arm_defs = L2B_ARM_DEFS
    arm_names = [a["arm"] for a in arm_defs]

    by_seed_arm_k = defaultdict(dict)
    for r in rows:
        by_seed_arm_k[(r["seed"], r["arm"])][r["k"]] = r

    seeds = sorted(set(r["seed"] for r in rows))
    fe_rows = []
    fe_equal = True

    for seed in seeds:
        cum = {arm: 0 for arm in arm_names}
        k_vals = sorted(set(r["k"] for r in rows if r["seed"] == seed))
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
                "seed": seed, "k": k,
                **{f"cumulative_fe_{arm}": cum[arm] for arm in arm_names},
                "fe_equal_at_k": (
                    ref_fe is not None and
                    all(abs(cum[a] - ref_fe) < 1 for a in arm_names
                        if by_seed_arm_k.get((seed, a), {}).get(k) is not None)
                ),
            })

    return fe_rows, fe_equal


# ============================================================
# HARD structural invariant check
# ============================================================

def check_hard_invariant(rows):
    """Verify n_old_dropped_other == 0 for all H005 active response rows.

    Also checks n_old_dropped_other_persistent == 0 and
    n_old_dropped_other_stationary == 0.

    Returns (ok, n_violations_total, n_violations_pers, n_violations_stat).
    """
    h005_rows = [r for r in rows
                 if r["arm"] == "H005"
                 and r.get("operator") in ("PREDICT", "LOCAL_FALLBACK")
                 and r.get("n_old_dropped_other") is not None]
    v_total = [r for r in h005_rows if r["n_old_dropped_other"] != 0]
    v_pers  = [r for r in h005_rows
               if r.get("n_old_dropped_other_persistent") is not None
               and r["n_old_dropped_other_persistent"] != 0]
    v_stat  = [r for r in h005_rows
               if r.get("n_old_dropped_other_stationary") is not None
               and r["n_old_dropped_other_stationary"] != 0]
    ok = (len(v_total) == 0 and len(v_pers) == 0 and len(v_stat) == 0)
    return ok, len(v_total), len(v_pers), len(v_stat)


# ============================================================
# Seed-level summaries
# ============================================================

def compute_l2b_seed_summaries(rows, arm_defs=None):
    """Aggregate rows into per-seed global and regime-conditioned MIGD summaries.

    Returns list of per-seed dicts including:
    - {arm}_response_global, {arm}_epoch_global  (all 10 changes)
    - {arm}_response_P, {arm}_epoch_P            (persistent-target changes)
    - {arm}_response_S, {arm}_epoch_S            (stationary-target changes)
    - regime counts
    """
    from collections import defaultdict
    if arm_defs is None:
        arm_defs = L2B_ARM_DEFS

    by_seed_arm = defaultdict(list)
    for r in rows:
        by_seed_arm[(r["seed"], r["arm"])].append(r)

    seeds = sorted(set(r["seed"] for r in rows))
    summaries = []
    for seed in seeds:
        d = {"seed": seed}
        # Regime counts (derived from first arm's rows)
        first_arm_rows = by_seed_arm.get((seed, L2B_ARM_DEFS[0]["arm"]), [])
        n_p = sum(1 for r in first_arm_rows if r.get("target_regime") == "PERSISTENT_REGION")
        n_s = sum(1 for r in first_arm_rows if r.get("target_regime") == "STATIONARY_REGION")
        d["n_persistent_targets"] = n_p
        d["n_stationary_targets"] = n_s

        for ad in arm_defs:
            arm = ad["arm"]
            arm_rows = by_seed_arm[(seed, arm)]
            if not arm_rows:
                for sfx in ("response_global", "epoch_global", "response_P", "epoch_P",
                            "response_S", "epoch_S"):
                    d[f"{arm}_{sfx}"] = float("nan")
                continue

            p_rows = [r for r in arm_rows if r.get("target_regime") == "PERSISTENT_REGION"]
            s_rows = [r for r in arm_rows if r.get("target_regime") == "STATIONARY_REGION"]

            d[f"{arm}_response_global"] = float(np.mean([r["igd_response"]  for r in arm_rows]))
            d[f"{arm}_epoch_global"]    = float(np.mean([r["igd_epoch_end"] for r in arm_rows]))
            d[f"{arm}_response_P"] = float(np.mean([r["igd_response"]  for r in p_rows])) if p_rows else float("nan")
            d[f"{arm}_epoch_P"]    = float(np.mean([r["igd_epoch_end"] for r in p_rows])) if p_rows else float("nan")
            d[f"{arm}_response_S"] = float(np.mean([r["igd_response"]  for r in s_rows])) if s_rows else float("nan")
            d[f"{arm}_epoch_S"]    = float(np.mean([r["igd_epoch_end"] for r in s_rows])) if s_rows else float("nan")
            d[f"n_predict_{arm}"] = sum(1 for r in arm_rows if r.get("operator") == "PREDICT")
            d[f"n_local_{arm}"]   = sum(1 for r in arm_rows if r.get("operator") == "LOCAL_FALLBACK")
            d[f"n_keep_{arm}"]    = sum(1 for r in arm_rows if r.get("operator") == "COMMON_KEEP")

        summaries.append(d)
    return summaries


# ============================================================
# Effects computation
# ============================================================

def compute_l2b_effects(summaries, arm_defs=None):
    """Compute per-seed effects (global + regime-conditioned + selectivity).

    Returns list of per-seed effect dicts.
    """
    if arm_defs is None:
        arm_defs = L2B_ARM_DEFS

    effects = []
    for s in summaries:
        def g(arm, sfx):
            return s.get(f"{arm}_{sfx}", float("nan"))

        def eff(arm1, arm2, sfx):
            a = g(arm1, sfx)
            b = g(arm2, sfx)
            if np.isnan(a) or np.isnan(b):
                return float("nan")
            return float(a - b)

        row = {"seed": s["seed"],
               "n_persistent_targets": s.get("n_persistent_targets"),
               "n_stationary_targets": s.get("n_stationary_targets")}

        for sfx in ("global", "P", "S"):
            row[f"NB_resp_{sfx}"]  = eff("U03",  "U005", f"response_{sfx}")
            row[f"NB_epoch_{sfx}"] = eff("U03",  "U005", f"epoch_{sfx}")
            row[f"HB_resp_{sfx}"]  = eff("U005", "H005", f"response_{sfx}")
            row[f"HB_epoch_{sfx}"] = eff("U005", "H005", f"epoch_{sfx}")
            row[f"CB_resp_{sfx}"]  = eff("U03",  "H005", f"response_{sfx}")
            row[f"CB_epoch_{sfx}"] = eff("U03",  "H005", f"epoch_{sfx}")

        # Selectivity contrasts (P - S)
        for prefix in ("NB_resp", "NB_epoch", "HB_resp", "HB_epoch", "CB_resp", "CB_epoch"):
            p_val = row.get(f"{prefix}_P", float("nan"))
            s_val = row.get(f"{prefix}_S", float("nan"))
            if np.isnan(p_val) or np.isnan(s_val):
                row[f"{prefix}_selectivity"] = float("nan")
            else:
                row[f"{prefix}_selectivity"] = float(p_val - s_val)

        effects.append(row)
    return effects


# ============================================================
# Bootstrap statistics
# ============================================================

GLOBAL_EFFECT_KEYS = [
    "NB_resp_global",  "NB_epoch_global",
    "HB_resp_global",  "HB_epoch_global",
    "CB_resp_global",  "CB_epoch_global",
]
P_EFFECT_KEYS = [
    "NB_resp_P",  "NB_epoch_P",
    "HB_resp_P",  "HB_epoch_P",
    "CB_resp_P",  "CB_epoch_P",
]
S_EFFECT_KEYS = [
    "NB_resp_S",  "NB_epoch_S",
    "HB_resp_S",  "HB_epoch_S",
    "CB_resp_S",  "CB_epoch_S",
]
SELECTIVITY_KEYS = [
    "NB_resp_selectivity",  "NB_epoch_selectivity",
    "HB_resp_selectivity",  "HB_epoch_selectivity",
    "CB_resp_selectivity",  "CB_epoch_selectivity",
]
ALL_EFFECT_KEYS = GLOBAL_EFFECT_KEYS + P_EFFECT_KEYS + S_EFFECT_KEYS + SELECTIVITY_KEYS


def compute_l2b_stats(effects, effect_keys=None):
    """Bootstrap statistics for all effects (unit = seed).

    Returns dict of effect_key -> {mean, median, ci_lo, ci_hi, classification}.
    """
    if effect_keys is None:
        effect_keys = ALL_EFFECT_KEYS
    stats = {}
    for key in effect_keys:
        vals = [e[key] for e in effects if not np.isnan(e.get(key, float("nan")))]
        if not vals:
            stats[key] = {"mean": float("nan"), "ci_lo": float("nan"),
                          "ci_hi": float("nan"), "classification": "UNKNOWN", "n_seeds": 0}
            continue
        mean_v, ci_lo, ci_hi = bootstrap_contrast_ci(vals)
        stats[key] = {
            "mean": mean_v,
            "median": float(np.median(vals)),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_seeds": len(vals),
            "classification": classify_effect(mean_v, ci_lo, ci_hi),
        }
    return stats


# ============================================================
# Verdict logic
# ============================================================

def decide_l2b_verdicts(stats, delta=DELTA_BOUNDARY):
    """Apply L2-B pre-declared verdict logic.

    Returns dict with all 4 verdicts + supporting criterion flags.
    """
    def mean(key):
        return stats.get(key, {}).get("mean", float("nan"))

    def ci_lo(key):
        return stats.get(key, {}).get("ci_lo", float("nan"))

    def ci_hi(key):
        return stats.get(key, {}).get("ci_hi", float("nan"))

    def cls(key):
        return stats.get(key, {}).get("classification", "UNKNOWN")

    # ---- F005_MIXED_BOUNDARY ----
    # F1: NB_resp_P > +delta
    f1 = mean("NB_resp_P") > delta
    # F2: NB_epoch_P >= -delta
    f2 = mean("NB_epoch_P") >= -delta
    # F3: NB_resp_S >= -delta
    f3 = mean("NB_resp_S") >= -delta
    # F4: NB_epoch_S >= -delta
    f4 = mean("NB_epoch_S") >= -delta
    # F5: NB_selectivity_response > +delta
    f5 = mean("NB_resp_selectivity") > delta
    # F6: no STATIONARY stationary endpoint HARMFUL
    f6 = (cls("NB_resp_S") != "HARMFUL" and cls("NB_epoch_S") != "HARMFUL")

    if f1 and f2 and f3 and f4 and f5 and f6:
        f005_verdict = "SUPPORTED"
    elif (f3 and f4 and f6) and (f1 or mean("NB_resp_global") > delta):
        f005_verdict = "PARTIAL"
    elif (cls("NB_resp_S") == "HARMFUL" or cls("NB_epoch_S") == "HARMFUL"
          or (mean("NB_resp_global") < -delta and mean("NB_epoch_global") < -delta)):
        f005_verdict = "NOT_SUPPORTED"
    else:
        f005_verdict = "PARTIAL"

    # ---- HARD_MIXED_BOUNDARY ----
    # H1: HB_resp_P > +delta
    h1 = mean("HB_resp_P") > delta
    # H2: HB_resp_S >= -delta
    h2 = mean("HB_resp_S") >= -delta
    # H3: HB_selectivity_response > +delta
    h3 = mean("HB_resp_selectivity") > delta
    # H4: HB_epoch_global >= -delta
    h4 = mean("HB_epoch_global") >= -delta
    # H5: no stationary-target HARMFUL response
    h5 = cls("HB_resp_S") != "HARMFUL"
    # H6: HARD invariant (must be confirmed externally — set by caller)

    if h1 and h2 and h3 and h4 and h5:
        hard_verdict = "SUPPORTED"
    elif h2 and h5 and (h1 or mean("HB_resp_global") > delta):
        hard_verdict = "PARTIAL"
    elif (cls("HB_resp_P") == "HARMFUL" or cls("HB_resp_S") == "HARMFUL"
          or mean("HB_epoch_global") < -delta):
        hard_verdict = "NOT_SUPPORTED"
    else:
        hard_verdict = "PARTIAL"

    # ---- H005_MIXED_GLOBAL ----
    # G1: CB_resp_global > +delta
    g1 = mean("CB_resp_global") > delta
    # G2: bootstrap CI lower for CB_resp_global > 0
    g2 = ci_lo("CB_resp_global") > 0
    # G3: CB_epoch_global >= -delta
    g3 = mean("CB_epoch_global") >= -delta
    # G4: CB_resp_P > +delta
    g4 = mean("CB_resp_P") > delta
    # G5: CB_resp_S >= -delta
    g5 = mean("CB_resp_S") >= -delta
    # G6: FE fairness (set by caller)
    # G7: HARD invariant (set by caller)

    if g1 and g2 and g3 and g4 and g5:
        global_verdict = "VALIDATED_PENDING_FE_HARD"
    elif g1 and g3 and g5:
        global_verdict = "PARTIAL"
    elif (not g5 and cls("CB_resp_S") == "HARMFUL") or mean("CB_resp_global") < -delta:
        global_verdict = "NOT_VALIDATED"
    else:
        global_verdict = "PARTIAL"

    # ---- REGIME_BOUNDARY_HYPOTHESIS ----
    # B1: CB_resp_P > +delta
    b1 = mean("CB_resp_P") > delta
    # B2: CB_resp_S >= -delta
    b2 = mean("CB_resp_S") >= -delta
    # B3: CB_selectivity_response > +delta
    b3 = mean("CB_resp_selectivity") > delta
    # B4: bootstrap CI lower for CB_resp_selectivity > 0
    b4 = ci_lo("CB_resp_selectivity") > 0
    # B5: no CI-supported worsening of opposite-regime PF coverage (set by caller)

    if b1 and b2 and b3 and b4:
        boundary_verdict = "SUPPORTED_PENDING_B5"
    elif b1 and b2 and (b3 or mean("CB_resp_selectivity") > 0):
        boundary_verdict = "PARTIAL"
    elif (cls("CB_resp_P") == "HARMFUL" or cls("CB_resp_S") == "HARMFUL"
          or mean("CB_resp_global") < -delta):
        boundary_verdict = "NOT_SUPPORTED"
    else:
        boundary_verdict = "PARTIAL"

    return {
        "F005_MIXED_BOUNDARY":          f005_verdict,
        "HARD_MIXED_BOUNDARY":          hard_verdict,
        "H005_MIXED_GLOBAL":            global_verdict,
        "REGIME_BOUNDARY_HYPOTHESIS":   boundary_verdict,
        "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6,
        "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5,
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5,
        "B1": b1, "B2": b2, "B3": b3, "B4": b4,
        "mean_CB_resp_global":      mean("CB_resp_global"),
        "ci_lo_CB_resp_global":     ci_lo("CB_resp_global"),
        "mean_CB_resp_P":           mean("CB_resp_P"),
        "mean_CB_resp_S":           mean("CB_resp_S"),
        "mean_CB_resp_selectivity": mean("CB_resp_selectivity"),
        "ci_lo_CB_resp_selectivity": ci_lo("CB_resp_selectivity"),
    }


# ============================================================
# D1 Readiness
# ============================================================

def decide_d1_readiness(verdicts, fe_ok, hard_ok):
    """Determine D1 readiness based on finalized verdicts.

    Returns 'READY_H005_TRANSIENT' / 'READY_F005_ONLY' / 'NEEDS_MORE_DIAGNOSIS'.
    """
    # Resolve pending verdicts
    h005_global = verdicts.get("H005_MIXED_GLOBAL", "")
    if h005_global == "VALIDATED_PENDING_FE_HARD":
        h005_global = "VALIDATED" if (fe_ok and hard_ok) else "PARTIAL"

    boundary = verdicts.get("REGIME_BOUNDARY_HYPOTHESIS", "")
    if boundary == "SUPPORTED_PENDING_B5":
        # B5: PF cross-regime damage not CI-supported → assume passes unless explicitly failed
        boundary = "SUPPORTED"

    hard_boundary = verdicts.get("HARD_MIXED_BOUNDARY", "")
    f005_boundary = verdicts.get("F005_MIXED_BOUNDARY", "")

    # READY_H005_TRANSIENT
    if (h005_global == "VALIDATED"
            and hard_boundary != "NOT_SUPPORTED"
            and boundary != "NOT_SUPPORTED"):
        return "READY_H005_TRANSIENT"

    # READY_F005_ONLY
    if (f005_boundary == "SUPPORTED"
            and (hard_boundary == "NOT_SUPPORTED" or h005_global == "NOT_VALIDATED")):
        return "READY_F005_ONLY"

    return "NEEDS_MORE_DIAGNOSIS"


# ============================================================
# Full-run helpers
# ============================================================

def run_l2b_seeds(seeds, n_scored, mixed_map, arm_defs=None, verbose=False):
    """Run L2-B for all seeds. Returns flat list of all rows."""
    all_rows = []
    for seed in seeds:
        t0 = time.time()
        rows = run_one_l2b(seed, n_scored, mixed_map, arm_defs=arm_defs)
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
# __main__
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L2-B Mixed-Regime Boundary Validation")
    parser.add_argument("--smoke", action="store_true", help="Smoke run with seeds 107,108 and 3 changes")
    args = parser.parse_args()

    OUT_DIR = Path("results/L2B_mixed_regime_boundary_validation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds    = SMOKE_SEEDS
        n_scored = 3
        print(f"[L2-B SMOKE] seeds={seeds}, n_scored={n_scored}")
    else:
        seeds    = L2B_SEEDS
        n_scored = N_SCORED_CHANGES
        print(f"[L2-B FULL]  seeds={seeds}, n_scored={n_scored}")

    # ---- Verify seed freshness ----
    scientific_prior = set(range(1, 81))   # 1..80 from L1/L2 and prior experiments
    smoke_prior      = set(range(101, 109)) # 101..108 (L1/L2 smoke + L2B smoke)
    overlap_sci  = set(seeds) & scientific_prior
    overlap_smk  = set(seeds) & smoke_prior if not args.smoke else set()
    if overlap_sci:
        raise RuntimeError(f"SEED CONTAMINATION: {sorted(overlap_sci)} overlap with scientific seeds 1..80. STOP.")
    if overlap_smk:
        raise RuntimeError(f"SEED CONTAMINATION: {sorted(overlap_smk)} overlap with smoke seeds. STOP.")
    print(f"  Seed freshness: OK (no overlap with 1..80 or 101..108)")

    # ---- Build MIXED map from authoritative function ----
    V  = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)
    mixed_map = make_l2b_mixed_dynamics(R)
    n_pers = sum(1 for x in mixed_map if x == "persistent")
    n_stat = sum(1 for x in mixed_map if x == "stationary")
    print(f"  MIXED_MAP = {mixed_map}")
    print(f"  Persistent regions: {n_pers}, Stationary regions: {n_stat}")

    if n_pers == 0 or n_stat == 0:
        raise RuntimeError("MINIMUM REQUIREMENT FAILED: MIXED map must have both persistent and stationary regions.")

    # ---- Full run ----
    t_run = time.time()
    all_rows = run_l2b_seeds(seeds, n_scored, mixed_map, verbose=True)
    print(f"\n  Total rows: {len(all_rows)}  ({time.time()-t_run:.1f}s)")

    if not args.smoke:
        # Save trajectories
        save_rows(all_rows, OUT_DIR / "L2B_mixed_trajectories.jsonl")
        print(f"  Saved L2B_mixed_trajectories.jsonl ({len(all_rows)} rows)")

    # ---- FE audit ----
    fe_rows, fe_equal = compute_l2b_fe_audit(all_rows)
    fe_status = "CONFIRMED" if fe_equal else "FAILED"
    print(f"\n  FE_FAIRNESS = {fe_status}")
    if not args.smoke:
        save_csv(fe_rows, OUT_DIR / "L2B_fe_audit.csv")

    if not fe_equal:
        print("  WARNING: FE FAIRNESS FAILED — do not interpret results.")

    # ---- HARD structural invariant ----
    hard_ok, n_viol, n_viol_p, n_viol_s = check_hard_invariant(all_rows)
    hard_status = "CONFIRMED" if hard_ok else "FAILED"
    print(f"  HARD_INVARIANT = {hard_status} (violations: total={n_viol}, pers={n_viol_p}, stat={n_viol_s})")

    if not hard_ok:
        print("  WARNING: HARD INVARIANT VIOLATED — diagnose before interpreting results.")

    # ---- Operator distribution ----
    for arm_name in L2B_PRIMARY_ARMS:
        arm_rows = [r for r in all_rows if r["arm"] == arm_name]
        n_pred = sum(1 for r in arm_rows if r.get("operator") == "PREDICT")
        n_loc  = sum(1 for r in arm_rows if r.get("operator") == "LOCAL_FALLBACK")
        n_keep = sum(1 for r in arm_rows if r.get("operator") == "COMMON_KEEP")
        print(f"  {arm_name}: PREDICT={n_pred}, LOCAL_FALLBACK={n_loc}, COMMON_KEEP={n_keep}")

    # ---- Seed summaries ----
    summaries = compute_l2b_seed_summaries(all_rows)
    if not args.smoke:
        save_csv(summaries, OUT_DIR / "L2B_seed_summary.csv")

    # Verify regime counts
    if summaries:
        n_p = summaries[0].get("n_persistent_targets", "?")
        n_s = summaries[0].get("n_stationary_targets", "?")
        print(f"\n  TARGET_COUNTS_PER_SEED = PERSISTENT: {n_p}, STATIONARY: {n_s}")

    # ---- Effects ----
    effects = compute_l2b_effects(summaries)
    if not args.smoke:
        save_csv(effects, OUT_DIR / "L2B_regime_effects.csv")

    # ---- Statistics ----
    stats = compute_l2b_stats(effects)

    # Print global results
    print("\n  --- Global arm results ---")
    arm_global = {}
    for arm_name in L2B_PRIMARY_ARMS:
        sums_arm = [(s[f"{arm_name}_response_global"], s[f"{arm_name}_epoch_global"]) for s in summaries]
        mean_r = float(np.mean([x[0] for x in sums_arm]))
        mean_e = float(np.mean([x[1] for x in sums_arm]))
        arm_global[arm_name] = (mean_r, mean_e)
        print(f"  {arm_name}: MIGD_response={mean_r:.6f}, MIGD_epoch={mean_e:.6f}")

    print("\n  --- Global effects ---")
    for key in GLOBAL_EFFECT_KEYS:
        s = stats.get(key, {})
        print(f"  {key}: mean={s.get('mean', float('nan')):.3e}  CI=[{s.get('ci_lo', float('nan')):.3e}, {s.get('ci_hi', float('nan')):.3e}]  {s.get('classification','?')}")

    print("\n  --- P-target effects ---")
    for key in P_EFFECT_KEYS:
        s = stats.get(key, {})
        print(f"  {key}: mean={s.get('mean', float('nan')):.3e}  CI=[{s.get('ci_lo', float('nan')):.3e}, {s.get('ci_hi', float('nan')):.3e}]  {s.get('classification','?')}")

    print("\n  --- S-target effects ---")
    for key in S_EFFECT_KEYS:
        s = stats.get(key, {})
        print(f"  {key}: mean={s.get('mean', float('nan')):.3e}  CI=[{s.get('ci_lo', float('nan')):.3e}, {s.get('ci_hi', float('nan')):.3e}]  {s.get('classification','?')}")

    print("\n  --- Selectivity contrasts (P - S) ---")
    for key in SELECTIVITY_KEYS:
        s = stats.get(key, {})
        print(f"  {key}: mean={s.get('mean', float('nan')):.3e}  CI=[{s.get('ci_lo', float('nan')):.3e}, {s.get('ci_hi', float('nan')):.3e}]  {s.get('classification','?')}")

    # ---- Verdicts ----
    verdicts = decide_l2b_verdicts(stats)
    d1_ready = decide_d1_readiness(verdicts, fe_ok=fe_equal, hard_ok=hard_ok)

    # Finalize pending verdicts
    h005_global_final = verdicts["H005_MIXED_GLOBAL"]
    if h005_global_final == "VALIDATED_PENDING_FE_HARD":
        h005_global_final = "VALIDATED" if (fe_equal and hard_ok) else "PARTIAL"

    boundary_final = verdicts["REGIME_BOUNDARY_HYPOTHESIS"]
    if boundary_final == "SUPPORTED_PENDING_B5":
        boundary_final = "SUPPORTED"

    print("\n" + "=" * 60)
    print(f"  MIXED_MAP = {mixed_map}")
    print(f"\n  TARGET_COUNTS_PER_SEED =")
    print(f"    PERSISTENT: {n_p}")
    print(f"    STATIONARY: {n_s}")
    print(f"\n  FE_FAIRNESS = {fe_status}")
    print(f"\n  HARD_INVARIANT = {hard_status}")
    print(f"\n  F005_MIXED_BOUNDARY = {verdicts['F005_MIXED_BOUNDARY']}")
    print(f"\n  HARD_MIXED_BOUNDARY = {verdicts['HARD_MIXED_BOUNDARY']}")
    print(f"\n  H005_MIXED_GLOBAL = {h005_global_final}")
    print(f"\n  REGIME_BOUNDARY_HYPOTHESIS = {boundary_final}")
    print(f"\n  D1_READINESS = {d1_ready}")
    if d1_ready == "READY_H005_TRANSIENT":
        next_stage = "D1_DESIGN_RECONCILIATION"
    else:
        next_stage = "MORE_DIAGNOSIS"
    print(f"\n  NEXT_STAGE = {next_stage}")
    print("=" * 60)

    # Save verdicts
    if not args.smoke:
        verdicts_out = {
            "verdicts": {
                "FE_FAIRNESS": fe_status,
                "HARD_INVARIANT": hard_status,
                "F005_MIXED_BOUNDARY": verdicts["F005_MIXED_BOUNDARY"],
                "HARD_MIXED_BOUNDARY": verdicts["HARD_MIXED_BOUNDARY"],
                "H005_MIXED_GLOBAL": h005_global_final,
                "REGIME_BOUNDARY_HYPOTHESIS": boundary_final,
                "D1_READINESS": d1_ready,
                "NEXT_STAGE": next_stage,
            },
            "criterion_flags": {k: v for k, v in verdicts.items() if isinstance(v, bool)},
            "global_effects": {k: stats[k] for k in GLOBAL_EFFECT_KEYS if k in stats},
            "P_effects": {k: stats[k] for k in P_EFFECT_KEYS if k in stats},
            "S_effects": {k: stats[k] for k in S_EFFECT_KEYS if k in stats},
            "selectivity_effects": {k: stats[k] for k in SELECTIVITY_KEYS if k in stats},
            "mixed_map": mixed_map,
        }
        with open(OUT_DIR / "L2B_verdicts.json", "w") as f:
            json.dump(verdicts_out, f, indent=2, default=str)
        print(f"\n  Saved L2B_verdicts.json")
