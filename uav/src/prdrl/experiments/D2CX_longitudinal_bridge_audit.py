"""
D2-CX — Longitudinal Path Bridge Audit.

Scientific question:
    Why does the frozen L2-B MIXED result (CB≈+2.264e-4, seeds 81-100, DIAG path)
    contradict the fresh D2-C MIXED result (CB≈-1.301e-4, seeds 109-128, PROD path)?

Four branches from a shared initial state per seed:
    DIAG_U03:  L2B CRN children + unrestricted survival, sigma=0.03
    PROD_O0:   oracle respond_population, UNRESTRICTED, sigma=0.03
    DIAG_H005: L2B CRN children + HARD survival, sigma=0.005
    PROD_O3:   oracle respond_population, HARD_RESPONSE_ONLY, sigma=0.005

Checkpoint levels:
    C0: pre-change (X fingerprint before evaluating F_minus)
    C1: post-env/pre-response (F_plus + region_ids fingerprint)
    C2: post-child-generation (X_children fingerprint — DIAG only; PROD internal)
    C3: post-response (X_surv fingerprint)
    C4: post-evolution (X fingerprint)

Block A: seeds 81-100 (frozen L2-B seeds)
Block B: seeds 109-128 (D2-C holdout seeds)
Both: MIXED scenario, 10 scored changes.

Verdict classification:
    SEED_BLOCK_HETEROGENEITY:       DIAG_U03 (Block B) gives negative CB too
    PRODUCTION_PROTOCOL_MISMATCH:   DIAG_U03 positive on both blocks, PROD negative
    RNG_COUPLING_MISMATCH:          C3 diverges only due to C2 RNG difference
    HISTORY_UPDATE_MISMATCH:        C5 (history state) differs unexpectedly
    STANDARD_EVOLUTION_MISMATCH:    C4 diverges when C3 agrees
    RESPONSE_PATH_MISMATCH:         C3 diverges with same C2 (within same arm type)
    UNRESOLVED:                     Inconclusive evidence

CONSTRAINTS:
    - Production default O0 (sigma=0.03, UNRESTRICTED) must NOT change.
    - No RL. No frozen results overwritten. No new scenarios invented.
    - HARD structural invariant: n_old_dropped_other == 0; STOP if violated.
    - FE fairness: all arms must have identical extra_fe per change; STOP if violated.
    - Seeds 81-100 are diagnostic-only (no retroactive change to L2-B verdict).
    - Seeds 109-128 are diagnostic-only (no retroactive change to D2-C verdict).
    - Pooled 40-seed results are post-hoc descriptive only.
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.core.survival import region_protected_survival
from prdrl.features.forecast_state import get_predict_motion_from_history
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import (
    PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    PREDICT_SURVIVAL_UNRESTRICTED,
    respond_population,
)
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)

# ============================================================
# Config — locked to production values
# ============================================================
LAMBDA_PRED = 1.0
SIGMA_L = 0.02

BUDGET = e2p3.BUDGET                    # 10
POP_SIZE = e2p3.POP_SIZE               # 100
D = e2p3.D                             # 10
N_OBJ = e2p3.N_OBJ                    # 2
N_PARTITIONS = e2p3.N_PARTITIONS       # 9 → R=10
WARMUP_GENERATIONS = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

compute_time = e2p3.compute_time
evolve_nsga2 = e2p3.evolve_nsga2
make_rng = e2p3.make_rng
nsga2_survival_traced = e2p3.nsga2_survival_traced
_e2p3_common_surv_seed = e2p3._e2p3_common_surv_seed
apply_samples = e2p3.apply_samples

N_SCORED_CHANGES = 10
BLOCK_A_SEEDS = list(range(81, 101))    # 20 seeds — frozen L2-B
BLOCK_B_SEEDS = list(range(109, 129))   # 20 seeds — D2-C holdout

# Four-arm definitions
ARM_DEFS = [
    {
        "arm": "DIAG_U03",
        "sigma": 0.03,
        "use_diag_crn": True,
        "use_hard": False,
        "mode": PREDICT_SURVIVAL_UNRESTRICTED,
    },
    {
        "arm": "PROD_O0",
        "sigma": 0.03,
        "use_diag_crn": False,
        "use_hard": False,
        "mode": PREDICT_SURVIVAL_UNRESTRICTED,
    },
    {
        "arm": "DIAG_H005",
        "sigma": 0.005,
        "use_diag_crn": True,
        "use_hard": True,
        "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    },
    {
        "arm": "PROD_O3",
        "sigma": 0.005,
        "use_diag_crn": False,
        "use_hard": True,
        "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    },
]


# ============================================================
# MIXED map helpers (authoritative existing function)
# ============================================================

def make_d2cx_mixed_dynamics(R):
    """Return MIXED dynamics map: regions 0..R//2-1 persistent, R//2..R-1 stationary."""
    return make_region_dynamics_map(R)


def get_target_regime(tgt_r, mixed_map):
    label = mixed_map[tgt_r]
    return "PERSISTENT_REGION" if label == "persistent" else "STATIONARY_REGION"


def target_region(seed, k, R):
    return (seed + k) % R


def make_pf_zones(pf_star, mixed_map):
    R = len(mixed_map)
    theta = np.arccos(np.clip(pf_star[:, 0], 0.0, 1.0))
    x1 = 2.0 * theta / np.pi
    bins = np.minimum(np.floor(R * x1).astype(int), R - 1)
    return np.array([mixed_map[b] == "persistent" for b in bins])


# ============================================================
# Fingerprint helpers
# ============================================================

def _fp(arr, label=""):
    """Compact fingerprint dict for an ndarray."""
    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
        return None
    a = np.asarray(arr, dtype=float)
    raw = a.tobytes()
    return {
        "label": label,
        "shape": list(a.shape),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "md5": hashlib.md5(raw).hexdigest()[:12],
    }


def _fp_pair(F_plus, region_ids):
    """C1 fingerprint: F_plus + region_ids combined."""
    return {
        "F_plus": _fp(F_plus, "F_plus"),
        "region_ids": _fp(region_ids.astype(float), "region_ids"),
    }


def _fps_diverge(fp_a, fp_b):
    """Return True if two fingerprint dicts have different md5 (or different shape)."""
    if fp_a is None and fp_b is None:
        return False
    if fp_a is None or fp_b is None:
        return True
    return fp_a.get("md5") != fp_b.get("md5")


# ============================================================
# HARD invariant check for PROD arms
# (respond_population doesn't return indices, so check by row matching)
# ============================================================

def _count_old_dropped_other(X_old, region_ids, tgt_r, X_surv):
    """Count old non-target rows absent from X_surv (O(N^2) but N=100)."""
    old_other = X_old[region_ids != tgt_r]
    dropped = 0
    for row in old_other:
        found = any(np.allclose(row, s, atol=1e-12) for s in X_surv)
        if not found:
            dropped += 1
    return dropped


# ============================================================
# Main per-seed runner
# ============================================================

def run_one_d2cx(seed, n_scored, mixed_map, pf_star, is_persistent_zone, block_label):
    """Run D2-CX 4-arm bridge for one seed on MIXED scenario.

    Returns list of row dicts, one per (arm, scored_change_k).
    """
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    region_dynamics = mixed_map

    # ── Shared warmup ──────────────────────────────────────────────────────
    rng_init = np.random.default_rng(seed)
    X_pop = rng_init.uniform(xl, xu, size=(POP_SIZE, D))
    t_float_0 = compute_time(0)
    X_pop = evolve_nsga2(
        X_pop, t_float_0, region_dynamics, WARMUP_GENERATIONS, seed * 10000, xl, xu
    )

    # ── Shared burn-in change (change_idx=1, not scored) ───────────────────
    history_shared = RegionHistory(n_regions=R)
    history_t_shared = -1
    gen = WARMUP_GENERATIONS

    F_minus_bi = cp_evaluate(X_pop, t_float_0, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t_float_1 = compute_time(gen)
    F_plus_bi = cp_evaluate(X_pop, t_float_1, region_dynamics)
    F_m_bi, _ = normalize(F_minus_bi, F_plus_bi)
    rids_bi = np.asarray(partition(np.asarray(F_m_bi), V), dtype=int)
    history_t_shared += 1   # → 0
    history_shared.update(
        t=history_t_shared, X=X_pop, F=F_minus_bi,
        region_ids=rids_bi, xl=xl, xu=xu,
    )
    X_pop = evolve_nsga2(
        X_pop, t_float_1, region_dynamics,
        GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu,
    )
    t_float_current = t_float_1

    # ── Clone into 4 independent arm states ───────────────────────────────
    arm_states = {}
    for ad in ARM_DEFS:
        arm_states[ad["arm"]] = {
            "X":           X_pop.copy(),
            "history":     copy.deepcopy(history_shared),
            "history_t":   history_t_shared,
            "sigma":       ad["sigma"],
            "use_diag_crn": ad["use_diag_crn"],
            "use_hard":    ad["use_hard"],
            "mode":        ad["mode"],
        }

    rows_out = []

    # ── Scored changes ─────────────────────────────────────────────────────
    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r = target_region(seed, k, R)
        tgt_regime = get_target_regime(tgt_r, mixed_map)

        gen += GENERATIONS_PER_CHANGE
        t_float_new = compute_time(gen)
        evo_seed = seed * 10000 + scored_change_idx
        surv_seed_diag = _e2p3_common_surv_seed(seed, scored_change_idx, tgt_r)

        # ── C0: pre-change fingerprints ──────────────────────────────────
        c0_fps = {name: _fp(arm["X"], "C0_X") for name, arm in arm_states.items()}

        # ── Per-arm: evaluate at t_new, partition, update history ─────────
        arm_eval = {}
        for arm_name, arm in arm_states.items():
            F_minus = cp_evaluate(arm["X"], t_float_current, region_dynamics)
            F_plus  = cp_evaluate(arm["X"], t_float_new,     region_dynamics)
            F_m, _  = normalize(F_minus, F_plus)
            region_ids = np.asarray(partition(np.asarray(F_m), V), dtype=int)
            n_r = int(np.sum(region_ids == tgt_r))
            igd_pre = float(igd(F_plus, pf_star))

            arm["history_t"] += 1
            arm["history"].update(
                t=arm["history_t"], X=arm["X"], F=F_minus,
                region_ids=region_ids, xl=xl, xu=xu,
            )
            arm_eval[arm_name] = {
                "F_plus":     F_plus,
                "region_ids": region_ids,
                "n_r":        n_r,
                "igd_pre":    igd_pre,
            }

        # ── C1: post-env fingerprints ─────────────────────────────────────
        c1_fps = {
            name: _fp_pair(arm_eval[name]["F_plus"], arm_eval[name]["region_ids"])
            for name in arm_states
        }

        # ── COMMON_KEEP check ─────────────────────────────────────────────
        any_empty = any(arm_eval[a]["n_r"] == 0 for a in arm_eval)

        if any_empty:
            for arm_name, arm in arm_states.items():
                ev = arm_eval[arm_name]
                arm["X"] = evolve_nsga2(
                    arm["X"], t_float_new, region_dynamics,
                    GENERATIONS_PER_CHANGE, evo_seed, xl, xu,
                )
                F_epoch_end = cp_evaluate(arm["X"], t_float_new, region_dynamics)
                rows_out.append({
                    "seed": seed, "block": block_label,
                    "arm": arm_name, "change_k": k,
                    "scored_change_idx": scored_change_idx,
                    "tgt_r": tgt_r, "tgt_regime": tgt_regime,
                    "sigma": arm["sigma"],
                    "use_diag_crn": arm["use_diag_crn"],
                    "use_hard": arm["use_hard"],
                    "operator": "COMMON_KEEP", "realized_op": "KEEP",
                    "n_r_before": ev["n_r"], "extra_fe": 0,
                    "igd_pre": ev["igd_pre"],
                    "igd_response": ev["igd_pre"],
                    "igd_epoch_end": float(igd(F_epoch_end, pf_star)),
                    "n_old_dropped_other": None,
                    "hard_invariant_ok": None,
                    "c0_fp": c0_fps[arm_name],
                    "c1_F_fp": c1_fps[arm_name]["F_plus"],
                    "c1_rids_fp": c1_fps[arm_name]["region_ids"],
                    "c2_fp": None,
                    "c2_rng_scheme": None,
                    "c3_fp": None,
                    "c4_fp": None,
                    "surv_seed_diag": None,
                })
            t_float_current = t_float_new
            continue

        # ── CRN base samples for DIAG arms (shared u_parent + xi) ─────────
        rng_crn = make_rng(seed, scored_change_idx, tgt_r, "L2B_RESPONSE")
        u_parent = rng_crn.uniform(size=BUDGET)
        xi_mat   = rng_crn.standard_normal(size=(BUDGET, D))
        xis_crn  = list(xi_mat)

        # ── Per-arm: response ─────────────────────────────────────────────
        for arm_name, arm in arm_states.items():
            ev         = arm_eval[arm_name]
            F_plus     = ev["F_plus"]
            region_ids = ev["region_ids"]
            n_r        = ev["n_r"]
            igd_pre    = ev["igd_pre"]
            X_old      = arm["X"].copy()
            X_region   = X_old[region_ids == tgt_r]

            if arm["use_diag_crn"]:
                # ── DIAG path: replicate L2B CRN child generation ────────
                parent_idxs = [min(int(u * n_r), n_r - 1) for u in u_parent]

                motion = get_predict_motion_from_history(
                    arm["history"], tgt_r,
                    t=arm["history_t"] + 1,
                    lambda_pred=LAMBDA_PRED,
                )
                if motion is None:
                    realized_op  = "LOCAL_FALLBACK"
                    delta_hat    = np.zeros(D)
                    sigma_used   = SIGMA_L
                else:
                    realized_op  = "PREDICT"
                    delta_hat    = motion["delta_hat"]
                    sigma_used   = arm["sigma"]

                X_children = apply_samples(
                    parent_idxs, xis_crn, X_region, delta_hat, sigma_used, xl, xu
                )
                F_children = cp_evaluate(X_children, t_float_new, region_dynamics)
                extra_fe   = len(X_children)

                c2_fp = _fp(X_children, "C2_children")
                c2_rng_scheme = f"L2B_CRN(seed={seed},sc={scored_change_idx},r={tgt_r})"

                if arm["use_hard"] and realized_op == "PREDICT":
                    X_s, F_s, idx_s = region_protected_survival(
                        X_old, F_plus, X_children, F_children,
                        tgt_r, region_ids, n_r, xl, xu, surv_seed_diag,
                    )
                    N_old = len(X_old)
                    surv_old_set = set(int(i) for i in idx_s if i < N_old)
                    old_other_ids = [i for i in range(N_old) if region_ids[i] != tgt_r]
                    n_dropped = sum(1 for i in old_other_ids if i not in surv_old_set)
                elif arm["use_hard"] and realized_op == "LOCAL_FALLBACK":
                    # HARD fallback rule: key on realized_op, not requested action
                    X_all = np.vstack([X_old, X_children])
                    F_all = np.vstack([F_plus, F_children])
                    X_s, F_s, idx_s = nsga2_survival_traced(
                        X_all, F_all, POP_SIZE, xl, xu, surv_seed_diag,
                    )
                    n_dropped = None  # HARD invariant N/A for LOCAL_FALLBACK
                else:
                    X_all = np.vstack([X_old, X_children])
                    F_all = np.vstack([F_plus, F_children])
                    X_s, F_s, idx_s = nsga2_survival_traced(
                        X_all, F_all, POP_SIZE, xl, xu, surv_seed_diag,
                    )
                    n_dropped = None

                arm["X"] = X_s

            else:
                # ── PROD path: oracle respond_population ─────────────────
                t_oracle = arm["history_t"] + 1
                oracle_base_seed = seed * 100000 + k * 100 + tgt_r

                result = respond_population(
                    X=X_old,
                    F_plus=F_plus,
                    region_ids=region_ids,
                    V=V,
                    actions={tgt_r: "PREDICT"},
                    history=arm["history"],
                    t=t_oracle,
                    evaluate_fn=lambda Xc: cp_evaluate(Xc, t_float_new, region_dynamics),
                    xl=xl,
                    xu=xu,
                    budget=BUDGET,
                    sigma_local=SIGMA_L,
                    sigma_predict=arm["sigma"],
                    sigma_diversify=0.15,
                    lambda_pred=LAMBDA_PRED,
                    base_seed=oracle_base_seed,
                    predict_response_survival_mode=arm["mode"],
                )

                X_s    = result["X"]
                F_s    = result["F"]
                extra_fe = result["extra_fe"]
                realized_op = "PREDICT"

                c2_fp = None  # children internal to dispatcher
                c2_rng_scheme = f"ORACLE_PREDICT(base={oracle_base_seed},t={t_oracle},r={tgt_r})"

                if arm["use_hard"]:
                    n_dropped = _count_old_dropped_other(X_old, region_ids, tgt_r, X_s)
                else:
                    n_dropped = None

                arm["X"] = X_s

            # ── C3 fingerprint (post-response) ────────────────────────────
            c3_fp = _fp(arm["X"], "C3_Xsurv")
            igd_response = float(igd(F_s, pf_star))

            # HARD invariant check — STOP if violated
            if arm["use_hard"] and realized_op == "PREDICT" and n_dropped is not None:
                if n_dropped != 0:
                    raise RuntimeError(
                        f"HARD INVARIANT VIOLATED: seed={seed} arm={arm_name} "
                        f"k={k} tgt_r={tgt_r} n_old_dropped_other={n_dropped}"
                    )
                hard_ok = True
            else:
                hard_ok = None

            # ── Evolve (C4) ───────────────────────────────────────────────
            arm["X"] = evolve_nsga2(
                arm["X"], t_float_new, region_dynamics,
                GENERATIONS_PER_CHANGE, evo_seed, xl, xu,
            )
            c4_fp = _fp(arm["X"], "C4_Xevo")
            F_epoch = cp_evaluate(arm["X"], t_float_new, region_dynamics)
            igd_epoch_end = float(igd(F_epoch, pf_star))

            rows_out.append({
                "seed":              seed,
                "block":             block_label,
                "arm":               arm_name,
                "change_k":          k,
                "scored_change_idx": scored_change_idx,
                "tgt_r":             tgt_r,
                "tgt_regime":        tgt_regime,
                "sigma":             arm["sigma"],
                "use_diag_crn":      arm["use_diag_crn"],
                "use_hard":          arm["use_hard"],
                "operator":          "PREDICT",
                "realized_op":       realized_op,
                "n_r_before":        n_r,
                "extra_fe":          extra_fe,
                "igd_pre":           igd_pre,
                "igd_response":      igd_response,
                "igd_epoch_end":     igd_epoch_end,
                "n_old_dropped_other": n_dropped,
                "hard_invariant_ok": hard_ok,
                "c0_fp":             c0_fps[arm_name],
                "c1_F_fp":           c1_fps[arm_name]["F_plus"],
                "c1_rids_fp":        c1_fps[arm_name]["region_ids"],
                "c2_fp":             c2_fp,
                "c2_rng_scheme":     c2_rng_scheme,
                "c3_fp":             c3_fp,
                "c4_fp":             c4_fp,
                "surv_seed_diag":    int(surv_seed_diag) if arm["use_diag_crn"] else "PROD_DEFAULT_NONE",
            })

        # ── FE fairness check per change_k ────────────────────────────────
        k_rows = [r for r in rows_out
                  if r["seed"] == seed and r["change_k"] == k
                  and r["operator"] != "COMMON_KEEP"]
        if len(k_rows) == len(ARM_DEFS):
            fe_vals = {r["arm"]: r["extra_fe"] for r in k_rows}
            if len(set(fe_vals.values())) > 1:
                raise RuntimeError(
                    f"FE FAIRNESS VIOLATED: seed={seed} k={k} "
                    f"fe_vals={fe_vals}"
                )

        t_float_current = t_float_new

    return rows_out


# ============================================================
# First-divergence detection
# ============================================================

_CHECKPOINT_KEYS = ["c0_md5", "c1_F_md5", "c1_rids_md5", "c2_md5", "c3_md5", "c4_md5"]
_CHECKPOINT_LABELS = ["C0", "C1_F", "C1_RIDS", "C2", "C3", "C4"]


def _extract_md5(fp_dict_or_none):
    if fp_dict_or_none is None:
        return None
    return fp_dict_or_none.get("md5")


def compute_divergence_report(all_rows):
    """For each (seed, change_k), find first checkpoint where each arm pair diverges."""
    from collections import defaultdict

    by_seed_k_arm = defaultdict(dict)
    for row in all_rows:
        if row["operator"] == "COMMON_KEEP":
            continue
        key = (row["seed"], row["change_k"])
        by_seed_k_arm[key][row["arm"]] = row

    arm_names = [ad["arm"] for ad in ARM_DEFS]
    pairs = [
        (arm_names[i], arm_names[j])
        for i in range(len(arm_names))
        for j in range(i + 1, len(arm_names))
    ]

    divergence_rows = []
    for (seed, k), arm_rows in sorted(by_seed_k_arm.items()):
        if len(arm_rows) < len(ARM_DEFS):
            continue

        first_divs = {}
        for (a1, a2) in pairs:
            if a1 not in arm_rows or a2 not in arm_rows:
                first_divs[f"{a1}_vs_{a2}"] = "MISSING_ARM"
                continue

            r1, r2 = arm_rows[a1], arm_rows[a2]
            checkpoints = [
                ("C0",      _extract_md5(r1["c0_fp"]),      _extract_md5(r2["c0_fp"])),
                ("C1_F",    _extract_md5(r1["c1_F_fp"]),    _extract_md5(r2["c1_F_fp"])),
                ("C1_RIDS", _extract_md5(r1["c1_rids_fp"]), _extract_md5(r2["c1_rids_fp"])),
                ("C2",      _extract_md5(r1["c2_fp"]),      _extract_md5(r2["c2_fp"])),
                ("C3",      _extract_md5(r1["c3_fp"]),      _extract_md5(r2["c3_fp"])),
                ("C4",      _extract_md5(r1["c4_fp"]),      _extract_md5(r2["c4_fp"])),
            ]
            first_div = "AGREE"
            for cp_label, m1, m2 in checkpoints:
                if m1 is None and m2 is None:
                    continue  # both unavailable — skip
                if m1 is None or m2 is None:
                    first_div = f"UNAVAILABLE@{cp_label}"
                    break
                if m1 != m2:
                    first_div = f"DIVERGE@{cp_label}"
                    break
            first_divs[f"{a1}_vs_{a2}"] = first_div

        divergence_rows.append({
            "seed": seed, "change_k": k,
            **first_divs,
        })

    return divergence_rows


# ============================================================
# CB computation and verdict
# ============================================================

def compute_arm_cb(all_rows, arm_name, baseline_arm="DIAG_U03"):
    """Compute CombinedBenefit = mean_igd_baseline_response - mean_igd_arm_response."""
    from collections import defaultdict

    by_block_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in all_rows:
        if row["operator"] == "COMMON_KEEP":
            continue
        by_block_seed[row["block"]][row["seed"]][row["arm"]].append(row["igd_response"])

    results = {}
    for block in ["A", "B"]:
        seed_cbs = []
        for seed_v, arm_d in by_block_seed[block].items():
            if baseline_arm in arm_d and arm_name in arm_d:
                m_base = float(np.mean(arm_d[baseline_arm]))
                m_arm  = float(np.mean(arm_d[arm_name]))
                seed_cbs.append(m_base - m_arm)  # positive = arm better than baseline

        if not seed_cbs:
            results[block] = {"n_seeds": 0, "mean_CB": None, "ci_low": None, "ci_high": None}
            continue

        arr = np.asarray(seed_cbs)
        rng_boot = np.random.default_rng(42)
        boot_means = [
            np.mean(rng_boot.choice(arr, size=len(arr), replace=True))
            for _ in range(10000)
        ]
        results[block] = {
            "n_seeds": len(seed_cbs),
            "mean_CB": float(np.mean(arr)),
            "ci_low": float(np.percentile(boot_means, 2.5)),
            "ci_high": float(np.percentile(boot_means, 97.5)),
        }

    return results


def compute_scenario_cb_by_arm(all_rows):
    """Compute per-arm, per-block CB vs DIAG_U03 baseline."""
    arms = [ad["arm"] for ad in ARM_DEFS if ad["arm"] != "DIAG_U03"]
    return {
        arm: compute_arm_cb(all_rows, arm, baseline_arm="DIAG_U03")
        for arm in arms
    }


def classify_discrepancy(all_rows, divergence_report):
    """
    Classify the MIXED sign reversal discrepancy.

    Compares:
      L2B frozen result: DIAG_U03 Block A CB ≈ +2.264e-4
      D2C result:        PROD_O3  Block B CB ≈ -1.301e-4 (vs PROD_O0)

    Uses arm CB vs DIAG_U03 baseline to isolate where divergence originates.
    """
    # Reference: frozen L2B CB = +2.264e-4 (Block A DIAG_U03 = U03 baseline = 0)
    L2B_MIXED_CB = 2.264e-4   # H005/HARD component vs U03

    # H005 CB vs U03 baseline (DIAG_H005 Block A should replicate L2B)
    h005_cb = compute_arm_cb(all_rows, "DIAG_H005", baseline_arm="DIAG_U03")
    prod_o3_cb = compute_arm_cb(all_rows, "PROD_O3", baseline_arm="PROD_O0")
    diag_u03_self = {"A": 0.0, "B": 0.0}  # baseline is itself

    # DIAG_U03 absolute IGD by block (for SEED_BLOCK_HETEROGENEITY)
    diag_u03_by_block = {}
    from collections import defaultdict
    by_block = defaultdict(list)
    for row in all_rows:
        if row["arm"] == "DIAG_U03" and row["operator"] != "COMMON_KEEP":
            by_block[row["block"]].append(row["igd_response"])
    for block in ["A", "B"]:
        if by_block[block]:
            diag_u03_by_block[block] = float(np.mean(by_block[block]))
        else:
            diag_u03_by_block[block] = None

    # H005 CB direction per block
    h005_a = h005_cb.get("A", {})
    h005_b = h005_cb.get("B", {})
    h005_a_positive = (h005_a.get("mean_CB") or 0) > 1e-5
    h005_b_positive = (h005_b.get("mean_CB") or 0) > 1e-5
    h005_b_negative = (h005_b.get("mean_CB") or 0) < -1e-5

    # First-divergence pattern (between DIAG and PROD)
    diag_prod_pair = "DIAG_U03_vs_PROD_O0"
    divs = [r.get(diag_prod_pair, "UNKNOWN") for r in divergence_report]
    first_divs_set = set(divs)
    c2_only = all("C2" in d or d == "AGREE" for d in divs if d not in ("UNKNOWN", "MISSING_ARM"))
    c3_divs = any("C3" in d for d in divs)

    # Classification logic
    if h005_b_negative and not h005_a_positive:
        verdict = "SEED_BLOCK_HETEROGENEITY"
        explanation = (
            "DIAG_H005 gives negative CB on Block B AND non-positive on Block A. "
            "Both blocks show H005 non-beneficial in MIXED — the sign reversal is "
            "attributable to seed selection (81-100 vs 109-128)."
        )
    elif h005_a_positive and h005_b_negative:
        verdict = "SEED_BLOCK_HETEROGENEITY"
        explanation = (
            "DIAG_H005 (CRN/L2B path) shows positive CB on Block A but negative CB "
            "on Block B. The sign reversal is entirely explained by seed heterogeneity "
            "between the two blocks — the L2B measurement was seed-specific, not "
            "generalizing to seeds 109-128."
        )
    elif h005_a_positive and h005_b_positive:
        # DIAG is consistently positive but PROD shows negative
        prod_o3_a = prod_o3_cb.get("A", {})
        prod_o3_b = prod_o3_cb.get("B", {})
        prod_a_positive = (prod_o3_a.get("mean_CB") or 0) > 1e-5
        prod_b_negative = (prod_o3_b.get("mean_CB") or 0) < -1e-5
        if prod_b_negative and c2_only:
            verdict = "RNG_COUPLING_MISMATCH"
            explanation = (
                "DIAG path consistently positive (both blocks) but PROD shows negative "
                "CB on Block B. First divergence is at C2 (child generation — different "
                "RNG schemes). HARD survivors on PROD path use oracle RNG (not L2B CRN), "
                "producing different children that underperform under HARD survival."
            )
        elif prod_b_negative:
            verdict = "PRODUCTION_PROTOCOL_MISMATCH"
            explanation = (
                "DIAG path consistently positive but PROD shows negative CB. "
                "The diagnostic CRN path and production oracle path produce different "
                "outcomes even for the same (seed, change) due to protocol differences "
                "(RNG scheme, surv_seed, or internal state)."
            )
        else:
            verdict = "UNRESOLVED"
            explanation = "DIAG positive both blocks; PROD not negative — inconsistent pattern."
    else:
        verdict = "UNRESOLVED"
        explanation = (
            f"Inconclusive pattern. "
            f"DIAG_H005 Block A CB={h005_a.get('mean_CB'):.3e}, "
            f"Block B CB={h005_b.get('mean_CB'):.3e}. "
            f"No clear seed-block or protocol mismatch pattern."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "frozen_l2b_mixed_cb": L2B_MIXED_CB,
        "DIAG_H005_block_A": h005_a,
        "DIAG_H005_block_B": h005_b,
        "PROD_O3_vs_PROD_O0_block_A": prod_o3_cb.get("A"),
        "PROD_O3_vs_PROD_O0_block_B": prod_o3_cb.get("B"),
        "DIAG_U03_mean_igd_block_A": diag_u03_by_block.get("A"),
        "DIAG_U03_mean_igd_block_B": diag_u03_by_block.get("B"),
        "divergence_c2_only": c2_only,
        "divergence_c3_present": c3_divs,
    }


# ============================================================
# JSON serialisation helper
# ============================================================

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
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import csv

    OUT_DIR = Path("results/D2CX_longitudinal_bridge_audit")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    V_ref = reference(N_OBJ, N_PARTITIONS)
    R_ref = len(V_ref)
    mixed_map = make_d2cx_mixed_dynamics(R_ref)
    pf_star = get_true_pf(n_points=500)
    is_persistent_zone = make_pf_zones(pf_star, mixed_map)

    all_rows = []
    t0_wall = time.time()

    print("=== D2-CX LONGITUDINAL BRIDGE AUDIT ===")
    print(f"MIXED map: {mixed_map}")
    print(f"Block A seeds: {BLOCK_A_SEEDS}")
    print(f"Block B seeds: {BLOCK_B_SEEDS}\n")

    for block_label, seeds in [("A", BLOCK_A_SEEDS), ("B", BLOCK_B_SEEDS)]:
        print(f"--- Block {block_label} ---")
        for seed in seeds:
            t1 = time.time()
            try:
                rows = run_one_d2cx(
                    seed=seed,
                    n_scored=N_SCORED_CHANGES,
                    mixed_map=mixed_map,
                    pf_star=pf_star,
                    is_persistent_zone=is_persistent_zone,
                    block_label=block_label,
                )
                all_rows.extend(rows)
                n_arm_rows = len(rows)
                arms_done = {r["arm"] for r in rows if r["operator"] != "COMMON_KEEP"}
                print(
                    f"  seed={seed}: {n_arm_rows} rows, arms={sorted(arms_done)}, "
                    f"{time.time()-t1:.1f}s"
                )
            except RuntimeError as e:
                print(f"STOP (seed={seed} block={block_label}): {e}")
                sys.exit(1)

    print(f"\nTotal rows: {len(all_rows)}")
    print(f"Total elapsed: {time.time()-t0_wall:.1f}s\n")

    # ── Write bridge rows JSONL ─────────────────────────────────────────────
    jsonl_path = OUT_DIR / "D2CX_bridge_rows.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            safe_r = json_safe({
                k: v for k, v in r.items()
                if k not in ("c0_fp", "c1_F_fp", "c1_rids_fp", "c2_fp", "c3_fp", "c4_fp")
            })
            # Flatten fingerprint md5s only (not full dicts) to keep JSONL compact
            for cp, fp_key in [("c0", "c0_fp"), ("c1_F", "c1_F_fp"),
                                ("c1_rids", "c1_rids_fp"), ("c2", "c2_fp"),
                                ("c3", "c3_fp"), ("c4", "c4_fp")]:
                raw_fp = r.get(fp_key)
                safe_r[f"{cp}_md5"] = raw_fp.get("md5") if raw_fp else None
            f.write(json.dumps(safe_r) + "\n")
    print(f"Wrote bridge rows: {jsonl_path}")

    # ── Divergence report ───────────────────────────────────────────────────
    div_report = compute_divergence_report(all_rows)
    div_path = OUT_DIR / "D2CX_divergence_report.jsonl"
    with open(div_path, "w") as f:
        for r in div_report:
            f.write(json.dumps(json_safe(r)) + "\n")
    print(f"Wrote divergence report: {div_path}")

    # ── Arm summary CSV ─────────────────────────────────────────────────────
    from collections import defaultdict
    arm_block_rows = []
    by_block_arm = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        if row["operator"] != "COMMON_KEEP":
            by_block_arm[row["block"]][row["arm"]].append(row["igd_response"])

    for block in ["A", "B"]:
        for ad in ARM_DEFS:
            arm = ad["arm"]
            igds = by_block_arm[block].get(arm, [])
            arm_block_rows.append({
                "block": block,
                "arm": arm,
                "sigma": ad["sigma"],
                "use_diag_crn": ad["use_diag_crn"],
                "use_hard": ad["use_hard"],
                "n_rows": len(igds),
                "mean_igd_response": float(np.mean(igds)) if igds else None,
                "std_igd_response": float(np.std(igds)) if igds else None,
            })

    arm_csv = OUT_DIR / "D2CX_arm_summary.csv"
    if arm_block_rows:
        keys = list(arm_block_rows[0].keys())
        with open(arm_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(arm_block_rows)
    print(f"Wrote arm summary: {arm_csv}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    verdict_dict = classify_discrepancy(all_rows, div_report)
    verdict_path = OUT_DIR / "D2CX_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(json_safe(verdict_dict), f, indent=2)
    print(f"Wrote verdict: {verdict_path}")

    # ── Terminal output ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D2-CX LONGITUDINAL BRIDGE AUDIT RESULTS")
    print("=" * 60)
    print(f"\nFrozen L2-B MIXED CB (Block A, DIAG path): +2.264e-04")
    print(f"D2-C MIXED CB (Block B, PROD path):          -1.301e-04")
    print(f"\nBridge audit verdict: {verdict_dict['verdict']}")
    print(f"Explanation: {verdict_dict['explanation']}")

    for ad in ARM_DEFS:
        arm = ad["arm"]
        for block in ["A", "B"]:
            igds = by_block_arm[block].get(arm, [])
            mean_igd = float(np.mean(igds)) if igds else float("nan")
            print(f"  {arm} Block {block}: mean_igd_response={mean_igd:.6f}")

    h005_a = verdict_dict.get("DIAG_H005_block_A") or {}
    h005_b = verdict_dict.get("DIAG_H005_block_B") or {}
    p_a = verdict_dict.get("PROD_O3_vs_PROD_O0_block_A") or {}
    p_b = verdict_dict.get("PROD_O3_vs_PROD_O0_block_B") or {}
    print(f"\nDIAG_H005 vs DIAG_U03 CB:")
    print(f"  Block A: {h005_a.get('mean_CB', 'N/A'):.3e}  CI=[{h005_a.get('ci_low','N/A'):.3e}, {h005_a.get('ci_high','N/A'):.3e}]")
    print(f"  Block B: {h005_b.get('mean_CB', 'N/A'):.3e}  CI=[{h005_b.get('ci_low','N/A'):.3e}, {h005_b.get('ci_high','N/A'):.3e}]")
    print(f"\nPROD_O3 vs PROD_O0 CB:")
    print(f"  Block A: {p_a.get('mean_CB', 'N/A'):.3e}  CI=[{p_a.get('ci_low','N/A'):.3e}, {p_a.get('ci_high','N/A'):.3e}]")
    print(f"  Block B: {p_b.get('mean_CB', 'N/A'):.3e}  CI=[{p_b.get('ci_low','N/A'):.3e}, {p_b.get('ci_high','N/A'):.3e}]")
    print(f"\nFirst-divergence DIAG_U03 vs PROD_O0:")
    pair_key = "DIAG_U03_vs_PROD_O0"
    div_summary = {}
    for r in div_report:
        d = r.get(pair_key, "UNKNOWN")
        div_summary[d] = div_summary.get(d, 0) + 1
    for label, count in sorted(div_summary.items()):
        print(f"  {label}: {count}")

    print(f"\nD2CX_VERDICT: {verdict_dict['verdict']}")
    print(f"D2CX_DIVERGENCE_C2_ONLY: {verdict_dict.get('divergence_c2_only')}")
    print(f"D2CX_DIVERGENCE_C3_PRESENT: {verdict_dict.get('divergence_c3_present')}")
    print(f"D2CX_DIAG_H005_BLOCK_A_CB: {h005_a.get('mean_CB')}")
    print(f"D2CX_DIAG_H005_BLOCK_B_CB: {h005_b.get('mean_CB')}")
    print(f"D2CX_PROD_O3_BLOCK_A_CB: {p_a.get('mean_CB')}")
    print(f"D2CX_PROD_O3_BLOCK_B_CB: {p_b.get('mean_CB')}")
    print("=" * 60)
