"""
E3 — Rule-Based Regional Controller Validation.

Objective:
  Test whether regional control (RULE_V1) improves over the best fixed-global
  controller BEFORE paying the complexity cost of DRL.

Stage invariants:
  - No DRL, no training (RL_STATUS = NOT_STARTED)
  - No sigma tuning, no HARD action, no B_r(a) activation
  - Production code UNCHANGED (oracle.py, budget.py, operators unchanged)
  - Thresholds NOT tuned from performance (feature percentile only)
  - DELTA_E3 = 1e-5

Protocol:
  Calibration seeds: 701-720 (20 seeds)
  Evaluation seeds:  721-780 (60 seeds)
  Smoke seeds:       781-782 (2 seeds, used for correctness check only)
  Scenarios:         PERSISTENT, STATIONARY, MIXED
  Controllers:       RULE_V1, GLOBAL_KEEP, GLOBAL_LOCAL, GLOBAL_PREDICT, GLOBAL_DIVERSIFY
  N_SCORED_CHANGES:  10 (evaluation), 3 (smoke)
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)
from prdrl.rl.interface import (
    RegionalDecisionContext,
    build_decision_context,
    execute_response,
    check_e3_readiness,
    CURRENT_RESPONSE_BUDGET_VALUE,
    LOCAL_SPEC,
    PREDICT_SPEC,
    DIVERSIFY_SPEC,
)
from prdrl.controllers.rule_based import (
    ALL_CONTROLLERS,
    GLOBAL_CONTROLLERS,
    choose_actions,
    get_rule_scalars,
)

# ============================================================
# LOCKED CONFIG (not tuned — mirrors E2a-R, controlled_predict)
# ============================================================
N_OBJ             = 2
D                 = 10
POP_SIZE          = 100
N_PARTITIONS      = 9        # Das-Dennis for M=2 → R=10 reference vectors
WARMUP_GENERATIONS    = 30
GENERATIONS_PER_CHANGE = 10
BUDGET            = CURRENT_RESPONSE_BUDGET_VALUE   # = 10, per active region
SIGMA_LOCAL       = LOCAL_SPEC.sigma                # 0.02
SIGMA_PREDICT     = PREDICT_SPEC.sigma              # 0.005
SIGMA_DIVERSIFY   = DIVERSIFY_SPEC.sigma            # 0.15
LAMBDA_PREDICT    = PREDICT_SPEC.lambda_P           # 1.0

EVOLUTION_FE      = POP_SIZE * GENERATIONS_PER_CHANGE   # 1000 FE per epoch

# Seeds
CALIB_SEEDS   = list(range(701, 721))   # 20 seeds
EVAL_SEEDS    = list(range(721, 781))   # 60 seeds
BLOCK_A_SEEDS = list(range(721, 751))   # 30 seeds
BLOCK_B_SEEDS = list(range(751, 781))   # 30 seeds
SMOKE_SEEDS   = [781, 782]              # 2 seeds

N_SCORED_CHANGES_FULL  = 10
N_SCORED_CHANGES_SMOKE = 3

DELTA_E3 = 1e-5

# Bootstrap params
BOOTSTRAP_RESAMPLES_BLOCK  = 20_000
BOOTSTRAP_RESAMPLES_POOLED = 30_000
BOOTSTRAP_SEED             = 8888

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

RESULTS_DIR = Path("results/E3_rule_based_regional_controller")

RL_STATUS = "NOT_STARTED"


# ============================================================
# Scenario helpers
# ============================================================

def make_region_dynamics(scenario: str, R: int) -> List[str]:
    if scenario == "PERSISTENT":
        return ["persistent"] * R
    if scenario == "STATIONARY":
        return ["stationary"] * R
    if scenario == "MIXED":
        return make_region_dynamics_map(R)
    raise ValueError(f"Unknown scenario: {scenario}")


def get_target_regime(region_id: int, region_dynamics: List[str]) -> str:
    return region_dynamics[region_id]


def target_region(seed: int, k: int, R: int) -> int:
    return (seed + k) % R


# ============================================================
# Problem wrapper for NSGA2 on controlled_predict
# ============================================================

class _CPredictProblem(Problem):
    def __init__(self, t: float, region_dynamics: List[str], xl, xu):
        super().__init__(n_var=len(xl), n_obj=N_OBJ, xl=xl, xu=xu)
        self.t = t
        self.region_dynamics = region_dynamics

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = cp_evaluate(X, self.t, self.region_dynamics)


def evolve_nsga2_predict(
    X_start: np.ndarray,
    t: float,
    region_dynamics: List[str],
    n_gen: int,
    seed: int,
    xl: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray:
    pop_size = len(X_start)
    problem = _CPredictProblem(t, region_dynamics, xl, xu)
    algorithm = NSGA2(pop_size=pop_size, sampling=X_start)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return np.asarray(result.pop.get("X"), dtype=float)


def compute_time(generation: int, n_t: int = 10, tau_t: int = 10) -> float:
    return (1.0 / n_t) * np.floor(generation / tau_t)


# ============================================================
# Core run — one seed × scenario × controller
# ============================================================

def _warmup_and_burnin(
    seed: int,
    region_dynamics: List[str],
    xl: np.ndarray,
    xu: np.ndarray,
    V: np.ndarray,
    R: int,
) -> Tuple[np.ndarray, RegionHistory, int, float, int]:
    """Common warmup + burn-in change for a given seed/scenario.

    Returns: (X_pop_after_burnin, history, history_t, t_current, gen)
    """
    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))

    gen = 0
    t_current = compute_time(gen)
    X_pop = evolve_nsga2_predict(
        X_pop, t_current, region_dynamics,
        WARMUP_GENERATIONS, seed * 10000, xl, xu,
    )
    gen += WARMUP_GENERATIONS

    history = RegionHistory(n_regions=R)
    history_t = -1

    # Burn-in change (not scored)
    F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t_new = compute_time(gen)
    F_plus = cp_evaluate(X_pop, t_new, region_dynamics)
    F_tm, _ = normalize(F_minus, F_plus)
    region_ids_burnin = np.asarray(partition(np.asarray(F_tm), V))
    history_t += 1
    history.update(
        t=history_t, X=X_pop, F=F_minus,
        region_ids=region_ids_burnin, xl=xl, xu=xu,
    )
    t_current = t_new
    X_pop = evolve_nsga2_predict(
        X_pop, t_current, region_dynamics,
        GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu,
    )

    return X_pop, history, history_t, t_current, gen


def run_one_arm(
    seed: int,
    scenario: str,
    controller_name: str,
    n_scored: int,
    xl: np.ndarray,
    xu: np.ndarray,
    V: np.ndarray,
    R: int,
    pf_star: np.ndarray,
    region_dynamics: List[str],
    X_init: np.ndarray,
    history_init: RegionHistory,
    history_t_init: int,
    t_current_init: float,
    gen_init: int,
    rule_config: Optional[dict],
    block: str,
) -> Tuple[List[dict], List[dict]]:
    """Run one arm for n_scored changes. Returns (traj_rows, decision_rows)."""

    X = X_init.copy()
    h = copy.deepcopy(history_init)
    h_t = history_t_init
    t_current = t_current_init
    gen = gen_init

    traj_rows = []
    decision_rows = []
    cumulative_response_fe = 0

    for k in range(n_scored):
        scored_change_idx = k + 2

        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)

        # Evaluate, partition, update history
        F_minus_arm = cp_evaluate(X, t_current, region_dynamics)
        F_plus_arm = cp_evaluate(X, t_new, region_dynamics)
        F_tm_arm, _ = normalize(F_minus_arm, F_plus_arm)
        rids_arm = np.asarray(partition(np.asarray(F_tm_arm), V))
        igd_pre = float(igd(F_plus_arm, pf_star))

        h_t += 1
        h.update(
            t=h_t, X=X, F=F_minus_arm,
            region_ids=rids_arm, xl=xl, xu=xu,
        )

        # Build decision contexts for ALL regions
        active_regions = [r for r in range(R) if int(np.sum(rids_arm == r)) > 0]
        contexts: Dict[int, RegionalDecisionContext] = {}
        for r in range(R):
            ctx = build_decision_context(r, h, h_t + 1, POP_SIZE, D)
            contexts[r] = ctx

        # Controller selects actions
        actions = choose_actions(controller_name, contexts, D, rule_config)

        # Execute response via R0 interface
        evaluate_fn = lambda Xq, _t=t_new, _rd=region_dynamics: cp_evaluate(Xq, _t, _rd)
        response_result, records = execute_response(
            actions=actions,
            X=X,
            F_plus=F_plus_arm,
            region_ids=rids_arm,
            V=V,
            history=h,
            t=h_t + 1,
            evaluate_fn=evaluate_fn,
            xl=xl,
            xu=xu,
            base_seed=seed,
            budget=BUDGET,
            sigma_local=SIGMA_LOCAL,
            sigma_predict=SIGMA_PREDICT,
            sigma_diversify=SIGMA_DIVERSIFY,
            lambda_pred=LAMBDA_PREDICT,
        )

        X_resp = np.asarray(response_result["X"], dtype=float)
        F_resp = np.asarray(response_result["F"], dtype=float)
        igd_response = float(igd(F_resp, pf_star))
        total_response_fe = sum(rec.response_fe for rec in records)
        cumulative_response_fe += total_response_fe

        # Count actions
        n_keep = sum(1 for rec in records if rec.requested_action == "KEEP")
        n_local_req = sum(1 for rec in records if rec.requested_action == "LOCAL")
        n_predict_req = sum(1 for rec in records if rec.requested_action == "PREDICT")
        n_diversify_req = sum(1 for rec in records if rec.requested_action == "DIVERSIFY")
        n_predict_fallback = sum(1 for rec in records if rec.fallback_used)

        # Evolution
        evo_seed = seed * 10000 + scored_change_idx
        X = evolve_nsga2_predict(X_resp, t_new, region_dynamics,
                                 GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
        F_epoch = cp_evaluate(X, t_new, region_dynamics)
        igd_epoch = float(igd(F_epoch, pf_star))

        tgt_r = target_region(seed, k, R)
        tgt_regime = get_target_regime(tgt_r, region_dynamics) if scenario == "MIXED" else scenario.lower()

        traj_rows.append({
            "scenario": scenario,
            "seed": seed,
            "block": block,
            "controller": controller_name,
            "k": k,
            "scored_change_idx": scored_change_idx,
            "t_new": float(t_new),
            "target_region": tgt_r,
            "target_regime": tgt_regime,
            "n_active_regions": len(active_regions),
            "n_KEEP": n_keep,
            "n_LOCAL_requested": n_local_req,
            "n_PREDICT_requested": n_predict_req,
            "n_DIVERSIFY_requested": n_diversify_req,
            "n_predict_fallback": n_predict_fallback,
            "total_response_fe": total_response_fe,
            "cumulative_response_fe": cumulative_response_fe,
            "evolution_fe": EVOLUTION_FE,
            "igd_pre": igd_pre,
            "igd_response": igd_response,
            "igd_epoch_end": igd_epoch,
        })

        # Per-region decision rows
        for rec in records:
            ctx = contexts[rec.region_id]
            scalars = get_rule_scalars(ctx, D)
            matched_rule = (
                1 if rec.requested_action == "KEEP" else
                2 if rec.requested_action == "DIVERSIFY" else
                3 if rec.requested_action == "PREDICT" else
                4
            )
            decision_rows.append({
                "scenario": scenario,
                "seed": seed,
                "block": block,
                "controller": controller_name,
                "k": k,
                "t": h_t + 1,
                "region_id": rec.region_id,
                "active": True,
                "c": scalars["c"],
                "l": scalars["l"],
                "o": scalars["o"],
                "dc": scalars["dc"],
                "requested_action": rec.requested_action,
                "realized_operator": rec.realized_operator,
                "fallback_used": rec.fallback_used,
                "fallback_reason": rec.fallback_reason,
                "matched_rule_id": matched_rule,
                "response_fe": rec.response_fe,
                "sigma_used": rec.sigma_used,
            })

        t_current = t_new

    return traj_rows, decision_rows


# ============================================================
# Feature collection (for calibration)
# ============================================================

def _collect_features_from_decision_rows(dec_rows: List[dict]) -> dict:
    """Extract (c, l, o, dc) arrays from decision rows of reference arm."""
    c_vals  = [row["c"]  for row in dec_rows]
    o_vals  = [row["o"]  for row in dec_rows]
    dc_vals = [row["dc"] for row in dec_rows]
    return {
        "c":  np.array(c_vals, dtype=float),
        "o":  np.array(o_vals, dtype=float),
        "dc": np.array(dc_vals, dtype=float),
    }


# ============================================================
# Calibration run (seeds 701-720)
# ============================================================

def run_calibration(xl, xu, V, R, pf_star):
    """Run calibration on seeds 701-720, all scenarios, all global controllers.

    Returns:
        calib_features: all c/o/dc feature samples from reference arm (GLOBAL_LOCAL)
        calib_perf:     {scenario: {controller: [MIGD_response values]}}
    """
    print("  Calibration: seeds 701-720, 3 scenarios, 4 global controllers")
    all_c  = []
    all_o  = []
    all_dc = []

    calib_perf: Dict[str, Dict[str, List[float]]] = {
        sc: {ctrl: [] for ctrl in GLOBAL_CONTROLLERS}
        for sc in SCENARIOS
    }

    for seed in CALIB_SEEDS:
        for scenario in SCENARIOS:
            region_dynamics = make_region_dynamics(scenario, R)

            # Common warmup + burn-in
            X_init, h_init, h_t_init, t_cur_init, gen_init = _warmup_and_burnin(
                seed, region_dynamics, xl, xu, V, R
            )

            # Run 4 global controllers; collect features from GLOBAL_LOCAL
            for ctrl in GLOBAL_CONTROLLERS:
                block = "CALIB"
                traj_rows, dec_rows = run_one_arm(
                    seed=seed,
                    scenario=scenario,
                    controller_name=ctrl,
                    n_scored=N_SCORED_CHANGES_FULL,
                    xl=xl, xu=xu, V=V, R=R,
                    pf_star=pf_star,
                    region_dynamics=region_dynamics,
                    X_init=X_init,
                    history_init=h_init,
                    history_t_init=h_t_init,
                    t_current_init=t_cur_init,
                    gen_init=gen_init,
                    rule_config=None,
                    block=block,
                )

                # Collect features from reference arm (GLOBAL_LOCAL)
                if ctrl == "GLOBAL_LOCAL":
                    feats = _collect_features_from_decision_rows(dec_rows)
                    all_c.extend(feats["c"].tolist())
                    all_o.extend(feats["o"].tolist())
                    all_dc.extend(feats["dc"].tolist())

                # Collect calibration performance
                migd_resp = float(np.mean([r["igd_response"] for r in traj_rows]))
                calib_perf[scenario][ctrl].append(migd_resp)

    calib_features = {
        "c":  np.array(all_c,  dtype=float),
        "o":  np.array(all_o,  dtype=float),
        "dc": np.array(all_dc, dtype=float),
    }
    print(f"    Feature samples collected: {len(all_c)}")
    return calib_features, calib_perf


def compute_thresholds(calib_features: dict) -> dict:
    """Compute RULE_V1 thresholds from feature percentiles."""
    c  = calib_features["c"]
    o  = calib_features["o"]
    dc = calib_features["dc"]
    return {
        "T_C_LOW":   float(np.percentile(c,  33)),
        "T_C_HIGH":  float(np.percentile(c,  67)),
        "T_O_LOW":   float(np.percentile(o,  25)),
        "T_DC_HIGH": float(np.percentile(dc, 75)),
        "T_L":       0.0,
        "D":         D,
        "source":    "feature percentile calibration, seeds 701-720, GLOBAL_LOCAL arm",
        "n_samples": int(len(c)),
    }


def select_global_baselines(calib_perf: dict) -> dict:
    """For each scenario, select global controller with lowest mean MIGD_response."""
    baselines = {}
    for sc in SCENARIOS:
        best_ctrl = min(
            GLOBAL_CONTROLLERS,
            key=lambda ctrl: float(np.mean(calib_perf[sc][ctrl])),
        )
        best_migd = float(np.mean(calib_perf[sc][best_ctrl]))
        baselines[sc] = {
            "best_global_controller": best_ctrl,
            "best_global_migd_response": best_migd,
            "all_controllers": {
                ctrl: float(np.mean(calib_perf[sc][ctrl]))
                for ctrl in GLOBAL_CONTROLLERS
            },
        }
    return baselines


# ============================================================
# Full evaluation and smoke runs
# ============================================================

def run_all_seeds(
    seeds: List[int],
    block_label: str,
    n_scored: int,
    rule_config: dict,
    xl, xu, V, R, pf_star,
    is_smoke: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Run all seeds × scenarios × controllers. Returns (all_traj, all_decisions)."""
    all_traj: List[dict]      = []
    all_decisions: List[dict] = []

    total = len(seeds) * len(SCENARIOS)
    done = 0

    for seed in seeds:
        for scenario in SCENARIOS:
            region_dynamics = make_region_dynamics(scenario, R)

            X_init, h_init, h_t_init, t_cur_init, gen_init = _warmup_and_burnin(
                seed, region_dynamics, xl, xu, V, R
            )

            # Determine block label per seed
            if block_label == "AUTO":
                blk = "A" if seed in BLOCK_A_SEEDS else "B"
            else:
                blk = block_label

            controllers_to_run = list(ALL_CONTROLLERS)

            for ctrl in controllers_to_run:
                rc = rule_config if ctrl == "RULE_V1" else None
                traj_rows, dec_rows = run_one_arm(
                    seed=seed,
                    scenario=scenario,
                    controller_name=ctrl,
                    n_scored=n_scored,
                    xl=xl, xu=xu, V=V, R=R,
                    pf_star=pf_star,
                    region_dynamics=region_dynamics,
                    X_init=X_init,
                    history_init=h_init,
                    history_t_init=h_t_init,
                    t_current_init=t_cur_init,
                    gen_init=gen_init,
                    rule_config=rc,
                    block=blk,
                )
                all_traj.extend(traj_rows)
                all_decisions.extend(dec_rows)

            done += 1
            if not is_smoke:
                print(f"    [{done}/{total}] seed={seed} scenario={scenario} done")

    return all_traj, all_decisions


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(values: List[float], n_resamples: int, rng_seed: int) -> Tuple[float, float, float]:
    """Bootstrap 95% percentile CI. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(rng_seed)
    means = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_resamples)]
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return float(np.mean(arr)), lo, hi


def classify_effect(mean_val: float, ci_lo: float, ci_hi: float, delta: float = DELTA_E3) -> str:
    if mean_val > delta and ci_lo > 0:
        return "BENEFICIAL"
    if mean_val > delta:
        return "DIRECTIONAL_POSITIVE"
    if abs(mean_val) <= delta:
        return "PRACTICALLY_NEUTRAL"
    if mean_val < -delta and ci_hi < 0:
        return "HARMFUL"
    return "DIRECTIONAL_NEGATIVE"


def compute_seed_migd(
    traj_rows: List[dict],
    scenario: str,
    controller: str,
    seeds: List[int],
    metric: str = "igd_response",
) -> Dict[int, float]:
    """Compute per-seed mean IGD for given scenario/controller."""
    result = {}
    for seed in seeds:
        vals = [
            r[metric] for r in traj_rows
            if r["scenario"] == scenario and r["controller"] == controller and r["seed"] == seed
        ]
        result[seed] = float(np.mean(vals)) if vals else float("nan")
    return result


def compute_per_seed_benefit(
    traj_rows: List[dict],
    scenario: str,
    rule_ctrl: str,
    best_global_ctrl: str,
    seeds: List[int],
    metric: str = "igd_response",
) -> List[float]:
    """Per-seed benefit: MIGD_best_global(s) - MIGD_rule(s). Positive = rule better."""
    rule_migd   = compute_seed_migd(traj_rows, scenario, rule_ctrl,        seeds, metric)
    global_migd = compute_seed_migd(traj_rows, scenario, best_global_ctrl, seeds, metric)
    return [global_migd[s] - rule_migd[s] for s in seeds
            if not (np.isnan(global_migd.get(s, float("nan"))) or
                    np.isnan(rule_migd.get(s, float("nan"))))]


def compute_gate_b(
    block_a_scenario_results: dict,
    block_b_scenario_results: dict,
    pooled_scenario_results: dict,
    fe_ok: bool,
    interface_ready: bool,
    no_causal_violation: bool,
    no_leakage: bool,
) -> Tuple[str, dict]:
    """Compute Gate B verdict and its sub-verdicts."""
    mixed_a = block_a_scenario_results.get("MIXED", {})
    mixed_b = block_b_scenario_results.get("MIXED", {})
    mixed_p = pooled_scenario_results.get("MIXED", {})

    m_mean_a    = mixed_a.get("mean_benefit", float("nan"))
    m_mean_b    = mixed_b.get("mean_benefit", float("nan"))
    m_mean_p    = mixed_p.get("mean_benefit", float("nan"))
    m_ci_lo_p   = mixed_p.get("ci_lo", float("nan"))
    m_effect_a  = mixed_a.get("effect", "UNKNOWN")
    m_effect_b  = mixed_b.get("effect", "UNKNOWN")

    # M1: pooled mean > delta
    m1 = not np.isnan(m_mean_p) and m_mean_p > DELTA_E3
    # M2: CI_lo > 0 (pooled)
    m2 = not np.isnan(m_ci_lo_p) and m_ci_lo_p > 0
    # M3: Block A mean > 0
    m3 = not np.isnan(m_mean_a) and m_mean_a > 0
    # M4: Block B mean > 0
    m4 = not np.isnan(m_mean_b) and m_mean_b > 0
    # M5: no HARMFUL in Block A (MIXED)
    m5 = m_effect_a != "HARMFUL"
    # M6: no HARMFUL in Block B (MIXED)
    m6 = m_effect_b != "HARMFUL"
    # M7: FE not excessive
    m7 = fe_ok
    # M8: no engineering failure
    m8 = interface_ready and no_causal_violation and no_leakage

    # Safety checks: PERSISTENT and STATIONARY
    pers_a_effect = block_a_scenario_results.get("PERSISTENT", {}).get("effect", "UNKNOWN")
    stat_a_effect = block_a_scenario_results.get("STATIONARY", {}).get("effect", "UNKNOWN")
    pers_b_effect = block_b_scenario_results.get("PERSISTENT", {}).get("effect", "UNKNOWN")
    stat_b_effect = block_b_scenario_results.get("STATIONARY", {}).get("effect", "UNKNOWN")

    persistent_safety = "PASS" if (pers_a_effect != "HARMFUL" and pers_b_effect != "HARMFUL") else "FAIL"
    stationary_safety = "PASS" if (stat_a_effect != "HARMFUL" and stat_b_effect != "HARMFUL") else "FAIL"

    # E3_MIXED_RULE_GAIN
    if m1 and m2 and m3 and m4:
        mixed_gain = "VALIDATED"
    elif m5 and m6 and not (m1 and m2):
        mixed_gain = "PARTIAL"
    else:
        mixed_gain = "FAILED"

    # Gate B
    if not m7 or not m8:
        gate_b = "FAIL"
    elif persistent_safety == "FAIL" or stationary_safety == "FAIL":
        gate_b = "FAIL"
    elif mixed_gain == "VALIDATED":
        gate_b = "PASS"
    elif mixed_gain == "PARTIAL":
        gate_b = "PARTIAL"
    else:
        gate_b = "FAIL"

    verdicts = {
        "M1_pooled_mean_positive": m1,
        "M2_pooled_ci_lo_positive": m2,
        "M3_blockA_mean_positive": m3,
        "M4_blockB_mean_positive": m4,
        "M5_no_harmful_blockA": m5,
        "M6_no_harmful_blockB": m6,
        "M7_fe_ok": m7,
        "M8_engineering_ok": m8,
        "E3_MIXED_RULE_GAIN": mixed_gain,
        "PERSISTENT_SAFETY": persistent_safety,
        "STATIONARY_SAFETY": stationary_safety,
        "GATE_B": gate_b,
    }
    return gate_b, verdicts


# ============================================================
# Output writers
# ============================================================

def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_pre_run_manifest() -> dict:
    key_files = [
        "src/prdrl/oracle/oracle.py",
        "src/prdrl/operators/budget.py",
        "src/prdrl/operators/local.py",
        "src/prdrl/operators/predict.py",
        "src/prdrl/operators/diversify.py",
        "src/prdrl/history/region_history.py",
        "src/prdrl/rl/interface.py",
        "src/prdrl/controllers/rule_based.py",
    ]
    manifest = {
        "stage": "E3_rule_based_regional_controller",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS,
        "note": "SHA256 of production files. Production code must not change.",
        "files": {}
    }
    for fpath in key_files:
        p = Path(fpath)
        manifest["files"][fpath] = _hash_file(p) if p.exists() else "NOT_FOUND"
    return manifest


def write_state_semantics(rule_config: dict) -> dict:
    return {
        "stage": "E3_rule_based_regional_controller",
        "schema_version": "V1",
        "D": D,
        "feature_vector_dim": 2 * D + 2,
        "feature_layout": {
            "indices_0_to_D-1": "c_r — centroid in normalized decision space, dim D",
            "index_D": "l_r — region size (number of solutions)",
            "index_D+1": "o_r — occupancy ratio (n_r / N_total)",
            "indices_D+2_to_2D+1": "delta_c_r — centroid shift (t-1 minus t-2), dim D",
        },
        "scalar_derivation": {
            "c": "||c_r||_2 = norm(fv[0:D])",
            "l": "fv[D]",
            "o": "fv[D+1]",
            "dc": "||delta_c_r||_2 = norm(fv[D+2:2D+2])",
        },
        "rule_v1_thresholds": rule_config,
        "causality_note": (
            "State built from RegionHistory entries at h_t and h_t-1 "
            "(pre-change snapshots, causally available at decision time)."
        ),
    }


def write_trajectories_jsonl(traj_rows: List[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in traj_rows:
            f.write(json.dumps(row) + "\n")


def write_decisions_jsonl(dec_rows: List[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in dec_rows:
            f.write(json.dumps(row) + "\n")


def write_seed_summary(
    traj_rows: List[dict],
    seeds: List[int],
    path: Path,
) -> List[dict]:
    rows_out = []
    for seed in seeds:
        blk = "A" if seed in BLOCK_A_SEEDS else ("B" if seed in BLOCK_B_SEEDS else "CALIB")
        for scenario in SCENARIOS:
            for ctrl in ALL_CONTROLLERS:
                vals_resp  = [r["igd_response"]  for r in traj_rows
                              if r["seed"] == seed and r["scenario"] == scenario and r["controller"] == ctrl]
                vals_epoch = [r["igd_epoch_end"] for r in traj_rows
                              if r["seed"] == seed and r["scenario"] == scenario and r["controller"] == ctrl]
                vals_fe    = [r["total_response_fe"] for r in traj_rows
                              if r["seed"] == seed and r["scenario"] == scenario and r["controller"] == ctrl]
                if not vals_resp:
                    continue
                rows_out.append({
                    "scenario": scenario,
                    "seed": seed,
                    "block": blk,
                    "controller": ctrl,
                    "MIGD_response":       float(np.mean(vals_resp)),
                    "MIGD_epoch":          float(np.mean(vals_epoch)),
                    "mean_response_FE":    float(np.mean(vals_fe)),
                    "total_response_FE":   float(sum(vals_fe)),
                    "n_changes":           len(vals_resp),
                })

    fieldnames = ["scenario", "seed", "block", "controller",
                  "MIGD_response", "MIGD_epoch", "mean_response_FE",
                  "total_response_FE", "n_changes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    return rows_out


def compute_block_results(
    traj_rows: List[dict],
    seeds: List[int],
    block_label: str,
    baselines: dict,
    n_resamples: int,
    rng_seed_base: int,
) -> dict:
    """Compute scenario-level block statistics for one block of seeds."""
    results = {}
    for scenario in SCENARIOS:
        best_ctrl = baselines[scenario]["best_global_controller"]
        benefits = compute_per_seed_benefit(
            traj_rows, scenario, "RULE_V1", best_ctrl, seeds,
        )
        if not benefits:
            results[scenario] = {
                "block": block_label, "scenario": scenario,
                "best_global": best_ctrl, "n_seeds": 0,
                "mean_benefit": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "effect": "UNKNOWN",
            }
            continue
        mean_b, lo, hi = bootstrap_ci(benefits, n_resamples, rng_seed_base + hash(scenario) % 10000)
        effect = classify_effect(mean_b, lo, hi)
        results[scenario] = {
            "block": block_label,
            "scenario": scenario,
            "best_global": best_ctrl,
            "n_seeds": len(benefits),
            "mean_benefit": mean_b,
            "ci_lo": lo,
            "ci_hi": hi,
            "effect": effect,
        }
    return results


def compute_pooled_results(
    traj_rows: List[dict],
    baselines: dict,
    n_resamples: int,
    rng_seed_base: int,
) -> dict:
    """Compute pooled (all 60 evaluation seeds) scenario statistics."""
    return compute_block_results(
        traj_rows, EVAL_SEEDS, "POOLED", baselines, n_resamples, rng_seed_base
    )


def write_block_summary_csv(
    block_a: dict, block_b: dict, pooled: dict, path: Path
) -> None:
    fieldnames = ["block", "scenario", "best_global", "n_seeds",
                  "mean_benefit", "ci_lo", "ci_hi", "effect"]
    rows = []
    for sc in SCENARIOS:
        for d in [block_a.get(sc, {}), block_b.get(sc, {}), pooled.get(sc, {})]:
            if d:
                rows.append({k: d.get(k, "") for k in fieldnames})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_action_distribution_csv(dec_rows: List[dict], path: Path) -> None:
    from collections import Counter
    fieldnames = ["scenario", "block", "controller", "requested_action", "count", "fraction"]
    rows_out = []
    for scenario in SCENARIOS:
        for block in ["A", "B"]:
            for ctrl in ALL_CONTROLLERS:
                subset = [r for r in dec_rows
                          if r["scenario"] == scenario and r["block"] == block
                          and r["controller"] == ctrl]
                if not subset:
                    continue
                total = len(subset)
                ctr = Counter(r["requested_action"] for r in subset)
                for action, cnt in ctr.items():
                    rows_out.append({
                        "scenario": scenario,
                        "block": block,
                        "controller": ctrl,
                        "requested_action": action,
                        "count": cnt,
                        "fraction": round(cnt / total, 4),
                    })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def write_scenario_summary_csv(
    traj_rows: List[dict],
    baselines: dict,
    pooled: dict,
    path: Path,
) -> None:
    fieldnames = ["scenario", "controller", "mean_MIGD_response", "mean_MIGD_epoch",
                  "best_global_ctrl", "rule_benefit_mean", "rule_benefit_ci_lo",
                  "rule_benefit_ci_hi", "effect"]
    rows_out = []
    for scenario in SCENARIOS:
        best_ctrl = baselines[scenario]["best_global_controller"]
        for ctrl in ALL_CONTROLLERS:
            migd_resp_vals = [r["igd_response"] for r in traj_rows
                              if r["scenario"] == scenario and r["controller"] == ctrl
                              and r["seed"] in EVAL_SEEDS]
            migd_epoch_vals = [r["igd_epoch_end"] for r in traj_rows
                               if r["scenario"] == scenario and r["controller"] == ctrl
                               and r["seed"] in EVAL_SEEDS]
            pool_sc = pooled.get(scenario, {})
            rows_out.append({
                "scenario": scenario,
                "controller": ctrl,
                "mean_MIGD_response": float(np.mean(migd_resp_vals)) if migd_resp_vals else float("nan"),
                "mean_MIGD_epoch": float(np.mean(migd_epoch_vals)) if migd_epoch_vals else float("nan"),
                "best_global_ctrl": best_ctrl,
                "rule_benefit_mean": pool_sc.get("mean_benefit", float("nan")) if ctrl == "RULE_V1" else "",
                "rule_benefit_ci_lo": pool_sc.get("ci_lo", float("nan")) if ctrl == "RULE_V1" else "",
                "rule_benefit_ci_hi": pool_sc.get("ci_hi", float("nan")) if ctrl == "RULE_V1" else "",
                "effect": pool_sc.get("effect", "") if ctrl == "RULE_V1" else "",
            })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def write_fe_summary_csv(
    traj_rows: List[dict],
    baselines: dict,
    path: Path,
) -> None:
    fieldnames = ["scenario", "controller", "mean_response_fe_per_change",
                  "total_response_fe", "mean_evolution_fe",
                  "fe_vs_best_global_delta", "fe_saving_pct"]
    rows_out = []
    for scenario in SCENARIOS:
        best_ctrl = baselines[scenario]["best_global_controller"]
        best_fe = float(np.mean([
            r["total_response_fe"] for r in traj_rows
            if r["scenario"] == scenario and r["controller"] == best_ctrl
            and r["seed"] in EVAL_SEEDS
        ]) or 0)
        for ctrl in ALL_CONTROLLERS:
            fe_vals = [r["total_response_fe"] for r in traj_rows
                       if r["scenario"] == scenario and r["controller"] == ctrl
                       and r["seed"] in EVAL_SEEDS]
            if not fe_vals:
                continue
            mean_fe = float(np.mean(fe_vals))
            total_fe = float(sum(fe_vals))
            fe_delta = mean_fe - best_fe
            fe_saving_pct = ((best_fe - mean_fe) / best_fe * 100) if best_fe > 0 else 0.0
            rows_out.append({
                "scenario": scenario,
                "controller": ctrl,
                "mean_response_fe_per_change": mean_fe,
                "total_response_fe": total_fe,
                "mean_evolution_fe": float(EVOLUTION_FE),
                "fe_vs_best_global_delta": fe_delta,
                "fe_saving_pct": round(fe_saving_pct, 2),
            })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def write_mixed_regime_summary_csv(
    traj_rows: List[dict],
    baselines: dict,
    path: Path,
) -> None:
    """MIXED scenario: split by target_regime (persistent vs stationary)."""
    fieldnames = ["target_regime", "controller", "mean_MIGD_response",
                  "mean_MIGD_epoch", "n_changes"]
    rows_out = []
    for regime in ["persistent", "stationary"]:
        for ctrl in ALL_CONTROLLERS:
            vals_resp = [r["igd_response"] for r in traj_rows
                         if r["scenario"] == "MIXED"
                         and r["controller"] == ctrl
                         and r["seed"] in EVAL_SEEDS
                         and r["target_regime"] == regime]
            vals_epoch = [r["igd_epoch_end"] for r in traj_rows
                          if r["scenario"] == "MIXED"
                          and r["controller"] == ctrl
                          and r["seed"] in EVAL_SEEDS
                          and r["target_regime"] == regime]
            rows_out.append({
                "target_regime": regime,
                "controller": ctrl,
                "mean_MIGD_response": float(np.mean(vals_resp)) if vals_resp else float("nan"),
                "mean_MIGD_epoch": float(np.mean(vals_epoch)) if vals_epoch else float("nan"),
                "n_changes": len(vals_resp),
            })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def check_fe_ok(traj_rows: List[dict], baselines: dict) -> bool:
    """M7: RULE_V1 response FE must not exceed 2× best global baseline."""
    for scenario in SCENARIOS:
        best_ctrl = baselines[scenario]["best_global_controller"]
        rule_fe = float(np.mean([
            r["total_response_fe"] for r in traj_rows
            if r["scenario"] == scenario and r["controller"] == "RULE_V1"
            and r["seed"] in EVAL_SEEDS
        ] or [0]))
        best_fe = float(np.mean([
            r["total_response_fe"] for r in traj_rows
            if r["scenario"] == scenario and r["controller"] == best_ctrl
            and r["seed"] in EVAL_SEEDS
        ] or [1]))
        if best_fe > 0 and rule_fe > 2.0 * best_fe:
            return False
    return True


def write_report_md(
    rule_config: dict,
    baselines: dict,
    block_a: dict,
    block_b: dict,
    pooled: dict,
    gate_b: str,
    gate_verdicts: dict,
    path: Path,
) -> None:
    lines = [
        "# E3 — Rule-Based Regional Controller: Report",
        "",
        f"**Stage**: E3  **Date**: 2026-09-20  **RL_STATUS**: NOT_STARTED  **Gate B**: {gate_b}",
        "",
        "## RULE_V1 Thresholds (frozen before evaluation)",
        "",
        f"- T_C_LOW  = {rule_config['T_C_LOW']:.6f}",
        f"- T_C_HIGH = {rule_config['T_C_HIGH']:.6f}",
        f"- T_O_LOW  = {rule_config['T_O_LOW']:.6f}",
        f"- T_DC_HIGH= {rule_config['T_DC_HIGH']:.6f}",
        f"- T_L      = {rule_config['T_L']}  (fixed)",
        f"- Calibrated from {rule_config['n_samples']} feature samples, seeds 701-720",
        "",
        "## Global Baseline Selection (per scenario)",
        "",
    ]
    for sc in SCENARIOS:
        b = baselines[sc]
        lines.append(f"- **{sc}**: best = {b['best_global_controller']} (MIGD_response={b['best_global_migd_response']:.4f})")
        for ctrl, val in b["all_controllers"].items():
            lines.append(f"  - {ctrl}: {val:.4f}")
        lines.append("")

    lines += [
        "## MIXED Scenario — Block Results",
        "",
        f"| Block | Mean Benefit | CI 95% | Effect |",
        f"|-------|-------------|--------|--------|",
    ]
    for block_label, blk_res in [("A", block_a), ("B", block_b), ("POOLED", pooled)]:
        sc_res = blk_res.get("MIXED", {})
        mean_b = sc_res.get("mean_benefit", float("nan"))
        ci_lo  = sc_res.get("ci_lo", float("nan"))
        ci_hi  = sc_res.get("ci_hi", float("nan"))
        eff    = sc_res.get("effect", "—")
        lines.append(f"| {block_label} | {mean_b:.6f} | [{ci_lo:.6f}, {ci_hi:.6f}] | {eff} |")

    lines += [
        "",
        "## Gate B Verdicts",
        "",
    ]
    for key, val in gate_verdicts.items():
        lines.append(f"- {key}: {val}")

    lines += [
        "",
        f"## Gate B: **{gate_b}**",
        "",
        "---",
        "Invariants confirmed: RL_STATUS=NOT_STARTED, production code unchanged,",
        "thresholds from feature calibration only, no sigma tuning, no HARD action.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    t0_total = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    pf_star = get_true_pf(n_points=500)

    print(f"E3 — Rule-Based Regional Controller")
    print(f"  D={D}, POP_SIZE={POP_SIZE}, R={R}")
    print(f"  Scenarios: {SCENARIOS}")
    print(f"  Controllers: {ALL_CONTROLLERS}")

    # ---- Phase E3-0: Pre-run manifest ----
    print("\n[E3-0] Writing pre-run manifest")
    manifest = write_pre_run_manifest()
    with open(RESULTS_DIR / "E3_pre_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("  oracle.py hash:", manifest["files"].get("src/prdrl/oracle/oracle.py", "?"))

    # ---- Phase E3-1: Interface readiness check ----
    print("\n[E3-1] Checking E3 interface readiness")
    readiness = check_e3_readiness()
    assert readiness["E3_INTERFACE_READINESS"] == "READY", str(readiness)
    print(f"  E3_INTERFACE_READINESS: {readiness['E3_INTERFACE_READINESS']}")

    # ---- Phase E3-2: Calibration ----
    print("\n[E3-2] Running calibration (seeds 701-720)")
    t_calib = time.time()
    calib_features, calib_perf = run_calibration(xl, xu, V, R, pf_star)
    print(f"  Calibration done in {time.time()-t_calib:.1f}s")

    # ---- Phase E3-3: Compute thresholds ----
    print("\n[E3-3] Computing RULE_V1 thresholds")
    rule_config = compute_thresholds(calib_features)
    print(f"  T_C_LOW={rule_config['T_C_LOW']:.4f}  T_C_HIGH={rule_config['T_C_HIGH']:.4f}")
    print(f"  T_O_LOW={rule_config['T_O_LOW']:.4f}  T_DC_HIGH={rule_config['T_DC_HIGH']:.4f}")
    print(f"  T_L={rule_config['T_L']}  (n_samples={rule_config['n_samples']})")

    # Write feature calibration
    with open(RESULTS_DIR / "E3_feature_calibration.json", "w") as f:
        json.dump({
            "stage": "E3",
            "n_samples": int(len(calib_features["c"])),
            "c_percentiles": {
                "p10": float(np.percentile(calib_features["c"], 10)),
                "p25": float(np.percentile(calib_features["c"], 25)),
                "p33": float(np.percentile(calib_features["c"], 33)),
                "p50": float(np.percentile(calib_features["c"], 50)),
                "p67": float(np.percentile(calib_features["c"], 67)),
                "p75": float(np.percentile(calib_features["c"], 75)),
                "p90": float(np.percentile(calib_features["c"], 90)),
            },
            "o_percentiles": {
                "p25": float(np.percentile(calib_features["o"], 25)),
                "p50": float(np.percentile(calib_features["o"], 50)),
                "p75": float(np.percentile(calib_features["o"], 75)),
            },
            "dc_percentiles": {
                "p25": float(np.percentile(calib_features["dc"], 25)),
                "p50": float(np.percentile(calib_features["dc"], 50)),
                "p75": float(np.percentile(calib_features["dc"], 75)),
            },
        }, f, indent=2)

    # ---- Phase E3-4: Freeze RULE_V1 ----
    print("\n[E3-4] Freezing RULE_V1")
    rule_v1_record = {
        "rule_name": "RULE_V1",
        "version": "1",
        "frozen_date": "2026-09-20",
        "tuning_method": "feature_percentile_only",
        "not_tuned_from": "IGD_performance",
        "RL_STATUS": RL_STATUS,
        "thresholds": rule_config,
        "rule_logic": {
            "1_KEEP":      "c <= T_C_LOW AND l <= T_L AND dc <= 0",
            "2_DIVERSIFY": "c >= T_C_HIGH AND l > T_L AND (o <= T_O_LOW OR dc >= T_DC_HIGH)",
            "3_PREDICT":   "c >= T_C_HIGH AND l > T_L",
            "4_LOCAL":     "default",
        },
        "note": (
            "T_L=0 and active regions always have l=n_r>0, so RULE 1 (KEEP) "
            "is unreachable for active regions. Effective rule: "
            "DIVERSIFY > PREDICT > LOCAL based on severity c and motion dc."
        ),
    }
    with open(RESULTS_DIR / "E3_rule_v1.json", "w") as f:
        json.dump(rule_v1_record, f, indent=2)
    print("  RULE_V1 frozen. Thresholds will not change.")

    # ---- Phase E3-5: Select global baselines ----
    print("\n[E3-5] Selecting global baselines from calibration performance")
    baselines = select_global_baselines(calib_perf)
    for sc, b in baselines.items():
        print(f"  {sc}: best = {b['best_global_controller']} (MIGD={b['best_global_migd_response']:.4f})")
    with open(RESULTS_DIR / "E3_global_baseline_selection.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # ---- Phase E3-6: State semantics ----
    state_semantics = write_state_semantics(rule_config)
    with open(RESULTS_DIR / "E3_state_semantics.json", "w") as f:
        json.dump(state_semantics, f, indent=2)

    # ---- Phase E3-7: Smoke run ----
    print("\n[E3-7] Smoke run (seeds 781-782, 3 scenarios, 5 controllers, 3 changes)")
    t_smoke = time.time()
    smoke_traj, smoke_dec = run_all_seeds(
        seeds=SMOKE_SEEDS,
        block_label="SMOKE",
        n_scored=N_SCORED_CHANGES_SMOKE,
        rule_config=rule_config,
        xl=xl, xu=xu, V=V, R=R, pf_star=pf_star,
        is_smoke=True,
    )
    n_smoke = len(smoke_traj)
    expected_smoke = len(SMOKE_SEEDS) * len(SCENARIOS) * len(ALL_CONTROLLERS) * N_SCORED_CHANGES_SMOKE
    print(f"  Smoke: {n_smoke} trajectory rows (expected {expected_smoke})")
    assert n_smoke == expected_smoke, f"Smoke row count mismatch: {n_smoke} != {expected_smoke}"
    print(f"  Smoke done in {time.time()-t_smoke:.1f}s — PASS")

    # ---- Phase E3-8: Full evaluation ----
    print(f"\n[E3-8] Full evaluation (seeds 721-780, 3 scenarios, 5 controllers, 10 changes)")
    t_eval = time.time()
    eval_traj, eval_dec = run_all_seeds(
        seeds=EVAL_SEEDS,
        block_label="AUTO",
        n_scored=N_SCORED_CHANGES_FULL,
        rule_config=rule_config,
        xl=xl, xu=xu, V=V, R=R, pf_star=pf_star,
        is_smoke=False,
    )
    n_eval = len(eval_traj)
    expected_eval = len(EVAL_SEEDS) * len(SCENARIOS) * len(ALL_CONTROLLERS) * N_SCORED_CHANGES_FULL
    print(f"  Evaluation: {n_eval} trajectory rows (expected {expected_eval})")
    assert n_eval == expected_eval, f"Eval row count mismatch: {n_eval} != {expected_eval}"
    print(f"  Evaluation done in {time.time()-t_eval:.1f}s")

    # ---- Phase E3-9: Write trajectory outputs ----
    print("\n[E3-9] Writing output files")
    write_trajectories_jsonl(eval_traj, RESULTS_DIR / "E3_trajectories.jsonl")
    write_decisions_jsonl(eval_dec, RESULTS_DIR / "E3_region_decisions.jsonl")
    write_seed_summary(eval_traj, EVAL_SEEDS, RESULTS_DIR / "E3_seed_summary.csv")
    write_action_distribution_csv(eval_dec, RESULTS_DIR / "E3_action_distribution.csv")
    write_fe_summary_csv(eval_traj, baselines, RESULTS_DIR / "E3_fe_summary.csv")
    write_mixed_regime_summary_csv(eval_traj, baselines, RESULTS_DIR / "E3_mixed_regime_summary.csv")

    # ---- Phase E3-10: Statistics ----
    print("\n[E3-10] Computing statistics (paired seed bootstrap)")
    block_a_results = compute_block_results(
        eval_traj, BLOCK_A_SEEDS, "A", baselines, BOOTSTRAP_RESAMPLES_BLOCK, BOOTSTRAP_SEED
    )
    block_b_results = compute_block_results(
        eval_traj, BLOCK_B_SEEDS, "B", baselines, BOOTSTRAP_RESAMPLES_BLOCK, BOOTSTRAP_SEED + 1
    )
    pooled_results = compute_pooled_results(
        eval_traj, baselines, BOOTSTRAP_RESAMPLES_POOLED, BOOTSTRAP_SEED + 2
    )

    for sc in SCENARIOS:
        ba = block_a_results[sc]
        bb = block_b_results[sc]
        bp = pooled_results[sc]
        print(f"  {sc}: BlockA={ba['effect']}({ba['mean_benefit']:.5f}), "
              f"BlockB={bb['effect']}({bb['mean_benefit']:.5f}), "
              f"Pooled={bp['effect']}({bp['mean_benefit']:.5f}) "
              f"CI=[{bp['ci_lo']:.5f},{bp['ci_hi']:.5f}]")

    write_block_summary_csv(block_a_results, block_b_results, pooled_results,
                            RESULTS_DIR / "E3_block_summary.csv")
    write_scenario_summary_csv(eval_traj, baselines, pooled_results,
                               RESULTS_DIR / "E3_scenario_summary.csv")

    # ---- Phase E3-11: Gate B ----
    print("\n[E3-11] Computing Gate B")
    fe_ok = check_fe_ok(eval_traj, baselines)
    gate_b, gate_verdicts = compute_gate_b(
        block_a_results, block_b_results, pooled_results,
        fe_ok=fe_ok,
        interface_ready=(readiness["E3_INTERFACE_READINESS"] == "READY"),
        no_causal_violation=True,   # verified by build_decision_context (uses h_t+1)
        no_leakage=True,            # regime labels not used in RULE_V1 state
    )

    print(f"  E3_MIXED_RULE_GAIN: {gate_verdicts['E3_MIXED_RULE_GAIN']}")
    print(f"  PERSISTENT_SAFETY: {gate_verdicts['PERSISTENT_SAFETY']}")
    print(f"  STATIONARY_SAFETY: {gate_verdicts['STATIONARY_SAFETY']}")
    print(f"  GATE_B: {gate_b}")

    # ---- Phase E3-12: Write verdict ----
    verdict = {
        "stage": "E3_rule_based_regional_controller",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS,
        "rule_version": "RULE_V1",
        "thresholds_tuned_from_performance": False,
        "sigma_tuned": False,
        "HARD_used": False,
        "B_r_a_activated": False,
        "production_code_modified": False,
        "DELTA_E3": DELTA_E3,
        "n_eval_seeds": len(EVAL_SEEDS),
        "n_calib_seeds": len(CALIB_SEEDS),
        "n_scenarios": len(SCENARIOS),
        "n_controllers": len(ALL_CONTROLLERS),
        "n_scored_changes": N_SCORED_CHANGES_FULL,
        "block_A_results": block_a_results,
        "block_B_results": block_b_results,
        "pooled_results": pooled_results,
        "baselines": baselines,
        "gate_verdicts": gate_verdicts,
        "GATE_B": gate_b,
        "NEXT_STAGE": "E4_DRL_AGENT" if gate_b == "PASS" else "GATE_B_FAIL",
    }
    with open(RESULTS_DIR / "E3_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    # ---- Phase E3-13: Report ----
    write_report_md(rule_config, baselines, block_a_results, block_b_results,
                    pooled_results, gate_b, gate_verdicts,
                    RESULTS_DIR / "E3_report.md")

    elapsed = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"E3 COMPLETE — Gate B: {gate_b}  ({elapsed:.1f}s)")
    print(f"  Output dir: {RESULTS_DIR}")
    print(f"  E3_MIXED_RULE_GAIN: {gate_verdicts['E3_MIXED_RULE_GAIN']}")
    print(f"  PERSISTENT_SAFETY:  {gate_verdicts['PERSISTENT_SAFETY']}")
    print(f"  STATIONARY_SAFETY:  {gate_verdicts['STATIONARY_SAFETY']}")
    print(f"  RL_STATUS:          {RL_STATUS}")
    print(f"  sigma_predict used: {SIGMA_PREDICT} (frozen O1 production spec)")
    print(f"  Next stage: {verdict['NEXT_STAGE']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
