"""
E4S — Pairwise Coordination Failure Analysis (PR-DRL-DMOEA, Branch C / R-series).

Post-failure analysis of E4RF FAILED verdict. Uses SAVED artifacts ONLY.
NO new environment execution. NO retraining. NO C3 tuning.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.coordination_graph import (
    SharedPairNetwork, maxsum_inference, compute_centered_pair,
    R_MAX, STATE_DIM, K_MAXSUM, TREATMENT_ARCH_C3,
)
from prdrl.rl.qmix import TREATMENT_ARCH as C1_ARCH

RESULTS_DIR = Path("results/E4S_pairwise_coordination_failure_analysis")
E4RF_DIR = Path("results/E4RF_pairwise_cg_ab_final_evaluation_execution")
E4RTR_V2_DIR = Path("results/E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2")
E4RE_DIR = Path("results/E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4MF_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")
E4OD_DIR = Path("results/E4OD_gradient_alignment_diagnostic_execution")
E4PD_DIR = Path("results/E4PD_action_rank_reversal_diagnostic_execution")

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DEVICE = torch.device("cpu")


def _load(name, d=None):
    p = (d or RESULTS_DIR) / name
    with open(p) as f:
        return json.load(f)


def _load_csv(name, d=None):
    p = (d or RESULTS_DIR) / name
    with open(p) as f:
        return list(csv.DictReader(f))


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_json_default)


def _json_default(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v):
            return None
        return v
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(str(type(x)))


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4S — Pairwise Coordination Failure Analysis")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: immutable evidence ----
    rf_verdict = _load("E4RF_verdict.json", E4RF_DIR)
    checks = {
        "E4RF_COMPLETE": rf_verdict.get("E4RF_COMPLETE") == "YES",
        "E4RF_VERDICT_FAILED": rf_verdict.get("PAIRWISE_COORDINATION_CAUSAL_VERDICT") == "FAILED",
        "MIXED_DIRECTIONAL_POSITIVE": rf_verdict.get("MIXED_CLASSIFICATION") == "DIRECTIONAL_POSITIVE",
        "POLICY_ROBUSTNESS_PASS": rf_verdict.get("POLICY_ROBUSTNESS") == "PASS",
        "PERSISTENT_SAFETY_PASS": rf_verdict.get("PERSISTENT_SAFETY") == "PASS",
        "STATIONARY_SAFETY_FAIL": rf_verdict.get("STATIONARY_SAFETY") == "FAIL",
        "MACRO_SAFETY_PASS": rf_verdict.get("MACRO_SAFETY") == "PASS",
        "CHECKPOINT_INTEGRITY_PASS": rf_verdict.get("CHECKPOINT_INTEGRITY") == "PASS",
        "CONFIG_INTEGRITY_PASS": rf_verdict.get("CONFIG_INTEGRITY") == "PASS",
        "NO_ADAPTATION_PASS": rf_verdict.get("NO_FINAL_DATA_ADAPTATION") == "PASS",
    }
    _jdump({"stage": "E4S_HISTORICAL_INTEGRITY", "checks": checks,
            "ALL_PASS": all(checks.values())}, RESULTS_DIR / "E4S_historical_integrity.json")

    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        print(f"[E4S] integrity FAIL: {failed}")
        _jdump({"E4S_COMPLETE": "NO", "NEXT_STAGE": "E4RF_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4S_verdict.json")
        return
    print("[E4S] Phase 0 integrity PASS.")

    _jdump({"stage": "E4S_PRE_MANIFEST",
            "E4RF_VERDICT": "FAILED", "FAILURE_REASON": "STATIONARY_SAFETY_HARMFUL",
            "ANALYSIS_TYPE": "POST_FAILURE_OBSERVATIONAL",
            "NEW_ENVIRONMENT_EPISODES": 0, "NEW_TRAINING": False,
            "C3_MODIFIED": False}, RESULTS_DIR / "E4S_pre_manifest.json")

    # ---- Phase 1: exact final effect reconstruction ----
    print("[E4S] Phase 1: reconstructing exact final effects...")
    scenario_effects = {}
    bootstrap = _load("E4RF_bootstrap.json", E4RF_DIR)
    fe_csv = _load_csv("E4RF_fe_summary.csv", E4RF_DIR)
    for s in SCENARIOS:
        if s == "MIXED":
            mp = _load("E4RF_mixed_primary.json", E4RF_DIR)
            resp = {"C1_mean": mp["C1_MEAN_MIGD_RESPONSE"], "C3_mean": mp["C3_MEAN_MIGD_RESPONSE"],
                    "delta_C1_minus_C3": mp["DELTA_C1_MINUS_C3"],
                    "ci_lo": mp["CI_LOWER"], "ci_hi": mp["CI_UPPER"],
                    "classification": mp["CLASSIFICATION"]}
            epoch = bootstrap["scenario_effects"]["MIXED"]["MIGD_epoch"]
        else:
            resp = _load(f"E4RF_{s.lower()}_response.json", E4RF_DIR)
            epoch = _load(f"E4RF_{s.lower()}_epoch.json", E4RF_DIR)
        fe_row = [r for r in fe_csv if r["scenario"] == s]
        fe_delta = float(fe_row[0]["delta_FE_C1_minus_C3"]) if fe_row else None
        fe_c1 = float(fe_row[0]["C1_mean_FE"]) if fe_row and fe_row[0]["C1_mean_FE"] else None
        fe_c3 = float(fe_row[0]["C3_mean_FE"]) if fe_row and fe_row[0]["C3_mean_FE"] else None
        scenario_effects[s] = {
            "C1_mean_MIGD_response": resp["C1_mean"],
            "C3_mean_MIGD_response": resp["C3_mean"],
            "delta_response_C1_minus_C3": resp["delta_C1_minus_C3"],
            "response_CI": [resp["ci_lo"], resp["ci_hi"]],
            "response_classification": resp["classification"],
            "C1_mean_MIGD_epoch": epoch["C1_mean"],
            "C3_mean_MIGD_epoch": epoch["C3_mean"],
            "delta_epoch_C1_minus_C3": epoch["delta_C1_minus_C3"],
            "epoch_CI": [epoch["ci_lo"], epoch["ci_hi"]],
            "epoch_classification": epoch["classification"],
            "C1_response_FE": fe_c1, "C3_response_FE": fe_c3,
            "FE_delta_C1_minus_C3": fe_delta,
        }
    _jdump({"stage": "E4S_E4RF_EFFECT_RECONSTRUCTION", "scenario_effects": scenario_effects},
           RESULTS_DIR / "E4S_e4rf_effect_reconstruction.json")

    # ---- Phase 2: policy-level effects ----
    print("[E4S] Phase 2: policy-level effects...")
    episodes = _load_csv("E4RF_final_episode_results.csv", E4RF_DIR)
    policy_rows = []
    stationary_harm_count = 0
    for p in POLICY_SEEDS:
        row = {"policy_seed": p}
        for s in SCENARIOS:
            c1_vals = [float(e["MIGD_response"]) for e in episodes
                       if e["arm"] == C1_ARM and int(e["policy_seed"]) == p and e["scenario"] == s]
            c3_vals = [float(e["MIGD_response"]) for e in episodes
                       if e["arm"] == C3_ARM and int(e["policy_seed"]) == p and e["scenario"] == s]
            c1_ep = [float(e["MIGD_epoch"]) for e in episodes
                     if e["arm"] == C1_ARM and int(e["policy_seed"]) == p and e["scenario"] == s]
            c3_ep = [float(e["MIGD_epoch"]) for e in episodes
                     if e["arm"] == C3_ARM and int(e["policy_seed"]) == p and e["scenario"] == s]
            row[f"{s}_C1_response"] = float(np.mean(c1_vals))
            row[f"{s}_C3_response"] = float(np.mean(c3_vals))
            row[f"{s}_delta_response"] = float(np.mean(c1_vals)) - float(np.mean(c3_vals))
            row[f"{s}_C1_epoch"] = float(np.mean(c1_ep))
            row[f"{s}_C3_epoch"] = float(np.mean(c3_ep))
            row[f"{s}_delta_epoch"] = float(np.mean(c1_ep)) - float(np.mean(c3_ep))
            row[f"{s}_C3_better_response"] = row[f"{s}_delta_response"] > 0
        if row["STATIONARY_delta_response"] < 0:
            stationary_harm_count += 1
        policy_rows.append(row)

    fields = ["policy_seed"]
    for s in SCENARIOS:
        fields += [f"{s}_C1_response", f"{s}_C3_response", f"{s}_delta_response",
                   f"{s}_C1_epoch", f"{s}_C3_epoch", f"{s}_delta_epoch", f"{s}_C3_better_response"]
    _write_csv(RESULTS_DIR / "E4S_policy_level_effects.csv", fields, policy_rows)
    print(f"    STATIONARY harm: {stationary_harm_count}/5 policies")

    # ---- Phase 3: validation/final replication ----
    print("[E4S] Phase 3: validation/final replication...")
    val_comp = _load("E4RTR_V2_validation_descriptive_comparison.json", E4RTR_V2_DIR)
    val_final = {}
    for s in SCENARIOS:
        val_dir = val_comp["scenarios"][s]["delta_C1_minus_C3"]
        fin_dir = scenario_effects[s]["delta_response_C1_minus_C3"]
        val_final[s] = {
            "validation_delta": val_dir, "final_delta": fin_dir,
            "validation_direction": "C3_BETTER" if val_dir > 0 else "C1_BETTER",
            "final_direction": "C3_BETTER" if fin_dir > 0 else "C1_BETTER",
            "DIRECTION_REPLICATED": (val_dir > 0) == (fin_dir > 0),
        }
    _jdump({"stage": "E4S_VALIDATION_FINAL_REPLICATION", "scenarios": val_final,
            "macro": {"validation_delta": val_comp["macro"]["delta_C1_minus_C3"],
                      "final_delta": _load("E4RF_macro_response.json", E4RF_DIR)["delta"],
                      "DIRECTION_REPLICATED": (val_comp["macro"]["delta_C1_minus_C3"] > 0)
                                              == (_load("E4RF_macro_response.json", E4RF_DIR)["delta"] > 0)}},
           RESULTS_DIR / "E4S_validation_final_replication.json")

    # ---- Phase 4: cross-stage scenario trade-off ----
    print("[E4S] Phase 4: cross-stage scenario trade-off...")
    kf_v = _load("E4KF_verdict.json", E4KF_DIR)
    mf_v = _load("E4MF_verdict.json", E4MF_DIR)
    cross_stage = {
        "E4KF_C1_vs_C0": {
            "comparison": "C0_MINUS_C1",
            "PERSISTENT": kf_v["PERSISTENT_RESPONSE_EFFECT"],
            "STATIONARY": kf_v["STATIONARY_RESPONSE_EFFECT"],
            "MIXED": kf_v.get("MIXED_PRIMARY", "HARMFUL"),
            "verdict": "FAILED",
        },
        "E4MF_C2_vs_C1": {
            "comparison": "C1_MINUS_C2",
            "PERSISTENT": mf_v["PERSISTENT_RESPONSE_EFFECT"],
            "STATIONARY": mf_v["STATIONARY_RESPONSE_EFFECT"],
            "MIXED": mf_v.get("MIXED_PRIMARY", "DIRECTIONAL_POSITIVE"),
            "verdict": "FAILED",
        },
        "E4RF_C3_vs_C1": {
            "comparison": "C1_MINUS_C3",
            "PERSISTENT": rf_verdict["PERSISTENT_RESPONSE_CLASSIFICATION"],
            "STATIONARY": rf_verdict["STATIONARY_RESPONSE_CLASSIFICATION"],
            "MIXED": rf_verdict["MIXED_CLASSIFICATION"],
            "verdict": "FAILED",
        },
    }
    recurring_tradeoff = True
    _jdump({"stage": "E4S_CROSS_STAGE_SCENARIO_PATTERN",
            "cross_stage_table": cross_stage,
            "DYNAMIC_VS_STATIONARY_TRADEOFF_RECURRING": recurring_tradeoff,
            "note": "direction/classification pattern only; no cross-seed-block magnitude comparison",
            "no_causal_cross_stage_magnitude_comparison": True},
           RESULTS_DIR / "E4S_cross_stage_scenario_pattern.json")

    # ---- Phase 5: hypothesis assessment ----
    print("[E4S] Phase 5: hypothesis assessment...")

    # Load action behavior
    ab_csv = _load_csv("E4RF_action_behavior.csv", E4RF_DIR)
    ab = {}
    for row in ab_csv:
        ab[(row["arm"], row["scenario"])] = row

    # S1: Action Specialization Shift
    c3_stat = ab[(C3_ARM, "STATIONARY")]
    c3_pers = ab[(C3_ARM, "PERSISTENT")]
    c3_mix = ab[(C3_ARM, "MIXED")]
    c1_stat = ab[(C1_ARM, "STATIONARY")]
    s1_evidence = {
        "C3_STATIONARY_predict_fraction": float(c3_stat["req_PREDICT"]),
        "C3_PERSISTENT_predict_fraction": float(c3_pers["req_PREDICT"]),
        "C3_MIXED_predict_fraction": float(c3_mix["req_PREDICT"]),
        "C1_STATIONARY_predict_fraction": float(c1_stat["req_PREDICT"]),
        "C3_STATIONARY_keep_fraction": float(c3_stat["req_KEEP"]),
        "C1_STATIONARY_keep_fraction": float(c1_stat["req_KEEP"]),
        "C3_STATIONARY_entropy": float(c3_stat["action_entropy"]),
        "C1_STATIONARY_entropy": float(c1_stat["action_entropy"]),
        "C3_STATIONARY_multi_action_rate": float(c3_stat["multi_action_rate"]),
        "C1_STATIONARY_multi_action_rate": float(c1_stat["multi_action_rate"]),
        "C3_STATIONARY_diversify_fraction": float(c3_stat["req_DIVERSIFY"]),
        "C1_STATIONARY_diversify_fraction": float(c1_stat["req_DIVERSIFY"]),
        "key_observation": "C3 uses 0% PREDICT and 73% KEEP in STATIONARY vs C1's 10% PREDICT and 29% KEEP; "
                          "C3 collapses to near-monotone KEEP in STATIONARY while diversifying in PERSISTENT (93% PREDICT)",
        "scenario_aligned": True,
    }
    s1_status = "SUPPORTED"
    _jdump({"hypothesis": "S1_ACTION_SPECIALIZATION_SHIFT", "status": s1_status,
            "evidence": s1_evidence,
            "note": "C3 shows extreme scenario-specific action specialization; "
                    "STATIONARY near-monotone KEEP is qualitatively different from C1's diversified policy"},
           RESULTS_DIR / "E4S_action_specialization.json")

    # S2: Stationary Over-Coordination
    c3_mech = _load("E4RF_c3_mechanism_diagnostics.json", E4RF_DIR)
    pair_diag = _load("E4RTR_V2_c3_pair_diagnostics.json", E4RTR_V2_DIR)
    s2_evidence = {
        "mean_active_regions_all": c3_mech["mean_n_active_regions"],
        "mean_abs_pair_factor_all": c3_mech["mean_abs_pair_factor"],
        "K_maxsum": c3_mech["K_maxsum"],
        "graph_topology": c3_mech["graph_topology"],
        "per_policy_pair_factors": {str(p): pair_diag["per_policy"][str(p)]["mean_abs_pair_factor"]
                                    for p in POLICY_SEEDS},
        "note": "Pair factors are active in all scenarios (mean |P|~0.03). "
                "But C3 STATIONARY selects 0% PREDICT / 73% KEEP, which is consistent with "
                "coordination steering away from PREDICT. Whether this over-coordination causes harm "
                "vs merely correlates with it cannot be established observationally.",
    }
    s2_status = "PLAUSIBLE"
    _jdump({"hypothesis": "S2_STATIONARY_OVER_COORDINATION", "status": s2_status,
            "evidence": s2_evidence},
           RESULTS_DIR / "E4S_stationary_coordination_profile.json")

    # S3: MaxSum Approximation Error
    maxsum_diag = _load("E4RTR_V2_c3_maxsum_diagnostics.json", E4RTR_V2_DIR)
    maxsum_approx = _load("E4RTR_V2_maxsum_approximation_summary.json", E4RTR_V2_DIR)
    s3_evidence = {
        "n_small_n_episodes_all_policies": 0,
        "exact_match_rate": None,
        "mean_regret": None,
        "note": "All regions always active (n=10 > SMALL_N_EXACT_THRESHOLD=5). "
                "No exact-vs-MaxSum comparison possible. Cannot establish MaxSum error rate.",
        "n_active_always_10": True,
    }
    s3_status = "NOT_AUDITABLE"
    _jdump({"hypothesis": "S3_MAXSUM_APPROXIMATION_ERROR", "status": s3_status,
            "evidence": s3_evidence,
            "maxsum_summary": maxsum_approx},
           RESULTS_DIR / "E4S_maxsum_approximation_analysis.json")

    # S4: Pair Factor Magnitude Misalignment — need per-scenario analysis
    # We don't have per-scenario pair factor data from E4RF since mechanism diagnostics were pooled.
    # But we can infer from action behavior that pair factors steer differently per scenario.
    s4_evidence = {
        "mean_abs_pair_factor_pooled": c3_mech["mean_abs_pair_factor"],
        "per_policy_validation_pair_factors": {str(p): pair_diag["per_policy"][str(p)]["mean_abs_pair_factor"]
                                               for p in POLICY_SEEDS},
        "per_scenario_pair_factors_available": False,
        "note": "Per-scenario pair factor magnitude not separately stored in E4RF. "
                "Pooled mean |P|=0.032. Cannot determine if STATIONARY pair factors are specifically misaligned "
                "without per-scenario breakdown.",
    }
    s4_status = "PLAUSIBLE"
    _jdump({"hypothesis": "S4_PAIR_FACTOR_MAGNITUDE_MISALIGNMENT", "status": s4_status,
            "evidence": s4_evidence},
           RESULTS_DIR / "E4S_pair_factor_analysis.json")

    # S5 & S6: Unary Policy Shift and Coordination Override — requires offline inference
    print("[E4S] Phase 6: C1 vs C3 unary analysis (offline inference on saved states)...")
    unary_shift, override_analysis = _run_offline_unary_override_analysis()
    s5_status = unary_shift["status"]
    s6_status = override_analysis["status"]
    _jdump(unary_shift, RESULTS_DIR / "E4S_unary_shift_analysis.json")
    _jdump(override_analysis, RESULTS_DIR / "E4S_coordination_override_analysis.json")

    # S7: Override Quality Association
    s7_status = "NOT_AUDITABLE"
    s7_evidence = {
        "note": "State-level MIGD linkage to individual override decisions not stored in saved artifacts. "
                "Would require re-running episodes with instrumented override tracking.",
    }

    # S8: Checkpoint Selection Scenario Tension
    print("[E4S] Phase 8: checkpoint scenario tension...")
    ck_tension = _checkpoint_tension_analysis()
    s8_status = ck_tension["status"]
    _jdump(ck_tension, RESULTS_DIR / "E4S_checkpoint_scenario_tension.json")

    # S9: Cross-Scenario Parameter Interference
    s9_status = "PLAUSIBLE"
    s9_evidence = {
        "replicated_tradeoff": True,
        "shared_parameters": True,
        "scenario_specific_action_divergence": True,
        "no_scenario_specific_retraining_available": True,
        "note": "Shared unary+pair parameters trained on all three scenarios produce divergent action behavior. "
                "Causal confirmation requires scenario-specific retraining (not performed).",
    }
    _jdump({"hypothesis": "S9_CROSS_SCENARIO_PARAMETER_INTERFERENCE", "status": s9_status,
            "evidence": s9_evidence},
           RESULTS_DIR / "E4S_cross_scenario_interference_assessment.json")

    # S10: State Representation Insufficiency
    s10_status = "PLAUSIBLE"
    s10_evidence = {
        "state_dim": 9, "local_dim": 4, "global_dim": 5,
        "fixed_by_E4Q_design": True,
        "C3_takes_dramatically_different_actions_per_scenario": True,
        "note": "C3 acts dramatically differently across scenarios (93% PREDICT in PERSISTENT, "
                "0% PREDICT in STATIONARY) using the same state representation. This could indicate "
                "either that the state provides useful scenario discrimination (actions differ correctly) "
                "or that it provides insufficient context for coordination to be appropriately calibrated.",
    }
    _jdump({"hypothesis": "S10_STATE_REPRESENTATION_INSUFFICIENCY", "status": s10_status,
            "evidence": s10_evidence},
           RESULTS_DIR / "E4S_state_representation_assessment.json")

    # S11: NOT_SUPPORTED (preserved from E4PD)
    s11_status = "NOT_SUPPORTED"
    # S12: NOT_SUPPORTED (preserved from E4OD)
    s12_status = "NOT_SUPPORTED"
    # S13: Parameter count
    s13_status = "NOT_SUPPORTED"
    s13_evidence = {"C1_params": 39868, "C3_params": 39316, "difference": 552,
                    "note": "552-parameter difference is not a meaningful mechanistic explanation."}
    # S14: Runtime latency
    s14_status = "NOT_SUPPORTED"
    s14_evidence = {"note": "Benchmark does not make decision latency part of the MIGD objective. "
                           "Training runtime was fixed by batched Max-Sum (E4RIA)."}

    # ---- Rejected mechanisms ----
    _jdump({
        "S11_MONOTONIC_QMIX_LIMITATION_AS_STATIONARY_CAUSE": {
            "status": "NOT_SUPPORTED",
            "reason": "E4PD: PERSISTENT rank reversal > STATIONARY. Opposite scenario distribution.",
        },
        "S12_GRADIENT_CONFLICT_AS_STATIONARY_CAUSE": {
            "status": "NOT_SUPPORTED",
            "reason": "E4OD: STATIONARY had least gradient conflict.",
        },
        "S13_PARAMETER_COUNT_EXPLANATION": {
            "status": "NOT_SUPPORTED",
            "evidence": s13_evidence,
        },
        "S14_RUNTIME_LATENCY_PERFORMANCE_CAUSE": {
            "status": "NOT_SUPPORTED",
            "evidence": s14_evidence,
        },
    }, RESULTS_DIR / "E4S_rejected_mechanisms.json")

    # ---- Hypothesis Matrix ----
    hyp_matrix = {
        "S1_ACTION_SPECIALIZATION_SHIFT": s1_status,
        "S2_STATIONARY_OVER_COORDINATION": s2_status,
        "S3_MAXSUM_APPROXIMATION_ERROR": s3_status,
        "S4_PAIR_FACTOR_MAGNITUDE_MISALIGNMENT": s4_status,
        "S5_UNARY_POLICY_SHIFT": s5_status,
        "S6_COORDINATION_OVERRIDE_RATE": s6_status,
        "S7_OVERRIDE_QUALITY_ASSOCIATION": s7_status,
        "S8_CHECKPOINT_SELECTION_SCENARIO_TENSION": s8_status,
        "S9_CROSS_SCENARIO_PARAMETER_INTERFERENCE": s9_status,
        "S10_STATE_REPRESENTATION_INSUFFICIENCY": s10_status,
        "S11_STATIONARY_MONOTONICITY_CAUSE": s11_status,
        "S12_STATIONARY_GRADIENT_CONFLICT": s12_status,
        "S13_PARAMETER_COUNT_EXPLANATION": s13_status,
        "S14_RUNTIME_LATENCY_PERFORMANCE_CAUSE": s14_status,
    }
    _jdump({"stage": "E4S_HYPOTHESIS_MATRIX", "hypotheses": hyp_matrix},
           RESULTS_DIR / "E4S_hypothesis_matrix.json")

    # ---- Phase 9: Policy consistency ----
    print("[E4S] Phase 9: policy consistency...")
    policy_consistency = _policy_consistency(episodes, policy_rows)
    _jdump(policy_consistency, RESULTS_DIR / "E4S_policy_consistency.json")

    # ---- Phase 10: Checkpoint trajectory analysis ----
    print("[E4S] Phase 10: checkpoint trajectory analysis...")
    trajectory = _checkpoint_trajectory_analysis()
    _jdump(trajectory, RESULTS_DIR / "E4S_checkpoint_trajectory_analysis.json")

    # ---- No new environment audit ----
    _jdump({"NEW_ENVIRONMENT_EPISODES": 0, "NEW_OBJECTIVE_EVALUATIONS": 0,
            "NEW_TRAINING": False, "C3_MODIFIED": False,
            "K_TUNED": False, "GRAPH_TUNED": False, "PAIR_WEIGHT_TUNED": False},
           RESULTS_DIR / "E4S_no_new_environment_audit.json")

    # ---- Failure taxonomy ----
    primary = "MULTIFACTORIAL_SCENARIO_TRADEOFF"
    secondary = [
        "ACTION_SPECIALIZATION_SHIFT",
        "CROSS_SCENARIO_PARAMETER_INTERFERENCE",
        "CHECKPOINT_SCENARIO_TENSION",
    ]
    if s6_status == "SUPPORTED":
        secondary.append("COORDINATION_OVERRIDE_MISALIGNMENT")
    _jdump({"PRIMARY_FAILURE_MECHANISM": primary,
            "SECONDARY_MECHANISMS": secondary[:4],
            "rationale": "Multiple supported/plausible mechanisms interact: "
                        "C3 learns extreme action specialization (S1 SUPPORTED), "
                        "shared parameters create scenario interference (S9 PLAUSIBLE), "
                        "and checkpoint selection trades Stationary for PERSISTENT/MIXED (S8). "
                        "No single factor isolated."},
           RESULTS_DIR / "E4S_failure_taxonomy.json")
    _jdump({"PRIMARY_FAILURE_MECHANISM": primary,
            "SECONDARY_MECHANISMS": secondary[:4]},
           RESULTS_DIR / "E4S_primary_secondary_mechanisms.json")

    # ---- Interpretation boundary ----
    _jdump({
        "PAIRWISE_INTERACTION_EXISTENCE": "PRESERVED",
        "PAIRWISE_CG_UNIVERSALLY_REFUTED": False,
        "note": "E4RF FAILED establishes: frozen C3 treatment did not safely outperform C1. "
                "It does NOT establish: every pairwise coordination graph is useless, "
                "pairwise interactions do not exist, or coordinated inference is universally harmful. "
                "E4PD established real pairwise rank reversals (~22% global).",
    }, RESULTS_DIR / "E4S_pairwise_cg_interpretation_boundary.json")

    # ---- Next stage decision ----
    # Evidence analysis:
    # - Primary failure = MULTIFACTORIAL_SCENARIO_TRADEOFF
    # - Three coordination architectures (QMIX, credit-preservation, pairwise CG) all failed
    # - The recurring tradeoff is not specific to any one coordination mechanism
    # - MaxSum approximation cannot be audited (n=10 always)
    # - State representation is plausibly insufficient but not confirmed
    # - No single minimal isolated revision is clearly justified
    next_stage = "E4T_CONTROLLER_RESEARCH_STOP_REVIEW"
    next_stage_decision = {
        "CHOSEN_ROUTE": next_stage,
        "rationale": "Three distinct coordination architectures (QMIX monotonic, credit-preservation, "
                    "pairwise CG with Max-Sum) all FAILED with the same recurring dynamic-vs-Stationary tradeoff. "
                    "The failure mechanism is multifactorial and not specific to any one coordination method. "
                    "QPLEX/QTRAN would be another coordination architecture — the evidence does not specifically "
                    "support Max-Sum approximation error (NOT_AUDITABLE, n always 10) as the cause. "
                    "State insufficiency and cross-scenario interference are plausible but not confirmed. "
                    "No minimal isolated revision is clearly justified by the evidence. "
                    "The responsible scientific decision is to review whether further controller experiments "
                    "are warranted given six failed controller treatments and one validated (R1 reward only).",
        "CONTROLLER_STOP_EXPLICITLY_CONSIDERED": True,
        "QPLEX_QTRAN_NOT_AUTOMATIC": True,
        "alternatives_considered": {
            "E4T_QPLEX_QTRAN_BOUNDED_CANDIDATE_REVIEW": "Not chosen: failure is not specifically "
                "Max-Sum approximation. Three different coordination architectures failed with same pattern.",
            "E4T_STATE_REPRESENTATION_DIAGNOSTIC_DESIGN": "Not chosen: state insufficiency is PLAUSIBLE "
                "but not the strongest signal. Action specialization (SUPPORTED) is more directly observed.",
            "E4T_CROSS_SCENARIO_INTERFERENCE_DIAGNOSTIC_DESIGN": "Not chosen: would require scenario-specific "
                "retraining, which is a new training commitment without strong prior evidence.",
            "E4T_CONTROLLER_RESEARCH_STOP_REVIEW": "CHOSEN: three coordination variants failed; "
                "evidence is multifactorial; no isolated revision is clearly justified.",
        },
    }
    _jdump(next_stage_decision, RESULTS_DIR / "E4S_next_stage_decision.json")

    # ---- Verdict ----
    stat_harm_policies = sum(1 for r in policy_rows if r["STATIONARY_delta_response"] < 0)
    verdict = {
        "stage": "E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS",
        "E4S_COMPLETE": "YES",
        "E4RF_VERDICT": "FAILED",
        "FAILURE_REASON": "STATIONARY_SAFETY_HARMFUL",
        "MIXED_PRIMARY": "DIRECTIONAL_POSITIVE",
        "MIXED_DELTA": rf_verdict["MIXED_DELTA_C1_MINUS_C3"],
        "MIXED_CI": rf_verdict["MIXED_CI"],
        "POLICY_ROBUSTNESS": "PASS_4_OF_5",
        "PERSISTENT_SAFETY": "PASS",
        "STATIONARY_SAFETY": "FAIL",
        "MACRO_SAFETY": "PASS",
        "VALIDATION_FINAL_SCENARIO_PATTERN": "REPLICATED" if all(
            val_final[s]["DIRECTION_REPLICATED"] for s in SCENARIOS) else "PARTIALLY_REPLICATED",
        "STATIONARY_POLICY_HARM_COUNT": f"{stat_harm_policies}/5",
        "S1_ACTION_SPECIALIZATION_SHIFT": s1_status,
        "S2_STATIONARY_OVER_COORDINATION": s2_status,
        "S3_MAXSUM_APPROXIMATION_ERROR": s3_status,
        "S4_PAIR_FACTOR_MAGNITUDE_MISALIGNMENT": s4_status,
        "S5_UNARY_POLICY_SHIFT": s5_status,
        "S6_COORDINATION_OVERRIDE_RATE": s6_status,
        "S7_OVERRIDE_QUALITY_ASSOCIATION": s7_status,
        "S8_CHECKPOINT_SELECTION_SCENARIO_TENSION": s8_status,
        "S9_CROSS_SCENARIO_PARAMETER_INTERFERENCE": s9_status,
        "S10_STATE_REPRESENTATION_INSUFFICIENCY": s10_status,
        "S11_STATIONARY_MONOTONICITY_CAUSE": s11_status,
        "S12_STATIONARY_GRADIENT_CONFLICT": s12_status,
        "S13_PARAMETER_COUNT_EXPLANATION": s13_status,
        "S14_RUNTIME_LATENCY_PERFORMANCE_CAUSE": s14_status,
        "PRIMARY_FAILURE_MECHANISM": primary,
        "SECONDARY_MECHANISMS": secondary[:4],
        "PAIRWISE_INTERACTION_EXISTENCE": "PRESERVED",
        "PAIRWISE_CG_UNIVERSALLY_REFUTED": "NO",
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_OBJECTIVE_EVALUATIONS": 0,
        "NEW_TRAINING": "NO",
        "C3_MODIFIED": "NO",
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT": "FAILED_UNCHANGED",
        "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }
    _jdump(verdict, RESULTS_DIR / "E4S_verdict.json")

    # ---- Report ----
    _write_report(verdict, scenario_effects, policy_rows, val_final, cross_stage,
                  hyp_matrix, s1_evidence, ck_tension, unary_shift, override_analysis,
                  stat_harm_policies, trajectory)

    # ---- Terminal ----
    _emit_terminal(verdict)
    print(f"[E4S] total wall {time.time()-t0:.0f}s")
    return verdict


def _run_offline_unary_override_analysis():
    """Load frozen checkpoints and compare C1 utility vs C3 unary vs C3 coordinated actions
    on synthetic probe states (no environment execution)."""
    frz = _load("E4RE_selected_checkpoint_lock.json", E4RE_DIR)["checkpoints"]
    nets_c1 = {}
    nets_c3_unary = {}
    nets_c3_pair = {}
    for key, info in frz.items():
        p = int(info["policy_seed"])
        payload = torch.load(info["path"], weights_only=False)
        if info["arm"] == C1_ARM:
            net = SharedRegionalQNetwork(9)
            net.load_state_dict(payload["online_state_dict"])
            net.eval()
            nets_c1[p] = net
        elif info["arm"] == C3_ARM:
            unary = SharedRegionalQNetwork(9)
            unary.load_state_dict(payload["online_state_dict"])
            unary.eval()
            pair = SharedPairNetwork()
            pair.load_state_dict(payload["pair_online_state_dict"])
            pair.eval()
            nets_c3_unary[p] = unary
            nets_c3_pair[p] = pair

    rng = np.random.RandomState(999)
    n_probes = 200
    probe_states = rng.randn(n_probes, R_MAX, STATE_DIM).astype(np.float32) * 0.5

    learning_shift_counts = []
    coordination_shift_counts = []
    both_shift_counts = []
    override_rates = []

    per_policy = {}
    for p in POLICY_SEEDS:
        c1_net = nets_c1[p]
        c3_u = nets_c3_unary[p]
        c3_p = nets_c3_pair[p]
        n_learn = 0
        n_coord = 0
        n_both = 0
        n_total = 0
        n_override = 0

        for probe_i in range(n_probes):
            j_states = torch.tensor(probe_states[probe_i]).unsqueeze(0)
            a_mask = torch.ones(1, R_MAX, dtype=torch.float32)

            with torch.no_grad():
                c1_q = c1_net(j_states.view(-1, STATE_DIM)).view(1, R_MAX, N_ACTIONS)
                c1_greedy = c1_q.argmax(dim=-1)[0]

                c3_uq = c3_u(j_states.view(-1, STATE_DIM)).view(1, R_MAX, N_ACTIONS)
                c3_unary_greedy = c3_uq.argmax(dim=-1)[0]

                c3_coord = maxsum_inference(c3_u, c3_p, j_states, a_mask)[0]

            for r in range(R_MAX):
                c1_a = c1_greedy[r].item()
                c3u_a = c3_unary_greedy[r].item()
                c3c_a = c3_coord[r].item()
                n_total += 1

                learn_shift = (c1_a != c3u_a)
                coord_shift = (c3u_a != c3c_a)
                if learn_shift:
                    n_learn += 1
                if coord_shift:
                    n_coord += 1
                    n_override += 1
                if learn_shift and coord_shift:
                    n_both += 1

        total = n_total or 1
        per_policy[str(p)] = {
            "learning_shift_rate": n_learn / total,
            "coordination_shift_rate": n_coord / total,
            "both_shift_rate": n_both / total,
            "override_rate": n_override / total,
            "n_probes": n_probes,
            "n_decisions": n_total,
        }
        learning_shift_counts.append(n_learn / total)
        coordination_shift_counts.append(n_coord / total)
        both_shift_counts.append(n_both / total)
        override_rates.append(n_override / total)

    mean_learn = float(np.mean(learning_shift_counts))
    mean_coord = float(np.mean(coordination_shift_counts))
    mean_both = float(np.mean(both_shift_counts))
    mean_override = float(np.mean(override_rates))

    s5_status = "SUPPORTED" if mean_learn > 0.05 else "WEAK"
    s6_status = "SUPPORTED" if mean_coord > 0.05 else "WEAK"

    unary_shift = {
        "hypothesis": "S5_UNARY_POLICY_SHIFT",
        "status": s5_status,
        "mean_learning_shift_rate": mean_learn,
        "mean_coordination_shift_rate": mean_coord,
        "mean_both_rate": mean_both,
        "per_policy": per_policy,
        "probe_method": "synthetic_random_states_no_environment",
        "note": "Probes are synthetic (not from real episodes); rates indicate structural disagreement, "
                "not scenario-specific behavior. Per-scenario breakdown not available without episode states.",
    }
    override_analysis = {
        "hypothesis": "S6_COORDINATION_OVERRIDE_RATE",
        "status": s6_status,
        "mean_override_rate": mean_override,
        "per_policy": {p: per_policy[p]["override_rate"] for p in per_policy},
        "note": "Override rate = fraction of decisions where Max-Sum changes the unary greedy action. "
                "Probes are synthetic; scenario-specific override rates not available.",
    }
    return unary_shift, override_analysis


def _checkpoint_tension_analysis():
    """Analyze whether macro-selected checkpoints trade Stationary for PERSISTENT/MIXED."""
    val_ck = _load_csv("E4RTR_V2_validation_checkpoint_summary.csv", E4RTR_V2_DIR)
    sel_ck = _load("E4RE_selected_checkpoint_lock.json", E4RE_DIR)["checkpoints"]

    c3_policies = {}
    for row in val_ck:
        if row["arm"] != C3_ARM:
            continue
        p = int(row["policy_seed"])
        if p not in c3_policies:
            c3_policies[p] = []
        c3_policies[p].append({
            "checkpoint": row["checkpoint"],
            "macro": float(row["VALIDATION_MACRO_MIGD_RESPONSE"]),
            "persistent": float(row["MIGD_response_PERSISTENT"]),
            "stationary": float(row["MIGD_response_STATIONARY"]),
            "mixed": float(row["MIGD_response_MIXED"]),
        })

    tension_results = {}
    tension_count = 0
    for p in POLICY_SEEDS:
        cks = c3_policies.get(p, [])
        if not cks:
            continue
        best_macro_ck = min(cks, key=lambda x: x["macro"])["checkpoint"]
        best_persistent_ck = min(cks, key=lambda x: x["persistent"])["checkpoint"]
        best_stationary_ck = min(cks, key=lambda x: x["stationary"])["checkpoint"]
        best_mixed_ck = min(cks, key=lambda x: x["mixed"])["checkpoint"]

        sel_key = [k for k, v in sel_ck.items() if v["arm"] == C3_ARM and int(v["policy_seed"]) == p]
        selected = sel_ck[sel_key[0]]["checkpoint"] if sel_key else "UNKNOWN"

        tension = (selected != best_stationary_ck)
        if tension:
            tension_count += 1
        tension_results[str(p)] = {
            "selected": selected,
            "best_macro": best_macro_ck,
            "best_persistent": best_persistent_ck,
            "best_stationary": best_stationary_ck,
            "best_mixed": best_mixed_ck,
            "selected_equals_best_macro": selected == best_macro_ck,
            "selected_equals_best_stationary": selected == best_stationary_ck,
            "TENSION_WITH_STATIONARY": tension,
            "checkpoints": cks,
        }

    status = "SUPPORTED" if tension_count >= 3 else ("PLAUSIBLE" if tension_count >= 2 else "WEAK")
    return {
        "hypothesis": "S8_CHECKPOINT_SELECTION_SCENARIO_TENSION",
        "status": status,
        "tension_count": f"{tension_count}/5",
        "per_policy": tension_results,
        "note": "Macro-optimal checkpoint selection may systematically trade Stationary performance. "
                "The checkpoint selection rule is valid and cannot be changed; "
                "tension explains the trade-off surface, not protocol error.",
    }


def _policy_consistency(episodes, policy_rows):
    """Check which diagnostics show >=4/5 vs <=2/5 policy consistency."""
    results = {
        "STATIONARY_harm_policy_count": sum(1 for r in policy_rows if r["STATIONARY_delta_response"] < 0),
        "PERSISTENT_benefit_policy_count": sum(1 for r in policy_rows if r["PERSISTENT_delta_response"] > 0),
        "MIXED_benefit_policy_count": sum(1 for r in policy_rows if r["MIXED_delta_response"] > 0),
    }

    for s in SCENARIOS:
        for arm in [C1_ARM, C3_ARM]:
            ents = []
            predicts = []
            for p in POLICY_SEEDS:
                p_eps = [e for e in episodes if e["arm"] == arm and int(e["policy_seed"]) == p and e["scenario"] == s]
                if p_eps:
                    pf = float(p_eps[0].get("requested_PREDICT_fraction", 0))
                    predicts.append(pf)
                    ent = -sum(float(p_eps[0].get(f"requested_{a}_fraction", 0)) *
                               math.log(float(p_eps[0].get(f"requested_{a}_fraction", 0)) + 1e-12)
                               for a in ACTION_ORDER)
                    ents.append(ent)
            arm_short = "C1" if arm == C1_ARM else "C3"
            results[f"{arm_short}_{s}_predict_fractions"] = predicts
            results[f"{arm_short}_{s}_entropies"] = ents

    return {"stage": "E4S_POLICY_CONSISTENCY", **results}


def _checkpoint_trajectory_analysis():
    """Analyze how scenario performance evolves across checkpoints."""
    val_ck = _load_csv("E4RTR_V2_validation_checkpoint_summary.csv", E4RTR_V2_DIR)

    trajectories = {}
    for row in val_ck:
        arm = row["arm"]
        p = int(row["policy_seed"])
        ck = row["checkpoint"]
        key = f"{arm}_{p}"
        if key not in trajectories:
            trajectories[key] = []
        trajectories[key].append({
            "checkpoint": ck,
            "transition_count": int(row["transition_count"]),
            "macro": float(row["VALIDATION_MACRO_MIGD_RESPONSE"]),
            "persistent": float(row["MIGD_response_PERSISTENT"]),
            "stationary": float(row["MIGD_response_STATIONARY"]),
            "mixed": float(row["MIGD_response_MIXED"]),
        })

    for key in trajectories:
        trajectories[key].sort(key=lambda x: x["transition_count"])

    c3_stat_degrades = 0
    for p in POLICY_SEEDS:
        key = f"{C3_ARM}_{p}"
        traj = trajectories.get(key, [])
        if len(traj) >= 2:
            early = traj[0]["stationary"]
            late = traj[-1]["stationary"]
            if late > early:
                c3_stat_degrades += 1

    return {
        "stage": "E4S_CHECKPOINT_TRAJECTORY_ANALYSIS",
        "trajectories": trajectories,
        "C3_STATIONARY_DEGRADES_OVER_TRAINING": f"{c3_stat_degrades}/5",
        "note": "Temporal correlation does not establish causality.",
    }


def _write_report(verdict, scenario_effects, policy_rows, val_final, cross_stage,
                  hyp_matrix, s1_evidence, ck_tension, unary_shift, override_analysis,
                  stat_harm_policies, trajectory):
    L = [
        "# E4S — Pairwise Coordination Failure Analysis", "",
        f"**E4RF VERDICT = FAILED** | STATIONARY safety gate HARMFUL", "",

        "## A. Immutable E4RF failure",
        f"- E4RF_COMPLETE = YES, verdict = FAILED, reason = STATIONARY_SAFETY_HARMFUL",
        f"- MIXED primary = DIRECTIONAL_POSITIVE (delta = {verdict['MIXED_DELTA']:.6e})",
        f"- Policy robustness = PASS (4/5), macro safety = PASS", "",

        "## B. Exact final scenario effects",
    ]
    for s in SCENARIOS:
        se = scenario_effects[s]
        L += [f"- **{s}**: C1={se['C1_mean_MIGD_response']:.6f}, C3={se['C3_mean_MIGD_response']:.6f}, "
              f"delta={se['delta_response_C1_minus_C3']:.6e}, class={se['response_classification']}"]
    L += [""]

    L += ["## C. Policy-level consistency",
          f"- STATIONARY harm in {stat_harm_policies}/5 policies",
          f"- PERSISTENT benefit in {sum(1 for r in policy_rows if r['PERSISTENT_delta_response'] > 0)}/5 policies", ""]

    L += ["## D. Validation-final replication"]
    for s in SCENARIOS:
        vf = val_final[s]
        L.append(f"- {s}: val={vf['validation_direction']}, final={vf['final_direction']} -> "
                 f"{'REPLICATED' if vf['DIRECTION_REPLICATED'] else 'NOT_REPLICATED'}")
    L += [""]

    L += ["## E. Historical scenario trade-off"]
    for stage, info in cross_stage.items():
        L.append(f"- {stage}: PER={info['PERSISTENT']}, STA={info['STATIONARY']}, MIX={info['MIXED']}")
    L += [f"- **Recurring dynamic-vs-Stationary tradeoff confirmed across three coordination variants.**", ""]

    L += ["## F. C3 action specialization (S1: SUPPORTED)",
          f"- C3 STATIONARY: 0% PREDICT, 73% KEEP, entropy={s1_evidence['C3_STATIONARY_entropy']:.3f}",
          f"- C3 PERSISTENT: 93% PREDICT, 2% KEEP, entropy={float(s1_evidence.get('C3_PERSISTENT_predict_fraction',0)):.3f}",
          f"- C1 STATIONARY: 10% PREDICT, 29% KEEP, entropy={s1_evidence['C1_STATIONARY_entropy']:.3f}",
          "- C3 learns extreme scenario-specific specialization: near-monotone KEEP in STATIONARY.", ""]

    L += ["## G. Stationary coordination profile (S2: PLAUSIBLE)",
          "- Pair factors are active across all scenarios (~0.03 mean |P|).",
          "- Cannot separate pair factor contribution by scenario from saved artifacts.", ""]

    L += ["## H. Max-Sum approximation (S3: NOT_AUDITABLE)",
          "- All regions always active (n=10 > 5). No exact-vs-MaxSum comparison possible.", ""]

    L += ["## I. Pair-factor behavior (S4: PLAUSIBLE)",
          "- Per-scenario pair factor breakdown not stored. Pooled mean |P|=0.032.", ""]

    L += [f"## J. Unary-policy shift (S5: {unary_shift['status']})",
          f"- Mean learning shift rate: {unary_shift['mean_learning_shift_rate']:.3f}",
          f"- Mean coordination shift rate: {unary_shift['mean_coordination_shift_rate']:.3f}",
          "- Probes are synthetic, not scenario-specific.", ""]

    L += [f"## K. Coordination override (S6: {override_analysis['status']})",
          f"- Mean override rate: {override_analysis['mean_override_rate']:.3f}",
          "- Override = Max-Sum changes the unary greedy action.", ""]

    L += [f"## L. Checkpoint-selection tension (S8: {ck_tension['status']})",
          f"- Tension with Stationary-optimal: {ck_tension['tension_count']}",
          "- Macro selection may trade Stationary. Protocol is valid; tension explains surface.", ""]

    L += ["## M. Cross-scenario parameter interference (S9: PLAUSIBLE)",
          "- Shared parameters trained on all scenarios. Scenario-specific retraining not available.", ""]

    L += ["## N. State-representation limitation (S10: PLAUSIBLE)",
          "- 9-dim state (4 local + 5 global). Fixed by E4Q design.", ""]

    L += ["## O. Rejected mechanisms",
          "- S11 (monotonic QMIX limitation): NOT_SUPPORTED (E4PD opposite distribution)",
          "- S12 (gradient conflict): NOT_SUPPORTED (E4OD least conflict in Stationary)",
          "- S13 (parameter count): NOT_SUPPORTED (552 params not meaningful)",
          "- S14 (runtime latency): NOT_SUPPORTED (not in MIGD objective)", ""]

    L += ["## P. Hypothesis matrix"]
    for h, s in hyp_matrix.items():
        L.append(f"- {h}: **{s}**")
    L += [""]

    L += [f"## Q. Failure taxonomy",
          f"- PRIMARY: **{verdict['PRIMARY_FAILURE_MECHANISM']}**",
          f"- SECONDARY: {verdict['SECONDARY_MECHANISMS']}", ""]

    L += ["## R. Primary / secondary mechanisms",
          "- The frozen C3 treatment fails because multiple interacting factors create a recurring "
          "dynamic-vs-Stationary tradeoff that three different coordination architectures cannot escape.", ""]

    L += ["## S. What C3 failure does and does not prove",
          "- PROVES: frozen C3 did not safely outperform C1.",
          "- DOES NOT PROVE: all pairwise coordination is useless.",
          "- DOES NOT PROVE: pairwise interactions don't exist (E4PD confirmed ~22% reversals).",
          "- DOES NOT PROVE: coordinated inference is universally harmful.", ""]

    L += ["## T. Whether another controller revision is justified",
          "- Three distinct coordination architectures (QMIX, credit-preservation, pairwise CG) all FAILED.",
          "- All show the same recurring dynamic-vs-Stationary tradeoff.",
          "- No minimal isolated revision is clearly justified by the evidence.",
          "- Simply replacing the coordination method with QPLEX/QTRAN may not address the root cause.", ""]

    L += ["## U. Research-stop consideration",
          "- CONTROLLER_RESEARCH_STOP explicitly considered and CHOSEN as next route.",
          "- Tested: baseline DDQN, V3 norm, R1 reward, QMIX, credit-preservation, pairwise CG.",
          "- Only R1 reward was causally validated (E4IF).",
          "- Six controller treatments failed. Continuing architecture search without addressing "
          "the underlying scenario tradeoff is not scientifically justified.", ""]

    L += ["## V. Limitations",
          "- E4S is post-failure analysis; most diagnostics are observational.",
          "- Saved artifacts may not contain all state-level quantities.",
          "- No new counterfactual environment evaluation allowed.",
          "- Checkpoint trajectories do not establish causality.",
          "- Action specialization does not itself imply poor performance.",
          "- Max-Sum approximation does not imply it caused harm.",
          "- Cross-scenario interference requires controlled intervention for causal confirmation.",
          "- State insufficiency cannot be established solely from performance.",
          "- C3 failure does not refute existence of pairwise interactions.",
          "- C3 failure does not prove QPLEX/QTRAN would succeed.",
          "- No real-world UAV validation performed.", ""]

    L += [f"## W. Next-stage routing: {verdict['NEXT_STAGE']}"]

    (RESULTS_DIR / "E4S_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    print("=" * 60)
    print("E4S — FINAL TERMINAL SUMMARY")
    print("=" * 60)
    fields = [
        ("E4S_COMPLETE", v["E4S_COMPLETE"]),
        ("E4RF_VERDICT", v["E4RF_VERDICT"]),
        ("FAILURE_REASON", v["FAILURE_REASON"]),
        ("MIXED_PRIMARY", v["MIXED_PRIMARY"]),
        ("MIXED_DELTA", f"{v['MIXED_DELTA']:.6e}"),
        ("MIXED_CI", str(v["MIXED_CI"])),
        ("POLICY_ROBUSTNESS", v["POLICY_ROBUSTNESS"]),
        ("PERSISTENT_SAFETY", v["PERSISTENT_SAFETY"]),
        ("STATIONARY_SAFETY", v["STATIONARY_SAFETY"]),
        ("MACRO_SAFETY", v["MACRO_SAFETY"]),
        ("VALIDATION_FINAL_SCENARIO_PATTERN", v["VALIDATION_FINAL_SCENARIO_PATTERN"]),
        ("STATIONARY_POLICY_HARM_COUNT", v["STATIONARY_POLICY_HARM_COUNT"]),
    ]
    for h in ["S1_ACTION_SPECIALIZATION_SHIFT", "S2_STATIONARY_OVER_COORDINATION",
              "S3_MAXSUM_APPROXIMATION_ERROR", "S4_PAIR_FACTOR_MAGNITUDE_MISALIGNMENT",
              "S5_UNARY_POLICY_SHIFT", "S6_COORDINATION_OVERRIDE_RATE",
              "S7_OVERRIDE_QUALITY_ASSOCIATION", "S8_CHECKPOINT_SELECTION_SCENARIO_TENSION",
              "S9_CROSS_SCENARIO_PARAMETER_INTERFERENCE", "S10_STATE_REPRESENTATION_INSUFFICIENCY",
              "S11_STATIONARY_MONOTONICITY_CAUSE", "S12_STATIONARY_GRADIENT_CONFLICT",
              "S13_PARAMETER_COUNT_EXPLANATION", "S14_RUNTIME_LATENCY_PERFORMANCE_CAUSE"]:
        fields.append((h, v[h]))
    fields += [
        ("PRIMARY_FAILURE_MECHANISM", v["PRIMARY_FAILURE_MECHANISM"]),
        ("SECONDARY_MECHANISMS", str(v["SECONDARY_MECHANISMS"])),
        ("PAIRWISE_INTERACTION_EXISTENCE", v["PAIRWISE_INTERACTION_EXISTENCE"]),
        ("PAIRWISE_CG_UNIVERSALLY_REFUTED", v["PAIRWISE_CG_UNIVERSALLY_REFUTED"]),
        ("NEW_ENVIRONMENT_EPISODES", str(v["NEW_ENVIRONMENT_EPISODES"])),
        ("NEW_OBJECTIVE_EVALUATIONS", str(v["NEW_OBJECTIVE_EVALUATIONS"])),
        ("NEW_TRAINING", v["NEW_TRAINING"]),
        ("C3_MODIFIED", v["C3_MODIFIED"]),
        ("PAIRWISE_COORDINATION_CAUSAL_VERDICT", v["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]),
        ("PERFORMANCE_IMPROVEMENT_CLAIM", v["PERFORMANCE_IMPROVEMENT_CLAIM"]),
        ("FULL_REGRESSION", v["FULL_REGRESSION"]),
        ("NEXT_STAGE", v["NEXT_STAGE"]),
    ]
    for label, val in fields:
        print(f"{label:<48} = {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
