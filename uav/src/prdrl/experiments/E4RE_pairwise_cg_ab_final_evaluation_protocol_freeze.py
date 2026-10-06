"""
E4RE — Pairwise CG A/B Final Evaluation Protocol Freeze.

PROTOCOL FREEZE ONLY. No environment execution. No final seeds opened.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent.parent
V2_DIR = ROOT / "results" / "E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2"
E4RE_DIR = ROOT / "results" / "E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze"
E4RF_DIR_NAME = "E4RF_pairwise_cg_ab_final_evaluation_execution"

# ============================================================
# FROZEN CONSTANTS
# ============================================================
C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
ARMS = [C1_ARM, C3_ARM]
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
TRAIN_SEEDS = list(range(2501, 2561))
VAL_SEEDS = list(range(2561, 2581))
FINAL_SEEDS = list(range(2581, 2641))
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

FROZEN_QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C3_SHA = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"
ORIGINAL_E4RT_SHA = "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6"
E4RTA_AMENDMENT_SHA = "ac5c93f7c68c304d1dae3dbf4c4de957c440d3ddf181a7870e68564c4d0bad91"
BATCHED_IMPL_SHA = "c8c455a05b7cb8f7b57ae840a077af90b0470e2c336920a7dbd41bee52c0f97f"

DELTA_PRACTICAL = 1e-5
BOOTSTRAP_N = 50000
BOOTSTRAP_SEED = 136001
BOOTSTRAP_POLICY_AXIS = 5
BOOTSTRAP_ENV_AXIS = 60
ROBUSTNESS_THRESHOLD = 4  # at least 4 of 5

SELECTED_C1 = {
    7001: "ckpt_final", 7002: "ckpt_final", 7003: "ckpt_final",
    7004: "ckpt_final", 7005: "ckpt_final",
}
SELECTED_C3 = {
    7001: "ckpt_50000", 7002: "ckpt_final", 7003: "ckpt_final",
    7004: "ckpt_40000", 7005: "ckpt_40000",
}

FINAL_MATRIX_SIZE = len(ARMS) * len(POLICY_SEEDS) * len(FINAL_SEEDS) * len(SCENARIOS)
assert FINAL_MATRIX_SIZE == 1800

OPERATORS = {
    "KEEP": {"FE": 0},
    "LOCAL": {"sigma_L": 0.02},
    "PREDICT": {"sigma_P": 0.005, "lambda_P": 1, "fallback": "LOCAL"},
    "DIVERSIFY": {"sigma_D": 0.15},
    "budget": 10,
    "budget_scope": "PER_ACTIVE_REGION",
    "survival": "unrestricted_standard_NSGA_II",
    "HARD": False,
}

EPISODE_SCHEMA = [
    "arm", "policy_seed", "selected_checkpoint", "environment_seed",
    "scenario", "MIGD_response", "MIGD_epoch", "response_FE",
    "requested_KEEP_fraction", "requested_LOCAL_fraction",
    "requested_PREDICT_fraction", "requested_DIVERSIFY_fraction",
    "realized_KEEP_fraction", "realized_LOCAL_fraction",
    "realized_PREDICT_fraction", "realized_DIVERSIFY_fraction",
    "multi_action_rate", "action_entropy", "epsilon", "learning_updates",
    "checkpoint_SHA256", "episode_complete",
]


def _jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jdefault)
        f.write("\n")


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(delta, ci_lo, ci_hi, margin=DELTA_PRACTICAL):
    if abs(delta) <= margin:
        return "PRACTICALLY_NEUTRAL"
    if delta > margin:
        if ci_lo > 0:
            return "BENEFICIAL"
        return "DIRECTIONAL_POSITIVE"
    if delta < -margin:
        if ci_hi < 0:
            return "HARMFUL"
        return "DIRECTIONAL_NEGATIVE"
    return "PRACTICALLY_NEUTRAL"


def paired_crossed_bootstrap(c1_matrix, c3_matrix, n_policies, n_envs,
                             n_boot, seed):
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.randint(0, n_policies, size=n_policies)
        ei = rng.randint(0, n_envs, size=n_envs)
        c1_sample = c1_matrix[pi][:, ei].mean()
        c3_sample = c3_matrix[pi][:, ei].mean()
        deltas[b] = c1_sample - c3_sample
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    point = float(np.mean(deltas))
    return point, ci_lo, ci_hi


def main():
    E4RE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ================================================================
    # PHASE E4RE-0: PRE-FREEZE INTEGRITY
    # ================================================================
    print("[E4RE-0] Pre-freeze integrity checks ...")
    v2_verdict = json.load(open(V2_DIR / "E4RTR_V2_verdict.json"))
    v2_completion = json.load(open(V2_DIR / "E4RTR_V2_training_completion.json"))
    v2_selected = json.load(open(V2_DIR / "E4RTR_V2_selected_checkpoint_manifest.json"))
    v2_sel_hashes = json.load(open(V2_DIR / "E4RTR_V2_selected_checkpoint_hashes.json"))
    v2_config = json.load(open(V2_DIR / "E4RTR_V2_execution_config.json"))
    v2_ck_integrity = json.load(open(V2_DIR / "E4RTR_V2_checkpoint_integrity_post_validation.json"))
    v2_old_excl = json.load(open(V2_DIR / "E4RTR_V2_old_run_exclusion.json"))
    v2_no_adapt = json.load(open(V2_DIR / "E4RTR_V2_no_adaptation_audit.json"))
    v2_firewall = json.load(open(V2_DIR / "E4RTR_V2_final_seed_firewall.json"))

    checks = {}
    checks["E4RTR_V2_COMPLETE"] = v2_verdict["E4RTR_V2_COMPLETE"] == "YES"
    checks["TRAINING_10800"] = v2_verdict["TRAIN_EPISODES"] == "10800/10800"
    checks["VALIDATION_3600"] = v2_verdict["VALIDATION_EPISODES"] == "3600/3600"
    checks["CHECKPOINTS_60"] = v2_verdict["CHECKPOINTS"] == "60/60"
    checks["SELECTED_10"] = v2_sel_hashes["n_selected"] == 10
    checks["SELECTED_HASHES_RESOLVE"] = all(
        Path(v2_selected["selected"][k]["checkpoint_path"]).exists()
        for k in v2_selected["selected"]
    )

    # Verify hashes unchanged since validation
    hash_ok = True
    for k, info in v2_selected["selected"].items():
        p = info["checkpoint_path"]
        if Path(p).exists():
            actual = _hash_file(p)
            if actual != info["sha256"]:
                hash_ok = False
        else:
            hash_ok = False
    checks["CHECKPOINT_HASHES_UNCHANGED"] = hash_ok
    checks["OLD_CHECKPOINTS_ABSENT"] = v2_old_excl["old_checkpoint_reuse"] is False
    checks["NO_FINAL_ADAPTATION"] = all(
        v is False for k, v in v2_no_adapt.items()
        if k not in ("stage", "NO_POST_VALIDATION_ADAPTATION")
    )
    checks["C1_CONFIG_SHA_EXACT"] = v2_config["QMIX_CONFIG_SHA256"] == FROZEN_QMIX_SHA
    checks["C3_CONFIG_SHA_EXACT"] = v2_config["C3_CONFIG_SHA256"] == FROZEN_C3_SHA
    checks["E4RT_SHA_EXACT"] = v2_config["original_E4RT_protocol_SHA256"] == ORIGINAL_E4RT_SHA
    checks["E4RTA_SHA_EXACT"] = v2_config["E4RTA_amendment_SHA256"] == E4RTA_AMENDMENT_SHA
    checks["FINAL_SEEDS_EXACT"] = (v2_config["final_seeds_firewalled"] == [2581, 2640])
    checks["FINAL_ACCESS_COUNT_ZERO"] = v2_firewall["FINAL_ACCESS_COUNT"] == 0
    checks["POLICY_SEEDS_EXACT"] = v2_config["policy_seeds"] == POLICY_SEEDS
    checks["FLOAT32_EXCEPTION_DOCUMENTED"] = v2_verdict["FLOAT32_TIE_EXCEPTION"] == "DOCUMENTED"
    checks["NON_TIE_MISMATCH_ZERO"] = v2_verdict["NON_TIE_ACTION_MISMATCH_COUNT"] == 0

    all_pass = all(checks.values())
    _jdump({
        "stage": "E4RE_PRE_FREEZE_INTEGRITY",
        "checks": checks,
        "ALL_PASS": all_pass,
    }, E4RE_DIR / "E4RE_pre_manifest.json")

    _jdump({
        "stage": "E4RE_HISTORICAL_INTEGRITY",
        "E4RTR_V2_COMPLETE": v2_verdict["E4RTR_V2_COMPLETE"],
        "TRAINING": v2_verdict["TRAIN_EPISODES"],
        "VALIDATION": v2_verdict["VALIDATION_EPISODES"],
        "CHECKPOINTS": v2_verdict["CHECKPOINTS"],
        "SELECTED": v2_sel_hashes["n_selected"],
        "FULL_REGRESSION": v2_verdict.get("FULL_REGRESSION", "N/A"),
        "INTEGRITY": "PASS" if all_pass else "FAIL",
    }, E4RE_DIR / "E4RE_historical_integrity.json")

    if not all_pass:
        print(f"[E4RE-0] FAIL: {[k for k,v in checks.items() if not v]}")
        _jdump({"E4RE_COMPLETE": "NO", "NEXT_STAGE": "E4RE_INTEGRITY_REVIEW"},
               E4RE_DIR / "E4RE_verdict.json")
        return
    print("[E4RE-0] All 18 integrity checks PASS.")

    # ================================================================
    # PHASE E4RE-1: SELECTED CHECKPOINT LOCK
    # ================================================================
    print("[E4RE-1] Locking selected checkpoints ...")
    ck_lock = {}
    ck_hash_lock = {}
    for arm, sel_map in [(C1_ARM, SELECTED_C1), (C3_ARM, SELECTED_C3)]:
        for p, ckpt_name in sel_map.items():
            key = f"{arm}_{p}"
            info = v2_selected["selected"][key]
            assert info["selected_checkpoint"] == ckpt_name
            sha = info["sha256"]
            ck_lock[key] = {
                "arm": arm,
                "policy_seed": p,
                "checkpoint": ckpt_name,
                "transition_count": info["transition_count"],
                "SHA256": sha,
                "path": info["checkpoint_path"],
            }
            ck_hash_lock[key] = sha

    _jdump({
        "stage": "E4RE_SELECTED_CHECKPOINT_LOCK",
        "n_selected": len(ck_lock),
        "n_C1": 5, "n_C3": 5,
        "all_policies_retained": True,
        "checkpoints": ck_lock,
        "NO_RESELECTION": True,
        "NO_FALLBACK": True,
    }, E4RE_DIR / "E4RE_selected_checkpoint_lock.json")

    _jdump({
        "stage": "E4RE_SELECTED_CHECKPOINT_HASHES",
        "hashes": ck_hash_lock,
        "n_selected": len(ck_hash_lock),
        "all_verified": hash_ok,
    }, E4RE_DIR / "E4RE_selected_checkpoint_hashes.json")

    # ================================================================
    # E4RE-2: TRAINING/VALIDATION LOCK
    # ================================================================
    _jdump({
        "stage": "E4RE_TRAINING_VALIDATION_LOCK",
        "TRAINING_COMPLETE": True,
        "TRAINING_EPISODES": 10800,
        "VALIDATION_COMPLETE": True,
        "VALIDATION_EPISODES": 3600,
        "VALIDATION_STATUS": "NON_CONFIRMATORY_DESCRIPTIVE_ONLY",
        "SELECTION_CRITERION": "validation_macro_MIGD_response",
        "TIEBREAK_1": "macro_MIGD_epoch",
        "TIEBREAK_2": "response_FE",
        "TIEBREAK_3": "earlier_checkpoint",
        "NO_REVISIT_SELECTION": True,
    }, E4RE_DIR / "E4RE_training_validation_lock.json")

    # ================================================================
    # E4RE-3: FINAL SEED CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_FINAL_SEED_CONTRACT",
        "final_seeds": FINAL_SEEDS,
        "n_final": len(FINAL_SEEDS),
        "range": "2581..2640",
        "status_at_entry": "NEVER_ACCESSED",
        "ACCESS_COUNT_AT_E4RE": 0,
        "NO_REPLACEMENT": True,
        "NO_SUPPLEMENTATION": True,
        "NO_EXTENSION": True,
    }, E4RE_DIR / "E4RE_final_seed_contract.json")

    _jdump({
        "stage": "E4RE_FINAL_SEED_FIREWALL",
        "FINAL_ACCESS_COUNT": 0,
        "FINAL_SEED_STATUS": "RESERVED_UNTOUCHED",
        "seeds": "2581..2640",
        "NO_SMOKE_TEST": True,
        "NO_DRY_RUN": True,
        "NO_TIMING": True,
        "NO_DEBUG": True,
    }, E4RE_DIR / "E4RE_final_seed_firewall.json")

    # ================================================================
    # E4RE-6: FINAL MATRIX CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_FINAL_MATRIX_CONTRACT",
        "arms": ARMS,
        "policies": POLICY_SEEDS,
        "environments": FINAL_SEEDS,
        "scenarios": SCENARIOS,
        "total_episodes": FINAL_MATRIX_SIZE,
        "per_arm": FINAL_MATRIX_SIZE // 2,
        "per_scenario": FINAL_MATRIX_SIZE // 3,
        "per_arm_per_scenario": FINAL_MATRIX_SIZE // 6,
        "REQUIRE_COMPLETE": True,
        "NO_MISSING_CELL": True,
    }, E4RE_DIR / "E4RE_final_matrix_contract.json")

    # ================================================================
    # E4RE-7: PAIRING CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_PAIRING_CONTRACT",
        "paired_unit": "policy × environment × scenario",
        "both_arms_per_unit": True,
        "deterministic_environment": True,
        "NO_UNMATCHED_REPLACEMENT": True,
    }, E4RE_DIR / "E4RE_pairing_contract.json")

    # ================================================================
    # E4RE-8/9/10: EVALUATION MODE + ACTION PATHS
    # ================================================================
    _jdump({
        "stage": "E4RE_EVALUATION_MODE",
        "epsilon": 0,
        "model_eval": True,
        "no_optimizer": True,
        "no_replay_append": True,
        "no_target_update": True,
        "no_gradient": True,
        "no_training": True,
        "no_normalizer_update": True,
        "no_checkpoint_mutation": True,
    }, E4RE_DIR / "E4RE_evaluation_mode.json")

    _jdump({
        "stage": "E4RE_ACTION_PATH_CONTRACT",
        "C1_action_path": {
            "shared_utility": True,
            "independent_regional_greedy_argmax": True,
            "no_mixer_deployment": True,
        },
        "C3_action_path": {
            "same_scientific_C3_config": True,
            "complete_pairwise_graph": True,
            "centered_pair_factors": True,
            "mean_unary_plus_mean_pair_value": True,
            "maxsum_K": 10,
            "frozen_float32_semantics": True,
            "no_tie_specific_scalar_fallback": True,
        },
    }, E4RE_DIR / "E4RE_action_path_contract.json")

    # ================================================================
    # E4RE-11: OPERATORS
    # ================================================================
    _jdump({
        "stage": "E4RE_OPERATOR_CONTRACT",
        "operators": OPERATORS,
    }, E4RE_DIR / "E4RE_operator_contract.json")

    # ================================================================
    # E4RE-13/14/15: PRIMARY ENDPOINT
    # ================================================================
    _jdump({
        "stage": "E4RE_PRIMARY_ENDPOINT",
        "primary_scenario": "MIXED",
        "primary_endpoint": "MIGD_response",
        "effect_direction": "C1_MINUS_C3",
        "positive_means": "C3_BETTER",
        "delta_formula": "mean(MIGD_C1 - MIGD_C3) over 5×60 paired units",
    }, E4RE_DIR / "E4RE_primary_endpoint.json")

    _jdump({
        "stage": "E4RE_PRACTICAL_MARGIN",
        "DELTA_PRACTICAL": DELTA_PRACTICAL,
        "NO_RETUNING": True,
    }, E4RE_DIR / "E4RE_practical_margin.json")

    # ================================================================
    # E4RE-17/18: BOOTSTRAP CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_BOOTSTRAP_CONTRACT",
        "method": "paired_crossed_two_axis_bootstrap",
        "policy_axis": BOOTSTRAP_POLICY_AXIS,
        "environment_axis": BOOTSTRAP_ENV_AXIS,
        "resamples": BOOTSTRAP_N,
        "master_seed": BOOTSTRAP_SEED,
        "CI_method": "95_percentile",
        "shared_draws_across_scenarios": True,
        "do_not_bootstrap_episode_rows_independently": True,
    }, E4RE_DIR / "E4RE_bootstrap_contract.json")

    # ================================================================
    # E4RE-19: POLICY ROBUSTNESS
    # ================================================================
    _jdump({
        "stage": "E4RE_POLICY_ROBUSTNESS_CONTRACT",
        "threshold": f"at_least_{ROBUSTNESS_THRESHOLD}_of_5_positive",
        "per_policy_delta": "mean over final environments C1 - C3",
        "PASS_if": f">= {ROBUSTNESS_THRESHOLD} of 5 policy deltas > 0",
    }, E4RE_DIR / "E4RE_policy_robustness_contract.json")

    # ================================================================
    # E4RE-20/21/22: SAFETY CONTRACTS
    # ================================================================
    for sc in ["persistent", "stationary", "macro"]:
        _jdump({
            "stage": f"E4RE_{sc.upper()}_SAFETY_CONTRACT",
            "endpoints": ["MIGD_response", "MIGD_epoch"],
            "PASS_if": "neither endpoint is HARMFUL",
            "FAIL_if": "any endpoint is HARMFUL",
        }, E4RE_DIR / f"E4RE_{sc}_safety_contract.json")

    # ================================================================
    # E4RE-16: CLASSIFICATION CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_CLASSIFICATION_CONTRACT",
        "margin": DELTA_PRACTICAL,
        "BENEFICIAL": "delta > margin AND CI_lower > 0",
        "DIRECTIONAL_POSITIVE": "delta > margin AND CI crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(delta) <= margin",
        "DIRECTIONAL_NEGATIVE": "delta < -margin AND CI crosses 0",
        "HARMFUL": "delta < -margin AND CI_upper < 0",
    }, E4RE_DIR / "E4RE_classification_contract.json")

    # ================================================================
    # E4RE-24/25/26: CAUSAL VERDICT CONTRACT
    # ================================================================
    _jdump({
        "stage": "E4RE_CAUSAL_VERDICT_CONTRACT",
        "VALIDATED": {
            "requires_ALL": [
                "MIXED MIGD_response = BENEFICIAL",
                "policy robustness = PASS",
                "persistent safety = PASS",
                "stationary safety = PASS",
                "macro safety = PASS",
                "final matrix integrity = PASS",
                "checkpoint integrity = PASS",
                "config integrity = PASS",
                "no final-data adaptation = PASS",
            ],
        },
        "PARTIAL": {
            "requires": "safety PASS AND one of: DIRECTIONAL_POSITIVE, BENEFICIAL+robustness_fail, PRACTICALLY_NEUTRAL",
        },
        "FAILED": {
            "if_ANY": [
                "MIXED = DIRECTIONAL_NEGATIVE",
                "MIXED = HARMFUL",
                "persistent safety FAIL",
                "stationary safety FAIL",
                "macro safety FAIL",
                "checkpoint integrity FAIL",
                "config integrity FAIL",
                "scientific protocol integrity FAIL",
                "final-data adaptation detected",
            ],
        },
    }, E4RE_DIR / "E4RE_causal_verdict_contract.json")

    # ================================================================
    # E4RE-23: FE REPORTING BOUNDARY
    # ================================================================
    _jdump({
        "stage": "E4RE_FE_REPORTING_CONTRACT",
        "FE_is": "DESCRIPTIVE_EFFICIENCY_CONTEXT",
        "lower_FE_does_NOT_rescue_HARMFUL": True,
        "higher_FE_does_NOT_invalidate_treatment": True,
    }, E4RE_DIR / "E4RE_fe_reporting_contract.json")

    # ================================================================
    # E4RE-28/29: MECHANISM DIAGNOSTIC BOUNDARY
    # ================================================================
    _jdump({
        "stage": "E4RE_MECHANISM_DIAGNOSTIC_BOUNDARY",
        "entropy_is_non_gating": True,
        "multi_action_is_non_gating": True,
        "maxsum_regret_is_non_gating": True,
        "pair_factor_magnitude_is_non_gating": True,
        "message_magnitude_is_non_gating": True,
        "these_cannot_alter_final_verdict": True,
    }, E4RE_DIR / "E4RE_mechanism_diagnostic_boundary.json")

    # ================================================================
    # E4RE-27: VALIDATION NON-ALTERATION
    # ================================================================
    _jdump({
        "stage": "E4RE_NO_VALIDATION_ADAPTATION",
        "validation_descriptively_favored": {
            "PERSISTENT": "C3",
            "STATIONARY": "C1",
            "MIXED": "C3",
            "macro": "C3",
        },
        "MUST_NOT_CHANGE": [
            "primary scenario", "safety rules", "practical margin",
            "bootstrap", "policy robustness", "selected checkpoints",
        ],
        "VALIDATION_INFORMATION_USED_TO_CHANGE_PROTOCOL": False,
    }, E4RE_DIR / "E4RE_no_validation_adaptation.json")

    # ================================================================
    # E4RE-34: NO FINAL ADAPTATION
    # ================================================================
    _jdump({
        "stage": "E4RE_NO_FINAL_ADAPTATION",
        "FORBIDDEN_AFTER_SEED_2581_OPENED": [
            "retraining", "checkpoint_reselection", "architecture_changes",
            "K_changes", "graph_changes", "pair_weight_changes",
            "centering_changes", "optimizer_changes", "reward_changes",
            "state_changes", "operator_changes", "bootstrap_changes",
            "endpoint_changes", "safety_gate_changes",
        ],
    }, E4RE_DIR / "E4RE_no_final_adaptation.json")

    # ================================================================
    # E4RE-35/36: INTEGRITY CONTRACTS
    # ================================================================
    _jdump({
        "stage": "E4RE_CHECKPOINT_INTEGRITY_CONTRACT",
        "pre_post_hash_required": True,
        "no_checkpoint_write": True,
    }, E4RE_DIR / "E4RE_checkpoint_integrity_contract.json")

    _jdump({
        "stage": "E4RE_CONFIG_INTEGRITY_CONTRACT",
        "C1_CONFIG_SHA256": FROZEN_QMIX_SHA,
        "C3_CONFIG_SHA256": FROZEN_C3_SHA,
        "ORIGINAL_E4RT_SHA256": ORIGINAL_E4RT_SHA,
        "E4RTA_SHA256": E4RTA_AMENDMENT_SHA,
    }, E4RE_DIR / "E4RE_config_integrity_contract.json")

    # ================================================================
    # E4RE-37: ENGINEERING LINEAGE
    # ================================================================
    _jdump({
        "stage": "E4RE_ENGINEERING_LINEAGE",
        "batched_implementation_SHA256": BATCHED_IMPL_SHA,
        "float32_B_gt_1_action_mismatch_rate": "0.1082%",
        "strict_non_tie_mismatches": 0,
        "lineage": [
            "E4R_DESIGN", "E4RI_IMPLEMENTATION", "E4RT_PROTOCOL",
            "E4RIA_BATCHED_EQUIVALENCE", "E4RTA_PROTOCOL_AMENDMENT",
            "E4RTAR_REGRESSION_RECONCILIATION", "E4RTR_V2_BATCHED_RESTART",
        ],
        "ENGINEERING_PROVENANCE_ONLY": True,
    }, E4RE_DIR / "E4RE_engineering_lineage.json")

    # ================================================================
    # E4RE-43/44/45: FUTURE ROUTING
    # ================================================================
    _jdump({
        "stage": "E4RE_FUTURE_ROUTING",
        "IF_VALIDATED": "E4S_PAIRWISE_COORDINATION_VS_C0_DESIGN",
        "IF_PARTIAL": "E4S_PAIRWISE_COORDINATION_RESULT_REVIEW",
        "IF_FAILED": "E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS",
        "NOTE_VALIDATED": "C3>C1 does NOT imply C3>C0; fresh C3-vs-C0 required",
        "NOTE_FAILED": "no post-hoc tuning of K, pair width, graph, pair weight, centering, replay",
    }, E4RE_DIR / "E4RE_future_routing.json")

    # ================================================================
    # E4RE-52: E4RF OUTPUT CONTRACT
    # ================================================================
    e4rf_outputs = [
        "E4RF_pre_manifest.json", "E4RF_checkpoint_lock.json",
        "E4RF_final_seed_usage.json", "E4RF_final_episode_results.csv",
        "E4RF_matrix_integrity.json", "E4RF_scenario_summary.csv",
        "E4RF_mixed_primary.json", "E4RF_policy_robustness.csv",
        "E4RF_persistent_safety.json", "E4RF_stationary_safety.json",
        "E4RF_macro_safety.json", "E4RF_bootstrap.json",
        "E4RF_fe_summary.json", "E4RF_action_behavior.json",
        "E4RF_c3_mechanism_diagnostics.json",
        "E4RF_checkpoint_integrity.json", "E4RF_config_integrity.json",
        "E4RF_no_adaptation_audit.json", "E4RF_causal_verdict.json",
        "E4RF_verdict.json", "E4RF_report.md",
    ]
    _jdump({
        "stage": "E4RE_E4RF_OUTPUT_CONTRACT",
        "output_directory": f"results/{E4RF_DIR_NAME}/",
        "required_outputs": e4rf_outputs,
        "episode_schema": EPISODE_SCHEMA,
        "epsilon_required": 0,
        "learning_updates_required": 0,
    }, E4RE_DIR / "E4RE_e4rf_output_contract.json")

    # ================================================================
    # E4RE-49: STATISTICAL VALIDATION (synthetic)
    # ================================================================
    print("[E4RE-49] Statistical synthetic validation ...")
    synth_n_pol = 5
    synth_n_env = 60
    rng_synth = np.random.RandomState(99999)
    c1_synth = rng_synth.randn(synth_n_pol, synth_n_env) * 0.001 + 0.007
    c3_synth = rng_synth.randn(synth_n_pol, synth_n_env) * 0.001 + 0.006
    pt_synth, lo_synth, hi_synth = paired_crossed_bootstrap(
        c1_synth, c3_synth, synth_n_pol, synth_n_env, BOOTSTRAP_N, BOOTSTRAP_SEED)

    synth_class = _classify(pt_synth, lo_synth, hi_synth)

    synth_ok = (
        abs(pt_synth - (c1_synth.mean() - c3_synth.mean())) < 1e-3
        and lo_synth < hi_synth
        and synth_class == "BENEFICIAL"
    )

    _jdump({
        "stage": "E4RE_STATISTICAL_VALIDATION",
        "synthetic_delta": pt_synth,
        "synthetic_CI": [lo_synth, hi_synth],
        "synthetic_classification": synth_class,
        "expected_classification": "BENEFICIAL",
        "PASS": synth_ok,
        "policy_axis": synth_n_pol,
        "env_axis": synth_n_env,
        "bootstrap_N": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }, E4RE_DIR / "E4RE_statistical_validation.json")

    # ================================================================
    # E4RE-50: CLASSIFIER TRUTH TABLE
    # ================================================================
    print("[E4RE-50] Classifier truth table ...")
    truth_cases = [
        {"delta": 5e-4, "ci_lo": 1e-4, "ci_hi": 9e-4, "expected": "BENEFICIAL"},
        {"delta": 5e-4, "ci_lo": -1e-4, "ci_hi": 9e-4, "expected": "DIRECTIONAL_POSITIVE"},
        {"delta": 5e-6, "ci_lo": -1e-4, "ci_hi": 1e-4, "expected": "PRACTICALLY_NEUTRAL"},
        {"delta": -5e-4, "ci_lo": -9e-4, "ci_hi": 1e-4, "expected": "DIRECTIONAL_NEGATIVE"},
        {"delta": -5e-4, "ci_lo": -9e-4, "ci_hi": -1e-4, "expected": "HARMFUL"},
    ]
    classifier_results = []
    for tc in truth_cases:
        actual = _classify(tc["delta"], tc["ci_lo"], tc["ci_hi"])
        classifier_results.append({**tc, "actual": actual, "PASS": actual == tc["expected"]})

    _jdump({
        "stage": "E4RE_CLASSIFIER_TRUTH_TABLE",
        "cases": classifier_results,
        "ALL_PASS": all(r["PASS"] for r in classifier_results),
    }, E4RE_DIR / "E4RE_classifier_truth_table.json")

    # ================================================================
    # E4RE-51: VERDICT TRUTH TABLE
    # ================================================================
    print("[E4RE-51] Verdict truth table ...")

    def _verdict(primary_class, robustness_pass, safety_pass, integrity_pass):
        if not integrity_pass:
            return "FAILED"
        if not safety_pass:
            return "FAILED"
        if primary_class in ("DIRECTIONAL_NEGATIVE", "HARMFUL"):
            return "FAILED"
        if primary_class == "BENEFICIAL" and robustness_pass:
            return "VALIDATED"
        if primary_class == "DIRECTIONAL_POSITIVE" and safety_pass:
            return "PARTIAL"
        if primary_class == "BENEFICIAL" and not robustness_pass:
            return "PARTIAL"
        if primary_class == "PRACTICALLY_NEUTRAL" and safety_pass:
            return "PARTIAL"
        return "FAILED"

    verdict_cases = [
        {"primary": "BENEFICIAL", "robust": True, "safe": True, "integ": True, "expected": "VALIDATED"},
        {"primary": "DIRECTIONAL_POSITIVE", "robust": True, "safe": True, "integ": True, "expected": "PARTIAL"},
        {"primary": "BENEFICIAL", "robust": False, "safe": True, "integ": True, "expected": "PARTIAL"},
        {"primary": "PRACTICALLY_NEUTRAL", "robust": True, "safe": True, "integ": True, "expected": "PARTIAL"},
        {"primary": "DIRECTIONAL_NEGATIVE", "robust": True, "safe": True, "integ": True, "expected": "FAILED"},
        {"primary": "HARMFUL", "robust": True, "safe": True, "integ": True, "expected": "FAILED"},
        {"primary": "BENEFICIAL", "robust": True, "safe": False, "integ": True, "expected": "FAILED"},
        {"primary": "BENEFICIAL", "robust": True, "safe": True, "integ": False, "expected": "FAILED"},
    ]
    verdict_results = []
    for vc in verdict_cases:
        actual = _verdict(vc["primary"], vc["robust"], vc["safe"], vc["integ"])
        verdict_results.append({**vc, "actual": actual, "PASS": actual == vc["expected"]})

    _jdump({
        "stage": "E4RE_VERDICT_TRUTH_TABLE",
        "cases": verdict_results,
        "ALL_PASS": all(r["PASS"] for r in verdict_results),
    }, E4RE_DIR / "E4RE_verdict_truth_table.json")

    # ================================================================
    # PROTOCOL FROZEN
    # ================================================================
    protocol_sha_input = json.dumps({
        "primary": "MIXED_MIGD_response",
        "effect": "C1_MINUS_C3",
        "margin": DELTA_PRACTICAL,
        "bootstrap": {"N": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                       "axes": [BOOTSTRAP_POLICY_AXIS, BOOTSTRAP_ENV_AXIS]},
        "robustness": ROBUSTNESS_THRESHOLD,
        "safety": ["PERSISTENT", "STATIONARY", "MACRO"],
        "final_seeds": FINAL_SEEDS,
        "selected_hashes": ck_hash_lock,
    }, sort_keys=True)
    protocol_sha = hashlib.sha256(protocol_sha_input.encode()).hexdigest()

    _jdump({
        "stage": "E4RE_PROTOCOL_FROZEN",
        "PROTOCOL_SHA256": protocol_sha,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "NO_MODIFICATION_AFTER_FREEZE": True,
    }, E4RE_DIR / "E4RE_protocol_frozen.json")

    # ================================================================
    # VERDICT
    # ================================================================
    all_synth_pass = (
        synth_ok
        and all(r["PASS"] for r in classifier_results)
        and all(r["PASS"] for r in verdict_results)
    )

    e4re_complete = all_pass and all_synth_pass

    _jdump({
        "stage": "E4RE_PAIRWISE_CG_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "E4RE_COMPLETE": "YES" if e4re_complete else "NO",
        "CONTROL": C1_ARM,
        "TREATMENT": C3_ARM,
        "CAUSAL_FACTOR": "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE",
        "C1_CONFIG_SHA256": FROZEN_QMIX_SHA,
        "C3_CONFIG_SHA256": FROZEN_C3_SHA,
        "ORIGINAL_E4RT_PROTOCOL_SHA256": ORIGINAL_E4RT_SHA,
        "E4RTA_AMENDED_PROTOCOL_SHA256": E4RTA_AMENDMENT_SHA,
        "E4RE_PROTOCOL_SHA256": protocol_sha,
        "TRAINING_LOCK": "10800_COMPLETE",
        "VALIDATION_LOCK": "3600_COMPLETE_NONCONFIRMATORY",
        "SELECTED_CHECKPOINTS": 10,
        "SELECTED_C1": {str(p): SELECTED_C1[p] for p in POLICY_SEEDS},
        "SELECTED_C3": {str(p): SELECTED_C3[p] for p in POLICY_SEEDS},
        "CHECKPOINT_HASH_INTEGRITY": "PASS" if hash_ok else "FAIL",
        "FINAL_SEEDS": "2581..2640",
        "FINAL_ACCESS_COUNT": 0,
        "FINAL_MATRIX": 1800,
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_response",
        "PRIMARY_EFFECT": "C1_MINUS_C3",
        "POSITIVE_MEANS": "C3_BETTER",
        "PRACTICAL_MARGIN": DELTA_PRACTICAL,
        "BOOTSTRAP": "PAIRED_CROSSED_5x60",
        "BOOTSTRAP_N": BOOTSTRAP_N,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "POLICY_ROBUSTNESS": f"AT_LEAST_{ROBUSTNESS_THRESHOLD}_OF_5_POSITIVE",
        "PERSISTENT_SAFETY": "RESPONSE_AND_EPOCH_NOT_HARMFUL",
        "STATIONARY_SAFETY": "RESPONSE_AND_EPOCH_NOT_HARMFUL",
        "MACRO_SAFETY": "RESPONSE_AND_EPOCH_NOT_HARMFUL",
        "VALIDATION_USED_TO_CHANGE_PROTOCOL": False,
        "NO_FINAL_DATA_ADAPTATION": "FROZEN",
        "STATISTICAL_SYNTHETIC_PASS": synth_ok,
        "CLASSIFIER_SYNTHETIC_PASS": all(r["PASS"] for r in classifier_results),
        "VERDICT_SYNTHETIC_PASS": all(r["PASS"] for r in verdict_results),
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": "E4RF_PAIRWISE_CG_AB_FINAL_EVALUATION_EXECUTION"
        if e4re_complete else "E4RE_INTEGRITY_REVIEW",
    }, E4RE_DIR / "E4RE_verdict.json")

    # ================================================================
    # REPORT
    # ================================================================
    report = f"""# E4RE — Pairwise CG A/B Final Evaluation Protocol Freeze

## A. Training/Validation Completion
- Training: 10800/10800 episodes, 60/60 checkpoints
- Validation: 3600/3600 episodes (NON_CONFIRMATORY)
- Restart version: E4RTR_BATCHED_RESTART_V2

## B. Selected Checkpoint Lock
- 10 checkpoints frozen (5 C1 + 5 C3)
- C1: all ckpt_final
- C3: 7001:ckpt_50000, 7002:ckpt_final, 7003:ckpt_final, 7004:ckpt_40000, 7005:ckpt_40000
- All hashes verified against post-validation state

## C. Final Seed Firewall
- Seeds: 2581..2640 (60 environments)
- Access count: 0 (NEVER_ACCESSED)
- No smoke testing, dry run, timing, or debugging on final seeds

## D. Final Matrix
- 2 arms × 5 policies × 60 environments × 3 scenarios = 1800 episodes
- Require 1800/1800 complete

## E. Paired Evaluation Design
- Primary unit: policy × environment × scenario
- Both arms evaluated under matched deterministic environment

## F. C1/C3 Final Action Paths
- C1: shared utility → independent regional greedy argmax (no mixer)
- C3: complete pairwise graph + centered pair factors + Max-Sum K=10

## G. Primary MIXED Endpoint
- Scenario: MIXED
- Endpoint: MIGD_response
- Effect: C1 - C3 (positive = C3 better)

## H. Practical Margin
- DELTA_PRACTICAL = 1e-5
- No retuning

## I. Crossed Bootstrap
- Paired crossed two-axis: 5 policies × 60 environments
- N = 50000, seed = 136001
- 95% percentile CI
- Shared draws across scenarios

## J. Policy Robustness
- PASS if ≥ 4 of 5 policy-level deltas > 0

## K. Persistent Safety
- PASS if neither MIGD_response nor MIGD_epoch is HARMFUL

## L. Stationary Safety
- PASS if neither endpoint is HARMFUL
- Validation descriptively favored C1 on STATIONARY; this does NOT change the gate

## M. Macro Safety
- Equal-weight (PERSISTENT + STATIONARY + MIXED) / 3
- PASS if neither endpoint is HARMFUL

## N. Classification Logic
- BENEFICIAL: delta > 1e-5 AND CI_lower > 0
- DIRECTIONAL_POSITIVE: delta > 1e-5 AND CI crosses 0
- PRACTICALLY_NEUTRAL: |delta| ≤ 1e-5
- DIRECTIONAL_NEGATIVE: delta < -1e-5 AND CI crosses 0
- HARMFUL: delta < -1e-5 AND CI_upper < 0

## O. Causal Verdict Logic
- VALIDATED: all gates pass (BENEFICIAL + robustness + safety + integrity)
- PARTIAL: safety pass + (DIRECTIONAL_POSITIVE or BENEFICIAL+robustness_fail or NEUTRAL)
- FAILED: any of NEGATIVE/HARMFUL/safety_fail/integrity_fail

## P. FE Reporting Boundary
- FE is descriptive/efficiency context
- Lower FE does NOT rescue HARMFUL; higher FE does NOT invalidate treatment

## Q. Action/Mechanism Diagnostics
- Entropy, multi-action rate, Max-Sum regret: non-gating
- Cannot alter final verdict

## R. Validation Non-Confirmatory Boundary
- Validation favored: C3 on PERSISTENT/MIXED/macro, C1 on STATIONARY
- This information MUST NOT change any frozen protocol parameter

## S. Checkpoint/Config Integrity
- Pre/post hash required for all 10 selected checkpoints
- Config SHAs frozen: C1={FROZEN_QMIX_SHA[:16]}..., C3={FROZEN_C3_SHA[:16]}...

## T. Engineering Lineage
- Batched Max-Sum implementation: {BATCHED_IMPL_SHA[:16]}...
- Float32 tie-only mismatch: 0.1082% (non-tie = 0)

## U. No-Adaptation Rule
- After seed 2581 opened: no retraining, reselection, architecture/K/graph/reward/state/operator/bootstrap/endpoint/safety changes

## V. Future Routing
- VALIDATED → E4S_PAIRWISE_COORDINATION_VS_C0_DESIGN (C3>C1 does NOT imply C3>C0)
- PARTIAL → E4S_PAIRWISE_COORDINATION_RESULT_REVIEW
- FAILED → E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS (no post-hoc tuning)

## W. Limitations
- E4RE is protocol freeze only; no final outcome observed
- Validation descriptively favored C3 overall but is non-confirmatory
- Validation descriptively favored C1 on STATIONARY; this does not change the preregistered safety gate
- C3 bundles pairwise representation and coordinated inference
- Max-Sum remains loopy approximate
- Training seeds reused after engineering-only interruption
- Validation seeds were first-use in E4RTR V2
- Final seeds remain untouched
- Float32 tie-only engineering discrepancy is documented
- Pairwise modeling does not capture arbitrary higher-order interactions
- No real-world UAV validation is performed

## X. Final Execution Authorization
- E4RE protocol is frozen
- E4RF may now execute the final evaluation on seeds 2581..2640
- The result must be accepted whatever it is
"""
    with open(E4RE_DIR / "E4RE_report.md", "w") as f:
        f.write(report)

    dt = time.time() - t0
    print(f"[E4RE] Protocol freeze complete in {dt:.1f}s")
    print(f"[E4RE] E4RE_COMPLETE = {'YES' if e4re_complete else 'NO'}")
    print(f"[E4RE] NEXT = {'E4RF' if e4re_complete else 'E4RE_INTEGRITY_REVIEW'}")


if __name__ == "__main__":
    main()
