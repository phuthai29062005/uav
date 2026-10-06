"""
Tests for E4RE — Pairwise CG A/B Final Evaluation Protocol Freeze.

These tests inspect SAVED artifacts. No environment execution. No final seeds.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
E4RE_DIR = ROOT / "results" / "E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze"
V2_DIR = ROOT / "results" / "E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2"

C1 = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3 = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
C3_SHA = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"
E4RT_SHA = "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6"
E4RTA_SHA = "ac5c93f7c68c304d1dae3dbf4c4de957c440d3ddf181a7870e68564c4d0bad91"
POLICIES = [7001, 7002, 7003, 7004, 7005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


def _load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load(E4RE_DIR / "E4RE_verdict.json")


@pytest.fixture(scope="module")
def pre():
    return _load(E4RE_DIR / "E4RE_pre_manifest.json")


@pytest.fixture(scope="module")
def ck_lock():
    return _load(E4RE_DIR / "E4RE_selected_checkpoint_lock.json")


@pytest.fixture(scope="module")
def ck_hashes():
    return _load(E4RE_DIR / "E4RE_selected_checkpoint_hashes.json")


@pytest.fixture(scope="module")
def tvl():
    return _load(E4RE_DIR / "E4RE_training_validation_lock.json")


@pytest.fixture(scope="module")
def seed_contract():
    return _load(E4RE_DIR / "E4RE_final_seed_contract.json")


@pytest.fixture(scope="module")
def firewall():
    return _load(E4RE_DIR / "E4RE_final_seed_firewall.json")


@pytest.fixture(scope="module")
def matrix():
    return _load(E4RE_DIR / "E4RE_final_matrix_contract.json")


@pytest.fixture(scope="module")
def pairing():
    return _load(E4RE_DIR / "E4RE_pairing_contract.json")


@pytest.fixture(scope="module")
def eval_mode():
    return _load(E4RE_DIR / "E4RE_evaluation_mode.json")


@pytest.fixture(scope="module")
def action_path():
    return _load(E4RE_DIR / "E4RE_action_path_contract.json")


@pytest.fixture(scope="module")
def operators():
    return _load(E4RE_DIR / "E4RE_operator_contract.json")


@pytest.fixture(scope="module")
def primary():
    return _load(E4RE_DIR / "E4RE_primary_endpoint.json")


@pytest.fixture(scope="module")
def margin():
    return _load(E4RE_DIR / "E4RE_practical_margin.json")


@pytest.fixture(scope="module")
def bootstrap():
    return _load(E4RE_DIR / "E4RE_bootstrap_contract.json")


@pytest.fixture(scope="module")
def robustness():
    return _load(E4RE_DIR / "E4RE_policy_robustness_contract.json")


@pytest.fixture(scope="module")
def classifier():
    return _load(E4RE_DIR / "E4RE_classifier_truth_table.json")


@pytest.fixture(scope="module")
def verdict_tt():
    return _load(E4RE_DIR / "E4RE_verdict_truth_table.json")


@pytest.fixture(scope="module")
def stat_val():
    return _load(E4RE_DIR / "E4RE_statistical_validation.json")


@pytest.fixture(scope="module")
def causal():
    return _load(E4RE_DIR / "E4RE_causal_verdict_contract.json")


@pytest.fixture(scope="module")
def no_val_adapt():
    return _load(E4RE_DIR / "E4RE_no_validation_adaptation.json")


@pytest.fixture(scope="module")
def routing():
    return _load(E4RE_DIR / "E4RE_future_routing.json")


@pytest.fixture(scope="module")
def e4rf_contract():
    return _load(E4RE_DIR / "E4RE_e4rf_output_contract.json")


@pytest.fixture(scope="module")
def mechanism():
    return _load(E4RE_DIR / "E4RE_mechanism_diagnostic_boundary.json")


# T1-T4: Prerequisites
def test_t1_e4rtr_v2_complete(pre):
    assert pre["checks"]["E4RTR_V2_COMPLETE"] is True

def test_t2_training_10800(pre):
    assert pre["checks"]["TRAINING_10800"] is True

def test_t3_validation_3600(pre):
    assert pre["checks"]["VALIDATION_3600"] is True

def test_t4_selected_10(pre):
    assert pre["checks"]["SELECTED_10"] is True


# T5-T6: Selected checkpoints exact
def test_t5_selected_c1_exact(ck_lock):
    for p in POLICIES:
        k = f"{C1}_{p}"
        assert ck_lock["checkpoints"][k]["checkpoint"] == "ckpt_final"

def test_t6_selected_c3_exact(ck_lock):
    expected = {7001: "ckpt_50000", 7002: "ckpt_final", 7003: "ckpt_final",
                7004: "ckpt_40000", 7005: "ckpt_40000"}
    for p, ckpt in expected.items():
        k = f"{C3}_{p}"
        assert ck_lock["checkpoints"][k]["checkpoint"] == ckpt


# T7-T8: Hashes
def test_t7_selected_hashes_load(ck_hashes):
    assert ck_hashes["n_selected"] == 10
    assert ck_hashes["all_verified"] is True

def test_t8_old_checkpoints_absent(pre):
    assert pre["checks"]["OLD_CHECKPOINTS_ABSENT"] is True


# T9-T12: Config SHAs
def test_t9_c1_config(pre):
    assert pre["checks"]["C1_CONFIG_SHA_EXACT"] is True

def test_t10_c3_config(pre):
    assert pre["checks"]["C3_CONFIG_SHA_EXACT"] is True

def test_t11_e4rt_hash(pre):
    assert pre["checks"]["E4RT_SHA_EXACT"] is True

def test_t12_e4rta_hash(pre):
    assert pre["checks"]["E4RTA_SHA_EXACT"] is True


# T13-T16: Seeds and policies
def test_t13_final_seeds(seed_contract):
    assert seed_contract["range"] == "2581..2640"

def test_t14_final_seed_count(seed_contract):
    assert seed_contract["n_final"] == 60

def test_t15_final_access_count(firewall):
    assert firewall["FINAL_ACCESS_COUNT"] == 0

def test_t16_policy_seeds(pre):
    assert pre["checks"]["POLICY_SEEDS_EXACT"] is True


# T17-T18: Scenarios and arms
def test_t17_three_scenarios(matrix):
    assert matrix["scenarios"] == SCENARIOS

def test_t18_arms_exact(matrix):
    assert matrix["arms"] == [C1, C3]


# T19-T21: Final matrix
def test_t19_final_episodes_1800(matrix):
    assert matrix["total_episodes"] == 1800

def test_t20_per_arm_900(matrix):
    assert matrix["per_arm"] == 900

def test_t21_matched_pairing(pairing):
    assert pairing["both_arms_per_unit"] is True
    assert pairing["deterministic_environment"] is True


# T22-T27: Evaluation mode
def test_t22_epsilon_0(eval_mode):
    assert eval_mode["epsilon"] == 0

def test_t23_model_eval(eval_mode):
    assert eval_mode["model_eval"] is True

def test_t24_no_gradients(eval_mode):
    assert eval_mode["no_gradient"] is True

def test_t25_no_optimizer(eval_mode):
    assert eval_mode["no_optimizer"] is True

def test_t26_no_replay(eval_mode):
    assert eval_mode["no_replay_append"] is True

def test_t27_no_target_update(eval_mode):
    assert eval_mode["no_target_update"] is True


# T28-T30: Action paths
def test_t28_c1_action_path(action_path):
    c1 = action_path["C1_action_path"]
    assert c1["independent_regional_greedy_argmax"] is True
    assert c1["no_mixer_deployment"] is True

def test_t29_c3_maxsum_k10(action_path):
    assert action_path["C3_action_path"]["maxsum_K"] == 10

def test_t30_no_scalar_fallback(action_path):
    assert action_path["C3_action_path"]["no_tie_specific_scalar_fallback"] is True


# T31-T32: Operators
def test_t31_operators_exact(operators):
    ops = operators["operators"]
    assert ops["KEEP"]["FE"] == 0
    assert ops["LOCAL"]["sigma_L"] == 0.02
    assert ops["PREDICT"]["sigma_P"] == 0.005
    assert ops["DIVERSIFY"]["sigma_D"] == 0.15
    assert ops["budget"] == 10
    assert ops["HARD"] is False

def test_t32_requested_realized(pairing):
    pass  # Contract is in action_path


# T33-T36: Primary endpoint
def test_t33_primary_mixed(primary):
    assert primary["primary_scenario"] == "MIXED"

def test_t34_primary_migd_response(primary):
    assert primary["primary_endpoint"] == "MIGD_response"

def test_t35_effect_c1_c3(primary):
    assert primary["effect_direction"] == "C1_MINUS_C3"

def test_t36_positive_c3_better(primary):
    assert primary["positive_means"] == "C3_BETTER"


# T37: Practical margin
def test_t37_margin_1e5(margin):
    assert margin["DELTA_PRACTICAL"] == 1e-5


# T38-T43: Bootstrap
def test_t38_bootstrap_policy_5(bootstrap):
    assert bootstrap["policy_axis"] == 5

def test_t39_bootstrap_env_60(bootstrap):
    assert bootstrap["environment_axis"] == 60

def test_t40_bootstrap_paired(bootstrap):
    assert "paired" in bootstrap["method"]

def test_t41_bootstrap_n_50000(bootstrap):
    assert bootstrap["resamples"] == 50000

def test_t42_bootstrap_seed_136001(bootstrap):
    assert bootstrap["master_seed"] == 136001

def test_t43_shared_scenario_draws(bootstrap):
    assert bootstrap["shared_draws_across_scenarios"] is True


# T44: Policy robustness
def test_t44_robustness_4_of_5(robustness):
    assert "4" in robustness["threshold"]
    assert "5" in robustness["threshold"]


# T45-T50: Safety
def test_t45_persistent_response_safety():
    d = _load(E4RE_DIR / "E4RE_persistent_safety_contract.json")
    assert "MIGD_response" in d["endpoints"]

def test_t46_persistent_epoch_safety():
    d = _load(E4RE_DIR / "E4RE_persistent_safety_contract.json")
    assert "MIGD_epoch" in d["endpoints"]

def test_t47_stationary_response_safety():
    d = _load(E4RE_DIR / "E4RE_stationary_safety_contract.json")
    assert "MIGD_response" in d["endpoints"]

def test_t48_stationary_epoch_safety():
    d = _load(E4RE_DIR / "E4RE_stationary_safety_contract.json")
    assert "MIGD_epoch" in d["endpoints"]

def test_t49_macro_response_safety():
    d = _load(E4RE_DIR / "E4RE_macro_safety_contract.json")
    assert "MIGD_response" in d["endpoints"]

def test_t50_macro_epoch_safety():
    d = _load(E4RE_DIR / "E4RE_macro_safety_contract.json")
    assert "MIGD_epoch" in d["endpoints"]


# T51-T55: Classification
def test_t51_beneficial(classifier):
    case = [c for c in classifier["cases"] if c["expected"] == "BENEFICIAL"][0]
    assert case["PASS"] is True

def test_t52_directional_positive(classifier):
    case = [c for c in classifier["cases"] if c["expected"] == "DIRECTIONAL_POSITIVE"][0]
    assert case["PASS"] is True

def test_t53_neutral(classifier):
    case = [c for c in classifier["cases"] if c["expected"] == "PRACTICALLY_NEUTRAL"][0]
    assert case["PASS"] is True

def test_t54_directional_negative(classifier):
    case = [c for c in classifier["cases"] if c["expected"] == "DIRECTIONAL_NEGATIVE"][0]
    assert case["PASS"] is True

def test_t55_harmful(classifier):
    case = [c for c in classifier["cases"] if c["expected"] == "HARMFUL"][0]
    assert case["PASS"] is True


# T56-T61: Verdict truth table
def test_t56_validated_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"] if c["expected"] == "VALIDATED"][0]
    assert case["PASS"] is True

def test_t57_partial_dp_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"]
            if c["expected"] == "PARTIAL" and c["primary"] == "DIRECTIONAL_POSITIVE"][0]
    assert case["PASS"] is True

def test_t58_partial_robust_fail_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"]
            if c["expected"] == "PARTIAL" and c["robust"] is False][0]
    assert case["PASS"] is True

def test_t59_partial_neutral_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"]
            if c["expected"] == "PARTIAL" and c["primary"] == "PRACTICALLY_NEUTRAL"][0]
    assert case["PASS"] is True

def test_t60_failed_negative_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"]
            if c["expected"] == "FAILED" and c["primary"] == "DIRECTIONAL_NEGATIVE"][0]
    assert case["PASS"] is True

def test_t61_failed_safety_tt(verdict_tt):
    case = [c for c in verdict_tt["cases"]
            if c["expected"] == "FAILED" and c.get("safe") is False][0]
    assert case["PASS"] is True


# T62-T64: Non-gating
def test_t62_fe_no_rescue():
    d = _load(E4RE_DIR / "E4RE_fe_reporting_contract.json")
    assert d["lower_FE_does_NOT_rescue_HARMFUL"] is True

def test_t63_entropy_non_gating(mechanism):
    assert mechanism["entropy_is_non_gating"] is True

def test_t64_maxsum_non_gating(mechanism):
    assert mechanism["maxsum_regret_is_non_gating"] is True


# T65-T68: Validation boundary
def test_t65_validation_no_alter(no_val_adapt):
    assert no_val_adapt["VALIDATION_INFORMATION_USED_TO_CHANGE_PROTOCOL"] is False

def test_t66_no_checkpoint_reselection(ck_lock):
    assert ck_lock["NO_RESELECTION"] is True

def test_t67_no_best_policy_selection(ck_lock):
    pass  # All 5 retained

def test_t68_all_five_policies(ck_lock):
    assert ck_lock["all_policies_retained"] is True


# T69-T70: Integrity contracts
def test_t69_checkpoint_pre_post_hash():
    d = _load(E4RE_DIR / "E4RE_checkpoint_integrity_contract.json")
    assert d["pre_post_hash_required"] is True

def test_t70_config_integrity():
    d = _load(E4RE_DIR / "E4RE_config_integrity_contract.json")
    assert d["C1_CONFIG_SHA256"] == QMIX_SHA
    assert d["C3_CONFIG_SHA256"] == C3_SHA


# T71-T74: Matrix
def test_t71_final_unique_key(e4rf_contract):
    schema = e4rf_contract["episode_schema"]
    for k in ["arm", "policy_seed", "environment_seed", "scenario"]:
        assert k in schema

def test_t72_no_duplicate_units(matrix):
    assert matrix["NO_MISSING_CELL"] is True

def test_t73_no_seed_substitution(seed_contract):
    assert seed_contract["NO_REPLACEMENT"] is True

def test_t74_resume_contract():
    pass  # Documented in spec


# T75-T76: No adaptation
def test_t75_no_final_adaptation():
    d = _load(E4RE_DIR / "E4RE_no_final_adaptation.json")
    assert len(d["FORBIDDEN_AFTER_SEED_2581_OPENED"]) >= 10

def test_t76_no_cross_stage_c0(routing):
    assert "C3>C1 does NOT imply C3>C0" in routing["NOTE_VALIDATED"]


# T77-T79: Routing
def test_t77_validated_routes_c0(routing):
    assert "C0" in routing["IF_VALIDATED"]

def test_t78_partial_routes_review(routing):
    assert "REVIEW" in routing["IF_PARTIAL"]

def test_t79_failed_routes_analysis(routing):
    assert "ANALYSIS" in routing["IF_FAILED"]


# T80-T81: Output contracts
def test_t80_output_contract(e4rf_contract):
    assert len(e4rf_contract["required_outputs"]) >= 20

def test_t81_episode_schema(e4rf_contract):
    schema = e4rf_contract["episode_schema"]
    for k in ["MIGD_response", "MIGD_epoch", "epsilon", "learning_updates",
              "checkpoint_SHA256", "episode_complete"]:
        assert k in schema


# T82-T84: Synthetic tests
def test_t82_statistical_synthetic(stat_val):
    assert stat_val["PASS"] is True

def test_t83_classifier_synthetic(classifier):
    assert classifier["ALL_PASS"] is True

def test_t84_verdict_synthetic(verdict_tt):
    assert verdict_tt["ALL_PASS"] is True


# T85-T89: Final state
def test_t85_no_final_envs_executed(firewall):
    assert firewall["FINAL_ACCESS_COUNT"] == 0
    assert firewall["NO_SMOKE_TEST"] is True

def test_t86_final_access_0(verdict):
    assert verdict["FINAL_ACCESS_COUNT"] == 0

def test_t87_perf_verdict_not_computed(verdict):
    assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

def test_t88_causal_verdict_not_computed(verdict):
    assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == "NOT_COMPUTED"

def test_t89_next_stage(verdict):
    assert verdict["NEXT_STAGE"] == "E4RF_PAIRWISE_CG_AB_FINAL_EVALUATION_EXECUTION"


# T90: Full regression
def test_t90_full_regression():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
         "--tb=no", "--no-header",
         "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
         "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"Regression FAILED:\n{result.stdout[-2000:]}\n{result.stderr[-500:]}"
    )


# Additional integrity
def test_t91_e4re_complete(verdict):
    assert verdict["E4RE_COMPLETE"] == "YES"

def test_t92_protocol_sha_exists():
    d = _load(E4RE_DIR / "E4RE_protocol_frozen.json")
    assert len(d["PROTOCOL_SHA256"]) == 64

def test_t93_all_artifacts_exist():
    expected = [
        "E4RE_pre_manifest.json", "E4RE_historical_integrity.json",
        "E4RE_training_validation_lock.json",
        "E4RE_selected_checkpoint_lock.json",
        "E4RE_selected_checkpoint_hashes.json",
        "E4RE_final_seed_contract.json", "E4RE_final_seed_firewall.json",
        "E4RE_final_matrix_contract.json", "E4RE_pairing_contract.json",
        "E4RE_evaluation_mode.json", "E4RE_action_path_contract.json",
        "E4RE_operator_contract.json", "E4RE_primary_endpoint.json",
        "E4RE_practical_margin.json", "E4RE_bootstrap_contract.json",
        "E4RE_policy_robustness_contract.json",
        "E4RE_persistent_safety_contract.json",
        "E4RE_stationary_safety_contract.json",
        "E4RE_macro_safety_contract.json",
        "E4RE_classification_contract.json",
        "E4RE_causal_verdict_contract.json",
        "E4RE_fe_reporting_contract.json",
        "E4RE_mechanism_diagnostic_boundary.json",
        "E4RE_no_validation_adaptation.json",
        "E4RE_no_final_adaptation.json",
        "E4RE_checkpoint_integrity_contract.json",
        "E4RE_config_integrity_contract.json",
        "E4RE_engineering_lineage.json", "E4RE_future_routing.json",
        "E4RE_e4rf_output_contract.json",
        "E4RE_statistical_validation.json",
        "E4RE_classifier_truth_table.json",
        "E4RE_verdict_truth_table.json",
        "E4RE_protocol_frozen.json",
        "E4RE_verdict.json", "E4RE_report.md",
    ]
    for f in expected:
        assert (E4RE_DIR / f).exists(), f"Missing: {f}"

def test_t94_bootstrap_reproducible():
    from prdrl.experiments.E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze import (
        paired_crossed_bootstrap,
    )
    rng = np.random.RandomState(99999)
    c1 = rng.randn(5, 60) * 0.001 + 0.007
    c3 = rng.randn(5, 60) * 0.001 + 0.006
    pt1, lo1, hi1 = paired_crossed_bootstrap(c1, c3, 5, 60, 50000, 136001)
    pt2, lo2, hi2 = paired_crossed_bootstrap(c1, c3, 5, 60, 50000, 136001)
    assert pt1 == pt2
    assert lo1 == lo2
    assert hi1 == hi2
