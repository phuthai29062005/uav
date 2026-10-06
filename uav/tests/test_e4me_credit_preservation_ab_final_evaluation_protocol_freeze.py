"""
Tests for E4ME — Credit Preservation A/B Final Evaluation Protocol Freeze.

Validates frozen artifacts WITHOUT running environments, training, or final seeds.
NO recursive subprocess regression test (per spec: run externally).
"""

import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "E4ME_credit_preservation_ab_final_evaluation_protocol_freeze"
E4MTR_DIR = BASE / "results" / "E4MTR_credit_preservation_ab_training_execution"
E4MT_DIR = BASE / "results" / "E4MT_credit_preservation_ab_training_protocol_freeze"
E4MI_DIR = BASE / "results" / "E4MI_joint_credit_preservation_implementation_validation"

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2_ARM = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
FROZEN_QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C2_CREDIT_SHA = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"
FROZEN_R1_SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
FROZEN_V3_SHA = "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"


def _load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load("E4ME_verdict.json")


@pytest.fixture(scope="module")
def protocol():
    return _load("E4ME_protocol_frozen.json")


@pytest.fixture(scope="module")
def checkpoint_freeze():
    return _load("E4ME_checkpoint_freeze.json")


@pytest.fixture(scope="module")
def synth():
    return _load("E4ME_synthetic_validation.json")


@pytest.fixture(scope="module")
def val_lock():
    return _load("E4ME_validation_observation_lock.json")


@pytest.fixture(scope="module")
def loss_scale():
    return _load("E4ME_loss_scale_observation_lock.json")


@pytest.fixture(scope="module")
def loss_spec():
    return _load("E4ME_loss_spec_integrity.json")


@pytest.fixture(scope="module")
def success_logic():
    return _load("E4ME_success_logic.json")


@pytest.fixture(scope="module")
def classification():
    return _load("E4ME_classification_logic.json")


@pytest.fixture(scope="module")
def bootstrap_contract():
    return _load("E4ME_paired_bootstrap_contract.json")


@pytest.fixture(scope="module")
def matrix():
    return _load("E4ME_final_matrix_contract.json")


@pytest.fixture(scope="module")
def effect_def():
    return _load("E4ME_effect_definition.json")


@pytest.fixture(scope="module")
def safety_p():
    return _load("E4ME_persistent_safety.json")


@pytest.fixture(scope="module")
def safety_s():
    return _load("E4ME_stationary_safety.json")


@pytest.fixture(scope="module")
def safety_m():
    return _load("E4ME_macro_safety.json")


@pytest.fixture(scope="module")
def robustness():
    return _load("E4ME_policy_robustness.json")


@pytest.fixture(scope="module")
def routing():
    return _load("E4ME_routing.json")


@pytest.fixture(scope="module")
def arch():
    return _load("E4ME_architecture_integrity.json")


@pytest.fixture(scope="module")
def vli():
    return _load("E4ME_validation_lock_invariance.json")


# ---- T1-T4: Prerequisites ----

def test_T1_e4mtr_complete(verdict):
    assert verdict["E4MTR_COMPLETE"] == "YES"


def test_T2_e4mi_valid():
    mi = _load("E4ME_pre_manifest.json")
    assert mi["CREDIT_PRESERVATION_IMPLEMENTATION_VALID"] == "YES"


def test_T3_e4kf_failed_preserved(verdict):
    assert verdict["ORIGINAL_E4KF_VERDICT"] in ("FAILED", "COORDINATION_CAUSAL_FAILED")


def test_T4_h4_supported(verdict):
    assert verdict["H4_STATUS"] == "SUPPORTED"


# ---- T5-T9: Checkpoint freeze ----

def test_T5_exact_ten_selected(checkpoint_freeze):
    assert checkpoint_freeze["n_checkpoints"] == 10


def test_T6_five_c1(checkpoint_freeze):
    assert checkpoint_freeze["n_c1"] == 5


def test_T7_five_c2(checkpoint_freeze):
    assert checkpoint_freeze["n_c2"] == 5


def test_T8_hashes_exact(checkpoint_freeze):
    for key, info in checkpoint_freeze["checkpoints"].items():
        assert info["hash_match"] is True, f"{key} hash mismatch"


def test_T9_no_checkpoint_reselection(checkpoint_freeze):
    assert checkpoint_freeze["no_reselection"] is True


# ---- T10-T13: Architecture ----

def test_T10_c1_architecture_exact(arch):
    assert arch["C1_CONTROL"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"


def test_T11_c2_architecture_exact(arch):
    assert arch["C2_TREATMENT"] == C2_ARM


def test_T12_architecture_identity(arch):
    assert arch["parameter_count_identity"] is True


def test_T13_parameter_count_identity(arch):
    assert arch["both_arms_same_utility"] is True
    assert arch["both_arms_same_mixer"] is True


# ---- T14-T18: Loss spec ----

def test_T14_c1_loss_joint_only(loss_spec):
    assert loss_spec["C1_loss"] == "L_C1 = L_joint"


def test_T15_c2_joint_component(loss_spec):
    assert loss_spec["joint_weight"] == 0.5


def test_T16_c2_regional_component(loss_spec):
    assert loss_spec["regional_weight"] == 0.5


def test_T17_c2_half_half_exact(loss_spec):
    assert loss_spec["C2_loss"] == "L_C2 = 0.5*L_joint + 0.5*L_regional"


def test_T18_half_half_immutable(loss_spec):
    assert loss_spec["weights_tunable"] is False
    assert loss_spec["weights_immutable"] is True


# ---- T19: Mixer-gradient consequence ----

def test_T19_mixer_gradient_consequence(loss_spec):
    assert "0.5*L_joint only" in loss_spec["mixer_gradient_consequence"]


# ---- T20-T24: V3/R1/QMIX/C2 ----

def test_T20_v3_exact(checkpoint_freeze):
    for key, info in checkpoint_freeze["checkpoints"].items():
        assert info["normalization_metadata_ok"] is True, f"{key} V3 mismatch"


def test_T21_r1_exact(checkpoint_freeze):
    for key, info in checkpoint_freeze["checkpoints"].items():
        assert info["reward_metadata_ok"] is True, f"{key} R1 mismatch"


def test_T22_alpha_01():
    reward = _load("E4ME_reward_integrity.json")
    assert reward["alpha_local_both_arms"] == 0.1


def test_T23_qmix_exact(verdict):
    assert verdict["QMIX_CONFIG_INTEGRITY"] == "PASS"


def test_T24_c2_credit_hash_exact(verdict):
    assert verdict["C2_CREDIT_CONFIG_INTEGRITY"] == "PASS"


# ---- T25-T26: Replay ----

def test_T25_replay_exact():
    hist = _load("E4ME_historical_integrity.json")
    assert hist["no_post_e4mtr_training"] is True


def test_T26_no_sibling_dedup():
    replay = json.load(open(E4MT_DIR / "E4MT_replay_lock.json"))
    assert replay["sibling_deduplication"] is False


# ---- T27-T30: Validation/loss-scale locks ----

def test_T27_validation_observation_locked(val_lock):
    assert val_lock["status"] == "NON_CONFIRMATORY_LOCKED"


def test_T28_validation_near_zero_cannot_change_protocol(val_lock):
    assert "composite_weights" in val_lock["must_not_alter"]
    assert "primary_endpoint" in val_lock["must_not_alter"]


def test_T29_loss_scale_descriptive_only(loss_scale):
    assert loss_scale["status"] == "DESCRIPTIVE_NON_CONFIRMATORY_LOCKED"


def test_T30_loss_scale_cannot_change_weights(loss_scale):
    assert loss_scale["cannot_tune_weights"] is True
    assert loss_scale["cannot_rescale_regional"] is True


# ---- T31-T37: Final seeds/matrix ----

def test_T31_final_seeds_2181_2240():
    seeds = _load("E4ME_final_seed_contract.json")
    assert seeds["range"] == [2181, 2240]
    assert seeds["n"] == 60
    assert seeds["final_seeds"] == list(range(2181, 2241))


def test_T32_final_access_zero(verdict):
    assert verdict["M_FINAL_ACCESS_COUNT"] == 0


def test_T33_scenarios_exact():
    ep = _load("E4ME_endpoint_contract.json")
    matrix = _load("E4ME_final_matrix_contract.json")
    assert matrix["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T34_final_matrix_1800(matrix):
    assert matrix["total_episodes"] == 1800


def test_T35_c1_900(matrix):
    assert matrix["C1_rows"] == 900


def test_T36_c2_900(matrix):
    assert matrix["C2_rows"] == 900


def test_T37_pairing_exact():
    pair = _load("E4ME_pairing_contract.json")
    assert set(pair["pair_by"]) == {"policy_seed", "environment_seed", "scenario"}


# ---- T38-T41: Primary/effect ----

def test_T38_primary_mixed(verdict):
    assert verdict["PRIMARY_SCENARIO"] == "MIXED"


def test_T39_endpoint_migd_response(verdict):
    assert verdict["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"


def test_T40_effect_c1_minus_c2(effect_def):
    assert "C1" in effect_def["cell_effect"] and "C2" in effect_def["cell_effect"]
    assert effect_def["positive_means"] == "C2_BETTER"


def test_T41_positive_means_c2_better(verdict):
    assert verdict["POSITIVE_MEANS"] == "C2_BETTER"


# ---- T42-T51: Bootstrap/robustness/macro ----

def test_T42_margin_1e5(verdict):
    assert verdict["PRACTICAL_MARGIN"] == 1e-5


def test_T43_crossed_bootstrap(bootstrap_contract):
    assert bootstrap_contract["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"


def test_T44_policy_axis_5(bootstrap_contract):
    assert bootstrap_contract["policy_axis_n"] == 5


def test_T45_env_axis_60(bootstrap_contract):
    assert bootstrap_contract["environment_axis_n"] == 60


def test_T46_bootstrap_n_50000(bootstrap_contract):
    assert bootstrap_contract["N_BOOTSTRAP"] == 50000


def test_T47_bootstrap_seed_106001(bootstrap_contract):
    assert bootstrap_contract["BOOTSTRAP_MASTER_SEED"] == 106001


def test_T48_iid_bootstrap_forbidden(bootstrap_contract):
    assert "naive_300_row_bootstrap" in bootstrap_contract["forbidden"]


def test_T49_independent_arm_forbidden(bootstrap_contract):
    assert "independent_arm_bootstrap" in bootstrap_contract["forbidden"]


def test_T50_macro_one_third():
    mac = _load("E4ME_macro_contract.json")
    assert abs(mac["scenario_weight"] - 1/3) < 1e-12


def test_T51_macro_shared_draws():
    mac = _load("E4ME_macro_contract.json")
    assert mac["shared_bootstrap_draws_across_scenarios"] is True


# ---- T52-T55: Robustness/safety ----

def test_T52_robustness_4_of_5(robustness):
    assert robustness["threshold"] == 4


def test_T53_persistent_safety(safety_p):
    assert safety_p["safety_pass"] == "neither endpoint HARMFUL"


def test_T54_stationary_safety(safety_s):
    assert safety_s["safety_pass"] == "neither endpoint HARMFUL"


def test_T55_macro_safety(safety_m):
    assert safety_m["safety_pass"] == "neither macro endpoint HARMFUL"


# ---- T56-T60: FE/diagnostics/mechanism gate ----

def test_T56_fe_secondary():
    fe = _load("E4ME_fe_contract.json")
    assert fe["secondary"] is True


def test_T57_fe_cannot_rescue():
    fe = _load("E4ME_fe_contract.json")
    assert fe["cannot_rescue_harmful_quality"] is True


def test_T58_action_diagnostics_descriptive():
    ac = _load("E4ME_action_context_contract.json")
    assert ac["descriptive_only"] is True
    assert ac["no_diversity_success_criterion"] is True


def test_T59_q_margin_descriptive():
    qm = _load("E4ME_q_margin_contract.json")
    assert qm["descriptive_only"] is True
    assert qm["no_threshold"] is True


def test_T60_no_credit_mechanism_gate():
    cb = _load("E4ME_credit_diagnostics_boundary.json")
    assert cb["CREDIT_PRESERVATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"


# ---- T61-T66: Verdict logic ----

def test_T61_validated_logic(success_logic):
    req = success_logic["CREDIT_PRESERVATION_CAUSAL_VALIDATED_requires_all"]
    assert any("BENEFICIAL" in r for r in req)
    assert any("robustness" in r.lower() for r in req)
    assert any("safety" in r.lower() for r in req)


def test_T62_performance_partial(success_logic):
    assert "DIRECTIONAL_POSITIVE" in success_logic["PERFORMANCE_PARTIAL"]


def test_T63_neutral_partial(success_logic):
    assert "NOT_SUPPORTED" in success_logic["PRACTICALLY_NEUTRAL"]


def test_T64_failed_primary_negative(success_logic):
    assert "HARMFUL" in success_logic["FAILED"]
    assert "DIRECTIONAL_NEGATIVE" in success_logic["FAILED"]


def test_T65_failed_safety(success_logic):
    assert "safety" in success_logic["FAILED"].lower()


def test_T66_failed_integrity(success_logic):
    assert "integrity" in success_logic["FAILED"].lower()


# ---- T67-T80: Synthetic validation ----

def test_T67_synthetic_zero_effect(synth):
    assert synth["checks"]["1_zero_effect"]["PASS"] is True


def test_T68_synthetic_positive_effect(synth):
    assert synth["checks"]["2_c2_better_const"]["PASS"] is True


def test_T69_synthetic_negative_effect(synth):
    assert synth["checks"]["3_c2_worse_const"]["PASS"] is True


def test_T70_policy_axis_variance(synth):
    assert synth["checks"]["4_policy_axis_variance"]["PASS"] is True


def test_T71_env_axis_variance(synth):
    assert synth["checks"]["5_env_axis_variance"]["PASS"] is True


def test_T72_pairing_synthetic(synth):
    assert synth["checks"]["6_pairing_cancels_common"]["PASS"] is True


def test_T73_macro_synthetic(synth):
    assert synth["checks"]["9_equal_macro_weighting"]["PASS"] is True


def test_T74_robustness4_pass(synth):
    assert synth["checks"]["11_robustness4_PASS"]["PASS"] is True


def test_T75_robustness3_fail(synth):
    assert synth["checks"]["12_robustness3_FAIL"]["PASS"] is True


def test_T76_beneficial_classification(synth):
    assert synth["checks"]["13_beneficial_classification"]["PASS"] is True


def test_T77_directional_positive(synth):
    assert synth["checks"]["14_directional_positive"]["PASS"] is True


def test_T78_neutral(synth):
    assert synth["checks"]["15_practically_neutral"]["PASS"] is True


def test_T79_directional_negative(synth):
    assert synth["checks"]["16_directional_negative"]["PASS"] is True


def test_T80_harmful(synth):
    assert synth["checks"]["17_harmful"]["PASS"] is True


# ---- T81-T88: Verdict branches + overrides ----

def test_T81_validated_branch(synth):
    assert synth["checks"]["18_validated_branch"]["PASS"] is True


def test_T82_partial_branch(synth):
    assert synth["checks"]["19_perf_partial_dirpos"]["PASS"] is True


def test_T83_neutral_partial_branch(synth):
    assert synth["checks"]["20_neutral_partial"]["PASS"] is True


def test_T84_safety_override(synth):
    assert synth["checks"]["22_persistent_safety_override"]["PASS"] is True
    assert synth["checks"]["23_stationary_safety_override"]["PASS"] is True
    assert synth["checks"]["24_macro_safety_override"]["PASS"] is True


def test_T85_integrity_override(synth):
    assert synth["checks"]["25_loss_integrity_override"]["PASS"] is True


def test_T86_diagnostics_cannot_rescue(synth):
    assert synth["checks"]["30_diagnostics_cannot_rescue_harmful"]["PASS"] is True


def test_T87_beneficial_robustness_fail_partial(synth):
    assert synth["checks"]["27_beneficial_robustness_fail_partial"]["PASS"] is True


def test_T88_directional_positive_cannot_validate(synth):
    assert synth["checks"]["28_directional_positive_cannot_validate"]["PASS"] is True


# ---- T89-T90: Lock invariance ----

def test_T89_validation_lock_invariance(vli):
    assert vli["ALL_PASS"] is True


def test_T90_loss_scale_lock_invariance(loss_scale):
    assert loss_scale["cannot_modify_treatment"] is True


# ---- T91-T95: Exclusions ----

def test_T91_no_c0_planned(matrix):
    assert C1_ARM in matrix["arms"]
    assert C2_ARM in matrix["arms"]
    assert "C0" not in str(matrix["arms"])


def test_T92_no_global_comparator(matrix):
    assert "GLOBAL" not in str(matrix["arms"])


def test_T93_no_final_environment_execution(verdict):
    assert verdict["FINAL_EVALUATION"] == "NOT_RUN"
    assert verdict["NEW_ENVIRONMENT_EXECUTION"] == "NO"


def test_T94_no_training(verdict):
    assert verdict["NEW_TRAINING"] == "NO"


def test_T95_no_optimizer_steps():
    exec_contract = _load("E4ME_final_execution_contract.json")
    audit = exec_contract["no_learning_audit"]
    assert audit["optimizer_steps"] == 0
    assert audit["gradient_updates"] == 0
    assert audit["weight_changes"] == 0


# ---- T96-T98: Output schema / routing / historical ----

def test_T96_final_output_schema_complete():
    schema = _load("E4ME_output_schema.json")
    required = schema["e4mf_required_outputs"]
    assert len(required) >= 20
    assert "E4MF_verdict.json" in required
    assert "E4MF_report.md" in required
    assert "E4MF_episode_results.csv" in required


def test_T97_future_result_routing(routing):
    assert "CREDIT_PRESERVATION_CAUSAL_VALIDATED" in routing
    assert "CREDIT_PRESERVATION_CAUSAL_PARTIAL" in routing
    assert "CREDIT_PRESERVATION_CAUSAL_FAILED_integrity_pass" in routing
    assert routing["no_transitive_inference"] is True


def test_T98_historical_artifacts_unchanged():
    hist = _load("E4ME_historical_integrity.json")
    assert hist["E4F"] == "FAILED"
    assert hist["E4HF_normalization"] == "FAILED"
    assert hist["E4IF_reward"] == "VALIDATED"
    assert hist["E4MT_COMPLETE"] == "YES"
    assert hist["E4MTR_COMPLETE"] == "YES"
    assert hist["E4MI_COMPLETE"] == "YES"


# ---- Additional: overall completeness ----

def test_e4me_complete(verdict):
    assert verdict["E4ME_COMPLETE"] == "YES"


def test_synth_all_pass(synth):
    assert synth["ALL_PASS"] is True
    assert synth["n_checks"] == 30


def test_all_required_files_exist():
    required = [
        "E4ME_pre_manifest.json", "E4ME_historical_integrity.json",
        "E4ME_checkpoint_freeze.json", "E4ME_architecture_integrity.json",
        "E4ME_loss_spec_integrity.json", "E4ME_qmix_config_integrity.json",
        "E4ME_credit_config_integrity.json", "E4ME_normalization_integrity.json",
        "E4ME_reward_integrity.json", "E4ME_validation_observation_lock.json",
        "E4ME_loss_scale_observation_lock.json", "E4ME_final_seed_contract.json",
        "E4ME_final_matrix_contract.json", "E4ME_pairing_contract.json",
        "E4ME_execution_order.json", "E4ME_endpoint_contract.json",
        "E4ME_effect_definition.json", "E4ME_practical_margin.json",
        "E4ME_classification_logic.json", "E4ME_paired_bootstrap_contract.json",
        "E4ME_macro_contract.json", "E4ME_policy_robustness.json",
        "E4ME_persistent_safety.json", "E4ME_stationary_safety.json",
        "E4ME_macro_safety.json", "E4ME_fe_contract.json",
        "E4ME_action_context_contract.json", "E4ME_q_margin_contract.json",
        "E4ME_credit_diagnostics_boundary.json", "E4ME_success_logic.json",
        "E4ME_synthetic_validation.json", "E4ME_validation_lock_invariance.json",
        "E4ME_final_execution_contract.json", "E4ME_output_schema.json",
        "E4ME_final_firewall.json", "E4ME_protocol_frozen.json",
        "E4ME_verdict.json", "E4ME_report.md",
    ]
    for f in required:
        assert (RESULTS / f).exists(), f"Missing: {f}"


def test_protocol_frozen_next_stage(protocol):
    assert protocol["next_stage"] == "E4MF_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_EXECUTION"


def test_report_exists():
    assert (RESULTS / "E4ME_report.md").exists()
    text = (RESULTS / "E4ME_report.md").read_text()
    assert "E4ME_COMPLETE" in text
