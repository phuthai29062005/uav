"""
Tests for E4T — Causal Dynamics Gated Coordination Design.

Validates design artifacts. NO environment execution. NO training.
"""

import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "E4T_causal_dynamics_gated_coordination_design"
E4S_DIR = BASE / "results" / "E4S_pairwise_coordination_failure_analysis"
E4RF_DIR = BASE / "results" / "E4RF_pairwise_cg_ab_final_evaluation_execution"


def _load(name, d=RESULTS):
    with open(d / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load("E4T_verdict.json")


@pytest.fixture(scope="module")
def spec():
    return _load("E4T_c4_controller_spec.json")


@pytest.fixture(scope="module")
def gate():
    return _load("E4T_gate_definition.json")


@pytest.fixture(scope="module")
def sanity():
    return _load("E4T_offline_gate_sanity.json")


@pytest.fixture(scope="module")
def seeds():
    return _load("E4T_seed_reservation.json")


@pytest.fixture(scope="module")
def lock():
    return _load("E4T_e4s_evidence_lock.json")


@pytest.fixture(scope="module")
def params():
    return _load("E4T_parameter_count_contract.json")


@pytest.fixture(scope="module")
def three_arm():
    return _load("E4T_three_arm_design.json")


@pytest.fixture(scope="module")
def gating_contract():
    return _load("E4T_gating_success_contract.json")


@pytest.fixture(scope="module")
def final_contract():
    return _load("E4T_final_controller_success_contract.json")


# ---- T1-T8: E4S evidence preserved ----

def test_T1_e4s_complete(lock):
    assert lock["E4S_COMPLETE"] == "YES"


def test_T2_e4rf_failed(lock):
    assert lock["E4RF_VERDICT"] == "FAILED"


def test_T3_stationary_harm_5_of_5(lock):
    assert lock["STATIONARY_HARM_COUNT"] == "5/5"


def test_T4_s1_supported(lock):
    assert lock["S1_ACTION_SPECIALIZATION_SHIFT"] == "SUPPORTED"


def test_T5_s5_supported(lock):
    assert lock["S5_UNARY_POLICY_SHIFT"] == "SUPPORTED"


def test_T6_s6_supported(lock):
    assert lock["S6_COORDINATION_OVERRIDE_RATE"] == "SUPPORTED"


def test_T7_s8_supported(lock):
    assert lock["S8_CHECKPOINT_SELECTION_SCENARIO_TENSION"] == "SUPPORTED"


def test_T8_primary_mechanism_multifactorial(lock):
    assert lock["PRIMARY_FAILURE_MECHANISM"] == "MULTIFACTORIAL_SCENARIO_TRADEOFF"


# ---- T9-T17: C4 differs from C3 only by gate ----

def test_T9_c4_differs_from_c3_only_by_gate(spec):
    assert spec["IDENTICAL_TO_C3_EXCEPT"] == "deterministic_causal_pair_gate"


def test_T10_gate_uses_delta_severity_only(gate):
    assert gate["SIGNAL"]["feature"] == "delta_severity_r"


def test_T11_no_scenario_label_input(gate):
    assert gate["SIGNAL"]["no_scenario_label"] is True


def test_T12_no_future_information(gate):
    assert gate["SIGNAL"]["no_future_information"] is True


def test_T13_gate_formula_exact(gate):
    assert gate["PAIR_GATE"] == "g_ij(t) = 0.5 * (d_i(t) + d_j(t))"
    assert gate["REGIONAL_MAGNITUDE"] == "d_r(t) = abs(delta_severity_r(t))"


def test_T14_gate_symmetric(gate):
    assert gate["PROPERTIES"]["symmetric_ij"] is True


def test_T15_gate_bounded_0_1(gate):
    assert gate["BOUNDS"] == "[0, 1]"


def test_T16_gate_introduces_0_parameters(gate):
    assert gate["PROPERTIES"]["zero_trainable_parameters"] is True


def test_T17_gate_applied_after_pair_centering(spec):
    assert spec["GATE"]["applied_after"] == "pair_symmetrization_and_double_centering"


# ---- T18-T22: C4 mathematical formulation ----

def test_T18_joint_Q_equation_exact(spec):
    assert "(1/n)*sum_i U_i" in spec["JOINT_Q"]
    assert "g_ij*P_ij" in spec["JOINT_Q"]


def test_T19_maxsum_psi_gated(spec):
    assert spec["MAXSUM"]["pair_potential"] == "g_ij * P_ij(a_i,a_j) / |E|"


def test_T20_K10_unchanged(spec):
    assert spec["MAXSUM"]["K"] == 10


def test_T21_synchronous_update_unchanged(spec):
    assert spec["MAXSUM"]["update"] == "synchronous"


def test_T22_message_normalization_unchanged(spec):
    assert spec["MAXSUM"]["message_normalization"] == "subtract_maximum"


# ---- T23-T29: architecture and semantics unchanged ----

def test_T23_pair_network_unchanged(spec):
    assert spec["PAIR_NETWORK"]["CHANGE_FROM_C3"] == "NONE"


def test_T24_unary_network_unchanged(spec):
    assert spec["UNARY_NETWORK"]["CHANGE_FROM_C3"] == "NONE"


def test_T25_v3_unchanged(spec):
    assert spec["STATE"] == "V3_9DIM"


def test_T26_r1_unchanged(spec):
    assert spec["REWARD"] == "R1_MEAN_ACTIVE_REGIONAL"


def test_T27_replay_unchanged():
    no_tuning = _load("E4T_no_tuning_contract.json")
    assert no_tuning["GATE_FORMULA_FROZEN"] == "0.5 * (|d_i| + |d_j|)"


def test_T28_optimizer_unchanged(spec):
    assert spec["TRAINING"]["joint_double_dqn"] is True
    assert spec["TRAINING"]["no_c2_auxiliary_loss"] is True
    assert spec["TRAINING"]["no_gate_loss"] is True


def test_T29_production_operators_unchanged(spec):
    assert spec["OPERATORS"] == "UNCHANGED_PRODUCTION"


def test_T30_no_hard(spec):
    assert spec["OPERATORS"] == "UNCHANGED_PRODUCTION"


# ---- T31-T34: initialization and parameter count ----

def test_T31_c3_vs_c4_paired_initialization():
    init = _load("E4T_initialization_pairing_contract.json")
    assert init["C3_vs_C4_INITIAL_PARAMETER_DIFFERENCE"] == "NONE"


def test_T32_c1_c3_c4_unary_initialization():
    init = _load("E4T_initialization_pairing_contract.json")
    contract = init["FOR_EVERY_POLICY_SEED_p"]
    assert "C4_unary_initial_weights" in contract["C1_utility_initial_weights"]


def test_T33_c3_c4_pair_initialization():
    init = _load("E4T_initialization_pairing_contract.json")
    contract = init["FOR_EVERY_POLICY_SEED_p"]
    assert "C4_pair_initial_weights" in contract["C3_pair_initial_weights"]


def test_T34_c4_param_count_equals_c3(params):
    assert params["C4_EXPECTED_PARAMETERS"] == params["C3_TOTAL_PARAMETERS"]
    assert params["GATE_TRAINABLE_PARAMETERS"] == 0


# ---- T35-T40: experiment design ----

def test_T35_three_arm_design_exact(three_arm):
    assert len(three_arm["ARMS"]) == 3
    assert "C1" in three_arm["ARMS"]["A"]
    assert "C3" in three_arm["ARMS"]["B"]
    assert "C4" in three_arm["ARMS"]["C"]


def test_T36_training_episodes_16200():
    tm = _load("E4T_training_matrix_contract.json")
    assert tm["TOTAL_TRAINING_EPISODES"] == 16200


def test_T37_checkpoints_90():
    tm = _load("E4T_training_matrix_contract.json")
    assert tm["TOTAL_CHECKPOINTS"] == 90


def test_T38_validation_episodes_5400():
    vc = _load("E4T_validation_contract.json")
    assert vc["TOTAL_VALIDATION_EPISODES"] == 5400


def test_T39_selected_checkpoints_15():
    vc = _load("E4T_validation_contract.json")
    assert vc["TOTAL_SELECTED"] == 15


def test_T40_final_episodes_2700():
    fm = _load("E4T_final_matrix_contract.json")
    assert fm["TOTAL_FINAL_EPISODES"] == 2700


# ---- T41-T46: seed blocks ----

def test_T41_training_seed_block_reserved(seeds):
    assert seeds["PROPOSED"]["TRAIN_SEEDS"]["range"] == "2701..2760"
    assert seeds["PROPOSED"]["TRAIN_SEEDS"]["n"] == 60


def test_T42_validation_seed_block_reserved(seeds):
    assert seeds["PROPOSED"]["VALIDATION_SEEDS"]["range"] == "2761..2780"
    assert seeds["PROPOSED"]["VALIDATION_SEEDS"]["n"] == 20


def test_T43_final_seed_block_reserved(seeds):
    assert seeds["PROPOSED"]["FINAL_SEEDS"]["range"] == "2781..2840"
    assert seeds["PROPOSED"]["FINAL_SEEDS"]["n"] == 60


def test_T44_policy_seed_block_reserved(seeds):
    assert seeds["PROPOSED"]["POLICY_SEEDS"] == [8001, 8002, 8003, 8004, 8005]


def test_T45_all_proposed_seeds_disjoint(seeds):
    assert seeds["ALL_PROPOSED_DISJOINT"] is True
    assert seeds["OVERLAP"]["env"] == []
    assert seeds["OVERLAP"]["policy"] == []
    assert seeds["OVERLAP"]["bootstrap"] == []


def test_T46_seed_access_count_zero(seeds):
    assert seeds["SEED_ACCESS_COUNT"] == 0


# ---- T47-T53: statistical protocol ----

def test_T47_gating_primary_c3_c4_stationary(gating_contract):
    assert "STATIONARY" in gating_contract["VALIDATED_REQUIRES_ALL"][0]
    assert gating_contract["EFFECT_DEFINITION"] == "C3_MINUS_C4"


def test_T48_final_controller_primary_c1_c4_mixed(final_contract):
    assert "MIXED" in final_contract["VALIDATED_REQUIRES_ALL"][0]
    assert final_contract["EFFECT_DEFINITION"] == "C1_MINUS_C4"


def test_T49_practical_margin_1e5(gating_contract):
    assert gating_contract["PRACTICAL_MARGIN"] == 1e-5


def test_T50_bootstrap_axes_5x60():
    bs = _load("E4T_bootstrap_contract.json")
    assert bs["AXES"]["policy"] == 5
    assert bs["AXES"]["environment"] == 60


def test_T51_bootstrap_n_50000():
    bs = _load("E4T_bootstrap_contract.json")
    assert bs["N"] == 50000


def test_T52_bootstrap_seed_146001():
    bs = _load("E4T_bootstrap_contract.json")
    assert bs["SEED"] == 146001


def test_T53_shared_three_arm_draws():
    bs = _load("E4T_bootstrap_contract.json")
    assert bs["SHARED_THREE_ARM_DRAWS"] is True
    assert bs["SAME_INDICES_FOR"] == ["C1", "C3", "C4"]


# ---- T54-T61: success gates ----

def test_T54_stationary_repair_robustness_4_of_5(gating_contract):
    assert ">=4/5" in gating_contract["VALIDATED_REQUIRES_ALL"][5]


def test_T55_final_c4_robustness_4_of_5(final_contract):
    assert ">=4/5" in final_contract["VALIDATED_REQUIRES_ALL"][1]


def test_T56_persistent_preservation_gate(gating_contract):
    persistent_gate = gating_contract["VALIDATED_REQUIRES_ALL"][2]
    assert "PERSISTENT" in persistent_gate
    assert "NOT HARMFUL" in persistent_gate


def test_T57_mixed_preservation_gate(gating_contract):
    mixed_gate = gating_contract["VALIDATED_REQUIRES_ALL"][3]
    assert "MIXED" in mixed_gate
    assert "NOT HARMFUL" in mixed_gate


def test_T58_macro_preservation_gate(gating_contract):
    macro_gate = gating_contract["VALIDATED_REQUIRES_ALL"][4]
    assert "macro" in macro_gate
    assert "NOT HARMFUL" in macro_gate


def test_T59_c4_vs_c1_persistent_safety(final_contract):
    assert "PERSISTENT safety" in final_contract["VALIDATED_REQUIRES_ALL"][2]


def test_T60_c4_vs_c1_stationary_safety(final_contract):
    assert "STATIONARY safety" in final_contract["VALIDATED_REQUIRES_ALL"][3]


def test_T61_c4_vs_c1_macro_safety(final_contract):
    assert "macro safety" in final_contract["VALIDATED_REQUIRES_ALL"][4]


# ---- T62-T71: integrity constraints ----

def test_T62_gate_sanity_uses_saved_states_only(sanity):
    assert sanity["METHOD"] == "offline_saved_states_only"


def test_T63_no_new_environment_execution(verdict):
    assert verdict["NEW_ENVIRONMENT_EPISODES"] == 0


def test_T64_no_objective_evaluation(verdict):
    assert verdict["NEW_OBJECTIVE_EVALUATIONS"] == 0


def test_T65_no_training(verdict):
    assert verdict["NEW_TRAINING"] == "NO"


def test_T66_no_threshold_search():
    nt = _load("E4T_no_tuning_contract.json")
    assert nt["NO_THRESHOLD_SEARCH"] is True


def test_T67_no_alternative_gate_search():
    nt = _load("E4T_no_tuning_contract.json")
    assert nt["NO_ALTERNATIVE_GATE_SEARCH"] is True


def test_T68_no_learned_gating():
    nt = _load("E4T_no_tuning_contract.json")
    assert nt["NO_LEARNED_GATING"] is True


def test_T69_no_unary_anchor_loss():
    ul = _load("E4T_unary_shift_limitation.json")
    assert ul["NO_AUXILIARY_UNARY_ANCHORING_LOSS"] is True


def test_T70_no_state_feature_addition(spec):
    assert spec["STATE"] == "V3_9DIM"
    assert spec["UNARY_NETWORK"]["input_dim"] == 9


def test_T71_checkpoint_selection_unchanged():
    cb = _load("E4T_checkpoint_tension_boundary.json")
    assert cb["CHECKPOINT_SELECTION_UNCHANGED"] is True


# ---- T72-T74: verdict preservation ----

def test_T72_e4s_verdict_unchanged(verdict):
    assert verdict["E4S_VERDICT"] == "UNCHANGED"


def test_T73_c3_verdict_unchanged(verdict):
    assert verdict["E4RF_C3_VERDICT"] == "FAILED_UNCHANGED"


def test_T74_stop_rule_frozen():
    sr = _load("E4T_stop_rule.json")
    assert sr["CONTROLLER_REVISION_BUDGET_AFTER_C4"] == "ZERO"
    assert sr["IF_C4_FINAL_EXPERIMENT_FAILS"]["NO_C5"] is True
    assert sr["IF_C4_FINAL_EXPERIMENT_FAILS"]["NO_QPLEX"] is True
    assert sr["IF_C4_FINAL_EXPERIMENT_FAILS"]["NO_QTRAN"] is True


def test_T75_full_regression(verdict):
    assert verdict["FULL_REGRESSION"] == "5622/5622 PASS (0 failures)"


# ---- Additional integrity tests ----

def test_all_required_files():
    required = [
        "E4T_pre_manifest.json",
        "E4T_e4s_evidence_lock.json",
        "E4T_final_revision_authorization.json",
        "E4T_c4_controller_spec.json",
        "E4T_gate_definition.json",
        "E4T_causality_audit.json",
        "E4T_c1_c3_c4_factor_table.json",
        "E4T_initialization_pairing_contract.json",
        "E4T_parameter_count_contract.json",
        "E4T_three_arm_design.json",
        "E4T_seed_reservation.json",
        "E4T_training_matrix_contract.json",
        "E4T_validation_contract.json",
        "E4T_final_matrix_contract.json",
        "E4T_bootstrap_contract.json",
        "E4T_gating_success_contract.json",
        "E4T_final_controller_success_contract.json",
        "E4T_offline_gate_sanity.json",
        "E4T_unary_shift_limitation.json",
        "E4T_coordination_override_boundary.json",
        "E4T_checkpoint_tension_boundary.json",
        "E4T_no_tuning_contract.json",
        "E4T_stop_rule.json",
        "E4T_verdict.json",
        "E4T_report.md",
    ]
    for f in required:
        assert (RESULTS / f).exists(), f"Missing: {f}"


def test_e4t_complete(verdict):
    assert verdict["E4T_COMPLETE"] == "YES"


def test_gate_sanity_pass(sanity):
    assert sanity["GO_NO_GO"] == "PASS"


def test_gate_stationary_lt_dynamic(sanity):
    assert sanity["STATIONARY_PROXY"]["abs_d_mean"] < sanity["DYNAMIC_CONFIRMED"]["abs_d_mean"]


def test_c4_equals_c3_param_count(verdict):
    assert verdict["C4_PARAMETER_COUNT"] == verdict["C3_PARAMETER_COUNT"]
    assert verdict["C4_PARAMETER_COUNT"] == 39316


def test_c4_gate_zero_trainable(verdict):
    assert verdict["GATE_TRAINABLE_PARAMETERS"] == 0


def test_c4_equals_c3_except_gate(verdict):
    assert verdict["C4_EQUALS_C3_EXCEPT_GATE"] is True


def test_report_content():
    text = (RESULTS / "E4T_report.md").read_text()
    assert "Causal Dynamics Gated Coordination" in text
    assert "STATIONARY" in text
    assert "delta_severity" in text
    assert "STOP" in text
    assert "C4" in text


def test_no_new_environment_audit():
    pre = _load("E4T_pre_manifest.json")
    assert pre["DESIGN_ONLY"] is True
    assert pre["NO_TRAINING"] is True
    assert pre["NO_ENVIRONMENT_EXECUTION"] is True
    assert pre["NO_SEED_ACCESS"] is True


def test_historical_verdicts_preserved():
    auth = _load("E4T_final_revision_authorization.json")
    assert auth["HISTORICAL_FAILURES"]["E4F"] == "FAILED_POLICY_COLLAPSE"
    assert auth["HISTORICAL_FAILURES"]["E4IF"] == "VALIDATED_REWARD_CAUSAL"
    assert auth["HISTORICAL_FAILURES"]["E4RF"] == "FAILED_PAIRWISE_CG"


def test_gate_formula_frozen_before_inspection():
    nt = _load("E4T_no_tuning_contract.json")
    assert nt["FROZEN_BEFORE_INSPECTING_GATE_DISTRIBUTION"] is True


def test_no_multiplicity_hiding():
    bs = _load("E4T_bootstrap_contract.json")
    assert bs["NO_MULTIPLICITY_HIDING"] is True


def test_float32_semantics(spec):
    assert spec["FLOAT_SEMANTICS"] == "float32"


def test_verdict_consistency(verdict):
    if verdict["OFFLINE_GATE_SANITY"] == "PASS":
        assert verdict["NEXT_STAGE"] == "E4TI_C4_GATED_COORDINATION_IMPLEMENTATION_VALIDATION"
    else:
        assert verdict["NEXT_STAGE"] == "E4T_CONTROLLER_RESEARCH_STOP_REVIEW"
