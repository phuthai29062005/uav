"""
E4ID — Reward Scale Diagnostic Design: contract + freeze tests.

Design/freeze only. Verifies historical preservation, frozen V3 behavior policies,
fresh disjoint diagnostic seeds 1701..1720, the 300-episode matrix, the active-local /
conditional-magnitude / subset-matching definitions, the deterministic alpha rule +
bounds + scenario/tail gates, and that nothing was executed or trained.
"""

import json
from pathlib import Path

import pytest

import prdrl.experiments.E4ID_reward_scale_diagnostic_design as E4ID
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME, EPS_R1

RES = Path("results/E4ID_reward_scale_diagnostic_design")
E4I = Path("results/E4I_reward_credit_revision_design")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4ID_verdict.json").exists():
        pytest.skip("E4ID not run yet")


# ---- prerequisites ----

def test_T1_e4i_complete():
    assert _load(E4I / "E4I_verdict.json")["E4I_COMPLETE"] == "YES"
    assert _load(E4I / "E4I_verdict.json")["R1_REWARD_READY"] == "NO"


def test_T2_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T3_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T4_branch_b_selected():
    _require()
    assert _load(RES / "E4ID_verdict.json")["SELECTED_BRANCH"] == "B_REWARD_CREDIT"


def test_T5_r1_reward_version_exact():
    _require()
    assert _load(RES / "E4ID_verdict.json")["R1_REWARD"] == REWARD_R1_NAME
    assert REWARD_R1_NAME == "REWARD_R1_REGIONAL_QUALITY_GAIN"


def test_T6_r1_formula_exact():
    _require()
    # E4ID preserves E4I formula (does not rewrite it)
    f = _load(E4I / "E4I_reward_r1_formula.json")
    assert f["g_r_formula"] == "if L_pre==0 and L_post==0: 0 else (L_pre-L_post)/(L_pre+L_post+eps)"
    assert f["eps"] == 1e-12


def test_T7_fixed_v3_normalization():
    _require()
    assert _load(RES / "E4ID_verdict.json")["FIXED_NORMALIZATION"] == NORM_V3
    assert _load(RES / "E4ID_execution_mode_contract.json")["normalization"] == NORM_V3


def test_T8_behavior_policy_set_exact():
    _require()
    bp = _load(RES / "E4ID_behavior_policy_contract.json")
    assert bp["policy_seeds"] == [3001, 3002, 3003, 3004, 3005]
    assert bp["arm"] == "V3_TREATMENT"


def test_T9_no_policy_reselection():
    _require()
    bp = _load(RES / "E4ID_behavior_policy_contract.json")
    assert bp["not_retrained"] and bp["not_fine_tuned"] and bp["not_reselected"]


def test_T10_diagnostic_seeds_exact():
    _require()
    assert _load(RES / "E4ID_diagnostic_seed_contract.json")["diagnostic_seeds"] == [1701, 1720]
    assert E4ID.DIAG_SEEDS == list(range(1701, 1721))


def test_T11_diagnostic_label():
    _require()
    assert _load(RES / "E4ID_diagnostic_seed_contract.json")["label"] == "POST_HOC_REWARD_SCALE_DIAGNOSTIC_ONLY"


def test_T12_diagnostic_seeds_disjoint_history():
    _require()
    sc = _load(RES / "E4ID_diagnostic_seed_contract.json")
    assert sc["disjoint_from_all_history"] is True
    # independent recompute
    diag = set(range(1701, 1721))
    consumed = set()
    for v in E4ID.CONSUMED_SEED_SETS.values():
        consumed |= set(v)
    assert len(diag & consumed) == 0


def test_T13_diagnostic_seeds_disjoint_future_reward():
    _require()
    assert _load(RES / "E4ID_diagnostic_seed_contract.json")["disjoint_from_future_reward_1501_1640"] is True
    diag = set(range(1701, 1721))
    future = set(range(1501, 1641))
    assert len(diag & future) == 0


def test_T14_train_seeds_untouched():
    _require()
    fw = _load(RES / "E4ID_future_seed_firewall.json")
    assert fw["future_reward_train"] == [1501, 1560]
    assert fw["REWARD_TRAIN_ACCESS_COUNT"] == 0


def test_T15_validation_seeds_untouched():
    _require()
    fw = _load(RES / "E4ID_future_seed_firewall.json")
    assert fw["future_reward_validation"] == [1561, 1580]
    assert fw["REWARD_VALIDATION_ACCESS_COUNT"] == 0


def test_T16_final_seeds_untouched():
    _require()
    fw = _load(RES / "E4ID_future_seed_firewall.json")
    assert fw["future_reward_final"] == [1581, 1640]
    assert fw["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T17_diagnostic_matrix_300():
    _require()
    m = _load(RES / "E4ID_diagnostic_matrix_contract.json")
    assert m["total_diagnostic_episodes"] == 300
    assert E4ID.DIAG_EPISODES == 5 * 20 * 3


def test_T18_scenarios_exact():
    _require()
    assert _load(RES / "E4ID_diagnostic_matrix_contract.json")["scenarios"] == \
        ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T19_epsilon_zero_diagnostic():
    _require()
    assert _load(RES / "E4ID_execution_mode_contract.json")["epsilon"] == 0.0


def test_T20_no_gradient_diagnostic():
    _require()
    assert _load(RES / "E4ID_execution_mode_contract.json")["gradients"] == "disabled"


def test_T21_no_replay_diagnostic():
    _require()
    assert _load(RES / "E4ID_execution_mode_contract.json")["replay"] == "disabled"


def test_T22_no_optimizer_diagnostic():
    _require()
    em = _load(RES / "E4ID_execution_mode_contract.json")
    assert em["optimizer"] == "disabled" and em["learning"] == "none"


def test_T23_same_transition_computes_r0_and_r1():
    _require()
    m = _load(RES / "E4ID_diagnostic_matrix_contract.json")
    assert m["R0_and_R1_computed_on_same_trajectory"] is True


def test_T24_r1_does_not_alter_action():
    _require()
    em = _load(RES / "E4ID_execution_mode_contract.json")
    assert em["R1_influences_actions"] is False
    assert em["R1_is_observational_logging_only"] is True


def test_T25_l_pre_logged():
    _require()
    log = _load(RES / "E4ID_reward_logging_contract.json")
    assert "L_pre" in log["quality"]


def test_T26_l_post_logged():
    _require()
    log = _load(RES / "E4ID_reward_logging_contract.json")
    assert "L_post" in log["quality"]


def test_T27_local_active_threshold_exact():
    _require()
    a = _load(RES / "E4ID_active_local_definition.json")
    assert a["EPS_ACTIVE"] == 1e-12 == EPS_R1


def test_T28_activity_uses_all_valid_transitions():
    _require()
    a = _load(RES / "E4ID_active_local_definition.json")
    assert a["LOCAL_ACTIVE_FRACTION"] == "count(LOCAL_ACTIVE)/count(valid regional transitions)"


def test_T29_active_magnitude_uses_only_active():
    _require()
    a = _load(RES / "E4ID_active_local_definition.json")
    assert a["MEDIAN_ABS_G_ACTIVE"] == "median(|g_r| over LOCAL_ACTIVE only)"


def test_T30_global_active_magnitude_same_subset():
    _require()
    s = _load(RES / "E4ID_scale_metric_contract.json")
    assert "SAME LOCAL_ACTIVE subset" in s["MATCHING_RULE"]
    assert "LOCAL_ACTIVE" in s["MEDIAN_ABS_GLOBAL_ACTIVE"]


def test_T31_unconditional_median_not_used_for_alpha():
    _require()
    s = _load(RES / "E4ID_scale_metric_contract.json")
    assert s["transparency_not_used_for_alpha"] is True


def test_T32_initial_alpha_exact():
    _require()
    assert _load(RES / "E4ID_initial_alpha_contract.json")["INITIAL_ALPHA_LOCAL"] == 0.1


def test_T33_global_overwhelms_threshold_exact():
    _require()
    assert _load(RES / "E4ID_initial_alpha_contract.json")["GLOBAL_OVERWHELMS_THRESHOLD"] == 10.0


def test_T34_deterministic_alpha_formula_exact():
    _require()
    r = _load(RES / "E4ID_alpha_selection_rule.json")
    assert r["single_deterministic_formula"] == "alpha_scale = G/(M+1e-12)"


def test_T35_no_alpha_sweep():
    _require()
    r = _load(RES / "E4ID_alpha_selection_rule.json")
    assert r["no_performance_metric_in_calculation"] is True
    assert len(r["forbidden_sweep"]) >= 1


def test_T36_alpha_min_exact():
    _require()
    assert _load(RES / "E4ID_alpha_bounds.json")["ALPHA_LOCAL_MIN"] == 0.1


def test_T37_alpha_max_exact():
    _require()
    assert _load(RES / "E4ID_alpha_bounds.json")["ALPHA_LOCAL_MAX"] == 1.0


def test_T38_no_silent_alpha_clipping():
    _require()
    b = _load(RES / "E4ID_alpha_bounds.json")
    assert b["no_silent_clipping"] is True
    assert "FAIL" in b["ALPHA_BOUND_GATE"]


def test_T39_scenario_ratio_gate_exact():
    _require()
    g = _load(RES / "E4ID_scenario_scale_gate.json")
    assert g["SCENARIO_RATIO_MAX"] == 10.0
    assert "< 10" in g["rule"]


def test_T40_q95_tail_ratio():
    _require()
    t = _load(RES / "E4ID_tail_safety_gate.json")
    assert t["Q95_RATIO_MAX"] == 5.0


def test_T41_q99_tail_ratio():
    _require()
    t = _load(RES / "E4ID_tail_safety_gate.json")
    assert t["Q99_RATIO_MAX"] == 10.0


def test_T42_r1_extra_fe_zero():
    _require()
    assert _load(RES / "E4ID_fe_contract.json")["ADDITIONAL_OBJECTIVE_EVALUATIONS_FOR_R1"] == 0


def test_T43_reward_sign_semantics_exact():
    _require()
    s = _load(RES / "E4ID_sign_semantics_contract.json")
    assert s["L_post<L_pre"] == "g_r>0" and s["L_post>L_pre"] == "g_r<0"
    assert s["require_zero_sign_violations"] is True


def test_T44_execution_order_exact():
    _require()
    o = _load(RES / "E4ID_execution_order.json")["order"]
    assert "scenario" in o[0] and "1701..1720" in o[1] and "3001..3005" in o[2]


def test_T45_no_early_stop():
    _require()
    assert _load(RES / "E4ID_execution_order.json")["no_outcome_dependent_early_stopping"] is True


def test_T46_full_matrix_required():
    _require()
    m = _load(RES / "E4ID_matrix_completeness_contract.json")
    assert m["expected_diagnostic_episodes"] == 300
    assert m["DIAGNOSTIC_MATRIX_COMPLETE_required"] is True


def test_T47_diagnostic_data_never_confirmatory():
    _require()
    sc = _load(RES / "E4ID_diagnostic_seed_contract.json")
    assert sc["never_confirmatory_final"] is True
    assert sc["never_training"] is True and sc["never_validation"] is True


def test_T48_future_experiment_v3_r0_vs_v3_r1():
    _require()
    sf = _load(RES / "E4ID_single_factor_reaffirmation.json")
    assert sf["CONTROL"]["reward"] == "ORIGINAL_REWARD_R0" and sf["CONTROL"]["normalization"] == NORM_V3
    assert sf["TREATMENT"]["reward"] == REWARD_R1_NAME and sf["TREATMENT"]["normalization"] == NORM_V3
    assert sf["R0_alpha_local_unchanged"] is True


def test_T49_network_unchanged():
    _require()
    assert _load(RES / "E4ID_single_factor_reaffirmation.json")["network"] == "9-128-128-4 Shared Double DQN"


def test_T50_joint_architecture_unchanged():
    _require()
    sf = _load(RES / "E4ID_single_factor_reaffirmation.json")
    assert sf["joint_representation"] == "IMPLICIT_STATE_ONLY"
    assert sf["no_CTDE"] and sf["no_QMIX"] and sf["no_VDN"] and sf["no_autoregressive"]


def test_T51_training_hyperparameters_unchanged():
    _require()
    # E4ID does not alter the E4I future training contract
    hp = _load(E4I / "E4I_future_training_contract.json")["hyperparameters"]
    assert hp["gamma"] == 0.99 and hp["lr"] == 1e-4 and hp["batch"] == 128 and hp["epochs"] == 6


def test_T52_no_new_training():
    _require()
    v = _load(RES / "E4ID_verdict.json")
    assert v["NEW_TRAINING"] == "NO"
    assert _load(RES / "E4ID_pre_manifest.json")["no_training"] is True


def test_T53_no_environment_run_in_e4id():
    _require()
    pm = _load(RES / "E4ID_pre_manifest.json")
    assert pm["no_diagnostic_execution"] is True
    assert _load(RES / "E4ID_verdict.json")["DIAGNOSTIC_EXECUTION"] == "NOT_RUN"


def test_T54_historical_checkpoint_hashes_unchanged():
    _require()
    h = _load(RES / "E4ID_historical_integrity.json")
    assert h["checkpoint_all_match"] is True
    assert h["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert h["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert all(r["match"] for r in h["behavior_checkpoints"].values())


def test_T55_verdict_alpha_not_computed_and_next_stage():
    _require()
    v = _load(RES / "E4ID_verdict.json")
    assert v["E4ID_COMPLETE"] == "YES"
    assert v["ALPHA_FINAL"] == "NOT_YET_COMPUTED"
    assert v["R1_REWARD_READY"] == "NO"
    assert v["NEXT_STAGE"] == "E4IDD_REWARD_SCALE_DIAGNOSTIC_EXECUTION"


def test_T56_e4idd_output_schema_frozen():
    _require()
    sch = _load(RES / "E4ID_e4idd_output_schema.json")
    assert len(sch["artifacts"]) == 20
    assert "E4IDD_alpha_freeze.json" in sch["artifacts"]
    assert "REWARD_R1_CONFIG_SHA256" in sch["reward_r1_config_must_contain"]


def test_T57_readiness_logic_frozen():
    _require()
    rl = _load(RES / "E4ID_readiness_logic.json")
    assert len(rl["R1_REWARD_READY_requires_all"]) == 15
    assert rl["e4id_does_not_make_r1_ready"] is True


def test_T58_magnitude_estimator_conditional():
    _require()
    v = _load(RES / "E4ID_verdict.json")
    assert v["MAGNITUDE_ESTIMATOR"] == "MEDIAN_ABS_CONDITIONAL_ON_LOCAL_ACTIVE"
    assert v["GLOBAL_LOCAL_SUBSET_MATCHED"] == "YES"
