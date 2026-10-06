"""
E4IT — Reward A/B Training Protocol Freeze: contract + freeze tests.

Freeze-only (no training). Verifies the matched V3+R0 vs V3+R1 training protocol: single
reward factor, identical state/network/normalization/hyperparameters/operators, alpha_local=0.1
both arms, fresh disjoint seeds, paired initialization, budget, logging, checkpoint selection,
mechanism/statistical/robustness/safety/success logic, and the untouched final-seed firewall.
"""

import json
from pathlib import Path

import pytest

import prdrl.experiments.E4IT_reward_ab_training_protocol_freeze as E4IT
from prdrl.rl import reward_integration as RI
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME
from prdrl.rl.normalization import NORM_V3
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4II = Path("results/E4II_reward_implementation_validation")
E4IDD = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4IT_verdict.json").exists():
        pytest.skip("E4IT not run yet")


# ---- prerequisites ----

def test_T1_e4ii_complete():
    assert _load(E4II / "E4II_implementation_verdict.json")["E4II_COMPLETE"] == "YES"


def test_T2_r1_implementation_valid():
    assert _load(E4II / "E4II_implementation_verdict.json")["R1_IMPLEMENTATION_VALID"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T4_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T5_branch_b():
    _require()
    assert _load(RES / "E4IT_verdict.json")["SELECTED_BRANCH"] == "B_REWARD_CREDIT"


def test_T6_exactly_two_arms():
    _require()
    rv = _load(RES / "E4IT_reward_versions.json")
    assert set(rv.keys()) >= {"CONTROL", "TREATMENT"}
    assert rv["no_third_candidate"] and rv["no_adjusted_alpha_variant"]


def test_T7_r0_control_exact():
    _require()
    c = _load(RES / "E4IT_reward_versions.json")["CONTROL"]
    assert c["reward_version"] == REWARD_R0_NAME and c["arm"] == "R0_CONTROL"


def test_T8_r1_treatment_exact():
    _require()
    t = _load(RES / "E4IT_reward_versions.json")["TREATMENT"]
    assert t["reward_version"] == REWARD_R1_NAME and t["arm"] == "R1_TREATMENT"
    assert t["reward_config_sha256"] == SHA


def test_T9_T10_both_use_v3():
    _require()
    nc = _load(RES / "E4IT_normalization_contract.json")
    assert nc["normalization"] == NORM_V3 and nc["identical_both_arms"] is True
    assert nc["no_refit"] is True


def test_T11_state_dim_9_both():
    _require()
    assert _load(RES / "E4IT_state_contract.json")["state_dim"] == 9


def test_T12_network_same_both():
    _require()
    assert _load(RES / "E4IT_network_contract.json")["network"] == "9-128-128-4"


def test_T13_joint_representation_same():
    _require()
    n = _load(RES / "E4IT_network_contract.json")
    assert n["joint_representation"] == "IMPLICIT_STATE_ONLY"
    assert n["no_QMIX"] and n["no_VDN"] and n["no_centralized_critic"] and n["no_autoregressive"]


def test_T14_alpha_local_01_both():
    _require()
    v = _load(RES / "E4IT_verdict.json")
    assert v["R0_ALPHA_LOCAL"] == 0.1 and v["R1_ALPHA_LOCAL"] == 0.1


def test_T15_T16_T17_single_factor_reward_only():
    _require()
    f = _load(RES / "E4IT_ab_factor_contract.json")
    assert f["experiment_factor"] == "REWARD_ONLY"
    assert f["single_factor_diff"]["single_factor"] is True
    assert f["single_factor_diff"]["differing_fields"] == []


def test_T18_r1_config_sha_exact():
    _require()
    assert _load(RES / "E4IT_r1_config_integrity.json")["reward_config_sha256"] == SHA
    assert _load(RES / "E4IT_r1_config_integrity.json")["verified"] is True


def test_T19_no_runtime_alpha_derivation():
    _require()
    assert _load(RES / "E4IT_r1_config_integrity.json")["no_runtime_alpha_derivation"] is True


def test_T20_T21_T22_seeds_exact():
    _require()
    s = _load(RES / "E4IT_seed_split.json")
    assert s["train"] == [1501, 1560] and s["validation"] == [1561, 1580] and s["final"] == [1581, 1640]


def test_T23_policy_seeds_exact():
    _require()
    assert _load(RES / "E4IT_seed_split.json")["policy_seeds"] == [4001, 4002, 4003, 4004, 4005]


def test_T24_T25_seeds_disjoint_and_diag_excluded():
    _require()
    s = _load(RES / "E4IT_seed_split.json")
    assert s["env_disjoint_from_all_history"] and s["policy_disjoint"] and s["internal_split_disjoint"]
    assert s["diagnostic_1701_1720_excluded"] is True
    new = set(range(1501, 1641))
    consumed = set()
    for v in E4IT.CONSUMED_SEED_SETS.values():
        consumed |= set(v)
    assert len(new & consumed) == 0 and (set(range(1701, 1721)) <= consumed)


def test_T26_T27_paired_initialization():
    _require()
    ip = _load(RES / "E4IT_initialization_pairing_contract.json")
    assert ip["all_paired"] is True
    for p in ["4001", "4002", "4003", "4004", "4005"]:
        row = ip["per_policy"][p]
        assert row["online_equals_target"] is True and row["R0_equals_R1_init"] is True


def test_T28_T29_separate_replay_no_sharing():
    _require()
    b = _load(RES / "E4IT_training_budget.json")
    assert "replay" in b["run_isolation"]
    assert b["no_replay_sharing_across_arms_or_policies"] is True


def test_T30_T31_T32_budget():
    _require()
    b = _load(RES / "E4IT_training_budget.json")
    assert b["episodes_per_run"] == 1080 and b["runs"] == 10 and b["total_training_episodes"] == 10800


def test_T33_scenario_balance():
    _require()
    assert "equal" in _load(RES / "E4IT_training_budget.json")["scenario_balance"].lower()


def test_T34_epsilon_schedule():
    _require()
    h = _load(RES / "E4IT_hyperparameters.json")
    assert "1.0->0.05" in h["epsilon_train"] and h["epsilon_validation"] == 0.0


def test_T35_checkpoint_schedule():
    _require()
    c = _load(RES / "E4IT_checkpoint_contract.json")
    assert c["schedule_transitions"] == [10000, 20000, 30000, 40000, 50000]
    assert c["total_future_checkpoints"] == 60


def test_T36_validation_epsilon_zero():
    _require()
    assert _load(RES / "E4IT_validation_contract.json")["epsilon"] == 0.0


def test_T37_validation_matrix_3600():
    _require()
    assert _load(RES / "E4IT_validation_contract.json")["validation_matrix"]["total_validation_episodes"] == 3600


def test_T38_T39_checkpoint_selection():
    _require()
    ph = _load(RES / "E4IT_primary_hypothesis.json")
    assert ph["checkpoint_selection_not_affected_by_reward"] is True
    assert _load(RES / "E4IT_verdict.json")["CHECKPOINT_SELECTION"] == "VALIDATION_MACRO_MIGD_RESPONSE"


def test_T40_T41_all_policies_retained_no_best_selection():
    _require()
    # 5 matched pairs, all retained (checkpoint_contract has 60 = 10 runs x 6)
    assert _load(RES / "E4IT_policy_pairing_contract.json")["matched_pairs"] == 5


def test_T42_T43_T44_T45_primary_endpoint():
    _require()
    ph = _load(RES / "E4IT_primary_hypothesis.json")
    assert ph["primary_scenario"] == "MIXED" and ph["primary_endpoint"] == "MIGD_response"
    assert "MIGD_response_R0" in ph["cell_effect"] and ph["positive_means"] == "R1_better"


def test_T46_practical_margin():
    _require()
    assert _load(RES / "E4IT_primary_hypothesis.json")["DELTA_NORM"] == 1e-5


def test_T47_mechanism_endpoint_1():
    _require()
    m = _load(RES / "E4IT_mechanism_endpoints.json")["endpoint_1_LOCAL_CREDIT_ACTIVITY"]
    assert "R1 active fraction > R0" in m["expected"]


def test_T48_mechanism_endpoint_2():
    _require()
    m = _load(RES / "E4IT_mechanism_endpoints.json")["endpoint_2_GLOBAL_TO_LOCAL_WEIGHTED_MAGNITUDE_RATIO"]
    assert m["expected"] == "R1 ratio < R0 ratio"
    assert "NOT_APPLICABLE" in m["zero_active_handling"]


def test_T49_T51_mechanism_endpoint_3_std_ddof0():
    _require()
    m = _load(RES / "E4IT_mechanism_endpoints.json")["endpoint_3_REGIONAL_REWARD_DISPERSION_WITHIN_EVENT"]
    assert "std (ddof=0)" in m["metric"] and m["expected"] == "R1 > R0"
    assert m["single_active_region_event"] == 0.0 and m["no_switch_to_MAD"] is True


def test_T50_mechanism_gate_2_of_3():
    _require()
    m = _load(RES / "E4IT_mechanism_endpoints.json")
    assert m["REWARD_CREDIT_MECHANISM_SUPPORTED"].startswith("YES iff >=2 of 3")


def test_T52_zero_active_not_applicable():
    _require()
    m = _load(RES / "E4IT_mechanism_endpoints.json")["endpoint_2_GLOBAL_TO_LOCAL_WEIGHTED_MAGNITUDE_RATIO"]
    assert "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL" in m["zero_active_handling"]


def test_T53_action_diversity_not_primary():
    _require()
    assert _load(RES / "E4IT_logging_contract.json")["action_diagnostics_not_primary_mechanism"] is True


def test_T54_shadow_reward_cannot_affect_replay():
    _require()
    sh = _load(RES / "E4IT_logging_contract.json")["shadow_logging"]
    assert sh["replay_uses_arm_specific_reward_only"] is True
    assert sh["no_cross_arm_replay_reward_influence"] is True


def test_T55_T56_T57_bootstrap():
    _require()
    s = _load(RES / "E4IT_statistical_design.json")
    assert s["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
    assert s["N_BOOTSTRAP"] == 50000 and s["BOOTSTRAP_MASTER_SEED"] == 66001
    assert s["policy_axis"] == 5 and s["environment_axis"] == 60


def test_T58_policy_robustness():
    _require()
    r = _load(RES / "E4IT_policy_robustness.json")
    assert r["threshold"] == 4 and ">=4 of 5" in r["rule"]


def test_T59_T60_safety():
    _require()
    s = _load(RES / "E4IT_safety_contract.json")
    assert s["scenarios"] == ["PERSISTENT", "STATIONARY"]
    assert s["safety_pass"] == "neither endpoint HARMFUL"


def test_T61_T62_macro_shared_draws():
    _require()
    s = _load(RES / "E4IT_statistical_design.json")
    assert "1/3" in s["macro"] and "SAME resampled" in s["macro"]


def test_T63_fe_cannot_rescue():
    _require()
    assert _load(RES / "E4IT_safety_contract.json")["fe_cannot_rescue_harmful"] is True
    assert _load(RES / "E4IT_primary_hypothesis.json")["fe_cannot_rescue_harmful_migd"] is True


def test_T64_validated_logic():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.BENEFICIAL, mechanism_supported=True,
                             policy_robustness_pass=True, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["verdict"] == "REWARD_CAUSAL_VALIDATED"


def test_T65_performance_partial_logic():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.BENEFICIAL, mechanism_supported=False,
                             policy_robustness_pass=True, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["verdict"] == "REWARD_CAUSAL_PARTIAL" and v["subtype"] == "PERFORMANCE_PARTIAL"


def test_T66_mechanistic_partial_logic():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.PRACTICALLY_NEUTRAL, mechanism_supported=True,
                             policy_robustness_pass=False, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["subtype"] == "MECHANISTIC_PARTIAL" and v["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"


def test_T67_failed_logic():
    for lab in (FV.DIRECTIONAL_NEGATIVE, FV.HARMFUL):
        v = RI.reward_ab_verdict(mixed_primary_label=lab, mechanism_supported=True,
                                 policy_robustness_pass=True, persistent_safety_pass=True,
                                 stationary_safety_pass=True, macro_safety_pass=True,
                                 checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                                 version_integrity_pass=True, no_final_tuning=True)
        assert v["verdict"] == "REWARD_CAUSAL_FAILED"
    # safety fail also FAILED even with beneficial
    v = RI.reward_ab_verdict(mixed_primary_label=FV.BENEFICIAL, mechanism_supported=True,
                             policy_robustness_pass=True, persistent_safety_pass=False,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["verdict"] == "REWARD_CAUSAL_FAILED"


def test_T68_T69_final_firewall():
    _require()
    fw = _load(RES / "E4IT_final_firewall.json")
    assert fw["final_seeds"] == [1581, 1640]
    assert fw["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T70_T71_T72_T73_T74_no_training_audit():
    _require()
    a = _load(RES / "E4IT_no_training_audit.json")
    for k in ["real_environment_episodes", "training_episodes", "optimizer_steps",
              "replay_training_appends", "new_policy_checkpoints", "model_weight_changes"]:
        assert a[k] == 0
    assert _load(RES / "E4IT_verdict.json")["NEW_TRAINING"] == "NO"


def test_T75_validation_may_not_produce_validated():
    _require()
    assert _load(RES / "E4IT_success_logic.json")["validation_may_not_produce_validated"] is True


def test_T76_e4it_complete_and_next_stage():
    _require()
    v = _load(RES / "E4IT_verdict.json")
    assert v["E4IT_COMPLETE"] == "YES"
    assert v["NEXT_STAGE"] == "E4ITR_REWARD_AB_TRAINING_EXECUTION"


def test_T77_r0_source_byte_stable():
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "REGIONAL_QUALITY_GAIN" not in src and "reward_r1" not in src
