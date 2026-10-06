"""
Tests for E4M — Joint Credit Preservation Design.

Design/freeze stage only. No training, no environment execution.
No skip. No xfail. No weakened assertions.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

E4M_DIR = Path("results/E4M_joint_credit_preservation_design")
E4L_DIR = Path("results/E4L_coordination_failure_analysis")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")


def _load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load(E4M_DIR / "E4M_verdict.json")


@pytest.fixture(scope="module")
def control():
    return _load(E4M_DIR / "E4M_control_contract.json")


@pytest.fixture(scope="module")
def treatment():
    return _load(E4M_DIR / "E4M_treatment_contract.json")


@pytest.fixture(scope="module")
def composite():
    return _load(E4M_DIR / "E4M_composite_loss_contract.json")


@pytest.fixture(scope="module")
def gradient():
    return _load(E4M_DIR / "E4M_gradient_path_contract.json")


@pytest.fixture(scope="module")
def seeds():
    return _load(E4M_DIR / "E4M_future_seed_contract.json")


@pytest.fixture(scope="module")
def selection():
    return _load(E4M_DIR / "E4M_revision_selection.json")


@pytest.fixture(scope="module")
def candidates():
    return _load(E4M_DIR / "E4M_candidate_revision_review.json")


@pytest.fixture(scope="module")
def protocol():
    return _load(E4M_DIR / "E4M_protocol_frozen.json")


@pytest.fixture(scope="module")
def budget():
    return _load(E4M_DIR / "E4M_future_training_budget.json")


@pytest.fixture(scope="module")
def ckpt():
    return _load(E4M_DIR / "E4M_future_checkpoint_contract.json")


@pytest.fixture(scope="module")
def primary():
    return _load(E4M_DIR / "E4M_future_primary_hypothesis.json")


@pytest.fixture(scope="module")
def bootstrap():
    return _load(E4M_DIR / "E4M_future_statistical_contract.json")


@pytest.fixture(scope="module")
def e4mi():
    return _load(E4M_DIR / "E4M_e4mi_validation_contract.json")


# =====================================================================
# T1-T9: Historical integrity / prerequisites
# =====================================================================
class TestHistoricalIntegrity:
    def test_t1_e4l_complete(self):
        j = _load(E4M_DIR / "E4M_historical_integrity.json")
        assert j["checks"]["E4L_COMPLETE"] is True

    def test_t2_e4kf_failed_preserved(self, verdict):
        assert verdict["ORIGINAL_E4KF_VERDICT"] == "FAILED"

    def test_t3_e4kf_mixed_harmful_preserved(self, verdict):
        assert verdict["E4KF_VERDICT_CHANGED"] == "NO"

    def test_t4_h4_supported_preserved(self, verdict):
        assert verdict["H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION"] == "SUPPORTED"

    def test_t5_h2_weak_preserved(self, verdict):
        assert verdict["H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT"] == "WEAK"

    def test_t6_h3_weak_preserved(self, verdict):
        assert verdict["H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION"] == "WEAK"

    def test_t7_h5_not_supported_preserved(self, verdict):
        assert verdict["H5_MIXER_DEGENERACY"] == "NOT_SUPPORTED"

    def test_t8_h6_plausible_preserved(self, verdict):
        assert verdict["H6_MONOTONIC_LIMITATION"] == "PLAUSIBLE"

    def test_t9_primary_failure_multifactorial(self, verdict):
        assert verdict["E4L_PRIMARY_FAILURE_HYPOTHESIS"] == "MULTIFACTORIAL_OR_UNRESOLVED"


# =====================================================================
# T10-T15: Candidate revision review + selection
# =====================================================================
class TestCandidateReview:
    def test_t10_m0_reviewed(self, candidates):
        assert "M0_ORIGINAL_QMIX_C1" in candidates

    def test_t11_m1_reviewed(self, candidates):
        assert "M1_QMIX_PLUS_REGIONAL_AUXILIARY_TD" in candidates
        assert candidates["M1_QMIX_PLUS_REGIONAL_AUXILIARY_TD"]["selected"] is True

    def test_t12_m2_reviewed(self, candidates):
        assert "M2_QMIX_EVENT_DEDUPLICATED_JOINT_LOSS" in candidates
        assert candidates["M2_QMIX_EVENT_DEDUPLICATED_JOINT_LOSS"]["selected"] is False

    def test_t13_m3_reviewed(self, candidates):
        assert "M3_QMIX_MODIFIED_TEAM_REWARD" in candidates
        assert candidates["M3_QMIX_MODIFIED_TEAM_REWARD"]["selected"] is False

    def test_t14_m4_reviewed(self, candidates):
        assert "M4_NON_MONOTONIC_CENTRALIZED_Q" in candidates
        assert candidates["M4_NON_MONOTONIC_CENTRALIZED_Q"]["selected"] is False

    def test_t15_selected_revision_m1(self, selection, verdict):
        assert selection["SELECTED_REVISION"] == "M1_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
        assert verdict["SELECTED_REVISION"] == "M1_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"


# =====================================================================
# T16-T18: Future control and treatment
# =====================================================================
class TestFutureArms:
    def test_t16_future_control_c1(self, verdict):
        assert verdict["FUTURE_CONTROL"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"

    def test_t17_future_treatment_c2(self, verdict):
        assert verdict["FUTURE_TREATMENT"] == "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"

    def test_t18_factor_exact(self, verdict):
        assert verdict["FUTURE_EXPERIMENT_FACTOR"] == "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX"


# =====================================================================
# T19-T34: Preservation contracts
# =====================================================================
class TestPreservation:
    def test_t19_v3_same(self, verdict):
        assert verdict["FIXED_NORMALIZATION"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"

    def test_t20_r1_same(self, verdict):
        assert verdict["FIXED_REWARD"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"

    def test_t21_alpha_01(self, verdict):
        assert verdict["FIXED_ALPHA_LOCAL"] == 0.1

    def test_t22_utility_architecture(self, verdict):
        assert verdict["UTILITY_NETWORK"] == "9-128-128-4"

    def test_t23_mixer_config(self, verdict):
        assert verdict["QMIX_MIXER"] == "QMIX_MIXER_V1"

    def test_t24_central_state(self, control):
        assert control["central_state_dim"] == 55

    def test_t25_joint_reward(self, verdict):
        assert verdict["JOINT_REWARD"] == "ACTIVE_REGION_MEAN_OF_R1"

    def test_t26_replay_sampling_unit(self, verdict):
        assert verdict["REPLAY_SAMPLING_UNIT"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

    def test_t27_siblings_not_deduplicated(self, verdict):
        assert verdict["SIBLING_ANCHOR_DEDUPLICATION"] == "NO"

    def test_t28_replay_capacity(self, control):
        assert control["replay_capacity"] == 50000

    def test_t29_batch_size(self, control):
        assert control["batch_size"] == 128

    def test_t30_optimizer(self, control):
        assert control["optimizer"] == "Adam"

    def test_t31_lr(self, control):
        assert control["lr"] == 1e-4

    def test_t32_gamma(self, control):
        assert control["gamma"] == 0.99

    def test_t33_tau(self, control):
        assert control["tau"] == 0.005

    def test_t34_epsilon_schedule(self, control):
        assert control["epsilon_schedule"] == "FROZEN_FROM_E4KT"


# =====================================================================
# T35-T52: Loss contracts
# =====================================================================
class TestLossContracts:
    def test_t35_existing_joint_loss_exact(self, verdict):
        assert verdict["CONTROL_LOSS"] == "JOINT_QMIX_SMOOTHL1"
        assert verdict["TREATMENT_JOINT_LOSS"] == "JOINT_QMIX_SMOOTHL1"

    def test_t36_regional_online_action_selection(self):
        j = _load(E4M_DIR / "E4M_regional_target_contract.json")
        assert j["q_local"] == "Q_online(s_r, a_r)"

    def test_t37_regional_target_evaluation(self):
        j = _load(E4M_DIR / "E4M_regional_target_contract.json")
        assert j["a_local_star"] == "argmax_a Q_online(s'_r, a)"
        assert "Q_target" in j["y_local"]

    def test_t38_regional_r1_reward(self):
        j = _load(E4M_DIR / "E4M_regional_target_contract.json")
        assert "r_r^R1" in j["y_local"]

    def test_t39_regional_smoothl1(self):
        j = _load(E4M_DIR / "E4M_regional_target_contract.json")
        assert j["L_regional"] == "SmoothL1(q_local, y_local)"

    def test_t40_composite_half_half(self, composite):
        assert composite["joint_weight"] == 0.5
        assert composite["regional_weight"] == 0.5

    def test_t41_equal_weighting_not_tunable(self, composite, verdict):
        assert composite["weights_tunable"] is False
        assert composite["lambda_aux_exposed"] is False
        assert verdict["AUXILIARY_WEIGHT_TUNING"] == "NO"

    def test_t42_local_loss_does_not_use_mixer(self, gradient):
        assert gradient["L_regional_depends_on_mixer"] is False

    def test_t43_joint_loss_uses_mixer(self):
        j = _load(E4M_DIR / "E4M_joint_loss_preservation.json")
        assert "mixer" in j["Q_tot_online"].lower()

    def test_t44_utility_receives_both_gradient_paths(self, gradient):
        assert "L_joint" in gradient["online_utility_receives_gradients_from"]
        assert "L_regional" in gradient["online_utility_receives_gradients_from"]

    def test_t45_mixer_receives_joint_path_only(self, gradient):
        assert gradient["online_mixer_receives_gradients_from"] == ["L_joint"]

    def test_t46_one_optimizer(self):
        j = _load(E4M_DIR / "E4M_optimizer_contract.json")
        assert j["one_step_per_update_event"] is True
        assert j["separate_local_optimizer"] is False
        assert j["separate_mixer_optimizer"] is False

    def test_t47_no_auxiliary_network(self, treatment):
        assert treatment["action_selection"] == "ARGMAX_REGIONAL_UTILITY"
        assert treatment["mixer_in_action_selection"] is False

    def test_t48_no_reward_aggregation_change(self, verdict):
        assert verdict["JOINT_REWARD"] == "ACTIVE_REGION_MEAN_OF_R1"

    def test_t49_no_event_dedup(self, verdict):
        assert verdict["SIBLING_ANCHOR_DEDUPLICATION"] == "NO"

    def test_t50_no_joint_loss_1k_reweighting(self):
        j = _load(E4M_DIR / "E4M_joint_loss_preservation.json")
        assert j["unchanged_from_C1"] is True

    def test_t51_no_mixer_modification(self):
        j = _load(E4M_DIR / "E4M_mixer_preservation.json")
        assert j["unchanged"] is True

    def test_t52_no_non_monotonic_critic(self):
        j = _load(E4M_DIR / "E4M_mixer_preservation.json")
        assert j["monotonic"] is True


# =====================================================================
# T53-T58: FE and seeds
# =====================================================================
class TestFEAndSeeds:
    def test_t53_extra_fe_zero(self, verdict):
        assert verdict["EXTRA_OBJECTIVE_FE"] == 0

    def test_t54_train_seeds(self, verdict):
        assert verdict["FUTURE_TRAIN_SEEDS"] == "2101..2160"

    def test_t55_val_seeds(self, verdict):
        assert verdict["FUTURE_VALIDATION_SEEDS"] == "2161..2180"

    def test_t56_final_seeds(self, verdict):
        assert verdict["FUTURE_FINAL_SEEDS"] == "2181..2240"

    def test_t57_policy_seeds(self, verdict):
        assert verdict["FUTURE_POLICY_SEEDS"] == [6001, 6002, 6003, 6004, 6005]

    def test_t58_seed_disjointness(self, seeds, verdict):
        assert seeds["SEED_DISJOINTNESS"] == "PASS"
        assert verdict["SEED_DISJOINTNESS"] == "PASS"
        assert seeds["env_overlap_with_prior"] == []
        assert seeds["policy_overlap_with_prior"] == []


# =====================================================================
# T59-T61: Firewall / pairing
# =====================================================================
class TestFirewallPairing:
    def test_t59_all_access_counts_zero(self, verdict):
        assert verdict["M_TRAIN_ACCESS_COUNT"] == 0
        assert verdict["M_VALIDATION_ACCESS_COUNT"] == 0
        assert verdict["M_FINAL_ACCESS_COUNT"] == 0

    def test_t60_utility_init_pairing(self):
        j = _load(E4M_DIR / "E4M_future_pairing_contract.json")
        assert j["at_t0_require"]["HASH_C1_utility_online_eq_HASH_C2_utility_online"] is True

    def test_t61_mixer_init_pairing(self):
        j = _load(E4M_DIR / "E4M_future_pairing_contract.json")
        assert j["at_t0_require"]["HASH_C1_mixer_online_eq_HASH_C2_mixer_online"] is True


# =====================================================================
# T62-T65: Training budget / checkpoints
# =====================================================================
class TestBudgetCheckpoints:
    def test_t62_future_runs_10(self, budget):
        assert budget["total_runs"] == 10

    def test_t63_episodes_per_run_1080(self, budget):
        assert budget["episodes_per_run"] == 1080

    def test_t64_total_training_10800(self, budget):
        assert budget["total_training_episodes"] == 10800

    def test_t65_checkpoint_schedule(self, ckpt):
        assert ckpt["checkpoint_schedule"] == [10000, 20000, 30000, 40000, 50000, "final"]


# =====================================================================
# T66-T75: Future statistical design
# =====================================================================
class TestFutureStatistics:
    def test_t66_primary_mixed_migd_response(self, verdict):
        assert verdict["FUTURE_PRIMARY"] == "MIXED_MIGD_RESPONSE"

    def test_t67_effect_c1_minus_c2(self, verdict):
        assert verdict["FUTURE_EFFECT"] == "C1_MINUS_C2"

    def test_t68_positive_means_c2_better(self, verdict):
        assert verdict["POSITIVE_MEANS"] == "CREDIT_PRESERVING_C2_BETTER"

    def test_t69_practical_margin(self, verdict):
        assert verdict["PRACTICAL_MARGIN"] == 1e-5

    def test_t70_policy_robustness_4_of_5(self, verdict):
        assert verdict["FUTURE_POLICY_ROBUSTNESS"] == "AT_LEAST_4_OF_5_C2_BETTER"

    def test_t71_safety_contracts_frozen(self):
        j = _load(E4M_DIR / "E4M_future_safety_contract.json")
        assert "PERSISTENT" in j
        assert "STATIONARY" in j
        assert "MACRO" in j

    def test_t72_bootstrap_two_axis(self, bootstrap):
        assert bootstrap["method"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"

    def test_t73_bootstrap_n_50000(self, bootstrap):
        assert bootstrap["resamples"] == 50000

    def test_t74_bootstrap_seed_106001(self, bootstrap, verdict):
        assert bootstrap["BOOTSTRAP_MASTER_SEED"] == 106001
        assert verdict["FUTURE_BOOTSTRAP_MASTER_SEED"] == 106001

    def test_t75_no_mechanism_performance_gate(self):
        j = _load(E4M_DIR / "E4M_credit_diagnostics_contract.json")
        assert j["no_post_hoc_performance_mechanism_gate"] is True


# =====================================================================
# T76-T80: E4MI contract / environment / regression
# =====================================================================
class TestE4MIAndRegression:
    def test_t76_e4mi_validation_contract_frozen(self, e4mi):
        assert len(e4mi["required_validations"]) >= 24
        assert "synthetic_fixtures" in e4mi
        assert len(e4mi["synthetic_fixtures"]) >= 5

    def test_t77_no_new_environments(self, verdict):
        assert verdict["NEW_ENVIRONMENT_EXECUTION"] == "NO"

    def test_t78_no_training(self, verdict):
        assert verdict["NEW_TRAINING"] == "NO"

    def test_t79_historical_artifacts_unchanged(self):
        e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
        assert e4kf["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
        assert e4kf["MIXED_PRIMARY"] == "HARMFUL"
        e4l = _load(E4L_DIR / "E4L_verdict.json")
        assert e4l["E4L_COMPLETE"] == "YES"
        assert e4l["ORIGINAL_E4KF_VERDICT"] == "FAILED"

    def test_t80_full_regression_placeholder(self, verdict):
        assert "FULL_REGRESSION" in verdict


# =====================================================================
# T81-T85: Additional contract checks
# =====================================================================
class TestAdditionalContracts:
    def test_t81_e4m_complete_yes(self, verdict):
        assert verdict["E4M_COMPLETE"] == "YES"

    def test_t82_next_stage_e4mi(self, verdict):
        assert verdict["NEXT_STAGE"] == "E4MI_JOINT_CREDIT_PRESERVATION_IMPLEMENTATION_VALIDATION"

    def test_t83_report_exists(self):
        assert (E4M_DIR / "E4M_report.md").exists()
        assert (E4M_DIR / "E4M_report.md").stat().st_size > 500

    def test_t84_protocol_frozen_exists(self, protocol):
        assert protocol["SELECTED_REVISION"] == "M1_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
        assert protocol["FUTURE_EFFECT"] == "C1_MINUS_C2"

    def test_t85_all_output_files_exist(self):
        expected = [
            "E4M_pre_manifest.json", "E4M_historical_integrity.json",
            "E4M_e4l_evidence_lock.json", "E4M_candidate_revision_review.json",
            "E4M_revision_selection.json", "E4M_control_contract.json",
            "E4M_treatment_contract.json", "E4M_joint_loss_preservation.json",
            "E4M_regional_target_contract.json", "E4M_composite_loss_contract.json",
            "E4M_gradient_path_contract.json", "E4M_optimizer_contract.json",
            "E4M_target_network_contract.json", "E4M_replay_preservation.json",
            "E4M_sibling_semantics.json", "E4M_reward_preservation.json",
            "E4M_normalization_preservation.json", "E4M_mixer_preservation.json",
            "E4M_action_execution_contract.json", "E4M_extra_fe_contract.json",
            "E4M_future_ab_factor_contract.json", "E4M_future_seed_contract.json",
            "E4M_seed_disjointness.json", "E4M_future_pairing_contract.json",
            "E4M_future_training_budget.json", "E4M_future_checkpoint_contract.json",
            "E4M_future_primary_hypothesis.json", "E4M_future_statistical_contract.json",
            "E4M_future_safety_contract.json", "E4M_future_success_logic_skeleton.json",
            "E4M_credit_diagnostics_contract.json", "E4M_e4mi_validation_contract.json",
            "E4M_future_firewall.json", "E4M_protocol_frozen.json",
            "E4M_verdict.json", "E4M_report.md",
        ]
        for f in expected:
            assert (E4M_DIR / f).exists(), f"missing: {f}"

    def test_t86_sibling_semantics_correct(self):
        j = _load(E4M_DIR / "E4M_sibling_semantics.json")
        assert j["siblings_deduplicated"] is False
        assert j["regional_term_differs_by_focal_region"] is True
        assert j["C2_does_NOT_eliminate_repeated_joint_targets"] is True

    def test_t87_treatment_regional_loss(self, verdict):
        assert verdict["TREATMENT_REGIONAL_LOSS"] == "FOCAL_R1_DOUBLE_DQN_SMOOTHL1"

    def test_t88_treatment_composite_loss(self, verdict):
        assert verdict["TREATMENT_COMPOSITE_LOSS"] == "EQUAL_COMPONENT_MEAN"
        assert verdict["LOSS_FORMULA"] == "0.5*L_joint + 0.5*L_regional"

    def test_t89_mixer_hidden_dim_32(self, verdict):
        assert verdict["MIXER_HIDDEN_DIM"] == 32

    def test_t90_action_execution_unchanged(self):
        j = _load(E4M_DIR / "E4M_action_execution_contract.json")
        assert j["action_selection"] == "ARGMAX_REGIONAL_UTILITY"
        assert j["mixer_in_action_selection"] is False
        assert j["decentralized_execution"] is True
