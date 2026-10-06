"""Tests for E4R Pairwise Coordination Graph Experiment Design."""
import json
import os
import pytest

BASE = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "E4R_pairwise_coordination_graph_experiment_design",
)

def _load(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Pre-manifest and historical integrity
# ---------------------------------------------------------------------------
class TestPreManifest:
    def test_t01_stage_type(self):
        v = _load("E4R_pre_manifest.json")
        assert v["stage_type"] == "DESIGN_PROTOCOL_FREEZE_ONLY"

    def test_t02_no_training(self):
        v = _load("E4R_pre_manifest.json")
        assert v["constraints"]["NO_TRAINING"] is True

    def test_t03_no_environment(self):
        v = _load("E4R_pre_manifest.json")
        assert v["constraints"]["NO_ENVIRONMENT_EXECUTION"] is True

    def test_t04_historical_integrity_all_pass(self):
        v = _load("E4R_historical_integrity.json")
        assert v["ALL_PASS"] is True
        for name, check in v["checks"].items():
            assert check["PASS"] is True, f"Failed: {name}"

    def test_t05_historical_integrity_count(self):
        v = _load("E4R_historical_integrity.json")
        assert len(v["checks"]) >= 11


# ---------------------------------------------------------------------------
# 2. E4Q selection lock
# ---------------------------------------------------------------------------
class TestE4QSelectionLock:
    def test_t06_selected_candidate(self):
        v = _load("E4R_e4q_selection_lock.json")
        assert v["PRIMARY_SELECTED_CANDIDATE"] == "PAIRWISE_COORDINATION_GRAPH_VALUE_FACTOR"

    def test_t07_control_is_c1(self):
        v = _load("E4R_e4q_selection_lock.json")
        assert "C1" in v["FUTURE_CONTROL"]

    def test_t08_c2_loss_not_carried(self):
        v = _load("E4R_e4q_selection_lock.json")
        assert v["C2_REGIONAL_AUXILIARY_LOSS_CARRIED_FORWARD"] == "NO"


# ---------------------------------------------------------------------------
# 3. Scientific question and factor boundary
# ---------------------------------------------------------------------------
class TestScientificQuestion:
    def test_t09_question_exists(self):
        v = _load("E4R_scientific_question.json")
        assert len(v["question"]) > 20

    def test_t10_not_stationary_specific(self):
        v = _load("E4R_scientific_question.json")
        assert v["NOT_a"] == "STATIONARY_SPECIFIC_TREATMENT"

    def test_t11_factor_name(self):
        v = _load("E4R_factor_boundary.json")
        assert v["factor_name"] == "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"

    def test_t12_factor_bundle_count(self):
        v = _load("E4R_factor_boundary.json")
        assert len(v["C3_complete_bundle"]) == 7

    def test_t13_factor_causal_wording(self):
        v = _load("E4R_factor_boundary.json")
        assert "pairwise" in v["causal_wording"].lower()


# ---------------------------------------------------------------------------
# 4. Control C1 contract
# ---------------------------------------------------------------------------
class TestControlC1:
    def test_t14_c1_utility_input(self):
        v = _load("E4R_control_c1_contract.json")
        assert v["utility_network"]["input"] == 9

    def test_t15_c1_utility_output(self):
        v = _load("E4R_control_c1_contract.json")
        assert v["utility_network"]["output"] == 4

    def test_t16_c1_mixer_type(self):
        v = _load("E4R_control_c1_contract.json")
        assert v["mixer"]["type"] == "QMIX"

    def test_t17_c1_mixer_hidden(self):
        v = _load("E4R_control_c1_contract.json")
        assert v["mixer"]["hidden"] == 32

    def test_t18_c1_loss_joint(self):
        v = _load("E4R_control_c1_contract.json")
        assert "SmoothL1" in v["loss"]

    def test_t19_c1_no_regional_loss(self):
        v = _load("E4R_control_c1_contract.json")
        assert v["NO_regional_auxiliary_loss"] is True


# ---------------------------------------------------------------------------
# 5. Treatment C3 contract
# ---------------------------------------------------------------------------
class TestTreatmentC3:
    def test_t20_c3_treatment_name(self):
        v = _load("E4R_treatment_c3_contract.json")
        assert v["treatment"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_t21_c3_components_count(self):
        v = _load("E4R_treatment_c3_contract.json")
        assert len(v["components"]) >= 6

    def test_t22_c3_not_full_q(self):
        v = _load("E4R_treatment_c3_contract.json")
        assert v["NOT_claimed_to_represent_every_joint_Q"] is True


# ---------------------------------------------------------------------------
# 6. Pair network contract
# ---------------------------------------------------------------------------
class TestPairNetwork:
    def test_t23_pair_input_dim(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["input_dim"] == 18

    def test_t24_pair_output_dim(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["output_dim"] == 16

    def test_t25_pair_reshape(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["output_reshape"] == [4, 4]

    def test_t26_pair_shared(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["shared"] is True

    def test_t27_pair_no_region_id(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["no_region_id_embedding"] is True

    def test_t28_pair_hidden_dims(self):
        v = _load("E4R_pair_network_contract.json")
        assert v["C3_pair_network"]["hidden_dims"] == [128, 128]


# ---------------------------------------------------------------------------
# 7. Pair symmetry contract
# ---------------------------------------------------------------------------
class TestPairSymmetry:
    def test_t29_symmetry_formula(self):
        v = _load("E4R_pair_symmetry_contract.json")
        assert "0.5" in v["symmetrization"]["M_ij"]

    def test_t30_symmetry_property(self):
        v = _load("E4R_pair_symmetry_contract.json")
        assert "M_ij" in v["symmetrization"]["property"]
        assert "M_ji" in v["symmetrization"]["property"]


# ---------------------------------------------------------------------------
# 8. Pair centering contract
# ---------------------------------------------------------------------------
class TestPairCentering:
    def test_t31_centering_formula(self):
        v = _load("E4R_pair_centering_contract.json")
        assert "row_mean" in v["centering"]["P_ij"]
        assert "col_mean" in v["centering"]["P_ij"]
        assert "grand_mean" in v["centering"]["P_ij"]

    def test_t32_centering_type(self):
        v = _load("E4R_pair_centering_contract.json")
        assert v["centering"]["type"] == "ZERO_ROW_COLUMN_MEAN"


# ---------------------------------------------------------------------------
# 9. Complete graph contract
# ---------------------------------------------------------------------------
class TestCompleteGraph:
    def test_t33_graph_type(self):
        v = _load("E4R_complete_graph_contract.json")
        assert v["graph"]["type"] == "COMPLETE_ACTIVE_REGION_GRAPH"

    def test_t34_max_edges(self):
        v = _load("E4R_complete_graph_contract.json")
        assert v["graph"]["max_edges"] == 45

    def test_t35_no_learned_sparse(self):
        v = _load("E4R_complete_graph_contract.json")
        assert v["graph"]["no_learned_sparse_graph"] is True


# ---------------------------------------------------------------------------
# 10. Joint value contract
# ---------------------------------------------------------------------------
class TestJointValue:
    def test_t36_normalization_type(self):
        v = _load("E4R_joint_value_contract.json")
        assert v["joint_value"]["normalization"] == "MEAN_UNARY_PLUS_MEAN_PAIR"

    def test_t37_beta_fixed(self):
        v = _load("E4R_joint_value_contract.json")
        assert v["joint_value"]["beta_pair"] == 1
        assert v["joint_value"]["beta_tunable"] is False

    def test_t38_n1_pair_zero(self):
        v = _load("E4R_joint_value_contract.json")
        assert v["joint_value"]["Q_pair_n_eq_1"] == "0"


# ---------------------------------------------------------------------------
# 11. Max-sum inference contract
# ---------------------------------------------------------------------------
class TestMaxSumInference:
    def test_t39_algorithm(self):
        v = _load("E4R_maxsum_inference_contract.json")
        assert v["inference"]["method"] == "MAX_SUM_MESSAGE_PASSING_V1"

    def test_t40_iterations(self):
        v = _load("E4R_maxsum_iteration_contract.json")
        assert v["message_passing"]["K_MAXSUM"] == 10

    def test_t41_synchronous(self):
        v = _load("E4R_maxsum_iteration_contract.json")
        assert v["message_passing"]["update_type"] == "synchronous"

    def test_t42_message_normalization(self):
        v = _load("E4R_maxsum_iteration_contract.json")
        assert "max" in v["message_passing"]["message_normalization"].lower()

    def test_t43_k_not_tunable(self):
        v = _load("E4R_maxsum_iteration_contract.json")
        assert v["message_passing"]["K_NOT_tunable"] is True


# ---------------------------------------------------------------------------
# 12. Loopy inference boundary
# ---------------------------------------------------------------------------
class TestLoopyInference:
    def test_t44_loopy_for_n_ge_3(self):
        v = _load("E4R_loopy_inference_boundary.json")
        assert "3" in v["loopy_status"]["complete_graph_has_loops_when"]

    def test_t45_approximate(self):
        v = _load("E4R_loopy_inference_boundary.json")
        assert "APPROXIMATE" in v["loopy_status"]["fixed_iteration_maxsum_is"]

    def test_t46_must_state_in_paper(self):
        v = _load("E4R_loopy_inference_boundary.json")
        assert v["loopy_status"]["must_be_explicit_in_paper_and_report"] is True


# ---------------------------------------------------------------------------
# 13. Small-n exact diagnostic
# ---------------------------------------------------------------------------
class TestSmallNExact:
    def test_t47_n_threshold_5(self):
        v = _load("E4R_small_n_exact_diagnostic.json")
        assert "5" in v["exact_diagnostic"]["for_states_with"]

    def test_t48_descriptive_only(self):
        v = _load("E4R_small_n_exact_diagnostic.json")
        assert v["exact_diagnostic"]["status"] == "DESCRIPTIVE_ONLY"

    def test_t49_no_action_replacement(self):
        v = _load("E4R_small_n_exact_diagnostic.json")
        assert v["exact_diagnostic"]["no_action_path_replacement"] is True


# ---------------------------------------------------------------------------
# 14. Complexity contract
# ---------------------------------------------------------------------------
class TestComplexity:
    def test_t50_maxsum_ops(self):
        v = _load("E4R_complexity_contract.json")
        assert "7200" in v["complexity"]["approximate_operations"]

    def test_t51_exact_ops(self):
        v = _load("E4R_complexity_contract.json")
        assert "1,048,576" in v["complexity"]["exhaustive_comparison"] or "1048576" in v["complexity"]["exhaustive_comparison"]


# ---------------------------------------------------------------------------
# 15. Reward contract
# ---------------------------------------------------------------------------
class TestReward:
    def test_t52_alpha_local(self):
        v = _load("E4R_reward_contract.json")
        assert v["reward"]["alpha_local"] == 0.1

    def test_t53_r1_validated(self):
        v = _load("E4R_reward_contract.json")
        assert v["reward"]["R1_validated"] is True

    def test_t54_no_reward_redesign(self):
        v = _load("E4R_reward_contract.json")
        assert v["reward"]["no_reward_redesign"] is True

    def test_t55_joint_reward_mean(self):
        v = _load("E4R_reward_contract.json")
        assert "mean" in v["reward"]["joint_reward"].lower()


# ---------------------------------------------------------------------------
# 16. Training loss contract
# ---------------------------------------------------------------------------
class TestTrainingLoss:
    def test_t56_loss_type(self):
        v = _load("E4R_training_loss_contract.json")
        assert v["C3_loss"]["type"] == "JOINT_DOUBLE_DQN_ONLY"

    def test_t57_smoothl1(self):
        v = _load("E4R_training_loss_contract.json")
        assert "SmoothL1" in v["C3_loss"]["loss"]

    def test_t58_no_regional_loss(self):
        v = _load("E4R_training_loss_contract.json")
        assert v["C3_loss"]["no_L_regional"] is True

    def test_t59_no_auxiliary_pair_loss(self):
        v = _load("E4R_training_loss_contract.json")
        assert v["C3_loss"]["no_auxiliary_pair_loss"] is True


# ---------------------------------------------------------------------------
# 17. Target network contract
# ---------------------------------------------------------------------------
class TestTargetNetwork:
    def test_t60_gamma(self):
        v = _load("E4R_target_network_contract.json")
        assert v["target_networks"]["gamma"] == 0.99

    def test_t61_tau(self):
        v = _load("E4R_target_network_contract.json")
        assert v["target_networks"]["tau"] == 0.005


# ---------------------------------------------------------------------------
# 18. Optimizer contract
# ---------------------------------------------------------------------------
class TestOptimizer:
    def test_t62_adam(self):
        v = _load("E4R_optimizer_contract.json")
        assert v["optimizer"]["type"] == "Adam"

    def test_t63_lr(self):
        v = _load("E4R_optimizer_contract.json")
        assert v["optimizer"]["lr"] == 1e-4

    def test_t64_batch_size(self):
        v = _load("E4R_optimizer_contract.json")
        assert v["optimizer"]["batch_size"] == 128

    def test_t65_replay_capacity(self):
        v = _load("E4R_optimizer_contract.json")
        assert v["optimizer"]["replay_capacity"] == 50000

    def test_t66_no_mixer_in_c3(self):
        v = _load("E4R_optimizer_contract.json")
        assert v["optimizer"]["no_mixer_in_C3"] is True


# ---------------------------------------------------------------------------
# 19. Replay contract
# ---------------------------------------------------------------------------
class TestReplay:
    def test_t67_replay_type(self):
        v = _load("E4R_replay_contract.json")
        assert v["replay"]["type"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

    def test_t68_no_sibling_dedup(self):
        v = _load("E4R_replay_contract.json")
        assert v["replay"]["sibling_deduplication"] is False

    def test_t69_retained_intentionally(self):
        v = _load("E4R_replay_contract.json")
        assert v["replay"]["sibling_duplication_known_limitation"] is True


# ---------------------------------------------------------------------------
# 20. Exploration contract
# ---------------------------------------------------------------------------
class TestExploration:
    def test_t70_epsilon_schedule(self):
        v = _load("E4R_exploration_contract.json")
        sched = v["exploration"]["schedule"]
        assert "1.0" in sched
        assert "0.05" in sched
        assert "50000" in sched

    def test_t71_same_as_c1(self):
        v = _load("E4R_exploration_contract.json")
        assert v["exploration"]["same_as_C1"] is True


# ---------------------------------------------------------------------------
# 21. Initialization pairing contract
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_t72_c1_eq_c3_unary(self):
        v = _load("E4R_initialization_pairing_contract.json")
        assert "C1_UTILITY_INIT == C3_UNARY_INIT" in v["initialization"]["requirement"]

    def test_t73_policy_seeds(self):
        v = _load("E4R_initialization_pairing_contract.json")
        assert v["initialization"]["policy_seeds"] == [7001, 7002, 7003, 7004, 7005]

    def test_t74_no_dummy_matching(self):
        v = _load("E4R_initialization_pairing_contract.json")
        assert v["initialization"]["no_dummy_parameter_matching"] is True


# ---------------------------------------------------------------------------
# 22. Normalization and operator contracts
# ---------------------------------------------------------------------------
class TestNormalizationOperator:
    def test_t75_normalization_v3(self):
        v = _load("E4R_normalization_contract.json")
        assert v["normalization"]["type"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"

    def test_t76_state_dim_9(self):
        v = _load("E4R_normalization_contract.json")
        assert v["normalization"]["total_per_region"] == 9

    def test_t77_both_arms_identical_norm(self):
        v = _load("E4R_normalization_contract.json")
        assert v["normalization"]["both_arms_identical"] is True

    def test_t78_operators_identical(self):
        v = _load("E4R_operator_contract.json")
        assert v["operators"]["both_arms_identical"] is True

    def test_t79_keep_fe_zero(self):
        v = _load("E4R_operator_contract.json")
        assert v["operators"]["KEEP"]["FE"] == 0


# ---------------------------------------------------------------------------
# 23. Seed contracts
# ---------------------------------------------------------------------------
class TestSeeds:
    def test_t80_train_seeds(self):
        v = _load("E4R_seed_contract.json")
        assert v["seeds"]["train"]["range"] == [2501, 2560]
        assert v["seeds"]["train"]["n"] == 60

    def test_t81_val_seeds(self):
        v = _load("E4R_seed_contract.json")
        assert v["seeds"]["validation"]["range"] == [2561, 2580]
        assert v["seeds"]["validation"]["n"] == 20

    def test_t82_final_seeds(self):
        v = _load("E4R_seed_contract.json")
        assert v["seeds"]["final"]["range"] == [2581, 2640]
        assert v["seeds"]["final"]["n"] == 60

    def test_t83_policy_seeds(self):
        v = _load("E4R_seed_contract.json")
        assert v["seeds"]["policy"] == [7001, 7002, 7003, 7004, 7005]

    def test_t84_bootstrap_seed(self):
        v = _load("E4R_seed_contract.json")
        assert v["seeds"]["bootstrap_seed"] == 136001

    def test_t85_all_disjoint(self):
        v = _load("E4R_seed_disjointness.json")
        assert v["ALL_DISJOINT"] is True
        assert v["env_overlap"] == 0
        assert v["policy_overlap"] == 0
        assert v["bootstrap_overlap"] == 0


# ---------------------------------------------------------------------------
# 24. Training matrix
# ---------------------------------------------------------------------------
class TestTrainingMatrix:
    def test_t86_arms(self):
        v = _load("E4R_training_matrix.json")
        assert v["training_matrix"]["n_arms"] == 2
        assert "C1_QMIX" in v["training_matrix"]["arms"]
        assert "C3_PAIRWISE_CG" in v["training_matrix"]["arms"]

    def test_t87_total_episodes(self):
        v = _load("E4R_training_matrix.json")
        assert v["training_matrix"]["total_episodes"] == 10800

    def test_t88_total_runs(self):
        v = _load("E4R_training_matrix.json")
        assert v["training_matrix"]["total_runs"] == 10


# ---------------------------------------------------------------------------
# 25. Checkpoint and validation contracts
# ---------------------------------------------------------------------------
class TestCheckpointValidation:
    def test_t89_checkpoint_count(self):
        v = _load("E4R_checkpoint_contract.json")
        assert v["checkpoints"]["n_per_run"] == 6
        assert v["checkpoints"]["total"] == 60

    def test_t90_no_checkpoint_modification(self):
        v = _load("E4R_checkpoint_contract.json")
        assert v["checkpoints"]["no_checkpoint_modification"] is True

    def test_t91_validation_non_confirmatory(self):
        v = _load("E4R_validation_contract.json")
        assert v["validation"]["confirmatory"] is False

    def test_t92_validation_metric(self):
        v = _load("E4R_validation_contract.json")
        assert v["validation"]["metric"] == "macro_MIGD_response"

    def test_t93_selection_per_arm(self):
        v = _load("E4R_checkpoint_selection_contract.json")
        assert v["checkpoint_selection"]["n_selected_per_arm"] == 5


# ---------------------------------------------------------------------------
# 26. Final evaluation contract
# ---------------------------------------------------------------------------
class TestFinalEvaluation:
    def test_t94_final_seeds(self):
        v = _load("E4R_final_matrix_contract.json")
        assert v["final_evaluation"]["env_seeds"] == [2581, 2640]

    def test_t95_final_episodes(self):
        v = _load("E4R_final_matrix_contract.json")
        assert v["final_evaluation"]["total_episodes"] == 1800

    def test_t96_final_confirmatory(self):
        v = _load("E4R_final_matrix_contract.json")
        assert v["final_evaluation"]["confirmatory"] is True

    def test_t97_seeds_unopened(self):
        v = _load("E4R_final_matrix_contract.json")
        assert v["final_evaluation"]["seeds_unopened_until_E4RF"] is True


# ---------------------------------------------------------------------------
# 27. Endpoint and statistical contracts
# ---------------------------------------------------------------------------
class TestEndpointStatistical:
    def test_t98_primary_mixed(self):
        v = _load("E4R_endpoint_contract.json")
        assert v["primary_endpoint"]["scenario"] == "MIXED"

    def test_t99_effect_direction(self):
        v = _load("E4R_endpoint_contract.json")
        assert v["primary_endpoint"]["effect_direction"] == "C1_MINUS_C3"
        assert v["primary_endpoint"]["positive_means"] == "C3_BETTER"

    def test_t100_margin(self):
        v = _load("E4R_endpoint_contract.json")
        assert v["primary_endpoint"]["margin"] == 1e-5

    def test_t101_bootstrap_n(self):
        v = _load("E4R_statistical_contract.json")
        assert v["bootstrap"]["N"] == 50000

    def test_t102_bootstrap_seed(self):
        v = _load("E4R_statistical_contract.json")
        assert v["bootstrap"]["seed"] == 136001

    def test_t103_bootstrap_paired(self):
        v = _load("E4R_statistical_contract.json")
        assert v["bootstrap"]["paired"] is True

    def test_t104_robustness_gate(self):
        v = _load("E4R_statistical_contract.json")
        assert v["robustness"]["threshold"] == 4


# ---------------------------------------------------------------------------
# 28. Success logic
# ---------------------------------------------------------------------------
class TestSuccessLogic:
    def test_t105_validated_requires_all(self):
        v = _load("E4R_success_logic.json")
        assert v["verdict_logic"]["VALIDATED"]["all_must_hold"] is True
        assert len(v["verdict_logic"]["VALIDATED"]["requires"]) == 4

    def test_t106_failed_any_of(self):
        v = _load("E4R_success_logic.json")
        assert len(v["verdict_logic"]["FAILED"]["any_of"]) >= 3


# ---------------------------------------------------------------------------
# 29. Mechanism diagnostic boundary
# ---------------------------------------------------------------------------
class TestMechanismDiagnostic:
    def test_t107_mechanism_not_used(self):
        v = _load("E4R_mechanism_diagnostic_boundary.json")
        assert v["mechanism_diagnostic"]["gate"] == "NOT_USED_IN_VERDICT"

    def test_t108_small_n_descriptive(self):
        v = _load("E4R_mechanism_diagnostic_boundary.json")
        d = v["mechanism_diagnostic"]["small_n_exact_diagnostic"]
        assert d["purpose"] == "DESCRIPTIVE_ONLY"
        assert d["cannot_rescue_verdict"] is True


# ---------------------------------------------------------------------------
# 30. Future routing
# ---------------------------------------------------------------------------
class TestFutureRouting:
    def test_t109_validated_route(self):
        v = _load("E4R_future_routing.json")
        assert v["routing"]["VALIDATED"]["next"] == "E4S_CG_VS_GLOBAL_COMPARATOR"
        assert v["routing"]["VALIDATED"]["NOT_inferred_transitively"] is True

    def test_t110_failed_no_tuning(self):
        v = _load("E4R_future_routing.json")
        f = v["routing"]["FAILED"]
        assert f["no_post_hoc_tuning"] is True
        assert f["no_hyperparameter_search"] is True
        assert f["no_K_adjustment"] is True
        assert f["no_graph_sparsification"] is True


# ---------------------------------------------------------------------------
# 31. E4RI validation contract
# ---------------------------------------------------------------------------
class TestE4RIContract:
    def test_t111_gate_count(self):
        v = _load("E4R_e4ri_validation_contract.json")
        assert v["e4ri_validation_requirements"]["total_gates"] == 28
        assert len(v["e4ri_validation_requirements"]["gates"]) == 28


# ---------------------------------------------------------------------------
# 32. Firewall
# ---------------------------------------------------------------------------
class TestFirewall:
    def test_t112_all_access_zero(self):
        v = _load("E4R_future_firewall.json")
        assert v["firewall"]["total_access"] == 0
        assert v["firewall"]["train_seed_access"] == 0
        assert v["firewall"]["val_seed_access"] == 0
        assert v["firewall"]["final_seed_access"] == 0
        assert v["firewall"]["environment_instantiation"] == 0
        assert v["firewall"]["training_runs"] == 0
        assert v["firewall"]["checkpoint_creation"] == 0
        assert v["firewall"]["checkpoint_modification"] == 0
        assert v["firewall"]["optimizer_steps"] == 0

    def test_t113_absolute_firewall(self):
        v = _load("E4R_future_firewall.json")
        assert v["firewall"]["ABSOLUTE_FIREWALL"] is True


# ---------------------------------------------------------------------------
# 33. Protocol frozen
# ---------------------------------------------------------------------------
class TestProtocolFrozen:
    def test_t114_frozen(self):
        v = _load("E4R_protocol_frozen.json")
        assert v["protocol_frozen"] is True

    def test_t115_frozen_components_count(self):
        v = _load("E4R_protocol_frozen.json")
        assert len(v["frozen_components"]) == 45

    def test_t116_no_modification(self):
        v = _load("E4R_protocol_frozen.json")
        assert v["no_modification_after_freeze"] is True


# ---------------------------------------------------------------------------
# 34. Verdict
# ---------------------------------------------------------------------------
class TestVerdict:
    def test_t117_e4r_complete(self):
        v = _load("E4R_verdict.json")
        assert v["E4R_COMPLETE"] == "YES"

    def test_t118_design_type(self):
        v = _load("E4R_verdict.json")
        assert v["design_type"] == "DESIGN_PROTOCOL_FREEZE_ONLY"

    def test_t119_treatment(self):
        v = _load("E4R_verdict.json")
        assert v["treatment"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_t120_control(self):
        v = _load("E4R_verdict.json")
        assert v["control"] == "C1_QMIX_V1"

    def test_t121_factor(self):
        v = _load("E4R_verdict.json")
        assert v["factor"] == "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"

    def test_t122_performance_not_computed(self):
        v = _load("E4R_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_t123_no_training(self):
        v = _load("E4R_verdict.json")
        assert v["training_executed"] is False
        assert v["final_evaluation_executed"] is False
        assert v["environment_instantiated"] is False

    def test_t124_next_stage(self):
        v = _load("E4R_verdict.json")
        assert v["NEXT_STAGE"] == "E4RI_PAIRWISE_CG_IMPLEMENTATION_VALIDATION"

    def test_t125_historical_e4f(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4F"] == "FAILED"

    def test_t126_historical_e4hf(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4HF"] == "FAILED"

    def test_t127_historical_e4if(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4IF"] == "REWARD_CAUSAL_VALIDATED"

    def test_t128_historical_e4jf(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4JF"] == "GLOBAL_COMPARATOR_FAILED"

    def test_t129_historical_e4kf(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4KF"] == "COORDINATION_CAUSAL_FAILED"

    def test_t130_historical_e4mf(self):
        v = _load("E4R_verdict.json")
        assert v["historical_verdicts_preserved"]["E4MF"] == "FAILED"


# ---------------------------------------------------------------------------
# 35. Report existence
# ---------------------------------------------------------------------------
class TestReport:
    def test_t131_report_exists(self):
        path = os.path.join(BASE, "E4R_report.md")
        assert os.path.exists(path)

    def test_t132_report_has_sections(self):
        path = os.path.join(BASE, "E4R_report.md")
        with open(path) as f:
            content = f.read()
        for section in ["## A.", "## B.", "## C.", "## Z.", "## AE."]:
            assert section in content, f"Missing section {section}"


# ---------------------------------------------------------------------------
# 36. Cross-file consistency
# ---------------------------------------------------------------------------
class TestCrossFileConsistency:
    def test_t133_seed_consistency(self):
        seed = _load("E4R_seed_contract.json")
        matrix = _load("E4R_training_matrix.json")
        assert seed["seeds"]["train"]["range"] == matrix["training_matrix"]["env_seeds_train"]

    def test_t134_policy_seed_consistency(self):
        seed = _load("E4R_seed_contract.json")
        init = _load("E4R_initialization_pairing_contract.json")
        assert seed["seeds"]["policy"] == init["initialization"]["policy_seeds"]

    def test_t135_bootstrap_consistency(self):
        seed = _load("E4R_seed_contract.json")
        stat = _load("E4R_statistical_contract.json")
        assert seed["seeds"]["bootstrap_seed"] == stat["bootstrap"]["seed"]

    def test_t136_effect_direction_consistency(self):
        endpoint = _load("E4R_endpoint_contract.json")
        stat = _load("E4R_statistical_contract.json")
        assert endpoint["primary_endpoint"]["effect_direction"] == stat["bootstrap"]["effect"]

    def test_t137_frozen_component_count_matches_files(self):
        frozen = _load("E4R_protocol_frozen.json")
        n_files = len([f for f in os.listdir(BASE) if f.startswith("E4R_")])
        assert len(frozen["frozen_components"]) == n_files

    def test_t138_parameter_count_descriptive(self):
        v = _load("E4R_parameter_count_contract.json")
        assert v["parameter_count"]["difference_is_intrinsic"] is True
        assert v["parameter_count"]["no_dummy_parameter_matching"] is True
