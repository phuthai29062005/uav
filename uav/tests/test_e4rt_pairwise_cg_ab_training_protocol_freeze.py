"""E4RT: Pairwise CG A/B Training Protocol Freeze — 100+ tests."""

import json
import os
import hashlib
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "results")
E4RT = os.path.join(BASE, "E4RT_pairwise_cg_ab_training_protocol_freeze")
E4R = os.path.join(BASE, "E4R_pairwise_coordination_graph_experiment_design")
E4RI = os.path.join(BASE, "E4RI_pairwise_coordination_graph_implementation_validation")

C1_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
C3_SHA = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"


def load(directory, name):
    with open(os.path.join(directory, name)) as f:
        return json.load(f)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def e4r_verdict():
    return load(E4R, "E4R_verdict.json")

@pytest.fixture(scope="module")
def e4ri_verdict():
    return load(E4RI, "E4RI_verdict.json")

@pytest.fixture(scope="module")
def pre():
    return load(E4RT, "E4RT_pre_manifest.json")

@pytest.fixture(scope="module")
def hist():
    return load(E4RT, "E4RT_historical_integrity.json")

@pytest.fixture(scope="module")
def arms():
    return load(E4RT, "E4RT_arm_contract.json")

@pytest.fixture(scope="module")
def c1lock():
    return load(E4RT, "E4RT_c1_config_lock.json")

@pytest.fixture(scope="module")
def c3lock():
    return load(E4RT, "E4RT_c3_config_lock.json")

@pytest.fixture(scope="module")
def factor():
    return load(E4RT, "E4RT_causal_factor_lock.json")

@pytest.fixture(scope="module")
def init_pair():
    return load(E4RT, "E4RT_initialization_pairing.json")

@pytest.fixture(scope="module")
def param_disc():
    return load(E4RT, "E4RT_parameter_count_disclosure.json")

@pytest.fixture(scope="module")
def seeds():
    return load(E4RT, "E4RT_seed_contract.json")

@pytest.fixture(scope="module")
def seed_disj():
    return load(E4RT, "E4RT_seed_disjointness.json")

@pytest.fixture(scope="module")
def train_mat():
    return load(E4RT, "E4RT_training_matrix.json")

@pytest.fixture(scope="module")
def train_ord():
    return load(E4RT, "E4RT_training_order.json")

@pytest.fixture(scope="module")
def pair_con():
    return load(E4RT, "E4RT_pairing_contract.json")

@pytest.fixture(scope="module")
def obj():
    return load(E4RT, "E4RT_training_objective_contract.json")

@pytest.fixture(scope="module")
def reward():
    return load(E4RT, "E4RT_reward_contract.json")

@pytest.fixture(scope="module")
def optim():
    return load(E4RT, "E4RT_optimizer_contract.json")

@pytest.fixture(scope="module")
def replay():
    return load(E4RT, "E4RT_replay_contract.json")

@pytest.fixture(scope="module")
def explore():
    return load(E4RT, "E4RT_exploration_contract.json")

@pytest.fixture(scope="module")
def maxsum():
    return load(E4RT, "E4RT_maxsum_training_contract.json")

@pytest.fixture(scope="module")
def ckpt():
    return load(E4RT, "E4RT_checkpoint_contract.json")

@pytest.fixture(scope="module")
def ckpt_schema():
    return load(E4RT, "E4RT_checkpoint_schema.json")

@pytest.fixture(scope="module")
def val_mat():
    return load(E4RT, "E4RT_validation_matrix.json")

@pytest.fixture(scope="module")
def val_mode():
    return load(E4RT, "E4RT_validation_mode.json")

@pytest.fixture(scope="module")
def ckpt_sel():
    return load(E4RT, "E4RT_checkpoint_selection.json")

@pytest.fixture(scope="module")
def val_bound():
    return load(E4RT, "E4RT_validation_status_boundary.json")

@pytest.fixture(scope="module")
def diag_log():
    return load(E4RT, "E4RT_diagnostic_logging_contract.json")

@pytest.fixture(scope="module")
def mech_gate():
    return load(E4RT, "E4RT_mechanism_gate_boundary.json")

@pytest.fixture(scope="module")
def train_comp():
    return load(E4RT, "E4RT_training_completion_contract.json")

@pytest.fixture(scope="module")
def val_comp():
    return load(E4RT, "E4RT_validation_completion_contract.json")

@pytest.fixture(scope="module")
def firewall():
    return load(E4RT, "E4RT_final_seed_firewall.json")

@pytest.fixture(scope="module")
def future_final():
    return load(E4RT, "E4RT_future_final_contract.json")

@pytest.fixture(scope="module")
def future_stat():
    return load(E4RT, "E4RT_future_statistical_contract.json")

@pytest.fixture(scope="module")
def success():
    return load(E4RT, "E4RT_future_success_logic.json")

@pytest.fixture(scope="module")
def routing():
    return load(E4RT, "E4RT_future_routing.json")

@pytest.fixture(scope="module")
def resume():
    return load(E4RT, "E4RT_resume_contract.json")

@pytest.fixture(scope="module")
def exec_cfg():
    return load(E4RT, "E4RT_execution_config.json")

@pytest.fixture(scope="module")
def proto_hash():
    return load(E4RT, "E4RT_protocol_hash.json")

@pytest.fixture(scope="module")
def frozen():
    return load(E4RT, "E4RT_protocol_frozen.json")

@pytest.fixture(scope="module")
def verdict():
    return load(E4RT, "E4RT_verdict.json")


# ── T1-T3: Prior stage completion ────────────────────────

class TestPriorStageCompletion:
    def test_t1_e4r_complete(self, e4r_verdict):
        assert e4r_verdict["E4R_COMPLETE"] == "YES"

    def test_t2_e4ri_complete(self, e4ri_verdict):
        assert e4ri_verdict["E4RI_COMPLETE"] == "YES"

    def test_t3_c3_implementation_valid(self, e4ri_verdict):
        assert e4ri_verdict["ALL_GATES_PASS"] is True


# ── T4: Historical verdicts ──────────────────────────────

class TestHistoricalVerdicts:
    def test_t4_e4mf_failed_preserved(self, hist):
        assert hist["preserved_verdicts"]["E4MF"] == "FAILED"

    def test_t4b_all_historical(self, hist):
        v = hist["preserved_verdicts"]
        assert v["E4F"] == "FAILED"
        assert v["E4HF"] == "FAILED"
        assert v["E4IF"] == "REWARD_CAUSAL_VALIDATED"
        assert v["E4JF"] == "GLOBAL_COMPARATOR_FAILED"
        assert v["E4KF"] == "COORDINATION_CAUSAL_FAILED"

    def test_t4c_performance_not_computed(self, hist):
        assert hist["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"


# ── T5-T7: Config SHAs and causal factor ─────────────────

class TestConfigIntegrity:
    def test_t5_c1_config_sha(self, pre):
        assert pre["QMIX_CONFIG_SHA256"] == C1_SHA

    def test_t6_c3_config_sha(self, pre):
        assert pre["C3_CONFIG_SHA256"] == C3_SHA

    def test_t7_causal_factor(self, factor):
        assert factor["factor"] == "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"

    def test_t7b_not_described_as_pair_vs_mixer_only(self, factor):
        assert factor["NOT_described_as"] == "pair_network_vs_QMIX_mixer_only"


# ── T8-T11: Arms ─────────────────────────────────────────

class TestArms:
    def test_t8_arms_exactly_c1_c3(self, arms):
        assert arms["arms"]["CONTROL"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
        assert arms["arms"]["TREATMENT"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_t9_no_c0(self, arms):
        assert "C0_INDEPENDENT_SHARED_DDQN" in arms["excluded"]

    def test_t10_no_c2(self, arms):
        assert "C2_FOCAL_REGIONAL_CREDIT" in arms["excluded"]
        assert arms["C2_loss_NOT_carried"] is True

    def test_t11_no_global_comparator(self, arms):
        assert "GLOBAL_FIXED_COMPARATOR" in arms["excluded"]


# ── T12-T15: Parameter counts ────────────────────────────

class TestParameterCounts:
    def test_t12_utility_unary_both_18308(self, param_disc):
        assert param_disc["C1"]["utility"] == 18308
        assert param_disc["C3"]["unary"] == 18308
        assert param_disc["unary_equals_utility"] is True

    def test_t13_c1_total_39868(self, param_disc):
        assert param_disc["C1"]["total"] == 39868

    def test_t14_c3_total_39316(self, param_disc):
        assert param_disc["C3"]["total"] == 39316

    def test_t15_mismatch_disclosed(self, param_disc):
        assert param_disc["difference"] == -552
        assert param_disc["parameter_count_is"] == "DESCRIPTIVE_ONLY"


# ── T16-T23: Seeds ────────────────────────────────────────

class TestSeeds:
    def test_t16_policy_seeds(self, seeds):
        assert seeds["policy"] == [7001, 7002, 7003, 7004, 7005]

    def test_t17_train_seeds(self, seeds):
        assert seeds["train"]["range"] == [2501, 2560]
        assert seeds["train"]["n"] == 60

    def test_t18_val_seeds(self, seeds):
        assert seeds["validation"]["range"] == [2561, 2580]
        assert seeds["validation"]["n"] == 20

    def test_t19_final_seeds(self, seeds):
        assert seeds["final"]["range"] == [2581, 2640]
        assert seeds["final"]["n"] == 60

    def test_t20_seed_disjointness(self, seed_disj):
        assert seed_disj["ALL_DISJOINT"] is True
        assert seed_disj["env_overlap"] == 0
        assert seed_disj["policy_overlap"] == 0
        assert seed_disj["bootstrap_overlap"] == 0

    def test_t21_train_access_zero(self, seeds):
        assert seeds["train"]["access_count"] == 0

    def test_t22_validation_access_zero(self, seeds):
        assert seeds["validation"]["access_count"] == 0

    def test_t23_final_access_zero(self, seeds):
        assert seeds["final"]["access_count"] == 0


# ── T24-T25: Initialization ──────────────────────────────

class TestInitialization:
    def test_t24_paired_unary_init(self, init_pair):
        assert init_pair["paired_unary_initialization"] is True

    def test_t25_pair_seed_offset_1(self, init_pair):
        assert init_pair["pair_seed_offset"] == 1
        assert init_pair["pair_seeds"] == [7002, 7003, 7004, 7005, 7006]


# ── T26-T33: Training matrix ─────────────────────────────

class TestTrainingMatrix:
    def test_t26_training_epochs_6(self, train_mat):
        assert train_mat["epochs_per_run"] == 6

    def test_t27_60_training_envs(self, train_mat):
        assert train_mat["env_seeds_per_epoch"] == 60

    def test_t28_3_scenarios(self, train_mat):
        assert train_mat["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]
        assert len(train_mat["scenarios"]) == 3

    def test_t29_1080_episodes_per_run(self, train_mat):
        assert train_mat["episodes_per_run"] == 1080

    def test_t30_10_runs(self, train_mat):
        assert train_mat["total_runs"] == 10

    def test_t31_training_total_10800(self, train_mat):
        assert train_mat["TRAIN_EPISODES_PLANNED"] == 10800

    def test_t32_training_order_frozen(self, train_ord):
        assert train_ord["deterministic"] is True

    def test_t33_environment_pairing_frozen(self, pair_con):
        assert pair_con["environment_pairing"]["C1_C3_same_deterministic_environment"] is True
        assert pair_con["environment_pairing"]["no_forced_identical_policy_actions"] is True


# ── T34-T38: Production operators and semantics ──────────

class TestProductionAndSemantics:
    def test_t34_production_actions_exact(self):
        e4r_ops = load(E4R, "E4R_operator_contract.json")
        ops = e4r_ops["operators"]
        assert ops["KEEP"]["FE"] == 0
        assert ops["LOCAL"]["sigma_L"] == 0.02
        assert ops["PREDICT"]["sigma_P"] == 0.005
        assert ops["PREDICT"]["lambda_P"] == 1.0
        assert ops["DIVERSIFY"]["sigma_D"] == 0.15

    def test_t35_operator_constants_exact(self):
        e4r_ops = load(E4R, "E4R_operator_contract.json")
        assert e4r_ops["operators"]["LOCAL"]["sigma_L"] == 0.02

    def test_t36_unrestricted_survival(self):
        e4r_ops = load(E4R, "E4R_operator_contract.json")
        assert e4r_ops["operators"]["survival"] == "UNRESTRICTED_STANDARD_NSGA_II"
        assert e4r_ops["operators"]["no_HARD"] is True

    def test_t37_v3_exact(self, c3lock):
        assert c3lock["C3_CONFIG_SHA256"] == C3_SHA

    def test_t38_requested_realized_semantics(self, explore):
        sem = explore["requested_action_semantics"]
        assert sem["if_requested_PREDICT_realized_LOCAL"] == "stored action = PREDICT"
        assert sem["performance_reflects"] == "realized response"


# ── T39-T41: Training objectives ─────────────────────────

class TestTrainingObjectives:
    def test_t39_c1_joint_objective(self, obj):
        assert obj["C1_objective"]["no_modification"] is True

    def test_t40_c3_joint_double_q(self, obj):
        c3 = obj["C3_objective"]
        assert c3["next_action"] == "a* = MaxSum_online(s')"
        assert c3["loss"] == "SmoothL1(q_online, y)"

    def test_t41_c3_no_regional_auxiliary(self, obj):
        c3 = obj["C3_objective"]
        assert c3["no_L_regional"] is True
        assert c3["no_C2_composite_loss"] is True
        assert c3["no_pair_imitation_loss"] is True
        assert c3["no_E4PD_supervision"] is True


# ── T42: Reward ───────────────────────────────────────────

class TestReward:
    def test_t42_reward_mean_r1(self, reward):
        assert "mean" in reward["joint_reward"].lower()
        assert "R1" in reward["joint_reward"]
        assert reward["both_arms_identical_reward"] is True
        assert reward["alpha_local"] == 0.1


# ── T43-T51: Optimizer ────────────────────────────────────

class TestOptimizer:
    def test_t43_adam(self, optim):
        assert optim["optimizer"] == "Adam"

    def test_t44_lr_1e4(self, optim):
        assert optim["lr"] == 1e-4

    def test_t45_gamma_99(self, optim):
        assert optim["gamma"] == 0.99

    def test_t46_tau_005(self, optim):
        assert optim["tau"] == 0.005

    def test_t47_batch_128(self, optim):
        assert optim["batch_size"] == 128

    def test_t48_replay_50000(self, optim):
        assert optim["replay_capacity"] == 50000

    def test_t49_warmup_1000(self, optim):
        assert optim["warmup"] == 1000

    def test_t50_update_every_4(self, optim):
        assert optim["update_every"] == 4

    def test_t51_no_clipping(self, optim):
        assert optim["gradient_clipping"] == "NONE"


# ── T52-T54: Replay and exploration ──────────────────────

class TestReplayExploration:
    def test_t52_regional_anchor_replay(self, replay):
        assert replay["type"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

    def test_t53_sibling_dedup_false(self, replay):
        assert replay["SIBLING_DEDUPLICATION"] is False

    def test_t54_epsilon_1_to_05(self, explore):
        assert explore["epsilon_start"] == 1.0
        assert explore["epsilon_end"] == 0.05
        assert explore["epsilon_decay_transitions"] == 50000


# ── T55-T56: Max-Sum ─────────────────────────────────────

class TestMaxSum:
    def test_t55_maxsum_k10(self, maxsum):
        assert maxsum["K"] == 10
        assert maxsum["method"] == "MAX_SUM_MESSAGE_PASSING_V1"

    def test_t56_no_exhaustive_production(self, maxsum):
        assert maxsum["no_exhaustive_enumeration_in_action_path"] is True


# ── T57-T58: Checkpoints ─────────────────────────────────

class TestCheckpoints:
    def test_t57_checkpoint_cadence_six(self, ckpt):
        assert ckpt["n_per_run"] == 6

    def test_t58_total_checkpoints_60(self, ckpt):
        assert ckpt["CHECKPOINTS_PLANNED"] == 60


# ── T59-T65: Validation ──────────────────────────────────

class TestValidation:
    def test_t59_validation_seeds_untouched_in_freeze(self, seeds):
        assert seeds["validation"]["access_count"] == 0

    def test_t60_validation_total_3600(self, val_mat):
        assert val_mat["VALIDATION_EPISODES_PLANNED"] == 3600

    def test_t61_eval_epsilon_0(self, val_mode):
        assert val_mode["epsilon"] == 0.0
        assert val_mode["no_gradients"] is True

    def test_t62_macro_response_checkpoint_selection(self, ckpt_sel):
        assert ckpt_sel["metric"] == "MACRO_MIGD_RESPONSE"

    def test_t63_tie_macro_epoch(self, ckpt_sel):
        assert "lowest macro MIGD_epoch" in ckpt_sel["tie_break_order"]

    def test_t64_tie_fe(self, ckpt_sel):
        assert "lowest response FE" in ckpt_sel["tie_break_order"]

    def test_t65_tie_earlier_checkpoint(self, ckpt_sel):
        assert "earlier checkpoint" in ckpt_sel["tie_break_order"]


# ── T66-T70: Selection and validation boundary ───────────

class TestSelectionBoundary:
    def test_t66_exactly_10_selected(self, ckpt_sel):
        assert ckpt_sel["total_selected"] == 10
        assert ckpt_sel["n_selected_per_arm"] == 5

    def test_t67_no_best_policy_selection(self, ckpt_sel):
        assert ckpt_sel["no_best_policy_selection"] is True

    def test_t68_validation_non_confirmatory(self, val_bound):
        assert val_bound["validation_is"] == "NON_CONFIRMATORY"
        assert "causal_superiority_claim" in val_bound["forbidden"]

    def test_t69_c3_diagnostics_descriptive(self, diag_log):
        assert diag_log["all_diagnostics_are"] == "DESCRIPTIVE_ONLY"

    def test_t70_no_mechanism_success_gate(self, mech_gate):
        assert mech_gate["COORDINATION_MECHANISM_GATE"] == "NOT_USED"
        assert mech_gate["verdict_based_on"] == "FROZEN_MIGD_CRITERIA_ONLY"


# ── T71-T73: Completion and firewall ─────────────────────

class TestCompletionFirewall:
    def test_t71_training_completion_frozen(self, train_comp):
        r = train_comp["requires"]
        assert r["runs_complete"] == "10/10"
        assert r["training_episodes"] == "10800/10800"
        assert r["checkpoints"] == "60/60"

    def test_t72_validation_completion_frozen(self, val_comp):
        r = val_comp["requires"]
        assert r["validation_episodes"] == "3600/3600"
        assert r["selected_checkpoints"] == 10

    def test_t73_final_firewall_exact(self, firewall):
        assert firewall["FINAL_SEEDS"] == [2581, 2640]
        assert firewall["FINAL_ACCESS_COUNT"] == 0
        assert firewall["no_smoke_test"] is True
        assert firewall["no_integrity_test"] is True


# ── T74-T81: Future final statistical design ─────────────

class TestFutureStatistical:
    def test_t74_primary_final_mixed(self, future_stat):
        assert future_stat["primary_scenario"] == "MIXED"

    def test_t75_primary_final_migd_response(self, future_stat):
        assert future_stat["primary_endpoint"] == "MIGD_RESPONSE"

    def test_t76_effect_c1_c3(self, future_stat):
        assert future_stat["effect"] == "C1_MINUS_C3"

    def test_t77_positive_means_c3_better(self, future_stat):
        assert future_stat["positive_means"] == "C3_BETTER"

    def test_t78_practical_margin_1e5(self, future_stat):
        assert future_stat["practical_margin"] == 1e-5

    def test_t79_bootstrap_5x60(self, future_stat):
        b = future_stat["bootstrap"]
        assert b["axes"]["policy"] == 5
        assert b["axes"]["environment"] == 60

    def test_t80_bootstrap_50000(self, future_stat):
        assert future_stat["bootstrap"]["N"] == 50000

    def test_t81_bootstrap_seed_136001(self, future_stat):
        assert future_stat["bootstrap"]["seed"] == 136001


# ── T82-T85: Robustness and safety ───────────────────────

class TestRobustnessSafety:
    def test_t82_robustness_4_of_5(self, future_stat):
        assert future_stat["robustness"]["gate"] == "at_least_4_of_5_policies_positive"

    def test_t83_persistent_safety(self, future_stat):
        assert "PERSISTENT_MIGD_response" in future_stat["safety"]["no_harmful_in"]

    def test_t84_stationary_safety(self, future_stat):
        assert "STATIONARY_MIGD_response" in future_stat["safety"]["no_harmful_in"]

    def test_t85_macro_safety(self, future_stat):
        assert "MACRO_MIGD_response" in future_stat["safety"]["no_harmful_in"]


# ── T86-T88: Verdict logic ───────────────────────────────

class TestVerdictLogic:
    def test_t86_validated_logic(self, success):
        v = success["VALIDATED"]
        assert "primary MIXED = BENEFICIAL" in v["all_required"]
        assert "policy robustness >= 4/5" in v["all_required"]
        assert "PERSISTENT safety PASS" in v["all_required"]

    def test_t87_partial_logic(self, success):
        p = success["PARTIAL"]
        assert "primary = DIRECTIONAL_POSITIVE" in p["safety_and_integrity_pass_AND_one_of"]
        assert p["no_equivalence_claim_from_neutral"] is True

    def test_t88_failed_logic(self, success):
        f = success["FAILED"]
        assert "primary = HARMFUL" in f["any_of"]
        assert "any safety FAIL" in f["any_of"]


# ── T89-T93: Routing ─────────────────────────────────────

class TestRouting:
    def test_t89_validated_requires_c3_vs_c0(self, routing):
        assert "C3_VS_C0" in routing["VALIDATED"]["route"]
        assert routing["VALIDATED"]["no_transitive_inference"] is True

    def test_t90_no_transitive_claim(self, routing):
        assert routing["VALIDATED"]["no_transitive_inference"] is True

    def test_t91_failure_no_k_tuning(self, routing):
        assert routing["FAILED"]["no_K_adjustment"] is True

    def test_t92_failure_no_pair_width_tuning(self, routing):
        assert routing["FAILED"]["no_pair_width_tuning"] is True

    def test_t93_failure_no_graph_topology_tuning(self, routing):
        assert routing["FAILED"]["no_graph_topology_tuning"] is True


# ── T94-T95: Resume and protocol hash ────────────────────

class TestResumeHash:
    def test_t94_resume_contract_frozen(self, resume):
        assert resume["deterministic_resume"] is True
        assert "arm" in resume["training_progress_persists"]
        assert "policy_seed" in resume["training_progress_persists"]

    def test_t95_protocol_hash_stable(self, proto_hash, exec_cfg):
        config_str = json.dumps(exec_cfg["config"], sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(config_str.encode()).hexdigest()
        assert exec_cfg["E4RT_EXECUTION_CONFIG_SHA256"] == recomputed

        hash_input = json.dumps({
            "E4RT_EXECUTION_CONFIG_SHA256": recomputed,
            "C1_CONFIG_SHA256": C1_SHA,
            "C3_CONFIG_SHA256": C3_SHA,
            "stage": "E4RT_PAIRWISE_CG_AB_TRAINING_PROTOCOL_FREEZE"
        }, sort_keys=True, separators=(",", ":"))
        proto_recomputed = hashlib.sha256(hash_input.encode()).hexdigest()
        assert proto_hash["E4RT_PROTOCOL_SHA256"] == proto_recomputed


# ── T96-T99: No-change audits ────────────────────────────

class TestNoChangeAudits:
    def test_t96_no_c1_scientific_change(self, c1lock):
        assert c1lock["no_modification"] is True
        assert c1lock["QMIX_CONFIG_SHA256"] == C1_SHA

    def test_t97_no_c3_scientific_change(self, c3lock):
        assert c3lock["no_modification"] is True
        assert c3lock["C3_CONFIG_SHA256"] == C3_SHA

    def test_t98_no_training(self, verdict):
        assert verdict["NEW_TRAINING"] is False
        assert verdict["TRAIN_ACCESS_COUNT"] == 0

    def test_t99_no_environment_execution(self, verdict):
        assert verdict["NEW_ENVIRONMENT_EXECUTION"] is False


# ── T100: Full regression delegated to external run ──────

class TestRegressionMarker:
    def test_t100_full_regression_marker(self, verdict):
        assert verdict["E4RT_COMPLETE"] == "YES"
        assert verdict["NEXT_STAGE"] == "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION"


# ── Additional checks ────────────────────────────────────

class TestAdditionalChecks:
    def test_all_required_files_exist(self):
        required = [
            "E4RT_pre_manifest.json",
            "E4RT_historical_integrity.json",
            "E4RT_arm_contract.json",
            "E4RT_c1_config_lock.json",
            "E4RT_c3_config_lock.json",
            "E4RT_causal_factor_lock.json",
            "E4RT_initialization_pairing.json",
            "E4RT_parameter_count_disclosure.json",
            "E4RT_seed_contract.json",
            "E4RT_seed_disjointness.json",
            "E4RT_training_matrix.json",
            "E4RT_training_order.json",
            "E4RT_pairing_contract.json",
            "E4RT_training_objective_contract.json",
            "E4RT_reward_contract.json",
            "E4RT_optimizer_contract.json",
            "E4RT_replay_contract.json",
            "E4RT_exploration_contract.json",
            "E4RT_maxsum_training_contract.json",
            "E4RT_checkpoint_contract.json",
            "E4RT_checkpoint_schema.json",
            "E4RT_validation_matrix.json",
            "E4RT_validation_mode.json",
            "E4RT_checkpoint_selection.json",
            "E4RT_validation_status_boundary.json",
            "E4RT_diagnostic_logging_contract.json",
            "E4RT_mechanism_gate_boundary.json",
            "E4RT_training_completion_contract.json",
            "E4RT_validation_completion_contract.json",
            "E4RT_final_seed_firewall.json",
            "E4RT_future_final_contract.json",
            "E4RT_future_statistical_contract.json",
            "E4RT_future_success_logic.json",
            "E4RT_future_routing.json",
            "E4RT_resume_contract.json",
            "E4RT_execution_config.json",
            "E4RT_protocol_hash.json",
            "E4RT_protocol_frozen.json",
            "E4RT_verdict.json",
            "E4RT_report.md",
        ]
        for f in required:
            assert os.path.isfile(os.path.join(E4RT, f)), f"Missing: {f}"

    def test_verdict_preserved_verdicts(self, verdict):
        v = verdict["PRESERVED_VERDICTS"]
        assert v["E4F"] == "FAILED"
        assert v["E4HF"] == "FAILED"
        assert v["E4IF"] == "REWARD_CAUSAL_VALIDATED"
        assert v["E4JF"] == "GLOBAL_COMPARATOR_FAILED"
        assert v["E4KF"] == "COORDINATION_CAUSAL_FAILED"
        assert v["E4MF"] == "FAILED"

    def test_verdict_performance_not_computed(self, verdict):
        assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_c3_checkpoint_schema(self, ckpt_schema):
        c3 = ckpt_schema["C3_checkpoint_contains"]
        assert "unary_online_state_dict" in c3
        assert "pair_online_state_dict" in c3
        assert "C3_CONFIG_SHA256" in c3

    def test_future_final_contract(self, future_final):
        assert future_final["FINAL_EPISODES_PLANNED"] == 1800
        assert future_final["do_NOT_immediately_run_final"] is True

    def test_protocol_frozen(self, frozen):
        assert frozen["E4RT_PROTOCOL_FROZEN"] is True
        assert frozen["no_scientific_implementation_change"] is True
        assert frozen["no_training"] is True

    def test_e4r_e4ri_sha_match(self):
        e4ri_c3 = load(E4RI, "E4RI_c3_config_hash.json")
        assert e4ri_c3["C3_CONFIG_SHA256"] == C3_SHA
        e4ri_c1 = load(E4RI, "E4RI_c1_backward_equivalence.json")
        assert e4ri_c1["QMIX_CONFIG_SHA256"] == C1_SHA

    def test_training_matrix_consistency(self, train_mat):
        assert train_mat["n_arms"] * train_mat["policies_per_arm"] * train_mat["episodes_per_run"] == 10800
        assert train_mat["epochs_per_run"] * train_mat["episodes_per_epoch"] == train_mat["episodes_per_run"]

    def test_validation_matrix_consistency(self, val_mat):
        expected = 2 * 5 * 6 * 20 * 3
        assert val_mat["VALIDATION_EPISODES_PLANNED"] == expected

    def test_no_c2_loss_in_c3(self, obj):
        assert obj["C3_objective"]["no_C2_composite_loss"] is True

    def test_c3_maxsum_remains_deployment(self, val_mode):
        assert val_mode["C3_maxsum_K10_remains_deployment"] is True

    def test_no_replay_reweighting(self, replay):
        assert replay["no_replay_event_reweighting"] is True

    def test_exploration_no_boltzmann(self, explore):
        assert explore["no_Boltzmann"] is True

    def test_no_checkpoint_modification(self, ckpt):
        assert ckpt["no_checkpoint_modification"] is True

    def test_c3_no_auxiliary(self, c3lock):
        assert c3lock["no_regional_auxiliary_loss"] is True
        assert c3lock["no_C2_composite_loss"] is True

    def test_safety_includes_epoch(self, future_stat):
        safety = future_stat["safety"]["no_harmful_in"]
        assert "PERSISTENT_MIGD_epoch" in safety
        assert "STATIONARY_MIGD_epoch" in safety
        assert "MACRO_MIGD_epoch" in safety

    def test_classification_labels(self, success):
        c = success["classification"]
        assert "BENEFICIAL" in c
        assert "DIRECTIONAL_POSITIVE" in c
        assert "PRACTICALLY_NEUTRAL" in c
        assert "DIRECTIONAL_NEGATIVE" in c
        assert "HARMFUL" in c

    def test_no_scenario_specific_checkpoint(self, ckpt_sel):
        assert ckpt_sel["no_scenario_specific_checkpoint"] is True

    def test_future_routing_no_post_hoc(self, routing):
        f = routing["FAILED"]
        assert f["no_post_hoc_tuning"] is True
        assert f["no_pair_coefficient_tuning"] is True
        assert f["no_centering_change"] is True
        assert f["no_replay_change"] is True
