"""E4TT: C4 Gated Coordination Training Protocol Freeze — 22-audit validation.

Tests that the frozen three-arm (C1/C3/C4) training protocol is internally
consistent, matches upstream frozen artifacts, and preserves all invariants.
"""

import json
import hashlib
import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
E4TT_DIR = RESULTS_DIR / "E4TT_c4_gated_coordination_training_protocol_freeze"
E4RT_DIR = RESULTS_DIR / "E4RT_pairwise_cg_ab_training_protocol_freeze"
E4T_DIR = RESULTS_DIR / "E4T_causal_dynamics_gated_coordination_design"
E4TI_DIR = RESULTS_DIR / "E4TI_c4_gated_coordination_implementation_validation"


def _load(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# T1: Arm training fairness
# ============================================================================

class TestT1ArmTrainingFairness:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.arm = _load(E4TT_DIR / "E4TT_arm_contract.json")
        self.init = _load(E4TT_DIR / "E4TT_initialization_pairing.json")

    def test_three_arms(self):
        assert self.arm["n_arms"] == 3

    def test_arm_names(self):
        arms = self.arm["arms"]
        assert "A_CONTROL" in arms
        assert "B_TREATMENT" in arms
        assert "C_TREATMENT" in arms
        assert arms["A_CONTROL"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
        assert arms["B_TREATMENT"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"
        assert arms["C_TREATMENT"] == "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1"

    def test_shared_hyperparameters(self):
        hp = self.arm["SHARED_HYPERPARAMETERS"]
        assert hp["optimizer"] == "Adam"
        assert hp["lr"] == 1e-4
        assert hp["gamma"] == 0.99
        assert hp["tau"] == 0.005
        assert hp["batch_size"] == 128
        assert hp["replay_capacity"] == 50000
        assert hp["warmup"] == 1000
        assert hp["update_every"] == 4
        assert hp["gradient_clipping"] == "NONE"
        assert hp["epsilon_start"] == 1.0
        assert hp["epsilon_end"] == 0.05
        assert hp["epsilon_decay_transitions"] == 50000

    def test_c4_params_equal_c3(self):
        assert self.arm["ARM_SPECIFIC"]["C4"]["total_params"] == 39316
        assert self.arm["ARM_SPECIFIC"]["C3"]["total_params"] == 39316

    def test_c4_gate_zero_params(self):
        assert self.arm["ARM_SPECIFIC"]["C4"]["gate"] == "deterministic_causal_pair_gate_0_trainable_params"

    def test_paired_unary_init(self):
        assert self.init["paired_unary_initialization"] is True
        assert "C1_UTILITY_INIT_SHA[p] == C3_UNARY_INIT_SHA[p] == C4_UNARY_INIT_SHA[p]" in self.init["requirement"]

    def test_paired_pair_init(self):
        assert "C3_PAIR_INIT_SHA[p] == C4_PAIR_INIT_SHA[p]" in self.init["pair_pairing"]

    def test_c3_c4_no_initial_difference(self):
        assert self.init["C3_vs_C4_INITIAL_PARAMETER_DIFFERENCE"] == "NONE"

    def test_only_gate_difference(self):
        assert self.init["ONLY_EXECUTION_TRAINING_DIFFERENCE"] == "deterministic_causal_gate"

    def test_excluded_controllers(self):
        excluded = self.arm["excluded"]
        for c in ["C0_INDEPENDENT_SHARED_DDQN", "C2_FOCAL_REGIONAL_CREDIT",
                   "GLOBAL_FIXED_COMPARATOR", "QPLEX", "QTRAN", "C5_ANY"]:
            assert c in excluded

    def test_controller_revision_budget_zero(self):
        assert self.arm["CONTROLLER_REVISION_BUDGET_AFTER_C4"] == "ZERO"

    def test_upstream_e4t_three_arm(self):
        e4t = _load(E4T_DIR / "E4T_three_arm_design.json")
        assert e4t["ARMS"]["A"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
        assert e4t["ARMS"]["B"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"
        assert e4t["ARMS"]["C"] == "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1"

    def test_upstream_e4t_init_pairing(self):
        e4t_init = _load(E4T_DIR / "E4T_initialization_pairing_contract.json")
        assert e4t_init["C3_vs_C4_INITIAL_PARAMETER_DIFFERENCE"] == "NONE"

    def test_t1_pass(self):
        assert self.arm["T1_PASS"] is True


# ============================================================================
# T2: Train seed mapping
# ============================================================================

class TestT2TrainSeedMapping:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seed = _load(E4TT_DIR / "E4TT_seed_contract.json")

    def test_train_range(self):
        assert self.seed["train"]["range"] == [2701, 2760]
        assert self.seed["train"]["n"] == 60

    def test_validation_range(self):
        assert self.seed["validation"]["range"] == [2761, 2780]
        assert self.seed["validation"]["n"] == 20

    def test_final_range(self):
        assert self.seed["final"]["range"] == [2781, 2840]
        assert self.seed["final"]["n"] == 60

    def test_policy_seeds(self):
        assert self.seed["policy"] == [8001, 8002, 8003, 8004, 8005]

    def test_bootstrap_seed(self):
        assert self.seed["bootstrap_seed"] == 146001

    def test_shared_across_arms(self):
        assert self.seed["train"]["SHARED_ACROSS_ALL_THREE_ARMS"] is True
        assert self.seed["validation"]["SHARED_ACROSS_ALL_THREE_ARMS"] is True
        assert self.seed["final"]["SHARED_ACROSS_ALL_THREE_ARMS"] is True

    def test_disjoint_from_e4rt(self):
        disjoint = self.seed["DISJOINT_FROM_E4RT"]
        assert disjoint["OVERLAP"] == "NONE"
        assert disjoint["E4RT_train"] == [2501, 2560]
        assert disjoint["E4RT_policy"] == [7001, 7005]
        assert disjoint["E4RT_bootstrap"] == 136001

    def test_no_seed_overlap_numerical(self):
        e4tt_train = set(range(2701, 2761))
        e4tt_val = set(range(2761, 2781))
        e4tt_final = set(range(2781, 2841))
        e4rt_train = set(range(2501, 2561))
        e4rt_val = set(range(2561, 2581))
        e4rt_final = set(range(2581, 2641))
        all_e4tt = e4tt_train | e4tt_val | e4tt_final
        all_e4rt = e4rt_train | e4rt_val | e4rt_final
        assert len(all_e4tt & all_e4rt) == 0

    def test_e4tt_seeds_internally_disjoint(self):
        e4tt_train = set(range(2701, 2761))
        e4tt_val = set(range(2761, 2781))
        e4tt_final = set(range(2781, 2841))
        assert len(e4tt_train & e4tt_val) == 0
        assert len(e4tt_train & e4tt_final) == 0
        assert len(e4tt_val & e4tt_final) == 0

    def test_access_count_zero(self):
        assert self.seed["train"]["access_count"] == 0
        assert self.seed["validation"]["access_count"] == 0
        assert self.seed["final"]["access_count"] == 0

    def test_t2_pass(self):
        assert self.seed["T2_PASS"] is True


# ============================================================================
# T3: Policy seed mapping and RNG hierarchy
# ============================================================================

class TestT3PolicySeedMapping:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.expl = _load(E4TT_DIR / "E4TT_exploration_contract.json")
        self.init = _load(E4TT_DIR / "E4TT_initialization_pairing.json")

    def test_policy_seeds(self):
        assert self.expl["policy_seeds"] == [8001, 8002, 8003, 8004, 8005]

    def test_pair_seed_derivation(self):
        assert self.expl["pair_seed_derivation"] == "policy_seed + 1"

    def test_pair_seeds(self):
        assert self.init["pair_seeds"] == [8002, 8003, 8004, 8005, 8006]

    def test_epsilon_greedy_rng(self):
        assert "EVENT_SEED_XOR_REGION_ID" in self.expl["epsilon_greedy_rng"]

    def test_c4_exploration(self):
        assert "gated-MaxSum" in self.expl["C4_exploration"]

    def test_no_boltzmann(self):
        assert self.expl["no_Boltzmann"] is True

    def test_t3_pass(self):
        assert self.expl["T3_PASS"] is True


# ============================================================================
# T4: Train episodes = 16200
# ============================================================================

class TestT4TrainEpisodes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.matrix = _load(E4TT_DIR / "E4TT_training_matrix.json")

    def test_total_episodes(self):
        assert self.matrix["TRAIN_EPISODES_PLANNED"] == 16200

    def test_formula(self):
        n = (self.matrix["n_arms"] * self.matrix["policies_per_arm"] *
             self.matrix["epochs_per_run"] * self.matrix["env_seeds_per_epoch"] *
             self.matrix["scenarios_per_epoch"])
        assert n == 16200

    def test_total_runs(self):
        assert self.matrix["total_runs"] == 15

    def test_episodes_per_run(self):
        assert self.matrix["episodes_per_run"] == 1080

    def test_n_arms(self):
        assert self.matrix["n_arms"] == 3

    def test_scenarios(self):
        assert self.matrix["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]

    def test_t4_pass(self):
        assert self.matrix["T4_PASS"] is True


# ============================================================================
# T5: Checkpoints = 90
# ============================================================================

class TestT5Checkpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ckpt = _load(E4TT_DIR / "E4TT_checkpoint_contract.json")

    def test_total_checkpoints(self):
        assert self.ckpt["CHECKPOINTS_PLANNED"] == 90

    def test_formula(self):
        assert self.ckpt["total_runs"] * self.ckpt["n_per_run"] == 90

    def test_schedule(self):
        assert self.ckpt["schedule_transitions"] == [10000, 20000, 30000, 40000, 50000, "FINAL"]

    def test_n_per_run(self):
        assert self.ckpt["n_per_run"] == 6

    def test_no_additional_checkpoint(self):
        assert self.ckpt["no_additional_best_train_checkpoint"] is True

    def test_t5_pass(self):
        assert self.ckpt["T5_PASS"] is True


# ============================================================================
# T6: Validation episodes = 5400
# ============================================================================

class TestT6ValidationEpisodes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.val = _load(E4TT_DIR / "E4TT_validation_matrix.json")

    def test_total_validation(self):
        assert self.val["VALIDATION_EPISODES_PLANNED"] == 5400

    def test_formula(self):
        n = (self.val["arms"] * self.val["policies_per_arm"] *
             self.val["checkpoints_per_run"] * self.val["n_seeds"] *
             self.val["n_scenarios"])
        assert n == 5400

    def test_t6_pass(self):
        assert self.val["T6_PASS"] is True


# ============================================================================
# T7: Final episodes = 2700
# ============================================================================

class TestT7FinalEpisodes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.final = _load(E4TT_DIR / "E4TT_final_matrix.json")

    def test_total_final(self):
        assert self.final["FINAL_EPISODES_PLANNED"] == 2700

    def test_formula(self):
        n = (self.final["arms"] * self.final["policies_per_arm"] *
             self.final["n_seeds"] * self.final["n_scenarios"])
        assert n == 2700

    def test_t7_pass(self):
        assert self.final["T7_PASS"] is True


# ============================================================================
# T8: Checkpoint selection rule
# ============================================================================

class TestT8CheckpointSelection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sel = _load(E4TT_DIR / "E4TT_checkpoint_selection.json")

    def test_metric(self):
        assert self.sel["metric"] == "MACRO_MIGD_RESPONSE"

    def test_lower_is_better(self):
        assert self.sel["lower_is_better"] is True

    def test_selection_per(self):
        assert self.sel["selection_per"] == "arm x policy_seed"

    def test_total_selected(self):
        assert self.sel["total_selected"] == 15

    def test_n_selected_per_arm(self):
        assert self.sel["n_selected_per_arm"] == 5

    def test_tie_break(self):
        assert len(self.sel["tie_break_order"]) == 3

    def test_no_best_policy_selection(self):
        assert self.sel["no_best_policy_selection"] is True

    def test_matches_e4rt(self):
        e4rt = _load(E4RT_DIR / "E4RT_checkpoint_selection.json")
        assert self.sel["metric"] == e4rt["metric"]
        assert self.sel["lower_is_better"] == e4rt["lower_is_better"]
        assert self.sel["tie_break_order"] == e4rt["tie_break_order"]

    def test_t8_pass(self):
        assert self.sel["T8_PASS"] is True


# ============================================================================
# T9: Training failure policy
# ============================================================================

class TestT9FailurePolicy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fp = _load(E4TT_DIR / "E4TT_failure_policy.json")

    def test_absolute_rule(self):
        assert self.fp["ABSOLUTE_RULE"] == "ACCEPT_RESULT_EXACTLY_AS_OBSERVED"

    def test_no_adaptation(self):
        assert self.fp["NO_ADAPTATION_AFTER_ANY_SEED_OPENED"] is True

    def test_no_seed_replacement(self):
        assert self.fp["on_training_crash"]["no_seed_replacement"] is True

    def test_no_nan_recovery(self):
        assert self.fp["on_nan_loss"]["no_automatic_recovery"] is True

    def test_t9_pass(self):
        assert self.fp["T9_PASS"] is True


# ============================================================================
# T10: Determinism policy
# ============================================================================

class TestT10DeterminismPolicy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.det = _load(E4TT_DIR / "E4TT_determinism_policy.json")

    def test_network_init(self):
        assert self.det["training_determinism"]["network_init"] == "DETERMINISTIC_FROM_POLICY_SEED"

    def test_gate_determinism(self):
        assert self.det["C4_gate_determinism"]["no_stochastic_gate"] is True
        assert self.det["C4_gate_determinism"]["no_gate_dropout"] is True
        assert self.det["C4_gate_determinism"]["no_gate_noise"] is True

    def test_reproducibility(self):
        assert self.det["reproducibility"]["same_seeds_same_result"] is True

    def test_t10_pass(self):
        assert self.det["T10_PASS"] is True


# ============================================================================
# T11: Replay buffer isolation
# ============================================================================

class TestT11ReplayIsolation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rep = _load(E4TT_DIR / "E4TT_replay_contract.json")

    def test_type(self):
        assert self.rep["type"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

    def test_isolation(self):
        iso = self.rep["ISOLATION_POLICY"]
        assert iso["no_cross_arm_sharing"] is True
        assert iso["no_cross_policy_sharing"] is True
        assert iso["C1_replay_isolated_from_C3"] is True
        assert iso["C1_replay_isolated_from_C4"] is True
        assert iso["C3_replay_isolated_from_C4"] is True

    def test_no_sibling_dedup(self):
        assert self.rep["SIBLING_DEDUPLICATION"] is False

    def test_t11_pass(self):
        assert self.rep["T11_PASS"] is True


# ============================================================================
# T12: Training condition schedule
# ============================================================================

class TestT12TrainingOrder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.order = _load(E4TT_DIR / "E4TT_training_order.json")

    def test_outer_order(self):
        assert "8001..8005" in self.order["canonical_order"]["outer"]

    def test_paired_order(self):
        assert self.order["canonical_order"]["paired"] == "C1 then C3 then C4 per policy seed"

    def test_epoch_order(self):
        assert self.order["canonical_order"]["epoch"] == "1..6"

    def test_env_order(self):
        assert "2701..2760" in self.order["canonical_order"]["environment"]

    def test_scenarios(self):
        assert self.order["canonical_order"]["scenario"] == ["PERSISTENT", "STATIONARY", "MIXED"]

    def test_deterministic(self):
        assert self.order["deterministic"] is True

    def test_t12_pass(self):
        assert self.order["T12_PASS"] is True


# ============================================================================
# T13: C4 gate training behavior
# ============================================================================

class TestT13GateTrainingBehavior:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.gate = _load(E4TT_DIR / "E4TT_gate_training_behavior.json")

    def test_zero_trainable_params(self):
        assert self.gate["gate_trainable_parameters"] == 0

    def test_not_in_optimizer(self):
        assert self.gate["gate_in_optimizer"] is False

    def test_no_gradients(self):
        assert self.gate["gate_receives_gradients"] is False

    def test_no_temperature(self):
        assert self.gate["no_temperature"] is True

    def test_no_threshold(self):
        assert self.gate["no_threshold"] is True

    def test_no_gate_loss(self):
        assert self.gate["no_gate_loss"] is True

    def test_gate_formula(self):
        assert self.gate["gate_formula"] == "g_ij = 0.5 * (|d_i| + |d_j|)"

    def test_stationary_behavior(self):
        assert "g_ij=0" in self.gate["STATIONARY_BEHAVIOR"]
        assert "unary-only" in self.gate["STATIONARY_BEHAVIOR"]

    def test_maxsum_k10(self):
        assert self.gate["maxsum_K"] == 10

    def test_code_verify(self):
        from prdrl.rl.coordination_graph import compute_pair_gate, GATE_SIGNAL_DIM
        import torch
        s_i = torch.zeros(1, 9)
        s_j = torch.zeros(1, 9)
        g = compute_pair_gate(s_i, s_j)
        assert g.item() == 0.0
        s_i[0, GATE_SIGNAL_DIM] = 0.5
        s_j[0, GATE_SIGNAL_DIM] = -0.3
        g = compute_pair_gate(s_i, s_j)
        expected = 0.5 * (0.5 + 0.3)
        assert abs(g.item() - expected) < 1e-6

    def test_t13_pass(self):
        assert self.gate["T13_PASS"] is True


# ============================================================================
# T14: Loss equations
# ============================================================================

class TestT14LossEquations:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.obj = _load(E4TT_DIR / "E4TT_training_objective_contract.json")

    def test_c1_objective(self):
        assert self.obj["C1_objective"]["type"] == "JOINT_DOUBLE_DQN_QMIX"

    def test_c3_objective(self):
        assert self.obj["C3_objective"]["type"] == "JOINT_DOUBLE_DQN_CG"
        assert self.obj["C3_objective"]["loss"] == "SmoothL1(q_online, y)"

    def test_c4_objective(self):
        assert self.obj["C4_objective"]["type"] == "JOINT_DOUBLE_DQN_GATED_CG"
        assert self.obj["C4_objective"]["loss"] == "SmoothL1(q_online, y)"
        assert self.obj["C4_objective"]["IDENTICAL_TO_C3_LOSS_FUNCTION"] is True

    def test_c4_only_difference(self):
        assert "gated pair potentials" in self.obj["C4_objective"]["ONLY_DIFFERENCE"]

    def test_no_auxiliary_losses(self):
        assert self.obj["C3_objective"]["no_C2_composite_loss"] is True
        assert self.obj["C4_objective"]["no_C2_composite_loss"] is True
        assert self.obj["C4_objective"]["no_gate_loss"] is True

    def test_code_loss_identical(self):
        assert self.obj["CODE_VERIFIED"]["IDENTICAL"] is True

    def test_t14_pass(self):
        assert self.obj["T14_PASS"] is True


# ============================================================================
# T15: Optimizer state policy
# ============================================================================

class TestT15OptimizerPolicy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.opt = _load(E4TT_DIR / "E4TT_optimizer_contract.json")

    def test_adam(self):
        assert self.opt["optimizer"] == "Adam"

    def test_lr(self):
        assert self.opt["lr"] == 1e-4

    def test_c4_optimizer_includes(self):
        assert self.opt["C4_optimizer_includes"] == ["unary_online", "pair_online"]

    def test_c4_gate_not_in_optimizer(self):
        assert self.opt["C4_gate_in_optimizer"] is False

    def test_c3_c4_identical(self):
        assert self.opt["C3_C4_optimizer_params_identical"] is True

    def test_no_gradient_clipping(self):
        assert self.opt["gradient_clipping"] == "NONE"

    def test_code_verified(self):
        assert self.opt["CODE_VERIFIED"]["IDENTICAL"] is True

    def test_t15_pass(self):
        assert self.opt["T15_PASS"] is True


# ============================================================================
# T16: Logging protocol
# ============================================================================

class TestT16LoggingProtocol:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.log = _load(E4TT_DIR / "E4TT_diagnostic_logging_contract.json")

    def test_all_arms_base_fields(self):
        base = self.log["all_arms"]
        for field in ["TD_loss", "Q_magnitudes", "action_fractions", "replay_size"]:
            assert field in base

    def test_c4_extra_gate_fields(self):
        c4_extra = self.log["C4_EXTRA_OVER_C3"]
        for field in ["gate_mean", "gate_std", "gate_fraction_zero",
                       "gated_pair_factor_mean", "gated_pair_factor_std"]:
            assert field in c4_extra

    def test_c4_includes_c3_fields(self):
        c3 = set(self.log["C3_additionally"])
        c4 = set(self.log["C4_additionally"])
        assert c3.issubset(c4)

    def test_descriptive_only(self):
        assert self.log["all_diagnostics_are"] == "DESCRIPTIVE_ONLY"

    def test_t16_pass(self):
        assert self.log["T16_PASS"] is True


# ============================================================================
# T17: Run-ID / artifact naming
# ============================================================================

class TestT17RunIdNaming:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.naming = _load(E4TT_DIR / "E4TT_run_id_naming.json")

    def test_format(self):
        assert self.naming["run_id_format"] == "{arm}_{policy_seed}"

    def test_total_runs(self):
        assert self.naming["total_runs"] == 15

    def test_examples_count(self):
        assert len(self.naming["examples"]) == 15

    def test_c4_runs_present(self):
        examples = self.naming["examples"]
        c4_runs = [e for e in examples if e.startswith("C4_")]
        assert len(c4_runs) == 5

    def test_t17_pass(self):
        assert self.naming["T17_PASS"] is True


# ============================================================================
# T18: Configuration hash lock
# ============================================================================

class TestT18ConfigurationHash:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.hash = _load(E4TT_DIR / "E4TT_protocol_hash.json")

    def test_protocol_hash_present(self):
        assert len(self.hash["E4TT_PROTOCOL_SHA256"]) == 64

    def test_exec_config_hash_present(self):
        assert len(self.hash["E4TT_EXECUTION_CONFIG_SHA256"]) == 64

    def test_c1_sha_matches(self):
        assert self.hash["C1_CONFIG_SHA256"] == "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"

    def test_c3_sha_matches(self):
        assert self.hash["C3_CONFIG_SHA256"] == "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"

    def test_c4_sha_matches(self):
        assert self.hash["C4_CONFIG_SHA256"] == "a478392e8c0be2bbf911fca4e69aeb209cec31009fb61a4af746277d9043c4f2"

    def test_exec_config_hash_reproducible(self):
        exec_cfg = _load(E4TT_DIR / "E4TT_execution_config.json")
        payload = json.dumps(exec_cfg, sort_keys=True, separators=(",", ":"))
        computed = hashlib.sha256(payload.encode()).hexdigest()
        assert computed == self.hash["E4TT_EXECUTION_CONFIG_SHA256"]

    def test_protocol_hash_reproducible(self):
        protocol_input = json.dumps({
            "exec_config_hash": self.hash["E4TT_EXECUTION_CONFIG_SHA256"],
            "C1_SHA": self.hash["C1_CONFIG_SHA256"],
            "C3_SHA": self.hash["C3_CONFIG_SHA256"],
            "C4_SHA": self.hash["C4_CONFIG_SHA256"],
            "stage": "E4TT_C4_GATED_COORDINATION_TRAINING_PROTOCOL_FREEZE"
        }, sort_keys=True, separators=(",", ":"))
        computed = hashlib.sha256(protocol_input.encode()).hexdigest()
        assert computed == self.hash["E4TT_PROTOCOL_SHA256"]

    def test_c4_sha_matches_code(self):
        from prdrl.rl.coordination_graph import c4_config_sha256
        assert c4_config_sha256() == self.hash["C4_CONFIG_SHA256"]

    def test_c3_sha_matches_code(self):
        from prdrl.rl.coordination_graph import c3_config_sha256
        assert c3_config_sha256() == self.hash["C3_CONFIG_SHA256"]

    def test_t18_pass(self):
        assert self.hash["T18_PASS"] is True


# ============================================================================
# T19: Statistical protocol preservation
# ============================================================================

class TestT19StatisticalProtocol:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.stat = _load(E4TT_DIR / "E4TT_statistical_protocol.json")

    def test_bootstrap_method(self):
        assert self.stat["bootstrap_method"] == "paired_crossed_two_axis_bootstrap"

    def test_bootstrap_n(self):
        assert self.stat["bootstrap_N"] == 50000

    def test_bootstrap_seed(self):
        assert self.stat["bootstrap_seed"] == 146001

    def test_bootstrap_axes(self):
        assert self.stat["bootstrap_axes"] == "5 x 60"

    def test_gating_comparison(self):
        gating = self.stat["comparisons"]["GATING_MECHANISM"]
        assert gating["effect"] == "C3_MINUS_C4"
        assert gating["primary_scenario"] == "STATIONARY"

    def test_final_controller_comparison(self):
        final = self.stat["comparisons"]["FINAL_CONTROLLER"]
        assert final["effect"] == "C1_MINUS_C4"
        assert final["primary_scenario"] == "MIXED"

    def test_classification(self):
        cls = self.stat["classification"]
        assert cls["DELTA_PRACTICAL"] == 1e-5
        assert len(cls["categories"]) == 5
        assert cls["lower_is_better"] is True

    def test_t19_pass(self):
        assert self.stat["T19_PASS"] is True


# ============================================================================
# T20: Final seed firewall
# ============================================================================

class TestT20SeedFirewall:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fw = _load(E4TT_DIR / "E4TT_seed_firewall.json")

    def test_all_sealed(self):
        for phase in ["train", "validation", "final"]:
            assert self.fw["RESERVED_SEEDS"][phase]["accessed"] == 0
            assert self.fw["RESERVED_SEEDS"][phase]["status"] == "SEALED"

    def test_no_overlap(self):
        assert self.fw["DISJOINT_FROM_ALL_PRIOR_EXPERIMENTS"]["OVERLAP_WITH_E4TT"] == "NONE"

    def test_firewall_rules(self):
        rules = self.fw["FIREWALL_RULES"]
        assert rules["no_seed_accessed_before_training_execution"] is True
        assert rules["no_seed_replacement_after_opened"] is True
        assert rules["no_cherry_picking_seeds"] is True
        assert rules["no_training_runs_in_E4TT"] is True

    def test_t20_pass(self):
        assert self.fw["T20_PASS"] is True


# ============================================================================
# T21: No-training firewall
# ============================================================================

class TestT21NoTrainingFirewall:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ntf = _load(E4TT_DIR / "E4TT_no_training_firewall.json")

    def test_protocol_freeze_only(self):
        assert self.ntf["E4TT_IS_PROTOCOL_FREEZE_ONLY"] is True

    def test_no_training(self):
        assert self.ntf["NO_TRAINING_RUNS_IN_THIS_STAGE"] is True

    def test_no_seeds_accessed(self):
        assert self.ntf["NO_SEEDS_ACCESSED"] is True

    def test_no_gradient_updates(self):
        assert self.ntf["NO_GRADIENT_UPDATES"] is True

    def test_next_stage(self):
        assert self.ntf["TRAINING_BEGINS_IN"] == "E4TTR_C4_GATED_COORDINATION_TRAINING_EXECUTION"

    def test_t21_pass(self):
        assert self.ntf["T21_PASS"] is True


# ============================================================================
# Verdict file integrity
# ============================================================================

class TestVerdict:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.verdict = _load(E4TT_DIR / "E4TT_verdict.json")

    def test_stage(self):
        assert self.verdict["stage"] == "E4TT_C4_GATED_COORDINATION_TRAINING_PROTOCOL_FREEZE"

    def test_total_audits(self):
        assert self.verdict["total_audits"] == 22

    def test_passed_plus_pending(self):
        assert self.verdict["passed"] + self.verdict["pending"] + self.verdict["failed"] == 22

    def test_no_failures(self):
        assert self.verdict["failed"] == 0

    def test_protocol_hash(self):
        assert len(self.verdict["E4TT_PROTOCOL_SHA256"]) == 64

    def test_next_stage(self):
        assert self.verdict["NEXT_STAGE"] == "E4TTR_C4_GATED_COORDINATION_TRAINING_EXECUTION"


# ============================================================================
# Execution config consistency
# ============================================================================

class TestExecutionConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = _load(E4TT_DIR / "E4TT_execution_config.json")["config"]

    def test_n_arms(self):
        assert self.cfg["n_arms"] == 3

    def test_total_training_episodes(self):
        assert self.cfg["total_training_episodes"] == 16200

    def test_total_checkpoints(self):
        assert self.cfg["total_checkpoints"] == 90

    def test_validation_episodes(self):
        assert self.cfg["validation_episodes"] == 5400

    def test_final_episodes(self):
        assert self.cfg["final_episodes"] == 2700

    def test_c4_sha(self):
        assert self.cfg["C4_CONFIG_SHA256"] == "a478392e8c0be2bbf911fca4e69aeb209cec31009fb61a4af746277d9043c4f2"

    def test_seeds_match_contract(self):
        seed = _load(E4TT_DIR / "E4TT_seed_contract.json")
        assert self.cfg["train_seeds"] == seed["train"]["range"]
        assert self.cfg["validation_seeds"] == seed["validation"]["range"]
        assert self.cfg["final_seeds"] == seed["final"]["range"]
        assert self.cfg["policy_seeds"] == seed["policy"]
        assert self.cfg["bootstrap_seed"] == seed["bootstrap_seed"]

    def test_comparisons(self):
        assert self.cfg["comparisons"]["GATING_MECHANISM"]["effect"] == "C3_MINUS_C4"
        assert self.cfg["comparisons"]["FINAL_CONTROLLER"]["effect"] == "C1_MINUS_C4"

    def test_hyperparams_match_e4rt(self):
        e4rt = _load(E4RT_DIR / "E4RT_execution_config.json")["config"]
        for key in ["optimizer", "lr", "gamma", "tau", "batch_size",
                     "replay_capacity", "warmup", "update_every",
                     "gradient_clipping", "epsilon_start", "epsilon_end",
                     "epsilon_decay_transitions", "maxsum_K"]:
            assert self.cfg[key] == e4rt[key], f"Mismatch on {key}: {self.cfg[key]} != {e4rt[key]}"


# ============================================================================
# Cross-file consistency
# ============================================================================

class TestCrossFileConsistency:
    def test_all_artifacts_exist(self):
        expected = [
            "E4TT_arm_contract.json",
            "E4TT_seed_contract.json",
            "E4TT_initialization_pairing.json",
            "E4TT_training_matrix.json",
            "E4TT_checkpoint_contract.json",
            "E4TT_validation_matrix.json",
            "E4TT_final_matrix.json",
            "E4TT_checkpoint_selection.json",
            "E4TT_failure_policy.json",
            "E4TT_determinism_policy.json",
            "E4TT_replay_contract.json",
            "E4TT_training_order.json",
            "E4TT_gate_training_behavior.json",
            "E4TT_training_objective_contract.json",
            "E4TT_optimizer_contract.json",
            "E4TT_exploration_contract.json",
            "E4TT_diagnostic_logging_contract.json",
            "E4TT_run_id_naming.json",
            "E4TT_maxsum_training_contract.json",
            "E4TT_validation_mode.json",
            "E4TT_resume_contract.json",
            "E4TT_statistical_protocol.json",
            "E4TT_seed_firewall.json",
            "E4TT_no_training_firewall.json",
            "E4TT_protocol_hash.json",
            "E4TT_execution_config.json",
            "E4TT_verdict.json",
            "E4TT_report.md",
        ]
        for name in expected:
            assert (E4TT_DIR / name).exists(), f"Missing: {name}"

    def test_all_policy_seeds_consistent(self):
        seeds = [8001, 8002, 8003, 8004, 8005]
        assert _load(E4TT_DIR / "E4TT_seed_contract.json")["policy"] == seeds
        assert _load(E4TT_DIR / "E4TT_exploration_contract.json")["policy_seeds"] == seeds
        assert _load(E4TT_DIR / "E4TT_initialization_pairing.json")["policy_seeds"] == seeds
        assert _load(E4TT_DIR / "E4TT_execution_config.json")["config"]["policy_seeds"] == seeds

    def test_all_stage_fields_match(self):
        stage = "E4TT_C4_GATED_COORDINATION_TRAINING_PROTOCOL_FREEZE"
        for name in ["E4TT_arm_contract.json", "E4TT_seed_contract.json",
                      "E4TT_training_matrix.json", "E4TT_checkpoint_contract.json",
                      "E4TT_protocol_hash.json", "E4TT_verdict.json"]:
            data = _load(E4TT_DIR / name)
            assert data["stage"] == stage, f"Stage mismatch in {name}"
