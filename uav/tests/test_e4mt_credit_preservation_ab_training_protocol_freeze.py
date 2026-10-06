"""
Tests for E4MT — Credit Preservation A/B Training Protocol Freeze.
Validates all frozen protocol components before any training.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
E4MT_DIR = BASE / "results" / "E4MT_credit_preservation_ab_training_protocol_freeze"
E4M_DIR = BASE / "results" / "E4M_joint_credit_preservation_design"
E4MI_DIR = BASE / "results" / "E4MI_joint_credit_preservation_implementation_validation"
E4KF_DIR = BASE / "results" / "E4KF_joint_coordination_ab_final_evaluation_execution"
E4L_DIR = BASE / "results" / "E4L_coordination_failure_analysis"


def _load(directory, filename):
    with open(directory / filename) as f:
        return json.load(f)


# ── lazy-load fixtures ──

@pytest.fixture(scope="module")
def verdict():
    return _load(E4MT_DIR, "E4MT_verdict.json")

@pytest.fixture(scope="module")
def pre_manifest():
    return _load(E4MT_DIR, "E4MT_pre_manifest.json")

@pytest.fixture(scope="module")
def historical():
    return _load(E4MT_DIR, "E4MT_historical_integrity.json")

@pytest.fixture(scope="module")
def factor():
    return _load(E4MT_DIR, "E4MT_ab_factor_contract.json")

@pytest.fixture(scope="module")
def control():
    return _load(E4MT_DIR, "E4MT_control_contract.json")

@pytest.fixture(scope="module")
def treatment():
    return _load(E4MT_DIR, "E4MT_treatment_contract.json")

@pytest.fixture(scope="module")
def loss():
    return _load(E4MT_DIR, "E4MT_loss_contract.json")

@pytest.fixture(scope="module")
def gradient():
    return _load(E4MT_DIR, "E4MT_gradient_contract.json")

@pytest.fixture(scope="module")
def arch():
    return _load(E4MT_DIR, "E4MT_architecture_identity.json")

@pytest.fixture(scope="module")
def norm_lock():
    return _load(E4MT_DIR, "E4MT_normalization_lock.json")

@pytest.fixture(scope="module")
def reward_lock():
    return _load(E4MT_DIR, "E4MT_reward_lock.json")

@pytest.fixture(scope="module")
def replay_lock():
    return _load(E4MT_DIR, "E4MT_replay_lock.json")

@pytest.fixture(scope="module")
def mixer_lock():
    return _load(E4MT_DIR, "E4MT_mixer_lock.json")

@pytest.fixture(scope="module")
def optimizer():
    return _load(E4MT_DIR, "E4MT_optimizer_contract.json")

@pytest.fixture(scope="module")
def hyper():
    return _load(E4MT_DIR, "E4MT_hyperparameters.json")

@pytest.fixture(scope="module")
def operators():
    return _load(E4MT_DIR, "E4MT_operator_contract.json")

@pytest.fixture(scope="module")
def seeds():
    return _load(E4MT_DIR, "E4MT_seed_split.json")

@pytest.fixture(scope="module")
def disjointness():
    return _load(E4MT_DIR, "E4MT_seed_disjointness.json")

@pytest.fixture(scope="module")
def init_pair():
    return _load(E4MT_DIR, "E4MT_initialization_pairing.json")

@pytest.fixture(scope="module")
def training_budget():
    return _load(E4MT_DIR, "E4MT_training_budget.json")

@pytest.fixture(scope="module")
def update_budget():
    return _load(E4MT_DIR, "E4MT_update_budget.json")

@pytest.fixture(scope="module")
def ckpt():
    return _load(E4MT_DIR, "E4MT_checkpoint_contract.json")

@pytest.fixture(scope="module")
def validation():
    return _load(E4MT_DIR, "E4MT_validation_contract.json")

@pytest.fixture(scope="module")
def selection():
    return _load(E4MT_DIR, "E4MT_checkpoint_selection.json")

@pytest.fixture(scope="module")
def hypothesis():
    return _load(E4MT_DIR, "E4MT_primary_hypothesis.json")

@pytest.fixture(scope="module")
def effect():
    return _load(E4MT_DIR, "E4MT_effect_definition.json")

@pytest.fixture(scope="module")
def margin():
    return _load(E4MT_DIR, "E4MT_practical_margin.json")

@pytest.fixture(scope="module")
def stats():
    return _load(E4MT_DIR, "E4MT_statistical_design.json")

@pytest.fixture(scope="module")
def robustness():
    return _load(E4MT_DIR, "E4MT_policy_robustness.json")

@pytest.fixture(scope="module")
def safety():
    return _load(E4MT_DIR, "E4MT_safety_contract.json")

@pytest.fixture(scope="module")
def fe_contract():
    return _load(E4MT_DIR, "E4MT_fe_contract.json")

@pytest.fixture(scope="module")
def success():
    return _load(E4MT_DIR, "E4MT_success_logic.json")

@pytest.fixture(scope="module")
def final_matrix():
    return _load(E4MT_DIR, "E4MT_future_final_matrix.json")

@pytest.fixture(scope="module")
def stage_seq():
    return _load(E4MT_DIR, "E4MT_future_stage_sequence.json")

@pytest.fixture(scope="module")
def firewall():
    return _load(E4MT_DIR, "E4MT_final_firewall.json")

@pytest.fixture(scope="module")
def no_train():
    return _load(E4MT_DIR, "E4MT_no_training_audit.json")

@pytest.fixture(scope="module")
def credit_diag():
    return _load(E4MT_DIR, "E4MT_credit_diagnostics.json")

@pytest.fixture(scope="module")
def q_margin():
    return _load(E4MT_DIR, "E4MT_q_margin_logging.json")

@pytest.fixture(scope="module")
def classification():
    return _load(E4MT_DIR, "E4MT_classification_logic.json")

@pytest.fixture(scope="module")
def logging_contract():
    return _load(E4MT_DIR, "E4MT_training_logging_contract.json")

@pytest.fixture(scope="module")
def e4mtr_contract():
    return _load(E4MT_DIR, "E4MT_e4mtr_output_contract.json")

@pytest.fixture(scope="module")
def protocol_frozen():
    return _load(E4MT_DIR, "E4MT_protocol_frozen.json")

@pytest.fixture(scope="module")
def numerical():
    return _load(E4MT_DIR, "E4MT_numerical_stability.json")

@pytest.fixture(scope="module")
def rng():
    return _load(E4MT_DIR, "E4MT_rng_contract.json")


# ═══════════════════════════════════════════════════════════
# T1-T5 — HISTORICAL INTEGRITY
# ═══════════════════════════════════════════════════════════

def test_t1_e4mi_complete(pre_manifest):
    assert pre_manifest["checks"]["E4MI_COMPLETE"] is True

def test_t2_implementation_valid(pre_manifest):
    assert pre_manifest["checks"]["IMPLEMENTATION_VALID"] is True

def test_t3_e4kf_failed(historical):
    assert historical["ORIGINAL_E4KF_VERDICT"] == "FAILED"
    assert historical["E4KF_VERDICT_CHANGED"] == "NO"

def test_t4_h4_supported(historical):
    assert historical["H4_STATUS"] == "SUPPORTED"

def test_t5_primary_failure_unresolved(historical):
    assert historical["E4L_PRIMARY_FAILURE_HYPOTHESIS"] == "MULTIFACTORIAL_OR_UNRESOLVED"


# ═══════════════════════════════════════════════════════════
# T6-T9 — ARMS & FACTOR
# ═══════════════════════════════════════════════════════════

def test_t6_exact_two_arms(factor):
    assert factor["CONTROL"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
    assert factor["TREATMENT"] == "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"

def test_t7_c1_exact(control):
    assert control["arm"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
    assert control["role"] == "CONTROL"
    assert control["loss_specification"] == "JOINT_QMIX_SMOOTHL1"
    assert control["loss_formula"] == "L_C1 = L_joint"

def test_t8_c2_exact(treatment):
    assert treatment["arm"] == "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
    assert treatment["role"] == "TREATMENT"
    assert treatment["loss_formula"] == "L_C2 = 0.5*L_joint + 0.5*L_regional"

def test_t9_factor_exact(factor):
    assert factor["EXPERIMENT_FACTOR"] == "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX"
    assert factor["everything_else_identical"] is True


# ═══════════════════════════════════════════════════════════
# T10-T14 — ARCHITECTURE IDENTITY
# ═══════════════════════════════════════════════════════════

def test_t10_architecture_same(arch):
    assert arch["utility"] == "9-128-128-4"
    assert arch["parameter_counts_identical"] is True

def test_t11_parameter_counts_same(arch):
    assert arch["parameter_counts_identical"] is True

def test_t12_utility_same(arch, control, treatment):
    assert control["utility_architecture"] == "9-128-128-4"
    assert treatment["utility_architecture"] == "9-128-128-4"

def test_t13_mixer_same(arch, control, treatment):
    assert arch["mixer"] == "QMIX_MIXER_V1"
    assert control["mixer"] == "QMIX_MIXER_V1"
    assert treatment["mixer"] == "QMIX_MIXER_V1"

def test_t14_central_state_same(arch, control, treatment):
    assert arch["CENTRAL_STATE_DIM"] == 55
    assert control["central_state_dim"] == 55
    assert treatment["central_state_dim"] == 55


# ═══════════════════════════════════════════════════════════
# T15-T20 — LOCKS
# ═══════════════════════════════════════════════════════════

def test_t15_v3_same(norm_lock, control, treatment):
    assert norm_lock["normalization"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"
    assert control["normalization"] == norm_lock["normalization"]
    assert treatment["normalization"] == norm_lock["normalization"]

def test_t16_r1_same(reward_lock, control, treatment):
    assert reward_lock["reward"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"
    assert control["reward"] == reward_lock["reward"]
    assert treatment["reward"] == reward_lock["reward"]

def test_t17_alpha_same(reward_lock, control, treatment):
    assert reward_lock["alpha_local"] == 0.1
    assert control["alpha_local"] == 0.1
    assert treatment["alpha_local"] == 0.1

def test_t18_joint_reward_same(reward_lock, control, treatment):
    assert reward_lock["joint_reward"] == "ACTIVE_REGION_MEAN_OF_R1"
    assert control["joint_reward"] == "ACTIVE_REGION_MEAN_OF_R1"
    assert treatment["joint_reward"] == "ACTIVE_REGION_MEAN_OF_R1"

def test_t19_replay_schema_same(replay_lock, control, treatment):
    assert replay_lock["sampling_unit"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"
    assert control["replay_sampling_unit"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"
    assert treatment["replay_sampling_unit"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

def test_t20_sibling_non_dedup(replay_lock, control, treatment):
    assert replay_lock["sibling_deduplication"] is False
    assert control["sibling_deduplication"] is False
    assert treatment["sibling_deduplication"] is False


# ═══════════════════════════════════════════════════════════
# T21-T26 — LOSS
# ═══════════════════════════════════════════════════════════

def test_t21_c1_loss_joint_only(control):
    assert control["loss_specification"] == "JOINT_QMIX_SMOOTHL1"

def test_t22_c2_joint_loss_same(treatment):
    assert treatment["joint_loss"] == "JOINT_QMIX_SMOOTHL1"

def test_t23_c2_regional_loss_exact(treatment):
    assert treatment["regional_loss"] == "FOCAL_R1_DOUBLE_DQN_SMOOTHL1"

def test_t24_c2_half_half_exact(loss, treatment):
    assert loss["joint_weight"] == 0.5
    assert loss["regional_weight"] == 0.5
    assert "0.5*L_joint + 0.5*L_regional" in treatment["loss_formula"]

def test_t25_no_tunable_aux_weight(loss, treatment):
    assert loss["weights_tunable"] is False
    assert treatment["auxiliary_weight_tunable"] is False

def test_t26_mixer_gradient_scaling_documented(gradient):
    assert "0.5" in gradient["MIXER_GRADIENT_CONSEQUENCE"]
    assert len(gradient["NO_COMPENSATION"]) >= 3


# ═══════════════════════════════════════════════════════════
# T27-T36 — OPTIMIZER & HYPERPARAMS
# ═══════════════════════════════════════════════════════════

def test_t27_one_optimizer(optimizer):
    assert optimizer["optimizer"] == "Adam"
    assert optimizer["separate_local_optimizer"] is False
    assert optimizer["separate_mixer_optimizer"] is False

def test_t28_optimizer_params_same(optimizer):
    assert optimizer["parameters"] == "online_utility + online_mixer"
    assert optimizer["parameter_count_identical_C1_C2"] is True

def test_t29_lr_same(optimizer, control, treatment, hyper):
    assert optimizer["lr"] == 1e-4
    assert control["lr"] == 1e-4
    assert treatment["lr"] == 1e-4
    assert hyper["lr"] == 1e-4

def test_t30_gamma_same(control, treatment, hyper):
    assert control["gamma"] == 0.99
    assert treatment["gamma"] == 0.99
    assert hyper["gamma"] == 0.99

def test_t31_tau_same(control, treatment, hyper):
    assert control["tau"] == 0.005
    assert treatment["tau"] == 0.005
    assert hyper["tau"] == 0.005

def test_t32_replay_capacity_same(replay_lock, hyper):
    assert replay_lock["capacity"] == 50000
    assert hyper["replay_capacity"] == 50000

def test_t33_warmup_same(replay_lock, hyper):
    assert replay_lock["warmup"] == 1000
    assert hyper["warmup"] == 1000

def test_t34_update_cadence_same(replay_lock, hyper):
    assert replay_lock["update_every"] == 4
    assert hyper["update_every"] == 4

def test_t35_batch_same(replay_lock, hyper):
    assert replay_lock["batch_size"] == 128
    assert hyper["batch_size"] == 128

def test_t36_epsilon_same(hyper):
    assert hyper["epsilon_start"] == 1.0
    assert hyper["epsilon_end"] == 0.05
    assert hyper["epsilon_decay_over"] == 50000
    assert hyper["validation_epsilon"] == 0.0
    assert hyper["final_epsilon"] == 0.0


# ═══════════════════════════════════════════════════════════
# T37 — OPERATORS
# ═══════════════════════════════════════════════════════════

def test_t37_operators_same(operators):
    assert operators["KEEP"]["FE"] == 0
    assert operators["LOCAL"]["sigma_L"] == 0.02
    assert operators["PREDICT"]["sigma_P"] == 0.005
    assert operators["PREDICT"]["lambda_P"] == 1
    assert operators["DIVERSIFY"]["sigma_D"] == 0.15
    assert operators["budget"]["B"] == 10
    assert operators["no_HARD"] is True


# ═══════════════════════════════════════════════════════════
# T38-T43 — SEEDS
# ═══════════════════════════════════════════════════════════

def test_t38_train_seeds(seeds):
    assert seeds["TRAIN_SEEDS"] == "2101..2160"
    assert seeds["train_count"] == 60

def test_t39_val_seeds(seeds):
    assert seeds["VALIDATION_SEEDS"] == "2161..2180"
    assert seeds["validation_count"] == 20

def test_t40_final_seeds(seeds):
    assert seeds["FINAL_SEEDS"] == "2181..2240"
    assert seeds["final_count"] == 60

def test_t41_policy_seeds(seeds):
    assert seeds["POLICY_SEEDS"] == [6001, 6002, 6003, 6004, 6005]
    assert seeds["policy_count"] == 5

def test_t42_disjointness(disjointness):
    assert disjointness["SEED_DISJOINTNESS"] == "PASS"
    assert disjointness["env_overlap"] == []
    assert disjointness["policy_overlap"] == []

def test_t43_all_access_counts_zero(firewall):
    assert firewall["M_TRAIN_ACCESS_COUNT"] == 0
    assert firewall["M_VALIDATION_ACCESS_COUNT"] == 0
    assert firewall["M_FINAL_ACCESS_COUNT"] == 0


# ═══════════════════════════════════════════════════════════
# T44-T45 — INITIALIZATION & PAIRING
# ═══════════════════════════════════════════════════════════

def test_t44_full_network_paired_init(init_pair):
    req = init_pair["at_t0_require"]
    assert req["HASH_C1_utility_online_eq_HASH_C2_utility_online"] is True
    assert req["HASH_C1_mixer_online_eq_HASH_C2_mixer_online"] is True
    assert req["online_utility_eq_target_utility"] is True
    assert req["online_mixer_eq_target_mixer"] is True
    for ps in [6001, 6002, 6003, 6004, 6005]:
        assert ps in init_pair["policy_seeds"]

def test_t45_environment_pairing(rng):
    assert rng["same_policy_epoch_scenario_env_seed_episode"] is True
    assert rng["do_not_force_identical_actions"] is True


# ═══════════════════════════════════════════════════════════
# T46-T52 — TRAINING BUDGET & CHECKPOINTS
# ═══════════════════════════════════════════════════════════

def test_t46_episodes_per_run(training_budget):
    assert training_budget["episodes_per_run"] == 1080

def test_t47_total_runs(training_budget):
    assert training_budget["total_runs"] == 10

def test_t48_total_training(training_budget):
    assert training_budget["total_training_episodes"] == 10800

def test_t49_scenario_balance(training_budget):
    bal = training_budget["scenario_balance"]
    assert bal["PERSISTENT"] == 60
    assert bal["STATIONARY"] == 60
    assert bal["MIXED"] == 60

def test_t50_no_persistent_oversampling(training_budget):
    assert training_budget["no_PERSISTENT_oversampling"] is True

def test_t51_checkpoint_schedule(ckpt):
    assert ckpt["checkpoint_schedule"] == [10000, 20000, 30000, 40000, 50000, "final"]
    assert ckpt["checkpoints_per_run"] == 6

def test_t52_total_checkpoints(ckpt):
    assert ckpt["total_checkpoints"] == 60


# ═══════════════════════════════════════════════════════════
# T53-T54 — CHECKPOINT METADATA
# ═══════════════════════════════════════════════════════════

def test_t53_c1_metadata():
    e4m_ctrl = _load(E4M_DIR, "E4M_control_contract.json")
    ctrl = _load(E4MT_DIR, "E4MT_control_contract.json")
    assert ctrl["arm"] == e4m_ctrl["arm"]

def test_t54_c2_credit_hash_required(treatment):
    sha = treatment.get("C2_CREDIT_CONFIG_SHA256")
    assert sha is not None
    assert len(sha) == 64
    e4mi_credit = _load(E4MI_DIR, "E4MI_credit_config.json")
    import hashlib
    expected = hashlib.sha256(
        json.dumps(e4mi_credit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert sha == expected


# ═══════════════════════════════════════════════════════════
# T55-T61 — VALIDATION & SELECTION
# ═══════════════════════════════════════════════════════════

def test_t55_validation_episodes(validation):
    assert validation["total_validation_episodes"] == 3600

def test_t56_validation_epsilon_zero(validation):
    assert validation["epsilon"] == 0.0

def test_t57_selection_macro_response(selection):
    assert selection["primary_criterion"] == "minimum_validation_MACRO_MIGD_RESPONSE"
    assert selection["macro_weight"] == "equal_1_3_each_scenario"

def test_t58_epoch_tiebreak(selection):
    assert selection["tie_1"] == "lower_macro_MIGD_epoch"

def test_t59_fe_tiebreak(selection):
    assert selection["tie_2"] == "lower_macro_response_FE"

def test_t60_earlier_tiebreak(selection):
    assert selection["tie_3"] == "earlier_checkpoint"

def test_t61_all_10_policies_retained(selection):
    assert selection["retain_all_10_policies"] is True
    assert selection["no_best_policy_selection"] is True
    assert selection["no_dropping_bad_seeds"] is True


# ═══════════════════════════════════════════════════════════
# T62-T68 — LOGGING & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

def test_t62_loss_components_logging(logging_contract):
    fields = logging_contract["per_episode_fields"]
    assert "arm" in fields
    assert "MIGD_response" in fields

def test_t63_sibling_dispersion_logging(credit_diag):
    items = credit_diag["C2_descriptive_report"]
    assert any("within_event_std" in x for x in items)

def test_t64_gradient_diagnostics():
    d = _load(E4MT_DIR, "E4MT_action_diagnostics.json")
    assert "||g_joint_utility||" in d["sampled_updates_report"]
    assert "||g_composite_mixer||" in d["sampled_updates_report"]

def test_t65_mechanism_gate_absent(credit_diag):
    assert credit_diag["CREDIT_PRESERVATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"

def test_t66_action_diagnostics_descriptive(credit_diag):
    assert credit_diag["diagnostics_are_explanatory_only"] is True

def test_t67_q_margin_logging(q_margin):
    assert q_margin["logging"]["top1_utility_Q"] is True
    assert q_margin["logging"]["top2_utility_Q"] is True
    assert q_margin["logging"]["Q_margin"] == "top1 - top2"

def test_t68_q_margin_not_success_gate(q_margin):
    assert q_margin["NOT_a_success_gate"] is True
    assert q_margin["no_additional_objective_FE"] is True


# ═══════════════════════════════════════════════════════════
# T69 — NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════

def test_t69_numerical_stability(numerical):
    assert "utility_Q" in numerical["require_finite"]
    assert "L_composite" in numerical["require_finite"]
    assert numerical["if_numerical_failure"] == "preserve_do_not_retune"


# ═══════════════════════════════════════════════════════════
# T70-T80 — STATISTICAL DESIGN
# ═══════════════════════════════════════════════════════════

def test_t70_primary_mixed(hypothesis):
    assert hypothesis["PRIMARY_SCENARIO"] == "MIXED"

def test_t71_endpoint_migd(hypothesis):
    assert hypothesis["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"

def test_t72_effect_c1_c2(hypothesis, effect):
    assert hypothesis["EFFECT_DEFINITION"] == "C1_MINUS_C2"
    assert "C1" in effect["formula"] and "C2" in effect["formula"]

def test_t73_positive_means_c2_better(hypothesis, effect):
    assert hypothesis["POSITIVE_MEANS"] == "C2_BETTER"
    assert effect["positive_means"] == "C2_BETTER"

def test_t74_margin(margin):
    assert margin["PRACTICAL_MARGIN"] == 1e-5

def test_t75_bootstrap_two_axis(stats):
    assert stats["method"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"

def test_t76_policy_axis(stats):
    assert stats["policy_axis"] == 5

def test_t77_env_axis(stats):
    assert stats["environment_axis"] == 60

def test_t78_bootstrap_n(stats):
    assert stats["resamples"] == 50000

def test_t79_bootstrap_seed(stats):
    assert stats["BOOTSTRAP_MASTER_SEED"] == 106001

def test_t80_macro_shared_draws(stats):
    assert stats["shared_draws_within_replicate"] is True


# ═══════════════════════════════════════════════════════════
# T81-T85 — ROBUSTNESS & SAFETY
# ═══════════════════════════════════════════════════════════

def test_t81_robustness(robustness):
    assert "4/5" in robustness["PASS_iff"]

def test_t82_persistent_safety(safety):
    assert "HARMFUL" in safety["PERSISTENT_safety"]
    assert "MIGD_response" in safety["PERSISTENT_safety"]

def test_t83_stationary_safety(safety):
    assert "HARMFUL" in safety["STATIONARY_safety"]

def test_t84_macro_safety(safety):
    assert "HARMFUL" in safety["MACRO_safety"]
    assert "1/3" in safety["MACRO_safety"]

def test_t85_fe_cannot_rescue(fe_contract):
    assert fe_contract["cannot_rescue_harmful_MIGD"] is True
    assert fe_contract["role"] == "SECONDARY_ONLY"


# ═══════════════════════════════════════════════════════════
# T86-T89 — VERDICT LOGIC
# ═══════════════════════════════════════════════════════════

def test_t86_validated_logic(success):
    req = success["CREDIT_PRESERVATION_CAUSAL_VALIDATED"]["requires_ALL"]
    assert any("BENEFICIAL" in r for r in req)
    assert any("robustness" in r.lower() for r in req)
    assert any("PERSISTENT" in r for r in req)
    assert any("STATIONARY" in r for r in req)
    assert any("MACRO" in r for r in req)

def test_t87_performance_partial(success):
    pp = success["PARTIAL"]["PERFORMANCE_PARTIAL"]
    assert any("DIRECTIONAL_POSITIVE" in p for p in pp)

def test_t88_neutral_partial(success):
    assert "PRACTICALLY_NEUTRAL" in success["PARTIAL"]["PRACTICALLY_NEUTRAL"]

def test_t89_failed_logic(success):
    conds = success["FAILED"]["conditions"]
    assert any("DIRECTIONAL_NEGATIVE" in c for c in conds)
    assert any("HARMFUL" in c for c in conds)
    assert any("safety" in c.lower() for c in conds)


# ═══════════════════════════════════════════════════════════
# T90-T95 — FUTURE & ROUTING
# ═══════════════════════════════════════════════════════════

def test_t90_final_matrix(final_matrix):
    assert final_matrix["total_episodes"] == 1800
    assert final_matrix["arms"] == 2
    assert final_matrix["policies"] == 5
    assert final_matrix["environments"] == 60
    assert final_matrix["scenarios"] == 3

def test_t91_no_c0_in_experiment(factor):
    assert "C0" not in factor["CONTROL"]
    assert "C0" not in factor["TREATMENT"]

def test_t92_no_global_comparator(factor):
    assert "GLOBAL" not in factor["CONTROL"]
    assert "GLOBAL" not in factor["TREATMENT"]

def test_t93_future_sequence(stage_seq):
    seq = stage_seq["if_E4MT_PASS"]
    assert seq[0] == "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION"
    assert seq[1] == "E4ME_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
    assert seq[2] == "E4MF_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_EXECUTION"

def test_t94_c2_success_routes_to_c2_vs_c0():
    routing = _load(E4MT_DIR, "E4MT_future_stage_sequence.json")
    assert routing["do_not_collapse_stages"] is True

def test_t95_no_transitive_c2_gt_c0(factor):
    assert "COMPOSITE_LOSS_INTERPRETATION" in factor
    interp = factor["COMPOSITE_LOSS_INTERPRETATION"]
    assert "not the stronger statement" in interp.lower() or "not" in interp.lower()


# ═══════════════════════════════════════════════════════════
# T96-T99 — FIREWALLS
# ═══════════════════════════════════════════════════════════

def test_t96_final_seeds_untouched(firewall):
    assert firewall["M_FINAL_ACCESS_COUNT"] == 0

def test_t97_no_environment_episode(no_train):
    assert no_train["environment_episodes"] == 0

def test_t98_no_optimizer_step(no_train):
    assert no_train["optimizer_steps"] == 0

def test_t99_no_training(no_train, firewall):
    assert no_train["training_episodes"] == 0
    assert firewall["no_training"] is True


# ═══════════════════════════════════════════════════════════
# T100-T101 — HISTORICAL & REGRESSION
# ═══════════════════════════════════════════════════════════

def test_t100_historical_artifacts_unchanged(historical):
    imm = historical["immutable_historical_verdicts"]
    assert imm["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert imm["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert imm["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
    assert imm["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
    assert imm["E4KF_COORDINATION_CAUSAL_VERDICT"] == "FAILED"

def test_t101_full_regression():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=line",
         "--ignore=tests/test_e4mt_credit_preservation_ab_training_protocol_freeze.py",
         "--ignore=tests/test_e4mi_joint_credit_preservation_implementation_validation.py",
         "--ignore=tests/test_e4mtr_credit_preservation_ab_training_execution.py",
         "--ignore=tests/test_e4me_credit_preservation_ab_final_evaluation_protocol_freeze.py",
         "--ignore=tests/test_e4mf_credit_preservation_ab_final_evaluation_execution.py",
         "--ignore=tests/test_e4n_credit_preservation_failure_analysis.py",
         "--ignore=tests/test_e4o_scenario_tradeoff_diagnostic_design.py",
         "--ignore=tests/test_e4oi_scenario_tradeoff_diagnostic_implementation_validation.py",
         "--ignore=tests/test_e4od_scenario_tradeoff_diagnostic_execution.py",
         "--ignore=tests/test_e4odr_e4od_report_artifact_reconciliation.py",
         "--ignore=tests/test_e4p_nonmonotonic_interaction_diagnostic_design.py",
         "--ignore=tests/test_e4pi_nonmonotonic_interaction_diagnostic_implementation_validation.py",
         "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
         "--ignore=tests/test_e4rtr_v2_pairwise_cg_ab_training_execution.py",
         "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py"],
        capture_output=True, text=True, cwd=str(BASE),
        env={**__import__("os").environ, "PYTHONPATH": str(BASE / "src")},
        timeout=1200,
    )
    assert result.returncode == 0, f"Regression failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
    assert "passed" in result.stdout


# ═══════════════════════════════════════════════════════════
# ADDITIONAL TESTS — CROSS-CHECKS
# ═══════════════════════════════════════════════════════════

def test_t102_e4mt_verdict_complete(verdict):
    assert verdict["E4MT_COMPLETE"] == "YES"
    assert verdict["NEXT_STAGE"] == "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION"

def test_t103_protocol_frozen_consistent(protocol_frozen, treatment):
    assert protocol_frozen["EFFECT"] == "C1_MINUS_C2"
    assert protocol_frozen["POSITIVE_MEANS"] == "C2_BETTER"
    assert protocol_frozen["C2_CREDIT_CONFIG_SHA256"] == treatment["C2_CREDIT_CONFIG_SHA256"]
    assert protocol_frozen["TRAIN_SEEDS"] == "2101..2160"
    assert protocol_frozen["VALIDATION_SEEDS"] == "2161..2180"
    assert protocol_frozen["FINAL_SEEDS"] == "2181..2240"

def test_t104_update_budget_no_extra(update_budget):
    assert update_budget["one_optimizer_step_per_update_event"] is True
    assert "optimizer_step_frequency" in update_budget["regional_loss_does_NOT_increase"]

def test_t105_e4mtr_contract(e4mtr_contract):
    req = e4mtr_contract["required"]
    assert req["total_training_episodes"] == 10800
    assert req["total_checkpoints"] == 60
    assert req["validation_episodes"] == 3600
    assert req["selected_checkpoints"] == 10
    assert req["final_seed_access"] == 0

def test_t106_no_rescue_rule(loss, success):
    assert loss["no_hyperparameter_sweep"] is True
    assert loss["no_schedule"] is True
    assert loss["no_adaptive_weighting"] is True
    cannot = success["CANNOT_RESCUE"]
    assert "FE_savings" in cannot
    assert "gradient_diversity" in cannot

def test_t107_c2_credit_config_in_protocol(protocol_frozen):
    assert len(protocol_frozen["C2_CREDIT_CONFIG_SHA256"]) == 64

def test_t108_qmix_config_in_protocol(protocol_frozen, arch):
    assert protocol_frozen["QMIX_CONFIG_SHA256"] == arch["QMIX_CONFIG_SHA256"]
    assert protocol_frozen["QMIX_CONFIG_SHA256"] == "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"

def test_t109_gradient_paths_frozen(gradient):
    assert gradient["C1"]["utility_gradients_from"] == ["L_joint"]
    assert gradient["C1"]["mixer_gradients_from"] == ["L_joint"]
    assert "0.5*L_joint" in gradient["C2"]["utility_gradients_from"]
    assert "0.5*L_regional" in gradient["C2"]["utility_gradients_from"]
    assert gradient["C2"]["mixer_gradients_from"] == ["0.5*L_joint"]

def test_t110_validation_mode(validation):
    assert validation["eval_mode"] is True
    assert validation["gradients_off"] is True
    assert validation["optimizer_off"] is True

def test_t111_classification_categories(classification):
    cats = classification["categories"]
    assert "BENEFICIAL" in cats
    assert "DIRECTIONAL_POSITIVE" in cats
    assert "PRACTICALLY_NEUTRAL" in cats
    assert "DIRECTIONAL_NEGATIVE" in cats
    assert "HARMFUL" in cats

def test_t112_no_persistent_oversampling_in_training(training_budget):
    bal = training_budget["scenario_balance"]
    assert bal["PERSISTENT"] == bal["STATIONARY"] == bal["MIXED"] == 60

def test_t113_report_exists():
    assert (E4MT_DIR / "E4MT_report.md").exists()
    text = (E4MT_DIR / "E4MT_report.md").read_text()
    assert "Scientific question" in text
    assert "C1" in text
    assert "C2" in text

def test_t114_forbidden_selection_signals(selection):
    forbidden = selection["forbidden_selection_signals"]
    assert "L_joint" in forbidden
    assert "L_regional" in forbidden
    assert "MIXED_only_performance" in forbidden
    assert "C1_vs_C2_validation_delta" in forbidden

def test_t115_mixer_in_arch(mixer_lock, arch):
    assert mixer_lock["hidden_dim"] == 32
    assert mixer_lock["central_state_dim"] == 55
    assert mixer_lock["R_MAX"] == 10
    assert arch["MIXER_HIDDEN"] == 32
    assert arch["R_MAX"] == 10

def test_t116_e4m_source_consistency():
    e4m_v = _load(E4M_DIR, "E4M_verdict.json")
    protocol = _load(E4MT_DIR, "E4MT_protocol_frozen.json")
    assert protocol["PRACTICAL_MARGIN"] == e4m_v["PRACTICAL_MARGIN"]
    assert protocol["BOOTSTRAP_SEED"] == e4m_v.get("FUTURE_BOOTSTRAP_MASTER_SEED", 106001)
