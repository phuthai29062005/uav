"""
Tests for E4K — Joint Coordination Revision Design (Branch C, DESIGN/FREEZE ONLY).

Verifies historical preservation (E4F/E4HF/E4IF/E4JF), the routing-amendment audit, the
source-derived current-architecture audit, the candidate review + QMIX selection, and every
frozen Branch-C contract (region slots, utility net, central state, mixer, monotonicity,
Double-Q joint target, R1 preservation, joint reward mean, regional-anchor replay, losses,
optimizer, exploration, decentralized execution, masking, fresh seeds/firewall, pairing,
training budget, checkpoint, statistics, success skeleton, diagnostics, E4KI contract), plus
the design self-consistency synthetic validation. No training; no environment execution.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from prdrl.experiments import E4K_joint_coordination_revision_design as E4K
from prdrl.evaluation.final_verdict import (
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)

RES = Path("results/E4K_joint_coordination_revision_design")


def _load(name):
    with open(RES / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load("E4K_verdict.json")


@pytest.fixture(scope="module")
def hist():
    return _load("E4K_historical_integrity.json")


# ----------------------------------------------------------------------------
# T1-T9  historical preservation + routing + branch
# ----------------------------------------------------------------------------
def test_T1_e4jf_complete(hist):
    assert hist["E4JF_COMPLETE"] == "YES"


def test_T2_e4jf_verdict_failed(hist):
    assert hist["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"


def test_T3_e4jf_integrity_pass(hist):
    for k in ["E4JF_CHECKPOINT_INTEGRITY", "E4JF_COMPARATOR_MAP_INTEGRITY",
              "E4JF_FINAL_PROTOCOL_INTEGRITY", "E4JF_SHARED_BASELINE_HANDLING"]:
        assert hist[k] == "PASS"


def test_T4_mixed_primary_directional_positive_preserved(hist, verdict):
    assert hist["E4JF_MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"
    assert verdict["E4JF_MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"


def test_T5_persistent_safety_fail_preserved(hist, verdict):
    assert hist["E4JF_PERSISTENT_GLOBAL_SAFETY"] == "FAIL"
    assert verdict["E4JF_PERSISTENT_GLOBAL_SAFETY"] == "FAIL"


def test_T6_e4if_reward_validated_preserved(hist, verdict):
    assert hist["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
    assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T7_e4hf_normalization_failed_preserved(hist, verdict):
    assert hist["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"


def test_T8_routing_amendment_non_verdict_change(verdict):
    ra = _load("E4K_routing_amendment_audit.json")
    assert ra["POST_RESULT_ROUTING_AMENDMENT"] == "DOCUMENTED_NON_VERDICT_CHANGE"
    assert verdict["POST_RESULT_ROUTING_AMENDMENT"] == "DOCUMENTED_NON_VERDICT_CHANGE"
    assert ra["changed_only"] == "NEXT_STAGE"
    for f in ["episode_data", "delta", "CI", "classification", "safety", "verdict", "superiority_claim"]:
        assert f in ra["did_not_change"]
    assert ra["corrected_route"] == "E4K_JOINT_COORDINATION_REVISION_DESIGN"


def test_T9_branch_c_selected(verdict):
    sel = _load("E4K_architecture_selection.json")
    assert sel["SELECTED_BRANCH"] == "C_JOINT_COORDINATION"
    assert verdict["SELECTED_BRANCH"] == "C_JOINT_COORDINATION"


# ----------------------------------------------------------------------------
# T10-T15  current architecture audit + candidate review + selection
# ----------------------------------------------------------------------------
def test_T10_current_architecture_audit_complete():
    a = _load("E4K_current_architecture_audit.json")
    for k in ["per_region_state", "per_region_Q", "regional_action_selection", "replay_transition",
              "reward_per_region", "td_target_form", "gradient_update_form", "source_files"]:
        assert k in a and a[k]


def test_T11_current_joint_representation_implicit_state_only(verdict):
    a = _load("E4K_current_architecture_audit.json")
    assert a["CURRENT_JOINT_ACTION_REPRESENTATION"] == "IMPLICIT_STATE_ONLY"
    assert verdict["CURRENT_JOINT_REPRESENTATION"] == "IMPLICIT_STATE_ONLY"
    assert verdict["CURRENT_ARCHITECTURE"] == "INDEPENDENT_SHARED_DDQN"


def test_T12_candidate_review_has_all_families():
    cr = _load("E4K_candidate_architecture_review.json")
    fams = cr["families"]
    for f in ["A_CURRENT_SHARED_DDQN", "B_CTDE_CENTRALIZED_CRITIC", "C_VDN_VALUE_DECOMPOSITION",
              "D_QMIX_MONOTONIC_VALUE_DECOMPOSITION", "E_AUTOREGRESSIVE_JOINT_POLICY"]:
        assert f in fams


def test_T13_no_candidate_trained():
    cr = _load("E4K_candidate_architecture_review.json")
    assert cr["trained"] is False
    assert cr["review_type"] == "ARCHITECTURE_PROPERTY_REVIEW"


def test_T14_selected_treatment_qmix_v1(verdict):
    assert verdict["SELECTED_TREATMENT"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
    sel = _load("E4K_architecture_selection.json")
    assert sel["SELECTED_TREATMENT"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"


def test_T15_selection_architecture_properties_only():
    sel = _load("E4K_architecture_selection.json")
    assert "ARCHITECTURE_PROPERTIES_ONLY" in sel["selection_basis"]
    rc = sel["rule_conditions"]
    assert rc["preserves_decentralized_simultaneous_execution"]
    assert rc["adds_explicit_joint_value_training"]
    assert rc["does_not_require_actor_critic_replacement"]
    assert rc["supports_fixed_region_slots_masks"]
    assert rc["smallest_candidate_satisfying_all"]


# ----------------------------------------------------------------------------
# T16-T20  V3/R1/alpha/utility preserved across both future arms
# ----------------------------------------------------------------------------
def test_T16_v3_same_both_arms():
    fa = _load("E4K_future_ab_factor_contract.json")
    assert fa["CONTROL"]["normalization"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"
    assert fa["TREATMENT"]["normalization"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T17_r1_same_both_arms():
    fa = _load("E4K_future_ab_factor_contract.json")
    assert fa["CONTROL"]["reward"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"
    assert fa["TREATMENT"]["reward_primitives"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"


def test_T18_alpha_same_both_arms():
    fa = _load("E4K_future_ab_factor_contract.json")
    assert fa["CONTROL"]["alpha_local"] == 0.1
    assert fa["TREATMENT"]["alpha_local"] == 0.1


def test_T19_utility_input_dim_9():
    u = _load("E4K_utility_network_contract.json")
    assert u["input_dim"] == 9


def test_T20_utility_net_9_128_128_4():
    u = _load("E4K_utility_network_contract.json")
    assert u["UTILITY_NETWORK"] == "9-128-128-4"
    assert u["hidden"] == [128, 128] and u["output"] == 4


# ----------------------------------------------------------------------------
# T21-T27  region slots + central state
# ----------------------------------------------------------------------------
def test_T21_exact_r_max_from_canonical_regions(verdict):
    from prdrl.experiments.E3R_rule_v2_semantic_rerun import N_OBJ, N_PARTITIONS
    from prdrl.core.reference import reference
    assert len(reference(N_OBJ, N_PARTITIONS)) == 10
    assert E4K.R_MAX == 10
    assert verdict["REGION_SLOT_COUNT"] == 10


def test_T22_region_slot_order_fixed():
    rs = _load("E4K_region_slot_contract.json")
    assert rs["slot_order"] == "ASCENDING_CANONICAL_REGION_ID"
    for bad in ["occupancy", "severity", "action", "centroid", "current_quality"]:
        assert bad in rs["forbidden_orderings"]


def test_T23_central_state_formula():
    cs = _load("E4K_central_state_contract.json")
    assert E4K.CENTRAL_STATE_DIM == 5 * E4K.R_MAX + 5
    assert cs["CENTRAL_STATE_DIM"] == 5 * E4K.R_MAX + 5


def test_T24_central_dim_55_when_r10(verdict):
    assert verdict["CENTRAL_STATE_DIM"] == 55


def test_T25_active_mask_included():
    cs = _load("E4K_central_state_contract.json")
    assert "active_mask" in cs and "mandatory" in cs["active_mask"].lower()
    assert cs["why_mask_required"]


def test_T26_inactive_local_slots_zero_padded():
    cs = _load("E4K_central_state_contract.json")
    assert cs["inactive_slot_local"] == [0, 0, 0, 0]


def test_T27_no_new_scientific_state_feature():
    cs = _load("E4K_central_state_contract.json")
    for bad in ["region_id_embeddings", "previous_actions", "future_actions", "oracle_actions",
                "reward_history", "counterfactual_outcomes", "future_environment_information"]:
        assert bad in cs["no_new_state_features"]
    assert cs["only_reorganizes_current_information"] is True


# ----------------------------------------------------------------------------
# T28-T34  mixer architecture
# ----------------------------------------------------------------------------
def test_T28_mixer_hidden_dim_32(verdict):
    m = _load("E4K_mixer_architecture.json")
    assert m["MIX_HIDDEN_DIM"] == 32
    assert verdict["MIXER_HIDDEN_DIM"] == 32


def test_T29_w1_state_conditioned():
    m = _load("E4K_mixer_architecture.json")
    assert "Linear(CENTRAL_STATE_DIM" in m["layer1"]["W1"]


def test_T30_w1_nonnegative_contract():
    m = _load("E4K_mixer_architecture.json")
    assert "abs(" in m["layer1"]["W1"]


def test_T31_w2_state_conditioned():
    m = _load("E4K_mixer_architecture.json")
    assert "Linear(CENTRAL_STATE_DIM" in m["layer2"]["W2"]


def test_T32_w2_nonnegative_contract():
    m = _load("E4K_mixer_architecture.json")
    assert "abs(" in m["layer2"]["W2"]


def test_T33_elu_mixing_hidden():
    m = _load("E4K_mixer_architecture.json")
    assert "ELU" in m["layer1"]["h"]


def test_T34_scalar_q_tot():
    m = _load("E4K_mixer_architecture.json")
    assert "Q_tot" in m["layer2"] and "b2" in m["layer2"]["Q_tot"]


# ----------------------------------------------------------------------------
# T35-T37  online/target mixer + double-Q joint target
# ----------------------------------------------------------------------------
def test_T35_online_target_mixer_contract():
    tm = _load("E4K_target_mixer_contract.json")
    assert "online_mixer" in tm["networks"] and "target_mixer" in tm["networks"]
    assert "online_mixer == target_mixer" in tm["t0"]


def test_T36_target_tau_005(verdict):
    tm = _load("E4K_target_mixer_contract.json")
    assert tm["soft_update"]["tau"] == 0.005
    assert tm["soft_update"]["no_separate_mixer_tau"] is True


def test_T37_double_q_joint_target_exact(verdict):
    jt = _load("E4K_joint_double_q_target.json")
    assert jt["double_q"] is True
    assert jt["gamma"] == 0.99
    assert "argmax_a Q_u_online" in jt["steps"][1]
    assert "Q_u_target" in jt["steps"][2]
    assert "Q_tot_target" in jt["y_joint"]
    assert verdict["JOINT_TARGET"] == "DOUBLE_Q_QMIX"


# ----------------------------------------------------------------------------
# T38-T41  R1 + joint reward
# ----------------------------------------------------------------------------
def test_T38_r1_formula_unchanged():
    r = _load("E4K_r1_reward_preservation.json")
    assert r["R1_REWARD_FORMULA_CHANGED"] == "NO"
    assert r["alpha_local"] == 0.1
    assert r["R1_CONFIG_SHA256"] == E4K.FROZEN_R1_CONFIG_SHA256
    for u in ["g_r", "alpha_local", "global_reward", "FE_penalty"]:
        assert u in r["unchanged"]


def test_T39_joint_reward_active_region_mean(verdict):
    j = _load("E4K_joint_reward_aggregation.json")
    assert j["JOINT_REWARD_AGGREGATION"] == "ACTIVE_REGION_MEAN_OF_R1"
    assert verdict["JOINT_REWARD"] == "ACTIVE_REGION_MEAN_OF_R1"
    assert "(1/|A_t|)" in j["r_joint"]


def test_T40_no_sum_reward_scaling():
    j = _load("E4K_joint_reward_aggregation.json")
    assert j["forbidden"] == "sum_of_regional_rewards"
    # numeric: mean not sum, invariant to active-count duplication
    per = np.array([0.2, -0.1, 0.05, 0.4])
    assert abs(float(np.mean(per)) - float(np.sum(per))) > 1e-9
    assert abs(float(np.mean(np.concatenate([per, per]))) - float(np.mean(per))) < 1e-12


def test_T41_no_extra_reward_coefficient():
    j = _load("E4K_joint_reward_aggregation.json")
    assert j["R1_REWARD_FORMULA_CHANGED"] == "NO"
    r = _load("E4K_r1_reward_preservation.json")
    # exactly the three frozen R1 coefficients (1.0 global, 0.1 local, 0.01 FE); no new coefficient
    assert "1.0*global_reward" in r["per_region_reward"]
    assert "0.1*g_r" in r["per_region_reward"]
    assert "0.01*fe_penalty_r" in r["per_region_reward"]
    assert r["alpha_local"] == 0.1


# ----------------------------------------------------------------------------
# T42-T47  replay + sampling
# ----------------------------------------------------------------------------
def test_T42_regional_anchor_replay_exact(verdict):
    ra = _load("E4K_replay_anchor_contract.json")
    assert ra["name"] == "REGIONAL_ANCHOR_JOINT_CONTEXT_REPLAY_V1"
    assert verdict["REPLAY_SAMPLING_UNIT"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"
    for grp in ["FOCAL", "JOINT_CURRENT", "JOINT_NEXT", "DONE"]:
        assert grp in ra["logical_record"]


def test_T43_replay_capacity_50000_anchors(verdict):
    s = _load("E4K_sampling_semantics.json")
    assert s["REPLAY_CAPACITY"] == 50000 and s["capacity_unit"] == "regional_anchors"
    assert verdict["REPLAY_CAPACITY"] == 50000


def test_T44_warmup_1000_anchors():
    s = _load("E4K_sampling_semantics.json")
    assert s["WARMUP_TRANSITIONS"] == 1000 and s["warmup_unit"] == "regional_anchors"


def test_T45_update_every_4_anchors():
    s = _load("E4K_sampling_semantics.json")
    assert s["UPDATE_FREQUENCY"] == 4 and s["update_unit"] == "appended_regional_anchors"


def test_T46_batch_128_anchors(verdict):
    s = _load("E4K_sampling_semantics.json")
    assert s["BATCH_SIZE"] == 128 and s["batch_unit"] == "sampled_regional_anchors"
    assert verdict["BATCH_SIZE"] == 128


def test_T47_sibling_anchors_not_deduplicated():
    s = _load("E4K_sampling_semantics.json")
    assert "do NOT silently deduplicate" in s["sibling_anchors"]
    tl = _load("E4K_treatment_loss_contract.json")
    assert tl["no_dedup_sibling_anchors"] is True


# ----------------------------------------------------------------------------
# T48-T53  losses + optimizer + exploration
# ----------------------------------------------------------------------------
def test_T48_control_loss_exact_existing_ddqn():
    cl = _load("E4K_control_loss_contract.json")
    assert cl["identical_to_existing_R1_DDQN"] is True
    assert cl["optimizer_params"] == "utility_network_only"
    assert cl["no_dummy_mixer"] is True


def test_T49_treatment_loss_joint_smoothl1():
    tl = _load("E4K_treatment_loss_contract.json")
    assert tl["LOSS"] == "SmoothL1"
    assert "Q_tot_online" in tl["per_sampled_anchor"] and "y_joint" in tl["per_sampled_anchor"]
    assert tl["optimizer_params"] == "utility_network + mixer"


def test_T50_no_auxiliary_loss():
    tl = _load("E4K_treatment_loss_contract.json")
    assert tl["auxiliary_losses"].startswith("NONE")


def test_T51_same_gamma():
    o = _load("E4K_optimizer_contract.json")
    assert o["gamma"] == 0.99


def test_T52_same_optimizer_type_lr():
    o = _load("E4K_optimizer_contract.json")
    assert o["optimizer"] == "Adam" and o["lr"] == 1e-4
    assert o["GRAD_CLIP"] == "NONE" and o["TARGET_TAU"] == 0.005


def test_T53_same_epsilon_schedule():
    e = _load("E4K_exploration_contract.json")
    assert e["eps_start"] == 1.0 and e["eps_end"] == 0.05 and e["eps_decay_transitions"] == 50000


# ----------------------------------------------------------------------------
# T54-T60  execution + no-order + no-family-change + no oracle
# ----------------------------------------------------------------------------
def test_T54_mixer_absent_from_execution(verdict):
    ex = _load("E4K_execution_contract.json")
    assert ex["mixer_in_execution_path"] is False
    assert verdict["DECENTRALIZED_EXECUTION"] == "YES"


def test_T55_simultaneous_regional_execution():
    ex = _load("E4K_execution_contract.json")
    assert ex["actions_assembled_simultaneously_before_response"] is True
    assert ex["production_decision_time_interface_remains_regional"] is True


def test_T56_no_centralized_joint_action_search():
    e = _load("E4K_exploration_contract.json")
    assert "centralized_joint_action_search" in e["forbidden"]


def test_T57_no_autoregressive_order():
    cr = _load("E4K_candidate_architecture_review.json")
    fam = cr["families"]["E_AUTOREGRESSIVE_JOINT_POLICY"]["criteria"]
    assert fam["introduces_action_order_dependence"] is True
    sel = _load("E4K_architecture_selection.json")
    assert "AUTOREGRESSIVE" in sel["why_not_others"]


def test_T58_no_ctde_actor_critic_change():
    cr = _load("E4K_candidate_architecture_review.json")
    fam = cr["families"]["B_CTDE_CENTRALIZED_CRITIC"]["criteria"]
    assert fam["requires_switch_away_from_value_based_DDQN"] is True


def test_T59_no_vdn_selected_as_treatment():
    sel = _load("E4K_architecture_selection.json")
    assert sel["SELECTED_TREATMENT"] != "VDN"
    assert "VDN" in sel["why_not_others"]
    cr = _load("E4K_candidate_architecture_review.json")
    assert cr["families"]["C_VDN_VALUE_DECOMPOSITION"]["role"] == "reviewed"


def test_T60_no_oracle_labels():
    nfi = _load("E4K_no_future_information.json")
    assert nfi["no_D5_counterfactual_as_training"][
        "D5_counterfactual_utilities_as_targets_labels_supervision_rewards"] == "FORBIDDEN"
    assert "oracle action" in nfi["forbidden"]


# ----------------------------------------------------------------------------
# T61-T68  fresh seeds + firewall + pairing + mixer init
# ----------------------------------------------------------------------------
def test_T61_future_train_seeds():
    assert E4K.FUTURE_TRAIN_SEEDS == list(range(1901, 1961))


def test_T62_future_validation_seeds():
    assert E4K.FUTURE_VALIDATION_SEEDS == list(range(1961, 1981))


def test_T63_future_final_seeds():
    assert E4K.FUTURE_FINAL_SEEDS == list(range(1981, 2041))


def test_T64_future_policy_seeds():
    assert E4K.FUTURE_POLICY_SEEDS == [5001, 5002, 5003, 5004, 5005]


def test_T65_all_fresh_sets_disjoint(verdict):
    sc = _load("E4K_future_seed_contract.json")
    assert sc["env_seeds_disjoint_from_all_prior"] is True
    assert sc["policy_seeds_disjoint_from_all_prior"] is True
    assert verdict["ENV_SEEDS_DISJOINT"] is True and verdict["POLICY_SEEDS_DISJOINT"] is True
    # explicit numeric disjointness against every consumed env set + final firewalled sets
    fresh = set(E4K.FUTURE_TRAIN_SEEDS) | set(E4K.FUTURE_VALIDATION_SEEDS) | set(E4K.FUTURE_FINAL_SEEDS)
    consumed = set().union(*[set(v) for v in E4K.CONSUMED_SEED_SETS.values()])
    assert len(fresh & consumed) == 0


def test_T66_future_access_counts_zero(verdict):
    fw = _load("E4K_future_firewall.json")
    assert fw["C_TRAIN_ACCESS_COUNT"] == 0
    assert fw["C_VALIDATION_ACCESS_COUNT"] == 0
    assert fw["C_FINAL_ACCESS_COUNT"] == 0
    assert verdict["C_TRAIN_ACCESS_COUNT"] == 0
    assert verdict["C_VALIDATION_ACCESS_COUNT"] == 0
    assert verdict["C_FINAL_ACCESS_COUNT"] == 0


def test_T67_future_utility_init_pairing_frozen():
    p = _load("E4K_future_pairing_contract.json")
    assert "HASH(C0_utility_online_t0) == " in p["utility_init_pairing"]
    assert p["t0_online_equals_target_both_arms"] is True


def test_T68_deterministic_mixer_init_frozen():
    p = _load("E4K_future_pairing_contract.json")
    assert p["mixer_seed_derivation"] == "mixer_seed = utility_initialization_seed + 1 (deterministic)"
    assert p["no_mixer_init_tuning"] is True
    assert p["t0_online_mixer_equals_target_mixer"] is True


# ----------------------------------------------------------------------------
# T69-T75  budget + checkpoint + primary/effect/robustness/safety
# ----------------------------------------------------------------------------
def test_T69_future_training_1080_episodes_per_run():
    b = _load("E4K_future_training_budget.json")
    assert b["episodes_per_run"] == 1080


def test_T70_future_runs_10():
    b = _load("E4K_future_training_budget.json")
    assert b["runs"] == 10 and b["total_training_episodes"] == 10800


def test_T71_future_checkpoint_rule_unchanged():
    c = _load("E4K_future_checkpoint_contract.json")
    assert c["selection_criterion"] == "VALIDATION_MACRO_MIGD_RESPONSE"
    assert c["tie_breakers"] == ["macro_MIGD_epoch", "response_FE", "earlier_checkpoint"]
    for bad in ["Q_tot_loss", "mixer_sensitivity", "persistent_only_MIGD", "global_comparator_performance"]:
        assert bad in c["forbidden_selection_by"]


def test_T72_future_primary_mixed_migd_response(verdict):
    s = _load("E4K_future_statistical_contract.json")
    assert s["FUTURE_PRIMARY"] == "MIXED_MIGD_RESPONSE"
    assert verdict["FUTURE_PRIMARY"] == "MIXED_MIGD_RESPONSE"


def test_T73_future_effect_c0_minus_c1(verdict):
    s = _load("E4K_future_statistical_contract.json")
    assert s["FUTURE_EFFECT"] == "C0_MINUS_C1" and s["POSITIVE_MEANS"] == "QMIX_BETTER"
    assert verdict["FUTURE_EFFECT"] == "C0_MINUS_C1" and verdict["POSITIVE_MEANS"] == "QMIX_BETTER"


def test_T74_future_policy_robustness_4of5(verdict):
    s = _load("E4K_future_statistical_contract.json")
    assert "AT_LEAST_4_OF_5" in s["FUTURE_POLICY_ROBUSTNESS"]
    assert verdict["FUTURE_POLICY_ROBUSTNESS"] == "AT_LEAST_4_OF_5_QMIX_BETTER"


def test_T75_persistent_remains_safety_not_new_primary():
    s = _load("E4K_future_statistical_contract.json")
    assert "PERSISTENT" in s["safety"]
    assert s["primary_scenario"] == "MIXED"
    assert "no outcome-driven endpoint switching" in s["primary_not_switched_to_persistent"]


# ----------------------------------------------------------------------------
# T76-T79  diagnostics + mechanism gate + bootstrap seed + no global-in-final
# ----------------------------------------------------------------------------
def test_T76_coordination_diagnostics_descriptive_only():
    d = _load("E4K_coordination_diagnostics.json")
    assert d["status"] == "DESCRIPTIVE + ENGINEERING_INTEGRITY ONLY"
    for k in ["DIAGNOSTIC_A_monotonic_gradient", "DIAGNOSTIC_B_central_state_conditioning",
              "DIAGNOSTIC_C_regional_joint_credit_dispersion", "DIAGNOSTIC_D_other_region_sensitivity"]:
        assert k in d


def test_T77_no_new_mechanism_success_gate(verdict):
    d = _load("E4K_coordination_diagnostics.json")
    assert d["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"
    assert verdict["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"


def test_T78_bootstrap_master_seed_86001(verdict):
    s = _load("E4K_future_statistical_contract.json")
    assert s["BOOTSTRAP_MASTER_SEED"] == 86001
    assert verdict["FUTURE_BOOTSTRAP_MASTER_SEED"] == 86001
    assert s["bootstrap"] == "PAIRED_CROSSED_TWO_AXIS"


def test_T79_no_global_comparator_in_same_causal_final():
    fw = _load("E4K_future_firewall.json")
    assert fw["after_coordination_ab"]["no_global_comparator_in_same_causal_final"] is True


# ----------------------------------------------------------------------------
# T80-T83  no env exec / no training / historical unchanged / verdict complete
# ----------------------------------------------------------------------------
def test_T80_no_new_environment_execution(verdict):
    assert verdict["NEW_ENVIRONMENT_EXECUTION"] == "NO"
    fw = _load("E4K_future_firewall.json")
    assert fw["NEW_ENVIRONMENT_EXECUTION"] == "NO"


def test_T81_no_training(verdict):
    assert verdict["NEW_TRAINING"] == "NO"
    pm = _load("E4K_pre_manifest.json")
    assert pm["no_training"] is True and pm["no_new_environment_execution"] is True


def test_T82_historical_artifacts_unchanged(hist):
    # 5 R1 checkpoints re-hashed against the E4J freeze; all must still match
    assert hist["R1_CHECKPOINTS_HASH_MATCH"] is True
    for ps, row in hist["R1_CHECKPOINTS"].items():
        assert row["hash_match"] is True and row["metadata_ok"] is True
    assert hist["V3_CONSTANTS_SHA256"] == E4K.FROZEN_V3_CONSTANTS_SHA256
    assert hist["R1_CONFIG_SHA256"] == E4K.FROZEN_R1_CONFIG_SHA256
    assert hist["ALPHA_LOCAL"] == 0.1


def test_T83_verdict_complete_and_next_stage(verdict):
    assert verdict["E4K_COMPLETE"] == "YES"
    assert verdict["NEXT_STAGE"] == "E4KI_JOINT_COORDINATION_IMPLEMENTATION_VALIDATION"
    assert verdict["HISTORICAL_INTEGRITY"] == "PASS"
    assert verdict["QMIX_SELECTION_SUPPORTED"] == "YES"
    assert verdict["no_placeholder_tokens"] is True


# ----------------------------------------------------------------------------
# T84-T92  additional: preservation locks, mixer math, verdict skeleton, E4KI, factor name
# ----------------------------------------------------------------------------
def test_T84_fixed_norm_reward_alpha_utility_in_verdict(verdict):
    assert verdict["FIXED_NORMALIZATION"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"
    assert verdict["FIXED_REWARD"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"
    assert verdict["FIXED_ALPHA_LOCAL"] == 0.1
    assert verdict["UTILITY_NETWORK"] == "9-128-128-4"


def test_T85_future_factor_name(verdict):
    fa = _load("E4K_future_ab_factor_contract.json")
    assert fa["FUTURE_EXPERIMENT_FACTOR"] == "JOINT_COORDINATION_LEARNER_SPECIFICATION"
    for bad in ["NETWORK_SIZE", "REWARD", "NORMALIZATION"]:
        assert bad in fa["factor_is_not"]
    assert verdict["FUTURE_EXPERIMENT_FACTOR"] == "JOINT_COORDINATION_LEARNER_SPECIFICATION"


def test_T86_mixer_monotonic_flag(verdict):
    assert verdict["MIXER_MONOTONIC"] == "YES"
    mono = _load("E4K_monotonicity_contract.json")
    assert "dQ_tot/dq_r >= 0" in mono["contract"]
    assert "abs()" in mono["mechanism"]


def test_T87_verdict_skeleton_branches():
    f = E4K.coordination_causal_verdict_skeleton
    assert f(BENEFICIAL, True, True, True, True, True, True, True)["verdict"] == "COORDINATION_CAUSAL_VALIDATED"
    assert f(DIRECTIONAL_POSITIVE, False, True, True, True, True, True, True)["subtype"] == "PERFORMANCE_PARTIAL"
    assert f(PRACTICALLY_NEUTRAL, False, True, True, True, True, True, True)["subtype"] == "PRACTICALLY_NEUTRAL"
    assert f(BENEFICIAL, True, False, True, True, True, True, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"
    assert f(DIRECTIONAL_NEGATIVE, True, True, True, True, True, True, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"
    assert f(HARMFUL, True, True, True, True, True, True, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"
    assert f(BENEFICIAL, True, True, True, True, False, True, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"


def test_T88_e4ki_contract_covers_required_items():
    e = _load("E4K_e4ki_validation_contract.json")
    assert e["next_stage"] == "E4KI_JOINT_COORDINATION_IMPLEMENTATION_VALIDATION"
    for item in ["mixer monotonicity", "decentralized argmax equivalence", "joint reward aggregation",
                 "joint Double-Q target", "regional-anchor replay semantics", "control-loss equivalence",
                 "optimizer parameter partition", "no future seed access"]:
        assert item in e["must_validate"]
    assert len(e["synthetic_fixtures"]) >= 20


def test_T89_synthetic_design_validation_all_pass(verdict):
    s = _load("E4K_synthetic_validation.json")
    assert s["ALL_PASS"] is True
    assert s["n_pass"] == s["n_checks"]
    assert verdict["SYNTHETIC_DESIGN_VALIDATION"].startswith("PASS")


def test_T90_monotonic_mixer_reference_is_monotone_and_scalar():
    # re-run the pure-numpy reference: monotonic derivative >= 0 and scalar Q_tot
    rng = np.random.default_rng(11)
    r = np.random.default_rng(3)
    params = {"A1": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.R_MAX * E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A1b": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A2": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A2b": r.standard_normal((E4K.CENTRAL_STATE_DIM,)) * 0.1}
    for _ in range(50):
        q = rng.standard_normal(E4K.R_MAX)
        c = rng.standard_normal(E4K.CENTRAL_STATE_DIM)
        base, _, _ = E4K._monotonic_mixer_reference(q, c, params)
        assert np.isscalar(base)
        for i in range(E4K.R_MAX):
            qp = q.copy(); qp[i] += 1e-4
            up, _, _ = E4K._monotonic_mixer_reference(qp, c, params)
            assert (up - base) >= -1e-9


def test_T91_decentralized_argmax_equivalence_small_r():
    import itertools
    r = np.random.default_rng(21)
    Q_r = r.standard_normal((3, E4K.N_ACTIONS))
    c = r.standard_normal(E4K.CENTRAL_STATE_DIM)
    params = {"A1": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.R_MAX * E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A1b": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A2": r.standard_normal((E4K.CENTRAL_STATE_DIM, E4K.MIX_HIDDEN_DIM)) * 0.1,
              "A2b": r.standard_normal((E4K.CENTRAL_STATE_DIM,)) * 0.1}
    greedy = tuple(Q_r.argmax(axis=1).tolist())
    best_val, best = -np.inf, None
    for combo in itertools.product(range(E4K.N_ACTIONS), repeat=3):
        q = np.zeros(E4K.R_MAX)
        for i in range(3):
            q[i] = Q_r[i, combo[i]]
        v, _, _ = E4K._monotonic_mixer_reference(q, c, params)
        if v > best_val:
            best_val, best = v, combo
    assert greedy == best


def test_T92_no_future_information_and_current_action_vector():
    nfi = _load("E4K_no_future_information.json")
    assert "current active mask" in nfi["mixer_central_state_may_use_only"]
    assert nfi["current_action_vector"]["use"].startswith("the actually requested joint action vector")
    assert nfi["zero_active_semantics"]["no_fake_joint_transition_for_zero_reward"] is True
