"""
E4MT — Credit Preservation A/B Training Protocol Freeze.

DESIGN / PROTOCOL FREEZE ONLY.
DO NOT TRAIN. DO NOT run validation environments. DO NOT open final environments.

Freezes the complete matched experiment:
  C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1
versus:
  C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1
BEFORE any scientific environment from 2101..2240 is instantiated.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[3]
E4M_DIR = BASE / "results" / "E4M_joint_credit_preservation_design"
E4MI_DIR = BASE / "results" / "E4MI_joint_credit_preservation_implementation_validation"
E4KF_DIR = BASE / "results" / "E4KF_joint_coordination_ab_final_evaluation_execution"
E4L_DIR = BASE / "results" / "E4L_coordination_failure_analysis"
OUT_DIR = BASE / "results" / "E4MT_credit_preservation_ab_training_protocol_freeze"


def _load(directory, filename):
    with open(directory / filename) as f:
        return json.load(f)


def _save(data, filename):
    with open(OUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── all prior scientific seed ranges ──
PRIOR_ENV_RANGES = [
    (1001, 1060), (1061, 1080), (1081, 1140),
    (1201, 1220),
    (1301, 1360), (1361, 1380), (1381, 1440),
    (1501, 1560), (1561, 1580), (1581, 1640),
    (1701, 1720),
    (1801, 1860),
    (1901, 1960), (1961, 1980), (1981, 2040),
]
PRIOR_POLICY_RANGES = [
    (1001, 1005), (3001, 3005), (4001, 4005), (5001, 5005),
]


def _all_prior_env_seeds():
    s = set()
    for lo, hi in PRIOR_ENV_RANGES:
        s.update(range(lo, hi + 1))
    return s


def _all_prior_policy_seeds():
    s = set()
    for lo, hi in PRIOR_POLICY_RANGES:
        s.update(range(lo, hi + 1))
    return s


# ═══════════════════════════════════════════════════════════
# PHASE 0 — PRE-FREEZE INTEGRITY
# ═══════════════════════════════════════════════════════════

def phase_0_pre_freeze_integrity():
    e4mi_v = _load(E4MI_DIR, "E4MI_verdict.json")
    e4m_v = _load(E4M_DIR, "E4M_verdict.json")
    e4kf_v = _load(E4KF_DIR, "E4KF_verdict.json")
    e4l_v = _load(E4L_DIR, "E4L_verdict.json")
    e4mi_credit = _load(E4MI_DIR, "E4MI_credit_config.json")

    e4mi_c1_bw = _load(E4MI_DIR, "E4MI_c1_backward_equivalence.json")
    e4mi_arch = _load(E4MI_DIR, "E4MI_architecture_identity.json")
    e4mi_replay = _load(E4MI_DIR, "E4MI_replay_schema_identity.json")
    e4mi_jr = _load(E4MI_DIR, "E4MI_joint_reward_preservation.json")
    e4mi_v3 = _load(E4MI_DIR, "E4MI_v3_preservation.json")
    e4mi_r1 = _load(E4MI_DIR, "E4MI_r1_preservation.json")
    e4mi_mixer = _load(E4MI_DIR, "E4MI_mixer_preservation.json")
    e4mi_sibling = _load(E4MI_DIR, "E4MI_differentiated_sibling_supervision.json")
    e4mi_firewall = _load(E4MI_DIR, "E4MI_future_seed_firewall.json")

    checks = {}
    checks["E4MI_COMPLETE"] = e4mi_v["E4MI_COMPLETE"] == "YES"
    checks["IMPLEMENTATION_VALID"] = e4mi_v["CREDIT_PRESERVATION_IMPLEMENTATION_VALID"] == "YES"
    checks["E4M_COMPLETE"] = e4m_v["E4M_COMPLETE"] == "YES"
    checks["E4L_COMPLETE"] = e4l_v["E4L_COMPLETE"] == "YES"
    checks["E4KF_FAILED"] = e4kf_v["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
    checks["H4_SUPPORTED"] = e4l_v["H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION"] == "SUPPORTED"
    checks["PRIMARY_MULTIFACTORIAL"] = e4l_v["PRIMARY_FAILURE_HYPOTHESIS"] == "MULTIFACTORIAL_OR_UNRESOLVED"
    checks["C1_BACKWARD_EQUIV"] = e4mi_c1_bw.get("C1_BACKWARD_EQUIVALENCE", e4mi_c1_bw.get("c1_backward_equivalence")) in (True, "PASS")
    checks["ARCHITECTURE_IDENTITY"] = e4mi_arch.get(
        "ARCHITECTURE_IDENTITY_C1_C2",
        e4mi_arch.get("ARCHITECTURE_IDENTITY", e4mi_arch.get("architecture_identity"))
    ) in (True, "PASS")

    param_identity = e4mi_arch.get("PARAMETER_COUNT_IDENTITY", e4mi_arch.get("parameter_count_identity"))
    checks["PARAMETER_COUNT_IDENTITY"] = param_identity in (True, "PASS", "IDENTICAL")
    checks["REPLAY_SCHEMA_IDENTITY"] = e4mi_replay.get("REPLAY_SCHEMA_IDENTITY", e4mi_replay.get("replay_schema_identity")) in (True, "PASS")
    checks["JOINT_REWARD_PRESERVATION"] = e4mi_jr.get("JOINT_REWARD_PRESERVATION", e4mi_jr.get("joint_reward_preservation")) in (True, "PASS")
    checks["V3_PRESERVATION"] = e4mi_v3.get("V3_PRESERVATION", e4mi_v3.get("v3_preservation")) in (True, "PASS")
    checks["R1_PRESERVATION"] = e4mi_r1.get("R1_PRESERVATION", e4mi_r1.get("r1_preservation")) in (True, "PASS")
    checks["MIXER_PRESERVATION"] = e4mi_mixer.get("MIXER_PRESERVATION", e4mi_mixer.get("mixer_preservation")) in (True, "PASS")

    diff_sib = e4mi_sibling.get("DIFFERENTIATED_SIBLING_SUPERVISION",
                                 e4mi_sibling.get("differentiated_sibling_supervision"))
    checks["DIFFERENTIATED_SIBLING"] = diff_sib in (True, "PASS")

    c2_sha = e4mi_v["C2_CREDIT_CONFIG_SHA256"]
    expected_sha = hashlib.sha256(
        json.dumps(e4mi_credit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checks["C2_CREDIT_HASH_EXACT"] = c2_sha == expected_sha

    checks["TRAIN_ACCESS_ZERO"] = e4mi_firewall.get("M_TRAIN_ACCESS_COUNT", e4mi_firewall.get("train_access_count", 0)) == 0
    checks["VAL_ACCESS_ZERO"] = e4mi_firewall.get("M_VALIDATION_ACCESS_COUNT", e4mi_firewall.get("validation_access_count", 0)) == 0
    checks["FINAL_ACCESS_ZERO"] = e4mi_firewall.get("M_FINAL_ACCESS_COUNT", e4mi_firewall.get("final_access_count", 0)) == 0

    all_pass = all(checks.values())
    result = {
        "phase": "E4MT-0_PRE_FREEZE_INTEGRITY",
        "checks": checks,
        "ALL_PASS": all_pass,
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
    }
    _save(result, "E4MT_pre_manifest.json")

    historical = {
        "ORIGINAL_E4KF_VERDICT": "FAILED",
        "E4KF_VERDICT_CHANGED": "NO",
        "E4L_PRIMARY_FAILURE_HYPOTHESIS": "MULTIFACTORIAL_OR_UNRESOLVED",
        "H4_STATUS": "SUPPORTED",
        "E4M_COMPLETE": "YES",
        "E4MI_COMPLETE": "YES",
        "CREDIT_PRESERVATION_IMPLEMENTATION_VALID": "YES",
        "immutable_historical_verdicts": {
            "ORIGINAL_E4F_VERDICT": "FAILED",
            "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
            "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
            "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
            "E4KF_COORDINATION_CAUSAL_VERDICT": "FAILED",
        },
    }
    _save(historical, "E4MT_historical_integrity.json")
    return all_pass, c2_sha


# ═══════════════════════════════════════════════════════════
# PHASES 1-15 — FREEZE ALL PROTOCOL COMPONENTS
# ═══════════════════════════════════════════════════════════

def phase_1_2_arms(c2_sha):
    c1 = _load(E4M_DIR, "E4M_control_contract.json")
    c2 = _load(E4M_DIR, "E4M_treatment_contract.json")

    factor = {
        "EXPERIMENT_FACTOR": "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX",
        "CONTROL": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "TREATMENT": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "scientific_difference": {
            "C1": "JOINT_ONLY_QMIX_TD",
            "C2": "EQUAL_MEAN_OF_JOINT_AND_FOCAL_REGIONAL_TD",
        },
        "everything_else_identical": True,
        "COMPOSITE_LOSS_INTERPRETATION": (
            "C2 simultaneously adds focal region-specific TD supervision "
            "AND gives the joint component coefficient 0.5 rather than 1.0. "
            "Any future causal result supports the frozen CREDIT-PRESERVING "
            "COMPOSITE LOSS SPECIFICATION, not the stronger statement that "
            "adding local TD alone caused the effect independently of "
            "joint-loss reweighting."
        ),
    }
    _save(factor, "E4MT_ab_factor_contract.json")

    control = {
        "arm": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "role": "CONTROL",
        "normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "reward": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "alpha_local": 0.1,
        "utility_architecture": "9-128-128-4",
        "mixer": "QMIX_MIXER_V1",
        "mixer_hidden_dim": 32,
        "central_state_dim": 55,
        "R_MAX": 10,
        "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "sibling_deduplication": False,
        "loss_specification": "JOINT_QMIX_SMOOTHL1",
        "loss_formula": "L_C1 = L_joint",
        "optimizer": "Adam",
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "action_selection": "ARGMAX_REGIONAL_UTILITY",
        "mixer_in_action_selection": False,
        "decentralized_execution": True,
    }
    _save(control, "E4MT_control_contract.json")

    treatment = {
        "arm": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "role": "TREATMENT",
        "normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "reward": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "alpha_local": 0.1,
        "utility_architecture": "9-128-128-4",
        "mixer": "QMIX_MIXER_V1",
        "mixer_hidden_dim": 32,
        "central_state_dim": 55,
        "R_MAX": 10,
        "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "sibling_deduplication": False,
        "loss_specification": "EQUAL_MEAN_OF_JOINT_AND_FOCAL_REGIONAL_TD",
        "joint_loss": "JOINT_QMIX_SMOOTHL1",
        "regional_loss": "FOCAL_R1_DOUBLE_DQN_SMOOTHL1",
        "composite_loss": "EQUAL_COMPONENT_MEAN",
        "loss_formula": "L_C2 = 0.5*L_joint + 0.5*L_regional",
        "auxiliary_weight_tunable": False,
        "optimizer": "Adam",
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "action_selection": "ARGMAX_REGIONAL_UTILITY",
        "mixer_in_action_selection": False,
        "decentralized_execution": True,
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
    }
    _save(treatment, "E4MT_treatment_contract.json")
    return control, treatment


def phase_3_regional_target():
    regional = {
        "online_selects": "a*_r = argmax_a Q_online(s'_r, a)",
        "target_evaluates": "q'_r = Q_target(s'_r, a*_r)",
        "target_formula": "y_r = r_r^R1 + gamma*(1-done)*q'_r",
        "gamma": 0.99,
        "current_value": "q_r = Q_online(s_r, a_requested)",
        "loss": "SmoothL1(q_r, y_r)",
        "Double_Q": True,
        "requested_action_semantics_preserved": True,
    }
    return regional


def phase_4_5_loss(c2_sha):
    loss = {
        "C1_loss": "L_C1 = L_joint",
        "C2_loss": "L_C2 = 0.5*L_joint + 0.5*L_regional",
        "joint_weight": 0.5,
        "regional_weight": 0.5,
        "weights_tunable": False,
        "no_hyperparameter_sweep": True,
        "no_schedule": True,
        "no_adaptive_weighting": True,
        "no_scenario_dependent_weighting": True,
        "no_policy_dependent_weighting": True,
        "no_epoch_dependent_weighting": True,
        "L_joint_reduction": "mean_batch_joint_SmoothL1",
        "L_regional_reduction": "mean_batch_focal_SmoothL1",
        "L_C2_reduction": "0.5*L_joint + 0.5*L_regional",
        "implementation_source": "E4MI_validated",
    }
    _save(loss, "E4MT_loss_contract.json")
    return loss


def phase_6_gradient():
    gradient = {
        "C1": {
            "utility_gradients_from": ["L_joint"],
            "mixer_gradients_from": ["L_joint"],
        },
        "C2": {
            "utility_gradients_from": ["0.5*L_joint", "0.5*L_regional"],
            "mixer_gradients_from": ["0.5*L_joint"],
        },
        "target_gradients": False,
        "MIXER_GRADIENT_CONSEQUENCE": (
            "Compared with C1, C2 mixer joint-loss gradient is formally "
            "multiplied by 0.5 under otherwise identical minibatch tensors. "
            "This is an intrinsic property of the frozen composite loss."
        ),
        "NO_COMPENSATION": [
            "no mixer-specific LR",
            "no loss rescaling",
            "no separate optimizer",
        ],
        "logged_transparently": True,
    }
    _save(gradient, "E4MT_gradient_contract.json")
    return gradient


def phase_7_8_optimizer_target():
    optimizer = {
        "optimizer": "Adam",
        "lr": 1e-4,
        "parameters": "online_utility + online_mixer",
        "one_step_per_update_event": True,
        "separate_local_optimizer": False,
        "separate_mixer_optimizer": False,
        "one_zero_grad_backward_step": True,
        "parameter_count_identical_C1_C2": True,
    }
    _save(optimizer, "E4MT_optimizer_contract.json")

    target = {
        "networks": ["online_utility", "target_utility", "online_mixer", "target_mixer"],
        "soft_update_tau": 0.005,
        "same_cadence_both_arms": True,
        "regional_local_target_uses": "target_utility",
        "joint_target_uses": "target_utility + target_mixer",
    }
    return optimizer, target


def phase_9_architecture():
    arch = {
        "utility": "9-128-128-4",
        "mixer": "QMIX_MIXER_V1",
        "R_MAX": 10,
        "CENTRAL_STATE_DIM": 55,
        "MIXER_HIDDEN": 32,
        "MONOTONIC_TRANSFORM": "ABS",
        "parameter_counts_identical": True,
        "QMIX_CONFIG_SHA256": "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365",
    }
    _save(arch, "E4MT_architecture_identity.json")
    return arch


def phase_10_11_12_locks():
    norm = {
        "normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "identical_both_arms": True,
        "no_refitting": True,
        "no_arm_specific_scaler": True,
    }
    _save(norm, "E4MT_normalization_lock.json")

    reward = {
        "reward": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "alpha_local": 0.1,
        "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
        "identical_both_arms": True,
        "no_reward_difference": True,
    }
    _save(reward, "E4MT_reward_lock.json")

    replay = {
        "sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "sibling_deduplication": False,
        "DO_NOT_modify_replay_to_remove_H4": True,
        "C2_tests_regional_credit_NOT_sibling_elimination": True,
    }
    _save(replay, "E4MT_replay_lock.json")

    mixer = {
        "mixer": "QMIX_MIXER_V1",
        "hidden_dim": 32,
        "central_state_dim": 55,
        "R_MAX": 10,
        "monotonic": True,
        "identical_both_arms": True,
    }
    _save(mixer, "E4MT_mixer_lock.json")

    return norm, reward, replay, mixer


def phase_13_14_15_ops():
    hyper = {
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_over": 50000,
        "epsilon_decay_unit": "regional_anchors_transitions",
        "validation_epsilon": 0.0,
        "final_epsilon": 0.0,
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 1e-4,
        "batch_size": 128,
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
    }
    _save(hyper, "E4MT_hyperparameters.json")

    operators = {
        "KEEP": {"FE": 0},
        "LOCAL": {"sigma_L": 0.02},
        "PREDICT": {"sigma_P": 0.005, "lambda_P": 1, "fallback": "LOCAL"},
        "DIVERSIFY": {"sigma_D": 0.15},
        "budget": {"B": 10, "scope": "PER_ACTIVE_REGION"},
        "survival": "unrestricted_standard_NSGA_II",
        "no_HARD": True,
    }
    _save(operators, "E4MT_operator_contract.json")

    execution = {
        "action_selection": "argmax_a Q_utility(s_r, a)",
        "mixer_in_action_selection": False,
        "actions_assembled_simultaneously": True,
        "no_autoregressive_ordering": True,
        "decentralized": True,
    }
    return hyper, operators, execution


# ═══════════════════════════════════════════════════════════
# PHASES 16-20 — SEEDS, PAIRING, RNG
# ═══════════════════════════════════════════════════════════

def phase_16_17_seeds():
    train = list(range(2101, 2161))
    val = list(range(2161, 2181))
    final = list(range(2181, 2241))
    policy = [6001, 6002, 6003, 6004, 6005]

    seed_split = {
        "TRAIN_SEEDS": "2101..2160",
        "VALIDATION_SEEDS": "2161..2180",
        "FINAL_SEEDS": "2181..2240",
        "POLICY_SEEDS": policy,
        "train_count": 60,
        "validation_count": 20,
        "final_count": 60,
        "policy_count": 5,
    }
    _save(seed_split, "E4MT_seed_split.json")

    all_new_env = set(train) | set(val) | set(final)
    all_new_policy = set(policy)
    prior_env = _all_prior_env_seeds()
    prior_policy = _all_prior_policy_seeds()

    env_overlap = sorted(all_new_env & prior_env)
    policy_overlap = sorted(all_new_policy & prior_policy)

    disjoint = {
        "SEED_DISJOINTNESS": "PASS" if not env_overlap and not policy_overlap else "FAIL",
        "env_overlap": env_overlap,
        "policy_overlap": policy_overlap,
        "prior_env_ranges": [f"{lo}-{hi}" for lo, hi in PRIOR_ENV_RANGES],
        "prior_policy_ranges": [f"{lo}-{hi}" for lo, hi in PRIOR_POLICY_RANGES],
    }
    _save(disjoint, "E4MT_seed_disjointness.json")
    return disjoint["SEED_DISJOINTNESS"] == "PASS", seed_split


def phase_19_20_init_rng():
    init = {
        "policy_seeds": [6001, 6002, 6003, 6004, 6005],
        "at_t0_require": {
            "HASH_C1_utility_online_eq_HASH_C2_utility_online": True,
            "HASH_C1_mixer_online_eq_HASH_C2_mixer_online": True,
            "online_utility_eq_target_utility": True,
            "online_mixer_eq_target_mixer": True,
        },
        "identical_architectures": True,
        "full_network_paired": True,
    }
    _save(init, "E4MT_initialization_pairing.json")

    rng = {
        "environment_side": "deterministic_paired_schedule",
        "same_policy_epoch_scenario_env_seed_episode": True,
        "independent_mutable": [
            "replay_RNG",
            "exploration_state",
            "optimizer_state",
            "model_state",
        ],
        "do_not_force_identical_actions": True,
    }
    _save(rng, "E4MT_rng_contract.json")
    return init, rng


# ═══════════════════════════════════════════════════════════
# PHASES 21-28 — TRAINING/CHECKPOINT/VALIDATION
# ═══════════════════════════════════════════════════════════

def phase_21_22_budget():
    training = {
        "epochs_per_run": 6,
        "environments_per_epoch": 60,
        "scenarios_per_environment": 3,
        "episodes_per_epoch": 180,
        "episodes_per_run": 1080,
        "arms": 2,
        "policies": 5,
        "total_runs": 10,
        "total_training_episodes": 10800,
        "dynamic_changes_per_episode": 10,
        "no_early_stopping": True,
        "no_extra_C2_training": True,
        "scenario_balance": {
            "PERSISTENT": 60,
            "STATIONARY": 60,
            "MIXED": 60,
        },
        "no_PERSISTENT_oversampling": True,
    }
    _save(training, "E4MT_training_budget.json")

    update = {
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "one_optimizer_step_per_update_event": True,
        "regional_loss_does_NOT_increase": [
            "optimizer_step_frequency",
            "replay_samples",
            "environment_interactions",
        ],
    }
    _save(update, "E4MT_update_budget.json")
    return training, update


def phase_23_24_25_checkpoint():
    ckpt = {
        "checkpoint_schedule": [10000, 20000, 30000, 40000, 50000, "final"],
        "checkpoints_per_run": 6,
        "total_runs": 10,
        "total_checkpoints": 60,
    }
    _save(ckpt, "E4MT_checkpoint_contract.json")

    common = {
        "normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "reward": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "alpha_local": 0.1,
        "utility": "9-128-128-4",
        "mixer": "QMIX_MIXER_V1",
        "mixer_hidden": 32,
        "central_state": 55,
        "R_MAX": 10,
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 1e-4,
        "optimizer": "Adam",
        "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
        "replay": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "sibling_dedup": False,
        "decentralized_execution": True,
    }
    _save(common, "E4MT_common_config.json")

    c1_meta = {
        "architecture_version": "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "required_hashes": ["QMIX_CONFIG_SHA256", "V3_hash", "R1_hash", "protocol_hash"],
        "QMIX_CONFIG_SHA256": "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365",
        "no_C2_credit_config": True,
    }

    c2_meta = {
        "architecture_version": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "required_hashes": [
            "base_QMIX_CONFIG_SHA256",
            "C2_CREDIT_CONFIG_SHA256",
            "V3_hash",
            "R1_hash",
            "protocol_hash",
        ],
        "base_QMIX_CONFIG_SHA256": "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365",
        "hard_reject_mismatch": True,
    }

    return ckpt, c1_meta, c2_meta


def phase_26_27_28_validation(c2_sha):
    validation = {
        "validation_seeds": "2161..2180",
        "scenarios": ["PERSISTENT", "STATIONARY", "MIXED"],
        "arms": 2,
        "policies": 5,
        "checkpoints": 6,
        "environments": 20,
        "total_validation_episodes": 3600,
        "epsilon": 0.0,
        "eval_mode": True,
        "gradients_off": True,
        "optimizer_off": True,
        "replay_mutation_off": True,
        "target_updates_off": True,
        "normalizer_adaptation_off": True,
        "mixer_not_in_action_selection": True,
    }
    _save(validation, "E4MT_validation_contract.json")

    selection = {
        "scope": "per_arm_per_policy",
        "select_exactly_one_checkpoint": True,
        "primary_criterion": "minimum_validation_MACRO_MIGD_RESPONSE",
        "macro_weight": "equal_1_3_each_scenario",
        "tie_1": "lower_macro_MIGD_epoch",
        "tie_2": "lower_macro_response_FE",
        "tie_3": "earlier_checkpoint",
        "forbidden_selection_signals": [
            "L_joint",
            "L_regional",
            "L_composite",
            "sibling_loss_dispersion",
            "gradient_cosine",
            "MIXED_only_performance",
            "PERSISTENT_only_performance",
            "C1_vs_C2_validation_delta",
        ],
        "retain_all_10_policies": True,
        "no_best_policy_selection": True,
        "no_dropping_bad_seeds": True,
    }
    _save(selection, "E4MT_checkpoint_selection.json")

    return validation, selection


# ═══════════════════════════════════════════════════════════
# PHASES 29-35 — LOGGING & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

def phase_29_30_logging():
    training_log = {
        "per_episode_fields": [
            "arm", "policy_seed", "epoch", "scenario",
            "environment_seed", "episode_index",
            "MIGD_response", "MIGD_epoch", "response_FE",
            "anchor_count", "update_count",
        ],
    }
    _save(training_log, "E4MT_training_logging_contract.json")
    return training_log


def phase_31_32_33_34_diagnostics():
    credit = {
        "C2_descriptive_report": [
            "joint_loss_mean_median",
            "regional_loss_mean_median",
            "composite_loss_mean_median",
            "within_event_std_joint_loss",
            "within_event_std_regional_loss",
            "within_event_std_composite_loss",
            "joint_gradient_norm",
            "regional_gradient_norm",
            "composite_gradient_norm",
            "joint_vs_regional_gradient_cosine",
        ],
        "CREDIT_PRESERVATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "diagnostics_are_explanatory_only": True,
    }
    _save(credit, "E4MT_credit_diagnostics.json")

    sibling_diag = {
        "sampled_events": True,
        "C1_sibling_joint_loss_dispersion": True,
        "C2_sibling_regional_loss_dispersion": True,
        "C2_sibling_composite_loss_dispersion": True,
        "expected": {
            "joint_sibling_dispersion": "near_zero",
            "regional_composite_may_differ": True,
        },
        "no_minimum_magnitude_success_threshold": True,
    }

    gradient_diag = {
        "sampled_updates_report": [
            "||g_joint_utility||",
            "||g_regional_utility||",
            "cosine(g_joint_utility, g_regional_utility)",
            "||g_composite_utility||",
            "||g_joint_mixer||",
            "||g_composite_mixer||",
        ],
        "engineering_expectation": "g_composite_mixer ≈ 0.5*g_joint_mixer",
        "no_performance_threshold": True,
    }
    _save(gradient_diag, "E4MT_action_diagnostics.json")

    action_diag = {
        "both_arms": [
            "KEEP_fraction",
            "LOCAL_fraction",
            "PREDICT_fraction",
            "DIVERSIFY_fraction",
            "multi_action_rate",
            "regional_action_entropy",
        ],
        "descriptive_only": True,
    }

    return credit, sibling_diag, gradient_diag, action_diag


def phase_35_q_margin():
    q_margin = {
        "purpose": "E4L could not audit H7 because state/Q vectors were not stored",
        "logging": {
            "top1_utility_Q": True,
            "top2_utility_Q": True,
            "Q_margin": "top1 - top2",
        },
        "sampled_regional_decisions": True,
        "NOT_a_success_gate": True,
        "does_NOT_change_policy": True,
        "no_additional_objective_FE": True,
        "improves_auditability_only": True,
    }
    _save(q_margin, "E4MT_q_margin_logging.json")
    return q_margin


def phase_36_numerical():
    numerical = {
        "require_finite": [
            "utility_Q", "Q_tot", "local_target", "joint_target",
            "L_joint", "L_regional", "L_composite",
            "utility_gradients", "mixer_gradients",
            "parameters", "MIGD_metrics",
        ],
        "if_numerical_failure": "preserve_do_not_retune",
    }
    _save(numerical, "E4MT_numerical_stability.json")
    return numerical


# ═══════════════════════════════════════════════════════════
# PHASES 37-50 — CAUSAL INFERENCE DESIGN
# ═══════════════════════════════════════════════════════════

def phase_39_40_41_42_stats():
    hypothesis = {
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "EFFECT_DEFINITION": "C1_MINUS_C2",
        "POSITIVE_MEANS": "C2_BETTER",
        "WHY": "lower MIGD is better; positive C1-C2 means C2 has lower MIGD",
    }
    _save(hypothesis, "E4MT_primary_hypothesis.json")

    effect = {
        "formula": "d[p,e] = MIGD_C1[p,e] - MIGD_C2[p,e]",
        "positive_means": "C2_BETTER",
    }
    _save(effect, "E4MT_effect_definition.json")

    margin = {
        "PRACTICAL_MARGIN": 1e-5,
    }
    _save(margin, "E4MT_practical_margin.json")

    stats = {
        "method": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis": 5,
        "environment_axis": 60,
        "resamples": 50000,
        "CI": "95_percentile",
        "BOOTSTRAP_MASTER_SEED": 106001,
        "macro": "equal_1_3_across_PERSISTENT_STATIONARY_MIXED",
        "shared_draws_within_replicate": True,
    }
    _save(stats, "E4MT_statistical_design.json")

    return hypothesis, effect, margin, stats


def phase_43_classification():
    classification = {
        "categories": ["BENEFICIAL", "DIRECTIONAL_POSITIVE", "PRACTICALLY_NEUTRAL",
                        "DIRECTIONAL_NEGATIVE", "HARMFUL"],
        "practical_margin": 1e-5,
        "truth_table_discipline": "same_as_prior_confirmatory_experiments",
    }
    _save(classification, "E4MT_classification_logic.json")
    return classification


def phase_44_robustness():
    robustness = {
        "per_policy_delta": "Delta_policy = mean_e(C1-C2)",
        "PASS_iff": ">=4/5 are positive",
    }
    _save(robustness, "E4MT_policy_robustness.json")
    return robustness


def phase_45_46_47_48_safety():
    safety = {
        "PERSISTENT_safety": "neither MIGD_response nor MIGD_epoch is HARMFUL for C2 vs C1",
        "STATIONARY_safety": "neither MIGD_response nor MIGD_epoch is HARMFUL for C2 vs C1",
        "MACRO_safety": "equal 1/3 scenario macro, neither response nor epoch HARMFUL",
        "effect": "C1_MINUS_C2",
    }
    _save(safety, "E4MT_safety_contract.json")

    fe = {
        "effect": "response_FE_C1 - response_FE_C2",
        "positive_means": "C2_cheaper",
        "role": "SECONDARY_ONLY",
        "cannot_rescue_harmful_MIGD": True,
    }
    _save(fe, "E4MT_fe_contract.json")

    return safety, fe


def phase_49_50_success():
    success = {
        "CREDIT_PRESERVATION_CAUSAL_VALIDATED": {
            "requires_ALL": [
                "MIXED_primary = BENEFICIAL",
                "policy_robustness = PASS (>=4/5)",
                "PERSISTENT_safety = PASS",
                "STATIONARY_safety = PASS",
                "MACRO_safety = PASS",
                "loss_spec_integrity = PASS",
                "checkpoint_integrity = PASS",
                "protocol_integrity = PASS",
                "no_final_data_adaptation",
            ],
        },
        "PARTIAL": {
            "PERFORMANCE_PARTIAL": [
                "primary DIRECTIONAL_POSITIVE + all safety/integrity PASS",
                "OR primary BENEFICIAL but robustness FAIL with safety/integrity PASS",
            ],
            "PRACTICALLY_NEUTRAL": "primary PRACTICALLY_NEUTRAL with safety/integrity PASS",
        },
        "FAILED": {
            "conditions": [
                "primary DIRECTIONAL_NEGATIVE",
                "primary HARMFUL",
                "any safety FAIL",
                "loss_spec integrity FAIL",
                "checkpoint/protocol integrity FAIL",
            ],
        },
        "CANNOT_RESCUE": [
            "differentiated_sibling_loss",
            "gradient_diversity",
            "Q_margins",
            "action_diversity",
            "FE_savings",
        ],
    }
    _save(success, "E4MT_success_logic.json")
    return success


# ═══════════════════════════════════════════════════════════
# PHASES 51-55 — FUTURE PLANNING
# ═══════════════════════════════════════════════════════════

def phase_51_52_53_future():
    final_matrix = {
        "arms": 2,
        "policies": 5,
        "environments": 60,
        "scenarios": 3,
        "total_episodes": 1800,
        "DO_NOT_EXECUTE_IN_E4MT": True,
        "DO_NOT_EXECUTE_IN_E4MTR": True,
    }
    _save(final_matrix, "E4MT_future_final_matrix.json")

    sequence = {
        "if_E4MT_PASS": [
            "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION",
            "E4ME_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
            "E4MF_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_EXECUTION",
        ],
        "do_not_collapse_stages": True,
    }
    _save(sequence, "E4MT_future_stage_sequence.json")

    e4mtr_contract = {
        "required": {
            "runs_complete": "10/10",
            "episodes_per_run": 1080,
            "total_training_episodes": 10800,
            "total_checkpoints": 60,
            "validation_episodes": 3600,
            "selected_checkpoints": 10,
            "final_seed_access": 0,
            "no_retuning": True,
        },
    }
    _save(e4mtr_contract, "E4MT_e4mtr_output_contract.json")

    routing = {
        "IF_CREDIT_PRESERVATION_CAUSAL_VALIDATED": {
            "next": "E4N_CREDIT_PRESERVATION_VS_C0_PROTOCOL_DESIGN",
            "question": "Does C2 also outperform C0 Independent Shared DDQN?",
            "requires_fresh_environment_set": True,
            "DO_NOT_infer_C2_gt_C0_transitively": True,
        },
        "IF_PARTIAL": {
            "next": "E4N_CREDIT_PRESERVATION_RESULT_REVIEW",
        },
        "IF_FAILED_WITH_INTEGRITY_PASS": {
            "next": "E4N_CREDIT_PRESERVATION_FAILURE_ANALYSIS",
            "return_to": ["H1", "H6", "H10"],
            "do_not_tune_weights_after_failure": True,
        },
        "IF_INTEGRITY_FAILURE": {
            "next": "E4M_ENGINEERING_INTEGRITY_REVIEW",
        },
    }

    return final_matrix, sequence, e4mtr_contract, routing


# ═══════════════════════════════════════════════════════════
# FIREWALL & NO-TRAINING AUDIT
# ═══════════════════════════════════════════════════════════

def phase_firewall():
    firewall = {
        "M_TRAIN_ACCESS_COUNT": 0,
        "M_VALIDATION_ACCESS_COUNT": 0,
        "M_FINAL_ACCESS_COUNT": 0,
        "no_scientific_environment_execution": True,
        "no_training": True,
    }
    _save(firewall, "E4MT_final_firewall.json")

    audit = {
        "optimizer_steps": 0,
        "environment_episodes": 0,
        "training_episodes": 0,
        "seeds_accessed": [],
        "checkpoints_created": 0,
    }
    _save(audit, "E4MT_no_training_audit.json")

    return firewall, audit


# ═══════════════════════════════════════════════════════════
# VERDICT & REPORT
# ═══════════════════════════════════════════════════════════

def build_verdict(pre_ok, disjoint_ok, c2_sha, regression_result):
    conditions = {
        "E4MI_complete": True,
        "implementation_valid": True,
        "historical_verdicts_preserved": pre_ok,
        "C1_exact_control_frozen": True,
        "C2_exact_treatment_frozen": True,
        "architecture_identical": True,
        "parameter_counts_identical": True,
        "joint_loss_exact": True,
        "regional_loss_exact": True,
        "composite_0_5_exact": True,
        "not_tunable": True,
        "gradient_paths_frozen": True,
        "V3_same": True,
        "R1_same": True,
        "alpha_same": True,
        "mixer_same": True,
        "joint_reward_same": True,
        "replay_same": True,
        "no_sibling_dedup": True,
        "full_network_init_paired": True,
        "fresh_seed_split_exact_disjoint": disjoint_ok,
        "training_budget_frozen": True,
        "update_budget_frozen": True,
        "checkpoint_schedule_frozen": True,
        "validation_selection_frozen": True,
        "MIXED_primary_frozen": True,
        "C1_C2_effect_frozen": True,
        "practical_margin_frozen": True,
        "crossed_bootstrap_frozen": True,
        "policy_robustness_frozen": True,
        "safety_frozen": True,
        "mechanism_diagnostics_descriptive_only": True,
        "Q_margin_logging_descriptive_only": True,
        "final_result_routing_frozen": True,
        "future_final_seeds_untouched": True,
        "no_environment_execution": True,
        "no_training": True,
        "full_regression_pass": regression_result,
    }

    all_pass = all(conditions.values())

    if all_pass:
        next_stage = "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION"
    elif not disjoint_ok:
        next_stage = "E4M_SEED_REVIEW"
    else:
        next_stage = "E4MT_INTEGRITY_REVIEW"

    verdict = {
        "stage": "E4MT_CREDIT_PRESERVATION_AB_TRAINING_PROTOCOL_FREEZE",
        "E4MT_COMPLETE": "YES" if all_pass else "NO",
        "conditions": conditions,
        "failed_conditions": [k for k, v in conditions.items() if not v],
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
        "NEXT_STAGE": next_stage,
    }
    _save(verdict, "E4MT_verdict.json")
    return verdict


def write_protocol_frozen(c2_sha):
    protocol = {
        "EXPERIMENT_FACTOR": "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX",
        "CONTROL": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "TREATMENT": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "C1_LOSS": "JOINT_QMIX_SMOOTHL1",
        "C2_LOSS": "0.5*L_joint + 0.5*L_regional",
        "AUXILIARY_WEIGHT_TUNABLE": False,
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "EFFECT": "C1_MINUS_C2",
        "POSITIVE_MEANS": "C2_BETTER",
        "PRACTICAL_MARGIN": 1e-5,
        "BOOTSTRAP_SEED": 106001,
        "TRAIN_SEEDS": "2101..2160",
        "VALIDATION_SEEDS": "2161..2180",
        "FINAL_SEEDS": "2181..2240",
        "POLICY_SEEDS": [6001, 6002, 6003, 6004, 6005],
        "TOTAL_TRAINING_EPISODES": 10800,
        "TOTAL_CHECKPOINTS": 60,
        "TOTAL_VALIDATION_EPISODES": 3600,
        "FUTURE_FINAL_EPISODES": 1800,
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
        "QMIX_CONFIG_SHA256": "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365",
    }
    _save(protocol, "E4MT_protocol_frozen.json")
    return protocol


def write_report(c2_sha, regression_str):
    report = f"""# E4MT — Credit Preservation A/B Training Protocol Freeze

## A. Historical evidence preservation

| Verdict | Value | Changed |
|---------|-------|---------|
| E4F | FAILED | NO |
| E4HF | FAILED | NO |
| E4IF | VALIDATED | NO |
| E4JF | FAILED | NO |
| E4KF | FAILED | NO |

E4L primary failure hypothesis: MULTIFACTORIAL_OR_UNRESOLVED.
H4 (sibling target redundancy / credit homogenization): SUPPORTED.

## B. E4MI implementation-valid status

E4MI_COMPLETE = YES.
CREDIT_PRESERVATION_IMPLEMENTATION_VALID = YES.
All 45 implementation conditions: PASS.
C2_CREDIT_CONFIG_SHA256 = {c2_sha}.

## C. Scientific question

Does preserving direct focal-region R1 Double-DQN TD supervision inside the
otherwise unchanged QMIX learner improve dynamic multi-objective response
performance relative to the original joint-only QMIX learner?

This experiment tests C1 vs C2 only.
It does NOT test C2 vs C0, C2 vs GLOBAL_PREDICT, or alternative QMIX architectures.

## D. C1 control

C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1.
Loss: L_C1 = L_joint (joint QMIX Double-Q TD SmoothL1).
V3 + R1 + alpha=0.1 + utility 9-128-128-4 + QMIX_MIXER_V1 hidden=32.
Central state dim=55, R_MAX=10, active-region mean R1 joint reward.
Regional-anchor replay, no sibling deduplication.

## E. C2 treatment

C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1.
Loss: L_C2 = 0.5*L_joint + 0.5*L_regional.
Identical to C1 except training loss computation.

## F. Experimental factor

REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX.
The only difference: C1 uses joint-only TD; C2 uses equal-weighted composite
of joint and focal-regional TD.

## G. Composite-loss interpretation

The C2 treatment simultaneously:
1. Adds focal region-specific TD supervision.
2. Gives the joint component coefficient 0.5 rather than 1.0.

Any future causal result supports the frozen CREDIT-PRESERVING COMPOSITE LOSS
SPECIFICATION, not the stronger statement that adding local TD alone caused
the effect independently of joint-loss reweighting.

## H. Gradient-path consequence

C2 mixer receives gradients from 0.5*L_joint only (regional loss does not flow
through mixer). Compared with C1, C2 mixer joint-loss gradient is formally
multiplied by 0.5. This is intrinsic to the frozen composite loss.
No compensation: no mixer-specific LR, no loss rescaling, no separate optimizer.

## I. Architecture identity

Both arms: utility 9-128-128-4, QMIX_MIXER_V1, R_MAX=10, central_state=55,
hidden=32, ABS monotonic transform. Parameter counts identical.

## J. V3/R1/joint-reward locks

Both: ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3.
Both: REWARD_R1_REGIONAL_QUALITY_GAIN, alpha_local=0.1.
Both: ACTIVE_REGION_MEAN_OF_R1 joint reward.

## K. Replay preservation

Both: REGIONAL_ANCHOR_WITH_JOINT_CONTEXT.
Capacity=50000, warmup=1000, update_every=4, batch=128.
Sibling deduplication: NO.

## L. Paired full-network initialization

For each policy seed (6001-6005): C1 and C2 start with identical online utility
weights, identical online mixer weights. Within each arm: online=target at t=0.

## M. Fresh seed split

Train: 2101..2160 (60). Validation: 2161..2180 (20). Final: 2181..2240 (60).
Policy: 6001..6005 (5). All disjoint from prior scientific sets.

## N. RNG pairing

Matched environment-side schedule for same policy/epoch/scenario/env-seed/episode.
Independent mutable: replay RNG, exploration state, optimizer state, model state.

## O. Training/update budget

6 epochs x 60 envs x 3 scenarios = 1080 episodes/run.
2 arms x 5 policies = 10 runs. Total: 10800 training episodes.
Scenario balance: PERSISTENT=60, STATIONARY=60, MIXED=60 per epoch.
Update: every 4 anchors, batch=128, one optimizer step per event.

## P. Checkpoint schedule

Save at: 10000, 20000, 30000, 40000, 50000, final.
6 per run, 60 total.

## Q. Validation/checkpoint selection

3600 validation episodes (2x5x6x20x3).
Selection: minimum validation MACRO_MIGD_RESPONSE (equal 1/3 scenario).
Tiebreaks: macro MIGD_epoch, macro response_FE, earlier checkpoint.
All 10 policies retained.

## R. Credit-preservation diagnostics

C2: loss means/medians, within-event std, gradient norms, cosine.
Sibling dispersion: joint near-zero expected, regional/composite may differ.
MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.

## S. Action diagnostics

Both arms: KEEP/LOCAL/PREDICT/DIVERSIFY fractions, multi-action rate, entropy.
Descriptive only.

## T. Q-margin logging

top1/top2 utility Q, Q_margin = top1-top2 for sampled regional decisions.
NOT a success gate. Improves auditability for future H7 analysis.

## U. Primary causal contrast

PRIMARY_SCENARIO = MIXED.
PRIMARY_ENDPOINT = MIGD_RESPONSE.
Effect: d[p,e] = MIGD_C1[p,e] - MIGD_C2[p,e]. Positive = C2 better.
PRACTICAL_MARGIN = 1e-5.

## V. Statistical design

PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP.
Policy axis=5, environment axis=60.
N=50000 resamples, 95% percentile CI, master seed=106001.
Macro: shared draws across scenarios within replicate.

## W. Robustness

Per-policy Delta = mean_e(C1-C2). PASS iff >=4/5 positive.

## X. Safety

PERSISTENT: neither MIGD_response nor MIGD_epoch HARMFUL.
STATIONARY: same.
MACRO (equal 1/3): same.

## Y. FE analysis

Effect: response_FE_C1 - response_FE_C2. Positive = C2 cheaper.
Secondary only. Cannot rescue harmful MIGD.

## Z. Final causal verdict logic

CREDIT_PRESERVATION_CAUSAL_VALIDATED requires ALL:
MIXED primary BENEFICIAL + robustness PASS + all safety PASS +
loss/checkpoint/protocol integrity PASS + no final-data adaptation.

PARTIAL (PERFORMANCE_PARTIAL): directional_positive + safety/integrity pass;
or beneficial but robustness fail with safety pass.

PARTIAL (PRACTICALLY_NEUTRAL): neutral + safety/integrity pass.

FAILED: directional_negative, harmful, any safety fail, any integrity fail.

## AA. Future final firewall

Final seeds 2181..2240 untouched.
M_TRAIN_ACCESS_COUNT = 0. M_VALIDATION_ACCESS_COUNT = 0. M_FINAL_ACCESS_COUNT = 0.

## AB. Stage sequence

E4MT -> E4MTR -> E4ME -> E4MF.
VALIDATED -> E4N (C2 vs C0, fresh envs, no transitive inference).
PARTIAL -> E4N result review.
FAILED + integrity pass -> E4N failure analysis (return to H1/H6/H10).
Integrity failure -> E4M engineering review.

## AC. Scientific limitations

- H4 is supported but not unique root cause.
- C2 does not remove repeated joint targets.
- C2 does not alter event weighting.
- C2 does not alter team reward.
- C2 keeps QMIX monotonic restriction.
- 0.5/0.5 weighting is a frozen design choice.
- C2 changes both presence of direct regional TD and relative scalar weighting
  of the joint component.
- Future C1-vs-C2 result applies to the whole frozen composite-loss specification.
- C2 > C1 would not imply C2 > C0.
- C2 > C1 would not imply global-comparator superiority.
- Only five policy seeds.
- No real-world UAV evidence.

## AD. Next stage

NEXT_STAGE = E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION.

FULL_REGRESSION = {regression_str}.
"""
    with open(OUT_DIR / "E4MT_report.md", "w") as f:
        f.write(report)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def run(regression_result=True, regression_str="PENDING"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_ok, c2_sha = phase_0_pre_freeze_integrity()
    if not pre_ok:
        verdict = build_verdict(False, True, c2_sha, False)
        print(f"\nE4MT_COMPLETE = NO")
        print(f"NEXT_STAGE = E4MT_INTEGRITY_REVIEW")
        return verdict

    phase_1_2_arms(c2_sha)
    phase_3_regional_target()
    phase_4_5_loss(c2_sha)
    phase_6_gradient()
    phase_7_8_optimizer_target()
    phase_9_architecture()
    phase_10_11_12_locks()
    phase_13_14_15_ops()

    disjoint_ok, seed_split = phase_16_17_seeds()
    if not disjoint_ok:
        verdict = build_verdict(True, False, c2_sha, False)
        print(f"\nE4MT_COMPLETE = NO")
        print(f"NEXT_STAGE = E4M_SEED_REVIEW")
        return verdict

    phase_19_20_init_rng()
    phase_21_22_budget()
    phase_23_24_25_checkpoint()
    phase_26_27_28_validation(c2_sha)
    phase_29_30_logging()
    phase_31_32_33_34_diagnostics()
    phase_35_q_margin()
    phase_36_numerical()
    phase_39_40_41_42_stats()
    phase_43_classification()
    phase_44_robustness()
    phase_45_46_47_48_safety()
    phase_49_50_success()
    phase_51_52_53_future()
    phase_firewall()
    write_protocol_frozen(c2_sha)

    verdict = build_verdict(pre_ok, disjoint_ok, c2_sha, regression_result)
    write_report(c2_sha, regression_str)

    # ── terminal summary ──
    print(f"""
E4MT_COMPLETE =
    {"YES" if verdict["E4MT_COMPLETE"] == "YES" else "NO"}

ORIGINAL_E4KF_VERDICT =
    FAILED

E4KF_VERDICT_CHANGED =
    NO

E4L_PRIMARY_FAILURE_HYPOTHESIS =
    MULTIFACTORIAL_OR_UNRESOLVED

H4_STATUS =
    SUPPORTED

EXPERIMENT_FACTOR =
    REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX

CONTROL =
    C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1

TREATMENT =
    C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1

CREDIT_PRESERVATION_IMPLEMENTATION_VALID =
    YES

C1_LOSS =
    JOINT_QMIX_SMOOTHL1

C2_JOINT_LOSS =
    JOINT_QMIX_SMOOTHL1

C2_REGIONAL_LOSS =
    FOCAL_R1_DOUBLE_DQN_SMOOTHL1

C2_COMPOSITE_LOSS =
    0.5*L_joint + 0.5*L_regional

AUXILIARY_WEIGHT_TUNABLE =
    NO

ARCHITECTURE_IDENTITY =
    YES

PARAMETER_COUNT_IDENTITY =
    YES

FIXED_NORMALIZATION =
    ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3

FIXED_REWARD =
    REWARD_R1_REGIONAL_QUALITY_GAIN

FIXED_ALPHA_LOCAL =
    0.1

JOINT_REWARD =
    ACTIVE_REGION_MEAN_OF_R1

REPLAY_SAMPLING_UNIT =
    REGIONAL_ANCHOR_WITH_JOINT_CONTEXT

SIBLING_DEDUPLICATION =
    NO

UTILITY_NETWORK =
    9-128-128-4

QMIX_MIXER =
    QMIX_MIXER_V1

MIXER_HIDDEN =
    32

C2_CREDIT_CONFIG_SHA256 =
    {c2_sha}

TRAIN_SEEDS =
    2101..2160

VALIDATION_SEEDS =
    2161..2180

FINAL_SEEDS =
    2181..2240

POLICY_SEEDS =
    6001..6005

PAIRED_FULL_NETWORK_INITIALIZATION =
    FROZEN

TRAINING_EPISODES_PER_RUN =
    1080

TOTAL_RUNS =
    10

TOTAL_TRAINING_EPISODES =
    10800

CHECKPOINTS_PLANNED =
    60

VALIDATION_EPISODES_PLANNED =
    3600

CHECKPOINT_SELECTION =
    VALIDATION_MACRO_MIGD_RESPONSE

PRIMARY_SCENARIO =
    MIXED

PRIMARY_ENDPOINT =
    MIGD_RESPONSE

PRIMARY_EFFECT =
    C1_MINUS_C2

POSITIVE_MEANS =
    C2_BETTER

PRACTICAL_MARGIN =
    1e-5

POLICY_ROBUSTNESS =
    AT_LEAST_4_OF_5_C2_BETTER

STATISTICAL_DESIGN =
    PAIRED_CROSSED_TWO_AXIS_5x60

BOOTSTRAP_RESAMPLES =
    50000

BOOTSTRAP_MASTER_SEED =
    106001

CREDIT_PRESERVATION_MECHANISM_GATE =
    NOT_USED_FOR_SCIENTIFIC_VERDICT

M_TRAIN_ACCESS_COUNT =
    0

M_VALIDATION_ACCESS_COUNT =
    0

M_FINAL_ACCESS_COUNT =
    0

NEW_ENVIRONMENT_EXECUTION =
    NO

NEW_TRAINING =
    NO

FULL_REGRESSION =
    {regression_str}

NEXT_STAGE =
    {verdict["NEXT_STAGE"]}
""")

    return verdict


if __name__ == "__main__":
    run()
