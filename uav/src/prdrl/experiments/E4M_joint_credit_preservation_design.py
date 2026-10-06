"""
E4M — Joint Credit Preservation Design (PR-DRL-DMOEA, Branch C).

Design/freeze stage only. No training, no environment execution, no new seeds used.
Freezes C1 (control) vs C2 (treatment: QMIX + focal regional R1 TD) for the next
causal A/B experiment targeting H4 sibling-target redundancy / credit homogenization.

E4KF verdict = FAILED (immutable). E4L PRIMARY = MULTIFACTORIAL_OR_UNRESOLVED.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path("results/E4M_joint_credit_preservation_design")
E4L_DIR = Path("results/E4L_coordination_failure_analysis")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


# =====================================================================
# Phase E4M-0: historical integrity
# =====================================================================
def phase_0_historical_integrity():
    e4l = _load(E4L_DIR / "E4L_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
    hyp = _load(E4L_DIR / "E4L_failure_hypothesis_summary.json")

    checks = {
        "E4L_COMPLETE": e4l["E4L_COMPLETE"] == "YES",
        "E4KF_FAILED": e4kf["COORDINATION_CAUSAL_VERDICT"] == "FAILED",
        "E4KF_MIXED_HARMFUL": e4kf["MIXED_PRIMARY"] == "HARMFUL",
        "E4KF_ROBUSTNESS_1_OF_5": e4kf["POLICY_PAIR_POSITIVE_COUNT"] == 1,
        "E4KF_PERSISTENT_SAFETY_FAIL": e4kf["PERSISTENT_SAFETY"] == "FAIL",
        "E4KF_STATIONARY_SAFETY_PASS": e4kf["STATIONARY_SAFETY"] == "PASS",
        "E4KF_MACRO_SAFETY_FAIL": e4kf["MACRO_SAFETY"] == "FAIL",
        "E4IF_VALIDATED": e4kf["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED",
        "E4HF_FAILED": e4kf["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED",
        "E4F_FAILED": e4kf["ORIGINAL_E4F_VERDICT"] == "FAILED",
        "H4_SUPPORTED": hyp["hypothesis_statuses"]["H4"] == "SUPPORTED",
        "H2_WEAK": hyp["hypothesis_statuses"]["H2"] == "WEAK",
        "H3_WEAK": hyp["hypothesis_statuses"]["H3"] == "WEAK",
        "H5_NOT_SUPPORTED": hyp["hypothesis_statuses"]["H5"] == "NOT_SUPPORTED",
        "H6_PLAUSIBLE": hyp["hypothesis_statuses"]["H6"] == "PLAUSIBLE",
        "PRIMARY_MULTIFACTORIAL": hyp["PRIMARY_FAILURE_HYPOTHESIS"] == "MULTIFACTORIAL_OR_UNRESOLVED",
    }
    return {"all_pass": all(checks.values()), "checks": checks}


# =====================================================================
# Phase E4M-1: candidate revision review
# =====================================================================
def phase_1_candidate_review():
    return {
        "M0_ORIGINAL_QMIX_C1": {
            "description": "Failed control C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
            "role": "FUTURE_CONTROL",
            "selected": False,
            "reason": "E4KF verdict = FAILED; remains as causal control",
        },
        "M1_QMIX_PLUS_REGIONAL_AUXILIARY_TD": {
            "description": "Preserve exact C1 architecture + joint loss, add focal-region "
                           "Double-DQN TD supervision for each sampled regional anchor",
            "mechanism_tested": "H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
            "selected": True,
            "reason": "Directly restores region-specific R1 training credit without "
                      "removing explicit joint QMIX learning; smallest isolated revision "
                      "targeting strongest directly actionable supported mechanism",
            "conditions_met": {
                "retains_exact_joint_TD": True,
                "retains_exact_replay": True,
                "retains_exact_team_reward": True,
                "reuses_existing_regional_R1_target_path": True,
                "requires_no_new_network": True,
                "preserves_decentralized_execution": True,
                "preserves_mixer_architecture": True,
                "preserves_V3_R1": True,
                "adds_no_environment_FE": True,
            },
        },
        "M2_QMIX_EVENT_DEDUPLICATED_JOINT_LOSS": {
            "description": "Deduplicates or reweights joint events",
            "selected": False,
            "reason": "H3 = WEAK; introduces less-supported sampling-factor change",
        },
        "M3_QMIX_MODIFIED_TEAM_REWARD": {
            "description": "Changes team reward aggregation",
            "selected": False,
            "reason": "H2 = WEAK; no strong evidence for reward aggregation change",
        },
        "M4_NON_MONOTONIC_CENTRALIZED_Q": {
            "description": "Replaces QMIX with non-monotonic centralized critic",
            "selected": False,
            "reason": "H6 = PLAUSIBLE only; larger architecture-family change; "
                      "isolate credit-preservation first",
        },
    }


# =====================================================================
# Phase E4M-2: selection
# =====================================================================
def phase_2_selection(candidates):
    m1 = candidates["M1_QMIX_PLUS_REGIONAL_AUXILIARY_TD"]
    all_conditions = all(m1["conditions_met"].values())
    return {
        "SELECTED_REVISION": "M1_QMIX_REGIONAL_CREDIT_PRESERVATION_V1" if all_conditions else "NONE",
        "all_selection_conditions_met": all_conditions,
        "conditions": m1["conditions_met"],
    }


# =====================================================================
# Contracts
# =====================================================================
def build_control_contract():
    return {
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
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "optimizer": "Adam",
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "loss": "SmoothL1",
        "loss_specification": "JOINT_QMIX_SMOOTHL1",
        "epsilon_schedule": "FROZEN_FROM_E4KT",
        "action_selection": "ARGMAX_REGIONAL_UTILITY",
        "mixer_in_action_selection": False,
        "decentralized_execution": True,
    }


def build_treatment_contract():
    control = build_control_contract()
    treatment = dict(control)
    treatment.update({
        "arm": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "role": "TREATMENT",
        "loss_specification": "EQUAL_MEAN_OF_JOINT_AND_FOCAL_REGIONAL_TD",
        "joint_loss": "JOINT_QMIX_SMOOTHL1",
        "regional_loss": "FOCAL_R1_DOUBLE_DQN_SMOOTHL1",
        "composite_loss": "EQUAL_COMPONENT_MEAN",
        "loss_formula": "0.5*L_joint + 0.5*L_regional",
        "auxiliary_weight_tuning": False,
        "auxiliary_coefficient_tunable": False,
    })
    return treatment


def build_joint_loss_preservation():
    return {
        "description": "For each sampled regional anchor i, compute exact existing C1 joint loss",
        "L_joint_definition": "SmoothL1(Q_tot_online_i, y_joint_i)",
        "Q_tot_online": "QMIX mixer output from online utility Q-values + online mixer",
        "y_joint": "r_joint + gamma*(1-done)*Q_tot_target(s', argmax_a Q_tot_online(s',a))",
        "Double_Q": True,
        "unchanged_from_C1": True,
    }


def build_regional_target_contract():
    return {
        "description": "Focal regional Double-DQN TD target for sampled anchor i with focal region r_i",
        "focal_transition": "s_r, a_r, r_r^R1, s'_r, done",
        "a_local_star": "argmax_a Q_online(s'_r, a)",
        "y_local": "r_r^R1 + gamma*(1-done)*Q_target(s'_r, a_local_star)",
        "gamma": 0.99,
        "q_local": "Q_online(s_r, a_r)",
        "action_convention": "REQUESTED policy action (not realized operator)",
        "L_regional": "SmoothL1(q_local, y_local)",
        "identical_to_C0_regional_TD": True,
        "no_new_reward": True,
        "no_counterfactual_label": True,
        "no_oracle_target": True,
    }


def build_composite_loss_contract():
    return {
        "formula": "L_C2 = 0.5 * L_joint + 0.5 * L_regional",
        "LOSS_COMBINATION": "EQUAL_COMPONENT_MEAN",
        "joint_weight": 0.5,
        "regional_weight": 0.5,
        "weights_tunable": False,
        "lambda_aux_exposed": False,
        "interpretation": "unweighted arithmetic mean of the two frozen supervision components",
        "rationale": "avoids doubling total scalar loss; gives equal formal weight to joint "
                     "coordination supervision and focal region-specific R1 supervision",
    }


def build_gradient_path_contract():
    return {
        "online_utility_receives_gradients_from": ["L_joint", "L_regional"],
        "online_mixer_receives_gradients_from": ["L_joint"],
        "L_regional_depends_on_mixer": False,
        "target_utility_gradients": False,
        "target_mixer_gradients": False,
        "one_backward_pass": True,
        "composite_loss_backward_once": True,
    }


def build_optimizer_contract():
    return {
        "optimizer": "Adam",
        "lr": 1e-4,
        "parameters": "online_utility + online_mixer",
        "one_step_per_update_event": True,
        "separate_local_optimizer": False,
        "separate_mixer_optimizer": False,
        "alternating_optimization": False,
    }


def build_target_network_contract():
    return {
        "target_utility": True,
        "target_mixer": True,
        "soft_update_tau": 0.005,
        "same_cadence_as_C1": True,
        "regional_local_target_uses": "target_utility",
        "joint_target_uses": "target_utility + target_mixer",
    }


def build_replay_preservation():
    return {
        "sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "each_anchor_contains": [
            "focal_state", "focal_action", "focal_R1_reward",
            "focal_next_state", "joint_event_context",
        ],
        "no_new_replay_information_needed": True,
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
    }


def build_sibling_semantics():
    return {
        "siblings_deduplicated": False,
        "C1_sibling_supervision": "[L_joint, L_joint, ..., L_joint] (approximately identical)",
        "C2_sibling_supervision": "[mean(L_joint, L_regional_1), mean(L_joint, L_regional_2), ..., "
                                  "mean(L_joint, L_regional_k)]",
        "joint_term_shared": True,
        "regional_term_differs_by_focal_region": True,
        "C2_does_NOT_eliminate_repeated_joint_targets": True,
        "C2_restores_differentiated_regional_credit": True,
    }


def build_seed_contract():
    prior_env = set()
    prior_env.update(range(1001, 1141))
    prior_env.update(range(1201, 1221))
    prior_env.update(range(1301, 1441))
    prior_env.update(range(1501, 1641))
    prior_env.update(range(1701, 1721))
    prior_env.update(range(1901, 2041))
    prior_policy = {1001, 1002, 1003, 1004, 1005, 3001, 3002, 3003, 3004, 3005,
                    4001, 4002, 4003, 4004, 4005, 5001, 5002, 5003, 5004, 5005}

    proposed_train = set(range(2101, 2161))
    proposed_val = set(range(2161, 2181))
    proposed_final = set(range(2181, 2241))
    proposed_policy = {6001, 6002, 6003, 6004, 6005}
    proposed_all_env = proposed_train | proposed_val | proposed_final

    overlap_env = prior_env & proposed_all_env
    overlap_pol = prior_policy & proposed_policy

    return {
        "FUTURE_TRAIN_SEEDS": "2101..2160",
        "FUTURE_VALIDATION_SEEDS": "2161..2180",
        "FUTURE_FINAL_SEEDS": "2181..2240",
        "FUTURE_POLICY_SEEDS": [6001, 6002, 6003, 6004, 6005],
        "train_count": 60,
        "validation_count": 20,
        "final_count": 60,
        "policy_count": 5,
        "env_overlap_with_prior": sorted(overlap_env),
        "policy_overlap_with_prior": sorted(overlap_pol),
        "SEED_DISJOINTNESS": "PASS" if not overlap_env and not overlap_pol else "FAIL",
        "prior_env_ranges": ["1001-1140", "1201-1220", "1301-1440", "1501-1640", "1701-1720", "1901-2040"],
        "prior_policy_ranges": ["1001-1005", "3001-3005", "4001-4005", "5001-5005"],
    }


def build_future_ab_factor():
    return {
        "FUTURE_EXPERIMENT_FACTOR": "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX",
        "FUTURE_CONTROL": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "FUTURE_TREATMENT": "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "scientific_difference": {
            "C1": "JOINT_ONLY_QMIX_TD",
            "C2": "EQUAL_MEAN_OF_JOINT_AND_FOCAL_REGIONAL_TD",
        },
        "everything_else_identical": True,
    }


def build_future_pairing():
    return {
        "policy_seeds": [6001, 6002, 6003, 6004, 6005],
        "at_t0_require": {
            "HASH_C1_utility_online_eq_HASH_C2_utility_online": True,
            "HASH_C1_mixer_online_eq_HASH_C2_mixer_online": True,
            "online_utility_eq_target_utility": True,
            "online_mixer_eq_target_mixer": True,
        },
        "identical_architectures": True,
    }


def build_future_training_budget():
    return {
        "epochs_per_run": 6,
        "episodes_per_epoch": 180,
        "episodes_per_run": 1080,
        "arms": 2,
        "policy_seeds": 5,
        "total_runs": 10,
        "total_training_episodes": 10800,
        "no_extra_training_for_C2": True,
    }


def build_future_checkpoint_contract():
    return {
        "checkpoint_schedule": [10000, 20000, 30000, 40000, 50000, "final"],
        "checkpoints_per_run": 6,
        "total_runs": 10,
        "total_checkpoints": 60,
        "selection_rule": "minimum_validation_MACRO_MIGD_RESPONSE",
        "scenario_weight": "1/3_each",
        "tie_1": "macro_MIGD_epoch",
        "tie_2": "macro_response_FE",
        "tie_3": "earlier_checkpoint",
        "no_selection_from": [
            "local_loss", "joint_loss", "sibling_dispersion",
            "MIXED_only", "PERSISTENT_only",
        ],
    }


def build_future_primary():
    return {
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "EFFECT_DEFINITION": "C1_MINUS_C2",
        "POSITIVE_MEANS": "CREDIT_PRESERVING_C2_BETTER",
        "PRACTICAL_MARGIN": 1e-5,
        "WHY_C1_MINUS_C2": "lower MIGD is better; positive C1-C2 means C2 has lower MIGD",
    }


def build_future_statistical_contract():
    return {
        "method": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis": 5,
        "environment_axis": 60,
        "resamples": 50000,
        "CI": "95_percentile",
        "BOOTSTRAP_MASTER_SEED": 106001,
        "effect": "C1_MINUS_C2",
        "positive_means": "C2_BETTER",
    }


def build_future_safety_contract():
    return {
        "PERSISTENT": "neither MIGD_response nor MIGD_epoch HARMFUL",
        "STATIONARY": "neither MIGD_response nor MIGD_epoch HARMFUL",
        "MACRO": "equal 1/3 scenarios, neither response nor epoch HARMFUL",
        "POLICY_ROBUSTNESS": "at_least_4_of_5_policies_positive_effect",
    }


def build_future_success_logic():
    return {
        "CREDIT_PRESERVATION_CAUSAL_VALIDATED": {
            "requires": [
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
            "PERFORMANCE_PARTIAL": "directional_positive + safety/integrity pass, "
                                   "OR beneficial but robustness fails with safety pass",
            "PRACTICALLY_NEUTRAL": "practically_neutral with safety/integrity pass",
        },
        "FAILED": {
            "conditions": [
                "primary directional_negative",
                "primary harmful",
                "any safety fail",
                "loss_spec integrity fail",
                "checkpoint/protocol integrity fail",
            ],
            "FE_cannot_rescue": True,
            "loss_diagnostics_cannot_rescue": True,
        },
    }


def build_credit_diagnostics_contract():
    return {
        "A": "sibling_composite_loss_dispersion",
        "B": "focal_regional_TD_loss_dispersion",
        "C": "joint_TD_loss_dispersion",
        "D": "joint_vs_regional_gradient_norm_ratio",
        "E": "joint_vs_regional_gradient_cosine",
        "F": "action_diversity",
        "G": "Q_margin_if_future_logging_permits",
        "diagnostic_role": "DESCRIPTIVE_MECHANISTIC",
        "no_post_hoc_performance_mechanism_gate": True,
    }


def build_e4mi_validation_contract():
    return {
        "required_validations": [
            "C1_implementation_unchanged",
            "C2_architecture_identical_to_C1",
            "utility_forward_identical_at_t0",
            "mixer_forward_identical_at_t0",
            "exact_existing_L_joint",
            "exact_existing_focal_regional_DDQN_target",
            "exact_regional_SmoothL1",
            "composite_0.5_joint_plus_0.5_regional",
            "local_loss_does_not_depend_on_mixer",
            "joint_loss_still_depends_on_mixer",
            "utility_receives_both_gradient_paths",
            "mixer_receives_joint_path_only",
            "target_utility_mixer_no_gradient",
            "one_optimizer_step",
            "no_extra_network",
            "no_change_replay_schema",
            "no_sibling_dedup",
            "no_team_reward_change",
            "no_V3_change",
            "no_R1_change",
            "no_alpha_change",
            "no_action_path_change",
            "no_extra_FE",
            "deterministic_paired_C1_C2_initialization",
        ],
        "synthetic_fixtures": {
            "A": "two sibling anchors, same joint event, different focal regional rewards → "
                 "C1 L_joint identical but C2 composite losses differ if local TD errors differ",
            "B": "identical local TD errors → sibling C2 composite losses equal",
            "C": "joint_loss=2, local_loss=4 → composite=3",
            "D": "joint_loss=0, local_loss>0 → utility receives regional gradient, "
                 "mixer receives zero/none from local component",
            "E": "local_loss=0, joint_loss>0 → existing QMIX gradient path unchanged",
        },
    }


def build_future_firewall():
    return {
        "M_TRAIN_ACCESS_COUNT": 0,
        "M_VALIDATION_ACCESS_COUNT": 0,
        "M_FINAL_ACCESS_COUNT": 0,
        "no_scientific_environment_execution": True,
    }


# =====================================================================
# Report
# =====================================================================
def write_report(verdict):
    L = [
        "# E4M — Joint Credit Preservation Design", "",
        "## A. Immutable E4KF / E4L evidence",
        "- E4KF verdict = FAILED (immutable)",
        "- E4L PRIMARY_FAILURE_HYPOTHESIS = MULTIFACTORIAL_OR_UNRESOLVED",
        "- H4 SIBLING_TARGET_REDUNDANCY = SUPPORTED (sibling redundancy ratio 10.0, duplicate mass 0.90)",
        "- E4IF R1 reward = VALIDATED; E4HF normalization = FAILED; E4JF global comparator = FAILED", "",
        "## B. Why H4 is the next isolated target",
        "- H4 is SUPPORTED and directly testable with a small isolated revision",
        "- Each joint event with k active regions produces k replay anchors sharing identical joint TD target",
        "- Region-specific R1 supervision (causally validated E4IF) is not directly used as TD target in C1",
        "- Restoring focal regional TD tests whether differentiated credit improves the failed learner", "",
        "## C. Why H2/H3/H5 are not changed",
        "- H2 (joint reward aggregation) = WEAK → do not alter team reward",
        "- H3 (event weighting distortion) = WEAK → do not change replay sampling",
        "- H5 (mixer degeneracy) = NOT_SUPPORTED → do not change mixer", "",
        "## D. Candidate revision review",
        "- M0: Original QMIX C1 → future control",
        "- M1: QMIX + regional auxiliary TD → SELECTED (conditions met)",
        "- M2: Event-deduplicated QMIX → rejected (H3 WEAK)",
        "- M3: Modified team reward → rejected (H2 WEAK)",
        "- M4: Non-monotonic critic → rejected (H6 only PLAUSIBLE)", "",
        "## E. Selected credit-preservation revision",
        "- M1_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "- Factor: REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX", "",
        "## F. Exact C1 control",
        "- C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 (unchanged from E4KF)", "",
        "## G. Exact C2 treatment",
        "- C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1",
        "- Identical to C1 EXCEPT: each sampled anchor receives BOTH joint QMIX TD AND focal-region R1 DDQN TD", "",
        "## H. Existing joint loss preservation",
        "- L_joint = SmoothL1(Q_tot_online, y_joint) — exact C1 definition, unchanged", "",
        "## I. Focal regional R1 Double-DQN target",
        "- y_local = r_r^R1 + gamma*(1-done)*Q_target(s'_r, argmax_a Q_online(s'_r, a))",
        "- L_regional = SmoothL1(Q_online(s_r, a_r), y_local)",
        "- Identical to C0 regional TD (E4KF control)", "",
        "## J. Equal-component composite loss",
        "- L_C2 = 0.5*L_joint + 0.5*L_regional",
        "- NOT tunable; equal arithmetic mean of two frozen components", "",
        "## K. Gradient paths",
        "- Online utility ← L_joint + L_regional",
        "- Online mixer ← L_joint only",
        "- L_regional does NOT depend on mixer parameters", "",
        "## L. Replay and sibling semantics",
        "- REGIONAL_ANCHOR_WITH_JOINT_CONTEXT (unchanged)",
        "- Siblings NOT deduplicated",
        "- C1 siblings: [L_joint, L_joint, ..., L_joint] ≈ identical",
        "- C2 siblings: [mean(L_joint, L_regional_1), ..., mean(L_joint, L_regional_k)] — differ by region", "",
        "## M. Joint reward preservation",
        "- r_joint = ACTIVE_REGION_MEAN_OF_R1 (unchanged)", "",
        "## N. Mixer / state preservation",
        "- QMIX_MIXER_V1, hidden 32, ABS monotonic, central state 55, R_MAX 10", "",
        "## O. Decentralized execution",
        "- argmax Q_online(s_r, a) per region; no mixer in action selection", "",
        "## P. Credit-preservation diagnostics",
        "- Sibling composite-loss dispersion, focal regional TD-loss dispersion",
        "- Joint vs regional gradient norm ratio and cosine similarity",
        "- Action diversity, Q margin if logging permits",
        "- All descriptive/mechanistic — no performance gate", "",
        "## Q. Future C1-vs-C2 causal experiment",
        "- Effect: C1 − C2 (positive = C2 better)",
        "- Primary: MIXED MIGD_response",
        "- Practical margin: 1e-5",
        "- Robustness: ≥4/5 policies positive",
        "- Safety: PERSISTENT/STATIONARY/MACRO all pass", "",
        "## R. Fresh-seed firewall",
        "- Train: 2101-2160, Val: 2161-2180, Final: 2181-2240",
        "- Policy: 6001-6005",
        "- All disjoint from prior; access counts = 0", "",
        "## S. Statistical design",
        "- Paired crossed two-axis bootstrap, 5×60, N=50000, seed 106001", "",
        "## T. Safety / robustness",
        "- Per-scenario safety (neither response nor epoch HARMFUL)",
        "- Policy robustness ≥4/5", "",
        "## U. E4MI implementation-validation contract",
        "- 24 required validations + 5 synthetic fixtures", "",
        "## V. Limitations",
        "- E4KF failure remains multifactorial/unresolved",
        "- H4 is supported but not established as the unique cause",
        "- C2 does not remove repeated joint targets",
        "- C2 only restores direct focal regional TD supervision",
        "- Equal joint/local loss weighting is a frozen design choice, not proven optimal",
        "- H1 and H10 remain supported competing explanations",
        "- H6 monotonic limitation remains plausible",
        "- No performance claim is made in E4M",
        "- No new environment was run",
        "- No real-world UAV validation", "",
        "## W. Next stage",
        f"- NEXT_STAGE = {verdict['NEXT_STAGE']}",
    ]
    (RESULTS_DIR / "E4M_report.md").write_text("\n".join(L))


# =====================================================================
# Main
# =====================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4M — Joint Credit Preservation Design")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0
    integrity = phase_0_historical_integrity()
    if not integrity["all_pass"]:
        _jdump({"E4M_COMPLETE": "NO", "NEXT_STAGE": "E4L_INTEGRITY_REVIEW",
                "integrity": integrity}, RESULTS_DIR / "E4M_verdict.json")
        print("[E4M-0] HISTORICAL INTEGRITY FAIL")
        return
    _jdump(integrity, RESULTS_DIR / "E4M_historical_integrity.json")
    print("[E4M-0] historical integrity PASS")

    # E4L evidence lock
    e4l = _load(E4L_DIR / "E4L_verdict.json")
    hyp = _load(E4L_DIR / "E4L_failure_hypothesis_summary.json")
    _jdump({
        "E4L_COMPLETE": e4l["E4L_COMPLETE"],
        "PRIMARY_FAILURE_HYPOTHESIS": hyp["PRIMARY_FAILURE_HYPOTHESIS"],
        "H4_STATUS": hyp["hypothesis_statuses"]["H4"],
        "H2_STATUS": hyp["hypothesis_statuses"]["H2"],
        "H3_STATUS": hyp["hypothesis_statuses"]["H3"],
        "H5_STATUS": hyp["hypothesis_statuses"]["H5"],
        "H6_STATUS": hyp["hypothesis_statuses"]["H6"],
        "SIBLING_REDUNDANCY_RATIO": e4l["SIBLING_REDUNDANCY_RATIO"],
    }, RESULTS_DIR / "E4M_e4l_evidence_lock.json")

    # Pre-manifest
    _jdump({
        "stage": "E4M_JOINT_CREDIT_PRESERVATION_DESIGN",
        "new_training": False,
        "new_environment_episodes": 0,
        "new_diagnostic_seeds": "NONE",
        "design_only": True,
    }, RESULTS_DIR / "E4M_pre_manifest.json")

    # Phase 1-2
    candidates = phase_1_candidate_review()
    _jdump(candidates, RESULTS_DIR / "E4M_candidate_revision_review.json")
    selection = phase_2_selection(candidates)
    _jdump(selection, RESULTS_DIR / "E4M_revision_selection.json")
    print(f"[E4M-2] SELECTED_REVISION = {selection['SELECTED_REVISION']}")

    # Contracts
    control = build_control_contract()
    _jdump(control, RESULTS_DIR / "E4M_control_contract.json")
    treatment = build_treatment_contract()
    _jdump(treatment, RESULTS_DIR / "E4M_treatment_contract.json")
    _jdump(build_joint_loss_preservation(), RESULTS_DIR / "E4M_joint_loss_preservation.json")
    _jdump(build_regional_target_contract(), RESULTS_DIR / "E4M_regional_target_contract.json")
    composite = build_composite_loss_contract()
    _jdump(composite, RESULTS_DIR / "E4M_composite_loss_contract.json")
    _jdump(build_gradient_path_contract(), RESULTS_DIR / "E4M_gradient_path_contract.json")
    _jdump(build_optimizer_contract(), RESULTS_DIR / "E4M_optimizer_contract.json")
    _jdump(build_target_network_contract(), RESULTS_DIR / "E4M_target_network_contract.json")
    _jdump(build_replay_preservation(), RESULTS_DIR / "E4M_replay_preservation.json")
    _jdump(build_sibling_semantics(), RESULTS_DIR / "E4M_sibling_semantics.json")

    # Preservation locks
    _jdump({"reward": "REWARD_R1_REGIONAL_QUALITY_GAIN", "unchanged": True,
            "alpha_local": 0.1, "no_new_reward": True},
           RESULTS_DIR / "E4M_reward_preservation.json")
    _jdump({"normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3", "unchanged": True},
           RESULTS_DIR / "E4M_normalization_preservation.json")
    _jdump({"mixer": "QMIX_MIXER_V1", "hidden_dim": 32, "central_state_dim": 55,
            "R_MAX": 10, "monotonic": True, "unchanged": True},
           RESULTS_DIR / "E4M_mixer_preservation.json")
    _jdump({"action_selection": "ARGMAX_REGIONAL_UTILITY", "mixer_in_action_selection": False,
            "decentralized_execution": True, "unchanged": True},
           RESULTS_DIR / "E4M_action_execution_contract.json")
    _jdump({"EXTRA_OBJECTIVE_FE": 0, "reason": "regional auxiliary target uses data already in replay anchor"},
           RESULTS_DIR / "E4M_extra_fe_contract.json")

    # Future experiment
    factor = build_future_ab_factor()
    _jdump(factor, RESULTS_DIR / "E4M_future_ab_factor_contract.json")
    seeds = build_seed_contract()
    _jdump(seeds, RESULTS_DIR / "E4M_future_seed_contract.json")
    _jdump({"SEED_DISJOINTNESS": seeds["SEED_DISJOINTNESS"],
            "env_overlap": seeds["env_overlap_with_prior"],
            "policy_overlap": seeds["policy_overlap_with_prior"]},
           RESULTS_DIR / "E4M_seed_disjointness.json")
    print(f"[E4M-19] seed disjointness = {seeds['SEED_DISJOINTNESS']}")

    _jdump(build_future_pairing(), RESULTS_DIR / "E4M_future_pairing_contract.json")
    _jdump(build_future_training_budget(), RESULTS_DIR / "E4M_future_training_budget.json")
    _jdump(build_future_checkpoint_contract(), RESULTS_DIR / "E4M_future_checkpoint_contract.json")
    _jdump(build_future_primary(), RESULTS_DIR / "E4M_future_primary_hypothesis.json")
    _jdump(build_future_statistical_contract(), RESULTS_DIR / "E4M_future_statistical_contract.json")
    _jdump(build_future_safety_contract(), RESULTS_DIR / "E4M_future_safety_contract.json")
    _jdump(build_future_success_logic(), RESULTS_DIR / "E4M_future_success_logic_skeleton.json")
    _jdump(build_credit_diagnostics_contract(), RESULTS_DIR / "E4M_credit_diagnostics_contract.json")
    _jdump(build_e4mi_validation_contract(), RESULTS_DIR / "E4M_e4mi_validation_contract.json")
    _jdump(build_future_firewall(), RESULTS_DIR / "E4M_future_firewall.json")

    # Protocol frozen
    _jdump({
        "SELECTED_REVISION": selection["SELECTED_REVISION"],
        "FUTURE_CONTROL": factor["FUTURE_CONTROL"],
        "FUTURE_TREATMENT": factor["FUTURE_TREATMENT"],
        "FUTURE_EXPERIMENT_FACTOR": factor["FUTURE_EXPERIMENT_FACTOR"],
        "LOSS_COMBINATION": composite["LOSS_COMBINATION"],
        "LOSS_FORMULA": composite["formula"],
        "FUTURE_PRIMARY": "MIXED_MIGD_RESPONSE",
        "FUTURE_EFFECT": "C1_MINUS_C2",
        "POSITIVE_MEANS": "CREDIT_PRESERVING_C2_BETTER",
        "PRACTICAL_MARGIN": 1e-5,
        "FUTURE_BOOTSTRAP_SEED": 106001,
        "SEED_DISJOINTNESS": seeds["SEED_DISJOINTNESS"],
    }, RESULTS_DIR / "E4M_protocol_frozen.json")

    # Verdict
    next_stage = "E4MI_JOINT_CREDIT_PRESERVATION_IMPLEMENTATION_VALIDATION"
    if seeds["SEED_DISJOINTNESS"] != "PASS":
        next_stage = "E4M_SEED_REVIEW"
    if not selection["all_selection_conditions_met"]:
        next_stage = "E4M_CREDIT_ARCHITECTURE_REVIEW"

    verdict = {
        "stage": "E4M_JOINT_CREDIT_PRESERVATION_DESIGN",
        "E4M_COMPLETE": "YES",
        "ORIGINAL_E4KF_VERDICT": "FAILED",
        "E4KF_VERDICT_CHANGED": "NO",
        "E4L_PRIMARY_FAILURE_HYPOTHESIS": "MULTIFACTORIAL_OR_UNRESOLVED",
        "H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION": "SUPPORTED",
        "H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT": "WEAK",
        "H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION": "WEAK",
        "H5_MIXER_DEGENERACY": "NOT_SUPPORTED",
        "H6_MONOTONIC_LIMITATION": "PLAUSIBLE",
        "SELECTED_REVISION": selection["SELECTED_REVISION"],
        "FUTURE_EXPERIMENT_FACTOR": factor["FUTURE_EXPERIMENT_FACTOR"],
        "FUTURE_CONTROL": factor["FUTURE_CONTROL"],
        "FUTURE_TREATMENT": factor["FUTURE_TREATMENT"],
        "FIXED_NORMALIZATION": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "FIXED_REWARD": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "FIXED_ALPHA_LOCAL": 0.1,
        "UTILITY_NETWORK": "9-128-128-4",
        "QMIX_MIXER": "QMIX_MIXER_V1",
        "MIXER_HIDDEN_DIM": 32,
        "JOINT_REWARD": "ACTIVE_REGION_MEAN_OF_R1",
        "REPLAY_SAMPLING_UNIT": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "SIBLING_ANCHOR_DEDUPLICATION": "NO",
        "CONTROL_LOSS": "JOINT_QMIX_SMOOTHL1",
        "TREATMENT_JOINT_LOSS": "JOINT_QMIX_SMOOTHL1",
        "TREATMENT_REGIONAL_LOSS": "FOCAL_R1_DOUBLE_DQN_SMOOTHL1",
        "TREATMENT_COMPOSITE_LOSS": "EQUAL_COMPONENT_MEAN",
        "LOSS_FORMULA": "0.5*L_joint + 0.5*L_regional",
        "AUXILIARY_WEIGHT_TUNING": "NO",
        "EXTRA_OBJECTIVE_FE": 0,
        "FUTURE_TRAIN_SEEDS": "2101..2160",
        "FUTURE_VALIDATION_SEEDS": "2161..2180",
        "FUTURE_FINAL_SEEDS": "2181..2240",
        "FUTURE_POLICY_SEEDS": [6001, 6002, 6003, 6004, 6005],
        "SEED_DISJOINTNESS": seeds["SEED_DISJOINTNESS"],
        "FUTURE_PRIMARY": "MIXED_MIGD_RESPONSE",
        "FUTURE_EFFECT": "C1_MINUS_C2",
        "POSITIVE_MEANS": "CREDIT_PRESERVING_C2_BETTER",
        "PRACTICAL_MARGIN": 1e-5,
        "FUTURE_POLICY_ROBUSTNESS": "AT_LEAST_4_OF_5_C2_BETTER",
        "FUTURE_BOOTSTRAP": "PAIRED_CROSSED_TWO_AXIS_5x60",
        "FUTURE_BOOTSTRAP_MASTER_SEED": 106001,
        "M_TRAIN_ACCESS_COUNT": 0,
        "M_VALIDATION_ACCESS_COUNT": 0,
        "M_FINAL_ACCESS_COUNT": 0,
        "NEW_ENVIRONMENT_EXECUTION": "NO",
        "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }
    _jdump(verdict, RESULTS_DIR / "E4M_verdict.json")
    write_report(verdict)

    # Terminal
    _emit_terminal(verdict)
    print(f"\n[E4M] wall time {time.time()-t0:.1f}s")
    return verdict


def _emit_terminal(v):
    print()
    fields = [
        "E4M_COMPLETE", "ORIGINAL_E4KF_VERDICT", "E4KF_VERDICT_CHANGED",
        "E4L_PRIMARY_FAILURE_HYPOTHESIS",
        "H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
        "H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT",
        "H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION",
        "H5_MIXER_DEGENERACY", "H6_MONOTONIC_LIMITATION",
        "SELECTED_REVISION", "FUTURE_EXPERIMENT_FACTOR",
        "FUTURE_CONTROL", "FUTURE_TREATMENT",
        "FIXED_NORMALIZATION", "FIXED_REWARD", "FIXED_ALPHA_LOCAL",
        "UTILITY_NETWORK", "QMIX_MIXER", "MIXER_HIDDEN_DIM",
        "JOINT_REWARD", "REPLAY_SAMPLING_UNIT", "SIBLING_ANCHOR_DEDUPLICATION",
        "CONTROL_LOSS", "TREATMENT_JOINT_LOSS", "TREATMENT_REGIONAL_LOSS",
        "TREATMENT_COMPOSITE_LOSS", "LOSS_FORMULA", "AUXILIARY_WEIGHT_TUNING",
        "EXTRA_OBJECTIVE_FE",
        "FUTURE_TRAIN_SEEDS", "FUTURE_VALIDATION_SEEDS", "FUTURE_FINAL_SEEDS",
        "FUTURE_POLICY_SEEDS",
        "FUTURE_PRIMARY", "FUTURE_EFFECT", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
        "FUTURE_POLICY_ROBUSTNESS", "FUTURE_BOOTSTRAP", "FUTURE_BOOTSTRAP_MASTER_SEED",
        "M_TRAIN_ACCESS_COUNT", "M_VALIDATION_ACCESS_COUNT", "M_FINAL_ACCESS_COUNT",
        "NEW_ENVIRONMENT_EXECUTION", "NEW_TRAINING",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in fields:
        val = v.get(k, "?")
        if isinstance(val, float):
            val = f"{val:.6g}"
        elif isinstance(val, list):
            val = str(val)
        print(f"{k:<55} = {val}")


if __name__ == "__main__":
    main()
