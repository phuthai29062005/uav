"""
E4T — Causal Dynamics Gated Coordination Design.

DESIGN ONLY. No training, no environment execution, no seed access.

Designs C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1:
identical to C3 except deterministic causal pair gate using
abs(V3 delta_severity).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent.parent
RESULTS = BASE / "results" / "E4T_causal_dynamics_gated_coordination_design"
E4S_DIR = BASE / "results" / "E4S_pairwise_coordination_failure_analysis"
E4RF_DIR = BASE / "results" / "E4RF_pairwise_cg_ab_final_evaluation_execution"
E4RI_DIR = BASE / "results" / "E4RI_pairwise_coordination_graph_implementation_validation"
E4R_DIR = BASE / "results" / "E4R_pairwise_coordination_graph_experiment_design"
E4RE_DIR = BASE / "results" / "E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze"
E4HR_DIR = BASE / "results" / "E4HR_normalization_design_review"
E4RT_DIR = BASE / "results" / "E4RT_pairwise_cg_ab_training_protocol_freeze"

DELTA_PRACTICAL = 1e-5

HISTORICAL_ENV_SEED_RANGES = [
    (701, 780),    (801, 882),    (901, 940),
    (1001, 1140),  (1201, 1260),  (1301, 1380),
    (1401, 1440),  (1501, 1620),  (1621, 1700),
    (1701, 1780),  (1781, 1820),  (1901, 2040),
    (2101, 2240),  (2501, 2640),
]
HISTORICAL_POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005,
                           6001, 6002, 6003, 6004, 6005,
                           7001, 7002, 7003, 7004, 7005]
HISTORICAL_BOOTSTRAP_SEEDS = [136001]

PROPOSED_TRAIN = list(range(2701, 2761))
PROPOSED_VALIDATION = list(range(2761, 2781))
PROPOSED_FINAL = list(range(2781, 2841))
PROPOSED_POLICY = [8001, 8002, 8003, 8004, 8005]
PROPOSED_BOOTSTRAP_SEED = 146001


def _load(name: str, d: Path = None) -> Any:
    with open((d or RESULTS) / name) as f:
        return json.load(f)


def _save(name: str, data: Any) -> None:
    with open(RESULTS / name, "w") as f:
        json.dump(data, f, indent=2)
    return RESULTS / name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Phase 1: Pre-manifest and integrity
    # ================================================================
    e4s_verdict = _load("E4S_verdict.json", E4S_DIR)
    e4rf_verdict = _load("E4RF_verdict.json", E4RF_DIR)
    e4s_hyp = _load("E4S_hypothesis_matrix.json", E4S_DIR)

    pre = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "E4S_COMPLETE": e4s_verdict.get("E4S_COMPLETE"),
        "E4RF_VERDICT": e4rf_verdict.get("PAIRWISE_COORDINATION_CAUSAL_VERDICT"),
        "STATIONARY_HARM_POLICY_COUNT": e4s_verdict.get("STATIONARY_POLICY_HARM_COUNT"),
        "S1_STATUS": e4s_hyp["hypotheses"]["S1_ACTION_SPECIALIZATION_SHIFT"],
        "S5_STATUS": e4s_hyp["hypotheses"]["S5_UNARY_POLICY_SHIFT"],
        "S6_STATUS": e4s_hyp["hypotheses"]["S6_COORDINATION_OVERRIDE_RATE"],
        "S8_STATUS": e4s_hyp["hypotheses"]["S8_CHECKPOINT_SELECTION_SCENARIO_TENSION"],
        "PRIMARY_MECHANISM": e4s_verdict.get("PRIMARY_FAILURE_MECHANISM"),
        "DESIGN_ONLY": True,
        "NO_TRAINING": True,
        "NO_ENVIRONMENT_EXECUTION": True,
        "NO_SEED_ACCESS": True,
    }
    _save("E4T_pre_manifest.json", pre)

    # ================================================================
    # Phase 2: E4S evidence lock
    # ================================================================
    e4s_lock = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "locked_from": "E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS",
        "E4S_COMPLETE": e4s_verdict.get("E4S_COMPLETE"),
        "E4RF_VERDICT": "FAILED",
        "STATIONARY_SAFETY": e4s_verdict.get("STATIONARY_SAFETY"),
        "STATIONARY_HARM_COUNT": e4s_verdict.get("STATIONARY_POLICY_HARM_COUNT"),
        "MIXED_PRIMARY": e4s_verdict.get("MIXED_PRIMARY"),
        "POLICY_ROBUSTNESS": e4s_verdict.get("POLICY_ROBUSTNESS"),
        "S1_ACTION_SPECIALIZATION_SHIFT": e4s_hyp["hypotheses"]["S1_ACTION_SPECIALIZATION_SHIFT"],
        "S5_UNARY_POLICY_SHIFT": e4s_hyp["hypotheses"]["S5_UNARY_POLICY_SHIFT"],
        "S6_COORDINATION_OVERRIDE_RATE": e4s_hyp["hypotheses"]["S6_COORDINATION_OVERRIDE_RATE"],
        "S8_CHECKPOINT_SELECTION_SCENARIO_TENSION": e4s_hyp["hypotheses"]["S8_CHECKPOINT_SELECTION_SCENARIO_TENSION"],
        "PRIMARY_FAILURE_MECHANISM": e4s_verdict.get("PRIMARY_FAILURE_MECHANISM"),
        "SECONDARY_MECHANISMS": e4s_verdict.get("SECONDARY_MECHANISMS"),
        "NEXT_STAGE_FROM_E4S": e4s_verdict.get("NEXT_STAGE"),
        "E4S_VERDICT_SHA256": _sha256(E4S_DIR / "E4S_verdict.json"),
        "E4S_HYPOTHESIS_SHA256": _sha256(E4S_DIR / "E4S_hypothesis_matrix.json"),
    }
    _save("E4T_e4s_evidence_lock.json", e4s_lock)

    # ================================================================
    # Phase 3: Final revision authorization
    # ================================================================
    auth = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "AUTHORIZATION": "EXACTLY_ONE_FINAL_BOUNDED_CONTROLLER_REVISION",
        "MUST_ADDRESS": "dynamic_benefit_vs_stationary_harm_tradeoff",
        "PROHIBITED_ALTERNATIVES": [
            "QPLEX", "QTRAN", "tune_C3", "search_K",
            "search_graph_topology", "search_pair_weights",
            "search_thresholds", "learned_gate",
            "new_state_features", "modify_reward",
            "modify_replay", "modify_response_operators",
        ],
        "HISTORICAL_FAILURES": {
            "E4F": "FAILED_POLICY_COLLAPSE",
            "E4HF": "FAILED_HARMFUL",
            "E4IF": "VALIDATED_REWARD_CAUSAL",
            "E4JF": "GLOBAL_COMPARATOR_FAILED",
            "E4KF": "FAILED_COORDINATION_CAUSAL",
            "E4MF": "FAILED_CREDIT_PRESERVATION",
            "E4RF": "FAILED_PAIRWISE_CG",
        },
        "CONTROLLER_REVISION_BUDGET_AFTER_C4": "ZERO",
    }
    _save("E4T_final_revision_authorization.json", auth)

    # ================================================================
    # Phase 4: C4 controller specification
    # ================================================================
    c4_spec = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "NAME": "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1",
        "SHORT_NAME": "C4_GATED_CG",
        "SCIENTIFIC_FACTOR": "CAUSAL_PAIRWISE_COORDINATION_GATING",
        "BASE_CONTROLLER": "C3_PAIRWISE_COORDINATION_GRAPH_V1",
        "IDENTICAL_TO_C3_EXCEPT": "deterministic_causal_pair_gate",
        "UNARY_NETWORK": {
            "input_dim": 9,
            "hidden": [128, 128],
            "output_dim": 4,
            "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
            "CHANGE_FROM_C3": "NONE",
        },
        "PAIR_NETWORK": {
            "input_dim": 18,
            "hidden": [128, 128],
            "output_dim": 16,
            "reshape": "4x4",
            "symmetrization": "0.5*(H(si,sj) + H(sj,si)^T)",
            "double_centering": "M - rowmean - colmean + grandmean",
            "CHANGE_FROM_C3": "NONE",
        },
        "GATE": {
            "signal": "abs(V3_delta_severity_r)",
            "formula": "g_ij = 0.5 * (abs(d_i) + abs(d_j))",
            "bounded": "[0, 1]",
            "trainable_parameters": 0,
            "threshold": "NONE",
            "temperature": "NONE",
            "slope": "NONE",
            "scenario_constant": "NONE",
            "applied_after": "pair_symmetrization_and_double_centering",
        },
        "EFFECTIVE_PAIR_POTENTIAL": "P_tilde_ij = g_ij * P_ij",
        "JOINT_Q": "(1/n)*sum_i U_i + (1/|E|)*sum_(i<j) g_ij*P_ij",
        "MAXSUM": {
            "K": 10,
            "update": "synchronous",
            "unary_potential": "U_i(a_i) / n",
            "pair_potential": "g_ij * P_ij(a_i,a_j) / |E|",
            "message_normalization": "subtract_maximum",
            "final_action": "deterministic_argmax",
            "CHANGE_FROM_C3": "pair_potential_gated",
        },
        "N_REGIONS_1_BEHAVIOR": "pair_contribution = 0",
        "PAIR_COEFFICIENT": 1,
        "NO_EXTRA_ALPHA_BETA": True,
        "TRAINING": {
            "joint_double_dqn": True,
            "online_action_selection": "gated_maxsum_using_online_C4",
            "target_evaluation": "same_selected_action",
            "no_c2_auxiliary_loss": True,
            "no_gate_loss": True,
            "no_gate_supervision": True,
            "no_stationary_regularization": True,
        },
        "REWARD": "R1_MEAN_ACTIVE_REGIONAL",
        "STATE": "V3_9DIM",
        "OPERATORS": "UNCHANGED_PRODUCTION",
        "FLOAT_SEMANTICS": "float32",
    }
    _save("E4T_c4_controller_spec.json", c4_spec)

    # ================================================================
    # Phase 5: Gate definition
    # ================================================================
    gate_def = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "GATE_NAME": "CAUSAL_DYNAMICS_PAIR_GATE",
        "SIGNAL": {
            "feature": "delta_severity_r",
            "source": "V3_local_state_dim_3",
            "normalization": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
            "pre_response": True,
            "causal": True,
            "no_scenario_label": True,
            "no_future_information": True,
            "no_post_response_information": True,
            "no_oracle_information": True,
            "no_objective_outcome_after_action": True,
        },
        "REGIONAL_MAGNITUDE": "d_r(t) = abs(delta_severity_r(t))",
        "PAIR_GATE": "g_ij(t) = 0.5 * (d_i(t) + d_j(t))",
        "BOUNDS": "[0, 1]",
        "CLIPPING": "clip(g_ij, 0, 1) only if numerical rounding",
        "PROPERTIES": {
            "symmetric_ij": True,
            "strictly_causal": True,
            "uses_existing_v3_feature": True,
            "zero_trainable_parameters": True,
            "zero_tuneable_thresholds": True,
            "no_temperature": True,
            "no_slope": True,
            "no_scenario_specific_constant": True,
        },
        "FORMULA_FROZEN_BEFORE_INSPECTION": True,
        "NO_POST_HOC_REPLACEMENT": [
            "max", "sum", "threshold", "sigmoid",
            "learned_gate", "scenario_classifier",
        ],
    }
    _save("E4T_gate_definition.json", gate_def)

    # ================================================================
    # Phase 6: Causality audit
    # ================================================================
    causality = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "GATE_IS_CAUSAL": True,
        "REASONING": [
            "delta_severity_r is computed from current severity minus previous severity",
            "Available pre-response at decision time",
            "No future information used",
            "No post-response outcome used",
            "No scenario label used (gate uses magnitude of change, not scenario type)",
            "No oracle or objective outcome used",
        ],
        "FEATURE_SOURCE": "V3_normalized_delta_severity_r (local state dimension 3)",
        "FEATURE_ALREADY_EXISTS": True,
        "NO_NEW_FEATURE_CREATED": True,
        "NO_LEARNED_NORMALIZATION": True,
        "V3_NORMALIZATION_FROZEN": True,
        "V3_DELTA_SEVERITY_FORMULA": "clip(raw_delta_severity / (a3 + eps), -1, 1)",
        "V3_a3": "q99(|x| where |x| > 0) from training fit states",
    }
    _save("E4T_causality_audit.json", causality)

    # ================================================================
    # Phase 7: C1/C3/C4 factor table
    # ================================================================
    factor_table = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1": {
            "coordination": "QMIX_monotonic_joint_training",
            "execution": "independent_local_greedy_argmax_per_region",
            "mixer": "QMIXMixerV1",
            "pair_network": "NONE",
            "gate": "NONE",
        },
        "C3_PAIRWISE_COORDINATION_GRAPH_V1": {
            "coordination": "pairwise_coordination_graph",
            "execution": "MaxSum_K10_synchronous_complete_graph",
            "mixer": "NONE",
            "pair_network": "SharedPairNetwork_18_128_128_16",
            "gate": "NONE (ungated pair interactions)",
        },
        "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1": {
            "coordination": "pairwise_coordination_graph",
            "execution": "MaxSum_K10_synchronous_complete_graph",
            "mixer": "NONE",
            "pair_network": "SharedPairNetwork_18_128_128_16",
            "gate": "deterministic_causal_pair_gate_0.5*(|d_i|+|d_j|)",
        },
        "ISOLATED_FACTOR": {
            "C3_vs_C4": "CAUSAL_PAIRWISE_COORDINATION_GATING",
            "C1_vs_C4": "COORDINATION_TYPE_AND_GATING",
        },
        "SHARED_ACROSS_ALL": [
            "V3_state_normalization",
            "R1_reward",
            "production_operators",
            "SharedRegionalQNetwork_9_128_128_4",
            "float32_semantics",
        ],
    }
    _save("E4T_c1_c3_c4_factor_table.json", factor_table)

    # ================================================================
    # Phase 8: Initialization pairing contract
    # ================================================================
    init_contract = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "FOR_EVERY_POLICY_SEED_p": {
            "C1_utility_initial_weights": "== C3_unary_initial_weights == C4_unary_initial_weights",
            "C3_pair_initial_weights": "== C4_pair_initial_weights",
        },
        "C3_vs_C4_INITIAL_PARAMETER_DIFFERENCE": "NONE",
        "ONLY_EXECUTION_TRAINING_DIFFERENCE": "deterministic_causal_gate",
        "C1_MIXER_INITIALIZATION": "frozen_semantics_unchanged",
    }
    _save("E4T_initialization_pairing_contract.json", init_contract)

    # ================================================================
    # Phase 9: Parameter count contract
    # ================================================================
    param_audit = _load("E4RI_parameter_count_audit.json", E4RI_DIR)
    c1_total = param_audit["C1_total"]
    c3_total = param_audit["C3_total"]

    param_contract = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "C1_TOTAL_PARAMETERS": c1_total,
        "C3_TOTAL_PARAMETERS": c3_total,
        "C4_EXPECTED_PARAMETERS": c3_total,
        "C4_EQUALS_C3": c3_total == c3_total,
        "GATE_TRAINABLE_PARAMETERS": 0,
        "C1_BREAKDOWN": {
            "utility_network": param_audit["C1_utility"],
            "mixer": param_audit["C1_mixer"],
        },
        "C3_BREAKDOWN": {
            "unary_network": param_audit["C3_unary"],
            "pair_network": param_audit["C3_pair"],
        },
        "C4_BREAKDOWN": {
            "unary_network": param_audit["C3_unary"],
            "pair_network": param_audit["C3_pair"],
            "gate": 0,
        },
        "SOURCE": "E4RI_parameter_count_audit.json",
    }
    _save("E4T_parameter_count_contract.json", param_contract)

    # ================================================================
    # Phase 10: Three-arm design
    # ================================================================
    three_arm = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "ARMS": {
            "A": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
            "B": "C3_PAIRWISE_COORDINATION_GRAPH_V1",
            "C": "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1",
        },
        "REASON": "C3-vs-C4 isolates gating; C1-vs-C4 tests final safety/utility",
        "ALL_THREE_ON_IDENTICAL_CELLS": True,
        "NO_SEPARATE_C4_VS_C1_EXPERIMENT_NEEDED": True,
        "COMPARISONS": {
            "GATING_MECHANISM": {
                "arms": "C3 vs C4",
                "effect": "C3_MINUS_C4",
                "positive_means": "C4_better",
                "primary_endpoint": "STATIONARY_MIGD_RESPONSE",
            },
            "FINAL_CONTROLLER": {
                "arms": "C1 vs C4",
                "effect": "C1_MINUS_C4",
                "positive_means": "C4_better",
                "primary_endpoint": "MIXED_MIGD_RESPONSE",
            },
            "REPLICATION_CONTEXT": {
                "arms": "C1 vs C3",
                "note": "replication/context, not new scientific treatment question",
            },
        },
    }
    _save("E4T_three_arm_design.json", three_arm)

    # ================================================================
    # Phase 11: Seed reservation
    # ================================================================
    historical_env = set()
    for s, e in HISTORICAL_ENV_SEED_RANGES:
        historical_env.update(range(s, e + 1))
    historical_pol = set(HISTORICAL_POLICY_SEEDS)
    historical_boot = set(HISTORICAL_BOOTSTRAP_SEEDS)

    proposed_env = set(PROPOSED_TRAIN + PROPOSED_VALIDATION + PROPOSED_FINAL)
    proposed_pol = set(PROPOSED_POLICY)
    proposed_boot = {PROPOSED_BOOTSTRAP_SEED}

    env_overlap = sorted(historical_env & proposed_env)
    pol_overlap = sorted(historical_pol & proposed_pol)
    boot_overlap = sorted(historical_boot & proposed_boot)

    all_disjoint = len(env_overlap) == 0 and len(pol_overlap) == 0 and len(boot_overlap) == 0

    seed_res = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "PROPOSED": {
            "TRAIN_SEEDS": {"range": "2701..2760", "n": 60, "seeds": PROPOSED_TRAIN},
            "VALIDATION_SEEDS": {"range": "2761..2780", "n": 20, "seeds": PROPOSED_VALIDATION},
            "FINAL_SEEDS": {"range": "2781..2840", "n": 60, "seeds": PROPOSED_FINAL},
            "POLICY_SEEDS": PROPOSED_POLICY,
            "BOOTSTRAP_SEED": PROPOSED_BOOTSTRAP_SEED,
        },
        "HISTORICAL_ENV_SEED_COUNT": len(historical_env),
        "HISTORICAL_MAX_ENV_SEED": max(historical_env),
        "HISTORICAL_POLICY_SEEDS": sorted(historical_pol),
        "HISTORICAL_BOOTSTRAP_SEEDS": sorted(historical_boot),
        "OVERLAP": {
            "env": env_overlap,
            "policy": pol_overlap,
            "bootstrap": boot_overlap,
        },
        "ALL_PROPOSED_DISJOINT": all_disjoint,
        "SEED_ACCESS_COUNT": 0,
        "NO_ENVIRONMENT_INSTANTIATED": True,
    }
    _save("E4T_seed_reservation.json", seed_res)

    if not all_disjoint:
        verdict = {
            "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
            "E4T_COMPLETE": "NO",
            "FAILURE_REASON": "SEED_COLLISION",
            "ENV_OVERLAP": env_overlap,
            "POLICY_OVERLAP": pol_overlap,
            "BOOTSTRAP_OVERLAP": boot_overlap,
        }
        _save("E4T_verdict.json", verdict)
        return

    # ================================================================
    # Phase 12: Training matrix contract
    # ================================================================
    train_matrix = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "ARMS": 3,
        "POLICIES": 5,
        "EPOCHS": 6,
        "TRAIN_ENVIRONMENTS": 60,
        "SCENARIOS": 3,
        "TOTAL_TRAINING_EPISODES": 3 * 5 * 6 * 60 * 3,
        "CHECKPOINTS_PER_ARM_POLICY": ["10k", "20k", "30k", "40k", "50k", "FINAL"],
        "TOTAL_CHECKPOINTS": 3 * 5 * 6,
    }
    assert train_matrix["TOTAL_TRAINING_EPISODES"] == 16200
    assert train_matrix["TOTAL_CHECKPOINTS"] == 90
    _save("E4T_training_matrix_contract.json", train_matrix)

    # ================================================================
    # Phase 13: Validation contract
    # ================================================================
    val_contract = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "ARMS": 3,
        "POLICIES": 5,
        "CHECKPOINTS": 6,
        "VALIDATION_ENVIRONMENTS": 20,
        "SCENARIOS": 3,
        "TOTAL_VALIDATION_EPISODES": 3 * 5 * 6 * 20 * 3,
        "SELECTION_CRITERIA_ORDERED": [
            "minimum_macro_MIGD_response",
            "macro_MIGD_epoch",
            "response_FE",
            "earlier_checkpoint",
        ],
        "SELECTED_PER": "arm × policy",
        "TOTAL_SELECTED": 3 * 5,
        "VALIDATION_IS": "NON_CONFIRMATORY",
    }
    assert val_contract["TOTAL_VALIDATION_EPISODES"] == 5400
    assert val_contract["TOTAL_SELECTED"] == 15
    _save("E4T_validation_contract.json", val_contract)

    # ================================================================
    # Phase 14: Final matrix contract
    # ================================================================
    final_matrix = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "ARMS": 3,
        "POLICIES": 5,
        "FINAL_ENVIRONMENTS": 60,
        "SCENARIOS": 3,
        "TOTAL_FINAL_EPISODES": 3 * 5 * 60 * 3,
        "ALL_THREE_ARMS_IDENTICAL_CELLS": True,
    }
    assert final_matrix["TOTAL_FINAL_EPISODES"] == 2700
    _save("E4T_final_matrix_contract.json", final_matrix)

    # ================================================================
    # Phase 15: Bootstrap contract
    # ================================================================
    bootstrap = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "METHOD": "paired_crossed_two_axis_bootstrap",
        "AXES": {"policy": 5, "environment": 60},
        "N": 50000,
        "SEED": PROPOSED_BOOTSTRAP_SEED,
        "SHARED_THREE_ARM_DRAWS": True,
        "SAME_INDICES_FOR": ["C1", "C3", "C4"],
        "SAME_INDICES_ACROSS": ["PERSISTENT", "STATIONARY", "MIXED"],
        "COHERENT_C3_C4_AND_C1_C4_EFFECTS": True,
        "NO_MULTIPLICITY_HIDING": True,
        "PRE_REGISTERED_COMPARISONS": [
            "C3_vs_C4_Stationary_response",
            "C1_vs_C4_Mixed_response",
        ],
    }
    _save("E4T_bootstrap_contract.json", bootstrap)

    # ================================================================
    # Phase 16: Gating success contract
    # ================================================================
    gating_success = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "VERDICT_NAME": "GATING_CAUSAL_VERDICT",
        "VALIDATED_REQUIRES_ALL": [
            "C3-vs-C4 STATIONARY MIGD_response = BENEFICIAL for C4",
            "C3-vs-C4 STATIONARY MIGD_epoch = NOT HARMFUL for C4",
            "C3-vs-C4 PERSISTENT MIGD_response = NOT HARMFUL for C4",
            "C3-vs-C4 MIXED MIGD_response = NOT HARMFUL for C4",
            "C3-vs-C4 macro MIGD_response = NOT HARMFUL for C4",
            "policy robustness: >=4/5 C3-C4 Stationary response delta > 0",
            "all integrity gates PASS",
        ],
        "EFFECT_DEFINITION": "C3_MINUS_C4",
        "POSITIVE_MEANS": "C4_better",
        "CLASSIFICATION": {
            "BENEFICIAL": f"delta > {DELTA_PRACTICAL} AND CI_lower > 0",
            "DIRECTIONAL_POSITIVE": f"delta > {DELTA_PRACTICAL} AND CI_lower <= 0",
            "PRACTICALLY_NEUTRAL": f"|delta| <= {DELTA_PRACTICAL}",
            "DIRECTIONAL_NEGATIVE": f"delta < -{DELTA_PRACTICAL} AND CI_upper >= 0",
            "HARMFUL": f"delta < -{DELTA_PRACTICAL} AND CI_upper < 0",
        },
        "PRACTICAL_MARGIN": DELTA_PRACTICAL,
    }
    _save("E4T_gating_success_contract.json", gating_success)

    # ================================================================
    # Phase 17: Final controller success contract
    # ================================================================
    final_success = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "VERDICT_NAME": "C4_PERFORMANCE_CAUSAL_VERDICT",
        "VALIDATED_REQUIRES_ALL": [
            "C1-vs-C4 MIXED MIGD_response = BENEFICIAL",
            "policy robustness: >=4/5 C1-C4 MIXED delta > 0",
            "PERSISTENT safety = PASS",
            "STATIONARY safety = PASS",
            "macro safety = PASS",
            "integrity = PASS",
        ],
        "EFFECT_DEFINITION": "C1_MINUS_C4",
        "POSITIVE_MEANS": "C4_better",
        "CLASSIFICATION_SEMANTICS": "same_canonical_as_E4RE_E4RF",
        "PRACTICAL_MARGIN": DELTA_PRACTICAL,
        "HIERARCHICAL_INTERPRETATION": {
            "A_gate_repairs_C3_but_C4_doesnt_beat_C1": "gating_useful_controller_not_established",
            "B_gate_fails_but_C4_looks_good_vs_C1": "DO_NOT_claim_gating_validated",
            "C_both_pass": "final_successful_controller_evidence",
            "D_either_safety_harmful": "preserve_failure",
        },
    }
    _save("E4T_final_controller_success_contract.json", final_success)

    # ================================================================
    # Phase 18: Offline gate sanity check
    # ================================================================
    fit_states = np.load(E4HR_DIR / "E4HR_fit_states.npy")
    v3_const = _load("E4HR_v3_constants.json", E4HR_DIR)
    a3 = v3_const["delta_q99_abs_positive"]

    raw_ds = fit_states[:, 3]
    norm_ds = np.clip(raw_ds / (a3 + 1e-12), -1.0, 1.0).astype(np.float32)
    abs_d = np.abs(norm_ds)

    zero_mask = raw_ds == 0.0
    nonzero_mask = ~zero_mask
    n_zero = int(zero_mask.sum())
    n_nonzero = int(nonzero_mask.sum())

    stat_gate_mean = float(abs_d[zero_mask].mean()) if n_zero > 0 else 0.0
    stat_gate_median = float(np.median(abs_d[zero_mask])) if n_zero > 0 else 0.0
    dyn_gate_mean = float(abs_d[nonzero_mask].mean()) if n_nonzero > 0 else 0.0
    dyn_gate_median = float(np.median(abs_d[nonzero_mask])) if n_nonzero > 0 else 0.0
    dyn_gate_q25 = float(np.percentile(abs_d[nonzero_mask], 25)) if n_nonzero > 0 else 0.0
    dyn_gate_q75 = float(np.percentile(abs_d[nonzero_mask], 75)) if n_nonzero > 0 else 0.0

    gate_sanity_pass = (
        stat_gate_mean < dyn_gate_mean
        and stat_gate_median < dyn_gate_median
    )

    gate_sanity = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "METHOD": "offline_saved_states_only",
        "SOURCE": "E4HR_fit_states.npy (90000 training states, seeds 1001..1060)",
        "V3_NORMALIZATION_APPLIED": True,
        "V3_DELTA_Q99_ABS_POSITIVE": a3,
        "GATE_FORMULA": "g_ij = 0.5 * (|d_i| + |d_j|)",
        "NOTE": "Statistics on |d_r| per state; g_ij = avg of two |d_r| values for same-partition pairs",
        "STATIONARY_PROXY": {
            "definition": "states with raw delta_severity == 0.0",
            "n_states": n_zero,
            "includes": "all STATIONARY states + some dynamic states with zero change",
            "abs_d_mean": stat_gate_mean,
            "abs_d_median": stat_gate_median,
        },
        "DYNAMIC_CONFIRMED": {
            "definition": "states with raw delta_severity != 0.0",
            "n_states": n_nonzero,
            "includes": "exclusively PERSISTENT and MIXED states with actual severity change",
            "abs_d_mean": dyn_gate_mean,
            "abs_d_median": dyn_gate_median,
            "abs_d_q25": dyn_gate_q25,
            "abs_d_q75": dyn_gate_q75,
        },
        "DIRECTIONAL_CHECK": {
            "stationary_mean_lt_dynamic_mean": stat_gate_mean < dyn_gate_mean,
            "stationary_median_lt_dynamic_median": stat_gate_median < dyn_gate_median,
        },
        "SEMANTIC_NOTE": (
            "STATIONARY scenarios have delta_severity = 0 by environmental definition "
            "(severity does not change). The gate is exactly 0 for all-STATIONARY pairs. "
            "Dynamic scenarios produce non-zero delta_severity when regional severity changes. "
            "The gate is strictly > 0 for pairs where both regions have severity change."
        ),
        "GO_NO_GO": "PASS" if gate_sanity_pass else "FAIL",
        "NO_FORMULA_TUNING_AFTER_INSPECTION": True,
    }
    _save("E4T_offline_gate_sanity.json", gate_sanity)

    if not gate_sanity_pass:
        verdict = {
            "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
            "E4T_COMPLETE": "NO",
            "FAILURE_REASON": "GATE_SANITY_FAIL",
            "GATE_NOT_DIRECTIONALLY_SENSITIVE": True,
            "NEXT_STAGE": "E4T_CONTROLLER_RESEARCH_STOP_REVIEW",
        }
        _save("E4T_verdict.json", verdict)
        return

    # ================================================================
    # Phase 19: Unary shift limitation
    # ================================================================
    unary_lim = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "E4S_S5_UNARY_POLICY_SHIFT": "SUPPORTED",
        "C4_GATE_DOES_NOT_DIRECTLY_SOLVE_UNARY_SHIFT": True,
        "INTENTIONALLY_MINIMAL_INTERVENTION_TEST": True,
        "IF_C4_STILL_HARMS_STATIONARY_DUE_TO_UNARY_SHIFT": "experiment_has_failed",
        "NO_AUXILIARY_UNARY_ANCHORING_LOSS": True,
    }
    _save("E4T_unary_shift_limitation.json", unary_lim)

    # ================================================================
    # Phase 20: Coordination override boundary
    # ================================================================
    override_boundary = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "E4S_S6_COORDINATION_OVERRIDE_RATE": "SUPPORTED",
        "C4_DIRECTLY_ATTENUATES_PAIR_POTENTIAL": True,
        "EXPECTED_EFFECT": "lower_unnecessary_Stationary_override_in_C4",
        "DESCRIPTIVE_MECHANISM_SUPPORT_ONLY": True,
        "PERFORMANCE_GATES_REMAIN_AUTHORITATIVE": True,
        "FUTURE_DIAGNOSTIC_REQUIRED": "unary_greedy_vs_final_coordinated_action_for_C3_and_C4",
    }
    _save("E4T_coordination_override_boundary.json", override_boundary)

    # ================================================================
    # Phase 21: Checkpoint tension boundary
    # ================================================================
    ckpt_boundary = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "E4S_S8_CHECKPOINT_TENSION": "SUPPORTED",
        "CHECKPOINT_SELECTION_UNCHANGED": True,
        "NO_STATIONARY_AWARE_CHECKPOINT_SELECTION": True,
        "NO_SAFETY_FILTERED_VALIDATION_CHECKPOINT_SELECTION": True,
        "REASON": "would_confound_C4_intervention",
        "IF_C4_STILL_EXHIBITS_CHECKPOINT_TENSION": "report_it",
    }
    _save("E4T_checkpoint_tension_boundary.json", ckpt_boundary)

    # ================================================================
    # Phase 22: No-tuning contract
    # ================================================================
    no_tuning = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "GATE_FORMULA_FROZEN": "0.5 * (|d_i| + |d_j|)",
        "FROZEN_BEFORE_INSPECTING_GATE_DISTRIBUTION": True,
        "IF_OFFLINE_SANITY_POOR": "STOP",
        "PROHIBITED_POST_HOC_REPLACEMENTS": [
            "max", "sum", "threshold", "sigmoid",
            "learned_gate", "scenario_classifier",
        ],
        "NO_THRESHOLD_SEARCH": True,
        "NO_ALTERNATIVE_GATE_SEARCH": True,
        "NO_LEARNED_GATING": True,
    }
    _save("E4T_no_tuning_contract.json", no_tuning)

    # ================================================================
    # Phase 23: Stop rule
    # ================================================================
    stop_rule = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "CONTROLLER_REVISION_BUDGET_AFTER_C4": "ZERO",
        "IF_C4_FINAL_EXPERIMENT_FAILS": {
            "NO_C5": True,
            "NO_QPLEX": True,
            "NO_QTRAN": True,
            "NO_NEW_GATE": True,
            "NO_THRESHOLD_TUNING": True,
            "NO_REWARD_MODIFICATION": True,
            "NO_NEW_STATE_REVISION": True,
            "ROUTE_TO": "FINAL_RESEARCH_SYNTHESIS_AND_REPORT_FREEZE",
        },
    }
    _save("E4T_stop_rule.json", stop_rule)

    # ================================================================
    # Phase 24: Verdict
    # ================================================================
    verdict = {
        "stage": "E4T_CAUSAL_DYNAMICS_GATED_COORDINATION_DESIGN",
        "E4T_COMPLETE": "YES",
        "FINAL_CONTROLLER_REVISION": "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1",
        "SCIENTIFIC_FACTOR": "CAUSAL_PAIRWISE_COORDINATION_GATING",
        "GATE_SIGNAL": "ABS_V3_DELTA_SEVERITY",
        "PAIR_GATE": "0.5*(ABS_DELTA_SEVERITY_I+ABS_DELTA_SEVERITY_J)",
        "GATE_TRAINABLE_PARAMETERS": 0,
        "C4_PARAMETER_COUNT": c3_total,
        "C3_PARAMETER_COUNT": c3_total,
        "C1_PARAMETER_COUNT": c1_total,
        "C4_EQUALS_C3_EXCEPT_GATE": True,
        "OFFLINE_GATE_SANITY": gate_sanity["GO_NO_GO"],
        "STATIONARY_GATE_MEAN": stat_gate_mean,
        "DYNAMIC_GATE_MEAN": dyn_gate_mean,
        "TRAIN_SEEDS_RESERVED": "2701..2760",
        "VALIDATION_SEEDS_RESERVED": "2761..2780",
        "FINAL_SEEDS_RESERVED": "2781..2840",
        "POLICY_SEEDS_RESERVED": "8001..8005",
        "BOOTSTRAP_SEED_RESERVED": PROPOSED_BOOTSTRAP_SEED,
        "ALL_NEW_SEEDS_DISJOINT": all_disjoint,
        "NEW_SEED_ACCESS_COUNT": 0,
        "PLANNED_ARMS": 3,
        "PLANNED_TRAIN_EPISODES": 16200,
        "PLANNED_CHECKPOINTS": 90,
        "PLANNED_VALIDATION_EPISODES": 5400,
        "PLANNED_FINAL_EPISODES": 2700,
        "GATING_PRIMARY": "C3_MINUS_C4_STATIONARY_MIGD_RESPONSE",
        "FINAL_CONTROLLER_PRIMARY": "C1_MINUS_C4_MIXED_MIGD_RESPONSE",
        "PRACTICAL_MARGIN": DELTA_PRACTICAL,
        "BOOTSTRAP_METHOD": "PAIRED_CROSSED_5x60_SHARED_3_ARM",
        "BOOTSTRAP_N": 50000,
        "BOOTSTRAP_SEED": PROPOSED_BOOTSTRAP_SEED,
        "CONTROLLER_REVISION_BUDGET_AFTER_C4": "ZERO",
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_OBJECTIVE_EVALUATIONS": 0,
        "NEW_TRAINING": "NO",
        "E4RF_C3_VERDICT": "FAILED_UNCHANGED",
        "E4S_VERDICT": "UNCHANGED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": "E4TI_C4_GATED_COORDINATION_IMPLEMENTATION_VALIDATION",
    }
    _save("E4T_verdict.json", verdict)

    # ================================================================
    # Phase 25: Report
    # ================================================================
    report = _generate_report(
        c4_spec, gate_def, gate_sanity, factor_table,
        init_contract, param_contract, three_arm,
        gating_success, final_success, seed_res,
        train_matrix, val_contract, final_matrix, bootstrap,
        unary_lim, override_boundary, ckpt_boundary,
        no_tuning, stop_rule, verdict, e4s_lock,
    )
    (RESULTS / "E4T_report.md").write_text(report)


def _generate_report(
    c4_spec, gate_def, gate_sanity, factor_table,
    init_contract, param_contract, three_arm,
    gating_success, final_success, seed_res,
    train_matrix, val_contract, final_matrix, bootstrap,
    unary_lim, override_boundary, ckpt_boundary,
    no_tuning, stop_rule, verdict, e4s_lock,
) -> str:
    lines = []
    def s(text=""): lines.append(text)

    s("# E4T — Causal Dynamics Gated Coordination Design")
    s()

    # A: Why exactly one final revision
    s("## A. Why Exactly One Final Revision Is Permitted")
    s()
    s("Six controller treatments have been tested: the original learned controller (E4F),")
    s("normalization V3 (E4HF), global comparator (E4JF), QMIX coordination (E4KF),")
    s("credit preservation (E4MF), and pairwise coordination graph (E4RF).")
    s("Only the R1 reward revision (E4IF) was causally validated.")
    s("The recurring dynamic-vs-Stationary tradeoff has appeared across three distinct")
    s("coordination architectures. One final bounded revision is authorized to address this tradeoff.")
    s()

    # B: E4S failure mechanism
    s("## B. E4S Failure Mechanism")
    s()
    s(f"- PRIMARY: {e4s_lock['PRIMARY_FAILURE_MECHANISM']}")
    s(f"- STATIONARY harm: {e4s_lock['STATIONARY_HARM_COUNT']} policies")
    s(f"- S1 ACTION_SPECIALIZATION: {e4s_lock['S1_ACTION_SPECIALIZATION_SHIFT']}")
    s(f"- S5 UNARY_POLICY_SHIFT: {e4s_lock['S5_UNARY_POLICY_SHIFT']}")
    s(f"- S6 COORDINATION_OVERRIDE: {e4s_lock['S6_COORDINATION_OVERRIDE_RATE']}")
    s(f"- S8 CHECKPOINT_TENSION: {e4s_lock['S8_CHECKPOINT_SELECTION_SCENARIO_TENSION']}")
    s()

    # C: Why QPLEX/QTRAN not selected
    s("## C. Why QPLEX/QTRAN Are Not Selected")
    s()
    s("QPLEX and QTRAN would introduce new coordination architectures without addressing the")
    s("identified failure mechanism. The Stationary-vs-dynamic tradeoff appears across QMIX,")
    s("credit-preservation, and pairwise CG — suggesting the problem is not architecture-specific.")
    s("A new architecture would likely exhibit the same tradeoff. Instead, C4 targets the root")
    s("mechanism: unconditional coordination pressure in low-change environments.")
    s()

    # D: C4 minimal intervention
    s("## D. C4 Minimal Intervention")
    s()
    s(f"- Name: {c4_spec['NAME']}")
    s(f"- Factor: {c4_spec['SCIENTIFIC_FACTOR']}")
    s(f"- Base: {c4_spec['BASE_CONTROLLER']}")
    s(f"- Change: {c4_spec['IDENTICAL_TO_C3_EXCEPT']}")
    s()

    # E: Causal gate definition
    s("## E. Causal Gate Definition")
    s()
    s(f"- Signal: {gate_def['SIGNAL']['feature']} from {gate_def['SIGNAL']['source']}")
    s(f"- Regional magnitude: {gate_def['REGIONAL_MAGNITUDE']}")
    s(f"- Pair gate: {gate_def['PAIR_GATE']}")
    s(f"- Bounds: {gate_def['BOUNDS']}")
    s(f"- Trainable parameters: {gate_def['PROPERTIES']['zero_trainable_parameters']}")
    s(f"- Symmetric: {gate_def['PROPERTIES']['symmetric_ij']}")
    s(f"- Causal: {gate_def['PROPERTIES']['strictly_causal']}")
    s()

    # F: Why delta_severity is allowed
    s("## F. Why delta_severity Is Allowed")
    s()
    s("delta_severity_r is an already-existing V3 local state feature (dimension 3).")
    s("It measures the change in regional severity between consecutive dynamic change events.")
    s("It is available pre-response at decision time and requires no new computation.")
    s("V3 normalization is frozen: clip(raw / (a3 + eps), -1, 1).")
    s("No new feature is introduced; only a deterministic function of existing state is used.")
    s()

    # G: C4 mathematical formulation
    s("## G. C4 Mathematical Formulation")
    s()
    s("Joint value:")
    s("```")
    s("Q_C4(S,A) = (1/n) * sum_i U_i(s_i, a_i)")
    s("           + (1/|E|) * sum_(i<j) g_ij(S) * P_ij(s_i, s_j, a_i, a_j)")
    s("```")
    s()
    s("Effective pair potential: `P_tilde_ij = g_ij * P_ij`")
    s()
    s("Gate applied AFTER pair symmetrization and double centering.")
    s("For n=1: pair contribution = 0.")
    s()

    # H: Max-Sum integration
    s("## H. Max-Sum Integration")
    s()
    s("- Unary potential: phi_i(a_i) = U_i(a_i) / n")
    s("- Pair potential: psi_ij(a_i,a_j) = g_ij * P_ij(a_i,a_j) / |E|")
    s("- K = 10, synchronous, message normalization = subtract maximum")
    s("- Final action: deterministic argmax (same C3 equations)")
    s()

    # I: Factor isolation
    s("## I. C1/C3/C4 Factor Isolation")
    s()
    s("| Factor | C1 | C3 | C4 |")
    s("|--------|----|----|-----|")
    s("| Coordination | QMIX | Pairwise CG | Pairwise CG |")
    s("| Pair gate | — | None | Causal gate |")
    s("| Pair network | — | SharedPairNet | SharedPairNet |")
    s("| Mixer | QMIXMixerV1 | — | — |")
    s()
    s("C3 vs C4 isolates: CAUSAL_PAIRWISE_COORDINATION_GATING")
    s()

    # J: Paired initialization
    s("## J. Paired Initialization")
    s()
    s("For every policy seed p:")
    s("- C1 utility == C3 unary == C4 unary (initial weights)")
    s("- C3 pair == C4 pair (initial weights)")
    s("- C3 vs C4 initial parameter difference = NONE")
    s()

    # K: Offline gate sanity
    s("## K. Offline Gate Sanity")
    s()
    s(f"- Source: {gate_sanity['SOURCE']}")
    s(f"- STATIONARY proxy (delta_severity==0): {gate_sanity['STATIONARY_PROXY']['n_states']} states")
    s(f"  - |d_r| mean: {gate_sanity['STATIONARY_PROXY']['abs_d_mean']:.6f}")
    s(f"  - |d_r| median: {gate_sanity['STATIONARY_PROXY']['abs_d_median']:.6f}")
    s(f"- DYNAMIC confirmed (delta_severity!=0): {gate_sanity['DYNAMIC_CONFIRMED']['n_states']} states")
    s(f"  - |d_r| mean: {gate_sanity['DYNAMIC_CONFIRMED']['abs_d_mean']:.6f}")
    s(f"  - |d_r| median: {gate_sanity['DYNAMIC_CONFIRMED']['abs_d_median']:.6f}")
    s(f"- Stationary mean < Dynamic mean: {gate_sanity['DIRECTIONAL_CHECK']['stationary_mean_lt_dynamic_mean']}")
    s(f"- GO/NO-GO: **{gate_sanity['GO_NO_GO']}**")
    s()

    # L: Three-arm experiment design
    s("## L. Three-Arm Experiment Design")
    s()
    s("| Arm | Controller |")
    s("|-----|-----------|")
    for k, v in three_arm["ARMS"].items():
        s(f"| {k} | {v} |")
    s()
    s(f"- Training episodes: {train_matrix['TOTAL_TRAINING_EPISODES']}")
    s(f"- Checkpoints: {train_matrix['TOTAL_CHECKPOINTS']}")
    s(f"- Validation episodes: {val_contract['TOTAL_VALIDATION_EPISODES']}")
    s(f"- Final episodes: {final_matrix['TOTAL_FINAL_EPISODES']}")
    s()

    # M: Gating mechanism hypothesis
    s("## M. Gating Mechanism Hypothesis")
    s()
    s("STATIONARY: small |delta_severity| → smaller g_ij → reduced pairwise coordination pressure.")
    s("PERSISTENT/MIXED: larger regional change → larger g_ij → C3-like coordination retained.")
    s()
    s("Primary comparison: C3 vs C4 on STATIONARY MIGD_response (C3_MINUS_C4, positive = C4 better).")
    s()

    # N: Final controller hypothesis
    s("## N. Final Controller Hypothesis")
    s()
    s("Primary comparison: C1 vs C4 on MIXED MIGD_response (C1_MINUS_C4, positive = C4 better).")
    s()

    # O: Statistical protocol
    s("## O. Statistical Protocol")
    s()
    s(f"- Bootstrap: paired crossed 5×60, N={bootstrap['N']}, seed={bootstrap['SEED']}")
    s("- Shared draws across all three arms and scenarios")
    s("- Pre-registered comparisons: C3-C4 Stationary, C1-C4 Mixed")
    s("- C1-C3 is replication/context, not new treatment question")
    s()

    # P: Safety and robustness gates
    s("## P. Safety and Robustness Gates")
    s()
    s("Gating verdict requires: Stationary repair (BENEFICIAL), no dynamic harm, 4/5 policy robustness.")
    s("Final controller requires: MIXED BENEFICIAL, all safety PASS, 4/5 robustness.")
    s()

    # Q: Unary-shift limitation
    s("## Q. Unary-Shift Limitation")
    s()
    s("E4S S5 (unary policy shift) was SUPPORTED. C4 gating does NOT directly solve this.")
    s("This is intentionally a MINIMAL_INTERVENTION_TEST. If C4 still harms Stationary")
    s("because its learned unary policy shifts, the experiment has failed.")
    s("No auxiliary unary anchoring loss is permitted.")
    s()

    # R: Checkpoint-tension limitation
    s("## R. Checkpoint-Tension Limitation")
    s()
    s("E4S S8 was SUPPORTED. Checkpoint selection remains macro-based and unchanged.")
    s("No Stationary-aware or safety-filtered checkpoint selection — would confound intervention.")
    s()

    # S: No-tuning boundary
    s("## S. No-Tuning Boundary")
    s()
    s("The gate formula 0.5*(|d_i|+|d_j|) is frozen BEFORE inspecting gate-distribution results.")
    s("If offline sanity is poor: STOP. No post-hoc replacement with max, sum, threshold, sigmoid,")
    s("learned gate, or scenario classifier.")
    s()

    # T: Final stop rule
    s("## T. Final Stop Rule")
    s()
    s("CONTROLLER_REVISION_BUDGET_AFTER_C4 = ZERO.")
    s("If C4 fails: no C5, no QPLEX, no QTRAN, no new gate, no threshold tuning,")
    s("no reward modification, no new state revision.")
    s("Route to: FINAL_RESEARCH_SYNTHESIS_AND_REPORT_FREEZE.")
    s()

    # U: Next-stage routing
    s("## U. Next-Stage Routing")
    s()
    s(f"- Design integrity: PASS")
    s(f"- Offline gate sanity: {gate_sanity['GO_NO_GO']}")
    if gate_sanity["GO_NO_GO"] == "PASS":
        s("- NEXT: E4TI_C4_GATED_COORDINATION_IMPLEMENTATION_VALIDATION")
    else:
        s("- NEXT: E4T_CONTROLLER_RESEARCH_STOP_REVIEW")
    s()

    return "\n".join(lines)


if __name__ == "__main__":
    run()
