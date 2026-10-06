"""E4O — Scenario Trade-off Diagnostic Design.

Design and freeze diagnostic experiment contracts to explain the systematic
scenario trade-off produced by C2 (credit-preserving QMIX). DESIGN/FREEZE ONLY.
No training, no environment execution, no seed opening.
"""

import json
import csv
import os
from pathlib import Path

STAGE = "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN"
OUT_DIR = "results/E4O_scenario_tradeoff_diagnostic_design"

E4N_DIR = "results/E4N_credit_preservation_failure_analysis"
E4MF_DIR = "results/E4MF_credit_preservation_ab_final_evaluation_execution"
E4MTR_DIR = "results/E4MTR_credit_preservation_ab_training_execution"
E4MI_DIR = "results/E4MI_joint_credit_preservation_implementation_validation"
E4MT_DIR = "results/E4MT_credit_preservation_ab_training_protocol_freeze"
E4ME_DIR = "results/E4ME_credit_preservation_ab_final_evaluation_protocol_freeze"

C1 = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2 = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
EPSILON = 1e-12
DIAGNOSTIC_SEEDS = list(range(2301, 2361))
DIAGNOSTIC_BOOTSTRAP_N = 50000
DIAGNOSTIC_BOOTSTRAP_SEED = 116001
STATE_DIM = 9
CENTRAL_STATE_DIM = 55
N_ACTIONS = 4


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # =================================================================
    # Phase 0 — Pre-manifest: verify all E4N artifacts exist
    # =================================================================
    e4n_files = [
        "E4N_historical_integrity.json",
        "E4N_e4mf_verdict_preservation.json",
        "E4N_scenario_effect_localization.csv",
        "E4N_cross_stage_scenario_tradeoff.csv",
        "E4N_gradient_conflict_analysis.csv",
        "E4N_loss_vs_gradient_scale.csv",
        "E4N_joint_supervision_scaling_analysis.csv",
        "E4N_action_specialization_analysis.csv",
        "E4N_q_margin_analysis.csv",
        "E4N_state_conditional_action_analysis.csv",
        "E4N_stationary_regional_reward_analysis.csv",
        "E4N_checkpoint_selection_tension.csv",
        "E4N_policy_seed_heterogeneity.csv",
        "E4N_validation_final_consistency.csv",
        "E4N_h4_targeted_revision_assessment.json",
        "E4N_monotonic_limitation_review.json",
        "E4N_hypothesis_evidence_table.csv",
        "E4N_failure_hypothesis_summary.json",
        "E4N_next_stage_decision.json",
        "E4N_verdict.json",
    ]
    pre_manifest = {}
    for fn in e4n_files:
        path = os.path.join(E4N_DIR, fn)
        pre_manifest[path] = "AVAILABLE" if os.path.isfile(path) else "MISSING"

    upstream_dirs = [E4MF_DIR, E4MTR_DIR, E4MI_DIR, E4MT_DIR, E4ME_DIR]
    for d in upstream_dirs:
        vf = os.path.join(d, os.path.basename(d).replace("results/", "").split("/")[0])
        prefix = d.split("/")[-1].split("_")[0]
        verdict_file = os.path.join(d, f"{prefix}_verdict.json")
        if os.path.isfile(verdict_file):
            pre_manifest[verdict_file] = "AVAILABLE"
        else:
            for fn in os.listdir(d) if os.path.isdir(d) else []:
                if fn.endswith("_verdict.json"):
                    pre_manifest[os.path.join(d, fn)] = "AVAILABLE"
                    break

    ckpt_file = os.path.join(E4MF_DIR, "E4MF_checkpoint_integrity.json")
    pre_manifest[ckpt_file] = "AVAILABLE" if os.path.isfile(ckpt_file) else "MISSING"

    all_available = all(v == "AVAILABLE" for v in pre_manifest.values())
    missing = [k for k, v in pre_manifest.items() if v != "AVAILABLE"]
    save_json(os.path.join(OUT_DIR, "E4O_pre_manifest.json"), {
        "stage": STAGE,
        "artifacts": pre_manifest,
        "all_available": all_available,
        "missing": missing,
    })

    # =================================================================
    # Phase E4O-0 — Historical integrity (17 checks)
    # =================================================================
    verdict = load_json(os.path.join(E4N_DIR, "E4N_verdict.json"))
    preservation = load_json(os.path.join(E4N_DIR, "E4N_e4mf_verdict_preservation.json"))
    historical = load_json(os.path.join(E4N_DIR, "E4N_historical_integrity.json"))

    checks = {}
    checks["C01_E4N_COMPLETE"] = verdict["E4N_COMPLETE"] == "YES"
    checks["C02_E4MF_VERDICT_FAILED"] = verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"
    checks["C03_MIXED_DIRECTIONAL_POSITIVE"] = verdict["E4MF_MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"
    checks["C04_ROBUSTNESS_FAIL"] = verdict["E4MF_POLICY_ROBUSTNESS"] == "FAIL_3_OF_5"
    checks["C05_PERSISTENT_SAFETY_PASS"] = verdict["E4MF_PERSISTENT_SAFETY"] == "PASS"
    checks["C06_STATIONARY_SAFETY_FAIL"] = verdict["E4MF_STATIONARY_SAFETY"] == "FAIL"
    checks["C07_MACRO_SAFETY_PASS"] = verdict["E4MF_MACRO_SAFETY"] == "PASS"
    checks["C08_E4KF_PERSISTENT_SAFETY_FAIL"] = historical["E4KF_PERSISTENT_SAFETY"] == "FAIL"
    checks["C09_E4KF_STATIONARY_SAFETY_PASS"] = historical["E4KF_STATIONARY_SAFETY"] == "PASS"
    checks["C10_N10_SUPPORTED"] = verdict["N10_MULTIFACTORIAL_SCENARIO_TRADEOFF"] == "SUPPORTED"
    checks["C11_N1_NOT_AUDITABLE"] = verdict["N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT"] == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
    checks["C12_N2_NOT_SUPPORTED"] = verdict["N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS"] == "NOT_SUPPORTED"
    checks["C13_N6_SUPPORTED"] = verdict["N6_CHECKPOINT_SELECTION_SCENARIO_TENSION"] == "SUPPORTED"
    checks["C14_N7_NOT_SUPPORTED"] = verdict["N7_POLICY_SEED_HETEROGENEITY"] == "NOT_SUPPORTED"
    checks["C15_N8_PLAUSIBLE"] = verdict["N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF"] == "PLAUSIBLE"
    checks["C16_N9_PLAUSIBLE"] = verdict["N9_MONOTONIC_QMIX_LIMITATION_REMAINS"] == "PLAUSIBLE"
    h4 = load_json(os.path.join(E4N_DIR, "E4N_h4_targeted_revision_assessment.json"))
    checks["C17_H4_REVISION_NOT_SUFFICIENT"] = h4.get("H4_TARGETED_REVISION_SUFFICIENT") == "NO" or h4.get("targeted_revision_sufficient") == "NO"

    all_pass = all(checks.values())
    save_json(os.path.join(OUT_DIR, "E4O_historical_integrity.json"), {
        "stage": STAGE,
        "checks": checks,
        "all_pass": all_pass,
        "total_checks": len(checks),
        "passed_checks": sum(checks.values()),
    })

    if not all_pass:
        save_json(os.path.join(OUT_DIR, "E4O_verdict.json"), {
            "stage": STAGE,
            "E4O_COMPLETE": "NO",
            "REASON": "HISTORICAL_INTEGRITY_FAILURE",
            "failed_checks": [k for k, v in checks.items() if not v],
            "NEXT_STAGE": "E4N_INTEGRITY_REVIEW",
        })
        return

    # =================================================================
    # Phase — E4N evidence lock
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_e4n_evidence_lock.json"), {
        "stage": STAGE,
        "E4N_PRIMARY_FAILURE_HYPOTHESIS": "MULTIFACTORIAL_SCENARIO_TRADEOFF",
        "E4N_CROSS_STAGE_PATTERN": "SCENARIO_TRADEOFF_SHIFT",
        "E4N_N1_STATUS": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
        "E4N_N2_STATUS": "NOT_SUPPORTED",
        "E4N_N3_STATUS": "PLAUSIBLE",
        "E4N_N4_STATUS": "PLAUSIBLE",
        "E4N_N5_STATUS": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
        "E4N_N6_STATUS": "SUPPORTED",
        "E4N_N7_STATUS": "NOT_SUPPORTED",
        "E4N_N8_STATUS": "PLAUSIBLE",
        "E4N_N9_STATUS": "PLAUSIBLE",
        "E4N_N10_STATUS": "SUPPORTED",
        "PERSISTENT_POLICY_PATTERN": "C2_BETTER_5_OF_5",
        "STATIONARY_POLICY_PATTERN": "C2_WORSE_5_OF_5",
        "REGIONAL_TO_JOINT_LOSS_RATIO": 0.0103,
        "REGIONAL_TO_JOINT_GRAD_NORM_RATIO": 0.0508,
        "locked": True,
    })

    # =================================================================
    # Phase — Diagnostic question
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_diagnostic_question.json"), {
        "stage": STAGE,
        "scientific_question": (
            "Why does the same frozen C2 credit-preserving specification "
            "produce opposite scenario effects? Are L_joint and L_regional "
            "more compatible under PERSISTENT/MIXED dynamics but more "
            "antagonistic under STATIONARY dynamics? Do those optimization "
            "differences correspond to scenario-specific changes in actions, "
            "utility Q margins, focal R1 signal, and TD targets?"
        ),
        "NOT_A_PERFORMANCE_TEST": True,
        "E4MF_REMAINS_AUTHORITATIVE": True,
        "E4O_CANNOT_RESCUE_E4MF": True,
    })

    # =================================================================
    # Phase — Hypothesis contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_hypothesis_contract.json"), {
        "stage": STAGE,
        "primary_diagnostic_mechanism": "O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT",
        "secondary_diagnostic_mechanism": "O2_SCENARIO_SPECIFIC_REGIONAL_CREDIT_STRUCTURE",
        "addresses_N1": True,
        "addresses_N3": True,
        "addresses_N4": True,
        "addresses_N5": True,
        "addresses_N8": True,
        "preserves_N2_NOT_SUPPORTED": True,
        "preserves_N6_SUPPORTED": True,
        "preserves_N7_NOT_SUPPORTED": True,
        "N9_H6_MONOTONICITY_CAUSAL_TEST": "NOT_INCLUDED",
        "hypothesis_may_update": ["N1", "N3", "N4", "N5", "N8"],
        "hypothesis_may_NOT_alter": "E4MF_VERDICT",
    })

    # =================================================================
    # Phase E4O-1 — Gradient alignment contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_gradient_alignment_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-1",
        "mechanism": "O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT",
        "definition": {
            "g_J": "gradient of UNWEIGHTED L_joint with respect to online utility parameters",
            "g_R": "gradient of UNWEIGHTED L_regional with respect to online utility parameters",
            "coefficients_excluded": True,
            "note": "Do NOT include 0.5 coefficients when defining component gradients",
        },
        "cosine_JR": {
            "formula": "dot(g_J, g_R) / (||g_J|| * ||g_R|| + epsilon)",
            "epsilon": EPSILON,
            "interpretation": {
                "positive": "locally aligned gradient directions",
                "negative": "locally conflicting gradient directions",
                "near_zero": "approximately orthogonal",
            },
            "is_optimization_diagnostic": True,
            "is_NOT_causal_outcome": True,
        },
    })

    # =================================================================
    # Phase E4O-2 — Gradient norm ratio contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_gradient_norm_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-2",
        "rho_g": {
            "formula": "||g_R|| / (||g_J|| + epsilon)",
            "epsilon": EPSILON,
            "purpose": "Distinguish scalar loss magnitude from optimization influence",
        },
        "additional_records": [
            "||g_J||",
            "||g_R||",
            "||0.5*g_J + 0.5*g_R||",
        ],
        "historical_context": {
            "regional_to_joint_loss_ratio": 0.0103,
            "regional_to_joint_grad_norm_ratio": 0.0508,
            "N2_NOT_SUPPORTED": True,
        },
    })

    # =================================================================
    # Phase E4O-3 — Gradient conflict fraction contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_gradient_conflict_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-3",
        "conflict_fraction": {
            "formula": "fraction of diagnostic batches/events with cosine_JR < 0",
            "aggregation": "within policy × environment × scenario FIRST",
            "note": "Do not treat individual anchors as independent scientific replicates",
        },
        "per_scenario": True,
        "scenarios": SCENARIOS,
    })

    # =================================================================
    # Phase E4O-4 — Primary scenario gap
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_primary_scenario_gap.json"), {
        "stage": STAGE,
        "phase": "E4O-4",
        "G_cos": {
            "formula": "mean_cosine_STATIONARY - 0.5 * (mean_cosine_PERSISTENT + mean_cosine_MIXED)",
            "interpretation": "G_cos < 0 means joint and regional gradients are less aligned / more conflicting in STATIONARY than in the two dynamic scenarios",
        },
        "G_conflict": {
            "formula": "conflict_fraction_STATIONARY - 0.5 * (conflict_fraction_PERSISTENT + conflict_fraction_MIXED)",
            "interpretation": "Positive means more gradient conflict in STATIONARY",
        },
        "primary_diagnostic_contrast": "G_cos",
        "secondary_diagnostic_contrast": "G_conflict",
        "no_MIGD_primary": True,
    })

    # =================================================================
    # Phase E4O-5 — O1 support logic
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_hypothesis_support_logic.json"), {
        "stage": STAGE,
        "phase": "E4O-5",
        "O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT": {
            "SUPPORTED": {
                "requires_ALL": [
                    "G_cos < 0",
                    "crossed policy/environment bootstrap 95% CI upper for G_cos < 0",
                    "at least 4/5 policy-level G_cos values < 0",
                    "G_conflict > 0",
                    "at least 4/5 policy-level G_conflict values > 0",
                ],
            },
            "PLAUSIBLE": "directional pattern exists but CI/policy consistency is incomplete",
            "NOT_SUPPORTED": "near-zero / no consistent pattern",
            "NOT_AUDITABLE": "data pipeline cannot compute required gradients",
        },
        "important_caveat": (
            "Even O1 SUPPORTED does NOT mean gradient conflict caused STATIONARY harm. "
            "Allowed conclusion: the frozen C2 optimization objective exhibits "
            "scenario-specific gradient conflict consistent with the observed "
            "performance trade-off."
        ),
        "frozen_BEFORE_new_data": True,
    })

    # =================================================================
    # Phase E4O-6 — R1 signal diagnostic contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_r1_signal_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-6",
        "mechanism": "O2_SCENARIO_SPECIFIC_REGIONAL_CREDIT_STRUCTURE",
        "per_scenario_records": {
            "zero_fraction": "fraction |r_r^R1| == 0 (exact zero)",
            "positive_fraction": "fraction r_r^R1 > 0",
            "negative_fraction": "fraction r_r^R1 < 0",
            "mean": "mean(r_r^R1)",
            "median": "median(r_r^R1)",
            "mean_absolute_value": "mean(|r_r^R1|)",
            "q10": "10th percentile",
            "q25": "25th percentile",
            "q75": "75th percentile",
            "q90": "90th percentile",
            "variance": "var(r_r^R1)",
            "TD_target_magnitude": "mean(|y_local|)",
        },
        "R1_activity": {
            "formula": "fraction |r_r^R1| > eps_reward",
            "eps_reward": "exact zero if reward representation is exact/stable",
            "note": "Do NOT invent a large post-hoc threshold",
        },
        "scenarios": SCENARIOS,
        "N5_addressable": True,
        "N5_support_rule": (
            "N5 can become SUPPORTED only if multiple pre-frozen measures "
            "consistently show a distinct STATIONARY regional-credit structure "
            "associated with the trade-off."
        ),
    })

    # =================================================================
    # Phase E4O-7 — Regional TD error contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_td_error_contract.json"), {
        "stage": STAGE,
        "phases": ["E4O-7", "E4O-8"],
        "regional_td_error": {
            "delta_R": "y_local - q_local",
            "records": ["|delta_R|", "sign(delta_R)", "regional SmoothL1"],
            "per_scenario": True,
        },
        "joint_td_error": {
            "delta_J": "y_joint - Q_tot",
            "records": ["|delta_J|", "joint SmoothL1"],
            "ratio": "|delta_R| / (|delta_J| + eps)",
            "note": "Do not treat this as direct gradient magnitude",
        },
        "scenarios": SCENARIOS,
    })

    # =================================================================
    # Phase E4O-9/10 — Common-state bank contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_common_state_bank_contract.json"), {
        "stage": STAGE,
        "phases": ["E4O-9", "E4O-10"],
        "purpose": "Address N3 with COMMON-STATE offline evaluation design",
        "state_bank": {
            "sources": ["SOURCE_C1", "SOURCE_C2"],
            "why_two_sources": (
                "Using states only from C1 could bias toward C1's trajectory "
                "distribution. Using states only from C2 could bias toward C2. "
                "Report separately for SOURCE_C1, SOURCE_C2, and pooled."
            ),
            "per_state_fields": [
                "source_arm",
                "policy",
                "scenario",
                "diagnostic_environment",
                "dynamic_change_index",
                "region_id",
                "local_state_9D",
                "central_state_55D_if_available",
                "active_mask",
            ],
        },
        "cross_evaluation": {
            "for_each_stored_state": [
                "evaluate C1 utility network on same 9-D regional state",
                "evaluate C2 utility network on same 9-D regional state",
            ],
            "record": [
                "argmax_action_C1",
                "argmax_action_C2",
                "action_disagreement_indicator",
                "C1_Q_margin",
                "C2_Q_margin",
            ],
        },
    })

    # =================================================================
    # Phase E4O-11 — Action disagreement contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_action_specialization_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-11",
        "D_action": {
            "formula": "fraction of identical stored states where argmax_C1 != argmax_C2",
            "per_scenario": True,
        },
        "transition_matrix": {
            "shape": "4x4",
            "rows": "C1 action",
            "cols": "C2 action",
            "actions": ACTIONS,
            "per_scenario": True,
        },
        "addresses_N3": True,
    })

    # =================================================================
    # Phase E4O-12 — Q-margin diagnostic contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_q_margin_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-12",
        "margin_C1": "top1(Q_C1) - top2(Q_C1)",
        "margin_C2": "top1(Q_C2) - top2(Q_C2)",
        "Delta_margin": "margin_C2 - margin_C1",
        "report_by": ["scenario", "source_arm", "policy"],
        "no_success_threshold": True,
        "addresses_N4": True,
    })

    # =================================================================
    # Phase E4O-13 — State stratification contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_state_stratification_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-13",
        "motivation": "STATIONARY is the harmful scenario — stratify by V3 state features",
        "stratification_features": [
            "delta_severity",
            "severity",
            "convergence_loss",
            "occupancy",
        ],
        "minimum_strata": {
            "delta_severity_zero": "delta_severity == 0 (structural zero)",
            "delta_severity_positive": "delta_severity > 0",
        },
        "note": "Do not invent post-hoc state features",
    })

    # =================================================================
    # Phase E4O-14 — Joint/regional supervision trade-off contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_joint_regional_tradeoff_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-14",
        "addresses_N8": True,
        "per_batch_records": {
            "||g_J||": "joint gradient norm",
            "||g_R||": "regional gradient norm",
            "cosine_JR": "cosine(g_J, g_R)",
            "||g_C2||": "composite gradient norm where g_C2 = 0.5*g_J + 0.5*g_R",
        },
        "relative_contributions": {
            "rho_joint_to_composite": "||0.5*g_J|| / (||g_C2|| + eps)",
            "rho_regional_to_composite": "||0.5*g_R|| / (||g_C2|| + eps)",
            "note": "Interpret carefully because vector components can cancel",
        },
    })

    # =================================================================
    # Phase E4O-15 — Mixer gradient contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_mixer_gradient_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-15",
        "mixer_gradient_source": "0.5*L_joint ONLY",
        "no_regional_path": True,
        "records": [
            "||g_mixer_joint_unweighted||",
            "||g_mixer_C2||",
        ],
        "engineering_expectation": "||g_mixer_C2|| ≈ 0.5 * ||g_mixer_joint_unweighted|| on same batch",
        "purpose": "integrity, NOT scientific mechanism gate",
    })

    # =================================================================
    # Phase E4O-16 — Monotonicity boundary
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_monotonicity_boundary.json"), {
        "stage": STAGE,
        "phase": "E4O-16",
        "H6_N9_STATUS": "PLAUSIBLE",
        "O_MONOTONICITY_CAUSAL_TEST": "NOT_INCLUDED",
        "reason": (
            "E4O scenario-tradeoff diagnostic does NOT perform new "
            "joint-action counterfactual environment rollouts."
        ),
        "runtime_monotonicity": "engineering integrity check only",
        "promotion_rule": (
            "Do NOT promote N9 to SUPPORTED from E4O unless the frozen "
            "design later explicitly acquires valid counterfactual evidence."
        ),
    })

    # =================================================================
    # Phase E4O-17/18 — Checkpoint lock
    # =================================================================
    ckpt_data = load_json(os.path.join(E4MF_DIR, "E4MF_checkpoint_integrity.json"))
    checkpoint_hashes = ckpt_data["checkpoint_pre_hashes"]

    save_json(os.path.join(OUT_DIR, "E4O_checkpoint_lock.json"), {
        "stage": STAGE,
        "phases": ["E4O-17", "E4O-18"],
        "N6_SUPPORTED_preserved": True,
        "checkpoint_selection_conclusion": (
            "Checkpoint selection lies on a scenario trade-off surface "
            "but did not create the harm."
        ),
        "no_checkpoint_redesign": True,
        "no_checkpoint_reselection": True,
        "no_alternate_epoch": True,
        "no_new_training": True,
        "frozen_checkpoints": {
            "count": 10,
            "arms": [C1, C2],
            "policy_seeds": POLICY_SEEDS,
            "hashes": checkpoint_hashes,
        },
        "QMIX_CONFIG_SHA256": "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365",
        "C2_CREDIT_CONFIG_SHA256": "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8",
    })

    # =================================================================
    # Phase E4O-19 — Fresh diagnostic seed contract
    # =================================================================
    all_prior_seeds = set()
    prior_seed_ranges = {
        "E4TR_TRAIN": (1001, 1060),
        "E4TR_VAL": (1061, 1080),
        "E4E_FINAL_RESERVED": (1081, 1140),
        "E4GD_DIAGNOSTIC": (1201, 1220),
        "E4HT_TRAIN": (1301, 1360),
        "E4HT_VAL": (1361, 1380),
        "E4HE_FINAL": (1381, 1440),
        "E4IT_TRAIN": (1501, 1560),
        "E4IT_VAL": (1561, 1580),
        "E4IE_FINAL": (1581, 1640),
        "E4IDD_DIAGNOSTIC": (1701, 1720),
        "E4JF_COMPARATOR": (1801, 1860),
        "E4KT_TRAIN": (1901, 1960),
        "E4KT_VAL": (1961, 1980),
        "E4KF_FINAL": (1981, 2040),
        "E4MT_TRAIN": (2101, 2160),
        "E4MT_VAL": (2161, 2180),
        "E4MF_FINAL": (2181, 2240),
    }
    for label, (lo, hi) in prior_seed_ranges.items():
        for s in range(lo, hi + 1):
            all_prior_seeds.add(s)

    diag_set = set(DIAGNOSTIC_SEEDS)
    overlap = diag_set & all_prior_seeds
    disjoint = len(overlap) == 0

    save_json(os.path.join(OUT_DIR, "E4O_diagnostic_seed_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-19",
        "diagnostic_seeds": "2301..2360",
        "diagnostic_seed_list": DIAGNOSTIC_SEEDS,
        "n_seeds": len(DIAGNOSTIC_SEEDS),
        "purpose": "DIAGNOSTIC_ONLY",
        "NOT_TRAIN": True,
        "NOT_VALIDATION": True,
        "NOT_CONFIRMATORY_FINAL": True,
    })

    # =================================================================
    # Seed disjointness verification
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_seed_disjointness.json"), {
        "stage": STAGE,
        "diagnostic_range": "2301..2360",
        "prior_seed_ranges": {k: f"{lo}..{hi}" for k, (lo, hi) in prior_seed_ranges.items()},
        "total_prior_seeds": len(all_prior_seeds),
        "overlap": sorted(overlap) if overlap else [],
        "disjoint": disjoint,
        "SEED_DISJOINTNESS_PASS": disjoint,
    })

    if not disjoint:
        save_json(os.path.join(OUT_DIR, "E4O_verdict.json"), {
            "stage": STAGE,
            "E4O_COMPLETE": "NO",
            "REASON": "SEED_OVERLAP",
            "overlapping_seeds": sorted(overlap),
            "NEXT_STAGE": "E4O_SEED_REVIEW",
        })
        return

    # =================================================================
    # Phase E4O-20 — Diagnostic firewall
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_future_firewall.json"), {
        "stage": STAGE,
        "phase": "E4O-20",
        "O_DIAGNOSTIC_ACCESS_COUNT": 0,
        "no_seed_instantiated": True,
        "seeds_reserved_but_not_opened": "2301..2360",
        "design_only": True,
    })

    # =================================================================
    # Phase E4O-21 — Diagnostic matrix
    # =================================================================
    n_arms = 2
    n_policies = len(POLICY_SEEDS)
    n_envs = len(DIAGNOSTIC_SEEDS)
    n_scenarios = len(SCENARIOS)
    total_episodes = n_arms * n_policies * n_envs * n_scenarios

    save_json(os.path.join(OUT_DIR, "E4O_diagnostic_matrix.json"), {
        "stage": STAGE,
        "phase": "E4O-21",
        "source_arms": [C1, C2],
        "policy_seeds": POLICY_SEEDS,
        "diagnostic_environment_seeds": "2301..2360",
        "n_diagnostic_environments": n_envs,
        "scenarios": SCENARIOS,
        "total_diagnostic_episodes": total_episodes,
        "expected_total": 1800,
        "matches_expected": total_episodes == 1800,
        "evaluation_only": True,
        "epsilon": 0.0,
        "no_learning": True,
        "NOT_PERFORMANCE_CONFIRMATORY": True,
        "purpose": "collect transitions, states, rewards, TD quantities for mechanism diagnosis",
    })

    assert total_episodes == 1800, f"Matrix size mismatch: {total_episodes} != 1800"

    # =================================================================
    # Phase E4O-22 — Transition logging schema
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_transition_logging_schema.json"), {
        "stage": STAGE,
        "phase": "E4O-22",
        "per_dynamic_change_event": {
            "source_arm": "C1 or C2",
            "policy_seed": "int from POLICY_SEEDS",
            "scenario": "PERSISTENT / STATIONARY / MIXED",
            "diagnostic_environment_seed": "int from 2301..2360",
            "episode_change_index": "int",
            "region_id": "int",
            "active_mask": "bool",
            "local_state_9D": f"float[{STATE_DIM}]",
            "central_state_55D": f"float[{CENTRAL_STATE_DIM}]",
            "requested_action": "int (0-3)",
            "realized_operator": "str",
            "focal_R1_reward": "float",
            "all_active_regional_R1_rewards": "list[float]",
            "r_joint": "float",
            "next_local_state_9D": f"float[{STATE_DIM}]",
            "next_central_state_55D": f"float[{CENTRAL_STATE_DIM}]",
            "done": "bool",
            "C1_utility_Q_vector": f"float[{N_ACTIONS}]",
            "C2_utility_Q_vector": f"float[{N_ACTIONS}]",
        },
    })

    # =================================================================
    # Phase E4O-23 — Offline gradient computation contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_offline_gradient_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-23",
        "procedure": [
            "freeze environment interactions after transition collection",
            "gradient diagnostics operate ONLY on stored transitions",
            "for each deterministic diagnostic batch:",
            "  1. load frozen checkpoint",
            "  2. forward pass",
            "  3. compute L_joint",
            "  4. compute L_regional",
            "  5. call backward/autograd ONLY for diagnostic gradients",
            "  6. DO NOT optimizer.step()",
        ],
        "no_model_mutation": {
            "optimizer_steps": 0,
            "parameter_changes": 0,
            "target_updates": 0,
            "replay_mutation": 0,
        },
        "gradient_is_diagnostic_only": True,
    })

    # =================================================================
    # Phase E4O-24 — Inferential unit contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_inferential_unit_contract.json"), {
        "stage": STAGE,
        "phase": "E4O-24",
        "aggregation_rule": (
            "First aggregate gradient diagnostics within "
            "policy × environment × scenario. "
            "Then treat policy axis (5) and environment axis (60) "
            "as inferential axes."
        ),
        "policy_axis": n_policies,
        "environment_axis": n_envs,
        "no_anchor_pseudoreplication": True,
    })

    # =================================================================
    # Phase E4O-25 — Statistical design
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_statistical_design.json"), {
        "stage": STAGE,
        "phase": "E4O-25",
        "bootstrap": {
            "method": "paired crossed two-axis bootstrap",
            "policy_axis": n_policies,
            "environment_axis": n_envs,
            "N": DIAGNOSTIC_BOOTSTRAP_N,
            "confidence_level": 0.95,
            "CI_method": "percentile",
            "seed": DIAGNOSTIC_BOOTSTRAP_SEED,
        },
        "purpose": "mechanism contrasts only",
        "do_NOT_reuse_E4MF_bootstrap": True,
    })

    # =================================================================
    # Phase E4O-26 — Primary diagnostic contrast (reaffirm)
    # =================================================================
    # Already in E4O_primary_scenario_gap.json

    # =================================================================
    # Phase E4O-27 — No performance verdict
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_no_performance_verdict.json"), {
        "stage": STAGE,
        "phase": "E4O-27",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "E4MF_REMAINS_AUTHORITATIVE": True,
        "rule": (
            "Future diagnostic execution MUST output "
            "PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED. "
            "E4MF remains the authoritative performance experiment."
        ),
    })

    # =================================================================
    # Phase E4O-28 — N5 diagnostic note
    # Already covered in R1 signal contract

    # Phase E4O-29 — Hypothesis update rules
    # Already in hypothesis_contract.json

    # =================================================================
    # Phase E4O-30 — Future routing
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_future_routing.json"), {
        "stage": STAGE,
        "phase": "E4O-30",
        "routing_rules": {
            "O1_SUPPORTED": {
                "NEXT_STAGE": "E4P_GRADIENT_CONFLICT_REVISION_DESIGN",
                "note": (
                    "Do NOT yet choose PCGrad, GradNorm, adaptive loss "
                    "weighting, or projection methods. First compare "
                    "candidate minimal revisions."
                ),
            },
            "O1_NOT_SUPPORTED_AND_N3_STRONGLY_SUPPORTED": {
                "NEXT_STAGE": "E4P_ACTION_SPECIALIZATION_REVISION_DESIGN",
            },
            "O1_NOT_SUPPORTED_AND_N4_STRONGLY_SUPPORTED": {
                "NEXT_STAGE": "E4P_POLICY_DECISIVENESS_REVISION_DESIGN",
            },
            "N5_STRONGEST": {
                "NEXT_STAGE": "E4P_SCENARIO_CONDITIONAL_CREDIT_REVIEW",
                "note": "Do NOT create scenario labels as controller inputs automatically",
            },
            "N8_STRONGEST_WITHOUT_GRADIENT_CONFLICT": {
                "NEXT_STAGE": "E4P_JOINT_REGIONAL_OBJECTIVE_REVIEW",
            },
            "ALL_MECHANISMS_INSUFFICIENT": {
                "NEXT_STAGE": "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN",
                "note": "Address remaining N9/H6 hypothesis with separate design",
            },
        },
    })

    # =================================================================
    # Phase E4O-31/32 — E4OI validation contract
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_e4oi_validation_contract.json"), {
        "stage": STAGE,
        "phases": ["E4O-31", "E4O-32"],
        "next_immediate_stage": "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION",
        "then": "E4OD_SCENARIO_TRADEOFF_DIAGNOSTIC_EXECUTION",
        "then_after": "evidence-based E4P stage",
        "do_not_open_seeds_during": ["E4O", "E4OI"],
        "E4OI_validation_requirements": {
            "V01_transition_logging_complete": "all schema fields present",
            "V02_no_model_mutation": "optimizer_steps=0, parameter_changes=0",
            "V03_exact_g_J_extraction": "verified on synthetic data",
            "V04_exact_g_R_extraction": "verified on synthetic data",
            "V05_gradient_cosine_correct": "verified against known vectors",
            "V06_gradient_norm_ratio_correct": "verified against known norms",
            "V07_composite_gradient_identity": "g_C2 = 0.5*g_J + 0.5*g_R verified",
            "V08_mixer_gradient_regional_path_absent": "no regional gradient through mixer",
            "V09_common_state_dual_network_evaluation": "both networks on same state",
            "V10_action_disagreement_exact": "verified on known Q tables",
            "V11_Q_margins_exact": "verified on known Q tables",
            "V12_scenario_gap_formulas_exact": "G_cos and G_conflict verified",
            "V13_policy_environment_aggregation_exact": "correct mean within cells",
            "V14_crossed_bootstrap_exact": "verified on known matrix",
            "V15_diagnostic_seed_firewall_intact": "no seed from 2301..2360 instantiated",
        },
    })

    # =================================================================
    # Protocol frozen marker
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_protocol_frozen.json"), {
        "stage": STAGE,
        "all_contracts_frozen": True,
        "gradient_alignment_contract": "FROZEN",
        "gradient_norm_contract": "FROZEN",
        "gradient_conflict_contract": "FROZEN",
        "primary_scenario_gap": "FROZEN",
        "hypothesis_support_logic": "FROZEN",
        "r1_signal_contract": "FROZEN",
        "td_error_contract": "FROZEN",
        "common_state_bank_contract": "FROZEN",
        "action_specialization_contract": "FROZEN",
        "q_margin_contract": "FROZEN",
        "state_stratification_contract": "FROZEN",
        "joint_regional_tradeoff_contract": "FROZEN",
        "mixer_gradient_contract": "FROZEN",
        "monotonicity_boundary": "FROZEN",
        "checkpoint_lock": "FROZEN",
        "diagnostic_seed_contract": "FROZEN",
        "diagnostic_matrix": "FROZEN",
        "transition_logging_schema": "FROZEN",
        "offline_gradient_contract": "FROZEN",
        "inferential_unit_contract": "FROZEN",
        "statistical_design": "FROZEN",
        "no_performance_verdict": "FROZEN",
        "future_routing": "FROZEN",
        "e4oi_validation_contract": "FROZEN",
    })

    # =================================================================
    # Report
    # =================================================================
    report_lines = []
    report_lines.append("# E4O — Scenario Trade-off Diagnostic Design Report\n")

    report_lines.append("## A. Immutable E4MF failure\n")
    report_lines.append("E4MF verdict = FAILED. This cannot be altered by E4O or any future diagnostic.\n")
    report_lines.append(f"- MIXED primary: DIRECTIONAL_POSITIVE (delta C1-C2 = {verdict['E4MF_MIXED_DELTA_C1_MINUS_C2']:.6e})\n")
    report_lines.append("- Robustness: FAIL (3/5)\n")
    report_lines.append("- PERSISTENT safety: PASS\n")
    report_lines.append("- STATIONARY safety: FAIL\n\n")

    report_lines.append("## B. E4N scenario-tradeoff finding\n")
    report_lines.append("PRIMARY = MULTIFACTORIAL_SCENARIO_TRADEOFF (N10 SUPPORTED).\n")
    report_lines.append("- PERSISTENT: C2 BENEFICIAL across 5/5 policy pairs\n")
    report_lines.append("- STATIONARY: C2 HARMFUL across 5/5 policy pairs\n")
    report_lines.append("- MIXED: C2 DIRECTIONAL_POSITIVE (3/5 policies)\n")
    report_lines.append("- Cross-stage pattern: E4KF had PERSISTENT HARMFUL / STATIONARY BENEFICIAL\n")
    report_lines.append("- E4MF reversed: PERSISTENT BENEFICIAL / STATIONARY HARMFUL\n\n")

    report_lines.append("## C. Why E4O is diagnostic rather than another revision\n")
    report_lines.append("E4N identified that critical mechanism data was not stored in prior stages.\n")
    report_lines.append("In particular, per-scenario gradient alignment between L_joint and L_regional\n")
    report_lines.append("was never measured. Until this is measured, any revision would be speculative.\n\n")

    report_lines.append("## D. Missing per-scenario gradient evidence\n")
    report_lines.append("N1 = NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS.\n")
    report_lines.append("Stored gradient cosines were aggregate only (median -0.070, all negative).\n")
    report_lines.append("Per-scenario breakdown was not retained in E4MTR artifacts.\n\n")

    report_lines.append("## E. Gradient alignment hypothesis\n")
    report_lines.append("O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT is the primary diagnostic mechanism.\n")
    report_lines.append("- g_J = gradient of UNWEIGHTED L_joint w.r.t. online utility parameters\n")
    report_lines.append("- g_R = gradient of UNWEIGHTED L_regional w.r.t. online utility parameters\n")
    report_lines.append("- cosine_JR = dot(g_J, g_R) / (||g_J|| * ||g_R|| + eps)\n")
    report_lines.append("- eps = 1e-12\n\n")

    report_lines.append("## F. Gradient norm / conflict metrics\n")
    report_lines.append("- rho_g = ||g_R|| / (||g_J|| + eps)\n")
    report_lines.append("- conflict_fraction = fraction of events with cosine_JR < 0\n")
    report_lines.append("- Historical: loss ratio 0.0103, grad ratio 0.0508 (N2 NOT_SUPPORTED)\n\n")

    report_lines.append("## G. Primary Stationary-vs-dynamic contrast\n")
    report_lines.append("G_cos = mean_cosine_STATIONARY - 0.5*(mean_cosine_PERSISTENT + mean_cosine_MIXED)\n")
    report_lines.append("G_conflict = conflict_fraction_STATIONARY - 0.5*(conflict_fraction_PERSISTENT + conflict_fraction_MIXED)\n")
    report_lines.append("G_cos < 0 means gradients more conflicting in STATIONARY.\n\n")

    report_lines.append("## H. Regional R1 signal structure\n")
    report_lines.append("O2_SCENARIO_SPECIFIC_REGIONAL_CREDIT_STRUCTURE addresses N5.\n")
    report_lines.append("Per-scenario: zero fraction, positive/negative fraction, mean, median, MAV,\n")
    report_lines.append("quantiles (q10/q25/q75/q90), variance, TD-target magnitude.\n")
    report_lines.append("R1_activity = fraction |r_r^R1| > eps_reward.\n\n")

    report_lines.append("## I. Regional and joint TD errors\n")
    report_lines.append("- delta_R = y_local - q_local (regional TD error)\n")
    report_lines.append("- delta_J = y_joint - Q_tot (joint TD error)\n")
    report_lines.append("- Records: |delta_R|, sign(delta_R), SmoothL1 per scenario.\n\n")

    report_lines.append("## J. Common-state diagnostic bank\n")
    report_lines.append("States collected from BOTH C1 and C2 trajectories.\n")
    report_lines.append("Both frozen networks evaluated on each identical stored state.\n")
    report_lines.append("Reports separated by SOURCE_C1, SOURCE_C2, and pooled.\n\n")

    report_lines.append("## K. Action-specialization comparison\n")
    report_lines.append("D_action(scenario) = fraction where argmax_C1 != argmax_C2.\n")
    report_lines.append("4x4 transition matrix per scenario.\n\n")

    report_lines.append("## L. Q-margin / policy decisiveness\n")
    report_lines.append("Delta_margin = margin_C2 - margin_C1 per scenario/source/policy.\n")
    report_lines.append("No success threshold.\n\n")

    report_lines.append("## M. State stratification\n")
    report_lines.append("Stratify by delta_severity (zero vs positive), severity, convergence_loss, occupancy.\n")
    report_lines.append("Focus on STATIONARY where harm occurs.\n\n")

    report_lines.append("## N. Joint-vs-regional supervision trade-off\n")
    report_lines.append("g_C2 = 0.5*g_J + 0.5*g_R. Relative contribution magnitudes recorded.\n")
    report_lines.append("Addresses N8.\n\n")

    report_lines.append("## O. Mixer-gradient integrity\n")
    report_lines.append("Mixer gradient from 0.5*L_joint ONLY. No regional path.\n")
    report_lines.append("Engineering check: ||g_mixer_C2|| ≈ 0.5 * ||g_mixer_joint_unweighted||.\n\n")

    report_lines.append("## P. Monotonicity boundary\n")
    report_lines.append("N9/H6 remains PLAUSIBLE. O_MONOTONICITY_CAUSAL_TEST = NOT_INCLUDED.\n")
    report_lines.append("No counterfactual rollouts in E4O.\n\n")

    report_lines.append("## Q. Fresh diagnostic seed contract\n")
    report_lines.append("Seeds 2301..2360 reserved as DIAGNOSTIC_ONLY.\n")
    report_lines.append(f"Disjointness verified against {len(prior_seed_ranges)} prior seed ranges.\n")
    report_lines.append(f"Overlap: {len(overlap)} seeds. PASS: {disjoint}.\n\n")

    report_lines.append("## R. Transition logging\n")
    report_lines.append("Per dynamic-change event: source_arm, policy_seed, scenario, env_seed,\n")
    report_lines.append("episode/change index, region_id, active mask, local 9-D state,\n")
    report_lines.append("central 55-D state, action, operator, focal R1, all regional R1,\n")
    report_lines.append("r_joint, next states, done, C1 Q-vector, C2 Q-vector.\n\n")

    report_lines.append("## S. Offline gradient analysis\n")
    report_lines.append("Gradients computed on stored transitions only.\n")
    report_lines.append("No optimizer.step(). No parameter changes. No target updates.\n\n")

    report_lines.append("## T. Statistical design\n")
    report_lines.append(f"Paired crossed two-axis bootstrap: {n_policies}×{n_envs}.\n")
    report_lines.append(f"N = {DIAGNOSTIC_BOOTSTRAP_N}, seed = {DIAGNOSTIC_BOOTSTRAP_SEED}.\n")
    report_lines.append("95% percentile CI. Mechanism contrasts only.\n\n")

    report_lines.append("## U. Hypothesis support logic\n")
    report_lines.append("O1 SUPPORTED requires ALL: G_cos < 0, CI upper < 0,\n")
    report_lines.append("≥4/5 policy G_cos < 0, G_conflict > 0, ≥4/5 policy G_conflict > 0.\n")
    report_lines.append("Even SUPPORTED does NOT mean gradient conflict caused STATIONARY harm.\n\n")

    report_lines.append("## V. No-performance-verdict boundary\n")
    report_lines.append("PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED.\n")
    report_lines.append("E4MF remains the authoritative performance experiment.\n\n")

    report_lines.append("## W. Future diagnostic execution\n")
    report_lines.append("NEXT: E4OI → E4OD → evidence-based E4P.\n")
    report_lines.append("Seeds not opened during E4O or E4OI.\n\n")

    report_lines.append("## X. Limitations\n")
    report_lines.append("- E4MF performance verdict remains FAILED\n")
    report_lines.append("- E4O is a mechanism diagnostic design\n")
    report_lines.append("- Gradient conflict is not yet established\n")
    report_lines.append("- Action shifts do not imply causation\n")
    report_lines.append("- Q-margin changes do not imply causation\n")
    report_lines.append("- R1 magnitude alone does not measure reward quality\n")
    report_lines.append("- C2 still contains repeated sibling joint targets\n")
    report_lines.append("- C2 still uses monotonic QMIX\n")
    report_lines.append("- Checkpoint-selection tension is already supported but not sufficient to explain harm\n")
    report_lines.append("- Diagnostic seeds are not confirmatory final seeds\n")
    report_lines.append("- No real-world UAV evidence\n\n")

    report_lines.append("## Y. Next-stage routing\n")
    report_lines.append("NEXT_STAGE = E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION\n")

    with open(os.path.join(OUT_DIR, "E4O_report.md"), "w") as f:
        f.write("".join(report_lines))

    # =================================================================
    # Final verdict
    # =================================================================
    save_json(os.path.join(OUT_DIR, "E4O_verdict.json"), {
        "stage": STAGE,
        "E4O_COMPLETE": "YES",
        "ORIGINAL_E4MF_VERDICT": "FAILED",
        "E4MF_VERDICT_CHANGED": "NO",
        "E4N_PRIMARY_FAILURE_HYPOTHESIS": "MULTIFACTORIAL_SCENARIO_TRADEOFF",
        "CROSS_STAGE_PATTERN": "SCENARIO_TRADEOFF_SHIFT",
        "PERSISTENT_POLICY_PATTERN": "C2_BETTER_5_OF_5",
        "STATIONARY_POLICY_PATTERN": "C2_WORSE_5_OF_5",
        "N1_PREVIOUS_STATUS": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
        "N2_STATUS": "NOT_SUPPORTED",
        "N6_STATUS": "SUPPORTED",
        "N8_STATUS": "PLAUSIBLE",
        "N9_STATUS": "PLAUSIBLE",
        "PRIMARY_DIAGNOSTIC_MECHANISM": "O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT",
        "PRIMARY_GRADIENT_METRIC": "COSINE_G_JOINT_G_REGIONAL",
        "PRIMARY_SCENARIO_CONTRAST": "STATIONARY_MINUS_MEAN_PERSISTENT_MIXED",
        "SECONDARY_GRADIENT_METRIC": "NEGATIVE_COSINE_FRACTION",
        "REGIONAL_TO_JOINT_LOSS_RATIO_HISTORICAL": 0.0103,
        "REGIONAL_TO_JOINT_GRAD_NORM_RATIO_HISTORICAL": 0.0508,
        "COMMON_STATE_CROSS_EVALUATION": "FROZEN",
        "ACTION_SPECIALIZATION_DIAGNOSTIC": "FROZEN",
        "Q_MARGIN_DIAGNOSTIC": "FROZEN",
        "R1_SCENARIO_SIGNAL_DIAGNOSTIC": "FROZEN",
        "MONOTONICITY_CAUSAL_TEST": "NOT_INCLUDED",
        "FROZEN_CHECKPOINTS": "C1_C2_POLICIES_6001_TO_6005",
        "DIAGNOSTIC_SEEDS": "2301..2360",
        "DIAGNOSTIC_MATRIX_PLANNED": "1800_EPISODES",
        "DIAGNOSTIC_BOOTSTRAP": "PAIRED_CROSSED_TWO_AXIS_5x60",
        "DIAGNOSTIC_BOOTSTRAP_RESAMPLES": DIAGNOSTIC_BOOTSTRAP_N,
        "DIAGNOSTIC_BOOTSTRAP_SEED": DIAGNOSTIC_BOOTSTRAP_SEED,
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "O_DIAGNOSTIC_ACCESS_COUNT": 0,
        "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EXECUTION": "NO",
        "NEXT_STAGE": "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION",
    })


if __name__ == "__main__":
    run()
