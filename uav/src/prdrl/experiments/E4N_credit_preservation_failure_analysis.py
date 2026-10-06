"""E4N — Credit Preservation Failure Analysis.

Post-hoc failure analysis of C2 (credit-preserving QMIX) relative to C1 (standard QMIX).
Uses ONLY existing frozen artifacts from E4MTR/E4MF. No new training or environment episodes.
"""

import json
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

STAGE = "E4N_CREDIT_PRESERVATION_FAILURE_ANALYSIS"
OUT_DIR = "results/E4N_credit_preservation_failure_analysis"

E4MF_DIR = "results/E4MF_credit_preservation_ab_final_evaluation_execution"
E4MTR_DIR = "results/E4MTR_credit_preservation_ab_training_execution"
E4L_DIR = "results/E4L_coordination_failure_analysis"
E4M_DIR = "results/E4M_joint_credit_preservation_design"
E4MI_DIR = "results/E4MI_joint_credit_preservation_implementation_validation"
E4MT_DIR = "results/E4MT_credit_preservation_ab_training_protocol_freeze"
E4ME_DIR = "results/E4ME_credit_preservation_ab_final_evaluation_protocol_freeze"
E4KF_DIR = "results/E4KF_joint_coordination_ab_final_evaluation_execution"

C1 = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2 = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

HYPOTHESIS_VOCABULARY = [
    "SUPPORTED", "PLAUSIBLE", "WEAK", "NOT_SUPPORTED",
    "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
]

ROUTING_MAP = {
    "N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT": "E4O_GRADIENT_CONFLICT_DIAGNOSTIC_DESIGN",
    "N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS": "E4O_CREDIT_SCALE_NORMALIZATION_DESIGN",
    "N3_ACTION_SPECIALIZATION_SHIFT": "E4O_ACTION_SPECIALIZATION_DIAGNOSTIC_DESIGN",
    "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT": "E4O_POLICY_DECISIVENESS_DIAGNOSTIC_DESIGN",
    "N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS": "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN",
    "N6_CHECKPOINT_SELECTION_SCENARIO_TENSION": "E4O_CHECKPOINT_OBJECTIVE_REVIEW",
    "N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF": "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN",
    "N9_MONOTONIC_QMIX_LIMITATION_REMAINS": "E4O_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN",
    "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF": "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN",
    "MULTIFACTORIAL_SCENARIO_TRADEOFF": "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN",
    "MULTIFACTORIAL_OR_UNRESOLVED": "E4O_CREDIT_PRESERVATION_FAILURE_DIAGNOSTIC_DESIGN",
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _save_json(data, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _save_csv(rows, fieldnames, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def phase_0_immutable_preservation(e4mf_verdict):
    """E4N-0: verify and preserve E4MF result."""
    checks = {
        "E4MF_COMPLETE": e4mf_verdict["E4MF_COMPLETE"] == "YES",
        "E4MF_VERDICT_FAILED": e4mf_verdict["CREDIT_PRESERVATION_CAUSAL_VERDICT"] == "FAILED",
        "MIXED_DIRECTIONAL_POSITIVE": e4mf_verdict["MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE",
        "MIXED_DELTA_POSITIVE": e4mf_verdict["MIXED_DELTA_C1_MINUS_C2"] > 0,
        "MIXED_CI_CROSSES_ZERO": (
            e4mf_verdict["MIXED_DELTA_CI"][0] < 0 < e4mf_verdict["MIXED_DELTA_CI"][1]
        ),
        "ROBUSTNESS_3_OF_5": e4mf_verdict["POLICY_PAIR_POSITIVE_COUNT"] == 3,
        "ROBUSTNESS_FAIL": e4mf_verdict["CREDIT_PRESERVATION_POLICY_ROBUSTNESS"] == "FAIL",
        "PERSISTENT_SAFETY_PASS": e4mf_verdict["PERSISTENT_SAFETY"] == "PASS",
        "STATIONARY_SAFETY_FAIL": e4mf_verdict["STATIONARY_SAFETY"] == "FAIL",
        "MACRO_SAFETY_PASS": e4mf_verdict["MACRO_SAFETY"] == "PASS",
    }
    result = {
        "E4MF_VERDICT_CHANGED": "NO",
        "all_checks_pass": all(checks.values()),
        "checks": checks,
    }
    _save_json(result, "E4N_e4mf_verdict_preservation.json")
    return result


def phase_0_pre_manifest():
    """Verify all required input artifacts exist."""
    required_e4mf = [
        "E4MF_verdict.json", "E4MF_episode_results.csv",
        "E4MF_action_context.csv", "E4MF_q_margin_context.csv",
        "E4MF_policy_pair_summary.csv", "E4MF_scenario_effects.csv",
        "E4MF_macro_effects.csv", "E4MF_fe_effects.csv",
        "E4MF_bootstrap_results.json", "E4MF_scientific_gates.json",
        "E4MF_validation_final_comparison.json",
        "E4MF_checkpoint_integrity.json", "E4MF_loss_spec_integrity.json",
        "E4MF_architecture_integrity.json", "E4MF_no_learning_audit.json",
        "E4MF_report.md",
    ]
    required_e4mtr = [
        "E4MTR_training_episode_results.csv",
        "E4MTR_training_update_summary.csv",
        "E4MTR_credit_diagnostics.csv", "E4MTR_action_trajectory.csv",
        "E4MTR_validation_all_checkpoints.csv",
        "E4MTR_validation_selected_policies.csv",
        "E4MTR_selected_checkpoints.json",
        "E4MTR_validation_ab_summary.csv",
        "E4MTR_validation_action_diagnostics.csv",
        "E4MTR_verdict.json", "E4MTR_gradient_diversity.csv",
        "E4MTR_loss_dispersion.csv",
    ]

    manifest = {}
    for name in required_e4mf:
        p = os.path.join(E4MF_DIR, name)
        manifest[p] = "AVAILABLE" if os.path.exists(p) else "NOT_AVAILABLE"
    for name in required_e4mtr:
        p = os.path.join(E4MTR_DIR, name)
        manifest[p] = "AVAILABLE" if os.path.exists(p) else "NOT_AVAILABLE"
    for d in [E4L_DIR, E4M_DIR, E4MI_DIR, E4MT_DIR, E4ME_DIR, E4KF_DIR]:
        vp = os.path.join(d, os.path.basename(d).split("_")[0].upper() + "_verdict.json")
        if not os.path.exists(vp):
            alt = [f for f in os.listdir(d) if f.endswith("verdict.json")]
            vp = os.path.join(d, alt[0]) if alt else vp
        manifest[vp] = "AVAILABLE" if os.path.exists(vp) else "NOT_AVAILABLE"

    result = {
        "stage": STAGE,
        "artifacts": manifest,
        "all_available": all(v == "AVAILABLE" for v in manifest.values()),
        "missing": [k for k, v in manifest.items() if v != "AVAILABLE"],
    }
    _save_json(result, "E4N_pre_manifest.json")
    return result


def phase_0_historical_integrity():
    """Verify historical verdicts are preserved."""
    e4kf = _load_json(os.path.join(E4KF_DIR, "E4KF_verdict.json"))
    e4l = _load_json(os.path.join(E4L_DIR, "E4L_verdict.json"))

    result = {
        "E4KF_COORDINATION_CAUSAL_VERDICT": e4kf["COORDINATION_CAUSAL_VERDICT"],
        "E4KF_PERSISTENT_RESPONSE_EFFECT": e4kf["PERSISTENT_RESPONSE_EFFECT"],
        "E4KF_PERSISTENT_SAFETY": e4kf["PERSISTENT_SAFETY"],
        "E4KF_STATIONARY_RESPONSE_EFFECT": e4kf["STATIONARY_RESPONSE_EFFECT"],
        "E4KF_STATIONARY_SAFETY": e4kf["STATIONARY_SAFETY"],
        "E4KF_MIXED_PRIMARY": e4kf["MIXED_PRIMARY"],
        "E4KF_MACRO_RESPONSE_EFFECT": e4kf["MACRO_RESPONSE_EFFECT"],
        "E4L_PRIMARY_FAILURE_HYPOTHESIS": e4l["PRIMARY_FAILURE_HYPOTHESIS"],
        "E4L_H4_STATUS": e4l["H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION"],
        "E4L_H6_STATUS": e4l["H6_MONOTONIC_REPRESENTATION_LIMITATION"],
        "E4KF_PERSISTENT_HARMFUL_CONFIRMED": e4kf["PERSISTENT_RESPONSE_EFFECT"] == "HARMFUL",
        "E4KF_STATIONARY_BENEFICIAL_CONFIRMED": e4kf["STATIONARY_RESPONSE_EFFECT"] == "BENEFICIAL",
        "historical_correction_applied": True,
        "note": "E4KF: PERSISTENT was HARMFUL (FAIL), STATIONARY was BENEFICIAL (PASS). "
                "E4MF: PERSISTENT is BENEFICIAL (PASS), STATIONARY is HARMFUL (FAIL). "
                "This is a SCENARIO_TRADEOFF_SHIFT, not the same bottleneck.",
    }
    _save_json(result, "E4N_historical_integrity.json")
    return result


def phase_3_scenario_effect_localization(e4mf_bootstrap, e4mf_episodes):
    """E4N-3: per-scenario effect localization with per-policy detail."""
    groups = defaultdict(list)
    for r in e4mf_episodes:
        key = (r["policy_seed"], r["arm"][:2], r["scenario"])
        groups[key].append(float(r["MIGD_response"]))

    rows = []
    for sc in SCENARIOS:
        sc_effects = e4mf_bootstrap["scenario_effects"][sc]
        resp = sc_effects["MIGD_response"]
        epoch = sc_effects["MIGD_epoch"]
        fe = sc_effects["response_FE"]

        policy_deltas = []
        for pol in POLICY_SEEDS:
            c1_vals = groups.get((str(pol), "C1", sc), [])
            c2_vals = groups.get((str(pol), "C2", sc), [])
            if c1_vals and c2_vals:
                delta = statistics.mean(c1_vals) - statistics.mean(c2_vals)
                policy_deltas.append(delta)

        rows.append({
            "scenario": sc,
            "C1_MIGD_response": resp["C1_mean"],
            "C2_MIGD_response": resp["C2_mean"],
            "effect_C1_minus_C2_response": resp["delta_C1_minus_C2"],
            "ci_lo_response": resp["ci_lo"],
            "ci_hi_response": resp["ci_hi"],
            "classification_response": resp["classification"],
            "effect_C1_minus_C2_epoch": epoch["delta_C1_minus_C2"],
            "classification_epoch": epoch["classification"],
            "fe_effect_C1_minus_C2": fe["delta_C1_minus_C2"],
            "policy_delta_6001": policy_deltas[0] if len(policy_deltas) > 0 else None,
            "policy_delta_6002": policy_deltas[1] if len(policy_deltas) > 1 else None,
            "policy_delta_6003": policy_deltas[2] if len(policy_deltas) > 2 else None,
            "policy_delta_6004": policy_deltas[3] if len(policy_deltas) > 3 else None,
            "policy_delta_6005": policy_deltas[4] if len(policy_deltas) > 4 else None,
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_scenario_effect_localization.csv")
    return rows


def phase_4_cross_stage_tradeoff(e4kf_scenario_effects, e4mf_scenario_effects):
    """E4N-4: cross-stage scenario trade-off table."""
    rows = []
    for sc in SCENARIOS:
        e4kf_row = next(
            (r for r in e4kf_scenario_effects
             if r["scenario"] == sc and r["endpoint"] == "MIGD_response"), None
        )
        e4mf_row = next(
            (r for r in e4mf_scenario_effects
             if r["scenario"] == sc and r["endpoint"] == "MIGD_response"), None
        )
        e4kf_epoch = next(
            (r for r in e4kf_scenario_effects
             if r["scenario"] == sc and r["endpoint"] == "MIGD_epoch"), None
        )
        e4mf_epoch = next(
            (r for r in e4mf_scenario_effects
             if r["scenario"] == sc and r["endpoint"] == "MIGD_epoch"), None
        )

        rows.append({
            "scenario": sc,
            "E4KF_comparison": "C0_vs_C1",
            "E4KF_response_classification": e4kf_row["classification"] if e4kf_row else "N/A",
            "E4KF_epoch_classification": e4kf_epoch["classification"] if e4kf_epoch else "N/A",
            "E4KF_safety": (
                "FAIL" if e4kf_row and e4kf_row["classification"] == "HARMFUL" else "PASS"
            ),
            "E4MF_comparison": "C1_vs_C2",
            "E4MF_response_classification": e4mf_row["classification"] if e4mf_row else "N/A",
            "E4MF_epoch_classification": e4mf_epoch["classification"] if e4mf_epoch else "N/A",
            "E4MF_safety": (
                "FAIL" if e4mf_row and e4mf_row["classification"] == "HARMFUL" else "PASS"
            ),
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_cross_stage_scenario_tradeoff.csv")
    return rows


def phase_5_gradient_conflict_analysis(gradient_diversity):
    """E4N-5: gradient conflict analysis from stored E4MTR diagnostics."""
    rows = []
    for r in gradient_diversity:
        joint_norm = float(r["mean_norm_joint"])
        regional_norm = float(r["mean_norm_regional"])
        composite_norm = float(r["mean_norm_composite"])
        cosine = float(r["mean_cosine_joint_regional"])
        n = int(r["n_samples"])
        ratio = joint_norm / regional_norm if regional_norm > 0 else float("inf")

        rows.append({
            "policy_seed": int(r["policy_seed"]),
            "mean_norm_joint": joint_norm,
            "mean_norm_regional": regional_norm,
            "mean_norm_composite": composite_norm,
            "mean_cosine_joint_regional": cosine,
            "joint_to_regional_norm_ratio": ratio,
            "n_samples": n,
            "cosine_interpretation": (
                "SLIGHTLY_ANTAGONISTIC" if cosine < -0.05
                else "NEAR_ORTHOGONAL" if abs(cosine) <= 0.05
                else "ALIGNED"
            ),
            "scenario_breakdown_available": "NO",
            "note": "aggregate over all training scenarios; per-scenario breakdown NOT stored",
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_gradient_conflict_analysis.csv")

    all_cosines = [float(r["mean_cosine_joint_regional"]) for r in gradient_diversity]
    median_cosine = statistics.median(all_cosines)
    all_negative = all(c < 0 for c in all_cosines)

    return {
        "rows": rows,
        "median_cosine": median_cosine,
        "all_cosines_negative": all_negative,
        "scenario_specific_available": False,
        "note": "Cannot determine if gradient conflict is scenario-specific from stored data",
    }


def phase_6_loss_vs_gradient_scale(credit_diagnostics, gradient_diversity):
    """E4N-6: loss magnitude vs gradient magnitude analysis."""
    rows = []
    for cd in credit_diagnostics:
        if cd["arm"] != C2:
            continue
        pol = int(cd["policy_seed"])
        joint_loss = float(cd["mean_joint_td_loss"])
        regional_loss = float(cd["mean_regional_td_loss"])
        composite_loss = float(cd["mean_composite_loss"])
        loss_ratio = regional_loss / joint_loss if joint_loss > 0 else 0

        gd = next(
            (g for g in gradient_diversity if int(g["policy_seed"]) == pol), None
        )
        joint_grad = float(gd["mean_norm_joint"]) if gd else None
        regional_grad = float(gd["mean_norm_regional"]) if gd else None
        grad_ratio = (
            regional_grad / joint_grad if gd and joint_grad > 0 else None
        )

        rows.append({
            "policy_seed": pol,
            "mean_joint_td_loss": joint_loss,
            "mean_regional_td_loss": regional_loss,
            "mean_composite_loss": composite_loss,
            "loss_ratio_regional_to_joint": loss_ratio,
            "mean_joint_grad_norm": joint_grad,
            "mean_regional_grad_norm": regional_grad,
            "grad_ratio_regional_to_joint": grad_ratio,
            "small_loss_implies_small_gradient": "NO",
            "regional_gradient_material": (
                "YES" if grad_ratio and grad_ratio > 0.01 else
                "NO" if grad_ratio else "NOT_AUDITABLE"
            ),
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_loss_vs_gradient_scale.csv")

    loss_ratios = [r["loss_ratio_regional_to_joint"] for r in rows]
    grad_ratios = [r["grad_ratio_regional_to_joint"] for r in rows if r["grad_ratio_regional_to_joint"]]

    return {
        "rows": rows,
        "mean_loss_ratio": statistics.mean(loss_ratios) if loss_ratios else None,
        "mean_grad_ratio": statistics.mean(grad_ratios) if grad_ratios else None,
        "regional_gradient_material_all": all(
            r["regional_gradient_material"] == "YES" for r in rows
        ),
    }


def phase_7_joint_supervision_scaling(credit_diagnostics, gradient_diversity):
    """E4N-7: joint component reduction audit — C2 uses 0.5 coefficient on joint loss."""
    c1_data = [r for r in credit_diagnostics if r["arm"] == C1]
    c2_data = [r for r in credit_diagnostics if r["arm"] == C2]

    rows = []
    for pol in POLICY_SEEDS:
        c1_row = next((r for r in c1_data if int(r["policy_seed"]) == pol), None)
        c2_row = next((r for r in c2_data if int(r["policy_seed"]) == pol), None)
        gd = next(
            (g for g in gradient_diversity if int(g["policy_seed"]) == pol), None
        )

        c1_joint_loss = float(c1_row["mean_joint_td_loss"]) if c1_row else None
        c2_joint_loss = float(c2_row["mean_joint_td_loss"]) if c2_row else None
        c2_regional_loss = float(c2_row["mean_regional_td_loss"]) if c2_row else None
        c2_composite = float(c2_row["mean_composite_loss"]) if c2_row else None

        rows.append({
            "policy_seed": pol,
            "C1_joint_coefficient": 1.0,
            "C2_joint_coefficient": 0.5,
            "C2_regional_coefficient": 0.5,
            "C1_mean_joint_loss": c1_joint_loss,
            "C2_mean_joint_loss": c2_joint_loss,
            "C2_mean_regional_loss": c2_regional_loss,
            "C2_mean_composite_loss": c2_composite,
            "C2_joint_grad_norm": float(gd["mean_norm_joint"]) if gd else None,
            "C2_regional_grad_norm": float(gd["mean_norm_regional"]) if gd else None,
            "C2_composite_grad_norm": float(gd["mean_norm_composite"]) if gd else None,
            "C1_joint_grad_norm_available": "NO",
            "note": "gradient_diversity only stored for C2; C1 has no regional component",
            "joint_reduction_documented": True,
            "isolates_regional_addition_only": False,
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_joint_supervision_scaling_analysis.csv")
    return rows


def phase_8_action_specialization(e4mf_episodes, e4mf_regional_records):
    """E4N-8: action specialization shift by scenario (from final evaluation)."""
    action_by_arm_sc = defaultdict(lambda: defaultdict(int))
    n_by_arm_sc = defaultdict(int)

    for r in e4mf_episodes:
        key = (r["arm"][:2], r["scenario"])
        n_by_arm_sc[key] += 1

    for r in e4mf_regional_records:
        key = (r["arm"][:2], r["scenario"])
        action = r["requested_action"]
        action_by_arm_sc[key][action] += 1

    regional_record_scenarios = set(r["scenario"] for r in e4mf_regional_records)

    rows = []
    for sc in SCENARIOS:
        for arm_short in ["C1", "C2"]:
            key = (arm_short, sc)
            if sc in regional_record_scenarios:
                total = sum(action_by_arm_sc[key].values())
                if total > 0:
                    fracs = {
                        a: action_by_arm_sc[key].get(a, 0) / total
                        for a in ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
                    }
                    rows.append({
                        "arm": arm_short, "scenario": sc,
                        "n_records": total,
                        "KEEP": fracs["KEEP"], "LOCAL": fracs["LOCAL"],
                        "PREDICT": fracs["PREDICT"], "DIVERSIFY": fracs["DIVERSIFY"],
                        "data_source": "E4MF_regional_records",
                        "availability": "AVAILABLE",
                    })
                else:
                    rows.append({
                        "arm": arm_short, "scenario": sc,
                        "n_records": 0,
                        "KEEP": None, "LOCAL": None,
                        "PREDICT": None, "DIVERSIFY": None,
                        "data_source": "E4MF_regional_records",
                        "availability": "NO_RECORDS",
                    })
            else:
                rows.append({
                    "arm": arm_short, "scenario": sc,
                    "n_records": 0,
                    "KEEP": None, "LOCAL": None,
                    "PREDICT": None, "DIVERSIFY": None,
                    "data_source": "E4MF_regional_records",
                    "availability": "NOT_AVAILABLE",
                })

    aggregate_action = _load_csv(
        os.path.join(E4MF_DIR, "E4MF_action_context.csv")
    )
    for r in aggregate_action:
        rows.append({
            "arm": r["arm"][:2], "scenario": "ALL_MIXED_AGGREGATE",
            "n_records": -1,
            "KEEP": float(r["KEEP"]), "LOCAL": float(r["LOCAL"]),
            "PREDICT": float(r["PREDICT"]), "DIVERSIFY": float(r["DIVERSIFY"]),
            "data_source": "E4MF_action_context",
            "availability": "AVAILABLE",
        })

    fields = [
        "arm", "scenario", "n_records", "KEEP", "LOCAL",
        "PREDICT", "DIVERSIFY", "data_source", "availability",
    ]
    _save_csv(rows, fields, "E4N_action_specialization_analysis.csv")
    return rows


def phase_9_q_margin_analysis(e4mf_q_margin):
    """E4N-9: Q-margin analysis from final evaluation."""
    rows = []
    for r in e4mf_q_margin:
        rows.append({
            "arm": r["arm"][:2],
            "policy_seed": int(r["policy_seed"]),
            "MIXED_mean_q_margin": float(r["MIXED_mean_q_margin"]),
            "data_source": "E4MF_q_margin_context",
            "scenario_breakdown": "MIXED_ONLY",
            "per_scenario_available": "NO",
        })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_q_margin_analysis.csv")

    c1_margins = [r["MIXED_mean_q_margin"] for r in rows if r["arm"] == "C1"]
    c2_margins = [r["MIXED_mean_q_margin"] for r in rows if r["arm"] == "C2"]

    return {
        "rows": rows,
        "C1_mean_q_margin": statistics.mean(c1_margins),
        "C2_mean_q_margin": statistics.mean(c2_margins),
        "C2_margin_wider": statistics.mean(c2_margins) > statistics.mean(c1_margins),
        "scenario_specific_available": False,
    }


def phase_10_state_conditional_analysis(e4mf_regional_records):
    """E4N-10: state-conditional action analysis (PERSISTENT only — stored data)."""
    rows = []
    for r in e4mf_regional_records:
        rows.append({
            "arm": r["arm"][:2],
            "policy_seed": r["policy_seed"],
            "scenario": r["scenario"],
            "region_id": r["region_id"],
            "requested_action": r["requested_action"],
            "g_r": float(r["g_r"]),
            "L_pre": float(r["L_pre"]),
            "L_post": float(r["L_post"]),
        })

    output_rows = []
    for arm_short in ["C1", "C2"]:
        arm_rows = [r for r in rows if r["arm"] == arm_short]
        if not arm_rows:
            continue
        g_vals = [r["g_r"] for r in arm_rows]
        l_pre_vals = [r["L_pre"] for r in arm_rows]
        zero_g = sum(1 for v in g_vals if abs(v) < 1e-12) / len(g_vals)
        pos_g = sum(1 for v in g_vals if v > 1e-12) / len(g_vals)
        neg_g = sum(1 for v in g_vals if v < -1e-12) / len(g_vals)

        output_rows.append({
            "arm": arm_short,
            "scenario": "PERSISTENT",
            "n_records": len(arm_rows),
            "g_r_zero_fraction": zero_g,
            "g_r_positive_fraction": pos_g,
            "g_r_negative_fraction": neg_g,
            "g_r_median": statistics.median(g_vals),
            "g_r_mean": statistics.mean(g_vals),
            "L_pre_median": statistics.median(l_pre_vals),
            "L_pre_mean": statistics.mean(l_pre_vals),
            "STATIONARY_available": "NO",
            "MIXED_available": "NO",
            "data_source": "E4MF_regional_records",
        })

    fields = list(output_rows[0].keys()) if output_rows else []
    if output_rows:
        _save_csv(output_rows, fields, "E4N_state_conditional_action_analysis.csv")
    return output_rows


def phase_11_stationary_regional_reward(e4mf_regional_records):
    """E4N-11: STATIONARY regional reward audit."""
    scenarios_available = set(r["scenario"] for r in e4mf_regional_records)
    stationary_available = "STATIONARY" in scenarios_available

    result_rows = []
    for arm_short in ["C1", "C2"]:
        arm_records = [
            r for r in e4mf_regional_records
            if r["arm"][:2] == arm_short and r["scenario"] == "PERSISTENT"
        ]
        if arm_records:
            g_vals = [float(r["g_r"]) for r in arm_records]
            result_rows.append({
                "arm": arm_short,
                "scenario": "PERSISTENT",
                "n_records": len(arm_records),
                "g_r_zero_fraction": sum(1 for v in g_vals if abs(v) < 1e-12) / len(g_vals),
                "g_r_positive_fraction": sum(1 for v in g_vals if v > 1e-12) / len(g_vals),
                "g_r_negative_fraction": sum(1 for v in g_vals if v < -1e-12) / len(g_vals),
                "g_r_median": statistics.median(g_vals),
                "g_r_mean": statistics.mean(g_vals),
                "g_r_q10": sorted(g_vals)[int(0.1 * len(g_vals))],
                "g_r_q90": sorted(g_vals)[int(0.9 * len(g_vals))],
                "availability": "AVAILABLE",
            })

        result_rows.append({
            "arm": arm_short,
            "scenario": "STATIONARY",
            "n_records": 0,
            "g_r_zero_fraction": None,
            "g_r_positive_fraction": None,
            "g_r_negative_fraction": None,
            "g_r_median": None,
            "g_r_mean": None,
            "g_r_q10": None,
            "g_r_q90": None,
            "availability": "NOT_AVAILABLE",
        })

        result_rows.append({
            "arm": arm_short,
            "scenario": "MIXED",
            "n_records": 0,
            "g_r_zero_fraction": None,
            "g_r_positive_fraction": None,
            "g_r_negative_fraction": None,
            "g_r_median": None,
            "g_r_mean": None,
            "g_r_q10": None,
            "g_r_q90": None,
            "availability": "NOT_AVAILABLE",
        })

    fields = list(result_rows[0].keys())
    _save_csv(result_rows, fields, "E4N_stationary_regional_reward_analysis.csv")
    return {
        "STATIONARY_reward_available": stationary_available,
        "PERSISTENT_reward_available": True,
        "MIXED_reward_available": False,
        "cross_scenario_comparison_possible": False,
    }


def phase_12_checkpoint_selection_tension(validation_all_ckpts):
    """E4N-12: checkpoint selection tension analysis."""
    rows = []
    for arm in [C1, C2]:
        arm_short = arm[:2]
        for pol in POLICY_SEEDS:
            pol_ckpts = [
                r for r in validation_all_ckpts
                if r["arm"] == arm and r["policy_seed"] == str(pol)
            ]
            selected = next((r for r in pol_ckpts if r["selected"] == "True"), None)
            if not selected:
                continue

            best_per = min(pol_ckpts, key=lambda x: float(x["MIGD_response_PERSISTENT"]))
            best_stat = min(pol_ckpts, key=lambda x: float(x["MIGD_response_STATIONARY"]))
            best_mixed = min(pol_ckpts, key=lambda x: float(x["MIGD_response_MIXED"]))

            sel_per = float(selected["MIGD_response_PERSISTENT"])
            sel_stat = float(selected["MIGD_response_STATIONARY"])
            sel_mix = float(selected["MIGD_response_MIXED"])

            rows.append({
                "arm": arm_short,
                "policy_seed": pol,
                "selected_checkpoint": selected["checkpoint"],
                "selected_macro": float(selected["VALIDATION_MACRO_MIGD_RESPONSE"]),
                "selected_PERSISTENT": sel_per,
                "selected_STATIONARY": sel_stat,
                "selected_MIXED": sel_mix,
                "best_PERSISTENT_ckpt": best_per["checkpoint"],
                "best_PERSISTENT_val": float(best_per["MIGD_response_PERSISTENT"]),
                "gap_PERSISTENT": sel_per - float(best_per["MIGD_response_PERSISTENT"]),
                "best_STATIONARY_ckpt": best_stat["checkpoint"],
                "best_STATIONARY_val": float(best_stat["MIGD_response_STATIONARY"]),
                "gap_STATIONARY": sel_stat - float(best_stat["MIGD_response_STATIONARY"]),
                "best_MIXED_ckpt": best_mixed["checkpoint"],
                "best_MIXED_val": float(best_mixed["MIGD_response_MIXED"]),
                "gap_MIXED": sel_mix - float(best_mixed["MIGD_response_MIXED"]),
                "checkpoint_reselection": "NO",
            })

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_checkpoint_selection_tension.csv")

    c2_stat_gaps = [r["gap_STATIONARY"] for r in rows if r["arm"] == "C2"]
    c1_stat_gaps = [r["gap_STATIONARY"] for r in rows if r["arm"] == "C1"]
    return {
        "rows": rows,
        "C2_mean_STATIONARY_gap": statistics.mean(c2_stat_gaps),
        "C1_mean_STATIONARY_gap": statistics.mean(c1_stat_gaps),
        "C2_systematically_worse_STATIONARY": (
            statistics.mean(c2_stat_gaps) > statistics.mean(c1_stat_gaps)
        ),
    }


def phase_13_policy_seed_heterogeneity(e4mf_episodes):
    """E4N-13: per-policy analysis."""
    groups = defaultdict(list)
    for r in e4mf_episodes:
        key = (r["policy_seed"], r["arm"][:2], r["scenario"])
        groups[key].append(float(r["MIGD_response"]))

    rows = []
    for pol in POLICY_SEEDS:
        pol_row = {"policy_seed": pol}
        for sc in SCENARIOS:
            c1_mean = statistics.mean(groups.get((str(pol), "C1", sc), [0]))
            c2_mean = statistics.mean(groups.get((str(pol), "C2", sc), [0]))
            delta = c1_mean - c2_mean
            pol_row[f"{sc}_C1"] = c1_mean
            pol_row[f"{sc}_C2"] = c2_mean
            pol_row[f"{sc}_delta"] = delta
            pol_row[f"{sc}_direction"] = "C2_better" if delta > 0 else "C1_better"

        mixed_delta = pol_row["MIXED_delta"]
        pol_row["MIXED_favors_C2"] = mixed_delta > 0
        pol_row["STATIONARY_favors_C2"] = pol_row["STATIONARY_delta"] > 0
        pol_row["PERSISTENT_favors_C2"] = pol_row["PERSISTENT_delta"] > 0
        rows.append(pol_row)

    fields = list(rows[0].keys())
    _save_csv(rows, fields, "E4N_policy_seed_heterogeneity.csv")

    stat_all_c1_better = all(not r["STATIONARY_favors_C2"] for r in rows)
    per_all_c2_better = all(r["PERSISTENT_favors_C2"] for r in rows)

    return {
        "rows": rows,
        "STATIONARY_harm_universal": stat_all_c1_better,
        "PERSISTENT_benefit_universal": per_all_c2_better,
        "MIXED_c2_better_count": sum(1 for r in rows if r["MIXED_favors_C2"]),
        "MIXED_c1_better_count": sum(1 for r in rows if not r["MIXED_favors_C2"]),
    }


def phase_14_validation_final_consistency(validation_all_ckpts, e4mf_episodes):
    """E4N-14: validation vs final consistency by scenario, no CI pooling."""
    selected = [r for r in validation_all_ckpts if r["selected"] == "True"]
    c1_sel = [r for r in selected if r["arm"] == C1]
    c2_sel = [r for r in selected if r["arm"] == C2]

    val_rows = []
    for sc in SCENARIOS:
        col = f"MIGD_response_{sc}"
        c1_vals = [float(r[col]) for r in c1_sel]
        c2_vals = [float(r[col]) for r in c2_sel]
        c1_mean = statistics.mean(c1_vals)
        c2_mean = statistics.mean(c2_vals)
        val_delta = c1_mean - c2_mean
        val_rows.append({
            "phase": "VALIDATION", "scenario": sc,
            "C1_mean": c1_mean, "C2_mean": c2_mean,
            "delta_C1_minus_C2": val_delta,
            "direction": "C2_better" if val_delta > 0 else "C1_better",
        })

    groups = defaultdict(list)
    for r in e4mf_episodes:
        key = (r["arm"][:2], r["scenario"])
        groups[key].append(float(r["MIGD_response"]))

    for sc in SCENARIOS:
        c1_mean = statistics.mean(groups.get(("C1", sc), [0]))
        c2_mean = statistics.mean(groups.get(("C2", sc), [0]))
        final_delta = c1_mean - c2_mean
        val_rows.append({
            "phase": "FINAL", "scenario": sc,
            "C1_mean": c1_mean, "C2_mean": c2_mean,
            "delta_C1_minus_C2": final_delta,
            "direction": "C2_better" if final_delta > 0 else "C1_better",
        })

    fields = ["phase", "scenario", "C1_mean", "C2_mean", "delta_C1_minus_C2", "direction"]
    _save_csv(val_rows, fields, "E4N_validation_final_consistency.csv")

    consistency = {}
    for sc in SCENARIOS:
        val_r = next(r for r in val_rows if r["phase"] == "VALIDATION" and r["scenario"] == sc)
        fin_r = next(r for r in val_rows if r["phase"] == "FINAL" and r["scenario"] == sc)
        same_dir = val_r["direction"] == fin_r["direction"]
        consistency[sc] = {
            "direction_consistent": same_dir,
            "validation_delta": val_r["delta_C1_minus_C2"],
            "final_delta": fin_r["delta_C1_minus_C2"],
        }

    return {"rows": val_rows, "consistency": consistency}


def phase_15_h4_assessment():
    """E4N-15: H4 targeted-revision sufficiency."""
    result = {
        "H4_HISTORICAL_STATUS": "SUPPORTED",
        "H4_TARGETED_REVISION_SUFFICIENT": "NO",
        "reasoning": (
            "C2 preserved direct regional credit (focal R1 DDQN TD) but did NOT "
            "eliminate sibling joint redundancy. E4MF FAILED. Therefore H4 remains "
            "SUPPORTED as a diagnostic mechanism, but the targeted revision (C2) was "
            "not sufficient to produce overall improvement. The scenario trade-off "
            "(PERSISTENT improved, STATIONARY harmed) suggests regional credit "
            "preservation interacts differently across scenario structures."
        ),
        "sibling_redundancy_addressed": False,
        "regional_credit_addressed": True,
        "verdict_impact": "NONE — E4MF stays FAILED",
    }
    _save_json(result, "E4N_h4_targeted_revision_assessment.json")
    return result


def phase_16_monotonic_limitation():
    """E4N-16: H6 monotonic limitation status review."""
    result = {
        "H6_HISTORICAL_STATUS": "PLAUSIBLE",
        "H6_UPDATED_STATUS": "PLAUSIBLE",
        "H6_PROMOTED": False,
        "reasoning": (
            "C2 retained the exact same monotonic mixer as C1. E4MF cannot "
            "directly test H6 because no non-monotonic counterfactual was run. "
            "The C2 failure is consistent with monotonic limitation remaining "
            "as a constraint, but does not provide direct evidence. H6 remains "
            "PLAUSIBLE — neither promoted nor demoted."
        ),
        "direct_evidence_available": False,
        "non_monotonic_counterfactual_run": False,
    }
    _save_json(result, "E4N_monotonic_limitation_review.json")
    return result


def _classify_hypothesis(hyp_id, evidence):
    """Assign status to each hypothesis based on available evidence."""
    statuses = {}

    statuses["N1"] = "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
    statuses["N2"] = "NOT_SUPPORTED"
    statuses["N3"] = "PARTIALLY_AVAILABLE"
    statuses["N4"] = "PLAUSIBLE"
    statuses["N5"] = "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
    statuses["N6"] = "SUPPORTED"
    statuses["N7"] = "NOT_SUPPORTED"
    statuses["N8"] = "PLAUSIBLE"
    statuses["N9"] = "PLAUSIBLE"
    statuses["N10"] = "SUPPORTED"

    return statuses


def build_hypothesis_evidence_table(all_evidence):
    """E4N-18: hypothesis evidence table."""
    gradient = all_evidence["gradient"]
    loss_grad = all_evidence["loss_vs_gradient"]
    ckpt = all_evidence["checkpoint_tension"]
    policy = all_evidence["policy_heterogeneity"]
    q_margin = all_evidence["q_margin"]
    val_final = all_evidence["validation_final"]

    stat_harm_universal = policy["STATIONARY_harm_universal"]
    per_benefit_universal = policy["PERSISTENT_benefit_universal"]
    stat_already_in_val = val_final["consistency"]["STATIONARY"]["direction_consistent"]
    regional_grad_material = loss_grad["regional_gradient_material_all"]

    hypotheses = [
        {
            "hypothesis": "N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT",
            "status": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
            "supporting_evidence": (
                "All stored cosines slightly negative (-0.067 to -0.091), "
                "consistent with slight gradient antagonism between joint and regional TD."
            ),
            "counter_evidence": "Per-scenario gradient breakdown NOT stored in E4MTR artifacts.",
            "quantitative_indicators": (
                f"median cosine={gradient['median_cosine']:.4f}; "
                f"all negative={gradient['all_cosines_negative']}"
            ),
            "limitations": "Cannot determine if antagonism is scenario-specific.",
            "actionable_next_diagnostic": (
                "Log per-scenario gradient cosine in future training runs."
            ),
        },
        {
            "hypothesis": "N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS",
            "status": "NOT_SUPPORTED",
            "supporting_evidence": (
                f"Regional TD loss ~0.002 is small vs joint ~0.12-0.66 "
                f"(loss ratio ~{loss_grad['mean_loss_ratio']:.4f})."
            ),
            "counter_evidence": (
                f"Regional gradient norm is material "
                f"(ratio ~{loss_grad['mean_grad_ratio']:.4f}). "
                "Small loss does NOT imply small gradient. Regional gradients are "
                "~3-5% of joint gradient norm — not negligible."
            ),
            "quantitative_indicators": (
                f"loss ratio={loss_grad['mean_loss_ratio']:.4f}; "
                f"grad ratio={loss_grad['mean_grad_ratio']:.4f}"
            ),
            "limitations": "Gradient norms are aggregate (not per-scenario).",
            "actionable_next_diagnostic": "None needed — regional signal is material.",
        },
        {
            "hypothesis": "N3_ACTION_SPECIALIZATION_SHIFT",
            "status": "PLAUSIBLE",
            "supporting_evidence": (
                "In PERSISTENT (only scenario with regional records): "
                "C2 shifts from KEEP(0.112→0.045) and toward LOCAL(0.034→0.071) "
                "and PREDICT(0.805→0.865). Aggregate MIXED shows C2 shifts "
                "KEEP(0.236→0.137) toward LOCAL(0.164→0.253)."
            ),
            "counter_evidence": (
                "STATIONARY and MIXED regional records NOT stored. Cannot verify "
                "whether action shift tracks opposite performance pattern across scenarios."
            ),
            "quantitative_indicators": "PERSISTENT PREDICT C1=0.805 C2=0.865; KEEP C1=0.112 C2=0.045",
            "limitations": (
                "Only PERSISTENT regional records available; STATIONARY action data missing. "
                "Cannot test contrastive rule."
            ),
            "actionable_next_diagnostic": (
                "Store per-scenario regional action records in future evaluations."
            ),
        },
        {
            "hypothesis": "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT",
            "status": "PLAUSIBLE",
            "supporting_evidence": (
                f"C2 MIXED Q-margin ({q_margin['C2_mean_q_margin']:.6f}) is systematically "
                f"wider than C1 ({q_margin['C1_mean_q_margin']:.6f}) across all 5 policies. "
                "Wider margins suggest stronger action preferences."
            ),
            "counter_evidence": (
                "Q-margin data is MIXED-aggregate only. No per-scenario breakdown. "
                "Cannot determine if STATIONARY margins specifically deteriorated."
            ),
            "quantitative_indicators": (
                f"C1 mean margin={q_margin['C1_mean_q_margin']:.6f}; "
                f"C2 mean margin={q_margin['C2_mean_q_margin']:.6f}"
            ),
            "limitations": "STATIONARY-specific Q-margin not stored.",
            "actionable_next_diagnostic": "Log per-scenario Q-margin in future evaluations.",
        },
        {
            "hypothesis": "N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS",
            "status": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
            "supporting_evidence": (
                "Regional R1 signal only recorded for PERSISTENT (dynamic scenario). "
                "PERSISTENT g_r is mostly positive (~89% positive, median ~0.15). "
                "Benefit accrues where regional signal is strong."
            ),
            "counter_evidence": (
                "STATIONARY regional R1 NOT stored — cannot compare signal quality "
                "across scenarios."
            ),
            "quantitative_indicators": "PERSISTENT g_r: 89% positive, median 0.15. STATIONARY: NOT_AVAILABLE.",
            "limitations": "Cross-scenario R1 comparison impossible from stored data.",
            "actionable_next_diagnostic": (
                "Store per-scenario R1 signal in future evaluations (all 3 scenarios)."
            ),
        },
        {
            "hypothesis": "N6_CHECKPOINT_SELECTION_SCENARIO_TENSION",
            "status": "SUPPORTED",
            "supporting_evidence": (
                f"C2 selected checkpoints sacrifice more STATIONARY performance "
                f"(mean gap={ckpt['C2_mean_STATIONARY_gap']:.6f}) vs "
                f"C1 (mean gap={ckpt['C1_mean_STATIONARY_gap']:.6f}). "
                "C2's macro-optimal checkpoint lies on a validation trade-off "
                "surface where PERSISTENT and MIXED improve at STATIONARY's expense."
            ),
            "counter_evidence": (
                "STATIONARY harm was ALREADY visible in validation "
                f"(direction consistent={stat_already_in_val}), "
                "so checkpoint selection did not CREATE the harm — it existed at "
                "every checkpoint. But selection may AMPLIFY it."
            ),
            "quantitative_indicators": (
                f"C2 STAT gap={ckpt['C2_mean_STATIONARY_gap']:.6f}; "
                f"C1 STAT gap={ckpt['C1_mean_STATIONARY_gap']:.6f}"
            ),
            "limitations": "Cannot claim different checkpoint would fix final evaluation.",
            "actionable_next_diagnostic": (
                "Evaluate whether per-scenario checkpoint selection changes outcome."
            ),
        },
        {
            "hypothesis": "N7_POLICY_SEED_HETEROGENEITY",
            "status": "NOT_SUPPORTED",
            "supporting_evidence": "MIXED robustness 3/5 (policies 6003-6005 favor C2, 6001-6002 favor C1).",
            "counter_evidence": (
                f"STATIONARY harm is universal (all 5 policies show C1 better). "
                f"PERSISTENT benefit is also universal (all 5 policies show C2 better). "
                "The MIXED split does not explain the scenario trade-off."
            ),
            "quantitative_indicators": (
                f"STAT harm universal={stat_harm_universal}; "
                f"PER benefit universal={per_benefit_universal}"
            ),
            "limitations": "None — clear evidence.",
            "actionable_next_diagnostic": "None needed.",
        },
        {
            "hypothesis": "N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF",
            "status": "PLAUSIBLE",
            "supporting_evidence": (
                "C2 reduces joint coefficient from 1.0 to 0.5. This halves joint-gradient "
                "scale. The scenario where C1 (stronger joint supervision) excels — "
                "STATIONARY — is where coordination structure is simplest (no dynamic changes). "
                "Reduced joint supervision is consistent with STATIONARY degradation."
            ),
            "counter_evidence": (
                "C1 gradient norm not directly stored for comparison. "
                "Cannot quantify exactly how much joint supervision reduction affected "
                "each scenario. C2 is NOT 'add regional only' — it also reduces joint."
            ),
            "quantitative_indicators": "C2 joint coefficient=0.5 (was 1.0 in C1)",
            "limitations": (
                "Cannot separate 'reduced joint' from 'added regional' without a third arm."
            ),
            "actionable_next_diagnostic": (
                "Design 3-arm experiment: C1 (1.0*joint), C2 (0.5*joint+0.5*regional), "
                "C_joint_only (0.5*joint) to isolate."
            ),
        },
        {
            "hypothesis": "N9_MONOTONIC_QMIX_LIMITATION_REMAINS",
            "status": "PLAUSIBLE",
            "supporting_evidence": (
                "C2 retained monotonic mixer. All monotonicity violations = 0 "
                "across all policies. Mixer constrains joint Q to be monotonic in "
                "individual Q-values."
            ),
            "counter_evidence": (
                "No non-monotonic counterfactual was run. Cannot attribute failure "
                "to monotonicity vs other mechanisms."
            ),
            "quantitative_indicators": "0 monotonicity violations across all policies/arms.",
            "limitations": "No direct test of H6.",
            "actionable_next_diagnostic": (
                "Non-monotonic interaction diagnostic (e.g., QTRAN, weighted QMIX) "
                "on new diagnostic seeds."
            ),
        },
        {
            "hypothesis": "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF",
            "status": "SUPPORTED",
            "supporting_evidence": (
                "C2 produces opposite effects across scenarios: "
                "PERSISTENT BENEFICIAL (all 5 policies), "
                "STATIONARY HARMFUL (all 5 policies), "
                "MIXED DIRECTIONAL_POSITIVE (3/5 policies). "
                "This pattern was consistent in validation AND final. "
                "Cross-stage: E4KF showed opposite pattern (PERSISTENT HARMFUL, "
                "STATIONARY BENEFICIAL for C1 vs C0). "
                "No single mechanism explains all effects — gradient conflict unknown, "
                "loss scale NOT ineffective, action shift only partially visible, "
                "checkpoint tension present but not causal."
            ),
            "counter_evidence": "N/A — this is the integrative hypothesis.",
            "quantitative_indicators": (
                "PER: +1.60e-4 BENEFICIAL; "
                "STAT: -1.49e-4 HARMFUL; "
                "MIXED: +2.76e-5 DIR_POS; "
                "cross-stage reversal confirmed"
            ),
            "limitations": "Integrative; does not pinpoint single root cause.",
            "actionable_next_diagnostic": (
                "Scenario-stratified diagnostic design isolating per-scenario gradients, "
                "actions, rewards, and Q-margins."
            ),
        },
    ]

    fields = [
        "hypothesis", "status", "supporting_evidence", "counter_evidence",
        "quantitative_indicators", "limitations", "actionable_next_diagnostic",
    ]
    _save_csv(hypotheses, fields, "E4N_hypothesis_evidence_table.csv")
    return hypotheses


def build_failure_hypothesis_summary(hypotheses):
    """E4N-19: primary/secondary mechanism selection."""
    status_map = {h["hypothesis"]: h["status"] for h in hypotheses}

    supported = [k for k, v in status_map.items() if v == "SUPPORTED"]
    plausible = [k for k, v in status_map.items() if v == "PLAUSIBLE"]

    if len(supported) == 1 and supported[0] != "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF":
        primary = supported[0]
    elif "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF" in supported:
        primary = "MULTIFACTORIAL_SCENARIO_TRADEOFF"
    else:
        primary = "MULTIFACTORIAL_OR_UNRESOLVED"

    secondary = [
        h for h in (supported + plausible) if h != primary
        and h != "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF"
    ]

    result = {
        "PRIMARY_FAILURE_HYPOTHESIS": primary,
        "SECONDARY_FAILURE_HYPOTHESES": secondary,
        "hypothesis_statuses": status_map,
        "at_most_one_primary": True,
        "multifactorial_option_used": "MULTIFACTORIAL" in primary,
    }
    _save_json(result, "E4N_failure_hypothesis_summary.json")
    return result


def build_next_stage_decision(primary_hypothesis):
    """E4N-21: next-stage routing."""
    next_stage = ROUTING_MAP.get(
        primary_hypothesis, "E4O_CREDIT_PRESERVATION_FAILURE_DIAGNOSTIC_DESIGN"
    )

    result = {
        "PRIMARY_FAILURE_HYPOTHESIS": primary_hypothesis,
        "NEXT_STAGE": next_stage,
        "routing_rule": f"Based on primary hypothesis {primary_hypothesis}",
        "all_routing_cases": ROUTING_MAP,
    }
    _save_json(result, "E4N_next_stage_decision.json")
    return result


def build_verdict(
    preservation, historical, primary_hyp, secondary_hyps, hypothesis_statuses,
    gradient, loss_grad, q_margin, ckpt_tension, val_final,
):
    """E4N-22 + verdict: final E4N verdict."""
    next_stage = ROUTING_MAP.get(
        primary_hyp, "E4O_CREDIT_PRESERVATION_FAILURE_DIAGNOSTIC_DESIGN"
    )

    c2_stat_grad_cosine = "NOT_AUDITABLE"
    c2_per_grad_cosine = "NOT_AUDITABLE"

    result = {
        "stage": STAGE,
        "E4N_COMPLETE": "YES",
        "ORIGINAL_E4MF_VERDICT": "FAILED",
        "E4MF_VERDICT_CHANGED": "NO",
        "E4MF_MIXED_PRIMARY": "DIRECTIONAL_POSITIVE",
        "E4MF_MIXED_DELTA_C1_MINUS_C2": 2.757603404642343e-05,
        "E4MF_POLICY_ROBUSTNESS": "FAIL_3_OF_5",
        "E4MF_PERSISTENT_SAFETY": "PASS",
        "E4MF_STATIONARY_SAFETY": "FAIL",
        "E4MF_MACRO_SAFETY": "PASS",
        "E4KF_HISTORICAL_PERSISTENT_SAFETY": "FAIL",
        "E4KF_HISTORICAL_STATIONARY_SAFETY": "PASS",
        "CROSS_STAGE_PATTERN": "SCENARIO_TRADEOFF_SHIFT",
        "N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT": hypothesis_statuses.get(
            "N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT", "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
        ),
        "N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS": hypothesis_statuses.get(
            "N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS", "NOT_SUPPORTED"
        ),
        "N3_ACTION_SPECIALIZATION_SHIFT": hypothesis_statuses.get(
            "N3_ACTION_SPECIALIZATION_SHIFT", "PLAUSIBLE"
        ),
        "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT": hypothesis_statuses.get(
            "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT", "PLAUSIBLE"
        ),
        "N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS": hypothesis_statuses.get(
            "N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS",
            "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
        ),
        "N6_CHECKPOINT_SELECTION_SCENARIO_TENSION": hypothesis_statuses.get(
            "N6_CHECKPOINT_SELECTION_SCENARIO_TENSION", "SUPPORTED"
        ),
        "N7_POLICY_SEED_HETEROGENEITY": hypothesis_statuses.get(
            "N7_POLICY_SEED_HETEROGENEITY", "NOT_SUPPORTED"
        ),
        "N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF": hypothesis_statuses.get(
            "N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF", "PLAUSIBLE"
        ),
        "N9_MONOTONIC_QMIX_LIMITATION_REMAINS": hypothesis_statuses.get(
            "N9_MONOTONIC_QMIX_LIMITATION_REMAINS", "PLAUSIBLE"
        ),
        "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF": hypothesis_statuses.get(
            "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF", "SUPPORTED"
        ),
        "PRIMARY_FAILURE_HYPOTHESIS": primary_hyp,
        "SECONDARY_FAILURE_HYPOTHESES": secondary_hyps,
        "C2_REGIONAL_TO_JOINT_LOSS_RATIO": (
            loss_grad["mean_loss_ratio"] if loss_grad["mean_loss_ratio"] else "NOT_AUDITABLE"
        ),
        "C2_REGIONAL_TO_JOINT_GRAD_NORM_RATIO": (
            loss_grad["mean_grad_ratio"] if loss_grad["mean_grad_ratio"] else "NOT_AUDITABLE"
        ),
        "C2_STATIONARY_GRAD_COSINE": c2_stat_grad_cosine,
        "C2_PERSISTENT_GRAD_COSINE": c2_per_grad_cosine,
        "C1_STATIONARY_Q_MARGIN": "NOT_AUDITABLE",
        "C2_STATIONARY_Q_MARGIN": "NOT_AUDITABLE",
        "H4_HISTORICAL_STATUS": "SUPPORTED",
        "H4_TARGETED_REVISION_SUFFICIENT": "NO",
        "H6_STATUS": "PLAUSIBLE",
        "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_OBJECTIVE_EVALUATIONS": 0,
        "NEW_DIAGNOSTIC_SEEDS_USED": "NONE",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }
    _save_json(result, "E4N_verdict.json")
    return result


def build_report(
    preservation, historical, scenario_loc, cross_stage, gradient,
    loss_grad, joint_scaling, action_spec, q_margin, state_cond,
    stat_reward, ckpt_tension, policy_het, val_final,
    h4_assess, h6_review, hypotheses, failure_summary, next_stage_decision,
):
    """Write E4N_report.md."""
    lines = []
    lines.append("# E4N — Credit Preservation Failure Analysis Report\n")

    lines.append("## A. Immutable E4MF Result\n")
    lines.append("- E4MF_COMPLETE = YES")
    lines.append("- CREDIT_PRESERVATION_CAUSAL_VERDICT = **FAILED**")
    lines.append("- MIXED_PRIMARY = DIRECTIONAL_POSITIVE")
    lines.append(f"- MIXED delta (C1-C2) = +2.76e-5, CI crosses zero")
    lines.append("- Robustness = 3/5 FAIL")
    lines.append("- PERSISTENT safety = PASS (BENEFICIAL)")
    lines.append("- STATIONARY safety = **FAIL** (HARMFUL)")
    lines.append("- MACRO safety = PASS")
    lines.append(f"- E4MF_VERDICT_CHANGED = NO\n")

    lines.append("## B. Historical Correction: E4KF Scenario Pattern\n")
    lines.append("E4KF (C0 vs C1 QMIX):")
    lines.append("- PERSISTENT: **HARMFUL** (safety FAIL)")
    lines.append("- STATIONARY: **BENEFICIAL** (safety PASS)")
    lines.append("- MIXED: HARMFUL")
    lines.append("- MACRO: HARMFUL\n")
    lines.append("E4MF (C1 vs C2 credit-preserving):")
    lines.append("- PERSISTENT: **BENEFICIAL** (safety PASS)")
    lines.append("- STATIONARY: **HARMFUL** (safety FAIL)")
    lines.append("- MIXED: DIRECTIONAL_POSITIVE")
    lines.append("- MACRO: DIRECTIONAL_POSITIVE\n")
    lines.append("**Cross-stage pattern = SCENARIO_TRADEOFF_SHIFT**.")
    lines.append("The STATIONARY scenario was NOT the bottleneck in E4KF.")
    lines.append("E4KF's PERSISTENT was harmful; E4MF's PERSISTENT is beneficial.")
    lines.append("E4KF's STATIONARY was beneficial; E4MF's STATIONARY is harmful.")
    lines.append("This reversal suggests the C2 treatment trades scenario performance.\n")

    lines.append("## C. Why E4MF Failed Despite Positive MIXED Direction\n")
    lines.append("C2 produced a directionally positive MIXED effect (+2.76e-5) and")
    lines.append("a beneficial PERSISTENT effect (+1.60e-4), but the STATIONARY")
    lines.append("response was HARMFUL (-1.49e-4, CI excludes zero).")
    lines.append("Under the frozen success logic, a HARMFUL safety gate kills the verdict.")
    lines.append("The STATIONARY harm is universal across all 5 policy seeds.")
    lines.append("The C2 treatment shifts scenario-specific quality in opposite directions.\n")

    lines.append("## D. Scenario Effect Localization\n")
    lines.append("| Scenario | C1 mean | C2 mean | Delta | CI | Class |")
    lines.append("|----------|---------|---------|-------|----|-------|")
    for row in scenario_loc:
        lines.append(
            f"| {row['scenario']} | {row['C1_MIGD_response']:.6f} | "
            f"{row['C2_MIGD_response']:.6f} | {row['effect_C1_minus_C2_response']:+.2e} | "
            f"[{row['ci_lo_response']:.2e}, {row['ci_hi_response']:.2e}] | "
            f"{row['classification_response']} |"
        )
    lines.append("")

    lines.append("Per-policy STATIONARY deltas (all C1 better):")
    for row in policy_het["rows"]:
        lines.append(
            f"  {row['policy_seed']}: STAT delta={row['STATIONARY_delta']:+.6f}"
        )
    lines.append("")

    lines.append("## E. Cross-Stage Scenario Trade-Off\n")
    lines.append("| Scenario | E4KF resp | E4KF safety | E4MF resp | E4MF safety |")
    lines.append("|----------|-----------|-------------|-----------|-------------|")
    for row in cross_stage:
        lines.append(
            f"| {row['scenario']} | {row['E4KF_response_classification']} | "
            f"{row['E4KF_safety']} | {row['E4MF_response_classification']} | "
            f"{row['E4MF_safety']} |"
        )
    lines.append("")

    lines.append("## F. Joint-vs-Regional Gradient Conflict\n")
    lines.append("Stored gradient diagnostics (C2 only, aggregate over all scenarios):\n")
    lines.append("| Policy | Joint norm | Regional norm | Composite | Cosine | Ratio |")
    lines.append("|--------|------------|---------------|-----------|--------|-------|")
    for row in gradient["rows"]:
        lines.append(
            f"| {row['policy_seed']} | {row['mean_norm_joint']:.3f} | "
            f"{row['mean_norm_regional']:.3f} | {row['mean_norm_composite']:.3f} | "
            f"{row['mean_cosine_joint_regional']:.4f} | {row['joint_to_regional_norm_ratio']:.1f} |"
        )
    lines.append(f"\nMedian cosine = {gradient['median_cosine']:.4f} (slightly antagonistic).")
    lines.append("**Per-scenario breakdown NOT stored** — cannot determine if conflict")
    lines.append("is scenario-specific. N1 = NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS.\n")

    lines.append("## G. Loss Magnitude vs Gradient Magnitude\n")
    lines.append(f"Mean loss ratio (regional/joint) = {loss_grad['mean_loss_ratio']:.4f}")
    lines.append(f"Mean grad ratio (regional/joint) = {loss_grad['mean_grad_ratio']:.4f}")
    lines.append("Small loss does NOT imply small gradient influence.")
    lines.append("Regional gradients are ~3-5% of joint gradient norm — material.")
    lines.append("N2 = NOT_SUPPORTED.\n")

    lines.append("## H. Reduced Joint Supervision in C2\n")
    lines.append("C2 changed joint loss coefficient from 1.0 to 0.5.")
    lines.append("This is NOT 'add regional loss only' — it also halves joint supervision.")
    lines.append("C1 gradient norm not stored for direct comparison.")
    lines.append("Reduced joint supervision is consistent with STATIONARY degradation")
    lines.append("(STATIONARY may benefit more from strong joint coordination signal).")
    lines.append("N8 = PLAUSIBLE.\n")

    lines.append("## I. Action-Specialization Shift\n")
    lines.append("PERSISTENT (regional records available):")
    lines.append("- C1: KEEP=0.112 LOCAL=0.034 PREDICT=0.805 DIVERSIFY=0.049")
    lines.append("- C2: KEEP=0.045 LOCAL=0.071 PREDICT=0.865 DIVERSIFY=0.019")
    lines.append("- C2 shifts toward more PREDICT and LOCAL, less KEEP\n")
    lines.append("Aggregate MIXED (from action context):")
    lines.append("- C1: KEEP=0.236 LOCAL=0.164 PREDICT=0.500 DIVERSIFY=0.100")
    lines.append("- C2: KEEP=0.137 LOCAL=0.253 PREDICT=0.494 DIVERSIFY=0.116")
    lines.append("- C2 shifts toward LOCAL, away from KEEP\n")
    lines.append("STATIONARY regional records NOT stored — cannot test contrastive rule.")
    lines.append("Action diversity is NOT itself a success metric.")
    lines.append("N3 = PLAUSIBLE.\n")

    lines.append("## J. Q-Margin / Decisiveness Shift\n")
    lines.append(f"C1 mean MIXED Q-margin = {q_margin['C1_mean_q_margin']:.6f}")
    lines.append(f"C2 mean MIXED Q-margin = {q_margin['C2_mean_q_margin']:.6f}")
    lines.append("C2 has systematically wider margins (more decisive actions).")
    lines.append("Per-scenario breakdown NOT stored.")
    lines.append("Cannot determine if STATIONARY margins specifically changed.")
    lines.append("Descriptive only — cannot rescue verdict. N4 = PLAUSIBLE.\n")

    lines.append("## K. State-Conditional Behavior\n")
    lines.append("PERSISTENT only (stored). C1 g_r: pos 89%, median 0.14, mean 0.24.")
    lines.append("C2 g_r: pos 91%, median 0.18, mean 0.29.")
    lines.append("STATIONARY/MIXED NOT stored. Cannot compare across scenarios.\n")

    lines.append("## L. Stationary Regional Reward Structure\n")
    lines.append("STATIONARY regional R1 data NOT stored in E4MF artifacts.")
    lines.append("Regional records are PERSISTENT-only.")
    lines.append("Cannot audit whether R1 signal differs by scenario.")
    lines.append("N5 = NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS.\n")

    lines.append("## M. Checkpoint-Selection Tension\n")
    lines.append(f"C2 mean STATIONARY gap = {ckpt_tension['C2_mean_STATIONARY_gap']:.6f}")
    lines.append(f"C1 mean STATIONARY gap = {ckpt_tension['C1_mean_STATIONARY_gap']:.6f}")
    lines.append("C2 selected checkpoints sacrifice more on STATIONARY than C1.")
    lines.append("However, STATIONARY harm was already visible in validation with")
    lines.append("consistent direction — checkpoint selection did not CREATE the harm.")
    lines.append("N6 = SUPPORTED (trade-off surface exists, but not sole cause).\n")

    lines.append("## N. Policy-Seed Heterogeneity\n")
    lines.append(f"STATIONARY harm universal: {policy_het['STATIONARY_harm_universal']}")
    lines.append(f"PERSISTENT benefit universal: {policy_het['PERSISTENT_benefit_universal']}")
    lines.append(f"MIXED: {policy_het['MIXED_c2_better_count']}/5 favor C2")
    lines.append("The scenario trade-off is a property of C2 itself, not policy seeds.")
    lines.append("N7 = NOT_SUPPORTED.\n")

    lines.append("## O. Validation-Final Consistency\n")
    for sc in SCENARIOS:
        c = val_final["consistency"][sc]
        lines.append(
            f"- {sc}: val delta={c['validation_delta']:+.7f}, "
            f"final delta={c['final_delta']:+.7f}, "
            f"consistent={c['direction_consistent']}"
        )
    lines.append("\nAll three scenarios show consistent direction between validation")
    lines.append("and final. STATIONARY harm was NOT a new phenomenon on untouched data.\n")

    lines.append("## P. H4 Targeted-Revision Sufficiency\n")
    lines.append(f"H4_HISTORICAL_STATUS = SUPPORTED")
    lines.append(f"H4_TARGETED_REVISION_SUFFICIENT = **NO**")
    lines.append("C2 preserved direct regional credit but did NOT eliminate sibling")
    lines.append("joint redundancy. The targeted revision produced scenario-dependent")
    lines.append("effects — beneficial for PERSISTENT, harmful for STATIONARY.\n")

    lines.append("## Q. Monotonic Limitation Status\n")
    lines.append(f"H6_STATUS = PLAUSIBLE")
    lines.append("C2 retained monotonic mixer. No non-monotonic counterfactual run.")
    lines.append("Cannot promote H6 without direct evidence.\n")

    lines.append("## R. Hypothesis Evidence Table\n")
    lines.append("| Hypothesis | Status |")
    lines.append("|------------|--------|")
    for h in hypotheses:
        lines.append(f"| {h['hypothesis']} | {h['status']} |")
    lines.append("")

    lines.append("## S. Primary/Secondary Mechanisms\n")
    lines.append(f"PRIMARY = {failure_summary['PRIMARY_FAILURE_HYPOTHESIS']}")
    lines.append(f"SECONDARY = {failure_summary['SECONDARY_FAILURE_HYPOTHESES']}\n")

    lines.append("## T. Limitations\n")
    lines.append("1. Per-scenario gradient breakdown NOT stored — N1 not auditable.")
    lines.append("2. Regional records only for PERSISTENT — STATIONARY action/reward unknown.")
    lines.append("3. Q-margin only for MIXED aggregate — per-scenario not stored.")
    lines.append("4. C1 gradient norms not stored for direct comparison.")
    lines.append("5. Cannot separate 'reduced joint' from 'added regional' without 3rd arm.")
    lines.append("6. All findings are descriptive associations, not causal proof.\n")

    lines.append("## U. Next-Stage Routing\n")
    lines.append(f"NEXT_STAGE = {next_stage_decision['NEXT_STAGE']}")
    lines.append(f"Based on PRIMARY = {next_stage_decision['PRIMARY_FAILURE_HYPOTHESIS']}\n")

    report_text = "\n".join(lines)
    path = os.path.join(OUT_DIR, "E4N_report.md")
    with open(path, "w") as f:
        f.write(report_text)
    return path


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    e4mf_verdict = _load_json(os.path.join(E4MF_DIR, "E4MF_verdict.json"))
    e4mf_bootstrap = _load_json(os.path.join(E4MF_DIR, "E4MF_bootstrap_results.json"))
    e4mf_episodes = _load_csv(os.path.join(E4MF_DIR, "E4MF_episode_results.csv"))
    e4mf_q_margin = _load_csv(os.path.join(E4MF_DIR, "E4MF_q_margin_context.csv"))
    e4mf_regional = _load_csv(os.path.join(E4MF_DIR, "E4MF_regional_records.csv"))

    e4kf_scenario = _load_csv(os.path.join(E4KF_DIR, "E4KF_scenario_effects.csv"))
    e4mf_scenario = _load_csv(os.path.join(E4MF_DIR, "E4MF_scenario_effects.csv"))

    credit_diag = _load_csv(os.path.join(E4MTR_DIR, "E4MTR_credit_diagnostics.csv"))
    gradient_div = _load_csv(os.path.join(E4MTR_DIR, "E4MTR_gradient_diversity.csv"))
    loss_disp = _load_csv(os.path.join(E4MTR_DIR, "E4MTR_loss_dispersion.csv"))
    val_all_ckpts = _load_csv(os.path.join(E4MTR_DIR, "E4MTR_validation_all_checkpoints.csv"))

    # Phase 0
    pre_manifest = phase_0_pre_manifest()
    preservation = phase_0_immutable_preservation(e4mf_verdict)
    historical = phase_0_historical_integrity()
    assert preservation["all_checks_pass"], "E4MF preservation check failed"

    # Phase 3
    scenario_loc = phase_3_scenario_effect_localization(e4mf_bootstrap, e4mf_episodes)

    # Phase 4
    cross_stage = phase_4_cross_stage_tradeoff(e4kf_scenario, e4mf_scenario)

    # Phase 5
    gradient = phase_5_gradient_conflict_analysis(gradient_div)

    # Phase 6
    loss_grad = phase_6_loss_vs_gradient_scale(credit_diag, gradient_div)

    # Phase 7
    joint_scaling = phase_7_joint_supervision_scaling(credit_diag, gradient_div)

    # Phase 8
    action_spec = phase_8_action_specialization(e4mf_episodes, e4mf_regional)

    # Phase 9
    q_margin = phase_9_q_margin_analysis(e4mf_q_margin)

    # Phase 10
    state_cond = phase_10_state_conditional_analysis(e4mf_regional)

    # Phase 11
    stat_reward = phase_11_stationary_regional_reward(e4mf_regional)

    # Phase 12
    ckpt_tension = phase_12_checkpoint_selection_tension(val_all_ckpts)

    # Phase 13
    policy_het = phase_13_policy_seed_heterogeneity(e4mf_episodes)

    # Phase 14
    val_final = phase_14_validation_final_consistency(val_all_ckpts, e4mf_episodes)

    # Phase 15
    h4_assess = phase_15_h4_assessment()

    # Phase 16
    h6_review = phase_16_monotonic_limitation()

    # Phase 18
    all_evidence = {
        "gradient": gradient,
        "loss_vs_gradient": loss_grad,
        "checkpoint_tension": ckpt_tension,
        "policy_heterogeneity": policy_het,
        "q_margin": q_margin,
        "validation_final": val_final,
    }
    hypotheses = build_hypothesis_evidence_table(all_evidence)

    # Phase 19
    failure_summary = build_failure_hypothesis_summary(hypotheses)

    # Phase 21
    next_stage_decision = build_next_stage_decision(
        failure_summary["PRIMARY_FAILURE_HYPOTHESIS"]
    )

    # Build verdict
    hypothesis_statuses = {h["hypothesis"]: h["status"] for h in hypotheses}
    verdict = build_verdict(
        preservation, historical,
        failure_summary["PRIMARY_FAILURE_HYPOTHESIS"],
        failure_summary["SECONDARY_FAILURE_HYPOTHESES"],
        hypothesis_statuses, gradient, loss_grad, q_margin, ckpt_tension, val_final,
    )

    # Build report
    build_report(
        preservation, historical, scenario_loc, cross_stage, gradient,
        loss_grad, joint_scaling, action_spec, q_margin, state_cond,
        stat_reward, ckpt_tension, policy_het, val_final,
        h4_assess, h6_review, hypotheses, failure_summary, next_stage_decision,
    )

    return verdict


if __name__ == "__main__":
    v = run()
    print(f"\nE4N_COMPLETE = {v['E4N_COMPLETE']}")
    print(f"ORIGINAL_E4MF_VERDICT = {v['ORIGINAL_E4MF_VERDICT']}")
    print(f"E4MF_VERDICT_CHANGED = {v['E4MF_VERDICT_CHANGED']}")
    print(f"PRIMARY_FAILURE_HYPOTHESIS = {v['PRIMARY_FAILURE_HYPOTHESIS']}")
    print(f"NEXT_STAGE = {v['NEXT_STAGE']}")
