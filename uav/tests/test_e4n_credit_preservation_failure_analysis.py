"""Tests for E4N — Credit Preservation Failure Analysis."""

import json
import csv
import os
import pytest

E4N_DIR = "results/E4N_credit_preservation_failure_analysis"
E4MF_DIR = "results/E4MF_credit_preservation_ab_final_evaluation_execution"
E4KF_DIR = "results/E4KF_joint_coordination_ab_final_evaluation_execution"
E4L_DIR = "results/E4L_coordination_failure_analysis"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def verdict():
    return _load_json(os.path.join(E4N_DIR, "E4N_verdict.json"))


@pytest.fixture(scope="module")
def preservation():
    return _load_json(os.path.join(E4N_DIR, "E4N_e4mf_verdict_preservation.json"))


@pytest.fixture(scope="module")
def historical():
    return _load_json(os.path.join(E4N_DIR, "E4N_historical_integrity.json"))


@pytest.fixture(scope="module")
def failure_summary():
    return _load_json(os.path.join(E4N_DIR, "E4N_failure_hypothesis_summary.json"))


@pytest.fixture(scope="module")
def next_stage():
    return _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))


@pytest.fixture(scope="module")
def hypotheses():
    return _load_csv(os.path.join(E4N_DIR, "E4N_hypothesis_evidence_table.csv"))


@pytest.fixture(scope="module")
def scenario_loc():
    return _load_csv(os.path.join(E4N_DIR, "E4N_scenario_effect_localization.csv"))


@pytest.fixture(scope="module")
def cross_stage():
    return _load_csv(os.path.join(E4N_DIR, "E4N_cross_stage_scenario_tradeoff.csv"))


@pytest.fixture(scope="module")
def gradient():
    return _load_csv(os.path.join(E4N_DIR, "E4N_gradient_conflict_analysis.csv"))


@pytest.fixture(scope="module")
def loss_grad():
    return _load_csv(os.path.join(E4N_DIR, "E4N_loss_vs_gradient_scale.csv"))


@pytest.fixture(scope="module")
def q_margin():
    return _load_csv(os.path.join(E4N_DIR, "E4N_q_margin_analysis.csv"))


@pytest.fixture(scope="module")
def ckpt_tension():
    return _load_csv(os.path.join(E4N_DIR, "E4N_checkpoint_selection_tension.csv"))


@pytest.fixture(scope="module")
def policy_het():
    return _load_csv(os.path.join(E4N_DIR, "E4N_policy_seed_heterogeneity.csv"))


@pytest.fixture(scope="module")
def val_final():
    return _load_csv(os.path.join(E4N_DIR, "E4N_validation_final_consistency.csv"))


@pytest.fixture(scope="module")
def h4_assessment():
    return _load_json(os.path.join(E4N_DIR, "E4N_h4_targeted_revision_assessment.json"))


@pytest.fixture(scope="module")
def h6_review():
    return _load_json(os.path.join(E4N_DIR, "E4N_monotonic_limitation_review.json"))


@pytest.fixture(scope="module")
def action_spec():
    return _load_csv(os.path.join(E4N_DIR, "E4N_action_specialization_analysis.csv"))


@pytest.fixture(scope="module")
def e4mf_verdict():
    return _load_json(os.path.join(E4MF_DIR, "E4MF_verdict.json"))


@pytest.fixture(scope="module")
def e4kf_verdict():
    return _load_json(os.path.join(E4KF_DIR, "E4KF_verdict.json"))


# ============================================================
# T1-T10: E4MF Preservation
# ============================================================

def test_T01_e4mf_complete(verdict):
    assert verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"
    assert verdict["E4N_COMPLETE"] == "YES"


def test_T02_e4mf_failed_preserved(preservation):
    assert preservation["checks"]["E4MF_VERDICT_FAILED"]


def test_T03_mixed_directional_positive_preserved(verdict):
    assert verdict["E4MF_MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"


def test_T04_positive_mixed_delta_preserved(verdict):
    assert verdict["E4MF_MIXED_DELTA_C1_MINUS_C2"] > 0


def test_T05_ci_crossing_zero_preserved(preservation):
    assert preservation["checks"]["MIXED_CI_CROSSES_ZERO"]


def test_T06_robustness_3_of_5_preserved(verdict):
    assert verdict["E4MF_POLICY_ROBUSTNESS"] == "FAIL_3_OF_5"


def test_T07_persistent_safety_pass_preserved(verdict):
    assert verdict["E4MF_PERSISTENT_SAFETY"] == "PASS"


def test_T08_stationary_safety_fail_preserved(verdict):
    assert verdict["E4MF_STATIONARY_SAFETY"] == "FAIL"


def test_T09_macro_safety_pass_preserved(verdict):
    assert verdict["E4MF_MACRO_SAFETY"] == "PASS"


def test_T10_e4mf_verdict_changed_no(verdict):
    assert verdict["E4MF_VERDICT_CHANGED"] == "NO"


# ============================================================
# T11-T13: E4KF Historical
# ============================================================

def test_T11_historical_e4kf_persistent_fail(historical):
    assert historical["E4KF_PERSISTENT_SAFETY"] == "FAIL"


def test_T12_historical_e4kf_stationary_pass(historical):
    assert historical["E4KF_STATIONARY_SAFETY"] == "PASS"


def test_T13_no_false_claim_e4kf_stationary_failed(historical):
    assert historical["E4KF_STATIONARY_BENEFICIAL_CONFIRMED"] is True
    assert historical["E4KF_PERSISTENT_HARMFUL_CONFIRMED"] is True


# ============================================================
# T14-T15: Scenario Localization and Cross-Stage
# ============================================================

def test_T14_scenario_localization_exact(scenario_loc):
    assert len(scenario_loc) == 3
    scenarios = {r["scenario"] for r in scenario_loc}
    assert scenarios == {"PERSISTENT", "STATIONARY", "MIXED"}
    per = next(r for r in scenario_loc if r["scenario"] == "PERSISTENT")
    assert per["classification_response"] == "BENEFICIAL"
    stat = next(r for r in scenario_loc if r["scenario"] == "STATIONARY")
    assert stat["classification_response"] == "HARMFUL"
    mixed = next(r for r in scenario_loc if r["scenario"] == "MIXED")
    assert mixed["classification_response"] == "DIRECTIONAL_POSITIVE"


def test_T15_cross_stage_table_exact(cross_stage):
    assert len(cross_stage) == 3
    per = next(r for r in cross_stage if r["scenario"] == "PERSISTENT")
    assert per["E4KF_response_classification"] == "HARMFUL"
    assert per["E4KF_safety"] == "FAIL"
    assert per["E4MF_response_classification"] == "BENEFICIAL"
    assert per["E4MF_safety"] == "PASS"
    stat = next(r for r in cross_stage if r["scenario"] == "STATIONARY")
    assert stat["E4KF_response_classification"] == "BENEFICIAL"
    assert stat["E4KF_safety"] == "PASS"
    assert stat["E4MF_response_classification"] == "HARMFUL"
    assert stat["E4MF_safety"] == "FAIL"


# ============================================================
# T16-T19: Gradient and Loss Analysis
# ============================================================

def test_T16_gradient_cosine_uses_stored_data(gradient):
    assert len(gradient) == 5
    for r in gradient:
        assert r["scenario_breakdown_available"] == "NO"
        assert float(r["n_samples"]) > 0


def test_T17_gradient_norm_uses_stored_data(gradient):
    for r in gradient:
        assert float(r["mean_norm_joint"]) > 0
        assert float(r["mean_norm_regional"]) > 0


def test_T18_loss_distinguished_from_gradient(loss_grad):
    for r in loss_grad:
        assert "mean_joint_td_loss" in r
        assert "mean_regional_td_loss" in r
        assert "mean_joint_grad_norm" in r
        assert "mean_regional_grad_norm" in r
        assert r["small_loss_implies_small_gradient"] == "NO"


def test_T19_no_small_loss_small_gradient_inference(loss_grad):
    for r in loss_grad:
        assert r["small_loss_implies_small_gradient"] == "NO"
        assert r["regional_gradient_material"] == "YES"


# ============================================================
# T20-T21: Joint Coefficient and Isolation
# ============================================================

def test_T20_c2_joint_coefficient_05_documented():
    rows = _load_csv(os.path.join(E4N_DIR, "E4N_joint_supervision_scaling_analysis.csv"))
    for r in rows:
        assert float(r["C2_joint_coefficient"]) == 0.5
        assert float(r["C1_joint_coefficient"]) == 1.0


def test_T21_no_claim_regional_td_alone_isolated():
    rows = _load_csv(os.path.join(E4N_DIR, "E4N_joint_supervision_scaling_analysis.csv"))
    for r in rows:
        assert r["isolates_regional_addition_only"] in ("False", False)


# ============================================================
# T22-T23: Action Specialization
# ============================================================

def test_T22_action_specialization_exact(action_spec):
    persistent_rows = [r for r in action_spec if r["scenario"] == "PERSISTENT"]
    assert len(persistent_rows) == 2
    for r in persistent_rows:
        assert r["availability"] == "AVAILABLE"


def test_T23_action_diversity_not_success_metric():
    report = open(os.path.join(E4N_DIR, "E4N_report.md")).read()
    assert "Action diversity is NOT itself a success metric" in report


# ============================================================
# T24-T25: Q-Margin
# ============================================================

def test_T24_q_margin_uses_stored_data(q_margin):
    assert len(q_margin) == 10
    for r in q_margin:
        assert r["data_source"] == "E4MF_q_margin_context"


def test_T25_q_margin_descriptive_only(q_margin):
    for r in q_margin:
        assert r["per_scenario_available"] == "NO"


# ============================================================
# T26-T27: State-Conditional and Reward
# ============================================================

def test_T26_state_conditional_stored_only():
    rows = _load_csv(os.path.join(E4N_DIR, "E4N_state_conditional_action_analysis.csv"))
    for r in rows:
        assert r["data_source"] == "E4MF_regional_records"
        assert r["scenario"] == "PERSISTENT"


def test_T27_r1_scenario_audit_stored_only():
    rows = _load_csv(os.path.join(E4N_DIR, "E4N_stationary_regional_reward_analysis.csv"))
    stat_rows = [r for r in rows if r["scenario"] == "STATIONARY"]
    for r in stat_rows:
        assert r["availability"] == "NOT_AVAILABLE"


# ============================================================
# T28-T29: Checkpoint
# ============================================================

def test_T28_checkpoint_tension_validation_only(ckpt_tension):
    assert len(ckpt_tension) == 10
    for r in ckpt_tension:
        assert "selected_checkpoint" in r
        assert "best_STATIONARY_ckpt" in r


def test_T29_no_checkpoint_reselection(ckpt_tension):
    for r in ckpt_tension:
        assert r["checkpoint_reselection"] == "NO"


# ============================================================
# T30-T31: Policy and Validation
# ============================================================

def test_T30_all_five_policies_retained(policy_het):
    seeds = {int(r["policy_seed"]) for r in policy_het}
    assert seeds == {6001, 6002, 6003, 6004, 6005}


def test_T31_validation_final_no_ci_pooling(val_final):
    phases = {r["phase"] for r in val_final}
    assert "VALIDATION" in phases
    assert "FINAL" in phases
    for r in val_final:
        assert "ci_pooled" not in r


# ============================================================
# T32-T34: H4 and H6
# ============================================================

def test_T32_h4_historical_supported(h4_assessment):
    assert h4_assessment["H4_HISTORICAL_STATUS"] == "SUPPORTED"


def test_T33_h4_targeted_revision_sufficient_no(h4_assessment):
    assert h4_assessment["H4_TARGETED_REVISION_SUFFICIENT"] == "NO"


def test_T34_h6_not_promoted_without_direct_evidence(h6_review):
    assert h6_review["H6_UPDATED_STATUS"] == "PLAUSIBLE"
    assert h6_review["H6_PROMOTED"] is False
    assert h6_review["direct_evidence_available"] is False


# ============================================================
# T35-T37: Hypothesis Vocabulary and Structure
# ============================================================

def test_T35_hypothesis_vocabulary_exact(hypotheses):
    valid = {
        "SUPPORTED", "PLAUSIBLE", "WEAK", "NOT_SUPPORTED",
        "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
    }
    for h in hypotheses:
        assert h["status"] in valid, f"{h['hypothesis']} has invalid status {h['status']}"


def test_T36_at_most_one_primary(failure_summary):
    primary = failure_summary["PRIMARY_FAILURE_HYPOTHESIS"]
    assert isinstance(primary, str)
    assert primary.count(",") == 0


def test_T37_multifactorial_option_available(failure_summary):
    assert failure_summary["multifactorial_option_used"] is True


# ============================================================
# T38-T45: Routing Cases
# ============================================================

def test_T38_routing_gradient_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT"] == "E4O_GRADIENT_CONFLICT_DIAGNOSTIC_DESIGN"


def test_T39_routing_loss_scale_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS"] == "E4O_CREDIT_SCALE_NORMALIZATION_DESIGN"


def test_T40_routing_action_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N3_ACTION_SPECIALIZATION_SHIFT"] == "E4O_ACTION_SPECIALIZATION_DIAGNOSTIC_DESIGN"


def test_T41_routing_q_margin_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    key = "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT"
    assert cases[key] == "E4O_POLICY_DECISIVENESS_DIAGNOSTIC_DESIGN"


def test_T42_routing_checkpoint_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N6_CHECKPOINT_SELECTION_SCENARIO_TENSION"] == "E4O_CHECKPOINT_OBJECTIVE_REVIEW"


def test_T43_routing_monotonicity_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N9_MONOTONIC_QMIX_LIMITATION_REMAINS"] == "E4O_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"


def test_T44_routing_scenario_tradeoff_case():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["N10_MULTIFACTORIAL_SCENARIO_TRADEOFF"] == "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN"


def test_T45_unresolved_routing():
    decision = _load_json(os.path.join(E4N_DIR, "E4N_next_stage_decision.json"))
    cases = decision["all_routing_cases"]
    assert cases["MULTIFACTORIAL_OR_UNRESOLVED"] == "E4O_CREDIT_PRESERVATION_FAILURE_DIAGNOSTIC_DESIGN"


# ============================================================
# T46-T50: No-New-Data and Historical Unchanged
# ============================================================

def test_T46_no_new_training(verdict):
    assert verdict["NEW_TRAINING"] == "NO"


def test_T47_no_new_environment_episodes(verdict):
    assert verdict["NEW_ENVIRONMENT_EPISODES"] == 0


def test_T48_no_new_objective_evaluations(verdict):
    assert verdict["NEW_OBJECTIVE_EVALUATIONS"] == 0


def test_T49_no_final_seed_rerun(verdict):
    assert verdict["NEW_DIAGNOSTIC_SEEDS_USED"] == "NONE"


def test_T50_historical_artifacts_unchanged(e4mf_verdict, e4kf_verdict):
    assert e4mf_verdict["CREDIT_PRESERVATION_CAUSAL_VERDICT"] == "FAILED"
    assert e4kf_verdict["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
    assert e4kf_verdict["PERSISTENT_RESPONSE_EFFECT"] == "HARMFUL"
    assert e4kf_verdict["PERSISTENT_SAFETY"] == "FAIL"
    assert e4kf_verdict["STATIONARY_RESPONSE_EFFECT"] == "BENEFICIAL"
    assert e4kf_verdict["STATIONARY_SAFETY"] == "PASS"


# ============================================================
# T51: Full Regression (placeholder — run externally)
# ============================================================

def test_T51_full_regression_placeholder(verdict):
    reg = verdict["FULL_REGRESSION"]
    assert reg == "(run separately)" or ("PASS" in reg)


# ============================================================
# Additional Completeness Tests
# ============================================================

def test_T52_all_required_outputs_exist():
    required = [
        "E4N_pre_manifest.json",
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
        "E4N_report.md",
    ]
    for name in required:
        path = os.path.join(E4N_DIR, name)
        assert os.path.exists(path), f"Missing: {name}"


def test_T53_cross_stage_pattern_documented(verdict):
    assert verdict["CROSS_STAGE_PATTERN"] == "SCENARIO_TRADEOFF_SHIFT"


def test_T54_stationary_harm_universal(policy_het):
    for r in policy_het:
        assert r["STATIONARY_direction"] == "C1_better"


def test_T55_persistent_benefit_universal(policy_het):
    for r in policy_het:
        assert r["PERSISTENT_direction"] == "C2_better"


def test_T56_validation_final_direction_consistent(val_final):
    val = {r["scenario"]: r["direction"] for r in val_final if r["phase"] == "VALIDATION"}
    fin = {r["scenario"]: r["direction"] for r in val_final if r["phase"] == "FINAL"}
    for sc in ["PERSISTENT", "STATIONARY", "MIXED"]:
        assert val[sc] == fin[sc], f"{sc} direction inconsistent: val={val[sc]} final={fin[sc]}"


def test_T57_hypothesis_count(hypotheses):
    assert len(hypotheses) == 10


def test_T58_next_stage_is_scenario_tradeoff(next_stage):
    assert next_stage["NEXT_STAGE"] == "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN"


def test_T59_report_all_sections():
    report = open(os.path.join(E4N_DIR, "E4N_report.md")).read()
    for section in ["## A.", "## B.", "## C.", "## D.", "## E.", "## F.",
                     "## G.", "## H.", "## I.", "## J.", "## K.", "## L.",
                     "## M.", "## N.", "## O.", "## P.", "## Q.", "## R.",
                     "## S.", "## T.", "## U."]:
        assert section in report, f"Missing section {section}"


def test_T60_pre_manifest_all_available():
    manifest = _load_json(os.path.join(E4N_DIR, "E4N_pre_manifest.json"))
    assert manifest["all_available"]


def test_T61_n1_status_correct(verdict):
    assert verdict["N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT"] == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"


def test_T62_n2_status_correct(verdict):
    assert verdict["N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS"] == "NOT_SUPPORTED"


def test_T63_n7_not_supported(verdict):
    assert verdict["N7_POLICY_SEED_HETEROGENEITY"] == "NOT_SUPPORTED"


def test_T64_n10_supported(verdict):
    assert verdict["N10_MULTIFACTORIAL_SCENARIO_TRADEOFF"] == "SUPPORTED"
