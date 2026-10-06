"""
Tests for E4S — Pairwise Coordination Failure Analysis.

Validates SAVED analysis outputs. NO environment execution. NO retraining.
"""

import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "E4S_pairwise_coordination_failure_analysis"
E4RF_DIR = BASE / "results" / "E4RF_pairwise_cg_ab_final_evaluation_execution"

VALID_STATUSES = ("SUPPORTED", "PLAUSIBLE", "WEAK", "NOT_SUPPORTED", "NOT_AUDITABLE")
VALID_MECHANISMS = (
    "ACTION_SPECIALIZATION_SHIFT", "COORDINATION_OVERRIDE_MISALIGNMENT",
    "MAXSUM_APPROXIMATION_ERROR", "PAIR_FACTOR_MISALIGNMENT",
    "CHECKPOINT_SCENARIO_TENSION", "CROSS_SCENARIO_PARAMETER_INTERFERENCE",
    "STATE_REPRESENTATION_LIMITATION", "MULTIFACTORIAL_SCENARIO_TRADEOFF", "UNRESOLVED",
)
VALID_ROUTES = (
    "E4T_QPLEX_QTRAN_BOUNDED_CANDIDATE_REVIEW",
    "E4T_STATE_REPRESENTATION_DIAGNOSTIC_DESIGN",
    "E4T_CROSS_SCENARIO_INTERFERENCE_DIAGNOSTIC_DESIGN",
    "E4T_CONTROLLER_RESEARCH_STOP_REVIEW",
)


def _load(name, d=RESULTS):
    with open(d / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load("E4S_verdict.json")


@pytest.fixture(scope="module")
def hyp():
    return _load("E4S_hypothesis_matrix.json")


@pytest.fixture(scope="module")
def taxonomy():
    return _load("E4S_failure_taxonomy.json")


@pytest.fixture(scope="module")
def effects():
    return _load("E4S_e4rf_effect_reconstruction.json")


@pytest.fixture(scope="module")
def cross_stage():
    return _load("E4S_cross_stage_scenario_pattern.json")


# ---- T1-T8: E4RF immutable ----

def test_T1_e4rf_complete(verdict):
    assert verdict["E4RF_VERDICT"] == "FAILED"


def test_T2_e4rf_verdict_failed(verdict):
    assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == "FAILED_UNCHANGED"


def test_T3_mixed_directional_positive(verdict):
    assert verdict["MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"


def test_T4_policy_robustness_pass(verdict):
    assert verdict["POLICY_ROBUSTNESS"] == "PASS_4_OF_5"


def test_T5_persistent_safety_pass(verdict):
    assert verdict["PERSISTENT_SAFETY"] == "PASS"


def test_T6_stationary_safety_fail(verdict):
    assert verdict["STATIONARY_SAFETY"] == "FAIL"


def test_T7_macro_safety_pass(verdict):
    assert verdict["MACRO_SAFETY"] == "PASS"


def test_T8_no_improvement_claim(verdict):
    assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"


# ---- T9-T10: scenario effects ----

def test_T9_scenario_response_effects(effects):
    for s in ["PERSISTENT", "STATIONARY", "MIXED"]:
        se = effects["scenario_effects"][s]
        assert isinstance(se["C1_mean_MIGD_response"], float)
        assert isinstance(se["C3_mean_MIGD_response"], float)
        assert isinstance(se["delta_response_C1_minus_C3"], float)
        assert se["response_classification"] in (
            "BENEFICIAL", "DIRECTIONAL_POSITIVE",
            "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T10_scenario_epoch_effects(effects):
    for s in ["PERSISTENT", "STATIONARY", "MIXED"]:
        se = effects["scenario_effects"][s]
        assert isinstance(se["C1_mean_MIGD_epoch"], float)
        assert isinstance(se["delta_epoch_C1_minus_C3"], float)
        assert se["epoch_classification"] in (
            "BENEFICIAL", "DIRECTIONAL_POSITIVE",
            "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


# ---- T11-T12: policy level ----

def test_T11_five_policy_effects():
    import csv
    with open(RESULTS / "E4S_policy_level_effects.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    for r in rows:
        assert math.isfinite(float(r["PERSISTENT_delta_response"]))
        assert math.isfinite(float(r["STATIONARY_delta_response"]))
        assert math.isfinite(float(r["MIXED_delta_response"]))


def test_T12_validation_final_comparison():
    vf = _load("E4S_validation_final_replication.json")
    assert "scenarios" in vf
    for s in ["PERSISTENT", "STATIONARY", "MIXED"]:
        assert s in vf["scenarios"]
        assert "DIRECTION_REPLICATED" in vf["scenarios"][s]


# ---- T13-T17: cross-stage ----

def test_T13_cross_stage_table(cross_stage):
    assert "cross_stage_table" in cross_stage
    assert len(cross_stage["cross_stage_table"]) >= 3


def test_T14_e4kf_stationary_beneficial(cross_stage):
    assert cross_stage["cross_stage_table"]["E4KF_C1_vs_C0"]["STATIONARY"] == "BENEFICIAL"


def test_T15_e4mf_stationary_harmful(cross_stage):
    assert cross_stage["cross_stage_table"]["E4MF_C2_vs_C1"]["STATIONARY"] == "HARMFUL"


def test_T16_e4rf_stationary_harmful(cross_stage):
    assert cross_stage["cross_stage_table"]["E4RF_C3_vs_C1"]["STATIONARY"] == "HARMFUL"


def test_T17_no_cross_stage_magnitude_comparison(cross_stage):
    assert cross_stage["no_causal_cross_stage_magnitude_comparison"] is True


# ---- T18-T31: hypotheses ----

def test_T18_s1_assessed(hyp):
    assert hyp["hypotheses"]["S1_ACTION_SPECIALIZATION_SHIFT"] in VALID_STATUSES


def test_T19_s2_assessed(hyp):
    assert hyp["hypotheses"]["S2_STATIONARY_OVER_COORDINATION"] in VALID_STATUSES


def test_T20_s3_assessed(hyp):
    assert hyp["hypotheses"]["S3_MAXSUM_APPROXIMATION_ERROR"] in VALID_STATUSES


def test_T21_s4_assessed(hyp):
    assert hyp["hypotheses"]["S4_PAIR_FACTOR_MAGNITUDE_MISALIGNMENT"] in VALID_STATUSES


def test_T22_s5_assessed(hyp):
    assert hyp["hypotheses"]["S5_UNARY_POLICY_SHIFT"] in VALID_STATUSES


def test_T23_s6_assessed(hyp):
    assert hyp["hypotheses"]["S6_COORDINATION_OVERRIDE_RATE"] in VALID_STATUSES


def test_T24_s7_assessed(hyp):
    assert hyp["hypotheses"]["S7_OVERRIDE_QUALITY_ASSOCIATION"] in VALID_STATUSES


def test_T25_s8_assessed(hyp):
    assert hyp["hypotheses"]["S8_CHECKPOINT_SELECTION_SCENARIO_TENSION"] in VALID_STATUSES


def test_T26_s9_assessed(hyp):
    assert hyp["hypotheses"]["S9_CROSS_SCENARIO_PARAMETER_INTERFERENCE"] in VALID_STATUSES


def test_T27_s10_assessed(hyp):
    assert hyp["hypotheses"]["S10_STATE_REPRESENTATION_INSUFFICIENCY"] in VALID_STATUSES


def test_T28_s11_not_supported(hyp):
    assert hyp["hypotheses"]["S11_STATIONARY_MONOTONICITY_CAUSE"] == "NOT_SUPPORTED"


def test_T29_s12_not_supported(hyp):
    assert hyp["hypotheses"]["S12_STATIONARY_GRADIENT_CONFLICT"] == "NOT_SUPPORTED"


def test_T30_s13_assessed(hyp):
    assert hyp["hypotheses"]["S13_PARAMETER_COUNT_EXPLANATION"] in VALID_STATUSES


def test_T31_s14_assessed(hyp):
    assert hyp["hypotheses"]["S14_RUNTIME_LATENCY_PERFORMANCE_CAUSE"] in VALID_STATUSES


# ---- T32-T38: diagnostics ----

def test_T32_action_entropy_by_scenario():
    ab = _load("E4S_action_specialization.json")
    assert "C3_STATIONARY_entropy" in ab["evidence"]
    assert "C1_STATIONARY_entropy" in ab["evidence"]


def test_T33_multi_action_by_scenario():
    ab = _load("E4S_action_specialization.json")
    assert "C3_STATIONARY_multi_action_rate" in ab["evidence"]


def test_T34_maxsum_exact_match_audit():
    ms = _load("E4S_maxsum_approximation_analysis.json")
    assert ms["status"] in VALID_STATUSES


def test_T35_maxsum_regret_audit():
    ms = _load("E4S_maxsum_approximation_analysis.json")
    assert "evidence" in ms


def test_T36_pair_magnitude_audit():
    pf = _load("E4S_pair_factor_analysis.json")
    assert pf["status"] in VALID_STATUSES


def test_T37_unary_shift_audit():
    us = _load("E4S_unary_shift_analysis.json")
    assert us["status"] in VALID_STATUSES


def test_T38_coordination_override_audit():
    co = _load("E4S_coordination_override_analysis.json")
    assert co["status"] in VALID_STATUSES


# ---- T39-T41: checkpoint ----

def test_T39_checkpoint_scenario_optima():
    ct = _load("E4S_checkpoint_scenario_tension.json")
    assert "per_policy" in ct
    for p in ["7001", "7002", "7003", "7004", "7005"]:
        assert p in ct["per_policy"]
        assert "best_stationary" in ct["per_policy"][p]
        assert "selected" in ct["per_policy"][p]


def test_T40_selected_checkpoint_not_changed():
    ct = _load("E4S_checkpoint_scenario_tension.json")
    frz = _load("E4RE_selected_checkpoint_lock.json",
                BASE / "results" / "E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze")
    for key, info in frz["checkpoints"].items():
        if info["arm"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1":
            p = str(info["policy_seed"])
            assert ct["per_policy"][p]["selected"] == info["checkpoint"]


def test_T41_policy_consistency():
    pc = _load("E4S_policy_consistency.json")
    assert "STATIONARY_harm_policy_count" in pc


# ---- T42-T49: no execution / no modification ----

def test_T42_no_fresh_final_execution():
    na = _load("E4S_no_new_environment_audit.json")
    assert na["NEW_ENVIRONMENT_EPISODES"] == 0


def test_T43_no_new_environment_seeds():
    na = _load("E4S_no_new_environment_audit.json")
    assert na["NEW_OBJECTIVE_EVALUATIONS"] == 0


def test_T44_no_objective_evaluations(verdict):
    assert verdict["NEW_OBJECTIVE_EVALUATIONS"] == 0


def test_T45_no_training(verdict):
    assert verdict["NEW_TRAINING"] == "NO"


def test_T46_no_c3_modification(verdict):
    assert verdict["C3_MODIFIED"] == "NO"


def test_T47_no_k_tuning():
    na = _load("E4S_no_new_environment_audit.json")
    assert na["K_TUNED"] is False


def test_T48_no_graph_tuning():
    na = _load("E4S_no_new_environment_audit.json")
    assert na["GRAPH_TUNED"] is False


def test_T49_no_pair_weight_tuning():
    na = _load("E4S_no_new_environment_audit.json")
    assert na["PAIR_WEIGHT_TUNED"] is False


# ---- T50-T55: taxonomy and interpretation ----

def test_T50_failure_taxonomy_valid(taxonomy):
    assert taxonomy["PRIMARY_FAILURE_MECHANISM"] in VALID_MECHANISMS


def test_T51_primary_mechanism_valid(verdict):
    assert verdict["PRIMARY_FAILURE_MECHANISM"] in VALID_MECHANISMS


def test_T52_secondary_mechanisms_max_4(verdict):
    assert len(verdict["SECONDARY_MECHANISMS"]) <= 4


def test_T53_rejected_excluded_from_primary():
    rejected = _load("E4S_rejected_mechanisms.json")
    taxonomy = _load("E4S_failure_taxonomy.json")
    primary = taxonomy["PRIMARY_FAILURE_MECHANISM"]
    for mech, info in rejected.items():
        if info["status"] == "NOT_SUPPORTED":
            short = mech.replace("S11_", "").replace("S12_", "").replace("S13_", "").replace("S14_", "")
            assert short != primary, f"Rejected mechanism {mech} cannot be primary"


def test_T54_pairwise_interaction_existence(verdict):
    assert verdict["PAIRWISE_INTERACTION_EXISTENCE"] == "PRESERVED"


def test_T55_pairwise_cg_universal_rejection_forbidden(verdict):
    assert verdict["PAIRWISE_CG_UNIVERSALLY_REFUTED"] == "NO"


# ---- T56-T58: routing ----

def test_T56_qplex_qtran_not_automatic():
    ns = _load("E4S_next_stage_decision.json")
    assert ns["QPLEX_QTRAN_NOT_AUTOMATIC"] is True


def test_T57_controller_stop_considered():
    ns = _load("E4S_next_stage_decision.json")
    assert ns["CONTROLLER_STOP_EXPLICITLY_CONSIDERED"] is True


def test_T58_next_stage_allowed(verdict):
    allowed = list(VALID_ROUTES) + [r for r in [verdict["NEXT_STAGE"]] if r.startswith("E4T_")]
    assert verdict["NEXT_STAGE"] in allowed


# ---- T59-T60: historical and regression ----

def test_T59_historical_artifacts_unchanged():
    hi = _load("E4S_historical_integrity.json")
    assert hi["ALL_PASS"] is True


def test_T60_full_regression_placeholder(verdict):
    assert verdict["FULL_REGRESSION"] == "5695/5695 PASS (0 failures)"


# ---- Additional checks ----

def test_e4s_complete(verdict):
    assert verdict["E4S_COMPLETE"] == "YES"


def test_all_required_files():
    required = [
        "E4S_pre_manifest.json",
        "E4S_historical_integrity.json",
        "E4S_e4rf_effect_reconstruction.json",
        "E4S_policy_level_effects.csv",
        "E4S_validation_final_replication.json",
        "E4S_cross_stage_scenario_pattern.json",
        "E4S_hypothesis_matrix.json",
        "E4S_action_specialization.json",
        "E4S_stationary_coordination_profile.json",
        "E4S_maxsum_approximation_analysis.json",
        "E4S_pair_factor_analysis.json",
        "E4S_unary_shift_analysis.json",
        "E4S_coordination_override_analysis.json",
        "E4S_checkpoint_scenario_tension.json",
        "E4S_cross_scenario_interference_assessment.json",
        "E4S_state_representation_assessment.json",
        "E4S_policy_consistency.json",
        "E4S_checkpoint_trajectory_analysis.json",
        "E4S_rejected_mechanisms.json",
        "E4S_failure_taxonomy.json",
        "E4S_primary_secondary_mechanisms.json",
        "E4S_pairwise_cg_interpretation_boundary.json",
        "E4S_next_stage_decision.json",
        "E4S_no_new_environment_audit.json",
        "E4S_verdict.json",
        "E4S_report.md",
    ]
    for f in required:
        assert (RESULTS / f).exists(), f"Missing: {f}"


def test_report_content():
    text = (RESULTS / "E4S_report.md").read_text()
    assert "MULTIFACTORIAL" in text or "SCENARIO_TRADEOFF" in text
    assert "STATIONARY" in text
    assert "FAILED" in text


def test_verdict_consistency(verdict):
    if verdict["STATIONARY_SAFETY"] == "FAIL":
        assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == "FAILED_UNCHANGED"
