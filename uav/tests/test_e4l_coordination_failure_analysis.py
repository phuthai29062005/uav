"""
Tests for E4L — Coordination Failure Analysis.

Diagnostic-only post-hoc analysis: no training, no environment runs, no new seeds.
No skip. No xfail. No weakened assertions.
"""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

E4L_DIR = Path("results/E4L_coordination_failure_analysis")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")

ARMS = ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"]
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
HYPOTHESIS_VOCAB = {"SUPPORTED", "PLAUSIBLE", "WEAK", "NOT_SUPPORTED",
                    "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"}


def _load(p):
    with open(p) as f:
        return json.load(f)


def _csv_rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def verdict():
    return _load(E4L_DIR / "E4L_verdict.json")


@pytest.fixture(scope="module")
def hyp_summary():
    return _load(E4L_DIR / "E4L_failure_hypothesis_summary.json")


@pytest.fixture(scope="module")
def pres():
    return _load(E4L_DIR / "E4L_e4kf_verdict_preservation.json")


@pytest.fixture(scope="module")
def integrity():
    return _load(E4L_DIR / "E4L_historical_integrity.json")


@pytest.fixture(scope="module")
def manifest():
    return _load(E4L_DIR / "E4L_pre_manifest.json")


@pytest.fixture(scope="module")
def routing():
    return _load(E4L_DIR / "E4L_next_stage_decision.json")


@pytest.fixture(scope="module")
def e4kf_verdict():
    return _load(E4KF_DIR / "E4KF_verdict.json")


# ====================================================================
# T1-T10: Prerequisites / historical integrity
# ====================================================================
class TestPrerequisites:
    def test_t1_e4kf_complete(self, e4kf_verdict):
        assert e4kf_verdict["E4KF_COMPLETE"] == "YES"

    def test_t2_e4kf_verdict_failed(self, e4kf_verdict):
        assert e4kf_verdict["COORDINATION_CAUSAL_VERDICT"] == "FAILED"

    def test_t3_original_e4f_preserved(self, verdict):
        assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"

    def test_t4_e4hf_preserved(self, verdict):
        assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"

    def test_t5_e4if_preserved(self, verdict):
        assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"

    def test_t6_e4jf_preserved(self, verdict):
        assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_t7_e4kf_verdict_unchanged(self, verdict):
        assert verdict["E4KF_VERDICT_CHANGED"] == "NO"

    def test_t8_e4kf_mixed_primary_harmful(self, verdict):
        assert verdict["E4KF_MIXED_PRIMARY"] == "HARMFUL"

    def test_t9_preservation_all_pass(self, pres):
        assert pres["all_preserved"] is True
        for k, v in pres["checks"].items():
            assert v is True, f"preservation check {k} failed"

    def test_t10_historical_integrity(self, integrity):
        assert integrity["E4KF_verdict_verified"] is True
        assert integrity["ORIGINAL_E4F_VERDICT"] == "FAILED"
        assert integrity["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        assert integrity["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        assert integrity["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
        assert integrity["E4KF_COORDINATION_CAUSAL_VERDICT"] == "FAILED"


# ====================================================================
# T11-T15: Diagnostic-only constraints
# ====================================================================
class TestDiagnosticConstraints:
    def test_t11_no_training(self, verdict):
        assert verdict["NEW_TRAINING"] == "NO"

    def test_t12_no_new_episodes(self, verdict):
        assert verdict["NEW_ENVIRONMENT_EPISODES"] == 0

    def test_t13_no_new_seeds(self, verdict):
        assert verdict["NEW_DIAGNOSTIC_ENVIRONMENT_SEEDS_USED"] == "NONE"

    def test_t14_manifest_sources(self, manifest):
        assert manifest["new_training"] is False
        assert manifest["new_environment_episodes"] == 0
        assert manifest["new_diagnostic_seeds"] == "NONE"
        assert len(manifest["sources"]) >= 6

    def test_t15_e4l_complete(self, verdict):
        assert verdict["E4L_COMPLETE"] == "YES"


# ====================================================================
# T16-T18: Scenario failure localization
# ====================================================================
class TestScenarioLocalization:
    def test_t16_localization_csv_exists(self):
        assert (E4L_DIR / "E4L_scenario_failure_localization.csv").exists()

    def test_t17_three_scenarios(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        assert len(rows) == 3
        assert {r["scenario"] for r in rows} == set(SCENARIOS)

    def test_t18_persistent_harmful_mixed_harmful_stationary_beneficial(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        for r in rows:
            d = float(r["delta_response"])
            if r["scenario"] == "PERSISTENT":
                assert d < 0
            elif r["scenario"] == "STATIONARY":
                assert d > 0
            elif r["scenario"] == "MIXED":
                assert d < 0


# ====================================================================
# T19-T22: H1 action specialization
# ====================================================================
class TestH1ActionSpecialization:
    def test_t19_action_csv_exists(self):
        assert (E4L_DIR / "E4L_action_specialization_analysis.csv").exists()

    def test_t20_h1_in_valid_vocab(self, verdict):
        assert verdict["H1_ACTION_SPECIALIZATION_MISMATCH"] in HYPOTHESIS_VOCAB

    def test_t21_c1_persistent_less_predict(self):
        rows = _csv_rows(E4L_DIR / "E4L_action_specialization_analysis.csv")
        c0_pers = next((r for r in rows if r["arm"] == ARMS[0] and r["scenario"] == "PERSISTENT"), None)
        c1_pers = next((r for r in rows if r["arm"] == ARMS[1] and r["scenario"] == "PERSISTENT"), None)
        if c0_pers and c1_pers and float(c0_pers["total_decisions"]) > 0:
            assert float(c1_pers["PREDICT"]) < float(c0_pers["PREDICT"])

    def test_t22_h1_status(self, verdict):
        assert verdict["H1_ACTION_SPECIALIZATION_MISMATCH"] == "SUPPORTED"


# ====================================================================
# T23-T26: H2 joint reward aggregation
# ====================================================================
class TestH2RewardAggregation:
    def test_t23_aggregation_csv_exists(self):
        assert (E4L_DIR / "E4L_joint_reward_aggregation_analysis.csv").exists()

    def test_t24_cancellation_csv_exists(self):
        assert (E4L_DIR / "E4L_reward_cancellation_analysis.csv").exists()

    def test_t25_h2_in_valid_vocab(self, verdict):
        assert verdict["H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT"] in HYPOTHESIS_VOCAB

    def test_t26_cancellation_ratio_bounded(self):
        rows = _csv_rows(E4L_DIR / "E4L_reward_cancellation_analysis.csv")
        for r in rows:
            med = float(r["median_cancellation_ratio"])
            assert 0 <= med <= 1 + 1e-9


# ====================================================================
# T27-T29: H3 anchor weighting
# ====================================================================
class TestH3AnchorWeighting:
    def test_t27_anchor_csv_exists(self):
        assert (E4L_DIR / "E4L_anchor_event_weighting_analysis.csv").exists()

    def test_t28_h3_in_valid_vocab(self, verdict):
        assert verdict["H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION"] in HYPOTHESIS_VOCAB

    def test_t29_h3_status(self, verdict):
        assert verdict["H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION"] == "WEAK"


# ====================================================================
# T30-T33: H4 sibling target redundancy
# ====================================================================
class TestH4SiblingRedundancy:
    def test_t30_sibling_csv_exists(self):
        assert (E4L_DIR / "E4L_sibling_redundancy_analysis.csv").exists()

    def test_t31_h4_in_valid_vocab(self, verdict):
        assert verdict["H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION"] in HYPOTHESIS_VOCAB

    def test_t32_redundancy_ratio_positive(self, verdict):
        assert verdict["SIBLING_REDUNDANCY_RATIO"] > 0

    def test_t33_h4_supported(self, verdict):
        assert verdict["H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION"] == "SUPPORTED"


# ====================================================================
# T34-T36: H5 mixer degeneracy
# ====================================================================
class TestH5MixerDegeneracy:
    def test_t34_mixer_csv_exists(self):
        assert (E4L_DIR / "E4L_mixer_conditioning_analysis.csv").exists()

    def test_t35_zero_violations(self):
        rows = _csv_rows(E4L_DIR / "E4L_mixer_conditioning_analysis.csv")
        for r in rows:
            assert int(r["monotonicity_violations"]) == 0

    def test_t36_h5_status(self, verdict):
        assert verdict["H5_MIXER_DEGENERACY_WEAK_STATE_CONDITIONING"] == "NOT_SUPPORTED"


# ====================================================================
# T37-T38: H6 monotonic limitation
# ====================================================================
class TestH6MonotonicLimitation:
    def test_t37_monotonic_review_exists(self):
        assert (E4L_DIR / "E4L_monotonic_limitation_review.json").exists()

    def test_t38_h6_plausible(self, verdict):
        assert verdict["H6_MONOTONIC_REPRESENTATION_LIMITATION"] == "PLAUSIBLE"


# ====================================================================
# T39-T40: H7 Q-margin
# ====================================================================
class TestH7QMargin:
    def test_t39_q_margin_csv_exists(self):
        assert (E4L_DIR / "E4L_q_margin_analysis.csv").exists()

    def test_t40_h7_not_auditable(self, verdict):
        assert verdict["H7_UTILITY_Q_MARGIN_DEGRADATION"] == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"


# ====================================================================
# T41-T43: H8 training dynamics
# ====================================================================
class TestH8TrainingDynamics:
    def test_t41_dynamics_csv_exists(self):
        assert (E4L_DIR / "E4L_training_dynamics_analysis.csv").exists()

    def test_t42_h8_in_valid_vocab(self, verdict):
        assert verdict["H8_TRAINING_DYNAMICS_Q_SCALE_INSTABILITY"] in HYPOTHESIS_VOCAB

    def test_t43_dynamics_both_arms(self):
        rows = _csv_rows(E4L_DIR / "E4L_training_dynamics_analysis.csv")
        arms = {r["arm"] for r in rows}
        assert ARMS[0] in arms
        assert ARMS[1] in arms


# ====================================================================
# T44-T46: H9 checkpoint tension
# ====================================================================
class TestH9CheckpointTension:
    def test_t44_tension_csv_exists(self):
        assert (E4L_DIR / "E4L_checkpoint_selection_tension.csv").exists()

    def test_t45_tension_ten_rows(self):
        rows = _csv_rows(E4L_DIR / "E4L_checkpoint_selection_tension.csv")
        assert len(rows) == 10

    def test_t46_h9_in_valid_vocab(self, verdict):
        assert verdict["H9_CHECKPOINT_SELECTION_TENSION"] in HYPOTHESIS_VOCAB


# ====================================================================
# T47-T48: H10 scenario-specific
# ====================================================================
class TestH10ScenarioSpecific:
    def test_t47_h10_in_valid_vocab(self, verdict):
        assert verdict["H10_SCENARIO_SPECIFIC_COORDINATION_FAILURE"] in HYPOTHESIS_VOCAB

    def test_t48_h10_supported(self, verdict):
        assert verdict["H10_SCENARIO_SPECIFIC_COORDINATION_FAILURE"] == "SUPPORTED"


# ====================================================================
# T49-T55: Hypothesis table / primary / routing
# ====================================================================
class TestHypothesisAggregation:
    def test_t49_hypothesis_table_csv_exists(self):
        assert (E4L_DIR / "E4L_hypothesis_evidence_table.csv").exists()

    def test_t50_ten_hypotheses_in_table(self):
        rows = _csv_rows(E4L_DIR / "E4L_hypothesis_evidence_table.csv")
        assert len(rows) == 10
        ids = {r["hypothesis_id"] for r in rows}
        for i in range(1, 11):
            assert f"H{i}" in ids

    def test_t51_all_statuses_in_vocab(self):
        rows = _csv_rows(E4L_DIR / "E4L_hypothesis_evidence_table.csv")
        for r in rows:
            assert r["status"] in HYPOTHESIS_VOCAB, f"{r['hypothesis_id']} has invalid status {r['status']}"

    def test_t52_primary_hypothesis_set(self, hyp_summary):
        assert hyp_summary["PRIMARY_FAILURE_HYPOTHESIS"] is not None
        assert isinstance(hyp_summary["PRIMARY_FAILURE_HYPOTHESIS"], str)

    def test_t53_multifactorial_when_multiple_supported(self, hyp_summary):
        supported = hyp_summary["supported"]
        if len(supported) > 1:
            assert hyp_summary["PRIMARY_FAILURE_HYPOTHESIS"] == "MULTIFACTORIAL_OR_UNRESOLVED"

    def test_t54_routing_decision_exists(self, routing):
        assert "NEXT_STAGE" in routing
        assert routing["NEXT_STAGE"].startswith("E4M_")

    def test_t55_routing_consistent_with_verdict(self, verdict, routing):
        assert verdict["NEXT_STAGE"] == routing["NEXT_STAGE"]


# ====================================================================
# T56-T62: Output artifacts
# ====================================================================
class TestOutputArtifacts:
    EXPECTED_FILES = [
        "E4L_verdict.json",
        "E4L_e4kf_verdict_preservation.json",
        "E4L_historical_integrity.json",
        "E4L_pre_manifest.json",
        "E4L_scenario_failure_localization.csv",
        "E4L_action_specialization_analysis.csv",
        "E4L_joint_reward_aggregation_analysis.csv",
        "E4L_reward_cancellation_analysis.csv",
        "E4L_anchor_event_weighting_analysis.csv",
        "E4L_sibling_redundancy_analysis.csv",
        "E4L_mixer_conditioning_analysis.csv",
        "E4L_monotonic_limitation_review.json",
        "E4L_q_margin_analysis.csv",
        "E4L_action_disagreement_analysis.json",
        "E4L_training_dynamics_analysis.csv",
        "E4L_checkpoint_selection_tension.csv",
        "E4L_credit_structure_comparison.json",
        "E4L_policy_seed_heterogeneity.csv",
        "E4L_validation_final_consistency.csv",
        "E4L_scenario_state_distribution.csv",
        "E4L_hypothesis_evidence_table.csv",
        "E4L_failure_hypothesis_summary.json",
        "E4L_next_stage_decision.json",
        "E4L_report.md",
    ]

    def test_t56_all_output_files_exist(self):
        for f in self.EXPECTED_FILES:
            assert (E4L_DIR / f).exists(), f"missing output: {f}"

    def test_t57_report_md_exists_and_nonempty(self):
        p = E4L_DIR / "E4L_report.md"
        assert p.exists()
        assert p.stat().st_size > 500

    def test_t58_verdict_json_complete(self, verdict):
        required_keys = [
            "E4L_COMPLETE", "ORIGINAL_E4KF_VERDICT", "E4KF_VERDICT_CHANGED",
            "E4KF_MIXED_PRIMARY", "E4KF_MIXED_DELTA_C0_MINUS_C1",
            "PRIMARY_FAILURE_HYPOTHESIS", "NEXT_STAGE",
            "NEW_TRAINING", "NEW_ENVIRONMENT_EPISODES",
            "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
            "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
        ]
        for k in required_keys:
            assert k in verdict, f"verdict missing key: {k}"

    def test_t59_all_10_hypothesis_keys_in_verdict(self, verdict):
        expected = [
            "H1_ACTION_SPECIALIZATION_MISMATCH",
            "H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT",
            "H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION",
            "H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
            "H5_MIXER_DEGENERACY_WEAK_STATE_CONDITIONING",
            "H6_MONOTONIC_REPRESENTATION_LIMITATION",
            "H7_UTILITY_Q_MARGIN_DEGRADATION",
            "H8_TRAINING_DYNAMICS_Q_SCALE_INSTABILITY",
            "H9_CHECKPOINT_SELECTION_TENSION",
            "H10_SCENARIO_SPECIFIC_COORDINATION_FAILURE",
        ]
        for k in expected:
            assert k in verdict, f"verdict missing hypothesis key: {k}"
            assert verdict[k] in HYPOTHESIS_VOCAB

    def test_t60_credit_structure_json_valid(self):
        j = _load(E4L_DIR / "E4L_credit_structure_comparison.json")
        assert "summary_by_scenario" in j
        assert "C0_n_events" in j

    def test_t61_action_disagreement_not_auditable(self):
        j = _load(E4L_DIR / "E4L_action_disagreement_analysis.json")
        assert j["status"] == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"

    def test_t62_validation_final_consistency(self):
        rows = _csv_rows(E4L_DIR / "E4L_validation_final_consistency.csv")
        assert len(rows) >= 2


# ====================================================================
# T63-T68: Causal language / immutability
# ====================================================================
class TestCausalLanguageImmutability:
    def test_t63_verdict_e4kf_not_changed(self, verdict):
        assert verdict["ORIGINAL_E4KF_VERDICT"] == "FAILED"
        assert verdict["E4KF_VERDICT_CHANGED"] == "NO"

    def test_t64_delta_matches_e4kf(self, verdict, e4kf_verdict):
        assert abs(verdict["E4KF_MIXED_DELTA_C0_MINUS_C1"]
                   - e4kf_verdict["MIXED_DELTA_C0_MINUS_C1"]) < 1e-12

    def test_t65_ci_matches_e4kf(self, verdict, e4kf_verdict):
        for i in range(2):
            assert abs(verdict["E4KF_MIXED_DELTA_CI"][i]
                       - e4kf_verdict["MIXED_DELTA_CI"][i]) < 1e-12

    def test_t66_no_rescue_claim(self, verdict):
        assert verdict["ORIGINAL_E4KF_VERDICT"] == "FAILED"
        primary = verdict["PRIMARY_FAILURE_HYPOTHESIS"]
        assert primary != "NONE"

    def test_t67_report_no_prohibited_causal_language(self):
        text = (E4L_DIR / "E4L_report.md").read_text()
        for phrase in ["root cause is", "proved that", "caused the failure"]:
            assert phrase not in text.lower(), f"report contains prohibited causal language: '{phrase}'"

    def test_t68_not_auditable_fields_present(self, verdict):
        assert verdict["C0_PERSISTENT_Q_MARGIN"] == "NOT_AUDITABLE"
        assert verdict["C1_PERSISTENT_Q_MARGIN"] == "NOT_AUDITABLE"


# ====================================================================
# T69-T72: Routing / next stage
# ====================================================================
class TestRouting:
    def test_t69_next_stage_starts_with_e4m(self, verdict):
        assert verdict["NEXT_STAGE"].startswith("E4M_")

    def test_t70_routing_json_has_reason(self, routing):
        assert "routing_reason" in routing
        assert len(routing["routing_reason"]) > 10

    def test_t71_no_prohibited_routing(self, verdict):
        prohibited = ["PPO", "CTDE_ACTOR_CRITIC", "GNN", "ATTENTION"]
        nxt = verdict["NEXT_STAGE"]
        for p in prohibited:
            assert p not in nxt

    def test_t72_routing_consistent_with_supported(self, hyp_summary, routing):
        supported = hyp_summary["supported"]
        nxt = routing["NEXT_STAGE"]
        if "H2" in supported or "H4" in supported:
            assert "CREDIT" in nxt or "DIAGNOSTIC" in nxt


# ====================================================================
# T73-T78: Numerical correctness spot-checks
# ====================================================================
class TestNumericalCorrectness:
    def test_t73_localization_delta_negative_for_mixed(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        mixed = next(r for r in rows if r["scenario"] == "MIXED")
        assert float(mixed["delta_response"]) < 0

    def test_t74_localization_delta_positive_for_stationary(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        stat = next(r for r in rows if r["scenario"] == "STATIONARY")
        assert float(stat["delta_response"]) > 0

    def test_t75_persistent_zero_policy_positive(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        pers = next(r for r in rows if r["scenario"] == "PERSISTENT")
        assert int(pers["n_policy_positive"]) == 0

    def test_t76_stationary_all_policy_positive(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        stat = next(r for r in rows if r["scenario"] == "STATIONARY")
        assert int(stat["n_policy_positive"]) == 5

    def test_t77_mixed_one_policy_positive(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_failure_localization.csv")
        mixed = next(r for r in rows if r["scenario"] == "MIXED")
        assert int(mixed["n_policy_positive"]) == 1

    def test_t78_sibling_redundancy_ratio_equals_10(self, verdict):
        assert abs(verdict["SIBLING_REDUNDANCY_RATIO"] - 10.0) < 0.01


# ====================================================================
# T79-T83: Data integrity cross-checks
# ====================================================================
class TestDataIntegrity:
    def test_t79_episode_count_matches_e4kf(self):
        e4kf_rows = _csv_rows(E4KF_DIR / "E4KF_episode_results.csv")
        assert len(e4kf_rows) == 1800

    def test_t80_regional_records_count(self):
        reg_rows = _csv_rows(E4KF_DIR / "E4KF_regional_records.csv")
        assert len(reg_rows) == 8000

    def test_t81_training_rows_count(self):
        train_rows = _csv_rows(E4KTR_DIR / "E4KTR_training_episode_results.csv")
        assert len(train_rows) == 10800

    def test_t82_update_summary_both_arms(self):
        rows = _csv_rows(E4KTR_DIR / "E4KTR_training_update_summary.csv")
        arms = {r["arm"] for r in rows}
        assert ARMS[0] in arms
        assert ARMS[1] in arms

    def test_t83_coordination_diag_five_policies(self):
        rows = _csv_rows(E4KTR_DIR / "E4KTR_coordination_diagnostics.csv")
        assert len(rows) == 5
        seeds = {int(r["policy_seed"]) for r in rows}
        assert seeds == set(POLICY_SEEDS)


# ====================================================================
# T84-T88: Hypothesis summary internal consistency
# ====================================================================
class TestHypothesisSummaryConsistency:
    def test_t84_supported_list_matches_statuses(self, hyp_summary):
        statuses = hyp_summary["hypothesis_statuses"]
        supported = hyp_summary["supported"]
        expected = [h for h, s in statuses.items() if s == "SUPPORTED"]
        assert set(supported) == set(expected)

    def test_t85_plausible_list_matches(self, hyp_summary):
        statuses = hyp_summary["hypothesis_statuses"]
        plausible = hyp_summary["plausible"]
        expected = [h for h, s in statuses.items() if s == "PLAUSIBLE"]
        assert set(plausible) == set(expected)

    def test_t86_not_auditable_list_matches(self, hyp_summary):
        statuses = hyp_summary["hypothesis_statuses"]
        na = hyp_summary["not_auditable"]
        expected = [h for h, s in statuses.items() if s == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"]
        assert set(na) == set(expected)

    def test_t87_verdict_statuses_match_summary(self, verdict, hyp_summary):
        statuses = hyp_summary["hypothesis_statuses"]
        mapping = {
            "H1": "H1_ACTION_SPECIALIZATION_MISMATCH",
            "H2": "H2_JOINT_REWARD_AGGREGATION_MISALIGNMENT",
            "H3": "H3_REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION",
            "H4": "H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
            "H5": "H5_MIXER_DEGENERACY_WEAK_STATE_CONDITIONING",
            "H6": "H6_MONOTONIC_REPRESENTATION_LIMITATION",
            "H7": "H7_UTILITY_Q_MARGIN_DEGRADATION",
            "H8": "H8_TRAINING_DYNAMICS_Q_SCALE_INSTABILITY",
            "H9": "H9_CHECKPOINT_SELECTION_TENSION",
            "H10": "H10_SCENARIO_SPECIFIC_COORDINATION_FAILURE",
        }
        for short, full in mapping.items():
            assert verdict[full] == statuses[short], f"{short}: {verdict[full]} != {statuses[short]}"

    def test_t88_all_10_hypotheses_classified(self, hyp_summary):
        assert len(hyp_summary["hypothesis_statuses"]) == 10


# ====================================================================
# T89-T93: Specific hypothesis evidence checks
# ====================================================================
class TestHypothesisEvidence:
    def test_t89_h5_zero_monotonicity_violations(self):
        rows = _csv_rows(E4L_DIR / "E4L_mixer_conditioning_analysis.csv")
        total_v = sum(int(r["monotonicity_violations"]) for r in rows)
        assert total_v == 0

    def test_t90_h6_no_counterfactual_runs(self):
        j = _load(E4L_DIR / "E4L_monotonic_limitation_review.json")
        assert j["counterfactual_rollouts_available"] is False

    def test_t91_h7_no_direct_q_margin(self):
        rows = _csv_rows(E4L_DIR / "E4L_q_margin_analysis.csv")
        for r in rows:
            assert "mean_abs_q_early" in r

    def test_t92_policy_heterogeneity_five_rows(self):
        rows = _csv_rows(E4L_DIR / "E4L_policy_seed_heterogeneity.csv")
        assert len(rows) == 5

    def test_t93_state_distribution_csv(self):
        rows = _csv_rows(E4L_DIR / "E4L_scenario_state_distribution.csv")
        assert len(rows) >= 1
        for r in rows:
            assert float(r["n_records"]) > 0


# ====================================================================
# T94-T97: Post-analysis integrity / no side effects
# ====================================================================
class TestPostAnalysisIntegrity:
    def test_t94_e4kf_verdict_not_modified(self, e4kf_verdict):
        assert e4kf_verdict["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
        assert e4kf_verdict["E4KF_COMPLETE"] == "YES"
        assert e4kf_verdict["MIXED_PRIMARY"] == "HARMFUL"

    def test_t95_no_checkpoint_dirs_created(self):
        e4l_contents = list(E4L_DIR.iterdir())
        for p in e4l_contents:
            assert not p.is_dir(), f"unexpected directory in E4L output: {p}"

    def test_t96_no_training_artifacts(self):
        e4l_contents = [p.name for p in E4L_DIR.iterdir()]
        for name in e4l_contents:
            assert "checkpoint" not in name.lower() or name.endswith(".csv")
            assert not name.endswith(".pt")
            assert not name.endswith(".pth")

    def test_t97_full_regression_placeholder(self, verdict):
        assert "FULL_REGRESSION" in verdict
