"""Tests for E4Q General Interaction Representation Review."""
import json
import csv
import os
import pytest

BASE = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "E4Q_general_interaction_representation_review",
)

def _load(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)

def _load_csv(name):
    with open(os.path.join(BASE, name)) as f:
        return list(csv.DictReader(f))


class TestE4PDPreservation:
    def test_t01_e4pd_complete(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        assert v["E4PD_COMPLETE"] == "YES"

    def test_t02_e4mf_failed_preserved(self):
        v = _load("E4Q_verdict.json")
        assert v["ORIGINAL_E4MF_VERDICT"] == "FAILED"
        assert v["E4MF_VERDICT_CHANGED"] == "NO"

    def test_t03_p1_not_supported(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        assert v["P1_STATUS"] == "NOT_SUPPORTED"
        assert v["P1_CRITERIA_PASSED"] == 0

    def test_t04_g_rev_negative(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        assert v["G_REV"] < 0

    def test_t05_g_rev_ci_below_zero(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        ci = v["G_REV_CI"]
        assert ci[0] < 0
        assert ci[1] < 0

    def test_t06_global_reversal_observed(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        assert v["GLOBAL_RR"] > 0
        assert v["MONOTONIC_INCOMPATIBILITY"] == "YES"

    def test_t07_persistent_rr_gt_stationary(self):
        v = _load("E4Q_e4pd_evidence_lock.json")
        assert v["RR_PERSISTENT"] > v["RR_STATIONARY"]


class TestScientificBoundaries:
    def test_t08_no_stationary_monotonicity_causal_claim(self):
        v = _load("E4Q_verdict.json")
        assert v["STATIONARY_SPECIFIC_MONOTONIC_CAUSE"] == "NOT_SUPPORTED"

    def test_t09_qmix_monotonic_condition_exact(self):
        q = _load("E4Q_current_qmix_constraint.json")
        assert "dQ_tot/dq_r >= 0" in q["monotonic_constraint_exact_statement"]

    def test_t10_decentralized_utility_execution(self):
        q = _load("E4Q_current_qmix_constraint.json")
        arch = q["current_architecture"]
        assert arch["regional_action_selection"]["decentralized"] is True
        assert arch["QMIX_mixer"]["NOT_used_in_action_selection"] is True

    def test_t11_full_q_vs_igm_distinction(self):
        v = _load("E4Q_verdict.json")
        assert v["FULL_Q_VS_IGM_DISTINCTION"] == "PRESERVED"
        b = _load("E4Q_e4pd_implication_boundary.json")
        assert "A_FULL_JOINT_Q_REPRESENTABILITY" in b["critical_distinction"]
        assert "B_IGM_DECENTRALIZED_GREEDY_OPTIMUM_REPRESENTABILITY" in b["critical_distinction"]


class TestRequirements:
    def test_t12_action_space_fixed(self):
        r = _load("E4Q_representation_requirements.json")
        assert r["requirements"]["R1"]["A"] == 4

    def test_t13_r_max_10(self):
        r = _load("E4Q_representation_requirements.json")
        assert r["requirements"]["R2"]["R_max"] == 10

    def test_t14_no_scenario_label(self):
        r = _load("E4Q_representation_requirements.json")
        assert "No scenario label" in r["requirements"]["R5"]["description"]

    def test_t15_no_extra_fe(self):
        r = _load("E4Q_representation_requirements.json")
        assert "No extra objective evaluations" in r["requirements"]["R6"]["description"]


class TestCandidateReviews:
    def test_t16_qmix_control_reviewed(self):
        e = _load("E4Q_candidate_exclusion.json")
        assert "Q1_QMIX_CONTROL" in e["exclusions"]

    def test_t17_qplex_reviewed(self):
        q = _load("E4Q_qplex_review.json")
        assert q["candidate"] == "Q2_QPLEX_LIKE_DUPLEX_DUELING_VALUE_DECOMPOSITION"

    def test_t18_qtran_reviewed(self):
        q = _load("E4Q_qtran_review.json")
        assert q["candidate"] == "Q3_QTRAN_LIKE_IGM_CONSTRAINED_JOINT_VALUE"

    def test_t19_coordination_graph_reviewed(self):
        q = _load("E4Q_coordination_graph_review.json")
        assert q["candidate"] == "Q4_PAIRWISE_COORDINATION_GRAPH_VALUE_FACTOR"

    def test_t20_autoregressive_reviewed(self):
        q = _load("E4Q_autoregressive_review.json")
        assert q["candidate"] == "Q5_AUTOREGRESSIVE_SEQUENTIAL_JOINT_ACTION_POLICY"

    def test_t21_ctde_actor_critic_reviewed(self):
        q = _load("E4Q_ctde_actor_critic_review.json")
        assert q["candidate"] == "Q6_CENTRALIZED_CRITIC_DECENTRALIZED_POLICY"

    def test_t22_full_central_q_reviewed(self):
        q = _load("E4Q_central_joint_q_upper_bound.json")
        assert q["candidate"] == "Q7_FULL_CENTRAL_JOINT_Q"

    def test_t23_4_power_10_documented(self):
        q = _load("E4Q_central_joint_q_upper_bound.json")
        assert q["joint_action_space"]["total_joint_actions"] == 1048576
        assert q["joint_action_space"]["computation"] == "4^10 = 1,048,576"


class TestCandidateComparison:
    def test_t24_interaction_capability_matrix_complete(self):
        rows = _load_csv("E4Q_candidate_comparison.csv")
        assert len(rows) == 7
        expected = {"Q1_QMIX_CONTROL", "Q2_QPLEX_LIKE", "Q3_QTRAN_LIKE",
                    "Q4_PAIRWISE_COORD_GRAPH", "Q5_AUTOREGRESSIVE",
                    "Q6_CTDE_ACTOR_CRITIC", "Q7_FULL_CENTRAL_Q"}
        actual = {r["candidate"] for r in rows}
        assert actual == expected

    def test_t25_no_numeric_candidate_scoring(self):
        rows = _load_csv("E4Q_candidate_comparison.csv")
        for row in rows:
            for k, v in row.items():
                if k == "candidate":
                    continue
                try:
                    float(v)
                    pytest.fail(f"Numeric score found: {k}={v} for {row['candidate']}")
                except ValueError:
                    pass

    def test_t26_direct_e4pd_evidence_match_documented(self):
        rows = _load_csv("E4Q_candidate_comparison.csv")
        matches = {r["candidate"]: r["direct_E4PD_evidence_match"] for r in rows}
        assert matches["Q4_PAIRWISE_COORD_GRAPH"] == "DIRECT"

    def test_t27_no_candidate_guaranteed_fix(self):
        s = _load("E4Q_selected_candidate.json")
        assert s["NOT_guaranteed_to_fix"] is True
        assert s["NOT_a_STATIONARY_targeted_fix"] is True


class TestPairwiseGraphSpecifics:
    def test_t28_pairwise_graph_tied_to_e4pd(self):
        q = _load("E4Q_coordination_graph_review.json")
        assert "DIRECT" in q["E4PD_evidence_match"]
        assert "pairwise" in q["structural_alignment_with_E4PD"].lower()

    def test_t29_graph_inference_complexity_audited(self):
        q = _load("E4Q_coordination_graph_review.json")
        sel = q["action_selection"]
        assert "message passing" in sel["method"].lower() or "max-sum" in sel["method"].lower()
        assert "720" in sel["computational_cost"] or "7200" in sel["computational_cost"]

    def test_t30_qplex_igm_constraint_audited(self):
        q = _load("E4Q_qplex_review.json")
        assert "IGM" in q["IGM_constraint"]
        assert "advantage" in q["IGM_constraint"].lower()

    def test_t31_qtran_igm_constraint_audited(self):
        q = _load("E4Q_qtran_review.json")
        assert "IGM" in q["IGM_constraint"]
        assert "soft" in q["IGM_constraint"].lower() or "auxiliary" in q["IGM_constraint"].lower()

    def test_t32_rank_reversal_not_ruling_out_igm(self):
        b = _load("E4Q_e4pd_implication_boundary.json")
        assert "do not automatically rule out" in b["therefore"].lower()


class TestExecutionSemantics:
    def test_t33_autoregressive_ordering_cost_documented(self):
        q = _load("E4Q_autoregressive_review.json")
        assert q["execution_semantics_change"]["is_fundamental_change"] is True
        assert "ordering" in q["execution_semantics_change"]["region_ordering"].lower()

    def test_t34_centralized_critic_not_solving_actor(self):
        q = _load("E4Q_ctde_actor_critic_review.json")
        ae = q["actor_expressiveness"]
        assert "does NOT" in ae["CRITICAL_DISTINCTION"]


class TestFutureDesign:
    def test_t35_state_representation_boundary_documented(self):
        s = _load("E4Q_action_state_representation_boundary.json")
        assert s["alternative_explanation_preserved"] is True

    def test_t36_v3_held_fixed(self):
        s = _load("E4Q_selected_candidate.json")
        assert s["isolation_preserved"]["V3_normalization"] == "HELD_FIXED"

    def test_t37_r1_held_fixed(self):
        s = _load("E4Q_selected_candidate.json")
        assert s["isolation_preserved"]["R1_reward"] == "HELD_FIXED"

    def test_t38_production_operators_held_fixed(self):
        s = _load("E4Q_selected_candidate.json")
        assert s["isolation_preserved"]["production_operators"] == "UNCHANGED"


class TestControlAndLoss:
    def test_t39_c1_default_clean_control(self):
        c = _load("E4Q_control_selection.json")
        assert c["future_control"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"

    def test_t40_c2_auxiliary_loss_not_retained(self):
        l = _load("E4Q_loss_spec_boundary.json")
        assert l["C2_auxiliary_loss_carried_forward"] is False
        v = _load("E4Q_verdict.json")
        assert v["C2_REGIONAL_AUXILIARY_LOSS_CARRIED_FORWARD"] == "NO"


class TestSelectionConstraints:
    def test_t41_selected_candidates_le_1(self):
        v = _load("E4Q_verdict.json")
        assert isinstance(v["PRIMARY_SELECTED_CANDIDATE"], str)

    def test_t42_alternates_le_2(self):
        v = _load("E4Q_verdict.json")
        assert len(v["ALTERNATE_CANDIDATES"]) <= 2

    def test_t43_selected_is_allowed_category(self):
        allowed = {
            "QPLEX_LIKE_IGM_VALUE_DECOMPOSITION",
            "QTRAN_LIKE_IGM_VALUE_DECOMPOSITION",
            "PAIRWISE_COORDINATION_GRAPH_VALUE_FACTOR",
            "AUTOREGRESSIVE_REGIONAL_COORDINATION",
            "CENTRALIZED_CRITIC_DECENTRALIZED_POLICY",
            "NO_CANDIDATE_SUFFICIENT_REVIEW_STATE_FIRST",
        }
        v = _load("E4Q_verdict.json")
        assert v["PRIMARY_SELECTED_CANDIDATE"] in allowed

    def test_t44_selection_evidence_based(self):
        s = _load("E4Q_selected_candidate.json")
        assert "evidence_match" in s["rationale"]
        assert "parsimony" in s["rationale"]

    def test_t45_no_global_optimality_claim(self):
        s = _load("E4Q_selected_candidate.json")
        assert s["NOT_claimed"] == "globally_optimal"
        assert s["selected_for"] == "NEXT_ISOLATED_TEST"


class TestFutureQuestion:
    def test_t46_future_performance_question_correct(self):
        f = _load("E4Q_future_scientific_question.json")
        assert "pairwise coordination graph" in f["future_causal_question"].lower()
        assert "C1 QMIX" in f["future_causal_question"]

    def test_t47_reversal_frequency_not_success_metric(self):
        f = _load("E4Q_future_scientific_question.json")
        assert "environment property" in f["reason"]
        assert "NOT a valid architecture success metric" in f["reason"] or "not an architecture success metric" in f["reason"].lower()


class TestFirewalls:
    def test_t48_no_new_training(self):
        v = _load("E4Q_verdict.json")
        assert v["NEW_TRAINING"] == "NO"

    def test_t49_no_environment_execution(self):
        v = _load("E4Q_verdict.json")
        assert v["NEW_ENVIRONMENT_EPISODES"] == 0

    def test_t50_no_objective_evaluation(self):
        v = _load("E4Q_verdict.json")
        assert v["NEW_OBJECTIVE_EVALUATIONS"] == 0

    def test_t51_no_new_controller_implementation(self):
        v = _load("E4Q_verdict.json")
        assert v["NEW_CONTROLLER_IMPLEMENTATION"] == "NO"


class TestIntegrity:
    def test_t52_historical_artifacts_unchanged(self):
        h = _load("E4Q_historical_integrity.json")
        assert h["ALL_PASS"] is True
        assert h["HISTORICAL_INTEGRITY"] == "VERIFIED"

    def test_t53_next_stage_routing_valid(self):
        v = _load("E4Q_verdict.json")
        valid_stages = {
            "E4R_PAIRWISE_COORDINATION_GRAPH_EXPERIMENT_DESIGN",
            "E4R_QPLEX_EXPERIMENT_DESIGN",
            "E4R_QTRAN_EXPERIMENT_DESIGN",
            "E4R_AUTOREGRESSIVE_COORDINATION_EXPERIMENT_DESIGN",
            "E4R_CTDE_ACTOR_CRITIC_EXPERIMENT_DESIGN",
            "E4R_ACTION_STATE_REPRESENTATION_REVIEW",
            "E4R_CONTROLLER_RESEARCH_STOP_REVIEW",
        }
        assert v["NEXT_STAGE"] in valid_stages


class TestOutputCompleteness:
    REQUIRED_FILES = [
        "E4Q_pre_manifest.json",
        "E4Q_historical_integrity.json",
        "E4Q_e4pd_evidence_lock.json",
        "E4Q_representation_question.json",
        "E4Q_current_qmix_constraint.json",
        "E4Q_e4pd_implication_boundary.json",
        "E4Q_representation_requirements.json",
        "E4Q_candidate_families.json",
        "E4Q_qplex_review.json",
        "E4Q_qtran_review.json",
        "E4Q_coordination_graph_review.json",
        "E4Q_autoregressive_review.json",
        "E4Q_ctde_actor_critic_review.json",
        "E4Q_central_joint_q_upper_bound.json",
        "E4Q_action_state_representation_boundary.json",
        "E4Q_candidate_comparison.csv",
        "E4Q_candidate_exclusion.json",
        "E4Q_selection_hierarchy.json",
        "E4Q_control_selection.json",
        "E4Q_loss_spec_boundary.json",
        "E4Q_selected_candidate.json",
        "E4Q_alternates.json",
        "E4Q_future_scientific_question.json",
        "E4Q_future_mechanism_diagnostics.json",
        "E4Q_next_stage_decision.json",
        "E4Q_verdict.json",
        "E4Q_report.md",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_required_file_exists(self, fname):
        path = os.path.join(BASE, fname)
        assert os.path.isfile(path), f"Missing: {fname}"


class TestPerformanceVerdictBoundary:
    def test_performance_causal_verdict_not_computed(self):
        v = _load("E4Q_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_no_diagnostic_seeds_used(self):
        v = _load("E4Q_verdict.json")
        assert v["NEW_DIAGNOSTIC_SEEDS_USED"] == "NONE"

    def test_general_interaction_supported(self):
        v = _load("E4Q_verdict.json")
        assert v["GENERAL_INTERACTION_REPRESENTATION_ISSUE"] == "SUPPORTED"

    def test_stationary_specific_not_supported(self):
        v = _load("E4Q_verdict.json")
        assert v["STATIONARY_SPECIFIC_MONOTONIC_CAUSE"] == "NOT_SUPPORTED"

    def test_state_representation_held_fixed(self):
        v = _load("E4Q_verdict.json")
        assert v["STATE_REPRESENTATION_HELD_FIXED"] == "YES"
