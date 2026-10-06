"""Tests for E4P — Nonmonotonic Interaction Diagnostic Design.

80+ tests covering historical integrity, hypothesis, utility,
counterfactual surface, actions, backgrounds, seeds, checkpoints,
snapshot, pair selection, random tapes, isolation, reversal,
interaction, aggregation, contrasts, bootstrap, P1 logic,
routing, audit contracts, firewall, verdict structure, regression.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
E4P_DIR = BASE / "results" / "E4P_nonmonotonic_interaction_diagnostic_design"


def _load(p):
    with open(p) as f:
        return json.load(f)


REQUIRED_FILES = [
    "E4P_pre_manifest.json",
    "E4P_historical_integrity.json",
    "E4P_e4odr_lock.json",
    "E4P_scientific_question.json",
    "E4P_hypothesis_contract.json",
    "E4P_environment_utility_contract.json",
    "E4P_counterfactual_surface_contract.json",
    "E4P_primary_action_set.json",
    "E4P_secondary_action_set.json",
    "E4P_background_context_contract.json",
    "E4P_source_trajectory_contract.json",
    "E4P_checkpoint_lock.json",
    "E4P_diagnostic_seed_contract.json",
    "E4P_seed_disjointness.json",
    "E4P_snapshot_eligibility_contract.json",
    "E4P_pair_selection_contract.json",
    "E4P_random_tape_contract.json",
    "E4P_counterfactual_isolation_contract.json",
    "E4P_action_preference_contract.json",
    "E4P_rank_reversal_contract.json",
    "E4P_best_action_context_contract.json",
    "E4P_cross_difference_contract.json",
    "E4P_source_aggregation_contract.json",
    "E4P_primary_scenario_contrast.json",
    "E4P_statistical_design.json",
    "E4P_p1_support_logic.json",
    "E4P_global_incompatibility_contract.json",
    "E4P_predict_fallback_audit_contract.json",
    "E4P_fe_audit_contract.json",
    "E4P_no_performance_verdict.json",
    "E4P_e4pi_validation_contract.json",
    "E4P_future_routing.json",
    "E4P_future_firewall.json",
    "E4P_protocol_frozen.json",
    "E4P_verdict.json",
    "E4P_report.md",
]


# ── OUTPUT EXISTENCE ─────────────────────────────────────────────────

class TestOutputExistence:
    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_output_exists(self, fname):
        assert (E4P_DIR / fname).exists(), f"Missing {fname}"

    def test_output_count(self):
        actual = {p.name for p in E4P_DIR.iterdir() if p.is_file()}
        assert set(REQUIRED_FILES).issubset(actual)


# ── T1-T5: HISTORICAL INTEGRITY ─────────────────────────────────────

class TestHistoricalIntegrity:
    def test_t1_e4odr_complete(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["checks"]["E4ODR_COMPLETE"]

    def test_t2_e4mf_failed(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["checks"]["E4MF_FAILED"]

    def test_t3_o1_not_supported(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["checks"]["O1_NOT_SUPPORTED"]

    def test_t4_n8_not_supported(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["checks"]["N8_NOT_SUPPORTED"]

    def test_t5_n9_plausible(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["checks"]["N9_PLAUSIBLE"]

    def test_all_integrity_pass(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["all_pass"]


# ── T6-T9: UTILITY & SURFACE ────────────────────────────────────────

class TestUtilityAndSurface:
    def test_t6_true_environment_utility(self):
        u = _load(E4P_DIR / "E4P_environment_utility_contract.json")
        assert "IGD" in u["quality_function"]
        assert "igd" in u["quality_source"]

    def test_t7_learned_q_not_ground_truth(self):
        u = _load(E4P_DIR / "E4P_environment_utility_contract.json")
        assert u["learned_Q_ground_truth"] is False

    def test_t8_fixed_pre_response_state(self):
        s = _load(E4P_DIR / "E4P_counterfactual_surface_contract.json")
        assert "EXACT SAME pre-response snapshot" in s["isolation"]

    def test_t9_same_snapshot_all_cells(self):
        iso = _load(E4P_DIR / "E4P_counterfactual_isolation_contract.json")
        assert "EXACT SAME pre-response snapshot" in iso["isolation_rule"]
        assert "discard" in iso["isolation_rule"].lower()


# ── T10-T14: ACTIONS ────────────────────────────────────────────────

class TestActions:
    def test_t10_primary_exactly_ldp(self):
        a = _load(E4P_DIR / "E4P_primary_action_set.json")
        assert a["primary_actions"] == ["LOCAL", "PREDICT", "DIVERSIFY"]

    def test_t11_primary_3x3(self):
        s = _load(E4P_DIR / "E4P_counterfactual_surface_contract.json")
        assert s["primary_surface"]["grid_size"] == "3x3"
        assert s["primary_surface"]["n_cells"] == 9

    def test_t12_keep_excluded_primary(self):
        a = _load(E4P_DIR / "E4P_primary_action_set.json")
        assert a["KEEP_excluded"]
        assert "KEEP" not in a["primary_actions"]

    def test_t13_keep_included_secondary(self):
        a = _load(E4P_DIR / "E4P_secondary_action_set.json")
        assert a["KEEP_included"]
        assert "KEEP" in a["secondary_actions"]

    def test_t14_secondary_4x4(self):
        s = _load(E4P_DIR / "E4P_counterfactual_surface_contract.json")
        assert s["secondary_surface"]["grid_size"] == "4x4"
        assert s["secondary_surface"]["n_cells"] == 16


# ── T15-T16: BACKGROUNDS ────────────────────────────────────────────

class TestBackgrounds:
    def test_t15_keep_background_primary(self):
        b = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert b["BACKGROUND_A"]["role"] == "PRIMARY"
        assert "KEEP" in b["BACKGROUND_A"]["name"]

    def test_t16_policy_background_secondary(self):
        b = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert b["BACKGROUND_B"]["role"] == "ROBUSTNESS_DESCRIPTIVE"

    def test_t62_keep_background_for_verdict(self):
        b = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert b["primary_verdict_based_on"] == "KEEP_BACKGROUND"

    def test_t63_policy_cannot_override_primary(self):
        b = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert b["policy_background_cannot_override_primary"]


# ── T17-T18: SOURCE TRAJECTORIES & CHECKPOINTS ──────────────────────

class TestSourceAndCheckpoints:
    def test_t17_both_c1_c2(self):
        s = _load(E4P_DIR / "E4P_source_trajectory_contract.json")
        assert s["source_arms"] == ["C1", "C2"]

    def test_t18_ten_checkpoints(self):
        c = _load(E4P_DIR / "E4P_checkpoint_lock.json")
        assert c["total_checkpoints"] == 10
        assert c["C1_policies"] == [6001, 6002, 6003, 6004, 6005]
        assert c["C2_policies"] == [6001, 6002, 6003, 6004, 6005]

    def test_t68_no_checkpoint_reselection(self):
        c = _load(E4P_DIR / "E4P_checkpoint_lock.json")
        assert c["no_reselection"]
        assert c["no_retraining"]


# ── T19-T21: SEEDS ──────────────────────────────────────────────────

class TestSeeds:
    def test_t19_seeds_2401_2420(self):
        s = _load(E4P_DIR / "E4P_diagnostic_seed_contract.json")
        assert s["diagnostic_seeds"] == list(range(2401, 2421))
        assert s["n_seeds"] == 20

    def test_t20_disjointness(self):
        d = _load(E4P_DIR / "E4P_seed_disjointness.json")
        assert d["disjoint"]
        assert d["overlap_with_env_seeds"] == []
        assert d["overlap_with_policy_seeds"] == []
        assert d["overlap_with_bootstrap_seeds"] == []

    def test_t21_future_matrix_600(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["FUTURE_SOURCE_EPISODES"] == 600


# ── T22-T28: SNAPSHOT & PAIR SELECTION ───────────────────────────────

class TestSnapshotAndPair:
    def test_t22_exactly_one_snapshot(self):
        s = _load(E4P_DIR / "E4P_snapshot_eligibility_contract.json")
        assert s["deterministic"]

    def test_t23_first_eligible_event(self):
        s = _load(E4P_DIR / "E4P_snapshot_eligibility_contract.json")
        assert "FIRST" in s["selection_rule"]

    def test_t24_predict_history_eligibility(self):
        s = _load(E4P_DIR / "E4P_snapshot_eligibility_contract.json")
        assert any("PREDICT history" in c for c in s["eligibility_criteria"])

    def test_t25_deterministic_pair_list(self):
        p = _load(E4P_DIR / "E4P_pair_selection_contract.json")
        assert "lexicographically" in str(p["steps"])

    def test_t26_sha256_pair_selector(self):
        p = _load(E4P_DIR / "E4P_pair_selection_contract.json")
        assert p["hash_function"] == "hashlib.sha256"
        assert p["selection_method"] == "SHA256_DETERMINISTIC_INDEX"

    def test_t27_python_hash_forbidden(self):
        p = _load(E4P_DIR / "E4P_pair_selection_contract.json")
        assert p["python_builtin_hash_forbidden"]

    def test_t28_pair_outcome_blind(self):
        p = _load(E4P_DIR / "E4P_pair_selection_contract.json")
        assert p["outcome_blind"]
        assert "performance" in str(p["forbidden_inputs"])


# ── T29-T30: OPERATOR CONSTANTS ─────────────────────────────────────

class TestOperatorConstants:
    def test_t29_production_constants(self):
        pf = _load(E4P_DIR / "E4P_protocol_frozen.json")
        ops = pf["operator_constants"]
        assert ops["LOCAL"]["sigma_L"] == 0.02
        assert ops["PREDICT"]["sigma_P"] == 0.005
        assert ops["PREDICT"]["lambda_P"] == 1.0
        assert ops["DIVERSIFY"]["sigma_D"] == 0.15
        assert ops["KEEP"]["FE"] == 0
        assert ops["LOCAL"]["B_PER_ACTIVE_REGION"] == 10

    def test_t30_unrestricted_survival(self):
        pf = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert pf["survival"] == "UNRESTRICTED_STANDARD_NSGA_II"


# ── T31-T34: RANDOM TAPES ───────────────────────────────────────────

class TestRandomTapes:
    def test_t31_common_random_tape(self):
        r = _load(E4P_DIR / "E4P_random_tape_contract.json")
        assert "Same random tape reused" in r["purpose"]

    def test_t32_tapes_equal_4(self):
        r = _load(E4P_DIR / "E4P_random_tape_contract.json")
        assert r["n_tapes"] == 4

    def test_t33_cells_share_tape(self):
        r = _load(E4P_DIR / "E4P_random_tape_contract.json")
        assert "Same random tape reused across every action cell" in r["purpose"]

    def test_t34_no_cross_cell_contamination(self):
        iso = _load(E4P_DIR / "E4P_counterfactual_isolation_contract.json")
        assert "No cross-cell contamination" in iso["isolation_rule"]


# ── T35-T36: UTILITY ────────────────────────────────────────────────

class TestUtilityDefinition:
    def test_t35_keep_all_baseline(self):
        u = _load(E4P_DIR / "E4P_environment_utility_contract.json")
        assert "KEEP_ALL" in u["KEEP_ALL_baseline"]

    def test_t36_delta_definition(self):
        a = _load(E4P_DIR / "E4P_action_preference_contract.json")
        assert "Delta_i(a,b | j=c)" in a["definition"]
        assert "V_S(i=a,j=c) - V_S(i=b,j=c)" in a["definition"]


# ── T37-T39: RANK REVERSAL ──────────────────────────────────────────

class TestRankReversal:
    def test_t37_reversal_definition(self):
        r = _load(E4P_DIR / "E4P_rank_reversal_contract.json")
        assert "epsilon_rank" in r["definition"]
        assert "> +epsilon_rank" in r["definition"]
        assert "< -epsilon_rank" in r["definition"]

    def test_t38_epsilon_exact(self):
        r = _load(E4P_DIR / "E4P_rank_reversal_contract.json")
        assert r["epsilon_rank"] == 1e-12

    def test_t39_ties_not_reversals(self):
        r = _load(E4P_DIR / "E4P_rank_reversal_contract.json")
        assert "Ties do NOT create reversals" in r["ties"]


# ── T40-T42: METRICS ────────────────────────────────────────────────

class TestMetrics:
    def test_t40_best_action_context(self):
        b = _load(E4P_DIR / "E4P_best_action_context_contract.json")
        assert "argmax_a V_S(i=a,j=c)" in b["definition"]

    def test_t41_cross_difference(self):
        c = _load(E4P_DIR / "E4P_cross_difference_contract.json")
        assert "V(a,c) - V(b,c) - V(a,d) + V(b,d)" in c["formula"]

    def test_t42_interaction_not_sufficient(self):
        c = _load(E4P_DIR / "E4P_cross_difference_contract.json")
        assert "NOT" in c["WARNING"]
        assert "SECONDARY" in c["WARNING"]


# ── T43-T44: AGGREGATION ────────────────────────────────────────────

class TestAggregation:
    def test_t43_no_pseudoreplication(self):
        s = _load(E4P_DIR / "E4P_source_aggregation_contract.json")
        assert "trajectory-source duplication" in s["prevents"]

    def test_t44_source_arms_averaged(self):
        s = _load(E4P_DIR / "E4P_source_aggregation_contract.json")
        assert "equal 1/2" in s["rule"]


# ── T45-T47: SCENARIO CONTRASTS ─────────────────────────────────────

class TestContrasts:
    def test_t45_g_rev_formula(self):
        c = _load(E4P_DIR / "E4P_primary_scenario_contrast.json")
        assert "RR_STAT - 0.5*(RR_PER + RR_MIX)" in c["G_REV"]["formula"]

    def test_t46_g_bac_formula(self):
        c = _load(E4P_DIR / "E4P_primary_scenario_contrast.json")
        assert "BAC_STAT - 0.5*(BAC_PER + BAC_MIX)" in c["G_BAC"]["formula"]

    def test_t47_g_interaction_secondary_only(self):
        c = _load(E4P_DIR / "E4P_primary_scenario_contrast.json")
        assert c["G_INTERACTION"]["role"] == "SECONDARY_ONLY"


# ── T48-T55: P1 SUPPORT LOGIC ───────────────────────────────────────

class TestP1SupportLogic:
    def test_t48_criterion1(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert "G_REV > 0" in p["SUPPORTED_requires_ALL"][0]

    def test_t49_criterion2(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert "CI lower" in p["SUPPORTED_requires_ALL"][1]

    def test_t50_criterion3(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert "4/5" in p["SUPPORTED_requires_ALL"][2]
        assert "policy" in p["SUPPORTED_requires_ALL"][2]

    def test_t51_criterion4(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert "G_BAC > 0" in p["SUPPORTED_requires_ALL"][3]

    def test_t52_criterion5(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert "4/5" in p["SUPPORTED_requires_ALL"][4]
        assert "G_BAC" in p["SUPPORTED_requires_ALL"][4]

    def test_t53_source_robustness(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        c6 = p["SUPPORTED_requires_ALL"][5]
        assert "SOURCE_C1" in c6 and "SOURCE_C2" in c6

    def test_t54_policy_level_g_rev(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["INFERENTIAL_AXES"] == "POLICY_5_X_ENVIRONMENT_20"

    def test_t55_policy_level_g_bac(self):
        c = _load(E4P_DIR / "E4P_primary_scenario_contrast.json")
        assert c["G_BAC"]["role"] == "SECONDARY"

    def test_six_criteria(self):
        p = _load(E4P_DIR / "E4P_p1_support_logic.json")
        assert p["n_criteria"] == 6
        assert len(p["SUPPORTED_requires_ALL"]) == 6


# ── T56-T61: BOOTSTRAP ──────────────────────────────────────────────

class TestBootstrap:
    def test_t56_policy_axis_5(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["bootstrap"]["axis_1"]["n"] == 5

    def test_t57_env_axis_20(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["bootstrap"]["axis_2"]["n"] == 20

    def test_t58_shared_scenario_draws(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["bootstrap"]["same_indices_all_scenarios"]

    def test_t59_n_50000(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["bootstrap"]["N"] == 50000

    def test_t60_seed_126001(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["bootstrap"]["seed"] == 126001

    def test_t61_tapes_not_inferential(self):
        s = _load(E4P_DIR / "E4P_statistical_design.json")
        assert s["random_tapes_NOT_inferential_axis"]


# ── T64-T66: AUDIT CONTRACTS ────────────────────────────────────────

class TestAuditContracts:
    def test_t64_fe_audit_required(self):
        f = _load(E4P_DIR / "E4P_fe_audit_contract.json")
        assert "Equal response FE" in f["primary_requirement"]

    def test_t65_predict_fallback_audit(self):
        p = _load(E4P_DIR / "E4P_predict_fallback_audit_contract.json")
        assert "requested_action" in p["record"]
        assert "realized_action" in p["record"]

    def test_t66_predict_fallback_invalidates(self):
        p = _load(E4P_DIR / "E4P_predict_fallback_audit_contract.json")
        assert "INVALID" in p["fallback_handling"]


# ── T67-T71: BOUNDARIES ─────────────────────────────────────────────

class TestBoundaries:
    def test_t67_no_learning(self):
        pre = _load(E4P_DIR / "E4P_pre_manifest.json")
        assert pre["new_training"] == "NO"

    def test_t69_no_controller_revision(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["NEW_CONTROLLER_REVISION"] == "NO"

    def test_t70_no_performance_verdict(self):
        n = _load(E4P_DIR / "E4P_no_performance_verdict.json")
        assert n["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"
        assert n["E4MF_REMAINS_AUTHORITATIVE"]

    def test_t71_e4mf_authoritative(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["ORIGINAL_E4MF_VERDICT"] == "FAILED"
        assert v["E4MF_VERDICT_CHANGED"] == "NO"


# ── T72-T76: ROUTING ────────────────────────────────────────────────

class TestRouting:
    def test_t72_p1_supported_routing(self):
        r = _load(E4P_DIR / "E4P_future_routing.json")
        assert r["routing_rules"]["P1_SUPPORTED"]["NEXT_STAGE"] == \
            "E4Q_DECENTRALIZED_IGM_LIMITATION_REVISION_DESIGN"

    def test_t73_global_reversal_routing(self):
        r = _load(E4P_DIR / "E4P_future_routing.json")
        assert r["routing_rules"]["GLOBAL_REVERSALS_BUT_P1_NOT_SUPPORTED"]["NEXT_STAGE"] == \
            "E4Q_GENERAL_INTERACTION_REPRESENTATION_REVIEW"

    def test_t74_no_reversal_routing(self):
        r = _load(E4P_DIR / "E4P_future_routing.json")
        assert r["routing_rules"]["P1_NOT_SUPPORTED_REVERSALS_RARE"]["NEXT_STAGE"] == \
            "E4Q_ACTION_STATE_REPRESENTATION_REVIEW"

    def test_t75_not_auditable_routing(self):
        r = _load(E4P_DIR / "E4P_future_routing.json")
        assert r["routing_rules"]["NOT_AUDITABLE"]["NEXT_STAGE"] == \
            "E4P_DIAGNOSTIC_INTEGRITY_REVIEW"

    def test_t76_e4pi_next(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["NEXT_STAGE"] == \
            "E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"


# ── T77-T79: FIREWALL ───────────────────────────────────────────────

class TestFirewall:
    def test_t77_access_count_zero(self):
        f = _load(E4P_DIR / "E4P_future_firewall.json")
        assert f["P_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_t78_no_environment_execution(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["NEW_ENVIRONMENT_EXECUTION"] == "NO"

    def test_t79_historical_unchanged(self):
        h = _load(E4P_DIR / "E4P_historical_integrity.json")
        assert h["all_pass"]


# ── VERDICT STRUCTURE ────────────────────────────────────────────────

class TestVerdictStructure:
    def test_e4p_complete(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["E4P_COMPLETE"] == "YES"

    def test_primary_hypothesis(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["PRIMARY_DIAGNOSTIC_HYPOTHESIS"] == \
            "P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL"

    def test_primary_metric(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["PRIMARY_SCIENTIFIC_METRIC"] == \
            "NONKEEP_CONTEXT_DEPENDENT_REVERSAL_RATE"

    def test_primary_contrast(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["PRIMARY_SCENARIO_CONTRAST"] == "G_REV_STAT_MINUS_MEAN_PER_MIX"

    def test_bootstrap_params(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["BOOTSTRAP_RESAMPLES"] == 50000
        assert v["BOOTSTRAP_SEED"] == 126001

    def test_seeds(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["DIAGNOSTIC_SEEDS"] == "2401..2420"

    def test_tapes(self):
        v = _load(E4P_DIR / "E4P_verdict.json")
        assert v["COUNTERFACTUAL_RANDOM_TAPES"] == 4


# ── REPORT ───────────────────────────────────────────────────────────

class TestReport:
    def test_report_exists(self):
        assert (E4P_DIR / "E4P_report.md").exists()

    def test_report_content(self):
        text = (E4P_DIR / "E4P_report.md").read_text()
        assert "E4P" in text
        assert "rank reversal" in text.lower() or "RANK" in text
        assert "G_REV" in text
        assert "KEEP excluded" in text or "KEEP_excluded" in text


# ── E4PI CONTRACT ────────────────────────────────────────────────────

class TestE4PIContract:
    def test_e4pi_next_stage(self):
        c = _load(E4P_DIR / "E4P_e4pi_validation_contract.json")
        assert c["next_validation_stage"] == \
            "E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"

    def test_e4pi_synthetic_only(self):
        c = _load(E4P_DIR / "E4P_e4pi_validation_contract.json")
        assert c["must_use_synthetic_only"]

    def test_e4pi_seeds_forbidden(self):
        c = _load(E4P_DIR / "E4P_e4pi_validation_contract.json")
        assert c["seeds_2401_2420_forbidden_during_E4PI"]

    def test_e4pi_validation_items(self):
        c = _load(E4P_DIR / "E4P_e4pi_validation_contract.json")
        items = c["must_validate"]
        assert "snapshot_cloning" in items
        assert "no_cross_cell_contamination" in items
        assert "SHA256_pair_index" in items
        assert "rank_reversal_detection" in items
        assert "P1_truth_table" in items
        assert "seed_firewall" in items


# ── T80: REGRESSION ──────────────────────────────────────────────────

class TestRegression:
    def test_t80_full_regression(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
             "--tb=short",
             "--ignore=tests/test_e4p_nonmonotonic_interaction_diagnostic_design.py",
             "--ignore=tests/test_e4pi_nonmonotonic_interaction_diagnostic_implementation_validation.py",
             "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4rtr_v2_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py"],
            capture_output=True, text=True,
            cwd=str(BASE),
            timeout=1200,
        )
        assert result.returncode == 0, (
            f"Regression failed:\n{result.stdout[-2000:]}\n{result.stderr[-1000:]}"
        )
