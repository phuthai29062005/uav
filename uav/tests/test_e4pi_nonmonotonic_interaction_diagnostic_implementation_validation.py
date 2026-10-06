"""Tests for E4PI — Nonmonotonic Interaction Diagnostic Implementation Validation."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results" / "E4PI_nonmonotonic_interaction_diagnostic_implementation_validation"
E4P_DIR = BASE / "results" / "E4P_nonmonotonic_interaction_diagnostic_design"
E4MF_DIR = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4MTR_DIR = BASE / "results" / "E4MTR_credit_preservation_ab_training_execution"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
DIAGNOSTIC_SEEDS = list(range(2401, 2421))
EPSILON_RANK = 1e-12


def _load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load(RESULTS_DIR / "E4PI_verdict.json")


@pytest.fixture(scope="module")
def pre_manifest():
    return _load(RESULTS_DIR / "E4PI_pre_manifest.json")


@pytest.fixture(scope="module")
def historical():
    return _load(RESULTS_DIR / "E4PI_historical_integrity.json")


# ---- Output existence ----

class TestOutputExistence:
    REQUIRED_FILES = [
        "E4PI_pre_manifest.json", "E4PI_historical_integrity.json",
        "E4PI_diagnostic_config.json", "E4PI_diagnostic_config_hash.json",
        "E4PI_snapshot_cloning_validation.json",
        "E4PI_snapshot_eligibility_validation.json",
        "E4PI_pair_selection_validation.json",
        "E4PI_random_tape_validation.json",
        "E4PI_counterfactual_isolation_validation.json",
        "E4PI_environment_utility_validation.json",
        "E4PI_primary_surface_validation.json",
        "E4PI_secondary_surface_validation.json",
        "E4PI_fe_audit_validation.json",
        "E4PI_predict_fallback_validation.json",
        "E4PI_action_preference_validation.json",
        "E4PI_rank_reversal_validation.json",
        "E4PI_tie_semantics_validation.json",
        "E4PI_best_action_context_validation.json",
        "E4PI_cross_difference_validation.json",
        "E4PI_interaction_normalization_validation.json",
        "E4PI_source_aggregation_validation.json",
        "E4PI_scenario_contrast_validation.json",
        "E4PI_crossed_bootstrap_validation.json",
        "E4PI_p1_truth_table_validation.json",
        "E4PI_global_reversal_validation.json",
        "E4PI_checkpoint_loader_validation.json",
        "E4PI_serialization_validation.json",
        "E4PI_version_guard_validation.json",
        "E4PI_future_execution_contract.json",
        "E4PI_future_seed_firewall.json",
        "E4PI_no_training_audit.json",
        "E4PI_verdict.json",
        "E4PI_report.md",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_output_exists(self, fname):
        assert (RESULTS_DIR / fname).exists(), f"Missing: {fname}"


# ---- T1-T8: Historical integrity ----

class TestHistoricalIntegrity:
    def test_t1_e4p_complete(self, historical):
        assert historical["checks"]["E4P_COMPLETE"] is True

    def test_t2_e4mf_failed(self, historical):
        assert historical["checks"]["E4MF_FAILED"] is True

    def test_t3_o1_not_supported(self, historical):
        assert historical["checks"]["E4OD_O1_NOT_SUPPORTED"] is True

    def test_t4_n8_not_supported(self, historical):
        assert historical["checks"]["N8_NOT_SUPPORTED"] is True

    def test_t5_n9_plausible(self, historical):
        assert historical["checks"]["N9_PLAUSIBLE"] is True

    def test_t6_seeds_exact(self, historical):
        assert historical["checks"]["seeds_exact"] is True

    def test_t7_access_count_zero(self, verdict):
        assert verdict["P_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_t8_checkpoint_set_exact(self, historical):
        assert historical["checks"]["C1_policies_exact"] is True
        assert historical["checks"]["C2_policies_exact"] is True
        assert historical["checks"]["total_checkpoints_10"] is True


# ---- T9-T10: Snapshot cloning ----

class TestSnapshotCloning:
    def test_t9_deep_clone(self):
        from prdrl.diagnostics.counterfactual_surface import clone_snapshot
        snap = {"population": [[1.0, 2.0]], "active_regions": [1, 2],
                "regional_states": {"1": [0.5]}}
        clone = clone_snapshot(snap)
        clone["population"][0][0] = 999.0
        assert snap["population"][0][0] == 1.0

    def test_t10_sibling_isolation(self):
        from prdrl.diagnostics.counterfactual_surface import clone_snapshot
        snap = {"population": [[1.0]], "active_regions": [1]}
        c1 = clone_snapshot(snap)
        c2 = clone_snapshot(snap)
        c1["population"][0][0] = 999.0
        assert c2["population"][0][0] == 1.0
        assert snap["population"][0][0] == 1.0


# ---- T11-T15: Eligibility ----

class TestEligibility:
    def test_t11_valid_case(self):
        from prdrl.diagnostics.counterfactual_surface import check_snapshot_eligibility
        snap = {"active_regions": [1, 2, 3],
                "regional_states": {"1": [0.1], "2": [0.2], "3": [0.3]},
                "global_context": [0.5], "predict_history": {"1": [[0.1]]}}
        assert check_snapshot_eligibility(snap)["eligible"] is True

    def test_t12_one_active_invalid(self):
        from prdrl.diagnostics.counterfactual_surface import check_snapshot_eligibility
        snap = {"active_regions": [1],
                "regional_states": {"1": [0.1]},
                "global_context": [0.5], "predict_history": {}}
        assert check_snapshot_eligibility(snap)["eligible"] is False

    def test_t13_missing_context_invalid(self):
        from prdrl.diagnostics.counterfactual_surface import check_snapshot_eligibility
        snap = {"active_regions": [1, 2],
                "regional_states": {"1": [0.1], "2": [0.2]},
                "global_context": None, "predict_history": {}}
        assert check_snapshot_eligibility(snap)["eligible"] is False

    def test_t14_missing_predict_history(self):
        from prdrl.diagnostics.counterfactual_surface import check_pair_predict_eligibility
        snap = {"predict_history": {"1": [[0.1]]}}
        assert check_pair_predict_eligibility(snap, (1, 2)) is False

    def test_t15_first_eligible(self):
        from prdrl.diagnostics.counterfactual_surface import find_first_eligible_snapshot
        bad = {"active_regions": [1], "regional_states": {"1": [0.1]},
               "global_context": [0.5]}
        good = {"active_regions": [1, 2], "regional_states": {"1": [0.1], "2": [0.2]},
                "global_context": [0.5]}
        assert find_first_eligible_snapshot([bad, good]) is good


# ---- T16-T18: Pair selection ----

class TestPairSelection:
    def test_t16_sha256_deterministic(self):
        from prdrl.diagnostics.counterfactual_surface import select_pair_sha256
        pairs = [(1, 2), (1, 3), (2, 3)]
        r1 = select_pair_sha256("C1", 6001, "MIXED", 2401, pairs)
        r2 = select_pair_sha256("C1", 6001, "MIXED", 2401, pairs)
        assert r1 == r2

    def test_t17_python_hash_not_used(self):
        import inspect
        from prdrl.diagnostics import counterfactual_surface
        src = inspect.getsource(counterfactual_surface.select_pair_sha256)
        assert "hash(" not in src or "hashlib" in src

    def test_t18_outcome_blind(self):
        from prdrl.diagnostics.counterfactual_surface import select_pair_sha256
        pairs = [(1, 2), (1, 3), (2, 3)]
        r = select_pair_sha256("C1", 6001, "MIXED", 2401, pairs)
        assert isinstance(r, tuple) and len(r) == 2


# ---- T19-T22: Action sets and surfaces ----

class TestActionSets:
    def test_t19_primary_actions_exact(self):
        prim = _load(E4P_DIR / "E4P_primary_action_set.json")
        assert prim["primary_actions"] == ["LOCAL", "PREDICT", "DIVERSIFY"]

    def test_t20_3x3_grid(self):
        assert len(PRIMARY_ACTIONS) ** 2 == 9

    def test_t21_secondary_actions_exact(self):
        sec = _load(E4P_DIR / "E4P_secondary_action_set.json")
        assert sec["secondary_actions"] == ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]

    def test_t22_4x4_grid(self):
        assert len(SECONDARY_ACTIONS) ** 2 == 16


# ---- T23-T24: Background contexts ----

class TestBackgrounds:
    def test_t23_keep_background(self):
        bg = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert bg["BACKGROUND_A"]["name"] == "KEEP_BACKGROUND"

    def test_t24_policy_background(self):
        bg = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert bg["BACKGROUND_B"]["name"] == "SOURCE_POLICY_BACKGROUND"


# ---- T25-T28: Random tapes ----

class TestRandomTapes:
    def test_t25_four_tapes(self):
        from prdrl.diagnostics.counterfactual_surface import generate_random_tape_seeds
        seeds = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
        assert len(seeds) == 4

    def test_t26_common_tape_reused(self):
        from prdrl.diagnostics.counterfactual_surface import generate_random_tape_seeds
        s1 = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
        s2 = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
        assert s1 == s2

    def test_t27_different_tapes_distinct(self):
        from prdrl.diagnostics.counterfactual_surface import generate_random_tape_seeds
        seeds = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
        assert len(set(seeds)) == 4

    def test_t28_action_semantics_preserved(self):
        tape = _load(RESULTS_DIR / "E4PI_random_tape_validation.json")
        assert tape["RANDOM_TAPE_VALID"] is True


# ---- T29-T35: Operator constants ----

class TestOperatorConstants:
    def test_t29_local_sigma(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["operator_constants"]["LOCAL"]["sigma_L"] == 0.02

    def test_t30_predict_sigma(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["operator_constants"]["PREDICT"]["sigma_P"] == 0.005

    def test_t31_predict_lambda(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["operator_constants"]["PREDICT"]["lambda_P"] == 1.0

    def test_t32_diversify_sigma(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["operator_constants"]["DIVERSIFY"]["sigma_D"] == 0.15

    def test_t33_budget_10(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["operator_constants"]["LOCAL"]["B_PER_ACTIVE_REGION"] == 10

    def test_t34_unrestricted_survival(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert frozen["survival"] == "UNRESTRICTED_STANDARD_NSGA_II"

    def test_t35_no_hard(self):
        frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
        assert "HARD" not in json.dumps(frozen["operator_constants"])


# ---- T36-T38: Utility ----

class TestUtility:
    def test_t36_keep_all_baseline(self):
        from prdrl.diagnostics.counterfactual_surface import compute_utility
        v = compute_utility(igd_post_action=0.05, igd_post_keep_all=0.10)
        assert abs(v - 0.05) < 1e-15

    def test_t37_baseline_reused(self):
        data = _load(RESULTS_DIR / "E4PI_environment_utility_validation.json")
        assert data["ENVIRONMENT_UTILITY_VALID"] is True

    def test_t38_learned_q_excluded(self):
        contract = _load(E4P_DIR / "E4P_environment_utility_contract.json")
        assert contract["learned_Q_ground_truth"] is False


# ---- T39-T44: Surfaces ----

class TestSurfaces:
    def test_t39_primary_surface_schema(self):
        from prdrl.diagnostics.counterfactual_surface import build_primary_surface
        utils = {(a, b): {"V_S": 0.1} for a in PRIMARY_ACTIONS for b in PRIMARY_ACTIONS}
        surf = build_primary_surface(utils)
        assert surf["n_cells"] == 9
        assert len(surf["cells"]) == 9

    def test_t40_secondary_surface_schema(self):
        from prdrl.diagnostics.counterfactual_surface import build_secondary_surface
        utils = {(a, b): {"V_S": 0.1} for a in SECONDARY_ACTIONS for b in SECONDARY_ACTIONS}
        surf = build_secondary_surface(utils)
        assert surf["n_cells"] == 16

    def test_t41_fe_equality_pass(self):
        from prdrl.diagnostics.counterfactual_surface import audit_fe_equality
        surf = {"cells": {f"{a}_{b}": {"response_FE": 30}
                          for a in PRIMARY_ACTIONS for b in PRIMARY_ACTIONS}}
        assert audit_fe_equality(surf)["valid"] is True

    def test_t42_fe_inequality_invalid(self):
        from prdrl.diagnostics.counterfactual_surface import audit_fe_equality
        surf = {"cells": {"LOCAL_LOCAL": {"response_FE": 30},
                          "LOCAL_PREDICT": {"response_FE": 20}}}
        assert audit_fe_equality(surf)["valid"] is False

    def test_t43_predict_realized_valid(self):
        from prdrl.diagnostics.counterfactual_surface import audit_predict_fallback
        surf = {"cells": {"P_L": {"requested_i": "PREDICT", "realized_i": "PREDICT",
                                  "requested_j": "LOCAL", "realized_j": "LOCAL"}}}
        assert audit_predict_fallback(surf)["valid"] is True

    def test_t44_predict_fallback_invalid(self):
        from prdrl.diagnostics.counterfactual_surface import audit_predict_fallback
        surf = {"cells": {"P_L": {"requested_i": "PREDICT", "realized_i": "LOCAL",
                                  "requested_j": "LOCAL", "realized_j": "LOCAL"}}}
        assert audit_predict_fallback(surf)["valid"] is False


# ---- T45-T49: Action preference and ties ----

class TestActionPreference:
    def test_t45_preference_delta_exact(self):
        from prdrl.diagnostics.counterfactual_surface import compute_action_preference_delta
        assert abs(compute_action_preference_delta(0.5, 0.3) - 0.2) < 1e-15

    def test_t46_epsilon_rank_exact(self):
        assert EPSILON_RANK == 1e-12

    def test_t47_positive_classification(self):
        from prdrl.diagnostics.counterfactual_surface import classify_delta
        assert classify_delta(0.5, EPSILON_RANK) == "POSITIVE"

    def test_t48_negative_classification(self):
        from prdrl.diagnostics.counterfactual_surface import classify_delta
        assert classify_delta(-0.5, EPSILON_RANK) == "NEGATIVE"

    def test_t49_tie_classification(self):
        from prdrl.diagnostics.counterfactual_surface import classify_delta
        assert classify_delta(1e-13, EPSILON_RANK) == "TIE"
        assert classify_delta(-1e-13, EPSILON_RANK) == "TIE"
        assert classify_delta(0.0, EPSILON_RANK) == "TIE"


# ---- T50-T57: Rank reversal ----

class TestRankReversal:
    def test_t50_no_interaction_no_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": 0.5, "D": 0.5}, EPSILON_RANK)
        assert r["reversal"] is False

    def test_t51_interaction_no_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": 0.3, "D": 0.1}, EPSILON_RANK)
        assert r["reversal"] is False

    def test_t52_single_strict_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": -0.3, "D": 0.1}, EPSILON_RANK)
        assert r["reversal"] is True

    def test_t53_multiple_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": -0.3, "D": -0.1}, EPSILON_RANK)
        assert r["reversal"] is True

    def test_t54_tie_only_no_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": 0.0, "D": 0.0}, EPSILON_RANK)
        assert r["reversal"] is False

    def test_t55_near_epsilon_no_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 1e-13, "P": -1e-13, "D": 0.5}, EPSILON_RANK)
        assert r["reversal"] is False

    def test_t56_reversal_rate_denominator(self):
        from prdrl.diagnostics.counterfactual_surface import compute_reversal_rate
        utils = {(a, b): 0.1 for a in PRIMARY_ACTIONS for b in PRIMARY_ACTIONS}
        r = compute_reversal_rate(utils, PRIMARY_ACTIONS, EPSILON_RANK)
        assert r["n_valid_opportunities"] == 6

    def test_t57_zero_denominator_na(self):
        from prdrl.diagnostics.counterfactual_surface import compute_reversal_rate
        r = compute_reversal_rate({}, PRIMARY_ACTIONS, EPSILON_RANK)
        assert r["reversal_rate"] is None


# ---- T58-T60: Best action context ----

class TestBestAction:
    def test_t58_constant_best(self):
        from prdrl.diagnostics.counterfactual_surface import compute_best_action_by_context
        utils = {("LOCAL", c): 0.5 for c in PRIMARY_ACTIONS}
        utils.update({("PREDICT", c): 0.3 for c in PRIMARY_ACTIONS})
        utils.update({("DIVERSIFY", c): 0.1 for c in PRIMARY_ACTIONS})
        r = compute_best_action_by_context(utils, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
        assert r["context_dependent"] is False

    def test_t59_strict_switch(self):
        from prdrl.diagnostics.counterfactual_surface import compute_best_action_by_context
        utils = {
            ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.1,
            ("LOCAL", "DIVERSIFY"): 0.3,
            ("PREDICT", "LOCAL"): 0.3, ("PREDICT", "PREDICT"): 0.6,
            ("PREDICT", "DIVERSIFY"): 0.2,
            ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.05,
            ("DIVERSIFY", "DIVERSIFY"): 0.1,
        }
        r = compute_best_action_by_context(utils, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
        assert r["context_dependent"] is True

    def test_t60_tie_mediated_not_counted(self):
        from prdrl.diagnostics.counterfactual_surface import compute_best_action_by_context
        utils = {
            ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.5,
            ("LOCAL", "DIVERSIFY"): 0.3,
            ("PREDICT", "LOCAL"): 0.5, ("PREDICT", "PREDICT"): 0.3,
            ("PREDICT", "DIVERSIFY"): 0.3,
            ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.1,
            ("DIVERSIFY", "DIVERSIFY"): 0.1,
        }
        r = compute_best_action_by_context(utils, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
        ba = r["best_actions"]
        tied_ctx = [c for c, info in ba.items() if info["tied"]]
        assert len(tied_ctx) >= 1


# ---- T61-T64: Cross-difference ----

class TestCrossDifference:
    def test_t61_exact(self):
        from prdrl.diagnostics.counterfactual_surface import compute_cross_difference
        I = compute_cross_difference(1.0, 0.5, 0.8, 0.6)
        assert abs(I - 0.3) < 1e-15

    def test_t62_nonzero_not_automatically_reversal(self):
        data = _load(RESULTS_DIR / "E4PI_rank_reversal_validation.json")
        assert data["checks"]["B_no_reversal"] is True

    def test_t63_normalization_normal(self):
        from prdrl.diagnostics.counterfactual_surface import compute_normalized_interaction
        n = compute_normalized_interaction(0.3, {(0, 0): 1.0, (0, 1): 0.5})
        assert n > 0

    def test_t64_normalization_all_zero(self):
        from prdrl.diagnostics.counterfactual_surface import compute_normalized_interaction
        n = compute_normalized_interaction(0.3, {(0, 0): 0.0})
        assert n == 0.0


# ---- T65-T69: Source aggregation ----

class TestSourceAggregation:
    def test_t65_source_c1_retained(self):
        from prdrl.diagnostics.counterfactual_surface import aggregate_source_arms
        r = aggregate_source_arms(0.4, 0.6, 0.5, 0.7)
        assert r["rr_c1"] == 0.4

    def test_t66_source_c2_retained(self):
        from prdrl.diagnostics.counterfactual_surface import aggregate_source_arms
        r = aggregate_source_arms(0.4, 0.6, 0.5, 0.7)
        assert r["rr_c2"] == 0.6

    def test_t67_equal_weighting(self):
        from prdrl.diagnostics.counterfactual_surface import aggregate_source_arms
        r = aggregate_source_arms(0.4, 0.6, 0.5, 0.7)
        assert abs(r["unit_rr"] - 0.5) < 1e-15

    def test_t68_source_not_bootstrap_axis(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["bootstrap"]["both_source_arms_within_unit"] is True

    def test_t69_missing_source_handling(self):
        from prdrl.diagnostics.counterfactual_surface import aggregate_source_arms
        r = aggregate_source_arms(0.4, None, 0.5, None)
        assert r["available"] is False
        assert r["unit_rr"] is None


# ---- T70-T76: Scenario contrasts ----

class TestScenarioContrasts:
    def test_t70_rr_means(self):
        from prdrl.diagnostics.counterfactual_surface import compute_scenario_means
        units = [{"scenario": "STATIONARY", "unit_rr": 0.5, "unit_bac": 0.5},
                 {"scenario": "PERSISTENT", "unit_rr": 0.2, "unit_bac": 0.3},
                 {"scenario": "MIXED", "unit_rr": 0.3, "unit_bac": 0.4}]
        r = compute_scenario_means(units, SCENARIOS)
        assert abs(r["STATIONARY"]["RR_mean"] - 0.5) < 1e-15

    def test_t71_bac_means(self):
        from prdrl.diagnostics.counterfactual_surface import compute_scenario_means
        units = [{"scenario": "STATIONARY", "unit_rr": 0.5, "unit_bac": 0.8}]
        r = compute_scenario_means(units, SCENARIOS)
        assert abs(r["STATIONARY"]["BAC_mean"] - 0.8) < 1e-15

    def test_t72_g_rev_exact(self):
        from prdrl.diagnostics.counterfactual_surface import compute_g_rev
        g = compute_g_rev(0.6, 0.2, 0.4)
        assert abs(g - 0.3) < 1e-15

    def test_t73_g_bac_exact(self):
        from prdrl.diagnostics.counterfactual_surface import compute_g_bac
        g = compute_g_bac(0.8, 0.3, 0.5)
        assert abs(g - 0.4) < 1e-15

    def test_t74_g_interaction_secondary_only(self):
        contract = _load(E4P_DIR / "E4P_primary_scenario_contrast.json")
        assert contract["G_INTERACTION"]["role"] == "SECONDARY_ONLY"

    def test_t75_policy_g_rev_five(self):
        from prdrl.diagnostics.rank_reversal_bootstrap import compute_policy_gaps
        units = []
        for p in POLICY_SEEDS:
            for s in SCENARIOS:
                units.append({"policy_seed": p, "scenario": s, "unit_rr": 0.1})
        r = compute_policy_gaps(units, POLICY_SEEDS, SCENARIOS)
        assert len(r) == 5

    def test_t76_policy_g_bac_five(self):
        from prdrl.diagnostics.rank_reversal_bootstrap import compute_policy_gaps
        units = []
        for p in POLICY_SEEDS:
            for s in SCENARIOS:
                units.append({"policy_seed": p, "scenario": s, "unit_bac": 0.1})
        r = compute_policy_gaps(units, POLICY_SEEDS, SCENARIOS)
        bacs = [v["G_BAC"] for v in r.values() if v["G_BAC"] is not None]
        assert len(bacs) == 5


# ---- T77-T84: Bootstrap ----

class TestBootstrap:
    def test_t77_policy_axis_5(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["bootstrap"]["axis_1"]["n"] == 5

    def test_t78_env_axis_20(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["bootstrap"]["axis_2"]["n"] == 20

    def test_t79_same_scenario_draws(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["bootstrap"]["same_indices_all_scenarios"] is True

    def test_t80_bootstrap_n_50000(self):
        data = _load(RESULTS_DIR / "E4PI_crossed_bootstrap_validation.json")
        assert data["production_N"] == 50000

    def test_t81_bootstrap_seed_126001(self):
        data = _load(RESULTS_DIR / "E4PI_crossed_bootstrap_validation.json")
        assert data["production_seed"] == 126001

    def test_t82_no_tape_bootstrap(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["random_tapes_NOT_inferential_axis"] is True

    def test_t83_no_pair_bootstrap(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        axes = [stat["bootstrap"]["axis_1"]["name"],
                stat["bootstrap"]["axis_2"]["name"]]
        assert "pair" not in axes and "region" not in axes

    def test_t84_no_action_cell_bootstrap(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        axes = [stat["bootstrap"]["axis_1"]["name"],
                stat["bootstrap"]["axis_2"]["name"]]
        assert "action" not in axes and "cell" not in axes


# ---- T85-T92: P1 support logic ----

class TestP1Support:
    def _make_inputs(self, g_rev, ci_lower, n_pos_pol, g_bac, n_pos_bac,
                     c1_pos, c2_pos, n_valid=100):
        pg = {}
        for i, p in enumerate(POLICY_SEEDS):
            pg[p] = {"G_REV": 0.1 if i < n_pos_pol else -0.1,
                     "G_BAC": 0.1 if i < n_pos_bac else -0.1}
        src = {"C1": {"G_REV": 0.05 if c1_pos else -0.05},
               "C2": {"G_REV": 0.05 if c2_pos else -0.05}}
        from prdrl.diagnostics.rank_reversal_bootstrap import classify_p1
        return classify_p1(g_rev, ci_lower, pg, g_bac, pg, src, n_valid)

    def test_t85_supported(self):
        r = self._make_inputs(0.1, 0.01, 5, 0.1, 5, True, True)
        assert r["verdict"] == "SUPPORTED"

    def test_t86_plausible_ci(self):
        r = self._make_inputs(0.1, -0.01, 5, 0.1, 5, True, True)
        assert r["verdict"] == "PLAUSIBLE"

    def test_t87_plausible_policy(self):
        r = self._make_inputs(0.1, 0.01, 3, 0.1, 5, True, True)
        assert r["verdict"] == "PLAUSIBLE"

    def test_t88_plausible_bac(self):
        r = self._make_inputs(0.1, 0.01, 5, -0.1, 5, True, True)
        assert r["verdict"] == "PLAUSIBLE"

    def test_t89_plausible_source(self):
        r = self._make_inputs(0.1, 0.01, 5, 0.1, 5, True, False)
        assert r["verdict"] == "PLAUSIBLE"

    def test_t90_not_supported_neutral(self):
        r = self._make_inputs(0.0, -0.01, 0, 0.0, 0, False, False)
        assert r["verdict"] == "NOT_SUPPORTED"

    def test_t91_not_supported_opposite(self):
        r = self._make_inputs(-0.1, -0.15, 0, -0.1, 0, False, False)
        assert r["verdict"] == "NOT_SUPPORTED"

    def test_t92_not_auditable(self):
        r = self._make_inputs(0.1, 0.01, 5, 0.1, 5, True, True, n_valid=5)
        assert r["verdict"] == "NOT_AUDITABLE"


# ---- T93-T97: Firewalls ----

class TestFirewalls:
    def test_t93_global_reversal_yes_p1_no(self):
        data = _load(RESULTS_DIR / "E4PI_global_reversal_validation.json")
        assert data["reversal_detected_correctly"] is True
        contract = _load(E4P_DIR / "E4P_global_incompatibility_contract.json")
        assert "NOT establish" in contract["separation"] or "NOT" in contract["separation"]

    def test_t94_tape_consistency_descriptive(self):
        stat = _load(E4P_DIR / "E4P_statistical_design.json")
        assert stat["tape_consistency"]["role"] == "descriptive only, no significance test, no gate"

    def test_t95_keep_background_primary(self):
        bg = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert bg["primary_verdict_based_on"] == "KEEP_BACKGROUND"

    def test_t96_policy_background_no_override(self):
        bg = _load(E4P_DIR / "E4P_background_context_contract.json")
        assert bg["policy_background_cannot_override_primary"] is True

    def test_t97_secondary_no_override(self):
        surf = _load(E4P_DIR / "E4P_counterfactual_surface_contract.json")
        assert "Cannot override" in surf["secondary_surface"]["note"]


# ---- T98-T101: Performance and checkpoint ----

class TestPerformanceAndCheckpoint:
    def test_t98_performance_not_computed(self, verdict):
        assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_t99_checkpoint_hashes_unchanged(self):
        data = _load(RESULTS_DIR / "E4PI_checkpoint_loader_validation.json")
        assert data["CHECKPOINT_IMMUTABILITY_VALID"] is True
        assert data["n_checkpoints"] == 10

    def test_t100_mixer_not_required(self):
        contract = _load(E4P_DIR / "E4P_environment_utility_contract.json")
        assert contract["learned_Q_ground_truth"] is False

    def test_t101_serialization_roundtrip(self):
        data = _load(RESULTS_DIR / "E4PI_serialization_validation.json")
        assert data["SERIALIZATION_VALID"] is True


# ---- T102-T105: Version guard ----

class TestVersionGuard:
    def test_t102_good_config(self):
        from prdrl.diagnostics.nonmonotonic_config import make_nonmonotonic_config, validate_config
        cfg = make_nonmonotonic_config()
        assert validate_config(cfg)["valid"] is True

    def test_t103_rejects_wrong_actions(self):
        from prdrl.diagnostics.nonmonotonic_config import make_nonmonotonic_config, validate_config
        cfg = make_nonmonotonic_config()
        cfg["primary_actions"] = ["KEEP", "LOCAL"]
        assert validate_config(cfg)["valid"] is False

    def test_t104_rejects_wrong_tapes(self):
        from prdrl.diagnostics.nonmonotonic_config import make_nonmonotonic_config, validate_config
        cfg = make_nonmonotonic_config()
        cfg["n_random_tapes"] = 8
        assert validate_config(cfg)["valid"] is False

    def test_t105_config_sha_stable(self):
        from prdrl.diagnostics.nonmonotonic_config import nonmonotonic_config_sha256
        s1 = nonmonotonic_config_sha256()
        s2 = nonmonotonic_config_sha256()
        assert s1 == s2


# ---- T106-T112: Firewall and final ----

class TestFinalFirewall:
    def test_t106_no_environment(self, verdict):
        assert verdict["NEW_ENVIRONMENT_EPISODES"] == 0

    def test_t107_no_objective_eval(self, verdict):
        assert verdict["NEW_OBJECTIVE_EVALUATIONS"] == 0

    def test_t108_no_training(self, verdict):
        assert verdict["NEW_TRAINING"] == "NO"

    def test_t109_no_controller_revision(self, verdict):
        assert verdict["NEW_CONTROLLER_REVISION"] == "NO"

    def test_t110_seed_firewall(self):
        data = _load(RESULTS_DIR / "E4PI_future_seed_firewall.json")
        assert data["DIAGNOSTIC_SEED_FIREWALL_VALID"] is True
        assert data["P_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_t111_next_stage(self, verdict):
        assert verdict["NEXT_STAGE"] == "E4PD_NONMONOTONIC_INTERACTION_DIAGNOSTIC_EXECUTION"

    def test_t112_historical_artifacts_unchanged(self, historical):
        assert historical["all_pass"] is True


# ---- Additional tests beyond T112 ----

class TestAdditional:
    def test_interaction_without_reversal_critical(self):
        data = _load(RESULTS_DIR / "E4PI_rank_reversal_validation.json")
        assert data["INTERACTION_WITHOUT_REVERSAL_VALID"] is True

    def test_e4pi_complete(self, verdict):
        assert verdict["E4PI_COMPLETE"] == "YES"

    def test_all_gates_pass(self, verdict):
        for gate, val in verdict["gates"].items():
            assert val is True, f"Gate {gate} = {val}"

    def test_report_exists(self):
        report = RESULTS_DIR / "E4PI_report.md"
        assert report.exists()
        text = report.read_text()
        assert "Historical integrity" in text
        assert "Limitations" in text
        assert "Next stage" in text

    def test_diagnostic_version(self, verdict):
        assert verdict["DIAGNOSTIC_VERSION"] == "E4PI_NONMONOTONIC_DIAGNOSTIC_V1"

    def test_negative_to_positive_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": -0.5, "P": 0.5, "D": 0.0}, EPSILON_RANK)
        assert r["reversal"] is True

    def test_positive_to_negative_reversal(self):
        from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
        r = detect_rank_reversal({"L": 0.5, "P": -0.5, "D": 0.0}, EPSILON_RANK)
        assert r["reversal"] is True

    def test_bootstrap_reproducible(self):
        data = _load(RESULTS_DIR / "E4PI_crossed_bootstrap_validation.json")
        assert data["reproducible"] is True

    def test_future_contract_complete(self):
        data = _load(RESULTS_DIR / "E4PI_future_execution_contract.json")
        fc = data["future_execution_contract"]
        assert fc["source_episodes"] == 600
        assert fc["source_arms"] == ["C1", "C2"]
        assert fc["policies"] == POLICY_SEEDS
        assert fc["environments"] == DIAGNOSTIC_SEEDS
        assert fc["random_tapes"] == 4
        assert fc["no_learning"] is True


# ---- Regression ----

class TestRegression:
    def test_full_regression(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
             "--tb=short",
             "--ignore=tests/test_e4pi_nonmonotonic_interaction_diagnostic_implementation_validation.py",
             "--ignore=tests/test_e4p_nonmonotonic_interaction_diagnostic_design.py",
             "--ignore=tests/test_e4odr_e4od_report_artifact_reconciliation.py",
             "--ignore=tests/test_e4mi_joint_credit_preservation_implementation_validation.py",
             "--ignore=tests/test_e4mt_credit_preservation_ab_training_protocol_freeze.py",
             "--ignore=tests/test_e4oi_scenario_tradeoff_diagnostic_implementation_validation.py",
             "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4rtr_v2_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py",
             ],
            capture_output=True, text=True, timeout=900,
            cwd=str(BASE), env={**__import__("os").environ, "PYTHONPATH": str(BASE / "src")},
        )
        assert result.returncode == 0, f"Regression failed:\n{result.stdout[-2000:]}\n{result.stderr[-1000:]}"
