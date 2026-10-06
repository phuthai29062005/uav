"""Tests for E4PD — Nonmonotonic Interaction Diagnostic Execution."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results" / "E4PD_nonmonotonic_interaction_diagnostic_execution"
E4MF_DIR = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4MTR_DIR = BASE / "results" / "E4MTR_credit_preservation_ab_training_execution"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
DIAGNOSTIC_SEEDS = list(range(2401, 2421))
EPSILON_RANK = 1e-12
EXPECTED_CONFIG_SHA = "ee1f730fd742f099efb87b2bf3fa8c1c43c3bc565e3c6fb5699dab89fac47cd0"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def verdict():
    return _load(RESULTS_DIR / "E4PD_verdict.json")


@pytest.fixture(scope="module")
def p0():
    return _load(RESULTS_DIR / "E4PD_pre_execution_manifest.json")


@pytest.fixture(scope="module")
def p1_verdict():
    return _load(RESULTS_DIR / "E4PD_p1_verdict.json")


@pytest.fixture(scope="module")
def bootstrap():
    return _load(RESULTS_DIR / "E4PD_bootstrap.json")


@pytest.fixture(scope="module")
def completeness():
    return _load(RESULTS_DIR / "E4PD_completeness_audit.json")


@pytest.fixture(scope="module")
def global_incompat():
    return _load(RESULTS_DIR / "E4PD_global_incompatibility.json")


@pytest.fixture(scope="module")
def post_integrity():
    return _load(RESULTS_DIR / "E4PD_post_integrity.json")


@pytest.fixture(scope="module")
def source_rows():
    return _load_csv(RESULTS_DIR / "E4PD_source_episode_manifest.csv")


@pytest.fixture(scope="module")
def surface_rows():
    return _load_csv(RESULTS_DIR / "E4PD_surface_reversal_summary.csv")


@pytest.fixture(scope="module")
def matched_rows():
    return _load_csv(RESULTS_DIR / "E4PD_matched_inferential_units.csv")


@pytest.fixture(scope="module")
def fe_rows():
    return _load_csv(RESULTS_DIR / "E4PD_fe_audit.csv")


@pytest.fixture(scope="module")
def predict_fb_rows():
    return _load_csv(RESULTS_DIR / "E4PD_predict_fallback_audit.csv")


@pytest.fixture(scope="module")
def scenario_summary():
    return _load_csv(RESULTS_DIR / "E4PD_scenario_summary.csv")


@pytest.fixture(scope="module")
def policy_contrasts():
    return _load_csv(RESULTS_DIR / "E4PD_policy_level_contrasts.csv")


@pytest.fixture(scope="module")
def source_arm():
    return _load(RESULTS_DIR / "E4PD_source_arm_contrasts.json")


# ===========================================================================
# OUTPUT EXISTENCE (18 tests)
# ===========================================================================

class TestOutputExistence:
    REQUIRED_FILES = [
        "E4PD_pre_execution_manifest.json",
        "E4PD_checkpoint_integrity.json",
        "E4PD_execution_config.json",
        "E4PD_verdict.json",
        "E4PD_p1_verdict.json",
        "E4PD_bootstrap.json",
        "E4PD_completeness_audit.json",
        "E4PD_global_incompatibility.json",
        "E4PD_post_integrity.json",
        "E4PD_no_learning_audit.json",
        "E4PD_no_performance_verdict.json",
        "E4PD_diagnostic_seed_usage.json",
        "E4PD_source_arm_contrasts.json",
        "E4PD_background_robustness.json",
        "E4PD_secondary_surface_summary.json",
        "E4PD_report.md",
        "E4PD_progress.jsonl",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_json_exists(self, fname):
        assert (RESULTS_DIR / fname).exists(), f"Missing: {fname}"

    REQUIRED_CSVS = [
        "E4PD_source_episode_manifest.csv",
        "E4PD_snapshot_manifest.csv",
        "E4PD_pair_selection.csv",
        "E4PD_random_tape_manifest.csv",
        "E4PD_primary_surface_cells.csv",
        "E4PD_secondary_surface_cells.csv",
        "E4PD_policy_background_cells.csv",
        "E4PD_fe_audit.csv",
        "E4PD_predict_fallback_audit.csv",
        "E4PD_surface_reversal_summary.csv",
        "E4PD_rank_reversal_opportunities.csv",
        "E4PD_best_action_context.csv",
        "E4PD_cross_difference_interactions.csv",
        "E4PD_tape_consistency.csv",
        "E4PD_source_specific_units.csv",
        "E4PD_matched_inferential_units.csv",
        "E4PD_scenario_summary.csv",
        "E4PD_policy_level_contrasts.csv",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_CSVS)
    def test_csv_exists(self, fname):
        assert (RESULTS_DIR / fname).exists(), f"Missing: {fname}"


# ===========================================================================
# PRE-EXECUTION INTEGRITY (6 tests)
# ===========================================================================

class TestPreExecutionIntegrity:
    def test_all_pass(self, p0):
        assert p0["ALL_INTEGRITY_PASS"] is True

    def test_e4pi_complete(self, p0):
        assert p0["checks"]["E4PI_COMPLETE"] is True

    def test_e4p_complete(self, p0):
        assert p0["checks"]["E4P_COMPLETE"] is True

    def test_config_sha(self, p0):
        assert p0["checks"]["CONFIG_SHA_MATCH"] is True

    def test_checkpoints_readable(self, p0):
        assert p0["checks"]["ALL_CHECKPOINTS_READABLE"] is True

    def test_seed_disjointness(self, p0):
        assert p0["checks"]["SEED_DISJOINTNESS"] is True


# ===========================================================================
# E4MF PRESERVATION (4 tests)
# ===========================================================================

class TestE4MFPreservation:
    def test_e4mf_failed(self, verdict):
        assert verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"

    def test_e4mf_not_changed(self, verdict):
        assert verdict["E4MF_VERDICT_CHANGED"] == "NO"

    def test_performance_not_computed(self, verdict):
        assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_no_performance_verdict_file(self):
        npv = _load(RESULTS_DIR / "E4PD_no_performance_verdict.json")
        assert npv["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"
        assert npv["E4MF_REMAINS_AUTHORITATIVE"] is True


# ===========================================================================
# SOURCE EPISODE COMPLETENESS (8 tests)
# ===========================================================================

class TestSourceEpisodeCompleteness:
    def test_total_episodes(self, completeness):
        assert completeness["SOURCE_DONE"] == 600

    def test_c1_episodes(self, completeness):
        assert completeness["C1"] == 300

    def test_c2_episodes(self, completeness):
        assert completeness["C2"] == 300

    def test_diagnostic_matrix_complete(self, verdict):
        assert verdict["DIAGNOSTIC_MATRIX_COMPLETE"] == "YES"

    def test_source_manifest_rows(self, source_rows):
        assert len(source_rows) == 600

    def test_all_arms_present(self, source_rows):
        arms = set(r["source_arm"] for r in source_rows)
        assert arms == {"C1", "C2"}

    def test_all_scenarios_present(self, source_rows):
        scns = set(r["scenario"] for r in source_rows)
        assert scns == set(SCENARIOS)

    def test_all_seeds_present(self, source_rows):
        seeds = set(int(r["environment_seed"]) for r in source_rows)
        assert seeds == set(DIAGNOSTIC_SEEDS)


# ===========================================================================
# SNAPSHOT ELIGIBILITY (4 tests)
# ===========================================================================

class TestSnapshotEligibility:
    def test_eligible_count(self, completeness):
        assert completeness["ELIGIBLE"] >= 100

    def test_eligible_plus_no_eligible_equals_600(self, completeness):
        assert completeness["ELIGIBLE"] + completeness["NO_ELIGIBLE"] == 600

    def test_snapshot_manifest_matches(self):
        snaps = _load_csv(RESULTS_DIR / "E4PD_snapshot_manifest.csv")
        eligible = [r for r in _load_csv(RESULTS_DIR / "E4PD_source_episode_manifest.csv")
                    if r["eligible"] == "True"]
        assert len(snaps) == len(eligible)

    def test_n_active_at_least_2(self):
        snaps = _load_csv(RESULTS_DIR / "E4PD_snapshot_manifest.csv")
        for s in snaps:
            assert int(s["n_active"]) >= 2


# ===========================================================================
# PAIR SELECTION (3 tests)
# ===========================================================================

class TestPairSelection:
    def test_sha256_method(self):
        pairs = _load_csv(RESULTS_DIR / "E4PD_pair_selection.csv")
        for p in pairs:
            assert p["method"] == "SHA256"

    def test_pair_count_matches_eligible(self, completeness):
        pairs = _load_csv(RESULTS_DIR / "E4PD_pair_selection.csv")
        assert len(pairs) == completeness["ELIGIBLE"]

    def test_region_i_less_than_j(self):
        snaps = _load_csv(RESULTS_DIR / "E4PD_snapshot_manifest.csv")
        for s in snaps:
            assert int(s["region_i"]) < int(s["region_j"])


# ===========================================================================
# FE AUDIT (3 tests)
# ===========================================================================

class TestFEAudit:
    def test_fe_audit_rows(self, fe_rows, completeness):
        n_eligible = completeness["ELIGIBLE"]
        assert len(fe_rows) == n_eligible * 4

    def test_all_fe_equal(self, fe_rows):
        for r in fe_rows:
            assert r["FE_EQUAL"] == "True", \
                f"FE not equal: {r['source_arm']}_{r['policy_seed']}_{r['scenario']}_{r['environment_seed']} tape {r['tape_id']}"

    def test_no_invalid_fe(self, completeness):
        assert completeness["INVALID_FE"] == 0


# ===========================================================================
# PREDICT FALLBACK AUDIT (2 tests)
# ===========================================================================

class TestPredictFallbackAudit:
    def test_no_predict_fallback(self, predict_fb_rows):
        assert len(predict_fb_rows) == 0

    def test_no_invalid_predict(self, completeness):
        assert completeness["INVALID_PREDICT"] == 0


# ===========================================================================
# PRIMARY SURFACE VALIDITY (4 tests)
# ===========================================================================

class TestPrimarySurfaceValidity:
    def test_all_valid(self, completeness):
        assert completeness["VALID_PRIMARY"] == completeness["ELIGIBLE"]

    def test_surface_rows_count(self, surface_rows, completeness):
        assert len(surface_rows) == completeness["ELIGIBLE"]

    def test_all_surface_valid_flag(self, surface_rows):
        for r in surface_rows:
            assert r["surface_valid"] == "True"

    def test_primary_cells_count(self, completeness):
        cells = _load_csv(RESULTS_DIR / "E4PD_primary_surface_cells.csv")
        n_eligible = completeness["ELIGIBLE"]
        assert len(cells) == n_eligible * 4 * 9  # 4 tapes × 9 cells


# ===========================================================================
# SECONDARY / POLICY BACKGROUND (3 tests)
# ===========================================================================

class TestSecondarySurface:
    def test_secondary_cells_count(self, completeness):
        cells = _load_csv(RESULTS_DIR / "E4PD_secondary_surface_cells.csv")
        n_eligible = completeness["ELIGIBLE"]
        assert len(cells) == n_eligible * 4 * 16

    def test_policy_bg_cells_count(self, completeness):
        cells = _load_csv(RESULTS_DIR / "E4PD_policy_background_cells.csv")
        n_eligible = completeness["ELIGIBLE"]
        assert len(cells) == n_eligible * 4 * 9

    def test_secondary_cannot_alter_p1(self):
        s = _load(RESULTS_DIR / "E4PD_secondary_surface_summary.json")
        assert s["cannot_alter_P1"] is True


# ===========================================================================
# REVERSAL RATE (5 tests)
# ===========================================================================

class TestReversalRate:
    def test_reversal_rate_bounded(self, surface_rows):
        for r in surface_rows:
            rr = float(r["reversal_rate"])
            assert 0.0 <= rr <= 1.0

    def test_reversal_opps_exist(self):
        opps = _load_csv(RESULTS_DIR / "E4PD_rank_reversal_opportunities.csv")
        assert len(opps) > 0

    def test_global_rr_bounded(self, verdict):
        assert 0.0 <= verdict["GLOBAL_RR"] <= 1.0

    def test_mono_incompat_bool(self, verdict):
        assert verdict["MONOTONIC_INCOMPATIBILITY"] in ("YES", "NO")

    def test_global_incompat_file(self, global_incompat):
        assert global_incompat["GLOBAL_VALID"] > 0


# ===========================================================================
# MATCHED INFERENTIAL UNITS (5 tests)
# ===========================================================================

class TestMatchedUnits:
    def test_matched_rows_count(self, matched_rows):
        assert len(matched_rows) == 300  # 5 policies × 20 seeds × 3 scenarios

    def test_all_available(self, matched_rows, completeness):
        avail = sum(1 for r in matched_rows if r["available"] == "True")
        assert avail == completeness["MATCHED"]

    def test_unit_rr_bounded(self, matched_rows):
        for r in matched_rows:
            if r["available"] == "True":
                rr = float(r["unit_rr"])
                assert 0.0 <= rr <= 1.0

    def test_all_scenarios_covered(self, matched_rows):
        scns = set(r["scenario"] for r in matched_rows)
        assert scns == set(SCENARIOS)

    def test_all_policies_covered(self, matched_rows):
        pols = set(int(r["policy_seed"]) for r in matched_rows)
        assert pols == set(POLICY_SEEDS)


# ===========================================================================
# SCENARIO MEANS AND CONTRASTS (6 tests)
# ===========================================================================

class TestScenarioContrasts:
    def test_scenario_summary_rows(self, scenario_summary):
        assert len(scenario_summary) == 3

    def test_g_rev_formula(self, verdict, scenario_summary):
        scn = {r["scenario"]: float(r["RR_mean"]) for r in scenario_summary}
        expected = scn["STATIONARY"] - 0.5 * (scn["PERSISTENT"] + scn["MIXED"])
        assert abs(verdict["G_REV"] - expected) < 1e-10

    def test_g_bac_exists(self, verdict):
        assert verdict["G_BAC"] is not None

    def test_g_interaction_secondary(self, verdict):
        assert "G_INTERACTION" in verdict

    def test_g_rev_ci_not_empty(self, verdict):
        ci = verdict["G_REV_CI"]
        assert ci[0] is not None
        assert ci[1] is not None
        assert ci[0] < ci[1]

    def test_bootstrap_n(self, bootstrap):
        detail = bootstrap.get("detail", {})
        assert detail.get("n_resamples") == 50000


# ===========================================================================
# POLICY-LEVEL CONTRASTS (3 tests)
# ===========================================================================

class TestPolicyContrasts:
    def test_policy_rows_count(self, policy_contrasts):
        assert len(policy_contrasts) == 5

    def test_all_policies_present(self, policy_contrasts):
        pols = set(int(r["policy_seed"]) for r in policy_contrasts)
        assert pols == set(POLICY_SEEDS)

    def test_g_rev_values_not_none(self, policy_contrasts):
        for r in policy_contrasts:
            assert r["G_REV"] != "" and r["G_REV"] is not None


# ===========================================================================
# SOURCE-ARM ROBUSTNESS (3 tests)
# ===========================================================================

class TestSourceArmRobustness:
    def test_both_arms(self, source_arm):
        contrasts = source_arm["source_arm_contrasts"]
        assert "C1" in contrasts
        assert "C2" in contrasts

    def test_g_rev_c1_exists(self, source_arm):
        g = source_arm["source_arm_contrasts"]["C1"]["G_REV"]
        assert g is not None

    def test_g_rev_c2_exists(self, source_arm):
        g = source_arm["source_arm_contrasts"]["C2"]["G_REV"]
        assert g is not None


# ===========================================================================
# P1 VERDICT LOGIC (8 tests)
# ===========================================================================

class TestP1Verdict:
    def test_p1_status_valid(self, p1_verdict):
        assert p1_verdict["P1_STATUS"] in (
            "SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED", "NOT_AUDITABLE")

    def test_p1_in_verdict(self, verdict):
        assert verdict["P1_STATUS"] == \
            _load(RESULTS_DIR / "E4PD_p1_verdict.json")["P1_STATUS"]

    def test_criteria_keys(self, p1_verdict):
        criteria = p1_verdict["criteria"]
        assert "C1_G_REV_positive" in criteria
        assert "C2_CI_lower_positive" in criteria
        assert "C3_policy_4_of_5" in criteria
        assert "C4_G_BAC_positive" in criteria
        assert "C5_policy_BAC_4_of_5" in criteria
        assert "C6_both_sources_positive" in criteria

    def test_supported_requires_all_6(self, p1_verdict):
        if p1_verdict["P1_STATUS"] == "SUPPORTED":
            assert all(p1_verdict["criteria"].values())

    def test_not_supported_g_rev_nonpositive(self, p1_verdict):
        if p1_verdict["P1_STATUS"] == "NOT_SUPPORTED":
            g = p1_verdict["G_REV"]
            assert g is None or g <= 0

    def test_g_rev_matches(self, p1_verdict, verdict):
        assert p1_verdict["G_REV"] == verdict["G_REV"]

    def test_policy_g_rev_count(self, p1_verdict):
        assert len(p1_verdict["POLICY_G_REV"]) == 5

    def test_source_arms_both_present(self, p1_verdict):
        assert "G_REV_C1" in p1_verdict
        assert "G_REV_C2" in p1_verdict


# ===========================================================================
# NO LEARNING AUDIT (3 tests)
# ===========================================================================

class TestNoLearningAudit:
    def test_no_learning_file(self):
        d = _load(RESULTS_DIR / "E4PD_no_learning_audit.json")
        assert d["optimizer_steps"] == 0
        assert d["gradient_updates"] == 0
        assert d["replay_appends"] == 0

    def test_no_target_updates(self):
        d = _load(RESULTS_DIR / "E4PD_no_learning_audit.json")
        assert d["target_updates"] == 0

    def test_no_checkpoint_writes(self):
        d = _load(RESULTS_DIR / "E4PD_no_learning_audit.json")
        assert d["checkpoint_writes"] == 0


# ===========================================================================
# CHECKPOINT INTEGRITY (3 tests)
# ===========================================================================

class TestCheckpointIntegrity:
    def test_all_unchanged(self, post_integrity):
        assert post_integrity["ALL_UNCHANGED"] is True

    def test_10_checkpoints(self, post_integrity):
        assert len(post_integrity["hashes"]) == 10

    def test_each_unchanged(self, post_integrity):
        for key, info in post_integrity["hashes"].items():
            assert info["unchanged"] is True, f"Changed: {key}"


# ===========================================================================
# DIAGNOSTIC SEED ACCOUNTING (3 tests)
# ===========================================================================

class TestSeedAccounting:
    def test_seeds_consumed(self):
        d = _load(RESULTS_DIR / "E4PD_diagnostic_seed_usage.json")
        assert d["status"] == "CONSUMED_NONMONOTONIC_INTERACTION_DIAGNOSTIC"

    def test_seeds_list(self):
        d = _load(RESULTS_DIR / "E4PD_diagnostic_seed_usage.json")
        assert d["seeds"] == DIAGNOSTIC_SEEDS

    def test_seeds_disjoint(self, p0):
        assert p0["checks"]["SEED_DISJOINTNESS"] is True


# ===========================================================================
# ROUTING (3 tests)
# ===========================================================================

class TestRouting:
    def test_next_stage_exists(self, verdict):
        assert "NEXT_STAGE" in verdict
        assert verdict["NEXT_STAGE"] != ""

    def test_routing_consistent(self, verdict):
        p1s = verdict["P1_STATUS"]
        nxt = verdict["NEXT_STAGE"]
        if p1s == "SUPPORTED":
            assert "IGM" in nxt or "DECENTRALIZED" in nxt
        elif p1s == "NOT_AUDITABLE":
            assert "INTEGRITY" in nxt
        elif verdict["MONOTONIC_INCOMPATIBILITY"] == "YES":
            assert "GENERAL" in nxt or "INTERACTION" in nxt
        else:
            assert "ACTION" in nxt or "STATE" in nxt

    def test_e4pd_complete(self, verdict):
        assert verdict["E4PD_COMPLETE"] == "YES"


# ===========================================================================
# CROSS-DIFFERENCE INTERACTIONS (3 tests)
# ===========================================================================

class TestCrossDifference:
    def test_interactions_exist(self):
        rows = _load_csv(RESULTS_DIR / "E4PD_cross_difference_interactions.csv")
        assert len(rows) > 0

    def test_abs_I_nonneg(self):
        rows = _load_csv(RESULTS_DIR / "E4PD_cross_difference_interactions.csv")
        for r in rows:
            assert float(r["abs_I"]) >= 0

    def test_i_norm_nonneg(self):
        rows = _load_csv(RESULTS_DIR / "E4PD_cross_difference_interactions.csv")
        for r in rows:
            assert float(r["I_norm"]) >= 0


# ===========================================================================
# TAPE CONSISTENCY (2 tests)
# ===========================================================================

class TestTapeConsistency:
    def test_tape_rows_count(self, completeness):
        rows = _load_csv(RESULTS_DIR / "E4PD_tape_consistency.csv")
        assert len(rows) == completeness["ELIGIBLE"]

    def test_tape_seeds_deterministic(self):
        tapes = _load_csv(RESULTS_DIR / "E4PD_random_tape_manifest.csv")
        assert len(tapes) > 0
        first = tapes[0]["tape_seeds"]
        assert first.startswith("[")


# ===========================================================================
# REPORT (2 tests)
# ===========================================================================

class TestReport:
    def test_report_exists(self):
        assert (RESULTS_DIR / "E4PD_report.md").exists()

    def test_report_contains_p1(self):
        text = (RESULTS_DIR / "E4PD_report.md").read_text()
        assert "P1" in text
        assert "NOT_COMPUTED" in text


# ===========================================================================
# VERDICT CONSISTENCY (5 tests)
# ===========================================================================

class TestVerdictConsistency:
    def test_g_rev_sign_matches_p1(self, verdict):
        if verdict["P1_STATUS"] == "NOT_SUPPORTED":
            assert verdict["G_REV"] <= 0

    def test_checkpoint_pass(self, verdict):
        assert verdict["CHECKPOINT_INTEGRITY"] == "PASS"

    def test_no_learning_pass(self, verdict):
        assert verdict["NO_LEARNING_AUDIT"] == "PASS"

    def test_stage_name(self, verdict):
        assert verdict["stage"] == "E4PD_NONMONOTONIC_INTERACTION_DIAGNOSTIC_EXECUTION"

    def test_e4mf_verdict_in_verdict(self, verdict):
        assert verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"


# ===========================================================================
# CONFIG SHA (2 tests)
# ===========================================================================

class TestConfigSHA:
    def test_config_sha_in_execution(self):
        cfg = _load(RESULTS_DIR / "E4PD_execution_config.json")
        assert cfg["config_sha256"] == EXPECTED_CONFIG_SHA

    def test_diagnostic_version(self):
        cfg = _load(RESULTS_DIR / "E4PD_execution_config.json")
        assert cfg["diagnostic_version"] == "E4PI_NONMONOTONIC_DIAGNOSTIC_V1"
