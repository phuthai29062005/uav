"""Tests for E4ODR — E4OD Report and Derived-Artifact Reconciliation.

53+ tests covering: output existence, firewall, G_cos reconciliation,
G_conflict reconciliation, rho reconciliation, hypothesis label audit,
scientific wording, O1 recheck, bootstrap reproduction, hypothesis
status, routing, raw-hash integrity, correction manifest, report,
verdict structure, regression.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
E4ODR_DIR = BASE / "results" / "E4ODR_e4od_report_artifact_reconciliation"
E4OD_DIR = BASE / "results" / "E4OD_scenario_tradeoff_diagnostic_execution"
E4N_DIR = BASE / "results" / "E4N_credit_preservation_failure_analysis"
E4O_DIR = BASE / "results" / "E4O_scenario_tradeoff_diagnostic_design"


def _load(p):
    with open(p) as f:
        return json.load(f)


# ── helpers ──────────────────────────────────────────────────────────

E4N_AUTHORITATIVE = {
    "N1": "N1_SCENARIO_SPECIFIC_GRADIENT_CONFLICT",
    "N2": "N2_REGIONAL_LOSS_SCALE_INEFFECTIVENESS",
    "N3": "N3_ACTION_SPECIALIZATION_SHIFT",
    "N4": "N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT",
    "N5": "N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS",
    "N6": "N6_CHECKPOINT_SELECTION_SCENARIO_TENSION",
    "N7": "N7_POLICY_SEED_HETEROGENEITY",
    "N8": "N8_JOINT_VS_REGIONAL_SUPERVISION_TRADEOFF",
    "N9": "N9_MONOTONIC_QMIX_LIMITATION_REMAINS",
    "N10": "N10_MULTIFACTORIAL_SCENARIO_TRADEOFF",
}

REQUIRED_FILES = [
    "E4ODR_pre_manifest.json",
    "E4ODR_authoritative_source_map.json",
    "E4ODR_raw_artifact_hashes.json",
    "E4ODR_g_cos_reconciliation.json",
    "E4ODR_g_conflict_reconciliation.json",
    "E4ODR_rho_reconciliation.json",
    "E4ODR_hypothesis_label_audit.csv",
    "E4ODR_scientific_wording_audit.json",
    "E4ODR_recomputed_scenario_statistics.json",
    "E4ODR_recomputed_policy_gaps.json",
    "E4ODR_bootstrap_reproduction.json",
    "E4ODR_o1_criterion_recheck.json",
    "E4ODR_hypothesis_status_audit.json",
    "E4ODR_routing_audit.json",
    "E4ODR_correction_manifest.json",
    "E4ODR_post_hash_integrity.json",
    "E4ODR_verdict.json",
    "E4ODR_report.md",
]


# ══════════════════════════════════════════════════════════════════════
# GROUP 1: OUTPUT EXISTENCE
# ══════════════════════════════════════════════════════════════════════

class TestOutputExistence:
    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_output_exists(self, fname):
        assert (E4ODR_DIR / fname).exists(), f"Missing {fname}"

    def test_output_count(self):
        actual = set(p.name for p in E4ODR_DIR.iterdir() if p.is_file())
        expected = set(REQUIRED_FILES)
        assert expected.issubset(actual)


# ══════════════════════════════════════════════════════════════════════
# GROUP 2: FIREWALL
# ══════════════════════════════════════════════════════════════════════

class TestFirewall:
    def test_no_new_episodes(self):
        pre = _load(E4ODR_DIR / "E4ODR_pre_manifest.json")
        assert pre["new_environment_episodes"] == 0

    def test_no_seeds_reopened(self):
        pre = _load(E4ODR_DIR / "E4ODR_pre_manifest.json")
        assert pre["diagnostic_seeds_reopened"] == "NO"

    def test_no_training(self):
        pre = _load(E4ODR_DIR / "E4ODR_pre_manifest.json")
        assert pre["new_training"] == "NO"

    def test_no_checkpoints_changed(self):
        pre = _load(E4ODR_DIR / "E4ODR_pre_manifest.json")
        assert pre["checkpoints_changed"] == "NO"

    def test_reconciliation_type(self):
        pre = _load(E4ODR_DIR / "E4ODR_pre_manifest.json")
        assert pre["reconciliation_type"] == "REPORT_AND_DERIVED_ARTIFACT_ONLY"


# ══════════════════════════════════════════════════════════════════════
# GROUP 3: RAW HASH INTEGRITY
# ══════════════════════════════════════════════════════════════════════

class TestRawHashIntegrity:
    def test_pre_post_match(self):
        post = _load(E4ODR_DIR / "E4ODR_post_hash_integrity.json")
        assert post["all_unchanged"]

    def test_each_file_match(self):
        pre = _load(E4ODR_DIR / "E4ODR_raw_artifact_hashes.json")
        post = _load(E4ODR_DIR / "E4ODR_post_hash_integrity.json")
        for fname in pre["hashes"]:
            assert pre["hashes"][fname] == post["hashes"][fname], \
                f"{fname} hash changed"

    def test_no_file_not_found(self):
        pre = _load(E4ODR_DIR / "E4ODR_raw_artifact_hashes.json")
        for fname, h in pre["hashes"].items():
            assert h != "FILE_NOT_FOUND", f"Expected {fname} to exist"


# ══════════════════════════════════════════════════════════════════════
# GROUP 4: G_COS RECONCILIATION
# ══════════════════════════════════════════════════════════════════════

class TestGCosReconciliation:
    def test_both_positive(self):
        r = _load(E4ODR_DIR / "E4ODR_g_cos_reconciliation.json")
        assert r["BOTH_POSITIVE"]

    def test_discrepancy_explained(self):
        r = _load(E4ODR_DIR / "E4ODR_g_cos_reconciliation.json")
        assert r["DISCREPANCY_EXPLAINED"]

    def test_sample_vs_bootstrap_different(self):
        r = _load(E4ODR_DIR / "E4ODR_g_cos_reconciliation.json")
        assert r["G_COS_SAMPLE_POINT"] != r["G_COS_BOOTSTRAP_POINT"]

    def test_o1_uses_bootstrap(self):
        r = _load(E4ODR_DIR / "E4ODR_g_cos_reconciliation.json")
        assert r["O1_CRITERION_1_USES"] == "G_COS_BOOTSTRAP_POINT"
        assert r["O1_CRITERION_2_USES"] == "G_COS_CI_UPPER (from same bootstrap)"

    def test_bootstrap_n_and_seed(self):
        r = _load(E4ODR_DIR / "E4ODR_g_cos_reconciliation.json")
        assert r["BOOTSTRAP_N"] == 50000
        assert r["BOOTSTRAP_SEED"] == 116001


# ══════════════════════════════════════════════════════════════════════
# GROUP 5: G_CONFLICT RECONCILIATION
# ══════════════════════════════════════════════════════════════════════

class TestGConflictReconciliation:
    def test_both_negative(self):
        r = _load(E4ODR_DIR / "E4ODR_g_conflict_reconciliation.json")
        assert r["BOTH_NEGATIVE"]

    def test_discrepancy_explained(self):
        r = _load(E4ODR_DIR / "E4ODR_g_conflict_reconciliation.json")
        assert r["DISCREPANCY_EXPLAINED"]

    def test_sample_vs_bootstrap_different(self):
        r = _load(E4ODR_DIR / "E4ODR_g_conflict_reconciliation.json")
        assert r["G_CONFLICT_SAMPLE_POINT"] != r["G_CONFLICT_BOOTSTRAP_POINT"]

    def test_o1_uses_sample(self):
        r = _load(E4ODR_DIR / "E4ODR_g_conflict_reconciliation.json")
        assert r["O1_CRITERION_4_USES"] == "G_CONFLICT_SAMPLE_POINT"


# ══════════════════════════════════════════════════════════════════════
# GROUP 6: RHO RECONCILIATION
# ══════════════════════════════════════════════════════════════════════

class TestRhoReconciliation:
    def test_n8_verdict_stable(self):
        r = _load(E4ODR_DIR / "E4ODR_rho_reconciliation.json")
        assert r["N8_VERDICT_STABLE"]
        assert r["N8_VERDICT"] == "NOT_SUPPORTED"

    def test_stat_less_than_per_both_methods(self):
        r = _load(E4ODR_DIR / "E4ODR_rho_reconciliation.json")
        assert r["N8_CANONICAL_COMPARISON"]["stat_less_than_per_BOTH_METHODS"]

    def test_rho_176_is_mean_of_ratios(self):
        r = _load(E4ODR_DIR / "E4ODR_rho_reconciliation.json")
        assert r["RHO_1_76_TRACE"]["category"] == "MEAN_OF_RATIOS_JENSEN_INFLATION"
        assert not r["RHO_1_76_TRACE"]["is_typo"]

    def test_per_scenario_count(self):
        r = _load(E4ODR_DIR / "E4ODR_rho_reconciliation.json")
        assert len(r["per_scenario"]) == 3


# ══════════════════════════════════════════════════════════════════════
# GROUP 7: HYPOTHESIS LABEL AUDIT
# ══════════════════════════════════════════════════════════════════════

class TestHypothesisLabelAudit:
    def test_csv_exists_and_parseable(self):
        with open(E4ODR_DIR / "E4ODR_hypothesis_label_audit.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_all_corrected_names_match_e4n(self):
        with open(E4ODR_DIR / "E4ODR_hypothesis_label_audit.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            hid = row["ID"]
            assert row["corrected_name"] == E4N_AUTHORITATIVE[hid]

    def test_n7_mismatch_flagged(self):
        with open(E4ODR_DIR / "E4ODR_hypothesis_label_audit.csv") as f:
            rows = list(csv.DictReader(f))
        n7 = [r for r in rows if r["ID"] == "N7"][0]
        assert n7["match"] == "NO"
        assert n7["E4OD_displayed_name"] == "N7_GLOBAL_DOMINANCE"
        assert n7["corrected_name"] == "N7_POLICY_SEED_HETEROGENEITY"

    def test_n9_mismatch_flagged(self):
        with open(E4ODR_DIR / "E4ODR_hypothesis_label_audit.csv") as f:
            rows = list(csv.DictReader(f))
        n9 = [r for r in rows if r["ID"] == "N9"][0]
        assert n9["match"] == "NO"
        assert n9["E4OD_displayed_name"] == "N9_NONMONOTONIC_INTERACTION"
        assert n9["corrected_name"] == "N9_MONOTONIC_QMIX_LIMITATION_REMAINS"


# ══════════════════════════════════════════════════════════════════════
# GROUP 8: CORRECTED E4OD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════

class TestCorrectedE4ODArtifacts:
    def test_hypothesis_reassessment_uses_e4n_names(self):
        h = _load(E4OD_DIR / "E4OD_hypothesis_reassessment.json")
        for hid in ["N3", "N4", "N5", "N6", "N7", "N8", "N9"]:
            assert E4N_AUTHORITATIVE[hid] in h, \
                f"{E4N_AUTHORITATIVE[hid]} not a key"

    def test_n7_corrected_to_not_supported(self):
        h = _load(E4OD_DIR / "E4OD_hypothesis_reassessment.json")
        assert h["N7_POLICY_SEED_HETEROGENEITY"]["verdict"] == "NOT_SUPPORTED"

    def test_verdict_hypothesis_reassessment_n7(self):
        v = _load(E4OD_DIR / "E4OD_verdict.json")
        assert v["HYPOTHESIS_REASSESSMENT"]["N7"] == "NOT_SUPPORTED"

    def test_verdict_o1_unchanged(self):
        v = _load(E4OD_DIR / "E4OD_verdict.json")
        assert v["O1_SUPPORT"] == "NOT_SUPPORTED"

    def test_verdict_next_stage_unchanged(self):
        v = _load(E4OD_DIR / "E4OD_verdict.json")
        assert v["NEXT_STAGE"] == "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"


# ══════════════════════════════════════════════════════════════════════
# GROUP 9: SCIENTIFIC WORDING AUDIT
# ══════════════════════════════════════════════════════════════════════

class TestScientificWordingAudit:
    def test_r5_overclaim_identified(self):
        w = _load(E4ODR_DIR / "E4ODR_scientific_wording_audit.json")
        r5 = w["R5_OVERCLAIM_PRESENCE_OF_REGIONAL_LOSS"]
        assert r5["claim_status"] == "NONE_ESTABLISHED"

    def test_r6_bounded_wording(self):
        w = _load(E4ODR_DIR / "E4ODR_scientific_wording_audit.json")
        r6 = w["R6_O1_INTERPRETATION"]
        assert r6["preserve_strong_negative"]
        assert "not supported" in r6["canonical_wording"]

    def test_r7_no_identity_symbol(self):
        w = _load(E4ODR_DIR / "E4ODR_scientific_wording_audit.json")
        r7 = w["R7_STATIONARY_R1_EXACTNESS"]
        assert not r7["use_identity_symbol"]
        assert r7["empirically_all_zero"]

    def test_overall_pass(self):
        w = _load(E4ODR_DIR / "E4ODR_scientific_wording_audit.json")
        assert w["SCIENTIFIC_WORDING_AUDIT"] == "PASS"


# ══════════════════════════════════════════════════════════════════════
# GROUP 10: O1 CRITERION RECHECK
# ══════════════════════════════════════════════════════════════════════

class TestO1CriterionRecheck:
    def test_o1_unchanged(self):
        o = _load(E4ODR_DIR / "E4ODR_o1_criterion_recheck.json")
        assert not o["O1_VERDICT_CHANGED"]
        assert o["O1_STATUS_BEFORE"] == o["O1_STATUS_AFTER_RECONCILIATION"]
        assert o["O1_STATUS_AFTER_RECONCILIATION"] == "NOT_SUPPORTED"

    def test_all_five_criteria_fail(self):
        o = _load(E4ODR_DIR / "E4ODR_o1_criterion_recheck.json")
        criteria = o["criteria"]
        assert not criteria["C1_G_cos_negative"]["pass"]
        assert not criteria["C2_CI_upper_negative"]["pass"]
        assert not criteria["C3_policy_consistency_cos"]["pass"]
        assert not criteria["C4_G_conflict_positive"]["pass"]
        assert not criteria["C5_policy_consistency_conflict"]["pass"]


# ══════════════════════════════════════════════════════════════════════
# GROUP 11: RECOMPUTED STATISTICS
# ══════════════════════════════════════════════════════════════════════

class TestRecomputedStatistics:
    def test_g_cos_match(self):
        r = _load(E4ODR_DIR / "E4ODR_recomputed_scenario_statistics.json")
        assert r["G_COS_match"]

    def test_g_conflict_match(self):
        r = _load(E4ODR_DIR / "E4ODR_recomputed_scenario_statistics.json")
        assert r["G_CONFLICT_match"]

    def test_policy_g_cos_match(self):
        r = _load(E4ODR_DIR / "E4ODR_recomputed_policy_gaps.json")
        assert r["policy_G_cos_match"]

    def test_policy_g_conflict_match(self):
        r = _load(E4ODR_DIR / "E4ODR_recomputed_policy_gaps.json")
        assert r["policy_G_conflict_match"]


# ══════════════════════════════════════════════════════════════════════
# GROUP 12: BOOTSTRAP REPRODUCTION
# ══════════════════════════════════════════════════════════════════════

class TestBootstrapReproduction:
    def test_g_cos_point_matches_per_scenario(self):
        b = _load(E4ODR_DIR / "E4ODR_bootstrap_reproduction.json")
        assert b["G_COS_BOOTSTRAP"]["match"]

    def test_ci_brackets_point(self):
        b = _load(E4ODR_DIR / "E4ODR_bootstrap_reproduction.json")
        assert b["G_COS_BOOTSTRAP"]["ci_brackets_point"]

    def test_n_matches_contract(self):
        b = _load(E4ODR_DIR / "E4ODR_bootstrap_reproduction.json")
        assert b["G_COS_BOOTSTRAP"]["N_matches"]

    def test_seed_matches_contract(self):
        b = _load(E4ODR_DIR / "E4ODR_bootstrap_reproduction.json")
        assert b["G_COS_BOOTSTRAP"]["seed_matches"]

    def test_status_verified(self):
        b = _load(E4ODR_DIR / "E4ODR_bootstrap_reproduction.json")
        assert b["REPRODUCTION_STATUS"] == "VERIFIED_INTERNAL_CONSISTENCY"


# ══════════════════════════════════════════════════════════════════════
# GROUP 13: ROUTING AUDIT
# ══════════════════════════════════════════════════════════════════════

class TestRoutingAudit:
    def test_routing_correct(self):
        r = _load(E4ODR_DIR / "E4ODR_routing_audit.json")
        assert r["routing_correct"]

    def test_next_stage(self):
        r = _load(E4ODR_DIR / "E4ODR_routing_audit.json")
        assert r["NEXT_STAGE"] == "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"


# ══════════════════════════════════════════════════════════════════════
# GROUP 14: CORRECTION MANIFEST
# ══════════════════════════════════════════════════════════════════════

class TestCorrectionManifest:
    def test_corrections_exist(self):
        c = _load(E4ODR_DIR / "E4ODR_correction_manifest.json")
        assert c["total_corrections"] >= 3

    def test_no_verdict_impact(self):
        c = _load(E4ODR_DIR / "E4ODR_correction_manifest.json")
        for corr in c["corrections"]:
            assert corr["verdict_impact"].startswith("NONE"), \
                f"Expected NONE but got {corr['verdict_impact']}"


# ══════════════════════════════════════════════════════════════════════
# GROUP 15: VERDICT STRUCTURE
# ══════════════════════════════════════════════════════════════════════

class TestVerdictStructure:
    def test_e4odr_complete(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["E4ODR_COMPLETE"] == "YES"

    def test_e4mf_unchanged(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["ORIGINAL_E4MF_VERDICT"] == "FAILED"
        assert v["E4MF_VERDICT_CHANGED"] == "NO"

    def test_performance_not_computed(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_raw_unchanged(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["RAW_SCIENTIFIC_ARTIFACTS_CHANGED"] == "NO"

    def test_reconciliation_status(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["RECONCILIATION_STATUS"] == "DOCUMENTED_NON_VERDICT_CHANGE"

    def test_o1_not_changed(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert not v["O1_VERDICT_CHANGED"]

    def test_next_stage(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["NEXT_STAGE"] == "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"

    def test_hypothesis_label_audit_fail(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["HYPOTHESIS_LABEL_AUDIT"] == "FAIL"

    def test_raw_hash_pass(self):
        v = _load(E4ODR_DIR / "E4ODR_verdict.json")
        assert v["RAW_HASH_INTEGRITY"] == "PASS"


# ══════════════════════════════════════════════════════════════════════
# GROUP 16: HISTORICAL VERDICTS PRESERVED
# ══════════════════════════════════════════════════════════════════════

class TestHistoricalVerdicts:
    @pytest.mark.parametrize("stage,verdict_file,key,expected", [
        ("E4HF", "E4HF_normalization_ab_final_evaluation_execution", "NORMALIZATION_CAUSAL_VERDICT", "FAILED"),
        ("E4IF", "E4IF_reward_ab_final_evaluation_execution", "REWARD_CAUSAL_VERDICT", "VALIDATED"),
        ("E4KF", "E4KF_joint_coordination_ab_final_evaluation_execution", "COORDINATION_CAUSAL_VERDICT", "FAILED"),
        ("E4MF", "E4MF_credit_preservation_ab_final_evaluation_execution", "CREDIT_PRESERVATION_CAUSAL_VERDICT", "FAILED"),
    ])
    def test_historical_verdict(self, stage, verdict_file, key, expected):
        p = BASE / "results" / verdict_file / f"{stage}_verdict.json"
        if p.exists():
            v = _load(p)
            assert v[key] == expected


# ══════════════════════════════════════════════════════════════════════
# GROUP 17: REPORT
# ══════════════════════════════════════════════════════════════════════

class TestReport:
    def test_e4odr_report_exists(self):
        assert (E4ODR_DIR / "E4ODR_report.md").exists()

    def test_corrected_e4od_report_exists(self):
        assert (E4OD_DIR / "E4OD_report.md").exists()

    def test_corrected_report_no_overclaim(self):
        text = (E4OD_DIR / "E4OD_report.md").read_text()
        assert "PRESENCE of regional objective interacts" not in text
        assert "proved impossible" not in text

    def test_corrected_report_has_labeled_estimators(self):
        text = (E4OD_DIR / "E4OD_report.md").read_text()
        assert "G_cos SAMPLE POINT" in text or "G_COS_SAMPLE_POINT" in text
        assert "G_cos BOOTSTRAP POINT" in text or "G_COS_BOOTSTRAP_POINT" in text


# ══════════════════════════════════════════════════════════════════════
# GROUP 18: REGRESSION
# ══════════════════════════════════════════════════════════════════════

class TestRegression:
    def test_full_regression(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
             "--tb=short",
             "--ignore=tests/test_e4odr_e4od_report_artifact_reconciliation.py",
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
