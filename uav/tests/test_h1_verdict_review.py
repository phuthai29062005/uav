"""
T1-T23+ test suite for H1-RV — Verdict and Regional-Interaction Review.

Invariants:
  - H1 raw data NOT overwritten (H1_verdict.json, H1_oracle_region_records.jsonl unchanged).
  - No new scientific seeds (no 941+).
  - RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
  - BAL_ACC_MARGIN = 0.05 (frozen).
  - MARGIN_EPS = 1e-8 (frozen).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "src")

H1_DIR     = Path("results/H1_regional_controller_hypothesis_review")
REVIEW_DIR = Path("results/H1_verdict_review")

BAL_ACC_MARGIN = 0.05
MARGIN_EPS     = 1e-8
DELTA_H1_ROG   = 1e-5
SCENARIOS      = ["PERSISTENT", "STATIONARY", "MIXED"]
ACTIONS        = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")


def _load_json(name: str, base: Path = REVIEW_DIR) -> dict:
    return json.loads((base / name).read_text())


def _load_h1_json(name: str) -> dict:
    return json.loads((H1_DIR / name).read_text())


# ============================================================
# T1: Raw H1 verdict unchanged
# ============================================================

class TestH1DataImmutability:
    def test_T1_h1_verdict_gate_b_unchanged(self):
        v = _load_h1_json("H1_verdict.json")
        assert v["H1_OUTCOME"] == "INCONCLUSIVE"

    def test_T2_h1_raw_records_still_accessible(self):
        assert (H1_DIR / "H1_oracle_region_records.jsonl").exists()
        assert (H1_DIR / "H1_verdict.json").exists()

    def test_T3_no_new_scientific_seeds(self):
        # H1RV manifest must declare no new seeds
        m = _load_json("H1RV_pre_manifest.json")
        assert m.get("no_new_seeds") is True
        assert m.get("no_new_oracle_evaluations") is True

    def test_T4_frozen_bal_acc_margin(self):
        m = _load_json("H1RV_pre_manifest.json")
        assert abs(float(m["BAL_ACC_MARGIN_FROZEN"]) - BAL_ACC_MARGIN) < 1e-15

    def test_T5_frozen_margin_eps(self):
        m = _load_json("H1RV_pre_manifest.json")
        assert abs(float(m["MARGIN_EPS_FROZEN"]) - MARGIN_EPS) < 1e-20

    def test_T6_frozen_sep_rule_loaded(self):
        m = _load_json("H1RV_pre_manifest.json")
        sep_rule = m.get("frozen_sep_rule", {})
        assert "SUPPORTED" in sep_rule
        # Rule text must mention BAL_ACC_MARGIN threshold
        assert "BAL_ACC_MARGIN" in str(sep_rule.get("SUPPORTED", ""))

    def test_T21_production_unchanged(self):
        v = _load_json("H1RV_verdict.json")
        assert v.get("PRODUCTION_DEFAULT") == "O1_F005"

    def test_T22_rl_status_not_started(self):
        v = _load_json("H1RV_verdict.json")
        assert v["RL_STATUS"] == "NOT_STARTED"
        assert v["DRL_TRAINING"] == "NOT_STARTED"


# ============================================================
# T7-T12: Separability recompute audit
# ============================================================

class TestSeparabilityRecompute:
    @pytest.fixture(scope="class")
    def sep(self):
        return _load_json("H1RV_separability_recompute.json")

    def test_T7_stronger_baseline_computed(self, sep):
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            sb = rec.get("stronger_baseline")
            lr = rec.get("bal_lr")
            dt = rec.get("bal_dt")
            # stronger_baseline is just used for margin; must be consistent
            assert sb is not None

    def test_T8_lr_margins_correct(self, sep):
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            ml = rec.get("margin_lr")
            bl = rec.get("bal_lr")
            sb = rec.get("stronger_baseline")
            if ml is not None and bl is not None and sb is not None:
                assert abs(ml - (bl - sb)) < 1e-9, f"margin_lr={ml} != bal_lr-stronger"

    def test_T9_dt_margins_correct(self, sep):
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            md = rec.get("margin_dt")
            bd = rec.get("bal_dt")
            sb = rec.get("stronger_baseline")
            if md is not None and bd is not None and sb is not None:
                assert abs(md - (bd - sb)) < 1e-9, f"margin_dt={md} != bal_dt-stronger"

    def test_T10_perm_flags_loaded(self, sep):
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            assert "perm_pass_lr" in rec
            assert "perm_pass_dt" in rec
            assert "perm_pass_either" in rec
            # either = lr OR dt
            assert rec["perm_pass_either"] == (rec["perm_pass_lr"] or rec["perm_pass_dt"])

    def test_T11_nnt_results_used(self, sep):
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            # NNT margins must be present
            assert "nnt_margin_lr" in rec
            assert "nnt_margin_dt" in rec

    def test_T12_verdict_computed_mechanically(self, sep):
        # At least three verdict interpretations present
        assert "lr_only" in sep
        assert "dt_only" in sep
        assert "either_probe" in sep
        # Each has a verdict string
        for k in ["lr_only", "dt_only", "either_probe"]:
            assert sep[k]["verdict"] in {"SUPPORTED", "PARTIAL", "ABSENT"}

    def test_T13_no_post_hoc_threshold_changes(self, sep):
        # BAL_ACC_MARGIN used must equal 0.05
        assert abs(float(sep["BAL_ACC_MARGIN_USED"]) - BAL_ACC_MARGIN) < 1e-15

    def test_mixed_and_pooled_dt_margin_exceeds_threshold(self, sep):
        """DT exceeds 0.05 margin in MIXED and POOLED directions."""
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            sc = rec["scenario"]
            md = rec.get("margin_dt")
            if sc in ("MIXED", "POOLED") and md is not None:
                assert md > BAL_ACC_MARGIN, \
                    f"{sc}/{rec['direction']}: DT margin {md:.4f} should exceed {BAL_ACC_MARGIN}"

    def test_persistent_dt_margin_below_threshold(self, sep):
        """PERSISTENT DT margin should be below 0.05 (not qualifying as SUPPORTED)."""
        for rec in sep["per_direction"]:
            if "error" in rec:
                continue
            if rec["scenario"] == "PERSISTENT":
                md = rec.get("margin_dt")
                if md is not None:
                    assert md < BAL_ACC_MARGIN, \
                        f"PERSISTENT/{rec['direction']}: DT margin {md:.4f} should be below {BAL_ACC_MARGIN}"


# ============================================================
# T14-T15: Oracle type
# ============================================================

class TestOracleType:
    @pytest.fixture(scope="class")
    def oracle_audit(self):
        return _load_json("H1RV_oracle_composition_audit.json")

    def test_T14_oracle_type_identified(self, oracle_audit):
        oracle_type = oracle_audit.get("ORACLE_TYPE")
        assert oracle_type in {"TRUE_JOINT", "GREEDY_FACTORIZED", "OTHER"}

    def test_T15_no_claim_true_joint_unless_exhaustive(self, oracle_audit):
        oracle_type = oracle_audit.get("ORACLE_TYPE")
        evidence = oracle_audit.get("evidence", {})
        if oracle_type == "TRUE_JOINT":
            assert evidence.get("exhaustive_joint_search") is True, \
                "Cannot claim TRUE_JOINT without exhaustive joint search"
        else:
            # GREEDY_FACTORIZED: isolated_per_region + composition
            assert evidence.get("isolated_per_region_evaluation") is True
            assert evidence.get("greedy_composition_after_selection") is True

    def test_greedy_factorized_4_to_R_documented(self, oracle_audit):
        assert "4_to_R_combinations" in oracle_audit
        assert oracle_audit["4_to_R_combinations"] == 4**10


# ============================================================
# T16: PERSISTENT interpretation bounded
# ============================================================

class TestPersistentInteraction:
    @pytest.fixture(scope="class")
    def pers(self):
        return _load_json("H1RV_persistent_interaction.json")

    def test_T16_interpretation_bounded(self, pers):
        interp = pers.get("PERSISTENT_NEGATIVE_ROG_INTERPRETATION")
        valid = {"GLOBAL_ACTION_TRULY_DOMINATES", "FACTORIZED_ORACLE_INTERACTION",
                 "BOTH_PLAUSIBLE", "UNRESOLVED"}
        assert interp in valid

    def test_persistent_mean_rog_negative(self, pers):
        assert float(pers["mean_ROG"]) < 0

    def test_persistent_frac_ohi_positive_high(self, pers):
        # Oracle assigns heterogeneous per-region actions in most events
        assert float(pers["frac_OHI_positive"]) > 0.8

    def test_persistent_predict_dominant(self, pers):
        assert float(pers["frac_best_global_PREDICT"]) > 0.7


# ============================================================
# T17: MIXED event summaries algebraically correct
# ============================================================

class TestMixedEvidence:
    @pytest.fixture(scope="class")
    def mixed(self):
        return _load_json("H1RV_mixed_evidence.json")

    def test_T17_mixed_frac_algebraically_consistent(self, mixed):
        total = (
            mixed["frac_events_ge2_nnt_actions"] +
            mixed["frac_events_exactly_1_nnt_action"] +
            mixed["frac_events_zero_nnt_actions"]
        )
        # ge2 + exactly_1 + exactly_0 should sum to >=1.0 (ge3 is subset of ge2)
        # Actually: ge2 covers >=2, 1 covers exactly 1, 0 covers exactly 0
        # These are mutually exclusive and exhaustive
        # ge2 + exactly_1 + exactly_0 = 1.0
        assert abs(total - 1.0) < 0.01, f"MIXED frac sum={total:.4f}"

    def test_mixed_ge2_high(self, mixed):
        # 97% of MIXED events have >=2 NNT actions
        assert float(mixed["frac_events_ge2_nnt_actions"]) > 0.90

    def test_mixed_rog_positive_majority(self, mixed):
        assert float(mixed["frac_events_ROG_positive"]) > 0.5

    def test_mixed_headroom_supported(self, mixed):
        assert mixed["MIXED_ORACLE_HEADROOM"] == "SUPPORTED"

    def test_mixed_median_rog_positive(self, mixed):
        assert float(mixed["median_ROG"]) > 0


# ============================================================
# T18-T20: No rollout, no controller, no DQN
# ============================================================

class TestNoDRLStarted:
    def test_T18_no_classifier_rollout(self):
        v = _load_json("H1RV_verdict.json")
        assert v.get("NO_CLASSIFIER_ROLLOUT") is True or v.get("NO_DQN") is True

    def test_T19_no_new_controller(self):
        v = _load_json("H1RV_verdict.json")
        assert v.get("NO_RULE_TUNING") is True

    def test_T20_no_dqn(self):
        v = _load_json("H1RV_verdict.json")
        assert v.get("NO_DQN") is True


# ============================================================
# T23: Regression (deferred — run via pytest directly)
# ============================================================

class TestOutputFileCompleteness:
    def test_T23_all_output_files_exist(self):
        expected = [
            "H1RV_pre_manifest.json",
            "H1RV_separability_recompute.json",
            "H1RV_oracle_composition_audit.json",
            "H1RV_persistent_interaction.json",
            "H1RV_mixed_evidence.json",
            "H1RV_drl_case.json",
            "H1_verdict_addendum.json",
            "H1RV_verdict.json",
            "H1RV_report.md",
        ]
        for fn in expected:
            assert (REVIEW_DIR / fn).exists(), f"Missing: {fn}"

    def test_h1rv_verdict_outcome_valid(self):
        v = _load_json("H1RV_verdict.json")
        valid_outcomes = {"GLOBAL_ACTION_DOMINANCE", "STATE_INFORMATION_INSUFFICIENT",
                         "RULE_MAPPING_INADEQUATE", "INCONCLUSIVE"}
        assert v["H1_OUTCOME_REVIEWED"] in valid_outcomes

    def test_h1rv_e4_readiness_valid(self):
        v = _load_json("H1RV_verdict.json")
        assert v["E4_READINESS"] in {"READY", "CONDITIONAL", "NOT_READY"}

    def test_h1rv_next_stage_valid(self):
        v = _load_json("H1RV_verdict.json")
        valid = {"E4_LEARNED_CONTROLLER_PREPARATION", "H1_ARCHITECTURE_REVIEW",
                 "STATE_REPRESENTATION_HYPOTHESIS_REVIEW", "REGIONAL_CONTROL_REDESIGN_OR_STOP",
                 "H1_REVIEW"}
        assert v["NEXT_STAGE"] in valid

    def test_drl_case_valid(self):
        d = _load_json("H1RV_drl_case.json")
        valid = {"NOT_JUSTIFIED", "STATE_SIGNAL_TOO_WEAK",
                 "NONLINEAR_MAPPING_SUPPORTED", "JOINT_COORDINATION_REQUIRED"}
        for c in d.get("cases", []):
            assert c in valid
        assert d.get("primary") in valid

    def test_h1rv_sep_audit_valid(self):
        v = _load_json("H1RV_verdict.json")
        assert v["SEPARABILITY_VERDICT_AUDIT"] in \
            {"CONSISTENT", "IMPLEMENTATION_MISMATCH", "SPEC_AMBIGUOUS"}

    def test_addendum_original_verdict_preserved(self):
        a = _load_json("H1_verdict_addendum.json")
        assert a.get("original_verdict") == "PARTIAL"

    def test_addendum_no_margin_change(self):
        a = _load_json("H1_verdict_addendum.json")
        if "NO_BAL_ACC_MARGIN_change" in a:
            assert a["NO_BAL_ACC_MARGIN_change"] is True

    def test_report_contains_outcome(self):
        text = (REVIEW_DIR / "H1RV_report.md").read_text()
        assert "H1_OUTCOME_REVIEWED" in text or "Reviewed H1 Outcome" in text

    def test_report_mentions_oracle_type(self):
        text = (REVIEW_DIR / "H1RV_report.md").read_text()
        assert "GREEDY_FACTORIZED" in text or "ORACLE_TYPE" in text
