"""
T1-T42 — R0-SM State Semantics Reconciliation tests.

No DRL. No performance experiment. No sigma tuning. No seed block.
RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
Production code NOT modified. E3 results NOT modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

OUT_DIR = Path("results/R0_state_semantics_reconciliation")
E3_DIR = Path("results/E3_rule_based_regional_controller")

# ──────────────────────────────────────────────────────────────
# T1-T5: Pre-run manifest invariants
# ──────────────────────────────────────────────────────────────

def test_T1_prechange_manifest_exists():
    assert (OUT_DIR / "SM_prechange_manifest.json").exists()


def test_T2_prechange_manifest_rl_status():
    data = json.loads((OUT_DIR / "SM_prechange_manifest.json").read_text())
    assert data["RL_STATUS"] == "NOT_STARTED"
    assert data["DRL_TRAINING"] == "NOT_STARTED"


def test_T3_prechange_manifest_oracle_hash():
    data = json.loads((OUT_DIR / "SM_prechange_manifest.json").read_text())
    assert data["oracle_sha256"] == "48ee63004f9a4d0d64af80fdc32cf3bdd6cff69c979b61991265bbb2a536453a"


def test_T4_prechange_manifest_no_production_change():
    data = json.loads((OUT_DIR / "SM_prechange_manifest.json").read_text())
    assert data["production_code_modified"] is False


def test_T5_prechange_manifest_no_e3_modified():
    data = json.loads((OUT_DIR / "SM_prechange_manifest.json").read_text())
    assert data["e3_results_modified"] is False


# ──────────────────────────────────────────────────────────────
# T6-T12: V1 feature lineage
# ──────────────────────────────────────────────────────────────

def test_T6_v1_lineage_exists():
    assert (OUT_DIR / "SM_v1_feature_lineage.json").exists()


def test_T7_v1_lineage_has_four_features():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    assert len(data["features"]) == 4


def test_T8_v1_lineage_c_r_is_centroid():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    c_r = next(f for f in data["features"] if f["symbol"] == "c_r")
    assert "centroid" in c_r["actual_semantics"].lower()
    assert c_r["dimension"] == "D (vector)"


def test_T9_v1_lineage_l_r_is_region_size():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    l_r = next(f for f in data["features"] if f["symbol"] == "l_r")
    assert "region size" in l_r["actual_semantics"].lower() or "number" in l_r["actual_semantics"].lower()
    assert l_r["dimension"] == "1 (scalar)"


def test_T10_v1_lineage_o_r_is_occupancy():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    o_r = next(f for f in data["features"] if f["symbol"] == "o_r")
    assert "occupancy" in o_r["actual_semantics"].lower() or "n_r" in o_r["actual_semantics"]
    assert o_r["status"] == "IMPLEMENTED_ASSEMBLED"


def test_T11_v1_lineage_total_dim_22_for_D10():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    assert data["total_dim_for_D10"] == 22


def test_T12_v1_lineage_delta_c_r_is_centroid_shift():
    data = json.loads((OUT_DIR / "SM_v1_feature_lineage.json").read_text())
    dc = next(f for f in data["features"] if f["symbol"] == "delta_c_r")
    assert "centroid" in dc["actual_semantics"].lower()
    assert dc["dimension"] == "D (vector)"


# ──────────────────────────────────────────────────────────────
# T13-T20: Manuscript → code mapping
# ──────────────────────────────────────────────────────────────

def test_T13_manuscript_mapping_exists():
    assert (OUT_DIR / "SM_manuscript_code_mapping.json").exists()


def test_T14_manuscript_c_r_maps_to_severity():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    c_r = next(f for f in data["features"] if f["symbol"] == "c_r")
    assert "severity" in c_r["code_file"]
    assert c_r["status"] == "IMPLEMENTED_NOT_ASSEMBLED"


def test_T15_manuscript_l_r_maps_to_quality_loss():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    l_r = next(f for f in data["features"] if f["symbol"] == "l_r")
    assert "quality_loss" in l_r["code_file"]
    assert l_r["status"] == "IMPLEMENTED_NOT_ASSEMBLED"


def test_T16_manuscript_o_r_matches_v1():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    o_r = next(f for f in data["features"] if f["symbol"] == "o_r")
    assert "MATCH" in o_r["status"]


def test_T17_manuscript_delta_c_r_not_implemented():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    dc = next(f for f in data["features"] if f["symbol"] == "delta_c_r")
    assert dc["status"] == "NOT_IMPLEMENTED_REQUIRES_SEVERITY_HISTORY"
    assert dc["code_file"] is None


def test_T18_manuscript_c_r_dim_is_scalar():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    c_r = next(f for f in data["features"] if f["symbol"] == "c_r")
    assert "scalar" in c_r["dimension"] or c_r["dimension"] == "1 (scalar)"


def test_T19_manuscript_severity_formula_in_semantics():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    c_r = next(f for f in data["features"] if f["symbol"] == "c_r")
    assert "exp" in c_r["manuscript_semantics"] or "C[r]" in c_r["manuscript_semantics"]


def test_T20_manuscript_four_features():
    data = json.loads((OUT_DIR / "SM_manuscript_code_mapping.json").read_text())
    assert len(data["features"]) == 4


# ──────────────────────────────────────────────────────────────
# T21-T27: Naming collision audit
# ──────────────────────────────────────────────────────────────

def test_T21_collision_audit_exists():
    assert (OUT_DIR / "SM_naming_collision_audit.json").exists()


def test_T22_collision_three_symbols():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    assert len(data["collisions"]) == 3


def test_T23_collision_c_r_critical():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    c_r = next(c for c in data["collisions"] if c["symbol"] == "c_r")
    assert c_r["severity"] == "CRITICAL"
    assert "DIM_MISMATCH" in c_r["collision_type"]


def test_T24_collision_l_r_critical():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    l_r = next(c for c in data["collisions"] if c["symbol"] == "l_r")
    assert l_r["severity"] == "CRITICAL"
    assert "OVERLOADED" in l_r["collision_type"]


def test_T25_collision_l_r_keep_unreachable_mentioned():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    l_r = next(c for c in data["collisions"] if c["symbol"] == "l_r")
    assert "unreachable" in l_r["impact"].lower() or "UNREACHABLE" in l_r["impact"]


def test_T26_collision_delta_c_r_critical():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    dc = next(c for c in data["collisions"] if c["symbol"] == "delta_c_r")
    assert dc["severity"] == "CRITICAL"


def test_T27_collision_all_symbols_covered():
    data = json.loads((OUT_DIR / "SM_naming_collision_audit.json").read_text())
    symbols = {c["symbol"] for c in data["collisions"]}
    assert symbols == {"c_r", "l_r", "delta_c_r"}


# ──────────────────────────────────────────────────────────────
# T28-T31: Causality audit
# ──────────────────────────────────────────────────────────────

def test_T28_causality_audit_exists():
    assert (OUT_DIR / "SM_causality_audit.json").exists()


def test_T29_causality_all_features_causal():
    data = json.loads((OUT_DIR / "SM_causality_audit.json").read_text())
    assert data["verdict"] == "ALL V2 FEATURES ARE CAUSAL"


def test_T30_causality_delta_severity_no_response_outcome():
    data = json.loads((OUT_DIR / "SM_causality_audit.json").read_text())
    ds = next(f for f in data["features"] if "delta" in f["symbol_v2"])
    assert ds.get("requires_response_outcome") is False
    assert ds.get("requires_severity_history") is True


def test_T31_causality_four_v2_features():
    data = json.loads((OUT_DIR / "SM_causality_audit.json").read_text())
    assert len(data["features"]) == 4


# ──────────────────────────────────────────────────────────────
# T32-T36: Numerical validation
# ──────────────────────────────────────────────────────────────

def test_T32_numerical_validation_exists():
    assert (OUT_DIR / "SM_numerical_validation.json").exists()


def test_T33_numerical_validation_all_passed():
    data = json.loads((OUT_DIR / "SM_numerical_validation.json").read_text())
    assert data["all_passed"] is True
    assert data["verdict"] == "VALIDATED"


def test_T34_numerical_severity_formula_match():
    data = json.loads((OUT_DIR / "SM_numerical_validation.json").read_text())
    sev = next(t for t in data["tests"] if t["test"] == "severity_formula")
    assert sev["passed"] is True
    assert sev["S_match"] is True
    assert sev["C_match"] is True
    assert sev["Q_match"] is True


def test_T35_numerical_quality_loss_formula_match():
    data = json.loads((OUT_DIR / "SM_numerical_validation.json").read_text())
    ql = next(t for t in data["tests"] if t["test"] == "quality_loss_formula")
    assert ql["passed"] is True
    assert ql["match"] is True


def test_T36_numerical_severity_c_r_in_0_1():
    data = json.loads((OUT_DIR / "SM_numerical_validation.json").read_text())
    sev = next(t for t in data["tests"] if t["test"] == "severity_formula")
    for c in sev["C_actual"]:
        assert 0.0 <= c <= 1.0, f"C[r]={c} out of [0,1]"


# ──────────────────────────────────────────────────────────────
# T37-T38: Temporal validation
# ──────────────────────────────────────────────────────────────

def test_T37_temporal_validation_exists():
    assert (OUT_DIR / "SM_temporal_validation.json").exists()


def test_T38_temporal_keep_reachability():
    data = json.loads((OUT_DIR / "SM_temporal_validation.json").read_text())
    kr = data["keep_reachability"]
    assert kr["v1_keep_reachable"] is False
    assert kr["v2_keep_reachable"] is True


# ──────────────────────────────────────────────────────────────
# T39-T40: V2 state candidate
# ──────────────────────────────────────────────────────────────

def test_T39_v2_candidate_exists():
    assert (OUT_DIR / "SM_state_candidate_v2.json").exists()


def test_T40_v2_candidate_dim_4():
    data = json.loads((OUT_DIR / "SM_state_candidate_v2.json").read_text())
    assert data["v2_feature_vector"]["total_dim"] == 4


# ──────────────────────────────────────────────────────────────
# T41: E3 admissibility
# ──────────────────────────────────────────────────────────────

def test_T41_e3_addendum_not_admissible():
    data = json.loads((OUT_DIR / "E3_semantics_addendum.json").read_text())
    assert data["e3_admissibility"] == "NOT_ADMISSIBLE_FOR_DRL_GATE"
    assert data["mismatch_confirmed"] is True


# ──────────────────────────────────────────────────────────────
# T42: Final verdict
# ──────────────────────────────────────────────────────────────

def test_T42_sm_verdict_complete():
    data = json.loads((OUT_DIR / "SM_verdict.json").read_text())
    assert data["SM_STATUS"] == "COMPLETE"
    assert data["V1_V2_MISMATCH_CONFIRMED"] is True
    assert data["E3_ADMISSIBLE_FOR_DRL_GATE"] is False
    assert data["E3R_REQUIRED"] is True
    assert data["E3R_READINESS"] == "READY"
    assert data["RL_STATUS"] == "NOT_STARTED"
    assert data["DRL_TRAINING"] == "NOT_STARTED"
    assert data["production_code_modified"] is False
    assert data["e3_results_modified"] is False
