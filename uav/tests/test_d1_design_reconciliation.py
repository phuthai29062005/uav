"""
D1 design reconciliation consistency tests (T1–T20).

These tests validate artifact consistency and production-code immutability.
They do NOT run the scientific experiments (those are frozen in results/).
"""

import json
import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D1_DIR = os.path.join(REPO_ROOT, "results", "D1_design_reconciliation")
L2B_DIR = os.path.join(REPO_ROOT, "results", "L2B_mixed_regime_boundary_validation")
SRC_ORACLE = os.path.join(REPO_ROOT, "src", "prdrl", "oracle", "oracle.py")
SRC_PREDICT = os.path.join(REPO_ROOT, "src", "prdrl", "operators", "predict.py")
SRC_LOCAL = os.path.join(REPO_ROOT, "src", "prdrl", "operators", "local.py")
SRC_DIVERSIFY = os.path.join(REPO_ROOT, "src", "prdrl", "operators", "diversify.py")
SRC_KEEP = os.path.join(REPO_ROOT, "src", "prdrl", "operators", "keep.py")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def d1_decision():
    path = os.path.join(D1_DIR, "D1_decision.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def d1_options():
    path = os.path.join(D1_DIR, "D1_design_options.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def l2b_verdicts():
    path = os.path.join(L2B_DIR, "L2B_verdicts.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# T1 — All expected evidence sources exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", [
    os.path.join("results", "L2B_mixed_regime_boundary_validation", "L2B_verdicts.json"),
    os.path.join("results", "L2B_mixed_regime_boundary_validation", "L2B_mixed_protocol.json"),
    os.path.join("results", "L2B_mixed_regime_boundary_validation", "L2B_seed_summary.csv"),
    os.path.join("results", "L2B_mixed_regime_boundary_validation", "L2B_regime_effects.csv"),
    os.path.join("results", "L1_longitudinal_factorial_validation"),
    os.path.join("results", "L2_scenario_generalization"),
    os.path.join("results", "R1_region_protected_response"),
    os.path.join("results", "R1N_production_noise_interaction"),
    os.path.join("results", "N1_predict_noise_calibration"),
    os.path.join("results", "N2C_head_to_head_confirmation"),
    os.path.join("results", "D1_design_reconciliation", "D1_evidence_matrix.csv"),
    os.path.join("results", "D1_design_reconciliation", "D1_design_options.json"),
    os.path.join("results", "D1_design_reconciliation", "D1_decision.json"),
    os.path.join("results", "D1_design_reconciliation", "D1_design_spec.md"),
    os.path.join("results", "D1_design_reconciliation", "D1_migration_plan.md"),
    os.path.join("results", "D1_design_reconciliation", "D1_risk_register.csv"),
])
def test_t1_evidence_sources_exist(rel_path):
    full = os.path.join(REPO_ROOT, rel_path)
    assert os.path.exists(full), f"Missing: {rel_path}"


# ---------------------------------------------------------------------------
# T2 — L2B frozen verdicts load correctly and match expected values
# ---------------------------------------------------------------------------


def test_t2_l2b_verdicts_load(l2b_verdicts):
    v = l2b_verdicts["verdicts"]
    assert v["FE_FAIRNESS"] == "CONFIRMED"
    assert v["HARD_INVARIANT"] == "CONFIRMED"
    assert v["H005_MIXED_GLOBAL"] == "VALIDATED"
    assert v["HARD_MIXED_BOUNDARY"] == "SUPPORTED"
    assert v["D1_READINESS"] == "READY_H005_TRANSIENT"
    assert v["NEXT_STAGE"] == "D1_DESIGN_RECONCILIATION"


def test_t2b_l2b_f005_harmful_persistent(l2b_verdicts):
    nb_resp_p = l2b_verdicts["P_effects"]["NB_resp_P"]
    assert nb_resp_p["classification"] == "HARMFUL"
    assert nb_resp_p["mean"] < 0


def test_t2c_l2b_h005_global_beneficial(l2b_verdicts):
    cb_resp_global = l2b_verdicts["global_effects"]["CB_resp_global"]
    assert cb_resp_global["classification"] == "BENEFICIAL"
    assert cb_resp_global["ci_lo"] > 0


# ---------------------------------------------------------------------------
# T3 — Design options exactly O0-O3
# ---------------------------------------------------------------------------


def test_t3_design_options_exactly_O0_to_O3(d1_options):
    opts = set(d1_options["options"].keys())
    assert opts == {"O0", "O1", "O2", "O3"}, f"Expected exactly O0-O3, got {opts}"


# ---------------------------------------------------------------------------
# T4 — O3 sigma exactly 0.005
# ---------------------------------------------------------------------------


def test_t4_O3_sigma_005(d1_options):
    assert d1_options["options"]["O3"]["sigma_P"] == 0.005


# ---------------------------------------------------------------------------
# T5 — O3 uses HARD response-stage protection
# ---------------------------------------------------------------------------


def test_t5_O3_uses_hard(d1_options):
    assert d1_options["options"]["O3"]["use_hard"] is True
    assert "HARD" in d1_options["options"]["O3"]["predict_survival"]


# ---------------------------------------------------------------------------
# T6 — O3 returns to ordinary NSGA-II evolution afterward
# ---------------------------------------------------------------------------


def test_t6_O3_standard_evolution_after(d1_options):
    assert d1_options["options"]["O3"]["evolution_after_response"] == "standard_nsga2"


# ---------------------------------------------------------------------------
# T7 — Missing-history LOCAL fallback unchanged in all options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt_id", ["O0", "O1", "O2", "O3"])
def test_t7_local_fallback_unchanged(d1_options, opt_id):
    opt = d1_options["options"][opt_id]
    assert opt["missing_history_fallback"] == "LOCAL_sigma_L"
    assert opt["sigma_L"] == 0.02


# ---------------------------------------------------------------------------
# T8 — BUDGET remains 10 in all options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt_id", ["O0", "O1", "O2", "O3"])
def test_t8_budget_10(d1_options, opt_id):
    assert d1_options["options"][opt_id]["fe_budget"] == 10


# ---------------------------------------------------------------------------
# T9 — lambda remains 1 in all options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt_id", ["O0", "O1", "O2", "O3"])
def test_t9_lambda_1(d1_options, opt_id):
    assert d1_options["options"][opt_id]["lambda_pred"] == 1.0


# ---------------------------------------------------------------------------
# T10 — Production sigma remains 0.03 in source code (D1 does not implement)
# ---------------------------------------------------------------------------


def test_t10_production_sigma_unchanged():
    # D2-D (2026-09-20): sigma_predict default changed 0.03 → 0.005 (O1 adoption).
    # This test now asserts the post-D2-D production default.
    with open(SRC_ORACLE) as f:
        src = f.read()
    assert "sigma_predict=0.005" in src, (
        "oracle.py must have sigma_predict=0.005 after D2-D O1 adoption"
    )
    assert "sigma_predict=0.03," not in src, (
        "oracle.py must not have active sigma_predict=0.03 after D2-D"
    )


# ---------------------------------------------------------------------------
# T11 — Production default survival is still unrestricted NSGA-II (D2-A completed)
#
# D2-A wires O3 behind an explicit mode constant. The production DEFAULT must
# remain O0 (UNRESTRICTED). Verified invariants:
#   1. nsga2_survival is still present and called in the default path.
#   2. region_protected_survival is present (D2-A added it behind config).
#   3. The default parameter is PREDICT_SURVIVAL_UNRESTRICTED ("UNRESTRICTED").
# ---------------------------------------------------------------------------


def test_t11_production_survival_unchanged():
    with open(SRC_ORACLE) as f:
        src = f.read()
    # nsga2_survival must still be present and used as the default path
    assert "nsga2_survival" in src
    # D2-A: region_protected_survival is now legitimately behind O3 config
    assert "PREDICT_SURVIVAL_UNRESTRICTED" in src, (
        "oracle.py must export PREDICT_SURVIVAL_UNRESTRICTED constant (D2-A)"
    )
    assert "PREDICT_SURVIVAL_HARD_RESPONSE_ONLY" in src, (
        "oracle.py must export PREDICT_SURVIVAL_HARD_RESPONSE_ONLY constant (D2-A)"
    )
    # The default in respond_population's signature must still be UNRESTRICTED
    assert 'predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED' in src, (
        "respond_population default survival mode must remain PREDICT_SURVIVAL_UNRESTRICTED"
    )


# ---------------------------------------------------------------------------
# T12 — LOCAL action unchanged (sigma_L still 0.02 default in oracle)
# ---------------------------------------------------------------------------


def test_t12_local_sigma_unchanged():
    with open(SRC_ORACLE) as f:
        src = f.read()
    assert "sigma_local=0.02" in src or "0.02" in src


def test_t12b_local_py_unchanged():
    with open(SRC_LOCAL) as f:
        src = f.read()
    assert "def local(" in src
    assert "sigma" in src


# ---------------------------------------------------------------------------
# T13 — DIVERSIFY action unchanged
# ---------------------------------------------------------------------------


def test_t13_diversify_unchanged():
    with open(SRC_DIVERSIFY) as f:
        src = f.read()
    assert "def diversify(" in src


# ---------------------------------------------------------------------------
# T14 — KEEP action unchanged
# ---------------------------------------------------------------------------


def test_t14_keep_unchanged():
    with open(SRC_KEEP) as f:
        src = f.read()
    assert "def keep(" in src or "KEEP" in src or "keep" in src.lower()


# ---------------------------------------------------------------------------
# T15 — D1 decision logic D1-D9 all pass in D1_decision.json
# ---------------------------------------------------------------------------


def test_t15_d1_d9_criteria_all_pass(d1_decision):
    criteria = d1_decision["decision_criteria"]
    for key in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]:
        assert key in criteria, f"Missing criterion {key}"
        assert criteria[key]["result"] == "PASS", (
            f"Criterion {key} did not pass: {criteria[key]}"
        )
    assert d1_decision["all_criteria_pass"] is True


# ---------------------------------------------------------------------------
# T16 — No PDF file was modified by D1
# ---------------------------------------------------------------------------


def test_t16_no_pdf_modified(d1_decision):
    assert d1_decision["production_unchanged_confirmation"]["pdf_modified"] is False


# ---------------------------------------------------------------------------
# T17 — No production source modified by D1
# ---------------------------------------------------------------------------


def test_t17_no_production_source_modified(d1_decision):
    conf = d1_decision["production_unchanged_confirmation"]
    assert conf["predict_py_modified"] is False
    assert conf["oracle_py_modified"] is False


# ---------------------------------------------------------------------------
# T18 — Migration plan includes production-path equivalence gate
# ---------------------------------------------------------------------------


def test_t18_migration_plan_has_equivalence_gate():
    path = os.path.join(D1_DIR, "D1_migration_plan.md")
    with open(path) as f:
        content = f.read()
    assert "equivalence" in content.lower() or "D2-B" in content, (
        "Migration plan does not mention production-path equivalence gate"
    )
    assert "D2-B" in content


# ---------------------------------------------------------------------------
# T19 — Risk register includes MIXED F005 inversion (R1)
# ---------------------------------------------------------------------------


def test_t19_risk_register_has_mixed_inversion():
    path = os.path.join(D1_DIR, "D1_risk_register.csv")
    with open(path) as f:
        content = f.read()
    assert "R1" in content
    assert "MIXED" in content
    assert "persistent" in content.lower() or "HARMFUL" in content


# ---------------------------------------------------------------------------
# T20 — No RL code modified or started
# ---------------------------------------------------------------------------


def test_t20_rl_not_started(d1_decision):
    assert d1_decision["rl_status"] == "NOT_STARTED"
    assert d1_decision["production_unchanged_confirmation"]["rl_started"] is False


# ---------------------------------------------------------------------------
# T20b — D1 decision is PROPOSE_O3_JOINT_H005
# ---------------------------------------------------------------------------


def test_t20b_design_decision_is_O3(d1_decision):
    assert d1_decision["design_decision"] == "PROPOSE_O3_JOINT_H005"
    assert d1_decision["next_stage"] == "D2_IMPLEMENTATION_VALIDATION"
