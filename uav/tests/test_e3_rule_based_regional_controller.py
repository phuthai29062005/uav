"""
E3 — Rule-Based Regional Controller: Test Suite (T1–T88).

Run after: python src/prdrl/experiments/E3_rule_based_regional_controller.py

All 88 tests must PASS. No skip, no xfail.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, "src")

RESULTS = Path("results/E3_rule_based_regional_controller")

# ============================================================
# Helpers
# ============================================================

def _read_json(name: str) -> dict:
    p = RESULTS / name
    assert p.exists(), f"Missing: {p}"
    with open(p) as f:
        return json.load(f)


def _read_jsonl(name: str) -> List[dict]:
    p = RESULTS / name
    assert p.exists(), f"Missing: {p}"
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(name: str) -> List[dict]:
    p = RESULTS / name
    assert p.exists(), f"Missing: {p}"
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


CALIB_SEEDS   = list(range(701, 721))
EVAL_SEEDS    = list(range(721, 781))
BLOCK_A_SEEDS = list(range(721, 751))
BLOCK_B_SEEDS = list(range(751, 781))
SCENARIOS     = ["PERSISTENT", "STATIONARY", "MIXED"]
ALL_CONTROLLERS = ["RULE_V1", "GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]
D = 10
R = 10  # reference vectors for M=2, N_PARTITIONS=9

# ============================================================
# T1-T10: Pre-run manifest and invariant checks
# ============================================================

def test_t1_pre_run_manifest_exists():
    """T1: E3_pre_run_manifest.json exists and lists oracle.py."""
    d = _read_json("E3_pre_run_manifest.json")
    assert "files" in d
    assert "src/prdrl/oracle/oracle.py" in d["files"]


def test_t2_rl_status_not_started():
    """T2: RL_STATUS = NOT_STARTED in all output files."""
    verdict = _read_json("E3_verdict.json")
    assert verdict["RL_STATUS"] == "NOT_STARTED"
    manifest = _read_json("E3_pre_run_manifest.json")
    assert manifest["RL_STATUS"] == "NOT_STARTED"
    rule_v1 = _read_json("E3_rule_v1.json")
    assert rule_v1["RL_STATUS"] == "NOT_STARTED"


def test_t3_no_hard_in_controller():
    """T3: rule_based.py does not import or call HARD production functions."""
    p = Path("src/prdrl/controllers/rule_based.py")
    src = p.read_text()
    # Check for actual code constructs, not documentation mentions
    assert "region_protected_survival" not in src
    assert "import region_protected" not in src
    assert "ACTION_HARD" not in src
    assert '"HARD"' not in src and "'HARD'" not in src


def test_t4_sigma_values_match_o1_production():
    """T4: E3 uses O1 production sigma values."""
    from prdrl.experiments.E3_rule_based_regional_controller import (
        SIGMA_LOCAL, SIGMA_PREDICT, SIGMA_DIVERSIFY, LAMBDA_PREDICT
    )
    assert SIGMA_LOCAL    == pytest.approx(0.02)
    assert SIGMA_PREDICT  == pytest.approx(0.005)
    assert SIGMA_DIVERSIFY == pytest.approx(0.15)
    assert LAMBDA_PREDICT == pytest.approx(1.0)


def test_t5_e3_interface_readiness():
    """T5: E3_INTERFACE_READINESS = READY from check_e3_readiness()."""
    from prdrl.rl.interface import check_e3_readiness
    result = check_e3_readiness()
    assert result["E3_INTERFACE_READINESS"] == "READY"


def test_t6_config_constants():
    """T6: D=10, R=10, POP_SIZE=100 as locked in E3."""
    from prdrl.experiments.E3_rule_based_regional_controller import (
        D as E3_D, POP_SIZE, N_OBJ, N_PARTITIONS, BUDGET
    )
    from prdrl.core.reference import reference
    V = reference(N_OBJ, N_PARTITIONS)
    assert E3_D == 10
    assert POP_SIZE == 100
    assert len(V) == 10   # R=10
    assert BUDGET == 10


def test_t7_seed_sets_disjoint():
    """T7: Calibration, evaluation, and smoke seed sets are pairwise disjoint."""
    smoke = {781, 782}
    calib = set(CALIB_SEEDS)
    evalu = set(EVAL_SEEDS)
    assert len(calib & evalu) == 0
    assert len(calib & smoke) == 0
    assert len(evalu & smoke) == 0


def test_t8_eval_seeds_count():
    """T8: Evaluation has 60 seeds, Block A=30, Block B=30."""
    assert len(EVAL_SEEDS) == 60
    assert len(BLOCK_A_SEEDS) == 30
    assert len(BLOCK_B_SEEDS) == 30
    assert set(BLOCK_A_SEEDS) | set(BLOCK_B_SEEDS) == set(EVAL_SEEDS)
    assert set(BLOCK_A_SEEDS) & set(BLOCK_B_SEEDS) == set()


def test_t9_rule_v1_priority_order():
    """T9: RULE_V1 priority order is KEEP > DIVERSIFY > PREDICT > LOCAL."""
    rule_v1 = _read_json("E3_rule_v1.json")
    logic = rule_v1["rule_logic"]
    assert "1_KEEP" in logic
    assert "2_DIVERSIFY" in logic
    assert "3_PREDICT" in logic
    assert "4_LOCAL" in logic


def test_t10_t_l_equals_zero():
    """T10: T_L = 0 (fixed, not calibrated)."""
    rule_v1 = _read_json("E3_rule_v1.json")
    assert rule_v1["thresholds"]["T_L"] == pytest.approx(0.0)
    manifest = _read_json("E3_rule_v1.json")
    assert manifest["thresholds"]["T_L"] == pytest.approx(0.0)


# ============================================================
# T11-T20: Controller module correctness
# ============================================================

def test_t11_controller_registry():
    """T11: ALL_CONTROLLERS contains exactly 5 expected controllers."""
    from prdrl.controllers.rule_based import ALL_CONTROLLERS, GLOBAL_CONTROLLERS
    assert set(ALL_CONTROLLERS) == {
        "RULE_V1", "GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"
    }
    assert "RULE_V1" not in GLOBAL_CONTROLLERS
    assert len(GLOBAL_CONTROLLERS) == 4


def test_t12_global_keep_always_keep():
    """T12: GLOBAL_KEEP assigns KEEP to all active regions."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    contexts = {
        r: RegionalDecisionContext(
            t=2, region_id=r, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
        for r in range(R)
    }
    actions = choose_actions("GLOBAL_KEEP", contexts, D)
    assert all(v == "KEEP" for v in actions.values())


def test_t13_global_local_always_local():
    """T13: GLOBAL_LOCAL assigns LOCAL to all active regions."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    contexts = {
        r: RegionalDecisionContext(
            t=2, region_id=r, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
        for r in range(R)
    }
    actions = choose_actions("GLOBAL_LOCAL", contexts, D)
    assert all(v == "LOCAL" for v in actions.values())


def test_t14_global_predict_always_predict():
    """T14: GLOBAL_PREDICT assigns PREDICT to all active regions."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    contexts = {
        r: RegionalDecisionContext(
            t=2, region_id=r, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
        for r in range(R)
    }
    actions = choose_actions("GLOBAL_PREDICT", contexts, D)
    assert all(v == "PREDICT" for v in actions.values())


def test_t15_global_diversify_always_diversify():
    """T15: GLOBAL_DIVERSIFY assigns DIVERSIFY to all active regions."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    contexts = {
        r: RegionalDecisionContext(
            t=2, region_id=r, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
        for r in range(R)
    }
    actions = choose_actions("GLOBAL_DIVERSIFY", contexts, D)
    assert all(v == "DIVERSIFY" for v in actions.values())


def test_t16_inactive_regions_excluded():
    """T16: Inactive regions receive no action from any controller."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    contexts = {
        0: RegionalDecisionContext(
            t=2, region_id=0, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        ),
        1: RegionalDecisionContext(
            t=2, region_id=1, active=False,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=False, delta_c_r_available=False,
            n_solutions=0, n_total=100, current_budget=10,
        ),
    }
    for ctrl in ["RULE_V1", "GLOBAL_KEEP", "GLOBAL_LOCAL"]:
        actions = choose_actions(ctrl, contexts, D, rule_config={"D": D, "T_C_LOW": 1.0,
            "T_C_HIGH": 2.0, "T_O_LOW": 0.05, "T_DC_HIGH": 0.5, "T_L": 0.0})
        assert 0 in actions
        assert 1 not in actions, f"Controller {ctrl} assigned action to inactive region"


def test_t17_rule_v1_diversify_fires_high_c_high_dc():
    """T17: RULE_V1 selects DIVERSIFY when c is high and dc is high."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    # c = norm of c_r: set c_r to large value
    fv[:D] = 3.0 / np.sqrt(D)  # norm = 3.0 > T_C_HIGH
    fv[D] = 10.0                 # l = 10 > T_L = 0
    fv[D + 1] = 0.001            # o = very low < T_O_LOW
    fv[D + 2:] = 0.1 / np.sqrt(D)  # dc small (doesn't matter since o < T_O_LOW)
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=10, n_total=100, current_budget=10,
    )
    rule_config = {"D": D, "T_C_LOW": 0.5, "T_C_HIGH": 2.0, "T_O_LOW": 0.01, "T_DC_HIGH": 1.0, "T_L": 0.0}
    action = choose_action_rule_v1(ctx, rule_config)
    assert action == "DIVERSIFY", f"Expected DIVERSIFY, got {action}"


def test_t18_rule_v1_predict_fires_high_c_no_diversify_trigger():
    """T18: RULE_V1 selects PREDICT when c is high but o and dc not extreme."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    fv[:D] = 3.0 / np.sqrt(D)   # c = 3.0 > T_C_HIGH = 2.0
    fv[D] = 10.0                  # l > T_L
    fv[D + 1] = 0.5               # o = 0.5 > T_O_LOW = 0.01 — no diversify from o
    fv[D + 2:] = 0.01 / np.sqrt(D)  # dc very small < T_DC_HIGH = 1.0 — no diversify from dc
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=10, n_total=100, current_budget=10,
    )
    rule_config = {"D": D, "T_C_LOW": 0.5, "T_C_HIGH": 2.0, "T_O_LOW": 0.01, "T_DC_HIGH": 1.0, "T_L": 0.0}
    action = choose_action_rule_v1(ctx, rule_config)
    assert action == "PREDICT", f"Expected PREDICT, got {action}"


def test_t19_rule_v1_local_default():
    """T19: RULE_V1 defaults to LOCAL when c is in mid range."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    # c = 1.0 — between T_C_LOW=0.5 and T_C_HIGH=2.0
    fv[:D] = 1.0 / np.sqrt(D)
    fv[D] = 10.0   # l > T_L
    fv[D + 1] = 0.5  # o normal
    fv[D + 2:] = 0.1 / np.sqrt(D)  # dc small
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=10, n_total=100, current_budget=10,
    )
    rule_config = {"D": D, "T_C_LOW": 0.5, "T_C_HIGH": 2.0, "T_O_LOW": 0.01, "T_DC_HIGH": 1.0, "T_L": 0.0}
    action = choose_action_rule_v1(ctx, rule_config)
    assert action == "LOCAL", f"Expected LOCAL, got {action}"


def test_t20_rule_v1_feature_vector_read_only():
    """T20: Feature vector is read-only (R0 invariant enforced by interface)."""
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=5, n_total=100, current_budget=10,
    )
    with pytest.raises((ValueError, TypeError)):
        ctx.feature_vector[0] = 99.0


# ============================================================
# T21-T30: Calibration and threshold outputs
# ============================================================

def test_t21_feature_calibration_file_exists():
    """T21: E3_feature_calibration.json exists."""
    d = _read_json("E3_feature_calibration.json")
    assert "n_samples" in d
    assert d["n_samples"] > 0


def test_t22_calibration_n_samples_reasonable():
    """T22: Feature calibration has at least 100 samples."""
    d = _read_json("E3_feature_calibration.json")
    assert d["n_samples"] >= 100


def test_t23_thresholds_order_consistent():
    """T23: T_C_LOW < T_C_HIGH and T_O_LOW > 0 and T_DC_HIGH > 0."""
    d = _read_json("E3_rule_v1.json")
    t = d["thresholds"]
    assert t["T_C_LOW"] < t["T_C_HIGH"], "T_C_LOW must be < T_C_HIGH"
    assert t["T_O_LOW"] >= 0.0
    assert t["T_DC_HIGH"] >= 0.0


def test_t24_thresholds_source_feature_only():
    """T24: Thresholds are derived from feature percentiles, NOT performance."""
    d = _read_json("E3_rule_v1.json")
    assert d["tuning_method"] == "feature_percentile_only"
    assert d["not_tuned_from"] == "IGD_performance"


def test_t25_rule_v1_json_frozen():
    """T25: E3_rule_v1.json has frozen_date field."""
    d = _read_json("E3_rule_v1.json")
    assert "frozen_date" in d
    assert d["frozen_date"] == "2026-09-20"


def test_t26_global_baseline_selection_exists():
    """T26: E3_global_baseline_selection.json exists with all 3 scenarios."""
    d = _read_json("E3_global_baseline_selection.json")
    for sc in SCENARIOS:
        assert sc in d, f"Missing scenario {sc}"
        assert "best_global_controller" in d[sc]


def test_t27_global_baseline_is_valid_controller():
    """T27: best_global_controller for each scenario is one of 4 global controllers."""
    d = _read_json("E3_global_baseline_selection.json")
    valid = {"GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"}
    for sc in SCENARIOS:
        assert d[sc]["best_global_controller"] in valid


def test_t28_calibration_percentiles_sensible():
    """T28: T_C_LOW = 33rd pctile and T_C_HIGH = 67th pctile from feature calibration."""
    calib = _read_json("E3_feature_calibration.json")
    rule_v1 = _read_json("E3_rule_v1.json")
    p33 = calib["c_percentiles"]["p33"]
    p67 = calib["c_percentiles"]["p67"]
    assert abs(rule_v1["thresholds"]["T_C_LOW"]  - p33) < 1e-10
    assert abs(rule_v1["thresholds"]["T_C_HIGH"] - p67) < 1e-10


def test_t29_t_o_low_is_25th_percentile():
    """T29: T_O_LOW = 25th percentile of o from calibration."""
    calib = _read_json("E3_feature_calibration.json")
    rule_v1 = _read_json("E3_rule_v1.json")
    p25 = calib["o_percentiles"]["p25"]
    assert abs(rule_v1["thresholds"]["T_O_LOW"] - p25) < 1e-10


def test_t30_t_dc_high_is_75th_percentile():
    """T30: T_DC_HIGH = 75th percentile of dc from calibration."""
    calib = _read_json("E3_feature_calibration.json")
    rule_v1 = _read_json("E3_rule_v1.json")
    p75 = calib["dc_percentiles"]["p75"]
    assert abs(rule_v1["thresholds"]["T_DC_HIGH"] - p75) < 1e-10


# ============================================================
# T31-T40: RULE_V1 logic correctness edge cases
# ============================================================

def test_t31_rule_v1_keep_unreachable_for_active():
    """T31: RULE_V1 KEEP is unreachable for active regions (l > T_L always)."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    rule_v1 = _read_json("E3_rule_v1.json")
    t = rule_v1["thresholds"]
    fv = np.zeros(2 * D + 2)
    # Set c very low (below T_C_LOW), dc=0, BUT l > T_L (active region)
    fv[:D] = 0.0   # c = 0 < T_C_LOW
    fv[D] = 1.0    # l = 1 > T_L = 0 — active region
    fv[D + 1] = 0.5
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=1, n_total=100, current_budget=10,
    )
    rc = {"D": D, "T_C_LOW": t["T_C_LOW"], "T_C_HIGH": t["T_C_HIGH"],
          "T_O_LOW": t["T_O_LOW"], "T_DC_HIGH": t["T_DC_HIGH"], "T_L": t["T_L"]}
    action = choose_action_rule_v1(ctx, rc)
    # Since l=1 > T_L=0, KEEP rule condition fails, so falls through
    assert action != "KEEP"


def test_t32_rule_v1_deterministic():
    """T32: RULE_V1 is deterministic — same context always gives same action."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    rng = np.random.default_rng(555)
    fv = rng.uniform(0, 0.5, size=(2 * D + 2,))
    fv[D] = 5.0
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=5, n_total=100, current_budget=10,
    )
    rc = {"D": D, "T_C_LOW": 0.5, "T_C_HIGH": 1.5, "T_O_LOW": 0.05, "T_DC_HIGH": 0.3, "T_L": 0.0}
    action_1 = choose_action_rule_v1(ctx, rc)
    action_2 = choose_action_rule_v1(ctx, rc)
    assert action_1 == action_2


def test_t33_rule_v1_no_rng():
    """T33: rule_based.py contains no RNG instantiation or random calls."""
    p = Path("src/prdrl/controllers/rule_based.py")
    src = p.read_text()
    assert "default_rng" not in src
    assert "np.random.seed" not in src
    assert "random.random" not in src
    assert "rng.uniform" not in src


def test_t34_choose_actions_returns_dict():
    """T34: choose_actions returns Dict[int, str] for valid controller."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    ctx = {
        5: RegionalDecisionContext(
            t=2, region_id=5, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
    }
    for ctrl in ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]:
        actions = choose_actions(ctrl, ctx, D)
        assert isinstance(actions, dict)
        assert 5 in actions
        assert isinstance(actions[5], str)


def test_t35_invalid_controller_raises():
    """T35: Unknown controller name raises ValueError."""
    from prdrl.controllers.rule_based import choose_actions
    with pytest.raises(ValueError, match="Unknown controller"):
        choose_actions("FAKE_CTRL", {}, D)


def test_t36_rule_v1_diversify_from_high_dc():
    """T36: RULE_V1 selects DIVERSIFY when dc >= T_DC_HIGH (even if o > T_O_LOW)."""
    from prdrl.controllers.rule_based import choose_action_rule_v1
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    fv[:D] = 3.0 / np.sqrt(D)   # c = 3 > T_C_HIGH = 2
    fv[D] = 10.0                  # l > T_L
    fv[D + 1] = 0.5               # o large (> T_O_LOW) — no o-based diversify
    fv[D + 2:] = 2.0 / np.sqrt(D)  # dc = 2.0 > T_DC_HIGH = 1.0 → diversify from dc
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=10, n_total=100, current_budget=10,
    )
    rc = {"D": D, "T_C_LOW": 0.5, "T_C_HIGH": 2.0, "T_O_LOW": 0.01, "T_DC_HIGH": 1.0, "T_L": 0.0}
    action = choose_action_rule_v1(ctx, rc)
    assert action == "DIVERSIFY"


def test_t37_get_rule_scalars_correct():
    """T37: get_rule_scalars extracts correct scalars from feature vector."""
    from prdrl.controllers.rule_based import get_rule_scalars
    from prdrl.rl.interface import RegionalDecisionContext
    fv = np.zeros(2 * D + 2)
    # Set c_r to unit vector → c_norm = 1
    fv[:D] = 1.0 / np.sqrt(D)
    fv[D] = 7.0      # l
    fv[D + 1] = 0.3  # o
    # delta_c_r = [2/sqrt(D), ...]
    fv[D + 2:] = 2.0 / np.sqrt(D)
    ctx = RegionalDecisionContext(
        t=2, region_id=0, active=True,
        feature_vector=fv,
        feature_names=[f"f{i}" for i in range(2 * D + 2)],
        c_r_available=True, delta_c_r_available=True,
        n_solutions=7, n_total=100, current_budget=10,
    )
    s = get_rule_scalars(ctx, D)
    assert abs(s["c"]  - 1.0) < 1e-10
    assert abs(s["l"]  - 7.0) < 1e-10
    assert abs(s["o"]  - 0.3) < 1e-10
    assert abs(s["dc"] - 2.0) < 1e-10


def test_t38_rule_v1_config_requires_d():
    """T38: rule_config must contain 'D' key for RULE_V1."""
    from prdrl.controllers.rule_based import choose_actions
    from prdrl.rl.interface import RegionalDecisionContext
    ctx = {
        0: RegionalDecisionContext(
            t=2, region_id=0, active=True,
            feature_vector=np.zeros(2 * D + 2),
            feature_names=[f"f{i}" for i in range(2 * D + 2)],
            c_r_available=True, delta_c_r_available=True,
            n_solutions=5, n_total=100, current_budget=10,
        )
    }
    with pytest.raises((ValueError, KeyError)):
        choose_actions("RULE_V1", ctx, D, rule_config=None)


def test_t39_no_b_r_a_in_controller():
    """T39: No B_r(a) action-dependent budget implementation in controller."""
    p = Path("src/prdrl/controllers/rule_based.py")
    src = p.read_text()
    # Check for actual code, not documentation mentions
    assert "action_dependent_budget" not in src
    assert "budget_per_action" not in src
    # KEEP_RESPONSE_FE and CURRENT_RESPONSE_BUDGET_VALUE are fine to import
    # but no custom action-dependent budget logic
    assert "def b_r" not in src.lower()


def test_t40_no_nsga3_in_controller():
    """T40: No NSGA-III in controller or experiment code."""
    for fpath in [
        "src/prdrl/controllers/rule_based.py",
        "src/prdrl/experiments/E3_rule_based_regional_controller.py",
    ]:
        src = Path(fpath).read_text()
        assert "nsga3" not in src.lower(), f"NSGA-III found in {fpath}"
        assert "NSGA3" not in src


# ============================================================
# T41-T55: Output files existence and schema
# ============================================================

def test_t41_trajectories_exists():
    """T41: E3_trajectories.jsonl exists."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    assert len(rows) > 0


def test_t42_trajectories_row_count():
    """T42: Trajectories has exactly 60×3×5×10 = 9000 rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    expected = 60 * 3 * 5 * 10
    assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"


def test_t43_trajectories_required_columns():
    """T43: Each trajectory row has required columns."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    required = {
        "scenario", "seed", "block", "controller", "k",
        "n_active_regions", "total_response_fe", "igd_pre",
        "igd_response", "igd_epoch_end",
    }
    for row in rows[:10]:
        missing = required - set(row.keys())
        assert not missing, f"Missing columns: {missing}"


def test_t44_carry_forward_implies_divergence():
    """T44: Different controllers show different igd_epoch_end (carry-forward diverges)."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    # For same seed/scenario/k=5, different controllers should generally differ
    seed, sc, k = 730, "MIXED", 5
    ctrl_vals = {}
    for row in rows:
        if row["seed"] == seed and row["scenario"] == sc and row["k"] == k:
            ctrl_vals[row["controller"]] = row["igd_epoch_end"]
    assert len(ctrl_vals) == 5, f"Missing controllers for seed={seed}/sc={sc}/k={k}"
    # At least some controllers should differ after 5 changes
    vals = list(ctrl_vals.values())
    assert not all(abs(v - vals[0]) < 1e-12 for v in vals[1:]), "All controllers identical — carry-forward not working"


def test_t45_decisions_exists():
    """T45: E3_region_decisions.jsonl exists."""
    rows = _read_jsonl("E3_region_decisions.jsonl")
    assert len(rows) > 0


def test_t46_decisions_required_columns():
    """T46: Each decision row has required columns."""
    rows = _read_jsonl("E3_region_decisions.jsonl")
    required = {
        "scenario", "seed", "block", "controller", "k",
        "region_id", "active", "c", "l", "o", "dc",
        "requested_action", "realized_operator", "fallback_used",
        "matched_rule_id", "response_fe",
    }
    for row in rows[:10]:
        missing = required - set(row.keys())
        assert not missing, f"Missing columns: {missing}"


def test_t47_decisions_only_active_regions():
    """T47: All decision rows have active=True."""
    rows = _read_jsonl("E3_region_decisions.jsonl")
    assert all(row["active"] for row in rows), "Inactive region found in decision log"


def test_t48_seed_summary_exists():
    """T48: E3_seed_summary.csv exists."""
    rows = _read_csv("E3_seed_summary.csv")
    assert len(rows) > 0


def test_t49_seed_summary_all_seeds():
    """T49: E3_seed_summary.csv covers all 60 eval seeds × 3 scenarios × 5 controllers."""
    rows = _read_csv("E3_seed_summary.csv")
    combos = {(int(r["seed"]), r["scenario"], r["controller"]) for r in rows}
    expected = len(EVAL_SEEDS) * len(SCENARIOS) * len(ALL_CONTROLLERS)
    assert len(combos) == expected, f"Expected {expected} combos, got {len(combos)}"


def test_t50_block_summary_exists():
    """T50: E3_block_summary.csv exists."""
    rows = _read_csv("E3_block_summary.csv")
    assert len(rows) >= len(SCENARIOS) * 3  # 3 blocks: A, B, POOLED


def test_t51_scenario_summary_exists():
    """T51: E3_scenario_summary.csv exists with all scenarios."""
    rows = _read_csv("E3_scenario_summary.csv")
    scenarios_found = {r["scenario"] for r in rows}
    assert scenarios_found == set(SCENARIOS)


def test_t52_action_distribution_exists():
    """T52: E3_action_distribution.csv exists."""
    rows = _read_csv("E3_action_distribution.csv")
    assert len(rows) > 0


def test_t53_fe_summary_exists():
    """T53: E3_fe_summary.csv exists."""
    rows = _read_csv("E3_fe_summary.csv")
    assert len(rows) > 0


def test_t54_mixed_regime_summary_exists():
    """T54: E3_mixed_regime_summary.csv exists with both regimes."""
    rows = _read_csv("E3_mixed_regime_summary.csv")
    regimes = {r["target_regime"] for r in rows}
    assert "persistent" in regimes
    assert "stationary" in regimes


def test_t55_verdict_exists():
    """T55: E3_verdict.json exists."""
    d = _read_json("E3_verdict.json")
    assert "GATE_B" in d
    assert d["GATE_B"] in {"PASS", "PARTIAL", "FAIL"}


# ============================================================
# T56-T65: Trajectory correctness
# ============================================================

def test_t56_igd_pre_non_negative():
    """T56: igd_pre >= 0 for all trajectory rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for row in rows:
        assert row["igd_pre"] >= 0, f"Negative igd_pre: {row}"


def test_t57_igd_response_non_negative():
    """T57: igd_response >= 0 for all trajectory rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for row in rows:
        assert row["igd_response"] >= 0


def test_t58_igd_epoch_non_negative():
    """T58: igd_epoch_end >= 0 for all trajectory rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for row in rows:
        assert row["igd_epoch_end"] >= 0


def test_t59_global_keep_zero_response_fe():
    """T59: GLOBAL_KEEP has total_response_fe = 0 for all changes."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for row in rows:
        if row["controller"] == "GLOBAL_KEEP":
            assert row["total_response_fe"] == 0, \
                f"GLOBAL_KEEP has nonzero response_fe: {row['total_response_fe']}"


def test_t60_global_local_response_fe_equals_budget_per_active():
    """T60: GLOBAL_LOCAL response FE = 10 × n_active_regions."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for row in rows:
        if row["controller"] == "GLOBAL_LOCAL":
            expected = 10 * row["n_active_regions"]
            assert row["total_response_fe"] == expected, \
                f"GLOBAL_LOCAL FE mismatch: {row['total_response_fe']} != {expected}"


def test_t61_scenarios_covered():
    """T61: All 3 scenarios appear in trajectory rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    scenarios = {r["scenario"] for r in rows}
    assert scenarios == set(SCENARIOS)


def test_t62_controllers_covered():
    """T62: All 5 controllers appear in trajectory rows."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    ctrls = {r["controller"] for r in rows}
    assert ctrls == set(ALL_CONTROLLERS)


def test_t63_blocks_a_and_b_covered():
    """T63: Both Block A and Block B appear in trajectory."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    blocks = {r["block"] for r in rows}
    assert "A" in blocks
    assert "B" in blocks


def test_t64_n_changes_per_arm_is_10():
    """T64: Each (seed, scenario, controller) has exactly 10 scored changes."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    from collections import Counter
    counts = Counter((r["seed"], r["scenario"], r["controller"]) for r in rows)
    for key, cnt in counts.items():
        assert cnt == 10, f"Unexpected n_changes={cnt} for {key}"


def test_t65_k_values_zero_to_nine():
    """T65: Change indices k run 0 to 9 for each arm."""
    rows = _read_jsonl("E3_trajectories.jsonl")
    for seed in [721, 730, 750, 751, 780]:
        for sc in SCENARIOS:
            for ctrl in ["RULE_V1", "GLOBAL_LOCAL"]:
                ks = sorted(r["k"] for r in rows
                            if r["seed"] == seed and r["scenario"] == sc and r["controller"] == ctrl)
                assert ks == list(range(10)), f"Bad k sequence for seed={seed}/{sc}/{ctrl}: {ks}"


# ============================================================
# T66-T75: Statistics correctness
# ============================================================

def test_t66_bootstrap_ci_function():
    """T66: bootstrap_ci returns (mean, lo, hi) with lo <= mean <= hi."""
    from prdrl.experiments.E3_rule_based_regional_controller import bootstrap_ci
    rng = np.random.default_rng(42)
    vals = rng.normal(0.01, 0.05, size=30).tolist()
    mean, lo, hi = bootstrap_ci(vals, 5000, 12345)
    assert lo <= mean <= hi


def test_t67_classify_effect_all_categories():
    """T67: classify_effect produces correct categories."""
    from prdrl.experiments.E3_rule_based_regional_controller import classify_effect, DELTA_E3
    d = DELTA_E3
    assert classify_effect(d * 10, d, d * 5) == "BENEFICIAL"
    assert classify_effect(d * 10, -d * 2, d * 20) == "DIRECTIONAL_POSITIVE"
    assert classify_effect(0.0, -d * 0.5, d * 0.5) == "PRACTICALLY_NEUTRAL"
    assert classify_effect(-d * 10, -d * 20, -d) == "HARMFUL"
    assert classify_effect(-d * 10, -d * 20, d * 5) == "DIRECTIONAL_NEGATIVE"


def test_t68_block_summary_has_all_scenarios():
    """T68: E3_block_summary.csv covers PERSISTENT, STATIONARY, MIXED."""
    rows = _read_csv("E3_block_summary.csv")
    scs = {r["scenario"] for r in rows}
    assert scs == set(SCENARIOS)


def test_t69_block_summary_has_a_b_pooled():
    """T69: E3_block_summary.csv has block A, B, POOLED rows."""
    rows = _read_csv("E3_block_summary.csv")
    blocks = {r["block"] for r in rows}
    assert "A" in blocks
    assert "B" in blocks
    assert "POOLED" in blocks


def test_t70_n_seeds_in_block_summary():
    """T70: Block A has n_seeds=30, Block B has n_seeds=30 in mixed scenario."""
    rows = _read_csv("E3_block_summary.csv")
    for row in rows:
        if row["scenario"] == "MIXED" and row["block"] == "A":
            assert int(row["n_seeds"]) == 30, f"Block A n_seeds={row['n_seeds']}"
        if row["scenario"] == "MIXED" and row["block"] == "B":
            assert int(row["n_seeds"]) == 30, f"Block B n_seeds={row['n_seeds']}"


def test_t71_mean_benefit_is_numeric():
    """T71: mean_benefit in block summary is a valid float."""
    rows = _read_csv("E3_block_summary.csv")
    for row in rows:
        val = float(row["mean_benefit"])
        assert np.isfinite(val), f"Non-finite mean_benefit: {val}"


def test_t72_ci_lo_le_mean_le_ci_hi():
    """T72: CI is ordered: ci_lo <= mean_benefit <= ci_hi."""
    rows = _read_csv("E3_block_summary.csv")
    for row in rows:
        lo   = float(row["ci_lo"])
        mean = float(row["mean_benefit"])
        hi   = float(row["ci_hi"])
        assert lo <= mean <= hi, f"CI ordering violated: lo={lo} mean={mean} hi={hi}"


def test_t73_effect_classification_valid():
    """T73: All effect values in block summary are valid effect names."""
    valid_effects = {"BENEFICIAL", "DIRECTIONAL_POSITIVE", "PRACTICALLY_NEUTRAL",
                     "DIRECTIONAL_NEGATIVE", "HARMFUL"}
    rows = _read_csv("E3_block_summary.csv")
    for row in rows:
        assert row["effect"] in valid_effects, f"Invalid effect: {row['effect']}"


def test_t74_fe_summary_rule_v1_fe_nonneg():
    """T74: RULE_V1 response FE is non-negative in all scenarios."""
    rows = _read_csv("E3_fe_summary.csv")
    for row in rows:
        if row["controller"] == "RULE_V1":
            assert float(row["mean_response_fe_per_change"]) >= 0


def test_t75_fe_summary_global_keep_zero():
    """T75: GLOBAL_KEEP has zero mean response FE in FE summary."""
    rows = _read_csv("E3_fe_summary.csv")
    for row in rows:
        if row["controller"] == "GLOBAL_KEEP":
            assert float(row["mean_response_fe_per_change"]) == pytest.approx(0.0)


# ============================================================
# T76-T82: Gate B and verdict correctness
# ============================================================

def test_t76_gate_b_verdict_valid():
    """T76: GATE_B is one of PASS, PARTIAL, FAIL."""
    d = _read_json("E3_verdict.json")
    assert d["GATE_B"] in {"PASS", "PARTIAL", "FAIL"}


def test_t77_mixed_rule_gain_valid():
    """T77: E3_MIXED_RULE_GAIN is one of VALIDATED, PARTIAL, FAILED."""
    d = _read_json("E3_verdict.json")
    verd = d["gate_verdicts"]
    assert verd["E3_MIXED_RULE_GAIN"] in {"VALIDATED", "PARTIAL", "FAILED"}


def test_t78_safety_verdicts_present():
    """T78: PERSISTENT_SAFETY and STATIONARY_SAFETY in verdict."""
    d = _read_json("E3_verdict.json")
    verd = d["gate_verdicts"]
    assert "PERSISTENT_SAFETY" in verd
    assert "STATIONARY_SAFETY" in verd
    assert verd["PERSISTENT_SAFETY"] in {"PASS", "FAIL"}
    assert verd["STATIONARY_SAFETY"] in {"PASS", "FAIL"}


def test_t79_verdict_invariant_fields():
    """T79: Verdict confirms no DRL, no sigma tuning, no HARD, no B_r(a)."""
    d = _read_json("E3_verdict.json")
    assert d["RL_STATUS"]                    == "NOT_STARTED"
    assert d["thresholds_tuned_from_performance"] is False
    assert d["sigma_tuned"]                  is False
    assert d["HARD_used"]                    is False
    assert d["B_r_a_activated"]              is False
    assert d["production_code_modified"]     is False


def test_t80_verdict_m_gate_verdicts_boolean():
    """T80: M1..M8 gate verdicts are boolean."""
    d = _read_json("E3_verdict.json")
    verd = d["gate_verdicts"]
    for m_key in ["M1_pooled_mean_positive", "M2_pooled_ci_lo_positive",
                  "M3_blockA_mean_positive", "M4_blockB_mean_positive",
                  "M5_no_harmful_blockA", "M6_no_harmful_blockB",
                  "M7_fe_ok", "M8_engineering_ok"]:
        assert isinstance(verd[m_key], bool), f"{m_key} is not bool"


def test_t81_gate_b_pass_requires_both_safety_pass():
    """T81: If GATE_B=PASS then both PERSISTENT_SAFETY and STATIONARY_SAFETY must be PASS."""
    d = _read_json("E3_verdict.json")
    if d["GATE_B"] == "PASS":
        verd = d["gate_verdicts"]
        assert verd["PERSISTENT_SAFETY"] == "PASS"
        assert verd["STATIONARY_SAFETY"] == "PASS"


def test_t82_gate_b_pass_requires_mixed_validated():
    """T82: If GATE_B=PASS then E3_MIXED_RULE_GAIN must be VALIDATED."""
    d = _read_json("E3_verdict.json")
    if d["GATE_B"] == "PASS":
        assert d["gate_verdicts"]["E3_MIXED_RULE_GAIN"] == "VALIDATED"


# ============================================================
# T83-T88: Regression and Oracle integrity
# ============================================================

def test_t83_oracle_hash_unchanged():
    """T83: oracle.py hash in E3 pre-run manifest matches R0 manifest."""
    e3_manifest = _read_json("E3_pre_run_manifest.json")
    r0_manifest_path = Path("results/R0_rl_interface_validation/R0_prechange_manifest.json")
    assert r0_manifest_path.exists(), "R0 prechange manifest missing"
    with open(r0_manifest_path) as f:
        r0_manifest = json.load(f)
    e3_hash = e3_manifest["files"].get("src/prdrl/oracle/oracle.py")
    r0_hash = r0_manifest.get("oracle_py_sha256") or (
        r0_manifest.get("files", {}).get("src/prdrl/oracle/oracle.py")
    )
    assert e3_hash is not None, "E3 manifest missing oracle.py hash"
    assert r0_hash is not None, "R0 manifest missing oracle.py hash"
    assert e3_hash == r0_hash, \
        f"oracle.py hash changed! E3={e3_hash} R0={r0_hash}"


def test_t84_predict_fallback_recorded_in_decisions():
    """T84: PREDICT fallback events are recorded in decision log with correct fallback_reason."""
    rows = _read_jsonl("E3_region_decisions.jsonl")
    fallback_rows = [r for r in rows if r["fallback_used"] and r["requested_action"] == "PREDICT"]
    # Fallbacks only happen on early changes (insufficient history)
    for row in fallback_rows:
        assert row["fallback_reason"] == "MISSING_PREDICT_HISTORY"
        assert row["realized_operator"] == "LOCAL"


def test_t85_action_distribution_fractions_sum_to_one():
    """T85: Within each (scenario, block, controller), action fractions sum to 1."""
    rows = _read_csv("E3_action_distribution.csv")
    from collections import defaultdict
    by_group = defaultdict(float)
    for row in rows:
        key = (row["scenario"], row["block"], row["controller"])
        by_group[key] += float(row["fraction"])
    for key, total in by_group.items():
        assert abs(total - 1.0) < 1e-4, f"Fractions don't sum to 1 for {key}: {total}"


def test_t86_report_md_exists():
    """T86: E3_report.md exists and mentions Gate B."""
    p = RESULTS / "E3_report.md"
    assert p.exists(), "E3_report.md missing"
    content = p.read_text()
    assert "Gate B" in content or "GATE_B" in content


def test_t87_state_semantics_has_correct_dim():
    """T87: E3_state_semantics.json has feature_vector_dim = 2D+2 = 22."""
    d = _read_json("E3_state_semantics.json")
    assert d["D"] == D
    assert d["feature_vector_dim"] == 2 * D + 2


@pytest.mark.parametrize("verdict_key,expected_type", [
    ("GATE_B", str),
    ("RL_STATUS", str),
    ("n_eval_seeds", int),
    ("n_scenarios", int),
    ("n_controllers", int),
    ("n_scored_changes", int),
])
def test_t88_verdict_field_types(verdict_key, expected_type):
    """T88: Verdict JSON has correct field types."""
    d = _read_json("E3_verdict.json")
    val = d[verdict_key]
    assert isinstance(val, expected_type), \
        f"verdict['{verdict_key}'] is {type(val)}, expected {expected_type}"
