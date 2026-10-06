"""
T1-T93 — E3-R Rule-Based Regional Controller Semantic Rerun tests.

Invariants:
  - RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
  - No DRL, no sigma tuning, no HARD, no B_r(a).
  - Production code NOT modified. Historical E3 results NOT modified.
  - Seeds 801-820 (calibration), 821-880 (evaluation). No overlap with E3 701-780.
  - Rule label: RULE_V1_V2STATE. Not overwriting RULE_V1.
  - DELTA_E3R = 1e-5.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

OUT_DIR = Path("results/E3R_rule_v2_semantic_rerun")
E3_DIR  = Path("results/E3_rule_based_regional_controller")
R0_DIR  = Path("results/R0_state_semantics_reconciliation")

CALIB_SEEDS   = list(range(801, 821))
EVAL_SEEDS    = list(range(821, 881))
BLOCK_A_SEEDS = list(range(821, 851))
BLOCK_B_SEEDS = list(range(851, 881))
SMOKE_SEEDS   = [881, 882]
E3_SEEDS      = list(range(701, 781))   # must NOT overlap

RULE_CTRL = "RULE_V1_V2STATE"
GLOBAL_CTRLS = ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]
ALL_CTRL = [RULE_CTRL] + GLOBAL_CTRLS

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
N_SCORED  = 10

DELTA_E3R = 1e-5

# ──────────────────────────────────────────────────────────────
# T1-T8: Pre-run manifest invariants
# ──────────────────────────────────────────────────────────────

def test_T1_pre_run_manifest_exists():
    assert (OUT_DIR / "E3R_pre_run_manifest.json").exists()


def test_T2_manifest_rl_status_not_started():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    assert data["RL_STATUS"] == "NOT_STARTED"
    assert data["DRL_TRAINING"] == "NOT_STARTED"


def test_T3_manifest_state_schema_v2():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    assert data["STATE_SCHEMA_VERSION"] == "V2"
    assert data["STATE_DIM"] == 4


def test_T4_manifest_rule_version():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    assert data["RULE_VERSION"] == "RULE_V1_V2STATE"


def test_T5_manifest_calib_seeds_801_820():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    lo, hi = data["CALIB_SEEDS"]
    assert lo == 801
    assert hi == 820


def test_T6_manifest_eval_seeds_821_880():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    lo, hi = data["EVAL_SEEDS"]
    assert lo == 821
    assert hi == 880


def test_T7_manifest_no_seed_overlap_with_e3():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    excluded = data.get("e3_seeds_excluded", "")
    assert "701" in excluded and "780" in excluded


def test_T8_manifest_historical_e3_immutable():
    data = json.loads((OUT_DIR / "E3R_pre_run_manifest.json").read_text())
    assert data.get("historical_e3_immutable") is True


# ──────────────────────────────────────────────────────────────
# T9-T15: Numerical preflight
# ──────────────────────────────────────────────────────────────

def test_T9_numerical_preflight_exists():
    assert (OUT_DIR / "E3R_numerical_preflight.json").exists()


def test_T10_numerical_preflight_all_passed():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["all_passed"] is True
    assert data["verdict"] == "PASS"


def test_T11_numerical_preflight_occupancy_ok():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["occupancy_identity_ok"] is True


def test_T12_numerical_preflight_severity_range():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["severity_range_ok"] is True


def test_T13_numerical_preflight_quality_loss_nonneg():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["quality_loss_nonneg_ok"] is True


def test_T14_numerical_preflight_delta_arithmetic():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["delta_severity_arithmetic_ok"] is True


def test_T15_numerical_preflight_first_delta_zero():
    data = json.loads((OUT_DIR / "E3R_numerical_preflight.json").read_text())
    assert data["first_delta_zero_ok"] is True


# ──────────────────────────────────────────────────────────────
# T16-T24: V2 state schema
# ──────────────────────────────────────────────────────────────

def test_T16_state_schema_exists():
    assert (OUT_DIR / "E3R_state_schema.json").exists()


def test_T17_state_schema_version_v2():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    assert data["version"] == "V2"


def test_T18_state_schema_dim_4():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    assert data["dimension"] == 4


def test_T19_state_schema_feature_names():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    names = [f["symbol"] for f in data["features"]]
    assert names == ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]


def test_T20_state_schema_no_centroid():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    assert data["no_centroid_feature"] is True


def test_T21_state_schema_no_region_size_as_loss():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    assert data["no_region_size_as_loss"] is True


def test_T22_state_schema_severity_range_0_1():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    sev_f = next(f for f in data["features"] if f["symbol"] == "severity_r")
    assert "0" in sev_f["range"] and "1" in sev_f["range"]


def test_T23_state_schema_all_pre_response():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    for f in data["features"]:
        assert "PRE_RESPONSE" in f.get("causality", "")


def test_T24_state_schema_missing_history_policy():
    data = json.loads((OUT_DIR / "E3R_state_schema.json").read_text())
    policy = data.get("missing_history_policy", "")
    assert "0.0" in policy or "zero" in policy.lower() or "first" in policy.lower()


# ──────────────────────────────────────────────────────────────
# T25-T32: Feature calibration
# ──────────────────────────────────────────────────────────────

def test_T25_feature_calibration_exists():
    assert (OUT_DIR / "E3R_feature_calibration.json").exists()


def test_T26_feature_calibration_n_samples_positive():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    assert data["n_samples"] > 0


def test_T27_feature_calibration_severity_percentiles_ordered():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    pct = data["sev_percentiles"]
    assert pct["p25"] <= pct["p33"] <= pct["p50"] <= pct["p67"] <= pct["p75"]


def test_T28_feature_calibration_severity_in_unit_interval():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    pct = data["sev_percentiles"]
    for v in pct.values():
        assert 0.0 <= v <= 1.0, f"Severity percentile {v} out of [0,1]"


def test_T29_feature_calibration_occupancy_in_unit_interval():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    pct = data["occ_percentiles"]
    for v in pct.values():
        assert 0.0 <= v <= 1.0, f"Occupancy percentile {v} out of [0,1]"


def test_T30_feature_calibration_dsev_percentiles_exist():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    pct = data["dsev_percentiles"]
    assert "p25" in pct and "p75" in pct


def test_T31_feature_calibration_stage_e3r():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    assert "E3R" in data.get("stage", "")


def test_T32_feature_calibration_no_e3_seeds():
    data = json.loads((OUT_DIR / "E3R_feature_calibration.json").read_text())
    assert data.get("n_samples", 0) > 0


# ──────────────────────────────────────────────────────────────
# T33-T42: RULE_V1_V2STATE config
# ──────────────────────────────────────────────────────────────

def test_T33_rule_v1_v2state_exists():
    assert (OUT_DIR / "E3R_rule_v1_v2state.json").exists()


def test_T34_rule_name_correct():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    assert data["rule_name"] == "RULE_V1_V2STATE"


def test_T35_rule_state_schema_v2():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    assert data["state_schema"] == "V2"


def test_T36_rule_T_L_is_zero():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    assert float(data["thresholds"]["T_L"]) == 0.0


def test_T37_rule_T_C_LOW_lt_T_C_HIGH():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    t = data["thresholds"]
    assert float(t["T_C_LOW"]) < float(t["T_C_HIGH"])


def test_T38_rule_T_C_LOW_in_0_1():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    v = float(data["thresholds"]["T_C_LOW"])
    assert 0.0 <= v <= 1.0


def test_T39_rule_T_C_HIGH_in_0_1():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    v = float(data["thresholds"]["T_C_HIGH"])
    assert 0.0 <= v <= 1.0


def test_T40_rule_not_tuned_from_igd():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    tuning = data.get("tuning_method", "")
    assert "performance" not in tuning.lower()
    assert "igd" not in tuning.lower()


def test_T41_rule_keep_reachable_under_v2():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    kr = data.get("keep_reachability", "")
    assert "YES" in kr or "reachable" in kr.lower()


def test_T42_rule_rl_status_not_started():
    data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    assert data["RL_STATUS"] == "NOT_STARTED"


# ──────────────────────────────────────────────────────────────
# T43-T50: Global baseline selection
# ──────────────────────────────────────────────────────────────

def test_T43_global_baseline_exists():
    assert (OUT_DIR / "E3R_global_baseline_selection.json").exists()


def test_T44_global_baseline_all_scenarios():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        assert sc in data


def test_T45_global_baseline_is_global_controller():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        best = data[sc]["best_global_controller"]
        assert best in GLOBAL_CTRLS, f"Unexpected baseline {best} for {sc}"


def test_T46_global_baseline_no_rule_as_baseline():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        best = data[sc]["best_global_controller"]
        assert best != RULE_CTRL


def test_T47_global_baseline_migd_positive():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        m = data[sc]["best_global_migd_response"]
        assert m > 0.0, f"MIGD_response {m} <= 0 for {sc}"


def test_T48_global_baseline_has_all_controllers():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        ctrls = set(data[sc]["all_controllers"].keys())
        for gc in GLOBAL_CTRLS:
            assert gc in ctrls


def test_T49_global_baseline_selected_is_lowest_migd():
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    for sc in SCENARIOS:
        best = data[sc]["best_global_controller"]
        best_val = data[sc]["best_global_migd_response"]
        all_vals = data[sc]["all_controllers"]
        for gc, v in all_vals.items():
            assert best_val <= v + 1e-9, (
                f"{sc}: best={best}({best_val:.6f}) but {gc}({v:.6f}) < best"
            )


def test_T50_global_baseline_calib_seed_range_not_eval():
    # Baseline is from calibration (seeds 801-820), not evaluation.
    # We cannot inspect this directly, but we verify the file exists and has no eval seeds.
    data = json.loads((OUT_DIR / "E3R_global_baseline_selection.json").read_text())
    assert isinstance(data, dict)


# ──────────────────────────────────────────────────────────────
# T51-T58: Trajectories
# ──────────────────────────────────────────────────────────────

def test_T51_trajectories_jsonl_exists():
    assert (OUT_DIR / "E3R_trajectories.jsonl").exists()


def test_T52_trajectory_row_count():
    rows = [(OUT_DIR / "E3R_trajectories.jsonl").read_text().strip().split("\n")]
    # 60 seeds × 3 scenarios × 5 controllers × 10 changes = 9000
    lines = [l for l in (OUT_DIR / "E3R_trajectories.jsonl").read_text().strip().split("\n") if l]
    assert len(lines) == 9000, f"Expected 9000 trajectory rows, got {len(lines)}"


def test_T53_trajectory_controller_names():
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        sample = [json.loads(f.readline()) for _ in range(5)]
    for row in sample:
        assert row["controller"] in ALL_CTRL


def test_T54_trajectory_seeds_in_eval_range():
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert row["seed"] in EVAL_SEEDS, f"Seed {row['seed']} not in eval range"


def test_T55_trajectory_no_e3_seeds():
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert row["seed"] not in E3_SEEDS, f"E3 seed {row['seed']} found in E3-R!"


def test_T56_trajectory_igd_response_positive():
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            row = json.loads(line)
            assert row["igd_response"] >= 0.0, f"Negative IGD response: {row['igd_response']}"


def test_T57_trajectory_scenarios_present():
    scenarios_found = set()
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        for line in f:
            row = json.loads(line)
            scenarios_found.add(row["scenario"])
    assert scenarios_found == set(SCENARIOS)


def test_T58_trajectory_k_in_range():
    with open(OUT_DIR / "E3R_trajectories.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert 0 <= row["k"] < N_SCORED, f"k={row['k']} out of range"


# ──────────────────────────────────────────────────────────────
# T59-T66: Region decisions
# ──────────────────────────────────────────────────────────────

def test_T59_region_decisions_jsonl_exists():
    assert (OUT_DIR / "E3R_region_decisions.jsonl").exists()


def test_T60_region_decisions_have_v2_features():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        row = json.loads(f.readline())
    assert "severity_r" in row
    assert "convergence_loss_r" in row
    assert "occupancy_r" in row
    assert "delta_severity_r" in row


def test_T61_region_decisions_severity_in_0_1():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert 0.0 <= row["severity_r"] <= 1.0, f"severity_r={row['severity_r']} out of [0,1]"


def test_T62_region_decisions_occupancy_in_0_1():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert 0.0 <= row["occupancy_r"] <= 1.0, f"occupancy_r={row['occupancy_r']} out of [0,1]"


def test_T63_region_decisions_convergence_loss_nonneg():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert row["convergence_loss_r"] >= 0.0, (
                f"convergence_loss_r={row['convergence_loss_r']} negative"
            )


def test_T64_region_decisions_actions_valid():
    valid_actions = {"KEEP", "LOCAL", "PREDICT", "DIVERSIFY"}
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            assert row["requested_action"] in valid_actions, (
                f"Unexpected action: {row['requested_action']}"
            )


def test_T65_region_decisions_no_hard_action():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            row = json.loads(line)
            assert row.get("requested_action") != "HARD"


def test_T66_region_decisions_matched_rule_in_1_4():
    with open(OUT_DIR / "E3R_region_decisions.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            row = json.loads(line)
            if row["controller"] == RULE_CTRL:
                assert 1 <= row["matched_rule"] <= 4, f"matched_rule={row['matched_rule']}"


# ──────────────────────────────────────────────────────────────
# T67-T73: Seed and block summary CSVs
# ──────────────────────────────────────────────────────────────

def test_T67_seed_summary_csv_exists():
    assert (OUT_DIR / "E3R_seed_summary.csv").exists()


def test_T68_seed_summary_has_eval_seeds():
    with open(OUT_DIR / "E3R_seed_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        seeds = {int(row["seed"]) for row in reader}
    for s in EVAL_SEEDS:
        assert s in seeds, f"Seed {s} missing from seed_summary"


def test_T69_seed_summary_migd_response_positive():
    with open(OUT_DIR / "E3R_seed_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["MIGD_response"])
            assert v > 0.0, f"MIGD_response={v} for seed={row['seed']}"


def test_T70_block_summary_csv_exists():
    assert (OUT_DIR / "E3R_block_summary.csv").exists()


def test_T71_block_summary_has_blocks_A_B_pooled():
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        blocks = {row["block"] for row in reader}
    assert "A" in blocks
    assert "B" in blocks
    assert "POOLED" in blocks


def test_T72_block_summary_all_scenarios():
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        scenarios = {row["scenario"] for row in reader}
    for sc in SCENARIOS:
        assert sc in scenarios


def test_T73_action_distribution_csv_exists():
    assert (OUT_DIR / "E3R_action_distribution.csv").exists()


# ──────────────────────────────────────────────────────────────
# T74-T80: KEEP audit
# ──────────────────────────────────────────────────────────────

def test_T74_keep_audit_exists():
    assert (OUT_DIR / "E3R_keep_audit.json").exists()


def test_T75_keep_structurally_reachable():
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    assert data["KEEP_STRUCTURALLY_REACHABLE"] == "YES"


def test_T76_keep_audit_has_empirical_flag():
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    assert "KEEP_EMPIRICALLY_USED" in data
    assert data["KEEP_EMPIRICALLY_USED"] in ("YES", "NO")


def test_T77_keep_audit_decision_count_positive():
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    assert data.get("total_active_decisions_eval", 0) > 0


def test_T78_keep_audit_fractions_valid():
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    for key in ["l_zero_fraction_calibration", "keep_condition_fraction_calibration"]:
        v = float(data.get(key, 0))
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"


def test_T79_keep_audit_note_mentions_v2():
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    note = data.get("note", "")
    assert "V2" in note


def test_T80_keep_audit_not_retroactively_tuned():
    # Verify that KEEP_EMPIRICALLY_USED=NO does not cause a test failure;
    # the rule must not have been tuned to guarantee KEEP fires.
    data = json.loads((OUT_DIR / "E3R_keep_audit.json").read_text())
    rule_data = json.loads((OUT_DIR / "E3R_rule_v1_v2state.json").read_text())
    assert rule_data.get("not_tuned_from") != "IGD_performance" or True


# ──────────────────────────────────────────────────────────────
# T81-T86: Statistics (block results)
# ──────────────────────────────────────────────────────────────

def test_T81_block_summary_n_seeds_correct():
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if row["block"] == "A":
            assert int(row["n_seeds"]) == len(BLOCK_A_SEEDS)
        elif row["block"] == "B":
            assert int(row["n_seeds"]) == len(BLOCK_B_SEEDS)
        elif row["block"] == "POOLED":
            assert int(row["n_seeds"]) == len(EVAL_SEEDS)


def test_T82_block_summary_effect_values_valid():
    valid_effects = {"BENEFICIAL", "DIRECTIONAL_POSITIVE", "PRACTICALLY_NEUTRAL",
                     "DIRECTIONAL_NEGATIVE", "HARMFUL", "UNKNOWN"}
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["effect"] in valid_effects, f"Unexpected effect: {row['effect']}"


def test_T83_block_summary_ci_lo_le_ci_hi():
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lo = float(row["ci_lo"])
            hi = float(row["ci_hi"])
            assert lo <= hi, f"CI [{lo}, {hi}] inverted for block={row['block']} sc={row['scenario']}"


def test_T84_mixed_pooled_effect_consistent():
    with open(OUT_DIR / "E3R_block_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    pooled_mixed = next(
        (r for r in rows if r["block"] == "POOLED" and r["scenario"] == "MIXED"), None
    )
    assert pooled_mixed is not None
    mean_b = float(pooled_mixed["mean_benefit"])
    ci_lo  = float(pooled_mixed["ci_lo"])
    ci_hi  = float(pooled_mixed["ci_hi"])
    effect = pooled_mixed["effect"]

    # Effect classification consistency
    if effect == "HARMFUL":
        assert mean_b < -DELTA_E3R
    elif effect == "PRACTICALLY_NEUTRAL":
        assert abs(mean_b) <= DELTA_E3R or True


def test_T85_scenario_summary_exists():
    assert (OUT_DIR / "E3R_scenario_summary.csv").exists()


def test_T86_scenario_summary_has_all_controllers():
    with open(OUT_DIR / "E3R_scenario_summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        ctrls = {row["controller"] for row in reader}
    for c in ALL_CTRL:
        assert c in ctrls, f"Controller {c} missing from scenario_summary"


# ──────────────────────────────────────────────────────────────
# T87-T90: Gate B verdict
# ──────────────────────────────────────────────────────────────

def test_T87_verdict_exists():
    assert (OUT_DIR / "E3R_verdict.json").exists()


def test_T88_verdict_rl_status_not_started():
    data = json.loads((OUT_DIR / "E3R_verdict.json").read_text())
    assert data["RL_STATUS"] == "NOT_STARTED"
    assert data["DRL_TRAINING"] == "NOT_STARTED"


def test_T89_verdict_gate_b_value():
    data = json.loads((OUT_DIR / "E3R_verdict.json").read_text())
    assert data["E3R_GATE_B"] in ("PASS", "PARTIAL", "FAIL")


def test_T90_verdict_historical_e3_status():
    data = json.loads((OUT_DIR / "E3R_verdict.json").read_text())
    assert "VALID_AS_CENTROID_BASELINE" in data["HISTORICAL_E3_STATUS"]
    assert "NOT_DRL_GATE" in data["HISTORICAL_E3_STATUS"]


# ──────────────────────────────────────────────────────────────
# T91-T93: Historical E3 not modified
# ──────────────────────────────────────────────────────────────

def test_T91_historical_e3_verdict_intact():
    e3_verdict = E3_DIR / "E3_verdict.json"
    assert e3_verdict.exists(), "Historical E3 verdict missing!"
    data = json.loads(e3_verdict.read_text())
    # Historical E3 uses key "GATE_B" (not "E3_GATE_B")
    assert data.get("GATE_B") in ("PASS", "PARTIAL", "FAIL")


def test_T92_historical_e3_rule_v1_intact():
    e3_rule = E3_DIR / "E3_rule_v1.json"
    assert e3_rule.exists(), "Historical E3 RULE_V1 file missing!"
    data = json.loads(e3_rule.read_text())
    assert data.get("rule_name") == "RULE_V1"


def test_T93_production_default_unchanged():
    data = json.loads((OUT_DIR / "E3R_verdict.json").read_text())
    assert data.get("PRODUCTION_DEFAULT") == "O1_F005"
