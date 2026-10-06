"""
E4ITR — Reward A/B Training Execution: artifact validation tests.

Validates the executed matched reward experiment (V3+R0 vs V3+R1) from saved artifacts
WITHOUT rerunning the 10800 training episodes: run/episode/checkpoint completeness, paired
initialization, single-factor + no-leak integrity, reward-version/config integrity, reward
records reconstruction, mechanism diagnostics, validation matrix + checkpoint selection, and
the untouched final-seed firewall (no final verdict / no final bootstrap).
"""

import csv
import json
import math
from pathlib import Path

import pytest

from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, regional_quality_gain_r1

RES = Path("results/E4ITR_reward_ab_training_execution")
E4IT = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4II = Path("results/E4II_reward_implementation_validation")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
POLICY = [4001, 4002, 4003, 4004, 4005]
SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4ITR_verdict.json").exists():
        pytest.skip("E4ITR not run yet")


def _rows(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


# ---- prerequisites ----

def test_T1_e4it_complete():
    assert _load(E4IT / "E4IT_verdict.json")["E4IT_COMPLETE"] == "YES"


def test_T2_e4ii_valid():
    assert _load(E4II / "E4II_implementation_verdict.json")["R1_IMPLEMENTATION_VALID"] == "YES"


def test_T3_exact_two_arms():
    _require()
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    arms = {v["arm"] for v in sel.values()}
    assert arms == {"R0_CONTROL", "R1_TREATMENT"}


def test_T4_five_policy_seeds_each():
    _require()
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    for arm in ("R0_CONTROL", "R1_TREATMENT"):
        assert sorted(v["policy_seed"] for v in sel.values() if v["arm"] == arm) == POLICY


def test_T5_T6_T7_runs_and_episodes():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["RUNS_COMPLETE"] == "10/10"
    assert v["R0_RUNS_COMPLETE"] == "5/5" and v["R1_RUNS_COMPLETE"] == "5/5"
    assert v["TRAINING_EPISODES_COMPLETE"] == "10800/10800"


def test_T8_paired_initialization_equal():
    _require()
    pi = _load(RES / "E4ITR_pair_initialization_manifest.json")
    assert pi["all_online_equals_target"] is True and pi["R0_equals_R1_init_all"] is True


def test_T9_distinct_policy_inits():
    _require()
    assert _load(RES / "E4ITR_pair_initialization_manifest.json")["distinct_across_policy_seeds"] is True


def test_T10_T11_v3_both_arms():
    _require()
    m = _load(RES / "E4ITR_pre_execution_manifest.json")
    assert m["v3_constants_sha256"]
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    for v in sel.values():
        assert v["normalization_version"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T12_T13_reward_versions():
    _require()
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    for v in sel.values():
        if v["arm"] == "R0_CONTROL":
            assert v["reward_version"] == REWARD_R0_NAME
        else:
            assert v["reward_version"] == REWARD_R1_NAME


def test_T14_r1_config_sha_exact():
    _require()
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    for v in sel.values():
        if v["arm"] == "R1_TREATMENT":
            assert v["reward_config_sha256"] == SHA


def test_T15_T16_T17_alpha():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["R0_ALPHA_LOCAL"] == 0.1 and v["R1_ALPHA_LOCAL"] == 0.1
    assert v["R1_RUNTIME_ALPHA_CHANGED"] == "NO"


def test_T23_joint_architecture_unchanged():
    _require()
    assert _load(E4IT / "E4IT_network_contract.json")["joint_representation"] == "IMPLICIT_STATE_ONLY"


def test_T24_train_seeds():
    _require()
    assert _load(RES / "E4ITR_execution_config.json")["train_seeds"] == [1501, 1560]


def test_T26_final_seeds_never_accessed():
    _require()
    assert _load(RES / "E4ITR_final_firewall.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert _load(RES / "E4ITR_verdict.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0
    # no final seed appears in any episode row
    for r in _rows("E4ITR_training_episode_results.csv"):
        assert not (1581 <= int(r["environment_seed"]) <= 1640)


def test_T27_scenario_balance():
    _require()
    rows = _rows("E4ITR_training_episode_results.csv")
    # per arm per scenario counts equal (each scenario 60 env x 6 epochs x 5 policies = 1800)
    from collections import Counter
    for arm in ("R0_CONTROL", "R1_TREATMENT"):
        c = Counter(r["scenario"] for r in rows if r["arm"] == arm)
        assert c["PERSISTENT"] == c["STATIONARY"] == c["MIXED"] == 1800


def test_T29_T30_separate_replay_no_cross_sharing():
    _require()
    # single-factor audit records shared-everything-else + arm isolation; leakage gate NO
    assert _load(RES / "E4ITR_verdict.json")["WRONG_ARM_REWARD_LEAKAGE"] == "NO"


def test_T31_T32_T33_reward_records_reconstruct_and_no_leak():
    _require()
    rows = _rows("E4ITR_training_reward_records.csv")
    assert len(rows) > 0
    for r in rows:
        wg = float(r["weighted_global_reward"]); wfe = float(r["weighted_fe_penalty"])
        actual = float(r["actual_total_reward"]); alocal = float(r["actual_local_reward"])
        # actual total reconstructs from global + arm-local - fe
        assert abs(actual - (wg + 0.1 * alocal - wfe)) < 1e-9
        if r["arm"] == "R0_CONTROL":
            assert r["actual_reward_version"] == REWARD_R0_NAME
            assert abs(alocal - float(r["raw_local_reward_R0"])) < 1e-12
        else:
            assert r["actual_reward_version"] == REWARD_R1_NAME
            assert abs(alocal - float(r["g_r"])) < 1e-12


def test_T34_T35_global_and_fe_same_spec():
    _require()
    # global term = raw_global (alpha_g=1); fe = beta*raw_fe (beta=0.01) — same columns both arms
    rows = _rows("E4ITR_training_reward_records.csv")
    for r in rows:
        assert abs(float(r["weighted_global_reward"]) - float(r["raw_global_reward"])) < 1e-12
        assert abs(float(r["weighted_fe_penalty"]) - 0.01 * float(r["raw_fe_cost"])) < 1e-12


def test_T36_T37_r1_bounds_and_sign():
    _require()
    a = _load(RES / "E4ITR_r1_formula_runtime_audit.json")
    assert a["R1_g_bound_violations"] == 0 and a["R1_sign_violations"] == 0
    # independent recompute of g on R1 sample rows
    for r in _rows("E4ITR_training_reward_records.csv"):
        if r["arm"] == "R1_TREATMENT":
            g = regional_quality_gain_r1(float(r["L_pre"]), float(r["L_post"]))
            assert abs(g - float(r["g_r"])) <= 1e-12 and -1 <= g <= 1


def test_T38_r1_extra_fe_zero():
    _require()
    assert _load(RES / "E4ITR_r1_formula_runtime_audit.json")["R1_extra_FE"] == 0


def test_T39_T40_reward_logs_reconstruct_totals():
    _require()
    for r in _rows("E4ITR_training_reward_records.csv"):
        wg = float(r["weighted_global_reward"]); wfe = float(r["weighted_fe_penalty"])
        if r["arm"] == "R0_CONTROL":
            assert abs(float(r["actual_total_reward"]) - (wg + float(r["weighted_local_reward_R0"]) - wfe)) < 1e-9
        else:
            assert abs(float(r["actual_total_reward"]) - (wg + float(r["weighted_local_reward_R1"]) - wfe)) < 1e-9


def test_T41_T42_local_active_and_zero_active():
    _require()
    # trajectory: STATIONARY zero-active rows carry NOT_APPLICABLE ratio (never numeric 0 claimed "better")
    rows = _rows("E4ITR_reward_trajectory.csv")
    for r in rows:
        if r["n_active"] == "0" or int(r["n_active"]) == 0:
            assert r["global_local_active_ratio"] == "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"


def test_T44_T45_numerical_stability():
    _require()
    assert _load(RES / "E4ITR_numerical_stability.json")["numerical_stability"] == "PASS"
    assert _load(RES / "E4ITR_verdict.json")["NUMERICAL_STABILITY"] == "PASS"


def test_T46_T47_checkpoint_schedule_and_count():
    _require()
    ck = _load(RES / "E4ITR_checkpoint_manifest.json")
    assert ck["n_checkpoints"] == 60 and ck["complete"] is True
    counts = {}
    for c in ck["checkpoints"]:
        counts[(c["arm"], c["policy_seed"])] = counts.get((c["arm"], c["policy_seed"]), 0) + 1
    assert all(n == 6 for n in counts.values()) and len(counts) == 10


def test_T48_checkpoint_metadata_valid():
    _require()
    for c in _load(RES / "E4ITR_checkpoint_manifest.json")["checkpoints"]:
        assert c["sha256"] and c["transition_count"] >= 0


def test_T49_validation_seeds_range():
    _require()
    assert _load(RES / "E4ITR_execution_config.json")["val_seeds"] == [1561, 1580]


def test_T54_T55_all_checkpoints_validated():
    _require()
    rows = _rows("E4ITR_validation_all_checkpoints.csv")
    assert len(rows) == 60
    assert _load(RES / "E4ITR_verdict.json")["VALIDATION_CHECKPOINTS_COMPLETE"] == "60/60"


def test_T57_T58_checkpoint_selection_rule():
    _require()
    # selected checkpoint is the min VALIDATION_MACRO_MIGD_RESPONSE for its run (ties allowed)
    valrows = _rows("E4ITR_validation_all_checkpoints.csv")
    sel = _load(RES / "E4ITR_selected_checkpoints.json")["selected"]
    by_run = {}
    for r in valrows:
        by_run.setdefault((r["arm"], int(r["policy_seed"])), []).append(r)
    for (arm, p), rs in by_run.items():
        best = min(float(r["VALIDATION_MACRO_MIGD_RESPONSE"]) for r in rs)
        chosen = sel[f"{arm}_{p}"]["validation_macro_MIGD_response"]
        assert abs(chosen - best) < 1e-12


def test_T59_T60_T61_ten_selected_all_retained():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["R0_SELECTED_CHECKPOINTS"] == "5/5" and v["R1_SELECTED_CHECKPOINTS"] == "5/5"
    assert len(_load(RES / "E4ITR_selected_checkpoints.json")["selected"]) == 10


def test_T62_selected_may_differ_transition_count():
    _require()
    # not required to match; just assert field present and integer
    for v in _load(RES / "E4ITR_selected_checkpoints.json")["selected"].values():
        assert isinstance(v["transition_count"], int)


def test_T63_T64_validation_non_confirmatory():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["REWARD_CAUSAL_CLAIM"] == "NOT_MADE" and v["FINAL_EVALUATION"] == "NOT_RUN"


def test_T65_no_final_bootstrap():
    _require()
    assert _load(RES / "E4ITR_final_firewall.json")["no_final_bootstrap"] is True


def test_T66_reward_final_access_zero():
    _require()
    assert _load(RES / "E4ITR_verdict.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T67_T68_historical_verdicts_preserved():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert v["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T69_selected_checkpoint_hashes_recorded():
    _require()
    for v in _load(RES / "E4ITR_selected_checkpoints.json")["selected"].values():
        assert len(v["sha256"]) == 64


def test_T71_T72_T73_T74_no_retuning():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    assert v["HYPERPARAMETER_RETUNING"] == "NO" and v["REWARD_RETUNING"] == "NO"
    assert v["CHECKPOINT_SELECTION_CHANGED"] == "NO" and v["BEST_POLICY_SELECTION"] == "NO"


def test_T75_single_factor_integrity():
    _require()
    assert _load(RES / "E4ITR_single_factor_audit.json")["SINGLE_FACTOR_INTEGRITY"] == "PASS"
    assert _load(RES / "E4ITR_verdict.json")["SINGLE_FACTOR_INTEGRITY"] == "PASS"


def test_T76_mechanism_summary_present():
    _require()
    rows = _rows("E4ITR_reward_mechanism_summary.csv")
    arms = {r["arm"] for r in rows}
    assert arms == {"R0_CONTROL", "R1_TREATMENT"}
    v = _load(RES / "E4ITR_verdict.json")
    assert v["VALIDATION_REWARD_MECHANISM_DIRECTION"] in (
        "EXPECTED_DIRECTION", "PARTIAL_DIRECTION", "OPPOSITE_DIRECTION", "RAW_ENDPOINTS_ONLY")


def test_T77_mixed_action_diagnostics_present():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    for k in ["R0_MIXED_PREDICT_FRACTION", "R1_MIXED_PREDICT_FRACTION", "R0_MIXED_MULTI_ACTION_RATE",
              "R1_MIXED_MULTI_ACTION_RATE", "R0_MIXED_ACTION_ENTROPY", "R1_MIXED_ACTION_ENTROPY"]:
        assert isinstance(v[k], (int, float))


def test_T78_e4itr_complete_next_stage():
    _require()
    v = _load(RES / "E4ITR_verdict.json")
    if v["E4ITR_COMPLETE"] == "YES":
        assert v["NEXT_STAGE"] == "E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"


def test_T79_r0_source_byte_stable():
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "REGIONAL_QUALITY_GAIN" not in src and "reward_r1" not in src
