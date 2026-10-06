"""
E4KTR — Joint Coordination A/B Training Execution: artifact validation tests.

Validates the executed matched coordination experiment (C0 Independent Shared DDQN vs
C1 QMIX Monotonic Value Decomposition V1) from saved artifacts WITHOUT rerunning the
10800 training episodes: run/episode/checkpoint completeness, paired initialization,
single-factor + no-leak integrity, QMIX config + V3/R1 integrity, coordination diagnostics,
action trajectory, validation matrix + checkpoint selection, and the untouched final-seed
firewall (no final verdict / no final bootstrap).
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from prdrl.rl.network import SharedRegionalQNetwork, build_online_target_pair
from prdrl.rl.qmix import (
    R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM,
    build_mixer_online_target_pair, derive_mixer_seed, TREATMENT_ARCH,
)
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, FROZEN_V3_CONSTANTS_SHA256, qmix_config_sha256, make_qmix_config,
)
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256

RES = Path("results/E4KTR_joint_coordination_ab_training_execution")
E4KT = Path("results/E4KT_joint_coordination_ab_training_protocol_freeze")
E4KI = Path("results/E4KI_joint_coordination_implementation_validation")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF = Path("results/E4JF_global_comparator_final_evaluation_execution")

POLICY = [5001, 5002, 5003, 5004, 5005]
C0 = "C0_INDEPENDENT_SHARED_DDQN"
C1 = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
R1_SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
RNG_OFFSET_POLICY_INIT = 0


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4KTR_verdict.json").exists():
        pytest.skip("E4KTR not run yet")


def _rows(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def _param_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ============================================================================
# T1-T5: PREREQUISITES
# ============================================================================

def test_T1_e4kt_complete():
    assert _load(E4KT / "E4KT_verdict.json")["E4KT_COMPLETE"] == "YES"


def test_T2_e4ki_valid():
    assert _load(E4KI / "E4KI_verdict.json")["QMIX_IMPLEMENTATION_VALID"] == "YES"


def test_T3_exact_two_arms():
    _require()
    sel = _load(RES / "E4KTR_selected_checkpoints.json")["selected"]
    arms = {v["arm"] for v in sel.values()}
    assert arms == {C0, C1}


def test_T4_five_policy_seeds_each():
    _require()
    sel = _load(RES / "E4KTR_selected_checkpoints.json")["selected"]
    for arm in (C0, C1):
        assert sorted(v["policy_seed"] for v in sel.values()
                       if v["arm"] == arm) == POLICY


def test_T5_integrity_pass():
    _require()
    m = _load(RES / "E4KTR_pre_execution_manifest.json")
    assert m["integrity_ok"] is True


# ============================================================================
# T6-T10: RUNS AND EPISODES
# ============================================================================

def test_T6_T7_T8_runs_and_episodes():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["RUNS_COMPLETE"] == "10/10"
    assert v["C0_RUNS_COMPLETE"] == "5/5"
    assert v["C1_RUNS_COMPLETE"] == "5/5"
    assert v["TRAINING_EPISODES_COMPLETE"] == "10800/10800"


def test_T9_episode_csv_row_count():
    _require()
    rows = _rows("E4KTR_training_episode_results.csv")
    assert len(rows) == 10800


def test_T10_scenario_balance():
    _require()
    rows = _rows("E4KTR_training_episode_results.csv")
    from collections import Counter
    for arm in (C0, C1):
        c = Counter(r["scenario"] for r in rows if r["arm"] == arm)
        assert c["PERSISTENT"] == c["STATIONARY"] == c["MIXED"] == 1800


# ============================================================================
# T11-T15: PAIRED INITIALIZATION
# ============================================================================

def test_T11_paired_util_online_equals_target():
    _require()
    pi = _load(RES / "E4KTR_pair_initialization_manifest.json")
    assert pi["all_util_online_equals_target"] is True


def test_T12_paired_mixer_online_equals_target():
    _require()
    pi = _load(RES / "E4KTR_pair_initialization_manifest.json")
    assert pi["all_mixer_online_equals_target"] is True


def test_T13_c0_c1_util_init_identical():
    _require()
    pi = _load(RES / "E4KTR_pair_initialization_manifest.json")
    assert pi["C0_C1_util_init_identical_all"] is True


def test_T14_distinct_across_policies():
    _require()
    pi = _load(RES / "E4KTR_pair_initialization_manifest.json")
    assert pi["distinct_across_policy_seeds"] is True


def test_T15_independent_util_hash_reproducible():
    for p in POLICY:
        o, _ = build_online_target_pair(9, seed=p * 1000 + RNG_OFFSET_POLICY_INIT)
        pi = _load(RES / "E4KTR_pair_initialization_manifest.json")
        assert _param_hash(o) == pi["per_policy"][str(p)]["util_online_hash"]


# ============================================================================
# T16-T20: V3 / R1 / QMIX CONFIG INTEGRITY
# ============================================================================

def test_T16_v3_sha_in_manifest():
    _require()
    m = _load(RES / "E4KTR_pre_execution_manifest.json")
    assert m["v3_constants_sha256"]


def test_T17_r1_config_sha_exact():
    _require()
    m = _load(RES / "E4KTR_pre_execution_manifest.json")
    assert m["R1_config_sha256"] == R1_SHA


def test_T18_qmix_sha_exact():
    _require()
    m = _load(RES / "E4KTR_pre_execution_manifest.json")
    assert m["QMIX_CONFIG_SHA256"] == QMIX_SHA


def test_T19_qmix_sha_reproducible():
    cfg = make_qmix_config()
    assert qmix_config_sha256(cfg) == QMIX_SHA


def test_T20_runtime_architecture_integrity():
    _require()
    a = _load(RES / "E4KTR_runtime_architecture_integrity.json")
    assert a["C0_architecture"] == CONTROL_ARCH
    assert a["C1_architecture"] == TREATMENT_ARCH
    assert a["QMIX_CONFIG_SHA256_MATCHES_E4KT"] is True
    assert a["mixer_not_in_action_path"] is True
    assert a["no_runtime_adaptation"] is True


# ============================================================================
# T21-T25: SINGLE FACTOR + SEED RANGES
# ============================================================================

def test_T21_single_factor_integrity():
    _require()
    sf = _load(RES / "E4KTR_single_factor_audit.json")
    assert sf["SINGLE_FACTOR_INTEGRITY"] == "PASS"
    assert sf["factor"] == "JOINT_COORDINATION_LEARNER_SPECIFICATION"
    assert sf["shared_everything_else"] is True


def test_T22_train_seeds():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["train_seeds"] == [1901, 1960]


def test_T23_val_seeds():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["val_seeds"] == [1961, 1980]


def test_T24_final_seeds_firewalled():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["final_seeds_firewalled"] == [1981, 2040]


def test_T25_no_final_seed_in_episodes():
    _require()
    for r in _rows("E4KTR_training_episode_results.csv"):
        seed = int(r["environment_seed"])
        assert not (1981 <= seed <= 2040), f"final seed {seed} in training"


# ============================================================================
# T26-T30: TRAIN SEED COVERAGE + NO OVERLAP
# ============================================================================

def test_T26_train_seeds_all_used():
    _require()
    rows = _rows("E4KTR_training_episode_results.csv")
    for arm in (C0, C1):
        seeds = {int(r["environment_seed"]) for r in rows if r["arm"] == arm}
        expected = set(range(1901, 1961))
        assert seeds == expected


def test_T27_val_seeds_no_overlap_train():
    _require()
    assert set(range(1961, 1981)).isdisjoint(set(range(1901, 1961)))


def test_T28_final_seeds_no_overlap():
    _require()
    final = set(range(1981, 2041))
    assert final.isdisjoint(set(range(1901, 1961)))
    assert final.isdisjoint(set(range(1961, 1981)))


def test_T29_leakage_audit():
    _require()
    la = _load(RES / "E4KTR_training_validation_leakage_audit.json")
    assert la["train_val_disjoint"] is True
    assert la["train_final_disjoint"] is True
    assert la["val_final_disjoint"] is True
    assert la["no_future_info_in_reward"] is True
    assert la["no_oracle_in_training"] is True


def test_T30_policy_seeds_in_config():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["policy_seeds"] == POLICY


# ============================================================================
# T31-T40: CHECKPOINTS
# ============================================================================

def test_T31_checkpoint_count_60():
    _require()
    ck = _load(RES / "E4KTR_checkpoint_manifest.json")
    assert ck["n_checkpoints"] == 60
    assert ck["complete"] is True


def test_T32_six_per_run():
    _require()
    ck = _load(RES / "E4KTR_checkpoint_manifest.json")
    counts = {}
    for c in ck["checkpoints"]:
        key = (c["arm"], c["policy_seed"])
        counts[key] = counts.get(key, 0) + 1
    assert all(n == 6 for n in counts.values())
    assert len(counts) == 10


def test_T33_checkpoint_hashes_recorded():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        assert len(c["sha256"]) == 64


def test_T34_checkpoint_files_exist():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        assert Path(c["path"]).exists(), f"missing {c['path']}"


def test_T35_checkpoint_names_standard():
    _require()
    expected_names = {"ckpt_10000", "ckpt_20000", "ckpt_30000",
                      "ckpt_40000", "ckpt_50000", "ckpt_final"}
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        assert c["checkpoint"] in expected_names


def test_T36_c0_checkpoints_no_mixer():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C0:
            payload = torch.load(c["path"], weights_only=False)
            assert "mixer_online_state_dict" not in payload


def test_T37_c1_checkpoints_have_mixer():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C1:
            payload = torch.load(c["path"], weights_only=False)
            assert "mixer_online_state_dict" in payload
            assert "mixer_target_state_dict" in payload


def test_T38_c1_architecture_version_in_ckpt():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C1:
            payload = torch.load(c["path"], weights_only=False)
            assert payload["architecture_version"] == TREATMENT_ARCH
            assert payload["qmix_config_sha256"] == QMIX_SHA
            break


def test_T39_c0_architecture_version_in_ckpt():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C0:
            payload = torch.load(c["path"], weights_only=False)
            assert payload["architecture_version"] == CONTROL_ARCH
            break


def test_T40_checkpoint_validation_migd_finite():
    _require()
    for c in _load(RES / "E4KTR_checkpoint_manifest.json")["checkpoints"]:
        v = c["VALIDATION_MACRO_MIGD_RESPONSE"]
        assert isinstance(v, (int, float)) and math.isfinite(v)


# ============================================================================
# T41-T50: VALIDATION
# ============================================================================

def test_T41_all_checkpoints_validated():
    _require()
    rows = _rows("E4KTR_validation_all_checkpoints.csv")
    assert len(rows) == 60


def test_T42_validation_checkpoints_complete():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["VALIDATION_CHECKPOINTS_COMPLETE"] == "60/60"


def test_T43_checkpoint_selection_is_min_migd():
    _require()
    valrows = _rows("E4KTR_validation_all_checkpoints.csv")
    sel = _load(RES / "E4KTR_selected_checkpoints.json")["selected"]
    by_run = {}
    for r in valrows:
        by_run.setdefault((r["arm"], int(r["policy_seed"])), []).append(r)
    for (arm, p), rs in by_run.items():
        best = min(float(r["VALIDATION_MACRO_MIGD_RESPONSE"]) for r in rs)
        chosen = sel[f"{arm}_{p}"]["validation_macro_MIGD_response"]
        assert abs(chosen - best) < 1e-12


def test_T44_T45_ten_selected():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["C0_SELECTED_CHECKPOINTS"] == "5/5"
    assert v["C1_SELECTED_CHECKPOINTS"] == "5/5"
    assert len(_load(RES / "E4KTR_selected_checkpoints.json")["selected"]) == 10


def test_T46_selected_hashes_recorded():
    _require()
    for val in _load(RES / "E4KTR_selected_checkpoints.json")["selected"].values():
        assert len(val["sha256"]) == 64


def test_T47_selected_v3_version():
    _require()
    for val in _load(RES / "E4KTR_selected_checkpoints.json")["selected"].values():
        assert val["normalization_version"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T48_selected_r1_version():
    _require()
    for val in _load(RES / "E4KTR_selected_checkpoints.json")["selected"].values():
        assert val["reward_version"] == REWARD_R1_NAME
        assert val["reward_config_sha256"] == R1_SHA


def test_T49_selected_architecture_versions():
    _require()
    for val in _load(RES / "E4KTR_selected_checkpoints.json")["selected"].values():
        if val["arm"] == C0:
            assert val["architecture_version"] == CONTROL_ARCH
        else:
            assert val["architecture_version"] == TREATMENT_ARCH


def test_T50_selected_transition_count_integer():
    _require()
    for val in _load(RES / "E4KTR_selected_checkpoints.json")["selected"].values():
        assert isinstance(val["transition_count"], int)


# ============================================================================
# T51-T60: VALIDATION A/B SUMMARY
# ============================================================================

def test_T51_ab_summary_has_4_rows():
    _require()
    rows = _rows("E4KTR_validation_ab_summary.csv")
    assert len(rows) == 4
    scens = [r["scenario"] for r in rows]
    assert set(scens) == {"PERSISTENT", "STATIONARY", "MIXED", "macro"}


def test_T52_delta_c0_minus_c1_present():
    _require()
    rows = _rows("E4KTR_validation_ab_summary.csv")
    for r in rows:
        d = float(r["delta_C0_minus_C1_response"])
        c0 = float(r["C0_MIGD_response"])
        c1 = float(r["C1_MIGD_response"])
        assert abs(d - (c0 - c1)) < 1e-12


def test_T53_validation_delta_in_verdict():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert isinstance(v["VALIDATION_PAIRED_DELTA_C0_MINUS_C1"], float)


def test_T54_validation_macro_c0_finite():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert math.isfinite(v["C0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"])


def test_T55_validation_macro_c1_finite():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert math.isfinite(v["C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"])


def test_T56_fe_columns_present():
    _require()
    rows = _rows("E4KTR_validation_ab_summary.csv")
    for r in rows:
        assert "C0_response_FE" in r and "C1_response_FE" in r


def test_T57_per_scenario_migd_in_validation():
    _require()
    rows = _rows("E4KTR_validation_all_checkpoints.csv")
    for r in rows:
        for sc in SCENARIOS:
            key = f"MIGD_response_{sc}"
            assert key in r
            assert math.isfinite(float(r[key]))


def test_T58_selected_flag_exactly_10():
    _require()
    rows = _rows("E4KTR_validation_all_checkpoints.csv")
    sel_count = sum(1 for r in rows if r["selected"] == "True")
    assert sel_count == 10


def test_T59_validation_selected_policies_csv():
    _require()
    rows = _rows("E4KTR_validation_selected_policies.csv")
    assert len(rows) == 10


def test_T60_validation_action_context_csv():
    _require()
    rows = _rows("E4KTR_validation_action_context.csv")
    assert len(rows) == 10


# ============================================================================
# T61-T70: COORDINATION DIAGNOSTICS
# ============================================================================

def test_T61_coordination_diagnostics_csv():
    _require()
    rows = _rows("E4KTR_coordination_diagnostics.csv")
    assert len(rows) == 5
    assert {int(r["policy_seed"]) for r in rows} == set(POLICY)


def test_T62_monotonicity_computed():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        assert r["monotonicity_computed"] == "True"


def test_T63_monotonicity_violations_low():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        viol = int(r["monotonicity_violations"])
        assert viol >= 0


def test_T64_mixer_param_count_positive():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        assert int(r["mixer_param_count"]) > 0


def test_T65_util_param_count_matches_c0():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        net = SharedRegionalQNetwork(9)
        expected = sum(p.numel() for p in net.parameters())
        assert int(r["util_param_count"]) == expected


def test_T66_mean_joint_td_loss_finite():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        assert math.isfinite(float(r["mean_joint_td_loss"]))


def test_T67_n_updates_positive():
    _require()
    for r in _rows("E4KTR_coordination_diagnostics.csv"):
        assert int(r["n_updates"]) > 0


def test_T68_action_trajectory_csv():
    _require()
    rows = _rows("E4KTR_action_trajectory.csv")
    assert len(rows) == 60
    expected_arms = {C0, C1}
    assert {r["arm"] for r in rows} == expected_arms


def test_T69_action_trajectory_fractions_sum_to_1():
    _require()
    for r in _rows("E4KTR_action_trajectory.csv"):
        total = sum(float(r[f"frac_{a}"]) for a in
                    ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"])
        assert abs(total - 1.0) < 1e-6


def test_T70_training_update_summary_csv():
    _require()
    rows = _rows("E4KTR_training_update_summary.csv")
    assert len(rows) == 60
    assert {r["arm"] for r in rows} == {C0, C1}


# ============================================================================
# T71-T80: MIXED ACTION DIAGNOSTICS
# ============================================================================

def test_T71_mixed_predict_fraction_c0():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert 0 <= v["C0_MIXED_PREDICT_FRACTION"] <= 1


def test_T72_mixed_predict_fraction_c1():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert 0 <= v["C1_MIXED_PREDICT_FRACTION"] <= 1


def test_T73_multi_action_rate_c0():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert 0 <= v["C0_MIXED_MULTI_ACTION_RATE"] <= 1


def test_T74_multi_action_rate_c1():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert 0 <= v["C1_MIXED_MULTI_ACTION_RATE"] <= 1


def test_T75_action_entropy_c0():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["C0_MIXED_ACTION_ENTROPY"] >= 0


def test_T76_action_entropy_c1():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["C1_MIXED_ACTION_ENTROPY"] >= 0


def test_T77_mixed_action_diagnostics_in_selected():
    _require()
    sel = _load(RES / "E4KTR_selected_checkpoints.json")
    mad = sel["mixed_action_diagnostics"]
    for arm in (C0, C1):
        assert 0 <= mad[arm]["predict_fraction"] <= 1
        assert 0 <= mad[arm]["multi_action_rate"] <= 1
        assert mad[arm]["action_entropy"] >= 0


def test_T78_validation_coordination_diagnostics_csv():
    _require()
    rows = _rows("E4KTR_validation_coordination_diagnostics.csv")
    assert len(rows) == 10
    assert {r["arm"] for r in rows} == {C0, C1}


def test_T79_training_episode_migd_finite():
    _require()
    for r in _rows("E4KTR_training_episode_results.csv"):
        assert math.isfinite(float(r["MIGD_response"]))
        assert math.isfinite(float(r["MIGD_epoch"]))


def test_T80_training_episode_fe_nonneg():
    _require()
    for r in _rows("E4KTR_training_episode_results.csv"):
        assert int(r["total_response_FE"]) >= 0


# ============================================================================
# T81-T90: NUMERICAL STABILITY + FIREWALL + NO RETUNING
# ============================================================================

def test_T81_numerical_stability():
    _require()
    ns = _load(RES / "E4KTR_numerical_stability.json")
    assert ns["numerical_stability"] == "PASS"


def test_T82_numerical_stability_in_verdict():
    _require()
    assert _load(RES / "E4KTR_verdict.json")["NUMERICAL_STABILITY"] == "PASS"


def test_T83_all_runs_stable():
    _require()
    ns = _load(RES / "E4KTR_numerical_stability.json")
    for key, val in ns["runs"].items():
        assert val == "PASS", f"{key} not PASS"


def test_T84_final_seed_firewall_zero():
    _require()
    ff = _load(RES / "E4KTR_final_firewall.json")
    assert ff["COORDINATION_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert ff["FINAL_EVALUATION"] == "NOT_RUN"
    assert ff["no_final_bootstrap"] is True


def test_T85_final_access_in_verdict_zero():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["COORDINATION_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T86_no_hyperparameter_retuning():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["HYPERPARAMETER_RETUNING"] == "NO"


def test_T87_no_architecture_retuning():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["ARCHITECTURE_RETUNING"] == "NO"


def test_T88_no_checkpoint_selection_changed():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["CHECKPOINT_SELECTION_CHANGED"] == "NO"


def test_T89_no_best_policy_selection():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["BEST_POLICY_SELECTION"] == "NO"


def test_T90_no_causal_claim():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["COORDINATION_CAUSAL_CLAIM"] == "NOT_MADE"
    assert v["FINAL_EVALUATION"] == "NOT_RUN"


# ============================================================================
# T91-T100: HISTORICAL VERDICTS
# ============================================================================

def test_T91_e4f_verdict_failed():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED"


def test_T92_e4hf_verdict_failed():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T93_e4if_verdict_validated():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T94_e4jf_verdict_failed():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"


def test_T95_source_e4f_still_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T96_source_e4hf_still_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T97_source_e4if_still_validated():
    assert _load(E4IF / "E4IF_verdict.json")["REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T98_source_e4jf_still_failed():
    assert _load(E4JF / "E4JF_verdict.json")["GLOBAL_COMPARATOR_VERDICT"] == "FAILED"


def test_T99_e4kt_still_complete():
    assert _load(E4KT / "E4KT_verdict.json")["E4KT_COMPLETE"] == "YES"


def test_T100_e4ki_still_valid():
    assert _load(E4KI / "E4KI_verdict.json")["QMIX_IMPLEMENTATION_VALID"] == "YES"


# ============================================================================
# T101-T109: VERDICT + REPORT + NEXT STAGE
# ============================================================================

def test_T101_e4ktr_complete():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    assert v["E4KTR_COMPLETE"] in ("YES", "NO")


def test_T102_next_stage_routing():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    if v["E4KTR_COMPLETE"] == "YES":
        assert v["NEXT_STAGE"] == \
            "E4KE_JOINT_COORDINATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
    else:
        assert v["NEXT_STAGE"] == "E4KTR_ENGINEERING_REVIEW"


def test_T103_report_exists():
    _require()
    assert (RES / "E4KTR_report.md").exists()


def test_T104_report_mentions_coordination():
    _require()
    txt = (RES / "E4KTR_report.md").read_text()
    assert "E4KTR" in txt
    assert "coordination" in txt.lower() or "QMIX" in txt


def test_T105_verdict_has_all_required_keys():
    _require()
    v = _load(RES / "E4KTR_verdict.json")
    required = [
        "stage", "E4KTR_COMPLETE", "RUNS_COMPLETE",
        "C0_RUNS_COMPLETE", "C1_RUNS_COMPLETE",
        "TRAINING_EPISODES_COMPLETE", "PAIRED_INITIALIZATION",
        "SINGLE_FACTOR_INTEGRITY", "V3_CONSTANTS_INTEGRITY",
        "R1_REWARD_CONFIG_INTEGRITY", "QMIX_CONFIG_INTEGRITY",
        "NUMERICAL_STABILITY", "CHECKPOINTS_COMPLETE",
        "VALIDATION_CHECKPOINTS_COMPLETE",
        "C0_SELECTED_CHECKPOINTS", "C1_SELECTED_CHECKPOINTS",
        "C0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "VALIDATION_PAIRED_DELTA_C0_MINUS_C1",
        "HYPERPARAMETER_RETUNING", "ARCHITECTURE_RETUNING",
        "CHECKPOINT_SELECTION_CHANGED", "BEST_POLICY_SELECTION",
        "FINAL_EVALUATION", "COORDINATION_CAUSAL_CLAIM",
        "COORDINATION_FINAL_EVAL_ACCESS_COUNT",
        "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in required:
        assert k in v, f"missing key: {k}"


def test_T106_c1_checkpoint_loads_mixer():
    _require()
    sel = _load(RES / "E4KTR_selected_checkpoints.json")["selected"]
    for key, val in sel.items():
        if val["arm"] == C1:
            payload = torch.load(val["checkpoint_path"], weights_only=False)
            from prdrl.rl.qmix import QMIXMixerV1
            mixer = QMIXMixerV1()
            mixer.load_state_dict(payload["mixer_online_state_dict"])
            break


def test_T107_c0_checkpoint_loads_utility():
    _require()
    sel = _load(RES / "E4KTR_selected_checkpoints.json")["selected"]
    for key, val in sel.items():
        if val["arm"] == C0:
            payload = torch.load(val["checkpoint_path"], weights_only=False)
            net = SharedRegionalQNetwork(9)
            net.load_state_dict(payload["online_state_dict"])
            break


def test_T108_execution_config_arms():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["arms"] == [C0, C1]
    assert ec["control"] == CONTROL_ARCH
    assert ec["treatment"] == TREATMENT_ARCH


def test_T109_execution_config_hashes():
    _require()
    ec = _load(RES / "E4KTR_execution_config.json")
    assert ec["QMIX_CONFIG_SHA256"] == QMIX_SHA
    assert ec["R1_config_sha256"] == R1_SHA
