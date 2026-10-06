"""
E4MTR — Credit Preservation A/B Training Execution: artifact validation tests.

Validates the executed matched credit-preservation experiment (C1 QMIX Joint TD vs
C2 QMIX Composite 0.5*L_joint+0.5*L_regional) from saved artifacts WITHOUT rerunning
the 10800 training episodes: run/episode/checkpoint completeness, paired initialization,
single-factor + no-leak integrity, QMIX config + C2 credit config + V3/R1 integrity,
credit diagnostics (loss dispersion, gradient diversity, monotonicity), action trajectory,
validation matrix + checkpoint selection, and the untouched final-seed firewall (no final
verdict / no final bootstrap).

NO recursive subprocess regression test (per E4MTR spec rule).
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
    CONTROL_ARCH, C2_CREDIT_ARCH, FROZEN_V3_CONSTANTS_SHA256,
    qmix_config_sha256, make_qmix_config,
    c2_credit_config_sha256, make_c2_credit_config,
    JOINT_WEIGHT, REGIONAL_WEIGHT,
)
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256

RES = Path("results/E4MTR_credit_preservation_ab_training_execution")
E4MT = Path("results/E4MT_credit_preservation_ab_training_protocol_freeze")
E4MI = Path("results/E4MI_joint_credit_preservation_implementation_validation")
E4KF = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF = Path("results/E4IF_reward_ab_final_evaluation_execution")

POLICY = [6001, 6002, 6003, 6004, 6005]
C1 = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2 = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
C2_CREDIT_SHA = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"
R1_SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
RNG_OFFSET_POLICY_INIT = 0


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4MTR_verdict.json").exists():
        pytest.skip("E4MTR not run yet")


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

def test_T1_e4mt_complete():
    assert _load(E4MT / "E4MT_verdict.json")["E4MT_COMPLETE"] == "YES"


def test_T2_e4mi_valid():
    assert _load(E4MI / "E4MI_verdict.json")["CREDIT_PRESERVATION_IMPLEMENTATION_VALID"] == "YES"


def test_T3_exact_two_arms():
    _require()
    sel = _load(RES / "E4MTR_selected_checkpoints.json")["selected"]
    arms = {v["arm"] for v in sel.values()}
    assert arms == {C1, C2}


def test_T4_five_policy_seeds_each():
    _require()
    sel = _load(RES / "E4MTR_selected_checkpoints.json")["selected"]
    for arm in (C1, C2):
        assert sorted(v["policy_seed"] for v in sel.values()
                       if v["arm"] == arm) == POLICY


def test_T5_integrity_pass():
    _require()
    m = _load(RES / "E4MTR_pre_execution_manifest.json")
    assert m["integrity_ok"] is True


# ============================================================================
# T6-T10: RUNS AND EPISODES
# ============================================================================

def test_T6_T7_T8_runs_and_episodes():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["RUNS_COMPLETE"] == "10/10"
    assert v["C1_RUNS_COMPLETE"] == "5/5"
    assert v["C2_RUNS_COMPLETE"] == "5/5"
    assert v["TRAINING_EPISODES_COMPLETE"] == "10800/10800"


def test_T9_episode_csv_row_count():
    _require()
    rows = _rows("E4MTR_training_episode_results.csv")
    assert len(rows) == 10800


def test_T10_scenario_balance():
    _require()
    rows = _rows("E4MTR_training_episode_results.csv")
    from collections import Counter
    for arm in (C1, C2):
        c = Counter(r["scenario"] for r in rows if r["arm"] == arm)
        assert c["PERSISTENT"] == c["STATIONARY"] == c["MIXED"] == 1800


# ============================================================================
# T11-T15: PAIRED INITIALIZATION
# ============================================================================

def test_T11_paired_util_online_equals_target():
    _require()
    pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
    assert pi["all_util_online_equals_target"] is True


def test_T12_paired_mixer_online_equals_target():
    _require()
    pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
    assert pi["all_mixer_online_equals_target"] is True


def test_T13_c1_c2_util_init_identical():
    _require()
    pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
    assert pi["C1_C2_util_init_identical_all"] is True


def test_T14_c1_c2_mixer_init_identical():
    _require()
    pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
    assert pi["C1_C2_mixer_init_identical_all"] is True


def test_T15_distinct_across_policies():
    _require()
    pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
    assert pi["distinct_across_policy_seeds"] is True


def test_T16_independent_util_hash_reproducible():
    for p in POLICY:
        o, _ = build_online_target_pair(9, seed=p * 1000 + RNG_OFFSET_POLICY_INIT)
        pi = _load(RES / "E4MTR_pair_initialization_manifest.json")
        assert _param_hash(o) == pi["per_policy"][str(p)]["util_online_hash"]


# ============================================================================
# T17-T22: V3 / R1 / QMIX / C2 CREDIT CONFIG INTEGRITY
# ============================================================================

def test_T17_v3_sha_in_manifest():
    _require()
    m = _load(RES / "E4MTR_pre_execution_manifest.json")
    assert m["v3_constants_sha256"]


def test_T18_r1_config_sha_exact():
    _require()
    m = _load(RES / "E4MTR_pre_execution_manifest.json")
    assert m["R1_config_sha256"] == R1_SHA


def test_T19_qmix_sha_exact():
    _require()
    m = _load(RES / "E4MTR_pre_execution_manifest.json")
    assert m["QMIX_CONFIG_SHA256"] == QMIX_SHA


def test_T20_qmix_sha_reproducible():
    cfg = make_qmix_config()
    assert qmix_config_sha256(cfg) == QMIX_SHA


def test_T21_c2_credit_sha_exact():
    _require()
    m = _load(RES / "E4MTR_pre_execution_manifest.json")
    assert m["C2_CREDIT_CONFIG_SHA256"] == C2_CREDIT_SHA


def test_T22_c2_credit_sha_reproducible():
    cfg = make_c2_credit_config()
    assert c2_credit_config_sha256(cfg) == C2_CREDIT_SHA


# ============================================================================
# T23-T27: RUNTIME ARCHITECTURE + SINGLE FACTOR
# ============================================================================

def test_T23_runtime_architecture_integrity():
    _require()
    a = _load(RES / "E4MTR_runtime_architecture_integrity.json")
    assert a["C1_architecture"] == TREATMENT_ARCH
    assert a["C2_architecture"] == C2_CREDIT_ARCH
    assert a["QMIX_CONFIG_SHA256_MATCHES_E4MT"] is True
    assert a["C2_CREDIT_CONFIG_SHA256_MATCHES_E4MT"] is True
    assert a["mixer_not_in_action_path"] is True
    assert a["no_runtime_adaptation"] is True
    assert a["C2_mixer_gradient_from_joint_only"] is True
    assert a["C2_joint_weight"] == JOINT_WEIGHT
    assert a["C2_regional_weight"] == REGIONAL_WEIGHT


def test_T24_single_factor_integrity():
    _require()
    sf = _load(RES / "E4MTR_single_factor_audit.json")
    assert sf["SINGLE_FACTOR_INTEGRITY"] == "PASS"
    assert sf["factor"] == "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX"
    assert sf["shared_everything_else"] is True


def test_T25_composite_weight_05():
    assert JOINT_WEIGHT == 0.5
    assert REGIONAL_WEIGHT == 0.5


# ============================================================================
# T26-T32: SEED RANGES
# ============================================================================

def test_T26_train_seeds():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["train_seeds"] == [2101, 2160]


def test_T27_val_seeds():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["val_seeds"] == [2161, 2180]


def test_T28_final_seeds_firewalled():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["final_seeds_firewalled"] == [2181, 2240]


def test_T29_no_final_seed_in_episodes():
    _require()
    for r in _rows("E4MTR_training_episode_results.csv"):
        seed = int(r["environment_seed"])
        assert not (2181 <= seed <= 2240), f"final seed {seed} in training"


def test_T30_train_seeds_all_used():
    _require()
    rows = _rows("E4MTR_training_episode_results.csv")
    for arm in (C1, C2):
        seeds = {int(r["environment_seed"]) for r in rows if r["arm"] == arm}
        expected = set(range(2101, 2161))
        assert seeds == expected


def test_T31_seed_disjointness():
    _require()
    assert set(range(2101, 2161)).isdisjoint(set(range(2161, 2181)))
    assert set(range(2181, 2241)).isdisjoint(set(range(2101, 2161)))
    assert set(range(2181, 2241)).isdisjoint(set(range(2161, 2181)))


def test_T32_leakage_audit():
    _require()
    la = _load(RES / "E4MTR_training_validation_leakage_audit.json")
    assert la["train_val_disjoint"] is True
    assert la["train_final_disjoint"] is True
    assert la["val_final_disjoint"] is True
    assert la["no_future_info_in_reward"] is True
    assert la["no_oracle_in_training"] is True


def test_T33_policy_seeds_in_config():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["policy_seeds"] == POLICY


# ============================================================================
# T34-T43: CHECKPOINTS
# ============================================================================

def test_T34_checkpoint_count_60():
    _require()
    ck = _load(RES / "E4MTR_checkpoint_manifest.json")
    assert ck["n_checkpoints"] == 60
    assert ck["complete"] is True


def test_T35_six_per_run():
    _require()
    ck = _load(RES / "E4MTR_checkpoint_manifest.json")
    counts = {}
    for c in ck["checkpoints"]:
        key = (c["arm"], c["policy_seed"])
        counts[key] = counts.get(key, 0) + 1
    assert all(n == 6 for n in counts.values())
    assert len(counts) == 10


def test_T36_checkpoint_hashes_recorded():
    _require()
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        assert len(c["sha256"]) == 64


def test_T37_checkpoint_files_exist():
    _require()
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        assert Path(c["path"]).exists(), f"missing {c['path']}"


def test_T38_checkpoint_names_standard():
    _require()
    expected_names = {"ckpt_10000", "ckpt_20000", "ckpt_30000",
                      "ckpt_40000", "ckpt_50000", "ckpt_final"}
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        assert c["checkpoint"] in expected_names


def test_T39_c1_checkpoints_have_mixer():
    _require()
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C1:
            payload = torch.load(c["path"], weights_only=False)
            assert "mixer_online_state_dict" in payload
            assert "mixer_target_state_dict" in payload
            assert payload["architecture_version"] == TREATMENT_ARCH
            assert payload["qmix_config_sha256"] == QMIX_SHA
            break


def test_T40_c2_checkpoints_have_mixer_and_credit():
    _require()
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        if c["arm"] == C2:
            payload = torch.load(c["path"], weights_only=False)
            assert "mixer_online_state_dict" in payload
            assert "mixer_target_state_dict" in payload
            assert payload["architecture_version"] == C2_CREDIT_ARCH
            assert payload["qmix_config_sha256"] == QMIX_SHA
            assert payload["c2_credit_config_sha256"] == C2_CREDIT_SHA
            break


def test_T41_checkpoint_validation_migd_finite():
    _require()
    for c in _load(RES / "E4MTR_checkpoint_manifest.json")["checkpoints"]:
        v = c["VALIDATION_MACRO_MIGD_RESPONSE"]
        assert isinstance(v, (int, float)) and math.isfinite(v)


# ============================================================================
# T42-T52: VALIDATION
# ============================================================================

def test_T42_all_checkpoints_validated():
    _require()
    rows = _rows("E4MTR_validation_all_checkpoints.csv")
    assert len(rows) == 60


def test_T43_validation_checkpoints_complete():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["VALIDATION_CHECKPOINTS_COMPLETE"] == "60/60"


def test_T44_checkpoint_selection_is_min_migd():
    _require()
    valrows = _rows("E4MTR_validation_all_checkpoints.csv")
    sel = _load(RES / "E4MTR_selected_checkpoints.json")["selected"]
    by_run = {}
    for r in valrows:
        by_run.setdefault((r["arm"], int(r["policy_seed"])), []).append(r)
    for (arm, p), rs in by_run.items():
        best = min(float(r["VALIDATION_MACRO_MIGD_RESPONSE"]) for r in rs)
        chosen = sel[f"{arm}_{p}"]["validation_macro_MIGD_response"]
        assert abs(chosen - best) < 1e-12


def test_T45_ten_selected():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["C1_SELECTED_CHECKPOINTS"] == "5/5"
    assert v["C2_SELECTED_CHECKPOINTS"] == "5/5"
    assert len(_load(RES / "E4MTR_selected_checkpoints.json")["selected"]) == 10


def test_T46_selected_hashes_recorded():
    _require()
    for val in _load(RES / "E4MTR_selected_checkpoints.json")["selected"].values():
        assert len(val["sha256"]) == 64


def test_T47_selected_v3_version():
    _require()
    for val in _load(RES / "E4MTR_selected_checkpoints.json")["selected"].values():
        assert val["normalization_version"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T48_selected_r1_version():
    _require()
    for val in _load(RES / "E4MTR_selected_checkpoints.json")["selected"].values():
        assert val["reward_version"] == REWARD_R1_NAME
        assert val["reward_config_sha256"] == R1_SHA


def test_T49_selected_architecture_versions():
    _require()
    for val in _load(RES / "E4MTR_selected_checkpoints.json")["selected"].values():
        if val["arm"] == C1:
            assert val["architecture_version"] == TREATMENT_ARCH
        else:
            assert val["architecture_version"] == C2_CREDIT_ARCH


def test_T50_selected_transition_count_integer():
    _require()
    for val in _load(RES / "E4MTR_selected_checkpoints.json")["selected"].values():
        assert isinstance(val["transition_count"], int)


def test_T51_per_scenario_migd_in_validation():
    _require()
    rows = _rows("E4MTR_validation_all_checkpoints.csv")
    for r in rows:
        for sc in SCENARIOS:
            key = f"MIGD_response_{sc}"
            assert key in r
            assert math.isfinite(float(r[key]))


def test_T52_selected_flag_exactly_10():
    _require()
    rows = _rows("E4MTR_validation_all_checkpoints.csv")
    sel_count = sum(1 for r in rows if r["selected"] == "True")
    assert sel_count == 10


# ============================================================================
# T53-T58: VALIDATION A/B SUMMARY
# ============================================================================

def test_T53_ab_summary_has_4_rows():
    _require()
    rows = _rows("E4MTR_validation_ab_summary.csv")
    assert len(rows) == 4
    scens = [r["scenario"] for r in rows]
    assert set(scens) == {"PERSISTENT", "STATIONARY", "MIXED", "macro"}


def test_T54_delta_c1_minus_c2_present():
    _require()
    rows = _rows("E4MTR_validation_ab_summary.csv")
    for r in rows:
        d = float(r["delta_C1_minus_C2_response"])
        c1_val = float(r["C1_MIGD_response"])
        c2_val = float(r["C2_MIGD_response"])
        assert abs(d - (c1_val - c2_val)) < 1e-12


def test_T55_validation_delta_in_verdict():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert isinstance(v["VALIDATION_PAIRED_DELTA_C1_MINUS_C2"], float)


def test_T56_validation_macro_c1_finite():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert math.isfinite(v["C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"])


def test_T57_validation_macro_c2_finite():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert math.isfinite(v["C2_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"])


def test_T58_fe_columns_present():
    _require()
    rows = _rows("E4MTR_validation_ab_summary.csv")
    for r in rows:
        assert "C1_response_FE" in r and "C2_response_FE" in r


# ============================================================================
# T59-T70: CREDIT DIAGNOSTICS
# ============================================================================

def test_T59_credit_diagnostics_csv():
    _require()
    rows = _rows("E4MTR_credit_diagnostics.csv")
    assert len(rows) == 10
    arms = {r["arm"] for r in rows}
    assert arms == {C1, C2}
    assert {int(r["policy_seed"]) for r in rows
            if r["arm"] == C1} == set(POLICY)
    assert {int(r["policy_seed"]) for r in rows
            if r["arm"] == C2} == set(POLICY)


def test_T60_monotonicity_computed():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        assert r["monotonicity_computed"] == "True"


def test_T61_monotonicity_violations_nonneg():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        viol = int(r["monotonicity_violations"])
        assert viol >= 0


def test_T62_mixer_param_count_positive():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        assert int(r["mixer_param_count"]) > 0


def test_T63_util_param_count_matches():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        net = SharedRegionalQNetwork(9)
        expected = sum(p.numel() for p in net.parameters())
        assert int(r["util_param_count"]) == expected


def test_T64_c1_mean_joint_td_loss_finite():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        if r["arm"] == C1:
            assert math.isfinite(float(r["mean_joint_td_loss"]))


def test_T65_c2_mean_composite_loss_finite():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        if r["arm"] == C2:
            assert math.isfinite(float(r["mean_composite_loss"]))
            assert math.isfinite(float(r["mean_joint_td_loss"]))
            assert math.isfinite(float(r["mean_regional_td_loss"]))


def test_T66_n_updates_positive():
    _require()
    for r in _rows("E4MTR_credit_diagnostics.csv"):
        assert int(r["n_updates"]) > 0


def test_T67_gradient_diversity_csv():
    _require()
    rows = _rows("E4MTR_gradient_diversity.csv")
    assert len(rows) == 5
    assert {int(r["policy_seed"]) for r in rows} == set(POLICY)
    for r in rows:
        assert math.isfinite(float(r["mean_norm_joint"]))
        assert math.isfinite(float(r["mean_norm_regional"]))
        assert math.isfinite(float(r["mean_norm_composite"]))
        assert math.isfinite(float(r["mean_cosine_joint_regional"]))
        assert int(r["n_samples"]) > 0


def test_T68_loss_dispersion_csv():
    _require()
    rows = _rows("E4MTR_loss_dispersion.csv")
    assert len(rows) == 5
    assert {int(r["policy_seed"]) for r in rows} == set(POLICY)
    for r in rows:
        assert float(r["within_event_std_joint_loss"]) >= 0
        assert float(r["within_event_std_regional_loss"]) >= 0
        assert float(r["within_event_std_composite_loss"]) >= 0


def test_T69_action_trajectory_csv():
    _require()
    rows = _rows("E4MTR_action_trajectory.csv")
    assert len(rows) == 60
    assert {r["arm"] for r in rows} == {C1, C2}


def test_T70_action_trajectory_fractions_sum_to_1():
    _require()
    for r in _rows("E4MTR_action_trajectory.csv"):
        total = sum(float(r[f"frac_{a}"]) for a in
                    ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"])
        assert abs(total - 1.0) < 1e-6


# ============================================================================
# T71-T78: MIXED ACTION DIAGNOSTICS
# ============================================================================

def test_T71_mixed_predict_fraction_c1():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert 0 <= v["C1_MIXED_PREDICT_FRACTION"] <= 1


def test_T72_mixed_predict_fraction_c2():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert 0 <= v["C2_MIXED_PREDICT_FRACTION"] <= 1


def test_T73_multi_action_rate_c1():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert 0 <= v["C1_MIXED_MULTI_ACTION_RATE"] <= 1


def test_T74_multi_action_rate_c2():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert 0 <= v["C2_MIXED_MULTI_ACTION_RATE"] <= 1


def test_T75_action_entropy_c1():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["C1_MIXED_ACTION_ENTROPY"] >= 0


def test_T76_action_entropy_c2():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["C2_MIXED_ACTION_ENTROPY"] >= 0


def test_T77_mixed_action_diagnostics_in_selected():
    _require()
    sel = _load(RES / "E4MTR_selected_checkpoints.json")
    mad = sel["mixed_action_diagnostics"]
    for arm in (C1, C2):
        assert 0 <= mad[arm]["predict_fraction"] <= 1
        assert 0 <= mad[arm]["multi_action_rate"] <= 1
        assert mad[arm]["action_entropy"] >= 0


def test_T78_validation_action_diagnostics_csv():
    _require()
    rows = _rows("E4MTR_validation_action_diagnostics.csv")
    assert len(rows) == 10
    assert {r["arm"] for r in rows} == {C1, C2}


# ============================================================================
# T79-T84: TRAINING EPISODE QUALITY
# ============================================================================

def test_T79_training_episode_migd_finite():
    _require()
    for r in _rows("E4MTR_training_episode_results.csv"):
        assert math.isfinite(float(r["MIGD_response"]))
        assert math.isfinite(float(r["MIGD_epoch"]))


def test_T80_training_episode_fe_nonneg():
    _require()
    for r in _rows("E4MTR_training_episode_results.csv"):
        assert int(r["total_response_FE"]) >= 0


def test_T81_training_update_summary_csv():
    _require()
    rows = _rows("E4MTR_training_update_summary.csv")
    assert len(rows) == 60
    assert {r["arm"] for r in rows} == {C1, C2}


def test_T82_validation_selected_policies_csv():
    _require()
    rows = _rows("E4MTR_validation_selected_policies.csv")
    assert len(rows) == 10


# ============================================================================
# T83-T90: NUMERICAL STABILITY + FIREWALL + NO RETUNING
# ============================================================================

def test_T83_numerical_stability():
    _require()
    ns = _load(RES / "E4MTR_numerical_stability.json")
    assert ns["numerical_stability"] == "PASS"


def test_T84_numerical_stability_in_verdict():
    _require()
    assert _load(RES / "E4MTR_verdict.json")["NUMERICAL_STABILITY"] == "PASS"


def test_T85_all_runs_stable():
    _require()
    ns = _load(RES / "E4MTR_numerical_stability.json")
    for key, val in ns["runs"].items():
        assert val == "PASS", f"{key} not PASS"


def test_T86_final_seed_firewall_zero():
    _require()
    ff = _load(RES / "E4MTR_final_firewall.json")
    assert ff["M_FINAL_ACCESS_COUNT"] == 0
    assert ff["FINAL_EVALUATION"] == "NOT_RUN"
    assert ff["no_final_bootstrap"] is True


def test_T87_final_access_in_verdict_zero():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["M_FINAL_ACCESS_COUNT"] == 0


def test_T88_no_hyperparameter_retuning():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["HYPERPARAMETER_RETUNING"] == "NO"


def test_T89_no_architecture_retuning():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["ARCHITECTURE_RETUNING"] == "NO"


def test_T90_no_composite_weight_tuned():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["COMPOSITE_WEIGHT_TUNED"] == "NO"


def test_T91_no_checkpoint_selection_changed():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["CHECKPOINT_SELECTION_CHANGED"] == "NO"


def test_T92_no_causal_claim():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["CREDIT_CAUSAL_CLAIM"] == "NOT_MADE"
    assert v["FINAL_EVALUATION"] == "NOT_RUN"


# ============================================================================
# T93-T100: HISTORICAL VERDICTS + VERDICT STRUCTURE
# ============================================================================

def test_T93_e4f_verdict_failed():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED"


def test_T94_e4hf_verdict_failed():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T95_e4if_verdict_validated():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T96_e4kf_verdict_failed():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["E4KF_COORDINATION_CAUSAL_VERDICT"] == "FAILED"


def test_T97_source_verdicts_preserved():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert _load(E4IF / "E4IF_verdict.json")["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
    assert _load(E4KF / "E4KF_verdict.json")["COORDINATION_CAUSAL_VERDICT"] == "FAILED"


def test_T98_e4mt_still_complete():
    assert _load(E4MT / "E4MT_verdict.json")["E4MT_COMPLETE"] == "YES"


def test_T99_e4mtr_complete():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    assert v["E4MTR_COMPLETE"] in ("YES", "NO")


def test_T100_next_stage_routing():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    if v["E4MTR_COMPLETE"] == "YES":
        assert v["NEXT_STAGE"] == \
            "E4ME_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
    else:
        assert v["NEXT_STAGE"] == "E4MTR_ENGINEERING_REVIEW"


def test_T101_report_exists():
    _require()
    assert (RES / "E4MTR_report.md").exists()
    txt = (RES / "E4MTR_report.md").read_text()
    assert "E4MTR" in txt
    assert "credit" in txt.lower() or "Credit" in txt


def test_T102_verdict_has_all_required_keys():
    _require()
    v = _load(RES / "E4MTR_verdict.json")
    required = [
        "stage", "E4MTR_COMPLETE", "RUNS_COMPLETE",
        "C1_RUNS_COMPLETE", "C2_RUNS_COMPLETE",
        "TRAINING_EPISODES_COMPLETE", "PAIRED_INITIALIZATION",
        "SINGLE_FACTOR_INTEGRITY", "V3_CONSTANTS_INTEGRITY",
        "R1_REWARD_CONFIG_INTEGRITY", "QMIX_CONFIG_INTEGRITY",
        "C2_CREDIT_CONFIG_INTEGRITY",
        "NUMERICAL_STABILITY", "CHECKPOINTS_COMPLETE",
        "VALIDATION_CHECKPOINTS_COMPLETE",
        "C1_SELECTED_CHECKPOINTS", "C2_SELECTED_CHECKPOINTS",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "C2_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "VALIDATION_PAIRED_DELTA_C1_MINUS_C2",
        "HYPERPARAMETER_RETUNING", "ARCHITECTURE_RETUNING",
        "CHECKPOINT_SELECTION_CHANGED", "COMPOSITE_WEIGHT_TUNED",
        "BEST_POLICY_SELECTION",
        "FINAL_EVALUATION", "CREDIT_CAUSAL_CLAIM",
        "M_FINAL_ACCESS_COUNT",
        "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4KF_COORDINATION_CAUSAL_VERDICT",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in required:
        assert k in v, f"missing key: {k}"


def test_T103_c1_checkpoint_loads_mixer():
    _require()
    sel = _load(RES / "E4MTR_selected_checkpoints.json")["selected"]
    for key, val in sel.items():
        if val["arm"] == C1:
            payload = torch.load(val["checkpoint_path"], weights_only=False)
            from prdrl.rl.qmix import QMIXMixerV1
            mixer = QMIXMixerV1()
            mixer.load_state_dict(payload["mixer_online_state_dict"])
            break


def test_T104_c2_checkpoint_loads_utility_and_mixer():
    _require()
    sel = _load(RES / "E4MTR_selected_checkpoints.json")["selected"]
    for key, val in sel.items():
        if val["arm"] == C2:
            payload = torch.load(val["checkpoint_path"], weights_only=False)
            net = SharedRegionalQNetwork(9)
            net.load_state_dict(payload["online_state_dict"])
            from prdrl.rl.qmix import QMIXMixerV1
            mixer = QMIXMixerV1()
            mixer.load_state_dict(payload["mixer_online_state_dict"])
            break


def test_T105_execution_config_arms():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["arms"] == [C1, C2]
    assert ec["control"] == TREATMENT_ARCH
    assert ec["treatment"] == C2_CREDIT_ARCH


def test_T106_execution_config_hashes():
    _require()
    ec = _load(RES / "E4MTR_execution_config.json")
    assert ec["QMIX_CONFIG_SHA256"] == QMIX_SHA
    assert ec["C2_CREDIT_CONFIG_SHA256"] == C2_CREDIT_SHA
    assert ec["R1_config_sha256"] == R1_SHA
