"""
E4II — Reward Implementation Validation: artifact + integration tests.

Validates the frozen R1 reward plumbing WITHOUT training: config hash integrity,
independent reference agreement over the full E4IDD record set, formula properties,
invalid-input / degenerate / timing semantics, R0/R1 invariance + single-factor arm
isolation, reward-version guards, and firewall/no-training audits.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, regional_quality_gain_r1, EPS_R1
from prdrl.rl import reward_integration as RI
from prdrl.rl.normalization import NORM_V3

RES = Path("results/E4II_reward_implementation_validation")
E4IDD = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4II_implementation_verdict.json").exists():
        pytest.skip("E4II not run yet")


def _g_ref(Lp, Lq):
    if math.isnan(Lp) or math.isnan(Lq):
        return 0.0
    if Lp == 0.0 and Lq == 0.0:
        return 0.0
    return (Lp - Lq) / (Lp + Lq + 1e-12)


# ---- prerequisites ----

def test_T1_e4idd_complete():
    assert _load(E4IDD / "E4IDD_verdict.json")["E4IDD_COMPLETE"] == "YES"


def test_T2_r1_reward_ready():
    assert _load(E4IDD / "E4IDD_verdict.json")["R1_REWARD_READY"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T4_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T5_branch_b():
    _require()
    assert _load(RES / "E4II_implementation_verdict.json")["SELECTED_BRANCH"] == "B_REWARD_CREDIT"


def test_T6_alpha_final_exact():
    _require()
    assert _load(RES / "E4II_implementation_verdict.json")["R1_ALPHA_LOCAL"] == 0.1
    assert RI.FROZEN_R1_ALPHA_LOCAL == 0.1


def test_T7_r1_config_sha_exact():
    _require()
    assert _load(RES / "E4II_implementation_verdict.json")["REWARD_R1_CONFIG_SHA256"] == SHA
    assert RI.FROZEN_R1_CONFIG_SHA256 == SHA


def test_T8_r0_reward_version_exact():
    assert REWARD_R0_NAME == "ORIGINAL_REWARD_R0"


def test_T9_r1_reward_version_exact():
    assert REWARD_R1_NAME == "REWARD_R1_REGIONAL_QUALITY_GAIN"


def test_T10_r0_source_unchanged():
    # R0 source contains no R1 symbols (byte-stable control)
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "REGIONAL_QUALITY_GAIN" not in src and "reward_r1" not in src


def test_T11_independent_reference_exists():
    _require()
    ref = _load(RES / "E4II_reference_implementation.json")
    assert "inline" in ref["independent_reference"]


def test_T12_all_e4idd_rows_replayed():
    _require()
    r = _load(RES / "E4II_e4idd_offline_replay.json")
    with open(E4IDD / "E4IDD_regional_reward_records.csv") as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert r["n_rows"] == n == 30000


def test_T13_T14_T15_production_reference_agreement():
    _require()
    a = _load(RES / "E4II_formula_agreement.json")
    assert a["REFERENCE_IMPLEMENTATION_MATCH"] == "PASS"
    assert a["max_g_impl_vs_reference"] <= 1e-12
    assert a["max_weighted_local_diff"] <= 1e-12
    assert a["max_total_diff"] <= 1e-9
    # independent recompute on a sample
    with open(E4IDD / "E4IDD_regional_reward_records.csv") as f:
        rows = list(csv.DictReader(f))[:2000]
    for r in rows:
        Lp, Lq = float(r["L_pre"]), float(r["L_post"])
        assert abs(regional_quality_gain_r1(Lp, Lq) - _g_ref(Lp, Lq)) <= 1e-12


def test_T16_boundedness():
    _require()
    assert _load(RES / "E4II_formula_properties.json")["R1_BOUNDS"] == "PASS"


def test_T17_T18_T19_T20_sign_and_zero():
    assert regional_quality_gain_r1(1.0, 0.2) > 0
    assert regional_quality_gain_r1(0.2, 1.0) < 0
    assert regional_quality_gain_r1(0.5, 0.5) == 0.0
    assert regional_quality_gain_r1(0.0, 0.0) == 0.0


def test_T21_fixture_2_1():
    _require()
    assert abs(_load(RES / "E4II_hand_fixtures.json")["D_2_1"] - 1/3) < 1e-6


def test_T22_fixture_1_2():
    _require()
    assert abs(_load(RES / "E4II_hand_fixtures.json")["E_1_2"] + 1/3) < 1e-6


def test_T23_invalid_nan_handled_as_degenerate():
    _require()
    # NaN is frozen degenerate -> g_r=0 (allowed, not a hard error)
    inv = _load(RES / "E4II_invalid_input_contract.json")
    assert inv["nan_L_post"]["raised"] is False
    assert regional_quality_gain_r1(1.0, float("nan")) == 0.0


def test_T24_invalid_inf_rejected():
    _require()
    inv = _load(RES / "E4II_invalid_input_contract.json")
    assert inv["pos_inf"]["raised"] and inv["neg_inf"]["raised"]
    with pytest.raises(RI.RewardConfigError):
        RI.validate_regional_losses(float("inf"), 1.0)


def test_T25_negative_loss_rejected():
    _require()
    with pytest.raises(RI.RewardConfigError):
        RI.validate_regional_losses(-1.0, 1.0)
    with pytest.raises(RI.RewardConfigError):
        RI.validate_regional_losses(1.0, -1.0)


def test_T26_degenerate_semantics():
    _require()
    assert _load(RES / "E4II_degenerate_region_contract.json")["DEGENERATE_REGION_SEMANTICS"] == "PASS"


def test_T27_region_identity():
    _require()
    assert _load(RES / "E4II_region_identity.json")["FIXED_REGION_IDENTITY"] == "PASS"


def test_T28_T29_reward_timing_after_joint_response_no_future():
    _require()
    t = _load(RES / "E4II_reward_timing.json")
    assert t["R1_CAUSALITY"] == "PASS"
    assert t["uses_future_env_state"] is False and t["uses_next_change_info"] is False
    assert t["joint_response_before_local_reward"] is True


def test_T30_requested_realized_semantics_retained():
    _require()
    log = _load(RES / "E4II_logging_schema.json")
    assert "requested_action" in log["action"] and "realized_operator" in log["action"]
    assert "fallback" in log["action"]


def test_T31_global_reward_same_r0_r1():
    _require()
    assert _load(RES / "E4II_global_term_invariance.json")["GLOBAL_REWARD_INVARIANCE"] == "PASS"


def test_T32_fe_term_same_r0_r1():
    _require()
    assert _load(RES / "E4II_fe_term_invariance.json")["FE_TERM_INVARIANCE"] == "PASS"


def test_T33_r1_extra_fe_zero():
    _require()
    assert _load(RES / "E4II_extra_fe_validation.json")["R1_EXTRA_FE"] == 0


def test_T34_T35_alpha_loaded_from_config_no_runtime_derivation():
    _require()
    c = _load(RES / "E4II_r1_config_integrity.json")
    assert c["ALPHA_CONFIG_INTEGRITY"] == "PASS" and c["alpha_local"] == 0.1
    assert c["runtime_alpha_derivation"] is False


def test_T36_wrong_config_hash_errors():
    with pytest.raises(RI.RewardConfigError):
        RI.validate_checkpoint_reward_version(
            {"reward_version": REWARD_R1_NAME, "reward_config_sha256": "deadbeef"},
            REWARD_R1_NAME, SHA)


def test_T37_reward_version_mismatch_errors():
    with pytest.raises(RI.RewardConfigError):
        RI.validate_checkpoint_reward_version({"reward_version": REWARD_R0_NAME}, REWARD_R1_NAME, SHA)
    with pytest.raises(RI.RewardConfigError):
        RI.validate_checkpoint_reward_version({"reward_version": REWARD_R1_NAME}, REWARD_R0_NAME)


def test_T38_unsupported_reward_version_errors():
    with pytest.raises(RI.RewardConfigError):
        RI.validate_checkpoint_reward_version({"reward_version": "REWARD_RX"}, REWARD_R0_NAME)


def test_T39_T40_both_arms_v3():
    _require()
    fa = _load(RES / "E4II_future_arm_configs.json")
    assert fa["R0_CONTROL"]["normalization"] == NORM_V3
    assert fa["R1_TREATMENT"]["normalization"] == NORM_V3


def test_T41_T42_state_dim_and_network():
    _require()
    fa = _load(RES / "E4II_future_arm_configs.json")
    assert fa["R0_CONTROL"]["state_dim"] == 9 and fa["R1_TREATMENT"]["state_dim"] == 9
    assert fa["R0_CONTROL"]["network"] == "9-128-128-4"


def test_T43_reward_not_appended_to_state():
    _require()
    # state_dim stays 9; reward is a training signal only
    assert _load(RES / "E4II_implementation_verdict.json")["STATE_DIM"] == 9
    ag = _load(RES / "E4II_autograd_boundary.json")
    assert ag["no_differentiable_reward_path"] is True


def test_T44_local_active_threshold():
    assert RI.LOCAL_ACTIVE_EPS == 1e-12 == EPS_R1
    assert RI.local_active(1e-6) is True and RI.local_active(0.0) is False


def test_T45_T46_logs_contain_l_pre_l_post():
    _require()
    log = _load(RES / "E4II_logging_schema.json")
    assert "L_pre" in log["quality"] and "L_post" in log["quality"]


def test_T47_T48_T49_log_reconstructability():
    _require()
    assert _load(RES / "E4II_log_reconstructability.json")["LOG_RECONSTRUCTABILITY"] == "PASS"


def test_T50_T51_stationary_zero_active():
    _require()
    s = _load(RES / "E4II_stationary_zero_active_contract.json")
    assert s["STATIONARY_ZERO_ACTIVE_HANDLING"] == "PASS"
    assert s["no_divide_by_zero"] is True and s["does_not_prove_materiality"] is True
    assert s["global_local_active_ratio"] == "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"


def test_T52_all_replay_finite():
    _require()
    assert _load(RES / "E4II_e4idd_offline_replay.json")["all_finite"] is True


def test_T53_near_zero_numerical_finite():
    _require()
    assert _load(RES / "E4II_numerical_stability.json")["NUMERICAL_STABILITY"] == "PASS"


def test_T54_reward_detached_from_autograd():
    _require()
    assert _load(RES / "E4II_autograd_boundary.json")["AUTOGRAD_BOUNDARY"] == "PASS"
    assert isinstance(regional_quality_gain_r1(2.0, 1.0), float)


def test_T55_replay_schema_unchanged():
    _require()
    assert _load(RES / "E4II_replay_schema.json")["REPLAY_SCHEMA"] == "PASS"


def test_T56_T57_synthetic_dispatchers():
    _require()
    d = _load(RES / "E4II_synthetic_dry_run.json")
    assert d["R0_dispatch_ok"] and d["R1_dispatch_ok"] and d["SYNTHETIC_DRY_RUN"] == "PASS"


def test_T58_T59_future_arms_exact():
    _require()
    fa = _load(RES / "E4II_future_arm_configs.json")
    assert fa["R0_CONTROL"]["reward_version"] == REWARD_R0_NAME
    assert fa["R1_TREATMENT"]["reward_version"] == REWARD_R1_NAME
    assert fa["R1_TREATMENT"]["reward_config_sha256"] == SHA


def test_T60_future_arm_diff_reward_only():
    _require()
    assert _load(RES / "E4II_single_factor_implementation.json")["SINGLE_FACTOR_IMPLEMENTATION"] == "PASS"
    assert _load(RES / "E4II_single_factor_implementation.json")["differing_fields"] == []


def test_T61_T62_T63_reward_seeds_untouched():
    _require()
    fw = _load(RES / "E4II_future_seed_firewall.json")
    assert fw["REWARD_TRAIN_ACCESS_COUNT"] == 0
    assert fw["REWARD_VALIDATION_ACCESS_COUNT"] == 0
    assert fw["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T64_diagnostic_not_rerun():
    _require()
    assert _load(RES / "E4II_future_seed_firewall.json")["diagnostic_1701_1720_rerun"] is False


def test_T65_T66_T67_T68_T69_T70_no_training_audit():
    _require()
    a = _load(RES / "E4II_no_training_audit.json")
    for k in ["real_environment_episodes", "optimizer_steps", "gradient_updates", "target_updates",
              "replay_training_appends", "new_policy_checkpoints", "model_weight_changes"]:
        assert a[k] == 0


def test_T71_future_training_contract_unchanged():
    # E4I future training contract still present + unchanged fields
    hp = _load("results/E4I_reward_credit_revision_design/E4I_future_training_contract.json")["hyperparameters"]
    assert hp["gamma"] == 0.99 and hp["batch"] == 128 and hp["epochs"] == 6


def test_T72_checkpoint_selection_rule_unchanged():
    fa = _load(RES / "E4II_future_arm_configs.json")
    assert fa["R0_CONTROL"]["checkpoint_selection"] == "VALIDATION_MACRO_MIGD_RESPONSE"
    assert fa["R1_TREATMENT"]["checkpoint_selection"] == "VALIDATION_MACRO_MIGD_RESPONSE"


def test_T73_historical_artifacts_unchanged():
    _require()
    h = _load(RES / "E4II_historical_integrity.json")
    assert h["reward_r1_formula_hash_matches_config"] is True
    assert h["v3_sha_matches_config"] is True
    assert h["ORIGINAL_E4F_VERDICT"] == "FAILED"


def test_T74_implementation_valid_logic_exact():
    _require()
    v = _load(RES / "E4II_implementation_verdict.json")
    valid = all(v["checks"].values())
    assert (v["R1_IMPLEMENTATION_VALID"] == "YES") == valid
    if valid:
        assert v["E4II_COMPLETE"] == "YES"
        assert v["NEXT_STAGE"] == "E4IT_REWARD_AB_TRAINING_PROTOCOL_FREEZE"


def test_T75_config_hash_reproducible_and_dispatch_matches():
    _require()
    cfg = _load(E4IDD / "E4IDD_reward_r1_config.json")
    stored = cfg.pop("REWARD_R1_CONFIG_SHA256")
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == stored == SHA
    # dispatch: R1 total = weighted_global + 0.1*g - weighted_fe
    out = RI.compute_regional_total(REWARD_R1_NAME, global_reward=0.03, L_pre=2.0, L_post=1.0,
                                    fe_cost=0.5, alpha_local=0.1)
    g = regional_quality_gain_r1(2.0, 1.0)
    assert abs(out["total_reward"] - (0.03 + 0.1 * g - 0.01 * 0.5)) < 1e-12


def test_T76_r1_alpha_must_be_frozen_value():
    # dispatcher rejects a non-frozen alpha for R1 (single-factor discipline)
    with pytest.raises(RI.RewardConfigError):
        RI.compute_regional_total(REWARD_R1_NAME, global_reward=0.0, L_pre=1.0, L_post=0.5,
                                  fe_cost=0.0, alpha_local=0.5)
