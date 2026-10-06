"""
E4IE — Reward A/B Final Evaluation Protocol Freeze: contract + unit tests.

No final-eval environment instantiated. Verifies the frozen R0-vs-R1 causal protocol, the
paired crossed-bootstrap + reward-mechanism + verdict logic (unit), the 24-check synthetic
validation, the 10 frozen checkpoints, and the non-confirmatory validation lock.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4IE_reward_ab_final_evaluation_protocol_freeze as E4IE
from prdrl.rl import reward_integration as RI
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME
from prdrl.rl.normalization import NORM_V3
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro,
    independent_arm_bootstrap, N_BOOTSTRAP, CI_LEVEL, CI_METHOD,
)
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4IE_reward_ab_final_evaluation_protocol_freeze")
E4IT = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4ITR = Path("results/E4ITR_reward_ab_training_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4IE_verdict.json").exists():
        pytest.skip("E4IE not run yet")


def test_T1_e4it():
    assert _load(E4IT / "E4IT_verdict.json")["E4IT_COMPLETE"] == "YES"


def test_T2_e4itr():
    assert _load(E4ITR / "E4ITR_verdict.json")["E4ITR_COMPLETE"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T4_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T5_branch_b():
    _require()
    assert _load(RES / "E4IE_verdict.json")["SELECTED_BRANCH"] == "B_REWARD_CREDIT"


def test_T6_exact_two_arms():
    _require()
    fm = _load(RES / "E4IE_final_matrix_contract.json")
    assert fm["arms"] == ["R0_CONTROL", "R1_TREATMENT"]


def test_T7_T8_five_each():
    _require()
    ck = _load(RES / "E4IE_checkpoint_freeze.json")
    assert ck["n_r0"] == 5 and ck["n_r1"] == 5 and ck["n_checkpoints"] == 10


def test_T9_checkpoint_hashes_exact():
    _require()
    ck = _load(RES / "E4IE_checkpoint_freeze.json")["checkpoints"]
    for v in ck.values():
        assert v["hash_match"] is True and v["sha256"] == v["sha256_recomputed"]


def test_T10_no_reselection():
    _require()
    ck = _load(RES / "E4IE_checkpoint_freeze.json")
    assert ck["no_reselection"] and ck["no_weight_averaging"] and ck["no_ensemble"] and ck["all_ten_advance"]


def test_T11_T12_reward_metadata():
    _require()
    for v in _load(RES / "E4IE_checkpoint_freeze.json")["checkpoints"].values():
        assert v["reward_metadata_ok"] is True
        if v["arm"] == "R0_CONTROL":
            assert v["reward_version"] == REWARD_R0_NAME
        else:
            assert v["reward_version"] == REWARD_R1_NAME


def test_T13_r1_config_sha():
    _require()
    assert _load(RES / "E4IE_reward_version_integrity.json")["R1_config_sha256"] == SHA


def test_T14_T15_alpha():
    _require()
    rv = _load(RES / "E4IE_reward_version_integrity.json")
    assert rv["alpha_local_R0"] == 0.1 and rv["alpha_local_R1"] == 0.1


def test_T16_T17_v3():
    _require()
    n = _load(RES / "E4IE_normalization_integrity.json")
    assert n["normalization"] == NORM_V3 and n["identical_both_arms"] is True


def test_T18_T19_state_network():
    _require()
    fe = _load(RES / "E4IE_final_execution_contract.json")
    assert fe["state_dim"] == 9 and fe["network"] == "9-128-128-4"


def test_T20_T21_final_seeds_and_access():
    _require()
    sc = _load(RES / "E4IE_final_seed_contract.json")
    assert sc["range"] == [1581, 1640] and sc["n"] == 60 and sc["never_instantiated"] is True
    assert _load(RES / "E4IE_final_firewall.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T22_scenarios():
    _require()
    assert _load(RES / "E4IE_final_matrix_contract.json")["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T23_T24_T25_matrix_1800():
    _require()
    fm = _load(RES / "E4IE_final_matrix_contract.json")
    assert fm["total_episodes"] == 1800 and fm["R0_rows"] == 900 and fm["R1_rows"] == 900


def test_T26_pairing_key():
    _require()
    assert _load(RES / "E4IE_final_matrix_contract.json")["pairing_key"] == \
        ["policy_seed", "environment_seed", "scenario"]


def test_T27_T28_T29_T30_endpoint():
    _require()
    v = _load(RES / "E4IE_verdict.json")
    assert v["PRIMARY_SCENARIO"] == "MIXED" and v["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"
    assert v["EFFECT_DEFINITION"] == "R0_MINUS_R1" and v["POSITIVE_MEANS"] == "R1_BETTER"


def test_T31_practical_margin():
    _require()
    assert _load(RES / "E4IE_practical_margin.json")["DELTA_NORM"] == 1e-5


def test_T32_T33_T34_T35_T36_bootstrap():
    _require()
    b = _load(RES / "E4IE_paired_bootstrap_contract.json")
    assert b["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
    assert b["policy_axis_n"] == 5 and b["environment_axis_n"] == 60
    assert b["N_BOOTSTRAP"] == 50000 and b["BOOTSTRAP_MASTER_SEED"] == 66001


def test_T37_T38_forbidden_bootstraps():
    _require()
    b = _load(RES / "E4IE_paired_bootstrap_contract.json")
    assert any("naive" in f for f in b["forbidden"]) and any("independent" in f for f in b["forbidden"])


def test_T39_policy_robustness():
    _require()
    r = _load(RES / "E4IE_policy_robustness.json")
    assert r["threshold"] == 4 and ">=4 of 5" in r["rule"]


def test_T40_T41_T42_mechanism_endpoints():
    _require()
    m = _load(RES / "E4IE_reward_mechanism_contract.json")
    assert m["endpoints"] == ["LOCAL_CREDIT_ACTIVITY", "GLOBAL_LOCAL_MAGNITUDE_RATIO", "REGIONAL_REWARD_DISPERSION"]
    assert ">0 expected" in m["endpoint_1"]["delta_activity"]
    assert ">0 expected" in m["endpoint_2"]["delta_ratio"]
    assert ">0 expected" in m["endpoint_3"]["delta_dispersion"]


def test_T43_dispersion_ddof0():
    _require()
    assert "ddof=0" in _load(RES / "E4IE_reward_mechanism_contract.json")["endpoint_3"]["metric"]


def test_T44_mechanism_2_of_3():
    _require()
    m = _load(RES / "E4IE_reward_mechanism_contract.json")
    assert ">=2 of 3" in m["gate"]
    # unit
    assert RI.reward_mechanism_supported(0.1, 0.1, -0.1)["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"
    assert RI.reward_mechanism_supported(0.1, -0.1, -0.1)["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "NO"


def test_T45_zero_active_not_applicable():
    _require()
    z = _load(RES / "E4IE_zero_active_contract.json")
    assert z["no_divide_by_zero"] and z["not_material"]
    r = RI.reward_mechanism_supported(0.02, "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL", 0.01)
    assert r["n_expected"] == 2 and r["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"


def test_T46_action_diversity_not_gate():
    _require()
    m = _load(RES / "E4IE_reward_mechanism_contract.json")
    assert "NOT action diversity" in m["primary_mechanism"]


def test_T47_T48_safety():
    _require()
    s = _load(RES / "E4IE_safety_contract.json")
    assert s["scenarios"] == ["PERSISTENT", "STATIONARY"]
    assert s["safety_pass"] == "neither endpoint HARMFUL"


def test_T49_T50_macro():
    _require()
    m = _load(RES / "E4IE_macro_contract.json")
    assert m["scenario_weight"] == 1/3 and m["shared_bootstrap_draws_across_scenarios"] is True


def test_T51_T52_fe_secondary():
    _require()
    fe = _load(RES / "E4IE_fe_contract.json")
    assert fe["secondary"] is True and fe["cannot_rescue_harmful_quality"] is True


def test_T53_validated_logic():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.BENEFICIAL, mechanism_supported=True,
                             policy_robustness_pass=True, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["verdict"] == "REWARD_CAUSAL_VALIDATED"


def test_T54_performance_partial():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.BENEFICIAL, mechanism_supported=False,
                             policy_robustness_pass=True, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["subtype"] == "PERFORMANCE_PARTIAL"


def test_T55_mechanistic_partial():
    v = RI.reward_ab_verdict(mixed_primary_label=FV.PRACTICALLY_NEUTRAL, mechanism_supported=True,
                             policy_robustness_pass=False, persistent_safety_pass=True,
                             stationary_safety_pass=True, macro_safety_pass=True,
                             checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                             version_integrity_pass=True, no_final_tuning=True)
    assert v["subtype"] == "MECHANISTIC_PARTIAL" and v["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"


def test_T56_T57_failed_and_mechanism_cannot_rescue():
    for lab in (FV.DIRECTIONAL_NEGATIVE, FV.HARMFUL):
        v = RI.reward_ab_verdict(mixed_primary_label=lab, mechanism_supported=True,
                                 policy_robustness_pass=True, persistent_safety_pass=True,
                                 stationary_safety_pass=True, macro_safety_pass=True,
                                 checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                                 version_integrity_pass=True, no_final_tuning=True)
        assert v["verdict"] == "REWARD_CAUSAL_FAILED"


@pytest.mark.parametrize("key", [f"{i}_" for i in range(1, 25)])
def test_T58_through_T70_synthetic_checks(key):
    _require()
    synth = _load(RES / "E4IE_synthetic_validation.json")
    matches = [k for k in synth["checks"] if k.startswith(key)]
    assert matches, f"missing synthetic check {key}"
    for k in matches:
        assert synth["checks"][k]["PASS"] is True


def test_T70b_synthetic_all_pass():
    _require()
    synth = _load(RES / "E4IE_synthetic_validation.json")
    assert synth["ALL_PASS"] is True and synth["n_checks"] == 24
    assert _load(RES / "E4IE_verdict.json")["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS"


def test_T71_T72_validation_non_confirmatory():
    _require()
    lock = _load(RES / "E4IE_validation_observation_lock.json")
    assert lock["status"] == "NON_CONFIRMATORY_LOCKED"
    for f in ["primary_endpoint", "practical_margin", "bootstrap", "safety", "robustness",
              "mechanism_gate", "verdict_logic", "checkpoints", "alpha"]:
        assert f in lock["must_not_alter"]
    assert _load(RES / "E4IE_verdict.json")["VALIDATION_OBSERVATION_STATUS"] == "NON_CONFIRMATORY_LOCKED"


def test_T73_T74_no_final_no_training():
    _require()
    fw = _load(RES / "E4IE_final_firewall.json")
    assert fw["FINAL_EVALUATION"] == "NOT_RUN" and fw["REWARD_CAUSAL_CLAIM"] == "NOT_MADE"
    assert fw["NEW_TRAINING"] == "NO"


def test_T75_historical_artifacts_unchanged():
    _require()
    h = _load(RES / "E4IE_historical_integrity.json")
    assert h["E4F"] == "FAILED" and h["E4HF_normalization"] == "FAILED"


def test_T76_e4if_output_schema_frozen():
    _require()
    sch = _load(RES / "E4IE_output_schema.json")
    assert len(sch["e4if_required_outputs"]) == 18
    assert "E4IF_verdict.json" in sch["e4if_required_outputs"]
    assert _load(RES / "E4IE_verdict.json")["NEXT_STAGE"] == "E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION"


def test_T77_paired_bootstrap_unit_66001():
    D = np.random.default_rng(0).normal(0.0002, 0.001, size=(5, 60))
    out = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=66001)
    assert abs(out["point_estimate"] - float(D.mean())) < 1e-12
    assert out["n_policy"] == 5 and out["n_env"] == 60
