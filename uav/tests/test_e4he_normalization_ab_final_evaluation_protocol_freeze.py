"""
E4HE — Normalization A/B Final Evaluation Protocol Freeze: contract + unit tests.

No final-eval environment instantiated. Verifies the frozen V1-vs-V3 causal protocol,
the paired crossed-bootstrap + verdict logic (unit), and the E4HE artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4HE_normalization_ab_final_evaluation_protocol_freeze as E4HE
from prdrl.rl.normalization import NORM_V1, NORM_V3
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro,
    independent_arm_bootstrap, N_BOOTSTRAP, CI_LEVEL, CI_METHOD,
)
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HT = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HTR = Path("results/E4HTR_normalization_ab_training_execution")
E4HR = Path("results/E4HR_normalization_design_review")
E4F = Path("results/E4F_final_evaluation")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _hash(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _require():
    if not (RES / "E4HE_verdict.json").exists():
        pytest.skip("E4HE not run yet")


# ---- prerequisites ----

def test_T1_e4ht():
    assert _load(E4HT / "E4HT_verdict.json")["E4HT_COMPLETE"] == "YES"


def test_T2_e4htr():
    assert _load(E4HTR / "E4HTR_verdict.json")["E4HTR_COMPLETE"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


# ---- unit: paired bootstrap + verdict logic ----

def test_T24_T25_T26_T27_T28_paired_bootstrap():
    D = np.random.default_rng(0).normal(0.0003, 0.001, size=(5, 60))
    out = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=56001)
    assert out["method"] == "paired_crossed_two_axis"
    assert out["n_policy"] == 5 and out["n_env"] == 60
    assert abs(out["point_estimate"] - float(D.mean())) < 1e-12


def test_T29_T30_forbidden_bootstraps_exist_as_marked():
    # independent-arm exists but is explicitly FORBIDDEN for inference
    V1 = np.random.default_rng(1).uniform(0.005, 0.009, (5, 60))
    V3 = V1 - 0.0003
    out = independent_arm_bootstrap(V1, V3, n_boot=500, seed=1)
    assert out["method"] == "independent_arm_FORBIDDEN"


def test_T31_policy_robustness_unit():
    assert FV.policy_robustness([1, 1, 1, 1, -1], threshold=4)["PASS"] is True
    assert FV.policy_robustness([1, 1, 1, -1, -1], threshold=4)["PASS"] is False


def test_T32_T33_T34_T35_T36_mechanism_gate_unit():
    assert FV.mechanism_decollapse_supported(0.1, 0.1, 0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"
    assert FV.mechanism_decollapse_supported(0.1, 0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"
    assert FV.mechanism_decollapse_supported(0.1, -0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"] == "NO"


def test_T41_T42_T43_T44_T45_T46_verdict_logic_unit():
    def V(mp, mech, rob, ps=True, ss=True, ms=True, eng=True):
        return FV.normalization_ab_verdict(
            mixed_primary_label=mp, mechanism_supported=mech, policy_robustness_pass=rob,
            persistent_safety_pass=ps, stationary_safety_pass=ss, macro_safety_pass=ms,
            checkpoint_integrity_pass=eng, protocol_integrity_pass=eng,
            version_integrity_pass=eng, no_final_tuning=eng)
    assert V(FV.BENEFICIAL, True, True)["verdict"] == "NORMALIZATION_CAUSAL_VALIDATED"
    assert V(FV.BENEFICIAL, False, True)["subtype"] == "PERFORMANCE_PARTIAL"
    mp = V(FV.PRACTICALLY_NEUTRAL, True, False)
    assert mp["subtype"] == "MECHANISTIC_PARTIAL"
    assert mp["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"
    assert V(FV.DIRECTIONAL_NEGATIVE, True, True)["verdict"] == "NORMALIZATION_CAUSAL_FAILED"
    assert V(FV.BENEFICIAL, True, True, ps=False)["verdict"] == "NORMALIZATION_CAUSAL_FAILED"
    assert V(FV.PRACTICALLY_NEUTRAL, False, True)["verdict"] == "NORMALIZATION_CAUSAL_FAILED"
    assert V(FV.HARMFUL, True, True)["verdict"] == "NORMALIZATION_CAUSAL_FAILED"  # not rescued


def test_T23_practical_margin():
    assert FV.DELTA_E4 == 1e-5


# ---- artifact tests ----

def test_T6_T7_ten_checkpoints_hashes():
    _require()
    d = _load(RES / "E4HE_checkpoint_freeze.json")
    assert d["n_checkpoints"] == 10
    for key, info in d["checkpoints"].items():
        assert info["hash_match"] is True
        assert info["sha256_recomputed"] == info["sha256"]


def test_T8_T9_T10_normalization_metadata():
    _require()
    d = _load(RES / "E4HE_checkpoint_freeze.json")["checkpoints"]
    v3_sha = _load(E4HR / "E4HR_v3_constants.json")["constants_sha256"]
    for key, info in d.items():
        assert info["normalization_metadata_ok"] is True
        if info["arm"] == "V1_CONTROL":
            assert info["normalization_version"] == NORM_V1
        else:
            assert info["normalization_version"] == NORM_V3
            assert info["normalizer_constants_sha256"] == v3_sha


def test_T11_no_reselection():
    _require()
    d = _load(RES / "E4HE_checkpoint_freeze.json")
    assert d["no_reselection"] is True
    assert d["no_alt_checkpoint_on_final"] is True
    assert d["immutable_for_final_evaluation"] is True


def test_T12_T13_final_seed_firewall():
    _require()
    d = _load(RES / "E4HE_final_seed_contract.json")
    assert d["final_seeds"] == list(range(1381, 1441))
    assert d["never_instantiated"] is True
    assert d["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T14_T15_T16_T17_T18_matrix():
    _require()
    d = _load(RES / "E4HE_final_matrix_contract.json")
    assert d["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]
    assert d["total_episodes"] == 1800
    assert d["V1_rows"] == 900 and d["V3_rows"] == 900
    assert d["pairing_key"] == ["policy_seed", "environment_seed", "scenario"]
    assert d["no_global_or_rule_in_primary_matrix"] is True


def test_T19_T20_T21_T22_endpoint_effect():
    _require()
    ec = _load(RES / "E4HE_endpoint_contract.json")
    assert ec["primary_endpoint"] == "MIGD_response"
    assert ec["primary_scenario"] == "MIXED"
    ed = _load(RES / "E4HE_effect_definition.json")
    assert "MIGD_response_V1" in ed["cell_effect"] and "MIGD_response_V3" in ed["cell_effect"]
    assert ed["positive_means"] == "V3 better"


def test_T24b_T27b_T28b_bootstrap_contract():
    _require()
    d = _load(RES / "E4HE_paired_bootstrap_contract.json")
    assert d["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
    assert d["policy_axis_n"] == 5 and d["environment_axis_n"] == 60
    assert d["N_BOOTSTRAP"] == 50000 and d["BOOTSTRAP_MASTER_SEED"] == 56001
    assert "naive_300_row_bootstrap" in d["forbidden"]
    assert "independent_arm_bootstrap" in d["forbidden"]


def test_T39_macro_shared_draws():
    _require()
    m = _load(RES / "E4HE_macro_contract.json")
    assert abs(m["scenario_weight"] - 1/3) < 1e-12
    assert m["shared_bootstrap_draws_across_scenarios"] is True


def test_T31b_policy_robustness_contract():
    _require()
    d = _load(RES / "E4HE_policy_robustness.json")
    assert d["threshold"] == 4
    assert "4 of 5" in d["rule"]


def test_T32b_T36b_mechanism_contract():
    _require()
    d = _load(RES / "E4HE_mechanism_contract.json")
    assert d["endpoints"] == ["PREDICT_fraction", "within_event_multi_action_rate", "regional_action_entropy"]
    assert ">=2 of 3" in d["gate"]
    assert d["uses_requested_action"] is True


def test_T37_T38_T40_safety_fe():
    _require()
    s = _load(RES / "E4HE_safety_contract.json")
    assert s["safety_pass"] == "neither endpoint HARMFUL"
    assert s["fe_cannot_rescue_harmful_migd"] is True
    fe = _load(RES / "E4HE_fe_contract.json")
    assert fe["cannot_rescue_harmful_quality"] is True


def test_T41b_success_logic_contract():
    _require()
    d = _load(RES / "E4HE_success_logic.json")
    assert "MIXED primary=BENEFICIAL" in d["VALIDATED_requires_all"]
    assert "MECHANISTIC_PARTIAL" in d
    assert d["mechanism_cannot_rescue_harmful"] is True
    assert d["beneficial_without_mechanism_is_partial_not_validated"] is True


def test_T47_T48_T49_T50_output_schema():
    _require()
    d = _load(RES / "E4HE_output_schema.json")
    assert d["episode_row_keys"] == ["arm", "policy_seed", "scenario", "environment_seed"]
    assert d["no_pseudoreplication"] is True
    assert "requested_actions" in d["change_level_diagnostics"]


def test_T51_T52_T53_T54_T55_T56_T57_execution_contract():
    _require()
    d = _load(RES / "E4HE_final_execution_contract.json")
    assert d["epsilon"] == 0.0 and d["learning"] == "none"
    assert d["gradients"] == "disabled" and d["replay"] == "disabled" and d["optimizer"] == "disabled"
    assert d["v3_frozen_constants_no_adaptation"] is True and d["v1_legacy_normalization"] is True


def test_T58_to_T67_synthetic_validation():
    _require()
    d = _load(RES / "E4HE_synthetic_validation.json")
    assert d["ALL_PASS"] is True
    assert d["n_checks"] == 18
    for k, c in d["checks"].items():
        assert c["PASS"] is True


def test_T68_T69_no_final_no_training():
    _require()
    fw = _load(RES / "E4HE_final_firewall.json")
    assert fw["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert fw["FINAL_EVALUATION"] == "NOT_RUN"
    assert fw["NORMALIZATION_CAUSAL_CLAIM"] == "NOT_MADE"
    v = _load(RES / "E4HE_verdict.json")
    assert v["NEW_TRAINING"] == "NO"


def test_T70_old_artifacts_unchanged():
    _require()
    frz = _load("results/E4E_final_evaluation_protocol_freeze/E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_verdict_complete_and_next():
    _require()
    v = _load(RES / "E4HE_verdict.json")
    assert v["E4HE_COMPLETE"] == "YES"
    assert v["CHECKPOINT_INTEGRITY"] == "PASS"
    assert v["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS"
    assert v["no_placeholder_tokens"] is True
    assert v["NEXT_STAGE"] == "E4HF_NORMALIZATION_AB_FINAL_EVALUATION_EXECUTION"


def test_classification_logic_source():
    _require()
    d = _load(RES / "E4HE_classification_logic.json")
    assert d["DELTA_NORM"] == 1e-5
    assert "classify_effect" in d["source"]
