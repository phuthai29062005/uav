"""
E4HI — Normalization Implementation Validation: unit + contract + artifact tests.

Independent-reference logic, structural-zero/positive/monotonicity properties, hand
fixtures, and the frozen-gate artifacts. No training; historical results preserved.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4HI_normalization_implementation_validation as E4HI
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, NORM_V3, normalize_v1, ZeroAwareRobustLogNormalizerV3,
    RAW_ORDER, ZERO_INFLATED_DIMS, DELTA_DIM, check_normalization_compatibility,
)

RES = Path("results/E4HI_normalization_implementation_validation")
E4HR = Path("results/E4HR_normalization_design_review")
E4F = Path("results/E4F_final_evaluation")
E4H = Path("results/E4H_state_normalization_revision_design")


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
    if not (RES / "E4HI_verdict.json").exists():
        pytest.skip("E4HI not run yet")


# ---- unit: independent reference vs production ----

def test_T11_T12_reference_matches_production():
    v3, m_by, c_by, a3, _ = E4HI.load_v3_from_e4hr()
    rng = np.random.default_rng(0)
    X = np.zeros((2000, 9))
    for d in ZERO_INFLATED_DIMS:
        mask = rng.random(2000) > 0.5
        X[mask, d] = rng.exponential(0.003, mask.sum())
    X[:, 2] = rng.uniform(0, 0.3, 2000); X[:, 4] = 1.0
    X[:, DELTA_DIM] = rng.normal(0, 0.002, 2000)
    Z_impl = v3.transform_batch(X)
    Z_ref = E4HI.v3_reference_transform(X, m_by, c_by, a3)
    assert float(np.max(np.abs(Z_impl - Z_ref))) <= 1e-12


def test_T16_structural_zero_maps_zero():
    v3, m_by, c_by, a3, _ = E4HI.load_v3_from_e4hr()
    z = v3.transform(np.zeros(9))
    for d in ZERO_INFLATED_DIMS:
        assert abs(z[d]) <= 1e-12


def test_T17_positive_maps_positive():
    v3, *_ = E4HI.load_v3_from_e4hr()
    xp = np.zeros(9); xp[0] = 1e-7; xp[1] = 1e-7
    z = v3.transform(xp)
    assert z[0] > 0 and z[1] > 0


def test_T18_monotonic():
    v3, *_ = E4HI.load_v3_from_e4hr()
    for d in ZERO_INFLATED_DIMS:
        xs = np.linspace(0, 3, 50); base = np.zeros((50, 9)); base[:, d] = xs
        z = v3.transform_batch(base)[:, d]
        assert np.all(np.diff(z) >= -1e-12)


def test_T19_T20_delta_zero_sign():
    v3, _, _, a3, _ = E4HI.load_v3_from_e4hr()
    assert abs(v3.transform(np.zeros(9))[DELTA_DIM]) <= 1e-12
    dp = np.zeros(9); dp[DELTA_DIM] = a3 / 2
    dn = np.zeros(9); dn[DELTA_DIM] = -a3 / 2
    assert v3.transform(dp)[DELTA_DIM] > 0 and v3.transform(dn)[DELTA_DIM] < 0


def test_T21_T22_identity():
    v3, *_ = E4HI.load_v3_from_e4hr()
    oc = np.zeros(9); oc[2] = 0.33
    ac = np.zeros(9); ac[4] = 0.55
    assert abs(v3.transform(oc)[2] - 0.33) < 1e-9
    assert abs(v3.transform(ac)[4] - 0.55) < 1e-9


def test_T43_T44_T45_T46_hand_fixtures():
    v3, m_by, c_by, a3, _ = E4HI.load_v3_from_e4hr()
    d = ZERO_INFLATED_DIMS[0]; m, c = m_by[d], c_by[d]
    def one(v):
        x = np.zeros(9); x[d] = v; return x
    assert abs(v3.transform(one(0.0))[d]) <= 1e-12                                  # x=0
    assert abs(v3.transform(one(m))[d] - np.log1p(1.0) / np.log1p(c / m)) < 1e-6     # x=m
    assert abs(v3.transform(one(c))[d] - 1.0) < 1e-6                                 # x=c
    assert abs(v3.transform(one(c * 5))[d] - 1.0) < 1e-9                             # x>c clips 1
    assert 0 < v3.transform(one(m / 2))[d] < v3.transform(one(m))[d]                 # 0<x<m


def test_T40_T41_T42_backward_compat_raises():
    v3, _, _, _, sha = E4HI.load_v3_from_e4hr()
    check_normalization_compatibility({}, NORM_V1)   # legacy default ok
    with pytest.raises(ValueError):
        check_normalization_compatibility({}, NORM_V3, sha)
    with pytest.raises(ValueError):
        check_normalization_compatibility(
            {"normalization_version": NORM_V3, "normalizer_constants_sha256": "bad"}, NORM_V3, sha)
    with pytest.raises(ValueError):
        check_normalization_compatibility({"normalization_version": "FOO"}, NORM_V3, sha)


# ---- integrity / history ----

def test_T1_e4hr_complete():
    assert _load(E4HR / "E4HR_verdict.json")["E4HR_COMPLETE"] == "YES"


def test_T2_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T3_e4h_no():
    assert _load(E4H / "E4H_verdict.json")["E4H_COMPLETE"] == "NO"


def test_T4_T37_v2_rejected_not_promoted():
    _require()
    v = _load(RES / "E4HI_verdict.json")
    assert v["V2_STATUS"] == "REJECTED_NOT_TRAINED"


def test_T5_v3_name():
    assert NORM_V3 == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T6_T7_source_integrity():
    _require()
    si = _load(RES / "E4HI_source_integrity.json")
    assert si["fit_states_hash_match"] is True
    assert si["v3_constants_sha256_recorded"] == \
        _load(E4HR / "E4HR_v3_constants.json")["constants_sha256"]


def test_T10_no_refit():
    _require()
    assert _load(RES / "E4HI_pre_manifest.json")["no_refit"] is True


# ---- artifact gate results ----

def test_T12_impl_reference_pass():
    _require()
    d = _load(RES / "E4HI_impl_reference_agreement.json")
    assert d["IMPLEMENTATION_REFERENCE_MATCH"] == "PASS"
    assert d["max_abs_diff"] <= 1e-12


def test_T8_T9_raw_audit():
    _require()
    a = _load(RES / "E4HI_raw_state_audit.json")
    assert a["input_dim"] == 9
    assert a["all_finite"] is True


def test_T16_structural_zero_gate():
    _require()
    d = _load(RES / "E4HI_structural_zero_audit.json")
    assert d["STRUCTURAL_ZERO_GATE"] == "PASS"
    for f, r in d["per_feature"].items():
        assert r["zero_preservation_error_count"] == 0


def test_T26_T27_T28_T29_T30_saturation_gate():
    _require()
    d = _load(RES / "E4HI_saturation_gate.json")
    assert d["gate"] == 0.05
    for f, s in d["per_feature"].items():
        assert s["positive_lower_saturation_fraction"] <= 0.05
        assert s["positive_upper_saturation_fraction"] <= 0.05
    assert d["delta_severity"]["nonzero_absolute_saturation_fraction"] <= 0.05
    assert d["ZERO_AWARE_SATURATION_GATE"] == "PASS"


def test_T23_T24_T25_signal_gate_consistency():
    _require()
    d = _load(RES / "E4HI_signal_gate.json")
    for feat, key in [("severity_r", "SEVERITY_SIGNAL_V3"),
                      ("convergence_loss_r", "CONVERGENCE_LOSS_SIGNAL_V3"),
                      ("delta_severity_r", "DELTA_SEVERITY_SIGNAL_V3")]:
        pf = d["per_feature"][feat]
        expect = "PASS" if (pf["range"] >= 0.05 and pf["std"] >= 0.02) else "FAIL"
        assert d[key] == expect


def test_T31_T32_T33_T34_T35_T36_separability():
    _require()
    d = _load(RES / "E4HI_separability_gate.json")
    rows = list(csv.DictReader(open(RES / "E4HI_separability_probe_results.csv")))
    assert {r["representation"] for r in rows} == {"RAW", "V1", "V2", "V3"}
    assert {r["probe"] for r in rows} == {"logistic", "tree", "kNN"}
    if d["V3_SEPARABILITY_VS_V1"] == "IMPROVED":
        assert d["mean_balanced_accuracy"]["V3"] >= d["mean_balanced_accuracy"]["V1"]


def test_T38_T39_semantic_invariance():
    _require()
    d = _load(RES / "E4HI_semantic_invariance.json")
    assert d["SEMANTIC_INVARIANCE"] == "PASS"
    assert d["state_dim"] == 9


def test_T48_T49_T50_T51_future_firewall():
    _require()
    fw = _load(RES / "E4HI_future_seed_firewall.json")
    assert fw["new_train_seed_access"] == 0
    assert fw["new_validation_seed_access"] == 0
    assert fw["new_final_eval_access_count"] == 0


def test_T52_T53_T54_T55_T56_no_training():
    _require()
    nt = _load(RES / "E4HI_no_training_audit.json")
    assert nt["optimizer_step_count"] == 0
    assert nt["gradient_update_count"] == 0
    assert nt["replay_append_count"] == 0
    assert nt["new_checkpoint_count"] == 0
    assert nt["model_weights_changed"] is False


def test_T57_historical_artifacts_unchanged():
    _require()
    frz = _load("results/E4E_final_evaluation_protocol_freeze/E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]
    for d in ["results/E4TR_shared_double_dqn_training", "results/E4F_final_evaluation",
              "results/E4GD_controlled_diagnostic_execution",
              "results/E4H_state_normalization_revision_design",
              "results/E4HR_normalization_design_review"]:
        assert Path(d).exists()


def test_T58_readiness_logic_exact():
    _require()
    v = _load(RES / "E4HI_verdict.json")
    ready = all(v["gates"].values())
    assert (v["NORMALIZATION_V3_READY"] == "YES") == ready
    expect = "E4HT_NORMALIZATION_AB_TRAINING_PROTOCOL_FREEZE" if ready else "E4HR_DESIGN_FIX_REQUIRED"
    assert v["NEXT_STAGE"] == expect
    assert v["E4HI_COMPLETE"] == "YES"   # stage completed regardless of readiness


def test_no_schema_or_model_change():
    _require()
    v = _load(RES / "E4HI_verdict.json")
    assert v["STATE_DIM"] == 9
    assert v["NETWORK_CHANGED"] == "NO"
    assert v["REWARD_CHANGED"] == "NO"
    assert v["JOINT_ARCHITECTURE_CHANGED"] == "NO"
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in \
        Path("src/prdrl/rl/reward.py").read_text()


def test_v2_included_only_as_diagnostic():
    _require()
    # V2 appears in separability comparison but is never the treatment/next stage
    v = _load(RES / "E4HI_verdict.json")
    assert v["V2_STATUS"] == "REJECTED_NOT_TRAINED"
    if v["NORMALIZATION_V3_READY"] == "YES":
        assert "V3" in v["NEXT_STAGE"] or "AB_TRAINING" in v["NEXT_STAGE"]
