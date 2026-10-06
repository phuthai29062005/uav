"""
E4HR — Zero-Aware Normalization Design Review: unit + contract + artifact tests.

Verify V3 is correctly designed (structural zero preserved, positive spread,
monotonic, delta sign/zero), fit train-only, gates frozen zero-aware, A/B updated
to V1-vs-V3, and that E4F=FAILED / E4H=NO / V2=REJECTED are all preserved.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER,
    ZERO_INFLATED_DIMS, DELTA_DIM, MIN_POSITIVE_SAMPLES, check_normalization_compatibility,
)

RES = Path("results/E4HR_normalization_design_review")
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
    if not (RES / "E4HR_verdict.json").exists():
        pytest.skip("E4HR not run yet")


def _make_fit(seed=0):
    # zero-inflated nonneg fixture: ~40% zeros + positive tail
    rng = np.random.default_rng(seed)
    n = 5000
    X = np.zeros((n, 9))
    for d in ZERO_INFLATED_DIMS:
        mask = rng.random(n) > 0.4
        X[mask, d] = rng.exponential(0.01, mask.sum())
    X[:, 2] = rng.uniform(0, 0.3, n)          # occupancy
    X[:, 4] = 1.0                              # active fraction
    X[:, DELTA_DIM] = rng.normal(0, 0.002, n)  # signed delta
    return X


# ---- outcome-independent unit tests ----

def test_T5_v3_version_name():
    assert NORM_V3 == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T14_T15_zero_to_zero_positive_to_positive():
    v3 = ZeroAwareRobustLogNormalizerV3.fit(_make_fit())
    z0 = v3.transform(np.zeros(9))
    for d in ZERO_INFLATED_DIMS:
        assert abs(z0[d]) < 1e-12                       # structural zero -> 0
    xp = np.zeros(9); xp[0] = 1e-7; xp[1] = 1e-7
    zp = v3.transform(xp)
    assert zp[0] > 0 and zp[1] > 0                       # positive -> > 0


def test_T16_nonneg_monotonic():
    v3 = ZeroAwareRobustLogNormalizerV3.fit(_make_fit())
    for d in ZERO_INFLATED_DIMS:
        xs = np.linspace(0, 5, 50); base = np.zeros((50, 9)); base[:, d] = xs
        z = v3.transform_batch(base)[:, d]
        assert np.all(np.diff(z) >= -1e-12)


def test_T17_T18_constants_deterministic():
    X = _make_fit(1)
    a = ZeroAwareRobustLogNormalizerV3.fit(X)
    b = ZeroAwareRobustLogNormalizerV3.fit(X)
    assert a.constants_sha256() == b.constants_sha256()
    # median/q99 match numpy on positive subset
    col = X[:, 0]; pos = col[col > 0]
    assert abs(a.to_dict()["zero_aware"]["0"]["m"] - float(np.median(pos))) < 1e-12
    assert abs(a.to_dict()["zero_aware"]["0"]["c"] - float(np.percentile(pos, 99))) < 1e-12


def test_T19_T20_delta_zero_and_sign():
    v3 = ZeroAwareRobustLogNormalizerV3.fit(_make_fit())
    assert abs(v3.transform(np.zeros(9))[DELTA_DIM]) < 1e-12
    dp = np.zeros(9); dp[DELTA_DIM] = 0.3
    dn = np.zeros(9); dn[DELTA_DIM] = -0.3
    assert v3.transform(dp)[DELTA_DIM] > 0 and v3.transform(dn)[DELTA_DIM] < 0


def test_T21_T22_identity_dims():
    v3 = ZeroAwareRobustLogNormalizerV3.fit(_make_fit())
    oc = np.zeros(9); oc[2] = 0.31
    ac = np.zeros(9); ac[4] = 0.44
    assert abs(v3.transform(oc)[2] - 0.31) < 1e-9
    assert abs(v3.transform(ac)[4] - 0.44) < 1e-9


def test_degenerate_detection_and_no_substitution():
    X = _make_fit()
    X[:, 1] = 0.0  # all-zero convergence_loss -> no positive samples -> invalid
    v3 = ZeroAwareRobustLogNormalizerV3.fit(X)
    assert "convergence_loss_r" in v3.to_dict()["invalid_features"]
    assert v3.to_dict()["V3_SCALER_VALID"] == "NO"


def test_min_positive_samples_frozen():
    assert MIN_POSITIVE_SAMPLES == 100


def test_T45_version_mismatch_raises_v3():
    with pytest.raises(ValueError):
        check_normalization_compatibility({}, NORM_V3, "abc")  # legacy ckpt, expect V3 -> raise
    with pytest.raises(ValueError):
        check_normalization_compatibility(
            {"normalization_version": NORM_V3, "normalizer_constants_sha256": "x"}, NORM_V3, "y")


# ---- integrity / preservation ----

def test_T1_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T2_e4h_no():
    assert _load(E4H / "E4H_verdict.json")["E4H_COMPLETE"] == "NO"


def test_T3_T4_v2_rejected_no_retroactive_pass():
    _require()
    p = _load(RES / "E4HR_e4h_failure_preservation.json")
    assert p["V2_STATUS"] == "REJECTED_NOT_TRAINED"
    assert p["no_retroactive_v2_pass"] is True
    assert p["E4H_ORIGINAL_RESULT"] == "NO"


# ---- artifact tests ----

def test_T6_T7_T8_T9_fit_train_only():
    _require()
    c = _load(RES / "E4HR_v3_constants.json")
    assert c["fit_seed_range"] == [1001, 1060]
    v = _load(RES / "E4HR_verdict.json")
    assert v["FIT_LEAKAGE"] == "NONE"
    assert v["V3_CONSTANTS_FIT_SEEDS"] == "1001..1060"


def test_T10_T11_structural_zero_separated():
    _require()
    sat = _load(RES / "E4HR_zero_aware_saturation_contract.json")
    assert sat["distinguishes_structural_zero_from_clipping"] is True
    assert "REPORTED, not gated" in sat["STRUCTURAL_ZERO_FRACTION"]


def test_T12_T13_clip_definitions_exact():
    _require()
    sat = _load(RES / "E4HR_zero_aware_saturation_contract.json")
    assert "raw x>0" in sat["POSITIVE_LOWER_CLIP_FRACTION"]
    assert "raw x>0" in sat["POSITIVE_UPPER_CLIP_FRACTION"]
    assert sat["gate"] == 0.05


def test_zero_inflation_audit_present():
    _require()
    a = _load(RES / "E4HR_zero_inflation_audit.json")
    for d in ZERO_INFLATED_DIMS:
        f = RAW_ORDER[d]
        assert f in a["per_feature"]
        assert "zero_fraction" in a["per_feature"][f]
        assert "n_positive" in a["per_feature"][f]


def test_T23_T24_T25_T26_no_schema_or_model_change():
    _require()
    v = _load(RES / "E4HR_verdict.json")
    assert v["STATE_DIM"] == 9
    assert v["STATE_SCHEMA_CHANGED"] == "NO"
    assert v["NETWORK_CHANGED"] == "NO"
    assert v["REWARD_CHANGED"] == "NO"
    assert v["JOINT_ARCHITECTURE_CHANGED"] == "NO"
    # reward formula + network dim still intact
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in \
        Path("src/prdrl/rl/reward.py").read_text()


def test_T27_T28_T29_future_control_treatment():
    _require()
    ab = _load(RES / "E4HR_ab_contract_supersession.json")
    assert ab["control_arm"]["normalization_version"] == NORM_V1
    assert ab["treatment_arm"]["normalization_version"] == NORM_V3
    assert ab["v2_not_a_training_candidate"] is True
    assert ab["only_experimental_factor"] == "normalization_version"


def test_T30_T31_T32_T33_future_seeds():
    _require()
    s = _load(RES / "E4HR_future_seed_contract.json")
    assert s["train"]["seeds"] == list(range(1301, 1361))
    assert s["validation"]["seeds"] == list(range(1361, 1381))
    assert s["final"]["seeds"] == list(range(1381, 1441))
    assert s["policy_seeds"] == [3001, 3002, 3003, 3004, 3005]
    assert s["disjoint_from_historical"] is True


def test_T34_T35_no_final_access_no_training():
    _require()
    v = _load(RES / "E4HR_verdict.json")
    assert v["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert v["NEW_TRAINING"] == "NO"


def test_T36_historical_artifacts_unchanged():
    _require()
    for d in ["results/E4TR_shared_double_dqn_training", "results/E4E_final_evaluation_protocol_freeze",
              "results/E4F_final_evaluation", "results/E4GD_controlled_diagnostic_execution",
              "results/E4H_state_normalization_revision_design"]:
        assert Path(d).exists()
    # old checkpoints untouched
    frz = _load("results/E4E_final_evaluation_protocol_freeze/E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_versioning_contract():
    _require()
    vc = _load(RES / "E4HR_versioning_contract.json")
    assert vc["old_checkpoints_use"] == NORM_V1
    assert vc["no_old_checkpoint_uses_V3"] is True
    assert vc["versions"]["V2"]["status"].startswith("REJECTED")


def test_semantic_invariance_and_next_stage():
    _require()
    s = _load(RES / "E4HR_semantic_invariance_contract.json")
    v = _load(RES / "E4HR_verdict.json")
    if v["V3_SCALER_VALID"] == "YES":
        assert s["all_invariants_pass"] is True
    ready = v["E4HR_COMPLETE"] == "YES"
    expect = "E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready else "E4HR_DESIGN_FIX_REQUIRED"
    assert v["NEXT_STAGE"] == expect


def test_gates_frozen_before_validation():
    _require()
    for f in ["E4HR_zero_aware_saturation_contract.json", "E4HR_signal_gate_contract.json",
              "E4HR_separability_gate_contract.json"]:
        d = _load(RES / f)
        assert d["frozen_before_validation"] is True
