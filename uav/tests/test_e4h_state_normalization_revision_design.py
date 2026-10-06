"""
E4H — State Normalization Revision Design: unit + contract + artifact tests.

Unit tests exercise the versioned normalizer directly (outcome-independent).
Artifact tests verify the frozen design: single-factor (normalization only), fit
from training seeds only, gates recorded consistently, fresh seed splits, and that
no historical artifact / the E4F verdict changed.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4H_state_normalization_revision_design as E4H
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, normalize_v1, RobustQuantileNormalizerV2, RAW_ORDER,
    ROBUST_MINMAX_DIMS, DELTA_DIM, check_normalization_compatibility,
)

RES = Path("results/E4H_state_normalization_revision_design")
E4F = Path("results/E4F_final_evaluation")
E4GD = Path("results/E4GD_controlled_diagnostic_execution")
E4E = Path("results/E4E_final_evaluation_protocol_freeze")


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
    if not (RES / "E4H_verdict.json").exists():
        pytest.skip("E4H not run yet")


# ---- outcome-independent unit tests (normalizer) ----

def test_T9_v1_matches_e4p():
    from prdrl.experiments.E4P_learned_controller_preparation import (
        normalize_local_state, normalize_global_context)
    rng = np.random.default_rng(0)
    for _ in range(300):
        sev, cl, occ, ds = rng.uniform(-1, 12), rng.uniform(-1, 15), rng.uniform(0, 1.2), rng.uniform(-2, 2)
        g = rng.uniform(0, 12, size=5)
        v2 = {"severity_r": sev, "convergence_loss_r": cl, "occupancy_r": occ, "delta_severity_r": ds}
        e4p = np.concatenate([normalize_local_state(v2), normalize_global_context(g)])
        assert np.allclose(normalize_v1(np.array([sev, cl, occ, ds, *g])), e4p, atol=1e-6)


def test_T7_T8_state_features_and_dim():
    assert RAW_ORDER == E4H.LOCAL_FEATURES + E4H.GLOBAL_FEATURES
    assert len(RAW_ORDER) == 9


def test_T14_T15_fit_deterministic():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 5, size=(2000, 9))
    n1 = RobustQuantileNormalizerV2.fit(X)
    n2 = RobustQuantileNormalizerV2.fit(X)
    assert n1.constants_sha256() == n2.constants_sha256()


def test_T16_T17_identity_dims():
    X = np.zeros((3, 9)); X[:, 2] = [0.1, 0.5, 0.9]; X[:, 4] = [0.2, 0.4, 0.8]
    n = RobustQuantileNormalizerV2.fit(np.random.default_rng(2).uniform(0, 5, (500, 9)))
    Z = n.transform_batch(X)
    assert np.allclose(Z[:, 2], [0.1, 0.5, 0.9])
    assert np.allclose(Z[:, 4], [0.2, 0.4, 0.8])


def test_T18_T19_delta_zero_and_sign():
    n = RobustQuantileNormalizerV2.fit(np.random.default_rng(3).uniform(-2, 5, (500, 9)))
    assert abs(n.transform(np.array([0, 0, 0, 0.0, 0, 0, 0, 0, 0]))[DELTA_DIM]) < 1e-9
    assert n.transform(np.array([0, 0, 0, 0.4, 0, 0, 0, 0, 0]))[DELTA_DIM] > 0
    assert n.transform(np.array([0, 0, 0, -0.4, 0, 0, 0, 0, 0]))[DELTA_DIM] < 0


def test_T20_robust_monotonic():
    n = RobustQuantileNormalizerV2.fit(np.random.default_rng(4).uniform(0, 5, (2000, 9)))
    for d in ROBUST_MINMAX_DIMS:
        xs = np.linspace(-1, 12, 40)
        base = np.zeros((40, 9)); base[:, d] = xs
        z = n.transform_batch(base)[:, d]
        assert np.all(np.diff(z) >= -1e-12)


def test_T45_version_mismatch_raises():
    # old checkpoint (no field) treated as V1; expecting V2 must raise
    with pytest.raises(ValueError):
        check_normalization_compatibility({}, NORM_V2, "abc")
    # V1 expected on legacy checkpoint passes
    check_normalization_compatibility({}, NORM_V1)
    # V2 constants mismatch raises
    with pytest.raises(ValueError):
        check_normalization_compatibility(
            {"normalization_version": NORM_V2, "normalizer_constants_sha256": "x"}, NORM_V2, "y")


def test_degenerate_detection():
    X = np.random.default_rng(5).uniform(0, 5, (500, 9))
    X[:, 1] = 3.0  # constant convergence_loss -> degenerate
    n = RobustQuantileNormalizerV2.fit(X)
    assert "convergence_loss_r" in n.to_dict()["degenerate_features"]


# ---- integrity ----

def test_T1_e4f_verdict_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T2_e4f_verdict_unchanged():
    # hash matches the one E4GD recorded (proves unchanged since E4GD)
    e4gd_man = _load(E4GD / "E4GD_pre_manifest.json")
    assert e4gd_man["e4f_verdict_hash"] == _hash(E4F / "E4F_verdict.json")


def test_T3_T4_e4gd_branch_a():
    v = _load(E4GD / "E4GD_verdict.json")
    assert v["E4GD_COMPLETE"] == "YES"
    assert v["SELECTED_BRANCH"] == "A_STATE_NORMALIZATION"


def test_T5_T6_no_branch_b_or_c_change():
    # reward formula unchanged (no B); network input dim unchanged (no C)
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in \
        Path("src/prdrl/rl/reward.py").read_text()
    import torch
    from prdrl.rl.network import SharedRegionalQNetwork
    net = SharedRegionalQNetwork(9)
    lin = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert lin[0].in_features == 9 and lin[-1].out_features == 4


# ---- artifact tests ----

def test_T10_T11_T12_T13_fit_training_only():
    _require()
    fm = _load(RES / "E4H_fit_data_manifest.json")
    assert fm["fit_seed_range"] == [1001, 1060]
    assert fm["validation_leakage"] is False
    assert fm["final_leakage"] is False
    assert fm["diagnostic_leakage"] is False


def test_T21_no_degenerate_scaler():
    _require()
    c = _load(RES / "E4H_normalizer_v2_constants.json")
    v = _load(RES / "E4H_verdict.json")
    # if degenerate, E4H must not be READY
    if c["degenerate_features"]:
        assert v["NORMALIZATION_V2_READY"] == "NO"


def test_T22_T23_T24_signal_gate_consistency():
    _require()
    sg = _load(RES / "E4H_signal_gate.json")
    # each recorded PASS/FAIL matches the near-constant recomputation stored per feature
    for feat, key in [("severity_r", "SEVERITY_SIGNAL_V2"),
                      ("convergence_loss_r", "CONVERGENCE_LOSS_SIGNAL_V2"),
                      ("delta_severity_r", "DELTA_SEVERITY_SIGNAL_V2")]:
        expect = "PASS" if not sg["per_feature"][feat]["v2_near_constant"] else "FAIL"
        assert sg[key] == expect


def test_T25_saturation_gate_consistency():
    _require()
    sg = _load(RES / "E4H_signal_gate.json")
    assert sg["SATURATION_GATE"] in ("PASS", "FAIL")


def test_T26_T27_separability_consistency():
    _require()
    sep = _load(RES / "E4H_separability_gate.json")
    assert sep["V2_SEPARABILITY_VS_V1"] in ("IMPROVED", "NOT_IMPROVED")
    if sep["V2_SEPARABILITY_VS_V1"] == "IMPROVED":
        assert sep["V2_gap"] < sep["V1_gap"]
    # probes recorded for RAW/V1/V2
    rows = list(csv.DictReader(open(RES / "E4H_separability_raw_v1_v2.csv")))
    if rows:
        assert {r["representation"] for r in rows} == {"RAW", "V1", "V2"}


def test_T28_T29_T30_T31_T32_T33_no_changes():
    _require()
    v = _load(RES / "E4H_verdict.json")
    assert v["NEW_TRAINING_RUN"] == "NO"
    assert v["STATE_SCHEMA_CHANGED"] == "NO"
    assert v["STATE_DIM"] == 9
    assert v["NETWORK_CHANGED"] == "NO"
    assert v["REWARD_CHANGED"] == "NO"
    assert v["JOINT_ARCHITECTURE_CHANGED"] == "NO"
    # old checkpoints unchanged
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_T34_T35_T36_T37_new_seed_split():
    _require()
    s = _load(RES / "E4H_new_experiment_seed_split.json")
    assert s["train"]["seeds"] == list(range(1301, 1361))
    assert s["validation"]["seeds"] == list(range(1361, 1381))
    assert s["final"]["seeds"] == list(range(1381, 1441))
    assert s["disjoint_from_historical"] is True
    for v in s["intersections"].values():
        assert v == []


def test_T38_T39_policy_seeds():
    _require()
    p = _load(RES / "E4H_new_policy_seed_contract.json")
    assert p["new_policy_seeds"] == [3001, 3002, 3003, 3004, 3005]
    assert 2001 not in p["new_policy_seeds"]
    assert "SAME policy seed" in p["paired_across_arms"]


def test_T40_T41_T42_future_contracts():
    _require()
    ft = _load(RES / "E4H_future_training_contract.json")
    assert ft["total_future_training_episodes"] == 10800
    assert ft["no_training_in_E4H"] is True
    cc = _load(RES / "E4H_future_checkpoint_contract.json")
    assert cc["checkpoint_transitions"] == [10000, 20000, 30000, 40000, 50000]
    assert "MACRO" in cc["selection_metric"]


def test_T43_T44_T45_backward_compat():
    _require()
    bc = _load(RES / "E4H_backward_compatibility_contract.json")
    assert bc["old_checkpoints_use"] == NORM_V1
    assert bc["future_v2_checkpoints_use"] == NORM_V2
    assert bc["mismatch_raises"] is True
    assert "normalizer_constants_sha256" in bc["checkpoint_metadata_required"]


def test_T46_new_final_firewall():
    _require()
    fw = _load(RES / "E4H_new_final_firewall.json")
    assert fw["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert fw["unopened"] is True


def test_T47_old_artifacts_present():
    _require()
    for d in ["results/E4TR_shared_double_dqn_training", "results/E4E_final_evaluation_protocol_freeze",
              "results/E4F_final_evaluation", "results/E4GD_controlled_diagnostic_execution"]:
        assert Path(d).exists()


def test_ab_single_factor():
    _require()
    ab = _load(RES / "E4H_ab_experiment_contract.json")
    assert ab["only_experimental_factor"] == "normalization_version"
    assert ab["design"] == "MATCHED_V1_VS_V2 A/B (single factor = normalization_version)"
    assert ab["control_arm"]["normalization_version"] == NORM_V1
    assert ab["treatment_arm"]["normalization_version"] == NORM_V2
    assert ab["identical_across_arms"]["state_dim"] == 9


def test_verdict_next_stage_consistent():
    _require()
    v = _load(RES / "E4H_verdict.json")
    ready = v["NORMALIZATION_V2_READY"] == "YES"
    assert (v["E4H_COMPLETE"] == "YES") == ready
    expect = "E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready else "E4H_NORMALIZATION_DESIGN_REVIEW"
    assert v["NEXT_STAGE"] == expect


def test_semantic_invariance_pass():
    _require()
    s = _load(RES / "E4H_semantic_invariance.json")
    assert s["all_invariants_pass"] is True
    assert s["state_dim_9"] is True
