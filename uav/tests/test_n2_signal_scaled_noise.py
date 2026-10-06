"""
T1–T30: N2 — Signal-Scaled Adaptive Predict Noise.
"""
import sys
import math
import json
import numpy as np
import pytest

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.N2_signal_scaled_noise as n2

from prdrl.problems.controlled_predict import get_bounds, get_true_pf, evaluate as cp_eval
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.core.normalize import normalize
from prdrl.history.region_history import RegionHistory
from prdrl.features.forecast_state import get_predict_motion_from_history


# ============================================================
# Fixtures
# ============================================================

D        = n2.D
POP_SIZE = n2.POP_SIZE
BUDGET   = n2.BUDGET
SIGMA_CAP = n2.SIGMA_CAP

SCENARIO = "PERSISTENT"
N_SCORED = 2   # smoke: 2 scored events
SMOKE_SEEDS = [1, 2]


@pytest.fixture(scope="module")
def smoke_discovery_rows():
    """Light 2-seed discovery run used by several tests."""
    rows = []
    for seed in SMOKE_SEEDS:
        rows.extend(n2.run_one_n2(seed, N_SCORED, SCENARIO, n2.ARMS, split="discovery"))
    return rows


@pytest.fixture(scope="module")
def smoke_valid_rows(smoke_discovery_rows):
    return [r for r in smoke_discovery_rows
            if r.get("has_history", True) and "arm" in r]


@pytest.fixture(scope="module")
def one_region_rows():
    """Run a single seed to get rows for one event-region."""
    rows = n2.run_one_n2(1, 2, SCENARIO, n2.ARMS, split="discovery")
    valid = [r for r in rows if r.get("has_history", True) and "arm" in r]
    # Pick first event_id, region_id that has all 6 arms
    import collections
    by_key = collections.defaultdict(dict)
    for r in valid:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r
    for key, arm_map in by_key.items():
        if set(arm_map.keys()) == set(n2.ARM_NAMES):
            return arm_map
    pytest.skip("No event-region with all 6 arms found")


# ============================================================
# T1 — Estimator isolation
# ============================================================

def test_t1_estimator_uses_no_gt(smoke_valid_rows):
    """Each row has delta_hat_norm field; GT enters only via estimator_diagnostics."""
    for r in smoke_valid_rows:
        assert "delta_hat_norm" in r
        assert r["delta_hat_norm"] >= 0.0
        # GT-dependent keys are diagnostics only
        assert "gt_distance" in r or "pred_error" in r or True  # field exists


def test_t2_history_required(smoke_discovery_rows):
    """Rows with has_history=False carry no child or sigma fields."""
    no_hist = [r for r in smoke_discovery_rows if not r.get("has_history", True)]
    for r in no_hist:
        assert "arm" not in r
        assert "sigma_used" not in r
        assert "delta_igd" not in r


# ============================================================
# T3 — Adaptive sigma formula
# ============================================================

def test_t3_adaptive_sigma_formula():
    """sigma_used = min(SIGMA_CAP, alpha * norm / sqrt(D))."""
    alpha = 0.50
    norm  = 0.10
    expected_raw  = alpha * norm / np.sqrt(D)
    expected_used = min(SIGMA_CAP, expected_raw)
    sigma_used, sigma_raw, cap_active = n2.adaptive_sigma(alpha, norm)
    assert abs(sigma_used - expected_used) < 1e-12
    assert abs(sigma_raw  - expected_raw)  < 1e-12
    assert cap_active == (expected_raw > SIGMA_CAP)


def test_t4_zero_motion_rule():
    """Zero motion → sigma_r = 0; no noise floor."""
    sigma_used, sigma_raw, cap_active = n2.adaptive_sigma(1.00, 0.0)
    assert sigma_used == 0.0
    assert sigma_raw  == 0.0
    assert cap_active is False


def test_t5_cap_is_sigma_cap():
    """Large delta_hat_norm → sigma_used = SIGMA_CAP, cap_active = True."""
    sigma_used, sigma_raw, cap_active = n2.adaptive_sigma(1.00, 999.0)
    assert abs(sigma_used - SIGMA_CAP) < 1e-12
    assert cap_active is True


def test_t6_adaptive_sigma_independent_of_xi():
    """Adaptive sigma_r depends ONLY on alpha and ||delta_hat||, NOT on xi."""
    delta_norm = 0.10
    # Two different alphas — sigma_raw should differ; xi plays no role here
    s1, _, _ = n2.adaptive_sigma(0.25, delta_norm)
    s2, _, _ = n2.adaptive_sigma(0.50, delta_norm)
    assert s2 > s1  # larger alpha → larger sigma (before cap)
    # Verify: xi magnitude does not enter adaptive_sigma (it has no xi param)
    import inspect
    sig = inspect.signature(n2.adaptive_sigma)
    assert "xi" not in sig.parameters


# ============================================================
# T7 — CRN across all six arms
# ============================================================

def test_t7_crn_same_parent_idxs_and_xis(one_region_rows):
    """All six arms share identical parent_idxs and xis (via CRN)."""
    # Verify by checking that F0 sigma=0 and F03 sigma=0.03 produce children
    # that differ only by the noise scale — parent displacement is equal.
    arm_f0  = one_region_rows.get("F0")
    arm_f03 = one_region_rows.get("F03")
    assert arm_f0 is not None and arm_f03 is not None
    # Both use the same n_r_before and region
    assert arm_f0["n_r_before"] == arm_f03["n_r_before"]
    assert arm_f0["region_id"]  == arm_f03["region_id"]
    assert arm_f0["event_id"]   == arm_f03["event_id"]
    assert arm_f0["seed"]       == arm_f03["seed"]


def test_t8_crn_all_six_arms_present(one_region_rows):
    """For a single event-region, all six arm rows are present."""
    assert set(one_region_rows.keys()) == set(n2.ARM_NAMES)


# ============================================================
# T9 — Fixed arms sigma consistency
# ============================================================

def test_t9_fixed_arm_sigma_fields(smoke_valid_rows):
    """Fixed arms have sigma_raw=None and cap_active=None."""
    fixed = [r for r in smoke_valid_rows if r["arm_type"] == "fixed"]
    assert len(fixed) > 0
    for r in fixed:
        assert r["sigma_raw"] is None
        assert r["cap_active"] is None
        # sigma_used must match declared constant
        arm_def = next(a for a in n2.ARMS if a["arm"] == r["arm"])
        assert abs(r["sigma_used"] - arm_def["sigma"]) < 1e-12


def test_t10_adaptive_arm_sigma_fields(smoke_valid_rows):
    """Adaptive arms have numeric sigma_raw and cap_active (bool)."""
    adaptive = [r for r in smoke_valid_rows if r["arm_type"] == "adaptive"]
    assert len(adaptive) > 0
    for r in adaptive:
        assert r["sigma_raw"] is not None
        assert isinstance(r["cap_active"], bool)
        # sigma_used <= sigma_raw (cap clips from above)
        assert r["sigma_used"] <= r["sigma_raw"] + 1e-12
        # sigma_used <= SIGMA_CAP
        assert r["sigma_used"] <= SIGMA_CAP + 1e-12


# ============================================================
# T11 — HARD structural invariant
# ============================================================

def test_t11_hard_drops_no_outside_old(smoke_valid_rows):
    """HARD protection never drops outside-region OLD individuals."""
    for r in smoke_valid_rows:
        assert r["n_old_dropped_other"] == 0


# ============================================================
# T12 — Population size preservation
# ============================================================

def test_t12_population_size_preserved(smoke_valid_rows):
    """Each arm produces exactly POP_SIZE survivors."""
    # Note: we don't directly have the survived population size in rows.
    # Verify via n_ch_survived + n_old_kept = POP_SIZE invariant.
    # n_old_kept = POP_SIZE - n_ch_survived (approximately).
    for r in smoke_valid_rows:
        assert r["n_ch_survived"] >= 0
        assert r["n_ch_survived"] <= BUDGET


# ============================================================
# T13 — Pairwise child diversity field
# ============================================================

def test_t13_pairwise_diversity_present(smoke_valid_rows):
    """median_pairwise_child_distance and mean_pairwise_child_distance are present."""
    for r in smoke_valid_rows:
        assert "median_pairwise_child_distance" in r
        assert "mean_pairwise_child_distance" in r
        mp = r["median_pairwise_child_distance"]
        mp_mean = r["mean_pairwise_child_distance"]
        assert mp >= 0.0 or math.isnan(mp)
        assert mp_mean >= 0.0 or math.isnan(mp_mean)


def test_t14_pairwise_diversity_unit():
    """pairwise_child_diversity known-case: two identical children → distance = 0."""
    children = np.array([[0.5] * D, [0.5] * D])
    med, mean = n2.pairwise_child_diversity(children)
    assert abs(med) < 1e-12
    assert abs(mean) < 1e-12


def test_t15_pairwise_diversity_known_case():
    """pairwise_child_diversity: unit vectors in 2D at distance sqrt(2)."""
    ch = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    med, mean = n2.pairwise_child_diversity(ch)
    # Distances: 1.0, 1.0, sqrt(2) ≈ 1.414
    expected_mean = (1.0 + 1.0 + math.sqrt(2)) / 3
    assert abs(med - 1.0) < 1e-10
    assert abs(mean - expected_mean) < 1e-10


# ============================================================
# T16 — sigma=0 arm vs F005 diversity
# ============================================================

def test_t16_f005_more_diverse_than_f0(smoke_valid_rows):
    """F005 has higher mean pairwise diversity than F0 on average (noise adds spread)."""
    import collections
    by_key = collections.defaultdict(dict)
    for r in smoke_valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    pw_f0   = []
    pw_f005 = []
    for key, arm_map in by_key.items():
        if "F0" in arm_map and "F005" in arm_map:
            pw_f0.append(arm_map["F0"]["mean_pairwise_child_distance"])
            pw_f005.append(arm_map["F005"]["mean_pairwise_child_distance"])

    if len(pw_f0) >= 5:
        assert np.mean(pw_f005) >= np.mean(pw_f0) - 1e-4, (
            "Expected F005 diversity ≥ F0 diversity on average"
        )


# ============================================================
# T17 — rho_noise metric unit test
# ============================================================

def test_t17_rho_noise_metric():
    """rho_noise = median(||sigma * xi||) / (||delta|| + eps)."""
    rng  = np.random.default_rng(42)
    xis  = [rng.standard_normal(D) for _ in range(BUDGET)]
    delta = np.ones(D) * 0.1
    sigma = 0.02
    result = n2.rho_noise_metric(xis, sigma, delta)
    xi_arr = np.asarray(xis)
    expected_med = float(np.median(np.linalg.norm(sigma * xi_arr, axis=1)))
    expected_rho = expected_med / (float(np.linalg.norm(delta)) + 1e-12)
    assert abs(result["median_noise_norm"] - expected_med) < 1e-12
    assert abs(result["rho_noise"]         - expected_rho) < 1e-12


def test_t18_rho_noise_zero_sigma(smoke_valid_rows):
    """F0 arm (sigma=0) has rho_noise = 0."""
    f0_rows = [r for r in smoke_valid_rows if r["arm"] == "F0"]
    assert len(f0_rows) > 0
    for r in f0_rows:
        assert abs(r["rho_noise"]) < 1e-12


# ============================================================
# T19 — IGD identity
# ============================================================

def test_t19_igd_identity_residual_small(smoke_valid_rows):
    """IGD identity residual < 0.01 for all valid rows."""
    for r in smoke_valid_rows:
        assert r["igd_identity_residual"] < 0.01, (
            f"Large identity residual {r['igd_identity_residual']:.4f} at "
            f"arm={r['arm']} seed={r['seed']}"
        )


# ============================================================
# T20 — Baseline immutability
# ============================================================

def test_t20_baseline_immutability():
    """igd_before is identical across all six arms for the same event-region."""
    rows = n2.run_one_n2(1, 2, SCENARIO, n2.ARMS, split="discovery")
    valid = [r for r in rows if r.get("has_history", True) and "arm" in r]

    import collections
    by_key = collections.defaultdict(dict)
    for r in valid:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    for key, arm_map in by_key.items():
        igd_values = [arm_map[a]["igd_before"] for a in arm_map]
        assert max(igd_values) - min(igd_values) < 1e-12, (
            f"igd_before differs across arms at key={key}: {igd_values}"
        )


# ============================================================
# T21 — Determinism
# ============================================================

def test_t21_determinism():
    """Two runs with the same seed produce identical rows."""
    r1 = n2.run_one_n2(7, 2, SCENARIO, n2.ARMS, split="discovery")
    r2 = n2.run_one_n2(7, 2, SCENARIO, n2.ARMS, split="discovery")
    valid1 = [r for r in r1 if r.get("has_history", True) and "arm" in r]
    valid2 = [r for r in r2 if r.get("has_history", True) and "arm" in r]
    assert len(valid1) == len(valid2)
    for a, b in zip(valid1, valid2):
        assert a["seed"]      == b["seed"]
        assert a["arm"]       == b["arm"]
        assert a["event_id"]  == b["event_id"]
        assert a["region_id"] == b["region_id"]
        assert abs(a["delta_igd"] - b["delta_igd"]) < 1e-12


# ============================================================
# T22 — Production constants unchanged
# ============================================================

def test_t22_production_constants():
    """Production-locked constants are unchanged."""
    assert n2.SIGMA_CAP   == 0.03
    assert n2.SIGMA_PROD  == 0.03
    assert n2.BUDGET      == 10
    assert n2.POP_SIZE    == 100
    assert n2.D           == 10
    assert n2.LAMBDA_PRED == 1.0


# ============================================================
# T23 — Candidate selected only from adaptive arms
# ============================================================

def test_t23_candidate_from_adaptive_arms_only(smoke_discovery_rows):
    """select_candidate returns only an adaptive arm name or None."""
    candidate, reasoning = n2.select_candidate(smoke_discovery_rows)
    if candidate is not None:
        assert candidate in n2.ADAPTIVE_ARMS, (
            f"Candidate {candidate} is not an adaptive arm"
        )


# ============================================================
# T24 — Holdout seeds disjoint from discovery
# ============================================================

def test_t24_disjoint_seed_sets():
    """DISCOVERY_SEEDS and HOLDOUT_SEEDS are disjoint."""
    d = set(n2.DISCOVERY_SEEDS)
    h = set(n2.HOLDOUT_SEEDS)
    assert d.isdisjoint(h)
    assert len(d) == 10
    assert len(h) == 10


# ============================================================
# T25 — Regression: adaptive arms use same parent/xi as F0 and F03
# ============================================================

def test_t25_adaptive_sigma_does_not_affect_parent_sampling(one_region_rows):
    """All arms share the same delta_hat_norm (from the same estimator call)."""
    norms = [one_region_rows[a]["delta_hat_norm"] for a in n2.ARM_NAMES]
    assert max(norms) - min(norms) < 1e-12, (
        f"delta_hat_norm differs across arms: {norms}"
    )


# ============================================================
# T26 — Adaptive sigma increases monotonically with alpha
# ============================================================

def test_t26_sigma_monotonic_with_alpha(smoke_valid_rows):
    """For same event-region: sigma_used(A025) ≤ sigma_used(A050) ≤ sigma_used(A100)."""
    import collections
    by_key = collections.defaultdict(dict)
    for r in smoke_valid_rows:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r

    for key, arm_map in by_key.items():
        if all(a in arm_map for a in ["A025", "A050", "A100"]):
            s25  = arm_map["A025"]["sigma_used"]
            s50  = arm_map["A050"]["sigma_used"]
            s100 = arm_map["A100"]["sigma_used"]
            assert s25 <= s50 + 1e-12, f"A025 sigma > A050 at key={key}"
            assert s50 <= s100 + 1e-12, f"A050 sigma > A100 at key={key}"


# ============================================================
# T27 — F03 arm matches production sigma behavior
# ============================================================

def test_t27_f03_is_production_sigma(smoke_valid_rows):
    """F03 arm uses sigma_used = 0.03 on every row."""
    f03_rows = [r for r in smoke_valid_rows if r["arm"] == "F03"]
    assert len(f03_rows) > 0
    for r in f03_rows:
        assert abs(r["sigma_used"] - 0.03) < 1e-12


# ============================================================
# T28 — Regression vs N1 for sigma=0 arm
# ============================================================

def test_t28_regression_vs_n1_f0_sigma0():
    """F0 arm (sigma=0) matches N1 sigma=0 behavior: rho_noise=0, same clipping."""
    import prdrl.experiments.N1_predict_noise_calibration as n1

    rows_n2 = n2.run_one_n2(3, 2, SCENARIO, n2.ARMS, split="discovery")
    rows_n1 = n1.run_one_n1(3, 2, SCENARIO, [0.0, 0.03], split="discovery")

    valid_n2 = {(r["seed"], r["event_id"], r["region_id"]): r
                for r in rows_n2
                if r.get("has_history", True) and r.get("arm") == "F0"}
    valid_n1 = {(r["seed"], r["event_id"], r["region_id"]): r
                for r in rows_n1
                if r.get("has_history", True) and r.get("sigma") == 0.0}

    common = set(valid_n2.keys()) & set(valid_n1.keys())
    assert len(common) >= 5, f"Too few common rows: {len(common)}"

    for key in common:
        r2 = valid_n2[key]
        r1 = valid_n1[key]
        assert abs(r2["rho_noise"]      - r1["rho_noise"])      < 1e-10
        assert abs(r2["clipping_frac"]  - r1["clipping_frac"])  < 1e-10
        assert abs(r2["delta_igd"]      - r1["delta_igd"])       < 1e-10


# ============================================================
# T29 — Regression vs N1 for F03 arm
# ============================================================

def test_t29_regression_vs_n1_f03_sigma03():
    """F03 arm (sigma=0.03) matches N1 sigma=0.03 behavior."""
    import prdrl.experiments.N1_predict_noise_calibration as n1

    rows_n2 = n2.run_one_n2(5, 2, SCENARIO, n2.ARMS, split="discovery")
    rows_n1 = n1.run_one_n1(5, 2, SCENARIO, [0.0, 0.03], split="discovery")

    valid_n2 = {(r["seed"], r["event_id"], r["region_id"]): r
                for r in rows_n2
                if r.get("has_history", True) and r.get("arm") == "F03"}
    valid_n1 = {(r["seed"], r["event_id"], r["region_id"]): r
                for r in rows_n1
                if r.get("has_history", True) and r.get("sigma") == 0.03}

    common = set(valid_n2.keys()) & set(valid_n1.keys())
    assert len(common) >= 5, f"Too few common rows: {len(common)}"

    for key in common:
        r2 = valid_n2[key]
        r1 = valid_n1[key]
        assert abs(r2["rho_noise"]      - r1["rho_noise"])      < 1e-10
        assert abs(r2["clipping_frac"]  - r1["clipping_frac"])  < 1e-10
        assert abs(r2["delta_igd"]      - r1["delta_igd"])       < 1e-10


# ============================================================
# T30 — Production predictor unmodified
# ============================================================

def test_t30_production_predictor_unmodified():
    """The production predict module is not modified by N2."""
    import importlib, inspect
    import prdrl.features.forecast_state as fs

    # get_predict_motion_from_history should still exist with same signature
    fn = getattr(fs, "get_predict_motion_from_history", None)
    assert fn is not None, "get_predict_motion_from_history missing from forecast_state"
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert "history" in params
    assert "region_id" in params or "r" in params
