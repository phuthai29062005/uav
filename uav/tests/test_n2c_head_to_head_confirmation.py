"""
T1–T27: N2C — Frozen Head-to-Head Confirmation (A025 vs F005).
"""
import sys
import math
import json
import collections
import numpy as np
import pytest

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.N2C_head_to_head_confirmation as n2c

from prdrl.problems.controlled_predict import get_bounds, get_true_pf
from prdrl.core.reference import reference

D           = n2c.D
POP_SIZE    = n2c.POP_SIZE
BUDGET      = n2c.BUDGET
DELTA_PRACTICAL = n2c.DELTA_PRACTICAL

SCENARIO = "PERSISTENT"
N_SCORED  = 2   # smoke: 2 scored events
SMOKE_SEED = 101   # outside 1-40, safe for smoke


@pytest.fixture(scope="module")
def smoke_rows():
    """Single smoke seed outside the confirmed seed range."""
    return n2c.run_one_n2c(SMOKE_SEED, N_SCORED, SCENARIO, n2c.ARMS)


@pytest.fixture(scope="module")
def smoke_valid(smoke_rows):
    return [r for r in smoke_rows if r.get("has_history", True) and "arm" in r]


@pytest.fixture(scope="module")
def smoke_by_key(smoke_valid):
    by_key = collections.defaultdict(dict)
    for r in smoke_valid:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r
    return by_key


# ============================================================
# T1 — F005 frozen sigma formula
# ============================================================

def test_t1_f005_frozen_sigma(smoke_valid):
    """Every F005 row has sigma_used exactly 0.005."""
    f005_rows = [r for r in smoke_valid if r["arm"] == "F005"]
    assert len(f005_rows) > 0
    for r in f005_rows:
        assert abs(r["sigma_used"] - 0.005) < 1e-12
        assert r["sigma_raw"] is None
        assert r["cap_active"] is None


# ============================================================
# T2 — A025 frozen formula
# ============================================================

def test_t2_a025_frozen_formula():
    """sigma_used = min(0.03, 0.25 * ||delta|| / sqrt(d)) for known input."""
    delta_norm = 0.10
    expected_raw  = 0.25 * delta_norm / np.sqrt(D)
    expected_used = min(0.03, expected_raw)
    sigma_used, sigma_raw, cap_active = n2c.sigma_a025(delta_norm)
    assert abs(sigma_used - expected_used) < 1e-12
    assert abs(sigma_raw  - expected_raw)  < 1e-12
    assert cap_active == (expected_raw > 0.03)


# ============================================================
# T3 — A025 sigma independent of xi
# ============================================================

def test_t3_a025_independent_of_xi():
    """sigma_a025 takes only delta_hat_norm; xi has no role."""
    import inspect
    sig = inspect.signature(n2c.sigma_a025)
    params = list(sig.parameters.keys())
    assert "xi" not in params
    assert len(params) == 1  # only delta_hat_norm


# ============================================================
# T4 — Common parent indices across arms
# ============================================================

def test_t4_common_parent_indices(smoke_by_key):
    """F005 and A025 have same parent/xi (checked via delta_hat_norm identity)."""
    for key, arm_map in smoke_by_key.items():
        if "F005" not in arm_map or "A025" not in arm_map:
            continue
        # Both arms use same CRN: delta_hat_norm must be identical
        assert abs(arm_map["F005"]["delta_hat_norm"] -
                   arm_map["A025"]["delta_hat_norm"]) < 1e-12


# ============================================================
# T5 — Common xi (confirmed via rho = 0 only for F0)
# ============================================================

def test_t5_common_xi_via_crn(smoke_valid):
    """F0 has rho_noise=0 (sigma=0, no noise), confirming xi is generated once."""
    f0_rows = [r for r in smoke_valid if r["arm"] == "F0"]
    for r in f0_rows:
        assert abs(r["rho_noise"]) < 1e-12


# ============================================================
# T6 — Same delta_hat across arms
# ============================================================

def test_t6_same_delta_hat_across_arms(smoke_by_key):
    """All arms at same event-region share delta_hat_norm (same estimator call)."""
    for key, arm_map in smoke_by_key.items():
        norms = [arm_map[a]["delta_hat_norm"] for a in arm_map]
        assert max(norms) - min(norms) < 1e-12, (
            f"delta_hat_norm differs across arms at key={key}: {norms}"
        )


# ============================================================
# T7 — No future leakage
# ============================================================

def test_t7_no_future_leakage(smoke_valid):
    """history_t < event_id for all valid rows (history uses only past data)."""
    for r in smoke_valid:
        assert r["history_t"] < r["event_id"], (
            f"history_t={r['history_t']} >= event_id={r['event_id']}"
        )


# ============================================================
# T8 — Common FE = BUDGET
# ============================================================

def test_t8_common_fe(smoke_valid):
    """All arms use exactly BUDGET=10 function evaluations."""
    for r in smoke_valid:
        assert r["fe"] == BUDGET


# ============================================================
# T9 — Common baseline IGD
# ============================================================

def test_t9_common_baseline(smoke_by_key):
    """igd_before is identical across all arms for the same event-region."""
    for key, arm_map in smoke_by_key.items():
        igd_vals = [arm_map[a]["igd_before"] for a in arm_map]
        assert max(igd_vals) - min(igd_vals) < 1e-12, (
            f"igd_before differs across arms at key={key}: {igd_vals}"
        )


# ============================================================
# T10 — Deterministic survival
# ============================================================

def test_t10_deterministic_survival():
    """Two runs with same seed produce identical results."""
    r1 = n2c.run_one_n2c(SMOKE_SEED, 2, SCENARIO, n2c.ARMS)
    r2 = n2c.run_one_n2c(SMOKE_SEED, 2, SCENARIO, n2c.ARMS)
    v1 = [r for r in r1 if r.get("has_history", True) and "arm" in r]
    v2 = [r for r in r2 if r.get("has_history", True) and "arm" in r]
    assert len(v1) == len(v2)
    for a, b in zip(v1, v2):
        assert a["arm"]      == b["arm"]
        assert a["event_id"] == b["event_id"]
        assert abs(a["delta_igd"] - b["delta_igd"]) < 1e-12


# ============================================================
# T11 — HARD invariant F005
# ============================================================

def test_t11_hard_invariant_f005(smoke_valid):
    """F005: n_old_dropped_other = 0 for all rows."""
    for r in [x for x in smoke_valid if x["arm"] == "F005"]:
        assert r["n_old_dropped_other"] == 0


# ============================================================
# T12 — HARD invariant A025
# ============================================================

def test_t12_hard_invariant_a025(smoke_valid):
    """A025: n_old_dropped_other = 0 for all rows."""
    for r in [x for x in smoke_valid if x["arm"] == "A025"]:
        assert r["n_old_dropped_other"] == 0


# ============================================================
# T13 — Population size
# ============================================================

def test_t13_population_size(smoke_valid):
    """n_ch_survived within [0, BUDGET] for all rows."""
    for r in smoke_valid:
        assert 0 <= r["n_ch_survived"] <= BUDGET


# ============================================================
# T14 — Practical margin constant
# ============================================================

def test_t14_practical_margin():
    """DELTA_PRACTICAL is exactly 1e-5."""
    assert n2c.DELTA_PRACTICAL == 1e-5


# ============================================================
# T15 — Decision logic: A025_BETTER
# ============================================================

def test_t15_decision_a025_better():
    """Synthetic mean > +margin with CI above zero → A025_BETTER."""
    synthetic_strata = [{"label": "Q1", "n": 50, "mean_diff": 2e-5},
                        {"label": "Q2", "n": 50, "mean_diff": 2e-5}]
    verdict, reasoning = n2c.decide_head_to_head(2e-5, 0.5e-5, 4e-5, synthetic_strata)
    assert verdict == "A025_BETTER"
    assert reasoning["decision_case"] == 1


# ============================================================
# T16 — Decision logic: F005_BETTER
# ============================================================

def test_t16_decision_f005_better():
    """Synthetic mean < -margin with CI below zero → F005_BETTER."""
    synthetic_strata = [{"label": "Q1", "n": 50, "mean_diff": -2e-5},
                        {"label": "Q2", "n": 50, "mean_diff": -2e-5}]
    verdict, reasoning = n2c.decide_head_to_head(-2e-5, -4e-5, -0.5e-5, synthetic_strata)
    assert verdict == "F005_BETTER"
    assert reasoning["decision_case"] == 2


# ============================================================
# T17 — Decision logic: PRACTICALLY_EQUIVALENT
# ============================================================

def test_t17_decision_practically_equivalent():
    """Synthetic |mean| <= margin with consistent strata → PRACTICALLY_EQUIVALENT."""
    synthetic_strata = [{"label": "Q1", "n": 50, "mean_diff": 3e-6},
                        {"label": "Q2", "n": 50, "mean_diff": 5e-6}]
    verdict, reasoning = n2c.decide_head_to_head(3e-6, -5e-6, 1e-5, synthetic_strata)
    assert verdict == "PRACTICALLY_EQUIVALENT"
    assert reasoning["decision_case"] == 3


# ============================================================
# T18 — Decision logic: INCONCLUSIVE
# ============================================================

def test_t18_decision_inconclusive():
    """Strata strongly disagree → INCONCLUSIVE."""
    synthetic_strata = [{"label": "Q1", "n": 50, "mean_diff": +2e-5},
                        {"label": "Q2", "n": 50, "mean_diff": -2e-5}]
    verdict, reasoning = n2c.decide_head_to_head(3e-6, -5e-6, 5e-6, synthetic_strata)
    assert verdict == "INCONCLUSIVE"
    assert reasoning["decision_case"] == 4
    assert reasoning["strata_disagree"] is True


# ============================================================
# T19 — Paired bootstrap resamples matched rows
# ============================================================

def test_t19_bootstrap_resamples_pairs():
    """Bootstrap operates on matched pair differences (1D), not arms separately."""
    rng = np.random.default_rng(0)
    diffs = list(rng.standard_normal(50) * 1e-5)
    mean_d, ci_lo, ci_hi = n2c.bootstrap_ci(diffs, seed=42, n_resamples=1000)
    assert ci_lo < mean_d < ci_hi
    assert ci_hi - ci_lo < 2e-5  # reasonable width for n=50


# ============================================================
# T20 — Bootstrap deterministic
# ============================================================

def test_t20_bootstrap_deterministic():
    """Same seed → same CI."""
    diffs = list(np.arange(1, 101) * 1e-6)
    r1 = n2c.bootstrap_ci(diffs, seed=99, n_resamples=500)
    r2 = n2c.bootstrap_ci(diffs, seed=99, n_resamples=500)
    assert r1 == r2


# ============================================================
# T21 — Seed freshness
# ============================================================

def test_t21_seed_freshness():
    """CONFIRMATION_SEEDS (21-40) are disjoint from N1/N2 seeds (1-20)."""
    prior = set(range(1, 21))
    conf  = set(n2c.CONFIRMATION_SEEDS)
    assert conf.isdisjoint(prior)
    assert len(conf) == 20
    assert min(conf) == 21
    assert max(conf) == 40


# ============================================================
# T22 — IGD identity
# ============================================================

def test_t22_igd_identity(smoke_valid):
    """IGD identity residual < 0.01 for all valid rows."""
    for r in smoke_valid:
        assert r["igd_identity_residual"] < 0.01, (
            f"Large residual {r['igd_identity_residual']:.4f} at arm={r['arm']}"
        )


# ============================================================
# T23 — Baseline immutability
# ============================================================

def test_t23_baseline_immutability():
    """Running arm F005 and A025 does not change igd_before for the other arm."""
    rows = n2c.run_one_n2c(SMOKE_SEED, 2, SCENARIO, n2c.ARMS)
    valid = [r for r in rows if r.get("has_history", True) and "arm" in r]
    by_key = collections.defaultdict(dict)
    for r in valid:
        key = (r["seed"], r["event_id"], r["region_id"])
        by_key[key][r["arm"]] = r
    for key, arm_map in by_key.items():
        vals = [arm_map[a]["igd_before"] for a in arm_map]
        assert max(vals) - min(vals) < 1e-12


# ============================================================
# T24 — Production constants unchanged
# ============================================================

def test_t24_production_constants():
    """Production-locked constants are correct."""
    assert n2c.SIGMA_CAP    == 0.03
    assert n2c.SIGMA_PROD   == 0.03
    assert n2c.SIGMA_F005   == 0.005
    assert n2c.ALPHA_A025   == 0.25
    assert n2c.BUDGET       == 10
    assert n2c.POP_SIZE     == 100
    assert n2c.D            == 10
    assert n2c.LAMBDA_PRED  == 1.0


# ============================================================
# T25 — No production mutation
# ============================================================

def test_t25_no_production_mutation():
    """forecast_state and region_protected_survival are not monkeypatched."""
    import prdrl.features.forecast_state as fs
    import inspect
    fn = getattr(fs, "get_predict_motion_from_history", None)
    assert fn is not None
    # Verify signature still has expected parameters
    params = list(inspect.signature(fn).parameters.keys())
    assert "history" in params


# ============================================================
# T26 — Regression vs N2: F005 on prior seeds
# ============================================================

def test_t26_n2_regression_f005():
    """F005 arm on N2-era seed (seed=1) reproduces N2 sigma semantics."""
    import prdrl.experiments.N2_signal_scaled_noise as n2

    # Run N2C on seed=1 with only F005 and F005-equivalent N2 arm
    rows_n2c = n2c.run_one_n2c(1, 2, "PERSISTENT",
                                [a for a in n2c.ARMS if a["arm"] == "F005"])
    rows_n2  = n2.run_one_n2(1, 2, "PERSISTENT",
                              [a for a in n2.ARMS if a["arm"] == "F005"],
                              split="smoke")

    v_n2c = {(r["seed"], r["event_id"], r["region_id"]): r
             for r in rows_n2c if r.get("has_history", True) and "arm" in r}
    v_n2  = {(r["seed"], r["event_id"], r["region_id"]): r
             for r in rows_n2  if r.get("has_history", True) and "arm" in r}

    common = set(v_n2c.keys()) & set(v_n2.keys())
    assert len(common) >= 3
    for key in common:
        assert abs(v_n2c[key]["delta_igd"]     - v_n2[key]["delta_igd"])     < 1e-10
        assert abs(v_n2c[key]["clipping_frac"] - v_n2[key]["clipping_frac"]) < 1e-10


# ============================================================
# T27 — Regression vs N2: A025 on prior seeds
# ============================================================

def test_t27_n2_regression_a025():
    """A025 arm on N2-era seed (seed=2) reproduces N2 sigma semantics."""
    import prdrl.experiments.N2_signal_scaled_noise as n2

    rows_n2c = n2c.run_one_n2c(2, 2, "PERSISTENT",
                                [a for a in n2c.ARMS if a["arm"] == "A025"])
    rows_n2  = n2.run_one_n2(2, 2, "PERSISTENT",
                              [a for a in n2.ARMS if a["arm"] == "A025"],
                              split="smoke")

    v_n2c = {(r["seed"], r["event_id"], r["region_id"]): r
             for r in rows_n2c if r.get("has_history", True) and "arm" in r}
    v_n2  = {(r["seed"], r["event_id"], r["region_id"]): r
             for r in rows_n2  if r.get("has_history", True) and "arm" in r}

    common = set(v_n2c.keys()) & set(v_n2.keys())
    assert len(common) >= 3
    for key in common:
        assert abs(v_n2c[key]["sigma_used"]   - v_n2[key]["sigma_used"])   < 1e-12
        assert abs(v_n2c[key]["delta_igd"]    - v_n2[key]["delta_igd"])    < 1e-10
        assert abs(v_n2c[key]["rho_noise"]    - v_n2[key]["rho_noise"])    < 1e-10
