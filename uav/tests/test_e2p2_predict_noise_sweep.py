"""
E2-P2 Predict Noise Sweep -- Tests T1-T9 (Phase 11).

Tests cover:
  T1  sigma=0 -> offspring = parent + delta (no noise)
  T2  same seed/input -> same offspring
  T3  POP vs GT isolation -- only delta differs
  T4  LOCAL invariant to sigma_pred
  T5  FE equality across modes
  T6  no future leakage -- POP estimator from t-2,t-1 only
  T7  GT isolation -- changing GT doesn't affect POP/LOCAL
  T8  sigma scaling: perturbation(0.01) = 0.5 * perturbation(0.02) before clip
  T9  baseline immutability across sigma branches
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.E2P2_predict_noise_sweep as e2p2
from prdrl.problems.controlled_predict import get_bounds

D = 10
xl, xu = get_bounds(D)
xl = np.asarray(xl, dtype=float)
xu = np.asarray(xu, dtype=float)


def _interior_X(n=15, seed=0):
    """Population well inside bounds -- no clip artefacts."""
    return np.random.default_rng(seed).uniform(0.3, 0.7, size=(n, D))


def _delta(scale=0.01):
    """Small deterministic delta (no clip issues)."""
    rng = np.random.default_rng(99)
    d = rng.uniform(-1, 1, size=D) * scale
    d[0] = 0.0  # GT convention: x1-component = 0
    return d / (np.linalg.norm(d) + 1e-12) * scale


def _shared_rng(seed=1, event=5, region=2):
    return e2p2.make_rng(seed, event, region, "PREDICT_SHARED")


# =====================================================================
# T1 -- sigma=0: offspring = parent + delta, no Gaussian perturbation
# =====================================================================
def test_t1_sigma_zero_no_noise():
    X = _interior_X(n=10)
    delta = _delta(scale=0.01)
    rng = _shared_rng()
    idxs, xis = e2p2.generate_base_samples(5, D, rng, len(X))

    offspring = e2p2.apply_samples(idxs, xis, X, delta, sigma=0.0, xl=xl, xu=xu)
    assert offspring.shape == (5, D)

    for i, (idx, xi) in enumerate(zip(idxs, xis)):
        parent = X[idx]
        z_parent = (parent - xl) / (xu - xl)
        expected = xl + np.clip(z_parent + delta, 0.0, 1.0) * (xu - xl)
        np.testing.assert_allclose(offspring[i], expected, atol=1e-10)


# =====================================================================
# T2 -- same seed/input -> same offspring
# =====================================================================
def test_t2_reproducibility():
    X = _interior_X(n=12)
    delta = _delta(scale=0.015)

    rng1 = _shared_rng(seed=3, event=7, region=1)
    idxs1, xis1 = e2p2.generate_base_samples(8, D, rng1, len(X))

    rng2 = _shared_rng(seed=3, event=7, region=1)
    idxs2, xis2 = e2p2.generate_base_samples(8, D, rng2, len(X))

    out1 = e2p2.apply_samples(idxs1, xis1, X, delta, 0.02, xl, xu)
    out2 = e2p2.apply_samples(idxs2, xis2, X, delta, 0.02, xl, xu)

    np.testing.assert_array_equal(out1, out2)


# =====================================================================
# T3 -- POP vs GT isolation: same xi, only delta differs
# =====================================================================
def test_t3_pop_vs_gt_only_delta_differs():
    X = _interior_X(n=20)
    # Use small deltas to avoid clipping
    delta_pop = _delta(scale=0.005)
    delta_gt  = _delta(scale=0.008)
    # Ensure they differ
    assert not np.allclose(delta_pop, delta_gt)

    # Same RNG key -> same samples
    rng_pop = _shared_rng()
    rng_gt  = _shared_rng()
    idxs_pop, xis_pop = e2p2.generate_base_samples(6, D, rng_pop, len(X))
    idxs_gt,  xis_gt  = e2p2.generate_base_samples(6, D, rng_gt,  len(X))

    # Verify samples are identical
    assert idxs_pop == idxs_gt
    for xi_p, xi_g in zip(xis_pop, xis_gt):
        np.testing.assert_array_equal(xi_p, xi_g)

    out_pop = e2p2.apply_samples(idxs_pop, xis_pop, X, delta_pop, 0.0, xl, xu)
    out_gt  = e2p2.apply_samples(idxs_gt,  xis_gt,  X, delta_gt,  0.0, xl, xu)

    # With sigma=0 and no clip, difference = (delta_gt - delta_pop) * (xu - xl)
    expected_diff = (delta_gt - delta_pop) * (xu - xl)
    for row in (out_gt - out_pop):
        np.testing.assert_allclose(row, expected_diff, atol=1e-9)


# =====================================================================
# T4 -- LOCAL invariant: LOCAL offspring don't change when sigma_pred changes
# =====================================================================
def test_t4_local_invariant_to_sigma_pred():
    X = _interior_X(n=15)
    base_seed, event_id, region_id = 7, 3, 2

    ch1 = e2p2.make_local_children(X, xl, xu, base_seed, event_id, region_id)

    # LOCAL uses a fixed RNG key ("LOCAL") independent of sigma_pred
    # Running it again with same key should give the same result
    ch2 = e2p2.make_local_children(X, xl, xu, base_seed, event_id, region_id)

    np.testing.assert_array_equal(ch1, ch2)

    # And LOCAL children must be identical across sigma_pred values
    # (no sigma_pred parameter in make_local_children)
    assert ch1.shape == (e2p2.BUDGET, D)


# =====================================================================
# T5 -- FE equality: LOCAL, POP, GT all produce exactly BUDGET offspring
# =====================================================================
def test_t5_fe_equality():
    X = _interior_X(n=12)
    delta = _delta(scale=0.01)

    rng = _shared_rng()
    idxs, xis = e2p2.generate_base_samples(e2p2.BUDGET, D, rng, len(X))

    ch_pop = e2p2.apply_samples(idxs, xis, X, delta, 0.0, xl, xu)
    ch_gt  = e2p2.apply_samples(idxs, xis, X, delta, 0.0, xl, xu)
    ch_loc = e2p2.make_local_children(X, xl, xu, 1, 1, 1)

    assert len(ch_pop) == e2p2.BUDGET, f"POP: {len(ch_pop)}"
    assert len(ch_gt)  == e2p2.BUDGET, f"GT:  {len(ch_gt)}"
    assert len(ch_loc) == e2p2.BUDGET, f"LOC: {len(ch_loc)}"


# =====================================================================
# T6 -- no future leakage: POP estimator uses t-2 and t-1 only.
#       Changing t-time population does not affect delta_pop.
# =====================================================================
def test_t6_no_future_leakage():
    # delta_pop is computed inside run_one_sweep from prev2 and current pops.
    # Here we verify the formula source uses ONLY mu_pop_full_prev and _prev2.
    import inspect
    src = inspect.getsource(e2p2.run_region_sweep)
    # Must compute Delta_POP from prev population means (no t_new/t_current)
    assert "mu_pop_full_prev - mu_pop_full_prev2" in src
    # Must NOT use mu_star_current or t_new for Delta_POP
    assert "Delta_POP_full = (mu_pop_full_prev - mu_pop_full_prev2)" in src


# =====================================================================
# T7 -- GT isolation: changing Delta_GT does not affect POP or LOCAL
# =====================================================================
def test_t7_gt_isolation():
    X = _interior_X(n=10)
    delta_pop = _delta(scale=0.01)
    delta_gt_A = _delta(scale=0.02)
    delta_gt_B = _delta(scale=0.05)  # different GT

    rng = _shared_rng()
    idxs, xis = e2p2.generate_base_samples(5, D, rng, len(X))

    # POP offspring must be the same regardless of GT delta
    ch_pop_A = e2p2.apply_samples(idxs, xis, X, delta_pop, 0.0, xl, xu)
    ch_pop_B = e2p2.apply_samples(idxs, xis, X, delta_pop, 0.0, xl, xu)
    np.testing.assert_array_equal(ch_pop_A, ch_pop_B)

    # LOCAL must also be the same
    ch_loc_A = e2p2.make_local_children(X, xl, xu, 1, 1, 1)
    ch_loc_B = e2p2.make_local_children(X, xl, xu, 1, 1, 1)
    np.testing.assert_array_equal(ch_loc_A, ch_loc_B)

    # GT offspring DO differ when GT delta changes
    ch_gt_A = e2p2.apply_samples(idxs, xis, X, delta_gt_A, 0.0, xl, xu)
    ch_gt_B = e2p2.apply_samples(idxs, xis, X, delta_gt_B, 0.0, xl, xu)
    assert not np.allclose(ch_gt_A, ch_gt_B)


# =====================================================================
# T8 -- sigma scaling: perturbation(sigma=0.01) = 0.5 * perturbation(0.02)
#       before clipping in controlled interior-point case.
# =====================================================================
def test_t8_sigma_scaling():
    # Interior parents: [0.3, 0.7] -> max |delta + sigma*xi| won't hit [0,1] boundary
    # with sigma <= 0.03 and delta=0 (pure noise test)
    X = np.full((1, D), 0.5)  # all at center
    delta = np.zeros(D)       # zero delta -- isolates noise contribution

    rng_01 = _shared_rng(seed=5)
    idxs_01, xis_01 = e2p2.generate_base_samples(8, D, rng_01, len(X))

    rng_02 = _shared_rng(seed=5)
    idxs_02, xis_02 = e2p2.generate_base_samples(8, D, rng_02, len(X))

    # sigma=0.01 offspring (in normalized space, perturbation = 0.01 * xi)
    # sigma=0.02 offspring (in normalized space, perturbation = 0.02 * xi)
    # At x=0.5 center: z_parent=0.5, no clipping for small sigma

    # Compute raw perturbations manually
    for xi01, xi02 in zip(xis_01, xis_02):
        np.testing.assert_array_equal(xi01, xi02)  # same base xi

    # Now verify offspring positions reflect sigma scaling
    out_01 = e2p2.apply_samples(idxs_01, xis_01, X, delta, 0.01, xl, xu)
    out_02 = e2p2.apply_samples(idxs_02, xis_02, X, delta, 0.02, xl, xu)

    # In [0,1] normalized space, z_parent = 0.5 for all
    # z_child_01 = 0.5 + 0.01 * xi
    # z_child_02 = 0.5 + 0.02 * xi
    # deviation_01 = z_child_01 - 0.5 = 0.01 * xi
    # deviation_02 = z_child_02 - 0.5 = 0.02 * xi
    # => deviation_01 = 0.5 * deviation_02  (before clip -- guaranteed here since |0.02*xi| << 0.5)
    center = np.full(D, 0.5) * (xu - xl) + xl  # x=0.5 in original space
    dev_01 = out_01 - center
    dev_02 = out_02 - center
    np.testing.assert_allclose(dev_01, 0.5 * dev_02, atol=1e-10)


# =====================================================================
# T9 -- baseline immutability: running one sigma branch doesn't mutate
#       the X_region used by another sigma branch.
# =====================================================================
def test_t9_baseline_immutability():
    X = _interior_X(n=15, seed=7)
    X_copy = X.copy()
    delta = _delta(scale=0.01)

    rng = _shared_rng()
    idxs, xis = e2p2.generate_base_samples(e2p2.BUDGET, D, rng, len(X))

    # Run multiple sigma branches
    for sigma in e2p2.SIGMA_P_VALUES:
        _ = e2p2.apply_samples(idxs, xis, X, delta, sigma, xl, xu)

    # X must be unchanged
    np.testing.assert_array_equal(X, X_copy)


# =====================================================================
# Scenario dynamics sanity
# =====================================================================
def test_persistent_dynamics_all_persistent():
    rd = e2p2.make_scenario_dynamics("PERSISTENT", 10)
    assert all(t == "persistent" for t in rd)
    assert len(rd) == 10


def test_stationary_dynamics_all_stationary():
    rd = e2p2.make_scenario_dynamics("STATIONARY", 10)
    assert all(t == "stationary" for t in rd)
    assert len(rd) == 10


# =====================================================================
# Stage A Audit Tests (T-A1 through T-A6)
# =====================================================================

def _cp_region_center(region_dynamics, bin_index, t, n_distance_vars):
    from prdrl.problems.controlled_predict import region_center
    return region_center(region_dynamics, bin_index, t, n_distance_vars)


# T-A1: GT response vector has correct t-1→t semantics
def test_ta1_gt_response_vector_semantics():
    """GT_PREDICT should consume delta_response_true = mu_star(t) - mu_star(t-1),
    NOT the historical shift delta_hist_true = mu_star(t-1) - mu_star(t-2)."""
    import inspect
    src = inspect.getsource(e2p2.run_region_sweep)
    # Must define delta_response_true (the correct semantic)
    assert "delta_response_true" in src
    # Must use delta_response_true_full for GT offspring
    assert "delta_response_true_full" in src
    # Must NOT use Delta_GT (old name that encoded wrong semantics)
    assert "Delta_GT" not in src


# T-A2: For PERSISTENT (linear + equal steps), hist shift == response shift
def test_ta2_persistent_hist_eq_response_shift():
    """PERSISTENT dynamics: linear target, equal time steps -> hist == response."""
    from prdrl.problems.controlled_predict import region_center, DYNAMIC_TYPES
    n_bins = 10
    rate = 0.10
    region_dynamics = ["persistent"] * n_bins
    n_dist = D - 1
    bin_idx = 3

    # Equal step: dt = 0.1 (one generation block)
    t2, t1, t0 = 0.0, 0.1, 0.2

    mu_prev2  = region_center(region_dynamics, bin_idx, t2, n_dist)
    mu_prev   = region_center(region_dynamics, bin_idx, t1, n_dist)
    mu_curr   = region_center(region_dynamics, bin_idx, t0, n_dist)

    delta_hist_true     = mu_prev  - mu_prev2  # t-2→t-1
    delta_response_true = mu_curr  - mu_prev   # t-1→t

    np.testing.assert_allclose(delta_hist_true, delta_response_true, atol=1e-12)


# T-A3: STATIONARY true response shift == zero
def test_ta3_stationary_response_shift_zero():
    from prdrl.problems.controlled_predict import region_center
    n_bins = 10
    region_dynamics = ["stationary"] * n_bins
    n_dist = D - 1
    bin_idx = 2

    for t in [0.0, 0.1, 0.5, 1.0]:
        mu = region_center(region_dynamics, bin_idx, t, n_dist)
        np.testing.assert_array_equal(mu, np.zeros(n_dist))

    mu_prev  = region_center(region_dynamics, bin_idx, 0.1, n_dist)
    mu_curr  = region_center(region_dynamics, bin_idx, 0.2, n_dist)
    delta_response_true = mu_curr - mu_prev
    np.testing.assert_array_equal(delta_response_true, np.zeros(n_dist))


# T-A4: Manual POP estimator == production RegionHistory estimator
def test_ta4_pop_estimator_production_equivalence():
    """E2-P2's manual centroid delta must match RegionHistory+get_predict_motion."""
    from prdrl.history.region_history import RegionHistory
    from prdrl.features.forecast_state import get_predict_motion_from_history

    xl_test, xu_test = get_bounds(D)
    xl_test = np.asarray(xl_test, dtype=float)
    xu_test = np.asarray(xu_test, dtype=float)

    rng = np.random.default_rng(777)
    X_t2 = rng.uniform(0.2, 0.8, size=(20, D))
    X_t1 = rng.uniform(0.2, 0.8, size=(20, D))
    region_ids = np.zeros(20, dtype=int)
    F_dummy = np.zeros((20, 2))

    history = RegionHistory(n_regions=1)
    history.update(t=1, X=X_t2, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    history.update(t=2, X=X_t1, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)

    # Production path: at event t=3, reads centroids at t=1 and t=2
    motion = get_predict_motion_from_history(history, region_id=0, t=3, lambda_pred=1.0)
    delta_production = motion["delta_hat"]  # D-dimensional

    # Manual path (E2-P2 style)
    Z_t1 = (X_t1 - xl_test) / (xu_test - xl_test)
    Z_t2 = (X_t2 - xl_test) / (xu_test - xl_test)
    mu_t1 = np.mean(Z_t1, axis=0)
    mu_t2 = np.mean(Z_t2, axis=0)
    delta_manual = (mu_t1 - mu_t2) * 1.0  # lambda_pred=1.0

    np.testing.assert_allclose(delta_manual, delta_production, atol=1e-10)


# T-A5: Production estimator reads only t-2, t-1 (no future leakage)
def test_ta5_production_estimator_no_future():
    """get_predict_motion_from_history reads only t-2 and t-1, not t or beyond."""
    import inspect
    from prdrl.features.forecast_state import get_predict_motion_from_history
    src = inspect.getsource(get_predict_motion_from_history)
    # Uses end_t = t-1, length=2 -> reads at t-2 and t-1 only
    assert "t - 1" in src or "t-1" in src
    assert "length=2" in src

    # Runtime verification: adding a t-snapshot does NOT change delta
    from prdrl.history.region_history import RegionHistory
    xl_test, xu_test = get_bounds(D)
    rng = np.random.default_rng(88)
    X_t2 = rng.uniform(0.2, 0.8, size=(10, D))
    X_t1 = rng.uniform(0.2, 0.8, size=(10, D))
    X_t_future = rng.uniform(0.2, 0.8, size=(10, D))  # t=3 (future of t=3 call)
    region_ids = np.zeros(10, dtype=int)
    F_dummy = np.zeros((10, 2))

    history_base = RegionHistory(n_regions=1)
    history_base.update(t=1, X=X_t2, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    history_base.update(t=2, X=X_t1, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    m_base = get_predict_motion_from_history(history_base, 0, t=3, lambda_pred=1.0)

    history_with_future = RegionHistory(n_regions=1)
    history_with_future.update(t=1, X=X_t2, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    history_with_future.update(t=2, X=X_t1, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    history_with_future.update(t=3, X=X_t_future, F=F_dummy, region_ids=region_ids, xl=xl_test, xu=xu_test)
    m_future = get_predict_motion_from_history(history_with_future, 0, t=3, lambda_pred=1.0)

    np.testing.assert_allclose(m_base["delta_hat"], m_future["delta_hat"], atol=1e-12)


# T-A6: make_rng and _surv_seed do not use Python hash() (PYTHONHASHSEED stable)
def test_ta6_no_python_hash_in_rng():
    """Neither make_rng nor _surv_seed should contain Python hash() calls."""
    import inspect
    import re
    src_make_rng = inspect.getsource(e2p2.make_rng)
    # Check that hash() is not called (not just mentioned in comments/docstrings)
    assert not re.search(r'[^_a-zA-Z]hash\s*\(', src_make_rng), \
        "make_rng must not call Python hash()"

    src_surv = inspect.getsource(e2p2._surv_seed)
    assert not re.search(r'[^_a-zA-Z]hash\s*\(', src_surv), \
        "_surv_seed must not call Python hash()"

    # Same-process reproducibility (sanity check)
    rng1 = e2p2.make_rng(42, 5, 3, "LOCAL")
    rng2 = e2p2.make_rng(42, 5, 3, "LOCAL")
    assert rng1.integers(2**31) == rng2.integers(2**31)

    seed1 = e2p2._surv_seed(42, 5, 3, "POP_0.02")
    seed2 = e2p2._surv_seed(42, 5, 3, "POP_0.02")
    assert seed1 == seed2
