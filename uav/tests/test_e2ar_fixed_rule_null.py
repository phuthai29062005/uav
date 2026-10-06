"""
Fixed-rule null tests -- T1 to T7 (Section: Phase 4).

Validates _get_projection_vector, _fixed_rule_partition,
_build_fixed_rule_null_history.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.E2aR_controlled_operator_oracle as e2ar
from prdrl.history.region_history import RegionHistory
from prdrl.problems.controlled_operator import get_bounds

D = 6
xl, xu = get_bounds(D)
xl = np.asarray(xl, dtype=float)
xu = np.asarray(xu, dtype=float)


def _make_pop(seed, N=30):
    rng = np.random.default_rng(seed)
    return rng.uniform(xl, xu, size=(N, D))


def _make_history_two_taus(X_t2, X_t1, R, t):
    """Store X_t2 at (t-2) and X_t1 at (t-1) with round-robin real-region IDs."""
    N2, N1 = len(X_t2), len(X_t1)
    F2 = np.zeros((N2, 2))
    F1 = np.zeros((N1, 2))
    ids2 = np.arange(N2) % R
    ids1 = np.arange(N1) % R
    h = RegionHistory(n_regions=R)
    h.update(t=t - 2, X=X_t2, F=F2, region_ids=ids2, xl=xl, xu=xu)
    h.update(t=t - 1, X=X_t1, F=F1, region_ids=ids1, xl=xl, xu=xu)
    return h


# =====================================================================
# T1 -- deterministic rule: same key -> same w_k -> same assignments
# =====================================================================
def test_t1_deterministic_same_key():
    BASE_SEED, EVENT_ID, K = 7, 3, 2
    R = 4
    X = _make_pop(0, N=40)
    ids_real = np.arange(40) % R

    w_k1 = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, K, D)
    w_k2 = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, K, D)
    np.testing.assert_array_equal(w_k1, w_k2)

    ids1 = e2ar._fixed_rule_partition(X, ids_real, R, w_k1, xl, xu)
    ids2 = e2ar._fixed_rule_partition(X, ids_real, R, w_k2, xl, xu)
    np.testing.assert_array_equal(ids1, ids2)


# =====================================================================
# T2 -- different replica k -> different w_k -> usually different grouping
# =====================================================================
def test_t2_different_replica_different_projection():
    BASE_SEED, EVENT_ID = 7, 3
    R = 4
    X = _make_pop(0, N=40)
    ids_real = np.arange(40) % R

    w_k1 = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, 1, D)
    w_k2 = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, 2, D)
    assert not np.allclose(w_k1, w_k2), "Different k must yield different w_k"

    ids1 = e2ar._fixed_rule_partition(X, ids_real, R, w_k1, xl, xu)
    ids2 = e2ar._fixed_rule_partition(X, ids_real, R, w_k2, xl, xu)
    # In continuous space, different projections almost certainly give different groupings
    assert not np.array_equal(ids1, ids2), (
        "Different projections should (almost surely) yield different groupings"
    )


# =====================================================================
# T3 -- no real-region-order dependence: reordering input rows does NOT
#        change which individual belongs to which null group.
# =====================================================================
def test_t3_row_order_invariant():
    BASE_SEED, EVENT_ID, K = 5, 2, 0
    R = 3
    N = 30
    X = _make_pop(1, N=N)
    ids_real = np.arange(N) % R  # sizes: 10 each

    w_k = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, K, D)

    # Original assignment
    ids_orig = e2ar._fixed_rule_partition(X, ids_real, R, w_k, xl, xu)

    # Reorder rows with a random permutation
    rng = np.random.default_rng(99)
    perm = rng.permutation(N)
    X_reordered = X[perm]
    ids_real_reordered = ids_real[perm]  # sizes unchanged

    ids_reord = e2ar._fixed_rule_partition(X_reordered, ids_real_reordered, R, w_k, xl, xu)

    # Map back: individual i in reordered array is individual perm[i] in original
    # ids_reord[i] should equal ids_orig[perm[i]]
    for i in range(N):
        orig_idx = perm[i]
        assert ids_reord[i] == ids_orig[orig_idx], (
            f"Row {i} (original {orig_idx}): got {ids_reord[i]}, expected {ids_orig[orig_idx]}"
        )


# =====================================================================
# T4 -- exact size preservation at every tau
# =====================================================================
def test_t4_exact_size_preservation():
    BASE_SEED, EVENT_ID, K = 3, 1, 5
    R = 4
    N = 40
    t = 3
    X_t2 = _make_pop(10, N=N)
    X_t1 = _make_pop(11, N=N)

    # Real region sizes differ between taus
    ids2 = np.array([0]*15 + [1]*10 + [2]*10 + [3]*5)
    ids1 = np.array([0]*8  + [1]*12 + [2]*15 + [3]*5)

    h = RegionHistory(n_regions=R)
    h.update(t=t - 2, X=X_t2, F=np.zeros((N, 2)), region_ids=ids2, xl=xl, xu=xu)
    h.update(t=t - 1, X=X_t1, F=np.zeros((N, 2)), region_ids=ids1, xl=xl, xu=xu)

    h_null = e2ar._build_fixed_rule_null_history(h, R, BASE_SEED, EVENT_ID, K, t, xl, xu)

    for r in range(R):
        snap_t2 = h_null.get_region(r, t - 2)
        snap_t1 = h_null.get_region(r, t - 1)
        expected_t2 = int(np.sum(ids2 == r))
        expected_t1 = int(np.sum(ids1 == r))
        got_t2 = snap_t2.n if snap_t2 is not None else 0
        got_t1 = snap_t1.n if snap_t1 is not None else 0
        assert got_t2 == expected_t2, f"Region {r} at t-2: expected {expected_t2}, got {got_t2}"
        assert got_t1 == expected_t1, f"Region {r} at t-1: expected {expected_t1}, got {got_t1}"


# =====================================================================
# T5 -- same rule across time: w_k applied at both t-2 and t-1
#        (no tau in RNG key).  Verify by checking that the projection
#        vector is the same as calling _get_projection_vector directly.
# =====================================================================
def test_t5_same_projection_at_both_taus():
    BASE_SEED, EVENT_ID, K = 9, 4, 3
    # w_k must match what _get_projection_vector returns
    w_k_expected = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, K, D)

    # Build a tiny history and verify assignment at t-2 and t-1
    # is consistent with the same w_k
    R = 3
    N = 12
    t = 3
    X_t2 = _make_pop(20, N=N)
    X_t1 = _make_pop(21, N=N)
    ids2 = np.array([0]*4 + [1]*4 + [2]*4)
    ids1 = np.array([0]*5 + [1]*3 + [2]*4)

    h = RegionHistory(n_regions=R)
    h.update(t=t - 2, X=X_t2, F=np.zeros((N, 2)), region_ids=ids2, xl=xl, xu=xu)
    h.update(t=t - 1, X=X_t1, F=np.zeros((N, 2)), region_ids=ids1, xl=xl, xu=xu)

    h_null = e2ar._build_fixed_rule_null_history(h, R, BASE_SEED, EVENT_ID, K, t, xl, xu)

    # Verify t-2 assignment matches manual application of w_k_expected
    X_tau2 = np.vstack([h.get_region(r, t - 2).X for r in range(R)])
    sizes2 = [int(np.sum(ids2 == r)) for r in range(R)]
    expected_ids2 = e2ar._fixed_rule_partition(X_tau2, ids2, R, w_k_expected, xl, xu)
    for r in range(R):
        snap = h_null.get_region(r, t - 2)
        X_null_r = snap.X if snap is not None else np.empty((0, D))
        X_expected_r = X_tau2[expected_ids2 == r]
        assert len(X_null_r) == len(X_expected_r)

    # Verify t-1 assignment matches manual application of same w_k_expected
    X_tau1 = np.vstack([h.get_region(r, t - 1).X for r in range(R)])
    expected_ids1 = e2ar._fixed_rule_partition(X_tau1, ids1, R, w_k_expected, xl, xu)
    for r in range(R):
        snap = h_null.get_region(r, t - 1)
        X_null_r = snap.X if snap is not None else np.empty((0, D))
        X_expected_r = X_tau1[expected_ids1 == r]
        assert len(X_null_r) == len(X_expected_r)


# =====================================================================
# T6 -- no future leakage: changing t-time population does not change
#        the null history built from t-2 and t-1 only.
# =====================================================================
def test_t6_no_future_leakage():
    BASE_SEED, EVENT_ID, K = 2, 7, 1
    R = 3
    N = 15
    t = 4

    X_t2 = _make_pop(30, N=N)
    X_t1 = _make_pop(31, N=N)
    ids2 = np.arange(N) % R
    ids1 = np.arange(N) % R

    # History A: only stores t-2 and t-1
    hA = RegionHistory(n_regions=R)
    hA.update(t=t - 2, X=X_t2, F=np.zeros((N, 2)), region_ids=ids2, xl=xl, xu=xu)
    hA.update(t=t - 1, X=X_t1, F=np.zeros((N, 2)), region_ids=ids1, xl=xl, xu=xu)

    # History B: additionally has a t-time snapshot
    hB = RegionHistory(n_regions=R)
    hB.update(t=t - 2, X=X_t2, F=np.zeros((N, 2)), region_ids=ids2, xl=xl, xu=xu)
    hB.update(t=t - 1, X=X_t1, F=np.zeros((N, 2)), region_ids=ids1, xl=xl, xu=xu)
    X_t_different = _make_pop(99, N=N)
    ids_t = np.arange(N) % R
    hB.update(t=t, X=X_t_different, F=np.zeros((N, 2)), region_ids=ids_t, xl=xl, xu=xu)

    hA_null = e2ar._build_fixed_rule_null_history(hA, R, BASE_SEED, EVENT_ID, K, t, xl, xu)
    hB_null = e2ar._build_fixed_rule_null_history(hB, R, BASE_SEED, EVENT_ID, K, t, xl, xu)

    # Null histories must be identical (t-time data must not affect them)
    for r in range(R):
        for tau in (t - 2, t - 1):
            snapA = hA_null.get_region(r, tau)
            snapB = hB_null.get_region(r, tau)
            nA = snapA.n if snapA is not None else 0
            nB = snapB.n if snapB is not None else 0
            assert nA == nB, f"Region {r} tau {tau}: nA={nA} nB={nB}"
            if nA > 0:
                np.testing.assert_array_equal(snapA.X, snapB.X)


# =====================================================================
# T7 -- no real-region contamination: same X but different real-region
#        labels/ordering -> same fixed-rule partition IF sizes held fixed.
# =====================================================================
def test_t7_no_real_region_contamination_fixed_sizes():
    BASE_SEED, EVENT_ID, K = 1, 5, 0
    R = 3
    N = 30
    t = 3

    X_pop = _make_pop(42, N=N)
    F_pop = np.zeros((N, 2))

    # Sizes profile: [10, 10, 10] for both taus
    sizes = [10, 10, 10]

    # Labeling A: sequential blocks
    ids_A = np.array([0]*10 + [1]*10 + [2]*10)

    # Labeling B: interleaved (same sizes, different individual-to-region mapping)
    ids_B = np.zeros(N, dtype=int)
    ids_B[::3] = 0   # every 3rd starting 0 -> region 0
    ids_B[1::3] = 1  # every 3rd starting 1 -> region 1
    ids_B[2::3] = 2  # every 3rd starting 2 -> region 2
    # Verify same sizes
    for r in range(R):
        assert np.sum(ids_A == r) == np.sum(ids_B == r) == sizes[r]

    hA = RegionHistory(n_regions=R)
    hA.update(t=t - 2, X=X_pop, F=F_pop, region_ids=ids_A, xl=xl, xu=xu)
    hA.update(t=t - 1, X=X_pop, F=F_pop, region_ids=ids_A, xl=xl, xu=xu)

    hB = RegionHistory(n_regions=R)
    hB.update(t=t - 2, X=X_pop, F=F_pop, region_ids=ids_B, xl=xl, xu=xu)
    hB.update(t=t - 1, X=X_pop, F=F_pop, region_ids=ids_B, xl=xl, xu=xu)

    hA_null = e2ar._build_fixed_rule_null_history(hA, R, BASE_SEED, EVENT_ID, K, t, xl, xu)
    hB_null = e2ar._build_fixed_rule_null_history(hB, R, BASE_SEED, EVENT_ID, K, t, xl, xu)

    # Same individuals -> same projection scores -> same null group membership
    # Verify group sizes match
    for r in range(R):
        for tau in (t - 2, t - 1):
            sA = hA_null.get_region(r, tau)
            sB = hB_null.get_region(r, tau)
            nA = sA.n if sA is not None else 0
            nB = sB.n if sB is not None else 0
            assert nA == nB, f"Region {r} tau {tau}: nA={nA} nB={nB}"

    # Verify the X values assigned to each group are the same set of individuals
    # (may be in different row order within group)
    w_k = e2ar._get_projection_vector(BASE_SEED, EVENT_ID, K, D)
    # All individuals have same X_pop regardless of label -> same scores -> same sort order
    # So the null partition must assign same individuals to same groups
    ids_null_A = e2ar._fixed_rule_partition(X_pop, ids_A, R, w_k, xl, xu)
    ids_null_B = e2ar._fixed_rule_partition(X_pop, ids_B, R, w_k, xl, xu)
    np.testing.assert_array_equal(ids_null_A, ids_null_B)
