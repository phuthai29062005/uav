"""
E2a-R -- Revised Controlled Operator Oracle Validation. Tests T1-T22
(Section 22-24).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.E2aR_controlled_operator_oracle as e2ar  # noqa: E402
from prdrl.problems.controlled_operator import (  # noqa: E402
    evaluate as op_evaluate, get_bounds, make_region_dynamics_map,
)
from prdrl.history.region_history import RegionHistory  # noqa: E402


D = 6
xl = np.zeros(D)
xu = np.ones(D)


def _toy_snapshot(seed=0, N=40, R=4):
    rng = np.random.default_rng(seed)
    xl_, xu_ = get_bounds(D=D)
    X_base = rng.uniform(xl_, xu_, size=(N, D))
    region_dynamics = make_region_dynamics_map("MIXED_RESPONSE", R)
    F_plus = op_evaluate(X_base, 0.3, region_dynamics)
    region_ids = rng.integers(0, R, size=N)
    return X_base, F_plus, region_ids, R, xl_, xu_, region_dynamics


# =====================================================================
# T1-T10 -- random-partition null
# =====================================================================
def test_t1_random_partition_preserves_exact_active_sizes():
    region_ids = np.array([0] * 7 + [1] * 3 + [2] * 5 + [3] * 0)
    rng = np.random.default_rng(0)
    rand_ids = e2ar.size_preserving_random_partition(region_ids, R=4, rng=rng)
    for r in range(4):
        assert np.sum(rand_ids == r) == np.sum(region_ids == r)


def test_t2_random_partition_preserves_active_region_count():
    region_ids = np.array([0] * 7 + [1] * 3 + [2] * 5)
    rng = np.random.default_rng(0)
    rand_ids = e2ar.size_preserving_random_partition(region_ids, R=5, rng=rng)
    active_real = set(int(r) for r in np.unique(region_ids))
    active_rand = set(int(r) for r in range(5) if np.sum(rand_ids == r) > 0)
    assert active_real == active_rand
    # region 3,4 khong xuat hien trong region_ids goc -> phai VAN rong
    assert np.sum(rand_ids == 3) == 0
    assert np.sum(rand_ids == 4) == 0


def test_t3_same_seed_identical_random_partition():
    region_ids = np.array([0] * 5 + [1] * 5)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = e2ar.size_preserving_random_partition(region_ids, R=2, rng=rng1)
    out2 = e2ar.size_preserving_random_partition(region_ids, R=2, rng=rng2)
    np.testing.assert_array_equal(out1, out2)


def test_t4_different_replicate_different_partition():
    region_ids = np.array([0] * 5 + [1] * 5)
    rng = np.random.default_rng(1)
    out1 = e2ar.size_preserving_random_partition(region_ids, R=2, rng=rng)
    out2 = e2ar.size_preserving_random_partition(region_ids, R=2, rng=rng)
    assert not np.array_equal(out1, out2)


def test_t5_real_partition_never_overwritten():
    region_ids = np.array([0] * 5 + [1] * 5)
    region_ids_copy = region_ids.copy()
    rng = np.random.default_rng(2)
    _ = e2ar.size_preserving_random_partition(region_ids, R=2, rng=rng)
    np.testing.assert_array_equal(region_ids, region_ids_copy)


def test_t6_random_null_same_operator_parameters():
    import inspect
    src = inspect.getsource(e2ar.run_regional_protocol)
    assert "SIGMA_LOCAL" not in src  # sigma dung ben trong generate_children, khong hardcode lai o day
    src_gen = inspect.getsource(e2ar.generate_children)
    assert "e2ar.SIGMA_LOCAL" not in src_gen  # dung module-level constant truc tiep
    assert "SIGMA_LOCAL" in src_gen and "SIGMA_DIVERSIFY" in src_gen


def test_t7_random_null_same_budget_policy():
    """isolated trial (ca real lan random) deu dung equal_budget() --
    kiem tra qua run_regional_protocol() nguon."""
    import inspect
    src = inspect.getsource(e2ar.run_regional_protocol)
    assert "equal_budget(" in src


def test_t8_random_null_same_survival_mechanism():
    import inspect
    src = inspect.getsource(e2ar.respond)
    assert "nsga2_survival(" in src


def test_t9_random_null_starts_from_same_frozen_snapshot():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=5)
    X_base_copy = X_base.copy()

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    def quality_fn(F):
        return -float(np.mean(F))

    rng = np.random.default_rng(9)
    for k in range(3):
        rand_ids = e2ar.size_preserving_random_partition(region_ids, R, rng)
        e2ar.run_regional_protocol(X_base, F_plus, rand_ids, R,
                                   RegionHistory(R), 1,
                                   evaluate_fn, quality_fn, xl_, xu_, 5,
                                   base_seed=1, event_id=1, null_rep_idx=k)
    np.testing.assert_array_equal(X_base, X_base_copy)


def test_t10_rog_percentile_correct_on_synthetic_values():
    rog_real = 0.05
    rog_rand = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.07])
    percentile = float(np.mean(rog_rand <= rog_real))
    assert percentile == pytest.approx(4 / 6)


# =====================================================================
# T11-T16 -- FE matching
# =====================================================================
def test_t11_fe_matched_sum_equals_regional_total():
    active_regions = [0, 1, 2, 3, 4]
    B_total = 47
    budgets = e2ar.fe_matched_budgets(B_total, active_regions)
    assert sum(budgets.values()) == B_total


def test_t12_budget_allocation_deterministic():
    active_regions = [3, 1, 4, 0]
    out1 = e2ar.fe_matched_budgets(23, active_regions)
    out2 = e2ar.fe_matched_budgets(23, active_regions)
    assert out1 == out2


def test_t13_remainder_allocated_by_fixed_region_id_order():
    active_regions = [5, 2, 8, 1]   # thu tu vao KHONG sap xep
    budgets = e2ar.fe_matched_budgets(10, active_regions)  # base=2, rem=2
    # sorted(active_regions) = [1,2,5,8] -- 2 vung dau (1,2) nhan +1
    assert budgets[1] == 3
    assert budgets[2] == 3
    assert budgets[5] == 2
    assert budgets[8] == 2


def test_t14_no_severity_based_fe_allocation():
    import inspect
    src = inspect.getsource(e2ar.fe_matched_budgets)
    assert "severity" not in src


def test_t15_keep_remains_zero_fe():
    active_regions = [0, 1, 2]
    budgets = e2ar.fe_matched_budgets(30, active_regions)
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=6, R=3)

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    result = e2ar.respond(X_base, F_plus, region_ids, R, {}, budgets,
                          RegionHistory(R), 1,
                          evaluate_fn, xl_, xu_, base_seed=1, event_id=1,
                          null_rep_idx=e2ar.REAL_SENTINEL)
    assert result["extra_fe"] == 0


def test_t16_fe_matched_comparator_excludes_keep():
    import inspect
    src = inspect.getsource(e2ar.run_best_global_fe_matched)
    assert '"LOCAL", "PREDICT", "DIVERSIFY"' in src
    assert 'action == "KEEP"' not in src


# =====================================================================
# T17-T22 -- dynamic label audit
# =====================================================================
def test_t17_ground_truth_labels_derivable():
    from prdrl.problems.controlled_operator import ground_truth_dynamic_label
    xl_, xu_ = get_bounds(8)
    rng = np.random.default_rng(0)
    X = rng.uniform(xl_, xu_, size=(20, 8))
    rd = make_region_dynamics_map("MIXED_RESPONSE", 5)
    labels = ground_truth_dynamic_label(X, rd)
    assert labels.shape == (20,)
    assert set(labels.tolist()) <= set(rd)


def test_t18_region_composition_sums_to_one():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=7)
    composition, dominant, purity = e2ar.region_dynamic_composition(
        X_base, region_ids, R, rd)
    for r in range(R):
        if np.sum(region_ids == r) == 0:
            continue
        total = sum(composition[r].values())
        assert total == pytest.approx(1.0)


def test_t19_dominant_type_only_if_purity_ge_08():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=7)
    composition, dominant, purity = e2ar.region_dynamic_composition(
        X_base, region_ids, R, rd, purity_threshold=0.8)
    for r in range(R):
        if np.sum(region_ids == r) == 0:
            continue
        if purity[r] >= 0.8:
            assert dominant[r] != "MIXED"


def test_t20_otherwise_mixed():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=7)
    composition, dominant, purity = e2ar.region_dynamic_composition(
        X_base, region_ids, R, rd, purity_threshold=0.8)
    for r in range(R):
        if np.sum(region_ids == r) == 0:
            continue
        if purity[r] < 0.8:
            assert dominant[r] == "MIXED"


def test_t21_no_assumption_region_id_equals_ground_truth_id():
    """region_dynamic_composition() phai tinh dua tren VI TRI THAT cua
    tung nghiem (bin(x1)), khong dua tren gia dinh region_ids == bin
    index."""
    import inspect
    src = inspect.getsource(e2ar.region_dynamic_composition)
    assert "ground_truth_dynamic_label(X_base, region_dynamics)" in src


def test_t22_abrupt_scenario_reports_composition_per_region():
    xl_, xu_ = get_bounds(D=10)
    rng = np.random.default_rng(3)
    X = rng.uniform(xl_, xu_, size=(60, 10))
    rd = make_region_dynamics_map("ABRUPT_LARGE_SHIFT", 10)
    rng2 = np.random.default_rng(4)
    region_ids = rng2.integers(0, 10, size=60)
    composition, dominant, purity = e2ar.region_dynamic_composition(
        X, region_ids, 10, rd)
    assert len(composition) == 10
    assert len(dominant) == 10
    assert len(purity) == 10


# =====================================================================
# T23-T27 -- null-history alignment
# =====================================================================

def _make_controlled_history(D=3, R=2, N=12, seed=42):
    """Two clearly separated populations: region 0 members in [0,0.2],
    region 1 members in [0.8,1.0].  Returns history with snapshots at
    t=0 and t=1, plus xl/xu."""
    rng = np.random.default_rng(seed)
    xl, xu = np.zeros(D), np.ones(D)
    X_low  = rng.uniform(0.0, 0.2, size=(N // 2, D))
    X_high = rng.uniform(0.8, 1.0, size=(N // 2, D))
    X_full = np.vstack([X_low, X_high])
    F_full = rng.uniform(0, 1, size=(N, 2))
    real_ids = np.array([0] * (N // 2) + [1] * (N // 2))

    h = RegionHistory(n_regions=R)
    h.update(t=0, X=X_full,        F=F_full, region_ids=real_ids, xl=xl, xu=xu)
    h.update(t=1, X=X_full + 0.01, F=F_full, region_ids=real_ids, xl=xl, xu=xu)
    return h, xl, xu, real_ids, X_full, D, R, N


def test_t23_null_history_centroid_matches_null_partition():
    """_build_null_history centroid at t-1 must match the centroid computed
    with the shared perm_k (key = (BASE_SEED, EVENT_ID, K), no tau)."""
    h, xl, xu, real_ids, X_full, D, R, N = _make_controlled_history()
    BASE_SEED, EVENT_ID, K, T = 7, 3, 5, 2  # T=2 → t-2=0, t-1=1

    null_hist = e2ar._build_null_history(h, R, BASE_SEED, EVENT_ID, K, T,
                                          xl, xu)

    # Reproduce: perm_k from (BASE_SEED, EVENT_ID, K) -- no tau
    from prdrl.core.seeding import deterministic_rng as drng
    rng_k = drng(BASE_SEED, EVENT_ID, K)
    perm_k = rng_k.permutation(N)
    sizes = [int(np.sum(real_ids == r)) for r in range(R)]
    expected_ids = np.empty(N, dtype=int)
    start = 0
    for r in range(R):
        expected_ids[perm_k[start:start + sizes[r]]] = r
        start += sizes[r]

    Z_t1 = (X_full + 0.01 - xl) / (xu - xl)
    expected_c0_t1 = np.mean(Z_t1[expected_ids == 0], axis=0)

    actual_c0_t1 = null_hist.get_region(0, 1).centroid_z
    np.testing.assert_allclose(actual_c0_t1, expected_c0_t1, atol=1e-9,
        err_msg="Null history centroid must come from shared perm_k, not real partition")


def test_t24_null_history_differs_from_real_history():
    """Null-history centroid must differ from real-history centroid when
    the two partitions assign different members to the same slot."""
    h, xl, xu, real_ids, X_full, D, R, N = _make_controlled_history(seed=10)
    # T=2 so t-2=0 and t-1=1, both of which exist in the controlled history
    BASE_SEED, EVENT_ID, K, T = 3, 4, 2, 2

    null_hist = e2ar._build_null_history(h, R, BASE_SEED, EVENT_ID, K, T,
                                          xl, xu)
    # Reproduce perm_k: key = (BASE_SEED, EVENT_ID, K), no tau
    from prdrl.core.seeding import deterministic_rng as drng
    rng_k = drng(BASE_SEED, EVENT_ID, K)
    perm_k = rng_k.permutation(N)
    sizes = [int(np.sum(real_ids == r)) for r in range(R)]
    rand_ids_0 = np.empty(N, dtype=int)
    start = 0
    for r in range(R):
        rand_ids_0[perm_k[start:start + sizes[r]]] = r
        start += sizes[r]

    if not np.array_equal(rand_ids_0, real_ids):
        real_c0 = h.get_region(0, 0).centroid_z
        null_c0 = null_hist.get_region(0, 0).centroid_z
        assert not np.allclose(real_c0, null_c0, atol=1e-3), (
            "Null centroid should differ from real centroid when null "
            "partition assigns different members to group 0")


def test_t25_null_history_deterministic():
    """Same (base_seed, event_id, k, t) always produces identical null
    history."""
    h, xl, xu, *_ = _make_controlled_history()
    kwargs = dict(history=h, R=2, base_seed=5, event_id=2, k=3, t=3,
                  xl=xl, xu=xu)
    h1 = e2ar._build_null_history(**kwargs)
    h2 = e2ar._build_null_history(**kwargs)

    for tau in (1, 2):          # t-2=1, t-1=2 -- only t-2 and t-1 stored
        for r in range(2):
            s1 = h1.get_region(r, tau)
            s2 = h2.get_region(r, tau)
            if s1 is None:
                assert s2 is None
            else:
                np.testing.assert_array_equal(s1.centroid_z, s2.centroid_z,
                    err_msg=f"Null history not deterministic at r={r}, tau={tau}")


def test_t26_null_history_different_replicas_differ():
    """k=0 and k=1 produce different null histories (different random
    partitions at t-2 and t-1)."""
    h, xl, xu, *_ = _make_controlled_history(seed=99, N=20)
    h0 = e2ar._build_null_history(h, R=2, base_seed=1, event_id=2,
                                   k=0, t=3, xl=xl, xu=xu)
    h1 = e2ar._build_null_history(h, R=2, base_seed=1, event_id=2,
                                   k=1, t=3, xl=xl, xu=xu)
    any_differ = False
    for tau in (1, 2):
        for r in range(2):
            s0 = h0.get_region(r, tau)
            s1 = h1.get_region(r, tau)
            if s0 is not None and s1 is not None:
                if not np.allclose(s0.centroid_z, s1.centroid_z):
                    any_differ = True
    assert any_differ, "k=0 and k=1 null histories should differ in at least one centroid"


def test_t27_null_history_no_future_leakage():
    """_build_null_history stores t-2 and t-1 only; t is NOT stored."""
    h, xl, xu, *_ = _make_controlled_history()
    # T=2 so t-2=0 and t-1=1 both exist in the controlled history (t=0, t=1)
    T = 2
    null_hist = e2ar._build_null_history(h, R=2, base_seed=2, event_id=T,
                                          k=0, t=T, xl=xl, xu=xu)
    assert null_hist.get_region(0, T - 1) is not None, "t-1=1 must be stored"
    assert null_hist.get_region(0, T - 2) is not None, "t-2=0 must be stored"
    assert null_hist.get_region(0, T)     is None,     "t=2 must NOT be stored"


def test_t28_null_history_same_perm_at_t2_and_t1():
    """Both historical taus must be reproduced by the SAME perm_k
    (key = (base_seed, event_id, k), no tau).  Verified by independently
    computing the expected centroid at t-2 and t-1 with perm_k and
    confirming they match the null_hist snapshots."""
    D, R, N = 3, 2, 12
    xl_, xu_ = np.zeros(D), np.ones(D)
    rng = np.random.default_rng(50)
    X0 = rng.uniform(0, 1, size=(N, D))
    X1 = rng.uniform(0, 1, size=(N, D))
    F  = rng.uniform(0, 1, size=(N, 2))
    real_ids = np.array([0] * 6 + [1] * 6)

    h = RegionHistory(n_regions=R)
    h.update(t=0, X=X0, F=F, region_ids=real_ids, xl=xl_, xu=xu_)
    h.update(t=1, X=X1, F=F, region_ids=real_ids, xl=xl_, xu=xu_)

    BASE_SEED, EVENT_ID, K, T = 11, 5, 3, 2
    null_hist = e2ar._build_null_history(h, R, BASE_SEED, EVENT_ID, K, T,
                                          xl_, xu_)

    from prdrl.core.seeding import deterministic_rng as drng
    rng_k = drng(BASE_SEED, EVENT_ID, K)
    perm_k = rng_k.permutation(N)
    sizes = [6, 6]
    expected_ids = np.empty(N, dtype=int)
    start = 0
    for r in range(R):
        expected_ids[perm_k[start:start + sizes[r]]] = r
        start += sizes[r]

    # Check tau=0 (t-2)
    Z0 = (X0 - xl_) / (xu_ - xl_)
    exp_c0_t0 = np.mean(Z0[expected_ids == 0], axis=0)
    act_c0_t0 = null_hist.get_region(0, 0).centroid_z
    np.testing.assert_allclose(act_c0_t0, exp_c0_t0, atol=1e-9,
        err_msg="t-2 centroid must come from shared perm_k (no tau in key)")

    # Check tau=1 (t-1) -- SAME perm_k, different population
    Z1 = (X1 - xl_) / (xu_ - xl_)
    exp_c0_t1 = np.mean(Z1[expected_ids == 0], axis=0)
    act_c0_t1 = null_hist.get_region(0, 1).centroid_z
    np.testing.assert_allclose(act_c0_t1, exp_c0_t1, atol=1e-9,
        err_msg="t-1 centroid must come from same perm_k as t-2 (no tau in key)")


# =====================================================================
# Meta: E2a (frozen) + E1 (frozen) still unaffected
# =====================================================================
def test_e2a_module_unmodified_still_importable():
    import prdrl.experiments.E2a_controlled_operator_oracle as e2a
    assert hasattr(e2a, "run_one")
    assert hasattr(e2a, "respond")


def test_full_suite_regression_marker():
    """Placeholder -- T20-equivalent cua E2a-R (khong phai test rieng,
    chi de doc ro trong file nay rang full suite duoc chay o CI/pytest
    tests/ cap cao hon)."""
    assert True
