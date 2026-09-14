"""Giai doan A -- unit tests cho core/normalize, core/reference,
core/partition, detectors/severity, detectors/rhi, metrics/igd.

Khoa dung cong thuc (eq. 24-25, 26-27, 28-29, 30-33, 38, 64-65) truoc
khi dung lam co so cho Gate A1/A2.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.core.normalize import compute_ideal_nadir, normalize, normalize_pair  # noqa: E402
from prdrl.core.partition import assign_regions, region_indices  # noqa: E402
from prdrl.core.reference import das_dennis_count, make_reference_directions  # noqa: E402
from prdrl.detectors.rhi import (percentile_rank, random_partition_same_sizes,  # noqa: E402
                                 rhi, rhi_null_distribution)
from prdrl.detectors.severity import CausalSeverityScale, displacement, region_severity_raw  # noqa: E402
from prdrl.metrics.igd import igd, migd  # noqa: E402


# ---------------------------------------------------------------------
# core/normalize.py -- eq. 24-25
# ---------------------------------------------------------------------
def test_ideal_nadir_is_over_union_not_either_alone():
    F_minus = np.array([[0.0, 1.0], [0.5, 0.5]])
    F_plus = np.array([[1.0, 0.0], [0.2, 0.8]])
    z_star, z_nad = compute_ideal_nadir(F_minus, F_plus)
    # min/max must consider BOTH F_minus and F_plus together
    np.testing.assert_allclose(z_star, [0.0, 0.0])
    np.testing.assert_allclose(z_nad, [1.0, 1.0])


def test_normalize_maps_ideal_to_0_and_nadir_to_1():
    F_minus = np.array([[0.0, 10.0], [5.0, 5.0]])
    F_plus = np.array([[10.0, 0.0], [5.0, 5.0]])
    z_star, z_nad = compute_ideal_nadir(F_minus, F_plus)
    Fn = normalize(F_minus, z_star, z_nad, eps=0.0)
    assert Fn[0, 0] == pytest.approx(0.0)   # was ideal in dim0
    assert Fn[0, 1] == pytest.approx(1.0)   # was nadir in dim1


def test_normalize_pair_consistent_with_compute_ideal_nadir():
    rng = np.random.default_rng(0)
    Fm = rng.uniform(0, 5, size=(10, 3))
    Fp = rng.uniform(0, 5, size=(10, 3))
    Fm_t, Fp_t, z_star, z_nad = normalize_pair(Fm, Fp)
    z_star2, z_nad2 = compute_ideal_nadir(Fm, Fp)
    np.testing.assert_allclose(z_star, z_star2)
    np.testing.assert_allclose(z_nad, z_nad2)
    assert Fm_t.min() >= -1e-9 and Fp_t.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------
# core/reference.py -- eq. 26-27, Bang 11.1
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n_obj,h,expected_R", [(2, 9, 10), (3, 4, 15), (5, 2, 15)])
def test_das_dennis_count_matches_spec_table(n_obj, h, expected_R):
    assert das_dennis_count(n_obj, h) == expected_R


def test_reference_directions_sum_to_one_and_nonnegative():
    V = make_reference_directions(2, 9)
    assert V.shape == (10, 2)
    np.testing.assert_allclose(V.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(V >= -1e-9)


# ---------------------------------------------------------------------
# core/partition.py -- eq. 28-29, "membership dong bang" nguyen tac
# ---------------------------------------------------------------------
def test_partition_uses_only_f_minus_frozen_membership():
    """API-level guard: assign_regions() chi nhan MOT tap diem -- khong
    co tham so F_plus nao de vo tinh dung nham (dung y 'bay nguy hiem
    nhat' trong dac ta)."""
    import inspect
    sig = inspect.signature(assign_regions)
    params = list(sig.parameters.keys())
    assert params == ["F_minus_tilde", "V", "eps"], params


def test_partition_assigns_axis_extreme_points_to_axis_regions():
    V = make_reference_directions(2, 9)   # region 0 = (0,1), region 9 = (1,0)
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    r = assign_regions(F, V)
    assert r[0] == 0
    assert r[1] == 9


def test_region_indices_covers_all_points_exactly_once():
    V = make_reference_directions(2, 9)
    rng = np.random.default_rng(1)
    F = rng.uniform(0, 1, size=(50, 2))
    r_of_i = assign_regions(F, V)
    idx = region_indices(r_of_i, len(V))
    all_idx = np.concatenate([v for v in idx.values() if len(v)])
    assert sorted(all_idx.tolist()) == list(range(50))


# ---------------------------------------------------------------------
# detectors/severity.py -- eq. 30-33
# ---------------------------------------------------------------------
def test_displacement_zero_for_identical_points():
    F = np.array([[0.1, 0.2], [0.3, 0.4]])
    d = displacement(F, F)
    np.testing.assert_allclose(d, 0.0)


def test_region_severity_raw_is_median_and_none_for_empty():
    d_i = np.array([1.0, 2.0, 3.0, 100.0])
    region_idx = {0: np.array([0, 1, 2]), 1: np.array([], dtype=int)}
    s = region_severity_raw(d_i, region_idx)
    assert s[0] == pytest.approx(2.0)   # median of [1,2,3]
    assert s[1] is None


def test_causal_severity_scale_cold_start_then_causal_update():
    scale = CausalSeverityScale(kappa=1.0, rho=0.5)
    c1 = scale.update({0: 1.0, 1: 3.0})
    assert c1[0] == 0.0 and c1[1] == 0.0          # cold start -> c=0
    assert scale.q == pytest.approx(2.0)          # median([1,3])

    c2 = scale.update({0: 1.0, 1: 3.0})
    # eq.32 using q_{t-1}=2.0 (NOT yet updated with this step's s_r)
    expected0 = 1 - np.exp(-1.0 / (1.0 * 2.0 + 1e-12))
    assert c2[0] == pytest.approx(expected0, rel=1e-6)
    # eq.33: q updated AFTER computing c
    assert scale.q == pytest.approx(0.5 * 2.0 + 0.5 * 2.0)  # unchanged since median same


def test_causal_severity_scale_none_for_inactive_region():
    scale = CausalSeverityScale()
    c = scale.update({0: 1.0, 1: None})
    assert c[1] is None


# ---------------------------------------------------------------------
# detectors/rhi.py -- eq. 38 + random-partition control
# ---------------------------------------------------------------------
def test_rhi_zero_when_all_severities_equal():
    s_r = {0: 1.0, 1: 1.0, 2: 1.0}
    assert rhi(s_r) == pytest.approx(0.0)


def test_rhi_positive_when_severities_differ():
    s_r = {0: 1.0, 1: 5.0, 2: 1.0}
    assert rhi(s_r) > 0.0


def test_rhi_ignores_inactive_regions():
    s_r_a = {0: 1.0, 1: 5.0}
    s_r_b = {0: 1.0, 1: 5.0, 2: None}
    assert rhi(s_r_a) == pytest.approx(rhi(s_r_b))


def test_random_partition_same_sizes_preserves_total_and_sizes():
    rng = np.random.default_rng(2)
    sizes = [3, 5, 2]
    out = random_partition_same_sizes(10, sizes, rng)
    assert [len(v) for v in out.values()] == sizes
    all_idx = np.concatenate(list(out.values()))
    assert sorted(all_idx.tolist()) == list(range(10))


def test_percentile_rank_basic():
    null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(3.0, null) == pytest.approx(0.6)   # 3 values <= 3.0
    assert percentile_rank(0.0, null) == pytest.approx(0.0)
    assert percentile_rank(10.0, null) == pytest.approx(1.0)


def test_rhi_null_distribution_shape_and_reproducibility():
    d_i = np.abs(np.random.default_rng(3).normal(size=30))
    sizes = [10, 10, 10]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = rhi_null_distribution(d_i, sizes, 50, rng1)
    out2 = rhi_null_distribution(d_i, sizes, 50, rng2)
    assert out1.shape == (50,)
    np.testing.assert_allclose(out1, out2)   # same seed -> reproducible


# ---------------------------------------------------------------------
# metrics/igd.py -- eq. 64-65
# ---------------------------------------------------------------------
def test_igd_zero_for_exact_match():
    P = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
    assert igd(P, P) == pytest.approx(0.0)


def test_igd_positive_offset():
    P = np.array([[0.0, 0.0]])
    A = np.array([[3.0, 4.0]])
    assert igd(P, A) == pytest.approx(5.0)   # 3-4-5 triangle


def test_migd_ignores_none_and_averages():
    assert migd([1.0, 2.0, None, 3.0]) == pytest.approx(2.0)


def test_migd_empty_is_inf():
    assert migd([]) == float("inf")
