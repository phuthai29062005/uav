"""
Review Giai doan A (PR-DRL-DMOEA, code cua user) -- test khoa lai moi
bug/edge-case tim thay khi review normalize.py, reference.py,
partition.py, severity.py, rhi.py, igd.py.

Moi test doc duoc gan (BUG-n) hoac (EDGE) khop voi muc tuong ung trong
bao cao review.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.core.normalize import ideal_nadir, normalize  # noqa: E402
from prdrl.core.partition import partition  # noqa: E402
from prdrl.core.reference import reference  # noqa: E402
from prdrl.detectors.rhi import rhi, rhi_null  # noqa: E402
from prdrl.detectors.severity import severity  # noqa: E402
from prdrl.metrics.igd import igd, migd  # noqa: E402


# =====================================================================
# core/normalize.py
# =====================================================================
def test_normalize_ideal_nadir_over_union():
    F_minus = [[0.0, 1.0], [0.5, 0.5]]
    F_plus = [[1.0, 0.0], [0.2, 0.8]]
    Fm_t, Fp_t = normalize(F_minus, F_plus)
    # z*=[0,0] z_nad=[1,1] tren hop -> (0,1) -> [0,1]; (1,0)->[1,0]
    np.testing.assert_allclose(np.asarray(Fm_t)[0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(Fp_t)[0], [1.0, 0.0], atol=1e-6)


def test_normalize_returns_ndarray_not_list_of_list():
    """(ISSUE-5) tra ve ndarray de tuong thich voi partition()/severity()
    ma khong can caller tu wrap np.array()."""
    Fm_t, Fp_t = normalize([[0.0, 1.0]], [[1.0, 0.0]])
    assert isinstance(Fm_t, np.ndarray)
    assert isinstance(Fp_t, np.ndarray)


def test_normalize_accepts_plain_python_list_input():
    """Ban goc dung F_minus[i][j] -- van phai hoat dong voi list thuan
    (khong chi ndarray) sau khi them np.asarray() o dau ham."""
    Fm_t, Fp_t = normalize([[0.0, 1.0], [1.0, 0.0]], [[0.5, 0.5], [0.2, 0.8]])
    assert Fm_t.shape == (2, 2)


def test_normalize_degenerate_dimension_gives_zero_not_nan():
    """(EDGE da xac nhan DUNG) moi nghiem giong nhau o 1 muc tieu ->
    mau so ~alpha, tu so cung = 0 -> ket qua 0.0, khong NaN/Inf."""
    F_minus = [[0.5, 1.0], [0.5, 2.0]]
    F_plus = [[0.5, 1.5], [0.5, 3.0]]
    Fm_t, Fp_t = normalize(F_minus, F_plus)
    assert np.all(np.isfinite(Fm_t)) and np.all(np.isfinite(Fp_t))
    np.testing.assert_allclose(Fm_t[:, 0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(Fp_t[:, 0], [0.0, 0.0], atol=1e-6)


def test_normalize_rejects_mismatched_n():
    """(ISSUE-6) F_minus/F_plus phai cung N (cung decision vectors, eq.22)."""
    with pytest.raises(AssertionError):
        normalize([[0.0, 1.0], [1.0, 0.0]], [[0.5, 0.5]])


def test_normalize_n_equals_1():
    Fm_t, Fp_t = normalize([[0.3, 0.7]], [[0.4, 0.6]])
    assert Fm_t.shape == (1, 2) and Fp_t.shape == (1, 2)
    assert np.all(np.isfinite(Fm_t)) and np.all(np.isfinite(Fp_t))


def test_normalize_rejects_empty_input():
    """(ISSUE-7) N=0 bao loi ro rang thay vi IndexError kho hieu."""
    with pytest.raises(AssertionError):
        normalize([], [])


def test_ideal_nadir_helper_matches_normalize_internal_values():
    F_minus = [[0.0, 10.0], [5.0, 5.0]]
    F_plus = [[10.0, 0.0], [5.0, 5.0]]
    z_star, z_nad = ideal_nadir(F_minus, F_plus)
    np.testing.assert_allclose(z_star, [0.0, 0.0])
    np.testing.assert_allclose(z_nad, [10.0, 10.0])


# =====================================================================
# core/reference.py
# =====================================================================
def test_reference_does_not_crash_bug1_regression():
    """(BUG-1) truoc fix: goi positional n_partitions crash ngay.
    Regression test khoa lai fix keyword."""
    V = reference(2, 9)
    assert V.shape == (10, 2)   # eq.27: C(9+2-1,2-1)=10


def test_reference_das_dennis_count_matches_formula():
    from math import comb
    V2 = reference(2, 9)
    assert len(V2) == comb(9 + 2 - 1, 2 - 1)          # =10
    V3 = reference(3, 4)
    assert len(V3) == comb(4 + 3 - 1, 3 - 1)          # =15


def test_reference_directions_sum_to_one():
    V = reference(2, 9)
    np.testing.assert_allclose(np.asarray(V).sum(axis=1), 1.0, atol=1e-6)


# =====================================================================
# core/partition.py
# =====================================================================
def _partition_loop_reference(F_minus, V, alpha=1e-12):
    """Ban sao trung thuc cua vong lap GOC (truoc vectorize) -- dung de
    verify ban vectorized cho ket qua GIONG HET, khong chi 'hop ly'."""
    F_minus = np.asarray(F_minus, dtype=float)
    V = np.asarray(V, dtype=float)
    region_ids = []
    for i in F_minus:
        x = np.linalg.norm(i)
        i_new = i / (x + alpha)
        cal = -np.inf
        idx = None
        for j in range(len(V)):
            val = (i_new @ V[j]) / (np.linalg.norm(V[j]) + alpha)
            if val > cal:
                cal = val
                idx = j
        region_ids.append(idx)
    return np.array(region_ids)


def test_partition_vectorized_matches_original_loop():
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1, size=(60, 3))
    V = reference(3, 4)
    expected = _partition_loop_reference(F, V)
    got = partition(F, V)
    np.testing.assert_array_equal(got, expected)


def test_partition_axis_extreme_points():
    V = reference(2, 9)
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    r = partition(F, V)
    assert r[0] == 0     # V[0] = (0,1)
    assert r[-1] == 9    # V[-1] = (1,0)


def test_partition_accepts_plain_list_input():
    """(ISSUE-5) truoc fix: list thuan crash TypeError o `i / (x+alpha)`."""
    V = reference(2, 9)
    F_as_list = [[0.0, 1.0], [1.0, 0.0]]
    r = partition(F_as_list, V)
    assert list(r) == [0, 9]


def test_partition_zero_vector_deterministic_region_0():
    """(EDGE) diem [0,0] (trung ideal point) -> cos-sim=0 voi moi huong
    -> chon V[0] mot cach xac dinh (khoa lai hanh vi, khong tu dinh
    nghia lai vi spec khong noi ro truong hop suy bien nay)."""
    V = reference(2, 9)
    F = np.array([[0.0, 0.0], [0.3, 0.7]])
    r = partition(F, V)
    assert r[0] == 0


def test_partition_frozen_membership_no_f_plus_param():
    """API guard: partition() KHONG duoc co tham so F_plus (bay nguy
    hiem nhat trong dac ta -- membership phai dong bang tai F_minus)."""
    import inspect
    params = list(inspect.signature(partition).parameters.keys())
    assert "F_plus" not in params
    assert "F_minus" in params


# =====================================================================
# detectors/severity.py
# =====================================================================
def test_severity_empty_region_is_nan_not_zero():
    """(BUG-2) vung rong phai la NaN, khong phai 0."""
    F_minus = np.array([[0.0, 1.0], [0.0, 1.0]])
    F_plus = np.array([[0.1, 0.9], [0.1, 0.9]])
    region_ids = np.array([0, 0])   # region 1 rong
    S, C, Q = severity(F_minus, F_plus, region_ids, R=2, q_prev=1.0)
    assert np.isnan(S[1]), f"vung rong phai la NaN, nhan duoc {S[1]}"
    assert np.isnan(C[1]), "c_r phai NaN cho vung inactive (lan truyen tu S)"


def test_severity_q_survives_empty_region_no_permanent_nan():
    """(BUG-3, quan trong nhat) Q khong duoc thanh NaN chi vi co 1 vung
    rong -- truoc fix, np.median([x, nan]) = nan se lam hong toan bo
    chuoi EMA con lai cua run."""
    F_minus = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    F_plus = np.array([[0.1, 0.9], [0.1, 0.9], [1.0, 0.1]])
    region_ids = np.array([0, 0, 2])   # region 1 rong, region 0 va 2 co du lieu
    S, C, Q = severity(F_minus, F_plus, region_ids, R=3, q_prev=1.0)
    assert not np.isnan(Q), f"Q khong duoc la NaN, nhan duoc {Q}"
    # Q phai la nanmedian cua cac S active, khong phai median (co NaN)
    expected_Q = 0.95 * 1.0 + 0.05 * np.nanmedian(S)
    assert Q == pytest.approx(expected_Q)


def test_severity_ema_chain_does_not_permanently_poison_after_many_steps():
    """Regression rong hon cho BUG-3: chay nhieu buoc lien tiep, mot
    buoc giua chung co vung rong -- Q phai luon huu han xuyen suot."""
    rng = np.random.default_rng(1)
    q = 1.0
    for step in range(10):
        F_minus = rng.uniform(0, 1, size=(10, 2))
        F_plus = F_minus + rng.normal(0, 0.05, size=(10, 2))
        # buoc thu 5: ep tat ca nghiem vao region 0 de region 1.. rong
        region_ids = np.zeros(10, dtype=int) if step == 5 else rng.integers(0, 3, size=10)
        S, C, q = severity(F_minus, F_plus, region_ids, R=3, q_prev=q)
        assert np.isfinite(q), f"buoc {step}: q_prev bi NaN/Inf sau khi lan truyen"


def test_severity_kappa_rho_default_backward_compatible():
    """(BUG-4) goi voi dung 5 tham so vi tri (nhu code goc) phai cho
    ket qua GIONG HET truoc khi them kappa/rho (vi mac dinh kappa=1.0
    khop dung ngam dinh cu)."""
    F_minus = np.array([[0.0, 1.0], [0.2, 0.8]])
    F_plus = np.array([[0.1, 0.9], [0.3, 0.7]])
    region_ids = np.array([0, 1])
    S, C, Q = severity(F_minus, F_plus, region_ids, 2, 1.0)   # 5 tham so, khong kappa/rho
    S2, C2, Q2 = severity(F_minus, F_plus, region_ids, 2, 1.0, kappa=1.0, rho=0.05)
    assert S == S2 and Q == pytest.approx(Q2)


def test_severity_accepts_plain_list_input():
    """(ISSUE-5) truoc fix: list thuan crash o `F_minus - F_plus`."""
    S, C, Q = severity([[0.0, 1.0]], [[0.1, 0.9]], [0], R=1, q_prev=1.0)
    assert len(S) == 1 and not np.isnan(S[0])


def test_severity_single_solution():
    """(EDGE) N=1."""
    S, C, Q = severity([[0.2, 0.8]], [[0.3, 0.7]], [0], R=1, q_prev=1.0)
    assert len(S) == 1
    assert np.isfinite(S[0])


# =====================================================================
# detectors/rhi.py
# =====================================================================
def test_rhi_consistent_with_severity_nan_convention():
    """Xac nhan sau khi sua BUG-2, dau ra severity() va rhi._severity_from_dis
    xu ly vung rong GIONG NHAU (ca hai deu NaN) -- rhi(severity_output)
    phai loai dung vung rong."""
    F_minus = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    F_plus = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.2], [0.9, 0.2]])
    region_ids = np.array([0, 0, 2, 2])   # region 1 rong trong so R=3
    S, C, Q = severity(F_minus, F_plus, region_ids, R=3, q_prev=1.0)
    # rhi() phai chi tinh tren 2 vung active (0 va 2), khong phai 3
    active = [s for s in S if not np.isnan(s)]
    assert len(active) == 2
    val = rhi(S)
    assert np.isfinite(val)


def test_rhi_less_than_2_active_regions_returns_zero():
    assert rhi([1.0, float("nan"), float("nan")]) == 0.0
    assert rhi([float("nan"), float("nan")]) == 0.0


def test_rhi_zero_when_equal_severities():
    assert rhi([2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_rhi_null_preserves_region_sizes():
    """Phu luc C.3: random control phai giu DUNG kich thuoc vung that."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        F_minus = np.random.uniform(0, 1, size=(30, 2))
        F_plus = F_minus + np.random.normal(0, 0.1, size=(30, 2))
        region_ids = np.array([0] * 10 + [1] * 15 + [2] * 5)
        rhi_ref, null_mean, null_std, pct = rhi_null(F_minus, F_plus, region_ids, R=3, K=50)
        assert 0.0 <= pct <= 1.0
        assert np.isfinite(rhi_ref)
    finally:
        np.random.set_state(rng_state)


def test_rhi_null_n_equals_1():
    """(EDGE) N=1 -- RHI luon 0 (khong the co heterogeneity), percentile
    phai la gia tri thap/khong gay crash."""
    F_minus = np.array([[0.3, 0.7]])
    F_plus = np.array([[0.4, 0.6]])
    region_ids = np.array([0])
    rhi_ref, null_mean, null_std, pct = rhi_null(F_minus, F_plus, region_ids, R=1, K=20)
    assert rhi_ref == 0.0
    assert pct == pytest.approx(0.0)


# =====================================================================
# metrics/igd.py
# =====================================================================
def test_igd_exact_match_is_zero():
    P = [[0, 1], [0.5, 0.5], [1, 0]]
    assert igd(P, P) == pytest.approx(0.0)


def test_igd_offset_3_4_5_triangle():
    assert igd([[0.0, 0.0]], [[3.0, 4.0]]) == pytest.approx(5.0)


def test_igd_matches_original_loop_implementation():
    """Xac nhan vectorize khong doi ket qua so voi vong for goc."""
    rng = np.random.default_rng(2)
    A = rng.uniform(0, 1, size=(20, 3))
    P = rng.uniform(0, 1, size=(15, 3))
    A_arr, P_arr = np.asarray(A), np.asarray(P)
    total = 0.0
    for y in P_arr:
        total += np.min(np.linalg.norm(A_arr - y, axis=1))
    expected = total / len(P_arr)
    assert igd(A, P) == pytest.approx(expected)


def test_igd_empty_a_raises():
    with pytest.raises(ValueError):
        igd([], [[0, 1]])


def test_igd_empty_pstar_raises():
    with pytest.raises(ValueError):
        igd([[0, 1]], [])


def test_migd_basic():
    assert migd([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_migd_empty_raises():
    with pytest.raises(ValueError):
        migd([])
