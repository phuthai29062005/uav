"""10 invariant cua temporal segment-change estimator (4.3)."""
import os
import sys

import copy

import numpy as np
import pytest
from pymoo.core.problem import Problem
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from change_detector import ChangeDetector  # noqa: E402

D = 10
SEG2 = [[0], list(range(1, D))]
REF = (2.0, 2.0)


def probe(n=10, seed=0):
    return np.random.RandomState(seed).rand(n, D)


def det(segments=SEG2, **kw):
    return ChangeDetector(segments=segments, obj_scale=np.asarray(REF),
                          lb=np.zeros(D), ub=np.ones(D),
                          eps=0.01, kappa=2.0, lam=0.05, **kw)


# --------------------------------------------------------- TEST 1: STATIC
def test_static_environment_gives_zero_change():
    X = probe()
    d = det()
    d.prime(X, DF1(time=0.5, n_var=D))
    res = d.compute(X, DF1(time=0.5, n_var=D))
    assert np.all(res["c_tilde"] < 1e-6), res["c_tilde"]


# ---------------------------------------------------------- TEST 2: LOCAL
class SyntheticLocal(Problem):
    """
    f1 = x0 * (1 + a*t)          -> gradient theo segment 1 doi khi a != 0
    f2 = sum((x[1:] - b*t)^2)    -> gradient theo segment 2 doi khi b != 0
    """

    def __init__(self, time=0.0, a=0.0, b=0.0):
        super().__init__(n_var=D, n_obj=2, xl=0.0, xu=1.0)
        self.t, self.a, self.b = time, a, b

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:, 0] * (1.0 + self.a * self.t)
        f2 = np.sum((X[:, 1:] - self.b * self.t) ** 2, axis=1)
        out["F"] = np.column_stack([f1, f2])


@pytest.mark.parametrize("a,b,hot,cold", [(1.0, 0.0, 0, 1),
                                          (0.0, 1.0, 1, 0)])
def test_change_is_localized_to_the_segment_that_moved(a, b, hot, cold):
    X = probe()
    d = det()
    d.prime(X, SyntheticLocal(time=0.0, a=a, b=b))
    ct = d.compute(X, SyntheticLocal(time=1.0, a=a, b=b))["c_tilde"]
    assert ct[hot] > 5 * ct[cold], ct


# ------------------------------------------------- TEST 3: S=1 KHONG COLLAPSE
def _two_step(segments, t_mid, t_end):
    """prime@0.4 -> compute@t_mid (khoi tao EMA) -> compute@t_end."""
    X = probe()
    d = det(segments)
    d.prime(X, DF1(time=0.4, n_var=D))
    d.compute(X, DF1(time=t_mid, n_var=D))
    return d.compute(X, DF1(time=t_end, n_var=D))


def test_single_segment_does_not_collapse_to_one():
    small = _two_step([list(range(D))], 0.5, 0.51)["c"][0]
    med = _two_step([list(range(D))], 0.5, 0.6)["c"][0]
    big = _two_step([list(range(D))], 0.5, 1.0)["c"][0]
    assert not np.allclose([small, med, big], 1.0), (small, med, big)
    assert small < med < big, (small, med, big)


# --------------------------------------------------- TEST 4: SEGMENT SIZE
class SyntheticEqualSensitivity(Problem):
    """
    f1 = (1+t) * (1/sqrt(|s1|)) * sum_{j in s1} x_j
    f2 = (1+t) * (1/sqrt(|s2|)) * sum_{j in s2} x_j

    Chuan hoa 1/sqrt(|s|) khien directional derivative theo huong don
    vi cua moi segment BANG NHAU, bat ke |s|. He so (1+t) tao temporal
    change cung bien do cho ca hai segment.
    """

    def __init__(self, time=0.0, segs=SEG2):
        super().__init__(n_var=D, n_obj=2, xl=0.0, xu=1.0)
        self.t, self.segs = time, segs

    def _evaluate(self, X, out, *args, **kwargs):
        cols = [(1.0 + self.t) * X[:, g].sum(axis=1) / np.sqrt(len(g))
                for g in self.segs]
        out["F"] = np.column_stack(cols)


def test_no_dimensionality_bias_on_equal_sensitivity():
    """
    Hai segment 1 bien vs 9 bien nhung directional sensitivity that su
    bang nhau -> c_tilde phai bang nhau. Day la test sach cho viec
    chuan hoa eps/sqrt(n_s), khong lan cau truc cua DF.
    """
    X = probe()
    d = det(SEG2)
    d.prime(X, SyntheticEqualSensitivity(time=0.0))
    ct = d.compute(X, SyntheticEqualSensitivity(time=1.0))["c_tilde"]
    print(f"\n  c_tilde 1-bien={ct[0]:.6f}  9-bien={ct[1]:.6f}")
    assert abs(ct[0] / ct[1] - 1.0) < 0.05, ct


def test_segment_size_df1_observed():
    """
    MO TA, khong phai gate. Tren DF1: g = sum (x_i - G)^2 nen
    dg/dx_i = 2(x_i - G) bang nhau moi i khi hoi tu. Huong don vi cua
    segment s la (1/sqrt(n_s))*(1,..,1), nen
        grad_g . d_s = n_s * 2(x-G)/sqrt(n_s) = sqrt(n_s) * 2(x-G)
    => ty so ky vong = sqrt(9/5) = 1.342. Do la DAC TINH CAU TRUC cua
    DF (thay doi dong pha tren moi toa do), KHONG phai bias cua
    implementation — xem test_no_dimensionality_bias_on_equal_sensitivity.
    """
    X = probe()
    dA = det([[0], list(range(1, D))])
    dA.prime(X, DF1(time=0.5, n_var=D))
    ctA = dA.compute(X, DF1(time=0.6, n_var=D))["c_tilde"]

    dB = det([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    dB.prime(X, DF1(time=0.5, n_var=D))
    ctB = dB.compute(X, DF1(time=0.6, n_var=D))["c_tilde"]

    ratio = ctA[1] / float(np.mean(ctB))
    print(f"\n  DF1 c_tilde A(9 bien)={ctA[1]:.6f} "
          f"mean B(5+5)={np.mean(ctB):.6f} ratio={ratio:.3f} "
          f"(ky vong cau truc sqrt(9/5)={np.sqrt(9/5):.3f})")
    assert np.all(np.isfinite(ctA)) and np.all(np.isfinite(ctB))


# ----------------------------------------------- TEST 5: MAGNITUDE MONOTONIC
def test_magnitude_monotonic():
    a = _two_step(SEG2, 0.5, 0.52)
    b = _two_step(SEG2, 0.5, 0.6)
    c = _two_step(SEG2, 0.5, 0.8)
    for i in range(len(SEG2)):
        assert a["c_tilde"][i] < b["c_tilde"][i] < c["c_tilde"][i], (
            i, a["c_tilde"], b["c_tilde"], c["c_tilde"])
        assert a["c"][i] < b["c"][i] < c["c"][i], (
            i, a["c"], b["c"], c["c"])


# ============================================ BOUNDARY (bound-aware stencil)
class _RecordingProblem:
    """Boc mot problem, ghi lai moi input truyen vao evaluate()."""

    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def evaluate(self, X):
        self.seen.append(np.array(X, copy=True))
        return self.inner.evaluate(X)


def _assert_all_in_bounds(rec):
    for blk in rec.seen:
        assert blk.min() >= -1e-12 and blk.max() <= 1 + 1e-12, (
            f"probe ra ngoai [0,1]: min={blk.min()} max={blk.max()}")


def _boundary_probe(value):
    """x[0] dung tai bien, cac bien khac o giua."""
    X = probe(n=6, seed=3)
    X[:, 0] = value
    return X


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_boundary_stays_finite_and_in_domain(value):
    """TEST 6/7: probe sat bien duoi / bien tren."""
    X = _boundary_probe(value)
    d = det()
    r0 = _RecordingProblem(DF1(time=0.5, n_var=D))
    d.prime(X, r0)
    r1 = _RecordingProblem(DF1(time=0.6, n_var=D))
    res = d.compute(X, r1)

    assert np.all(np.isfinite(res["c_tilde"])), res["c_tilde"]
    assert np.all(np.isfinite(res["c"])), res["c"]
    _assert_all_in_bounds(r0)
    _assert_all_in_bounds(r1)
    assert d.last_prime_stats["onesided2"] > 0, d.last_prime_stats


def test_static_boundary_gives_zero_change():
    """
    TEST 8: probe nam dung tren bien o nhieu toa do, moi truong KHONG
    doi. Chung minh one-sided stencil khong tu tao ra tin hieu gia.
    """
    X = probe(n=6, seed=4)
    X[:, 0] = 0.0
    X[:, 1] = 1.0
    X[:, 2] = 0.0
    d = det()
    d.prime(X, DF1(time=0.5, n_var=D))
    res = d.compute(X, DF1(time=0.5, n_var=D))
    assert np.all(np.isfinite(res["c_tilde"]))
    assert np.all(res["c_tilde"] < 1e-6), res["c_tilde"]


def test_boundary_still_localizes_change():
    """TEST 9: one-sided van bat duoc thay doi cuc bo cua segment 1."""
    X = _boundary_probe(0.0)
    d = det()
    d.prime(X, SyntheticLocal(time=0.0, a=1.0, b=0.0))
    ct = d.compute(X, SyntheticLocal(time=1.0, a=1.0, b=0.0))["c_tilde"]
    assert np.all(np.isfinite(ct)), ct
    assert ct[0] > 5 * ct[1], ct


def test_stencil_invariant_across_time():
    """TEST 10: stencil quyet dinh o prime() va KHONG BAO GIO doi."""
    X = _boundary_probe(0.0)
    d = det()
    d.prime(X, DF1(time=0.5, n_var=D))
    frozen = copy.deepcopy(d._stencils)

    for t in [0.6, 0.7, 0.8]:
        d.compute(X, DF1(time=t, n_var=D))
        for i in range(len(X)):
            for s in range(len(SEG2)):
                a, b = frozen[i][s], d._stencils[i][s]
                assert a.mode == b.mode, (t, i, s, a.mode, b.mode)
                assert np.allclose(a.direction, b.direction), (t, i, s)
                assert a.h == b.h, (t, i, s)


def test_compute_without_prime_raises():
    with pytest.raises(RuntimeError):
        det().compute(probe(), DF1(time=0.5, n_var=D))
