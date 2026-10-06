"""Cross-validation of CEC2018 DF1-DF14 against official MATLAB cec2018_DF.m.

Each test evaluates the problem at a known point and compares to the value
computed by hand from the MATLAB code. This catches translation errors
between MATLAB 1-based and Python 0-based indexing, and verifies every
formula term.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.problems.cec2018 import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14,
)
from src.problems.base import DynamicBenchmarkProblem, T0

ATOL = 1e-12


# =====================================================================
# T01-T03: Time model tests
# =====================================================================

class TestTimeModel:
    """Verify T0=50 time model matches MATLAB:
    tau_tmp = max(tau + taut - (T0+1), 0)
    t = (1/nt) * floor(tau_tmp / taut)
    """

    def test_t01_before_first_change(self):
        """tau < T0+1 should give t=0."""
        p = DF1(n_var=10, nt=10, taut=10)
        for tau in [0, 1, 10, 40, 50]:
            assert p.time(tau) == 0.0, f"tau={tau} should give t=0"

    def test_t02_first_change(self):
        """tau=T0+1=51 with taut=10: tau_tmp=max(51+10-51,0)=10, t=(1/10)*1=0.1."""
        p = DF1(n_var=10, nt=10, taut=10)
        assert p.time(51) == pytest.approx(0.1)

    def test_t03_multiple_changes(self):
        """Verify time progression at taut=10, nt=10."""
        p = DF1(n_var=10, nt=10, taut=10)
        # tau=51: tau_tmp=10, t=0.1
        # tau=61: tau_tmp=20, t=0.2
        # tau=71: tau_tmp=30, t=0.3
        assert p.time(61) == pytest.approx(0.2)
        assert p.time(71) == pytest.approx(0.3)

    def test_t03b_taut30(self):
        """Verify with taut=30."""
        p = DF1(n_var=10, nt=10, taut=30)
        # tau=51: tau_tmp=max(51+30-51,0)=30, t=(1/10)*1=0.1
        # tau=81: tau_tmp=max(81+30-51,0)=60, t=(1/10)*2=0.2
        assert p.time(51) == pytest.approx(0.1)
        assert p.time(81) == pytest.approx(0.2)

    def test_t03c_t0_constant(self):
        assert T0 == 50


# =====================================================================
# T04-T17: Individual problem formula tests (2-obj)
# =====================================================================

def _eval_at(prob_cls, x_vec, t_val, n_var=10):
    """Evaluate problem at a single point with given t."""
    p = prob_cls(n_var=n_var)
    X = np.array(x_vec).reshape(1, -1)
    return p.evaluate(X, t=t_val).ravel()


class TestDF1:
    def test_t04_formula(self):
        """DF1 at t=0.5: G=abs(sin(pi/4))=sqrt(2)/2, H=0.75*sin(pi/4)+1.25"""
        t = 0.5
        x = np.full(10, 0.3)
        G = np.abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        expected_f1 = x[0]
        expected_f2 = g * (1.0 - (x[0] / g) ** H)
        result = _eval_at(DF1, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)

    def test_t04b_pf_on_front(self):
        """On PF (g=1), f2 = 1 - f1^H."""
        t = 0.7
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        G = np.abs(np.sin(0.5 * np.pi * t))
        x = np.zeros(10)
        x[0] = 0.4
        x[1:] = G  # makes g=1
        result = _eval_at(DF1, x, t)
        assert result[0] == pytest.approx(0.4)
        assert result[1] == pytest.approx(1.0 - 0.4 ** H, abs=ATOL)


class TestDF2:
    def test_t05_formula(self):
        """DF2: r index changes with t."""
        t = 0.3
        n = 10
        G = np.abs(np.sin(0.5 * np.pi * t))
        r = int(np.floor((n - 1) * G))
        not_r = [k for k in range(n) if k != r]
        x = np.linspace(0.1, 0.9, n)
        g = 1.0 + np.sum((x[not_r] - G) ** 2)
        expected_f1 = x[r]
        expected_f2 = g * (1.0 - (x[r] / g) ** 0.5)
        result = _eval_at(DF2, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)


class TestDF3:
    def test_t06_formula(self):
        """DF3: g uses x(1)^H, not (x_i-G-x1)^H."""
        t = 0.4
        G = np.sin(0.5 * np.pi * t)
        H = G + 1.5
        x = np.linspace(0.1, 0.9, 10)
        g = 1.0 + np.sum((x[1:] - G - x[0] ** H) ** 2)
        expected_f1 = x[0]
        expected_f2 = g * (1.0 - (x[0] / g) ** H)
        result = _eval_at(DF3, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)

    def test_t06b_bounds(self):
        """DF3 bounds: x1 in [0,1], x2..xn in [-1,2]."""
        p = DF3()
        assert p.xl[0] == 0.0
        assert p.xu[0] == 1.0
        assert p.xl[1] == -1.0
        assert p.xu[1] == 2.0


class TestDF4:
    def test_t07_formula(self):
        """DF4: g divides by index [2:n], not by x values."""
        t = 0.6
        n = 10
        a = np.sin(0.5 * np.pi * t)
        b = 1.0 + np.abs(np.cos(0.5 * np.pi * t))
        c = max(np.abs(a), a + b)
        H = 1.5 + a
        x = np.linspace(0.1, 0.9, n)
        indices = np.arange(2, n + 1, dtype=float)
        g = 1.0 + np.sum((x[1:] - a * (x[0] / c) ** 2 / indices) ** 2)
        expected_f1 = g * np.abs(x[0] - a) ** H
        expected_f2 = g * np.abs(x[0] - a - b) ** H
        result = _eval_at(DF4, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)

    def test_t07b_different_from_df1(self):
        """DF4 must produce different values from DF1 at same point."""
        t = 0.5
        x = np.full(10, 0.5)
        r1 = _eval_at(DF1, x, t)
        r4 = _eval_at(DF4, x, t)
        assert not np.allclose(r1, r4), "DF4 must differ from DF1"

    def test_t07c_bounds(self):
        """DF4 bounds: all in [-2, 2]."""
        p = DF4()
        np.testing.assert_array_equal(p.xl, np.full(10, -2.0))
        np.testing.assert_array_equal(p.xu, np.full(10, 2.0))


class TestDF5:
    def test_t08_formula(self):
        """DF5: f1=g*(x1+0.02*sin(w*pi*x1)), f2=g*(1-x1+0.02*sin(w*pi*x1))."""
        t = 0.3
        G = np.sin(0.5 * np.pi * t)
        w = np.floor(10.0 * G)
        x = np.full(10, 0.5)
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        expected_f1 = g * (x[0] + 0.02 * np.sin(w * np.pi * x[0]))
        expected_f2 = g * (1.0 - x[0] + 0.02 * np.sin(w * np.pi * x[0]))
        result = _eval_at(DF5, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)


class TestDF6:
    def test_t09_formula(self):
        """DF6: Rastrigin-like g function."""
        t = 0.4
        G = np.sin(0.5 * np.pi * t)
        a = 0.2 + 2.8 * np.abs(G)
        x = np.full(10, 0.5)
        y = x[1:] - G
        g = 1.0 + np.sum(np.abs(G) * y ** 2 - 10.0 * np.cos(2.0 * np.pi * y) + 10.0)
        expected_f1 = g * (x[0] + 0.1 * np.sin(3.0 * np.pi * x[0])) ** a
        expected_f2 = g * (1.0 - x[0] + 0.1 * np.sin(3.0 * np.pi * x[0])) ** a
        result = _eval_at(DF6, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)


class TestDF7:
    def test_t10_formula(self):
        """DF7: sigmoid linkage, f1=g*(1+t)/x1, f2=g*x1/(1+t)."""
        t = 0.5
        x = np.full(10, 2.0)
        a = 5.0 * np.cos(0.5 * np.pi * t)
        tmp = 1.0 / (1.0 + np.exp(a * (x[0] - 2.5)))
        g = 1.0 + np.sum((x[1:] - tmp) ** 2)
        expected_f1 = g * (1.0 + t) / x[0]
        expected_f2 = g * x[0] / (1.0 + t)
        result = _eval_at(DF7, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)

    def test_t10b_bounds(self):
        """DF7: x1 in [1,4], x2..xn in [0,1]."""
        p = DF7()
        assert p.xl[0] == 1.0
        assert p.xu[0] == 4.0
        assert p.xl[1] == 0.0
        assert p.xu[1] == 1.0


class TestDF8:
    def test_t11_formula(self):
        """DF8: sin(4*pi*x1), NO exponent b; f2 uses x1 NOT x2."""
        t = 0.3
        G = np.sin(0.5 * np.pi * t)
        a = 2.25 + 2.0 * np.cos(2.0 * np.pi * t)
        x = np.linspace(0.1, 0.9, 10)
        tmp = G * np.sin(4.0 * np.pi * x[0]) / (1.0 + np.abs(G))
        g = 1.0 + np.sum((x[1:] - tmp) ** 2)
        expected_f1 = g * (x[0] + 0.1 * np.sin(3.0 * np.pi * x[0]))
        expected_f2 = g * (1.0 - x[0] + 0.1 * np.sin(3.0 * np.pi * x[0])) ** a
        result = _eval_at(DF8, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)


class TestDF9:
    def test_t12_formula(self):
        """DF9: sequential g with cos(4t+x1+x_{i-1})."""
        t = 0.4
        n = 10
        N = 1.0 + np.floor(10.0 * np.abs(np.sin(0.5 * np.pi * t)))
        x = np.linspace(0.1, 0.9, n)
        g = 1.0
        for i in range(1, n):
            tmp = x[i] - np.cos(4.0 * t + x[0] + x[i - 1])
            g += tmp ** 2
        expected_f1 = g * (x[0] + max(0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * x[0])))
        expected_f2 = g * (1.0 - x[0] + max(0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * x[0])))
        result = _eval_at(DF9, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2], atol=ATOL)


class TestDF10:
    def test_t13_formula(self):
        """DF10: tmp uses 4*pi NOT 2*pi; uses sin^H not cos."""
        t = 0.3
        G = np.sin(0.5 * np.pi * t)
        H = 2.25 + 2.0 * np.cos(0.5 * np.pi * t)
        x = np.linspace(0.1, 0.9, 10)
        tmp = np.sin(4.0 * np.pi * (x[0] + x[1])) / (1.0 + np.abs(G))
        g = 1.0 + np.sum((x[2:] - tmp) ** 2)
        expected_f1 = g * np.sin(0.5 * np.pi * x[0]) ** H
        expected_f2 = g * (np.sin(0.5 * np.pi * x[1]) * np.cos(0.5 * np.pi * x[0])) ** H
        expected_f3 = g * (np.cos(0.5 * np.pi * x[1]) * np.cos(0.5 * np.pi * x[0])) ** H
        result = _eval_at(DF10, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2, expected_f3], atol=ATOL)


class TestDF11:
    def test_t14_formula(self):
        """DF11: G=abs(sin(...)), g=1+G+sum(...)."""
        t = 0.4
        G = np.abs(np.sin(0.5 * np.pi * t))
        x = np.linspace(0.1, 0.9, 10)
        g = 1.0 + G + np.sum((x[2:] - 0.5 * G * x[0]) ** 2)
        y0 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * x[0]
        y1 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * x[1]
        expected = [g * np.sin(y0), g * np.sin(y1) * np.cos(y0), g * np.cos(y1) * np.cos(y0)]
        result = _eval_at(DF11, x, t)
        np.testing.assert_allclose(result, expected, atol=ATOL)


class TestDF12:
    def test_t15_formula(self):
        """DF12: uses abs(sin(...)) with BOTH x1,x2; f3=sin(0.5*pi*x1)."""
        t = 0.3
        k = 10.0 * np.sin(np.pi * t)
        x = np.linspace(0.1, 0.9, 10)
        tmp1 = x[2:] - np.sin(t * x[0])
        tmp2_0 = np.abs(np.sin(np.floor(k * (2.0 * x[0] - 1.0)) * np.pi / 2.0))
        tmp2_1 = np.abs(np.sin(np.floor(k * (2.0 * x[1] - 1.0)) * np.pi / 2.0))
        g = 1.0 + np.sum(tmp1 ** 2) + tmp2_0 * tmp2_1
        expected_f1 = g * np.cos(0.5 * np.pi * x[1]) * np.cos(0.5 * np.pi * x[0])
        expected_f2 = g * np.sin(0.5 * np.pi * x[1]) * np.cos(0.5 * np.pi * x[0])
        expected_f3 = g * np.sin(0.5 * np.pi * x[0])  # x(1) in MATLAB = x[0]
        result = _eval_at(DF12, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2, expected_f3], atol=ATOL)

    def test_t15b_different_from_df10(self):
        """DF12 must produce different values from DF10."""
        t = 0.5
        x = np.full(10, 0.5)
        r10 = _eval_at(DF10, x, t)
        r12 = _eval_at(DF12, x, t)
        assert not np.allclose(r10, r12), "DF12 must differ from DF10"


class TestDF13:
    def test_t16_formula(self):
        """DF13: cos^2/sin^2 with p-dependent knee."""
        t = 0.4
        G = np.sin(0.5 * np.pi * t)
        p = np.floor(6.0 * G)
        x = np.linspace(0.1, 0.9, 10)
        g = 1.0 + np.sum((x[2:] - G) ** 2)
        expected_f1 = g * np.cos(0.5 * np.pi * x[0]) ** 2
        expected_f2 = g * np.cos(0.5 * np.pi * x[1]) ** 2
        expected_f3 = (g * np.sin(0.5 * np.pi * x[0]) ** 2
                       + np.sin(0.5 * np.pi * x[0]) * np.cos(p * np.pi * x[0]) ** 2
                       + np.sin(0.5 * np.pi * x[1]) ** 2
                       + np.sin(0.5 * np.pi * x[1]) * np.cos(p * np.pi * x[1]) ** 2)
        result = _eval_at(DF13, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2, expected_f3], atol=ATOL)

    def test_t16b_different_from_df10(self):
        """DF13 must differ from DF10."""
        t = 0.5
        x = np.full(10, 0.5)
        r10 = _eval_at(DF10, x, t)
        r13 = _eval_at(DF13, x, t)
        assert not np.allclose(r10, r13), "DF13 must differ from DF10"


class TestDF14:
    def test_t17_formula(self):
        """DF14: y=0.5+G*(x1-0.5) with sinusoidal perturbation."""
        t = 0.4
        G = np.sin(0.5 * np.pi * t)
        x = np.linspace(0.1, 0.9, 10)
        g = 1.0 + np.sum((x[2:] - G) ** 2)
        y = 0.5 + G * (x[0] - 0.5)
        expected_f1 = g * (1.0 - y + 0.05 * np.sin(6.0 * np.pi * y))
        expected_f2 = g * (1.0 - x[1] + 0.05 * np.sin(6.0 * np.pi * x[1])) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        expected_f3 = g * (x[1] + 0.05 * np.sin(6.0 * np.pi * x[1])) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        result = _eval_at(DF14, x, t)
        np.testing.assert_allclose(result, [expected_f1, expected_f2, expected_f3], atol=ATOL)

    def test_t17b_different_from_df10(self):
        """DF14 must differ from DF10."""
        t = 0.5
        x = np.full(10, 0.5)
        r10 = _eval_at(DF10, x, t)
        r14 = _eval_at(DF14, x, t)
        assert not np.allclose(r10, r14), "DF14 must differ from DF10"


# =====================================================================
# T18-T21: Structural tests
# =====================================================================

class TestStructural:
    def test_t18_n_obj(self):
        """Verify n_obj for all problems."""
        for cls in [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9]:
            assert cls().n_obj == 2, f"{cls.__name__} should be 2-obj"
        for cls in [DF10, DF11, DF12, DF13, DF14]:
            assert cls().n_obj == 3, f"{cls.__name__} should be 3-obj"

    def test_t19_evaluate_shape(self):
        """Evaluate returns correct shape."""
        for cls in [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9]:
            p = cls()
            X = np.random.rand(5, 10) * (p.xu - p.xl) + p.xl
            F = p.evaluate(X, t=0.5)
            assert F.shape == (5, 2), f"{cls.__name__} shape"
        for cls in [DF10, DF11, DF12, DF13, DF14]:
            p = cls()
            X = np.random.rand(5, 10) * (p.xu - p.xl) + p.xl
            F = p.evaluate(X, t=0.5)
            assert F.shape == (5, 3), f"{cls.__name__} shape"

    def test_t20_pf_shape(self):
        """true_pf returns correct shape."""
        for cls in [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9]:
            p = cls()
            pf = p.true_pf(t=0.5, n_points=100)
            assert pf.shape[1] == 2, f"{cls.__name__} PF columns"
            assert pf.shape[0] > 0, f"{cls.__name__} PF non-empty"
        for cls in [DF10, DF11, DF12, DF13, DF14]:
            p = cls()
            pf = p.true_pf(t=0.5, n_points=100)
            assert pf.shape[1] == 3, f"{cls.__name__} PF columns"
            assert pf.shape[0] > 0, f"{cls.__name__} PF non-empty"

    def test_t21_pf_nonnegative(self):
        """PF values should be non-negative for all problems at t=0.5."""
        for cls in [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9,
                    DF10, DF11, DF12, DF13, DF14]:
            p = cls()
            pf = p.true_pf(t=0.5, n_points=200)
            assert np.all(pf >= -1e-10), f"{cls.__name__} PF has negatives"


# =====================================================================
# T22-T24: Difference tests (E4U defects)
# =====================================================================

class TestE4UDefects:
    def test_t22_df4_not_df1(self):
        """DF4 formula is completely different from DF1."""
        for t in [0.0, 0.3, 0.7, 1.0]:
            x = np.full(10, 0.5)
            r1 = _eval_at(DF1, x, t)
            r4 = _eval_at(DF4, x, t)
            assert not np.allclose(r1, r4, atol=1e-6), f"DF4=DF1 at t={t}"

    def test_t23_df12_not_df10(self):
        """DF12 formula differs from DF10."""
        for t in [0.3, 0.5, 0.7]:
            x = np.linspace(0.1, 0.9, 10)
            r10 = _eval_at(DF10, x, t)
            r12 = _eval_at(DF12, x, t)
            assert not np.allclose(r10, r12, atol=1e-6), f"DF12=DF10 at t={t}"

    def test_t24_df13_not_df10(self):
        """DF13 differs from DF10."""
        for t in [0.3, 0.5, 0.7]:
            x = np.linspace(0.1, 0.9, 10)
            r10 = _eval_at(DF10, x, t)
            r13 = _eval_at(DF13, x, t)
            assert not np.allclose(r10, r13, atol=1e-6), f"DF13=DF10 at t={t}"

    def test_t24b_df14_not_df10(self):
        """DF14 differs from DF10."""
        for t in [0.3, 0.5, 0.7]:
            x = np.linspace(0.1, 0.9, 10)
            r10 = _eval_at(DF10, x, t)
            r14 = _eval_at(DF14, x, t)
            assert not np.allclose(r10, r14, atol=1e-6), f"DF14=DF10 at t={t}"


# =====================================================================
# T25: No NaN test
# =====================================================================

class TestNoNaN:
    def test_t25_no_nan(self):
        """No NaN in any problem evaluation at multiple time steps."""
        for cls in [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9,
                    DF10, DF11, DF12, DF13, DF14]:
            p = cls()
            for t in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]:
                X = np.random.rand(20, 10) * (p.xu - p.xl) + p.xl
                F = p.evaluate(X, t=t)
                assert not np.any(np.isnan(F)), f"{cls.__name__} NaN at t={t}"
                pf = p.true_pf(t=t, n_points=100)
                assert not np.any(np.isnan(pf)), f"{cls.__name__} PF NaN at t={t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
