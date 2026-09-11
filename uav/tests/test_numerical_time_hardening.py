"""H1-H17: periodic time-boundary stabilization + ref_point fail-loud (4.10C.1)."""
import os
import sys

import numpy as np
import pytest
from pymoo.indicators.igd import IGD
from pymoo.problems.dynamic.df import DF1, DF7, DF10

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from baseline_runner import run_nsga2_baseline  # noqa: E402
from df_stable import (DF5Stable, DF12Stable, DF13Stable,  # noqa: E402
                       snap_near_zero)

D = 10
TIMES = [0.0, 4.0, 8.0]


def _X(p, seed=0):
    return p.xl + np.random.RandomState(seed).rand(24, D) * (p.xu - p.xl)


# ---------- H1-H3: internal discrete state identical at t=0,4,8
def test_h1_df5_w_periodic():
    w = [np.floor(10 * snap_near_zero(np.sin(0.5 * np.pi * t))) for t in TIMES]
    assert w[0] == w[1] == w[2] == 0.0


def test_h2_df12_k_periodic():
    k = [np.floor(10 * snap_near_zero(np.sin(np.pi * t))
                  * (2 * 0.3 - 1)) for t in TIMES]
    assert k[0] == k[1] == k[2]


def test_h3_df13_p_periodic():
    p = [np.floor(6 * snap_near_zero(np.sin(0.5 * np.pi * t))) for t in TIMES]
    assert p[0] == p[1] == p[2] == 0.0


# ---------- H4-H6: PF periodic after snap (tight tol)
@pytest.mark.parametrize("C", [DF5Stable, DF12Stable, DF13Stable])
def test_h4_h6_pf_periodic(C):
    pf0 = C(time=0.0, n_var=D).pareto_front()
    pf4 = C(time=4.0, n_var=D).pareto_front()
    assert max(IGD(pf0)(pf4), IGD(pf4)(pf0)) < 1e-10


# ---------- H7,H9: DF5/DF13 objective periodic; H8: DF12 PF-only
def test_h7_df5_objective_periodic():
    p0, p4 = DF5Stable(time=0.0, n_var=D), DF5Stable(time=4.0, n_var=D)
    X = _X(p0)
    assert np.allclose(p0.evaluate(X), p4.evaluate(X))


def test_h9_df13_objective_periodic():
    p0, p4 = DF13Stable(time=0.0, n_var=D), DF13Stable(time=4.0, n_var=D)
    X = _X(p0)
    assert np.allclose(p0.evaluate(X), p4.evaluate(X))


def test_h8_df12_pf_periodic_objective_not():
    # DF12 g co term sin(time*x0) PHI CHU KY (dinh nghia CEC2018) -> objective
    # KHONG periodic; chi PF (phu thuoc k) duoc stabilize. Snap sua discrete
    # flip cua k, khong bien DF12 thanh periodic.
    p0, p4 = DF12Stable(time=0.0, n_var=D), DF12Stable(time=4.0, n_var=D)
    X = _X(p0)
    assert not np.allclose(p0.evaluate(X), p4.evaluate(X))       # objective
    pf0, pf4 = p0.pareto_front(), p4.pareto_front()
    assert max(IGD(pf0)(pf4), IGD(pf4)(pf0)) < 1e-10             # PF


# ---------- H10: near-boundary NOT flattened
def test_h10_near_boundary_preserved():
    def p13(t):
        return np.floor(6 * snap_near_zero(np.sin(0.5 * np.pi * t)))
    assert p13(4 - 1e-6) == -1.0      # genuine negative giu nguyen
    assert p13(4.0) == 0.0            # exact boundary -> math value
    assert p13(4 + 1e-6) == 0.0
    # atol nho: 1e-7 residue KHONG bi snap
    assert snap_near_zero(1e-7) == 1e-7
    assert snap_near_zero(-2.4e-16) == 0.0


# ---------- H11: DF7 remains non-periodic (co (1+t))
def test_h11_df7_nonperiodic():
    pf0 = DF7(time=0.0, n_var=D).pareto_front()
    pf4 = DF7(time=4.0, n_var=D).pareto_front()
    assert max(IGD(pf0)(pf4), IGD(pf4)(pf0)) > 0.1


# ---------- H12-H15: ref_point fail-loud
def _sa(prob, rp):
    from dynamic_runner import run_sa_drl
    run_sa_drl(prob, 10, 3, 20, D, 2, 5, agent=None, segments=[[0]],
               seed=0, ref_point=rp)


def test_h12_sa_none_raises():
    with pytest.raises(ValueError, match="ref_point"):
        _sa(DF1, None)


def test_h13_sa_wrong_dim_raises():
    with pytest.raises(ValueError, match="n_obj"):
        _sa(DF10, (2.0, 2.0))


def test_h14_baseline_none_raises():
    with pytest.raises(ValueError, match="ref_point"):
        run_nsga2_baseline(DF1, 10, 3, 20, D, 2, 5, seed=0, ref_point=None)


def test_h15_baseline_wrong_dim_raises():
    with pytest.raises(ValueError, match="n_obj"):
        run_nsga2_baseline(DF10, 10, 3, 20, D, 2, 5, seed=0,
                           ref_point=(2.0, 2.0))


# ---------- H16,H17: valid ref runs (bi + tri)
def test_h16_valid_biobjective():
    res = run_nsga2_baseline(DF5Stable, 10, 3, 20, D, 2, 5, seed=0,
                             ref_point=(2.0, 2.0))
    assert np.isfinite(res["migd_end"])


def test_h17_valid_triobjective():
    res = run_nsga2_baseline(DF13Stable, 10, 3, 20, D, 2, 5, seed=0,
                             ref_point=(2.0, 2.0, 6.0))
    assert np.isfinite(res["migd_end"])
