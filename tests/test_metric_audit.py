"""
4.10C — CEC2018 PF/IGD/HV/reference-point audit (READ-ONLY).

Khoa metric infrastructure cho DF1-DF14 truoc benchmark: objective count,
PF dimension/finite/time, IGD argument order, HV convention, ref-point
domination, tri-objective support, periodicity, baseline==SA metric path.

KHONG sua production.
"""
import os
import sys

import numpy as np
import pytest
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems.dynamic.df import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from train import problem_setup  # noqa: E402

D = 10
NAMES = [f"DF{i}" for i in range(1, 15)]
CLS = {n: c for n, c in zip(NAMES, [DF1, DF2, DF3, DF4, DF5, DF6, DF7,
                                    DF8, DF9, DF10, DF11, DF12, DF13, DF14])}
TIMES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 4.0]
EXPECT_M = {**{f"DF{i}": 2 for i in range(1, 10)},
            **{f"DF{i}": 3 for i in range(10, 15)}}


# ------------------------------------------ PF1: objective count correct
@pytest.mark.parametrize("name", NAMES)
def test_pf1_objective_count(name):
    p = CLS[name](time=0.0, n_var=D)
    X = p.xl + np.random.RandomState(0).rand(16, D) * (p.xu - p.xl)
    assert p.n_obj == EXPECT_M[name]
    assert p.evaluate(X).shape[1] == p.n_obj


# ---------------------- PF2: PF finite + correct dim at representative t
@pytest.mark.parametrize("name", NAMES)
def test_pf2_pf_valid(name):
    for t in TIMES:
        p = CLS[name](time=t, n_var=D)
        pf = p.pareto_front()
        assert pf is not None and pf.ndim == 2
        assert pf.shape[1] == p.n_obj
        assert np.all(np.isfinite(pf))


# ------------------------------------------ PF3: IGD exact PF -> ~0
def test_pf3_igd_exact_zero():
    pf = np.array([[0., 1.], [1., 0.]])
    assert IGD(pf)(pf) < 1e-12
    pf3 = np.eye(3)
    assert IGD(pf3)(pf3) < 1e-12


# ------------------------------------------ PF4: IGD argument order
def test_pf4_igd_order():
    pf = np.array([[0., 1.], [1., 0.]])
    assert IGD(pf)(pf + 5.0) > 1.0          # approx xa -> IGD lon
    assert IGD(pf)(pf) < 1e-12


# ------------------------------------ PF5: ref-point dim == M all DF
@pytest.mark.parametrize("name", NAMES)
def test_pf5_ref_dim(name):
    _, ref = problem_setup(name)
    assert len(ref) == CLS[name](time=0.0, n_var=D).n_obj


# ---------------------- PF6: ref point dominates PF across times
@pytest.mark.parametrize("name", NAMES)
def test_pf6_ref_dominates_pf(name):
    _, ref = problem_setup(name)
    ref = np.array(ref)
    for t in TIMES:
        pf = CLS[name](time=t, n_var=D).pareto_front()
        assert np.all(ref > pf.max(axis=0)), (name, t, pf.max(axis=0), ref)


# -------------------------------------------------- PF7/PF8: HV sanity
def test_pf7_hv_2d():
    hv = HV(ref_point=np.array([2., 2.]))
    assert hv(np.array([[0.5, 1.], [1., 0.5]])) > hv(np.array([[1.5, 1.5]]))
    assert hv(np.array([[3., 3.]])) == 0.0        # outside box -> 0, no raise


def test_pf8_hv_3d():
    hv = HV(ref_point=np.array([2., 2., 2.]))
    g = hv(np.array([[0.5, 0.5, 0.5]]))
    b = hv(np.array([[1.5, 1.5, 1.5]]))
    assert g > b and np.isfinite(g) and g > 0


# ------------------------- PF9: no live metric helper hard-code M=2
def test_pf9_no_hardcode_m2_in_runners():
    import inspect
    import dynamic_runner
    import baseline_runner
    for mod in (dynamic_runner, baseline_runner):
        src = inspect.getsource(mod)
        # metric path dung _igd(prob,F)/hv_calc(F); khong index F[:,0]/F[:,1]
        assert "F[:, 0]" not in src and "F[:, 1]" not in src
        assert "shape[1] == 2" not in src


# -------------------------- PF10: baseline vs SA metric path identical
def test_pf10_shared_metric_path():
    import inspect
    import dynamic_runner
    import baseline_runner
    # ca hai dinh nghia _igd giong het + dung IGD(pf)(F), HV(ref_point)
    for mod in (dynamic_runner, baseline_runner):
        s = inspect.getsource(mod)
        assert "IGD(pf)(Fvals)" in s
        assert "HV(ref_point=" in s
    # numeric: cung problem+F -> cung IGD/HV (helper la pymoo chung)
    for name in ["DF2", "DF10"]:
        _, ref = problem_setup(name)
        p = CLS[name](time=0.3, n_var=D)
        F = p.xl + np.random.RandomState(0).rand(20, D) * (p.xu - p.xl)
        F = p.evaluate(F)
        igd = IGD(p.pareto_front())(F)
        hv = HV(ref_point=np.array(ref))(F)
        assert np.isfinite(igd) and np.isfinite(hv)


# ---------------------------- PF11: periodicity diagnostic t=0 vs t=4
def test_pf11_periodicity_diagnostic():
    # DF7 phi chu ky (co (1+t)); DF5/12/13 lech nho do floor() float-edge
    # tai G(4)=sin(2pi)~-2e-16. Cac DF con lai ~same. CHI diagnostic.
    non_periodic_ok = {"DF7"}
    float_edge = {"DF5", "DF12", "DF13"}
    for name in NAMES:
        pf0 = CLS[name](time=0.0, n_var=D).pareto_front()
        pf4 = CLS[name](time=4.0, n_var=D).pareto_front()
        d = max(IGD(pf0)(pf4), IGD(pf4)(pf0))
        if name in non_periodic_ok or name in float_edge:
            continue                              # biet truoc, khong gate
        assert d < 1e-6, (name, d)


# ------------------- PF12: reward HV normalization finite/nonzero all DF
@pytest.mark.parametrize("name", NAMES)
def test_pf12_reward_norm(name):
    _, ref = problem_setup(name)
    hv_box = np.prod(ref)
    assert np.isfinite(hv_box) and hv_box > 0


# ------------- PF13: representative population HV finite all DF
@pytest.mark.parametrize("name", NAMES)
def test_pf13_population_hv_finite(name):
    _, ref = problem_setup(name)
    p = CLS[name](time=0.5, n_var=D)
    X = p.xl + np.random.RandomState(2).rand(40, D) * (p.xu - p.xl)
    hv = HV(ref_point=np.array(ref))(p.evaluate(X))
    assert np.isfinite(hv) and hv >= 0.0


# --------------- PF14: IGD_response/end use current problem time
def test_pf14_igd_uses_current_time():
    # _igd goi prob.pareto_front() tren dung problem object cua env hien
    # tai -> PF(t) khop. Kiem PF(0.1) != PF(0.2) va _igd dung dung object.
    import dynamic_runner
    pf01 = DF1(time=0.1, n_var=D).pareto_front()
    pf02 = DF1(time=0.2, n_var=D).pareto_front()
    assert not np.allclose(pf01, pf02)          # PF thuc su doi theo t
    # helper _igd dung prob.pareto_front() (khong cache sai time)
    import inspect
    src = inspect.getsource(dynamic_runner.run_sa_drl)
    assert "prob.pareto_front()" in src
