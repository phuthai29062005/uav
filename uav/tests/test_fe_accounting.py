"""F1-F12: exact candidate-level function evaluation accounting."""
import os
import sys

import numpy as np
import pytest
from pymoo.core.problem import Problem
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from change_detector import ChangeDetector  # noqa: E402
from memory_archive import compute_signature  # noqa: E402
from nsga2_pymoo import nsga2_one_generation, seed_nsga2  # noqa: E402

D = 10
SEGS = [[0], list(range(1, D))]


class RecordingProblem(Problem):
    """Dem SO ROW candidate moi lan objective duoc evaluate (ground truth)."""

    def __init__(self, inner):
        super().__init__(n_var=inner.n_var, n_obj=inner.n_obj,
                         xl=inner.xl, xu=inner.xu)
        self.inner = inner
        self.total = 0

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        self.total += len(X)
        self.inner.time = self.time
        out["F"] = self.inner.evaluate(X)

    def _calc_pareto_front(self, *args, **kwargs):
        # PF la metric protocol, KHONG goi _evaluate -> khong tinh FE.
        self.inner.time = self.time
        return self.inner.pareto_front()

    @property
    def time(self):
        return getattr(self, "_time", 0.0)

    @time.setter
    def time(self, v):
        self._time = v


# ------------------------------------ F1: BATCH EVALUATION COUNTS ROWS
def test_f1_batch_counts_rows():
    rp = RecordingProblem(DF1(time=0.0, n_var=D))
    rp.evaluate(np.random.rand(7, D))
    assert rp.total == 7


# ---------------------------------------------- F4: DETECTOR CENTRAL EXACT
def test_f4_detector_central_exact():
    rp = RecordingProblem(DF1(time=0.5, n_var=D))
    X = np.full((3, D), 0.5)                    # xa bien -> central het
    d = ChangeDetector(segments=SEGS, obj_scale=np.array([2., 2.]),
                       lb=np.zeros(D), ub=np.ones(D), verbose=False)
    rp.total = 0
    fe = d.prime(X, rp)
    assert d.last_prime_stats["central"] == 6   # 3 probe * 2 seg
    assert fe == rp.total                        # central: 2/probe-seg = 12
    assert fe == 12


# ------------------------------ F5: DETECTOR MIXED STENCIL + BASE CACHE
def test_f5_detector_mixed_base_cached():
    rp = RecordingProblem(DF1(time=0.5, n_var=D))
    X = np.full((4, D), 0.5)
    X[0, 0] = 0.0                                # sat bien -> onesided
    d = ChangeDetector(segments=SEGS, obj_scale=np.array([2., 2.]),
                       lb=np.zeros(D), ub=np.ones(D), verbose=False)
    st = None
    rp.total = 0
    fe = d.prime(X, rp)
    stats = d.last_prime_stats
    assert stats["onesided2"] + stats["onesided1"] >= 1
    assert stats["central"] >= 1
    assert fe == rp.total                        # ground truth khop
    # base F(x) cua probe 0 chi eval 1 lan du 2 segment cung one-sided:
    # neu khong cache, so eval se cao hon. Kiem qua cong thuc:
    #   central pair -> 2 ; onesided2 -> 2 (+base share) ; onesided1 -> 1 (+base)
    n_base = sum(1 for i in range(4)
                 if any(s.mode != "central" for s in d._stencils[i]))
    per = 0
    for i in range(4):
        for s in d._stencils[i]:
            per += {"central": 2, "onesided2": 2, "onesided1": 1}[s.mode]
    assert fe == per + n_base


# ------------------------------------------- F6: SIGNATURE COUNTS N_PROBE
def test_f6_signature_counts_n_probe():
    rp = RecordingProblem(DF1(time=0.3, n_var=D))
    X = np.random.rand(5, D)
    rp.total = 0
    compute_signature(X, rp, np.array([2., 2.]))
    assert rp.total == 5


# -------------------------------------------- F3: ONE NSGA-II GENERATION
def test_f3_one_nsga_generation():
    rp = RecordingProblem(DF1(time=0.5, n_var=D))
    pop = np.random.rand(20, D)
    F = rp.evaluate(pop)
    rp.total = 0
    seed_nsga2(0)
    _, _, fe = nsga2_one_generation(pop, F, rp, 20, return_fe=True)
    assert fe == rp.total == 20


def _short_run(problem_cls, changes=2, warm_up=5, tau_t=3, N=20, seed=0):
    return dynamic_runner.run_sa_drl(
        problem_cls, n_t=10, tau_t=tau_t, N=N, D=D, n_changes=changes,
        warm_up=warm_up, agent=NSGA2Baseline(n_segments=2), segments=SEGS,
        seed=seed, training=False, ref_point=(2.0, 2.0), n_elite=5)


# -------------------------------------------------- F2: INITIAL POPULATION
def test_f2_initial_population():
    res = _short_run(DF1, changes=1, warm_up=1, tau_t=1, N=20)
    assert res["fe_breakdown"]["initial"] == 20


# ------------------------------------------------------- F9: LEDGER SUM
def test_f9_ledger_sum():
    res = _short_run(DF1, changes=2)
    bd = res["fe_breakdown"]
    assert res["fes_used"] == sum(bd.values())
    for v in bd.values():
        assert isinstance(v, int) and v >= 0


# ---------------------------------- F10: RECORDING PROBLEM GROUND TRUTH
def test_f10_ground_truth_matches_ledger():
    class Wrap:
        instances = []

        def __init__(self, time=0.0, n_var=D):
            self.rp = RecordingProblem(DF1(time=time, n_var=n_var))
            self.rp.time = time
            Wrap.instances.append(self)

        def __getattr__(self, k):
            return getattr(self.rp, k)

    total = {"n": 0}
    orig_eval = RecordingProblem._evaluate

    def counting(self, X, out, *a, **k):
        # tach metric: pareto_front khong goi _evaluate; chi optimization
        total["n"] += len(np.atleast_2d(X))
        orig_eval(self, X, out, *a, **k)
    RecordingProblem._evaluate = counting
    try:
        res = _short_run(Wrap, changes=2, warm_up=4, tau_t=3, N=20)
    finally:
        RecordingProblem._evaluate = orig_eval
    assert res["fes_used"] == total["n"], (res["fes_used"], total["n"])


# ------------------------- F11: TIMELINE CHANGE DOES NOT DOUBLE COUNT
def test_f11_no_double_count(monkeypatch):
    calls = {"sig": 0, "comp": 0}
    os_ = compute_signature
    oc = ChangeDetector.compute
    monkeypatch.setattr(dynamic_runner, "compute_signature",
                        lambda *a, **k: (calls.__setitem__("sig", calls["sig"] + 1)
                                         or os_(*a, **k)))
    monkeypatch.setattr(ChangeDetector, "compute",
                        lambda self, *a, **k: (calls.__setitem__("comp", calls["comp"] + 1)
                                               or oc(self, *a, **k)))
    _short_run(DF1, changes=3)
    assert calls["comp"] == 3            # + 0 fake t=0->0
    assert calls["sig"] == 3             # calibration dung calibrate_scale rieng


# ----------------------------------------------- F12: SAME SEED, SAME FE
def test_f12_same_seed_same_fe():
    a = _short_run(DF1, changes=2, seed=5)
    b = _short_run(DF1, changes=2, seed=5)
    assert a["fes_used"] == b["fes_used"]
    assert a["fe_breakdown"] == b["fe_breakdown"]


# --------- F7/F8: RESPONSE FE (KEEP vs CHANGED) — live runner luon eval N
def test_f7_f8_response_eval_is_actual():
    """
    Runner luon `problem.evaluate(pop)` sau response (audit dong 148),
    bat ke gate/seg -> response FE = 2N moi change (F_after_change + post).
    changes=2, N=20 -> response = 2*2*20 = 80.
    """
    res = _short_run(DF1, changes=2, N=20)
    assert res["fe_breakdown"]["response"] == 2 * 2 * 20
