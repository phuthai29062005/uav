"""BL1-BL15: clean Dynamic NSGA-II baseline isolated from SA-DRL (4.9)."""
import os
import sys

import numpy as np
import pytest
from pymoo.core.problem import Problem
from pymoo.problems.dynamic.df import DF1, DF4

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import baseline_runner  # noqa: E402
import dynamic_runner  # noqa: E402
from baseline_runner import run_nsga2_baseline  # noqa: E402


def run(problem=DF1, changes=3, warm_up=5, tau_t=3, N=20, seed=0):
    return run_nsga2_baseline(problem, n_t=10, tau_t=tau_t, N=N, D=10,
                              n_changes=changes, warm_up=warm_up, seed=seed,
                              ref_point=(2.0, 2.0))


class RecordingProblem(Problem):
    def __init__(self, inner):
        super().__init__(n_var=inner.n_var, n_obj=inner.n_obj,
                         xl=inner.xl, xu=inner.xu)
        self.inner = inner
        self.total = 0

    def _evaluate(self, X, out, *a, **k):
        X = np.atleast_2d(X)
        self.total += len(X)
        self.inner.time = self.time
        out["F"] = self.inner.evaluate(X)

    def _calc_pareto_front(self, *a, **k):
        self.inner.time = self.time
        return self.inner.pareto_front()

    @property
    def time(self):
        return getattr(self, "_time", 0.0)

    @time.setter
    def time(self, v):
        self._time = v


# ------------------------------------ BL1/2/3: BYPASS SA COMPONENTS
def test_bl1_bypass_detector(monkeypatch):
    monkeypatch.setattr(dynamic_runner, "ChangeDetector",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("detector goi trong baseline")))
    run()


def test_bl2_bypass_memory(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("memory goi trong baseline")
    monkeypatch.setattr(dynamic_runner, "MemoryArchive", boom)
    monkeypatch.setattr(dynamic_runner, "compute_signature", boom)
    run()


def test_bl3_bypass_dqn_state(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("DQN/state goi trong baseline")
    monkeypatch.setattr(dynamic_runner, "build_state", boom)
    monkeypatch.setattr(dynamic_runner, "apply_hierarchical_response", boom)
    run()


# ------------------------------------------------------ BL4: EXACT TIMELINE
def test_bl4_timeline():
    tl = run(changes=3)["timeline"]
    tr = [(e["t_old"], e["t_new"]) for e in tl]
    assert len(tr) == 3
    for got, exp in zip(tr, [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]):
        assert np.isclose(got[0], exp[0]) and np.isclose(got[1], exp[1])


# ---------------------- BL5: RESPONSE == PRE, N REEVAL (not 2N)
def test_bl5_response_equals_pre():
    res = run(changes=3)
    assert res["igd_response_history"] == res["igd_pre_history"]
    assert res["hv_response_history"] == res["hv_pre_history"]
    # reeval FE = N/change (khong 2N)
    assert res["fe_breakdown"]["environment_reeval"] == 3 * 20


# ------------------------------------------ BL6: FE LEDGER GROUND TRUTH
def test_bl6_fe_ground_truth():
    total = {"n": 0}
    orig = RecordingProblem._evaluate

    def counting(self, X, out, *a, **k):
        total["n"] += len(np.atleast_2d(X))
        orig(self, X, out, *a, **k)
    RecordingProblem._evaluate = counting
    try:
        res = run_nsga2_baseline(
            lambda time=0.0, n_var=10: RecordingProblem(DF1(time=time, n_var=n_var)),
            n_t=10, tau_t=3, N=20, D=10, n_changes=3, warm_up=4,
            seed=0, ref_point=(2.0, 2.0))
    finally:
        RecordingProblem._evaluate = orig
    assert res["fes_used"] == total["n"]
    assert res["fes_used"] == sum(res["fe_breakdown"].values())


# ------------------------------------------------ BL7: NO SA OVERHEAD KEYS
def test_bl7_no_sa_keys():
    bd = run()["fe_breakdown"]
    assert set(bd) == {"initial", "nsga2", "environment_reeval"}
    for k in ("detector", "signature", "response", "memory"):
        assert k not in bd


# ---------------------------------------------- BL8: SAME NSGA-II BACKBONE
def test_bl8_shared_nsga(monkeypatch):
    calls = {"n": 0}
    orig = baseline_runner.nsga2_one_generation
    monkeypatch.setattr(baseline_runner, "nsga2_one_generation",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1)
                                         or orig(*a, **k)))
    run(changes=3, warm_up=5, tau_t=3)
    # total_gens = 5 + (3+1)*3 = 17 nsga generations
    assert calls["n"] == 17


# ------------------------------------------------------ BL9: BOUNDS CORRECT
def test_bl9_bounds():
    # DF4 [-2,2]: moi diem evaluate phai trong [xl,xu]; co diem < 0 (thoat
    # [0,1]) -> bao ve 4.5B. Metric finite (khong assert nguong IGD vi run
    # ngan tren [-2,2] chua hoi tu).
    seen = []
    orig = RecordingProblem._evaluate

    def cap(self, X, out, *a, **k):
        X = np.atleast_2d(X); seen.append(X.copy()); orig(self, X, out, *a, **k)
    RecordingProblem._evaluate = cap
    try:
        res = run_nsga2_baseline(
            lambda time=0.0, n_var=10: RecordingProblem(DF4(time=time, n_var=n_var)),
            n_t=10, tau_t=2, N=20, D=10, n_changes=2, warm_up=3,
            seed=0, ref_point=(5.0, 5.0))
    finally:
        RecordingProblem._evaluate = orig
    allX = np.vstack(seen)
    assert np.all(allX >= -2 - 1e-9) and np.all(allX <= 2 + 1e-9)
    assert allX.min() < 0
    assert np.all(np.isfinite(res["igd_end_history"]))


# ---------------------------------------------- BL10: MIGD END SEMANTICS
def test_bl10_migd_semantics():
    res = run(changes=3)
    assert res["migd"] == res["migd_end"]
    assert np.isclose(res["migd_response"], np.mean(res["igd_response_history"]))
    assert np.isclose(res["migd_end"], np.mean(res["igd_end_history"]))


# ------------------------------------------------------- BL11: K METRIC ROWS
def test_bl11_k_rows():
    res = run(changes=4)
    assert len(res["igd_pre_history"]) == 4
    assert len(res["igd_response_history"]) == 4
    assert len(res["igd_end_history"]) == 4


# ------------------------------------------------------ BL12: NO REPLAY/LEARN
def test_bl12_no_rl():
    res = run(changes=3)
    assert res["reward_history"] == []
    # baseline runner khong import DQNAgent
    import inspect
    src = inspect.getsource(baseline_runner)
    assert "DQNAgent(" not in src
    assert "replay_buffer" not in src


# ------------------------------------------------------ BL13: REPRODUCIBLE
def test_bl13_reproducible():
    a, b = run(changes=3, seed=5), run(changes=3, seed=5)
    assert np.allclose(a["igd_end_history"], b["igd_end_history"])
    assert a["fe_breakdown"] == b["fe_breakdown"]
    assert a["fes_used"] == b["fes_used"]


# --------------------------- BL14: BASELINE FE < CONTROLLER (structural)
def test_bl14_baseline_fe_lower():
    from dqn_agent import DQNAgent
    import torch
    torch.manual_seed(0)
    ag = DQNAgent(state_dim=2 + 4, n_segments=2, eps_fixed=0.3)
    sa = dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=3, N=20, D=10, n_changes=3, warm_up=5,
        agent=ag, segments=[[0], list(range(1, 10))], seed=0,
        training=True, ref_point=(2.0, 2.0), n_elite=5)
    bl = run(changes=3, warm_up=5, tau_t=3, N=20)
    assert "detector" not in bl["fe_breakdown"]
    assert "signature" not in bl["fe_breakdown"]
    assert bl["fes_used"] < sa["fes_used"]


# ------------------------------------------------------ BL15: DF1 SMOKE
def test_bl15_smoke():
    res = run(changes=3)
    assert np.all(np.isfinite(res["igd_end_history"]))
    assert np.all(np.isfinite(res["hv_end_history"]))
    assert np.isfinite(res["migd_end"]) and np.isfinite(res["hv_final"])
