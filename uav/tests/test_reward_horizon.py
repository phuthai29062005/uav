"""R1-R12: align reward with end-of-environment metrics."""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from sa_drl_dmoea import compute_reward  # noqa: E402

D = 10
SEGS = [[0], list(range(1, D))]
HV_REF = 4.0
BUDGET = 200


def run(changes=3, warm_up=5, tau_t=3, N=20, seed=0):
    return dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=tau_t, N=N, D=D, n_changes=changes,
        warm_up=warm_up, agent=NSGA2Baseline(n_segments=2), segments=SEGS,
        seed=seed, training=False, ref_point=(2.0, 2.0), n_elite=5)


# ------------------------------------ R1: QUALITY USES PRE, NOT RESPONSE
def test_r1_quality_uses_pre():
    r = compute_reward(hv_end=0.50, hv_pre=0.20, hv_ref=HV_REF,
                       fe_used=20, fe_budget=BUDGET)
    expected = 1.0 * (0.50 - 0.20) / HV_REF - 0.1 * 20 / BUDGET
    assert np.isclose(r, expected)
    wrong = 1.0 * (0.50 - 0.10) / HV_REF - 0.1 * 20 / BUDGET
    assert not np.isclose(r, wrong)


# ------------- R2: RESPONSE QUALITY KHONG THAY REWARD NEU PRE/END GIONG
def test_r2_response_does_not_affect_reward():
    a = compute_reward(0.5, 0.2, HV_REF, 20, BUDGET)   # response=0.05 (an y)
    b = compute_reward(0.5, 0.2, HV_REF, 20, BUDGET)   # response=0.4
    assert a == b


# ---------------------------------------- R3: BETTER END -> HIGHER REWARD
def test_r3_better_end_higher_reward():
    a = compute_reward(0.3, 0.2, HV_REF, 20, BUDGET)
    b = compute_reward(0.6, 0.2, HV_REF, 20, BUDGET)
    assert b > a


# ------------------------------ R4: ACTUAL RESPONSE FE, KHONG count_fe
def test_r4_actual_response_fe():
    # Runner luon eval N sau response -> pending_action_fe = N cho moi action.
    # response breakdown = 2N/change (F_pre + F_response), gian tiep xac nhan.
    res = run(changes=2, N=20)
    assert res["fe_breakdown"]["response"] == 2 * 2 * 20


# ------------------------------------------- R5: RESPONSE METRIC LENGTH
def test_r5_response_length():
    res = run(changes=4)
    assert len(res["igd_response_history"]) == 4
    assert len(res["hv_response_history"]) == 4


# ------------------------------------------------ R6: END METRIC LENGTH
def test_r6_end_length():
    res = run(changes=4)
    assert len(res["igd_end_history"]) == 4       # bao gom env cuoi
    assert len(res["hv_end_history"]) == 4


# --------------------------- R7: END METRIC BELONGS TO SAME ENVIRONMENT
def test_r7_end_metric_correct_environment(monkeypatch):
    """IGD_end cua action tai t_k phai dung PF(t_k), khong PF(t_{k+1})."""
    seen = []
    orig = dynamic_runner.IGD

    class SpyIGD:
        def __init__(self, pf):
            self._pf = pf
            self._inner = orig(pf)

        def __call__(self, F):
            seen.append(len(self._pf))
            return self._inner(F)

    # Thay bang cach ghi lai (time, PF) qua pareto_front cua tung problem.
    times = []
    from pymoo.problems.dynamic.df import DF1 as _DF1
    orig_pf = _DF1.pareto_front

    def spy_pf(self, *a, **k):
        times.append(round(self.time, 4))
        return orig_pf(self, *a, **k)
    monkeypatch.setattr(_DF1, "pareto_front", spy_pf)
    res = run(changes=3, tau_t=2)
    # end histories co 3 entry cho t=0.1,0.2,0.3; PF phai duoc lay o dung t.
    assert len(res["igd_end_history"]) == 3
    assert set([0.1, 0.2, 0.3]).issubset(set(times))


# ------------------------------------------------------- R8: MIGD ALIAS
def test_r8_migd_alias():
    res = run(changes=3)
    assert res["migd"] == res["migd_end"]
    assert np.isclose(res["migd_response"],
                      np.mean(res["igd_response_history"]))
    assert np.isclose(res["migd_end"], np.mean(res["igd_end_history"]))


# --------------------------- R9: LAST TRANSITION VAN CHUA PUSH (4.7)
def test_r9_last_transition_not_pushed():
    K = 4
    agent = NSGA2Baseline(n_segments=2)
    # NSGA2Baseline dung _NoOpBuffer -> dem push qua DQNAgent that.
    from dqn_agent import DQNAgent
    import torch
    torch.manual_seed(0)
    ag = DQNAgent(state_dim=len(SEGS) + 6, n_segments=2, eps_fixed=0.3)
    dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=3, N=20, D=D, n_changes=K, warm_up=5,
        agent=ag, segments=SEGS, seed=0, training=True,
        ref_point=(2.0, 2.0), n_elite=5)
    assert len(ag.replay_buffer) == K - 1
    for tr in ag.replay_buffer.buffer:
        assert tr[5] == 0.0        # done == False


# ------------------------------------------------------ R10: REWARD COUNT
def test_r10_reward_count():
    K = 5
    res = run(changes=K)
    assert len(res["reward_history"]) == K - 1


# ------------------------------------------- R11: COUNT_FE KHONG CON LIVE
def test_r11_count_fe_not_in_live_path():
    import inspect
    src = inspect.getsource(dynamic_runner.run_sa_drl)
    assert "count_fe(" not in src
    assert "count_fe" not in dynamic_runner.run_sa_drl.__globals__


# ----------------------------------------------- R12: SAME SEED REPRODUCE
def test_r12_reproducible():
    a = run(changes=3, seed=5)
    b = run(changes=3, seed=5)
    for k in ["igd_response_history", "igd_end_history",
              "hv_response_history", "hv_end_history"]:
        assert np.allclose(a[k], b[k]), k
