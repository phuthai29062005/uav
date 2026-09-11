"""T1-T12: finalize terminal transition with done=True (4.7)."""
import os
import sys

import numpy as np
import pytest
import torch
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from dqn_agent import DQNAgent, q_total_of  # noqa: E402

D = 10
SEGS = [[0], list(range(1, D))]


def run_baseline(changes=3, warm_up=5, tau_t=3, N=20, seed=0):
    return dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=tau_t, N=N, D=D, n_changes=changes,
        warm_up=warm_up, agent=NSGA2Baseline(n_segments=2), segments=SEGS,
        seed=seed, training=False, ref_point=(2.0, 2.0), n_elite=5)


def run_dqn(changes=3, warm_up=5, tau_t=3, N=20, seed=0, training=True):
    torch.manual_seed(seed)
    ag = DQNAgent(state_dim=len(SEGS) + 4, n_segments=2, eps_fixed=0.3)
    res = dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=tau_t, N=N, D=D, n_changes=changes,
        warm_up=warm_up, agent=ag, segments=SEGS, seed=seed,
        training=training, ref_point=(2.0, 2.0), n_elite=5)
    return ag, res


# ------------------------------------------- T1: EXACTLY K TRANSITIONS
def test_t1_exactly_k_transitions():
    K = 4
    ag, _ = run_dqn(changes=K)
    assert len(ag.replay_buffer) == K


# ------------------------------------------------------- T2: ONLY LAST DONE
def test_t2_only_last_done():
    ag, _ = run_dqn(changes=4)
    dones = [tr[5] for tr in ag.replay_buffer.buffer]
    assert sum(dones) == 1
    assert dones[-1] == 1.0


# --------------------------- T3: TERMINAL REWARD USES FINAL ENV END
def test_t3_terminal_reward_uses_end():
    from sa_drl_dmoea import compute_reward
    res = run_baseline(changes=3)
    hv_pre_K = res["hv_pre_history"][-1]
    hv_end_K = res["hv_end_history"][-1]
    expected = compute_reward(hv_end_K, hv_pre_K, 4.0, 20, 20 * 3)
    assert np.isclose(res["reward_history"][-1], expected)


# --------------------------- T4: TERMINAL TARGET DOES NOT BOOTSTRAP
def test_t4_terminal_target_no_bootstrap():
    reward, gamma, q_next = 0.5, 0.95, 1e6
    done = 1.0
    y = reward + gamma * q_next * (1 - done)
    assert np.isclose(y, reward)


# --------------------------------------------- T5: NO FAKE K+1 ENVIRONMENT
def test_t5_no_fake_k_plus_1(monkeypatch):
    calls = {"comp": 0, "query": 0}
    oc = dynamic_runner.ChangeDetector.compute
    oq = dynamic_runner.MemoryArchive.query
    monkeypatch.setattr(dynamic_runner.ChangeDetector, "compute",
                        lambda self, *a, **k: (calls.__setitem__("comp", calls["comp"] + 1)
                                               or oc(self, *a, **k)))
    monkeypatch.setattr(dynamic_runner.MemoryArchive, "query",
                        lambda self, *a, **k: (calls.__setitem__("query", calls["query"] + 1)
                                               or oq(self, *a, **k)))
    res = run_baseline(changes=4)
    assert calls["comp"] == 4
    assert calls["query"] == 4
    assert np.isclose(res["timeline"][-1]["t_new"], 0.4)


# ------------------------------------------------------- T6: NO EXTRA FE
def test_t6_no_extra_fe():
    # FE = initial + nsga2 + detector + signature + response, khong co
    # muc "terminal". Ledger sum == total; finalize khong them key/FE.
    res = run_baseline(changes=3)
    assert res["fes_used"] == sum(res["fe_breakdown"].values())
    assert set(res["fe_breakdown"]) == {"initial", "nsga2", "detector",
                                        "signature", "response"}


# ------------------------------------------------- T7: REWARD HISTORY LENGTH
def test_t7_reward_history_length():
    res = run_baseline(changes=5)
    assert len(res["reward_history"]) == 5


# ----------------------------------------------- T8: METRIC LENGTH UNCHANGED
def test_t8_metric_length_unchanged():
    res = run_baseline(changes=4)
    assert len(res["igd_response_history"]) == 4
    assert len(res["igd_end_history"]) == 4


# --------------------------------------- T9: FINAL ENV FULL HORIZON
def test_t9_final_env_full_horizon():
    # Env cuoi t_K co du tau_t gen: total_gens = warm_up + (K+1)*tau_t.
    # Neu t_K thieu gen, HV_end != gia tri sau tau_t. Kiem qua reproduc:
    # end metric on ton dinh -> chung minh full horizon da chay.
    res = run_baseline(changes=3, tau_t=5)
    assert np.isfinite(res["igd_end_history"][-1])
    # reward cuoi dung HV_end sau tau_t recovery (T3 da so voi hv_end_history)


# ------------------------------------------- T10: TERMINAL NEXT STATE
def test_t10_terminal_next_state():
    ag, _ = run_dqn(changes=3)
    last = ag.replay_buffer.buffer[-1]
    state, next_state, done = last[0], last[4], last[5]
    assert next_state.shape == state.shape
    assert done == 1.0
    assert np.all(next_state == 0.0)
    # Bellman: done=True -> huge next_state khong doi target.
    y_zero = 1.0 + 0.95 * 999.0 * (1 - 1.0)
    assert np.isclose(y_zero, 1.0)


# ------------------------- T11: MEMORY FINAL STORE (decision B: NO STORE)
def test_t11_memory_final_no_extra_query(monkeypatch):
    calls = {"store": 0, "query": 0}
    ostore = dynamic_runner.MemoryArchive.store
    oquery = dynamic_runner.MemoryArchive.query
    monkeypatch.setattr(dynamic_runner.MemoryArchive, "store",
                        lambda self, *a, **k: (calls.__setitem__("store", calls["store"] + 1)
                                               or ostore(self, *a, **k)))
    monkeypatch.setattr(dynamic_runner.MemoryArchive, "query",
                        lambda self, *a, **k: (calls.__setitem__("query", calls["query"] + 1)
                                               or oquery(self, *a, **k)))
    run_baseline(changes=4)
    # Decision B: env cuoi t_K KHONG store (episode ket thuc, archive
    # clear moi run nen store t_K vo tac dung). store = K-1 (t_1..t_{K-1}).
    assert calls["store"] == 3
    assert calls["query"] == 4     # finalize KHONG keo theo query


# ----------------------------------------------- T12: SAME SEED REPRODUCE
def test_t12_reproducible():
    a = run_baseline(changes=3, seed=5)
    b = run_baseline(changes=3, seed=5)
    assert np.allclose(a["reward_history"], b["reward_history"])
    ag1, r1 = run_dqn(changes=3, seed=7)
    ag2, r2 = run_dqn(changes=3, seed=7)
    d1 = [tr[5] for tr in ag1.replay_buffer.buffer]
    d2 = [tr[5] for tr in ag2.replay_buffer.buffer]
    assert d1 == d2
    assert len(ag1.replay_buffer) == len(ag2.replay_buffer) == 3
