"""
4.10B — FE fairness protocol audit (READ-ONLY).

Khoa cac su kien ve so sanh cong bang SA-DRL vs clean NSGA-II baseline:
FE analytical == ledger, khong fake FE, RNG streams khong tao confound
cho paired-seed comparison, moi response action co FE = N.

KHONG test/sua production behavior.
"""
import os
import sys

import numpy as np
import pytest
import torch
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner as dr  # noqa: E402
import nsga2_pymoo as ng  # noqa: E402
from baseline_runner import run_nsga2_baseline  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from nsga2_pymoo import seed_nsga2  # noqa: E402

SEGS = [[0], list(range(1, 10))]


def sa_run(N=20, W=5, T=3, K=3, seed=0, training=True):
    torch.manual_seed(seed)
    ag = DQNAgent(state_dim=6, n_segments=2, eps_fixed=0.3)
    return dr.run_sa_drl(DF1, n_t=10, tau_t=T, N=N, D=10, n_changes=K,
                         warm_up=W, agent=ag, segments=SEGS, seed=seed,
                         training=training, ref_point=(2.0, 2.0), n_elite=5)


def bl_run(N=20, W=5, T=3, K=3, seed=0):
    return run_nsga2_baseline(DF1, n_t=10, tau_t=T, N=N, D=10, n_changes=K,
                              warm_up=W, seed=seed, ref_point=(2.0, 2.0))


# ------------------ FEF1: same paired seed -> identical initial population
def test_fef1_same_seed_identical_init():
    def init(seed):
        import random
        np.random.seed(seed); random.seed(seed); seed_nsga2(seed)
        p = DF1(time=0.0, n_var=10)
        return p.xl + np.random.rand(20, 10) * (p.xu - p.xl)
    assert np.allclose(init(11), init(11))
    assert not np.allclose(init(11), init(12))


# ------- FEF2: nsga2 _rng isolated from controller's global np.random use
def test_fef2_nsga_rng_isolated_from_controller():
    seed_nsga2(123)
    a = [ng._rng.randint(0, 10**6) for _ in range(8)]
    seed_nsga2(123)
    # gia lap controller tieu thu GLOBAL np.random giua cac generation
    np.random.seed(42)
    [np.random.rand() for _ in range(100)]
    np.random.randint(0, 4, size=5)
    b = [ng._rng.randint(0, 10**6) for _ in range(8)]
    assert a == b       # controller khong lam lech NSGA-II variation stream


# ---------------------------- FEF3: baseline FE analytical == ledger
def test_fef3_baseline_fe_analytical():
    N, W, T, K = 20, 5, 3, 3
    res = bl_run(N=N, W=W, T=T, K=K)
    analytical = N * (1 + W + (K + 1) * T + K)
    assert res["fes_used"] == analytical
    assert res["fes_used"] == sum(res["fe_breakdown"].values())
    assert res["fe_breakdown"]["environment_reeval"] == K * N


# ----------------------------------- FEF4: SA ledger decomposition
def test_fef4_sa_ledger_decomposition():
    N, W, T, K = 20, 5, 3, 3
    sa = sa_run(N=N, W=W, T=T, K=K)
    bd = sa["fe_breakdown"]
    assert bd["initial"] == N
    assert bd["nsga2"] == (W + (K + 1) * T) * N
    assert bd["response"] == 2 * K * N        # F_pre (common) + F_response (SA-only)
    assert bd["signature"] == 5 * (K + 1)     # n_elite=5, calib + K change
    assert bd["detector"] > 0
    assert sa["fes_used"] == sum(bd.values())


# ------------------- FEF5: moi response action co pending FE = N
def test_fef5_response_fe_is_N_all_actions():
    # Runner set pending_action_fe = len(pop) = N bat ke gate/seg -> SA
    # response breakdown = 2KN cho MOI policy. Kiem qua nhieu seed (action
    # paths khac nhau) van cho response = 2KN.
    N, K = 20, 3
    for seed in range(4):
        sa = sa_run(N=N, K=K, seed=seed)
        assert sa["fe_breakdown"]["response"] == 2 * K * N


# ------------------------------- FEF6: baseline khong co fake FE key
def test_fef6_baseline_no_fake_overhead():
    bd = bl_run()["fe_breakdown"]
    assert set(bd) == {"initial", "nsga2", "environment_reeval"}
    for k in ("detector", "signature", "response", "memory"):
        assert k not in bd
    # baseline reeval = KN (1x), KHONG 2KN
    assert bd["environment_reeval"] == 3 * 20


# ---------------------- FEF7: SA overhead = detector + signature + KN extra
def test_fef7_overhead_fully_attributed():
    N, W, T, K = 20, 5, 3, 3
    sa, bl = sa_run(N=N, W=W, T=T, K=K), bl_run(N=N, W=W, T=T, K=K)
    overhead = sa["fes_used"] - bl["fes_used"]
    sa_only = (sa["fe_breakdown"]["detector"]
               + sa["fe_breakdown"]["signature"]
               + K * N)                        # second response eval
    assert overhead == sa_only
    assert sa["fes_used"] > bl["fes_used"]
