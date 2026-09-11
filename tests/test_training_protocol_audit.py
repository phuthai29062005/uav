"""TR1-TR14: training/evaluation protocol audit (READ-ONLY, 4.10E)."""
import copy
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import dynamic_runner as dr  # noqa: E402
import train  # noqa: E402
from baseline_runner import run_nsga2_baseline  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from nsga2_pymoo import seed_nsga2  # noqa: E402
from pymoo.problems.dynamic.df import DF1  # noqa: E402

SEGS = [[0], list(range(1, 10))]
REF = (2.0, 2.0)


def frozen_eval(agent, seed=100000, K=6):
    return dr.run_sa_drl(DF1, 10, 3, 20, 10, K, 5, agent=agent,
                         segments=SEGS, seed=seed, training=False,
                         ref_point=REF, n_elite=5)


def trained_agent(seed=0, pushes=70):
    torch.manual_seed(seed)
    ag = DQNAgent(state_dim=6, n_segments=2)
    rng = np.random.RandomState(seed)
    for _ in range(pushes):
        s = rng.rand(6).astype(np.float32); s[-1] = 1
        ag.replay_buffer.push(s, 0, np.array([1, 2]), 1.0,
                              rng.rand(6).astype(np.float32), 0.0)
    ag.learn()
    ag.eps = 0.0
    return ag


# --------------------- TR1/TR13: train / eval seed sets disjoint
def test_tr1_tr13_seed_disjoint():
    train_seeds = set(range(30, 30 + 200))            # train_on seed=30 + ep
    eval_seeds = set(train.EVAL_SEED_BASE + r for r in range(train.N_EVAL_RUNS))
    assert train_seeds.isdisjoint(eval_seeds)
    assert train.EVAL_SEED_BASE == 100_000
    assert max(train_seeds) < train.EVAL_SEED_BASE


# ------------- TR2/TR3/TR4: frozen eval leaves params unchanged, no step
def test_tr2_tr3_tr4_frozen_params():
    ag = trained_agent()
    on0 = copy.deepcopy(ag.online.state_dict())
    tg0 = copy.deepcopy(ag.target.state_dict())
    step0 = ag.step_count
    frozen_eval(ag)
    assert all(torch.equal(on0[k], ag.online.state_dict()[k]) for k in on0)
    assert all(torch.equal(tg0[k], ag.target.state_dict()[k]) for k in tg0)
    assert ag.step_count == step0                     # no learn -> no step


# ------------------------- TR5: eps=0 greedy deterministic action
def test_tr5_greedy_deterministic():
    ag = trained_agent()
    st = np.random.RandomState(0).rand(6).astype(np.float32)
    st[-1] = 1.0
    a = [ag.select_action(st, training=False) for _ in range(20)]
    for g, s in a[1:]:
        assert g == a[0][0] and np.array_equal(s, a[0][1])


# ----------------- TR6/TR7: fresh archive + detector per run
def test_tr6_tr7_fresh_episode_state():
    news = {"archive": 0, "detector": 0}
    oa, od = dr.MemoryArchive.__init__, dr.ChangeDetector.__init__

    def a_init(self, *a, **k):
        news["archive"] += 1
        return oa(self, *a, **k)

    def d_init(self, *a, **k):
        news["detector"] += 1
        return od(self, *a, **k)
    dr.MemoryArchive.__init__ = a_init
    dr.ChangeDetector.__init__ = d_init
    try:
        ag = trained_agent()
        frozen_eval(ag, seed=100000)
        frozen_eval(ag, seed=100001)
    finally:
        dr.MemoryArchive.__init__ = oa
        dr.ChangeDetector.__init__ = od
    assert news["archive"] == 2 and news["detector"] == 2   # 1 moi run


# -------------------------- TR8: per-DF agent instances independent
def test_tr8_agent_instances_independent():
    a1 = train.new_agent(SEGS, seed=0)
    a2 = train.new_agent(SEGS, seed=1)
    assert a1 is not a2
    assert a1.online is not a2.online
    assert a1.replay_buffer is not a2.replay_buffer


# --------------- TR10/TR11: paired seed same init pop + same _rng
def test_tr10_tr11_paired_init():
    def init(seed):
        np.random.seed(seed)
        import random
        random.seed(seed)
        seed_nsga2(seed)
        p = DF1(time=0.0, n_var=10)
        return p.xl + np.random.rand(20, 10) * (p.xu - p.xl)
    assert np.allclose(init(100000), init(100000))
    # _rng deterministic tu seed_nsga2
    import nsga2_pymoo as ng
    seed_nsga2(100000); a = ng._rng.randint(0, 10**6, size=5)
    seed_nsga2(100000); b = ng._rng.randint(0, 10**6, size=5)
    assert np.array_equal(a, b)


# --------------------- TR12: result metadata records DF/seed/params
def test_tr12_metadata_recorded():
    from experiment_logger import ExperimentLogger
    import json
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "m.jsonl")
    lg = ExperimentLogger(path)
    with lg.run(algo="SA-DRL", problem="DF1", seed=100000,
                params={"N": 20, "tau_t": 3}):
        lg.record(migd=0.1, hv=1.0, fes=100, feasible_ratio=1.0)
    row = json.loads(open(path).readline())
    assert row["problem"] == "DF1" and row["seed"] == 100000
    assert row["params"]["N"] == 20 and "git_commit" in row


# ------------------------------ TR14: frozen eval reproducible
def test_tr14_frozen_reproducible():
    r1 = frozen_eval(trained_agent(seed=3), seed=100000)
    r2 = frozen_eval(trained_agent(seed=3), seed=100000)
    assert np.allclose(r1["igd_end_history"], r2["igd_end_history"])
    # baseline paired same seed reproducible
    b1 = run_nsga2_baseline(DF1, 10, 3, 20, 10, 6, 5, seed=100000, ref_point=REF)
    b2 = run_nsga2_baseline(DF1, 10, 3, 20, 10, 6, 5, seed=100000, ref_point=REF)
    assert np.allclose(b1["igd_end_history"], b2["igd_end_history"])
