"""FZ1-FZ20: per-DF frozen training/evaluation protocol (4.10E.1)."""
import copy
import hashlib
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
from dqn_agent import DQNAgent  # noqa: E402
from nsga2_pymoo import seed_nsga2  # noqa: E402
from pymoo.problems.dynamic.df import DF1  # noqa: E402


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def run_frozen(names=("DF1",), n_ep=3, n_runs=2, K=3, capture=None):
    train.TRAIN_CHANGES = 3
    agents, probs = [], []
    orig = train.run_sa_drl        # train.py import truc tiep -> patch o day

    def spy(problem_class, *a, **k):
        agent = k.get("agent")
        if k.get("training") is False:      # CHI capture eval runs
            agents.append(agent)
            probs.append(problem_class)
            if capture is not None:
                capture.append((agent, hsd(agent.online.state_dict()),
                                len(agent.replay_buffer)))
        return orig(problem_class, *a, **k)
    train.run_sa_drl = spy
    try:
        res = train.mode_frozen(list(names), n_episodes=n_ep, n_runs=n_runs,
                                n_changes=K, log_path="/tmp/fz.jsonl")
    finally:
        train.run_sa_drl = orig
    return res, agents, probs


# ---------------- FZ1/FZ13: train/test seeds disjoint; no test in train
def test_fz1_fz13_seeds_disjoint():
    train_seeds = [train.TRAIN_SEED_BASE + ep for ep in range(200)]
    test_seeds = [train.EVAL_SEED_BASE + r for r in range(train.N_EVAL_RUNS)]
    assert set(train_seeds).isdisjoint(test_seeds)
    assert all(s < train.EVAL_SEED_BASE for s in train_seeds)


# ------------------------------------- FZ2: one agent per DF
def test_fz2_one_agent_per_df():
    a1 = train.new_agent(train.make_segments(10), 0)
    a2 = train.new_agent(train.make_segments(10), 0)
    assert a1 is not a2 and a1.online is not a2.online
    assert a1.replay_buffer is not a2.replay_buffer


# ------------------------------ FZ3: training is single-DF
def test_fz3_single_df_training():
    import json
    _, agents, probs = run_frozen(names=("DF1",), n_ep=3, n_runs=1)
    # eval-run problem classes all DF1 (frozen eval on DF1 only)
    names = {p.__name__ for p in probs}
    assert names == {"DF1"}


# --------------------- FZ4/FZ18: selection rule + metadata complete
def test_fz4_fz18_metadata():
    import json
    run_frozen(names=("DF1",), n_ep=3, n_runs=1)
    row = json.loads(open("/tmp/fz.jsonl").readlines()[-1])
    p = row["params"]
    assert p["protocol"] == "frozen_per_df"
    assert p["selection_rule"] == "fixed_final_checkpoint"
    assert p["epsilon_eval"] == 0.0
    assert p["training_during_eval"] is False
    assert p["segmentation"] == "fixed_index_two_block"
    for key in ("model_seed", "evaluation_seed", "target_tau",
                "state_dim", "n_segments"):
        assert key in p


# --------- FZ5/FZ15: eval clones get identical frozen online weights
def test_fz5_fz15_identical_frozen_weights():
    cap = []
    run_frozen(names=("DF1",), n_ep=3, n_runs=3, capture=cap)
    hashes = [h for _, h, _ in cap]
    assert len(set(hashes)) == 1               # all eval clones same weights


# ------------------------------ FZ6: fresh replay each test run
def test_fz6_fresh_replay():
    cap = []
    run_frozen(names=("DF1",), n_ep=3, n_runs=3, capture=cap)
    for _, _, buflen in cap:
        assert buflen == 0                     # moi eval clone bat dau rong


# ------------------------ FZ7/FZ8/FZ9: frozen params + no learn
def test_fz7_fz8_fz9_frozen_no_learn():
    learn = {"n": 0}
    ol = DQNAgent.learn
    DQNAgent.learn = lambda self: (learn.__setitem__("n", learn["n"] + 1)
                                   or ol(self))
    snaps = []
    orig = train.run_sa_drl

    def spy(pc, *a, **k):
        ag = k.get("agent")
        if k.get("training") is not False:
            return orig(pc, *a, **k)       # bo qua training runs
        b = (copy.deepcopy(ag.online.state_dict()),
             copy.deepcopy(ag.target.state_dict()), ag.step_count)
        r = orig(pc, *a, **k)
        on_ok = all(torch.equal(b[0][kk], ag.online.state_dict()[kk])
                    for kk in b[0])
        tg_ok = all(torch.equal(b[1][kk], ag.target.state_dict()[kk])
                    for kk in b[1])
        snaps.append((on_ok, tg_ok, ag.step_count == b[2]))
        return r
    learn_before = learn["n"]
    train.run_sa_drl = spy
    try:
        run_frozen(names=("DF1",), n_ep=3, n_runs=2)
    finally:
        train.run_sa_drl = orig
        DQNAgent.learn = ol
    learn["n"] -= learn_before
    # eval runs: params unchanged, step_count unchanged
    for on_ok, tg_ok, step_ok in snaps:
        assert on_ok and tg_ok and step_ok
    # step_count bat bien trong moi eval run -> khong optimizer step/learn
    assert all(step_ok for _, _, step_ok in snaps)


# ----------------------------------- FZ10: epsilon zero deterministic
def test_fz10_epsilon_zero():
    seg = train.make_segments(10)
    ag = train.new_agent(seg, 0)
    ag.eps = 0.0
    st = np.random.RandomState(0).rand(6).astype(np.float32); st[-1] = 1
    outs = [ag.select_action(st, training=False) for _ in range(10)]
    assert all(o[0] == outs[0][0] and np.array_equal(o[1], outs[0][1])
               for o in outs)


# ----------------- FZ11/FZ12: paired init pop + NSGA _rng
def test_fz11_fz12_paired_init():
    import nsga2_pymoo as ng

    def init(seed):
        np.random.seed(seed)
        import random
        random.seed(seed)
        seed_nsga2(seed)
        p = DF1(time=0.0, n_var=10)
        return p.xl + np.random.rand(20, 10) * (p.xu - p.xl)
    assert np.allclose(init(100000), init(100000))
    seed_nsga2(100000); a = ng._rng.randint(0, 10**6, size=5)
    seed_nsga2(100000); b = ng._rng.randint(0, 10**6, size=5)
    assert np.array_equal(a, b)


# ----------------- FZ14: model seed independent of test seed
def test_fz14_model_seed_fixed():
    import json
    run_frozen(names=("DF1",), n_ep=3, n_runs=3)
    rows = [json.loads(l) for l in open("/tmp/fz.jsonl")][-3:]
    mseeds = {r["params"]["model_seed"] for r in rows}
    eseeds = {r["params"]["evaluation_seed"] for r in rows}
    assert len(mseeds) == 1                    # model seed co dinh
    assert len(eseeds) == 3                    # test seeds khac nhau
    assert train.TRAIN_SEED_BASE not in eseeds


# --------------- FZ16: fresh detector/memory per eval run
def test_fz16_fresh_episode_state():
    news = {"a": 0, "d": 0}
    oa, od = dr.MemoryArchive.__init__, dr.ChangeDetector.__init__
    dr.MemoryArchive.__init__ = lambda self, *a, **k: (
        news.__setitem__("a", news["a"] + 1) or oa(self, *a, **k))
    dr.ChangeDetector.__init__ = lambda self, *a, **k: (
        news.__setitem__("d", news["d"] + 1) or od(self, *a, **k))
    try:
        run_frozen(names=("DF1",), n_ep=3, n_runs=2)
    finally:
        dr.MemoryArchive.__init__ = oa
        dr.ChangeDetector.__init__ = od
    # 3 train episodes + 2 eval runs = 5 runs, each fresh archive+detector
    assert news["a"] == 5 and news["d"] == 5


# -------------------------------------- FZ17: segmentation lock
def test_fz17_segmentation():
    assert train.make_segments(10) == [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9]]


# --------------- FZ19: online mode unchanged (training=True, eps 0.3)
def test_fz19_online_unchanged():
    import inspect
    src = inspect.getsource(train.mode_online)
    assert "eps_fixed=0.3" in src
    assert "training=True" in src


# ------------------------------------ FZ20: frozen eval reproducible
def test_fz20_reproducible():
    r1, _, _ = run_frozen(names=("DF1",), n_ep=3, n_runs=2)
    r2, _, _ = run_frozen(names=("DF1",), n_ep=3, n_runs=2)
    assert np.allclose(r1["DF1"][2], r2["DF1"][2])
