"""
Test lich epsilon hai che do (A1).

- Mode online: eps CO DINH = 0.3, khong decay (chi 1 episode/agent).
- Mode debug/loo/ablation: eps decay per-episode nhu cu.

Chay:  python -m pytest tests/test_epsilon_schedule.py -v
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import train  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402


def test_online_eps_stays_fixed():
    """
    Lap lai dung quy trinh mode_online (3 run, moi run 1 agent moi,
    eps_fixed=0.3, training=True). Sau moi run eps phai van = 0.3
    (decay_epsilon o cuoi run la no-op khi eps_fixed).
    """
    segments = train.make_segments(train.CFG["D"])
    for run in range(3):
        seed = train.EVAL_SEED_BASE + run
        agent = train.new_agent(segments, seed, eps_fixed=0.3)
        initial_eps = agent.eps
        train.one_run("DF1", agent, segments, seed,
                      training=True, n_changes=5)
        assert agent.eps == 0.3, (
            f"run {run}: eps thay doi {initial_eps} -> {agent.eps} "
            "(le ra phai giu co dinh 0.3)"
        )


def test_debug_eps_decays(monkeypatch):
    """
    Nhanh huan luyen debug (train_on, eps_fixed=None) chay 5 episode.
    Chuoi eps ghi trong log phai GIAM don dieu, va eps cuoi < eps dau.
    Rut TRAIN_CHANGES cho nhanh.
    """
    monkeypatch.setattr(train, "TRAIN_CHANGES", 3)
    segments = train.make_segments(train.CFG["D"])

    _, log = train.train_on(["DF1"], 5, segments)
    eps_seq = [r["eps"] for r in log]

    assert len(eps_seq) == 5
    # giam don dieu (khong tang)
    for i in range(1, len(eps_seq)):
        assert eps_seq[i] <= eps_seq[i - 1] + 1e-12, (
            f"eps tang o episode {i}: {eps_seq}"
        )
    assert eps_seq[-1] < eps_seq[0], f"eps cuoi khong nho hon dau: {eps_seq}"


def test_eps_fixed_disables_decay():
    """
    Unit test cap thap:
    - eps_fixed=0.3: goi decay_epsilon() 10 lan -> eps van = 0.3.
    - khong eps_fixed: goi decay_epsilon() 10 lan -> eps giam.
    """
    fixed = DQNAgent(state_dim=6, n_segments=2, eps_fixed=0.3)
    for _ in range(10):
        fixed.decay_epsilon()
    assert fixed.eps == 0.3, f"eps_fixed nhung eps van doi: {fixed.eps}"

    normal = DQNAgent(state_dim=6, n_segments=2)  # eps_fixed=None
    start = normal.eps
    for _ in range(10):
        normal.decay_epsilon()
    assert normal.eps < start, (
        f"eps le ra phai giam: {start} -> {normal.eps}"
    )
