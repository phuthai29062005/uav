"""
Test baseline NSGA-II thuan (doi chung Bang 1).

Chay:  python -m pytest tests/test_baseline.py -v
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from baselines import NSGA2Baseline  # noqa: E402
import train  # noqa: E402


def test_baseline_always_keeps():
    """select_action luon tra ve keep-het (0) cho moi phan doan."""
    agent = NSGA2Baseline(n_segments=2)
    for _ in range(10):
        state = np.random.rand(6)
        actions = agent.select_action(state)
        assert np.all(actions == 0)
        assert actions.shape == (2,)
        assert actions.dtype in (np.int64, np.int32)


def test_baseline_noop_methods():
    """learn/decay_epsilon/replay_buffer.push la no-op, khong raise."""
    agent = NSGA2Baseline(n_segments=2)
    assert agent.learn() is None
    agent.decay_epsilon()                      # khong raise
    agent.replay_buffer.push(1, 2, 3, 4, 5)    # khong raise
    assert len(agent.replay_buffer) == 0


def test_baseline_runs_end_to_end():
    """mode_baseline chay duoc va tra ve dung format."""
    res = train.mode_baseline(["DF1"], n_runs=2, n_changes=5)

    assert "DF1" in res
    mean, std, migds = res["DF1"]
    assert np.isfinite(mean), f"mean khong huu han: {mean}"
    assert np.isfinite(std), f"std khong huu han: {std}"
    assert len(migds) == 2
