"""U1-U10: Polyak soft target update for hierarchical DQN (4.4)."""
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from dqn_agent import DQNAgent  # noqa: E402

S = 2
SD = S + 4


def agent(**kw):
    torch.manual_seed(0)
    return DQNAgent(state_dim=SD, n_segments=S, **kw)


def fill_buffer(ag, n):
    for i in range(n):
        s0 = np.random.rand(SD).astype(np.float32)
        s0[-1] = 1.0
        s1 = np.random.rand(SD).astype(np.float32)
        s1[-1] = 1.0
        ag.replay_buffer.push(s0, 0, np.array([1, 2]), float(i), s1, 0.0)


def set_all(net, val):
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(val)


# ------------------------------------------- U1: INITIAL TARGET == ONLINE
def test_u1_initial_equal():
    ag = agent()
    for po, pt in zip(ag.online.parameters(), ag.target.parameters()):
        assert torch.allclose(po, pt)


# ---------------------------------------------- U2: ONE SOFT UPDATE FORMULA
def test_u2_one_soft_update():
    ag = agent(target_tau=0.01)
    set_all(ag.online, 1.0)
    set_all(ag.target, 0.0)
    ag.soft_update_target()
    for pt in ag.target.parameters():
        assert torch.allclose(pt, torch.full_like(pt, 0.01), atol=1e-7)


# ------------------------------------------------- U3: MULTIPLE SOFT UPDATES
def test_u3_multiple():
    ag = agent(target_tau=0.01)
    set_all(ag.online, 1.0)
    set_all(ag.target, 0.0)
    n = 10
    for _ in range(n):
        ag.soft_update_target()
    expected = 1 - (1 - 0.01) ** n
    for pt in ag.target.parameters():
        assert torch.allclose(pt, torch.full_like(pt, expected), atol=1e-6)


# ------------------------------------------- U4: TARGET DOES NOT HARD COPY
def test_u4_no_hard_copy():
    ag = agent(target_tau=0.01)
    set_all(ag.online, 1.0)
    set_all(ag.target, 0.0)
    ag.soft_update_target()
    for pt in ag.target.parameters():
        assert torch.allclose(pt, torch.full_like(pt, 0.01), atol=1e-7)
        assert not torch.allclose(pt, torch.ones_like(pt))


# ---------------------------------------------- U5: LEARN UPDATES TARGET
def test_u5_learn_updates_target():
    ag = agent(batch_size=8)
    fill_buffer(ag, 8)
    before = [p.clone() for p in ag.target.parameters()]
    assert ag.learn() is not None
    changed = any(not torch.allclose(b, a)
                  for b, a in zip(before, ag.target.parameters()))
    assert changed


# ---------------------------------- U6: EXACTLY ONE SOFT UPDATE PER LEARN
def test_u6_one_per_learn(monkeypatch):
    ag = agent(batch_size=8)
    fill_buffer(ag, 8)
    calls = {"n": 0}
    orig = ag.soft_update_target
    monkeypatch.setattr(ag, "soft_update_target",
                        lambda: (calls.__setitem__("n", calls["n"] + 1) or orig()))
    for _ in range(5):
        ag.learn()
    assert calls["n"] == 5


# ------------------------------------- U7: NO UPDATE IF LEARN RETURNS EARLY
def test_u7_no_update_early_return(monkeypatch):
    ag = agent(batch_size=64)
    fill_buffer(ag, 10)              # < batch_size
    calls = {"n": 0}
    orig = ag.soft_update_target
    monkeypatch.setattr(ag, "soft_update_target",
                        lambda: (calls.__setitem__("n", calls["n"] + 1) or orig()))
    before = [p.clone() for p in ag.target.parameters()]
    assert ag.learn() is None
    assert calls["n"] == 0
    for b, a in zip(before, ag.target.parameters()):
        assert torch.allclose(b, a)


# --------------------------------------- U8: TERMINAL TRANSITION STILL WORKS
def test_u8_terminal_ok():
    ag = agent(batch_size=4)
    for i in range(4):
        s = np.random.rand(SD).astype(np.float32); s[-1] = 1.0
        ns = np.zeros(SD, dtype=np.float32)
        ag.replay_buffer.push(s, 0, np.array([1, 2]), 1.0, ns, 1.0)
    before = [p.clone() for p in ag.target.parameters()]
    assert ag.learn() is not None
    assert any(not torch.allclose(b, a)
               for b, a in zip(before, ag.target.parameters()))


# --------------------------------------- U9: HIERARCHICAL SHAPES UNCHANGED
@pytest.mark.parametrize("n_seg", [1, 2, 6])
def test_u9_shapes(n_seg):
    ag = DQNAgent(state_dim=n_seg + 4, n_segments=n_seg)
    x = torch.rand(3, n_seg + 4)
    for net in (ag.online, ag.target):
        qg, adv = net(x)
        assert qg.shape == (3, 2)
        assert adv.shape == (3, n_seg, 4)


# ------------------------------------------------------ U10: REPRODUCIBILITY
def test_u10_reproducible():
    def build_run():
        torch.manual_seed(3)
        np.random.seed(3)
        ag = DQNAgent(state_dim=SD, n_segments=S, batch_size=8)
        rng = np.random.RandomState(0)
        trans = [(rng.rand(SD).astype(np.float32), rng.rand(SD).astype(np.float32))
                 for _ in range(8)]
        for s0, s1 in trans:
            s0[-1] = 1.0; s1[-1] = 1.0
            ag.replay_buffer.push(s0, 0, np.array([1, 2]), 0.5, s1, 0.0)
        for _ in range(5):
            ag.learn()
        return ag
    a, b = build_run(), build_run()
    for pa, pb in zip(a.online.parameters(), b.online.parameters()):
        assert torch.allclose(pa, pb)
    for pa, pb in zip(a.target.parameters(), b.target.parameters()):
        assert torch.allclose(pa, pb)
