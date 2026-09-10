"""H1-H10: hierarchical gate+segment controller, single Q_total loss."""
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from dqn_agent import (DQNAgent, HierarchicalQNetwork,  # noqa: E402
                       best_hierarchical)
from dynamic_runner import validate_transition  # noqa: E402
from sa_drl_dmoea import (IDX_D_MEM, IDX_HAS_MEMORY,  # noqa: E402
                          apply_hierarchical_response, repair)

S = 2
STATE_DIM = S + 6


def make_state(has_memory, d_mem=0.5, seed=0):
    st = np.random.RandomState(seed).rand(STATE_DIM).astype(np.float32)
    st[IDX_D_MEM] = d_mem
    st[IDX_HAS_MEMORY] = float(has_memory)
    return st


def agent_with_huge_q_mem(seed=0):
    torch.manual_seed(seed)
    ag = DQNAgent(state_dim=STATE_DIM, n_segments=S)
    with torch.no_grad():
        ag.online.gate_head.bias[1] = 1e6
    return ag


# ------------------------------------------------- H1: MASK WHEN EMPTY
def test_h1_mask_when_archive_empty():
    ag = agent_with_huge_q_mem()
    st = make_state(has_memory=0)
    for _ in range(100):
        assert ag.select_action(st, training=False)[0] == 0
    ag.eps = 1.0
    np.random.seed(0)
    for _ in range(100):
        assert ag.select_action(st, training=True)[0] == 0


# --------------------------------- H2: MEMORY ALLOWED WHEN NONEMPTY
def test_h2_memory_allowed_when_nonempty():
    ag = agent_with_huge_q_mem()
    gate, seg = ag.select_action(make_state(has_memory=1), training=False)
    assert gate == 1
    assert seg.shape == (S,)


# --------------------------- H3: MEMORY EXECUTES WHOLE POP, NO SEG
def test_h3_memory_replaces_whole_pop_ignores_seg():
    rng = np.random.RandomState(1)
    pop = rng.rand(20, 10)
    pop_mem = rng.rand(20, 10)
    segs = [[0], list(range(1, 10))]
    out = apply_hierarchical_response(pop, pop.copy(), 1, np.array([3, 3]),
                                      segs, pop_mem)
    assert np.allclose(out, repair(pop_mem))
    assert not np.allclose(out, pop)


# ------------------------------ H4: NO_MEMORY LEAVES pop_mem UNTOUCHED
def test_h4_no_memory_ignores_pop_mem():
    rng = np.random.RandomState(2)
    pop = rng.rand(20, 10)
    pop_mem = rng.rand(20, 10)
    segs = [[0], list(range(1, 10))]
    out = apply_hierarchical_response(pop, pop.copy(), 0, np.array([0, 0]),
                                      segs, pop_mem)
    assert np.allclose(out, pop)


# ------------------------------------ H5/H6: gradient masking / flow
def _one_transition_learn(gate, seg):
    torch.manual_seed(0)
    ag = DQNAgent(state_dim=STATE_DIM, n_segments=S, batch_size=1)
    s0 = make_state(has_memory=1, seed=3)
    s1 = make_state(has_memory=1, seed=4)
    ag.replay_buffer.push(s0, gate, np.array(seg), 1.0, s1, 0.0)
    assert ag.learn() is not None
    return ag


def test_h5_gradient_masked_on_memory_transition():
    ag = _one_transition_learn(gate=1, seg=[0, 0])
    for p in ag.online.segment_head.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
    assert any(torch.any(p.grad != 0)
               for p in ag.online.gate_head.parameters())


def test_h6_gradient_flows_on_no_memory_transition():
    ag = _one_transition_learn(gate=0, seg=[1, 2])
    assert torch.any(ag.online.segment_head.weight.grad != 0)
    assert torch.any(ag.online.gate_head.weight.grad != 0)


# ------------------------------------------------------ H7: d_mem WIRED
def test_d_mem_is_wired_deterministic():
    """Mot duong dan duy nhat d_mem -> trunk -> gate[NO]; khong phu thuoc seed."""
    net = HierarchicalQNetwork(STATE_DIM, S, hidden=64)
    idx_dmem = STATE_DIM + IDX_D_MEM
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.trunk[0].weight[0, idx_dmem] = 1.0
        net.trunk[2].weight[0, 0] = 1.0
        net.gate_head.weight[0, 0] = 1.0

    base = torch.zeros(1, STATE_DIM)
    s_a = base.clone(); s_a[0, idx_dmem] = 0.0
    s_b = base.clone(); s_b[0, idx_dmem] = 1.0
    q_a, _ = net(s_a)
    q_b, _ = net(s_b)

    assert abs(q_a[0, 0].item() - 0.0) < 1e-6
    assert abs(q_b[0, 0].item() - 1.0) < 1e-6
    assert abs(q_a[0, 1].item()) < 1e-6
    assert abs(q_b[0, 1].item()) < 1e-6


# ------------------------------------ H11: MEMORY => has_memory INVARIANT
def test_memory_without_archive_raises():
    state = np.zeros(STATE_DIM); state[IDX_HAS_MEMORY] = 0.0
    with pytest.raises(RuntimeError, match="archive was empty"):
        validate_transition(state, gate=1)
    state[IDX_HAS_MEMORY] = 1.0
    validate_transition(state, gate=1)
    state[IDX_HAS_MEMORY] = 0.0
    validate_transition(state, gate=0)


# ----------------------------------------------- H8: SHAPES S=1,2,6
@pytest.mark.parametrize("n_seg", [1, 2, 6])
def test_h8_shapes(n_seg):
    ag = DQNAgent(state_dim=n_seg + 6, n_segments=n_seg)
    x = torch.rand(5, n_seg + 6)
    q_gate, adv = ag.online(x)
    assert q_gate.shape == (5, 2)
    assert adv.shape == (5, n_seg, 4)
    assert torch.allclose(adv.mean(dim=2), torch.zeros(5, n_seg), atol=1e-6)
    st = np.random.rand(n_seg + 6).astype(np.float32)
    st[IDX_HAS_MEMORY] = 1.0
    gate, seg = ag.select_action(st, training=False)
    assert gate in (0, 1) and seg.shape == (n_seg,)


# ------------------------- H9: DOUBLE DQN COMPARES FULL HIERARCHICAL
def test_h9_best_hierarchical_uses_full_q_total():
    adv = torch.zeros(1, 2, 4)
    adv[0, 0, 1] = 0.30
    adv[0, 1, 2] = 0.20
    adv = adv - adv.mean(dim=2, keepdim=True)     # dung dang mang tra ve
    # max cua adv sau khi tru mean: 0.30-0.075=0.225, 0.20-0.05=0.15
    q_gate = torch.tensor([[0.40, 0.62]])
    has = torch.tensor([1.0])
    q_no_best = 0.40 + (0.225 + 0.15) / 2      # 0.5875 < 0.62 -> gate 1?
    # De giu dung vi du so 9 (max A = 0.30/0.20), dung adv da la advantage:
    adv2 = torch.zeros(1, 2, 4)
    adv2[0, 0] = torch.tensor([0.30, -0.10, -0.10, -0.10])   # mean 0, max .30
    adv2[0, 1] = torch.tensor([0.20, -0.20, 0.00, 0.00])      # mean 0, max .20
    g, s, q = best_hierarchical(q_gate, adv2, has)
    assert g.item() == 0                       # 0.40+0.25=0.65 > 0.62
    assert abs(q.item() - 0.65) < 1e-6
    assert s.tolist() == [[0, 0]]
    g2, _, q2 = best_hierarchical(torch.tensor([[0.40, 0.80]]), adv2, has)
    assert g2.item() == 1 and abs(q2.item() - 0.80) < 1e-6
    # mask: has_memory=0 -> luon gate 0 du q_mem lon
    g3, _, _ = best_hierarchical(torch.tensor([[0.40, 0.80]]), adv2,
                                 torch.tensor([0.0]))
    assert g3.item() == 0
    del q_no_best, adv


# ----------------------------- H10: REPLAY STORES HIERARCHICAL ACTION
def test_h10_replay_stores_hierarchical_action():
    ag = DQNAgent(state_dim=STATE_DIM, n_segments=S)
    trans = [(0, [1, 2]), (1, [0, 0]), (0, [3, 0])]
    for i, (g, sg) in enumerate(trans):
        ag.replay_buffer.push(make_state(1, seed=i), g, np.array(sg),
                              float(i), make_state(1, seed=i + 10), 0.0)
    st, gates, segs, rw, _, _ = ag.replay_buffer.sample(3)
    by_reward = {int(r): (int(g), list(map(int, s)))
                 for r, g, s in zip(rw, gates, segs)}
    for i, (g, sg) in enumerate(trans):
        assert by_reward[i] == (g, sg)
