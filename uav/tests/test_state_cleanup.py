"""S1-S12: state semantics cleanup (4.8) — drop oracle/dead features."""
import os
import sys

import numpy as np
import pytest
import torch
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
import sa_drl_dmoea  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from dqn_agent import DQNAgent, HierarchicalQNetwork  # noqa: E402
from sa_drl_dmoea import (IDX_D_MEM, IDX_HAS_MEMORY,  # noqa: E402
                          build_state, compute_dispersion)

SEGS2 = [[0], list(range(1, 10))]


def run_short(changes=3, warm_up=5, tau_t=3, N=20, seed=0):
    return dynamic_runner.run_sa_drl(
        DF1, n_t=10, tau_t=tau_t, N=N, D=10, n_changes=changes,
        warm_up=warm_up, agent=NSGA2Baseline(n_segments=2), segments=SEGS2,
        seed=seed, training=False, ref_point=(2.0, 2.0), n_elite=5)


# --------------------------------------------------- S1: NEW STATE SHAPE
@pytest.mark.parametrize("S", [1, 2, 6])
def test_s1_shape(S):
    st = build_state(np.zeros(S), 0.3, 0.4, 0.5, 1.0)
    assert len(st) == S + 4


# ---------------------------------------------------- S2: EXACT ORDER
def test_s2_order():
    st = build_state(np.array([0.1, 0.2]), 0.3, 0.4, 0.5, 1.0)
    assert np.allclose(st, [0.1, 0.2, 0.3, 0.4, 0.5, 1.0])


# --------------------------------------------- S3: MEMORY INDICES STABLE
@pytest.mark.parametrize("S", [1, 2, 6])
def test_s3_memory_indices(S):
    st = build_state(np.zeros(S), 0.3, 0.4, d_mem=0.7, has_memory=1.0)
    assert st[IDX_D_MEM] == 0.7
    assert st[IDX_HAS_MEMORY] == 1.0


# ----------------------------------- S4: DISPERSION COLLAPSED POPULATION
def test_s4_collapsed():
    pop = np.full((30, 10), 0.5)
    assert compute_dispersion(pop, np.zeros(10), np.ones(10)) < 1e-6


# ------------------------------------- S5: DISPERSION UNIFORM POPULATION
@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 2.0)])
def test_s5_uniform(lo, hi):
    xl, xu = np.full(10, lo), np.full(10, hi)
    pop = xl + np.random.RandomState(0).rand(3000, 10) * (xu - xl)
    d = compute_dispersion(pop, xl, xu)
    assert 0.9 < d <= 1.0, d


# --------------------------------------- S6: DISPERSION SCALE INVARIANCE
def test_s6_scale_invariance():
    Z = np.random.RandomState(1).rand(500, 10)
    d1 = compute_dispersion(Z, np.zeros(10), np.ones(10))
    X2 = -2 + 4 * Z
    d2 = compute_dispersion(X2, np.full(10, -2.0), np.full(10, 2.0))
    assert np.isclose(d1, d2, atol=1e-9)


# ------------------------- S7: LIVE RUNNER KHONG CALL compute_entropy
def test_s7_no_compute_entropy(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("compute_entropy goi trong live path")
    monkeypatch.setattr(sa_drl_dmoea, "compute_entropy", boom)
    run_short()      # khong raise


# --------------------------- S8: LIVE RUNNER KHONG CALL compute_phase
def test_s8_no_compute_phase(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("compute_phase (oracle) goi trong live path")
    monkeypatch.setattr(sa_drl_dmoea, "compute_phase", boom)
    # runner khong import compute_phase nen patch tren module goc du
    run_short()


# ------------------------------------------ S9: NO g_last IN build_state
def test_s9_no_g_last():
    with pytest.raises(TypeError):
        build_state(np.zeros(2), 0.3, 0.4, 0, 3, 0.0, 0.5, 1.0)  # chu ky cu
    import inspect
    assert "g_last" not in inspect.signature(build_state).parameters
    assert "g_last" not in inspect.getsource(dynamic_runner.run_sa_drl)


# --------------------------- S10: HIERARCHICAL SHAPES AFTER SHRINK
@pytest.mark.parametrize("S", [1, 2, 6])
def test_s10_shapes(S):
    ag = DQNAgent(state_dim=S + 4, n_segments=S)
    x = torch.rand(4, S + 4)
    qg, adv = ag.online(x)
    assert qg.shape == (4, 2) and adv.shape == (4, S, 4)
    st = np.random.rand(S + 4).astype(np.float32)
    st[IDX_HAS_MEMORY] = 1.0
    gate, seg = ag.select_action(st, training=False)
    assert gate in (0, 1) and seg.shape == (S,)


# ----------------------- S11: d_mem DETERMINISTIC WIRING (state_dim=S+4)
def test_s11_d_mem_wired():
    S = 2
    sd = S + 4
    net = HierarchicalQNetwork(sd, S, hidden=64)
    idx_dmem = sd + IDX_D_MEM
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.trunk[0].weight[0, idx_dmem] = 1.0
        net.trunk[2].weight[0, 0] = 1.0
        net.gate_head.weight[0, 0] = 1.0
    a = torch.zeros(1, sd); b = torch.zeros(1, sd); b[0, idx_dmem] = 1.0
    qa, _ = net(a); qb, _ = net(b)
    assert abs(qa[0, 0].item() - 0.0) < 1e-6
    assert abs(qb[0, 0].item() - 1.0) < 1e-6


# --------------------------------------- S12: FULL SHORT RUN FINITE STATE
def test_s12_states_finite(monkeypatch):
    seen = []
    orig = dynamic_runner.build_state

    def spy(*a, **k):
        st = orig(*a, **k)
        seen.append(st)
        return st
    monkeypatch.setattr(dynamic_runner, "build_state", spy)
    run_short(changes=4)
    assert len(seen) == 4
    for st in seen:
        assert st.shape == (2 + 4,)
        assert np.all(np.isfinite(st))
        assert st[IDX_HAS_MEMORY] in (0.0, 1.0)
        assert 0.0 <= st[IDX_D_MEM] <= 1.0
