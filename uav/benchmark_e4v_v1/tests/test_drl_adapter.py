"""Tests for DRL adapter (T49-T60).

T49-T51: Reference direction generation
T52-T53: Region assignment
T54-T55: State computation
T56-T57: Controller loading and inference
T58-T60: Runner integration with DRL
"""

import glob
import numpy as np
import pytest
import torch

from benchmark_e4v_v1.src.controllers.drl_adapter import (
    generate_reference_directions, assign_regions, compute_severity,
    compute_quality_loss, build_state_9dim, load_controller,
    select_actions_greedy, execute_response, DRLController,
    N_REF_BI, N_REF_TRI, R_MAX, ACTION_NAMES,
)
from benchmark_e4v_v1.src.runner import run_single_experiment


def _find_checkpoint(arm_prefix: str, policy_seed: int = 8001) -> str:
    pattern = f"results/E4TTR_c4_gated_coordination_training_execution/{arm_prefix}*/policy_{policy_seed}/checkpoints/ckpt_final.pt"
    paths = glob.glob(pattern)
    if not paths:
        pytest.skip(f"No checkpoint found for {arm_prefix}")
    return paths[0]


# ── T49-T51: Reference directions ──────────────────────────────

class TestReferenceDirections:

    def test_t49_bi_objective_directions(self):
        """Bi-objective generates N_REF_BI unit-norm directions."""
        V = generate_reference_directions(2)
        assert V.shape == (N_REF_BI, 2)
        norms = np.linalg.norm(V, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_t50_tri_objective_directions(self):
        """Tri-objective generates >= N_REF_TRI unit-norm directions."""
        V = generate_reference_directions(3)
        assert V.shape[0] >= N_REF_TRI
        assert V.shape[1] == 3
        norms = np.linalg.norm(V, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_t51_directions_span_simplex(self):
        """Bi-obj directions cover extremes and middle."""
        V = generate_reference_directions(2)
        ratios = V[:, 0] / (V[:, 0] + V[:, 1])
        assert min(ratios) < 0.1
        assert max(ratios) > 0.9


# ── T52-T53: Region assignment ─────────────────────────────────

class TestRegionAssignment:

    def test_t52_all_assigned(self):
        """Every solution gets a region assignment."""
        V = generate_reference_directions(2)
        F = np.random.rand(50, 2)
        ids = assign_regions(F, V)
        assert len(ids) == 50
        assert all(0 <= r < len(V) for r in ids)

    def test_t53_extreme_points_assigned_correctly(self):
        """Solutions near axis extremes get assigned to matching directions."""
        V = generate_reference_directions(2)
        F = np.array([[0.01, 1.0], [1.0, 0.01]])
        ids = assign_regions(F, V)
        assert ids[0] != ids[1]


# ── T54-T55: State computation ─────────────────────────────────

class TestStateComputation:

    def test_t54_severity_finite(self):
        """Severity values are finite for valid inputs."""
        V = generate_reference_directions(2)
        F_m = np.random.rand(50, 2)
        F_p = F_m + np.random.normal(0, 0.01, (50, 2))
        ids = assign_regions(F_p, V)
        S, C, Q = compute_severity(F_m, F_p, ids, len(V), 1.0)
        for r in range(len(V)):
            if np.any(ids == r):
                assert np.isfinite(C[r])
                assert 0.0 <= C[r] <= 1.0

    def test_t55_state_9dim(self):
        """State vector has exactly 9 dimensions."""
        V = generate_reference_directions(2)
        F_m = np.random.rand(50, 2)
        F_p = F_m + np.random.normal(0, 0.01, (50, 2))
        ids = assign_regions(F_p, V)
        _, C, _ = compute_severity(F_m, F_p, ids, len(V), 1.0)
        ql = compute_quality_loss(F_m, F_p, ids, V)
        states = build_state_9dim(C, ql, ids, len(V), 50, None)
        for r, s in states.items():
            assert s.shape == (9,)
            assert np.all(np.isfinite(s))


# ── T56-T57: Controller loading and inference ──────────────────

class TestControllerInference:

    def test_t56_c1_greedy_actions_valid(self):
        """C1 produces valid action indices (0-3) for all active regions."""
        ckpt = _find_checkpoint("C1_QMIX")
        ctrl = DRLController("C1", ckpt, n_obj=2)
        rng = np.random.default_rng(42)
        F_m = rng.uniform(0, 1, (100, 2))
        F_p = F_m + rng.normal(0, 0.01, (100, 2))
        F_p = np.abs(F_p)
        ids = assign_regions(F_p, ctrl.V)
        _, C, _ = compute_severity(F_m, F_p, ids, ctrl.R, 1.0)
        ql = compute_quality_loss(F_m, F_p, ids, ctrl.V)
        states = build_state_9dim(C, ql, ids, ctrl.R, 100, None)
        actions = select_actions_greedy("C1", ctrl.unary_net, None, states, ctrl.R, ctrl.device)
        for r, a in actions.items():
            assert 0 <= a <= 3, f"Invalid action {a} for region {r}"
            assert a in ACTION_NAMES

    def test_t57_c3_c4_use_coordination(self):
        """C3 and C4 use pair network (not just unary)."""
        c3_ckpt = _find_checkpoint("C3_PAIR")
        c4_ckpt = _find_checkpoint("C4_CAUSAL")
        ctrl_c3 = DRLController("C3", c3_ckpt, n_obj=2)
        ctrl_c4 = DRLController("C4", c4_ckpt, n_obj=2)
        assert ctrl_c3.pair_net is not None
        assert ctrl_c4.pair_net is not None


# ── T58-T60: Runner integration ────────────────────────────────

class TestDRLRunner:

    def test_t58_drl_30_measurements(self):
        """DRL runner produces exactly 30 IGD measurements."""
        ckpt = _find_checkpoint("C1_QMIX")
        r = run_single_experiment("DF1", "B-C1", seed=42, taut=10, checkpoint_path=ckpt)
        assert r["n_measurements"] == 30
        assert len(r["igd_values"]) == 30

    def test_t59_drl_migd_finite(self):
        """DRL MIGD is finite and positive."""
        ckpt = _find_checkpoint("C4_CAUSAL")
        r = run_single_experiment("DF1", "B-C4", seed=42, taut=10, checkpoint_path=ckpt)
        assert 0 < r["migd"] < float("inf")

    def test_t60_drl_taut30(self):
        """DRL works with taut=30."""
        ckpt = _find_checkpoint("C1_QMIX")
        r = run_single_experiment("DF1", "B-C1", seed=42, taut=30, checkpoint_path=ckpt)
        assert r["n_measurements"] == 30
        assert 0 < r["migd"] < float("inf")
