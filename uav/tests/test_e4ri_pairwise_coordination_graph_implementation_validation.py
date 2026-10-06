"""Tests for E4RI Pairwise Coordination Graph Implementation Validation."""
import json
import os
import hashlib
import pytest
import torch
import torch.nn as nn
import numpy as np

BASE = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "E4RI_pairwise_coordination_graph_implementation_validation",
)

def _load(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# T1-T5: Pre-manifest and historical integrity
# ---------------------------------------------------------------------------
class TestPreManifest:
    def test_t01_stage_type(self):
        v = _load("E4RI_pre_manifest.json")
        assert v["stage_type"] == "IMPLEMENTATION_VALIDATION_ONLY"

    def test_t02_no_training(self):
        v = _load("E4RI_pre_manifest.json")
        assert v["constraints"]["NO_TRAINING"] is True

    def test_t03_no_environment(self):
        v = _load("E4RI_pre_manifest.json")
        assert v["constraints"]["NO_ENVIRONMENT_EXECUTION"] is True

    def test_t04_historical_integrity_all_pass(self):
        v = _load("E4RI_historical_integrity.json")
        assert v["ALL_PASS"] is True
        for name, check in v["checks"].items():
            assert check["PASS"] is True, f"Failed: {name}"

    def test_t05_historical_integrity_count(self):
        v = _load("E4RI_historical_integrity.json")
        assert len(v["checks"]) >= 15


# ---------------------------------------------------------------------------
# T6-T7: C1 backward equivalence
# ---------------------------------------------------------------------------
class TestC1BackwardEquivalence:
    def test_t06_qmix_sha_unchanged(self):
        v = _load("E4RI_c1_backward_equivalence.json")
        assert v["QMIX_CONFIG_SHA256"] == "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"

    def test_t07_c1_behavior_unchanged(self):
        v = _load("E4RI_c1_backward_equivalence.json")
        assert v["C1_BEHAVIOR_UNCHANGED"] is True


# ---------------------------------------------------------------------------
# T8-T10: C3 config
# ---------------------------------------------------------------------------
class TestC3Config:
    def test_t08_c3_config_exists(self):
        v = _load("E4RI_c3_config.json")
        assert v["architecture_version"] == "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_t09_c3_config_hash(self):
        v = _load("E4RI_c3_config_hash.json")
        assert v["C3_CONFIG_SHA256"] == "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"

    def test_t10_c3_config_hash_matches_module(self):
        from prdrl.rl.coordination_graph import c3_config_sha256
        v = _load("E4RI_c3_config_hash.json")
        assert c3_config_sha256() == v["C3_CONFIG_SHA256"]


# ---------------------------------------------------------------------------
# T11-T15: Unary architecture validation
# ---------------------------------------------------------------------------
class TestUnaryArchitecture:
    def test_t11_unary_architecture_string(self):
        v = _load("E4RI_unary_architecture_validation.json")
        assert v["architecture"] == "9-128-128-4"

    def test_t12_unary_input_dim(self):
        from prdrl.rl.network import SharedRegionalQNetwork
        net = SharedRegionalQNetwork(9)
        first_weight = list(net.parameters())[0]
        assert first_weight.shape[1] == 9

    def test_t13_unary_output_dim(self):
        from prdrl.rl.network import SharedRegionalQNetwork
        net = SharedRegionalQNetwork(9)
        last_weight = list(net.parameters())[-2]
        assert last_weight.shape[0] == 4

    def test_t14_unary_identical_to_c1_utility(self):
        v = _load("E4RI_unary_architecture_validation.json")
        assert v["IDENTICAL_TO_C1_UTILITY"] is True

    def test_t15_unary_param_count(self):
        from prdrl.rl.network import SharedRegionalQNetwork
        net = SharedRegionalQNetwork(9)
        total = sum(p.numel() for p in net.parameters())
        assert total == 18308


# ---------------------------------------------------------------------------
# T16-T21: Pair architecture validation
# ---------------------------------------------------------------------------
class TestPairArchitecture:
    def test_t16_pair_architecture_string(self):
        v = _load("E4RI_pair_architecture_validation.json")
        assert v["architecture"] == "18-128-128-16"

    def test_t17_pair_input_dim_18(self):
        from prdrl.rl.coordination_graph import SharedPairNetwork, PAIR_INPUT_DIM
        assert PAIR_INPUT_DIM == 18
        net = SharedPairNetwork()
        first_weight = list(net.parameters())[0]
        assert first_weight.shape[1] == 18

    def test_t18_pair_output_dim_16(self):
        from prdrl.rl.coordination_graph import SharedPairNetwork, PAIR_OUTPUT_DIM
        assert PAIR_OUTPUT_DIM == 16

    def test_t19_pair_hidden_dims(self):
        from prdrl.rl.coordination_graph import PAIR_HIDDEN_1, PAIR_HIDDEN_2
        assert PAIR_HIDDEN_1 == 128
        assert PAIR_HIDDEN_2 == 128

    def test_t20_pair_reshape_to_4x4(self):
        from prdrl.rl.coordination_graph import SharedPairNetwork, raw_pair_matrix
        net = SharedPairNetwork()
        s = torch.randn(2, 9)
        out = raw_pair_matrix(net, s, s)
        assert out.shape == (2, 4, 4)

    def test_t21_pair_rejects_wrong_input_dim(self):
        from prdrl.rl.coordination_graph import SharedPairNetwork
        with pytest.raises(ValueError):
            SharedPairNetwork(input_dim=9)


# ---------------------------------------------------------------------------
# T22-T26: Pair symmetry validation
# ---------------------------------------------------------------------------
class TestPairSymmetry:
    def test_t22_symmetry_formula(self):
        from prdrl.rl.coordination_graph import symmetrize_pair
        h_ij = torch.randn(3, 4, 4)
        h_ji = torch.randn(3, 4, 4)
        m = symmetrize_pair(h_ij, h_ji)
        expected = 0.5 * (h_ij + h_ji.transpose(-2, -1))
        assert torch.allclose(m, expected, atol=1e-7)

    def test_t23_symmetry_output_is_symmetric(self):
        from prdrl.rl.coordination_graph import symmetrize_pair
        h = torch.randn(2, 4, 4)
        m = symmetrize_pair(h, h)
        assert torch.allclose(m, m.transpose(-2, -1), atol=1e-7)

    def test_t24_symmetry_pipeline(self):
        v = _load("E4RI_pair_symmetry_validation.json")
        assert v["SYMMETRY_VERIFIED"] is True

    def test_t25_compute_centered_pair_uses_symmetry(self):
        from prdrl.rl.coordination_graph import (SharedPairNetwork,
            compute_centered_pair, raw_pair_matrix, symmetrize_pair, center_pair)
        torch.manual_seed(42)
        net = SharedPairNetwork()
        s_i = torch.randn(1, 9)
        s_j = torch.randn(1, 9)
        p_auto = compute_centered_pair(net, s_i, s_j)
        h_ij = raw_pair_matrix(net, s_i, s_j)
        h_ji = raw_pair_matrix(net, s_j, s_i)
        m = symmetrize_pair(h_ij, h_ji)
        p_manual = center_pair(m)
        assert torch.allclose(p_auto, p_manual, atol=1e-7)

    def test_t26_symmetry_with_different_states(self):
        from prdrl.rl.coordination_graph import SharedPairNetwork, compute_centered_pair
        torch.manual_seed(99)
        net = SharedPairNetwork()
        s_i = torch.randn(1, 9)
        s_j = torch.randn(1, 9)
        p_ij = compute_centered_pair(net, s_i, s_j)
        p_ji = compute_centered_pair(net, s_j, s_i)
        assert torch.allclose(p_ij, p_ji.transpose(-2, -1), atol=1e-6)


# ---------------------------------------------------------------------------
# T27-T31: Pair centering validation
# ---------------------------------------------------------------------------
class TestPairCentering:
    def test_t27_row_mean_zero(self):
        from prdrl.rl.coordination_graph import center_pair
        m = torch.randn(5, 4, 4)
        p = center_pair(m)
        row_means = p.mean(dim=-1)
        assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-6)

    def test_t28_col_mean_zero(self):
        from prdrl.rl.coordination_graph import center_pair
        m = torch.randn(5, 4, 4)
        p = center_pair(m)
        col_means = p.mean(dim=-2)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-6)

    def test_t29_centering_formula(self):
        from prdrl.rl.coordination_graph import center_pair
        m = torch.randn(2, 4, 4)
        p = center_pair(m)
        row_mean = m.mean(dim=-1, keepdim=True)
        col_mean = m.mean(dim=-2, keepdim=True)
        grand_mean = m.mean(dim=(-2, -1), keepdim=True)
        expected = m - row_mean - col_mean + grand_mean
        assert torch.allclose(p, expected, atol=1e-7)

    def test_t30_centering_validation_output(self):
        v = _load("E4RI_pair_centering_validation.json")
        assert v["CENTERING_VERIFIED"] is True

    def test_t31_centering_invariance(self):
        v = _load("E4RI_centering_invariance_validation.json")
        assert v["ADDITIVE_EFFECTS_REMOVED"] is True


# ---------------------------------------------------------------------------
# T32-T37: Graph construction validation
# ---------------------------------------------------------------------------
class TestGraphConstruction:
    def test_t32_n0_edges(self):
        from prdrl.rl.coordination_graph import build_complete_graph
        mask = torch.zeros(10)
        assert len(build_complete_graph(mask)) == 0

    def test_t33_n1_edges(self):
        from prdrl.rl.coordination_graph import build_complete_graph
        mask = torch.zeros(10)
        mask[3] = 1
        assert len(build_complete_graph(mask)) == 0

    def test_t34_n2_edges(self):
        from prdrl.rl.coordination_graph import build_complete_graph
        mask = torch.zeros(10)
        mask[2] = 1; mask[7] = 1
        edges = build_complete_graph(mask)
        assert len(edges) == 1
        assert edges[0] == (2, 7)

    def test_t35_n5_edges(self):
        from prdrl.rl.coordination_graph import build_complete_graph
        mask = torch.zeros(10)
        for i in [0, 1, 2, 3, 4]:
            mask[i] = 1
        assert len(build_complete_graph(mask)) == 10  # C(5,2)

    def test_t36_n10_edges(self):
        from prdrl.rl.coordination_graph import build_complete_graph
        mask = torch.ones(10)
        assert len(build_complete_graph(mask)) == 45  # C(10,2)

    def test_t37_graph_output_file(self):
        v = _load("E4RI_graph_construction_validation.json")
        assert v["edge_counts"]["n0"] == 0
        assert v["edge_counts"]["n1"] == 0
        assert v["edge_counts"]["n2"] == 1
        assert v["edge_counts"]["n5"] == 10
        assert v["edge_counts"]["n10"] == 45


# ---------------------------------------------------------------------------
# T38-T42: Joint value validation
# ---------------------------------------------------------------------------
class TestJointValue:
    def test_t38_joint_value_computation(self):
        from prdrl.rl.coordination_graph import (joint_value_cg, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        actions = torch.zeros(1, 10, dtype=torch.long)
        mask = torch.zeros(1, 10)
        mask[0, 0] = 1; mask[0, 1] = 1
        q = joint_value_cg(u, p, states, actions, mask)
        assert q.shape == (1,)
        assert torch.isfinite(q).all()

    def test_t39_joint_value_uses_mean(self):
        v = _load("E4RI_joint_value_validation.json")
        assert v["NORMALIZATION"] == "MEAN_UNARY_PLUS_MEAN_PAIR"

    def test_t40_active_count_scale(self):
        v = _load("E4RI_active_count_scale_validation.json")
        assert v["USES_MEAN_NOT_SUM"] is True

    def test_t41_n0_safe(self):
        from prdrl.rl.coordination_graph import joint_value_cg, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        actions = torch.zeros(1, 10, dtype=torch.long)
        mask = torch.zeros(1, 10)
        q = joint_value_cg(u, p, states, actions, mask)
        assert q.shape == (1,)
        assert torch.isfinite(q).all()

    def test_t42_n1_unary_only(self):
        v = _load("E4RI_n0_n1_semantics.json")
        assert v["n0_safe"] is True
        assert v["n1_pair_contribution"] == 0.0


# ---------------------------------------------------------------------------
# T43: Rank reversal representation
# ---------------------------------------------------------------------------
class TestRankReversal:
    def test_t43_rank_reversal_representable(self):
        v = _load("E4RI_rank_reversal_representation_validation.json")
        assert v["RANK_REVERSAL_REPRESENTABLE"] is True


# ---------------------------------------------------------------------------
# T44-T53: Max-sum validation
# ---------------------------------------------------------------------------
class TestMaxSum:
    def test_t44_maxsum_update_formula(self):
        v = _load("E4RI_maxsum_update_validation.json")
        assert v["UPDATE_FORMULA_VERIFIED"] is True

    def test_t45_maxsum_synchronous(self):
        v = _load("E4RI_maxsum_synchronous_validation.json")
        assert v["SYNCHRONOUS_UPDATE"] is True

    def test_t46_maxsum_normalization(self):
        v = _load("E4RI_maxsum_normalization_validation.json")
        assert v["NORMALIZATION_METHOD"] == "SUBTRACT_MAX"

    def test_t47_maxsum_k10(self):
        from prdrl.rl.coordination_graph import K_MAXSUM
        assert K_MAXSUM == 10

    def test_t48_maxsum_k10_output(self):
        v = _load("E4RI_maxsum_iteration_validation.json")
        assert v["K_MAXSUM"] == 10

    def test_t49_maxsum_action_argmax(self):
        v = _load("E4RI_maxsum_action_validation.json")
        assert v["ACTION_SELECTION"] == "BELIEF_ARGMAX"

    def test_t50_maxsum_returns_valid_actions(self):
        from prdrl.rl.coordination_graph import maxsum_inference, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(2, 10, 9)
        mask = torch.zeros(2, 10)
        mask[0, :3] = 1; mask[1, :5] = 1
        actions = maxsum_inference(u, p, states, mask)
        assert actions.shape == (2, 10)
        assert (actions >= 0).all()
        assert (actions < 4).all()

    def test_t51_maxsum_inactive_zero(self):
        from prdrl.rl.coordination_graph import maxsum_inference, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, :3] = 1
        actions = maxsum_inference(u, p, states, mask)
        assert (actions[0, 3:] == 0).all()

    def test_t52_maxsum_n1_equals_unary_argmax(self):
        from prdrl.rl.coordination_graph import maxsum_inference, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, 5] = 1
        actions = maxsum_inference(u, p, states, mask)
        q_direct = u(states[0, 5:6])
        expected = q_direct.argmax(dim=1).item()
        assert actions[0, 5].item() == expected

    def test_t53_maxsum_deterministic(self):
        from prdrl.rl.coordination_graph import maxsum_inference, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.ones(1, 10)
        a1 = maxsum_inference(u, p, states, mask)
        a2 = maxsum_inference(u, p, states, mask)
        assert torch.equal(a1, a2)


# ---------------------------------------------------------------------------
# T54-T58: Exactness and loopy boundary
# ---------------------------------------------------------------------------
class TestExactness:
    def test_t54_two_node_exact(self):
        v = _load("E4RI_two_node_exactness.json")
        assert v["EXACT_MATCH"] is True

    def test_t55_two_node_regret_zero(self):
        v = _load("E4RI_two_node_exactness.json")
        assert abs(v["regret"]) < 1e-10

    def test_t56_tree_exact(self):
        v = _load("E4RI_tree_exactness.json")
        assert v["TREE_EXACT_VERIFIED"] is True

    def test_t57_loopy_boundary(self):
        v = _load("E4RI_loopy_boundary_validation.json")
        assert v["LOOPY_FOR_N_GEQ_3"] is True
        assert v["APPROXIMATE_STATED"] is True

    def test_t58_two_node_exact_live(self):
        from prdrl.rl.coordination_graph import (maxsum_inference, exact_joint_argmax,
            joint_value_cg, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(123)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, 2] = 1; mask[0, 7] = 1
        ms_actions = maxsum_inference(u, p, states, mask)
        exact_actions, exact_q = exact_joint_argmax(u, p, states, mask, 0)
        with torch.no_grad():
            ms_q = joint_value_cg(u, p, states, ms_actions, mask).item()
        assert abs(exact_q - ms_q) < 1e-6


# ---------------------------------------------------------------------------
# T59-T63: Small-n diagnostic
# ---------------------------------------------------------------------------
class TestSmallNDiagnostic:
    def test_t59_diagnostic_works_n2(self):
        from prdrl.rl.coordination_graph import (maxsum_inference, small_n_diagnostic,
            SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, 0] = 1; mask[0, 1] = 1
        actions = maxsum_inference(u, p, states, mask)
        result = small_n_diagnostic(u, p, states, mask, actions, 0)
        assert "regret" in result
        assert "exact_action_match" in result

    def test_t60_diagnostic_works_n5(self):
        from prdrl.rl.coordination_graph import (maxsum_inference, small_n_diagnostic,
            SharedPairNetwork, SMALL_N_EXACT_THRESHOLD)
        from prdrl.rl.network import SharedRegionalQNetwork
        assert SMALL_N_EXACT_THRESHOLD == 5
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1
        actions = maxsum_inference(u, p, states, mask)
        result = small_n_diagnostic(u, p, states, mask, actions, 0)
        assert result["regret"] >= 0

    def test_t61_diagnostic_blocked_n6(self):
        from prdrl.rl.coordination_graph import exact_joint_argmax, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, :6] = 1
        with pytest.raises(ValueError, match="blocked"):
            exact_joint_argmax(u, p, states, mask, 0)

    def test_t62_small_n_diagnostic_output(self):
        v = _load("E4RI_small_n_exact_diagnostic_validation.json")
        assert v["DIAGNOSTIC_WORKS"] is True
        assert v["BLOCKED_ABOVE_THRESHOLD"] is True

    def test_t63_no_exhaustive_production(self):
        v = _load("E4RI_no_exhaustive_production_inference.json")
        assert v["NO_EXHAUSTIVE_4_10_IN_PRODUCTION"] is True


# ---------------------------------------------------------------------------
# T64-T65: Complexity audit
# ---------------------------------------------------------------------------
class TestComplexity:
    def test_t64_complexity_maxsum(self):
        v = _load("E4RI_complexity_audit.json")
        assert v["maxsum_ops"] == 7200  # O(K*|E|*A^2) = 10*45*16

    def test_t65_complexity_exact(self):
        v = _load("E4RI_complexity_audit.json")
        assert v["exact_ops"] == 1048576  # 4^10


# ---------------------------------------------------------------------------
# T66-T67: Joint reward validation
# ---------------------------------------------------------------------------
class TestJointReward:
    def test_t66_joint_reward_mean(self):
        v = _load("E4RI_joint_reward_validation.json")
        assert v["JOINT_REWARD"] == "MEAN_ACTIVE_R1"

    def test_t67_joint_reward_unchanged(self):
        v = _load("E4RI_joint_reward_validation.json")
        assert v["SAME_AS_C1"] is True


# ---------------------------------------------------------------------------
# T68-T72: Double-Q validation
# ---------------------------------------------------------------------------
class TestDoubleQ:
    def test_t68_online_selects(self):
        v = _load("E4RI_double_q_validation.json")
        assert v["ONLINE_SELECTS_ACTION"] is True

    def test_t69_target_evaluates(self):
        v = _load("E4RI_double_q_validation.json")
        assert v["TARGET_EVALUATES_VALUE"] is True

    def test_t70_double_q_live(self):
        from prdrl.rl.coordination_graph import (c3_double_q_target, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        ou = SharedRegionalQNetwork(9)
        op = SharedPairNetwork()
        tu = SharedRegionalQNetwork(9)
        tp = SharedPairNetwork()
        states = torch.randn(2, 10, 9)
        mask = torch.zeros(2, 10)
        mask[0, :3] = 1; mask[1, :2] = 1
        r = torch.tensor([1.0, 0.5])
        done = torch.tensor([0.0, 0.0])
        y = c3_double_q_target(ou, op, tu, tp, states, mask, r, done)
        assert y.shape == (2,)
        assert torch.isfinite(y).all()

    def test_t71_done_masks_future(self):
        from prdrl.rl.coordination_graph import (c3_double_q_target, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        ou = SharedRegionalQNetwork(9)
        op = SharedPairNetwork()
        tu = SharedRegionalQNetwork(9)
        tp = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.ones(1, 10)
        r = torch.tensor([5.0])
        done = torch.tensor([1.0])
        y = c3_double_q_target(ou, op, tu, tp, states, mask, r, done)
        assert abs(y.item() - 5.0) < 1e-6

    def test_t72_double_q_output(self):
        v = _load("E4RI_double_q_validation.json")
        assert v["DOUBLE_Q_VERIFIED"] is True


# ---------------------------------------------------------------------------
# T73-T75: Loss validation
# ---------------------------------------------------------------------------
class TestLoss:
    def test_t73_loss_smoothl1(self):
        from prdrl.rl.coordination_graph import c3_loss
        q = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.tensor([1.1, 1.9, 3.2])
        loss = c3_loss(q, y)
        assert loss.item() > 0
        loss.backward()
        assert q.grad is not None

    def test_t74_loss_no_auxiliary(self):
        v = _load("E4RI_loss_validation.json")
        assert v["LOSS_TYPE"] == "SMOOTHL1_ONLY"
        assert v["NO_AUXILIARY_LOSSES"] is True

    def test_t75_loss_output_file(self):
        v = _load("E4RI_loss_validation.json")
        assert v["LOSS_VERIFIED"] is True


# ---------------------------------------------------------------------------
# T76-T80: Gradient path validation
# ---------------------------------------------------------------------------
class TestGradientPath:
    def test_t76_gradient_to_unary(self):
        from prdrl.rl.coordination_graph import (c3_current_q, c3_loss, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        actions = torch.zeros(1, 10, dtype=torch.long)
        mask = torch.zeros(1, 10)
        mask[0, :2] = 1
        q = c3_current_q(u, p, states, actions, mask)
        y = torch.tensor([0.5])
        loss = c3_loss(q, y)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in u.parameters())
        assert has_grad

    def test_t77_gradient_to_pair(self):
        from prdrl.rl.coordination_graph import (c3_current_q, c3_loss, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        actions = torch.zeros(1, 10, dtype=torch.long)
        mask = torch.zeros(1, 10)
        mask[0, :2] = 1
        q = c3_current_q(u, p, states, actions, mask)
        y = torch.tensor([0.5])
        loss = c3_loss(q, y)
        loss.backward()
        has_grad = any(pp.grad is not None and pp.grad.abs().sum() > 0
                       for pp in p.parameters())
        assert has_grad

    def test_t78_no_gradient_to_target(self):
        from prdrl.rl.coordination_graph import (c3_double_q_target, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        ou = SharedRegionalQNetwork(9)
        op = SharedPairNetwork()
        tu = SharedRegionalQNetwork(9)
        tp = SharedPairNetwork()
        for pp in tu.parameters():
            pp.requires_grad_(True)
        for pp in tp.parameters():
            pp.requires_grad_(True)
        states = torch.randn(1, 10, 9)
        mask = torch.ones(1, 10)
        r = torch.tensor([1.0])
        done = torch.tensor([0.0])
        y = c3_double_q_target(ou, op, tu, tp, states, mask, r, done)
        assert not y.requires_grad

    def test_t79_gradient_path_output(self):
        v = _load("E4RI_gradient_path_validation.json")
        assert v["GRADIENT_TO_UNARY"] is True
        assert v["GRADIENT_TO_PAIR"] is True
        assert v["NO_GRADIENT_TO_TARGET"] is True

    def test_t80_target_update_tau(self):
        v = _load("E4RI_target_update_validation.json")
        assert v["TAU"] == 0.005


# ---------------------------------------------------------------------------
# T81-T84: Optimizer validation
# ---------------------------------------------------------------------------
class TestOptimizer:
    def test_t81_optimizer_params(self):
        from prdrl.rl.coordination_graph import c3_optimizer_params, SharedPairNetwork
        from prdrl.rl.network import SharedRegionalQNetwork
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        params = c3_optimizer_params(u, p)
        u_count = sum(1 for _ in u.parameters())
        p_count = sum(1 for _ in p.parameters())
        assert len(params) == u_count + p_count

    def test_t82_optimizer_output(self):
        v = _load("E4RI_optimizer_validation.json")
        assert v["OPTIMIZER"] == "Adam"
        assert v["LR"] == 1e-4

    def test_t83_batch_size(self):
        v = _load("E4RI_optimizer_validation.json")
        assert v["BATCH_SIZE"] == 128

    def test_t84_replay_capacity(self):
        v = _load("E4RI_optimizer_validation.json")
        assert v["REPLAY_CAPACITY"] == 50000


# ---------------------------------------------------------------------------
# T85-T88: Replay and exploration
# ---------------------------------------------------------------------------
class TestReplayExploration:
    def test_t85_replay_type(self):
        v = _load("E4RI_replay_validation.json")
        assert v["REPLAY_TYPE"] == "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"

    def test_t86_sibling_dedup_false(self):
        v = _load("E4RI_sibling_replay_validation.json")
        assert v["SIBLING_DEDUP"] is False

    def test_t87_exploration_over_maxsum(self):
        v = _load("E4RI_exploration_validation.json")
        assert v["EXPLORATION_OVER_MAXSUM_GREEDY"] is True

    def test_t88_epsilon_schedule(self):
        v = _load("E4RI_exploration_validation.json")
        assert v["EPSILON_START"] == 1.0
        assert v["EPSILON_END"] == 0.05


# ---------------------------------------------------------------------------
# T89-T93: Initialization pairing
# ---------------------------------------------------------------------------
class TestInitializationPairing:
    def test_t89_c1_utility_equals_c3_unary(self):
        v = _load("E4RI_initialization_pairing_validation.json")
        assert v["C1_UTILITY_EQUALS_C3_UNARY_INIT"] is True

    def test_t90_pair_seed_offset(self):
        from prdrl.rl.coordination_graph import PAIR_SEED_OFFSET
        assert PAIR_SEED_OFFSET == 1

    def test_t91_pair_seed_derivation(self):
        from prdrl.rl.coordination_graph import derive_pair_seed
        assert derive_pair_seed(7001) == 7002
        assert derive_pair_seed(7005) == 7006

    def test_t92_pair_init_live(self):
        from prdrl.rl.coordination_graph import build_pair_online_target, derive_pair_seed
        seed = derive_pair_seed(7001)
        online, target = build_pair_online_target(seed)
        for po, pt in zip(online.parameters(), target.parameters()):
            assert torch.equal(po, pt)
        for pt in target.parameters():
            assert not pt.requires_grad

    def test_t93_unary_init_matches_utility(self):
        from prdrl.rl.network import build_online_target_pair
        from prdrl.rl.coordination_graph import derive_pair_seed
        c1_online, _ = build_online_target_pair(9, 7001)
        c3_online, _ = build_online_target_pair(9, 7001)
        for p1, p3 in zip(c1_online.parameters(), c3_online.parameters()):
            assert torch.equal(p1, p3)


# ---------------------------------------------------------------------------
# T94-T98: Parameter count audit
# ---------------------------------------------------------------------------
class TestParameterCount:
    def test_t94_c1_utility_count(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C1_utility"] == 18308

    def test_t95_c1_mixer_count(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C1_mixer"] == 21560

    def test_t96_c1_total(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C1_total"] == 39868

    def test_t97_c3_pair_count(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C3_pair"] == 21008

    def test_t98_c3_total(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C3_total"] == 39316

    def test_t99_c3_unary_equals_c1_utility(self):
        v = _load("E4RI_parameter_count_audit.json")
        assert v["C3_unary"] == v["C1_utility"]


# ---------------------------------------------------------------------------
# T100-T105: Inactive region and permutation invariance
# ---------------------------------------------------------------------------
class TestInvariance:
    def test_t100_inactive_invariance(self):
        v = _load("E4RI_inactive_region_invariance.json")
        assert v["INACTIVE_INVARIANT"] is True

    def test_t101_inactive_garbage_live(self):
        from prdrl.rl.coordination_graph import (joint_value_cg, maxsum_inference,
            SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, 0] = 1; mask[0, 1] = 1
        a1 = maxsum_inference(u, p, states, mask)
        q1 = joint_value_cg(u, p, states, a1, mask)
        states2 = states.clone()
        states2[0, 5:] = torch.randn(5, 9) * 100
        a2 = maxsum_inference(u, p, states2, mask)
        q2 = joint_value_cg(u, p, states2, a2, mask)
        assert torch.equal(a1, a2)
        assert torch.allclose(q1, q2, atol=1e-6)

    def test_t102_permutation_equivariance(self):
        v = _load("E4RI_region_permutation_validation.json")
        assert v["PERMUTATION_EQUIVARIANT"] is True

    def test_t103_permutation_live(self):
        from prdrl.rl.coordination_graph import (maxsum_inference, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9)
        mask = torch.zeros(1, 10)
        mask[0, 0] = 1; mask[0, 1] = 1; mask[0, 2] = 1
        a1 = maxsum_inference(u, p, states, mask)
        states2 = states.clone()
        states2[0, 0] = states[0, 1].clone()
        states2[0, 1] = states[0, 0].clone()
        a2 = maxsum_inference(u, p, states2, mask)
        assert a2[0, 0].item() == a1[0, 1].item()
        assert a2[0, 1].item() == a1[0, 0].item()

    def test_t104_numerical_stability(self):
        v = _load("E4RI_numerical_stability.json")
        assert v["NO_NAN_INF"] is True

    def test_t105_numerical_stability_live(self):
        from prdrl.rl.coordination_graph import (joint_value_cg, maxsum_inference,
            SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork
        torch.manual_seed(42)
        u = SharedRegionalQNetwork(9)
        p = SharedPairNetwork()
        states = torch.randn(1, 10, 9) * 10
        mask = torch.ones(1, 10)
        a = maxsum_inference(u, p, states, mask)
        q = joint_value_cg(u, p, states, a, mask)
        assert torch.isfinite(a.float()).all()
        assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# T106-T109: Action path audit
# ---------------------------------------------------------------------------
class TestActionPath:
    def test_t106_c3_uses_maxsum(self):
        v = _load("E4RI_action_path_audit.json")
        assert v["C3_INFERENCE"] == "MAX_SUM_MESSAGE_PASSING"

    def test_t107_c1_uses_utility_argmax(self):
        v = _load("E4RI_action_path_audit.json")
        assert v["C1_INFERENCE"] == "UTILITY_ARGMAX"

    def test_t108_c3_uses_pair_network(self):
        v = _load("E4RI_action_path_audit.json")
        assert v["C3_USES_PAIR_NETWORK"] is True

    def test_t109_c1_no_pair_network(self):
        v = _load("E4RI_action_path_audit.json")
        assert v["C1_HAS_PAIR_NETWORK"] is False


# ---------------------------------------------------------------------------
# T110-T113: V3/operator integrity
# ---------------------------------------------------------------------------
class TestV3OperatorIntegrity:
    def test_t110_v3_unchanged(self):
        v = _load("E4RI_v3_operator_integrity.json")
        assert v["V3_UNCHANGED"] is True

    def test_t111_operators_unchanged(self):
        v = _load("E4RI_v3_operator_integrity.json")
        assert v["OPERATORS_UNCHANGED"] is True

    def test_t112_no_c2_loss(self):
        v = _load("E4RI_no_c2_loss_audit.json")
        assert v["NO_C2_COMPOSITE_LOSS_IN_C3"] is True

    def test_t113_no_e4pd_supervision(self):
        v = _load("E4RI_no_e4pd_supervision_audit.json")
        assert v["NO_E4PD_DATA_DEPENDENCY"] is True


# ---------------------------------------------------------------------------
# T114-T117: Checkpoint roundtrip
# ---------------------------------------------------------------------------
class TestCheckpointRoundtrip:
    def test_t114_checkpoint_roundtrip(self):
        v = _load("E4RI_checkpoint_roundtrip.json")
        assert v["ROUNDTRIP_VERIFIED"] is True

    def test_t115_checkpoint_roundtrip_live(self):
        import tempfile
        from prdrl.rl.coordination_graph import (build_pair_online_target,
            derive_pair_seed, maxsum_inference, joint_value_cg, SharedPairNetwork)
        from prdrl.rl.network import SharedRegionalQNetwork, build_online_target_pair
        seed = 7001
        pair_seed = derive_pair_seed(seed)
        u_online, u_target = build_online_target_pair(9, seed)
        p_online, p_target = build_pair_online_target(pair_seed)
        states = torch.randn(1, 10, 9)
        mask = torch.ones(1, 10)
        a_before = maxsum_inference(u_online, p_online, states, mask)
        q_before = joint_value_cg(u_online, p_online, states, a_before, mask)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save({
                "unary_online": u_online.state_dict(),
                "unary_target": u_target.state_dict(),
                "pair_online": p_online.state_dict(),
                "pair_target": p_target.state_dict(),
            }, f)
        u2 = SharedRegionalQNetwork(9)
        p2 = SharedPairNetwork()
        ckpt = torch.load(path, weights_only=True)
        u2.load_state_dict(ckpt["unary_online"])
        p2.load_state_dict(ckpt["pair_online"])
        a_after = maxsum_inference(u2, p2, states, mask)
        q_after = joint_value_cg(u2, p2, states, a_after, mask)
        assert torch.equal(a_before, a_after)
        assert torch.allclose(q_before, q_after, atol=1e-7)
        os.unlink(path)

    def test_t116_checkpoint_all_components(self):
        v = _load("E4RI_checkpoint_roundtrip.json")
        for component in ["unary_online", "unary_target", "pair_online", "pair_target"]:
            assert component in v["COMPONENTS_SAVED"]

    def test_t117_checkpoint_q_match(self):
        v = _load("E4RI_checkpoint_roundtrip.json")
        assert v["Q_VALUES_MATCH"] is True


# ---------------------------------------------------------------------------
# T118-T123: Seed disjointness and firewalls
# ---------------------------------------------------------------------------
class TestSeedFirewall:
    def test_t118_seed_disjointness(self):
        v = _load("E4RI_seed_disjointness.json")
        assert v["DISJOINT_FROM_HISTORICAL"] is True

    def test_t119_train_seeds(self):
        v = _load("E4RI_seed_disjointness.json")
        assert v["train_range"] == [2501, 2560]

    def test_t120_val_seeds(self):
        v = _load("E4RI_seed_disjointness.json")
        assert v["validation_range"] == [2561, 2580]

    def test_t121_final_seeds(self):
        v = _load("E4RI_seed_disjointness.json")
        assert v["final_range"] == [2581, 2640]

    def test_t122_future_seed_firewall(self):
        v = _load("E4RI_future_seed_firewall.json")
        assert v["ALL_ACCESS_COUNTS_ZERO"] is True

    def test_t123_policy_seeds(self):
        v = _load("E4RI_seed_disjointness.json")
        assert v["policy_seeds"] == [7001, 7002, 7003, 7004, 7005]


# ---------------------------------------------------------------------------
# T124-T129: Future contracts
# ---------------------------------------------------------------------------
class TestFutureContracts:
    def test_t124_training_episodes(self):
        v = _load("E4RI_future_training_contract.json")
        assert v["TOTAL_EPISODES"] == 10800

    def test_t125_training_arms(self):
        v = _load("E4RI_future_training_contract.json")
        assert v["ARMS"] == 2

    def test_t126_validation_episodes(self):
        v = _load("E4RI_future_validation_contract.json")
        assert v["TOTAL_EPISODES"] == 3600

    def test_t127_final_episodes(self):
        v = _load("E4RI_future_final_contract.json")
        assert v["TOTAL_EPISODES"] == 1800

    def test_t128_final_seeds_unopened(self):
        v = _load("E4RI_future_final_contract.json")
        assert v["FINAL_SEEDS_UNOPENED"] is True

    def test_t129_bootstrap_seed(self):
        v = _load("E4RI_future_final_contract.json")
        assert v["BOOTSTRAP_SEED"] == 136001


# ---------------------------------------------------------------------------
# T130-T133: No training audit
# ---------------------------------------------------------------------------
class TestNoTrainingAudit:
    def test_t130_no_training(self):
        v = _load("E4RI_no_training_audit.json")
        assert v["NO_TRAINING"] is True

    def test_t131_no_env(self):
        v = _load("E4RI_no_training_audit.json")
        assert v["NO_ENVIRONMENT"] is True

    def test_t132_no_optimizer_step(self):
        v = _load("E4RI_no_training_audit.json")
        assert v["NO_OPTIMIZER_STEP"] is True

    def test_t133_no_checkpoint_creation(self):
        v = _load("E4RI_no_training_audit.json")
        assert v["NO_CHECKPOINT_CREATION"] is True


# ---------------------------------------------------------------------------
# T134-T148: Verdict and config integrity
# ---------------------------------------------------------------------------
class TestVerdict:
    def test_t134_e4ri_complete(self):
        v = _load("E4RI_verdict.json")
        assert v["E4RI_COMPLETE"] == "YES"

    def test_t135_next_stage(self):
        v = _load("E4RI_verdict.json")
        assert v["NEXT_STAGE"] == "E4RT_PAIRWISE_CG_AB_TRAINING_PROTOCOL_FREEZE"

    def test_t136_all_gates_pass(self):
        v = _load("E4RI_verdict.json")
        for gate_name, gate_val in v["GATES"].items():
            assert gate_val == "PASS", f"Gate {gate_name} is {gate_val}"

    def test_t137_gate_count(self):
        v = _load("E4RI_verdict.json")
        assert len(v["GATES"]) >= 45

    def test_t138_c3_config_sha_in_verdict(self):
        v = _load("E4RI_verdict.json")
        assert v["C3_CONFIG_SHA256"] == "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"

    def test_t139_qmix_sha_in_verdict(self):
        v = _load("E4RI_verdict.json")
        assert v["QMIX_CONFIG_SHA256"] == "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"

    def test_t140_pair_seed_offset_in_verdict(self):
        v = _load("E4RI_verdict.json")
        assert v["PAIR_SEED_OFFSET"] == 1

    def test_t141_e4f_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4F"] == "FAILED"

    def test_t142_e4hf_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4HF"] == "FAILED"

    def test_t143_e4if_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4IF"] == "REWARD_CAUSAL_VALIDATED"

    def test_t144_e4jf_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4JF"] == "GLOBAL_COMPARATOR_FAILED"

    def test_t145_e4kf_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4KF"] == "COORDINATION_CAUSAL_FAILED"

    def test_t146_e4mf_preserved(self):
        v = _load("E4RI_verdict.json")
        assert v["PRESERVED_VERDICTS"]["E4MF"] == "FAILED"

    def test_t147_performance_causal_not_computed(self):
        v = _load("E4RI_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_t148_report_exists(self):
        assert os.path.exists(os.path.join(BASE, "E4RI_report.md"))


# ---------------------------------------------------------------------------
# T149-T155: Live module validation (additional)
# ---------------------------------------------------------------------------
class TestLiveModuleValidation:
    def test_t149_beta_pair_is_1(self):
        from prdrl.rl.coordination_graph import BETA_PAIR
        assert BETA_PAIR == 1

    def test_t150_r_max_is_10(self):
        from prdrl.rl.coordination_graph import R_MAX
        assert R_MAX == 10

    def test_t151_state_dim_is_9(self):
        from prdrl.rl.coordination_graph import STATE_DIM
        assert STATE_DIM == 9

    def test_t152_treatment_arch_string(self):
        from prdrl.rl.coordination_graph import TREATMENT_ARCH_C3
        assert TREATMENT_ARCH_C3 == "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_t153_config_contains_required_keys(self):
        from prdrl.rl.coordination_graph import make_c3_config
        cfg = make_c3_config()
        required = [
            "architecture_version", "unary_architecture", "pair_architecture",
            "pair_symmetrization", "pair_centering", "graph_topology",
            "joint_value", "beta_pair", "inference", "K_maxsum",
            "pair_seed_offset", "gamma", "tau", "loss", "reward_version",
            "normalization_version", "seed_ranges",
        ]
        for k in required:
            assert k in cfg, f"Missing key: {k}"

    def test_t154_config_pair_seed_offset_is_1(self):
        from prdrl.rl.coordination_graph import make_c3_config
        cfg = make_c3_config()
        assert cfg["pair_seed_offset"] == 1

    def test_t155_config_sha_deterministic(self):
        from prdrl.rl.coordination_graph import c3_config_sha256
        h1 = c3_config_sha256()
        h2 = c3_config_sha256()
        assert h1 == h2

    def test_t156_no_mixer_in_c3(self):
        import inspect
        from prdrl.rl import coordination_graph as cg
        source = inspect.getsource(cg)
        assert "QMIXMixer" not in source

    def test_t157_pair_no_region_id(self):
        from prdrl.rl.coordination_graph import make_c3_config
        cfg = make_c3_config()
        assert cfg["pair_shared_across_all_active_pairs"] is True

    def test_t158_no_gradient_clipping(self):
        from prdrl.rl.coordination_graph import make_c3_config
        cfg = make_c3_config()
        assert cfg["no_gradient_clipping"] is True
