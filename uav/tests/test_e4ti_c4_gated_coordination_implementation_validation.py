"""
Tests for E4TI: C4 Gated Coordination Implementation Validation.

Validates all 18 E4TI output artifacts and C4 implementation fidelity.
No training, no environment execution, no seed access.
"""

import json
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                           "E4TI_c4_gated_coordination_implementation_validation")
E4T_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "E4T_causal_dynamics_gated_coordination_design")


def _load(name):
    path = os.path.join(RESULTS_DIR, f"E4TI_{name}.json")
    with open(path) as f:
        return json.load(f)


# ===================================================================
# Output artifact existence
# ===================================================================

class TestOutputArtifacts:
    EXPECTED_FILES = [
        "E4TI_v1_structural_equivalence.json",
        "E4TI_v2_gate_signal_provenance.json",
        "E4TI_v3_stationary_gate_zeroing.json",
        "E4TI_v4_dynamic_gate_nonzero.json",
        "E4TI_v5_mixed_gate.json",
        "E4TI_v6_numerical_boundary.json",
        "E4TI_v7_symmetry.json",
        "E4TI_v8_gradient_audit.json",
        "E4TI_v9_zero_gate_unary_reduction.json",
        "E4TI_v10_unit_gate_c3_equivalence.json",
        "E4TI_v11_identical_weight_forward.json",
        "E4TI_v12_parameter_inventory.json",
        "E4TI_v13_checkpoint_roundtrip.json",
        "E4TI_v14_seed_firewall.json",
        "E4TI_v15_no_training_firewall.json",
        "E4TI_v16_regression.json",
        "E4TI_v17_statistical_protocol_audit.json",
        "E4TI_v18_episode_count_accounting.json",
        "E4TI_verdict.json",
        "E4TI_report.md",
    ]

    @pytest.mark.parametrize("fname", EXPECTED_FILES)
    def test_artifact_exists(self, fname):
        assert os.path.isfile(os.path.join(RESULTS_DIR, fname)), f"Missing: {fname}"

    def test_artifact_count(self):
        json_files = [f for f in os.listdir(RESULTS_DIR)
                      if f.startswith("E4TI_") and f.endswith(".json")]
        md_files = [f for f in os.listdir(RESULTS_DIR)
                    if f.startswith("E4TI_") and f.endswith(".md")]
        assert len(json_files) + len(md_files) == 20


# ===================================================================
# V1: Structural Equivalence
# ===================================================================

class TestV1StructuralEquivalence:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v1_structural_equivalence")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_unary_input_dim(self, data):
        assert data["checks"]["unary_input_dim"] is True

    def test_pair_input_dim(self, data):
        assert data["checks"]["pair_input_dim"] is True

    def test_gate_signal_dim(self, data):
        assert data["checks"]["gate_signal_dim"] is True

    def test_k_maxsum(self, data):
        assert data["checks"]["k_maxsum"] is True


# ===================================================================
# V2: Gate Signal Provenance
# ===================================================================

class TestV2GateSignalProvenance:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v2_gate_signal_provenance")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_dim_matches(self, data):
        assert data["checks"]["gate_signal_dim_matches_delta_dim"] is True

    def test_formula_correct(self, data):
        assert data["checks"]["gate_formula_correct"] is True

    def test_no_trainable_params(self, data):
        assert data["checks"]["no_trainable_parameters"] is True


# ===================================================================
# V3: Stationary Gate Zeroing
# ===================================================================

class TestV3StationaryGateZeroing:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v3_stationary_gate_zeroing")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_all_zero(self, data):
        assert data["checks"]["all_gate_values_zero"] is True

    def test_q_unary_only(self, data):
        assert data["checks"]["q_c4_equals_unary_only"] is True


# ===================================================================
# V4: Dynamic Gate Nonzero
# ===================================================================

class TestV4DynamicGateNonzero:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v4_dynamic_gate_nonzero")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_all_positive(self, data):
        assert data["checks"]["all_positive"] is True

    def test_min_positive(self, data):
        assert data["checks"]["min"] > 0


# ===================================================================
# V5: Mixed Gate
# ===================================================================

class TestV5MixedGate:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v5_mixed_gate")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_has_both(self, data):
        assert data["checks"]["has_both"] is True

    def test_value_correct(self, data):
        assert data["checks"]["nonzero_value_correct"] is True


# ===================================================================
# V6: Numerical Boundary
# ===================================================================

class TestV6NumericalBoundary:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v6_numerical_boundary")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_in_bounds(self, data):
        assert data["checks"]["in_bounds_0_1"] is True

    def test_all_values_correct(self, data):
        assert data["checks"]["all_values_correct"] is True


# ===================================================================
# V7: Symmetry
# ===================================================================

class TestV7Symmetry:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v7_symmetry")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_symmetric(self, data):
        assert data["checks"]["symmetric"] is True


# ===================================================================
# V8: Gradient Audit
# ===================================================================

class TestV8GradientAudit:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v8_gradient_audit")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_pair_grads(self, data):
        assert data["checks"]["pair_network_receives_gradients"] is True

    def test_unary_grads(self, data):
        assert data["checks"]["unary_network_receives_gradients"] is True

    def test_param_count_match(self, data):
        assert data["checks"]["c4_param_count_equals_c3"] is True

    def test_param_numel_match(self, data):
        assert data["checks"]["c4_param_numel_equals_c3"] is True


# ===================================================================
# V9: Zero-Gate Unary Reduction
# ===================================================================

class TestV9ZeroGateUnaryReduction:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v9_zero_gate_unary_reduction")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_reduces(self, data):
        assert data["checks"]["reduces_to_unary"] is True


# ===================================================================
# V10: Unit-Gate C3 Equivalence
# ===================================================================

class TestV10UnitGateC3Equivalence:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v10_unit_gate_c3_equivalence")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_value_match(self, data):
        assert data["checks"]["joint_value_match"] is True

    def test_action_match(self, data):
        assert data["checks"]["maxsum_action_match"] is True


# ===================================================================
# V11: Identical-Weight Forward
# ===================================================================

class TestV11IdenticalWeightForward:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v11_identical_weight_forward")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_scalar_value_match(self, data):
        assert data["checks"]["scalar_c4_c3_value_match_at_g1"] is True

    def test_batched_value_match(self, data):
        assert data["checks"]["batched_c4_c3_value_match_at_g1"] is True

    def test_scalar_batched_c4_match(self, data):
        assert data["checks"]["scalar_batched_c4_value_match"] is True

    def test_scalar_action_match(self, data):
        assert data["checks"]["scalar_c4_c3_action_match_at_g1"] is True

    def test_batched_action_match(self, data):
        assert data["checks"]["batched_c4_c3_action_match_at_g1"] is True

    def test_scalar_batched_c4_action_match(self, data):
        assert data["checks"]["scalar_batched_c4_action_match"] is True


# ===================================================================
# V12: Parameter Inventory
# ===================================================================

class TestV12ParameterInventory:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v12_parameter_inventory")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_c4_total(self, data):
        assert data["checks"]["c4_total"] == 39316

    def test_unary(self, data):
        assert data["checks"]["unary_params"] == 18308

    def test_pair(self, data):
        assert data["checks"]["pair_params"] == 21008

    def test_gate(self, data):
        assert data["checks"]["gate_params"] == 0


# ===================================================================
# V13: Checkpoint Roundtrip
# ===================================================================

class TestV13CheckpointRoundtrip:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v13_checkpoint_roundtrip")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_forward_match(self, data):
        assert data["checks"]["forward_match"] is True

    def test_unary_sd_match(self, data):
        assert data["checks"]["unary_state_dict_match"] is True

    def test_pair_sd_match(self, data):
        assert data["checks"]["pair_state_dict_match"] is True


# ===================================================================
# V14: Seed Firewall
# ===================================================================

class TestV14SeedFirewall:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v14_seed_firewall")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_zero_access(self, data):
        assert data["checks"]["new_seed_access_count"] == 0


# ===================================================================
# V15: No-Training Firewall
# ===================================================================

class TestV15NoTrainingFirewall:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v15_no_training_firewall")

    def test_pass(self, data):
        assert data["PASS"] is True


# ===================================================================
# V16: Regression
# ===================================================================

class TestV16Regression:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v16_regression")

    def test_status(self, data):
        assert data["status"] == "PASS"
        assert data["PASS"] is True
        assert data["total_tests"] == 6114
        assert data["failed"] == 0


# ===================================================================
# V17: Statistical Protocol Audit
# ===================================================================

class TestV17StatisticalProtocolAudit:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v17_statistical_protocol_audit")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_three_arm(self, data):
        assert data["checks"]["three_arm_preserved"] is True

    def test_gating_direction(self, data):
        assert data["checks"]["gating_effect_direction"] is True

    def test_final_direction(self, data):
        assert data["checks"]["final_effect_direction"] is True

    def test_seed_ranges(self, data):
        assert data["checks"]["seed_ranges_match_e4t"] is True

    def test_gate_no_params(self, data):
        assert data["checks"]["c4_gate_trainable_parameters"] == 0


# ===================================================================
# V18: Episode Count Accounting
# ===================================================================

class TestV18EpisodeCountAccounting:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("v18_episode_count_accounting")

    def test_pass(self, data):
        assert data["PASS"] is True

    def test_episodes_match(self, data):
        assert data["checks"]["episodes_match_e4t_design"] is True

    def test_train_16200(self, data):
        assert data["checks"]["planned_train_episodes"] == 16200

    def test_final_2700(self, data):
        assert data["checks"]["planned_final_episodes"] == 2700


# ===================================================================
# Verdict
# ===================================================================

class TestVerdict:
    @pytest.fixture(scope="class")
    def data(self):
        return _load("verdict")

    def test_verdict_valid(self, data):
        assert data["verdict"] == "C4_IMPLEMENTATION_VALID"

    def test_no_failures(self, data):
        assert data["failed"] == 0

    def test_passed_count(self, data):
        assert data["passed"] >= 17

    def test_c4_sha256_set(self, data):
        assert data["C4_SHA256"] is not None
        assert len(data["C4_SHA256"]) == 64

    def test_next_stage(self, data):
        assert data["NEXT_STAGE"] == "E4TT_C4_GATED_COORDINATION_TRAINING_PROTOCOL_FREEZE"


# ===================================================================
# Direct C4 implementation tests
# ===================================================================

class TestC4GateComputation:
    def test_gate_zero_both_zero(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        s_i = torch.zeros(1, 9)
        s_j = torch.zeros(1, 9)
        g = compute_pair_gate(s_i, s_j)
        assert g.item() == 0.0

    def test_gate_both_max(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        s_i = torch.zeros(1, 9)
        s_j = torch.zeros(1, 9)
        s_i[0, 3] = 1.0
        s_j[0, 3] = -1.0
        g = compute_pair_gate(s_i, s_j)
        assert abs(g.item() - 1.0) < 1e-7

    def test_gate_one_zero(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        s_i = torch.zeros(1, 9)
        s_j = torch.zeros(1, 9)
        s_i[0, 3] = 0.6
        g = compute_pair_gate(s_i, s_j)
        assert abs(g.item() - 0.3) < 1e-7

    def test_gate_symmetric(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        torch.manual_seed(100)
        s_i = torch.randn(10, 9)
        s_j = torch.randn(10, 9)
        g_ij = compute_pair_gate(s_i, s_j)
        g_ji = compute_pair_gate(s_j, s_i)
        assert torch.allclose(g_ij, g_ji)

    def test_gate_batch(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        s_i = torch.zeros(5, 3, 9)
        s_j = torch.zeros(5, 3, 9)
        s_i[:, :, 3] = 0.4
        s_j[:, :, 3] = 0.2
        g = compute_pair_gate(s_i, s_j)
        assert g.shape == (5, 3)
        assert torch.allclose(g, torch.tensor(0.3).expand(5, 3))


class TestC4Constants:
    def test_treatment_arch(self):
        from prdrl.rl.coordination_graph import TREATMENT_ARCH_C4
        assert TREATMENT_ARCH_C4 == "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1"

    def test_gate_signal_dim(self):
        from prdrl.rl.coordination_graph import GATE_SIGNAL_DIM
        from prdrl.rl.normalization import DELTA_DIM
        assert GATE_SIGNAL_DIM == 3
        assert GATE_SIGNAL_DIM == DELTA_DIM

    def test_c4_config_has_gate_fields(self):
        from prdrl.rl.coordination_graph import make_c4_config
        cfg = make_c4_config()
        assert cfg["gate_signal"] == "ABS_V3_DELTA_SEVERITY"
        assert cfg["gate_trainable_parameters"] == 0
        assert cfg["gate_formula"] == "0.5*(ABS_DELTA_SEVERITY_I+ABS_DELTA_SEVERITY_J)"
        assert cfg["identical_to_c3_except"] == "deterministic_causal_pair_gate"

    def test_c4_config_sha256_deterministic(self):
        from prdrl.rl.coordination_graph import c4_config_sha256
        sha1 = c4_config_sha256()
        sha2 = c4_config_sha256()
        assert sha1 == sha2
        assert len(sha1) == 64


class TestC4ScalarBatchedEquivalence:
    @pytest.fixture(scope="class")
    def setup(self):
        from prdrl.rl.coordination_graph import (
            joint_value_c4, joint_value_c4_batched,
            maxsum_inference_c4, maxsum_inference_c4_batched,
            STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        torch.manual_seed(200)
        seed = 99990
        online_u, _ = build_online_target_pair(STATE_DIM, seed)
        online_p, _ = build_pair_online_target(derive_pair_seed(seed))

        B = 3
        states = torch.randn(B, R_MAX, STATE_DIM)
        states[:, :, 3] = torch.randn(B, R_MAX) * 0.5
        mask = torch.zeros(B, R_MAX)
        mask[0, :2] = 1.0
        mask[1, :4] = 1.0
        mask[2, :3] = 1.0
        actions = torch.randint(0, N_ACTIONS, (B, R_MAX))

        with torch.no_grad():
            q_s = joint_value_c4(online_u, online_p, states, actions, mask)
            q_b = joint_value_c4_batched(online_u, online_p, states, actions, mask)
            a_s = maxsum_inference_c4(online_u, online_p, states, mask)
            a_b = maxsum_inference_c4_batched(online_u, online_p, states, mask)

        return q_s, q_b, a_s, a_b

    def test_value_equivalence(self, setup):
        q_s, q_b, _, _ = setup
        assert torch.allclose(q_s, q_b, atol=1e-5)

    def test_action_equivalence(self, setup):
        _, _, a_s, a_b = setup
        assert torch.equal(a_s, a_b)


class TestC4VsC3Divergence:
    def test_c4_c3_differ_with_nonunit_gate(self):
        from prdrl.rl.coordination_graph import (
            joint_value_c4, joint_value_cg,
            STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        torch.manual_seed(300)
        seed = 99989
        online_u, _ = build_online_target_pair(STATE_DIM, seed)
        online_p, _ = build_pair_online_target(derive_pair_seed(seed))

        states = torch.randn(2, R_MAX, STATE_DIM)
        states[:, :, 3] = 0.3
        mask = torch.zeros(2, R_MAX)
        mask[:, :3] = 1.0
        actions = torch.randint(0, N_ACTIONS, (2, R_MAX))

        with torch.no_grad():
            q_c4 = joint_value_c4(online_u, online_p, states, actions, mask)
            q_c3 = joint_value_cg(online_u, online_p, states, actions, mask)

        assert not torch.allclose(q_c4, q_c3, atol=1e-7), \
            "C4 and C3 should differ when gate != 1"

    def test_c4_c3_equal_with_unit_gate(self):
        from prdrl.rl.coordination_graph import (
            joint_value_c4, joint_value_cg,
            STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        torch.manual_seed(301)
        seed = 99988
        online_u, _ = build_online_target_pair(STATE_DIM, seed)
        online_p, _ = build_pair_online_target(derive_pair_seed(seed))

        states = torch.randn(2, R_MAX, STATE_DIM)
        states[:, :, 3] = 1.0
        mask = torch.zeros(2, R_MAX)
        mask[:, :3] = 1.0
        actions = torch.randint(0, N_ACTIONS, (2, R_MAX))

        with torch.no_grad():
            q_c4 = joint_value_c4(online_u, online_p, states, actions, mask)
            q_c3 = joint_value_cg(online_u, online_p, states, actions, mask)

        assert torch.allclose(q_c4, q_c3, atol=1e-6)


class TestC4DoubleQTarget:
    def test_c4_double_q_shape(self):
        from prdrl.rl.coordination_graph import (
            c4_double_q_target, STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        seed = 99987
        online_u, target_u = build_online_target_pair(STATE_DIM, seed)
        online_p, target_p = build_pair_online_target(derive_pair_seed(seed))

        B = 2
        states = torch.randn(B, R_MAX, STATE_DIM)
        states[:, :, 3] = 0.4
        mask = torch.zeros(B, R_MAX)
        mask[:, :3] = 1.0
        r = torch.randn(B)
        d = torch.zeros(B)

        y = c4_double_q_target(online_u, online_p, target_u, target_p,
                               states, mask, r, d)
        assert y.shape == (B,)

    def test_c4_batched_double_q_equivalence(self):
        from prdrl.rl.coordination_graph import (
            c4_double_q_target, c4_double_q_target_batched,
            STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        torch.manual_seed(302)
        seed = 99986
        online_u, target_u = build_online_target_pair(STATE_DIM, seed)
        online_p, target_p = build_pair_online_target(derive_pair_seed(seed))

        B = 2
        states = torch.randn(B, R_MAX, STATE_DIM)
        states[:, :, 3] = 0.5
        mask = torch.zeros(B, R_MAX)
        mask[:, :3] = 1.0
        r = torch.randn(B)
        d = torch.zeros(B)

        y_s = c4_double_q_target(online_u, online_p, target_u, target_p,
                                 states, mask, r, d)
        y_b = c4_double_q_target_batched(online_u, online_p, target_u, target_p,
                                         states, mask, r, d)
        assert torch.allclose(y_s, y_b, atol=1e-5)


class TestC4ExactJointArgmax:
    def test_exact_c4_matches_maxsum_small_n(self):
        from prdrl.rl.coordination_graph import (
            exact_joint_argmax_c4, maxsum_inference_c4,
            joint_value_c4, STATE_DIM, R_MAX, N_ACTIONS,
            build_pair_online_target, derive_pair_seed,
        )
        from prdrl.rl.network import build_online_target_pair

        torch.manual_seed(303)
        seed = 99985
        online_u, _ = build_online_target_pair(STATE_DIM, seed)
        online_p, _ = build_pair_online_target(derive_pair_seed(seed))

        states = torch.randn(1, R_MAX, STATE_DIM)
        states[:, :, 3] = 0.5
        mask = torch.zeros(1, R_MAX)
        mask[0, :2] = 1.0

        with torch.no_grad():
            exact_a, exact_q = exact_joint_argmax_c4(online_u, online_p, states, mask, 0)
            ms_a = maxsum_inference_c4(online_u, online_p, states, mask)
            ms_q = joint_value_c4(online_u, online_p, states,
                                  ms_a, mask).item()

        assert exact_q >= ms_q - 1e-6


class TestC4FullRegressionCount:
    @pytest.fixture(scope="class")
    def verdict(self):
        return _load("verdict")

    def test_full_regression_field(self, verdict):
        reg = verdict.get("FULL_REGRESSION", "")
        assert reg == "6114/6114 PASS (0 failures)"
