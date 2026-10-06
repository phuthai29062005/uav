"""
T1-T57+: Tests for E4-P — Learned Controller Preparation.

Each test is numbered to match the spec. Tests operate on:
  - The RL module code (network.py, replay.py, reward.py, agent.py)
  - The E4P experiment outputs (results/E4_learned_controller_preparation/)
  - Invariants: production code unchanged, oracle labels NOT in training path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair,
    hard_update, polyak_update, double_dqn_target,
    ACTION_ORDER, N_ACTIONS, HIDDEN_DIM_1, HIDDEN_DIM_2,
)
from prdrl.rl.replay import SharedReplayBuffer, RegionalTransition
from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward,
    compute_fe_cost, compute_regional_reward, RewardConfig,
    ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT, GAMMA_DEFAULT,
    BUDGET_PER_REGION,
)
from prdrl.rl.agent import (
    EpsilonGreedyConfig, select_regional_actions,
    ACTION_TO_IDX, IDX_TO_ACTION,
)
from prdrl.rl.interface import (
    ACTION_KEEP, ACTION_LOCAL, ACTION_PREDICT, ACTION_DIVERSIFY,
    CURRENT_RESPONSE_BUDGET_VALUE,
    STATE_SCHEMA_V2,
)

OUT_DIR = Path("results/E4_learned_controller_preparation")


# ────────────────────────────────────────────────────────────────
# T1-T8: Output file manifest
# ────────────────────────────────────────────────────────────────

EXPECTED_FILES = [
    "E4P_prechange_manifest.json",
    "E4P_rl_inventory.json",
    "E4P_global_context_audit.json",
    "E4P_state_schema.json",
    "E4P_reward_contract.json",
    "E4P_transition_contract.json",
    "E4P_joint_event_contract.json",
    "E4P_replay_contract.json",
    "E4P_network_spec.json",
    "E4P_exploration_contract.json",
    "E4P_rng_contract.json",
    "E4P_normalization_contract.json",
    "E4P_oracle_label_firewall.json",
    "E4P_smoke_validation.json",
    "E4_training_protocol.json",
    "E4P_verdict.json",
    "E4P_report.md",
]


@pytest.mark.parametrize("fname", EXPECTED_FILES[:8])
def test_T1_T8_output_files_exist(fname):
    assert (OUT_DIR / fname).exists(), f"Missing: {fname}"


@pytest.mark.parametrize("fname", EXPECTED_FILES[8:])
def test_T1_T8_remaining_output_files_exist(fname):
    assert (OUT_DIR / fname).exists(), f"Missing: {fname}"


# ────────────────────────────────────────────────────────────────
# T9-T13: Production code immutability
# ────────────────────────────────────────────────────────────────

def test_T9_rl_status_not_started():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data["RL_STATUS"] == "NOT_STARTED"


def test_T10_drl_training_not_started():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data["DRL_TRAINING"] == "NOT_STARTED"


def test_T11_production_default_unchanged():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data["PRODUCTION_DEFAULT"] == "O1_F005"


def test_T12_manifest_production_budget():
    data = json.loads((OUT_DIR / "E4P_prechange_manifest.json").read_text())
    assert data["production_constants"]["BUDGET_PER_ACTIVE_REGION"] == CURRENT_RESPONSE_BUDGET_VALUE


def test_T13_manifest_records_h1_outcome():
    data = json.loads((OUT_DIR / "E4P_prechange_manifest.json").read_text())
    assert data["H1_OUTCOME_REVIEWED"] == "RULE_MAPPING_INADEQUATE"
    assert data["E4_READINESS"] == "READY"


# ────────────────────────────────────────────────────────────────
# T14-T20: Network architecture
# ────────────────────────────────────────────────────────────────

def test_T14_network_action_order():
    assert ACTION_ORDER == ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")
    assert N_ACTIONS == 4


def test_T15_network_input_dim():
    spec = json.loads((OUT_DIR / "E4P_network_spec.json").read_text())
    assert spec["input_dim"] == 9
    assert spec["hidden_layers"] == [128, 128]
    assert spec["output_dim"] == 4


def test_T16_forward_pass_shape():
    online, _ = build_online_target_pair(9, seed=0)
    x = torch.randn(16, 9)
    q = online(x)
    assert q.shape == (16, 4)


def test_T17_forward_pass_finite():
    online, _ = build_online_target_pair(9, seed=0)
    x = torch.randn(8, 9)
    q = online(x)
    assert torch.isfinite(q).all()


def test_T18_deterministic_init():
    o1, _ = build_online_target_pair(9, seed=42)
    o2, _ = build_online_target_pair(9, seed=42)
    x = torch.randn(4, 9)
    assert torch.allclose(o1(x), o2(x)), "Same seed must produce same weights"


def test_T19_different_seed_different_init():
    o1, _ = build_online_target_pair(9, seed=0)
    o2, _ = build_online_target_pair(9, seed=99)
    x = torch.randn(4, 9)
    assert not torch.allclose(o1(x), o2(x)), "Different seeds should differ"


def test_T20_target_no_grad():
    _, target = build_online_target_pair(9, seed=0)
    for p in target.parameters():
        assert not p.requires_grad, "Target params must have requires_grad=False"


# ────────────────────────────────────────────────────────────────
# T21-T26: Double DQN algebra
# ────────────────────────────────────────────────────────────────

def test_T21_double_dqn_target_shape():
    online, target = build_online_target_pair(9, seed=0)
    r = torch.randn(8)
    ns = torch.randn(8, 9)
    d = torch.zeros(8)
    y = double_dqn_target(r, ns, d, online, target, gamma=0.99)
    assert y.shape == (8,)


def test_T22_double_dqn_target_finite():
    online, target = build_online_target_pair(9, seed=0)
    r = torch.randn(8)
    ns = torch.randn(8, 9)
    d = torch.zeros(8)
    y = double_dqn_target(r, ns, d, online, target, gamma=0.99)
    assert torch.isfinite(y).all()


def test_T23_double_dqn_terminal_ignores_next():
    online, target = build_online_target_pair(9, seed=1)
    r = torch.tensor([0.5])
    ns = torch.randn(1, 9)
    done = torch.tensor([1.0])   # terminal
    y = double_dqn_target(r, ns, done, online, target, gamma=0.99)
    assert abs(float(y[0]) - 0.5) < 1e-6, "Terminal: y must equal reward exactly"


def test_T24_double_dqn_nonterminal_adds_future():
    online, target = build_online_target_pair(9, seed=2)
    r = torch.zeros(1)
    ns = torch.randn(1, 9)
    done = torch.zeros(1)
    y = double_dqn_target(r, ns, done, online, target, gamma=0.99)
    # With r=0, y = gamma * Q_target(ns, a*); should be nonzero
    assert abs(float(y[0])) > 1e-6


def test_T25_double_dqn_uses_online_for_action_selection():
    """Verify: argmax comes from online, not target."""
    online, target = build_online_target_pair(9, seed=3)
    # Diverge online and target
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    r = torch.zeros(4)
    ns = torch.randn(4, 9)
    done = torch.zeros(4)
    y1 = double_dqn_target(r, ns, done, online, target, gamma=0.99)
    y2 = double_dqn_target(r, ns, done, target, online, gamma=0.99)
    # Should differ (different networks for action vs value)
    assert not torch.allclose(y1, y2), "Online vs target swap must change result"


def test_T26_hard_update_copies_weights():
    online, target = build_online_target_pair(9, seed=0)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.ones_like(p))
    hard_update(online, target)
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        assert torch.allclose(p_o.data, p_t.data)


# ────────────────────────────────────────────────────────────────
# T27-T31: Optimizer smoke
# ────────────────────────────────────────────────────────────────

def test_T27_optimizer_step_finite_loss():
    import torch.optim as optim
    online, target = build_online_target_pair(9, seed=0)
    opt = optim.Adam(online.parameters(), lr=1e-3)
    x = torch.randn(8, 9)
    r = torch.randn(8)
    ns = torch.randn(8, 9)
    done = torch.zeros(8)
    acts = torch.randint(0, 4, (8,))
    y = double_dqn_target(r, ns, done, online, target, 0.99)
    q_pred = online(x).gather(1, acts.unsqueeze(1)).squeeze(1)
    loss = nn.SmoothL1Loss()(q_pred, y.detach())
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_T28_optimizer_step_finite_gradients():
    import torch.optim as optim
    online, target = build_online_target_pair(9, seed=5)
    opt = optim.Adam(online.parameters(), lr=1e-3)
    x = torch.randn(8, 9)
    y = torch.randn(8)
    acts = torch.randint(0, 4, (8,))
    q_pred = online(x).gather(1, acts.unsqueeze(1)).squeeze(1)
    loss = nn.SmoothL1Loss()(q_pred, y)
    opt.zero_grad()
    loss.backward()
    for p in online.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_T29_loss_function_is_huber():
    spec = json.loads((OUT_DIR / "E4P_network_spec.json").read_text())
    assert "Huber" in spec["loss_function"] or "SmoothL1" in spec["loss_function"]


def test_T30_polyak_update_interpolates():
    online, target = build_online_target_pair(9, seed=0)
    tau = 0.1
    # Record old target params
    old_target = [p.data.clone() for p in target.parameters()]
    old_online = [p.data.clone() for p in online.parameters()]
    polyak_update(online, target, tau)
    for p_t, old_t, old_o in zip(target.parameters(), old_target, old_online):
        expected = tau * old_o + (1.0 - tau) * old_t
        assert torch.allclose(p_t.data, expected, atol=1e-6)


def test_T31_smoke_validation_overall_pass():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data.get("OVERALL") == "PASS", f"Smoke OVERALL={data.get('OVERALL')}"


# ────────────────────────────────────────────────────────────────
# T32-T37: Replay buffer
# ────────────────────────────────────────────────────────────────

def _make_transition(region_id=0, event_id="e0", action=2, fallback=False, oracle_action=None):
    return RegionalTransition(
        state=np.zeros(9, dtype=np.float32),
        requested_action=action,
        realized_operator=IDX_TO_ACTION.get(action, "LOCAL"),
        reward=0.1,
        next_state=np.ones(9, dtype=np.float32),
        done=False,
        region_id=region_id,
        event_id=event_id,
        fallback_applied=fallback,
        response_fe=10,
        diagnostic_oracle_action=oracle_action,
    )


def test_T32_buffer_push_and_len():
    buf = SharedReplayBuffer(capacity=100, rng_seed=0)
    buf.push(_make_transition())
    assert len(buf) == 1


def test_T33_buffer_capacity_limit():
    buf = SharedReplayBuffer(capacity=3, rng_seed=0)
    for i in range(5):
        buf.push(_make_transition(event_id=str(i)))
    assert len(buf) == 3


def test_T34_buffer_sample_size():
    buf = SharedReplayBuffer(capacity=100, rng_seed=0)
    for i in range(20):
        buf.push(_make_transition(event_id=str(i)))
    batch = buf.sample(8)
    assert len(batch) == 8


def test_T35_buffer_sample_too_large_raises():
    buf = SharedReplayBuffer(capacity=100, rng_seed=0)
    buf.push(_make_transition())
    with pytest.raises(ValueError):
        buf.sample(5)


def test_T36_buffer_to_arrays_shapes():
    buf = SharedReplayBuffer(capacity=100, rng_seed=0)
    for i in range(10):
        buf.push(_make_transition(action=i % 4))
    batch = buf.sample(5)
    s, a, r, ns, d = buf.to_arrays(batch)
    assert s.shape == (5, 9)
    assert a.shape == (5,)
    assert r.shape == (5,)
    assert ns.shape == (5, 9)
    assert d.shape == (5,)


def test_T37_oracle_label_not_in_training_arrays():
    buf = SharedReplayBuffer(capacity=100, rng_seed=0)
    t = _make_transition(oracle_action="KEEP")
    buf.push(t)
    batch = buf.sample(1)
    # to_arrays returns (s, a, r, ns, done) — exactly 5 arrays, no oracle column
    arrays = buf.to_arrays(batch)
    assert len(arrays) == 5, "to_arrays must return exactly 5 training arrays"
    # diagnostic_oracle_action is accessible via the transition object (not arrays)
    assert batch[0].diagnostic_oracle_action == "KEEP"


# ────────────────────────────────────────────────────────────────
# T38-T44: Reward functions
# ────────────────────────────────────────────────────────────────

def test_T38_global_reward_positive_for_improvement():
    r = compute_global_reward(q_before=-0.02, q_after=-0.01)
    assert r > 0


def test_T39_global_reward_negative_for_degradation():
    r = compute_global_reward(q_before=-0.01, q_after=-0.02)
    assert r < 0


def test_T40_global_reward_zero_for_no_change():
    r = compute_global_reward(q_before=-0.01, q_after=-0.01)
    assert abs(r) < 1e-9


def test_T41_keep_fe_cost_zero():
    assert compute_fe_cost(actual_fe=0, n_active_regions=5) == 0.0


def test_T42_fe_cost_bounded():
    c = compute_fe_cost(actual_fe=50, n_active_regions=5, budget_per_region=10)
    assert c == pytest.approx(1.0)


def test_T43_pbi_reward_improves_when_loss_decreases():
    r = compute_regional_pbi_reward(pbi_loss_before=0.5, pbi_loss_after=0.3)
    assert r > 0


def test_T44_team_dominant_enforced():
    """alpha_g < alpha_l must raise."""
    with pytest.raises(ValueError):
        compute_regional_reward(0.1, 0.0, 0.0, alpha_g=0.05, alpha_l=0.1, beta=0.01)


# ────────────────────────────────────────────────────────────────
# T45-T50: Agent / action selection
# ────────────────────────────────────────────────────────────────

def test_T45_action_to_idx_mapping():
    assert ACTION_TO_IDX["KEEP"] == 0
    assert ACTION_TO_IDX["LOCAL"] == 1
    assert ACTION_TO_IDX["PREDICT"] == 2
    assert ACTION_TO_IDX["DIVERSIFY"] == 3


def test_T46_idx_to_action_mapping():
    assert IDX_TO_ACTION[0] == "KEEP"
    assert IDX_TO_ACTION[3] == "DIVERSIFY"


def test_T47_epsilon_schedule_bounds():
    cfg = EpsilonGreedyConfig(eps_start=1.0, eps_end=0.05, n_steps=10000)
    assert cfg.get_epsilon(0) == pytest.approx(1.0)
    assert cfg.get_epsilon(10000) == pytest.approx(0.05)


def test_T48_epsilon_monotone_decreasing():
    cfg = EpsilonGreedyConfig()
    eps_vals = [cfg.get_epsilon(t) for t in range(0, cfg.n_steps + 1, cfg.n_steps // 10)]
    for i in range(len(eps_vals) - 1):
        assert eps_vals[i] >= eps_vals[i + 1]


def test_T49_select_actions_all_regions_covered():
    online, _ = build_online_target_pair(9, seed=0)
    gc = np.zeros(5, dtype=np.float32)
    local_states = {r: np.zeros(4, dtype=np.float32) for r in range(5)}
    act_idx, act_str = select_regional_actions(online, local_states, gc, epsilon=0.0,
                                               event_seed=42, device=torch.device("cpu"))
    assert set(act_idx.keys()) == set(range(5))
    assert set(act_str.keys()) == set(range(5))


def test_T50_select_actions_valid_action_ids():
    online, _ = build_online_target_pair(9, seed=0)
    gc = np.zeros(5, dtype=np.float32)
    local_states = {r: np.random.randn(4).astype(np.float32) for r in range(3)}
    act_idx, act_str = select_regional_actions(online, local_states, gc, epsilon=0.5,
                                               event_seed=99, device=torch.device("cpu"))
    for r, idx in act_idx.items():
        assert idx in (0, 1, 2, 3)
        assert act_str[r] in ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")


# ────────────────────────────────────────────────────────────────
# T51-T57: Contract JSON schemas
# ────────────────────────────────────────────────────────────────

def test_T51_state_schema_dims():
    data = json.loads((OUT_DIR / "E4P_state_schema.json").read_text())
    assert data["local_dim"] == 4
    assert data["global_dim"] == 5
    assert data["total_input_dim"] == 9


def test_T52_state_schema_local_features():
    data = json.loads((OUT_DIR / "E4P_state_schema.json").read_text())
    assert data["local_features"] == ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]


def test_T53_state_schema_global_features():
    data = json.loads((OUT_DIR / "E4P_state_schema.json").read_text())
    assert data["global_features"] == [
        "active_region_fraction", "mean_severity", "max_severity",
        "mean_convergence_loss", "max_convergence_loss",
    ]


def test_T54_state_schema_no_oracle_utility():
    data = json.loads((OUT_DIR / "E4P_state_schema.json").read_text())
    assert data.get("local_no_oracle_utility") is True
    assert data.get("no_NaN_guarantee") is not None


def test_T55_reward_contract_joint_outcome():
    data = json.loads((OUT_DIR / "E4P_reward_contract.json").read_text())
    assert data.get("joint_outcome_required") is True
    assert data.get("oracle_labels_as_target") is False


def test_T56_transition_contract_predict_fallback():
    data = json.loads((OUT_DIR / "E4P_transition_contract.json").read_text())
    fb = data["action_field_semantics"]["PREDICT_fallback_rule"]
    assert len(fb) > 10, "PREDICT fallback rule must be documented"


def test_T57_oracle_firewall_forbids_training():
    data = json.loads((OUT_DIR / "E4P_oracle_label_firewall.json").read_text())
    assert data.get("no_oracle_imitation_learning") is True
    assert data.get("no_oracle_pretraining") is True
    assert data.get("GREEDY_FACTORIZED_oracle_labels_are_not_ground_truth") is True


# ────────────────────────────────────────────────────────────────
# T58-T63: Verdict
# ────────────────────────────────────────────────────────────────

def test_T58_verdict_e4p_complete_yes():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("E4P_COMPLETE") == "YES", f"E4P_COMPLETE={data.get('E4P_COMPLETE')}"


def test_T59_verdict_next_stage():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("NEXT_STAGE") == "E4T_TRAINING_PROTOCOL_FREEZE"


def test_T60_verdict_architecture():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("E4_ARCHITECTURE") == "PARAMETER_SHARED_DOUBLE_DQN"


def test_T61_verdict_oracle_label_forbidden():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("ORACLE_LABEL_TRAINING") == "FORBIDDEN"


def test_T62_verdict_long_training_not_run():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("LONG_TRAINING_RUN") == "NOT_RUN"


def test_T63_verdict_reward_scope():
    data = json.loads((OUT_DIR / "E4P_verdict.json").read_text())
    assert data.get("REWARD_SCOPE") == "JOINT_GLOBAL_PLUS_LOCAL"


# ────────────────────────────────────────────────────────────────
# T64-T68: Smoke validation sub-tests
# ────────────────────────────────────────────────────────────────

def test_T64_smoke_forward_pass():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data["forward_pass"]["PASS"] is True


def test_T65_smoke_double_dqn_algebra():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data["double_dqn_target"]["PASS"] is True
    assert data["double_dqn_target"]["algebra_correct"] is True


def test_T66_smoke_optimizer():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data["optimizer_smoke"]["PASS"] is True


def test_T67_smoke_joint_coordination():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data["joint_coordination_fixture"]["PASS"] is True
    assert data["joint_coordination_fixture"]["joint_reward_captures_interaction"] is True


def test_T68_smoke_oracle_firewall():
    data = json.loads((OUT_DIR / "E4P_smoke_validation.json").read_text())
    assert data["oracle_label_firewall"]["PASS"] is True
    assert data["oracle_label_firewall"]["oracle_label_in_training_arrays"] is False


# ────────────────────────────────────────────────────────────────
# T69-T73: Training protocol draft
# ────────────────────────────────────────────────────────────────

def test_T69_training_protocol_not_executed():
    data = json.loads((OUT_DIR / "E4_training_protocol.json").read_text())
    assert data.get("all_fields_frozen_before_training") is False


def test_T70_training_protocol_excludes_historical_seeds():
    data = json.loads((OUT_DIR / "E4_training_protocol.json").read_text())
    policy = data.get("seeds_policy", "")
    assert "901-940" in policy, "Must exclude H1 seeds"
    assert "701-780" in policy or "E3" in policy


def test_T71_training_protocol_no_oracle_labels():
    data = json.loads((OUT_DIR / "E4_training_protocol.json").read_text())
    assert data.get("no_H1_oracle_labels_as_targets") is True


def test_T72_rng_contract_7_namespaces():
    data = json.loads((OUT_DIR / "E4P_rng_contract.json").read_text())
    assert len(data["namespaces"]) == 7


def test_T73_rng_isolation_guarantee():
    data = json.loads((OUT_DIR / "E4P_rng_contract.json").read_text())
    assert data.get("production_rng_frozen") is True


# ────────────────────────────────────────────────────────────────
# T74-T78: Global context contract
# ────────────────────────────────────────────────────────────────

def test_T74_global_context_5_features():
    data = json.loads((OUT_DIR / "E4P_global_context_audit.json").read_text())
    assert data["global_dim"] == 5
    assert len(data["GlobalContextV1_chosen"]) == 5


def test_T75_global_context_no_oracle():
    data = json.loads((OUT_DIR / "E4P_global_context_audit.json").read_text())
    assert data.get("no_oracle_features") is True
    assert data.get("no_future_information") is True


def test_T76_global_context_compute_zero_active():
    """Zero active regions → zero vector."""
    from prdrl.experiments.E4P_learned_controller_preparation import compute_global_context_v1
    gc = compute_global_context_v1({}, R=10)
    assert gc.shape == (5,)
    assert np.all(gc == 0.0)


def test_T77_global_context_compute_values():
    from prdrl.experiments.E4P_learned_controller_preparation import compute_global_context_v1
    v2 = {
        0: {"severity_r": 0.8, "convergence_loss_r": 0.5, "occupancy_r": 0.3, "delta_severity_r": 0.1},
        1: {"severity_r": 0.4, "convergence_loss_r": 0.2, "occupancy_r": 0.6, "delta_severity_r": -0.05},
    }
    gc = compute_global_context_v1(v2, R=10)
    assert abs(gc[0] - 0.2) < 1e-9   # 2/10 active
    assert abs(gc[1] - 0.6) < 1e-9   # mean_severity
    assert abs(gc[2] - 0.8) < 1e-9   # max_severity
    assert gc[3] > 0                  # mean_convergence_loss
    assert gc[4] > 0                  # max_convergence_loss


def test_T78_global_context_shared_per_event():
    data = json.loads((OUT_DIR / "E4P_global_context_audit.json").read_text())
    assert "active regions" in data.get("sharing", "") and "same event" in data.get("sharing", "")


# ────────────────────────────────────────────────────────────────
# T79-T83: Normalization + episode contract
# ────────────────────────────────────────────────────────────────

def test_T79_normalization_no_fitted_stats():
    data = json.loads((OUT_DIR / "E4P_normalization_contract.json").read_text())
    assert "NOT fitted" in data["normalization_fitted_on"]


def test_T80_normalization_local_clip():
    from prdrl.experiments.E4P_learned_controller_preparation import normalize_local_state
    v2 = {"severity_r": 2.0, "convergence_loss_r": 15.0, "occupancy_r": -0.5, "delta_severity_r": 5.0}
    out = normalize_local_state(v2)
    assert out[0] == pytest.approx(1.0)       # severity clipped to 1
    assert out[1] == pytest.approx(1.0)       # convergence_loss clipped at 10 then /10
    assert out[2] == pytest.approx(0.0)       # occupancy clipped to 0
    assert out[3] == pytest.approx(1.0)       # delta_severity clipped to 1


def test_T81_normalization_global_clip():
    from prdrl.experiments.E4P_learned_controller_preparation import normalize_global_context
    gc = np.array([1.5, 2.0, -0.3, 15.0, 0.0], dtype=np.float64)
    out = normalize_global_context(gc)
    assert out[0] == pytest.approx(1.0)   # active_region_fraction clipped
    assert out[1] == pytest.approx(1.0)   # mean_severity clipped
    assert out[2] == pytest.approx(0.0)   # max_severity clipped to 0
    assert out[3] == pytest.approx(1.0)   # mean_convergence_loss clipped at 10 /10
    assert out[4] == pytest.approx(0.0)   # max_convergence_loss clipped to 0


def test_T82_episode_length_10():
    data = json.loads((OUT_DIR / "E4P_normalization_contract.json").read_text())
    assert data["episode_boundary"]["length"] == 10


def test_T83_done_not_per_region():
    data = json.loads((OUT_DIR / "E4P_normalization_contract.json").read_text())
    assert data["episode_boundary"]["done_per_region"] is False


# ────────────────────────────────────────────────────────────────
# T84-T88: Action interface compatibility
# ────────────────────────────────────────────────────────────────

def test_T84_action_order_matches_interface():
    """ACTION_ORDER in network.py must match interface.py action IDs."""
    from prdrl.rl.interface import ACTION_IDS
    for i, name in enumerate(ACTION_ORDER):
        assert ACTION_IDS[name] == i, f"Action {name} ID mismatch"


def test_T85_v2_state_dim_matches_schema():
    assert STATE_SCHEMA_V2.dim == 4


def test_T86_v2_state_features_match_schema():
    assert STATE_SCHEMA_V2.feature_names == ("severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r")


def test_T87_budget_per_region_matches_interface():
    assert BUDGET_PER_REGION == CURRENT_RESPONSE_BUDGET_VALUE


def test_T88_transition_event_ids_returnable():
    buf = SharedReplayBuffer(capacity=50, rng_seed=0)
    for i in range(5):
        buf.push(_make_transition(event_id=f"ev_{i}"))
    ids = buf.event_ids()
    assert len(ids) == 5
    assert all(f"ev_{i}" in ids for i in range(5))


# ────────────────────────────────────────────────────────────────
# T89-T92: Invariants enforced
# ────────────────────────────────────────────────────────────────

def test_T89_no_hard_action():
    """HARD must not appear in any action list."""
    assert "HARD" not in ACTION_ORDER
    assert "HARD" not in ACTION_TO_IDX


def test_T90_no_nsga3():
    """NSGA-III must not appear in production spec."""
    data = json.loads((OUT_DIR / "E4P_prechange_manifest.json").read_text())
    assert data["production_constants"].get("BASE_EVOLUTION") == "NSGA-II"


def test_T91_reward_config_alpha_constraint():
    rc = RewardConfig(alpha_g=1.0, alpha_l=0.1)
    assert rc.alpha_g >= rc.alpha_l
    with pytest.raises(ValueError):
        RewardConfig(alpha_g=0.05, alpha_l=0.1)


def test_T92_training_protocol_not_run():
    """Training protocol must note it is a draft only."""
    data = json.loads((OUT_DIR / "E4_training_protocol.json").read_text())
    note = data.get("_note", "")
    assert "DO NOT EXECUTE" in note.upper() or "do not" in note.lower()


# ────────────────────────────────────────────────────────────────
# T93-T96: Report
# ────────────────────────────────────────────────────────────────

def test_T93_report_exists():
    assert (OUT_DIR / "E4P_report.md").exists()


def test_T94_report_mentions_rl_status():
    text = (OUT_DIR / "E4P_report.md").read_text()
    assert "NOT_STARTED" in text


def test_T95_report_mentions_architecture():
    text = (OUT_DIR / "E4P_report.md").read_text()
    assert "PARAMETER_SHARED_DOUBLE_DQN" in text or "Parameter-shared" in text


def test_T96_report_oracle_firewall():
    text = (OUT_DIR / "E4P_report.md").read_text()
    assert "FORBIDDEN" in text or "oracle" in text.lower()
