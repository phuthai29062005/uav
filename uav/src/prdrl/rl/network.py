"""
E4-P: Parameter-shared Double DQN network for PR-DRL-DMOEA.

Architecture:
  - Shared parameters across ALL active regions.
  - Per-region input = [local_state_V2 | global_context_V1].
  - Output = 4 Q-values (KEEP, LOCAL, PREDICT, DIVERSIFY).

Stage: PREPARATION. No substantive training in E4-P.
RL_STATUS = NOT_STARTED.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Action ordering must match interface.py
ACTION_ORDER = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")
N_ACTIONS    = 4

# Frozen architecture per manuscript starting design
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128


class SharedRegionalQNetwork(nn.Module):
    """
    Parameter-shared regional Q-network.

    Each active region at a decision event passes its own
    [local_state | global_context] vector through the SAME
    network weights. No cross-region attention or GNN in E4-P.

    Input dim  = local_dim + global_context_dim
    Output dim = N_ACTIONS = 4
    """

    def __init__(self, input_dim: int, hidden1: int = HIDDEN_DIM_1,
                 hidden2: int = HIDDEN_DIM_2, n_actions: int = N_ACTIONS):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim  = input_dim
        self.n_actions  = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → Q-values: (batch, n_actions)."""
        return self.net(x)


def build_online_target_pair(input_dim: int, seed: int = 0):
    """
    Create a deterministically initialized (online, target) network pair.

    Both networks start with identical weights.
    Target network parameters are frozen (no grad) until explicit hard update.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _make() -> SharedRegionalQNetwork:
        net = SharedRegionalQNetwork(input_dim)
        # deterministic init with the same generator state
        for p in net.parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        return net

    online = _make()
    # Re-seed to same value so target is identical
    gen.manual_seed(seed)
    target = _make()
    for p in target.parameters():
        p.requires_grad_(False)

    return online, target


def hard_update(online: nn.Module, target: nn.Module) -> None:
    """Copy online weights to target (hard update)."""
    target.load_state_dict(online.state_dict())


def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """Soft update: θ_tgt ← τ θ_online + (1-τ) θ_tgt."""
    with torch.no_grad():
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.copy_(tau * p_o.data + (1.0 - tau) * p_t.data)


def double_dqn_target(
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
    online_net: nn.Module,
    target_net: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """
    Double DQN target:
      a* = argmax_a Q_online(s', a)
      y  = r + gamma * (1 - done) * Q_target(s', a*)

    Args:
        reward:     (batch,)
        next_state: (batch, input_dim)
        done:       (batch,) float, 1.0 if terminal
        online_net, target_net: SharedRegionalQNetwork
        gamma: discount factor

    Returns:
        y: (batch,) target Q-values
    """
    with torch.no_grad():
        q_online_next = online_net(next_state)          # (batch, 4)
        a_star = q_online_next.argmax(dim=1, keepdim=True)  # (batch, 1)
        q_target_next = target_net(next_state)          # (batch, 4)
        q_next = q_target_next.gather(1, a_star).squeeze(1)  # (batch,)
        y = reward + gamma * (1.0 - done) * q_next
    return y
