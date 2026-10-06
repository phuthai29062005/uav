"""
Branch-C treatment: QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 (frozen by E4K).

Monotonic joint-value mixer used in TRAINING ONLY. Decentralized greedy per-region argmax on
the (unchanged) utility network remains the execution path; the mixer is never required to
select actions. Architecture frozen in results/E4K_.../E4K_mixer_architecture.json:

    inputs: q_selected [B, R_MAX], central_state [B, CENTRAL_STATE_DIM]
    W1 = abs(Linear(CENTRAL_STATE_DIM, R_MAX*MIX_HIDDEN))  -> reshape [B, R_MAX, MIX_HIDDEN]
    b1 = Linear(CENTRAL_STATE_DIM, MIX_HIDDEN)
    h  = ELU( bmm(q.unsqueeze(1), W1).squeeze(1) + b1 )                  # [B, MIX_HIDDEN]
    W2 = abs(Linear(CENTRAL_STATE_DIM, MIX_HIDDEN)) -> reshape [B, MIX_HIDDEN, 1]
    b2 = Linear(CENTRAL_STATE_DIM, 1)
    Q_tot = bmm(h.unsqueeze(1), W2).squeeze(1) + b2                      # [B, 1]

The non-negative (abs) hypernetwork weights guarantee dQ_tot/dq_r >= 0 for every region, so the
mixer is monotone in each regional utility. No softplus/exp/relu substitution; no mixer variant
search; no attention/transformer/recurrent/graph/embedding in V1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Frozen dimensions (must match E4K_region_slot_contract / E4K_central_state_contract / E4K_mixer_architecture)
R_MAX = 10
CENTRAL_STATE_DIM = 5 * R_MAX + 5      # 55
MIX_HIDDEN_DIM = 32
NONNEGATIVE_TRANSFORM = "ABS"
MIXER_SEED_OFFSET = 1                   # mixer_seed = utility_initialization_seed + 1 (frozen by E4K)

TREATMENT_ARCH = "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"


class QMIXMixerV1(nn.Module):
    """Monotonic QMIX mixer V1 (state-conditioned, non-additive, monotone in each q_r)."""

    def __init__(self, r_max: int = R_MAX, central_state_dim: int = CENTRAL_STATE_DIM,
                 mix_hidden: int = MIX_HIDDEN_DIM):
        super().__init__()
        self.r_max = r_max
        self.central_state_dim = central_state_dim
        self.mix_hidden = mix_hidden
        # Hypernetworks generating state-conditioned mixer parameters.
        self.hyper_w1 = nn.Linear(central_state_dim, r_max * mix_hidden)
        self.hyper_b1 = nn.Linear(central_state_dim, mix_hidden)
        self.hyper_w2 = nn.Linear(central_state_dim, mix_hidden)
        self.hyper_b2 = nn.Linear(central_state_dim, 1)

    def forward(self, q_selected: torch.Tensor, central_state: torch.Tensor) -> torch.Tensor:
        """q_selected: [B, R_MAX] (inactive regions must already be 0); central_state: [B, 55]
        -> Q_tot: [B, 1]."""
        if q_selected.dim() != 2 or q_selected.shape[1] != self.r_max:
            raise ValueError(f"q_selected must be [B,{self.r_max}], got {tuple(q_selected.shape)}")
        if central_state.dim() != 2 or central_state.shape[1] != self.central_state_dim:
            raise ValueError(
                f"central_state must be [B,{self.central_state_dim}], got {tuple(central_state.shape)}")
        B = q_selected.shape[0]
        # first monotonic layer
        w1 = torch.abs(self.hyper_w1(central_state)).view(B, self.r_max, self.mix_hidden)  # [B,R,H]
        b1 = self.hyper_b1(central_state)                                                  # [B,H]
        h = torch.nn.functional.elu(
            torch.bmm(q_selected.unsqueeze(1), w1).squeeze(1) + b1)                        # [B,H]
        # second monotonic layer
        w2 = torch.abs(self.hyper_w2(central_state)).view(B, self.mix_hidden, 1)           # [B,H,1]
        b2 = self.hyper_b2(central_state)                                                  # [B,1]
        q_tot = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2                              # [B,1]
        return q_tot

    def mixer_weights(self, central_state: torch.Tensor):
        """Return (W1[B,R,H], W2[B,H,1]) after the abs() transform — for diagnostics only."""
        B = central_state.shape[0]
        w1 = torch.abs(self.hyper_w1(central_state)).view(B, self.r_max, self.mix_hidden)
        w2 = torch.abs(self.hyper_w2(central_state)).view(B, self.mix_hidden, 1)
        return w1, w2


def derive_mixer_seed(utility_initialization_seed: int) -> int:
    """Frozen E4K rule: mixer_seed = utility_initialization_seed + 1."""
    return int(utility_initialization_seed) + MIXER_SEED_OFFSET


def _init_mixer_(mixer: QMIXMixerV1, gen: torch.Generator) -> None:
    for p in mixer.parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p, generator=gen)
        else:
            nn.init.zeros_(p)


def build_mixer_online_target_pair(seed: int, r_max: int = R_MAX,
                                   central_state_dim: int = CENTRAL_STATE_DIM,
                                   mix_hidden: int = MIX_HIDDEN_DIM):
    """Deterministically initialize an (online_mixer, target_mixer) pair with identical weights.
    Target mixer parameters are frozen (requires_grad=False)."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    online = QMIXMixerV1(r_max, central_state_dim, mix_hidden)
    _init_mixer_(online, gen)
    gen.manual_seed(int(seed))
    target = QMIXMixerV1(r_max, central_state_dim, mix_hidden)
    _init_mixer_(target, gen)
    for p in target.parameters():
        p.requires_grad_(False)
    return online, target
