"""
E4-P: Epsilon-greedy agent for parameter-shared regional Double DQN.

Design:
  - Same epsilon schedule for all regions at one event.
  - Independent per-region action RNG (seeded by event_id + region_id).
  - Does NOT access oracle labels, oracle utility, or regime labels.
  - Production response RNG is completely isolated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS

ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_ORDER)}
IDX_TO_ACTION: Dict[int, str] = {i: a for i, a in enumerate(ACTION_ORDER)}


class EpsilonGreedyConfig:
    """
    Epsilon schedule: linear decay from eps_start to eps_end over n_steps.
    Indexed by TRANSITION COUNT (not episode count).

    Predeclared defaults (UNFROZEN — not empirically tuned in E4-P):
        eps_start = 1.0
        eps_end   = 0.05
        n_steps   = to be specified in training protocol
    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_end:   float = 0.05,
        n_steps:   int   = 50_000,
    ):
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.n_steps   = n_steps

    def get_epsilon(self, n_transitions: int) -> float:
        frac = min(n_transitions / max(self.n_steps, 1), 1.0)
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))

    def to_dict(self) -> dict:
        return {
            "eps_start": self.eps_start,
            "eps_end":   self.eps_end,
            "n_steps":   self.n_steps,
            "schedule":  "linear_transition_count",
            "status":    "UNFROZEN_PREDECLARED_DEFAULTS",
        }


def select_regional_actions(
    online_net: SharedRegionalQNetwork,
    local_states: Dict[int, np.ndarray],
    global_context: np.ndarray,
    epsilon: float,
    event_seed: int,
    device: torch.device,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Epsilon-greedy action selection for all active regions at one event.

    Each region uses its OWN RNG key (event_seed XOR region_id * prime),
    ensuring independent exploration even at the same epsilon.

    Returns:
        action_indices:  {region_id: int action index}
        action_strings:  {region_id: str action name}
    """
    if not local_states:
        return {}, {}

    PRIME = 1_000_003   # arbitrary prime for namespace separation

    action_indices: Dict[int, int] = {}
    action_strings: Dict[int, str] = {}

    # Build batch for greedy pass (all regions at once)
    region_ids_ordered = sorted(local_states.keys())
    inputs = []
    for r in region_ids_ordered:
        obs = np.concatenate([local_states[r], global_context], axis=0)
        inputs.append(obs)
    x = torch.tensor(np.stack(inputs), dtype=torch.float32, device=device)

    with torch.no_grad():
        q_values = online_net(x)   # (n_regions, 4)

    greedy_actions = q_values.argmax(dim=1).cpu().numpy()   # (n_regions,)

    for i, r in enumerate(region_ids_ordered):
        region_rng = np.random.default_rng(event_seed ^ (r * PRIME))
        if region_rng.random() < epsilon:
            action_idx = int(region_rng.integers(0, N_ACTIONS))
        else:
            action_idx = int(greedy_actions[i])
        action_indices[r] = action_idx
        action_strings[r] = IDX_TO_ACTION[action_idx]

    return action_indices, action_strings
