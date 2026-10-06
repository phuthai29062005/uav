"""
E4-P: Shared replay buffer for parameter-shared regional Double DQN.

Design constraints (from H1-RV + spec):
  - Each transition retains event_id so regions from the same
    joint event can be identified for coordination analysis.
  - Oracle labels are NOT stored as training targets.
    They may exist only in offline diagnostic metadata.
  - No greedy oracle action in the training path.
  - Deterministic sampling under a fixed seed.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class RegionalTransition:
    """One region's transition at one environmental change."""

    __slots__ = (
        "state",
        "requested_action",       # action the controller requested (int index)
        "realized_operator",      # actual operator used (may differ if fallback)
        "reward",
        "next_state",
        "done",
        "region_id",
        "event_id",
        "fallback_applied",
        "response_fe",
        # Offline diagnostic only — NOT used in training
        "diagnostic_oracle_action",
    )

    def __init__(
        self,
        state: np.ndarray,
        requested_action: int,
        realized_operator: str,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        region_id: int,
        event_id: str,
        fallback_applied: bool,
        response_fe: int,
        diagnostic_oracle_action: Optional[str] = None,
    ):
        self.state             = np.asarray(state,      dtype=np.float32)
        self.requested_action  = int(requested_action)
        self.realized_operator = str(realized_operator)
        self.reward            = float(reward)
        self.next_state        = np.asarray(next_state, dtype=np.float32)
        self.done              = bool(done)
        self.region_id         = int(region_id)
        self.event_id          = str(event_id)
        self.fallback_applied  = bool(fallback_applied)
        self.response_fe       = int(response_fe)
        # Oracle label stored ONLY as offline diagnostic metadata, never as target
        self.diagnostic_oracle_action = diagnostic_oracle_action


class SharedReplayBuffer:
    """
    Circular replay buffer for regional transitions.

    Sampling is deterministic under a fixed rng_seed.
    All transitions share the same buffer regardless of region.
    event_id is retained for coordination analysis.
    """

    def __init__(self, capacity: int, rng_seed: int = 9999):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf: deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng(rng_seed)
        self.capacity  = capacity
        self.rng_seed  = rng_seed

    def push(self, transition: RegionalTransition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> List[RegionalTransition]:
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buf)} transitions, "
                f"need {batch_size}"
            )
        idx = self._rng.integers(0, len(self._buf), size=batch_size)
        return [self._buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)

    def event_ids(self) -> List[str]:
        return [t.event_id for t in self._buf]

    def to_arrays(self, batch: List[RegionalTransition]):
        """Convert a batch of RegionalTransition to numpy arrays."""
        import numpy as np
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.requested_action for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
