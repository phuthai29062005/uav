"""
Branch-C replay: REGIONAL_ANCHOR_JOINT_CONTEXT_REPLAY_V1 (frozen by E4K_replay_anchor_contract).

Sampling unit = REGIONAL_ANCHOR_WITH_JOINT_CONTEXT: one anchor per active regional transition, each
carrying the full joint event context (all fixed slots, masks, requested joint action vector, central
state, per-region R1 rewards, r_joint) for current and next state, plus the focal region's transition
and the terminal flag. Capacity/warmup/update/batch are all counted in REGIONAL ANCHORS (not joint
events), preserving the existing replay-capacity accounting and the regional sampling measure.

Sibling anchors from the same event are NOT scientifically deduplicated: if two anchors from one event
are sampled in one minibatch, both contribute independently (weighting proportional to active-region
count). Physical de-duplication of the shared event context is an implementation detail only.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence

import numpy as np

from prdrl.rl.joint_context import R_MAX, LOCAL_DIM, CENTRAL_STATE_DIM

STATE_DIM = 9
REPLAY_SAMPLING_UNIT = "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"
DEFAULT_CAPACITY = 50_000     # regional anchors
DEFAULT_WARMUP = 1000         # regional anchors
DEFAULT_UPDATE_EVERY = 4      # appended regional anchors
DEFAULT_BATCH = 128           # sampled regional anchors


class JointAnchor:
    """One regional anchor + its full joint event context (current and next)."""

    __slots__ = (
        # FOCAL
        "focal_region_id", "focal_state_9", "focal_action", "focal_R1_reward", "focal_next_state_9",
        # JOINT CURRENT
        "joint_states", "active_mask", "action_vector", "central_state", "regional_rewards", "r_joint",
        # JOINT NEXT
        "next_joint_states", "next_active_mask", "next_central_state",
        # DONE + provenance
        "done", "event_id",
    )

    def __init__(self, *, focal_region_id, focal_state_9, focal_action, focal_R1_reward,
                 focal_next_state_9, joint_states, active_mask, action_vector, central_state,
                 regional_rewards, r_joint, next_joint_states, next_active_mask, next_central_state,
                 done, event_id):
        self.focal_region_id = int(focal_region_id)
        self.focal_state_9 = np.asarray(focal_state_9, dtype=np.float32)
        self.focal_action = int(focal_action)
        self.focal_R1_reward = float(focal_R1_reward)
        self.focal_next_state_9 = np.asarray(focal_next_state_9, dtype=np.float32)
        self.joint_states = np.asarray(joint_states, dtype=np.float32)
        self.active_mask = np.asarray(active_mask, dtype=np.float32)
        self.action_vector = np.asarray(action_vector, dtype=np.int64)
        self.central_state = np.asarray(central_state, dtype=np.float32)
        self.regional_rewards = np.asarray(regional_rewards, dtype=np.float32)
        self.r_joint = float(r_joint)
        self.next_joint_states = np.asarray(next_joint_states, dtype=np.float32)
        self.next_active_mask = np.asarray(next_active_mask, dtype=np.float32)
        self.next_central_state = np.asarray(next_central_state, dtype=np.float32)
        self.done = bool(done)
        self.event_id = str(event_id)
        self._validate()

    def _validate(self):
        assert self.joint_states.shape == (R_MAX, STATE_DIM), self.joint_states.shape
        assert self.active_mask.shape == (R_MAX,), self.active_mask.shape
        assert self.action_vector.shape == (R_MAX,), self.action_vector.shape
        assert self.central_state.shape == (CENTRAL_STATE_DIM,), self.central_state.shape
        assert self.regional_rewards.shape == (R_MAX,), self.regional_rewards.shape
        assert self.next_joint_states.shape == (R_MAX, STATE_DIM), self.next_joint_states.shape
        assert self.next_active_mask.shape == (R_MAX,), self.next_active_mask.shape
        assert self.next_central_state.shape == (CENTRAL_STATE_DIM,), self.next_central_state.shape
        assert self.focal_state_9.shape == (STATE_DIM,), self.focal_state_9.shape
        assert self.focal_next_state_9.shape == (STATE_DIM,), self.focal_next_state_9.shape


def make_event_anchors(*, active_regions: Sequence[int], joint_states: np.ndarray,
                       active_mask: np.ndarray, action_vector: np.ndarray, central_state: np.ndarray,
                       regional_rewards: np.ndarray, r_joint: float, next_joint_states: np.ndarray,
                       next_active_mask: np.ndarray, next_central_state: np.ndarray, done: bool,
                       event_id: str) -> List[JointAnchor]:
    """Create exactly len(active_regions) anchors sharing one joint event context — one focal anchor
    per active regional transition (scientific sampling measure = regional anchors)."""
    anchors: List[JointAnchor] = []
    for r in sorted(active_regions):
        anchors.append(JointAnchor(
            focal_region_id=r, focal_state_9=joint_states[r], focal_action=int(action_vector[r]),
            focal_R1_reward=float(regional_rewards[r]), focal_next_state_9=next_joint_states[r],
            joint_states=joint_states, active_mask=active_mask, action_vector=action_vector,
            central_state=central_state, regional_rewards=regional_rewards, r_joint=r_joint,
            next_joint_states=next_joint_states, next_active_mask=next_active_mask,
            next_central_state=next_central_state, done=done, event_id=event_id))
    return anchors


class JointReplayBuffer:
    """Circular replay buffer of REGIONAL ANCHORS (deterministic sampling under a fixed seed)."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY, rng_seed: int = 9999,
                 warmup: int = DEFAULT_WARMUP, update_every: int = DEFAULT_UPDATE_EVERY):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf: deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng(rng_seed)
        self.capacity = capacity
        self.warmup = warmup
        self.update_every = update_every
        self.appended = 0                      # total anchors ever appended (counts anchors, not events)

    def push(self, anchor: JointAnchor) -> None:
        self._buf.append(anchor)
        self.appended += 1

    def push_event(self, anchors: Sequence[JointAnchor]) -> None:
        for a in anchors:
            self.push(a)

    def __len__(self) -> int:
        return len(self._buf)

    def ready(self) -> bool:
        return len(self._buf) >= self.warmup

    def should_update(self) -> bool:
        """Update trigger every `update_every` appended anchors, once warmed up."""
        return self.ready() and (self.appended % self.update_every == 0)

    def sample(self, batch_size: int = DEFAULT_BATCH) -> List[JointAnchor]:
        if len(self._buf) < batch_size:
            raise ValueError(f"buffer has {len(self._buf)} anchors, need {batch_size}")
        idx = self._rng.integers(0, len(self._buf), size=batch_size)
        # NB: sibling anchors from the same event are NOT deduplicated (intentional).
        return [self._buf[i] for i in idx]

    def sample_indices(self, indices: Sequence[int]) -> List[JointAnchor]:
        """Deterministic index-based sampling (for fixtures that force sibling co-occurrence)."""
        return [self._buf[i] for i in indices]
