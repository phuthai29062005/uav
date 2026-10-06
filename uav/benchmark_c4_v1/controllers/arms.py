"""Benchmark controller arms: B-C1, B-C3, B-C4.

These port the exact C4 controller mechanism (same architecture, gate formula,
0 trainable gate parameters) to work on standard DMOP benchmarks with 2-region
segmentation.

Architecture (frozen, identical to core):
  Unary: SharedRegionalQNetwork 9→128→128→4
  Pair:  SharedPairNetwork 18→128→128→16

B-C1: Independent greedy argmax per region (no coordination)
B-C3: Pairwise CG with MaxSum K=10 (full coordination)
B-C4: Gated CG with deterministic gate g_ij = 0.5*(|d_i|+|d_j|) where d_r = |state_r[3]|
"""

from __future__ import annotations

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair, N_ACTIONS, ACTION_ORDER,
    polyak_update, double_dqn_target,
)
from prdrl.rl.coordination_graph import (
    SharedPairNetwork, build_pair_online_target, derive_pair_seed,
    maxsum_inference, joint_value_cg, c3_double_q_target, c3_current_q, c3_loss,
    maxsum_inference_c4, joint_value_c4, c4_double_q_target, c4_current_q, c4_loss,
    compute_pair_gate, build_complete_graph,
    maxsum_inference_batched, maxsum_inference_c4_batched,
    c3_double_q_target_batched, c3_current_q_batched,
    c4_double_q_target_batched, c4_current_q_batched,
)
from prdrl.rl.qmix import (
    QMIXMixerV1, build_mixer_online_target_pair, derive_mixer_seed,
)

STATE_DIM = 9
R_MAX = 10
DEVICE = torch.device("cpu")
NETWORK_INPUT_DIM = 9
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
WARMUP = 64
UPDATE_EVERY = 2
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_TRANSITIONS = 50_000
K_MAXSUM = 10


class BenchmarkReplayBuffer:
    """Simple replay buffer for benchmark training (joint transitions)."""

    def __init__(self, capacity: int = REPLAY_CAPACITY, seed: int = 9999):
        self._buf = []
        self._pos = 0
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)

    def push(self, transition: dict):
        if len(self._buf) < self._capacity:
            self._buf.append(transition)
        else:
            self._buf[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> List[dict]:
        idx = self._rng.integers(0, len(self._buf), size=batch_size)
        return [self._buf[i] for i in idx]

    def __len__(self):
        return len(self._buf)


def _epsilon(step: int) -> float:
    frac = min(1.0, step / EPS_DECAY_TRANSITIONS)
    return EPS_START + (EPS_END - EPS_START) * frac


class BenchmarkC1Arm:
    """B-C1: Independent Double DQN per region (QMIX mixer for training)."""

    def __init__(self, policy_seed: int):
        self.policy_seed = policy_seed
        init_seed = policy_seed * 1000
        self.online, self.target = build_online_target_pair(NETWORK_INPUT_DIM, seed=init_seed)
        mixer_seed = derive_mixer_seed(init_seed)
        self.mixer_online, self.mixer_target = build_mixer_online_target_pair(mixer_seed)
        self.optimizer = torch.optim.Adam(
            list(self.online.parameters()) + list(self.mixer_online.parameters()), lr=LR
        )
        self.replay = BenchmarkReplayBuffer(REPLAY_CAPACITY, seed=policy_seed)
        self.step_count = 0

    def select_actions(self, states: Dict[int, np.ndarray], rng: np.random.Generator) -> Dict[int, int]:
        eps = _epsilon(self.step_count)
        actions = {}
        for r_id, state in states.items():
            if rng.random() < eps:
                actions[r_id] = rng.integers(0, N_ACTIONS)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q = self.online(s)
                    actions[r_id] = int(q.argmax(dim=1).item())
        return actions

    def select_actions_greedy(self, states: Dict[int, np.ndarray]) -> Dict[int, int]:
        actions = {}
        for r_id, state in states.items():
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q = self.online(s)
                actions[r_id] = int(q.argmax(dim=1).item())
        return actions

    def store_transition(self, transition: dict):
        self.replay.push(transition)
        self.step_count += 1

    def update(self) -> Optional[float]:
        if len(self.replay) < WARMUP:
            return None
        if self.step_count % UPDATE_EVERY != 0:
            return None

        batch = self.replay.sample(BATCH_SIZE)
        states = torch.tensor(np.array([t["states"] for t in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([t["actions"] for t in batch]), dtype=torch.long)
        rewards = torch.tensor(np.array([t["rewards"] for t in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t["next_states"] for t in batch]), dtype=torch.float32)
        active_mask = torch.tensor(np.array([t["active_mask"] for t in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([t["done"] for t in batch]), dtype=torch.float32)
        central_state = torch.tensor(np.array([t["central_state"] for t in batch]), dtype=torch.float32)

        B, R, D = states.shape
        q_all = self.online(states.reshape(B * R, D)).reshape(B, R, N_ACTIONS)
        q_sel = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)
        q_sel = q_sel * active_mask
        q_tot = self.mixer_online(q_sel, central_state)

        with torch.no_grad():
            q_next_all = self.online(next_states.reshape(B * R, D)).reshape(B, R, N_ACTIONS)
            a_star = q_next_all.argmax(dim=2)
            q_target_all = self.target(next_states.reshape(B * R, D)).reshape(B, R, N_ACTIONS)
            q_target_sel = q_target_all.gather(2, a_star.unsqueeze(2)).squeeze(2)
            q_target_sel = q_target_sel * active_mask
            q_tot_target = self.mixer_target(q_target_sel, central_state)
            r_joint = rewards.sum(dim=1, keepdim=True) / active_mask.sum(dim=1, keepdim=True).clamp(min=1)
            y = r_joint + GAMMA * (1.0 - dones.unsqueeze(1)) * q_tot_target

        loss = torch.nn.SmoothL1Loss()(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        polyak_update(self.online, self.target, TAU)
        polyak_update(self.mixer_online, self.mixer_target, TAU)
        return float(loss.item())


class BenchmarkC3Arm:
    """B-C3: Pairwise CG with MaxSum K=10."""

    def __init__(self, policy_seed: int):
        self.policy_seed = policy_seed
        init_seed = policy_seed * 1000
        self.online_unary, self.target_unary = build_online_target_pair(NETWORK_INPUT_DIM, seed=init_seed)
        pair_seed = derive_pair_seed(init_seed)
        self.online_pair, self.target_pair = build_pair_online_target(pair_seed)
        self.optimizer = torch.optim.Adam(
            list(self.online_unary.parameters()) + list(self.online_pair.parameters()), lr=LR
        )
        self.replay = BenchmarkReplayBuffer(REPLAY_CAPACITY, seed=policy_seed)
        self.step_count = 0

    def select_actions(self, states: Dict[int, np.ndarray], active_mask: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        eps = _epsilon(self.step_count)
        if rng.random() < eps:
            actions = rng.integers(0, N_ACTIONS, size=R_MAX)
            actions = actions * (active_mask > 0.5).astype(int)
            return actions

        joint_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
        for r_id, state in states.items():
            joint_states[r_id] = state

        with torch.no_grad():
            js = torch.tensor(joint_states, dtype=torch.float32).unsqueeze(0)
            am = torch.tensor(active_mask, dtype=torch.float32).unsqueeze(0)
            actions = maxsum_inference(self.online_unary, self.online_pair, js, am, K_MAXSUM)
        return actions.squeeze(0).numpy()

    def select_actions_greedy(self, states: Dict[int, np.ndarray],
                              active_mask: np.ndarray) -> np.ndarray:
        joint_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
        for r_id, state in states.items():
            joint_states[r_id] = state

        with torch.no_grad():
            js = torch.tensor(joint_states, dtype=torch.float32).unsqueeze(0)
            am = torch.tensor(active_mask, dtype=torch.float32).unsqueeze(0)
            actions = maxsum_inference(self.online_unary, self.online_pair, js, am, K_MAXSUM)
        return actions.squeeze(0).numpy()

    def store_transition(self, transition: dict):
        self.replay.push(transition)
        self.step_count += 1

    def update(self) -> Optional[float]:
        if len(self.replay) < WARMUP:
            return None
        if self.step_count % UPDATE_EVERY != 0:
            return None

        batch = self.replay.sample(BATCH_SIZE)
        joint_states = torch.tensor(np.array([t["joint_states"] for t in batch]), dtype=torch.float32)
        action_vectors = torch.tensor(np.array([t["action_vector"] for t in batch]), dtype=torch.long)
        active_masks = torch.tensor(np.array([t["active_mask"] for t in batch]), dtype=torch.float32)
        r_joints = torch.tensor(np.array([t["r_joint"] for t in batch]), dtype=torch.float32)
        next_joint_states = torch.tensor(np.array([t["next_joint_states"] for t in batch]), dtype=torch.float32)
        next_active_masks = torch.tensor(np.array([t["next_active_mask"] for t in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([t["done"] for t in batch]), dtype=torch.float32)

        q_current = c3_current_q_batched(self.online_unary, self.online_pair,
                                          joint_states, action_vectors, active_masks)
        y = c3_double_q_target_batched(self.online_unary, self.online_pair,
                                        self.target_unary, self.target_pair,
                                        next_joint_states, next_active_masks,
                                        r_joints, dones, GAMMA)
        loss = c3_loss(q_current, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        polyak_update(self.online_unary, self.target_unary, TAU)
        polyak_update(self.online_pair, self.target_pair, TAU)
        return float(loss.item())


class BenchmarkC4Arm:
    """B-C4: Gated CG with deterministic gate (0 trainable gate parameters).

    Gate: g_ij = 0.5 * (|d_i| + |d_j|) where d_r = abs(state_r[3]).
    Identical to C3 except pair potentials are multiplied by g_ij.
    """

    def __init__(self, policy_seed: int):
        self.policy_seed = policy_seed
        init_seed = policy_seed * 1000
        self.online_unary, self.target_unary = build_online_target_pair(NETWORK_INPUT_DIM, seed=init_seed)
        pair_seed = derive_pair_seed(init_seed)
        self.online_pair, self.target_pair = build_pair_online_target(pair_seed)
        self.optimizer = torch.optim.Adam(
            list(self.online_unary.parameters()) + list(self.online_pair.parameters()), lr=LR
        )
        self.replay = BenchmarkReplayBuffer(REPLAY_CAPACITY, seed=policy_seed)
        self.step_count = 0

    def select_actions(self, states: Dict[int, np.ndarray], active_mask: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        eps = _epsilon(self.step_count)
        if rng.random() < eps:
            actions = rng.integers(0, N_ACTIONS, size=R_MAX)
            actions = actions * (active_mask > 0.5).astype(int)
            return actions

        joint_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
        for r_id, state in states.items():
            joint_states[r_id] = state

        with torch.no_grad():
            js = torch.tensor(joint_states, dtype=torch.float32).unsqueeze(0)
            am = torch.tensor(active_mask, dtype=torch.float32).unsqueeze(0)
            actions = maxsum_inference_c4(self.online_unary, self.online_pair, js, am, K_MAXSUM)
        return actions.squeeze(0).numpy()

    def select_actions_greedy(self, states: Dict[int, np.ndarray],
                              active_mask: np.ndarray) -> np.ndarray:
        joint_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
        for r_id, state in states.items():
            joint_states[r_id] = state

        with torch.no_grad():
            js = torch.tensor(joint_states, dtype=torch.float32).unsqueeze(0)
            am = torch.tensor(active_mask, dtype=torch.float32).unsqueeze(0)
            actions = maxsum_inference_c4(self.online_unary, self.online_pair, js, am, K_MAXSUM)
        return actions.squeeze(0).numpy()

    def store_transition(self, transition: dict):
        self.replay.push(transition)
        self.step_count += 1

    def update(self) -> Optional[float]:
        if len(self.replay) < WARMUP:
            return None
        if self.step_count % UPDATE_EVERY != 0:
            return None

        batch = self.replay.sample(BATCH_SIZE)
        joint_states = torch.tensor(np.array([t["joint_states"] for t in batch]), dtype=torch.float32)
        action_vectors = torch.tensor(np.array([t["action_vector"] for t in batch]), dtype=torch.long)
        active_masks = torch.tensor(np.array([t["active_mask"] for t in batch]), dtype=torch.float32)
        r_joints = torch.tensor(np.array([t["r_joint"] for t in batch]), dtype=torch.float32)
        next_joint_states = torch.tensor(np.array([t["next_joint_states"] for t in batch]), dtype=torch.float32)
        next_active_masks = torch.tensor(np.array([t["next_active_mask"] for t in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([t["done"] for t in batch]), dtype=torch.float32)

        q_current = c4_current_q_batched(self.online_unary, self.online_pair,
                                          joint_states, action_vectors, active_masks)
        y = c4_double_q_target_batched(self.online_unary, self.online_pair,
                                        self.target_unary, self.target_pair,
                                        next_joint_states, next_active_masks,
                                        r_joints, dones, GAMMA)
        loss = c4_loss(q_current, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        polyak_update(self.online_unary, self.target_unary, TAU)
        polyak_update(self.online_pair, self.target_pair, TAU)
        return float(loss.item())
