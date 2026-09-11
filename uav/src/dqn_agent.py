import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

N_SEG_ACTIONS = 4        # KEEP=0 LOCAL=1 PREDICT=2 DIVERSIFY=3
GATE_NO_MEMORY = 0
GATE_MEMORY = 1
IDX_HAS_MEMORY = -1      # trung voi sa_drl_dmoea.IDX_HAS_MEMORY


class HierarchicalQNetwork(nn.Module):
    def __init__(self, state_dim, n_segments, n_seg_actions=N_SEG_ACTIONS,
                 hidden=64):
        super().__init__()
        self.n_segments = n_segments
        self.n_seg_actions = n_seg_actions
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden, 2)
        self.segment_head = nn.Linear(hidden, n_segments * n_seg_actions)

    def forward(self, x):
        """
        x: (B, state_dim)
        return: q_gate (B, 2); adv (B, S, 4) voi mean over actions = 0
        """
        h = self.trunk(x)
        q_gate = self.gate_head(h)
        q_seg = self.segment_head(h).view(-1, self.n_segments,
                                          self.n_seg_actions)
        adv = q_seg - q_seg.mean(dim=2, keepdim=True)
        return q_gate, adv


def q_total_of(q_gate, adv, gate_action, seg_actions):
    """
    Q cua hanh dong hierarchical DA CHON.
    q_gate (B,2)  adv (B,S,4)  gate_action (B,) long  seg_actions (B,S) long
    return: (B,)
    """
    q_mem = q_gate[:, 1]
    seg_val = adv.gather(2, seg_actions.unsqueeze(2)).squeeze(2).mean(dim=1)
    q_no = q_gate[:, 0] + seg_val
    return torch.where(gate_action == 1, q_mem, q_no)


def best_hierarchical(q_gate, adv, has_memory):
    """
    Hanh dong hierarchical TOT NHAT theo Q_total, co mask.
    has_memory: (B,) float 0/1. Mask duy nhat, dung ca select lan target.
    return: gate_best (B,) long, seg_best (B,S) long, q_best (B,)
    """
    seg_best = adv.argmax(dim=2)
    q_no_best = q_gate[:, 0] + adv.max(dim=2).values.mean(dim=1)
    q_mem = q_gate[:, 1].clone()
    q_mem[has_memory < 0.5] = -1e9
    gate_best = (q_mem > q_no_best).long()
    q_best = torch.where(gate_best == 1, q_mem, q_no_best)
    return gate_best, seg_best, q_best


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, gate, seg_actions, reward, next_state, done):
        self.buffer.append((np.asarray(state, dtype=np.float32), int(gate),
                            np.asarray(seg_actions, dtype=np.int64),
                            float(reward),
                            np.asarray(next_state, dtype=np.float32),
                            float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, gates, segs, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(gates, dtype=np.int64),
                np.array(segs, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, n_segments, n_seg_actions=N_SEG_ACTIONS,
                 lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 buffer_size=10000, batch_size=64, target_tau=0.01,
                 eps_fixed=None):
        self.n_segments = n_segments
        self.n_seg_actions = n_seg_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_tau = target_tau   # Polyak; KHONG nham voi tau_t

        # eps_fixed: khong decay — dung cho mode online 1 episode.
        self.eps_fixed = eps_fixed
        self.eps = eps_fixed if eps_fixed is not None else eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = HierarchicalQNetwork(state_dim, n_segments, n_seg_actions)
        self.target = HierarchicalQNetwork(state_dim, n_segments, n_seg_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.step_count = 0

    @torch.no_grad()
    def soft_update_target(self):
        """Polyak: theta_tgt <- (1-tau)*theta_tgt + tau*theta_online."""
        tau = self.target_tau
        for p_tgt, p_src in zip(self.target.parameters(),
                                self.online.parameters()):
            p_tgt.data.mul_(1.0 - tau)
            p_tgt.data.add_(tau * p_src.data)

    def select_action(self, state, training=True):
        """return: (gate: int, seg_actions: np.ndarray (S,) int64)"""
        has_mem = bool(state[IDX_HAS_MEMORY] > 0.5)

        if training and np.random.rand() < self.eps:
            gate = random.choice([0, 1] if has_mem else [0])
            seg = np.random.randint(0, self.n_seg_actions, size=self.n_segments)
            return gate, seg

        with torch.no_grad():
            q_gate, adv = self.online(torch.FloatTensor(state).unsqueeze(0))
            gate_best, seg_best, _ = best_hierarchical(
                q_gate, adv, torch.tensor([float(has_mem)]))
        return int(gate_best[0]), seg_best[0].numpy()

    def learn(self):
        """Mot buoc Double DQN tren Q_total — MOT loss duy nhat."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, gates, segs, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        gates = torch.LongTensor(gates)
        segs = torch.LongTensor(segs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_gate, adv = self.online(states)
        q_current = q_total_of(q_gate, adv, gates, segs)

        with torch.no_grad():
            has_mem_next = next_states[:, IDX_HAS_MEMORY]
            q_gate_on, adv_on = self.online(next_states)
            gate_star, seg_star, _ = best_hierarchical(q_gate_on, adv_on,
                                                       has_mem_next)
            q_gate_tg, adv_tg = self.target(next_states)
            q_next = q_total_of(q_gate_tg, adv_tg, gate_star, seg_star)
            y = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_current, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft target update sau MOI learn step (thay hard-copy moi 100
        # step, vi online run ngan co the khong bao gio dat 100).
        self.soft_update_target()
        self.step_count += 1
        return loss.item()

    def decay_epsilon(self):
        if self.eps_fixed is not None:
            return
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
