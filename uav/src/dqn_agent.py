import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

N_ACTIONS = 5


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_segments, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.n_segments = n_segments
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_segments * n_actions)
        )
        
    def forward(self, x):
        """
        x: (batch, state_dim)
        return: (batch, n_segments, n_actions)
        """
        batch_size = x.size(0)
        out = self.net(x)  # (batch, n_segments * n_actions)
        out = out.view(batch_size, self.n_segments, self.n_actions)  # (batch, n_segments, n_actions)
        return out
    

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_dim, n_segments, n_actions=N_ACTIONS,
                 lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100,
                 eps_fixed=None):

        self.n_segments = n_segments
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # eps_fixed: gia tri epsilon co dinh (khong decay) — dung cho
        # mode online chi co 1 episode. None => decay per-episode nhu cu.
        self.eps_fixed = eps_fixed
        if eps_fixed is not None:
            self.eps = eps_fixed
        else:
            self.eps = eps_start
        self.eps_end = eps_end      # chi y nghia khi eps_fixed is None
        self.eps_decay = eps_decay  # chi y nghia khi eps_fixed is None
        
        #2 network
        self.online = QNetwork(state_dim, n_segments, n_actions)
        self.target = QNetwork(state_dim, n_segments, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)  
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.step_count = 0
        
    def select_action(self, state, training = True):
        """
        state: np.array (state_dim,)
        return: np.array (n_segments,) — hành động cho từng segment
        """
        if training and np.random.rand() < self.eps:
            return np.random.randint(0, self.n_actions, size=self.n_segments)
        
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q = self.online(s)
            actions = q.argmax(dim=2).squeeze(0).numpy()
        return actions
    
    def learn(self):
        
        """Một bước cập nhật Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states      = torch.FloatTensor(states)        # (B, state_dim)
        actions     = torch.LongTensor(actions)        # (B, S)
        rewards     = torch.FloatTensor(rewards)       # (B,)
        next_states = torch.FloatTensor(next_states)   # (B, state_dim)
        dones       = torch.FloatTensor(dones)         # (B,)
        
        # --- Q hiện tại: lấy Q của hành động ĐÃ CHỌN ---
        q_all = self.online(states)                        # (B, S, 5)
        q_current = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, S)
        
        # --- Double DQN target ---
        with torch.no_grad():
            # 1. ONLINE chọn hành động tốt nhất ở next_state
            next_actions = self.online(next_states).argmax(dim=2)
            
            # 2. TARGET đánh giá hành động đó
            q_next_all = self.target(next_states)          # (B, S, 5)
            q_next = q_next_all.gather(2, next_actions.unsqueeze(2)).squeeze(2)  # (B, S)
            
            # 3. Công thức Bellman (reward chia đều cho các segment)
            r = rewards.unsqueeze(1)                       # (B, 1)
            d = dones.unsqueeze(1)                         # (B, 1)
            q_target = r + self.gamma * q_next * (1 - d)
            
        
                # --- Loss & cập nhật ---
        loss = nn.functional.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # --- Cập nhật target network định kỳ ---
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())  # copy online → target
        return loss.item()

    def decay_epsilon(self):
        if self.eps_fixed is not None:
            return
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        


