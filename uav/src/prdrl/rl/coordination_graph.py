"""
C3 treatment: PAIRWISE_COORDINATION_GRAPH_V1 (frozen by E4R).

Pairwise coordination graph with max-sum message passing for joint action inference.
Replaces QMIX's centralized mixer with explicit pairwise interaction factors over a
complete active-region graph. The unary network is architecturally identical to C1's
utility network (SharedRegionalQNetwork, 9→128→128→4).

    Q_CG(s,a) = (1/n) sum_i U_i(s_i,a_i) + (1/|E|) sum_{(i,j)} P_ij(s_i,s_j,a_i,a_j)

where P_ij is obtained by passing concat(s_i,s_j) through a shared pair network
(18→128→128→16), reshaping to 4×4, symmetrizing, and centering to zero row/column means.

Max-sum message passing (K=10 synchronous iterations) computes approximate coordinated
greedy joint actions. For n>=3 the complete graph is loopy and inference is APPROXIMATE.

Stage: E4RI IMPLEMENTATION VALIDATION. No fresh seeds accessed.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from prdrl.rl.network import SharedRegionalQNetwork, N_ACTIONS

R_MAX = 10
STATE_DIM = 9
PAIR_INPUT_DIM = 18
PAIR_OUTPUT_DIM = 16
PAIR_HIDDEN_1 = 128
PAIR_HIDDEN_2 = 128
K_MAXSUM = 10
PAIR_SEED_OFFSET = 1
BETA_PAIR = 1
TREATMENT_ARCH_C3 = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
SMALL_N_EXACT_THRESHOLD = 5


class SharedPairNetwork(nn.Module):
    """Shared pairwise interaction network: concat(s_i,s_j) -> 16 -> reshape 4x4."""

    def __init__(self, input_dim: int = PAIR_INPUT_DIM,
                 hidden1: int = PAIR_HIDDEN_1, hidden2: int = PAIR_HIDDEN_2,
                 output_dim: int = PAIR_OUTPUT_DIM):
        super().__init__()
        if input_dim != PAIR_INPUT_DIM:
            raise ValueError(f"pair input_dim must be {PAIR_INPUT_DIM}, got {input_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 18) -> (batch, 16)."""
        return self.net(x)


def _init_module(module: nn.Module, gen: torch.Generator) -> None:
    for p in module.parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p, generator=gen)
        else:
            nn.init.zeros_(p)


def derive_pair_seed(utility_initialization_seed: int) -> int:
    return int(utility_initialization_seed) + PAIR_SEED_OFFSET


def build_pair_online_target(seed: int) -> Tuple[SharedPairNetwork, SharedPairNetwork]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    online = SharedPairNetwork()
    _init_module(online, gen)
    gen.manual_seed(int(seed))
    target = SharedPairNetwork()
    _init_module(target, gen)
    for p in target.parameters():
        p.requires_grad_(False)
    return online, target


# ---------------------------------------------------------------------------
# Raw pair matrix, symmetrization, centering
# ---------------------------------------------------------------------------

def raw_pair_matrix(pair_net: SharedPairNetwork, s_i: torch.Tensor,
                    s_j: torch.Tensor) -> torch.Tensor:
    """Compute H_ij = pair_net(concat(s_i, s_j)), reshaped to (..., 4, 4).
    s_i, s_j: (..., 9). Returns (..., 4, 4)."""
    inp = torch.cat([s_i, s_j], dim=-1)
    leading = inp.shape[:-1]
    flat = inp.reshape(-1, PAIR_INPUT_DIM)
    out = pair_net(flat).reshape(*leading, N_ACTIONS, N_ACTIONS)
    return out


def symmetrize_pair(h_ij: torch.Tensor, h_ji: torch.Tensor) -> torch.Tensor:
    """M_ij = 0.5 * (H_ij + H_ji^T). h_ij, h_ji: (..., 4, 4) -> (..., 4, 4)."""
    return 0.5 * (h_ij + h_ji.transpose(-2, -1))


def center_pair(m: torch.Tensor) -> torch.Tensor:
    """Zero row/column mean centering. m: (..., A, A) -> P: (..., A, A).
    P[a,b] = M[a,b] - row_mean[a] - col_mean[b] + grand_mean."""
    row_mean = m.mean(dim=-1, keepdim=True)
    col_mean = m.mean(dim=-2, keepdim=True)
    grand_mean = m.mean(dim=(-2, -1), keepdim=True)
    return m - row_mean - col_mean + grand_mean


def compute_centered_pair(pair_net: SharedPairNetwork, s_i: torch.Tensor,
                          s_j: torch.Tensor) -> torch.Tensor:
    """Full pipeline: raw -> symmetrize -> center. Returns P_ij (..., 4, 4)."""
    h_ij = raw_pair_matrix(pair_net, s_i, s_j)
    h_ji = raw_pair_matrix(pair_net, s_j, s_i)
    m_ij = symmetrize_pair(h_ij, h_ji)
    return center_pair(m_ij)


# ---------------------------------------------------------------------------
# Complete graph construction
# ---------------------------------------------------------------------------

def build_complete_graph(active_mask: torch.Tensor) -> List[Tuple[int, int]]:
    """Build edge list for complete undirected graph over active regions.
    active_mask: (R_MAX,) binary. Returns list of (i,j) with i<j, both active."""
    active_indices = torch.where(active_mask > 0.5)[0].tolist()
    return list(combinations(active_indices, 2))


def build_complete_graph_batch(active_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """active_mask: (B, R_MAX). Returns list of edge lists per batch element."""
    return [build_complete_graph(active_mask[b]) for b in range(active_mask.shape[0])]


# ---------------------------------------------------------------------------
# Joint value computation
# ---------------------------------------------------------------------------

def joint_value_cg(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                   joint_states: torch.Tensor, action_vector: torch.Tensor,
                   active_mask: torch.Tensor) -> torch.Tensor:
    """Compute Q_CG for a given joint action vector.
    joint_states: (B, R_MAX, 9), action_vector: (B, R_MAX) int,
    active_mask: (B, R_MAX) binary -> Q_CG: (B,)."""
    B, R, D = joint_states.shape
    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, N_ACTIONS)
    q_sel = q_all.gather(2, action_vector.long().unsqueeze(2)).squeeze(2)
    q_sel = q_sel * active_mask
    n_active = active_mask.sum(dim=1).clamp(min=1)
    q_unary = q_sel.sum(dim=1) / n_active

    q_pair = torch.zeros(B, dtype=joint_states.dtype, device=joint_states.device)
    for b in range(B):
        edges = build_complete_graph(active_mask[b])
        if len(edges) == 0:
            continue
        pair_sum = torch.tensor(0.0, dtype=joint_states.dtype, device=joint_states.device)
        for (i, j) in edges:
            p_ij = compute_centered_pair(pair_net, joint_states[b, i:i+1],
                                         joint_states[b, j:j+1])
            a_i = action_vector[b, i].long()
            a_j = action_vector[b, j].long()
            pair_sum = pair_sum + p_ij[0, a_i, a_j]
        q_pair[b] = pair_sum / len(edges)

    return q_unary + BETA_PAIR * q_pair


# ---------------------------------------------------------------------------
# Max-sum message passing
# ---------------------------------------------------------------------------

def maxsum_inference(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                     joint_states: torch.Tensor, active_mask: torch.Tensor,
                     k: int = K_MAXSUM) -> torch.Tensor:
    """Max-sum message passing for coordinated greedy joint action.
    joint_states: (B, R_MAX, 9), active_mask: (B, R_MAX) -> actions: (B, R_MAX) int."""
    B, R, D = joint_states.shape
    device = joint_states.device
    dtype = joint_states.dtype
    actions = torch.zeros(B, R, dtype=torch.long, device=device)

    for b in range(B):
        active_idx = torch.where(active_mask[b] > 0.5)[0].tolist()
        n = len(active_idx)
        if n == 0:
            continue
        if n == 1:
            i = active_idx[0]
            q = unary_net(joint_states[b, i:i+1])
            actions[b, i] = q.argmax(dim=1).item()
            continue

        edges = list(combinations(active_idx, 2))
        n_edges = len(edges)
        idx_map = {r: pos for pos, r in enumerate(active_idx)}

        states_active = joint_states[b, active_idx]
        q_unary_all = unary_net(states_active)

        phi = q_unary_all / n

        pair_factors = {}
        for (i, j) in edges:
            p_ij = compute_centered_pair(pair_net, joint_states[b, i:i+1],
                                         joint_states[b, j:j+1])
            pair_factors[(i, j)] = p_ij.squeeze(0) / n_edges

        neighbors = {r: [] for r in active_idx}
        for (i, j) in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        msgs = {}
        for i in active_idx:
            for j in neighbors[i]:
                msgs[(i, j)] = torch.zeros(N_ACTIONS, dtype=dtype, device=device)

        for iteration in range(k):
            new_msgs = {}
            for i in active_idx:
                for j in neighbors[i]:
                    key_ij = (min(i, j), max(i, j))
                    if i < j:
                        psi = pair_factors[key_ij]
                    else:
                        psi = pair_factors[key_ij].T

                    incoming = torch.zeros(N_ACTIONS, dtype=dtype, device=device)
                    for h in neighbors[i]:
                        if h != j:
                            incoming = incoming + msgs[(h, i)]

                    phi_i = phi[idx_map[i]]
                    table = phi_i.unsqueeze(1) + psi + incoming.unsqueeze(1)
                    msg = table.max(dim=0).values
                    msg = msg - msg.max()
                    new_msgs[(i, j)] = msg

            msgs = new_msgs

        for i in active_idx:
            belief = phi[idx_map[i]].clone()
            for j in neighbors[i]:
                belief = belief + msgs[(j, i)]
            actions[b, i] = belief.argmax().item()

    return actions


# ---------------------------------------------------------------------------
# Small-n exact diagnostic (DESCRIPTIVE ONLY)
# ---------------------------------------------------------------------------

def exact_joint_argmax(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                       joint_states: torch.Tensor, active_mask: torch.Tensor,
                       b: int) -> Tuple[torch.Tensor, float]:
    """Exhaustive enumeration for one batch element. n_active must be <= SMALL_N_EXACT_THRESHOLD.
    Returns (best_action_vector [R_MAX], best_Q_CG scalar)."""
    active_idx = torch.where(active_mask[b] > 0.5)[0].tolist()
    n = len(active_idx)
    if n > SMALL_N_EXACT_THRESHOLD:
        raise ValueError(f"exact enumeration blocked for n={n} > {SMALL_N_EXACT_THRESHOLD}")
    if n == 0:
        return torch.zeros(R_MAX, dtype=torch.long, device=joint_states.device), 0.0

    best_q = float('-inf')
    best_actions = None
    total = N_ACTIONS ** n
    states_single = joint_states[b:b+1]
    mask_single = active_mask[b:b+1]

    for combo_idx in range(total):
        action_vec = torch.zeros(1, R_MAX, dtype=torch.long, device=joint_states.device)
        idx = combo_idx
        for pos, r in enumerate(active_idx):
            action_vec[0, r] = idx % N_ACTIONS
            idx //= N_ACTIONS
        with torch.no_grad():
            q = joint_value_cg(unary_net, pair_net, states_single, action_vec, mask_single)
        q_val = q.item()
        if q_val > best_q or (q_val == best_q and best_actions is None):
            if q_val > best_q:
                best_q = q_val
                best_actions = action_vec[0].clone()
            elif _action_vec_lt(action_vec[0], best_actions, active_idx):
                best_actions = action_vec[0].clone()

    return best_actions, best_q


def _action_vec_lt(a: torch.Tensor, b: torch.Tensor, active_idx: List[int]) -> bool:
    """Tie-breaking: prefer lower action indices (lexicographic over active slots)."""
    for r in active_idx:
        if a[r] < b[r]:
            return True
        if a[r] > b[r]:
            return False
    return False


def small_n_diagnostic(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                       joint_states: torch.Tensor, active_mask: torch.Tensor,
                       maxsum_actions: torch.Tensor, b: int) -> Dict:
    """Compute inference regret for one batch element. DESCRIPTIVE ONLY."""
    exact_actions, exact_q = exact_joint_argmax(unary_net, pair_net, joint_states, active_mask, b)
    with torch.no_grad():
        ms_q = joint_value_cg(unary_net, pair_net, joint_states[b:b+1],
                              maxsum_actions[b:b+1], active_mask[b:b+1]).item()
    regret = exact_q - ms_q
    match = torch.equal(exact_actions[active_mask[b] > 0.5].long(),
                        maxsum_actions[b][active_mask[b] > 0.5].long())
    return {
        "exact_q": exact_q, "maxsum_q": ms_q,
        "regret": regret, "exact_action_match": match,
        "exact_actions": exact_actions.tolist(),
        "maxsum_actions": maxsum_actions[b].tolist(),
    }


# ---------------------------------------------------------------------------
# Double-Q target for C3
# ---------------------------------------------------------------------------

def c3_double_q_target(online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
                       target_unary: SharedRegionalQNetwork, target_pair: SharedPairNetwork,
                       next_joint_states: torch.Tensor, next_active_mask: torch.Tensor,
                       r_joint: torch.Tensor, done: torch.Tensor,
                       gamma: float = 0.99) -> torch.Tensor:
    """C3 Double-Q: a* = MaxSum_online(s'); y = r + gamma*(1-done)*Q_CG_target(s',a*).
    Returns y: (B,)."""
    with torch.no_grad():
        a_star = maxsum_inference(online_unary, online_pair, next_joint_states, next_active_mask)
        q_target = joint_value_cg(target_unary, target_pair, next_joint_states, a_star, next_active_mask)
        y = r_joint + gamma * (1.0 - done) * q_target
    return y


def c3_current_q(online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
                 joint_states: torch.Tensor, action_vector: torch.Tensor,
                 active_mask: torch.Tensor) -> torch.Tensor:
    """Q_CG_online(s, a_requested). Returns (B,)."""
    return joint_value_cg(online_unary, online_pair, joint_states, action_vector, active_mask)


def c3_loss(q_current: torch.Tensor, y_joint: torch.Tensor) -> torch.Tensor:
    """L_C3 = SmoothL1(q_current, y_joint). No auxiliary losses."""
    return nn.SmoothL1Loss()(q_current, y_joint)


# ---------------------------------------------------------------------------
# C3 optimizer params
# ---------------------------------------------------------------------------

def c3_optimizer_params(online_unary: SharedRegionalQNetwork,
                        online_pair: SharedPairNetwork) -> List[nn.Parameter]:
    return list(online_unary.parameters()) + list(online_pair.parameters())


# ---------------------------------------------------------------------------
# C3 config + SHA256
# ---------------------------------------------------------------------------

def make_c3_config() -> Dict:
    from prdrl.rl.normalization import NORM_V3
    from prdrl.rl.reward_r1 import REWARD_R1_NAME
    from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256
    from prdrl.rl.coordination_integration import FROZEN_V3_CONSTANTS_SHA256

    return {
        "architecture_version": TREATMENT_ARCH_C3,
        "unary_architecture": "9-128-128-4",
        "pair_architecture": "18-128-128-16",
        "pair_output_reshape": [4, 4],
        "pair_shared_across_all_active_pairs": True,
        "pair_symmetrization": "0.5*(H_ij + transpose(H_ji))",
        "pair_centering": "ZERO_ROW_COLUMN_MEAN",
        "graph_topology": "COMPLETE_ACTIVE_REGION_GRAPH",
        "joint_value": "MEAN_UNARY_PLUS_MEAN_PAIR",
        "beta_pair": BETA_PAIR,
        "inference": "MAX_SUM_MESSAGE_PASSING_V1",
        "K_maxsum": K_MAXSUM,
        "message_normalization": "SUBTRACT_MAX",
        "action_ordering": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "R_MAX": R_MAX,
        "state_dim": STATE_DIM,
        "pair_seed_offset": PAIR_SEED_OFFSET,
        "gamma": 0.99,
        "tau": 0.005,
        "loss": "JOINT_DOUBLE_DQN_SMOOTHL1_ONLY",
        "no_auxiliary_losses": True,
        "no_regional_loss": True,
        "no_c2_composite": True,
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "optimizer": "Adam",
        "lr": 1e-4,
        "no_gradient_clipping": True,
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "normalization_version": NORM_V3,
        "normalizer_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "small_n_exact_threshold": SMALL_N_EXACT_THRESHOLD,
        "operators": {
            "KEEP_FE": 0, "LOCAL_sigma": 0.02, "PREDICT_sigma": 0.005,
            "PREDICT_lambda": 1.0, "DIVERSIFY_sigma": 0.15, "B": 10,
            "survival": "UNRESTRICTED_STANDARD_NSGA_II", "no_HARD": True,
        },
        "seed_ranges": {
            "train": [2501, 2560], "validation": [2561, 2580],
            "final": [2581, 2640], "policy": [7001, 7005],
        },
    }


def c3_config_sha256(config: Dict = None) -> str:
    cfg = make_c3_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ===========================================================================
# E4RIA: Batched / vectorized execution paths
# ---------------------------------------------------------------------------
# ENGINEERING-ONLY optimizations added by E4RIA.  The scalar functions above
# (maxsum_inference, joint_value_cg, c3_double_q_target, c3_current_q) are
# the IMMUTABLE REFERENCE and MUST NOT be modified.  The batched functions
# below produce IDENTICAL outputs — proven by exhaustive equivalence tests.
# They eliminate the Python ``for b in range(B)`` loop that caused ~350×
# slowdown in C3 training.
#
# Scientific config (make_c3_config / c3_config_sha256) is UNAFFECTED.
# ===========================================================================

# --- Static edge index for complete graph on R_MAX nodes ---

_ALL_EDGES: List[Tuple[int, int]] = list(combinations(range(R_MAX), 2))
_N_UNDIRECTED: int = len(_ALL_EDGES)   # C(10,2) = 45
_N_DIRECTED: int = 2 * _N_UNDIRECTED   # 90

_EDGE_SRC_NP = np.array([e[0] for e in _ALL_EDGES], dtype=np.int64)
_EDGE_DST_NP = np.array([e[1] for e in _ALL_EDGES], dtype=np.int64)

_REVERSE_IDX_NP = np.concatenate([
    np.arange(_N_UNDIRECTED, _N_DIRECTED, dtype=np.int64),
    np.arange(0, _N_UNDIRECTED, dtype=np.int64),
])

_DIR_SRC_NP = np.concatenate([_EDGE_SRC_NP, _EDGE_DST_NP])
_DIR_DST_NP = np.concatenate([_EDGE_DST_NP, _EDGE_SRC_NP])


def _get_edge_tensors(device: torch.device):
    """Return cached edge-index tensors on *device*."""
    return (
        torch.tensor(_EDGE_SRC_NP, dtype=torch.long, device=device),
        torch.tensor(_EDGE_DST_NP, dtype=torch.long, device=device),
        torch.tensor(_DIR_SRC_NP, dtype=torch.long, device=device),
        torch.tensor(_DIR_DST_NP, dtype=torch.long, device=device),
        torch.tensor(_REVERSE_IDX_NP, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Batched max-sum message passing
# ---------------------------------------------------------------------------

def maxsum_inference_batched(
    unary_net: SharedRegionalQNetwork,
    pair_net: SharedPairNetwork,
    joint_states: torch.Tensor,
    active_mask: torch.Tensor,
    k: int = K_MAXSUM,
) -> torch.Tensor:
    """Batched max-sum — mathematically equivalent to ``maxsum_inference``.

    Removes the Python ``for b in range(B)`` loop by operating on the full
    batch at once via tensor operations and scatter_add.

    Parameters match ``maxsum_inference`` exactly.
    joint_states: (B, R_MAX, 9), active_mask: (B, R_MAX) -> actions: (B, R_MAX).
    """
    B, R, D = joint_states.shape
    device = joint_states.device
    dtype = joint_states.dtype
    A = N_ACTIONS
    E = _N_UNDIRECTED   # 45
    DE = _N_DIRECTED    # 90

    edge_src, edge_dst, dir_src, dir_dst, reverse_idx = _get_edge_tensors(device)

    # --- Unary factors: phi[b,i] = U_i(s_i) / n_active[b] for active i ---
    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, A)
    n_active = active_mask.sum(dim=1).clamp(min=1)          # (B,)
    phi = (q_all / n_active.unsqueeze(1).unsqueeze(2)) * active_mask.unsqueeze(2)

    # --- Edge validity & per-sample n_edges ---
    edge_valid = active_mask[:, edge_src] * active_mask[:, edge_dst]   # (B, E)
    n_edges = edge_valid.sum(dim=1).clamp(min=1)                       # (B,)

    # --- Pair factors: P_ij for all 45 edges at once ---
    s_i = joint_states[:, edge_src]  # (B, E, D)
    s_j = joint_states[:, edge_dst]  # (B, E, D)
    p_ij = compute_centered_pair(pair_net, s_i, s_j)  # (B, E, A, A)
    p_scaled = (p_ij / n_edges.unsqueeze(1).unsqueeze(2).unsqueeze(3)) \
        * edge_valid.unsqueeze(2).unsqueeze(3)

    # --- Directed pair factors: fwd = P_ij, bwd = P_ij^T ---
    psi_dir = torch.cat([p_scaled, p_scaled.transpose(-2, -1)], dim=1)  # (B, DE, A, A)

    # --- Directed edge validity ---
    dir_valid = torch.cat([edge_valid, edge_valid], dim=1)  # (B, DE)

    # --- Message passing (K synchronous iterations) ---
    msgs = torch.zeros(B, DE, A, dtype=dtype, device=device)

    dst_idx_expand = dir_dst.unsqueeze(0).unsqueeze(2).expand(B, DE, A)

    for _iter in range(k):
        # Total incoming messages to each node
        masked_msgs = msgs * dir_valid.unsqueeze(2)
        total_incoming = torch.zeros(B, R, A, dtype=dtype, device=device)
        total_incoming.scatter_add_(1, dst_idx_expand, masked_msgs)

        # Incoming to src of each directed edge, excluding the reverse message
        incoming = total_incoming[:, dir_src] - msgs[:, reverse_idx]  # (B, DE, A)

        # Source unary factor
        phi_src = phi[:, dir_src]  # (B, DE, A)

        # Table[b, d, a_src, a_dst] = phi_src + psi_dir + incoming
        table = phi_src.unsqueeze(3) + psi_dir + incoming.unsqueeze(3)  # (B, DE, A, A)

        # Message = max over source actions, then normalize
        new_msgs = table.max(dim=2).values                            # (B, DE, A)
        new_msgs = new_msgs - new_msgs.max(dim=2, keepdim=True).values
        new_msgs = new_msgs * dir_valid.unsqueeze(2)

        msgs = new_msgs

    # --- Beliefs and action selection ---
    masked_msgs_final = msgs * dir_valid.unsqueeze(2)
    total_incoming_final = torch.zeros(B, R, A, dtype=dtype, device=device)
    total_incoming_final.scatter_add_(1, dst_idx_expand, masked_msgs_final)

    belief = phi + total_incoming_final  # (B, R, A)
    actions = belief.argmax(dim=2) * active_mask.long()  # (B, R)

    return actions


# ---------------------------------------------------------------------------
# Batched joint value computation
# ---------------------------------------------------------------------------

def joint_value_cg_batched(
    unary_net: SharedRegionalQNetwork,
    pair_net: SharedPairNetwork,
    joint_states: torch.Tensor,
    action_vector: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Q_CG — mathematically equivalent to ``joint_value_cg``.

    joint_states: (B, R_MAX, 9), action_vector: (B, R_MAX) int,
    active_mask: (B, R_MAX) -> Q_CG: (B,).
    """
    B, R, D = joint_states.shape
    device = joint_states.device
    A = N_ACTIONS
    E = _N_UNDIRECTED

    edge_src, edge_dst, _, _, _ = _get_edge_tensors(device)

    # --- Unary part (same as scalar) ---
    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, A)
    q_sel = q_all.gather(2, action_vector.long().unsqueeze(2)).squeeze(2)  # (B, R)
    q_sel = q_sel * active_mask
    n_active = active_mask.sum(dim=1).clamp(min=1)
    q_unary = q_sel.sum(dim=1) / n_active  # (B,)

    # --- Pair part (all 45 edges at once) ---
    edge_valid = active_mask[:, edge_src] * active_mask[:, edge_dst]  # (B, E)
    n_edges = edge_valid.sum(dim=1).clamp(min=1)

    s_i = joint_states[:, edge_src]  # (B, E, D)
    s_j = joint_states[:, edge_dst]  # (B, E, D)
    p_ij = compute_centered_pair(pair_net, s_i, s_j)  # (B, E, A, A)

    a_i = action_vector[:, edge_src].long()  # (B, E)
    a_j = action_vector[:, edge_dst].long()  # (B, E)

    B_range = torch.arange(B, device=device).unsqueeze(1)
    E_range = torch.arange(E, device=device).unsqueeze(0)
    p_selected = p_ij[B_range, E_range, a_i, a_j]  # (B, E)
    p_selected = p_selected * edge_valid

    q_pair = p_selected.sum(dim=1) / n_edges  # (B,)

    return q_unary + BETA_PAIR * q_pair


# ---------------------------------------------------------------------------
# Batched Double-Q target and current Q
# ---------------------------------------------------------------------------

def c3_double_q_target_batched(
    online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
    target_unary: SharedRegionalQNetwork, target_pair: SharedPairNetwork,
    next_joint_states: torch.Tensor, next_active_mask: torch.Tensor,
    r_joint: torch.Tensor, done: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Batched C3 Double-Q target — equivalent to ``c3_double_q_target``."""
    with torch.no_grad():
        a_star = maxsum_inference_batched(
            online_unary, online_pair, next_joint_states, next_active_mask)
        q_target = joint_value_cg_batched(
            target_unary, target_pair, next_joint_states, a_star, next_active_mask)
        y = r_joint + gamma * (1.0 - done) * q_target
    return y


def c3_current_q_batched(
    online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
    joint_states: torch.Tensor, action_vector: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Q_CG_online — equivalent to ``c3_current_q``."""
    return joint_value_cg_batched(
        online_unary, online_pair, joint_states, action_vector, active_mask)


# ===========================================================================
# C4: Causal Dynamics Gated Pairwise Coordination (E4T design, frozen)
# ---------------------------------------------------------------------------
# Identical to C3 except: each pair potential P_ij is multiplied by a
# deterministic causal gate g_ij = 0.5 * (|d_i| + |d_j|), where
# d_r = abs(state_r[3]) = abs(V3_normalized_delta_severity_r).
#
# Zero trainable gate parameters.  C4_param_count == C3_param_count.
# Gate applied AFTER symmetrization and double centering.
# ===========================================================================

TREATMENT_ARCH_C4 = "C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1"
GATE_SIGNAL_DIM = 3


def compute_pair_gate(s_i: torch.Tensor, s_j: torch.Tensor) -> torch.Tensor:
    """Causal dynamics pair gate: g_ij = 0.5 * (|d_i| + |d_j|).
    d_r = abs(s_r[..., 3]) (V3-normalized delta_severity).
    s_i, s_j: (..., 9) -> g_ij: (...)."""
    d_i = s_i[..., GATE_SIGNAL_DIM].abs()
    d_j = s_j[..., GATE_SIGNAL_DIM].abs()
    return 0.5 * (d_i + d_j)


# ---------------------------------------------------------------------------
# C4 scalar joint value
# ---------------------------------------------------------------------------

def joint_value_c4(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                   joint_states: torch.Tensor, action_vector: torch.Tensor,
                   active_mask: torch.Tensor) -> torch.Tensor:
    """Q_C4 = (1/n)*sum U_i + (1/|E|)*sum g_ij*P_ij(a_i,a_j).
    Identical to joint_value_cg except pair term is gated.
    joint_states: (B, R_MAX, 9), action_vector: (B, R_MAX) int,
    active_mask: (B, R_MAX) -> Q_C4: (B,)."""
    B, R, D = joint_states.shape
    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, N_ACTIONS)
    q_sel = q_all.gather(2, action_vector.long().unsqueeze(2)).squeeze(2)
    q_sel = q_sel * active_mask
    n_active = active_mask.sum(dim=1).clamp(min=1)
    q_unary = q_sel.sum(dim=1) / n_active

    q_pair = torch.zeros(B, dtype=joint_states.dtype, device=joint_states.device)
    for b in range(B):
        edges = build_complete_graph(active_mask[b])
        if len(edges) == 0:
            continue
        pair_sum = torch.tensor(0.0, dtype=joint_states.dtype, device=joint_states.device)
        for (i, j) in edges:
            p_ij = compute_centered_pair(pair_net, joint_states[b, i:i+1],
                                         joint_states[b, j:j+1])
            g = compute_pair_gate(joint_states[b, i:i+1], joint_states[b, j:j+1])
            a_i = action_vector[b, i].long()
            a_j = action_vector[b, j].long()
            pair_sum = pair_sum + g[0] * p_ij[0, a_i, a_j]
        q_pair[b] = pair_sum / len(edges)

    return q_unary + BETA_PAIR * q_pair


# ---------------------------------------------------------------------------
# C4 scalar MaxSum inference
# ---------------------------------------------------------------------------

def maxsum_inference_c4(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                        joint_states: torch.Tensor, active_mask: torch.Tensor,
                        k: int = K_MAXSUM) -> torch.Tensor:
    """C4 MaxSum: psi_ij(a_i,a_j) = g_ij * P_ij(a_i,a_j) / |E|.
    Identical to maxsum_inference except pair potentials are gated.
    joint_states: (B, R_MAX, 9), active_mask: (B, R_MAX) -> actions: (B, R_MAX)."""
    B, R, D = joint_states.shape
    device = joint_states.device
    dtype = joint_states.dtype
    actions = torch.zeros(B, R, dtype=torch.long, device=device)

    for b in range(B):
        active_idx = torch.where(active_mask[b] > 0.5)[0].tolist()
        n = len(active_idx)
        if n == 0:
            continue
        if n == 1:
            i = active_idx[0]
            q = unary_net(joint_states[b, i:i+1])
            actions[b, i] = q.argmax(dim=1).item()
            continue

        edges = list(combinations(active_idx, 2))
        n_edges = len(edges)
        idx_map = {r: pos for pos, r in enumerate(active_idx)}

        states_active = joint_states[b, active_idx]
        q_unary_all = unary_net(states_active)
        phi = q_unary_all / n

        pair_factors = {}
        for (i, j) in edges:
            p_ij = compute_centered_pair(pair_net, joint_states[b, i:i+1],
                                         joint_states[b, j:j+1])
            g = compute_pair_gate(joint_states[b, i:i+1], joint_states[b, j:j+1])
            pair_factors[(i, j)] = g.item() * p_ij.squeeze(0) / n_edges

        neighbors = {r: [] for r in active_idx}
        for (i, j) in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        msgs = {}
        for i in active_idx:
            for j in neighbors[i]:
                msgs[(i, j)] = torch.zeros(N_ACTIONS, dtype=dtype, device=device)

        for iteration in range(k):
            new_msgs = {}
            for i in active_idx:
                for j in neighbors[i]:
                    key_ij = (min(i, j), max(i, j))
                    if i < j:
                        psi = pair_factors[key_ij]
                    else:
                        psi = pair_factors[key_ij].T

                    incoming = torch.zeros(N_ACTIONS, dtype=dtype, device=device)
                    for h in neighbors[i]:
                        if h != j:
                            incoming = incoming + msgs[(h, i)]

                    phi_i = phi[idx_map[i]]
                    table = phi_i.unsqueeze(1) + psi + incoming.unsqueeze(1)
                    msg = table.max(dim=0).values
                    msg = msg - msg.max()
                    new_msgs[(i, j)] = msg

            msgs = new_msgs

        for i in active_idx:
            belief = phi[idx_map[i]].clone()
            for j in neighbors[i]:
                belief = belief + msgs[(j, i)]
            actions[b, i] = belief.argmax().item()

    return actions


# ---------------------------------------------------------------------------
# C4 Double-Q target and current Q
# ---------------------------------------------------------------------------

def c4_double_q_target(online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
                       target_unary: SharedRegionalQNetwork, target_pair: SharedPairNetwork,
                       next_joint_states: torch.Tensor, next_active_mask: torch.Tensor,
                       r_joint: torch.Tensor, done: torch.Tensor,
                       gamma: float = 0.99) -> torch.Tensor:
    """C4 Double-Q: a* = C4_MaxSum_online(s'); y = r + gamma*(1-done)*Q_C4_target(s',a*)."""
    with torch.no_grad():
        a_star = maxsum_inference_c4(online_unary, online_pair, next_joint_states, next_active_mask)
        q_target = joint_value_c4(target_unary, target_pair, next_joint_states, a_star, next_active_mask)
        y = r_joint + gamma * (1.0 - done) * q_target
    return y


def c4_current_q(online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
                 joint_states: torch.Tensor, action_vector: torch.Tensor,
                 active_mask: torch.Tensor) -> torch.Tensor:
    """Q_C4_online(s, a_requested)."""
    return joint_value_c4(online_unary, online_pair, joint_states, action_vector, active_mask)


def c4_loss(q_current: torch.Tensor, y_joint: torch.Tensor) -> torch.Tensor:
    """L_C4 = SmoothL1(q_current, y_joint). Identical to C3."""
    return nn.SmoothL1Loss()(q_current, y_joint)


def c4_optimizer_params(online_unary: SharedRegionalQNetwork,
                        online_pair: SharedPairNetwork) -> List[nn.Parameter]:
    """C4 optimizer parameters — identical to C3 (no gate parameters)."""
    return list(online_unary.parameters()) + list(online_pair.parameters())


# ---------------------------------------------------------------------------
# C4 exact enumeration (DESCRIPTIVE ONLY)
# ---------------------------------------------------------------------------

def exact_joint_argmax_c4(unary_net: SharedRegionalQNetwork, pair_net: SharedPairNetwork,
                          joint_states: torch.Tensor, active_mask: torch.Tensor,
                          b: int) -> Tuple[torch.Tensor, float]:
    """Exhaustive enumeration for C4. Uses joint_value_c4."""
    active_idx = torch.where(active_mask[b] > 0.5)[0].tolist()
    n = len(active_idx)
    if n > SMALL_N_EXACT_THRESHOLD:
        raise ValueError(f"exact enumeration blocked for n={n} > {SMALL_N_EXACT_THRESHOLD}")
    if n == 0:
        return torch.zeros(R_MAX, dtype=torch.long, device=joint_states.device), 0.0

    best_q = float('-inf')
    best_actions = None
    total = N_ACTIONS ** n
    states_single = joint_states[b:b+1]
    mask_single = active_mask[b:b+1]

    for combo_idx in range(total):
        action_vec = torch.zeros(1, R_MAX, dtype=torch.long, device=joint_states.device)
        idx = combo_idx
        for pos, r in enumerate(active_idx):
            action_vec[0, r] = idx % N_ACTIONS
            idx //= N_ACTIONS
        with torch.no_grad():
            q = joint_value_c4(unary_net, pair_net, states_single, action_vec, mask_single)
        q_val = q.item()
        if q_val > best_q or (q_val == best_q and best_actions is None):
            if q_val > best_q:
                best_q = q_val
                best_actions = action_vec[0].clone()
            elif _action_vec_lt(action_vec[0], best_actions, active_idx):
                best_actions = action_vec[0].clone()

    return best_actions, best_q


# ---------------------------------------------------------------------------
# C4 config + SHA256
# ---------------------------------------------------------------------------

def make_c4_config() -> Dict:
    from prdrl.rl.normalization import NORM_V3
    from prdrl.rl.reward_r1 import REWARD_R1_NAME
    from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256
    from prdrl.rl.coordination_integration import FROZEN_V3_CONSTANTS_SHA256

    return {
        "architecture_version": TREATMENT_ARCH_C4,
        "base_controller": TREATMENT_ARCH_C3,
        "identical_to_c3_except": "deterministic_causal_pair_gate",
        "unary_architecture": "9-128-128-4",
        "pair_architecture": "18-128-128-16",
        "pair_output_reshape": [4, 4],
        "pair_shared_across_all_active_pairs": True,
        "pair_symmetrization": "0.5*(H_ij + transpose(H_ji))",
        "pair_centering": "ZERO_ROW_COLUMN_MEAN",
        "gate_signal": "ABS_V3_DELTA_SEVERITY",
        "gate_signal_dim": GATE_SIGNAL_DIM,
        "gate_formula": "0.5*(ABS_DELTA_SEVERITY_I+ABS_DELTA_SEVERITY_J)",
        "gate_applied_after": "PAIR_SYMMETRIZATION_AND_DOUBLE_CENTERING",
        "gate_trainable_parameters": 0,
        "effective_pair": "g_ij * P_ij",
        "graph_topology": "COMPLETE_ACTIVE_REGION_GRAPH",
        "joint_value": "MEAN_UNARY_PLUS_GATED_MEAN_PAIR",
        "beta_pair": BETA_PAIR,
        "inference": "GATED_MAX_SUM_MESSAGE_PASSING_V1",
        "K_maxsum": K_MAXSUM,
        "message_normalization": "SUBTRACT_MAX",
        "action_ordering": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "R_MAX": R_MAX,
        "state_dim": STATE_DIM,
        "pair_seed_offset": PAIR_SEED_OFFSET,
        "gamma": 0.99,
        "tau": 0.005,
        "loss": "JOINT_DOUBLE_DQN_SMOOTHL1_ONLY",
        "no_auxiliary_losses": True,
        "no_regional_loss": True,
        "no_c2_composite": True,
        "no_gate_loss": True,
        "no_gate_supervision": True,
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "optimizer": "Adam",
        "lr": 1e-4,
        "no_gradient_clipping": True,
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "normalization_version": NORM_V3,
        "normalizer_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "small_n_exact_threshold": SMALL_N_EXACT_THRESHOLD,
        "operators": {
            "KEEP_FE": 0, "LOCAL_sigma": 0.02, "PREDICT_sigma": 0.005,
            "PREDICT_lambda": 1.0, "DIVERSIFY_sigma": 0.15, "B": 10,
            "survival": "UNRESTRICTED_STANDARD_NSGA_II", "no_HARD": True,
        },
        "seed_ranges": {
            "train": [2701, 2760], "validation": [2761, 2780],
            "final": [2781, 2840], "policy": [8001, 8005],
        },
    }


def c4_config_sha256(config: Dict = None) -> str:
    cfg = make_c4_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# C4 batched MaxSum inference
# ---------------------------------------------------------------------------

def maxsum_inference_c4_batched(
    unary_net: SharedRegionalQNetwork,
    pair_net: SharedPairNetwork,
    joint_states: torch.Tensor,
    active_mask: torch.Tensor,
    k: int = K_MAXSUM,
) -> torch.Tensor:
    """Batched C4 MaxSum — mathematically equivalent to ``maxsum_inference_c4``.

    Identical to ``maxsum_inference_batched`` except pair potentials are gated.
    joint_states: (B, R_MAX, 9), active_mask: (B, R_MAX) -> actions: (B, R_MAX).
    """
    B, R, D = joint_states.shape
    device = joint_states.device
    dtype = joint_states.dtype
    A = N_ACTIONS
    E = _N_UNDIRECTED
    DE = _N_DIRECTED

    edge_src, edge_dst, dir_src, dir_dst, reverse_idx = _get_edge_tensors(device)

    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, A)
    n_active = active_mask.sum(dim=1).clamp(min=1)
    phi = (q_all / n_active.unsqueeze(1).unsqueeze(2)) * active_mask.unsqueeze(2)

    edge_valid = active_mask[:, edge_src] * active_mask[:, edge_dst]
    n_edges = edge_valid.sum(dim=1).clamp(min=1)

    s_i = joint_states[:, edge_src]
    s_j = joint_states[:, edge_dst]
    p_ij = compute_centered_pair(pair_net, s_i, s_j)
    g = compute_pair_gate(s_i, s_j)

    p_gated = g.unsqueeze(2).unsqueeze(3) * p_ij
    p_scaled = (p_gated / n_edges.unsqueeze(1).unsqueeze(2).unsqueeze(3)) \
        * edge_valid.unsqueeze(2).unsqueeze(3)

    psi_dir = torch.cat([p_scaled, p_scaled.transpose(-2, -1)], dim=1)
    dir_valid = torch.cat([edge_valid, edge_valid], dim=1)

    msgs = torch.zeros(B, DE, A, dtype=dtype, device=device)
    dst_idx_expand = dir_dst.unsqueeze(0).unsqueeze(2).expand(B, DE, A)

    for _iter in range(k):
        masked_msgs = msgs * dir_valid.unsqueeze(2)
        total_incoming = torch.zeros(B, R, A, dtype=dtype, device=device)
        total_incoming.scatter_add_(1, dst_idx_expand, masked_msgs)

        incoming = total_incoming[:, dir_src] - msgs[:, reverse_idx]
        phi_src = phi[:, dir_src]
        table = phi_src.unsqueeze(3) + psi_dir + incoming.unsqueeze(3)
        new_msgs = table.max(dim=2).values
        new_msgs = new_msgs - new_msgs.max(dim=2, keepdim=True).values
        new_msgs = new_msgs * dir_valid.unsqueeze(2)
        msgs = new_msgs

    masked_msgs_final = msgs * dir_valid.unsqueeze(2)
    total_incoming_final = torch.zeros(B, R, A, dtype=dtype, device=device)
    total_incoming_final.scatter_add_(1, dst_idx_expand, masked_msgs_final)

    belief = phi + total_incoming_final
    actions = belief.argmax(dim=2) * active_mask.long()

    return actions


# ---------------------------------------------------------------------------
# C4 batched joint value
# ---------------------------------------------------------------------------

def joint_value_c4_batched(
    unary_net: SharedRegionalQNetwork,
    pair_net: SharedPairNetwork,
    joint_states: torch.Tensor,
    action_vector: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Q_C4 — mathematically equivalent to ``joint_value_c4``.

    joint_states: (B, R_MAX, 9), action_vector: (B, R_MAX) int,
    active_mask: (B, R_MAX) -> Q_C4: (B,).
    """
    B, R, D = joint_states.shape
    device = joint_states.device
    A = N_ACTIONS
    E = _N_UNDIRECTED

    edge_src, edge_dst, _, _, _ = _get_edge_tensors(device)

    q_all = unary_net(joint_states.reshape(B * R, D)).reshape(B, R, A)
    q_sel = q_all.gather(2, action_vector.long().unsqueeze(2)).squeeze(2)
    q_sel = q_sel * active_mask
    n_active = active_mask.sum(dim=1).clamp(min=1)
    q_unary = q_sel.sum(dim=1) / n_active

    edge_valid = active_mask[:, edge_src] * active_mask[:, edge_dst]
    n_edges = edge_valid.sum(dim=1).clamp(min=1)

    s_i = joint_states[:, edge_src]
    s_j = joint_states[:, edge_dst]
    p_ij = compute_centered_pair(pair_net, s_i, s_j)
    g = compute_pair_gate(s_i, s_j)

    a_i = action_vector[:, edge_src].long()
    a_j = action_vector[:, edge_dst].long()

    B_range = torch.arange(B, device=device).unsqueeze(1)
    E_range = torch.arange(E, device=device).unsqueeze(0)
    p_selected = p_ij[B_range, E_range, a_i, a_j]
    p_gated = p_selected * g
    p_gated = p_gated * edge_valid

    q_pair = p_gated.sum(dim=1) / n_edges

    return q_unary + BETA_PAIR * q_pair


# ---------------------------------------------------------------------------
# C4 batched Double-Q target and current Q
# ---------------------------------------------------------------------------

def c4_double_q_target_batched(
    online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
    target_unary: SharedRegionalQNetwork, target_pair: SharedPairNetwork,
    next_joint_states: torch.Tensor, next_active_mask: torch.Tensor,
    r_joint: torch.Tensor, done: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Batched C4 Double-Q target — equivalent to ``c4_double_q_target``."""
    with torch.no_grad():
        a_star = maxsum_inference_c4_batched(
            online_unary, online_pair, next_joint_states, next_active_mask)
        q_target = joint_value_c4_batched(
            target_unary, target_pair, next_joint_states, a_star, next_active_mask)
        y = r_joint + gamma * (1.0 - done) * q_target
    return y


def c4_current_q_batched(
    online_unary: SharedRegionalQNetwork, online_pair: SharedPairNetwork,
    joint_states: torch.Tensor, action_vector: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Q_C4_online — equivalent to ``c4_current_q``."""
    return joint_value_c4_batched(
        online_unary, online_pair, joint_states, action_vector, active_mask)
