"""DRL adapter: bridges frozen C1/C3/C4 controllers to CEC2018 benchmarks.

State features use PBI decomposition — NO true PF in state, reward, or detection.
V1 normalization (hardcoded clip bounds, no fitted constants) is used because:
  - V3 constants were fitted from UAV domain data, not CEC2018
  - V1 gives the controllers a clean, domain-independent normalization
  - This is a 'deploy without recalibration' evaluation

Reference directions partition the population into regions.
The controller observes per-region state and selects KEEP/LOCAL/PREDICT/DIVERSIFY.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from prdrl.rl.network import SharedRegionalQNetwork, N_ACTIONS
from prdrl.rl.coordination_graph import (
    SharedPairNetwork, maxsum_inference, maxsum_inference_c4,
    R_MAX, STATE_DIM, PAIR_SEED_OFFSET, K_MAXSUM,
)
from prdrl.rl.normalization import normalize_v1

ACTION_NAMES = {0: "KEEP", 1: "LOCAL", 2: "PREDICT", 3: "DIVERSIFY"}
N_REF_BI = 5
N_REF_TRI = 10
PBI_THETA = 5.0
SIGMA_LOCAL = 0.02
SIGMA_PREDICT = 0.005
SIGMA_DIVERSIFY = 0.15
LAMBDA_PRED = 1.0
RESPONSE_BUDGET = 10


def generate_reference_directions(n_obj: int, n_ref: Optional[int] = None) -> np.ndarray:
    """Generate uniform reference directions on the unit simplex."""
    if n_obj == 2:
        n = n_ref or N_REF_BI
        w = np.linspace(0, 1, n)
        V = np.column_stack([w, 1 - w])
    elif n_obj == 3:
        n = n_ref or N_REF_TRI
        pts = []
        p = 0
        while len(pts) < n:
            p += 1
            pts = []
            for i in range(p + 1):
                for j in range(p + 1 - i):
                    k = p - i - j
                    pts.append([i / p, j / p, k / p])
        V = np.array(pts[:n])
    else:
        raise ValueError(f"n_obj={n_obj} not supported")
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V


def assign_regions(F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Assign solutions to nearest reference direction by minimum angle."""
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    cos_sim = F_norm @ V_norm.T
    return np.argmax(cos_sim, axis=1)


def pbi_loss_per_solution(F: np.ndarray, V: np.ndarray, region_ids: np.ndarray,
                          theta: float = PBI_THETA) -> np.ndarray:
    """PBI loss for each solution relative to its assigned reference direction."""
    N = len(F)
    losses = np.zeros(N)
    for i in range(N):
        v = V[region_ids[i]]
        v_n = v / (np.linalg.norm(v) + 1e-12)
        d1 = np.dot(F[i], v_n)
        proj = d1 * v_n
        d2 = np.linalg.norm(F[i] - proj)
        losses[i] = d1 + theta * d2
    return losses


def compute_severity(F_minus: np.ndarray, F_plus: np.ndarray,
                     region_ids: np.ndarray, R: int,
                     q_prev: float, kappa: float = 1.0,
                     rho: float = 0.05):
    """Compute per-region severity C[r] and updated Q. Same as frozen severity.py."""
    Dis = np.linalg.norm(F_minus - F_plus, axis=1)
    S = []
    for i in range(R):
        mask = region_ids == i
        S.append(float(np.median(Dis[mask])) if np.any(mask) else float("nan"))
    C = [1 - np.exp(-S[i] / (kappa * q_prev + 1e-12)) for i in range(R)]
    Q = (1 - rho) * q_prev + rho * float(np.nanmedian(S))
    return S, C, Q


def compute_quality_loss(F_minus: np.ndarray, F_plus: np.ndarray,
                         region_ids: np.ndarray, V: np.ndarray,
                         theta: float = PBI_THETA) -> np.ndarray:
    """PBI-based quality loss per region. Same as frozen quality_loss.py."""
    R = len(V)
    losses = []
    for r in range(R):
        mask = region_ids == r
        if not np.any(mask):
            losses.append(float("nan"))
            continue
        F_m = F_minus[mask]
        F_p = F_plus[mask]
        v = V[r]
        v_n = v / (np.linalg.norm(v) + 1e-12)
        point_losses = []
        for fm, fp in zip(F_m, F_p):
            d1_m = np.dot(fm, v_n)
            d2_m = np.linalg.norm(fm - d1_m * v_n)
            g_m = d1_m + theta * d2_m
            d1_p = np.dot(fp, v_n)
            d2_p = np.linalg.norm(fp - d1_p * v_n)
            g_p = d1_p + theta * d2_p
            point_losses.append(max(0.0, g_p - g_m))
        losses.append(float(np.median(point_losses)))
    return np.array(losses)


def build_state_9dim(severity_C: List[float], quality_losses: np.ndarray,
                     region_ids: np.ndarray, R: int,
                     N_total: int, prev_C: Optional[List[float]]) -> Dict[int, np.ndarray]:
    """Build 9-dim state vector for each active region. Returns {region_id: raw9}."""
    active_regions = []
    for r in range(R):
        n_r = int(np.sum(region_ids == r))
        if n_r > 0:
            active_regions.append(r)

    active_fraction = len(active_regions) / R if R > 0 else 0.0
    active_severities = [severity_C[r] for r in active_regions
                         if not np.isnan(severity_C[r])]
    active_losses = [quality_losses[r] for r in active_regions
                     if not np.isnan(quality_losses[r])]

    mean_sev = float(np.mean(active_severities)) if active_severities else 0.0
    max_sev = float(np.max(active_severities)) if active_severities else 0.0
    mean_loss = float(np.mean(active_losses)) if active_losses else 0.0
    max_loss = float(np.max(active_losses)) if active_losses else 0.0

    states = {}
    for r in active_regions:
        sev_r = severity_C[r] if not np.isnan(severity_C[r]) else 0.0
        loss_r = quality_losses[r] if not np.isnan(quality_losses[r]) else 0.0
        n_r = int(np.sum(region_ids == r))
        occ_r = n_r / N_total if N_total > 0 else 0.0
        if prev_C is not None and not np.isnan(prev_C[r]):
            delta_sev = sev_r - prev_C[r]
        else:
            delta_sev = 0.0
        raw9 = np.array([sev_r, loss_r, occ_r, delta_sev,
                         active_fraction, mean_sev, max_sev,
                         mean_loss, max_loss], dtype=np.float64)
        states[r] = raw9
    return states


def load_controller(controller_type: str, checkpoint_path: str,
                    device: torch.device = torch.device("cpu")):
    """Load frozen controller weights.

    controller_type: 'C1', 'C3', or 'C4'
    Returns (unary_net, pair_net_or_None, controller_type)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unary = SharedRegionalQNetwork(input_dim=STATE_DIM)
    unary.load_state_dict(ckpt["online_state_dict"])
    unary.eval()

    pair_net = None
    if controller_type in ("C3", "C4"):
        pair_net = SharedPairNetwork()
        pair_net.load_state_dict(ckpt["pair_online_state_dict"])
        pair_net.eval()

    return unary, pair_net, controller_type


def select_actions_greedy(controller_type: str, unary_net: SharedRegionalQNetwork,
                          pair_net: Optional[SharedPairNetwork],
                          states: Dict[int, np.ndarray],
                          R: int, device: torch.device) -> Dict[int, int]:
    """Greedy action selection (epsilon=0) using frozen controller."""
    if not states:
        return {}

    if controller_type == "C1":
        actions = {}
        for r, raw9 in states.items():
            z = normalize_v1(raw9)
            x = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = unary_net(x)
            actions[r] = int(q.argmax(dim=1).item())
        return actions

    joint_states = torch.zeros(1, R_MAX, STATE_DIM, dtype=torch.float32, device=device)
    active_mask = torch.zeros(1, R_MAX, dtype=torch.float32, device=device)
    for r, raw9 in states.items():
        if r < R_MAX:
            z = normalize_v1(raw9)
            joint_states[0, r] = torch.tensor(z, dtype=torch.float32)
            active_mask[0, r] = 1.0

    with torch.no_grad():
        if controller_type == "C3":
            action_tensor = maxsum_inference(unary_net, pair_net, joint_states,
                                             active_mask, k=K_MAXSUM)
        elif controller_type == "C4":
            action_tensor = maxsum_inference_c4(unary_net, pair_net, joint_states,
                                                active_mask, k=K_MAXSUM)
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

    actions = {}
    for r in states:
        if r < R_MAX:
            actions[r] = int(action_tensor[0, r].item())
    return actions


def execute_response(X: np.ndarray, F: np.ndarray, region_ids: np.ndarray,
                     actions: Dict[int, int], V: np.ndarray,
                     xl: np.ndarray, xu: np.ndarray,
                     evaluate_fn, centroids_prev: Optional[Dict[int, np.ndarray]],
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
    """Execute response operators per region. Returns (X_new, F_new, fe_used)."""
    X_new = X.copy()
    F_new = F.copy()
    fe_total = 0

    for r, action_idx in actions.items():
        mask = region_ids == r
        if not np.any(mask):
            continue
        action = ACTION_NAMES.get(action_idx, "KEEP")
        if action == "KEEP":
            continue

        X_r = X[mask]
        n_r = len(X_r)
        n_var = X.shape[1]
        n_off = min(RESPONSE_BUDGET, n_r)

        if action == "PREDICT" and centroids_prev is not None and r in centroids_prev:
            centroid_now = X_r.mean(axis=0)
            delta = centroid_now - centroids_prev[r]
            parents = X_r[rng.choice(n_r, size=n_off, replace=True)]
            offspring = parents + LAMBDA_PRED * delta + rng.normal(0, SIGMA_PREDICT, (n_off, n_var))
        elif action == "DIVERSIFY":
            parents = X_r[rng.choice(n_r, size=n_off, replace=True)]
            offspring = parents + rng.normal(0, SIGMA_DIVERSIFY, (n_off, n_var))
        else:
            parents = X_r[rng.choice(n_r, size=n_off, replace=True)]
            offspring = parents + rng.normal(0, SIGMA_LOCAL, (n_off, n_var))

        offspring = np.clip(offspring, xl, xu)
        F_off = evaluate_fn(offspring)
        fe_total += n_off

        combined_X = np.vstack([X_r, offspring])
        combined_F = np.vstack([F[mask], F_off])
        selected_idx = _nsga2_select(combined_F, n_r)
        idx_original = np.where(mask)[0]
        X_new[idx_original] = combined_X[selected_idx]
        F_new[idx_original] = combined_F[selected_idx]

    return X_new, F_new, fe_total


def _nsga2_select(F: np.ndarray, n_select: int) -> np.ndarray:
    """NSGA-II environmental selection: NDS + crowding distance."""
    N = len(F)
    if N <= n_select:
        return np.arange(N)

    diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]
    dom = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
    np.fill_diagonal(dom, False)
    n_dom = dom.sum(axis=0)
    S = [np.where(dom[p])[0].tolist() for p in range(N)]
    fronts = [[]]
    for p in range(N):
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)

    selected = []
    for front in fronts:
        if not front:
            break
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            remaining = n_select - len(selected)
            cd = np.zeros(len(front))
            F_f = F[front]
            for m in range(F_f.shape[1]):
                order = np.argsort(F_f[:, m])
                cd[order[0]] = cd[order[-1]] = float("inf")
                f_range = F_f[order[-1], m] - F_f[order[0], m]
                if f_range < 1e-12:
                    continue
                for j in range(1, len(front) - 1):
                    cd[order[j]] += (F_f[order[j + 1], m] - F_f[order[j - 1], m]) / f_range
            best_idx = np.argsort(-cd)[:remaining]
            for j in best_idx:
                selected.append(front[j])
            break
    return np.array(selected)


class DRLController:
    """Wraps a frozen C1/C3/C4 controller for benchmark evaluation."""

    def __init__(self, controller_type: str, checkpoint_path: str,
                 n_obj: int, n_ref: Optional[int] = None,
                 device: torch.device = torch.device("cpu")):
        self.controller_type = controller_type
        self.device = device
        self.n_obj = n_obj
        self.V = generate_reference_directions(n_obj, n_ref)
        self.R = len(self.V)
        self.unary_net, self.pair_net, _ = load_controller(
            controller_type, checkpoint_path, device)
        self.q_prev = 1.0
        self.prev_C: Optional[List[float]] = None
        self.centroids_prev: Optional[Dict[int, np.ndarray]] = None

    def respond(self, X: np.ndarray, F_minus: np.ndarray, F_plus: np.ndarray,
                xl: np.ndarray, xu: np.ndarray, evaluate_fn,
                rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
        """Full response cycle: detect -> state -> action -> execute.

        F_minus: objectives at previous t (same X positions)
        F_plus: objectives at current t (same X positions, re-evaluated)

        Returns (X_new, F_new, fe_used, actions_taken)
        """
        region_ids = assign_regions(F_plus, self.V)

        _, severity_C, self.q_prev = compute_severity(
            F_minus, F_plus, region_ids, self.R, self.q_prev)
        quality_losses = compute_quality_loss(F_minus, F_plus, region_ids, self.V)

        states = build_state_9dim(
            severity_C, quality_losses, region_ids, self.R,
            len(X), self.prev_C)

        actions = select_actions_greedy(
            self.controller_type, self.unary_net, self.pair_net,
            states, self.R, self.device)

        X_new, F_new, fe_used = execute_response(
            X, F_plus, region_ids, actions, self.V, xl, xu,
            evaluate_fn, self.centroids_prev, rng)

        self.prev_C = severity_C
        self.centroids_prev = {}
        for r in range(self.R):
            mask = region_ids == r
            if np.any(mask):
                self.centroids_prev[r] = X_new[mask].mean(axis=0)

        return X_new, F_new, fe_used, actions
