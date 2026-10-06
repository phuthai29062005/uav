"""Common-state bank, action disagreement, Q-margin, and state stratification diagnostics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

STATE_DIM = 9
CENTRAL_STATE_DIM = 55
R_MAX = 10
N_ACTIONS = 4
ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]


def build_common_state_record(source_arm: str, policy: int, scenario: str,
                              env_seed: int, change_index: int, region_id: int,
                              local_state_9d: np.ndarray,
                              central_state_55d: Optional[np.ndarray],
                              active_mask: np.ndarray) -> Dict:
    """Build a single common-state bank record."""
    rec = {
        "source_arm": source_arm,
        "policy": policy,
        "scenario": scenario,
        "diagnostic_environment": env_seed,
        "dynamic_change_index": change_index,
        "region_id": region_id,
        "local_state_9D": local_state_9d.tolist() if isinstance(local_state_9d, np.ndarray) else list(local_state_9d),
        "active_mask": active_mask.tolist() if isinstance(active_mask, np.ndarray) else list(active_mask),
    }
    if central_state_55d is not None:
        rec["central_state_55D"] = (central_state_55d.tolist()
                                    if isinstance(central_state_55d, np.ndarray) else list(central_state_55d))
    return rec


def cross_evaluate(local_state: torch.Tensor,
                   c1_util: nn.Module,
                   c2_util: nn.Module) -> Dict:
    """Evaluate both C1 and C2 utility networks on the same 9-D state.

    local_state: [B, 9] or [9]. Returns per-state Q vectors, argmax actions,
    disagreement indicator, and Q-margins."""
    if local_state.dim() == 1:
        local_state = local_state.unsqueeze(0)
    with torch.no_grad():
        q_c1 = c1_util(local_state)
        q_c2 = c2_util(local_state)

    argmax_c1 = q_c1.argmax(dim=1)
    argmax_c2 = q_c2.argmax(dim=1)
    disagree = (argmax_c1 != argmax_c2).float()

    margin_c1 = _q_margin(q_c1)
    margin_c2 = _q_margin(q_c2)
    delta_margin = margin_c2 - margin_c1

    return {
        "Q_C1": q_c1,
        "Q_C2": q_c2,
        "argmax_C1": argmax_c1,
        "argmax_C2": argmax_c2,
        "disagreement": disagree,
        "margin_C1": margin_c1,
        "margin_C2": margin_c2,
        "Delta_margin": delta_margin,
    }


def _q_margin(q: torch.Tensor) -> torch.Tensor:
    """Q-margin = top1(Q) - top2(Q). Shape [B]."""
    sorted_q, _ = torch.sort(q, dim=1, descending=True)
    return sorted_q[:, 0] - sorted_q[:, 1]


def action_disagreement_fraction(argmax_c1: torch.Tensor,
                                 argmax_c2: torch.Tensor) -> float:
    """D_action = fraction where argmax_C1 != argmax_C2."""
    return float((argmax_c1 != argmax_c2).float().mean().item())


def action_transition_matrix(argmax_c1: torch.Tensor,
                             argmax_c2: torch.Tensor,
                             n_actions: int = N_ACTIONS) -> np.ndarray:
    """4x4 transition matrix: rows=C1 action, cols=C2 action.
    Entry [i,j] = count of states where C1 chose i and C2 chose j."""
    matrix = np.zeros((n_actions, n_actions), dtype=np.int64)
    c1_np = argmax_c1.cpu().numpy().astype(int)
    c2_np = argmax_c2.cpu().numpy().astype(int)
    for i in range(len(c1_np)):
        matrix[c1_np[i], c2_np[i]] += 1
    return matrix


def q_margin_stats(margins: torch.Tensor) -> Dict:
    """Summary statistics for Q-margin values."""
    m = margins.cpu().numpy().astype(float)
    return {
        "mean": float(np.mean(m)),
        "median": float(np.median(m)),
        "std": float(np.std(m, ddof=0)),
        "min": float(np.min(m)),
        "max": float(np.max(m)),
    }


def stratify_by_delta_severity(states_9d: np.ndarray,
                                delta_severity_index: int = 0) -> Dict:
    """Stratify states by delta_severity == 0 (structural zero) vs > 0."""
    ds = states_9d[:, delta_severity_index]
    zero_mask = ds == 0.0
    positive_mask = ds > 0.0
    return {
        "n_total": len(ds),
        "n_delta_severity_zero": int(np.sum(zero_mask)),
        "n_delta_severity_positive": int(np.sum(positive_mask)),
        "zero_mask": zero_mask,
        "positive_mask": positive_mask,
    }
