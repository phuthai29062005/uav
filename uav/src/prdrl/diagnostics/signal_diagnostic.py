"""R1 signal statistics and TD error diagnostics."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def r1_signal_statistics(r1_rewards: np.ndarray) -> Dict:
    """Compute per-scenario R1 signal statistics as frozen in E4O-6."""
    r = np.asarray(r1_rewards, dtype=np.float64)
    if len(r) == 0:
        return {"n": 0, "error": "empty"}
    n = len(r)
    zero_mask = r == 0.0
    pos_mask = r > 0.0
    neg_mask = r < 0.0
    abs_r = np.abs(r)
    return {
        "n": n,
        "zero_fraction": float(np.sum(zero_mask) / n),
        "positive_fraction": float(np.sum(pos_mask) / n),
        "negative_fraction": float(np.sum(neg_mask) / n),
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "mean_absolute_value": float(np.mean(abs_r)),
        "q10": float(np.percentile(r, 10)),
        "q25": float(np.percentile(r, 25)),
        "q75": float(np.percentile(r, 75)),
        "q90": float(np.percentile(r, 90)),
        "variance": float(np.var(r, ddof=0)),
    }


def r1_activity(r1_rewards: np.ndarray, eps_reward: float = 0.0) -> float:
    """R1 activity = fraction |r_r^R1| > eps_reward. Uses exact zero if eps_reward=0."""
    r = np.asarray(r1_rewards, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    return float(np.mean(np.abs(r) > eps_reward))


def regional_td_error(q_local: torch.Tensor,
                      y_local: torch.Tensor) -> Dict:
    """Compute regional TD error diagnostics.

    delta_R = y_local - q_local."""
    delta = (y_local - q_local).detach()
    abs_delta = torch.abs(delta)
    sign_delta = torch.sign(delta)
    smoothl1 = nn.SmoothL1Loss(reduction="none")(q_local.detach(), y_local.detach())
    return {
        "abs_delta_R": abs_delta,
        "sign_delta_R": sign_delta,
        "regional_smoothl1": smoothl1,
        "mean_abs_delta_R": float(abs_delta.mean().item()),
        "mean_regional_smoothl1": float(smoothl1.mean().item()),
    }


def joint_td_error(q_tot: torch.Tensor,
                   y_joint: torch.Tensor) -> Dict:
    """Compute joint TD error diagnostics.

    delta_J = y_joint - Q_tot."""
    delta = (y_joint - q_tot).detach()
    abs_delta = torch.abs(delta)
    smoothl1 = nn.SmoothL1Loss(reduction="none")(q_tot.detach(), y_joint.detach())
    return {
        "abs_delta_J": abs_delta,
        "joint_smoothl1": smoothl1,
        "mean_abs_delta_J": float(abs_delta.mean().item()),
        "mean_joint_smoothl1": float(smoothl1.mean().item()),
    }


def td_error_ratio(abs_delta_r: torch.Tensor,
                   abs_delta_j: torch.Tensor,
                   epsilon: float = 1e-12) -> torch.Tensor:
    """|delta_R| / (|delta_J| + eps)."""
    return abs_delta_r / (abs_delta_j + epsilon)


def td_error_summary(abs_delta: torch.Tensor) -> Dict:
    d = abs_delta.cpu().numpy().astype(float)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "std": float(np.std(d, ddof=0)),
        "q10": float(np.percentile(d, 10)),
        "q90": float(np.percentile(d, 90)),
    }
