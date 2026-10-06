"""Gradient alignment diagnostics for scenario trade-off analysis.

Implements g_J, g_R extraction, cosine similarity, gradient norms,
conflict fraction, composite gradient identity, and mixer gradient checks.
All computations are OFFLINE ONLY — optimizer_steps=0, parameter_changes=0.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

EPSILON_COSINE = 1e-12
COMPOSITE_RTOL = 1e-5
COMPOSITE_ATOL = 1e-7


def extract_utility_gradient(loss: torch.Tensor,
                             online_util: nn.Module,
                             retain_graph: bool = True) -> torch.Tensor:
    """Extract gradient of loss w.r.t. online utility parameters as flat 1-D vector.

    UNWEIGHTED: caller must pass the raw loss, NOT scaled by 0.5."""
    online_util.zero_grad()
    loss.backward(retain_graph=retain_graph)
    grads = []
    for p in online_util.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().clone().flatten())
        else:
            grads.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
    online_util.zero_grad()
    return torch.cat(grads)


def extract_mixer_gradient(loss: torch.Tensor,
                           online_mixer: nn.Module,
                           retain_graph: bool = True) -> torch.Tensor:
    """Extract gradient of loss w.r.t. mixer parameters as flat 1-D vector."""
    online_mixer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    grads = []
    for p in online_mixer.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().clone().flatten())
        else:
            grads.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
    online_mixer.zero_grad()
    return torch.cat(grads)


def gradient_norms(g: torch.Tensor) -> float:
    return float(torch.linalg.norm(g).item())


def cosine_similarity_jr(g_j: torch.Tensor,
                         g_r: torch.Tensor,
                         epsilon: float = EPSILON_COSINE) -> Dict:
    """Compute cosine similarity between g_J and g_R.

    Zero-gradient semantics: if either norm is zero, cosine_valid=False, cosine_JR=None (NA)."""
    norm_j = gradient_norms(g_j)
    norm_r = gradient_norms(g_r)
    if norm_j == 0.0 or norm_r == 0.0:
        return {
            "cosine_JR": None,
            "cosine_valid": False,
            "norm_g_J": norm_j,
            "norm_g_R": norm_r,
            "reason": "zero_gradient",
        }
    dot_val = float(torch.dot(g_j, g_r).item())
    cos_val = dot_val / (norm_j * norm_r + epsilon)
    return {
        "cosine_JR": cos_val,
        "cosine_valid": True,
        "norm_g_J": norm_j,
        "norm_g_R": norm_r,
    }


def gradient_norm_ratio(norm_r: float, norm_j: float,
                        epsilon: float = EPSILON_COSINE) -> float:
    """rho_g = ||g_R|| / (||g_J|| + epsilon)."""
    return norm_r / (norm_j + epsilon)


def composite_gradient_identity(g_j: torch.Tensor, g_r: torch.Tensor,
                                g_c2_direct: torch.Tensor,
                                joint_weight: float = 0.5,
                                regional_weight: float = 0.5) -> Dict:
    """Verify g_C2_direct == joint_weight*g_J + regional_weight*g_R."""
    g_c2_reconstructed = joint_weight * g_j + regional_weight * g_r
    max_diff = float(torch.max(torch.abs(g_c2_direct - g_c2_reconstructed)).item())
    close = torch.allclose(g_c2_direct, g_c2_reconstructed,
                           rtol=COMPOSITE_RTOL, atol=COMPOSITE_ATOL)
    return {
        "identity_holds": bool(close),
        "max_abs_diff": max_diff,
        "norm_direct": float(torch.linalg.norm(g_c2_direct).item()),
        "norm_reconstructed": float(torch.linalg.norm(g_c2_reconstructed).item()),
    }


def mixer_gradient_check(g_mixer_joint_unweighted: torch.Tensor,
                         g_mixer_c2: torch.Tensor,
                         joint_weight: float = 0.5) -> Dict:
    """Verify ||g_mixer_C2|| ≈ joint_weight * ||g_mixer_joint_unweighted||.

    Since L_regional does not flow through the mixer, g_mixer_C2 should equal
    joint_weight * g_mixer_joint_unweighted exactly."""
    expected = joint_weight * g_mixer_joint_unweighted
    max_diff = float(torch.max(torch.abs(g_mixer_c2 - expected)).item())
    close = torch.allclose(g_mixer_c2, expected, rtol=COMPOSITE_RTOL, atol=COMPOSITE_ATOL)
    return {
        "no_regional_path": bool(close),
        "max_abs_diff": max_diff,
        "norm_mixer_joint_unweighted": float(torch.linalg.norm(g_mixer_joint_unweighted).item()),
        "norm_mixer_c2": float(torch.linalg.norm(g_mixer_c2).item()),
        "expected_ratio": joint_weight,
    }


def conflict_fraction(cosine_values: List[Optional[float]]) -> float:
    """Fraction of valid cosine values that are negative (conflict)."""
    valid = [c for c in cosine_values if c is not None]
    if len(valid) == 0:
        return float("nan")
    return sum(1 for c in valid if c < 0) / len(valid)


def scenario_gap_cosine(mean_cosine_stat: float,
                        mean_cosine_per: float,
                        mean_cosine_mix: float) -> float:
    """G_cos = mean_cosine_STATIONARY - 0.5*(mean_cosine_PERSISTENT + mean_cosine_MIXED)."""
    return mean_cosine_stat - 0.5 * (mean_cosine_per + mean_cosine_mix)


def scenario_gap_conflict(conflict_stat: float,
                          conflict_per: float,
                          conflict_mix: float) -> float:
    """G_conflict = conflict_fraction_STATIONARY - 0.5*(conflict_fraction_PERSISTENT + conflict_fraction_MIXED)."""
    return conflict_stat - 0.5 * (conflict_per + conflict_mix)


def policy_level_gaps(per_policy_cosines: Dict[int, Dict[str, float]],
                      per_policy_conflicts: Dict[int, Dict[str, float]]) -> Dict:
    """Compute per-policy G_cos and G_conflict. Returns lists + count of consistent policies."""
    g_cos_list = []
    g_conflict_list = []
    for pseed in sorted(per_policy_cosines.keys()):
        cos_s = per_policy_cosines[pseed]
        g = scenario_gap_cosine(cos_s["STATIONARY"], cos_s["PERSISTENT"], cos_s["MIXED"])
        g_cos_list.append(g)
    for pseed in sorted(per_policy_conflicts.keys()):
        con_s = per_policy_conflicts[pseed]
        g = scenario_gap_conflict(con_s["STATIONARY"], con_s["PERSISTENT"], con_s["MIXED"])
        g_conflict_list.append(g)
    return {
        "policy_G_cos": g_cos_list,
        "policy_G_conflict": g_conflict_list,
        "n_policies_G_cos_negative": sum(1 for g in g_cos_list if g < 0),
        "n_policies_G_conflict_positive": sum(1 for g in g_conflict_list if g > 0),
    }
