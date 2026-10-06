"""
Effect definitions for PR-DRL-DMOEA final evaluation (frozen in E4E).

All effects are oriented so POSITIVE => favorable to the learned DDQN policy.

Primary  : Delta_response[s] = B_response[e]  - L_response[p,e]   (MIGD_response)
Secondary: Delta_epoch[s]    = B_epoch[e]     - L_epoch[p,e]      (MIGD_epoch)
           FE_saving[s]       = B_fe[e]        - L_fe[p,e]         (response_FE)

For MIGD (lower is better) a positive difference means DDQN achieved a lower
(better) MIGD than the comparator. For FE (lower is cheaper) a positive saving
means DDQN used fewer response function evaluations.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def cell_effects(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """d[p,e] = B[e] - L[p,e]  (n_policy x n_env)."""
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    assert L.ndim == 2 and B.ndim == 1 and B.shape[0] == L.shape[1]
    return B[None, :] - L


def scenario_mean_effect(L: np.ndarray, B: np.ndarray) -> float:
    """Equal-weight mean over all policy seeds and environment seeds."""
    return float(cell_effects(L, B).mean())


def per_policy_effects(L: np.ndarray, B: np.ndarray) -> List[float]:
    """Delta_policy[p] = mean_e (B[e] - L[p,e]) for each trained policy."""
    return [float(x) for x in cell_effects(L, B).mean(axis=1)]


def relative_improvement(delta: float, mean_B: float, eps: float = 1e-12) -> float:
    """RI = delta / mean_e B[e]  (descriptive only, expressed as fraction)."""
    return float(delta / (abs(mean_B) + eps))


def macro_effect(scenario_deltas: Dict[str, float]) -> float:
    """Equal 1/3-weight macro over the three fixed scientific scenarios."""
    vals = list(scenario_deltas.values())
    return float(np.mean(vals))


def summarize_scenario(L: np.ndarray, B: np.ndarray) -> Dict:
    """Point effect + per-policy effects + mean comparator level for one scenario."""
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    delta = scenario_mean_effect(L, B)
    per_policy = per_policy_effects(L, B)
    mean_B = float(B.mean())
    return {
        "delta": delta,
        "per_policy": per_policy,
        "mean_comparator": mean_B,
        "relative_improvement": relative_improvement(delta, mean_B),
        "mean_learned": float(L.mean()),
    }
