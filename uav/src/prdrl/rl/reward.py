"""
E4-P: Reward contract for parameter-shared regional Double DQN.

Design:
  - TEAM-DOMINANT reward: global component weighted >= local component.
  - Global reward = normalized IGD improvement of the JOINT population outcome.
  - Local reward = regional PBI quality improvement (from quality_loss module).
  - FE cost = actual response FE, normalized by budget.
  - Violation cost = NOT_APPLICABLE for current controlled benchmark.
  - Reward is based on ACTUAL JOINT EXECUTED outcome, NOT isolated per-region utility.

Coefficients (UNFROZEN — predeclared defaults, NOT empirically tuned in E4-P):
  alpha_g = 1.0   (global component weight)
  alpha_l = 0.1   (local component weight, must be <= alpha_g)
  beta    = 0.01  (FE cost coefficient)
  gamma   = 0.99  (discount factor, for training config only)

Convention: quality_fn = -IGD (same as oracle, E2, E3-R, H1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Unfrozen candidate defaults — not empirically tuned
ALPHA_G_DEFAULT = 1.0
ALPHA_L_DEFAULT = 0.1
BETA_DEFAULT    = 0.01
GAMMA_DEFAULT   = 0.99

# FE normalization denominator: B * n_active_regions
# (actual total budget consumed by all active regions)
BUDGET_PER_REGION = 10

# Quality convention (same as oracle)
_EPS = 1e-12


def compute_global_reward(
    q_before: float,
    q_after: float,
    normalization: float = 1.0,
) -> float:
    """
    Global reward component = (q_after - q_before) / normalization.

    q = -IGD (higher is better, same sign convention as oracle).
    q_before: quality BEFORE joint response.
    q_after:  quality AFTER joint response (actual executed outcome).
    normalization: scale factor (e.g. |q_before| + eps).

    Sign: positive if quality improved, negative if degraded.
    """
    norm = abs(normalization) + _EPS
    return float((q_after - q_before) / norm)


def compute_regional_pbi_reward(
    pbi_loss_before: float,
    pbi_loss_after: float,
) -> float:
    """
    Regional local reward = -(pbi_loss_after - pbi_loss_before).

    PBI loss is non-negative (larger = worse).
    Reducing loss = positive local reward.
    Increasing loss = negative local reward.

    Returns 0.0 if either value is NaN (inactive region).
    """
    if np.isnan(pbi_loss_before) or np.isnan(pbi_loss_after):
        return 0.0
    return float(-(pbi_loss_after - pbi_loss_before))


def compute_fe_cost(
    actual_fe: int,
    n_active_regions: int,
    budget_per_region: int = BUDGET_PER_REGION,
) -> float:
    """
    Normalized FE cost = actual_fe / (budget_per_region * n_active_regions).

    KEEP: actual_fe = 0 → FE cost = 0.
    Active response: FE cost ∈ (0, 1].
    """
    if n_active_regions <= 0 or budget_per_region <= 0:
        return 0.0
    max_fe = float(budget_per_region * n_active_regions)
    return float(actual_fe) / max_fe


def compute_regional_reward(
    global_reward: float,
    local_pbi_reward: float,
    fe_cost: float,
    alpha_g: float = ALPHA_G_DEFAULT,
    alpha_l: float = ALPHA_L_DEFAULT,
    beta:    float = BETA_DEFAULT,
) -> float:
    """
    r_r = alpha_g * r_global + alpha_l * r_local - beta * fe_cost

    Violation component = NOT_APPLICABLE (set to 0.0 for controlled benchmark).
    alpha_g >= alpha_l enforced (team-dominant).
    """
    if alpha_g < alpha_l:
        raise ValueError(
            f"alpha_g={alpha_g} must be >= alpha_l={alpha_l} "
            f"(team-dominant reward required)"
        )
    return float(alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost)


class RewardConfig:
    """Frozen reward configuration for one E4 training run."""

    def __init__(
        self,
        alpha_g: float = ALPHA_G_DEFAULT,
        alpha_l: float = ALPHA_L_DEFAULT,
        beta:    float = BETA_DEFAULT,
        gamma:   float = GAMMA_DEFAULT,
        normalization_mode: str = "abs_q_before",
    ):
        if alpha_g < alpha_l:
            raise ValueError("alpha_g must be >= alpha_l")
        self.alpha_g = alpha_g
        self.alpha_l = alpha_l
        self.beta    = beta
        self.gamma   = gamma
        self.normalization_mode = normalization_mode
        self.violation_component = "NOT_APPLICABLE"

    def to_dict(self) -> dict:
        return {
            "alpha_g":            self.alpha_g,
            "alpha_l":            self.alpha_l,
            "beta":               self.beta,
            "gamma":              self.gamma,
            "normalization_mode": self.normalization_mode,
            "violation_component": self.violation_component,
            "quality_fn":         "negative_IGD",
            "reward_scope":       "JOINT_GLOBAL_PLUS_LOCAL",
            "coefficients_status": "UNFROZEN_PREDECLARED_DEFAULTS",
            "oracle_labels_as_target": False,
        }
