"""
Reward version registry + REWARD_R1 candidate (frozen in E4I).

Single-factor discipline: the CONTROL arm reward (ORIGINAL_REWARD_R0) lives in
src/prdrl/rl/reward.py and is NOT modified. This module only ADDS the frozen R1
candidate so R0's source (and its code-manifest hash) stays byte-stable.

REWARD_R1_REGIONAL_QUALITY_GAIN replaces ONLY the inert local credit SIGNAL:
  r_r^R1 = alpha_g * r_global + alpha_local * g_r - beta * FE_cost_r
with g_r the SYMMETRIC_REGIONAL_QUALITY_GAIN of the same region (bounded, scale-free):

  L_pre  = regional quality loss immediately BEFORE the joint response
           (== the R0 pbi_before == the V2 state's convergence_loss_r)
  L_post = the SAME regional quality loss immediately AFTER the completed joint
           response (== the R0 pbi_after), same fixed reference-direction region ID.

  if L_pre == 0 and L_post == 0:  g_r = 0                       (no signal / exact tie)
  else:                           g_r = (L_pre - L_post) / (L_pre + L_post + eps)

For nonnegative losses g_r in [-1, 1]: >0 local quality improved, 0 tie, <0 worsened.
No fitted constants, no extra objective-function evaluation, no counterfactual rollout.
Uses only pre-response state + executed response + immediate post-response population.

Degenerate/empty-region convention is REUSED from R0 (compute_regional_pbi_reward):
an inactive/degenerate region whose L_pre or L_post is NaN yields g_r = 0.0. This is
the existing project convention (NaN -> 0 reward), not an invented penalty.
"""

from __future__ import annotations

import numpy as np

# Frozen reward version names (checkpoints must store reward_version; mismatch = hard error).
REWARD_R0_NAME = "ORIGINAL_REWARD_R0"
REWARD_R1_NAME = "REWARD_R1_REGIONAL_QUALITY_GAIN"

# Project-wide numerical epsilon (matches reward._EPS, quality_loss alpha, normalization EPS).
EPS_R1 = 1e-12

# Initial coefficients: preserve R0 exactly (smallest scientific modification).
ALPHA_G_R1 = 1.0
ALPHA_LOCAL_R1_INITIAL = 0.1
BETA_R1 = 0.01

# Non-performance-based engineering bounds for any scale-match coefficient (E4I-5 safety).
# alpha_local must keep the local term on a comparable magnitude to global without
# destabilising the Huber/SmoothL1 target range; derived from magnitude comparability,
# never from MIGD outcome. If a required scale-match exceeds these, R1 is NOT ready.
ALPHA_LOCAL_MIN = 0.01
ALPHA_LOCAL_MAX = 10.0


def regional_quality_gain_r1(L_pre, L_post, eps: float = EPS_R1) -> float:
    """Symmetric regional quality gain g_r in [-1, 1].

    L_pre, L_post are nonnegative regional quality losses (lower = better) for the
    SAME reference-direction region. NaN (inactive/degenerate region) -> 0.0 by the
    existing R0 convention. Exact double-zero -> 0.0.
    """
    if L_pre is None or L_post is None:
        return 0.0
    if np.isnan(L_pre) or np.isnan(L_post):
        return 0.0
    Lp = float(L_pre)
    Lq = float(L_post)
    if Lp == 0.0 and Lq == 0.0:
        return 0.0
    return float((Lp - Lq) / (Lp + Lq + eps))


def compute_regional_reward_r1(
    global_reward: float,
    g_r: float,
    fe_cost: float,
    alpha_g: float = ALPHA_G_R1,
    alpha_local: float = ALPHA_LOCAL_R1_INITIAL,
    beta: float = BETA_R1,
) -> float:
    """r_r^R1 = alpha_g * r_global + alpha_local * g_r - beta * fe_cost.

    Global term and FE penalty are IDENTICAL to R0; only the local signal (g_r)
    differs. alpha_g >= alpha_local retained by default (team-dominant)."""
    return float(alpha_g * global_reward + alpha_local * g_r - beta * fe_cost)
