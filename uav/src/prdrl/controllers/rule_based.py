"""
E3 — Rule-Based Regional Controller.

RULE_V1 (predeclared, not tuned from performance):
  Priority: 1=KEEP, 2=DIVERSIFY, 3=PREDICT, 4=LOCAL (default)
  Thresholds set from feature percentile calibration only.

Invariants:
  - No DRL, no training, no sigma tuning, no performance-tuned thresholds
  - No HARD action, no B_r(a) action-dependent budget
  - Production code UNCHANGED (respond_population called via R0 interface only)
  - RL_STATUS = NOT_STARTED
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from prdrl.rl.interface import (
    ACTION_KEEP,
    ACTION_LOCAL,
    ACTION_PREDICT,
    ACTION_DIVERSIFY,
    RegionalDecisionContext,
)

# -------------------------------------------------------------------
# Controller registry
# -------------------------------------------------------------------

GLOBAL_CONTROLLERS = (
    "GLOBAL_KEEP",
    "GLOBAL_LOCAL",
    "GLOBAL_PREDICT",
    "GLOBAL_DIVERSIFY",
)
ALL_CONTROLLERS = ("RULE_V1",) + GLOBAL_CONTROLLERS

_GLOBAL_ACTION: Dict[str, str] = {
    "GLOBAL_KEEP":      ACTION_KEEP,
    "GLOBAL_LOCAL":     ACTION_LOCAL,
    "GLOBAL_PREDICT":   ACTION_PREDICT,
    "GLOBAL_DIVERSIFY": ACTION_DIVERSIFY,
}

# -------------------------------------------------------------------
# Named state accessors — no hard-coded numeric indices.
# Feature vector layout: [c_r(D), l_r(1), o_r(1), delta_c_r(D)]
# -------------------------------------------------------------------

def _c_scalar(fv: np.ndarray, D: int) -> float:
    """L2 norm of c_r (centroid vector) as severity scalar."""
    return float(np.linalg.norm(fv[:D]))


def _l_scalar(fv: np.ndarray, D: int) -> float:
    """l_r = region size (number of solutions in region)."""
    return float(fv[D])


def _o_scalar(fv: np.ndarray, D: int) -> float:
    """o_r = occupancy ratio (n_r / N_total)."""
    return float(fv[D + 1])


def _dc_scalar(fv: np.ndarray, D: int) -> float:
    """L2 norm of delta_c_r (centroid shift) as motion magnitude."""
    return float(np.linalg.norm(fv[D + 2: D + 2 + D]))


# -------------------------------------------------------------------
# RULE_V1
# -------------------------------------------------------------------

def choose_action_rule_v1(
    context: RegionalDecisionContext,
    rule_config: dict,
) -> str:
    """Choose action for one ACTIVE region under RULE_V1.

    rule_config must contain:
        D         int    — decision space dimension
        T_C_LOW   float  — 33rd percentile of ||c_r|| from calibration
        T_C_HIGH  float  — 67th percentile of ||c_r|| from calibration
        T_O_LOW   float  — 25th percentile of o_r from calibration
        T_DC_HIGH float  — 75th percentile of ||delta_c_r|| from calibration
        T_L       float  — fixed at 0 (l_r>0 always True for active regions)

    Priority:
        1. KEEP      — if c<=T_C_LOW AND l<=T_L AND dc<=0
        2. DIVERSIFY — elif c>=T_C_HIGH AND l>T_L AND (o<=T_O_LOW OR dc>=T_DC_HIGH)
        3. PREDICT   — elif c>=T_C_HIGH AND l>T_L
        4. LOCAL     — default
    """
    D = int(rule_config["D"])
    fv = context.feature_vector   # read-only by R0 invariant

    T_C_LOW   = float(rule_config["T_C_LOW"])
    T_C_HIGH  = float(rule_config["T_C_HIGH"])
    T_O_LOW   = float(rule_config["T_O_LOW"])
    T_DC_HIGH = float(rule_config["T_DC_HIGH"])
    T_L       = float(rule_config["T_L"])   # spec: T_L = 0

    c  = _c_scalar(fv, D)
    l  = _l_scalar(fv, D)
    o  = _o_scalar(fv, D)
    dc = _dc_scalar(fv, D)

    # Rule 1 — KEEP
    if c <= T_C_LOW and l <= T_L and dc <= 0.0:
        return ACTION_KEEP

    # Rule 2 — DIVERSIFY
    if c >= T_C_HIGH and l > T_L and (o <= T_O_LOW or dc >= T_DC_HIGH):
        return ACTION_DIVERSIFY

    # Rule 3 — PREDICT
    if c >= T_C_HIGH and l > T_L:
        return ACTION_PREDICT

    # Rule 4 — LOCAL (default)
    return ACTION_LOCAL


def get_rule_scalars(
    context: RegionalDecisionContext,
    D: int,
) -> dict:
    """Extract (c, l, o, dc) scalars from context for logging."""
    fv = context.feature_vector
    return {
        "c":  _c_scalar(fv, D),
        "l":  _l_scalar(fv, D),
        "o":  _o_scalar(fv, D),
        "dc": _dc_scalar(fv, D),
    }


# -------------------------------------------------------------------
# Unified controller dispatch
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RULE_V1_V2STATE — same logical structure as RULE_V1, correct V2 semantics
# -------------------------------------------------------------------

# E3-R controller registry
ALL_CONTROLLERS_E3R = ("RULE_V1_V2STATE",) + GLOBAL_CONTROLLERS


def _sev_scalar(fv: np.ndarray) -> float:
    """severity_r from V2 feature vector index 0."""
    return float(fv[0])


def _closs_scalar(fv: np.ndarray) -> float:
    """convergence_loss_r from V2 feature vector index 1."""
    return float(fv[1])


def _occ_v2_scalar(fv: np.ndarray) -> float:
    """occupancy_r from V2 feature vector index 2."""
    return float(fv[2])


def _dsev_scalar(fv: np.ndarray) -> float:
    """delta_severity_r from V2 feature vector index 3."""
    return float(fv[3])


def choose_action_rule_v1_v2state(context, rule_config: dict) -> str:
    """RULE_V1_V2STATE: identical branch logic to RULE_V1, applied to V2 features.

    V2 feature vector: [severity_r, convergence_loss_r, occupancy_r, delta_severity_r]

    Same priority:
        1. KEEP      — if severity <= T_C_LOW AND convergence_loss <= 0 AND delta_severity <= 0
        2. DIVERSIFY — elif severity >= T_C_HIGH AND convergence_loss > 0 AND
                           (occupancy <= T_O_LOW OR delta_severity >= T_DC_HIGH)
        3. PREDICT   — elif severity >= T_C_HIGH AND convergence_loss > 0
        4. LOCAL     — default
    """
    fv = context.feature_vector  # shape (4,)

    c  = _sev_scalar(fv)
    l  = _closs_scalar(fv)
    o  = _occ_v2_scalar(fv)
    dc = _dsev_scalar(fv)

    T_C_LOW   = float(rule_config["T_C_LOW"])
    T_C_HIGH  = float(rule_config["T_C_HIGH"])
    T_O_LOW   = float(rule_config["T_O_LOW"])
    T_DC_HIGH = float(rule_config["T_DC_HIGH"])
    T_L       = float(rule_config.get("T_L", 0.0))  # fixed at 0.0

    # Rule 1 — KEEP (reachable under V2: convergence_loss can be 0)
    if c <= T_C_LOW and l <= T_L and dc <= 0.0:
        return ACTION_KEEP

    # Rule 2 — DIVERSIFY
    if c >= T_C_HIGH and l > T_L and (o <= T_O_LOW or dc >= T_DC_HIGH):
        return ACTION_DIVERSIFY

    # Rule 3 — PREDICT
    if c >= T_C_HIGH and l > T_L:
        return ACTION_PREDICT

    # Rule 4 — LOCAL (default)
    return ACTION_LOCAL


def get_v2_rule_scalars(context) -> dict:
    """Extract (c, l, o, dc) scalars from V2 context for logging."""
    fv = context.feature_vector
    return {
        "c":  _sev_scalar(fv),
        "l":  _closs_scalar(fv),
        "o":  _occ_v2_scalar(fv),
        "dc": _dsev_scalar(fv),
    }


def choose_actions_v2(
    controller_name: str,
    contexts: dict,  # Dict[int, RegionalDecisionContextV2]
    rule_config: Optional[dict] = None,
) -> Dict[int, str]:
    """Build action map for E3-R: dispatches RULE_V1_V2STATE or global controllers.

    Inactive regions (active=False) are excluded.

    Args:
        controller_name: one of ALL_CONTROLLERS_E3R
        contexts:        {region_id: RegionalDecisionContextV2}
        rule_config:     required for RULE_V1_V2STATE

    Returns:
        Dict[int, str] — only active regions
    """
    if controller_name not in ALL_CONTROLLERS_E3R:
        raise ValueError(
            f"Unknown controller '{controller_name}'. "
            f"Valid for E3-R: {ALL_CONTROLLERS_E3R}"
        )

    actions: Dict[int, str] = {}
    for region_id, ctx in contexts.items():
        if not ctx.active:
            continue
        if controller_name in _GLOBAL_ACTION:
            actions[region_id] = _GLOBAL_ACTION[controller_name]
        else:  # RULE_V1_V2STATE
            if rule_config is None:
                raise ValueError("rule_config is required for RULE_V1_V2STATE")
            actions[region_id] = choose_action_rule_v1_v2state(ctx, rule_config)

    return actions


def choose_actions(
    controller_name: str,
    contexts: Dict[int, RegionalDecisionContext],
    D: int,
    rule_config: Optional[dict] = None,
) -> Dict[int, str]:
    """Build action map {region_id: action_str} for all ACTIVE regions.

    Inactive regions (active=False) are excluded — they receive no action.
    The production respond_population defaults missing regions to KEEP.

    Args:
        controller_name: one of ALL_CONTROLLERS
        contexts:        {region_id: RegionalDecisionContext}
        D:               decision space dimension
        rule_config:     required if controller_name == "RULE_V1"

    Returns:
        Dict[int, str] — only active regions included
    """
    if controller_name not in ALL_CONTROLLERS:
        raise ValueError(
            f"Unknown controller '{controller_name}'. "
            f"Valid: {ALL_CONTROLLERS}"
        )

    actions: Dict[int, str] = {}
    for region_id, ctx in contexts.items():
        if not ctx.active:
            continue
        if controller_name in _GLOBAL_ACTION:
            actions[region_id] = _GLOBAL_ACTION[controller_name]
        else:  # RULE_V1
            if rule_config is None:
                raise ValueError("rule_config is required for RULE_V1")
            actions[region_id] = choose_action_rule_v1(ctx, rule_config)

    return actions
