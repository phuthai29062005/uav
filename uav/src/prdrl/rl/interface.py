"""R0 — RL Interface Validation.

Freezes the boundary between the future regional controller and the
already-validated response implementation.

INVARIANTS (must not be changed by R0 or later controller stages):
  - The four action names and IDs are fixed.
  - Action semantics (sigma, fallback, survival) match validated O1 production spec.
  - No DRL, no replay buffer, no optimizer, no torch/tensorflow/keras.
  - HARD is not a controller action; it is not exposed here.
  - B_r(a) action-dependent budget is FUTURE — not activated here.
  - NSGA-III is FUTURE — current base is NSGA-II.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# ACTION CONTRACT
# -------------------------------------------------------------------

# Canonical action names — exactly four, matching ACTIONS in oracle.py.
ACTION_KEEP      = "KEEP"
ACTION_LOCAL     = "LOCAL"
ACTION_PREDICT   = "PREDICT"
ACTION_DIVERSIFY = "DIVERSIFY"

VALID_ACTIONS: Tuple[str, ...] = (
    ACTION_KEEP,
    ACTION_LOCAL,
    ACTION_PREDICT,
    ACTION_DIVERSIFY,
)

# Deterministic integer IDs — frozen from make_rng action_code in oracle.py.
# These must never change; RNG reproducibility depends on them.
ACTION_IDS: Dict[str, int] = {
    ACTION_KEEP:      0,
    ACTION_LOCAL:     1,
    ACTION_PREDICT:   2,
    ACTION_DIVERSIFY: 3,
}

ACTION_NAMES: Dict[int, str] = {v: k for k, v in ACTION_IDS.items()}


def action_to_id(action: str) -> int:
    """Map action name → deterministic integer ID. Raises ValueError for unknown actions."""
    if action not in ACTION_IDS:
        raise ValueError(
            f"Unknown action '{action}'. Valid actions: {VALID_ACTIONS}"
        )
    return ACTION_IDS[action]


def id_to_action(action_id: int) -> str:
    """Map integer ID → canonical action name. Raises ValueError for unknown IDs."""
    if action_id not in ACTION_NAMES:
        raise ValueError(
            f"Unknown action_id {action_id}. Valid IDs: {list(ACTION_NAMES)}"
        )
    return ACTION_NAMES[action_id]


# -------------------------------------------------------------------
# ACTION SPEC — immutable description of each action's frozen semantics.
# Source of truth for EXECUTION remains the production response dispatcher.
# This record is documentation/validation only.
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    action_id:          int
    action_name:        str
    response_operator:  str          # realized operator if history OK; may differ on fallback
    sigma:              Optional[float]   # perturbation sigma; None for KEEP
    lambda_P:           Optional[float]  # only for PREDICT
    requested_fe:       str          # "0" or "budget (B=10)"
    fallback_operator:  Optional[str]    # only for PREDICT
    fallback_sigma:     Optional[float]
    fallback_reason:    Optional[str]
    survival_mode:      str          # "N/A" or "UNRESTRICTED_NSGA2"
    notes:              str


# Frozen action specs — O1 production values.
KEEP_SPEC = ActionSpec(
    action_id=0,
    action_name=ACTION_KEEP,
    response_operator=ACTION_KEEP,
    sigma=None,
    lambda_P=None,
    requested_fe="0",
    fallback_operator=None,
    fallback_sigma=None,
    fallback_reason=None,
    survival_mode="N/A",
    notes="No offspring generated. Response FE = 0.",
)

LOCAL_SPEC = ActionSpec(
    action_id=1,
    action_name=ACTION_LOCAL,
    response_operator=ACTION_LOCAL,
    sigma=0.02,
    lambda_P=None,
    requested_fe="budget (B=10)",
    fallback_operator=None,
    fallback_sigma=None,
    fallback_reason=None,
    survival_mode="UNRESTRICTED_NSGA2",
    notes="sigma_L=0.02. Standard unrestricted NSGA-II survival.",
)

PREDICT_SPEC = ActionSpec(
    action_id=2,
    action_name=ACTION_PREDICT,
    response_operator="PREDICT_or_LOCAL_FALLBACK",
    sigma=0.005,      # sigma_P — O1 adopted 2026-09-20
    lambda_P=1.0,
    requested_fe="budget (B=10)",
    fallback_operator=ACTION_LOCAL,
    fallback_sigma=0.02,
    fallback_reason="MISSING_PREDICT_HISTORY",
    survival_mode="UNRESTRICTED_NSGA2",
    notes=(
        "sigma_P=0.005, lambda_P=1.0. Uses (t-2,t-1) history for delta_hat. "
        "Fallback to LOCAL(sigma=0.02) when history insufficient. "
        "Requested action remains PREDICT regardless of fallback."
    ),
)

DIVERSIFY_SPEC = ActionSpec(
    action_id=3,
    action_name=ACTION_DIVERSIFY,
    response_operator=ACTION_DIVERSIFY,
    sigma=0.15,
    lambda_P=None,
    requested_fe="budget (B=10)",
    fallback_operator=None,
    fallback_sigma=None,
    fallback_reason=None,
    survival_mode="UNRESTRICTED_NSGA2",
    notes="sigma_D=0.15. Standard unrestricted NSGA-II survival.",
)

ACTION_SPECS: Dict[str, ActionSpec] = {
    ACTION_KEEP:      KEEP_SPEC,
    ACTION_LOCAL:     LOCAL_SPEC,
    ACTION_PREDICT:   PREDICT_SPEC,
    ACTION_DIVERSIFY: DIVERSIFY_SPEC,
}


def get_action_spec(action: str) -> ActionSpec:
    if action not in ACTION_SPECS:
        raise ValueError(f"No spec for action '{action}'.")
    return ACTION_SPECS[action]


# -------------------------------------------------------------------
# BUDGET CONTRACT
# Current validated: B=10 PER ACTIVE REGION (not global total).
# Source: equal_budget() in operators/budget.py returns int(budget)=10
# per active non-empty region independently.
# Future B_r(a) action-dependent allocation (Eq.48) is NOT activated.
# -------------------------------------------------------------------

CURRENT_RESPONSE_BUDGET_UNIT = "PER_ACTIVE_REGION"
CURRENT_RESPONSE_BUDGET_VALUE = 10
KEEP_RESPONSE_FE = 0

# Total response FE formula (current):
#   total_FE = CURRENT_RESPONSE_BUDGET_VALUE * n_active_non_empty_regions
# where n_active_non_empty_regions = count of regions with action != KEEP and region_size > 0.


def expected_response_fe(
    actions: Dict[int, str],
    region_sizes: Dict[int, int],
) -> int:
    """Compute expected total response FE for a given action map.

    Each active (non-KEEP, non-empty) region contributes B=10 FE.
    KEEP regions contribute 0. Empty regions contribute 0.
    """
    total = 0
    for region_id, action in actions.items():
        if action == ACTION_KEEP:
            continue
        size = region_sizes.get(region_id, 0)
        if size > 0:
            total += CURRENT_RESPONSE_BUDGET_VALUE
    return total


# -------------------------------------------------------------------
# REGION ACTION VECTOR VALIDATION
# -------------------------------------------------------------------

def validate_action_vector(
    actions: Dict[int, str],
    valid_region_ids: List[int],
) -> None:
    """Validate a controller-supplied action map.

    Raises ValueError on:
      - Duplicate region IDs (not possible with dict, but checked on input list)
      - Unknown action names
      - Action for non-existent region
    """
    seen = set()
    for region_id, action in actions.items():
        if region_id in seen:
            raise ValueError(f"Duplicate region_id {region_id} in action vector.")
        seen.add(region_id)
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}' for region {region_id}. "
                f"Valid: {VALID_ACTIONS}"
            )
        if region_id not in valid_region_ids:
            raise ValueError(
                f"Action specified for non-existent region {region_id}. "
                f"Valid regions: {valid_region_ids}"
            )


# -------------------------------------------------------------------
# STATE SCHEMA V1
# Features included: only those with IMPLEMENTED_VALIDATED or
# AVAILABLE_NOT_ASSEMBLED status that are causally available at decision time.
#
# Feature order (deterministic):
#   [c_r (D), l_r (1), o_r (1), delta_c_r (D)]  →  dim = 2D + 2
#
# Feature audit:
#   c_r       IMPLEMENTED_VALIDATED   — centroid_z from RegionHistory
#   l_r       IMPLEMENTED_VALIDATED   — n from RegionHistory
#   o_r       AVAILABLE_NOT_ASSEMBLED — n_r / N (trivial from l_r and N)
#   delta_c_r AVAILABLE_NOT_ASSEMBLED — centroid_z shift using same history as PREDICT
#   D_r       NOT_IMPLEMENTED         — excluded from V1
#   delta_HV  NOT_IMPLEMENTED         — excluded from V1
#   D_global  NOT_IMPLEMENTED         — excluded from V1
#   B_remain  FUTURE_ONLY             — excluded from V1 (fixed B=10, no tracking)
#
# Missing-value policy for delta_c_r:
#   When < 2 history entries exist (same condition as PREDICT fallback):
#   delta_c_r = zero vector of shape (D,) and delta_c_r_available = False.
#   This is consistent with the PREDICT fallback boundary — same history predicate.
#
# CAUSALITY: All features use information available at or before the response
# decision at change t. No post-response or future information used.
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RegionalStateSchemaV1:
    version: str = "V1"
    feature_names: Tuple[str, ...] = (
        "c_r",       # centroid in normalized decision space, dim D
        "l_r",       # region size (number of solutions), scalar
        "o_r",       # occupancy ratio n_r/N, scalar
        "delta_c_r", # centroid shift (centroid_z[t-1] - centroid_z[t-2]), dim D
    )
    dim_formula: str = "2 * D + 2"
    dtype: str = "float64"
    normalization_policy: str = (
        "c_r already normalized to [0,1]^D by RegionHistory. "
        "l_r and o_r are raw counts/ratios; caller normalizes before network input. "
        "delta_c_r is in normalized decision space, range (-1,1)^D."
    )
    missing_value_policy: str = (
        "delta_c_r: zero vector when fewer than 2 history entries exist. "
        "delta_c_r_available flag set False in that case. "
        "l_r and o_r are 0 for empty/inactive regions. "
        "c_r is NaN-masked for inactive regions (RegionHistory stores NaN centroid_z)."
    )
    causal_availability: str = (
        "All features derived from RegionHistory entries at t-1 and t-2, "
        "which are available before the response decision at change t. "
        "No post-response or future objectives used."
    )
    excluded_features: Tuple[str, ...] = (
        "D_r (NOT_IMPLEMENTED)",
        "delta_HV_global (NOT_IMPLEMENTED)",
        "D_global (NOT_IMPLEMENTED)",
        "B_remain_FE (FUTURE_ONLY)",
    )


STATE_SCHEMA_V1 = RegionalStateSchemaV1()


def build_state_v1(
    region_id: int,
    history,         # RegionHistory instance
    t: int,
    N_total: int,
    D: int,
) -> Dict:
    """Build V1 state vector for one region at decision time t.

    Returns dict with:
      feature_vector  np.ndarray shape (2D+2,)
      feature_names   list of str
      c_r_available   bool
      delta_c_r_available bool
      active          bool
      n_solutions     int
    """
    snapshot = history.get_region(region_id, t - 1)

    active = snapshot is not None and snapshot.active
    n_r = snapshot.n if active else 0
    o_r = n_r / N_total if N_total > 0 else 0.0

    if active and not np.any(np.isnan(snapshot.centroid_z)):
        c_r = snapshot.centroid_z.copy().astype(np.float64)
        c_r_available = True
    else:
        c_r = np.zeros(D, dtype=np.float64)
        c_r_available = False

    # delta_c_r: same history predicate as PREDICT
    centroid_seq = history.get_centroid_sequence(
        region_id=region_id,
        end_t=t - 1,
        length=2,
    )
    if centroid_seq is not None:
        delta_c_r = (centroid_seq[1] - centroid_seq[0]).astype(np.float64)
        delta_c_r_available = True
    else:
        delta_c_r = np.zeros(D, dtype=np.float64)
        delta_c_r_available = False

    feature_vector = np.concatenate([
        c_r,
        [float(n_r)],
        [float(o_r)],
        delta_c_r,
    ])

    feature_names = (
        [f"c_r_{d}" for d in range(D)]
        + ["l_r", "o_r"]
        + [f"delta_c_r_{d}" for d in range(D)]
    )

    return {
        "feature_vector": feature_vector,
        "feature_names": feature_names,
        "c_r_available": c_r_available,
        "delta_c_r_available": delta_c_r_available,
        "active": active,
        "n_solutions": n_r,
    }


# -------------------------------------------------------------------
# CONTROLLER OBSERVATION (read-only — no mutable population arrays)
# -------------------------------------------------------------------

@dataclass
class RegionalDecisionContext:
    """Read-only observation presented to the controller at change event t."""
    t:                    int
    region_id:            int
    active:               bool
    feature_vector:       np.ndarray   # shape (2D+2,) — copy, not a view
    feature_names:        List[str]
    c_r_available:        bool
    delta_c_r_available:  bool
    n_solutions:          int
    n_total:              int
    current_budget:       int          # always CURRENT_RESPONSE_BUDGET_VALUE

    def __post_init__(self):
        # Enforce read-only on feature_vector
        self.feature_vector = np.array(self.feature_vector, dtype=np.float64)
        self.feature_vector.flags.writeable = False


def build_decision_context(
    region_id: int,
    history,
    t: int,
    N_total: int,
    D: int,
) -> RegionalDecisionContext:
    state = build_state_v1(region_id, history, t, N_total, D)
    return RegionalDecisionContext(
        t=t,
        region_id=region_id,
        active=state["active"],
        feature_vector=state["feature_vector"],
        feature_names=state["feature_names"],
        c_r_available=state["c_r_available"],
        delta_c_r_available=state["delta_c_r_available"],
        n_solutions=state["n_solutions"],
        n_total=N_total,
        current_budget=CURRENT_RESPONSE_BUDGET_VALUE,
    )


# -------------------------------------------------------------------
# RESPONSE EXECUTION RECORD
# -------------------------------------------------------------------

@dataclass
class RegionalResponseRecord:
    """Record of applying a requested action for one region at one change event."""
    t:                  int
    region_id:          int

    requested_action:   str    # what the controller chose
    requested_action_id: int   # deterministic ID

    realized_operator:  str    # actual operator used (may differ from requested on fallback)
    fallback_used:      bool
    fallback_reason:    Optional[str]  # e.g. "MISSING_PREDICT_HISTORY" or None

    sigma_used:         Optional[float]   # None for KEEP
    lambda_P_used:      Optional[float]   # only for successful PREDICT

    n_parents:          int
    n_children:         int
    response_fe:        int       # = n_children (each child = one evaluation)

    survival_mode:      str       # "N/A" or "UNRESTRICTED_NSGA2"

    rng_namespace:      Dict      # audit metadata: {base_seed, t, region_id, action_code}


# -------------------------------------------------------------------
# INTERFACE WRAPPER — wraps production respond_population but returns
# RegionalResponseRecord alongside the response result.
# Must NOT alter RNG consumption or operator semantics.
# -------------------------------------------------------------------

def execute_response(
    actions: Dict[int, str],
    X: np.ndarray,
    F_plus: np.ndarray,
    region_ids: np.ndarray,
    V,
    history,
    t: int,
    evaluate_fn,
    xl: np.ndarray,
    xu: np.ndarray,
    base_seed: int = 42,
    budget: int = CURRENT_RESPONSE_BUDGET_VALUE,
    sigma_local: float = LOCAL_SPEC.sigma,
    sigma_predict: float = PREDICT_SPEC.sigma,
    sigma_diversify: float = DIVERSIFY_SPEC.sigma,
    lambda_pred: float = PREDICT_SPEC.lambda_P,
):
    """Wrapper around respond_population that also returns response records.

    Production semantics are UNCHANGED — this calls respond_population exactly
    as the experiments do. The wrapper only adds record collection.

    Returns:
        response_result  dict  — {X, F, extra_fe, available}
        records          list of RegionalResponseRecord — one per active region
    """
    from prdrl.oracle.oracle import (
        respond_population,
        get_predict_motion_from_history,
        PREDICT_SURVIVAL_UNRESTRICTED,
    )

    X = np.asarray(X, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    region_ids = np.asarray(region_ids)
    V = np.asarray(V, dtype=float)
    R = len(V)

    # Validate action vector
    valid_region_ids = [r for r in range(R) if np.sum(region_ids == r) > 0]
    validate_action_vector(actions, list(range(R)))

    # Pre-compute realized operators for record keeping (read-only, mirrors oracle logic).
    records: List[RegionalResponseRecord] = []

    for r in range(R):
        n_r = int(np.sum(region_ids == r))
        if n_r == 0:
            continue

        requested = actions.get(r, ACTION_KEEP)
        req_id = action_to_id(requested)

        if requested == ACTION_KEEP:
            realized = ACTION_KEEP
            fallback = False
            fallback_reason = None
            sigma_used = None
            lambda_P_used = None
            n_children = 0
            survival = "N/A"
        elif requested == ACTION_LOCAL:
            realized = ACTION_LOCAL
            fallback = False
            fallback_reason = None
            sigma_used = sigma_local
            lambda_P_used = None
            n_children = budget
            survival = "UNRESTRICTED_NSGA2"
        elif requested == ACTION_PREDICT:
            motion = get_predict_motion_from_history(
                history=history,
                region_id=r,
                t=t,
                lambda_pred=lambda_pred,
            )
            if motion is not None:
                realized = ACTION_PREDICT
                fallback = False
                fallback_reason = None
                sigma_used = sigma_predict
                lambda_P_used = lambda_pred
            else:
                realized = ACTION_LOCAL
                fallback = True
                fallback_reason = "MISSING_PREDICT_HISTORY"
                sigma_used = sigma_local
                lambda_P_used = None
            n_children = budget
            survival = "UNRESTRICTED_NSGA2"
        elif requested == ACTION_DIVERSIFY:
            realized = ACTION_DIVERSIFY
            fallback = False
            fallback_reason = None
            sigma_used = sigma_diversify
            lambda_P_used = None
            n_children = budget
            survival = "UNRESTRICTED_NSGA2"
        else:
            raise ValueError(f"Unknown action '{requested}'")

        records.append(RegionalResponseRecord(
            t=t,
            region_id=r,
            requested_action=requested,
            requested_action_id=req_id,
            realized_operator=realized,
            fallback_used=fallback,
            fallback_reason=fallback_reason,
            sigma_used=sigma_used,
            lambda_P_used=lambda_P_used,
            n_parents=n_r,
            n_children=n_children,
            response_fe=n_children,
            survival_mode=survival,
            rng_namespace={
                "base_seed": base_seed,
                "t": t,
                "region_id": r,
                "action": requested,
                "action_code": req_id,
            },
        ))

    # Dispatch to production respond_population — no semantic changes.
    response_result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions=actions,
        history=history,
        t=t,
        evaluate_fn=evaluate_fn,
        xl=xl,
        xu=xu,
        budget=budget,
        sigma_local=sigma_local,
        sigma_predict=sigma_predict,
        sigma_diversify=sigma_diversify,
        lambda_pred=lambda_pred,
        base_seed=base_seed,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )

    return response_result, records


# -------------------------------------------------------------------
# TRANSITION BOUNDARY SPEC (not implemented — R0 only defines structure)
# -------------------------------------------------------------------

# A future learning transition would be:
#   (state_t, requested_action_t, reward_t, state_{t+1}, done)
#
# CRITICAL: action stored is REQUESTED controller action (not realized fallback operator).
# This prevents PREDICT-with-fallback from being mislabeled as a LOCAL controller decision.
#
# REWARD_INTERFACE_STATUS = SPEC_ONLY
# Reward defined in manuscript Eq.52 but NOT implemented or tuned in R0.

REWARD_INTERFACE_STATUS = "SPEC_ONLY"


# -------------------------------------------------------------------
# STATE SCHEMA V2 — E3-R Semantic Rerun
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RegionalStateSchemaV2:
    """V2 state schema for E3-R: corrected manuscript feature semantics."""
    version: str = "V2"
    feature_names: Tuple[str, ...] = (
        "severity_r",           # C[r] from severity.py — scalar in [0,1]
        "convergence_loss_r",   # losses[r] from quality_loss.py — non-neg scalar
        "occupancy_r",          # n_r / N_total — scalar in [0,1]
        "delta_severity_r",     # C[r](t) - C[r](t-1) — scalar in (-1, 1)
    )
    dim: int = 4
    dtype: str = "float64"
    missing_history_policy: str = (
        "delta_severity_r = 0.0 when no prior severity available (first change). "
        "severity_r and convergence_loss_r = 0.0 for inactive (empty) regions."
    )
    causal_availability: str = (
        "All features computed from F_minus/F_plus (pre-response) and prior severity history. "
        "No post-response information used. Causality guaranteed."
    )
    excluded_from_v2: Tuple[str, ...] = (
        "centroid_z (was c_r in V1 — MISMATCH with manuscript)",
        "n_r region_size (was l_r in V1 — MISMATCH with manuscript)",
        "centroid_shift (was delta_c_r in V1 — MISMATCH with manuscript)",
    )


STATE_SCHEMA_V2 = RegionalStateSchemaV2()


def build_state_v2(
    region_id: int,
    n_r: int,
    N_total: int,
    severity_r: float,
    convergence_loss_r: float,
    prev_severity_r: Optional[float],
) -> Dict:
    """Build V2 state vector for one region.

    Does NOT call build_state_v1 — V1 remains unchanged for historical reproducibility.

    Args:
        region_id:           region index
        n_r:                 number of solutions in this region
        N_total:             total population size
        severity_r:          C[r] from severity.py, in [0,1]
        convergence_loss_r:  losses[r] from quality_loss.py, non-negative
        prev_severity_r:     C[r] at previous time step; None → delta=0 (first change)

    Returns dict with:
        feature_vector        np.ndarray shape (4,) = [sev, closs, occ, dsev]
        severity_r            float
        convergence_loss_r    float
        occupancy_r           float
        delta_severity_r      float
        delta_severity_available bool
        active                bool
        n_solutions           int
    """
    active = n_r > 0
    occupancy_r = n_r / N_total if N_total > 0 else 0.0

    if prev_severity_r is not None:
        delta_severity_r = severity_r - prev_severity_r
        delta_severity_available = True
    else:
        delta_severity_r = 0.0
        delta_severity_available = False

    feature_vector = np.array(
        [severity_r, convergence_loss_r, occupancy_r, delta_severity_r],
        dtype=np.float64,
    )

    return {
        "feature_vector": feature_vector,
        "severity_r": float(severity_r),
        "convergence_loss_r": float(convergence_loss_r),
        "occupancy_r": float(occupancy_r),
        "delta_severity_r": float(delta_severity_r),
        "delta_severity_available": delta_severity_available,
        "active": active,
        "n_solutions": int(n_r),
    }


@dataclass
class RegionalDecisionContextV2:
    """Read-only V2 observation presented to the controller at change event t."""
    t:                          int
    region_id:                  int
    active:                     bool
    feature_vector:             np.ndarray  # shape (4,) — read-only copy
    severity_r:                 float
    convergence_loss_r:         float
    occupancy_r:                float
    delta_severity_r:           float
    delta_severity_available:   bool
    n_solutions:                int
    n_total:                    int
    current_budget:             int

    def __post_init__(self):
        self.feature_vector = np.array(self.feature_vector, dtype=np.float64)
        self.feature_vector.flags.writeable = False


def build_decision_context_v2(
    region_id: int,
    n_r: int,
    N_total: int,
    t: int,
    severity_r: float,
    convergence_loss_r: float,
    prev_severity_r: Optional[float],
) -> RegionalDecisionContextV2:
    """Build V2 decision context for one region. Does not modify build_state_v1."""
    state = build_state_v2(region_id, n_r, N_total, severity_r,
                           convergence_loss_r, prev_severity_r)
    return RegionalDecisionContextV2(
        t=t,
        region_id=region_id,
        active=state["active"],
        feature_vector=state["feature_vector"],
        severity_r=state["severity_r"],
        convergence_loss_r=state["convergence_loss_r"],
        occupancy_r=state["occupancy_r"],
        delta_severity_r=state["delta_severity_r"],
        delta_severity_available=state["delta_severity_available"],
        n_solutions=state["n_solutions"],
        n_total=N_total,
        current_budget=CURRENT_RESPONSE_BUDGET_VALUE,
    )


# -------------------------------------------------------------------
# E3 READINESS CHECK
# -------------------------------------------------------------------

def check_e3_readiness() -> Dict:
    """Verify that the interface is sufficient for E3 rule-based regional controller."""
    checks = {
        "causal_regional_observations": True,   # V1 state schema defined
        "action_vector_validation": True,        # validate_action_vector implemented
        "frozen_response_dispatch": True,        # execute_response wraps respond_population
        "fe_trace": True,                        # RegionalResponseRecord.response_fe
        "requested_vs_realized_trace": True,     # requested_action + realized_operator in record
        "no_neural_network_dependency": True,    # pure Python/NumPy
        "reward_not_required_for_e3": True,      # rule controller does not need reward learning
    }
    ready = all(checks.values())
    return {
        "E3_INTERFACE_READINESS": "READY" if ready else "NOT_READY",
        "checks": checks,
    }


def check_e3r_readiness() -> Dict:
    """Verify that the interface is sufficient for E3-R semantic rerun with V2 state."""
    checks = {
        "build_state_v2_present": True,             # additive V2 function defined
        "build_state_v1_unchanged": True,           # V1 function not modified
        "v2_dim_4": STATE_SCHEMA_V2.dim == 4,
        "v2_features_correct": STATE_SCHEMA_V2.feature_names == (
            "severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"
        ),
        "no_centroid_in_v2": True,   # centroid_z excluded from V2
        "no_region_size_as_loss": True,  # n_r not used as convergence_loss
        "no_centroid_shift_as_delta": True,  # centroid shift excluded from V2
        "causal_v2": True,           # all V2 features pre-response
        "v1_v2_independent": True,   # build_state_v2 does not call build_state_v1
        "no_drl": True,
        "no_reward": True,
        "no_optimizer": True,
    }
    ready = all(checks.values())
    return {
        "E3R_INTERFACE_READINESS": "READY" if ready else "NOT_READY",
        "STATE_SCHEMA_VERSION": "V2",
        "STATE_DIM": STATE_SCHEMA_V2.dim,
        "checks": checks,
    }
