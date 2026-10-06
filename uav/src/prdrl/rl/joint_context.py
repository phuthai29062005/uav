"""
Branch-C centralized mixer state builder (frozen by E4K_central_state_contract).

central_state = concat(
    z_local_1 .. z_local_RMAX,     # 4 V3-normalized local features per fixed slot (inactive -> [0,0,0,0])
    active_mask_1 .. active_mask_RMAX,
    global_context_5               # one copy of the normalized global context
)
CENTRAL_STATE_DIM = 4*R_MAX + R_MAX + 5 = 5*R_MAX + 5 = 55 (R_MAX=10).

Slots are ordered by ASCENDING CANONICAL region_id — never by occupancy/severity/action/centroid/
quality/active-status. The active mask is mandatory: zero is a semantically valid V3 feature value,
so an inactive slot ([0,0,0,0], mask=0) must be distinguishable from an active slot whose normalized
features happen to be zero ([0,0,0,0], mask=1). No new state features are added (no region-id
embeddings, previous/future/oracle actions, reward history, or counterfactual outcomes).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

R_MAX = 10
LOCAL_DIM = 4          # severity, convergence_loss, occupancy, delta_severity (V3-normalized)
GLOBAL_DIM = 5         # active_region_fraction, mean_severity, max_severity, mean_conv_loss, max_conv_loss
CENTRAL_STATE_DIM = 5 * R_MAX + 5
LOCAL_FEATURE_ORDER = ("severity", "convergence_loss", "occupancy", "delta_severity")
GLOBAL_FEATURE_ORDER = ("active_region_fraction", "mean_severity", "max_severity",
                        "mean_convergence_loss", "max_convergence_loss")


def build_central_state(local4_slots: np.ndarray, active_mask: np.ndarray,
                        global_context: np.ndarray, r_max: int = R_MAX) -> np.ndarray:
    """Assemble the flat central state from already-slotted inputs.

    local4_slots: [r_max, 4] V3-normalized local features per canonical slot (inactive rows [0,0,0,0]).
    active_mask:  [r_max] in {0,1}. global_context: [5]. Returns [5*r_max+5] float32.
    """
    local4_slots = np.asarray(local4_slots, dtype=np.float32)
    active_mask = np.asarray(active_mask, dtype=np.float32)
    global_context = np.asarray(global_context, dtype=np.float32)
    if local4_slots.shape != (r_max, LOCAL_DIM):
        raise ValueError(f"local4_slots must be [{r_max},{LOCAL_DIM}], got {local4_slots.shape}")
    if active_mask.shape != (r_max,):
        raise ValueError(f"active_mask must be [{r_max}], got {active_mask.shape}")
    if global_context.shape != (GLOBAL_DIM,):
        raise ValueError(f"global_context must be [{GLOBAL_DIM}], got {global_context.shape}")
    # enforce masking: inactive slots contribute zero local features
    masked_local = local4_slots * active_mask[:, None]
    flat = np.concatenate([masked_local.reshape(-1), active_mask, global_context]).astype(np.float32)
    if flat.shape[0] != 5 * r_max + 5:
        raise ValueError(f"central_state dim {flat.shape[0]} != {5*r_max+5}")
    return flat


def central_state_from_regions(active_local4_by_region: Dict[int, Sequence[float]],
                               global_context: Sequence[float], r_max: int = R_MAX) -> np.ndarray:
    """Assemble the central state from a map {region_id: local4} of ACTIVE regions plus the global
    context. Slots are filled in ascending canonical region_id regardless of input dict ordering;
    absent (inactive) slots get local4=[0,0,0,0] and mask=0. This is the production builder used at
    decision time; it reorganizes only current pre-response information."""
    local4_slots = np.zeros((r_max, LOCAL_DIM), dtype=np.float32)
    active_mask = np.zeros((r_max,), dtype=np.float32)
    for rid in sorted(active_local4_by_region.keys()):   # canonical ascending order
        if not (0 <= rid < r_max):
            raise ValueError(f"region_id {rid} out of range [0,{r_max})")
        v = np.asarray(active_local4_by_region[rid], dtype=np.float32)
        if v.shape != (LOCAL_DIM,):
            raise ValueError(f"region {rid} local4 must be [{LOCAL_DIM}], got {v.shape}")
        local4_slots[rid] = v
        active_mask[rid] = 1.0
    return build_central_state(local4_slots, active_mask, np.asarray(global_context, dtype=np.float32), r_max)
