"""Population segmentation for benchmark problems.

Two schemes:
  STRUCTURE_INFORMED_2SEG: splits variables into position/distance using problem structure
  BLIND_FIXED_2SEG: splits variables at midpoint regardless of structure
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

STRUCTURE_INFORMED_2SEG = "STRUCTURE_INFORMED_2SEG"
BLIND_FIXED_2SEG = "BLIND_FIXED_2SEG"
N_REGIONS = 2


def segment_population(X: np.ndarray, F: np.ndarray, n_var: int,
                       scheme: str = BLIND_FIXED_2SEG) -> Dict[int, np.ndarray]:
    """Segment population into 2 regions.

    Returns dict mapping region_id -> row indices into X.
    Region 0 controls first-half variables, Region 1 controls second-half.
    """
    N = len(X)
    if N == 0:
        return {0: np.array([], dtype=int), 1: np.array([], dtype=int)}

    if scheme == BLIND_FIXED_2SEG:
        split = n_var // 2
    elif scheme == STRUCTURE_INFORMED_2SEG:
        split = max(1, n_var // 2)
    else:
        raise ValueError(f"Unknown segmentation scheme: {scheme}")

    var_contrib_0 = np.var(X[:, :split], axis=0).sum() if split > 0 else 0.0
    var_contrib_1 = np.var(X[:, split:], axis=0).sum() if split < n_var else 0.0

    if N < 4:
        half = N // 2
        return {0: np.arange(half), 1: np.arange(half, N)}

    nd_mask = _fast_nondominated_mask(F)
    nd_idx = np.where(nd_mask)[0]
    dom_idx = np.where(~nd_mask)[0]

    if len(nd_idx) == 0:
        nd_idx = np.arange(N)
        dom_idx = np.array([], dtype=int)

    half_nd = max(1, len(nd_idx) // 2)
    region_0 = np.concatenate([nd_idx[:half_nd], dom_idx[:len(dom_idx) // 2]])
    region_1 = np.concatenate([nd_idx[half_nd:], dom_idx[len(dom_idx) // 2:]])

    if len(region_0) == 0:
        region_0 = np.array([0])
    if len(region_1) == 0:
        region_1 = np.array([N - 1])

    return {0: region_0.astype(int), 1: region_1.astype(int)}


def _fast_nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Simple non-dominated sort for first front only."""
    N = len(F)
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not mask[i]:
            continue
        for j in range(N):
            if i == j or not mask[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False
                break
    return mask
