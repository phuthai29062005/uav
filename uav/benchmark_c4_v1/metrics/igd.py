"""Inverted Generational Distance (IGD) and Mean IGD (MIGD)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import List


def igd(approx: np.ndarray, reference: np.ndarray) -> float:
    """IGD: average minimum distance from each reference point to the approximation set.
    approx: (N, M), reference: (R, M) -> scalar."""
    if len(approx) == 0:
        return float("inf")
    approx = np.atleast_2d(approx)
    reference = np.atleast_2d(reference)
    dists = cdist(reference, approx, metric="euclidean")
    return float(np.mean(np.min(dists, axis=1)))


def migd(igd_values: List[float]) -> float:
    """Mean IGD across time steps."""
    if not igd_values:
        return float("inf")
    return float(np.mean(igd_values))
