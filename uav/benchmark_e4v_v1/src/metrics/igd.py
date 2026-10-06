"""Inverted Generational Distance (IGD) and Mean IGD (MIGD).

Standard IGD as defined in the CEC2018 competition:
IGD(A, P*) = (1/|P*|) * sum_{p in P*} min_{a in A} ||a - p||_2

MIGD = (1/T) * sum_{t=1}^{T} IGD(A_t, P*_t)
"""

from __future__ import annotations

import numpy as np
from typing import List


def igd(approx: np.ndarray, reference: np.ndarray) -> float:
    if len(approx) == 0:
        return float("inf")
    approx = np.atleast_2d(approx)
    reference = np.atleast_2d(reference)
    dists = np.linalg.norm(reference[:, np.newaxis, :] - approx[np.newaxis, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def migd(igd_values: List[float]) -> float:
    if not igd_values:
        return float("inf")
    return float(np.mean(igd_values))
