"""dMOP1-dMOP3 dynamic benchmark problems.

References:
    C.-K. Goh, K.C. Tan, "A competitive-cooperative coevolutionary paradigm
    for dynamic multiobjective optimization," IEEE TEVC, 2009.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import DynamicBenchmarkProblem


class dMOP1(DynamicBenchmarkProblem):
    """dMOP1: 2-obj, changing curvature of PF."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        return H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        H = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.sum(X[:, 1:] ** 2, axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        H = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class dMOP2(DynamicBenchmarkProblem):
    """dMOP2: 2-obj, PS changes while PF stays fixed (convex)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        return G

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.sum((X[:, 1:] - G) ** 2, axis=1) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class dMOP3(DynamicBenchmarkProblem):
    """dMOP3: 2-obj, both PS and PF change."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        r = int(np.floor(abs(G) * (self.n_var - 1))) + 1
        r = min(r, self.n_var - 1)
        r = max(r, 1)
        return G, r

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, r = self._params(t)
        f1 = np.mean(X[:, :r], axis=1)
        g = 1.0 + 9.0 * np.sum((X[:, r:] - G) ** 2, axis=1) / max(1, self.n_var - r)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])
