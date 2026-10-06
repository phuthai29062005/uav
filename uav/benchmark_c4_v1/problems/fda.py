"""FDA1-FDA5 dynamic benchmark problems.

References:
    M. Farina et al., "Dynamic multiobjective optimization problems: test cases,
    approximations, and applications," IEEE TEVC, 2004.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import DynamicBenchmarkProblem


class FDA1(DynamicBenchmarkProblem):
    """FDA1: 2-obj, PS changes but PF stays fixed (convex)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.full(n_var, -1.0)
        xl[0] = 0.0
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
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class FDA2(DynamicBenchmarkProblem):
    """FDA2: 2-obj, PF shape changes over time."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.full(n_var, -1.0)
        xl[0] = 0.0
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        H = 2.0 + np.sin(0.5 * np.pi * t)
        return H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        H = self._params(t)
        f1 = X[:, 0]
        n2 = max(1, self.n_var // 2)
        g = 1.0 + np.sum(X[:, 1:n2] ** 2, axis=1)
        h = 1.0 - (f1 / g) ** (1.0 / H)
        f2 = g * h
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        H = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** (1.0 / H)
        return np.column_stack([f1, f2])


class FDA3(DynamicBenchmarkProblem):
    """FDA3: 2-obj, both PS and PF change."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.full(n_var, -1.0)
        xl[0] = 0.0
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        F_t = 10.0 ** (2.0 * np.sin(0.5 * np.pi * t))
        return G, F_t

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, F_t = self._params(t)
        f1 = np.sum(X[:, :1] ** F_t, axis=1)
        g = 1.0 + G + np.sum((X[:, 1:] - G) ** 2, axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, F_t = self._params(t)
        G = np.sin(0.5 * np.pi * t)
        g_star = 1.0 + G
        if g_star < 1e-12:
            return np.zeros((n_points, 2))
        x1 = np.linspace(0, 1, n_points)
        f1 = x1 ** F_t
        f2 = g_star * (1.0 - np.sqrt(f1 / g_star))
        return np.column_stack([f1, f2])


class FDA4(DynamicBenchmarkProblem):
    """FDA4: 3-obj, PS changes but PF stays fixed (DTLZ-like)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        return G

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = self._params(t)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        f1 = (1.0 + g) * np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        f2 = (1.0 + g) * np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        f3 = (1.0 + g) * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        n_per_dim = int(np.ceil(np.sqrt(n_points)))
        u = np.linspace(0, 1, n_per_dim)
        v = np.linspace(0, 1, n_per_dim)
        U, V = np.meshgrid(u, v)
        U, V = U.ravel(), V.ravel()
        f1 = np.cos(0.5 * np.pi * U) * np.cos(0.5 * np.pi * V)
        f2 = np.cos(0.5 * np.pi * U) * np.sin(0.5 * np.pi * V)
        f3 = np.sin(0.5 * np.pi * U)
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf


class FDA5(DynamicBenchmarkProblem):
    """FDA5: 3-obj, PF shape changes (DTLZ-like with time-varying curvature)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        F_t = 1.0 + 100.0 * np.sin(0.5 * np.pi * t) ** 4
        return G, F_t

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, F_t = self._params(t)
        g = np.sum((X[:, 2:] - G) ** 2, axis=1)
        f1 = (1.0 + g) * np.cos(0.5 * np.pi * X[:, 0] ** F_t) * np.cos(0.5 * np.pi * X[:, 1] ** F_t)
        f2 = (1.0 + g) * np.cos(0.5 * np.pi * X[:, 0] ** F_t) * np.sin(0.5 * np.pi * X[:, 1] ** F_t)
        f3 = (1.0 + g) * np.sin(0.5 * np.pi * X[:, 0] ** F_t)
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, F_t = self._params(t)
        n_per_dim = int(np.ceil(np.sqrt(n_points)))
        u = np.linspace(0, 1, n_per_dim)
        v = np.linspace(0, 1, n_per_dim)
        U, V = np.meshgrid(u, v)
        U, V = U.ravel(), V.ravel()
        f1 = np.cos(0.5 * np.pi * U ** F_t) * np.cos(0.5 * np.pi * V ** F_t)
        f2 = np.cos(0.5 * np.pi * U ** F_t) * np.sin(0.5 * np.pi * V ** F_t)
        f3 = np.sin(0.5 * np.pi * U ** F_t)
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf
