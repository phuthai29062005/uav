"""CEC2018 DF1-DF14 dynamic benchmark problems.

References:
    S. Jiang et al., "Evolutionary Dynamic Multi-Objective Optimization:
    Benchmarks and Algorithm Comparisons," IEEE TEVC, 2017.
    CEC2018 Competition on Dynamic Multi-Objective Optimisation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import DynamicBenchmarkProblem


class DF1(DynamicBenchmarkProblem):
    """DF1: 2-obj, type I dynamics (shifting global optimum)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        H = 1.5 + G
        return G, H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, H = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, H = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class DF2(DynamicBenchmarkProblem):
    """DF2: 2-obj, type II dynamics (variable linkage changes)."""

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
        n = self.n_var
        r = int(np.floor(G * n / 2.0))
        r = max(1, min(r, n - 1))
        f1 = np.mean(X[:, :r], axis=1)
        g = 1.0 + np.sum((X[:, r:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** 0.5)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class DF3(DynamicBenchmarkProblem):
    """DF3: 2-obj, type III dynamics (changing density of solutions)."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        H = 1.5 + G
        return G, H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, H = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        h = g * (1.0 - np.sqrt(f1 / g))
        f2 = g - f1 ** H
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, H = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class DF4(DynamicBenchmarkProblem):
    """DF4: 2-obj, dynamics with changing PF shape."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        a = np.sin(0.5 * np.pi * t)
        H = 1.5 + a
        return a, H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        a, H = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + np.sum((X[:, 1:] - a) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, H = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class DF5(DynamicBenchmarkProblem):
    """DF5: 2-obj, dynamics with time-varying number of knees."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        w = int(np.floor(10.0 * G))
        w = max(1, abs(w))
        return G, w

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, w = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** 0.5 * np.cos(2.0 * w * np.pi * (f1 / g) ** 0.5) ** 2)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, w = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1) * np.cos(2.0 * w * np.pi * np.sqrt(f1)) ** 2
        return np.column_stack([f1, f2])


class DF6(DynamicBenchmarkProblem):
    """DF6: 2-obj, dynamics with time-varying disconnected PF."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        a = 0.5 + abs(G)
        return G, a

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, a = self._params(t)
        f1 = X[:, 0]
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** a)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, a = self._params(t)
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** a
        return np.column_stack([f1, f2])


class DF7(DynamicBenchmarkProblem):
    """DF7: 2-obj, dynamics with time-varying bias."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.ones(n_var) * 1.0
        xu = np.ones(n_var) * 4.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def _params(self, t):
        a = 2.0 + np.sin(0.5 * np.pi * t)
        return a

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        a = self._params(t)
        f1 = (1.0 + np.sum((X[:, 1:] - 1.0) ** 2, axis=1)) * X[:, 0]
        f2 = (1.0 + np.sum((X[:, 1:] - 1.0) ** 2, axis=1)) * (a / X[:, 0])
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        a = self._params(t)
        f1 = np.linspace(1.0, 4.0, n_points)
        f2 = a / f1
        return np.column_stack([f1, f2])


class DF8(DynamicBenchmarkProblem):
    """DF8: 2-obj, dynamics with time-varying curvature."""

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
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - f1 / g)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1
        return np.column_stack([f1, f2])


class DF9(DynamicBenchmarkProblem):
    """DF9: 2-obj, dynamics shifting both PS and PF."""

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
        f1 = X[:, 0] + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f2 = 1.0 - X[:, 0] + np.sum((X[:, 1:] - G) ** 2, axis=1)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1
        return np.column_stack([f1, f2])


class DF10(DynamicBenchmarkProblem):
    """DF10: 3-obj, extension of DF1 to three objectives."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        H = 2.5 + np.cos(0.5 * np.pi * t)
        return G, H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, H = self._params(t)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        f1 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        f2 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        f3 = g * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
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


class DF11(DynamicBenchmarkProblem):
    """DF11: 3-obj, time-varying convexity."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        H = 1.5 + G
        return G, H

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, H = self._params(t)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        f1 = g * np.cos(0.5 * np.pi * X[:, 0]) ** H * np.cos(0.5 * np.pi * X[:, 1]) ** H
        f2 = g * np.cos(0.5 * np.pi * X[:, 0]) ** H * np.sin(0.5 * np.pi * X[:, 1]) ** H
        f3 = g * np.sin(0.5 * np.pi * X[:, 0]) ** H
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, H = self._params(t)
        n_per_dim = int(np.ceil(np.sqrt(n_points)))
        u = np.linspace(0, 1, n_per_dim)
        v = np.linspace(0, 1, n_per_dim)
        U, V = np.meshgrid(u, v)
        U, V = U.ravel(), V.ravel()
        f1 = np.cos(0.5 * np.pi * U) ** H * np.cos(0.5 * np.pi * V) ** H
        f2 = np.cos(0.5 * np.pi * U) ** H * np.sin(0.5 * np.pi * V) ** H
        f3 = np.sin(0.5 * np.pi * U) ** H
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf


class DF12(DynamicBenchmarkProblem):
    """DF12: 3-obj, time-varying disconnected PF."""

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def _params(self, t):
        G = np.sin(0.5 * np.pi * t)
        k = int(np.floor(10.0 * abs(G)))
        k = max(1, k)
        return G, k

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G, k = self._params(t)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        f1 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        f2 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        f3 = g * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        _, k = self._params(t)
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


class DF13(DynamicBenchmarkProblem):
    """DF13: 3-obj, time-varying geometry."""

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
        f1 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        f2 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        f3 = g * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
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


class DF14(DynamicBenchmarkProblem):
    """DF14: 3-obj, time-varying degeneration."""

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
        f1 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        f2 = g * np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        f3 = g * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
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
