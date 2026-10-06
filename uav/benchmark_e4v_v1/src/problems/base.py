"""Base class for CEC2018 dynamic multi-objective benchmark problems.

Time model from official MATLAB cec2018_DF.m (S2):
    T0 = 50
    tau_tmp = max(tau + taut - (T0 + 1), 0)
    t = (1/nt) * floor(tau_tmp / taut)

The first change occurs after T0 generations, i.e. at generation (T0+1).
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

T0 = 50


class DynamicBenchmarkProblem(ABC):

    def __init__(self, n_var: int, n_obj: int, nt: int = 10, taut: int = 10,
                 xl: Optional[np.ndarray] = None, xu: Optional[np.ndarray] = None):
        self.n_var = n_var
        self.n_obj = n_obj
        self.nt = nt
        self.taut = taut
        self.xl = xl if xl is not None else np.zeros(n_var)
        self.xu = xu if xu is not None else np.ones(n_var)
        self._tau = 0

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def time(self, tau: Optional[int] = None) -> float:
        if tau is None:
            tau = self._tau
        tau_tmp = max(tau + self.taut - (T0 + 1), 0)
        return (1.0 / self.nt) * int(tau_tmp // self.taut)

    def set_tau(self, tau: int):
        self._tau = tau

    def advance(self):
        self._tau += 1

    def is_change_point(self) -> bool:
        if self._tau <= T0:
            return False
        return (self._tau - T0 - 1) % self.taut == 0 and self._tau > T0

    def total_fe(self, pop_size: int, n_changes: int) -> int:
        return pop_size * (n_changes * self.taut + T0)

    @abstractmethod
    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Evaluate objectives. X: (N, n_var) -> F: (N, n_obj)."""

    @abstractmethod
    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        """Sample from true Pareto front. Returns (n_points, n_obj)."""

    def reset(self):
        self._tau = 0
