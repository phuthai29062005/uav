"""Base class for dynamic multi-objective benchmark problems."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class DynamicBenchmarkProblem(ABC):
    """Abstract base for dynamic multi-objective optimization problems.

    Time formula (CEC2018 convention):
        t = (1/nt) * floor(tau / taut)
    where tau is the generation counter, nt controls speed, taut controls frequency.
    """

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
        return (1.0 / self.nt) * int(tau // self.taut)

    def set_tau(self, tau: int):
        self._tau = tau

    def advance(self):
        self._tau += 1

    def is_change_point(self) -> bool:
        return self._tau > 0 and (self._tau % self.taut == 0)

    @abstractmethod
    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Evaluate objectives. X: (N, n_var) -> F: (N, n_obj)."""

    @abstractmethod
    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        """Sample from true Pareto front in objective space. Returns (n_points, n_obj)."""

    def reset(self):
        self._tau = 0
