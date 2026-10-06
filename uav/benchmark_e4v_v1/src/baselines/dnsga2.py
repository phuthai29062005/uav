"""D-NSGA-II: NSGA-II + change detection + partial random reinitialization."""

from __future__ import annotations

import numpy as np
from typing import Optional

from .nsga2 import NSGA2Baseline


class DNSGA2Baseline(NSGA2Baseline):

    def __init__(self, pop_size: int = 100, eta_cross: float = 20.0,
                 eta_mut: float = 20.0, seed: int = 42,
                 reinit_fraction: float = 0.2, n_detectors: int = 5):
        super().__init__(pop_size, eta_cross, eta_mut, seed)
        self.reinit_fraction = reinit_fraction
        self.n_detectors = n_detectors
        self._detector_X: Optional[np.ndarray] = None
        self._detector_F: Optional[np.ndarray] = None

    def initialize(self, xl: np.ndarray, xu: np.ndarray, n_var: int) -> np.ndarray:
        X = super().initialize(xl, xu, n_var)
        self._init_detectors()
        return X

    def _init_detectors(self):
        idx = self.rng.choice(self.pop_size, size=min(self.n_detectors, self.pop_size), replace=False)
        self._detector_X = self.X[idx].copy()
        self._detector_F = None

    def detect_change(self, evaluate_fn) -> bool:
        if self._detector_X is None:
            return False
        new_F = evaluate_fn(self._detector_X)
        if self._detector_F is None:
            self._detector_F = new_F
            return False
        changed = not np.allclose(new_F, self._detector_F, atol=1e-10)
        self._detector_F = new_F
        return changed

    def respond_to_change(self):
        n_reinit = max(1, int(self.reinit_fraction * self.pop_size))
        idx = self.rng.choice(self.pop_size, size=n_reinit, replace=False)
        self.X[idx] = self.rng.uniform(self.xl, self.xu, size=(n_reinit, self.n_var))
        self._init_detectors()
