"""FE accounting ledger. Every objective-function call must be tracked."""

from __future__ import annotations

import numpy as np
from typing import Optional


class EvaluationLedger:

    def __init__(self, problem, budget: Optional[int] = None):
        self._problem = problem
        self._count = 0
        self._budget = budget

    @property
    def count(self) -> int:
        return self._count

    @property
    def budget(self) -> Optional[int]:
        return self._budget

    @property
    def remaining(self) -> Optional[int]:
        if self._budget is None:
            return None
        return max(0, self._budget - self._count)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        n = len(X)
        if self._budget is not None and self._count + n > self._budget:
            raise RuntimeError(
                f"FE budget exceeded: {self._count}+{n} > {self._budget}"
            )
        self._count += n
        return self._problem.evaluate(X, t=t)

    def reset(self):
        self._count = 0
