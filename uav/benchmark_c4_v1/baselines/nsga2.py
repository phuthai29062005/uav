"""NSGA-II baseline for dynamic benchmarks.

Standard NSGA-II with no change detection or response mechanism.
After each environmental change, the algorithm simply continues with
its current population. This is the static baseline.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def _fast_non_dominated_sort(F: np.ndarray):
    """Fully vectorized non-dominated sort using broadcasting."""
    N = len(F)
    if N == 0:
        return [], np.array([], dtype=int)

    # dom_matrix[i,j] = True if i dominates j
    # i dominates j iff all(F[i] <= F[j]) and any(F[i] < F[j])
    diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]  # (N, N, M)
    all_leq = np.all(diff <= 0, axis=2)  # (N, N)
    any_lt = np.any(diff < 0, axis=2)    # (N, N)
    dom_matrix = all_leq & any_lt        # (N, N)
    np.fill_diagonal(dom_matrix, False)

    n_dom = dom_matrix.sum(axis=0).astype(int)  # how many dominate each
    rank = np.full(N, -1, dtype=int)

    # Build S lists from dom_matrix
    S = [np.where(dom_matrix[p])[0].tolist() for p in range(N)]

    fronts = [[]]
    for p in range(N):
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    return [f for f in fronts if f], rank


def _crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, float("inf"))

    F_front = F[front]
    dist = np.zeros(n)

    for m in range(F_front.shape[1]):
        sorted_idx = np.argsort(F_front[:, m])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
        if f_range < 1e-12:
            continue
        for i in range(1, n - 1):
            dist[sorted_idx[i]] += (F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]) / f_range

    return dist


def _sbx_crossover_batch(parents1: np.ndarray, parents2: np.ndarray,
                         xl: np.ndarray, xu: np.ndarray, eta: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized SBX crossover for a batch of parent pairs."""
    n_pairs, n_var = parents1.shape
    children1, children2 = parents1.copy(), parents2.copy()

    # Which genes to crossover (50% chance per gene)
    cross_mask = rng.random((n_pairs, n_var)) <= 0.5
    # Skip genes where parents are identical
    diff_mask = np.abs(parents1 - parents2) >= 1e-14
    mask = cross_mask & diff_mask

    u = rng.random((n_pairs, n_var))
    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))

    c1 = 0.5 * ((1 + beta) * parents1 + (1 - beta) * parents2)
    c2 = 0.5 * ((1 - beta) * parents1 + (1 + beta) * parents2)

    children1 = np.where(mask, c1, children1)
    children2 = np.where(mask, c2, children2)

    return np.clip(children1, xl, xu), np.clip(children2, xl, xu)


def _polynomial_mutation_batch(X: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                               eta: float, prob: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Vectorized polynomial mutation for a batch of individuals."""
    children = X.copy()
    n_ind, n_var = X.shape

    mut_mask = rng.random((n_ind, n_var)) <= prob
    delta = xu - xl
    valid_mask = delta >= 1e-14
    mask = mut_mask & valid_mask[np.newaxis, :]

    u = rng.random((n_ind, n_var))
    deltaq = np.where(u < 0.5,
                      (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                      1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))

    children = np.where(mask, X + deltaq * delta[np.newaxis, :], children)
    return np.clip(children, xl, xu)


class NSGA2Baseline:
    """Standard NSGA-II with no dynamic response."""

    def __init__(self, pop_size: int = 100, eta_cross: float = 20.0,
                 eta_mut: float = 20.0, seed: int = 42):
        self.pop_size = pop_size
        self.eta_cross = eta_cross
        self.eta_mut = eta_mut
        self.rng = np.random.default_rng(seed)
        self.X: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self._rank: Optional[np.ndarray] = None

    def initialize(self, xl: np.ndarray, xu: np.ndarray, n_var: int) -> np.ndarray:
        self.xl = xl
        self.xu = xu
        self.n_var = n_var
        self.X = self.rng.uniform(xl, xu, size=(self.pop_size, n_var))
        return self.X

    def set_fitness(self, F: np.ndarray):
        self.F = F
        self._rank = None

    def step(self) -> np.ndarray:
        """One generation. Returns offspring to evaluate."""
        # Tournament selection: pick parent indices
        idx = self.rng.integers(0, self.pop_size, size=(self.pop_size // 2, 4))
        p1_idx = np.array([self._tournament(idx[i, 0], idx[i, 1])
                           for i in range(len(idx))])
        p2_idx = np.array([self._tournament(idx[i, 2], idx[i, 3])
                           for i in range(len(idx))])

        # Vectorized crossover
        children1, children2 = _sbx_crossover_batch(
            self.X[p1_idx], self.X[p2_idx], self.xl, self.xu,
            self.eta_cross, self.rng
        )

        # Vectorized mutation
        offspring = np.vstack([children1, children2])[:self.pop_size]
        offspring = _polynomial_mutation_batch(
            offspring, self.xl, self.xu, self.eta_mut,
            1.0 / self.n_var, self.rng
        )
        return offspring

    def select(self, offspring_X: np.ndarray, offspring_F: np.ndarray):
        """Environmental selection from combined parent+offspring."""
        combined_X = np.vstack([self.X, offspring_X])
        combined_F = np.vstack([self.F, offspring_F])

        fronts, rank = _fast_non_dominated_sort(combined_F)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
            else:
                remaining = self.pop_size - len(selected)
                cd = _crowding_distance(combined_F, front)
                sorted_by_cd = np.argsort(-cd)
                for idx in sorted_by_cd[:remaining]:
                    selected.append(front[idx])
                break

        self.X = combined_X[selected]
        self.F = combined_F[selected]
        self._rank = None

    def _tournament(self, i: int, j: int) -> int:
        if self._rank is None:
            _, self._rank = _fast_non_dominated_sort(self.F)
        if self._rank[i] < self._rank[j]:
            return i
        elif self._rank[j] < self._rank[i]:
            return j
        return i if self.rng.random() < 0.5 else j
