"""Statistical analysis: Wilcoxon signed-rank with Holm correction, Friedman ranking."""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    diff = x - y
    valid = np.isfinite(diff)
    diff = diff[valid & (np.abs(diff) > 1e-15)]
    if len(diff) == 0:
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        return 0.0, 1.0
    return float(stat), float(p)


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(m)
    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = min(1.0, p_values[idx] * (m - rank))
    cummax = corrected[sorted_idx].copy()
    for i in range(1, len(cummax)):
        cummax[i] = max(cummax[i], cummax[i - 1])
    corrected[sorted_idx] = cummax
    return corrected.tolist()


def friedman_ranking(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Friedman ranking across algorithms.

    results: {algorithm_name: [metric_per_problem]}
    Returns: {algorithm_name: mean_rank} (lower = better for MIGD).
    """
    alg_names = sorted(results.keys())
    n_problems = len(results[alg_names[0]])
    for name in alg_names:
        if len(results[name]) != n_problems:
            raise ValueError(f"All algorithms must have the same number of results")

    matrix = np.array([results[name] for name in alg_names])
    valid_cols = np.all(np.isfinite(matrix), axis=0)
    matrix = matrix[:, valid_cols]
    if matrix.shape[1] == 0:
        return {name: float("nan") for name in alg_names}

    ranks = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        ranks[:, j] = stats.rankdata(col)

    mean_ranks = ranks.mean(axis=1)
    return {name: float(mean_ranks[i]) for i, name in enumerate(alg_names)}


def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Friedman chi-squared test. Returns (statistic, p_value)."""
    alg_names = sorted(results.keys())
    matrix = np.array([results[name] for name in alg_names]).T
    valid_rows = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[valid_rows]
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return 0.0, 1.0
    try:
        stat, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    except ValueError:
        return 0.0, 1.0
    return float(stat), float(p)


def pairwise_comparisons(results: Dict[str, List[float]], alpha: float = 0.05
                          ) -> List[Dict]:
    """All pairwise Wilcoxon tests with Holm correction."""
    alg_names = sorted(results.keys())
    pairs = []
    p_values = []

    for i in range(len(alg_names)):
        for j in range(i + 1, len(alg_names)):
            x = np.array(results[alg_names[i]])
            y = np.array(results[alg_names[j]])
            stat, p = wilcoxon_signed_rank(x, y)
            pairs.append({"alg1": alg_names[i], "alg2": alg_names[j],
                           "statistic": stat, "p_value": p})
            p_values.append(p)

    corrected = holm_correction(p_values)
    for idx, pair in enumerate(pairs):
        pair["p_corrected"] = corrected[idx]
        pair["significant"] = corrected[idx] < alpha
        x = np.array(results[pair["alg1"]])
        y = np.array(results[pair["alg2"]])
        diff = x - y
        valid = np.isfinite(diff)
        pair["mean_diff"] = float(np.nanmean(diff[valid])) if valid.any() else 0.0
        pair["winner"] = pair["alg1"] if pair["mean_diff"] < 0 else pair["alg2"]

    return pairs
