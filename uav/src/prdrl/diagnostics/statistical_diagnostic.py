"""Aggregation, crossed bootstrap, and O1 support logic for scenario diagnostics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


def aggregate_policy_env_scenario(records: List[Dict],
                                  metric_key: str,
                                  policy_key: str = "policy",
                                  env_key: str = "diagnostic_environment",
                                  scenario_key: str = "scenario") -> Dict:
    """Aggregate metric within policy × environment × scenario cells.

    Returns a dict keyed by (policy, env, scenario) -> mean of metric within cell.
    Also returns per-scenario matrices [n_policies, n_envs] for bootstrap."""
    cells = {}
    for rec in records:
        val = rec.get(metric_key)
        if val is None:
            continue
        key = (rec[policy_key], rec[env_key], rec[scenario_key])
        cells.setdefault(key, []).append(val)

    cell_means = {k: float(np.mean(v)) for k, v in cells.items()}

    policies = sorted(set(k[0] for k in cells.keys()))
    envs = sorted(set(k[1] for k in cells.keys()))
    scenario_matrices = {}
    for scen in SCENARIOS:
        mat = np.full((len(policies), len(envs)), np.nan)
        for pi, p in enumerate(policies):
            for ei, e in enumerate(envs):
                key = (p, e, scen)
                if key in cell_means:
                    mat[pi, ei] = cell_means[key]
        scenario_matrices[scen] = mat

    return {
        "cell_means": cell_means,
        "scenario_matrices": scenario_matrices,
        "policies": policies,
        "environments": envs,
    }


def crossed_bootstrap_ci(matrix: np.ndarray,
                         n_bootstrap: int = 50000,
                         seed: int = 116001,
                         confidence: float = 0.95) -> Dict:
    """Paired crossed two-axis bootstrap on a [n_policies, n_envs] matrix.

    Resamples BOTH policy and environment axes with replacement.
    Returns point estimate (grand mean) and percentile CI."""
    rng = np.random.RandomState(seed)
    n_pol, n_env = matrix.shape
    valid_mask = ~np.isnan(matrix)
    if not np.any(valid_mask):
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_bootstrap": n_bootstrap, "seed": seed}

    point = float(np.nanmean(matrix))
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pol_idx = rng.randint(0, n_pol, size=n_pol)
        env_idx = rng.randint(0, n_env, size=n_env)
        sub = matrix[np.ix_(pol_idx, env_idx)]
        boots[b] = np.nanmean(sub)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boots, 100 * alpha / 2))
    ci_upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": confidence,
    }


def scenario_gap_bootstrap(scenario_matrices: Dict[str, np.ndarray],
                           n_bootstrap: int = 50000,
                           seed: int = 116001,
                           confidence: float = 0.95) -> Dict:
    """Bootstrap CI for G_cos = mean_STAT - 0.5*(mean_PER + mean_MIX).

    Shares scenario draws (same resampled indices) across all three scenarios
    within each bootstrap iteration."""
    mat_per = scenario_matrices["PERSISTENT"]
    mat_stat = scenario_matrices["STATIONARY"]
    mat_mix = scenario_matrices["MIXED"]

    n_pol = mat_per.shape[0]
    n_env = mat_per.shape[1]

    rng = np.random.RandomState(seed)
    point_per = float(np.nanmean(mat_per))
    point_stat = float(np.nanmean(mat_stat))
    point_mix = float(np.nanmean(mat_mix))
    point_gap = point_stat - 0.5 * (point_per + point_mix)

    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pol_idx = rng.randint(0, n_pol, size=n_pol)
        env_idx = rng.randint(0, n_env, size=n_env)
        m_per = float(np.nanmean(mat_per[np.ix_(pol_idx, env_idx)]))
        m_stat = float(np.nanmean(mat_stat[np.ix_(pol_idx, env_idx)]))
        m_mix = float(np.nanmean(mat_mix[np.ix_(pol_idx, env_idx)]))
        boots[b] = m_stat - 0.5 * (m_per + m_mix)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boots, 100 * alpha / 2))
    ci_upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "G_point": point_gap,
        "G_ci_lower": ci_lower,
        "G_ci_upper": ci_upper,
        "per_scenario_means": {
            "PERSISTENT": point_per,
            "STATIONARY": point_stat,
            "MIXED": point_mix,
        },
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": confidence,
    }


def o1_support_verdict(g_cos: float, g_cos_ci_upper: float,
                       n_policies_g_cos_neg: int, n_policies: int,
                       g_conflict: float, n_policies_g_conflict_pos: int) -> str:
    """O1 truth table: determine support level for scenario-specific gradient alignment.

    SUPPORTED requires ALL:
      - G_cos < 0
      - CI upper < 0
      - ≥4/5 policies G_cos < 0
      - G_conflict > 0
      - ≥4/5 policies G_conflict > 0

    Returns one of: SUPPORTED, PLAUSIBLE, NOT_SUPPORTED, NOT_AUDITABLE."""
    min_consistent = int(np.ceil(0.8 * n_policies))

    all_met = (
        g_cos < 0
        and g_cos_ci_upper < 0
        and n_policies_g_cos_neg >= min_consistent
        and g_conflict > 0
        and n_policies_g_conflict_pos >= min_consistent
    )
    if all_met:
        return "SUPPORTED"

    some_direction = (g_cos < 0 or g_conflict > 0)
    if some_direction:
        return "PLAUSIBLE"

    return "NOT_SUPPORTED"
