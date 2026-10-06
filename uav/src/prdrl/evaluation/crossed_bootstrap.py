"""
Crossed two-axis bootstrap for PR-DRL-DMOEA final evaluation (frozen in E4E).

The learned Shared Double-DQN has TWO genuine stochastic axes:
    - policy training seed  (5 independently trained policies)
    - environment seed      (60 reserved final environments)

The 5 x 60 = 300 learned-policy cells are NOT 300 independent replicates: they
share 5 trained policies and 60 environments. A naive row bootstrap over the 300
cells would treat repeated environment evaluations of the same trained policy as
independent policy replicates, understating variance. This module resamples BOTH
axes with replacement and forms the full sampled Cartesian product.

Effect convention:  d[p,e] = B[e] - L[p,e]   (positive => DDQN better; MIGD lower)
    L[p,e] : learned-policy metric (n_policy x n_env)
    B[e]   : preselected comparator metric (n_env,) -- no policy axis.

CI method: percentile. All randomness derives deterministically from a master
seed plus text labels via numpy SeedSequence -- no seeds are chosen after data.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Frozen bootstrap settings (E4E).
N_BOOTSTRAP = 50_000
CI_LEVEL = 0.95
CI_METHOD = "percentile"
BOOTSTRAP_MASTER_SEED = 46_001


def derive_seed(*labels) -> int:
    """Deterministic substream seed from the master seed + text/int labels.

    Uses SeedSequence entropy = [MASTER, *hashed-labels]. Same labels always
    yield the same stream; different labels are independent. No result-dependent
    seeding is possible.
    """
    entropy: List[int] = [BOOTSTRAP_MASTER_SEED]
    for lab in labels:
        if isinstance(lab, (int, np.integer)):
            entropy.append(int(lab))
        else:
            # stable 32-bit digest of the string label
            h = 0
            for ch in str(lab):
                h = (h * 131 + ord(ch)) & 0xFFFFFFFF
            entropy.append(int(h))
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def scenario_point_effect(L: np.ndarray, B: np.ndarray) -> float:
    """Mean crossed effect over all cells: mean_{p,e} (B[e] - L[p,e])."""
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    assert L.ndim == 2, "L must be (n_policy, n_env)"
    assert B.ndim == 1 and B.shape[0] == L.shape[1], "B must be (n_env,)"
    d = B[None, :] - L                      # (n_policy, n_env)
    return float(d.mean())


def _percentile_ci(samples: np.ndarray, ci_level: float) -> Tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(samples, 100.0 * alpha))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha)))
    return lo, hi


def crossed_two_axis_bootstrap(
    L: np.ndarray,
    B: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int | None = None,
    labels: Sequence = (),
) -> Dict:
    """Crossed two-axis percentile bootstrap for one scenario.

    Each replicate:
      1. sample n_policy policy indices WITH replacement,
      2. sample n_env environment indices WITH replacement (independent),
      3. take the full sampled Cartesian product,
      4. statistic = mean over that product of (B[e] - L[p,e]).
    """
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    n_policy, n_env = L.shape
    assert B.shape[0] == n_env
    if seed is None:
        seed = derive_seed("crossed", *labels)
    rng = np.random.default_rng(seed)

    point = scenario_point_effect(L, B)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p_idx = rng.integers(0, n_policy, size=n_policy)
        e_idx = rng.integers(0, n_env, size=n_env)
        Lsub = L[np.ix_(p_idx, e_idx)]          # (n_policy, n_env) sampled product
        Bsub = B[e_idx]                          # (n_env,)
        boot[b] = float((Bsub[None, :] - Lsub).mean())
    lo, hi = _percentile_ci(boot, ci_level)
    return {
        "point_estimate": point,
        "ci_lo": lo, "ci_hi": hi, "ci_level": ci_level,
        "n_boot": n_boot, "method": "percentile_crossed_two_axis",
        "n_policy": n_policy, "n_env": n_env, "seed": int(seed),
    }


def crossed_two_axis_bootstrap_macro(
    L_by_scenario: Dict[str, np.ndarray],
    B_by_scenario: Dict[str, np.ndarray],
    scenarios: Sequence[str],
    *,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int | None = None,
    labels: Sequence = (),
) -> Dict:
    """Macro (equal 1/3 scenario weight) crossed bootstrap.

    For each replicate a SINGLE set of policy indices and a SINGLE set of
    environment indices are drawn and REUSED across all scenarios, preserving the
    dependence induced by evaluating the same policy IDs and environment IDs
    across scenarios. Scenario effects are then averaged with equal weight.
    """
    scenarios = list(scenarios)
    shapes = {s: np.asarray(L_by_scenario[s]).shape for s in scenarios}
    n_policy = shapes[scenarios[0]][0]
    n_env = shapes[scenarios[0]][1]
    for s in scenarios:
        assert shapes[s] == (n_policy, n_env), "all scenarios must share shape"
        assert np.asarray(B_by_scenario[s]).shape[0] == n_env

    if seed is None:
        seed = derive_seed("crossed_macro", *labels)
    rng = np.random.default_rng(seed)

    Ls = {s: np.asarray(L_by_scenario[s], dtype=float) for s in scenarios}
    Bs = {s: np.asarray(B_by_scenario[s], dtype=float) for s in scenarios}

    point = float(np.mean([scenario_point_effect(Ls[s], Bs[s]) for s in scenarios]))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p_idx = rng.integers(0, n_policy, size=n_policy)
        e_idx = rng.integers(0, n_env, size=n_env)
        vals = []
        for s in scenarios:
            Lsub = Ls[s][np.ix_(p_idx, e_idx)]
            Bsub = Bs[s][e_idx]
            vals.append(float((Bsub[None, :] - Lsub).mean()))
        boot[b] = float(np.mean(vals))
    lo, hi = _percentile_ci(boot, ci_level)
    return {
        "point_estimate": point, "ci_lo": lo, "ci_hi": hi, "ci_level": ci_level,
        "n_boot": n_boot, "method": "percentile_crossed_two_axis_macro",
        "scenarios": scenarios, "n_policy": n_policy, "n_env": n_env,
        "seed": int(seed), "scenario_weight": 1.0 / len(scenarios),
    }


def paired_crossed_two_axis_bootstrap(
    D: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int | None = None,
    labels: Sequence = (),
) -> Dict:
    """Crossed two-axis percentile bootstrap on a PAIRED difference matrix.

    D[p,e] is the matched paired effect (e.g. V1[p,e] - V3[p,e]); both arms share
    the same policy seed p and environment seed e. Each replicate resamples policy
    indices and environment indices (each with replacement, independent) and takes
    the mean over the full sampled Cartesian product. Preserves both stochastic
    axes; the pairing is retained because D is already the within-cell difference.
    """
    D = np.asarray(D, dtype=float)
    assert D.ndim == 2, "D must be (n_policy, n_env)"
    n_policy, n_env = D.shape
    if seed is None:
        seed = derive_seed("paired_crossed", *labels)
    rng = np.random.default_rng(seed)
    point = float(D.mean())
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p_idx = rng.integers(0, n_policy, size=n_policy)
        e_idx = rng.integers(0, n_env, size=n_env)
        boot[b] = float(D[np.ix_(p_idx, e_idx)].mean())
    lo, hi = _percentile_ci(boot, ci_level)
    return {"point_estimate": point, "ci_lo": lo, "ci_hi": hi, "ci_level": ci_level,
            "n_boot": n_boot, "method": "paired_crossed_two_axis",
            "n_policy": n_policy, "n_env": n_env, "seed": int(seed)}


def paired_crossed_two_axis_bootstrap_macro(
    D_by_scenario: Dict[str, np.ndarray],
    scenarios: Sequence[str],
    *,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int | None = None,
    labels: Sequence = (),
) -> Dict:
    """Macro (equal 1/3) paired crossed bootstrap over paired difference matrices.

    Each replicate draws ONE set of policy indices and ONE set of environment indices
    and REUSES them across all scenarios (preserving cross-scenario dependence), then
    averages the per-scenario paired means with equal weight."""
    scenarios = list(scenarios)
    D = {s: np.asarray(D_by_scenario[s], dtype=float) for s in scenarios}
    n_policy, n_env = D[scenarios[0]].shape
    for s in scenarios:
        assert D[s].shape == (n_policy, n_env)
    if seed is None:
        seed = derive_seed("paired_crossed_macro", *labels)
    rng = np.random.default_rng(seed)
    point = float(np.mean([D[s].mean() for s in scenarios]))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p_idx = rng.integers(0, n_policy, size=n_policy)
        e_idx = rng.integers(0, n_env, size=n_env)
        vals = [float(D[s][np.ix_(p_idx, e_idx)].mean()) for s in scenarios]
        boot[b] = float(np.mean(vals))
    lo, hi = _percentile_ci(boot, ci_level)
    return {"point_estimate": point, "ci_lo": lo, "ci_hi": hi, "ci_level": ci_level,
            "n_boot": n_boot, "method": "paired_crossed_two_axis_macro", "scenarios": scenarios,
            "n_policy": n_policy, "n_env": n_env, "seed": int(seed)}


def independent_arm_bootstrap(V1: np.ndarray, V3: np.ndarray, *, n_boot: int = N_BOOTSTRAP,
                             ci_level: float = CI_LEVEL, seed: int | None = None) -> Dict:
    """FORBIDDEN for inference; provided only for synthetic tests to show it is WIDER
    than the paired bootstrap on shared-variance fixtures (destroys pairing)."""
    V1 = np.asarray(V1, dtype=float); V3 = np.asarray(V3, dtype=float)
    n_policy, n_env = V1.shape
    if seed is None:
        seed = derive_seed("independent_arm")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p1 = rng.integers(0, n_policy, size=n_policy); e1 = rng.integers(0, n_env, size=n_env)
        p2 = rng.integers(0, n_policy, size=n_policy); e2 = rng.integers(0, n_env, size=n_env)
        boot[b] = float(V1[np.ix_(p1, e1)].mean() - V3[np.ix_(p2, e2)].mean())
    lo, hi = _percentile_ci(boot, ci_level)
    return {"point_estimate": float(V1.mean() - V3.mean()), "ci_lo": lo, "ci_hi": hi,
            "method": "independent_arm_FORBIDDEN"}


def naive_row_bootstrap(
    L: np.ndarray,
    B: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int | None = None,
) -> Dict:
    """FORBIDDEN for inference; provided ONLY to demonstrate (in tests) that the
    crossed bootstrap yields a wider CI on clustered fixtures. Resamples the
    300 (p,e) cells as if they were independent rows."""
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    n_policy, n_env = L.shape
    d = (B[None, :] - L).reshape(-1)         # 300 rows
    n = d.shape[0]
    if seed is None:
        seed = derive_seed("naive_row")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(d[idx].mean())
    lo, hi = _percentile_ci(boot, ci_level)
    return {"point_estimate": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "ci_level": ci_level, "n_boot": n_boot, "method": "naive_row_FORBIDDEN"}
