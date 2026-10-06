"""
Core survival utilities for PR-DRL.

Promotes region_protected_survival and compute_ranks_crowding_all
from R1/E2P5 experimental path to a shared core module.

D2-A: This module is the canonical, single implementation of HARD
region-protected response survival. Both the production oracle path
(oracle.py) and the R1/L2B diagnostic helpers import from here.

INVARIANT (tested by T34 in test_d2_production_path_equivalence.py):
    For every successful PREDICT response targeting region r:
        n_old_dropped_other == 0
"""

import numpy as np

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _calc_crowding_dist(F_front):
    """Crowding distance for one Pareto front (same as pymoo NSGA-II)."""
    n, m = F_front.shape
    if n <= 2:
        return np.full(n, np.inf)

    crowd = np.zeros(n)
    for obj in range(m):
        order = np.argsort(F_front[:, obj])
        f_min = F_front[order[0], obj]
        f_max = F_front[order[-1], obj]
        span = f_max - f_min

        crowd[order[0]] = np.inf
        crowd[order[-1]] = np.inf

        if span < 1e-15:
            continue

        for k in range(1, n - 1):
            crowd[order[k]] += (
                F_front[order[k + 1], obj] - F_front[order[k - 1], obj]
            ) / span

    return crowd


def compute_ranks_crowding_all(F_pool):
    """Compute Pareto rank (0=best) and crowding distance for all pool members.

    Uses same NonDominatedSorting + standard NSGA-II crowding as pymoo's NSGA-II.
    Identical semantics to E2P5.compute_ranks_crowding_all — this is the canonical copy.

    Returns:
        ranks (N,)    — Pareto rank, 0 = non-dominated front
        crowding (N,) — crowding distance, inf at front boundaries
    """
    F = np.asarray(F_pool, dtype=float)
    N = len(F)
    nds = NonDominatedSorting()
    fronts = nds.do(F)

    ranks    = np.full(N, 0, dtype=int)
    crowding = np.zeros(N, dtype=float)

    for r_idx, front in enumerate(fronts):
        fa = np.array(front, dtype=int)
        ranks[fa] = r_idx
        crowding[fa] = _calc_crowding_dist(F[fa])

    return ranks, crowding


def region_protected_survival(
    X_pop, F_pop,
    X_children, F_children,
    response_region_r,
    old_region_ids,
    n_r_before,
    xl, xu,
    surv_seed=None,
):
    """HARD region-protected response survival.

    Identical semantics to R1_region_protected_response.region_protected_survival.
    This is the canonical single implementation — R1 and oracle.py both import here.

    Mechanism:
        - ALL OLD individuals outside region r are protected (guaranteed survivors).
        - Global NSGA-II rank+crowding computed on full merged pool (OLD + children).
        - Exactly n_r_before candidates selected from OLD_r ∪ CHILD_r using global
          rank then crowding as tie-break.
        - Final population = OLD_outside ∪ selected_r_candidates.

    HARD invariant:
        n_old_dropped_other == 0 for every call.

    Args:
        X_pop:             OLD population (N, D)
        F_pop:             OLD objectives at new time (N, M)
        X_children:        children for region r (B, D)
        F_children:        evaluated children (B, M)
        response_region_r: region index being responded to
        old_region_ids:    pre-response region assignments for OLD pop (N,)
        n_r_before:        slots reserved for region r (|OLD_r| before response)
        xl, xu:            bounds (not used in selection, kept for API consistency)
        surv_seed:         optional int for deterministic tie-breaking

    Returns:
        X_surv (POP_SIZE, D), F_surv (POP_SIZE, M),
        final_pool_indices (POP_SIZE,) — indices into merged pool [OLD; children]
    """
    N = len(X_pop)
    B = len(X_children)
    r = response_region_r
    old_region_ids = np.asarray(old_region_ids)

    # Build merged pool: OLD first, then children
    X_pool = np.vstack([X_pop, X_children])
    F_pool = np.vstack([F_pop, F_children])

    # Global ranks + crowding on full merged pool (same computation as unrestricted)
    ranks, crowding = compute_ranks_crowding_all(F_pool)

    # Partition OLD into protected (outside r) and candidate (inside r)
    old_outside = [i for i in range(N) if old_region_ids[i] != r]
    old_in_r    = [i for i in range(N) if old_region_ids[i] == r]
    children_idx = list(range(N, N + B))

    # Candidate pool: OLD_r ∪ CHILD_r
    candidates = old_in_r + children_idx

    # Sort candidates by (rank ASC, crowding DESC)
    def sort_key(i):
        cd = crowding[i]
        cd_sort = -cd if np.isfinite(cd) else -(1e15)
        return (int(ranks[i]), cd_sort)

    candidates_sorted = sorted(candidates, key=sort_key)
    selected_r = candidates_sorted[:n_r_before]

    # Final population: protected OLD_outside + best n_r_before from r-candidate pool
    final_idx = sorted(old_outside + selected_r)

    POP_SIZE = N
    assert len(final_idx) == POP_SIZE, (
        f"Region-protected survival produced {len(final_idx)} individuals, "
        f"expected {POP_SIZE} (|OLD_outside|={len(old_outside)}, n_r_before={n_r_before})"
    )

    final_idx_arr = np.array(final_idx, dtype=int)
    return X_pool[final_idx_arr], F_pool[final_idx_arr], final_idx_arr
