"""
E2-P3 -- Survival Translation Audit (corrected).

QUESTION: PREDICT creates better children. At which exact stage do those gains disappear?

Four-stage trace per mode (LOCAL / POP sigma=0 / GT sigma=0):
  Stage 1 -- Generated children     → child PS-distance (decision space)
  Stage 2 -- Evaluated children     → child PF-distance (objective space)
  Stage 3 -- Merged pool pre-survival → fraction rank-0, median rank
  Stage 4 -- Post-NSGA-II survival  → survival rate, surviving-child PF distance, IGD

Corrections vs initial version:
  - nsga2_survival_traced: provenance via pymoo pool_idx attribute (no tobytes matching)
  - Common survival RNG: LOCAL/POP/GT share same seed per (event, region)
  - region_type / dominant_bin embedded in every row
  - population_replacement_fraction = n_survived / POP_SIZE (not / BUDGET)
  - n_unique_children / unique_child_fraction diagnostic

Common-random-numbers: POP and GT share (parent_idx, xi) per offspring slot.
Sigma fixed: POP sigma=0, GT sigma=0, LOCAL sigma_local=0.02 (unchanged).

PREDICT_GT is a diagnostic oracle — NEVER goes to production.
Do NOT change: E2a/E2a-R, sigma_local, sigma_predict production, NSGA-II survival.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

import prdrl.experiments.E2P2_predict_noise_sweep as e2p2

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import nsga2_survival, split_by_region

from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    distance_to_region_ps, bin_of,
)

# ============================================================
# Locked config (same as E2-P2)
# ============================================================
SIGMA_LOCAL  = e2p2.SIGMA_LOCAL    # 0.02
SIGMA_POP    = 0.0                 # fixed for E2-P3
SIGMA_GT     = 0.0                 # fixed for E2-P3
BUDGET       = e2p2.BUDGET         # 10
N_OBJ        = e2p2.N_OBJ          # 2
D            = e2p2.D              # 10
POP_SIZE     = e2p2.POP_SIZE       # 100
N_PARTITIONS = e2p2.N_PARTITIONS   # 9 -> R=10
WARMUP_GENERATIONS     = e2p2.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p2.GENERATIONS_PER_CHANGE
LAMBDA_PRED  = e2p2.LAMBDA_PRED    # 1.0

compute_time = e2p2.compute_time
evolve_nsga2 = e2p2.evolve_nsga2

# Inherit stable RNG helpers from E2-P2
make_rng     = e2p2.make_rng
_surv_seed   = e2p2._surv_seed

make_scenario_dynamics   = e2p2.make_scenario_dynamics
generate_base_samples    = e2p2.generate_base_samples
apply_samples            = e2p2.apply_samples
make_local_children      = e2p2.make_local_children


# ============================================================
# Common survival seed — SAME for LOCAL/POP/GT, given event+region
# ============================================================

def _e2p3_common_surv_seed(base_seed, change_idx, region_id):
    """Mode-independent survival seed: LOCAL/POP/GT share this for same event/region.

    Differences between modes reflect only child quality, not survival RNG.
    PYTHONHASHSEED-stable.
    """
    ss = np.random.SeedSequence([int(base_seed), int(change_idx), int(region_id), 99997])
    return int(np.random.default_rng(ss).integers(2**31))


# ============================================================
# Objective-space child quality
# ============================================================

def median_nearest_pf_distance(F_children, pf_star):
    """Median distance from each child's objective vector to the nearest PF* point.
    Direction: child → PF (generational-distance style, NOT standard IGD).
    NOT called child_igd to avoid confusion with population IGD."""
    if F_children is None or len(F_children) == 0:
        return float("nan"), float("nan"), float("nan")
    F_c = np.asarray(F_children, dtype=float)
    P   = np.asarray(pf_star,    dtype=float)
    dists = np.sqrt(np.min(
        np.sum((F_c[:, None, :] - P[None, :, :]) ** 2, axis=2),
        axis=1
    ))
    return float(np.median(dists)), float(np.percentile(dists, 25)), float(np.percentile(dists, 75))


# ============================================================
# Rank assignment for merged pool (before survival)
# ============================================================

def assign_ranks(F_pool):
    """Return 0-indexed Pareto rank for each individual in F_pool."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    F_pool = np.asarray(F_pool, dtype=float)
    nds = NonDominatedSorting()
    fronts = nds.do(F_pool, return_rank=False)
    ranks = np.full(len(F_pool), -1, dtype=int)
    for r_idx, front in enumerate(fronts):
        for i in front:
            ranks[i] = r_idx
    return ranks


# ============================================================
# Traced NSGA-II survival — provenance via pymoo pool_idx attribute
#
# A1 guarantee: traced result == production result (T1).
# Duplicate-X safe: each pool member carries a unique integer index
# attached BEFORE survival; no X-vector equality used for matching.
# ============================================================

def nsga2_survival_traced(X_pool, F_pool, n_survive, xl, xu, surv_seed_val=None):
    """Identical survival semantics to nsga2_survival + survivor provenance.

    Provenance is carried via pymoo Population attribute 'pool_idx' attached
    before survival. Each pool member gets its original 0-based index stored
    in the Population object. After RankAndCrowding.do(), the surviving
    Population members still carry their original pool_idx — no X-vector
    equality or tobytes() comparison needed.

    Returns:
        X_surv, F_surv  -- same as nsga2_survival output (T1 guarantee)
        survived_orig_indices -- int array of shape (n_survive,):
                                 original index in X_pool for each survivor
    """
    X_pool = np.asarray(X_pool, dtype=float)
    F_pool = np.asarray(F_pool, dtype=float)
    N = len(X_pool)

    surv_rng = None
    if surv_seed_val is not None:
        seed_int = int(surv_seed_val) % (2 ** 31)
        np.random.seed(seed_int)
        # Pass a seeded Generator to pymoo's randomized_argsort so tie-breaking
        # in the truncation front is also deterministic (pymoo uses default_rng,
        # not the legacy np.random interface, so np.random.seed alone is not enough).
        surv_rng = np.random.default_rng(seed_int)

    problem = Problem(
        n_var=X_pool.shape[1],
        n_obj=F_pool.shape[1],
        xl=xl,
        xu=xu,
    )

    pop = Population.new(X=X_pool, F=F_pool)
    # Attach unique pool indices BEFORE survival so they travel with each individual
    pool_idx = np.arange(N, dtype=int)
    pop.set("pool_idx", pool_idx)

    survival = RankAndCrowding()
    survived = survival.do(problem, pop, n_survive=n_survive, random_state=surv_rng)

    X_surv = survived.get("X")
    F_surv = survived.get("F")
    survived_orig_indices = survived.get("pool_idx").astype(int)

    return X_surv, F_surv, survived_orig_indices


# ============================================================
# Per-region four-stage trace for one scored event
# ============================================================

def run_region_e2p3(
    r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
    prev_regions, region_dynamics, n_bins, pf_star,
    xl, xu, base_seed, change_idx,
):
    """Run LOCAL / POP / GT for one region at one event. Returns 3 row dicts."""
    mask  = region_ids == r
    n_old = int(np.sum(mask))
    if n_old == 0:
        return []

    X_region = X_pop[mask]

    # Region type / dominant bin
    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type = str(labels_r[np.argmax(counts_r)])

    bins_in_region = bin_of(X_region[:, 0], n_bins)
    bin_vals, bin_counts = np.unique(bins_in_region, return_counts=True)
    dominant_bin = int(bin_vals[np.argmax(bin_counts)])

    # True PS centers
    n_dist = D - 1
    mu_star_prev2   = region_center(region_dynamics, dominant_bin, t_prev2,  n_dist)
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_dist)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_dist)

    # GT uses delta_response_true (t-1 → t) per A1 semantics
    delta_response_true      = mu_star_current - mu_star_prev
    delta_response_true_full = np.concatenate([[0.0], delta_response_true])

    # POP estimator — causal: reads t-2 and t-1 only
    X_prev2_region = prev_regions.get(r)
    has_history = X_prev2_region is not None and len(X_prev2_region) > 0

    if has_history:
        Z_cur   = (X_region       - xl) / (xu - xl)
        Z_prev2 = (X_prev2_region - xl) / (xu - xl)
        Delta_POP_full = (np.mean(Z_cur, axis=0) - np.mean(Z_prev2, axis=0)) * LAMBDA_PRED
    else:
        Delta_POP_full = None

    # Pre-generate shared (parent_idx, xi) for POP and GT
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    base_idxs, base_xis = generate_base_samples(BUDGET, D, rng_shared, n_old)

    # Generate children for each mode (X_region NOT mutated)
    ch_local = make_local_children(X_region, xl, xu, base_seed, change_idx, r)
    ch_pop   = (apply_samples(base_idxs, base_xis, X_region, Delta_POP_full,
                              SIGMA_POP, xl, xu) if has_history else None)
    ch_gt    = (apply_samples(base_idxs, base_xis, X_region, delta_response_true_full,
                              SIGMA_GT,  xl, xu) if has_history else None)

    # Evaluate objectives
    def eval_fn(X_ch):
        return cp_evaluate(X_ch, t_new, region_dynamics)

    F_ch_local = eval_fn(ch_local)
    F_ch_pop   = eval_fn(ch_pop) if ch_pop is not None else None
    F_ch_gt    = eval_fn(ch_gt)  if ch_gt  is not None else None

    # IGD before (same for all modes — baseline population quality at t_new)
    igd_before = float(igd(F_plus, pf_star))

    # Common survival seed — SAME for LOCAL/POP/GT for this (event, region)
    common_sv_seed = _e2p3_common_surv_seed(base_seed, change_idx, r)

    rows = []
    modes = [("LOCAL", ch_local, F_ch_local),
             ("POP",   ch_pop,   F_ch_pop),
             ("GT",    ch_gt,    F_ch_gt)]

    for mode_name, ch, F_ch in modes:
        if ch is None or F_ch is None:
            rows.append({
                "mode": mode_name,
                "region_type": region_type,
                "dominant_bin": dominant_bin,
                "has_history": False,
                "n_children_generated": None,
                "n_unique_children": None,
                "unique_child_fraction": float("nan"),
                "child_ps_dist_median": float("nan"),
                "child_ps_dist_q25": float("nan"),
                "child_ps_dist_q75": float("nan"),
                "child_pf_dist_median": float("nan"),
                "child_pf_dist_q25": float("nan"),
                "child_pf_dist_q75": float("nan"),
                "frac_children_rank0": float("nan"),
                "median_child_rank": float("nan"),
                "n_children_survived": None,
                "survival_rate": float("nan"),
                "population_replacement_fraction": float("nan"),
                "surviving_child_pf_dist_median": float("nan"),
                "median_surviving_child_rank": float("nan"),
                "igd_before": igd_before,
                "igd_after": float("nan"),
                "delta_igd": float("nan"),
                "relative_igd_improvement": float("nan"),
                "fe": None,
            })
            continue

        n_ch = len(ch)

        # A7: unique children diagnostic
        ch_unique = np.unique(ch, axis=0)
        n_unique_ch = len(ch_unique)
        unique_ch_frac = n_unique_ch / n_ch if n_ch > 0 else float("nan")

        # Stage 1: PS distance
        ps_dists = distance_to_region_ps(ch, region_dynamics, dominant_bin, t_new)
        ps_med   = float(np.median(ps_dists))
        ps_q25   = float(np.percentile(ps_dists, 25))
        ps_q75   = float(np.percentile(ps_dists, 75))

        # Stage 2: PF distance
        pf_med, pf_q25, pf_q75 = median_nearest_pf_distance(F_ch, pf_star)

        # Stage 3: ranks in merged pool (old pop + children)
        N_total_old = len(X_pop)
        X_pool_full = np.vstack([X_pop, ch])
        F_pool_full = np.vstack([F_plus, F_ch])

        ranks_full = assign_ranks(F_pool_full)
        child_ranks = ranks_full[N_total_old:]

        frac_rank0        = float(np.mean(child_ranks == 0))
        median_child_rank = float(np.median(child_ranks))

        # Stage 4: survival with provenance via pymoo pool_idx attribute
        X_surv, F_surv, surv_orig_idx = nsga2_survival_traced(
            X_pool_full, F_pool_full, POP_SIZE, xl, xu, common_sv_seed
        )

        # Children in pool have indices >= N_total_old
        survived_child_mask = surv_orig_idx >= N_total_old
        n_children_survived = int(np.sum(survived_child_mask))
        survival_rate       = n_children_survived / n_ch if n_ch > 0 else float("nan")
        # A6: correct replacement denominator
        pop_replace_frac    = n_children_survived / POP_SIZE

        surviving_child_indices_in_pool = surv_orig_idx[survived_child_mask]
        if n_children_survived > 0:
            F_surviving_children = F_pool_full[surviving_child_indices_in_pool]
            surv_pf_med, _, _ = median_nearest_pf_distance(F_surviving_children, pf_star)
            surv_ranks = ranks_full[surviving_child_indices_in_pool]
            median_surv_rank = float(np.median(surv_ranks))
        else:
            surv_pf_med      = float("nan")
            median_surv_rank = float("nan")

        # Population IGD after survival
        igd_after = float(igd(F_surv, pf_star))
        delta_igd = igd_before - igd_after
        rel_igd   = delta_igd / (abs(igd_before) + 1e-12)

        rows.append({
            "mode": mode_name,
            "region_type": region_type,
            "dominant_bin": dominant_bin,
            "has_history": has_history,
            "n_children_generated": n_ch,
            "n_unique_children": n_unique_ch,
            "unique_child_fraction": unique_ch_frac,
            "child_ps_dist_median": ps_med,
            "child_ps_dist_q25": ps_q25,
            "child_ps_dist_q75": ps_q75,
            "child_pf_dist_median": pf_med,
            "child_pf_dist_q25": pf_q25,
            "child_pf_dist_q75": pf_q75,
            "frac_children_rank0": frac_rank0,
            "median_child_rank": median_child_rank,
            "n_children_survived": n_children_survived,
            "survival_rate": survival_rate,
            "population_replacement_fraction": pop_replace_frac,
            "surviving_child_pf_dist_median": surv_pf_med,
            "median_surviving_child_rank": median_surv_rank,
            "igd_before": igd_before,
            "igd_after": igd_after,
            "delta_igd": delta_igd,
            "relative_igd_improvement": rel_igd,
            "fe": n_ch,
        })

    return rows


# ============================================================
# One (seed, scenario) run
# ============================================================

def run_one_e2p3(seed, n_scored_changes, scenario_name):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R  = len(V)
    n_bins = R
    region_dynamics = make_scenario_dynamics(scenario_name, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_current = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                         WARMUP_GENERATIONS, seed * 10000, xl, xu)

    prev_regions = None
    t_prev2 = None

    rows_out = []
    gen = 0
    n_total = 1 + n_scored_changes

    for change_idx in range(1, n_total + 1):
        F_minus = cp_evaluate(X_pop, t_current, region_dynamics)
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)
        F_plus = cp_evaluate(X_pop, t_new, region_dynamics)

        F_tm, F_tp = normalize(F_minus, F_plus)
        F_tm = np.asarray(F_tm)
        region_ids  = np.asarray(partition(F_tm, V))
        cur_regions = split_by_region(X_pop, region_ids, R)

        if change_idx == 1:
            prev_regions = cur_regions
            t_prev2 = t_current
            t_current = t_new
            X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                                 GENERATIONS_PER_CHANGE,
                                 seed * 10000 + change_idx, xl, xu)
            continue

        t_prev = t_current

        for r in range(R):
            r_rows = run_region_e2p3(
                r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
                prev_regions, region_dynamics, n_bins, pf_star,
                xl, xu, seed, change_idx,
            )
            for row in r_rows:
                row.update({
                    "scenario":   scenario_name,
                    "seed":       seed,
                    "event_id":   change_idx,
                    "change_idx": change_idx,
                    "region_id":  r,
                    "t_prev2":    float(t_prev2),
                    "t_prev":     float(t_prev),
                    "t_new":      float(t_new),
                })
                rows_out.append(row)

        prev_regions = cur_regions
        t_prev2 = t_prev
        t_current = t_new
        X_pop = evolve_nsga2(X_pop, t_current, region_dynamics,
                             GENERATIONS_PER_CHANGE,
                             seed * 10000 + change_idx, xl, xu)

    return rows_out


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == "__main__":
    N_SEEDS   = 20
    N_SCORED  = 10
    SCENARIOS = ["PERSISTENT", "STATIONARY"]
    OUT_DIR   = Path("results/E2P3_survival_translation_corrected")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "E2P3_survival_translation.jsonl"

    all_rows = []
    t0 = time.time()

    for sc in SCENARIOS:
        for seed in range(1, N_SEEDS + 1):
            t1 = time.time()
            rows = run_one_e2p3(seed, N_SCORED, sc)
            all_rows.extend(rows)
            elapsed = time.time() - t1
            print(f"{sc} seed={seed}: {len(rows)} rows in {elapsed:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | Wrote {len(all_rows)} rows to {jsonl_path}")
