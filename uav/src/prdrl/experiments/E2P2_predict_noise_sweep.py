"""
E2-P2 -- PREDICT Noise Sweep.

Mục tiêu: trả lời riêng 3 câu hỏi về PREDICT bottleneck:

  Q1. Nếu giữ estimator hiện tại, giảm sigma_P -> PREDICT tốt hơn LOCAL không?
  Q2. Nếu dùng GT motion, sigma_P bao nhiêu còn giữ lợi ích?
  Q3. Nếu GT+sigma=0 vẫn không cải thiện IGD, bottleneck là survival interaction.

Protocol:
  - sigma sweep: [0.0, 0.005, 0.01, 0.02, 0.03]
  - modes: LOCAL (fixed sigma=0.02), POP_PREDICT, GT_PREDICT
  - scenarios: PERSISTENT, STATIONARY
  - common random numbers: POP và GT dùng cùng (parent_idx, xi) per offspring
  - sigma scaling: sigma * xi (same xi, different scale) -> T8 holds

FREEZE: E2-P cũ, E2a, E2a-R, lambda_pred=1.0, sigma_local=0.02,
        sigma_diversify, NSGA-II, FE accounting, PREDICT production semantics.

PREDICT_GT là oracle chẩn đoán -- KHÔNG vào production.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

import prdrl.experiments.E2P_predict_mechanism as e2p

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import nsga2_survival, split_by_region
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate, get_bounds, get_true_pf,
    region_center, ground_truth_dynamic_label,
    distance_to_region_ps, bin_of,
)


# ============================================================
# CONFIG (locked -- Section 2 E2-P2)
# ============================================================
SIGMA_LOCAL    = e2p.SIGMA_LOCAL    # 0.02
LAMBDA_PRED    = e2p.LAMBDA_PRED    # 1.0
BUDGET         = e2p.BUDGET         # 10
N_OBJ          = e2p.N_OBJ          # 2
D              = e2p.D              # 10
POP_SIZE       = e2p.POP_SIZE       # 100
N_PARTITIONS   = e2p.N_PARTITIONS   # 9 -> R=10
WARMUP_GENERATIONS   = e2p.WARMUP_GENERATIONS   # 30
GENERATIONS_PER_CHANGE = e2p.GENERATIONS_PER_CHANGE  # 10

SIGMA_P_VALUES = [0.0, 0.005, 0.01, 0.02, 0.03]

compute_time   = e2p.compute_time
evolve_nsga2   = e2p.evolve_nsga2

# ============================================================
# Stable RNG helpers -- no Python hash(). PYTHONHASHSEED-safe.
# e2p.make_rng uses hash(key) on strings which is NOT stable
# across separate Python processes. Define local versions here.
# ============================================================

_RNG_KEY_INTS = {"LOCAL": 1, "PREDICT_SHARED": 2, "GT": 3}

def make_rng(base_seed, event_id, region_id, key):
    """Cross-process deterministic RNG — encodes key as integer, PYTHONHASHSEED-stable."""
    key_int = _RNG_KEY_INTS.get(key, sum(b for b in key.encode()[:16]) % 10000)
    ss = np.random.SeedSequence([int(base_seed), int(event_id), int(region_id), key_int])
    return np.random.default_rng(ss)


_SURV_KEY_INTS = {"LOCAL": 1, "POP": 2, "GT": 3}

def _surv_seed(base_seed, change_idx, region_id, variant_key):
    """Deterministic survival np.random.seed — PYTHONHASHSEED-stable, no builtins.hash."""
    prefix = variant_key.split("_")[0] if "_" in variant_key else variant_key
    key_int = _SURV_KEY_INTS.get(prefix, sum(b for b in variant_key.encode()[:8]) % 1000)
    ss = np.random.SeedSequence([int(base_seed), int(change_idx), int(region_id), key_int])
    return int(np.random.default_rng(ss).integers(2**31))


# ============================================================
# Scenario dynamics factory
# ============================================================

def make_scenario_dynamics(scenario_name, n_bins):
    """Return region_dynamics list for scenario_name in {PERSISTENT, STATIONARY}."""
    if scenario_name == "PERSISTENT":
        return ["persistent"] * n_bins
    elif scenario_name == "STATIONARY":
        return ["stationary"] * n_bins
    raise ValueError(f"Unknown scenario: {scenario_name!r}. Use 'PERSISTENT' or 'STATIONARY'.")


# ============================================================
# Common-random-number offspring generation
#
# Pre-generate (parent_idx, xi) for all offspring slots from a
# shared RNG, then apply different delta / sigma without touching
# the base samples. This ensures:
#   * POP and GT use the same xi at the same sigma  -> isolates delta only
#   * sigma=0.01 offspring = sigma=0.02 offspring but with 0.5x noise  -> T8
#   * LOCAL uses its own RNG, independent of sigma sweep
# ============================================================

def generate_base_samples(n_offspring, D_, rng, n_parents):
    """Draw parent indices and unit-noise xi for all offspring slots."""
    parent_idxs = [int(rng.integers(n_parents)) for _ in range(n_offspring)]
    xis         = [rng.standard_normal(D_) for _ in range(n_offspring)]
    return parent_idxs, xis


def apply_samples(parent_idxs, xis, X_region, delta, sigma, xl, xu):
    """Translate offspring using pre-generated samples.

    z_child = clip(z_parent + delta + sigma * xi, 0, 1)
    """
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    offspring = []
    for idx, xi in zip(parent_idxs, xis):
        parent = X_region[idx]
        z_parent = (parent - xl) / (xu - xl)
        z_child  = np.clip(z_parent + delta + sigma * xi, 0.0, 1.0)
        offspring.append(xl + z_child * (xu - xl))
    return np.array(offspring)


def make_local_children(X_region, xl, xu, base_seed, event_id, region_id):
    """LOCAL offspring -- independent of sigma sweep."""
    rng = make_rng(base_seed, event_id, region_id, "LOCAL")
    from prdrl.operators.local import local
    return local(X_region, BUDGET, xl, xu, SIGMA_LOCAL, rng)


# ============================================================
# One scored event for a given region -- returns rows for all sigma.
# ============================================================

def run_region_sweep(
    r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
    prev_regions, region_dynamics, n_bins, pf_star,
    xl, xu, base_seed, change_idx, sigma_values
):
    """Run all sigma values for one region at one event. Returns list of row dicts."""
    mask    = region_ids == r
    n_prev  = int(np.sum(mask))
    if n_prev == 0:
        return []

    X_region = X_pop[mask]

    gt_labels = ground_truth_dynamic_label(X_pop, region_dynamics)
    labels_r, counts_r = np.unique(gt_labels[mask], return_counts=True)
    region_type = str(labels_r[np.argmax(counts_r)])

    from prdrl.problems.controlled_predict import bin_of
    bins_in_region = bin_of(X_region[:, 0], n_bins)
    bin_vals, bin_counts = np.unique(bins_in_region, return_counts=True)
    dominant_bin = int(bin_vals[np.argmax(bin_counts)])

    n_distance = D - 1
    mu_star_prev2   = region_center(region_dynamics, dominant_bin, t_prev2,  n_distance)
    mu_star_prev    = region_center(region_dynamics, dominant_bin, t_prev,   n_distance)
    mu_star_current = region_center(region_dynamics, dominant_bin, t_new,    n_distance)

    # Naming convention (A1 audit):
    #   delta_hist_true     = historical shift  (t-2 → t-1, observed)
    #   delta_response_true = response shift    (t-1 → t, what GT oracle should predict)
    # For PERSISTENT (linear dynamics, equal time steps): they are equal.
    # For STATIONARY: both are zero.
    # GT_PREDICT consumes delta_response_true (the correct semantic).
    delta_hist_true     = mu_star_prev - mu_star_prev2
    delta_response_true = mu_star_current - mu_star_prev
    delta_response_true_full = np.concatenate([[0.0], delta_response_true])

    X_prev2_region = prev_regions.get(r)
    n_prev2 = len(X_prev2_region) if X_prev2_region is not None else 0

    if n_prev2 > 0:
        Z_cur    = (X_region - xl) / (xu - xl)
        Z_prev2  = (X_prev2_region - xl) / (xu - xl)
        mu_pop_full_prev  = np.mean(Z_cur,   axis=0)
        mu_pop_full_prev2 = np.mean(Z_prev2, axis=0)
        Delta_POP_full = (mu_pop_full_prev - mu_pop_full_prev2) * LAMBDA_PRED
        Delta_POP      = Delta_POP_full[1:]
        has_history    = True
    else:
        Delta_POP_full = None
        Delta_POP      = None
        has_history    = False

    eps = 1e-12
    if has_history and np.linalg.norm(delta_response_true) > 1e-9:
        cos_val = float(np.dot(Delta_POP, delta_response_true)
                        / (np.linalg.norm(Delta_POP) * np.linalg.norm(delta_response_true) + eps)) \
                  if np.linalg.norm(Delta_POP) > 1e-9 else float("nan")
        rel_err = float(np.linalg.norm(Delta_POP - delta_response_true)
                        / (np.linalg.norm(delta_response_true) + eps))
        norm_rat = float(np.linalg.norm(Delta_POP)
                         / (np.linalg.norm(delta_response_true) + eps))
    else:
        cos_val  = float("nan")
        rel_err  = float("nan")
        norm_rat = float("nan")

    norm_delta_pop = float(np.linalg.norm(Delta_POP)) if has_history else float("nan")
    norm_delta_gt  = float(np.linalg.norm(delta_response_true))

    def evaluate_fn(X_new):
        return cp_evaluate(X_new, t_new, region_dynamics)

    def quality_for(ch, variant_key):
        if ch is None or len(ch) == 0:
            F_new = F_plus
        else:
            F_children = evaluate_fn(ch)
            X_pool = np.vstack([X_pop, ch])
            F_pool = np.vstack([F_plus, F_children])
            surv_seed = _surv_seed(base_seed, change_idx, r, variant_key)
            np.random.seed(surv_seed)
            _, F_new = nsga2_survival(X_pool, F_pool, n_survive=POP_SIZE, xl=xl, xu=xu)
        return float(igd(F_new, pf_star))

    # LOCAL children (independent of sigma sweep)
    ch_local   = make_local_children(X_region, xl, xu, base_seed, change_idx, r)
    dist_local = float(np.median(distance_to_region_ps(ch_local, region_dynamics, dominant_bin, t_new)))
    igd_local  = quality_for(ch_local, "LOCAL")

    # Pre-generate common base samples for POP/GT (same across all sigma)
    rng_shared = make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    base_idxs, base_xis = generate_base_samples(BUDGET, D, rng_shared, n_prev)

    rows = []
    for sigma in sigma_values:
        if has_history:
            ch_pop = apply_samples(base_idxs, base_xis, X_region,
                                   Delta_POP_full, sigma, xl, xu)
            ch_gt  = apply_samples(base_idxs, base_xis, X_region,
                                   delta_response_true_full, sigma, xl, xu)

            dist_pop = float(np.median(distance_to_region_ps(ch_pop, region_dynamics, dominant_bin, t_new)))
            dist_gt  = float(np.median(distance_to_region_ps(ch_gt,  region_dynamics, dominant_bin, t_new)))

            igd_pop  = quality_for(ch_pop, f"POP_{sigma}")
            igd_gt   = quality_for(ch_gt,  f"GT_{sigma}")

            fe_pop = fe_gt = BUDGET
        else:
            ch_pop = None
            ch_gt  = None
            dist_pop = float("nan")
            dist_gt  = float("nan")
            igd_pop  = float("nan")
            igd_gt   = float("nan")
            fe_pop   = None
            fe_gt    = None

        rows.append({
            "region_id":        int(r),
            "region_type":      region_type,
            "dominant_bin":     dominant_bin,
            "sigma_pred":       float(sigma),
            "has_history":      has_history,
            "n_prev":           n_prev,
            "n_prev2":          n_prev2,
            # child distances
            "child_distance_local": dist_local,
            "child_distance_pop":   dist_pop,
            "child_distance_gt":    dist_gt,
            # IGD
            "igd_local": igd_local,
            "igd_pop":   igd_pop,
            "igd_gt":    igd_gt,
            # estimator diagnostics
            "estimator_cosine":         cos_val,
            "estimator_relative_error": rel_err,
            "estimator_norm_ratio":     norm_rat,
            "norm_delta_pop":           norm_delta_pop,
            "norm_delta_gt":            norm_delta_gt,
            # FE
            "n_children":  BUDGET,
            "fe_local":    BUDGET,
            "fe_pop":      fe_pop,
            "fe_gt":       fe_gt,
        })

    return rows


# ============================================================
# One (seed, scenario) run -- all sigma values
# ============================================================

def run_one_sweep(seed, n_scored_changes, sigma_values, scenario_name):
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
            r_rows = run_region_sweep(
                r, X_pop, F_plus, region_ids, t_new, t_prev, t_prev2,
                prev_regions, region_dynamics, n_bins, pf_star,
                xl, xu, seed, change_idx, sigma_values,
            )
            for row in r_rows:
                row.update({
                    "scenario":   scenario_name,
                    "seed":       seed,
                    "event_id":   change_idx,
                    "change_idx": change_idx,
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
    OUT_DIR   = Path("results/E2P2_predict_noise_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "E2P2_predict_noise_sweep.jsonl"

    all_rows = []
    t0 = time.time()

    for sc in SCENARIOS:
        for seed in range(1, N_SEEDS + 1):
            t1 = time.time()
            rows = run_one_sweep(seed, N_SCORED, SIGMA_P_VALUES, sc)
            all_rows.extend(rows)
            elapsed = time.time() - t1
            print(f"{sc} seed={seed}: {len(rows)} rows in {elapsed:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s | Wrote {len(all_rows)} rows to {jsonl_path}")
