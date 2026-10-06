"""
E2a-R null A/B experiment: fixed_index vs fixed_rule.

Runs the same real events under two null construction modes and records
paired per-event metrics. Does NOT modify run_one() or run_event(); adds
run_event_ab() and run_one_ab() as standalone additions.

Output: results/null_ab_fixed_index_vs_fixed_rule/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

import prdrl.experiments.E2aR_controlled_operator_oracle as e2ar
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.operators.budget import equal_budget
from prdrl.core.seeding import deterministic_rng
from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.problems.controlled_operator import (
    evaluate as op_evaluate, get_bounds, get_true_pf,
    make_region_dynamics_map, SCENARIOS,
)

K = 50
REAL_SENTINEL = e2ar.REAL_SENTINEL

SCENARIOS_ALL = list(SCENARIOS)


def run_event_ab(X_base, F_plus, region_ids, R, history, t,
                 evaluate_fn, pf_star, xl, xu, budget,
                 base_seed, event_id, K):
    """Run one scored event under both null modes.

    Returns dict with real metrics (computed once) + null metrics
    for fixed_index and fixed_rule separately.
    """
    def quality_fn(F):
        return -igd(F, pf_star)

    real = e2ar.run_regional_protocol(
        X_base, F_plus, region_ids, R, history, t,
        evaluate_fn, quality_fn, xl, xu, budget,
        base_seed, event_id, REAL_SENTINEL)

    global_std_results, best_global_std_action, best_global_std_Q = \
        e2ar.run_best_global(
            X_base, F_plus, region_ids, R,
            real["active_regions"], history, t,
            evaluate_fn, quality_fn, xl, xu, budget,
            base_seed, event_id, real["Q_keep"])

    fe_results, best_fe_action, best_fe_Q, fe_budgets = \
        e2ar.run_best_global_fe_matched(
            X_base, F_plus, region_ids, R,
            real["active_regions"], history, t,
            evaluate_fn, quality_fn, xl, xu,
            real["regional_extra_fe"], base_seed, event_id)

    eps = 1e-12
    ROG_standard = ((real["Q_regional"] - best_global_std_Q)
                    / (abs(best_global_std_Q) + eps))

    if best_fe_action is not None:
        ROG_FE = (real["Q_regional"] - best_fe_Q) / (abs(best_fe_Q) + eps)
    else:
        ROG_FE = float("nan")

    # ---- Fixed-index nulls (same as run_event) ----
    null_rng_fi = deterministic_rng(base_seed, hash(str(event_id)) % 100000, event_id)
    rog_fi, ohi_fi, q_fi = [], [], []
    for k in range(K):
        rand_ids = e2ar.size_preserving_random_partition(region_ids, R, null_rng_fi)
        h_fi = e2ar._build_null_history(
            history, R, base_seed, event_id, k, t, xl, xu)
        result_k = e2ar.run_regional_protocol(
            X_base, F_plus, rand_ids, R, h_fi, t,
            evaluate_fn, quality_fn, xl, xu, budget,
            base_seed, event_id, k)
        rog_fi.append((result_k["Q_regional"] - best_global_std_Q)
                      / (abs(best_global_std_Q) + eps))
        ohi_fi.append(result_k["OHI"])
        q_fi.append(result_k["Q_regional"])

    rog_fi = np.array(rog_fi)
    ohi_fi = np.array(ohi_fi)

    # ---- Fixed-rule nulls ----
    rog_fr, ohi_fr, q_fr = [], [], []
    for k in range(K):
        w_k = e2ar._get_projection_vector(base_seed, event_id, k, len(xl))
        rand_ids = e2ar._fixed_rule_partition(X_base, region_ids, R, w_k, xl, xu)
        h_fr = e2ar._build_fixed_rule_null_history(
            history, R, base_seed, event_id, k, t, xl, xu)
        result_k = e2ar.run_regional_protocol(
            X_base, F_plus, rand_ids, R, h_fr, t,
            evaluate_fn, quality_fn, xl, xu, budget,
            base_seed, event_id, k)
        rog_fr.append((result_k["Q_regional"] - best_global_std_Q)
                      / (abs(best_global_std_Q) + eps))
        ohi_fr.append(result_k["OHI"])
        q_fr.append(result_k["Q_regional"])

    rog_fr = np.array(rog_fr)
    ohi_fr = np.array(ohi_fr)

    def _null_stats(rog_arr, ohi_arr, q_arr):
        med_rog = float(np.median(rog_arr))
        pct = float(np.mean(rog_arr <= ROG_standard))
        excess = ROG_standard - med_rog
        return {
            "rand_ROG_median": med_rog,
            "rand_ROG_mean": float(np.mean(rog_arr)),
            "rand_ROG_std": float(np.std(rog_arr)),
            "real_ROG_percentile": pct,
            "excess_ROG": float(excess),
            "rand_OHI_median": float(np.median(ohi_arr)),
            "real_OHI_percentile": float(np.mean(ohi_arr <= real["OHI"])),
            "excess_OHI": float(real["OHI"] - float(np.median(ohi_arr))),
        }

    fi_stats = _null_stats(rog_fi, ohi_fi, q_fi)
    fr_stats = _null_stats(rog_fr, ohi_fr, q_fr)

    delta_null_ROG = fr_stats["rand_ROG_median"] - fi_stats["rand_ROG_median"]
    delta_excess   = fr_stats["excess_ROG"] - fi_stats["excess_ROG"]

    return {
        "ROG_standard": float(ROG_standard),
        "ROG_FE": float(ROG_FE),
        "real_OHI": float(real["OHI"]),
        "fixed_index": fi_stats,
        "fixed_rule":  fr_stats,
        "delta_null_ROG": float(delta_null_ROG),
        "delta_excess":   float(delta_excess),
    }


def run_one_ab(scenario, seed, n_scored_changes, K, n_bins=None):
    xl, xu = get_bounds(e2ar.D)
    V = reference(e2ar.N_OBJ, e2ar.N_PARTITIONS)
    R = len(V)
    n_bins = R if n_bins is None else n_bins
    region_dynamics = make_region_dynamics_map(scenario, n_bins)
    pf_star = get_true_pf(n_points=500)

    rng = np.random.default_rng(seed)
    X_pop = rng.uniform(xl, xu, size=(e2ar.POP_SIZE, e2ar.D))
    t_current = e2ar.compute_time(0)
    X_pop = e2ar.evolve_nsga2(X_pop, t_current, region_dynamics,
                               e2ar.WARMUP_GENERATIONS, seed * 10000, xl, xu)

    history = RegionHistory(n_regions=R)
    rows = []
    gen = 0
    n_total_changes = 1 + n_scored_changes

    for change_idx in range(1, n_total_changes + 1):
        F_minus = op_evaluate(X_pop, t_current, region_dynamics)
        gen += e2ar.GENERATIONS_PER_CHANGE
        t_new = e2ar.compute_time(gen)
        F_plus = op_evaluate(X_pop, t_new, region_dynamics)

        F_tm, F_tp = normalize(F_minus, F_plus)
        F_tm = np.asarray(F_tm)
        region_ids = np.asarray(partition(F_tm, V))

        if change_idx == 1:
            history.update(t=0, X=X_pop, F=F_minus, region_ids=region_ids,
                           xl=xl, xu=xu)
            t_current = t_new
            X_pop = e2ar.evolve_nsga2(X_pop, t_current, region_dynamics,
                                       e2ar.GENERATIONS_PER_CHANGE,
                                       seed * 10000 + change_idx, xl, xu)
            continue

        def evaluate_fn(X_new, _t=t_new, _rd=region_dynamics):
            return op_evaluate(X_new, _t, _rd)

        history.update(t=change_idx - 1, X=X_pop, F=F_minus,
                       region_ids=region_ids, xl=xl, xu=xu)

        ab = run_event_ab(X_pop, F_plus, region_ids, R, history, change_idx,
                          evaluate_fn, pf_star, xl, xu, e2ar.BUDGET,
                          seed, change_idx, K)

        fi = ab["fixed_index"]
        fr = ab["fixed_rule"]

        row = {
            "scenario": scenario,
            "seed": seed,
            "change_idx": change_idx,
            "real_ROG": ab["ROG_standard"],
            "real_ROG_FE": ab["ROG_FE"],
            # fixed-index null
            "rand_ROG_fi_median": fi["rand_ROG_median"],
            "rand_ROG_fi_mean":   fi["rand_ROG_mean"],
            "rand_ROG_fi_std":    fi["rand_ROG_std"],
            "excess_ROG_fi":      fi["excess_ROG"],
            "real_ROG_pct_fi":    fi["real_ROG_percentile"],
            "rand_OHI_fi":        fi["rand_OHI_median"],
            "excess_OHI_fi":      fi["excess_OHI"],
            # fixed-rule null
            "rand_ROG_fr_median": fr["rand_ROG_median"],
            "rand_ROG_fr_mean":   fr["rand_ROG_mean"],
            "rand_ROG_fr_std":    fr["rand_ROG_std"],
            "excess_ROG_fr":      fr["excess_ROG"],
            "real_ROG_pct_fr":    fr["real_ROG_percentile"],
            "rand_OHI_fr":        fr["rand_OHI_median"],
            "excess_OHI_fr":      fr["excess_OHI"],
            # paired deltas
            "delta_null_ROG": ab["delta_null_ROG"],
            "delta_excess":   ab["delta_excess"],
        }
        rows.append(row)

        t_current = t_new
        X_pop = e2ar.evolve_nsga2(X_pop, t_current, region_dynamics,
                                   e2ar.GENERATIONS_PER_CHANGE,
                                   seed * 10000 + change_idx, xl, xu)

    return rows


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
    if obj is None:
        return None
    return obj


if __name__ == "__main__":
    N_SEEDS = 5
    N_SCORED = 5
    OUT_DIR = Path("results/null_ab_fixed_index_vs_fixed_rule")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "E2aR_null_ab.jsonl"

    all_rows = []
    t0 = time.time()

    for scenario in SCENARIOS_ALL:
        for seed in range(1, N_SEEDS + 1):
            t1 = time.time()
            rows = run_one_ab(scenario, seed, N_SCORED, K=K)
            all_rows.extend(rows)
            elapsed = time.time() - t1
            print(f"{scenario} seed={seed}: {len(rows)} events in {elapsed:.1f}s")

    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row)) + "\n")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Wrote {len(all_rows)} rows to {jsonl_path}")
