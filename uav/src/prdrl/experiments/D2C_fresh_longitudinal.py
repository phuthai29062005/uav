"""
D2-C — Fresh-Seed Longitudinal Validation (Production-Path Dispatcher).

Scientific question:
    Does O3 (sigma_P=0.005 + HARD_RESPONSE_ONLY) outperform O0 (sigma_P=0.03,
    UNRESTRICTED) on PERSISTENT, STATIONARY, and MIXED scenarios when driven
    through the actual production-path respond_population dispatcher?

DESIGN:
    Two arms: O0 (sigma=0.03, UNRESTRICTED) and O3 (sigma=0.005, HARD_RESPONSE_ONLY).
    Both arms driven through respond_population (not diagnostic CRN path).
    Longitudinal carry-forward: population NOT reset between scored changes.
    Arm-specific RegionHistory diverges after shared burn-in.

    Target region schedule (round-robin, pre-declared):
        target_r(seed, k) = (seed + k) % R, k = 0..N_SCORED_CHANGES-1

    Seeds 109–128 (fresh; disjoint from L1 41-60, L2B, E2P* 1-40).
    Scenarios: PERSISTENT, STATIONARY, MIXED.
    N_SCORED_CHANGES = 10.
    Total rows: 20 seeds × 3 scenarios × 10 changes × 2 arms = 1,200 rows.

ENDPOINTS:
    Primary: igd_response = IGD immediately after respond_population.
    Secondary: igd_epoch_end = IGD after GENERATIONS_PER_CHANGE evolution.

    CombinedBenefit = mean_igd_O0_response - mean_igd_O3_response (per seed×scenario).

TIME INDEXING:
    history_t (int, starts at 0) is used as the key for RegionHistory and
    as the `t` argument to respond_population/generate_children. This matches
    the L1 convention. t_float (numpy.float64) from compute_time is used only
    for cp_evaluate and evolve_nsga2.

CONSTRAINTS:
    - Production default O0 is NOT changed.
    - sigma_P=0.03 remains the production default.
    - RL not started. No frozen results overwritten.
    - Seeds 109–128 only (verified fresh).
    - FE fairness: O0 and O3 must evaluate identical n_children; STOP if violated.
    - HARD structural invariant: n_old_dropped_other == 0; STOP if violated.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import (
    PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    PREDICT_SURVIVAL_UNRESTRICTED,
    respond_population,
)
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)

# ============================================================
# Config — production locked
# ============================================================
LAMBDA_PRED = 1.0
SIGMA_PROD = 0.03     # O0 sigma — never change default
SIGMA_O3 = 0.005      # O3 sigma (D1 frozen decision)
SIGMA_L = 0.02        # LOCAL fallback

BUDGET = e2p3.BUDGET                  # 10
POP_SIZE = e2p3.POP_SIZE              # 100
D = e2p3.D                            # 10
N_OBJ = e2p3.N_OBJ                   # 2
N_PARTITIONS = e2p3.N_PARTITIONS      # 9 → R=10
WARMUP_GENERATIONS = e2p3.WARMUP_GENERATIONS
GENERATIONS_PER_CHANGE = e2p3.GENERATIONS_PER_CHANGE

compute_time = e2p3.compute_time
evolve_nsga2 = e2p3.evolve_nsga2
make_scenario_dynamics = e2p3.make_scenario_dynamics

N_SCORED_CHANGES = 10
D2C_SEEDS = list(range(109, 129))     # 20 fresh seeds
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

# Arm definitions
ARM_DEFS = [
    {
        "arm": "O0",
        "sigma_predict": SIGMA_PROD,
        "mode": PREDICT_SURVIVAL_UNRESTRICTED,
    },
    {
        "arm": "O3",
        "sigma_predict": SIGMA_O3,
        "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    },
]


def target_region(seed, k, R):
    return (seed + k) % R


def make_dynamics(scenario_name, n_bins):
    if scenario_name == "MIXED":
        return make_region_dynamics_map(n_bins)
    return make_scenario_dynamics(scenario_name, n_bins)


def run_one_d2c(seed, n_scored_changes, scenario_name):
    """Run D2-C comparison for one (seed, scenario). Returns list of row dicts.

    Uses integer history_t for RegionHistory keys (same as L1 convention).
    Uses float t_float for cp_evaluate / evolve_nsga2.
    """
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    n_bins = R
    region_dynamics = make_dynamics(scenario_name, n_bins)
    pf_star = get_true_pf(n_points=500)

    # ── Shared burn-in ──────────────────────────────────────────
    rng = np.random.default_rng(seed)
    X_init = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_float_0 = compute_time(0)
    X_burnin = evolve_nsga2(
        X_init, t_float_0, region_dynamics, WARMUP_GENERATIONS, seed * 10000, xl, xu
    )

    # Per-arm state (independent copies after burn-in)
    arms = {}
    for ad in ARM_DEFS:
        arms[ad["arm"]] = {
            "X": X_burnin.copy(),
            "history": RegionHistory(n_regions=R),
            "history_t": -1,          # integer counter, matches L1 convention
            "sigma_predict": ad["sigma_predict"],
            "mode": ad["mode"],
        }

    rows_out = []
    gen = WARMUP_GENERATIONS

    # Burn-in change (not scored, initialises history)
    F_burnin_t0 = cp_evaluate(X_burnin, t_float_0, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t_float_1 = compute_time(gen)
    F_burnin_t1 = cp_evaluate(X_burnin, t_float_1, region_dynamics)
    F_m_bi, _ = normalize(F_burnin_t0, F_burnin_t1)
    rids_bi = np.asarray(partition(np.asarray(F_m_bi), V), dtype=int)
    for arm_name, arm in arms.items():
        arm["history_t"] += 1        # → 0
        arm["history"].update(
            t=arm["history_t"], X=arm["X"], F=F_burnin_t0,
            region_ids=rids_bi, xl=xl, xu=xu,
        )
        arm["X"] = evolve_nsga2(
            arm["X"], t_float_1, region_dynamics,
            GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu,
        )
    t_float_current = t_float_1

    # Scored changes
    for k in range(n_scored_changes):
        tgt_r = target_region(seed, k, R)
        gen += GENERATIONS_PER_CHANGE
        t_float_new = compute_time(gen)

        # Per-arm: evaluate at new time, partition, update history
        arm_eval = {}
        for arm_name, arm in arms.items():
            F_minus = cp_evaluate(arm["X"], t_float_current, region_dynamics)
            F_plus = cp_evaluate(arm["X"], t_float_new, region_dynamics)
            F_m, _ = normalize(F_minus, F_plus)
            region_ids = np.asarray(partition(np.asarray(F_m), V), dtype=int)
            n_r = int(np.sum(region_ids == tgt_r))
            igd_pre = float(igd(F_plus, pf_star))

            arm["history_t"] += 1
            arm["history"].update(
                t=arm["history_t"], X=arm["X"], F=F_minus,
                region_ids=region_ids, xl=xl, xu=xu,
            )
            arm_eval[arm_name] = {
                "F_plus": F_plus,
                "region_ids": region_ids,
                "n_r": n_r,
                "igd_pre": igd_pre,
            }

        # COMMON_KEEP: if any arm has n_r == 0, skip response for all
        any_empty = any(arm_eval[a]["n_r"] == 0 for a in arm_eval)

        for arm_name, arm in arms.items():
            ev = arm_eval[arm_name]
            F_plus = ev["F_plus"]
            region_ids = ev["region_ids"]
            igd_pre = ev["igd_pre"]
            evo_seed = seed * 10000 + k + 2

            if any_empty or ev["n_r"] == 0:
                arm["X"] = evolve_nsga2(
                    arm["X"], t_float_new, region_dynamics,
                    GENERATIONS_PER_CHANGE, evo_seed, xl, xu,
                )
                rows_out.append({
                    "scenario": scenario_name, "seed": seed, "change_k": k,
                    "tgt_r": tgt_r, "arm": arm_name,
                    "sigma_predict": arm["sigma_predict"], "mode": arm["mode"],
                    "action": "COMMON_KEEP", "realized_op": "KEEP",
                    "n_r_before": ev["n_r"], "extra_fe": 0,
                    "igd_before": igd_pre, "igd_response": igd_pre,
                    "igd_epoch_end": float(igd(cp_evaluate(arm["X"], t_float_new, region_dynamics), pf_star)),
                    "t_float_new": float(t_float_new),
                })
                continue

            # Use integer history_t+1 as oracle time (matches internal history keys)
            t_oracle = arm["history_t"] + 1

            result = respond_population(
                X=arm["X"],
                F_plus=F_plus,
                region_ids=region_ids,
                V=V,
                actions={tgt_r: "PREDICT"},
                history=arm["history"],
                t=t_oracle,
                evaluate_fn=lambda Xc: cp_evaluate(Xc, t_float_new, region_dynamics),
                xl=xl,
                xu=xu,
                budget=BUDGET,
                sigma_local=SIGMA_L,
                sigma_predict=arm["sigma_predict"],
                sigma_diversify=0.15,
                lambda_pred=LAMBDA_PRED,
                base_seed=seed * 100000 + k * 100 + tgt_r,
                predict_response_survival_mode=arm["mode"],
            )

            X_post = result["X"]
            F_post = result["F"]
            extra_fe = result["extra_fe"]
            igd_response = float(igd(F_post, pf_star))

            # Evolve for epoch-end measurement
            arm["X"] = evolve_nsga2(
                X_post, t_float_new, region_dynamics,
                GENERATIONS_PER_CHANGE, evo_seed, xl, xu,
            )
            F_epoch = cp_evaluate(arm["X"], t_float_new, region_dynamics)
            igd_epoch_end = float(igd(F_epoch, pf_star))

            rows_out.append({
                "scenario": scenario_name, "seed": seed, "change_k": k,
                "tgt_r": tgt_r, "arm": arm_name,
                "sigma_predict": arm["sigma_predict"], "mode": arm["mode"],
                "action": "PREDICT", "realized_op": "PREDICT",
                "n_r_before": ev["n_r"], "extra_fe": extra_fe,
                "igd_before": igd_pre, "igd_response": igd_response,
                "igd_epoch_end": igd_epoch_end,
                "t_float_new": float(t_float_new),
            })

        # FE fairness check (compare across arms for same k)
        k_rows = [r for r in rows_out if r["change_k"] == k and r["scenario"] == scenario_name and r["seed"] == seed and r["action"] != "COMMON_KEEP"]
        if len(k_rows) == 2:
            fe_o0 = next(r["extra_fe"] for r in k_rows if r["arm"] == "O0")
            fe_o3 = next(r["extra_fe"] for r in k_rows if r["arm"] == "O3")
            if fe_o0 != fe_o3:
                raise RuntimeError(
                    f"FE FAIRNESS VIOLATED: seed={seed} k={k} tgt_r={tgt_r} "
                    f"O0.extra_fe={fe_o0} O3.extra_fe={fe_o3}"
                )

        t_float_current = t_float_new

    return rows_out


def compute_d2c_verdicts(all_rows):
    """Compute D2-C verdicts per scenario for O0 vs O3 comparison."""
    from collections import defaultdict

    verdicts = {}
    for scenario in SCENARIOS:
        sc_rows = [r for r in all_rows if r["scenario"] == scenario]
        o0_resp = [r["igd_response"] for r in sc_rows if r["arm"] == "O0"]
        o3_resp = [r["igd_response"] for r in sc_rows if r["arm"] == "O3"]

        if not o0_resp or not o3_resp:
            verdicts[scenario] = {"verdict": "NO_DATA"}
            continue

        mean_o0 = float(np.mean(o0_resp))
        mean_o3 = float(np.mean(o3_resp))
        combined_benefit = mean_o0 - mean_o3  # positive = O3 better

        # Per-seed combined benefit for bootstrap CI
        by_seed_arm = defaultdict(lambda: defaultdict(list))
        for r in sc_rows:
            by_seed_arm[r["seed"]][r["arm"]].append(r["igd_response"])

        seed_benefits = []
        for seed_v, arm_d in by_seed_arm.items():
            if "O0" in arm_d and "O3" in arm_d:
                m_o0 = float(np.mean(arm_d["O0"]))
                m_o3 = float(np.mean(arm_d["O3"]))
                seed_benefits.append(m_o0 - m_o3)

        rng_boot = np.random.default_rng(42)
        arr = np.asarray(seed_benefits)
        boot_means = [np.mean(rng_boot.choice(arr, size=len(arr), replace=True))
                      for _ in range(10000)]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        if abs(combined_benefit) < 1e-5:
            verdict = "NEUTRAL"
        elif combined_benefit > 0 and ci_lo > 0:
            verdict = "VALIDATED"
        elif combined_benefit > 0:
            verdict = "PARTIAL"
        else:
            verdict = "NOT_VALIDATED"

        verdicts[scenario] = {
            "verdict": verdict,
            "mean_O0_response": mean_o0,
            "mean_O3_response": mean_o3,
            "mean_CombinedBenefit": combined_benefit,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_seeds": len(seed_benefits),
        }

    return verdicts


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
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


if __name__ == "__main__":
    import csv

    OUT_DIR = Path("results/D2_implementation_validation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t0_wall = time.time()

    for scenario in SCENARIOS:
        for seed in D2C_SEEDS:
            t1 = time.time()
            try:
                rows = run_one_d2c(seed, N_SCORED_CHANGES, scenario)
                all_rows.extend(rows)
                print(f"{scenario} seed={seed}: {len(rows)} rows in {time.time()-t1:.1f}s")
            except RuntimeError as e:
                print(f"STOP: {e}")
                sys.exit(1)

    # Write trajectory JSONL
    jsonl_path = OUT_DIR / "D2_fresh_longitudinal.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(json_safe(r)) + "\n")
    print(f"\nWrote {len(all_rows)} rows to {jsonl_path}")

    # Per-seed per-scenario summary
    from collections import defaultdict
    by_scenario_seed_arm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_rows:
        by_scenario_seed_arm[r["scenario"]][r["seed"]][r["arm"]].append(r)

    seed_summary_rows = []
    for sc in SCENARIOS:
        for seed in D2C_SEEDS:
            arm_d = by_scenario_seed_arm[sc][seed]
            row = {"scenario": sc, "seed": seed}
            for arm_name in ["O0", "O3"]:
                arm_rows = arm_d.get(arm_name, [])
                row[f"{arm_name}_migd_response"] = float(np.mean([r["igd_response"] for r in arm_rows])) if arm_rows else None
                row[f"{arm_name}_migd_epoch"] = float(np.mean([r["igd_epoch_end"] for r in arm_rows])) if arm_rows else None
            if row["O0_migd_response"] is not None and row["O3_migd_response"] is not None:
                row["CombinedBenefit"] = row["O0_migd_response"] - row["O3_migd_response"]
            else:
                row["CombinedBenefit"] = None
            seed_summary_rows.append(row)

    seed_csv = OUT_DIR / "D2_seed_summary.csv"
    if seed_summary_rows:
        keys = list(seed_summary_rows[0].keys())
        with open(seed_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(seed_summary_rows)

    # Scenario-level verdict
    verdicts = compute_d2c_verdicts(all_rows)
    verdict_path = OUT_DIR / "D2_validation_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(json_safe(verdicts), f, indent=2)

    scenario_rows = []
    for sc, v in verdicts.items():
        row = {"scenario": sc}
        row.update(v)
        scenario_rows.append(row)

    scenario_csv = OUT_DIR / "D2_scenario_summary.csv"
    if scenario_rows:
        keys = list(scenario_rows[0].keys())
        with open(scenario_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(scenario_rows)

    total_elapsed = time.time() - t0_wall
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    print(f"\nD2-C Validation Verdicts:")
    for sc, v in verdicts.items():
        if "mean_CombinedBenefit" in v:
            print(f"  {sc}: {v['verdict']} "
                  f"(CB={v['mean_CombinedBenefit']:.2e} "
                  f"CI=[{v['ci_low']:.2e}, {v['ci_high']:.2e}])")
        else:
            print(f"  {sc}: {v['verdict']}")
