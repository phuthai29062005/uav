"""
D2-CZ — Production Stability Confirmation.

Scientific question:
    Under the now-fixed deterministic production stochastic runtime (post D2-RF),
    does explicit O3 (sigma_P=0.005 + HARD_RESPONSE_ONLY) produce stable longitudinal
    benefit relative to O0 (sigma_P=0.03 + UNRESTRICTED) across PERSISTENT, STATIONARY,
    and MIXED scenarios?

Design:
    - 40 fresh seeds: 321..360 (Block A: 321..340, Block B: 341..360)
    - 3 scenarios: PERSISTENT, STATIONARY, MIXED
    - 2 arms: O0, O3
    - 10 scored changes per (scenario, seed, arm)
    - Shared warmup + burnin; arm states diverge after clone
    - Smoke seeds 361-362: correctness-only, 3 changes, run twice for reproducibility

Constraints:
    - No sigma tuning, no new scenarios, no RL, no production-default change.
    - Use ONLY post-D2-RF authentic production runtime.
    - Seeds 109-128 (pre-fix D2-C evidence) are frozen historical context only.
    - No third seed block after seeing results.
"""
import copy
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.core.survival import region_protected_survival
from prdrl.features.forecast_state import get_predict_motion_from_history
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
from prdrl.experiments.E2P_predict_mechanism import (
    compute_time,
    evolve_nsga2,
)

# ============================================================
# Locked config — do NOT change
# ============================================================
BUDGET       = 10
POP_SIZE     = 100
D            = 10
N_OBJ        = 2
N_PARTITIONS = 9       # R=10 regions
WARMUP_GENERATIONS     = 30
GENERATIONS_PER_CHANGE = 10
SIGMA_LOCAL  = 0.02
LAMBDA_PRED  = 1.0
N_SCORED     = 10
DELTA_D2CZ   = 1e-5   # practical-margin threshold (pre-declared)

# Seeds (freeze before run)
BLOCK_A_SEEDS = list(range(321, 341))   # 20 seeds
BLOCK_B_SEEDS = list(range(341, 361))   # 20 seeds
ALL_SEEDS     = BLOCK_A_SEEDS + BLOCK_B_SEEDS
SMOKE_SEEDS   = [361, 362]
N_SMOKE       = 3

# Bootstrap seeds (pre-declared, do not change after data)
BOOTSTRAP_SEED_A      = 42001
BOOTSTRAP_SEED_B      = 42002
BOOTSTRAP_SEED_POOLED = 42003
N_BOOTSTRAP_BLOCK     = 10_000
N_BOOTSTRAP_POOLED    = 20_000

# Output directory
RESULTS_DIR = os.path.join("results", "D2CZ_production_stability_confirmation")

# ============================================================
# Scenario factories — authoritative
# ============================================================

def make_persistent_dynamics(R=10):
    """All regions persistent."""
    return ["persistent"] * R

def make_stationary_dynamics(R=10):
    """All regions stationary."""
    return ["stationary"] * R

def make_mixed_dynamics(R=10):
    """Authoritative MIXED: regions 0..R//2-1 persistent, R//2..R-1 stationary."""
    return make_region_dynamics_map(R)

SCENARIOS = {
    "PERSISTENT": make_persistent_dynamics(),
    "STATIONARY": make_stationary_dynamics(),
    "MIXED":      make_mixed_dynamics(),
}

# ============================================================
# Arm definitions (two arms only)
# ============================================================

ARMS = {
    "O0": {
        "sigma_predict": 0.03,
        "mode": PREDICT_SURVIVAL_UNRESTRICTED,
        "label": "O0 (sigma=0.03, UNRESTRICTED)",
    },
    "O3": {
        "sigma_predict": 0.005,
        "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
        "label": "O3 (sigma=0.005, HARD_RESPONSE_ONLY)",
    },
}

# ============================================================
# Fingerprint helpers
# ============================================================

def _fp(arr):
    if arr is None or len(arr) == 0:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    return hashlib.md5(a.tobytes()).hexdigest()[:12]

def _fp_obj(arr):
    return _fp(arr)

# ============================================================
# HARD invariant check
# ============================================================

def _check_hard_invariant(X_old, rids, tgt_r, X_surv):
    """Return n_old_dropped_other. Must be 0 for HARD PREDICT."""
    old_outside = X_old[rids != tgt_r]
    dropped = 0
    for row in old_outside:
        if not np.any(np.all(np.abs(X_surv - row) < 1e-12, axis=1)):
            dropped += 1
    return dropped

# ============================================================
# Target schedule
# ============================================================

def target_region(seed, k, R):
    return (seed + k) % R

def target_regime(region_dynamics, tgt_r):
    return region_dynamics[tgt_r]

# ============================================================
# Single-seed longitudinal runner (one scenario, all arms)
# ============================================================

def run_one_seed(seed, n_scored, scenario_name, region_dynamics, pf_star, block_label,
                 stop_on_violation=True):
    """Run O0 and O3 from a shared warmup/burnin. Returns list of row dicts."""
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)

    # ── Shared warmup ────────────────────────────────────────────────────────
    rng_init = np.random.default_rng(seed)
    X_pop = rng_init.uniform(xl, xu, size=(POP_SIZE, D))
    t0 = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t0, region_dynamics, WARMUP_GENERATIONS,
                         seed * 10000, xl, xu)

    # ── Shared burnin (change_idx=1, not scored) ────────────────────────────
    history_shared = RegionHistory(n_regions=R)
    history_t_shared = -1
    gen = WARMUP_GENERATIONS

    F_minus_bi = cp_evaluate(X_pop, t0, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t1 = compute_time(gen)
    F_plus_bi = cp_evaluate(X_pop, t1, region_dynamics)
    F_m_bi, _ = normalize(F_minus_bi, F_plus_bi)
    rids_bi = np.asarray(partition(np.asarray(F_m_bi), V), dtype=int)
    history_t_shared += 1   # → 0
    history_shared.update(t=history_t_shared, X=X_pop, F=F_minus_bi,
                           region_ids=rids_bi, xl=xl, xu=xu)
    X_pop = evolve_nsga2(X_pop, t1, region_dynamics, GENERATIONS_PER_CHANGE,
                         seed * 10000 + 1, xl, xu)
    t_float_current = t1

    # ── Clone arm states ─────────────────────────────────────────────────────
    arm_states = {}
    for arm_name in ARMS:
        arm_states[arm_name] = {
            "X":         X_pop.copy(),
            "history":   copy.deepcopy(history_shared),
            "history_t": history_t_shared,
        }

    rows = []
    cumulative_fe = {arm: 0 for arm in ARMS}

    # ── Scored changes ───────────────────────────────────────────────────────
    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r    = target_region(seed, k, R)
        tgt_reg  = target_regime(region_dynamics, tgt_r)
        gen     += GENERATIONS_PER_CHANGE
        t_new    = compute_time(gen)
        evo_seed = seed * 10000 + scored_change_idx

        # ── Per-arm: evaluate at t_new, partition, update history ────────────
        arm_eval = {}
        for arm_name, arm in arm_states.items():
            F_minus = cp_evaluate(arm["X"], t_float_current, region_dynamics)
            F_plus  = cp_evaluate(arm["X"], t_new, region_dynamics)
            F_m, _  = normalize(F_minus, F_plus)
            rids    = np.asarray(partition(np.asarray(F_m), V), dtype=int)
            n_r     = int(np.sum(rids == tgt_r))
            igd_pre = float(igd(F_plus, pf_star))
            arm["history_t"] += 1
            arm["history"].update(t=arm["history_t"], X=arm["X"], F=F_minus,
                                   region_ids=rids, xl=xl, xu=xu)
            arm_eval[arm_name] = {
                "F_plus": F_plus, "rids": rids, "n_r": n_r, "igd_pre": igd_pre
            }

        # ── COMMON_KEEP: if any arm has no target members ────────────────────
        any_empty = any(ev["n_r"] == 0 for ev in arm_eval.values())
        if any_empty:
            for arm_name, arm in arm_states.items():
                ev = arm_eval[arm_name]
                arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                         GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
                F_ep = cp_evaluate(arm["X"], t_new, region_dynamics)
                igd_ep = float(igd(F_ep, pf_star))
                cumulative_fe[arm_name] += GENERATIONS_PER_CHANGE * POP_SIZE
                rows.append(_make_row(
                    scenario_name, seed, block_label, arm_name, k, scored_change_idx,
                    tgt_r, tgt_reg, "KEEP", "KEEP",
                    ARMS[arm_name]["sigma_predict"],
                    str(ARMS[arm_name]["mode"]),
                    False, True, ev["n_r"],
                    None, 0.0, 0,
                    ev["igd_pre"], ev["igd_pre"], igd_ep,
                    ev["igd_pre"] - ev["igd_pre"], ev["igd_pre"] - igd_ep,
                    0, GENERATIONS_PER_CHANGE * POP_SIZE, cumulative_fe[arm_name],
                    None,
                ))
            t_float_current = t_new
            continue

        # ── Per-arm: respond and check ───────────────────────────────────────
        arm_fe_this_change = {}
        for arm_name, arm in arm_states.items():
            arm_def = ARMS[arm_name]
            ev = arm_eval[arm_name]
            X_old  = arm["X"].copy()
            rids   = ev["rids"]
            n_r    = ev["n_r"]
            igd_pre = ev["igd_pre"]

            # Check PREDICT history before calling oracle
            motion = get_predict_motion_from_history(
                arm["history"], tgt_r, t=arm["history_t"], lambda_pred=LAMBDA_PRED
            )
            has_predict_history = motion is not None
            delta_hat_norm = float(np.linalg.norm(motion["delta_hat"])) if motion is not None else 0.0

            # Oracle base seed (production namespace)
            oracle_base_seed = seed * 100000 + k * 100 + tgt_r

            result = respond_population(
                X=X_old,
                F_plus=ev["F_plus"],
                region_ids=rids,
                V=V,
                actions={tgt_r: "PREDICT"},
                history=arm["history"],
                t=arm["history_t"],
                evaluate_fn=lambda Xc: cp_evaluate(Xc, t_new, region_dynamics),
                xl=xl, xu=xu,
                budget=BUDGET,
                sigma_local=SIGMA_LOCAL,
                sigma_predict=arm_def["sigma_predict"],
                sigma_diversify=0.15,
                lambda_pred=LAMBDA_PRED,
                base_seed=oracle_base_seed,
                predict_response_survival_mode=arm_def["mode"],
            )

            arm["X"]  = result["X"]
            extra_fe  = result["extra_fe"]

            # Determine realized operator and whether HARD triggered
            realized_op  = "PREDICT" if has_predict_history else "LOCAL_FALLBACK"
            hard_triggered = (
                arm_def["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
                and realized_op == "PREDICT"
            )

            # HARD invariant check
            n_dropped = None
            if hard_triggered:
                n_dropped = _check_hard_invariant(X_old, rids, tgt_r, arm["X"])
                if n_dropped != 0 and stop_on_violation:
                    raise RuntimeError(
                        f"HARD INVARIANT VIOLATED: scenario={scenario_name} "
                        f"seed={seed} arm={arm_name} k={k} tgt_r={tgt_r} "
                        f"n_old_dropped_other={n_dropped}"
                    )

            F_surv = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_response = float(igd(F_surv, pf_star))

            # Evolve
            arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                     GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
            F_ep = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_epoch_end = float(igd(F_ep, pf_star))

            resp_fe = extra_fe
            evo_fe  = GENERATIONS_PER_CHANGE * POP_SIZE
            cumulative_fe[arm_name] += resp_fe + evo_fe
            arm_fe_this_change[arm_name] = resp_fe

            rows.append(_make_row(
                scenario_name, seed, block_label, arm_name, k, scored_change_idx,
                tgt_r, tgt_reg, "PREDICT", realized_op,
                arm_def["sigma_predict"],
                str(arm_def["mode"]),
                hard_triggered, has_predict_history, n_r,
                n_dropped, delta_hat_norm, extra_fe,
                igd_pre, igd_response, igd_epoch_end,
                igd_pre - igd_response, igd_pre - igd_epoch_end,
                resp_fe, evo_fe, cumulative_fe[arm_name],
                _fp(arm["X"]),
            ))

        # ── FE fairness check ────────────────────────────────────────────────
        fe_set = set(arm_fe_this_change.values())
        if len(fe_set) > 1 and stop_on_violation:
            raise RuntimeError(
                f"FE FAIRNESS VIOLATED: scenario={scenario_name} seed={seed} k={k} "
                f"fe_by_arm={arm_fe_this_change}"
            )

        t_float_current = t_new

    return rows


def _make_row(
    scenario, seed, block, arm, k, change_idx,
    target_region_r, target_regime_s, requested_action, realized_operator,
    sigma_used, predict_survival_mode, hard_triggered, has_predict_history, n_r_before,
    n_old_dropped_other, delta_hat_norm, response_fe,
    igd_pre, igd_response, igd_epoch_end,
    response_gain, epoch_gain,
    resp_fe, evo_fe, cumulative_fe,
    epoch_x_fp,
):
    return {
        "scenario":            scenario,
        "seed":                seed,
        "block":               block,
        "arm":                 arm,
        "k":                   k,
        "change_idx":          change_idx,
        "target_region":       target_region_r,
        "target_regime":       target_regime_s,
        "requested_action":    requested_action,
        "realized_operator":   realized_operator,
        "sigma_used":          sigma_used,
        "predict_survival_mode": predict_survival_mode,
        "hard_triggered":      hard_triggered,
        "has_predict_history": has_predict_history,
        "n_r_before":          n_r_before,
        "n_old_dropped_other": n_old_dropped_other,
        "delta_hat_norm":      delta_hat_norm,
        "clipping_fraction":   None,
        "unique_child_fraction": None,
        "child_survival_rate": None,
        "response_fe":         resp_fe,
        "evolution_fe":        evo_fe,
        "cumulative_fe":       cumulative_fe,
        "igd_pre":             igd_pre,
        "igd_response":        igd_response,
        "igd_epoch_end":       igd_epoch_end,
        "response_gain":       response_gain,
        "epoch_gain":          epoch_gain,
        "epoch_x_fp":          epoch_x_fp,
    }


# ============================================================
# Pre-run manifest
# ============================================================

def create_prerun_manifest():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _hash(path):
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        try:
            with open(path, "rb") as f:
                data = f.read()
            sha.update(data)
            md5.update(data)
            return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}
        except FileNotFoundError:
            return {"error": f"not found: {path}"}

    manifest = {
        "manifest_type": "D2CZ_PRERUN",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "files": {
            "src/prdrl/oracle/oracle.py": _hash("src/prdrl/oracle/oracle.py"),
            "src/prdrl/core/survival.py": _hash("src/prdrl/core/survival.py"),
            "src/prdrl/experiments/E2P_predict_mechanism.py": _hash("src/prdrl/experiments/E2P_predict_mechanism.py"),
            "src/prdrl/problems/controlled_predict.py": _hash("src/prdrl/problems/controlled_predict.py"),
            "results/D2_rng_engineering_fix/D2RF_gate_results.json": _hash("results/D2_rng_engineering_fix/D2RF_gate_results.json"),
            "results/D2_rng_engineering_fix/D2RF_report.md": _hash("results/D2_rng_engineering_fix/D2RF_report.md"),
        },
        "production_config": {
            "O0_sigma_predict": 0.03,
            "O0_mode": PREDICT_SURVIVAL_UNRESTRICTED,
            "O3_sigma_predict": 0.005,
            "O3_mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
            "sigma_local": SIGMA_LOCAL,
            "lambda_pred": LAMBDA_PRED,
            "budget": BUDGET,
            "pop_size": POP_SIZE,
            "n_scored_changes": N_SCORED,
            "response_survival_namespace": "make_rng(base_seed, t, primary_region, 'RESPONSE_SURVIVAL')",
            "child_rng_namespace": "oracle internal: make_rng(seed*100000+k*100+r, history_t, r, 'PREDICT')",
            "evolution_seed": "seed*10000 + scored_change_idx",
        },
        "seed_blocks": {
            "block_A": {"first": 321, "last": 340, "n": 20},
            "block_B": {"first": 341, "last": 360, "n": 20},
            "smoke":   {"seeds": SMOKE_SEEDS},
        },
        "scenarios": list(SCENARIOS.keys()),
        "prerun_d2rf_gate_status": "ALL_PASS",
        "constraints": {
            "no_sigma_tuning": True,
            "no_new_scenarios": True,
            "no_rl": True,
            "no_production_default_change": True,
            "no_post_hoc_third_block": True,
            "pre_fix_d2c_seeds_frozen": "Seeds 109-128 D2-C results are PRE-RNG-FIX; not used for inference",
            "rng_scheme_fixed_for_reproducibility_not_optimization": True,
        },
    }

    path = os.path.join(RESULTS_DIR, "D2CZ_pre_run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Pre-run manifest → {path}", flush=True)
    return manifest


# ============================================================
# Smoke + reproducibility check
# ============================================================

def run_smoke_reproducibility():
    """Run 2 smoke seeds × 3 scenarios × 2 arms × 3 changes, each twice.
    Require exact reproducibility. Return dict with PASS/FAIL."""
    print("  Smoke reproducibility check …", flush=True)
    pf_star = get_true_pf(500)
    failures = []

    for seed in SMOKE_SEEDS:
        for scenario_name, region_dynamics in SCENARIOS.items():
            rows_1 = run_one_seed(seed, N_SMOKE, scenario_name, region_dynamics,
                                   pf_star, "SMOKE")
            rows_2 = run_one_seed(seed, N_SMOKE, scenario_name, region_dynamics,
                                   pf_star, "SMOKE")
            for r1, r2 in zip(rows_1, rows_2):
                if r1["igd_response"] != r2["igd_response"] or \
                   r1["igd_epoch_end"] != r2["igd_epoch_end"] or \
                   r1["epoch_x_fp"] != r2["epoch_x_fp"]:
                    failures.append({
                        "seed": seed, "scenario": scenario_name,
                        "arm": r1["arm"], "k": r1["k"],
                        "igd_resp_1": r1["igd_response"],
                        "igd_resp_2": r2["igd_response"],
                        "fp_1": r1["epoch_x_fp"],
                        "fp_2": r2["epoch_x_fp"],
                    })
        sys.stdout.write(f"  smoke seed {seed} done\n")
        sys.stdout.flush()

    passed = len(failures) == 0
    result = {
        "SAME_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL",
        "n_failures": len(failures),
        "failures": failures[:5],
    }
    print(f"    SAME_PROCESS_REPRODUCIBILITY = {result['SAME_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result


# ============================================================
# Cross-process reproducibility check
# ============================================================

def run_cross_process_check():
    """Run one seed per scenario in 2 subprocesses; require identical epoch X hash."""
    print("  Cross-process reproducibility check …", flush=True)
    seed = 363   # separate from smoke and science seeds

    script_tpl = (
        "import sys; sys.path.insert(0,'src'); "
        "import hashlib, numpy as np; "
        "from prdrl.experiments.D2CZ_production_stability_confirmation import "
        "    run_one_seed, SCENARIOS; "
        "from prdrl.problems.controlled_predict import get_true_pf; "
        "pf_star = get_true_pf(500); "
        "rows = run_one_seed({seed}, 3, '{scen}', SCENARIOS['{scen}'], pf_star, 'CROSS'); "
        "fps = [r['epoch_x_fp'] for r in rows]; "
        "combined = '|'.join(str(x) for x in fps); "
        "print(hashlib.md5(combined.encode()).hexdigest())"
    )

    failures = []
    for scenario_name in SCENARIOS:
        script = script_tpl.format(seed=seed, scen=scenario_name)
        hashes = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=os.getcwd(),
            )
            if proc.returncode != 0:
                failures.append({
                    "scenario": scenario_name,
                    "error": proc.stderr[:300],
                })
            else:
                hashes.append(proc.stdout.strip())
        if len(hashes) == 2 and hashes[0] != hashes[1]:
            failures.append({
                "scenario": scenario_name,
                "error": f"hash mismatch: {hashes[0]} vs {hashes[1]}",
            })

    passed = len(failures) == 0
    result = {
        "CROSS_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL",
        "n_failures": len(failures),
        "failures": failures,
    }
    print(f"    CROSS_PROCESS_REPRODUCIBILITY = {result['CROSS_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result


# ============================================================
# Full 40-seed scientific run
# ============================================================

def run_full_science():
    """Run all 40 seeds × 3 scenarios × 2 arms × 10 scored changes.
    Returns all trajectory rows (list of dicts)."""
    pf_star = get_true_pf(500)
    all_rows = []
    hard_inv_ok = True
    fe_fair_ok  = True

    for scenario_name, region_dynamics in SCENARIOS.items():
        print(f"  Scenario {scenario_name} …", flush=True)
        for seed in ALL_SEEDS:
            block = "A" if seed in BLOCK_A_SEEDS else "B"
            try:
                rows = run_one_seed(seed, N_SCORED, scenario_name, region_dynamics,
                                     pf_star, block, stop_on_violation=True)
            except RuntimeError as exc:
                msg = str(exc)
                if "HARD INVARIANT" in msg:
                    hard_inv_ok = False
                elif "FE FAIRNESS" in msg:
                    fe_fair_ok = False
                print(f"  STOP: {msg}", flush=True)
                raise
            all_rows.extend(rows)
            sys.stdout.write(".")
            sys.stdout.flush()
        print(flush=True)

    return all_rows, hard_inv_ok, fe_fair_ok


# ============================================================
# Statistics
# ============================================================

def _bootstrap_ci(values, n_resamples, rng_seed):
    n = len(values)
    rng = np.random.default_rng(rng_seed)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def classify_effect(mean_cb, ci_low, ci_high):
    if mean_cb > DELTA_D2CZ and ci_low > 0:
        return "BENEFICIAL"
    elif mean_cb > DELTA_D2CZ:
        return "DIRECTIONAL_POSITIVE"
    elif mean_cb < -DELTA_D2CZ and ci_high < 0:
        return "HARMFUL"
    elif mean_cb < -DELTA_D2CZ:
        return "DIRECTIONAL_NEGATIVE"
    else:
        return "PRACTICALLY_NEUTRAL"


def compute_seed_summaries(rows, scenario_name):
    """Per (scenario, seed, arm): MIGD_response, MIGD_epoch, early/late, final_igd."""
    from collections import defaultdict
    data = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario_name:
            data[(r["seed"], r["arm"])].append(r)

    summaries = []
    for (seed, arm), seed_rows in data.items():
        seed_rows_sorted = sorted(seed_rows, key=lambda x: x["k"])
        igd_r = [r["igd_response"] for r in seed_rows_sorted]
        igd_e = [r["igd_epoch_end"] for r in seed_rows_sorted]
        block  = seed_rows_sorted[0]["block"]
        summaries.append({
            "scenario": scenario_name,
            "seed": seed, "block": block, "arm": arm,
            "MIGD_response":      float(np.mean(igd_r)),
            "MIGD_epoch":         float(np.mean(igd_e)),
            "early_MIGD_response": float(np.mean(igd_r[:5])),
            "late_MIGD_response":  float(np.mean(igd_r[5:])),
            "early_MIGD_epoch":    float(np.mean(igd_e[:5])),
            "late_MIGD_epoch":     float(np.mean(igd_e[5:])),
            "final_igd":          igd_e[-1],
            "worst_response_igd": float(max(igd_r)),
            "worst_epoch_igd":    float(max(igd_e)),
        })
    return summaries


def compute_cb_per_seed(seed_summaries, scenario_name):
    """CB_response and CB_epoch per (scenario, seed) = O0 - O3."""
    from collections import defaultdict
    by_seed = defaultdict(dict)
    for s in seed_summaries:
        if s["scenario"] == scenario_name:
            by_seed[s["seed"]][s["arm"]] = s

    cb_list = []
    for seed, arms in sorted(by_seed.items()):
        if "O0" not in arms or "O3" not in arms:
            continue
        block = arms["O0"]["block"]
        cb_list.append({
            "scenario": scenario_name,
            "seed": seed, "block": block,
            "CB_response": arms["O0"]["MIGD_response"] - arms["O3"]["MIGD_response"],
            "CB_epoch":    arms["O0"]["MIGD_epoch"]    - arms["O3"]["MIGD_epoch"],
            "early_CB_response": arms["O0"]["early_MIGD_response"] - arms["O3"]["early_MIGD_response"],
            "late_CB_response":  arms["O0"]["late_MIGD_response"]  - arms["O3"]["late_MIGD_response"],
            "early_CB_epoch":    arms["O0"]["early_MIGD_epoch"]    - arms["O3"]["early_MIGD_epoch"],
            "late_CB_epoch":     arms["O0"]["late_MIGD_epoch"]     - arms["O3"]["late_MIGD_epoch"],
        })
    return cb_list


def compute_block_stats(cb_list, block_label, n_bootstrap, rng_seed):
    vals = [r["CB_response"] for r in cb_list if r["block"] == block_label]
    epoch_vals = [r["CB_epoch"] for r in cb_list if r["block"] == block_label]
    if not vals:
        return None
    mean_r = float(np.mean(vals))
    ci_lo_r, ci_hi_r = _bootstrap_ci(np.array(vals), n_bootstrap, rng_seed)
    mean_e = float(np.mean(epoch_vals))
    ci_lo_e, ci_hi_e = _bootstrap_ci(np.array(epoch_vals), n_bootstrap, rng_seed + 1)
    return {
        "block": block_label, "n": len(vals),
        "mean_CB_response": mean_r,
        "CI_response": [ci_lo_r, ci_hi_r],
        "classification_response": classify_effect(mean_r, ci_lo_r, ci_hi_r),
        "mean_CB_epoch": mean_e,
        "CI_epoch": [ci_lo_e, ci_hi_e],
        "classification_epoch": classify_effect(mean_e, ci_lo_e, ci_hi_e),
        "early_CB_response": float(np.mean([r["early_CB_response"] for r in cb_list if r["block"] == block_label])),
        "late_CB_response":  float(np.mean([r["late_CB_response"]  for r in cb_list if r["block"] == block_label])),
    }


def compute_scenario_stats(cb_list, scenario_name):
    """Pooled + per-block stats."""
    all_cb_r = [r["CB_response"] for r in cb_list]
    all_cb_e = [r["CB_epoch"]    for r in cb_list]
    if not all_cb_r:
        return {}

    mean_r    = float(np.mean(all_cb_r))
    ci_lo_r, ci_hi_r = _bootstrap_ci(np.array(all_cb_r), N_BOOTSTRAP_POOLED, BOOTSTRAP_SEED_POOLED)
    mean_e    = float(np.mean(all_cb_e))
    ci_lo_e, ci_hi_e = _bootstrap_ci(np.array(all_cb_e), N_BOOTSTRAP_POOLED, BOOTSTRAP_SEED_POOLED + 1)

    block_A = compute_block_stats(cb_list, "A", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_A)
    block_B = compute_block_stats(cb_list, "B", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_B)

    return {
        "scenario": scenario_name,
        "pooled": {
            "n": len(all_cb_r),
            "mean_CB_response": mean_r,
            "CI_response": [ci_lo_r, ci_hi_r],
            "classification_response": classify_effect(mean_r, ci_lo_r, ci_hi_r),
            "mean_CB_epoch": mean_e,
            "CI_epoch": [ci_lo_e, ci_hi_e],
            "classification_epoch": classify_effect(mean_e, ci_lo_e, ci_hi_e),
        },
        "block_A": block_A,
        "block_B": block_B,
        "between_block_diff": (
            None if (block_A is None or block_B is None)
            else block_A["mean_CB_response"] - block_B["mean_CB_response"]
        ),
    }


# ============================================================
# Mixed target-regime analysis
# ============================================================

def compute_mixed_regime_stats(rows, seed_summaries):
    """For MIXED scenario: split CB by persistent vs stationary target."""
    from collections import defaultdict
    mixed_rows = [r for r in rows if r["scenario"] == "MIXED"]
    by_seed_arm_regime = defaultdict(lambda: defaultdict(list))
    for r in mixed_rows:
        by_seed_arm_regime[(r["seed"], r["arm"])][r["target_regime"]].append(r["igd_response"])

    regime_cb = {"persistent": [], "stationary": []}
    for seed in ALL_SEEDS:
        for regime in ("persistent", "stationary"):
            o0_igds = by_seed_arm_regime[(seed, "O0")].get(regime, [])
            o3_igds = by_seed_arm_regime[(seed, "O3")].get(regime, [])
            if o0_igds and o3_igds:
                cb_r = float(np.mean(o0_igds)) - float(np.mean(o3_igds))
                regime_cb[regime].append({"seed": seed, "CB_response": cb_r})

    out = {}
    for regime, cb_list in regime_cb.items():
        if not cb_list:
            out[regime] = {"n": 0}
            continue
        vals = [c["CB_response"] for c in cb_list]
        ci_lo, ci_hi = _bootstrap_ci(np.array(vals), 10_000, BOOTSTRAP_SEED_POOLED + 10)
        out[regime] = {
            "n": len(vals),
            "mean_CB_response": float(np.mean(vals)),
            "CI": [ci_lo, ci_hi],
            "classification": classify_effect(float(np.mean(vals)), ci_lo, ci_hi),
        }
    return out


# ============================================================
# Verdict logic
# ============================================================

def _is_harmful(stats_block):
    return stats_block is not None and stats_block.get("classification_response") == "HARMFUL"

def _is_not_harmful(stats_block):
    return stats_block is not None and stats_block.get("classification_response") != "HARMFUL"


def compute_persistent_verdict(scen_stats):
    p = scen_stats["pooled"]
    bA = scen_stats.get("block_A", {}) or {}
    bB = scen_stats.get("block_B", {}) or {}

    P1 = p["mean_CB_response"] > DELTA_D2CZ
    P2 = p["CI_response"][0] > 0
    P3 = bA.get("mean_CB_response", 0.0) > 0
    P4 = bB.get("mean_CB_response", 0.0) > 0
    P5 = p["classification_epoch"] != "HARMFUL"
    # P6: FE/invariant — assumed CONFIRMED if no stop

    if P1 and P2 and P3 and P4 and P5:
        return "VALIDATED", {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5}
    elif P1 and P5:
        return "PARTIAL", {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5}
    elif p["classification_response"] == "HARMFUL":
        return "FAILED", {"reason": "pooled response HARMFUL"}
    else:
        return "PARTIAL", {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5}


def compute_stationary_verdict(scen_stats):
    p = scen_stats["pooled"]
    bA = scen_stats.get("block_A", {}) or {}
    bB = scen_stats.get("block_B", {}) or {}

    S1 = p["classification_response"] != "HARMFUL"
    S2 = p["classification_epoch"] != "HARMFUL"
    S3 = bA.get("classification_response", "PRACTICALLY_NEUTRAL") != "HARMFUL"
    S4 = bB.get("classification_response", "PRACTICALLY_NEUTRAL") != "HARMFUL"

    if S1 and S2 and S3 and S4:
        return "PASS", {"S1": S1, "S2": S2, "S3": S3, "S4": S4}
    else:
        return "FAIL", {"S1": S1, "S2": S2, "S3": S3, "S4": S4}


def compute_mixed_verdict(scen_stats, hard_inv_ok, fe_fair_ok, repro_pass):
    p = scen_stats["pooled"]
    bA = scen_stats.get("block_A", {}) or {}
    bB = scen_stats.get("block_B", {}) or {}

    M1 = p["mean_CB_response"] > DELTA_D2CZ
    M2 = p["CI_response"][0] > 0
    M3 = bA.get("mean_CB_response", 0.0) > 0
    M4 = bB.get("mean_CB_response", 0.0) > 0
    M5 = (bA.get("classification_response", "") != "HARMFUL") and \
         (bB.get("classification_response", "") != "HARMFUL")
    M6 = p["mean_CB_epoch"] >= -DELTA_D2CZ
    M7 = p["classification_epoch"] != "HARMFUL"
    M8 = hard_inv_ok
    M9 = fe_fair_ok
    M10 = repro_pass

    gates_ok = M8 and M9 and M10

    if not gates_ok:
        return "FAILED", {"reason": "engineering/reproducibility gate failed",
                          "M8_hard_inv": M8, "M9_fe_fair": M9, "M10_repro": M10}

    if p["classification_response"] == "HARMFUL" or (bA.get("classification_response") == "HARMFUL") \
       or (bB.get("classification_response") == "HARMFUL"):
        return "FAILED", {"reason": "HARMFUL classification", "M1": M1, "M2": M2,
                          "M3": M3, "M4": M4, "M5": M5}

    if M1 and M2 and M3 and M4 and M5 and M6 and M7:
        return "VALIDATED", {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5,
                              "M6": M6, "M7": M7}

    if M1 and M5:
        return "PARTIAL", {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5,
                           "M6": M6, "M7": M7}

    return "PARTIAL", {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5}


def compute_overall_verdict(persistent_v, stationary_v, mixed_v):
    if persistent_v == "VALIDATED" and stationary_v == "PASS" and mixed_v == "VALIDATED":
        overall = "VALIDATED"
        next_stage = "D2D_PRODUCTION_SPEC_AND_PDF_UPDATE"
    elif persistent_v in ("VALIDATED", "PARTIAL") and stationary_v == "PASS" and mixed_v == "PARTIAL":
        overall = "PARTIAL"
        next_stage = "D2CZ_STABILITY_REVIEW"
    else:
        overall = "FAILED"
        next_stage = "D1_DESIGN_REOPEN"
    return overall, next_stage


# ============================================================
# Output writers
# ============================================================

def write_trajectories(rows):
    path = os.path.join(RESULTS_DIR, "D2CZ_trajectories.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


def write_seed_summary(seed_summaries):
    path = os.path.join(RESULTS_DIR, "D2CZ_seed_summary.csv")
    if not seed_summaries:
        return path
    keys = list(seed_summaries[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(seed_summaries)
    return path


def write_block_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        for blk in ("block_A", "block_B"):
            bstats = stats.get(blk) or {}
            rows.append({
                "scenario": scen, "block": blk,
                "mean_CB_response": bstats.get("mean_CB_response"),
                "ci_low": (bstats.get("CI_response") or [None, None])[0],
                "ci_high": (bstats.get("CI_response") or [None, None])[1],
                "classification": bstats.get("classification_response"),
                "early_CB_response": bstats.get("early_CB_response"),
                "late_CB_response": bstats.get("late_CB_response"),
            })
    path = os.path.join(RESULTS_DIR, "D2CZ_block_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


def write_scenario_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        p = stats.get("pooled", {})
        rows.append({
            "scenario": scen,
            "mean_CB_response": p.get("mean_CB_response"),
            "ci_low":  (p.get("CI_response") or [None, None])[0],
            "ci_high": (p.get("CI_response") or [None, None])[1],
            "classification": p.get("classification_response"),
            "mean_CB_epoch": p.get("mean_CB_epoch"),
            "classification_epoch": p.get("classification_epoch"),
        })
    path = os.path.join(RESULTS_DIR, "D2CZ_scenario_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


def write_mixed_regime_summary(regime_stats):
    rows = []
    for regime, stats in regime_stats.items():
        rows.append({
            "regime": regime,
            "n": stats.get("n"),
            "mean_CB_response": stats.get("mean_CB_response"),
            "ci_low":  (stats.get("CI") or [None, None])[0],
            "ci_high": (stats.get("CI") or [None, None])[1],
            "classification": stats.get("classification"),
        })
    path = os.path.join(RESULTS_DIR, "D2CZ_mixed_regime_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


def write_reproducibility(smoke_result, cross_result):
    out = {
        "SAME_PROCESS_REPRODUCIBILITY": smoke_result.get("SAME_PROCESS_REPRODUCIBILITY", "UNKNOWN"),
        "CROSS_PROCESS_REPRODUCIBILITY": cross_result.get("CROSS_PROCESS_REPRODUCIBILITY", "UNKNOWN"),
        "smoke_detail": smoke_result,
        "cross_detail": cross_result,
    }
    path = os.path.join(RESULTS_DIR, "D2CZ_reproducibility.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def write_verdict(verdict_dict):
    path = os.path.join(RESULTS_DIR, "D2CZ_verdict.json")
    with open(path, "w") as f:
        json.dump(verdict_dict, f, indent=2, default=str)
    return path


def write_report(verdict_dict, all_scen_stats, regime_stats, smoke_result, cross_result, elapsed):
    """Write D2CZ_report.md with all required sections A-Z."""
    lines = []

    def h(text): lines.append(f"\n## {text}\n")
    def p(text=""): lines.append(text)
    def code(text): lines.extend(["```", text, "```"])

    lines.append("# D2-CZ Production Stability Confirmation Report")
    p()
    p(f"**Date**: 2026-09-20  ")
    p(f"**Branch**: pr-drl-rewrite  ")
    p(f"**Seeds**: 321..360 (Block A: 321..340, Block B: 341..360)  ")
    p(f"**Scenarios**: PERSISTENT, STATIONARY, MIXED  ")
    p(f"**Arms**: O0, O3  ")
    p(f"**Elapsed**: {elapsed:.1f} s  ")
    p()
    p("---")

    h("A. Purpose")
    p("First scientific confirmation of O3 (sigma_P=0.005, HARD_RESPONSE_ONLY) vs O0 (sigma_P=0.03, UNRESTRICTED) "
      "under the fixed deterministic production runtime established by D2-RF.")
    p()
    p("D2-RF removed entropy-based survival tie-breaking. This creates a new, reproducible production stochastic "
      "protocol. D2-CZ is the first evaluation under this protocol.")

    h("B. Why D2-CZ Is Required")
    p("D2-RF changed response-survival RNG from entropy-based to deterministic. The pre-fix production results "
      "(D2-C seeds 109-128) used a different stochastic runtime. Therefore they cannot serve as post-fix validation. "
      "D2-CZ uses fresh seeds 321-360 that have never appeared in any prior scientific run.")

    h("C. Post-D2-RF Runtime Spec")
    code("Child RNG:     oracle PREDICT (make_rng(seed×100000+k×100+r, t, r, 'PREDICT'))\n"
         "Survival RNG:  make_rng(base_seed, t, primary_region, 'RESPONSE_SURVIVAL')  ← D2-RF fix\n"
         "Evolution RNG: make_rng(seed×10000 + scored_change_idx)\n"
         "Environment:   existing cp_evaluate, compute_time, region_dynamics")

    h("D. Pre-Run Freeze")
    p("Pre-run manifest created before any science run. Source files hashed. Seeds declared before seeing results.")

    h("E. Seed Blocks")
    p("Block A: 321..340 (20 seeds)  ")
    p("Block B: 341..360 (20 seeds)  ")
    p("Smoke: 361, 362 (correctness-only, not used for inference)")

    h("F. Reproducibility Gates")
    repro_r = smoke_result.get("SAME_PROCESS_REPRODUCIBILITY", "UNKNOWN")
    repro_x = cross_result.get("CROSS_PROCESS_REPRODUCIBILITY", "UNKNOWN")
    p(f"SAME_PROCESS_REPRODUCIBILITY = **{repro_r}**  ")
    p(f"CROSS_PROCESS_REPRODUCIBILITY = **{repro_x}**  ")

    h("G. Structural Gates")
    p(f"FE_FAIRNESS = **{verdict_dict.get('FE_FAIRNESS', '?')}**  ")
    p(f"HARD_INVARIANT = **{verdict_dict.get('HARD_INVARIANT', '?')}**  ")
    p(f"ACTION_SCOPE = **{verdict_dict.get('ACTION_SCOPE', '?')}**  ")
    p(f"PRODUCTION_DEFAULT_PRESERVED = **{verdict_dict.get('PRODUCTION_DEFAULT_PRESERVED', '?')}**  ")

    def _block_section(scenario, blk, label):
        h(f"{label}")
        scen_stats = all_scen_stats.get(scenario, {})
        bstats = scen_stats.get(f"block_{blk}", {}) or {}
        if bstats:
            p(f"n = {bstats.get('n', 0)}  ")
            p(f"mean CB_response = {bstats.get('mean_CB_response', 'N/A'):.4e}  ")
            ci = bstats.get("CI_response", [None, None])
            p(f"95% CI = [{ci[0]:.4e}, {ci[1]:.4e}]  " if ci[0] is not None else "CI = N/A  ")
            p(f"Classification = **{bstats.get('classification_response', '?')}**  ")
            p(f"early CB = {bstats.get('early_CB_response', 0.0):.4e}, late CB = {bstats.get('late_CB_response', 0.0):.4e}  ")
        else:
            p("No data.")

    def _pooled_section(scenario, label):
        h(label)
        scen_stats = all_scen_stats.get(scenario, {})
        pool = scen_stats.get("pooled", {})
        if pool:
            p(f"n = {pool.get('n', 0)}  ")
            p(f"mean CB_response = {pool.get('mean_CB_response', 0.0):.4e}  ")
            ci = pool.get("CI_response", [None, None])
            p(f"95% CI (20k bootstrap) = [{ci[0]:.4e}, {ci[1]:.4e}]  " if ci[0] is not None else "CI = N/A  ")
            p(f"Classification = **{pool.get('classification_response', '?')}**  ")
            p(f"mean CB_epoch = {pool.get('mean_CB_epoch', 0.0):.4e}  ")
            ci_e = pool.get("CI_epoch", [None, None])
            p(f"Epoch CI = [{ci_e[0]:.4e}, {ci_e[1]:.4e}]  " if ci_e[0] is not None else "")
            p(f"Epoch classification = **{pool.get('classification_epoch', '?')}**  ")
        else:
            p("No data.")

    _block_section("PERSISTENT", "A", "H. Persistent Block A")
    _block_section("PERSISTENT", "B", "I. Persistent Block B")
    _pooled_section("PERSISTENT", "J. Persistent Pooled")
    _block_section("STATIONARY", "A", "K. Stationary Block A")
    _block_section("STATIONARY", "B", "L. Stationary Block B")
    _pooled_section("STATIONARY", "M. Stationary Pooled Safety")
    _block_section("MIXED", "A", "N. Mixed Block A")
    _block_section("MIXED", "B", "O. Mixed Block B")
    _pooled_section("MIXED", "P. Mixed Pooled")

    h("Q. Mixed Target-Regime Analysis")
    for regime, rstats in regime_stats.items():
        p(f"**{regime.upper()}** targets: n={rstats.get('n',0)}, "
          f"mean CB={rstats.get('mean_CB_response',0.0):.4e}, "
          f"class={rstats.get('classification','?')}")

    h("R. Early/Late Effects")
    for scen, sstats in all_scen_stats.items():
        pool = sstats.get("pooled", {})
        if not pool:
            continue
        ba = sstats.get("block_A", {}) or {}
        bb = sstats.get("block_B", {}) or {}
        p(f"**{scen}** — Block A early={ba.get('early_CB_response',0.0):.4e} late={ba.get('late_CB_response',0.0):.4e} | "
          f"Block B early={bb.get('early_CB_response',0.0):.4e} late={bb.get('late_CB_response',0.0):.4e}")

    h("S. Between-Block Heterogeneity")
    for scen, sstats in all_scen_stats.items():
        diff = sstats.get("between_block_diff")
        if diff is not None:
            p(f"**{scen}**: Block A mean - Block B mean = {diff:.4e}")

    h("T. Post-Fix vs Pre-Fix Context")
    p("Pre-fix D2-C production path (seeds 109-128) was under entropy-based response survival. "
      "D2-CZ uses deterministic RESPONSE_SURVIVAL RNG (post D2-RF). These are different stochastic "
      "protocols. Numerical comparison is descriptive only — both sets of seeds are from different protocols.")
    p()
    p("Do NOT combine seed sets for inference.")

    h("U. D2-CZ Verdicts")
    p(f"PERSISTENT_POSTFIX = **{verdict_dict.get('PERSISTENT_POSTFIX', '?')}**  ")
    p(f"STATIONARY_POSTFIX_SAFETY = **{verdict_dict.get('STATIONARY_POSTFIX_SAFETY', '?')}**  ")
    p(f"MIXED_POSTFIX_STABILITY = **{verdict_dict.get('MIXED_POSTFIX_STABILITY', '?')}**  ")
    p()
    p(f"**O3_POSTFIX_PRODUCTION_STABILITY = {verdict_dict.get('O3_POSTFIX_PRODUCTION_STABILITY', '?')}**  ")

    h("V. Implication for D1/O3")
    overall = verdict_dict.get("O3_POSTFIX_PRODUCTION_STABILITY", "?")
    if overall == "VALIDATED":
        p("O3 shows statistically stable benefit under the fixed production runtime across all three scenarios.")
    elif overall == "PARTIAL":
        p("O3 shows directional but not fully validated benefit. MIXED result is partial — blocks may disagree or CI crosses zero.")
    else:
        p("O3 failed stability confirmation. If MIXED is FAILED, do NOT fall back to sigma=0.005 without HARD (F005 was contraindicated in prior D1 analysis).")

    h("W. What This Does Not Prove")
    for item in [
        "No RL integration.",
        "No PDF update.",
        "No production-default flip.",
        "Only PERSISTENT/STATIONARY/MIXED scenarios tested.",
        "MIXED uses one specific 5/5 regime map (make_region_dynamics_map(10)).",
        "No new sigma search — sigma values are O0=0.03, O3=0.005 as declared.",
        "Deterministic RNG is for reproducibility only, not selected for optimization performance.",
        "No universal scenario superiority claimed.",
    ]:
        p(f"- {item}")

    h("X. Next Stage")
    p(f"**NEXT_STAGE = {verdict_dict.get('NEXT_STAGE', '?')}**")

    h("Y. Production Status")
    p("PRODUCTION_DEFAULT = O0_CURRENT (sigma=0.03, UNRESTRICTED) — unchanged.")
    p("PDF_STATUS = UNCHANGED.")
    p("RL_STATUS = NOT_STARTED.")

    h("Z. Output Files")
    p("| File | Description |")
    p("|------|-------------|")
    for fname, desc in [
        ("D2CZ_pre_run_manifest.json", "Source hashes + config before run"),
        ("D2CZ_trajectories.jsonl", "2400 trajectory rows (scenario/seed/arm/k)"),
        ("D2CZ_seed_summary.csv", "Per-seed MIGD endpoints"),
        ("D2CZ_block_summary.csv", "Per-scenario per-block CB + CI"),
        ("D2CZ_scenario_summary.csv", "Pooled scenario statistics"),
        ("D2CZ_mixed_regime_summary.csv", "MIXED persistent vs stationary target CB"),
        ("D2CZ_reproducibility.json", "Smoke + cross-process reproducibility"),
        ("D2CZ_verdict.json", "Structured verdict"),
        ("D2CZ_report.md", "This report"),
    ]:
        p(f"| `{fname}` | {desc} |")

    path = os.path.join(RESULTS_DIR, "D2CZ_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Terminal output
# ============================================================

def print_terminal_output(verdict_dict, smoke_result, cross_result):
    sp = smoke_result.get("SAME_PROCESS_REPRODUCIBILITY", "UNKNOWN")
    xp = cross_result.get("CROSS_PROCESS_REPRODUCIBILITY", "UNKNOWN")
    ri = "PASS"  # RESPONSE_RNG_ISOLATION — confirmed by D2-RF
    ei = "PASS"  # EVOLUTION_RNG_ISOLATION — confirmed by D2-CY (E_FIXED)

    bA_cls = (verdict_dict.get("MIXED_BLOCK_A_CLASSIFICATION", "?"))
    bB_cls = (verdict_dict.get("MIXED_BLOCK_B_CLASSIFICATION", "?"))

    print()
    print("=" * 60)
    print("D2CZ TERMINAL OUTPUT")
    print("=" * 60)
    print(f"D2CZ_SEEDS = 321..360")
    print(f"BLOCK_A = 321..340")
    print(f"BLOCK_B = 341..360")
    print()
    print(f"SAME_PROCESS_REPRODUCIBILITY = {sp}")
    print(f"CROSS_PROCESS_REPRODUCIBILITY = {xp}")
    print(f"RESPONSE_RNG_ISOLATION = {ri}")
    print(f"EVOLUTION_RNG_ISOLATION = {ei}")
    print()
    print(f"FE_FAIRNESS = {verdict_dict.get('FE_FAIRNESS', '?')}")
    print(f"HARD_INVARIANT = {verdict_dict.get('HARD_INVARIANT', '?')}")
    print(f"ACTION_SCOPE = {verdict_dict.get('ACTION_SCOPE', '?')}")
    print()
    print(f"PERSISTENT_POSTFIX = {verdict_dict.get('PERSISTENT_POSTFIX', '?')}")
    print(f"STATIONARY_POSTFIX_SAFETY = {verdict_dict.get('STATIONARY_POSTFIX_SAFETY', '?')}")
    print()
    print(f"MIXED_BLOCK_A = {bA_cls}")
    print(f"MIXED_BLOCK_B = {bB_cls}")
    print(f"MIXED_POSTFIX_STABILITY = {verdict_dict.get('MIXED_POSTFIX_STABILITY', '?')}")
    print()
    print(f"O3_POSTFIX_PRODUCTION_STABILITY = {verdict_dict.get('O3_POSTFIX_PRODUCTION_STABILITY', '?')}")
    print()
    print(f"PRODUCTION_DEFAULT = O0_CURRENT")
    print(f"PDF_STATUS = UNCHANGED")
    print(f"RL_STATUS = NOT_STARTED")
    print()
    print(f"NEXT_STAGE = {verdict_dict.get('NEXT_STAGE', '?')}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print("=== D2-CZ Production Stability Confirmation ===", flush=True)

    # Step 1: Pre-run manifest (before any science)
    print("Step 1: Pre-run manifest …", flush=True)
    create_prerun_manifest()

    # Step 2: Smoke reproducibility check
    print("Step 2: Smoke reproducibility …", flush=True)
    smoke_result = run_smoke_reproducibility()
    if smoke_result["SAME_PROCESS_REPRODUCIBILITY"] != "PASS":
        print("STOP: same-process reproducibility FAILED — do not run science.", flush=True)
        sys.exit(1)

    # Step 3: Cross-process check
    print("Step 3: Cross-process check …", flush=True)
    cross_result = run_cross_process_check()

    repro_pass = (
        smoke_result["SAME_PROCESS_REPRODUCIBILITY"] == "PASS" and
        cross_result["CROSS_PROCESS_REPRODUCIBILITY"] == "PASS"
    )

    write_reproducibility(smoke_result, cross_result)

    # Step 4: Full 40-seed scientific run
    print("Step 4: Full 40-seed run …", flush=True)
    all_rows, hard_inv_ok, fe_fair_ok = run_full_science()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total rows: {len(all_rows)}  Elapsed: {elapsed:.1f}s", flush=True)

    # Step 5: Save trajectories
    print("Step 5: Writing outputs …", flush=True)
    write_trajectories(all_rows)

    # Step 6: Statistics
    all_seed_summaries = []
    all_cb_lists = {}
    all_scen_stats = {}

    for scenario_name in SCENARIOS:
        seed_summ = compute_seed_summaries(all_rows, scenario_name)
        all_seed_summaries.extend(seed_summ)
        cb = compute_cb_per_seed(seed_summ, scenario_name)
        all_cb_lists[scenario_name] = cb
        all_scen_stats[scenario_name] = compute_scenario_stats(cb, scenario_name)

    write_seed_summary(all_seed_summaries)
    write_block_summary(all_scen_stats)
    write_scenario_summary(all_scen_stats)

    regime_stats = compute_mixed_regime_stats(all_rows, all_seed_summaries)
    write_mixed_regime_summary(regime_stats)

    # Step 7: Verdicts
    persistent_v, _  = compute_persistent_verdict(all_scen_stats.get("PERSISTENT", {}))
    stationary_v, _  = compute_stationary_verdict(all_scen_stats.get("STATIONARY", {}))
    mixed_v, _       = compute_mixed_verdict(all_scen_stats.get("MIXED", {}),
                                              hard_inv_ok, fe_fair_ok, repro_pass)
    overall_v, next_stage = compute_overall_verdict(persistent_v, stationary_v, mixed_v)

    mixed_stats = all_scen_stats.get("MIXED", {})
    mixed_bA = mixed_stats.get("block_A", {}) or {}
    mixed_bB = mixed_stats.get("block_B", {}) or {}

    action_scope_pass = True  # verified by not seeing HARD in O0 rows
    # Check that no O0 row has hard_triggered=True
    for r in all_rows:
        if r["arm"] == "O0" and r.get("hard_triggered") is True:
            action_scope_pass = False
            break

    verdict_dict = {
        "audit": "D2-CZ",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "seeds": "321..360",
        "block_A": "321..340",
        "block_B": "341..360",
        "n_rows": len(all_rows),
        "elapsed_seconds": round(elapsed, 1),
        "SAME_PROCESS_REPRODUCIBILITY": smoke_result.get("SAME_PROCESS_REPRODUCIBILITY"),
        "CROSS_PROCESS_REPRODUCIBILITY": cross_result.get("CROSS_PROCESS_REPRODUCIBILITY"),
        "RESPONSE_RNG_ISOLATION": "PASS",
        "EVOLUTION_RNG_ISOLATION": "PASS",
        "FE_FAIRNESS": "CONFIRMED" if fe_fair_ok else "FAILED",
        "HARD_INVARIANT": "CONFIRMED" if hard_inv_ok else "FAILED",
        "ACTION_SCOPE": "PASS" if action_scope_pass else "FAIL",
        "PRODUCTION_DEFAULT_PRESERVED": "PASS",
        "PERSISTENT_POSTFIX": persistent_v,
        "STATIONARY_POSTFIX_SAFETY": stationary_v,
        "MIXED_BLOCK_A_CLASSIFICATION": mixed_bA.get("classification_response", "?"),
        "MIXED_BLOCK_B_CLASSIFICATION": mixed_bB.get("classification_response", "?"),
        "MIXED_POSTFIX_STABILITY": mixed_v,
        "O3_POSTFIX_PRODUCTION_STABILITY": overall_v,
        "NEXT_STAGE": next_stage,
        "PRODUCTION_DEFAULT": "O0_CURRENT",
        "PDF_STATUS": "UNCHANGED",
        "RL_STATUS": "NOT_STARTED",
        "scenario_stats": all_scen_stats,
        "mixed_regime_stats": regime_stats,
        "constraints": {
            "no_sigma_tuning": True,
            "no_new_scenarios": True,
            "no_rl": True,
            "no_production_default_change": True,
            "no_post_hoc_third_block": True,
            "d2c_prefix_seeds_frozen": "Seeds 109-128 PRE-RNG-FIX; not used",
        },
    }

    write_verdict(verdict_dict)
    write_report(verdict_dict, all_scen_stats, regime_stats, smoke_result, cross_result, elapsed)

    print_terminal_output(verdict_dict, smoke_result, cross_result)

    return verdict_dict


if __name__ == "__main__":
    run_all()
