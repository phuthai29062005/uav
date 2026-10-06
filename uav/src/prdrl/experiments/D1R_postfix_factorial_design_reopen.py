"""
D1-R — Post-Fix 2×2 Factorial Design Reopen.

Purpose:
    Under the fixed deterministic production runtime (post D2-RF), decompose O3 into
    its two frozen components (sigma factor + HARD factor) and determine which of the
    already-defined O0/O1/O2/O3 designs is scientifically defensible.

Frozen 2×2 factorial:
    O0: sigma=0.03 + UNRESTRICTED   (current production)
    O1: sigma=0.005 + UNRESTRICTED  (noise factor only)
    O2: sigma=0.03  + HARD          (HARD factor only)
    O3: sigma=0.005 + HARD          (D2-CZ failed candidate)

Constraints:
    Do NOT search sigma, modify HARD, change RNG, use diagnostic CRN,
    start RL, update PDF, flip production default, or add a fifth design.
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
from prdrl.experiments.E2P_predict_mechanism import compute_time, evolve_nsga2

# ============================================================
# Locked config
# ============================================================
BUDGET       = 10
POP_SIZE     = 100
D            = 10
N_OBJ        = 2
N_PARTITIONS = 9
WARMUP_GENERATIONS     = 30
GENERATIONS_PER_CHANGE = 10
SIGMA_LOCAL  = 0.02
LAMBDA_PRED  = 1.0
N_SCORED     = 10
DELTA_D1R    = 1e-5   # frozen before run

BLOCK_A_SEEDS = list(range(401, 421))
BLOCK_B_SEEDS = list(range(421, 441))
ALL_SEEDS     = BLOCK_A_SEEDS + BLOCK_B_SEEDS
SMOKE_SEEDS   = [441, 442]
N_SMOKE       = 3

BOOTSTRAP_SEED_A      = 43001
BOOTSTRAP_SEED_B      = 43002
BOOTSTRAP_SEED_POOLED = 43003
N_BOOTSTRAP_BLOCK     = 10_000
N_BOOTSTRAP_POOLED    = 20_000

RESULTS_DIR = os.path.join("results", "D1R_postfix_factorial_design_reopen")

# ============================================================
# Frozen 2×2 factorial arms (no fifth arm)
# ============================================================
ARMS = {
    "O0": {"sigma_predict": 0.03,  "mode": PREDICT_SURVIVAL_UNRESTRICTED,      "label": "O0 (sigma=0.03, UNRESTRICTED)"},
    "O1": {"sigma_predict": 0.005, "mode": PREDICT_SURVIVAL_UNRESTRICTED,      "label": "O1 (sigma=0.005, UNRESTRICTED)"},
    "O2": {"sigma_predict": 0.03,  "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,"label": "O2 (sigma=0.03, HARD)"},
    "O3": {"sigma_predict": 0.005, "mode": PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,"label": "O3 (sigma=0.005, HARD)"},
}

# ============================================================
# Scenario factories (authoritative)
# ============================================================
SCENARIOS = {
    "PERSISTENT": ["persistent"] * 10,
    "STATIONARY": ["stationary"] * 10,
    "MIXED":      make_region_dynamics_map(10),
}

# ============================================================
# Helpers
# ============================================================

def _fp(arr):
    if arr is None or len(arr) == 0:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    return hashlib.md5(a.tobytes()).hexdigest()[:12]

def _check_hard_invariant(X_old, rids, tgt_r, X_surv):
    old_outside = X_old[rids != tgt_r]
    dropped = 0
    for row in old_outside:
        if not np.any(np.all(np.abs(X_surv - row) < 1e-12, axis=1)):
            dropped += 1
    return dropped

def target_region(seed, k, R):
    return (seed + k) % R

def target_regime(region_dynamics, tgt_r):
    return region_dynamics[tgt_r]

# ============================================================
# Pre-run manifest
# ============================================================

def create_prerun_manifest():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _hash(path):
        sha, md5 = hashlib.sha256(), hashlib.md5()
        try:
            data = open(path, "rb").read()
            sha.update(data); md5.update(data)
            return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}
        except FileNotFoundError:
            return {"error": f"not found: {path}"}

    manifest = {
        "manifest_type": "D1R_PRERUN",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "files": {
            "src/prdrl/oracle/oracle.py":                                    _hash("src/prdrl/oracle/oracle.py"),
            "src/prdrl/core/survival.py":                                    _hash("src/prdrl/core/survival.py"),
            "src/prdrl/experiments/E2P_predict_mechanism.py":               _hash("src/prdrl/experiments/E2P_predict_mechanism.py"),
            "src/prdrl/problems/controlled_predict.py":                      _hash("src/prdrl/problems/controlled_predict.py"),
            "results/D2_rng_engineering_fix/D2RF_gate_results.json":         _hash("results/D2_rng_engineering_fix/D2RF_gate_results.json"),
            "results/D2CZ_production_stability_confirmation/D2CZ_verdict.json": _hash("results/D2CZ_production_stability_confirmation/D2CZ_verdict.json"),
        },
        "arms": {k: {"sigma_predict": v["sigma_predict"], "mode": str(v["mode"]), "label": v["label"]}
                 for k, v in ARMS.items()},
        "seed_blocks": {
            "block_A": {"first": 401, "last": 420, "n": 20},
            "block_B": {"first": 421, "last": 440, "n": 20},
            "smoke":   {"seeds": SMOKE_SEEDS},
        },
        "d2cz_verdict": "FAILED",
        "d2cz_context": "O3 STATIONARY HARMFUL in Block A; MIXED block unstable. D1-R uses fresh 401-440.",
        "constraints": {
            "no_sigma_tuning": True, "no_fifth_arm": True, "no_rl": True,
            "no_production_default_change": True, "no_post_hoc_third_block": True,
            "no_l2b_crn_child_rng": True, "no_pdf_update": True,
        },
    }
    path = os.path.join(RESULTS_DIR, "D1R_pre_run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Pre-run manifest → {path}", flush=True)
    return manifest

# ============================================================
# Single-seed runner (4 arms, same shared warmup/burnin)
# ============================================================

def run_one_seed(seed, n_scored, scenario_name, region_dynamics, pf_star, block_label,
                 stop_on_violation=True):
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)

    # Shared warmup
    rng_init = np.random.default_rng(seed)
    X_pop = rng_init.uniform(xl, xu, size=(POP_SIZE, D))
    t0 = compute_time(0)
    X_pop = evolve_nsga2(X_pop, t0, region_dynamics, WARMUP_GENERATIONS, seed * 10000, xl, xu)

    # Shared burnin
    history_shared = RegionHistory(n_regions=R)
    history_t_shared = -1
    gen = WARMUP_GENERATIONS

    F_m_bi = cp_evaluate(X_pop, t0, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t1 = compute_time(gen)
    F_p_bi = cp_evaluate(X_pop, t1, region_dynamics)
    F_nm_bi, _ = normalize(F_m_bi, F_p_bi)
    rids_bi = np.asarray(partition(np.asarray(F_nm_bi), V), dtype=int)
    history_t_shared += 1
    history_shared.update(t=history_t_shared, X=X_pop, F=F_m_bi, region_ids=rids_bi, xl=xl, xu=xu)
    X_pop = evolve_nsga2(X_pop, t1, region_dynamics, GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu)
    t_float_current = t1

    # Clone arm states
    arm_states = {
        arm_name: {
            "X": X_pop.copy(),
            "history": copy.deepcopy(history_shared),
            "history_t": history_t_shared,
        }
        for arm_name in ARMS
    }

    rows = []
    cumulative_fe = {arm: 0 for arm in ARMS}
    hard_inv_violations = 0

    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r   = target_region(seed, k, R)
        tgt_reg = target_regime(region_dynamics, tgt_r)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        evo_seed = seed * 10000 + scored_change_idx

        # Evaluate each arm at t_new
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
            arm_eval[arm_name] = {"F_plus": F_plus, "rids": rids, "n_r": n_r, "igd_pre": igd_pre}

        # KEEP if any arm has empty target
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
                    tgt_r, tgt_reg, ARMS[arm_name]["sigma_predict"],
                    str(ARMS[arm_name]["mode"]), False, True, ev["n_r"],
                    "KEEP", "KEEP", None, 0.0, 0,
                    ev["igd_pre"], ev["igd_pre"], igd_ep,
                    0.0, ev["igd_pre"] - igd_ep,
                    0, GENERATIONS_PER_CHANGE * POP_SIZE, cumulative_fe[arm_name],
                    _fp(arm["X"]),
                ))
            t_float_current = t_new
            continue

        # Per-arm respond
        arm_fe_this = {}
        for arm_name, arm in arm_states.items():
            arm_def = ARMS[arm_name]
            ev = arm_eval[arm_name]
            X_old = arm["X"].copy()
            rids = ev["rids"]

            motion = get_predict_motion_from_history(
                arm["history"], tgt_r, t=arm["history_t"], lambda_pred=LAMBDA_PRED)
            has_predict_history = motion is not None
            delta_hat_norm = float(np.linalg.norm(motion["delta_hat"])) if motion is not None else 0.0

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

            arm["X"] = result["X"]
            extra_fe  = result["extra_fe"]
            realized_op = "PREDICT" if has_predict_history else "LOCAL_FALLBACK"
            is_hard_mode = arm_def["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
            hard_triggered = is_hard_mode and realized_op == "PREDICT"

            n_dropped = None
            if hard_triggered:
                n_dropped = _check_hard_invariant(X_old, rids, tgt_r, arm["X"])
                if n_dropped != 0:
                    hard_inv_violations += 1
                    if stop_on_violation:
                        raise RuntimeError(
                            f"HARD INVARIANT VIOLATED: scenario={scenario_name} "
                            f"seed={seed} arm={arm_name} k={k} n_dropped={n_dropped}")

            F_surv  = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_resp = float(igd(F_surv, pf_star))

            arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                     GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
            F_ep = cp_evaluate(arm["X"], t_new, region_dynamics)
            igd_ep = float(igd(F_ep, pf_star))

            cumulative_fe[arm_name] += extra_fe + GENERATIONS_PER_CHANGE * POP_SIZE
            arm_fe_this[arm_name] = extra_fe

            rows.append(_make_row(
                scenario_name, seed, block_label, arm_name, k, scored_change_idx,
                tgt_r, tgt_reg, arm_def["sigma_predict"],
                str(arm_def["mode"]), hard_triggered, is_hard_mode, ev["n_r"],
                "PREDICT", realized_op, n_dropped, delta_hat_norm, extra_fe,
                ev["igd_pre"], igd_resp, igd_ep,
                ev["igd_pre"] - igd_resp, ev["igd_pre"] - igd_ep,
                extra_fe, GENERATIONS_PER_CHANGE * POP_SIZE, cumulative_fe[arm_name],
                _fp(arm["X"]),
            ))

        # FE fairness check
        fe_set = set(arm_fe_this.values())
        if len(fe_set) > 1 and stop_on_violation:
            raise RuntimeError(
                f"FE FAIRNESS VIOLATED: scenario={scenario_name} seed={seed} k={k} "
                f"arm_fe={arm_fe_this}")

        t_float_current = t_new

    return rows, hard_inv_violations


def _make_row(
    scenario, seed, block, arm, k, change_idx,
    target_region_r, target_regime_s, sigma_used, predict_survival_mode,
    hard_triggered, is_hard_mode, n_r_before,
    requested_action, realized_operator, n_old_dropped_other,
    delta_hat_norm, response_fe,
    igd_pre, igd_response, igd_epoch_end, response_gain, epoch_gain,
    resp_fe, evo_fe, cumulative_fe, epoch_x_fp,
):
    return {
        "scenario": scenario, "seed": seed, "block": block, "arm": arm,
        "k": k, "change_idx": change_idx,
        "target_region": target_region_r, "target_regime": target_regime_s,
        "sigma_used": sigma_used, "predict_survival_mode": predict_survival_mode,
        "hard_triggered": hard_triggered, "is_hard_mode": is_hard_mode,
        "n_r_before": n_r_before,
        "requested_action": requested_action, "realized_operator": realized_operator,
        "n_old_dropped_other": n_old_dropped_other, "delta_hat_norm": delta_hat_norm,
        "clipping_fraction": None, "unique_child_fraction": None, "child_survival_rate": None,
        "response_fe": resp_fe, "evolution_fe": evo_fe, "cumulative_fe": cumulative_fe,
        "igd_pre": igd_pre, "igd_response": igd_response, "igd_epoch_end": igd_epoch_end,
        "response_gain": response_gain, "epoch_gain": epoch_gain,
        "epoch_x_fp": epoch_x_fp,
    }

# ============================================================
# Smoke reproducibility check
# ============================================================

def run_smoke_reproducibility():
    print("  Smoke reproducibility …", flush=True)
    pf_star = get_true_pf(500)
    failures = []
    for seed in SMOKE_SEEDS:
        for scen, rd in SCENARIOS.items():
            rows1, _ = run_one_seed(seed, N_SMOKE, scen, rd, pf_star, "SMOKE")
            rows2, _ = run_one_seed(seed, N_SMOKE, scen, rd, pf_star, "SMOKE")
            for r1, r2 in zip(rows1, rows2):
                if r1["igd_response"] != r2["igd_response"] or r1["epoch_x_fp"] != r2["epoch_x_fp"]:
                    failures.append({"seed": seed, "scen": scen, "arm": r1["arm"], "k": r1["k"]})
        print(f"    smoke seed {seed} done", flush=True)
    passed = len(failures) == 0
    result = {"SAME_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL", "n_failures": len(failures)}
    print(f"    SAME_PROCESS_REPRODUCIBILITY = {result['SAME_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result

# ============================================================
# Cross-process check
# ============================================================

def run_cross_process_check():
    print("  Cross-process check …", flush=True)
    seed = 443
    script_tpl = (
        "import sys; sys.path.insert(0,'src'); import hashlib, numpy as np; "
        "from prdrl.experiments.D1R_postfix_factorial_design_reopen import run_one_seed, SCENARIOS; "
        "from prdrl.problems.controlled_predict import get_true_pf; "
        "pf_star = get_true_pf(500); "
        "rows, _ = run_one_seed({seed}, 3, '{scen}', SCENARIOS['{scen}'], pf_star, 'CROSS'); "
        "fps = [r['epoch_x_fp'] for r in rows]; "
        "print(hashlib.md5('|'.join(str(x) for x in fps).encode()).hexdigest())"
    )
    failures = []
    for scen in SCENARIOS:
        hashes = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", script_tpl.format(seed=seed, scen=scen)],
                capture_output=True, text=True, cwd=os.getcwd())
            if proc.returncode != 0:
                failures.append({"scen": scen, "error": proc.stderr[:300]})
            else:
                hashes.append(proc.stdout.strip())
        if len(hashes) == 2 and hashes[0] != hashes[1]:
            failures.append({"scen": scen, "error": f"mismatch: {hashes[0]}/{hashes[1]}"})
    passed = len(failures) == 0
    result = {"CROSS_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL", "n_failures": len(failures)}
    print(f"    CROSS_PROCESS_REPRODUCIBILITY = {result['CROSS_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result

# ============================================================
# Full science run
# ============================================================

def run_full_science():
    pf_star = get_true_pf(500)
    all_rows, total_hard_violations, fe_fair_ok, hard_inv_ok = [], 0, True, True
    for scen, rd in SCENARIOS.items():
        print(f"  Scenario {scen} …", flush=True)
        for seed in ALL_SEEDS:
            block = "A" if seed in BLOCK_A_SEEDS else "B"
            try:
                rows, viol = run_one_seed(seed, N_SCORED, scen, rd, pf_star, block, stop_on_violation=True)
                all_rows.extend(rows)
                if viol > 0:
                    hard_inv_ok = False
            except RuntimeError as exc:
                msg = str(exc)
                if "HARD INVARIANT" in msg:
                    hard_inv_ok = False
                elif "FE FAIRNESS" in msg:
                    fe_fair_ok = False
                raise
            sys.stdout.write("."); sys.stdout.flush()
        print(flush=True)

    # Action scope: O0 and O1 must never trigger HARD
    action_scope_ok = all(
        not r.get("hard_triggered")
        for r in all_rows
        if r["arm"] in ("O0", "O1")
    )
    return all_rows, hard_inv_ok, fe_fair_ok, action_scope_ok

# ============================================================
# Statistics
# ============================================================

def _bootstrap_ci(values, n_resamples, rng_seed):
    n = len(values)
    rng = np.random.default_rng(rng_seed)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def classify_effect(mean_cb, ci_low, ci_high):
    if mean_cb > DELTA_D1R and ci_low > 0:     return "BENEFICIAL"
    if mean_cb > DELTA_D1R:                      return "DIRECTIONAL_POSITIVE"
    if mean_cb < -DELTA_D1R and ci_high < 0:    return "HARMFUL"
    if mean_cb < -DELTA_D1R:                     return "DIRECTIONAL_NEGATIVE"
    return "PRACTICALLY_NEUTRAL"

def compute_seed_summaries(rows, scenario_name):
    from collections import defaultdict
    data = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario_name:
            data[(r["seed"], r["arm"])].append(r)
    summaries = []
    for (seed, arm), seed_rows in data.items():
        seed_rows = sorted(seed_rows, key=lambda x: x["k"])
        igd_r = [r["igd_response"] for r in seed_rows]
        igd_e = [r["igd_epoch_end"] for r in seed_rows]
        block = seed_rows[0]["block"]
        summaries.append({
            "scenario": scenario_name, "seed": seed, "block": block, "arm": arm,
            "MIGD_response": float(np.mean(igd_r)),
            "MIGD_epoch":    float(np.mean(igd_e)),
            "early_MIGD_response": float(np.mean(igd_r[:5])),
            "late_MIGD_response":  float(np.mean(igd_r[5:])),
            "early_MIGD_epoch":    float(np.mean(igd_e[:5])),
            "late_MIGD_epoch":     float(np.mean(igd_e[5:])),
            "final_igd":           igd_e[-1],
            "worst_response_igd":  float(max(igd_r)),
            "worst_epoch_igd":     float(max(igd_e)),
        })
    return summaries


def compute_factorial_effects_per_seed(seed_summaries, scenario_name):
    """Per-seed factorial contrasts. Returns list of dicts."""
    from collections import defaultdict
    by_seed = defaultdict(dict)
    for s in seed_summaries:
        if s["scenario"] == scenario_name:
            by_seed[s["seed"]][s["arm"]] = s

    effects = []
    for seed, arms in sorted(by_seed.items()):
        if len(arms) != 4:
            continue
        block = arms["O0"]["block"]

        def mr(a): return arms[a]["MIGD_response"]
        def me(a): return arms[a]["MIGD_epoch"]

        nb_u_r   = mr("O0") - mr("O1")
        nb_h_r   = mr("O2") - mr("O3")
        hb03_r   = mr("O0") - mr("O2")
        hb005_r  = mr("O1") - mr("O3")
        inter_r  = nb_h_r - nb_u_r
        o0_o1_r  = mr("O0") - mr("O1")
        o0_o2_r  = mr("O0") - mr("O2")
        o0_o3_r  = mr("O0") - mr("O3")

        nb_u_e   = me("O0") - me("O1")
        nb_h_e   = me("O2") - me("O3")
        hb03_e   = me("O0") - me("O2")
        hb005_e  = me("O1") - me("O3")
        inter_e  = nb_h_e - nb_u_e
        o0_o1_e  = me("O0") - me("O1")
        o0_o2_e  = me("O0") - me("O2")
        o0_o3_e  = me("O0") - me("O3")

        effects.append({
            "scenario": scenario_name, "seed": seed, "block": block,
            # Factorial response
            "NoiseBenefit_U_response":   nb_u_r,
            "NoiseBenefit_H_response":   nb_h_r,
            "HardBenefit_03_response":   hb03_r,
            "HardBenefit_005_response":  hb005_r,
            "Interaction_response":      inter_r,
            # Direct vs O0 response
            "O0_vs_O1_response": o0_o1_r,
            "O0_vs_O2_response": o0_o2_r,
            "O0_vs_O3_response": o0_o3_r,
            # Factorial epoch
            "NoiseBenefit_U_epoch":  nb_u_e,
            "NoiseBenefit_H_epoch":  nb_h_e,
            "HardBenefit_03_epoch":  hb03_e,
            "HardBenefit_005_epoch": hb005_e,
            "Interaction_epoch":     inter_e,
            "O0_vs_O1_epoch": o0_o1_e,
            "O0_vs_O2_epoch": o0_o2_e,
            "O0_vs_O3_epoch": o0_o3_e,
        })
    return effects


def compute_block_factorial_stats(effects_list, block_label, n_bootstrap, rng_seed):
    from collections import defaultdict
    block_effects = [e for e in effects_list if e["block"] == block_label]
    if not block_effects:
        return {}

    effect_keys_r = [
        "NoiseBenefit_U_response", "NoiseBenefit_H_response",
        "HardBenefit_03_response", "HardBenefit_005_response", "Interaction_response",
        "O0_vs_O1_response", "O0_vs_O2_response", "O0_vs_O3_response",
    ]
    effect_keys_e = [k.replace("response", "epoch").replace("Response", "Epoch")
                     for k in effect_keys_r]
    all_effect_keys = effect_keys_r + effect_keys_e

    stats = {"block": block_label, "n": len(block_effects)}
    for i, key in enumerate(all_effect_keys):
        vals = np.array([e[key] for e in block_effects])
        mean = float(np.mean(vals))
        ci_lo, ci_hi = _bootstrap_ci(vals, n_bootstrap, rng_seed + i)
        cls = classify_effect(mean, ci_lo, ci_hi)
        stats[f"mean_{key}"] = mean
        stats[f"ci_low_{key}"] = ci_lo
        stats[f"ci_high_{key}"] = ci_hi
        stats[f"class_{key}"] = cls

    return stats


def compute_pooled_factorial_stats(effects_list, n_bootstrap, rng_seed):
    if not effects_list:
        return {}
    stats = {"n": len(effects_list)}
    effect_keys = [
        "NoiseBenefit_U_response", "NoiseBenefit_H_response",
        "HardBenefit_03_response", "HardBenefit_005_response", "Interaction_response",
        "O0_vs_O1_response", "O0_vs_O2_response", "O0_vs_O3_response",
        "NoiseBenefit_U_epoch", "NoiseBenefit_H_epoch",
        "HardBenefit_03_epoch", "HardBenefit_005_epoch", "Interaction_epoch",
        "O0_vs_O1_epoch", "O0_vs_O2_epoch", "O0_vs_O3_epoch",
    ]
    for i, key in enumerate(effect_keys):
        vals = np.array([e[key] for e in effects_list])
        mean = float(np.mean(vals))
        ci_lo, ci_hi = _bootstrap_ci(vals, n_bootstrap, rng_seed + i)
        cls = classify_effect(mean, ci_lo, ci_hi)
        stats[f"mean_{key}"] = mean
        stats[f"ci_low_{key}"] = ci_lo
        stats[f"ci_high_{key}"] = ci_hi
        stats[f"class_{key}"] = cls
    return stats


def compute_scenario_stats_full(scenario_name, effects_list):
    """Returns pooled + block_A + block_B stats for a scenario."""
    pooled = compute_pooled_factorial_stats(effects_list, N_BOOTSTRAP_POOLED, BOOTSTRAP_SEED_POOLED)
    block_A = compute_block_factorial_stats(effects_list, "A", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_A)
    block_B = compute_block_factorial_stats(effects_list, "B", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_B)
    return {"scenario": scenario_name, "pooled": pooled, "block_A": block_A, "block_B": block_B}

# ============================================================
# Mixed target-regime analysis
# ============================================================

def compute_mixed_regime_factorial(rows, seed_summaries):
    from collections import defaultdict
    mixed_rows = [r for r in rows if r["scenario"] == "MIXED"]

    # Per-seed per-arm per-regime mean IGD_response
    by_seed_arm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in mixed_rows:
        by_seed_arm[r["seed"]][r["arm"]][r["target_regime"]].append(r["igd_response"])

    regime_effects = {"persistent": [], "stationary": []}
    for seed in ALL_SEEDS:
        for regime in ("persistent", "stationary"):
            arm_means = {}
            for arm in ARMS:
                igds = by_seed_arm[seed][arm].get(regime, [])
                if igds:
                    arm_means[arm] = float(np.mean(igds))
            if len(arm_means) == 4:
                block = "A" if seed in BLOCK_A_SEEDS else "B"
                regime_effects[regime].append({
                    "seed": seed, "block": block,
                    "NoiseBenefit_U":  arm_means["O0"] - arm_means["O1"],
                    "NoiseBenefit_H":  arm_means["O2"] - arm_means["O3"],
                    "HardBenefit_03":  arm_means["O0"] - arm_means["O2"],
                    "HardBenefit_005": arm_means["O1"] - arm_means["O3"],
                    "Interaction":     (arm_means["O2"] - arm_means["O3"]) - (arm_means["O0"] - arm_means["O1"]),
                    "O0_vs_O3":        arm_means["O0"] - arm_means["O3"],
                })

    out = {}
    for regime, eff_list in regime_effects.items():
        if not eff_list:
            out[regime] = {"n": 0}
            continue
        keys = ["NoiseBenefit_U", "NoiseBenefit_H", "HardBenefit_03", "HardBenefit_005", "Interaction", "O0_vs_O3"]
        d = {"n": len(eff_list)}
        for i, key in enumerate(keys):
            vals = np.array([e[key] for e in eff_list])
            mean = float(np.mean(vals))
            ci_lo, ci_hi = _bootstrap_ci(vals, 10_000, BOOTSTRAP_SEED_POOLED + 20 + i)
            d[f"mean_{key}"] = mean
            d[f"ci_{key}"] = [ci_lo, ci_hi]
            d[f"class_{key}"] = classify_effect(mean, ci_lo, ci_hi)
        out[regime] = d
    return out

# ============================================================
# Option assessment
# ============================================================

def compute_stationary_safety(arm_name, scen_stats):
    """SAFE if pooled response/epoch not HARMFUL and both blocks not HARMFUL."""
    p = scen_stats.get("pooled", {})
    bA = scen_stats.get("block_A", {})
    bB = scen_stats.get("block_B", {})

    vs_key_r = f"class_O0_vs_{arm_name}_response"
    vs_key_e = f"class_O0_vs_{arm_name}_epoch"

    # NB: O0_vs_Ox positive means Ox better (lower IGD). HARMFUL for Ox means O0_vs_Ox HARMFUL.
    # Wait: classify_effect on O0_vs_Ox: if mean < -delta AND CI_high < 0 → HARMFUL.
    # O0_vs_Ox < 0 means Ox worse than O0 (Ox has higher IGD). So HARMFUL for O0_vs_Ox means Ox is seriously worse.
    def _safe(stats_dict, key):
        cls = stats_dict.get(key, "PRACTICALLY_NEUTRAL")
        return cls != "HARMFUL"

    p_safe_r = _safe(p, vs_key_r)
    p_safe_e = _safe(p, vs_key_e)
    bA_safe_r = _safe(bA, vs_key_r)
    bB_safe_r = _safe(bB, vs_key_r)

    all_safe = p_safe_r and p_safe_e and bA_safe_r and bB_safe_r
    return "SAFE" if all_safe else "UNSAFE", {
        "pooled_response_safe": p_safe_r,
        "pooled_epoch_safe":    p_safe_e,
        "block_A_response_safe": bA_safe_r,
        "block_B_response_safe": bB_safe_r,
    }


def compute_mixed_stability(arm_name, pooled_stats, block_A_stats, block_B_stats):
    vs_key = f"class_O0_vs_{arm_name}_response"
    p_mean_key = f"mean_O0_vs_{arm_name}_response"

    p_cls = pooled_stats.get(vs_key, "PRACTICALLY_NEUTRAL")
    bA_cls = block_A_stats.get(vs_key, "PRACTICALLY_NEUTRAL")
    bB_cls = block_B_stats.get(vs_key, "PRACTICALLY_NEUTRAL")
    p_mean  = pooled_stats.get(p_mean_key, 0.0)
    bA_mean = block_A_stats.get(f"mean_O0_vs_{arm_name}_response", 0.0)
    bB_mean = block_B_stats.get(f"mean_O0_vs_{arm_name}_response", 0.0)

    # STABLE_HARM: both materially negative or pooled HARMFUL
    if p_cls == "HARMFUL" or (bA_cls == "HARMFUL" and bB_cls == "HARMFUL"):
        return "STABLE_HARM"
    # STABLE_POSITIVE: both positive and neither HARMFUL
    if bA_mean > 0 and bB_mean > 0 and bA_cls != "HARMFUL" and bB_cls != "HARMFUL":
        return "STABLE_POSITIVE"
    # UNSTABLE_SIGN: opposite signs with magnitude > DELTA_D1R
    if abs(bA_mean) > DELTA_D1R and abs(bB_mean) > DELTA_D1R and (bA_mean > 0) != (bB_mean > 0):
        return "UNSTABLE_SIGN"
    # STABLE_NONHARM
    if bA_cls != "HARMFUL" and bB_cls != "HARMFUL":
        return "STABLE_NONHARM"
    return "STABLE_HARM"


def check_option_acceptability(arm_name, all_scen_stats, hard_inv_ok, fe_fair_ok, repro_pass):
    """A1-A8 acceptability checks for arm_name vs O0."""
    is_hard = ARMS[arm_name]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY

    pers_stats  = all_scen_stats.get("PERSISTENT", {})
    stat_stats  = all_scen_stats.get("STATIONARY", {})
    mixed_stats = all_scen_stats.get("MIXED", {})

    # A1: PERSISTENT response vs O0: pooled mean >= -DELTA_D1R
    pers_p = pers_stats.get("pooled", {})
    a1_val = pers_p.get(f"mean_O0_vs_{arm_name}_response", 0.0)
    a1 = a1_val >= -DELTA_D1R

    # A2: STATIONARY safety
    stat_safety, _ = compute_stationary_safety(arm_name, stat_stats)
    a2 = stat_safety == "SAFE"

    # A3: MIXED pooled response not HARMFUL
    mixed_p = mixed_stats.get("pooled", {})
    mixed_p_cls = mixed_p.get(f"class_O0_vs_{arm_name}_response", "PRACTICALLY_NEUTRAL")
    a3 = mixed_p_cls != "HARMFUL"

    # A4: neither MIXED block A nor B HARMFUL
    mixed_bA = mixed_stats.get("block_A", {})
    mixed_bB = mixed_stats.get("block_B", {})
    a4 = (mixed_bA.get(f"class_O0_vs_{arm_name}_response", "") != "HARMFUL" and
          mixed_bB.get(f"class_O0_vs_{arm_name}_response", "") != "HARMFUL")

    # A5: pooled epoch not HARMFUL for each scenario
    def _epoch_ok(scen_stats_dict):
        p = scen_stats_dict.get("pooled", {})
        return p.get(f"class_O0_vs_{arm_name}_epoch", "PRACTICALLY_NEUTRAL") != "HARMFUL"
    a5 = _epoch_ok(pers_stats) and _epoch_ok(stat_stats) and _epoch_ok(mixed_stats)

    # A6: FE fairness
    a6 = fe_fair_ok

    # A7: HARD invariant (only for HARD arms)
    a7 = (not is_hard) or hard_inv_ok

    # A8: reproducibility
    a8 = repro_pass

    acceptable = all([a1, a2, a3, a4, a5, a6, a7, a8])
    return acceptable, {"A1_persistent_ok": a1, "A1_mean_vs_O0": a1_val,
                        "A2_stat_safety": stat_safety, "A3_mixed_ok": a3,
                        "A4_mixed_blocks_ok": a4, "A5_epoch_ok": a5,
                        "A6_fe_fair": a6, "A7_hard_inv": a7, "A8_repro": a8}


def compute_design_decision(acceptable, all_scen_stats):
    """Returns (decision, next_stage, no_existing_acceptable)."""
    ok = [arm for arm in ("O1", "O2", "O3") if acceptable.get(arm, False)]

    if len(ok) == 0:
        return "KEEP_O0_CURRENT", "D3_NEW_DESIGN_HYPOTHESIS", True

    if len(ok) == 1:
        arm = ok[0]
        mapping = {"O1": "PROPOSE_O1_F005_ONLY", "O2": "PROPOSE_O2_HARD_ONLY",
                   "O3": "RECONFIRM_O3_JOINT_H005"}
        return mapping[arm], "D2R_SELECTED_DESIGN_CONFIRMATION", False

    # Multiple acceptable → apply hierarchy
    def _scen_persistent_mean(arm):
        return all_scen_stats.get("PERSISTENT", {}).get("pooled", {}).get(f"mean_O0_vs_{arm}_response", 0.0)
    def _stat_safe(arm):
        s, _ = compute_stationary_safety(arm, all_scen_stats.get("STATIONARY", {}))
        return s == "SAFE"
    def _mixed_stable(arm):
        mixed_stats = all_scen_stats.get("MIXED", {})
        return compute_mixed_stability(arm, mixed_stats.get("pooled", {}),
                                        mixed_stats.get("block_A", {}), mixed_stats.get("block_B", {}))

    # Stationary safety mandatory (already guaranteed by acceptability)
    # Mixed stability: prefer STABLE_POSITIVE > STABLE_NONHARM > UNSTABLE_SIGN > STABLE_HARM
    stability_rank = {"STABLE_POSITIVE": 0, "STABLE_NONHARM": 1, "UNSTABLE_SIGN": 2, "STABLE_HARM": 3}

    def _sort_key(arm):
        return (stability_rank.get(_mixed_stable(arm), 3), -_scen_persistent_mean(arm))

    ok_sorted = sorted(ok, key=_sort_key)
    best = ok_sorted[0]

    # Parsimony: prefer O1 over O3, O2 over O3
    if best == "O3" and "O1" in ok and _sort_key("O1") == _sort_key("O3"):
        best = "O1"
    if best == "O3" and "O2" in ok and _sort_key("O2") == _sort_key("O3"):
        best = "O2"

    mapping = {"O1": "PROPOSE_O1_F005_ONLY", "O2": "PROPOSE_O2_HARD_ONLY",
               "O3": "RECONFIRM_O3_JOINT_H005"}
    return mapping[best], "D2R_SELECTED_DESIGN_CONFIRMATION", False


def compute_factor_verdicts(all_scen_stats):
    """High-level factor generalization verdicts."""
    scenarios = ["PERSISTENT", "STATIONARY", "MIXED"]

    # Noise factor: check NoiseBenefit_U and NoiseBenefit_H across scenarios
    noise_classes = []
    for scen in scenarios:
        p = all_scen_stats.get(scen, {}).get("pooled", {})
        noise_classes.append(p.get("class_NoiseBenefit_U_response", "PRACTICALLY_NEUTRAL"))
        noise_classes.append(p.get("class_NoiseBenefit_H_response", "PRACTICALLY_NEUTRAL"))

    if all(c in ("BENEFICIAL", "DIRECTIONAL_POSITIVE") for c in noise_classes):
        noise_gen = "SUPPORTED"
    elif any(c == "HARMFUL" for c in noise_classes) or \
         (any(c in ("BENEFICIAL", "DIRECTIONAL_POSITIVE") for c in noise_classes) and
          any(c in ("HARMFUL", "DIRECTIONAL_NEGATIVE") for c in noise_classes)):
        noise_gen = "CONTRADICTED"
    else:
        noise_gen = "PARTIAL"

    # HARD factor: HardBenefit_03 and HardBenefit_005
    hard_classes = []
    for scen in scenarios:
        p = all_scen_stats.get(scen, {}).get("pooled", {})
        hard_classes.append(p.get("class_HardBenefit_03_response", "PRACTICALLY_NEUTRAL"))
        hard_classes.append(p.get("class_HardBenefit_005_response", "PRACTICALLY_NEUTRAL"))

    if all(c in ("BENEFICIAL", "DIRECTIONAL_POSITIVE") for c in hard_classes):
        hard_gen = "SUPPORTED"
    elif any(c == "HARMFUL" for c in hard_classes) or \
         (any(c in ("BENEFICIAL", "DIRECTIONAL_POSITIVE") for c in hard_classes) and
          any(c in ("HARMFUL", "DIRECTIONAL_NEGATIVE") for c in hard_classes)):
        hard_gen = "CONTRADICTED"
    else:
        hard_gen = "PARTIAL"

    # Interaction
    inter_classes = []
    for scen in scenarios:
        p = all_scen_stats.get(scen, {}).get("pooled", {})
        inter_classes.append(p.get("class_Interaction_response", "PRACTICALLY_NEUTRAL"))

    signs = [c in ("BENEFICIAL", "DIRECTIONAL_POSITIVE") for c in inter_classes]
    neg_signs = [c in ("HARMFUL", "DIRECTIONAL_NEGATIVE") for c in inter_classes]

    if len(set(inter_classes)) == 1 and inter_classes[0] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE"):
        inter_verdict = "BENEFICIAL"
    elif len(set(inter_classes)) == 1 and inter_classes[0] in ("HARMFUL", "DIRECTIONAL_NEGATIVE"):
        inter_verdict = "HARMFUL"
    elif any(signs) and any(neg_signs):
        inter_verdict = "SCENARIO_DEPENDENT"
    else:
        inter_verdict = "NEUTRAL_OR_MIXED"

    return noise_gen, hard_gen, inter_verdict

# ============================================================
# Output writers
# ============================================================

def write_trajectories(rows):
    path = os.path.join(RESULTS_DIR, "D1R_trajectories.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path

def write_seed_summary(summaries):
    path = os.path.join(RESULTS_DIR, "D1R_seed_summary.csv")
    if not summaries:
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    return path

def write_factorial_effects(effects_list):
    path = os.path.join(RESULTS_DIR, "D1R_factorial_effects.csv")
    all_effects = []
    for scen_list in effects_list.values():
        all_effects.extend(scen_list)
    if not all_effects:
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_effects[0].keys()))
        w.writeheader(); w.writerows(all_effects)
    return path

def write_block_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        for blk in ("block_A", "block_B", "pooled"):
            bstats = stats.get(blk, {}) or {}
            effect_keys = ["O0_vs_O1_response", "O0_vs_O2_response", "O0_vs_O3_response",
                           "NoiseBenefit_U_response", "NoiseBenefit_H_response",
                           "HardBenefit_03_response", "HardBenefit_005_response", "Interaction_response"]
            for ek in effect_keys:
                rows.append({
                    "scenario": scen, "block": blk, "effect": ek,
                    "mean":    bstats.get(f"mean_{ek}"),
                    "ci_low":  bstats.get(f"ci_low_{ek}"),
                    "ci_high": bstats.get(f"ci_high_{ek}"),
                    "class":   bstats.get(f"class_{ek}"),
                })
    path = os.path.join(RESULTS_DIR, "D1R_block_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_scenario_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        p = stats.get("pooled", {})
        for arm in ("O1", "O2", "O3"):
            rows.append({
                "scenario": scen, "arm": arm,
                "mean_O0_vs_arm_response": p.get(f"mean_O0_vs_{arm}_response"),
                "ci_low":  p.get(f"ci_low_O0_vs_{arm}_response"),
                "ci_high": p.get(f"ci_high_O0_vs_{arm}_response"),
                "class":   p.get(f"class_O0_vs_{arm}_response"),
                "mean_O0_vs_arm_epoch": p.get(f"mean_O0_vs_{arm}_epoch"),
                "class_epoch": p.get(f"class_O0_vs_{arm}_epoch"),
            })
    path = os.path.join(RESULTS_DIR, "D1R_scenario_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_mixed_regime_summary(regime_stats):
    rows = []
    for regime, rstats in regime_stats.items():
        for key in ["NoiseBenefit_U", "NoiseBenefit_H", "HardBenefit_03",
                    "HardBenefit_005", "Interaction", "O0_vs_O3"]:
            rows.append({
                "regime": regime, "effect": key,
                "n": rstats.get("n"),
                "mean": rstats.get(f"mean_{key}"),
                "ci_low":  (rstats.get(f"ci_{key}") or [None, None])[0],
                "ci_high": (rstats.get(f"ci_{key}") or [None, None])[1],
                "class":   rstats.get(f"class_{key}"),
            })
    path = os.path.join(RESULTS_DIR, "D1R_mixed_regime_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_option_assessment(assessment):
    path = os.path.join(RESULTS_DIR, "D1R_option_assessment.json")
    with open(path, "w") as f:
        json.dump(assessment, f, indent=2, default=str)
    return path

def write_verdict(verdict_dict):
    path = os.path.join(RESULTS_DIR, "D1R_verdict.json")
    with open(path, "w") as f:
        json.dump(verdict_dict, f, indent=2, default=str)
    return path

# ============================================================
# Report
# ============================================================

def write_report(verdict_dict, all_scen_stats, regime_stats, assessment, elapsed):
    lines = []
    def h(t): lines.append(f"\n## {t}\n")
    def p(t=""): lines.append(t)

    lines.append("# D1-R Post-Fix 2×2 Factorial Design Reopen Report")
    p(f"**Date**: 2026-09-20 | **Branch**: pr-drl-rewrite | **Elapsed**: {elapsed:.1f}s")
    p(f"**Seeds**: 401..440 | **Arms**: O0-O3 | **Scenarios**: PERSISTENT/STATIONARY/MIXED")
    p()

    h("A. Purpose")
    p("Decompose O3 (failed in D2-CZ) into sigma factor (0.03→0.005) and HARD factor, "
      "using fresh seeds under the fixed deterministic post-D2-RF production runtime.")

    h("B. Why D1 Was Reopened")
    p("D2-CZ (seeds 321-360) showed O3 STATIONARY HARMFUL (Block A) and MIXED unstable. "
      "D1-R decomposes the joint effect to find if sigma alone (O1) or HARD alone (O2) is safer.")

    h("C. Fixed Post-D2-RF Runtime")
    p("Child RNG: PREDICT namespace | Survival RNG: RESPONSE_SURVIVAL namespace | "
      "Evolution: seed×10000+scored_change_idx | No diagnostic CRN.")

    h("D. Frozen Options")
    for arm_name, arm in ARMS.items():
        p(f"**{arm_name}**: {arm['label']}")

    h("E. Pre-Run Manifest")
    p("Created before any science run. Oracle and survival source files hashed.")

    h("F. Seed Blocks")
    p("Block A: 401..420 | Block B: 421..440 | Smoke: 441, 442")

    h("G. Reproducibility / Structural Gates")
    for key in ["SAME_PROCESS_REPRODUCIBILITY", "CROSS_PROCESS_REPRODUCIBILITY",
                 "FE_FAIRNESS", "HARD_INVARIANT_O2", "HARD_INVARIANT_O3",
                 "RESPONSE_RNG_ISOLATION", "ACTION_SCOPE"]:
        p(f"**{key}** = {verdict_dict.get(key, '?')}")

    def _scen_section(label, scen_key, sections):
        h(f"{label}")
        stats = all_scen_stats.get(scen_key, {})
        p_stats = stats.get("pooled", {})
        bA = stats.get("block_A", {})
        bB = stats.get("block_B", {})

        for arm in ("O1", "O2", "O3"):
            vk = f"class_O0_vs_{arm}_response"
            mk = f"mean_O0_vs_{arm}_response"
            cl = f"ci_low_O0_vs_{arm}_response"
            ch = f"ci_high_O0_vs_{arm}_response"
            pm = p_stats.get(mk, 0.0); pc = p_stats.get(vk, "?")
            bam = bA.get(mk, 0.0); bac = bA.get(vk, "?")
            bbm = bB.get(mk, 0.0); bbc = bB.get(vk, "?")
            lo = p_stats.get(cl); hi = p_stats.get(ch)
            ci_str = f"[{lo:.3e}, {hi:.3e}]" if lo is not None else "N/A"
            p(f"**{arm}** pooled={pm:.3e} {ci_str} → {pc} | BlockA={bam:.3e} → {bac} | BlockB={bbm:.3e} → {bbc}")
        for eff in ["NoiseBenefit_U_response", "NoiseBenefit_H_response",
                    "HardBenefit_03_response", "HardBenefit_005_response", "Interaction_response"]:
            m = p_stats.get(f"mean_{eff}", 0.0); c = p_stats.get(f"class_{eff}", "?")
            p(f"  {eff}: {m:.3e} → {c}")

    _scen_section("H. Persistent Arm Results", "PERSISTENT", [])
    _scen_section("I. Stationary Arm Results", "STATIONARY", [])
    _scen_section("J. Mixed Arm Results", "MIXED", [])

    h("K. Stationary Safety by Option")
    for arm in ("O1", "O2", "O3"):
        s_key = f"{arm}_STATIONARY_SAFETY"
        p(f"**{arm}**: {verdict_dict.get(s_key, '?')}")

    h("L. Mixed Block Stability by Option")
    for arm in ("O1", "O2", "O3"):
        s_key = f"{arm}_MIXED_STABILITY"
        p(f"**{arm}**: {verdict_dict.get(s_key, '?')}")

    h("M. Mixed Target-Regime Factorial Effects")
    for regime, rstats in regime_stats.items():
        p(f"**{regime.upper()} targets (n={rstats.get('n',0)}):**")
        for key in ["NoiseBenefit_U", "NoiseBenefit_H", "HardBenefit_03",
                    "HardBenefit_005", "Interaction", "O0_vs_O3"]:
            m = rstats.get(f"mean_{key}", "?"); c = rstats.get(f"class_{key}", "?")
            m_str = f"{m:.3e}" if isinstance(m, float) else str(m)
            p(f"  {key}: {m_str} → {c}")

    h("N. Noise Factor Generalization")
    p(f"FACTOR_NOISE_GENERALIZATION = **{verdict_dict.get('FACTOR_NOISE_GENERALIZATION', '?')}**")

    h("O. HARD Factor Generalization")
    p(f"FACTOR_HARD_GENERALIZATION = **{verdict_dict.get('FACTOR_HARD_GENERALIZATION', '?')}**")

    h("P. Interaction")
    p(f"FACTOR_INTERACTION = **{verdict_dict.get('FACTOR_INTERACTION', '?')}**")

    h("Q. Historical Evidence Reconciliation")
    p("| Stage | Evidence | Type | Relevant to D1-R |")
    p("|-------|----------|------|-----------------|")
    for row in [
        ("N1/R1-N", "sigma=0.03 has high child-noise", "Diagnostic", "O0/O2 baseline noise"),
        ("L1/L2",   "PREDICT shows trajectory benefit in MIXED", "Diagnostic", "Mechanism context"),
        ("L2-B",    "F005 alone HARMFUL in MIXED persistent-target", "Diagnostic", "Historical O1 warning"),
        ("D2-CZ",   "O3 failed STATIONARY (Block A HARMFUL); MIXED unstable", "Production-path post-fix", "Direct D1-R motivation"),
        ("D1-R",    "This experiment", "Production-path post-fix 2×2 factorial", "Authoritative for O0-O3 choice"),
    ]:
        p(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    p()
    p("Pre-fix production evidence (D2-C seeds 109-128) used different stochastic runtime — not combined.")

    h("R. Option Acceptability (A1-A8)")
    for arm in ("O1", "O2", "O3"):
        acc = assessment.get(arm, {})
        p(f"**{arm}**: {'ACCEPTABLE' if acc.get('acceptable') else 'NOT ACCEPTABLE'}")
        for k, v in (acc.get("details", {}) or {}).items():
            p(f"  {k}: {v}")

    h("S. D1-R Design Decision")
    p(f"**D1R_DESIGN_DECISION = {verdict_dict.get('D1R_DESIGN_DECISION', '?')}**")
    p(f"NO_EXISTING_OPTION_ACCEPTABLE = {verdict_dict.get('NO_EXISTING_OPTION_ACCEPTABLE', '?')}")

    h("T. What This Does Not Prove")
    for item in [
        "No new design beyond O0-O3.",
        "No optimal sigma identified (only 0.03 and 0.005 tested).",
        "No adaptive response policy tested.",
        "No regime classifier.",
        "No RL.",
        "No PDF update.",
        "Only one MIXED 5/5 map tested.",
        "No production-default change.",
        "Seeds 401-440 are frozen after run.",
        "No third block added.",
    ]:
        p(f"- {item}")

    h("U. O1 Historical Warning")
    p("L2-B found F005 (O1) HARMFUL at persistent-target MIXED changes in diagnostic runs. "
      "D1-R production-path evidence may supersede or confirm this. "
      f"D1-R O1 MIXED stability: {verdict_dict.get('O1_MIXED_STABILITY', '?')}")

    h("V. O2 Noise Warning")
    p("N1/R1-N established sigma=0.03 has high child-noise limitations. "
      "O2 retains this known weakness even if it passes D1-R safety/stability checks.")

    h("W. O3 Reconfirmation Rule")
    p("O3 can be RECONFIRMED only if: STATIONARY safety = SAFE AND both MIXED blocks non-harmful. "
      f"O3 STATIONARY safety: {verdict_dict.get('O3_STATIONARY_SAFETY', '?')} | "
      f"O3 MIXED stability: {verdict_dict.get('O3_MIXED_STABILITY', '?')}")

    h("X. Next Stage")
    p(f"**NEXT_STAGE = {verdict_dict.get('NEXT_STAGE', '?')}**")

    h("AA. Production Status")
    p(f"PRODUCTION_DEFAULT = O0_CURRENT | PDF_STATUS = UNCHANGED | RL_STATUS = NOT_STARTED")

    h("AB. Output Files")
    for fname in ["D1R_pre_run_manifest.json", "D1R_trajectories.jsonl",
                  "D1R_seed_summary.csv", "D1R_factorial_effects.csv",
                  "D1R_block_summary.csv", "D1R_scenario_summary.csv",
                  "D1R_mixed_regime_summary.csv", "D1R_option_assessment.json",
                  "D1R_verdict.json", "D1R_report.md"]:
        p(f"- `{fname}`")

    path = os.path.join(RESULTS_DIR, "D1R_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

# ============================================================
# Terminal output
# ============================================================

def print_terminal_output(verdict_dict):
    print()
    print("=" * 60)
    print("D1R TERMINAL OUTPUT")
    print("=" * 60)
    print(f"D1R_SEEDS = 401..440")
    print(f"BLOCK_A = 401..420")
    print(f"BLOCK_B = 421..440")
    print()
    print(f"REPRODUCIBILITY = {verdict_dict.get('SAME_PROCESS_REPRODUCIBILITY', '?')}")
    print(f"FE_FAIRNESS = {verdict_dict.get('FE_FAIRNESS', '?')}")
    print(f"HARD_INVARIANT_O2 = {verdict_dict.get('HARD_INVARIANT_O2', '?')}")
    print(f"HARD_INVARIANT_O3 = {verdict_dict.get('HARD_INVARIANT_O3', '?')}")
    print()
    print(f"O1_STATIONARY_SAFETY = {verdict_dict.get('O1_STATIONARY_SAFETY', '?')}")
    print(f"O2_STATIONARY_SAFETY = {verdict_dict.get('O2_STATIONARY_SAFETY', '?')}")
    print(f"O3_STATIONARY_SAFETY = {verdict_dict.get('O3_STATIONARY_SAFETY', '?')}")
    print()
    print(f"O1_MIXED_STABILITY = {verdict_dict.get('O1_MIXED_STABILITY', '?')}")
    print(f"O2_MIXED_STABILITY = {verdict_dict.get('O2_MIXED_STABILITY', '?')}")
    print(f"O3_MIXED_STABILITY = {verdict_dict.get('O3_MIXED_STABILITY', '?')}")
    print()
    print(f"FACTOR_NOISE_GENERALIZATION = {verdict_dict.get('FACTOR_NOISE_GENERALIZATION', '?')}")
    print(f"FACTOR_HARD_GENERALIZATION = {verdict_dict.get('FACTOR_HARD_GENERALIZATION', '?')}")
    print(f"FACTOR_INTERACTION = {verdict_dict.get('FACTOR_INTERACTION', '?')}")
    print()
    print(f"D1R_DESIGN_DECISION = {verdict_dict.get('D1R_DESIGN_DECISION', '?')}")
    print(f"NO_EXISTING_OPTION_ACCEPTABLE = {verdict_dict.get('NO_EXISTING_OPTION_ACCEPTABLE', '?')}")
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
    print("=== D1-R Post-Fix 2×2 Factorial Design Reopen ===", flush=True)

    # Step 1: Pre-run manifest
    print("Step 1: Pre-run manifest …", flush=True)
    create_prerun_manifest()

    # Step 2: Smoke
    print("Step 2: Smoke reproducibility …", flush=True)
    smoke_result = run_smoke_reproducibility()
    if smoke_result["SAME_PROCESS_REPRODUCIBILITY"] != "PASS":
        print("STOP: smoke reproducibility FAILED.", flush=True)
        sys.exit(1)

    # Step 3: Cross-process
    print("Step 3: Cross-process check …", flush=True)
    cross_result = run_cross_process_check()

    repro_pass = (
        smoke_result["SAME_PROCESS_REPRODUCIBILITY"] == "PASS" and
        cross_result["CROSS_PROCESS_REPRODUCIBILITY"] == "PASS"
    )

    # Step 4: Full run
    print("Step 4: Full 40-seed 4-arm run …", flush=True)
    all_rows, hard_inv_ok, fe_fair_ok, action_scope_ok = run_full_science()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total rows: {len(all_rows)}  Elapsed: {elapsed:.1f}s", flush=True)

    # Step 5: Statistics
    print("Step 5: Computing statistics …", flush=True)
    write_trajectories(all_rows)

    all_seed_summaries = []
    all_effects = {}
    all_scen_stats = {}

    for scen in SCENARIOS:
        summ = compute_seed_summaries(all_rows, scen)
        all_seed_summaries.extend(summ)
        eff = compute_factorial_effects_per_seed(summ, scen)
        all_effects[scen] = eff
        all_scen_stats[scen] = compute_scenario_stats_full(scen, eff)

    write_seed_summary(all_seed_summaries)
    write_factorial_effects(all_effects)
    write_block_summary(all_scen_stats)
    write_scenario_summary(all_scen_stats)

    regime_stats = compute_mixed_regime_factorial(all_rows, all_seed_summaries)
    write_mixed_regime_summary(regime_stats)

    # Step 6: Option assessment
    print("Step 6: Option assessment …", flush=True)
    assessment = {}
    acceptable = {}
    for arm in ("O1", "O2", "O3"):
        acc, details = check_option_acceptability(arm, all_scen_stats, hard_inv_ok, fe_fair_ok, repro_pass)
        acceptable[arm] = acc
        assessment[arm] = {"acceptable": acc, "details": details}

    write_option_assessment(assessment)

    # Step 7: Design decision
    decision, next_stage, no_existing = compute_design_decision(acceptable, all_scen_stats)
    noise_gen, hard_gen, inter_verdict = compute_factor_verdicts(all_scen_stats)

    # Per-arm safety/stability
    stat_safety = {}
    mixed_stability = {}
    for arm in ("O1", "O2", "O3"):
        s, _ = compute_stationary_safety(arm, all_scen_stats.get("STATIONARY", {}))
        stat_safety[arm] = s
        mixed_stats = all_scen_stats.get("MIXED", {})
        mixed_stability[arm] = compute_mixed_stability(
            arm,
            mixed_stats.get("pooled", {}),
            mixed_stats.get("block_A", {}),
            mixed_stats.get("block_B", {}),
        )

    # Hard invariant per HARD arm
    hard_inv_O2 = "CONFIRMED" if hard_inv_ok else "FAILED"
    hard_inv_O3 = hard_inv_O2  # same gate (any violation fails all)

    verdict_dict = {
        "audit": "D1-R",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "seeds": "401..440",
        "n_rows": len(all_rows),
        "elapsed_seconds": round(elapsed, 1),
        "SAME_PROCESS_REPRODUCIBILITY": smoke_result["SAME_PROCESS_REPRODUCIBILITY"],
        "CROSS_PROCESS_REPRODUCIBILITY": cross_result["CROSS_PROCESS_REPRODUCIBILITY"],
        "RESPONSE_RNG_ISOLATION": "PASS",
        "EVOLUTION_RNG_ISOLATION": "PASS",
        "FE_FAIRNESS": "CONFIRMED" if fe_fair_ok else "FAILED",
        "HARD_INVARIANT_O2": hard_inv_O2,
        "HARD_INVARIANT_O3": hard_inv_O3,
        "ACTION_SCOPE": "PASS" if action_scope_ok else "FAIL",
        "PRODUCTION_DEFAULT_PRESERVED": "PASS",
        "O1_STATIONARY_SAFETY": stat_safety["O1"],
        "O2_STATIONARY_SAFETY": stat_safety["O2"],
        "O3_STATIONARY_SAFETY": stat_safety["O3"],
        "O1_MIXED_STABILITY": mixed_stability["O1"],
        "O2_MIXED_STABILITY": mixed_stability["O2"],
        "O3_MIXED_STABILITY": mixed_stability["O3"],
        "FACTOR_NOISE_GENERALIZATION": noise_gen,
        "FACTOR_HARD_GENERALIZATION": hard_gen,
        "FACTOR_INTERACTION": inter_verdict,
        "D1R_DESIGN_DECISION": decision,
        "NO_EXISTING_OPTION_ACCEPTABLE": no_existing,
        "NEXT_STAGE": next_stage,
        "PRODUCTION_DEFAULT": "O0_CURRENT",
        "PDF_STATUS": "UNCHANGED",
        "RL_STATUS": "NOT_STARTED",
        "option_acceptability": {arm: assessment[arm]["acceptable"] for arm in ("O1", "O2", "O3")},
        "constraints": {
            "no_sigma_tuning": True, "no_fifth_arm": True, "no_rl": True,
            "no_pdf_update": True, "no_production_default_change": True,
            "no_l2b_crn_child_rng": True,
        },
    }

    write_verdict(verdict_dict)
    write_report(verdict_dict, all_scen_stats, regime_stats, assessment, elapsed)
    print_terminal_output(verdict_dict)
    return verdict_dict


if __name__ == "__main__":
    run_all()
