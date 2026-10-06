"""
D2-R — Selected O1 Design Confirmation.

Scientific question:
    Under the fixed deterministic production runtime (post D2-RF), does selected O1
    (sigma_P=0.005, UNRESTRICTED) provide a reproducible, safe, and sufficiently stable
    replacement candidate for O0 (sigma_P=0.03, UNRESTRICTED) across PERSISTENT,
    STATIONARY, and MIXED?

D1-R context:
    O1 was selected from a four-arm factorial (O0-O3). Both O1 and O2 passed A1-A8
    acceptability gates. O1 was preferred by the predeclared evidence hierarchy:
    HARD generalization = CONTRADICTED; O2 retains sigma=0.03 noise limitation;
    parsimony favors the simpler candidate. This is NOT because O1 was the only passer.

Constraints:
    Only two arms: O0 (current production) and O1 (selected candidate).
    No HARD, no sigma tuning, no diagnostic CRN, no RL, no PDF update,
    no production-default change, no post-hoc third block.
"""
import copy
import csv
import hashlib
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
from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED, respond_population
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
DELTA_D2R    = 1e-5   # frozen before run

BLOCK_A_SEEDS = list(range(501, 531))   # 30 seeds
BLOCK_B_SEEDS = list(range(531, 561))   # 30 seeds
ALL_SEEDS     = BLOCK_A_SEEDS + BLOCK_B_SEEDS
SMOKE_SEEDS   = [561, 562]
N_SMOKE       = 3

BOOTSTRAP_SEED_A      = 44001
BOOTSTRAP_SEED_B      = 44002
BOOTSTRAP_SEED_POOLED = 44003
N_BOOTSTRAP_BLOCK     = 20_000
N_BOOTSTRAP_POOLED    = 30_000

RESULTS_DIR = os.path.join("results", "D2R_selected_o1_confirmation")

# ============================================================
# Exactly two arms — no HARD
# ============================================================
ARMS = {
    "O0": {"sigma_predict": 0.03,  "mode": PREDICT_SURVIVAL_UNRESTRICTED, "label": "O0 (sigma=0.03, UNRESTRICTED)"},
    "O1": {"sigma_predict": 0.005, "mode": PREDICT_SURVIVAL_UNRESTRICTED, "label": "O1 (sigma=0.005, UNRESTRICTED)"},
}

# ============================================================
# Authoritative scenario factories
# ============================================================
SCENARIOS = {
    "PERSISTENT": ["persistent"] * 10,
    "STATIONARY": ["stationary"] * 10,
    "MIXED":      make_region_dynamics_map(10),
}

def _verify_mixed_map():
    m = make_region_dynamics_map(10)
    assert m[:5]  == ["persistent"] * 5
    assert m[5:]  == ["stationary"] * 5

_verify_mixed_map()

# ============================================================
# Helpers
# ============================================================

def _fp(arr):
    if arr is None or len(arr) == 0:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    return hashlib.md5(a.tobytes()).hexdigest()[:12]

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
        "manifest_type": "D2R_PRERUN",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "files": {
            "src/prdrl/oracle/oracle.py":                                    _hash("src/prdrl/oracle/oracle.py"),
            "src/prdrl/core/survival.py":                                    _hash("src/prdrl/core/survival.py"),
            "src/prdrl/experiments/E2P_predict_mechanism.py":               _hash("src/prdrl/experiments/E2P_predict_mechanism.py"),
            "src/prdrl/problems/controlled_predict.py":                      _hash("src/prdrl/problems/controlled_predict.py"),
            "results/D2_rng_engineering_fix/D2RF_gate_results.json":         _hash("results/D2_rng_engineering_fix/D2RF_gate_results.json"),
            "results/D1R_postfix_factorial_design_reopen/D1R_verdict.json":  _hash("results/D1R_postfix_factorial_design_reopen/D1R_verdict.json"),
            "results/D1R_postfix_factorial_design_reopen/D1R_option_assessment.json": _hash("results/D1R_postfix_factorial_design_reopen/D1R_option_assessment.json"),
        },
        "arms": {k: {"sigma_predict": v["sigma_predict"], "mode": str(v["mode"])}
                 for k, v in ARMS.items()},
        "d1r_decision": "PROPOSE_O1_F005_ONLY",
        "d1r_selection_note": (
            "Both O1 and O2 passed A1-A8 in D1-R. O1 selected by hierarchy: "
            "HARD_GENERALIZATION=CONTRADICTED; O2 retains sigma=0.03 limitation; parsimony."
        ),
        "seed_blocks": {
            "block_A": {"first": 501, "last": 530, "n": 30},
            "block_B": {"first": 531, "last": 560, "n": 30},
            "smoke":   {"seeds": SMOKE_SEEDS},
        },
        "scenarios": list(SCENARIOS.keys()),
        "bootstrap": {
            "block_n": 20_000, "pooled_n": 30_000,
            "seed_A": BOOTSTRAP_SEED_A, "seed_B": BOOTSTRAP_SEED_B, "seed_pooled": BOOTSTRAP_SEED_POOLED,
        },
        "delta_d2r": DELTA_D2R,
        "constraints": {
            "no_hard": True, "no_sigma_tuning": True, "no_rl": True,
            "no_diagnostic_crn": True, "no_pdf_update": True,
            "no_production_default_change": True, "no_post_hoc_third_block": True,
        },
    }
    path = os.path.join(RESULTS_DIR, "D2R_pre_run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Pre-run manifest → {path}", flush=True)
    return manifest

# ============================================================
# Single-seed runner
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

    F_m = cp_evaluate(X_pop, t0, region_dynamics)
    gen += GENERATIONS_PER_CHANGE
    t1 = compute_time(gen)
    F_p = cp_evaluate(X_pop, t1, region_dynamics)
    F_nm, _ = normalize(F_m, F_p)
    rids_bi = np.asarray(partition(np.asarray(F_nm), V), dtype=int)
    history_t_shared += 1
    history_shared.update(t=history_t_shared, X=X_pop, F=F_m, region_ids=rids_bi, xl=xl, xu=xu)
    X_pop = evolve_nsga2(X_pop, t1, region_dynamics, GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu)
    t_float_current = t1

    # Clone arm states
    arm_states = {
        arm: {
            "X": X_pop.copy(),
            "history": copy.deepcopy(history_shared),
            "history_t": history_t_shared,
            "n_predict": 0, "n_local_fallback": 0, "n_common_keep": 0,
        }
        for arm in ARMS
    }

    rows = []
    cumulative_fe = {arm: 0 for arm in ARMS}

    for k in range(n_scored):
        scored_change_idx = k + 2
        tgt_r   = target_region(seed, k, R)
        tgt_reg = target_regime(region_dynamics, tgt_r)
        gen    += GENERATIONS_PER_CHANGE
        t_new   = compute_time(gen)
        evo_seed = seed * 10000 + scored_change_idx

        arm_eval = {}
        for arm_name, arm in arm_states.items():
            F_minus = cp_evaluate(arm["X"], t_float_current, region_dynamics)
            F_plus  = cp_evaluate(arm["X"], t_new, region_dynamics)
            F_nm2, _ = normalize(F_minus, F_plus)
            rids    = np.asarray(partition(np.asarray(F_nm2), V), dtype=int)
            n_r     = int(np.sum(rids == tgt_r))
            igd_pre = float(igd(F_plus, pf_star))
            arm["history_t"] += 1
            arm["history"].update(t=arm["history_t"], X=arm["X"], F=F_minus,
                                   region_ids=rids, xl=xl, xu=xu)
            arm_eval[arm_name] = {"F_plus": F_plus, "rids": rids, "n_r": n_r, "igd_pre": igd_pre}

        any_empty = any(ev["n_r"] == 0 for ev in arm_eval.values())
        if any_empty:
            for arm_name, arm in arm_states.items():
                ev = arm_eval[arm_name]
                arm["X"] = evolve_nsga2(arm["X"], t_new, region_dynamics,
                                         GENERATIONS_PER_CHANGE, evo_seed, xl, xu)
                F_ep = cp_evaluate(arm["X"], t_new, region_dynamics)
                igd_ep = float(igd(F_ep, pf_star))
                cumulative_fe[arm_name] += GENERATIONS_PER_CHANGE * POP_SIZE
                arm["n_common_keep"] += 1
                rows.append(_make_row(
                    scenario_name, seed, block_label, arm_name, k, scored_change_idx,
                    tgt_r, tgt_reg, ARMS[arm_name]["sigma_predict"],
                    False, ev["n_r"], "KEEP", "KEEP", 0.0,
                    ev["igd_pre"], ev["igd_pre"], igd_ep, 0.0, ev["igd_pre"] - igd_ep,
                    0, GENERATIONS_PER_CHANGE * POP_SIZE, cumulative_fe[arm_name],
                    _fp(arm["X"]),
                ))
            t_float_current = t_new
            continue

        arm_fe_this = {}
        for arm_name, arm in arm_states.items():
            arm_def = ARMS[arm_name]
            ev = arm_eval[arm_name]
            X_old = arm["X"].copy()

            motion = get_predict_motion_from_history(
                arm["history"], tgt_r, t=arm["history_t"], lambda_pred=LAMBDA_PRED)
            has_predict_history = motion is not None
            delta_hat_norm = float(np.linalg.norm(motion["delta_hat"])) if motion is not None else 0.0

            oracle_base_seed = seed * 100000 + k * 100 + tgt_r

            result = respond_population(
                X=X_old,
                F_plus=ev["F_plus"],
                region_ids=ev["rids"],
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

            if realized_op == "PREDICT":
                arm["n_predict"] += 1
            else:
                arm["n_local_fallback"] += 1

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
                has_predict_history, ev["n_r"], "PREDICT", realized_op, delta_hat_norm,
                ev["igd_pre"], igd_resp, igd_ep,
                ev["igd_pre"] - igd_resp, ev["igd_pre"] - igd_ep,
                extra_fe, GENERATIONS_PER_CHANGE * POP_SIZE, cumulative_fe[arm_name],
                _fp(arm["X"]),
            ))

        fe_set = set(arm_fe_this.values())
        if len(fe_set) > 1 and stop_on_violation:
            raise RuntimeError(
                f"FE FAIRNESS VIOLATED: scenario={scenario_name} seed={seed} k={k} fe={arm_fe_this}")

        t_float_current = t_new

    return rows


def _make_row(
    scenario, seed, block, arm, k, change_idx,
    target_region_r, target_regime_s, sigma_used,
    has_predict_history, n_r_before,
    requested_action, realized_operator, delta_hat_norm,
    igd_pre, igd_response, igd_epoch_end, response_gain, epoch_gain,
    resp_fe, evo_fe, cumulative_fe, epoch_x_fp,
):
    return {
        "scenario": scenario, "seed": seed, "block": block, "arm": arm,
        "k": k, "change_idx": change_idx,
        "target_region": target_region_r, "target_regime": target_regime_s,
        "sigma_used": sigma_used,
        "has_predict_history": has_predict_history,
        "n_r_before": n_r_before,
        "requested_action": requested_action, "realized_operator": realized_operator,
        "delta_hat_norm": delta_hat_norm,
        "clipping_fraction": None, "unique_child_fraction": None, "child_survival_rate": None,
        "igd_pre": igd_pre, "igd_response": igd_response, "igd_epoch_end": igd_epoch_end,
        "response_gain": response_gain, "epoch_gain": epoch_gain,
        "response_fe": resp_fe, "evolution_fe": evo_fe, "cumulative_fe": cumulative_fe,
        "epoch_x_fp": epoch_x_fp,
    }

# ============================================================
# Reproducibility checks
# ============================================================

def run_smoke_reproducibility():
    print("  Smoke reproducibility …", flush=True)
    pf_star = get_true_pf(500)
    failures = []
    for seed in SMOKE_SEEDS:
        for scen, rd in SCENARIOS.items():
            rows1 = run_one_seed(seed, N_SMOKE, scen, rd, pf_star, "SMOKE")
            rows2 = run_one_seed(seed, N_SMOKE, scen, rd, pf_star, "SMOKE")
            for r1, r2 in zip(rows1, rows2):
                if r1["igd_response"] != r2["igd_response"] or r1["epoch_x_fp"] != r2["epoch_x_fp"]:
                    failures.append({"seed": seed, "scen": scen, "arm": r1["arm"], "k": r1["k"]})
        print(f"    smoke seed {seed} done", flush=True)
    passed = len(failures) == 0
    result = {"SAME_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL",
              "n_failures": len(failures), "failures": failures[:5]}
    print(f"    SAME_PROCESS_REPRODUCIBILITY = {result['SAME_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result


def run_cross_process_check():
    print("  Cross-process check …", flush=True)
    seed = 563
    script_tpl = (
        "import sys; sys.path.insert(0,'src'); import hashlib; "
        "from prdrl.experiments.D2R_selected_o1_confirmation import run_one_seed, SCENARIOS; "
        "from prdrl.problems.controlled_predict import get_true_pf; "
        "pf = get_true_pf(500); "
        "rows = run_one_seed({seed}, 3, '{scen}', SCENARIOS['{scen}'], pf, 'X'); "
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
            failures.append({"scen": scen, "error": f"mismatch: {hashes}"})
    passed = len(failures) == 0
    result = {"CROSS_PROCESS_REPRODUCIBILITY": "PASS" if passed else "FAIL",
              "n_failures": len(failures)}
    print(f"    CROSS_PROCESS_REPRODUCIBILITY = {result['CROSS_PROCESS_REPRODUCIBILITY']}", flush=True)
    return result

# ============================================================
# Full science run
# ============================================================

def run_full_science():
    pf_star = get_true_pf(500)
    all_rows, fe_fair_ok, action_ok = [], True, True
    for scen, rd in SCENARIOS.items():
        print(f"  Scenario {scen} …", flush=True)
        for seed in ALL_SEEDS:
            block = "A" if seed in BLOCK_A_SEEDS else "B"
            try:
                rows = run_one_seed(seed, N_SCORED, scen, rd, pf_star, block, stop_on_violation=True)
                all_rows.extend(rows)
            except RuntimeError as exc:
                if "FE FAIRNESS" in str(exc):
                    fe_fair_ok = False
                raise
            sys.stdout.write("."); sys.stdout.flush()
        print(flush=True)

    # No HARD arms — action_ok trivially true
    # Verify no HARD triggered in any row
    for r in all_rows:
        if r.get("hard_triggered"):
            action_ok = False
            break

    return all_rows, fe_fair_ok, action_ok

# ============================================================
# Statistics
# ============================================================

def _bootstrap_ci(values, n_resamples, rng_seed):
    n = len(values)
    rng = np.random.default_rng(rng_seed)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def classify_effect(mean_cb, ci_low, ci_high):
    if mean_cb > DELTA_D2R and ci_low > 0:     return "BENEFICIAL"
    if mean_cb > DELTA_D2R:                     return "DIRECTIONAL_POSITIVE"
    if mean_cb < -DELTA_D2R and ci_high < 0:   return "HARMFUL"
    if mean_cb < -DELTA_D2R:                   return "DIRECTIONAL_NEGATIVE"
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
        n_predict = sum(1 for r in seed_rows if r["realized_operator"] == "PREDICT")
        n_local   = sum(1 for r in seed_rows if r["realized_operator"] == "LOCAL_FALLBACK")
        n_keep    = sum(1 for r in seed_rows if r["realized_operator"] == "KEEP")
        summaries.append({
            "scenario": scenario_name, "seed": seed, "block": block, "arm": arm,
            "MIGD_response":      float(np.mean(igd_r)),
            "MIGD_epoch":         float(np.mean(igd_e)),
            "early_MIGD_response": float(np.mean(igd_r[:5])),
            "late_MIGD_response":  float(np.mean(igd_r[5:])),
            "early_MIGD_epoch":    float(np.mean(igd_e[:5])),
            "late_MIGD_epoch":     float(np.mean(igd_e[5:])),
            "final_igd": igd_e[-1],
            "worst_response_igd": float(max(igd_r)),
            "worst_epoch_igd":    float(max(igd_e)),
            "n_predict": n_predict, "n_local_fallback": n_local, "n_common_keep": n_keep,
        })
    return summaries


def compute_cb_per_seed(seed_summaries, scenario_name):
    from collections import defaultdict
    by_seed = defaultdict(dict)
    for s in seed_summaries:
        if s["scenario"] == scenario_name:
            by_seed[s["seed"]][s["arm"]] = s
    cb_list = []
    for seed, arms in sorted(by_seed.items()):
        if "O0" not in arms or "O1" not in arms:
            continue
        block = arms["O0"]["block"]
        cb_list.append({
            "scenario": scenario_name, "seed": seed, "block": block,
            "O1Benefit_response": arms["O0"]["MIGD_response"] - arms["O1"]["MIGD_response"],
            "O1Benefit_epoch":    arms["O0"]["MIGD_epoch"]    - arms["O1"]["MIGD_epoch"],
            "early_O1Benefit_response": arms["O0"]["early_MIGD_response"] - arms["O1"]["early_MIGD_response"],
            "late_O1Benefit_response":  arms["O0"]["late_MIGD_response"]  - arms["O1"]["late_MIGD_response"],
            "early_O1Benefit_epoch":    arms["O0"]["early_MIGD_epoch"]    - arms["O1"]["early_MIGD_epoch"],
            "late_O1Benefit_epoch":     arms["O0"]["late_MIGD_epoch"]     - arms["O1"]["late_MIGD_epoch"],
        })
    return cb_list


def _block_stats(cb_list, block_label, n_bootstrap, rng_seed):
    vals_r = np.array([c["O1Benefit_response"] for c in cb_list if c["block"] == block_label])
    vals_e = np.array([c["O1Benefit_epoch"]    for c in cb_list if c["block"] == block_label])
    if len(vals_r) == 0:
        return None
    m_r = float(np.mean(vals_r)); ci_lo_r, ci_hi_r = _bootstrap_ci(vals_r, n_bootstrap, rng_seed)
    m_e = float(np.mean(vals_e)); ci_lo_e, ci_hi_e = _bootstrap_ci(vals_e, n_bootstrap, rng_seed + 1)
    cls_r = classify_effect(m_r, ci_lo_r, ci_hi_r)
    cls_e = classify_effect(m_e, ci_lo_e, ci_hi_e)
    er = float(np.mean([c["early_O1Benefit_response"] for c in cb_list if c["block"] == block_label]))
    lr = float(np.mean([c["late_O1Benefit_response"]  for c in cb_list if c["block"] == block_label]))
    return {
        "block": block_label, "n": len(vals_r),
        "mean_O1Benefit_response": m_r, "CI_response": [ci_lo_r, ci_hi_r],
        "class_response": cls_r,
        "mean_O1Benefit_epoch": m_e, "CI_epoch": [ci_lo_e, ci_hi_e],
        "class_epoch": cls_e,
        "early_O1Benefit_response": er, "late_O1Benefit_response": lr,
    }


def _pooled_stats(cb_list, n_bootstrap, rng_seed):
    vals_r = np.array([c["O1Benefit_response"] for c in cb_list])
    vals_e = np.array([c["O1Benefit_epoch"]    for c in cb_list])
    if len(vals_r) == 0:
        return {}
    m_r = float(np.mean(vals_r)); ci_lo_r, ci_hi_r = _bootstrap_ci(vals_r, n_bootstrap, rng_seed)
    m_e = float(np.mean(vals_e)); ci_lo_e, ci_hi_e = _bootstrap_ci(vals_e, n_bootstrap, rng_seed + 1)
    cls_r = classify_effect(m_r, ci_lo_r, ci_hi_r)
    cls_e = classify_effect(m_e, ci_lo_e, ci_hi_e)
    ba = _block_stats(cb_list, "A", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_A)
    bb = _block_stats(cb_list, "B", N_BOOTSTRAP_BLOCK, BOOTSTRAP_SEED_B)
    return {
        "pooled": {
            "n": len(vals_r), "mean_O1Benefit_response": m_r,
            "CI_response": [ci_lo_r, ci_hi_r], "class_response": cls_r,
            "mean_O1Benefit_epoch": m_e, "CI_epoch": [ci_lo_e, ci_hi_e], "class_epoch": cls_e,
        },
        "block_A": ba, "block_B": bb,
        "between_block_response": (ba["mean_O1Benefit_response"] - bb["mean_O1Benefit_response"]
                                   if ba and bb else None),
    }

# ============================================================
# Mixed regime analysis
# ============================================================

def compute_mixed_regime_analysis(rows, seed_summaries):
    from collections import defaultdict
    mixed_rows = [r for r in rows if r["scenario"] == "MIXED"]
    by_seed_arm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in mixed_rows:
        by_seed_arm[r["seed"]][r["arm"]][r["target_regime"]].append(r["igd_response"])

    regime_cb = {"persistent": [], "stationary": []}
    for seed in ALL_SEEDS:
        block = "A" if seed in BLOCK_A_SEEDS else "B"
        for regime in ("persistent", "stationary"):
            o0 = by_seed_arm[seed]["O0"].get(regime, [])
            o1 = by_seed_arm[seed]["O1"].get(regime, [])
            if o0 and o1:
                regime_cb[regime].append({
                    "seed": seed, "block": block,
                    "O1Benefit_response": float(np.mean(o0)) - float(np.mean(o1)),
                })

    out = {}
    for regime, cb_list in regime_cb.items():
        if not cb_list:
            out[regime] = {"n": 0}
            continue
        vals = np.array([c["O1Benefit_response"] for c in cb_list])
        m = float(np.mean(vals))
        ci_lo, ci_hi = _bootstrap_ci(vals, 10_000, BOOTSTRAP_SEED_POOLED + (10 if regime == "persistent" else 20))
        cls = classify_effect(m, ci_lo, ci_hi)
        out[regime] = {
            "n": len(cb_list), "mean_O1Benefit_response": m,
            "CI": [ci_lo, ci_hi], "class_response": cls,
            "safety": "PASS" if cls != "HARMFUL" else "FAIL",
        }
    return out

# ============================================================
# Verdict logic
# ============================================================

def compute_persistent_verdict(stats):
    p = stats.get("pooled", {})
    bA = stats.get("block_A", {}) or {}
    bB = stats.get("block_B", {}) or {}
    m = p.get("mean_O1Benefit_response", 0.0)
    ci = p.get("CI_response", [0, 0])
    cls = p.get("class_response", "?")
    bA_m = bA.get("mean_O1Benefit_response", 0.0)
    bB_m = bB.get("mean_O1Benefit_response", 0.0)
    bA_cls = bA.get("class_response", "?")
    bB_cls = bB.get("class_response", "?")

    P1 = m > DELTA_D2R
    P2 = ci[0] > 0
    P3 = bA_m > 0
    P4 = bB_m > 0
    P5 = bA_cls != "HARMFUL" and bB_cls != "HARMFUL"
    P6 = p.get("class_epoch", "?") != "HARMFUL"

    if cls == "HARMFUL" or not P5:
        return "FAILED"
    if P1 and P2 and P3 and P4 and P5 and P6:
        return "VALIDATED"
    if P1 and P5 and P6:
        return "PARTIAL"
    return "PARTIAL"


def compute_stationary_verdict(stats):
    p   = stats.get("pooled", {})
    bA  = stats.get("block_A", {}) or {}
    bB  = stats.get("block_B", {}) or {}
    S1 = p.get("class_response", "") != "HARMFUL"
    S2 = p.get("class_epoch", "") != "HARMFUL"
    S3 = (bA.get("class_response", "") if bA else "PRACTICALLY_NEUTRAL") != "HARMFUL"
    S4 = (bB.get("class_response", "") if bB else "PRACTICALLY_NEUTRAL") != "HARMFUL"
    return "SAFE" if (S1 and S2 and S3 and S4) else "UNSAFE"


def compute_mixed_block_stability(stats):
    bA  = stats.get("block_A", {}) or {}
    bB  = stats.get("block_B", {}) or {}
    p   = stats.get("pooled", {})
    bA_m = bA.get("mean_O1Benefit_response", 0.0)
    bB_m = bB.get("mean_O1Benefit_response", 0.0)
    bA_c = bA.get("class_response", "PRACTICALLY_NEUTRAL")
    bB_c = bB.get("class_response", "PRACTICALLY_NEUTRAL")
    p_c  = p.get("class_response", "PRACTICALLY_NEUTRAL")

    if p_c == "HARMFUL" or (bA_c == "HARMFUL" and bB_c == "HARMFUL"):
        return "STABLE_HARM"
    if bA_m > 0 and bB_m > 0 and bA_c != "HARMFUL" and bB_c != "HARMFUL":
        return "STABLE_POSITIVE"
    if abs(bA_m) > DELTA_D2R and abs(bB_m) > DELTA_D2R and (bA_m > 0) != (bB_m > 0):
        return "UNSTABLE_SIGN"
    if bA_c != "HARMFUL" and bB_c != "HARMFUL":
        return "STABLE_NONHARM"
    return "STABLE_HARM"


def compute_mixed_verdict(stats, regime_stats, fe_fair_ok, repro_pass):
    p   = stats.get("pooled", {})
    bA  = stats.get("block_A", {}) or {}
    bB  = stats.get("block_B", {}) or {}
    p_c  = p.get("class_response", "PRACTICALLY_NEUTRAL")
    p_ec = p.get("class_epoch",    "PRACTICALLY_NEUTRAL")
    bA_c = bA.get("class_response", "PRACTICALLY_NEUTRAL")
    bB_c = bB.get("class_response", "PRACTICALLY_NEUTRAL")

    M1 = p_c != "HARMFUL"
    M2 = p_ec != "HARMFUL"
    M3 = bA_c != "HARMFUL"
    M4 = bB_c != "HARMFUL"
    M6 = fe_fair_ok and repro_pass

    regime_p_safe = regime_stats.get("persistent", {}).get("safety", "PASS") == "PASS"
    regime_s_safe = regime_stats.get("stationary", {}).get("safety", "PASS") == "PASS"

    if not M6:
        return "FAILED"
    if not (M1 and M2 and M3 and M4):
        return "FAILED"
    if M1 and M2 and M3 and M4:
        return "STABLE"
    return "PARTIAL"


def compute_overall_verdict(pers_v, stat_v, mixed_v, mixed_stab, p_target_safe, s_target_safe,
                             fe_fair_ok, repro_pass):
    C6 = repro_pass
    C7 = repro_pass
    C8 = fe_fair_ok

    if not (C6 and C7 and C8):
        return "FAILED"
    if pers_v == "FAILED" or stat_v == "UNSAFE" or mixed_v == "FAILED":
        return "FAILED"
    if not p_target_safe:
        return "FAILED"
    if pers_v == "VALIDATED" and stat_v == "SAFE" and mixed_v == "STABLE" and p_target_safe and s_target_safe:
        return "VALIDATED"
    if pers_v in ("VALIDATED", "PARTIAL") and stat_v == "SAFE" and mixed_v in ("STABLE", "PARTIAL"):
        return "PARTIAL"
    return "PARTIAL"

# ============================================================
# Output writers
# ============================================================

def write_trajectories(rows):
    path = os.path.join(RESULTS_DIR, "D2R_trajectories.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path

def write_seed_summary(summaries):
    path = os.path.join(RESULTS_DIR, "D2R_seed_summary.csv")
    if not summaries:
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    return path

def write_block_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        for blk_key, blk_label in [("block_A","A"), ("block_B","B"), ("pooled","pooled")]:
            bstats = stats.get(blk_key, {}) or {}
            rows.append({
                "scenario": scen, "block": blk_label,
                "mean_O1Benefit_response": bstats.get("mean_O1Benefit_response"),
                "ci_low":  (bstats.get("CI_response") or [None,None])[0],
                "ci_high": (bstats.get("CI_response") or [None,None])[1],
                "class_response": bstats.get("class_response"),
                "mean_O1Benefit_epoch": bstats.get("mean_O1Benefit_epoch"),
                "class_epoch": bstats.get("class_epoch"),
                "early_O1Benefit_response": bstats.get("early_O1Benefit_response"),
                "late_O1Benefit_response":  bstats.get("late_O1Benefit_response"),
            })
    path = os.path.join(RESULTS_DIR, "D2R_block_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_scenario_summary(all_scen_stats):
    rows = []
    for scen, stats in all_scen_stats.items():
        p = stats.get("pooled", {})
        rows.append({
            "scenario": scen,
            "mean_O1Benefit_response": p.get("mean_O1Benefit_response"),
            "ci_low":  (p.get("CI_response") or [None,None])[0],
            "ci_high": (p.get("CI_response") or [None,None])[1],
            "class_response": p.get("class_response"),
            "mean_O1Benefit_epoch": p.get("mean_O1Benefit_epoch"),
            "class_epoch": p.get("class_epoch"),
        })
    path = os.path.join(RESULTS_DIR, "D2R_scenario_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_mixed_regime_summary(regime_stats):
    rows = []
    for regime, rstats in regime_stats.items():
        rows.append({
            "regime": regime, "n": rstats.get("n"),
            "mean_O1Benefit_response": rstats.get("mean_O1Benefit_response"),
            "ci_low":  (rstats.get("CI") or [None,None])[0],
            "ci_high": (rstats.get("CI") or [None,None])[1],
            "class_response": rstats.get("class_response"),
            "safety": rstats.get("safety"),
        })
    path = os.path.join(RESULTS_DIR, "D2R_mixed_regime_summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return path

def write_reproducibility(smoke_result, cross_result):
    out = {
        "SAME_PROCESS_REPRODUCIBILITY": smoke_result.get("SAME_PROCESS_REPRODUCIBILITY"),
        "CROSS_PROCESS_REPRODUCIBILITY": cross_result.get("CROSS_PROCESS_REPRODUCIBILITY"),
        "smoke_detail": smoke_result, "cross_detail": cross_result,
    }
    path = os.path.join(RESULTS_DIR, "D2R_reproducibility.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path

def write_verdict(verdict_dict):
    path = os.path.join(RESULTS_DIR, "D2R_verdict.json")
    with open(path, "w") as f:
        json.dump(verdict_dict, f, indent=2, default=str)
    return path

# ============================================================
# Report
# ============================================================

def write_report(verdict_dict, all_scen_stats, regime_stats, elapsed):
    lines = []
    def h(t): lines.append(f"\n## {t}\n")
    def p(t=""): lines.append(t)

    lines.append("# D2-R Selected O1 Design Confirmation Report")
    p(f"**Date**: 2026-09-20 | **Branch**: pr-drl-rewrite | **Elapsed**: {elapsed:.1f}s")
    p(f"**Seeds**: 501..560 (Block A: 501..530, Block B: 531..560)")
    p(f"**Arms**: O0 (sigma=0.03, UNRESTRICTED) vs O1 (sigma=0.005, UNRESTRICTED)")
    p()

    h("A. Purpose")
    p("Fresh independent confirmation of selected O1 design on 60 seeds under the fixed "
      "deterministic post-D2-RF production runtime.")

    h("B. D1-R Selection Context")
    p("D1-R ran a 4-arm factorial (O0-O3) on seeds 401-440. Both O1 and O2 passed all A1-A8 "
      "acceptability gates. O1 was selected by the predeclared evidence hierarchy:")
    for item in ["FACTOR_HARD_GENERALIZATION = CONTRADICTED (O2 disadvantage)",
                 "O2 retains sigma=0.03 known noise limitation (N1/R1-N)",
                 "O1 is the simpler acceptable design (parsimony)"]:
        p(f"- {item}")
    p()
    p("**O1 was NOT the only option that passed A1-A8. O2 also passed.**")

    h("C. Traceability Correction")
    p("D1-R D1R_option_assessment.json records: O1 ACCEPTABLE = true, O2 ACCEPTABLE = true, "
      "O3 ACCEPTABLE = false. Any prose claiming O1 was the only A1-A8 passer is incorrect.")

    h("D. Selected Design")
    p("O1: sigma_P=0.005, PREDICT survival = UNRESTRICTED NSGA-II.")
    p("The only algorithmic difference from O0 (current production) is sigma_P: 0.03 → 0.005.")
    p("No HARD, no additional response FE, no new scenario, no RL.")

    h("E. Post-D2-RF Production Runtime")
    p("Child RNG: PREDICT namespace (make_rng(seed×100000+k×100+r, t, r, 'PREDICT'))")
    p("Survival RNG: RESPONSE_SURVIVAL namespace (D2-RF fix)")
    p("Evolution: seed×10000+scored_change_idx | No L2B_RESPONSE CRN.")

    h("F. Pre-Run Freeze")
    p("Manifest created before science run. Oracle/survival hashed. D1-R verdict and "
      "option assessment hashed. Protocol frozen.")

    h("G. Seed Blocks")
    p("Block A: 501..530 (n=30) | Block B: 531..560 (n=30) | Smoke: 561, 562")
    p("All seeds > 442: no overlap with prior experiments.")

    h("H. Reproducibility Gates")
    p(f"SAME_PROCESS_REPRODUCIBILITY = **{verdict_dict.get('SAME_PROCESS_REPRODUCIBILITY')}**")
    p(f"CROSS_PROCESS_REPRODUCIBILITY = **{verdict_dict.get('CROSS_PROCESS_REPRODUCIBILITY')}**")

    h("I. FE / Action Semantics")
    p(f"FE_FAIRNESS = **{verdict_dict.get('FE_FAIRNESS')}**")
    p(f"ACTION_SEMANTICS = **{verdict_dict.get('ACTION_SEMANTICS')}**")
    p("No HARD: neither O0 nor O1 uses HARD_RESPONSE_ONLY.")

    def _scen_block(section_label, scen, blk_key):
        h(section_label)
        stats = all_scen_stats.get(scen, {})
        bstats = stats.get(blk_key, {}) or {}
        if bstats:
            m = bstats.get("mean_O1Benefit_response", 0.0)
            ci = bstats.get("CI_response", [None, None])
            cls = bstats.get("class_response", "?")
            me = bstats.get("mean_O1Benefit_epoch", 0.0)
            clse = bstats.get("class_epoch", "?")
            ci_str = f"[{ci[0]:.3e}, {ci[1]:.3e}]" if ci[0] is not None else "N/A"
            p(f"O1Benefit_response: **{m:.3e}** {ci_str} → **{cls}**")
            p(f"O1Benefit_epoch:    {me:.3e} → {clse}")
            if blk_key != "pooled":
                er = bstats.get("early_O1Benefit_response", 0.0)
                lr = bstats.get("late_O1Benefit_response",  0.0)
                p(f"early={er:.3e} late={lr:.3e}")
        else:
            p("No data.")

    _scen_block("J. Persistent Block A", "PERSISTENT", "block_A")
    _scen_block("K. Persistent Block B", "PERSISTENT", "block_B")
    _scen_block("L. Persistent Pooled", "PERSISTENT", "pooled")
    _scen_block("M. Stationary Block A", "STATIONARY", "block_A")
    _scen_block("N. Stationary Block B", "STATIONARY", "block_B")
    _scen_block("O. Stationary Pooled Safety", "STATIONARY", "pooled")
    _scen_block("P. Mixed Block A", "MIXED", "block_A")
    _scen_block("Q. Mixed Block B", "MIXED", "block_B")
    _scen_block("R. Mixed Pooled Stability", "MIXED", "pooled")

    h("S. Mixed Persistent-Target Safety")
    rp = regime_stats.get("persistent", {})
    p(f"n={rp.get('n',0)}, mean={rp.get('mean_O1Benefit_response',0.0):.3e}, "
      f"CI={rp.get('CI', ['?','?'])}, class={rp.get('class_response','?')}")
    p(f"**O1_MIXED_P_TARGET_SAFETY = {verdict_dict.get('O1_MIXED_P_TARGET_SAFETY')}**")
    p("(Historical L2-B warning: F005 alone showed HARMFUL on persistent-target MIXED in diagnostic runs. "
      f"D2-R authentic production-path result: {rp.get('class_response','?')}.)")

    h("T. Mixed Stationary-Target Safety")
    rs = regime_stats.get("stationary", {})
    p(f"n={rs.get('n',0)}, mean={rs.get('mean_O1Benefit_response',0.0):.3e}, "
      f"class={rs.get('class_response','?')}")
    p(f"**O1_MIXED_S_TARGET_SAFETY = {verdict_dict.get('O1_MIXED_S_TARGET_SAFETY')}**")

    h("U. Early / Late")
    for scen in SCENARIOS:
        stats = all_scen_stats.get(scen, {})
        for blk in ("block_A", "block_B"):
            bstats = stats.get(blk, {}) or {}
            er = bstats.get("early_O1Benefit_response", "N/A")
            lr = bstats.get("late_O1Benefit_response",  "N/A")
            er_s = f"{er:.3e}" if isinstance(er, float) else str(er)
            lr_s = f"{lr:.3e}" if isinstance(lr, float) else str(lr)
            p(f"**{scen} {blk}**: early={er_s} late={lr_s}")

    h("V. Between-Block Heterogeneity")
    for scen, stats in all_scen_stats.items():
        diff = stats.get("between_block_response")
        if diff is not None:
            p(f"**{scen}**: Block A mean - Block B mean = {diff:.3e}")

    h("W. Historical Evidence Reconciliation")
    p("| Stage | Evidence | Type |")
    p("|-------|----------|------|")
    for row in [
        ("N1/R1-N", "sigma=0.03 high child-noise limitation", "Diagnostic"),
        ("L2-B", "F005/O1 persistent-target MIXED HARMFUL in diagnostic CRN runs", "Diagnostic"),
        ("D2-CZ", "O3 failed STATIONARY/MIXED; O1 not tested directly", "Post-fix production-path"),
        ("D1-R", "O1 and O2 both pass A1-A8; O1 selected by hierarchy/parsimony", "Post-fix 4-arm factorial"),
        ("D2-R", "Fresh confirmation of O1 on 501-560", "Post-fix selected-design"),
    ]:
        p(f"| {row[0]} | {row[1]} | {row[2]} |")
    p()
    p("L2-B used diagnostic CRN which is NOT the authentic production runtime. "
      "D2-R uses the actual production stochastic protocol on fresh seeds.")

    h("X. Selected-Design Verdict")
    for key in ["O1_PERSISTENT_CONFIRMATION", "O1_STATIONARY_CONFIRMATION",
                 "O1_MIXED_CONFIRMATION", "O1_MIXED_P_TARGET_SAFETY", "O1_MIXED_S_TARGET_SAFETY"]:
        p(f"**{key}** = {verdict_dict.get(key, '?')}")
    p()
    p(f"**O1_SELECTED_DESIGN_CONFIRMATION = {verdict_dict.get('O1_SELECTED_DESIGN_CONFIRMATION', '?')}**")

    h("Y. Production Recommendation")
    p(f"**D2R_PRODUCTION_RECOMMENDATION = {verdict_dict.get('D2R_PRODUCTION_RECOMMENDATION', '?')}**")
    rec = verdict_dict.get("D2R_PRODUCTION_RECOMMENDATION", "")
    if rec == "ADOPT_O1_SPEC":
        p("Sufficient evidence to proceed to production/PDF adoption stage (D2D).")
        p("This does NOT silently change production defaults — D2D does that explicitly.")
    elif rec == "FURTHER_REVIEW":
        p("Directional evidence but not full validation. Proceed to D2R_REVIEW.")
    else:
        p("O1 failed confirmation. Return to D1_DESIGN_REOPEN_AGAIN.")

    h("Z. What This Does Not Prove")
    for item in [
        "Only sigma 0.03 vs 0.005 compared — 0.005 is not claimed mathematically optimal.",
        "No adaptive sigma tested.",
        "No HARD in the proposed O1 specification.",
        "No RL.",
        "No PDF update (D2D does that).",
        "Only three scenario types (PERSISTENT/STATIONARY/MIXED) tested.",
        "One MIXED 5/5 map (make_region_dynamics_map(10)).",
        "Production default remains O0 throughout this experiment.",
        "Seeds 501-560 are frozen — no post-hoc third block.",
        "HARD code is not deleted; O1 simply does not use it.",
    ]:
        p(f"- {item}")

    h("AA. Next Stage")
    p(f"**NEXT_STAGE = {verdict_dict.get('NEXT_STAGE', '?')}**")

    h("AB. Production Status")
    p(f"PRODUCTION_DEFAULT = O0_CURRENT | PDF_STATUS = UNCHANGED | RL_STATUS = NOT_STARTED")

    h("AC. Output Files")
    for fname in ["D2R_pre_run_manifest.json", "D2R_trajectories.jsonl",
                  "D2R_seed_summary.csv", "D2R_block_summary.csv",
                  "D2R_scenario_summary.csv", "D2R_mixed_regime_summary.csv",
                  "D2R_reproducibility.json", "D2R_verdict.json", "D2R_report.md"]:
        p(f"- `{fname}`")

    path = os.path.join(RESULTS_DIR, "D2R_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

# ============================================================
# Terminal output
# ============================================================

def print_terminal_output(verdict_dict):
    print()
    print("=" * 60)
    print("D2R TERMINAL OUTPUT")
    print("=" * 60)
    print(f"D2R_SEEDS = 501..560")
    print(f"BLOCK_A = 501..530")
    print(f"BLOCK_B = 531..560")
    print()
    print(f"SAME_PROCESS_REPRODUCIBILITY = {verdict_dict.get('SAME_PROCESS_REPRODUCIBILITY')}")
    print(f"CROSS_PROCESS_REPRODUCIBILITY = {verdict_dict.get('CROSS_PROCESS_REPRODUCIBILITY')}")
    print(f"FE_FAIRNESS = {verdict_dict.get('FE_FAIRNESS')}")
    print(f"ACTION_SEMANTICS = {verdict_dict.get('ACTION_SEMANTICS')}")
    print()
    print(f"O1_PERSISTENT_CONFIRMATION = {verdict_dict.get('O1_PERSISTENT_CONFIRMATION')}")
    print(f"O1_STATIONARY_CONFIRMATION = {verdict_dict.get('O1_STATIONARY_CONFIRMATION')}")
    print()
    mixed_stats_bA = verdict_dict.get("mixed_block_A_class", "?")
    mixed_stats_bB = verdict_dict.get("mixed_block_B_class", "?")
    print(f"O1_MIXED_BLOCK_A = {mixed_stats_bA}")
    print(f"O1_MIXED_BLOCK_B = {mixed_stats_bB}")
    print(f"O1_MIXED_CONFIRMATION = {verdict_dict.get('O1_MIXED_CONFIRMATION')}")
    print()
    print(f"O1_MIXED_P_TARGET_SAFETY = {verdict_dict.get('O1_MIXED_P_TARGET_SAFETY')}")
    print(f"O1_MIXED_S_TARGET_SAFETY = {verdict_dict.get('O1_MIXED_S_TARGET_SAFETY')}")
    print()
    print(f"O1_SELECTED_DESIGN_CONFIRMATION = {verdict_dict.get('O1_SELECTED_DESIGN_CONFIRMATION')}")
    print(f"D2R_PRODUCTION_RECOMMENDATION = {verdict_dict.get('D2R_PRODUCTION_RECOMMENDATION')}")
    print()
    print(f"PRODUCTION_DEFAULT = O0_CURRENT")
    print(f"PDF_STATUS = UNCHANGED")
    print(f"RL_STATUS = NOT_STARTED")
    print()
    print(f"NEXT_STAGE = {verdict_dict.get('NEXT_STAGE')}")
    print("=" * 60)

# ============================================================
# Main
# ============================================================

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print("=== D2-R Selected O1 Design Confirmation ===", flush=True)

    print("Step 1: Pre-run manifest …", flush=True)
    create_prerun_manifest()

    print("Step 2: Smoke reproducibility …", flush=True)
    smoke_result = run_smoke_reproducibility()
    if smoke_result["SAME_PROCESS_REPRODUCIBILITY"] != "PASS":
        print("STOP: smoke reproducibility FAILED.", flush=True)
        sys.exit(1)

    print("Step 3: Cross-process check …", flush=True)
    cross_result = run_cross_process_check()

    repro_pass = (smoke_result["SAME_PROCESS_REPRODUCIBILITY"] == "PASS" and
                  cross_result["CROSS_PROCESS_REPRODUCIBILITY"] == "PASS")
    write_reproducibility(smoke_result, cross_result)

    print("Step 4: Full 60-seed run …", flush=True)
    all_rows, fe_fair_ok, action_ok = run_full_science()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total rows: {len(all_rows)}  Elapsed: {elapsed:.1f}s", flush=True)

    print("Step 5: Statistics …", flush=True)
    write_trajectories(all_rows)

    all_seed_summaries = []
    all_cb = {}
    all_scen_stats = {}
    for scen in SCENARIOS:
        summ = compute_seed_summaries(all_rows, scen)
        all_seed_summaries.extend(summ)
        cb = compute_cb_per_seed(summ, scen)
        all_cb[scen] = cb
        all_scen_stats[scen] = _pooled_stats(cb, N_BOOTSTRAP_POOLED, BOOTSTRAP_SEED_POOLED)

    write_seed_summary(all_seed_summaries)
    write_block_summary(all_scen_stats)
    write_scenario_summary(all_scen_stats)

    regime_stats = compute_mixed_regime_analysis(all_rows, all_seed_summaries)
    write_mixed_regime_summary(regime_stats)

    print("Step 6: Verdicts …", flush=True)
    pers_v  = compute_persistent_verdict(all_scen_stats.get("PERSISTENT", {}))
    stat_v  = compute_stationary_verdict(all_scen_stats.get("STATIONARY", {}))
    mixed_stab = compute_mixed_block_stability(all_scen_stats.get("MIXED", {}))
    mixed_v    = compute_mixed_verdict(all_scen_stats.get("MIXED", {}), regime_stats,
                                        fe_fair_ok, repro_pass)
    p_target_safe = regime_stats.get("persistent", {}).get("safety", "FAIL") == "PASS"
    s_target_safe = regime_stats.get("stationary", {}).get("safety", "FAIL") == "PASS"

    overall_v = compute_overall_verdict(pers_v, stat_v, mixed_v, mixed_stab,
                                         p_target_safe, s_target_safe, fe_fair_ok, repro_pass)

    if overall_v == "VALIDATED":
        rec, next_stage = "ADOPT_O1_SPEC", "D2D_O1_PRODUCTION_SPEC_UPDATE"
    elif overall_v == "PARTIAL":
        rec, next_stage = "FURTHER_REVIEW", "D2R_REVIEW"
    else:
        rec, next_stage = "RETAIN_O0", "D1_DESIGN_REOPEN_AGAIN"

    mixed_bA = all_scen_stats.get("MIXED", {}).get("block_A", {}) or {}
    mixed_bB = all_scen_stats.get("MIXED", {}).get("block_B", {}) or {}

    verdict_dict = {
        "audit": "D2-R",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "seeds": "501..560",
        "n_rows": len(all_rows),
        "elapsed_seconds": round(elapsed, 1),
        "SAME_PROCESS_REPRODUCIBILITY": smoke_result["SAME_PROCESS_REPRODUCIBILITY"],
        "CROSS_PROCESS_REPRODUCIBILITY": cross_result["CROSS_PROCESS_REPRODUCIBILITY"],
        "FE_FAIRNESS": "CONFIRMED" if fe_fair_ok else "FAILED",
        "ACTION_SEMANTICS": "PASS" if action_ok else "FAIL",
        "O1_PERSISTENT_CONFIRMATION": pers_v,
        "O1_STATIONARY_CONFIRMATION": stat_v,
        "O1_MIXED_CONFIRMATION": mixed_v,
        "mixed_block_A_class": mixed_bA.get("class_response", "?"),
        "mixed_block_B_class": mixed_bB.get("class_response", "?"),
        "O1_MIXED_BLOCK_STABILITY": mixed_stab,
        "O1_MIXED_P_TARGET_SAFETY": "PASS" if p_target_safe else "FAIL",
        "O1_MIXED_S_TARGET_SAFETY": "PASS" if s_target_safe else "FAIL",
        "O1_SELECTED_DESIGN_CONFIRMATION": overall_v,
        "D2R_PRODUCTION_RECOMMENDATION": rec,
        "NEXT_STAGE": next_stage,
        "PRODUCTION_DEFAULT": "O0_CURRENT",
        "PDF_STATUS": "UNCHANGED",
        "RL_STATUS": "NOT_STARTED",
        "d1r_selection_note": "Both O1 and O2 passed A1-A8 in D1-R; O1 selected by hierarchy/parsimony",
        "constraints": {
            "no_hard": True, "no_sigma_tuning": True, "no_rl": True,
            "no_pdf_update": True, "no_diagnostic_crn": True,
            "no_production_default_change": True, "no_post_hoc_third_block": True,
        },
    }

    write_verdict(verdict_dict)
    write_report(verdict_dict, all_scen_stats, regime_stats, elapsed)
    print_terminal_output(verdict_dict)
    return verdict_dict


if __name__ == "__main__":
    run_all()
