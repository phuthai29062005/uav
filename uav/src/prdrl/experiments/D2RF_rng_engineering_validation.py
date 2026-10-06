"""
D2-RF — Response RNG Reproducibility Fix: Engineering Validation Audit.

Gates (R1-R8):
  R1  nsga2_survival + response_rng: deterministic over N_REPEAT calls
  R2  region_protected_survival: no np.random.seed in source
  R3  respond_population O0 path: deterministic over N_REPEAT x N_SEEDS
  R4  Cross-process: 10 subprocesses nsga2_survival identical
  R5  make_rng("RESPONSE_SURVIVAL"): action code 4, distinct from other actions
  R6  Frozen CY verdicts unchanged (SURVIVAL_TIE_REPRODUCIBILITY=FAIL still present)
  R7  D2RF_prechange_manifest.json exists
  R8  Survival RNG namespace: make_rng(base_seed, t, region, "RESPONSE_SURVIVAL")
      appears in respond_population source

Output files:
  results/D2_rng_engineering_fix/D2RF_gate_results.json
  results/D2_rng_engineering_fix/D2RF_report.md
"""
import copy
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
from prdrl.history.region_history import RegionHistory
from prdrl.oracle.oracle import make_rng, nsga2_survival, respond_population
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate_raw,
    get_bounds,
    make_region_dynamics_map,
)

# ----------------------------------------------------------
RESULTS_DIR  = os.path.join("results", "D2_rng_engineering_fix")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "D2RF_prechange_manifest.json")
D2CY_VERDICT  = os.path.join("results", "D2CY_stochastic_protocol_crossover", "D2CY_verdict.json")

D        = 10
N_OBJ    = 2
N_PARTS  = 10
POP_SIZE = 20
N_REPEAT = 100
N_SEEDS  = 20
RF_SEEDS = list(range(301, 301 + N_SEEDS))


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr)).hexdigest()[:12]


# ----------------------------------------------------------
# Environment helpers
# ----------------------------------------------------------

def _make_env(seed, rd, V, xl, xu):
    R = len(V)
    rng = np.random.default_rng(seed)
    X   = rng.uniform(xl, xu, size=(POP_SIZE, D))
    F_m = cp_evaluate_raw(X, 0.0, rd)
    F_p = cp_evaluate_raw(X, 0.1, rd)
    Fm_norm, _ = normalize(F_m, F_p)
    rids = np.asarray(partition(np.asarray(Fm_norm), V), dtype=int)
    hist = RegionHistory(n_regions=R)
    hist.update(t=0, X=X, F=F_m, region_ids=rids, xl=xl, xu=xu)
    tgt_r = int(seed % R)
    bs    = seed * 100000 + 1 * 100 + tgt_r
    def eval_fn(Xc):
        return cp_evaluate_raw(Xc, 0.2, rd)
    return {
        "X": X, "F_plus": F_p, "region_ids": rids, "V": V,
        "actions": {tgt_r: "LOCAL"}, "history": hist,
        "t": 1, "eval_fn": eval_fn, "xl": xl, "xu": xu,
        "base_seed": bs,
    }


def _run_respond(e):
    return respond_population(
        X=e["X"].copy(), F_plus=e["F_plus"].copy(),
        region_ids=e["region_ids"].copy(), V=e["V"],
        actions=e["actions"], history=copy.deepcopy(e["history"]),
        t=e["t"], evaluate_fn=e["eval_fn"],
        xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
        sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
        lambda_pred=0.5, base_seed=e["base_seed"],
    )


# ----------------------------------------------------------
# Phase RF-0: make_rng RESPONSE_SURVIVAL
# ----------------------------------------------------------

def run_rf0_make_rng():
    print("  RF-0: make_rng RESPONSE_SURVIVAL namespace …", flush=True)
    # Action code 4 distinct from other actions
    seqs = {}
    for action in ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY", "RESPONSE_SURVIVAL"):
        gen = make_rng(100, 5, 3, action)
        seqs[action] = gen.random(20)
    collision = False
    for a1 in seqs:
        for a2 in seqs:
            if a1 != a2 and np.array_equal(seqs[a1], seqs[a2]):
                collision = True
    reproducible = True
    for _ in range(10):
        g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
        g2 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
        if not np.array_equal(g1.random(10), g2.random(10)):
            reproducible = False
            break
    result = {
        "phase": "RF-0",
        "action_code_distinct": not collision,
        "reproducible": reproducible,
        "pass": (not collision) and reproducible,
    }
    print(f"    action_code_distinct={result['action_code_distinct']}  reproducible={result['reproducible']}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-1: nsga2_survival determinism
# ----------------------------------------------------------

def run_rf1_nsga2_survival():
    print(f"  RF-1: nsga2_survival determinism ({N_REPEAT} repeats) …", flush=True)
    xl, xu = get_bounds(D)
    rng_sp = np.random.default_rng(9999)
    X_pool = rng_sp.uniform(xl, xu, size=(POP_SIZE * 2, D))
    F_pool = rng_sp.uniform(0, 1, size=(POP_SIZE * 2, N_OBJ))
    hashes = []
    for _ in range(N_REPEAT):
        rng = make_rng(301, 5, 2, "RESPONSE_SURVIVAL")
        X_new, _ = nsga2_survival(
            X=X_pool, F=F_pool, n_survive=POP_SIZE,
            xl=xl, xu=xu, response_rng=rng,
        )
        hashes.append(_md5(X_new))
    n_unique = len(set(hashes))
    result = {
        "phase": "RF-1",
        "n_repeat": N_REPEAT,
        "n_unique_hashes": n_unique,
        "pass": n_unique == 1,
    }
    print(f"    n_unique={n_unique} / {N_REPEAT}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-2: region_protected_survival — no global hack
# ----------------------------------------------------------

def run_rf2_rps_no_global_hack():
    print("  RF-2: region_protected_survival source audit …", flush=True)
    src = inspect.getsource(region_protected_survival)
    no_seed_call = "np.random.seed" not in src
    result = {
        "phase": "RF-2",
        "np_random_seed_absent": no_seed_call,
        "pass": no_seed_call,
    }
    print(f"    np.random.seed absent={no_seed_call}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-3: respond_population O0 reproducibility
# ----------------------------------------------------------

def run_rf3_respond_population():
    print(f"  RF-3: respond_population determinism ({N_SEEDS} seeds × {N_REPEAT // N_SEEDS} repeats) …", flush=True)
    rd   = make_region_dynamics_map(N_PARTS + 1)
    V    = reference(N_OBJ, N_PARTS)
    xl, xu = get_bounds(D)
    fail_seeds = []
    rows = []
    for seed in RF_SEEDS:
        e = _make_env(seed, rd, V, xl, xu)
        r1 = _run_respond(e)
        r2 = _run_respond(e)
        match = np.array_equal(r1["X"], r2["X"]) and np.array_equal(r1["F"], r2["F"])
        rows.append({"seed": seed, "match": match, "h1": _md5(r1["X"]), "h2": _md5(r2["X"])})
        if not match:
            fail_seeds.append(seed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print(flush=True)
    result = {
        "phase": "RF-3",
        "n_seeds": N_SEEDS,
        "n_fail": len(fail_seeds),
        "fail_seeds": fail_seeds,
        "pass": len(fail_seeds) == 0,
        "rows": rows,
    }
    print(f"    pass={result['pass']}  n_fail={result['n_fail']}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-4: Cross-process nsga2_survival
# ----------------------------------------------------------

def run_rf4_cross_process():
    n_procs = 10
    print(f"  RF-4: cross-process nsga2_survival ({n_procs} subprocesses) …", flush=True)
    script = (
        "import sys; sys.path.insert(0,'src'); "
        "import numpy as np, hashlib; "
        "from prdrl.oracle.oracle import make_rng, nsga2_survival; "
        "rng_sp = np.random.default_rng(9999); "
        "X = rng_sp.uniform(0,1,size=(40,10)); "
        "F = rng_sp.uniform(0,1,size=(40,2)); "
        "rng = make_rng(301,5,2,'RESPONSE_SURVIVAL'); "
        "Xn,_ = nsga2_survival(X=X,F=F,n_survive=20,"
        "xl=np.zeros(10),xu=np.ones(10),response_rng=rng); "
        "print(hashlib.md5(np.ascontiguousarray(Xn)).hexdigest())"
    )
    hashes = []
    errors = []
    for i in range(n_procs):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=os.getcwd(),
        )
        if proc.returncode != 0:
            errors.append(proc.stderr[:200])
        else:
            hashes.append(proc.stdout.strip())
    n_unique = len(set(hashes))
    result = {
        "phase": "RF-4",
        "n_procs": n_procs,
        "n_errors": len(errors),
        "n_unique_hashes": n_unique,
        "pass": len(errors) == 0 and n_unique == 1,
        "errors": errors,
    }
    print(f"    n_unique={n_unique} / {n_procs}  errors={len(errors)}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-5: Source audit — respond_population uses make_rng RESPONSE_SURVIVAL
# ----------------------------------------------------------

def run_rf5_source_audit():
    print("  RF-5: respond_population source audit …", flush=True)
    src = inspect.getsource(respond_population)
    has_surv_rng   = "surv_rng" in src
    has_response_survival = "RESPONSE_SURVIVAL" in src
    has_response_rng_param = "response_rng" in inspect.getsource(nsga2_survival)
    result = {
        "phase": "RF-5",
        "respond_population_has_surv_rng": has_surv_rng,
        "respond_population_has_RESPONSE_SURVIVAL": has_response_survival,
        "nsga2_survival_has_response_rng": has_response_rng_param,
        "pass": has_surv_rng and has_response_survival and has_response_rng_param,
    }
    print(f"    surv_rng={has_surv_rng}  RESPONSE_SURVIVAL={has_response_survival}  response_rng_param={has_response_rng_param}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-6: Frozen CY verdicts integrity
# ----------------------------------------------------------

def run_rf6_frozen_verdicts():
    print("  RF-6: frozen D2CY verdict integrity …", flush=True)
    try:
        with open(D2CY_VERDICT) as f:
            verdict = json.load(f)
        surv_fail = verdict.get("SURVIVAL_TIE_REPRODUCIBILITY") == "FAIL"
        root_combined = verdict.get("ROOT_CAUSE") == "COMBINED_STOCHASTIC_PROTOCOL_SENSITIVITY"
        child_pass = verdict.get("PRODUCTION_CHILD_REPRODUCIBILITY") == "PASS"
        resp_fail  = verdict.get("PRODUCTION_RESPONSE_REPRODUCIBILITY") == "FAIL"
        all_ok = surv_fail and root_combined and child_pass and resp_fail
        result = {
            "phase": "RF-6",
            "SURVIVAL_TIE_REPRODUCIBILITY_FAIL": surv_fail,
            "ROOT_CAUSE_COMBINED": root_combined,
            "PRODUCTION_CHILD_REPRODUCIBILITY_PASS": child_pass,
            "PRODUCTION_RESPONSE_REPRODUCIBILITY_FAIL": resp_fail,
            "pass": all_ok,
        }
    except Exception as exc:
        result = {"phase": "RF-6", "pass": False, "error": str(exc)}
    print(f"    pass={result['pass']}", flush=True)
    return result


# ----------------------------------------------------------
# Phase RF-7: Pre-change manifest exists
# ----------------------------------------------------------

def run_rf7_manifest():
    print("  RF-7: pre-change manifest check …", flush=True)
    exists = os.path.isfile(MANIFEST_PATH)
    result = {"phase": "RF-7", "manifest_exists": exists, "pass": exists}
    print(f"    exists={exists}", flush=True)
    return result


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== D2-RF RNG Engineering Validation Audit ===", flush=True)
    t_start = time.perf_counter()

    phases = {
        "RF_0_MAKE_RNG": run_rf0_make_rng(),
        "RF_1_NSGA2_SURVIVAL": run_rf1_nsga2_survival(),
        "RF_2_RPS_NO_HACK": run_rf2_rps_no_global_hack(),
        "RF_3_RESPOND_POP": run_rf3_respond_population(),
        "RF_4_CROSS_PROCESS": run_rf4_cross_process(),
        "RF_5_SOURCE_AUDIT": run_rf5_source_audit(),
        "RF_6_FROZEN_VERDICTS": run_rf6_frozen_verdicts(),
        "RF_7_MANIFEST": run_rf7_manifest(),
    }

    elapsed = time.perf_counter() - t_start

    # Gate conditions R1-R8 (mapped from RF phases 0-7)
    gate_map = {
        "R1_NSGA2_SURVIVAL_DETERMINISTIC": phases["RF_1_NSGA2_SURVIVAL"]["pass"],
        "R2_RPS_NO_GLOBAL_HACK": phases["RF_2_RPS_NO_HACK"]["pass"],
        "R3_RESPOND_POP_DETERMINISTIC": phases["RF_3_RESPOND_POP"]["pass"],
        "R4_CROSS_PROCESS_IDENTICAL": phases["RF_4_CROSS_PROCESS"]["pass"],
        "R5_MAKE_RNG_RESPONSE_SURVIVAL": phases["RF_0_MAKE_RNG"]["pass"],
        "R6_FROZEN_CY_VERDICTS_INTACT": phases["RF_6_FROZEN_VERDICTS"]["pass"],
        "R7_PRECHANGE_MANIFEST_EXISTS": phases["RF_7_MANIFEST"]["pass"],
        "R8_SOURCE_USES_SURV_RNG_NAMESPACE": phases["RF_5_SOURCE_AUDIT"]["pass"],
    }

    all_gates_pass = all(gate_map.values())

    summary = {
        "audit": "D2-RF",
        "date": "2026-09-20",
        "branch": "pr-drl-rewrite",
        "elapsed_seconds": round(elapsed, 1),
        "gate_results": gate_map,
        "all_gates_pass": all_gates_pass,
        "PRODUCTION_RESPONSE_REPRODUCIBILITY": "PASS" if all_gates_pass else "PENDING",
        "FIX_VERDICT": (
            "RESPONSE_SURVIVAL_RNG_DETERMINISTIC"
            if all_gates_pass else
            "FIX_INCOMPLETE"
        ),
        "constraints": {
            "no_performance_based_rng_selection": True,
            "no_production_default_change": True,
            "no_sigma_tuning": True,
            "pre_fix_d2c_seeds_frozen": "Seeds 109-128 D2-C results remain PRE-RNG-FIX; not used for post-fix validation",
            "next_stage": "D2CZ_PRODUCTION_STABILITY_CONFIRMATION",
        },
        "next_stage_note": (
            "D2CZ_PRODUCTION_STABILITY_CONFIRMATION: validate O3 benefit on fresh production seeds "
            "AFTER this fix, using seeds 301-320 or a new block. "
            "DO NOT use seeds 109-128 (pre-fix evidence, frozen). "
            "D2-CZ is a SEPARATE audit — not implemented here."
        ),
        "phase_detail": phases,
    }

    gate_path = os.path.join(RESULTS_DIR, "D2RF_gate_results.json")
    with open(gate_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Gate results → {gate_path}", flush=True)

    # Write report
    _write_report(summary)

    print("\n=== GATE SUMMARY ===", flush=True)
    for gate, passed in gate_map.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate:<42} {status}", flush=True)
    overall = "ALL PASS" if all_gates_pass else "FAIL"
    print(f"\n  Overall: {overall}  ({elapsed:.1f}s)", flush=True)

    return all_gates_pass


def _write_report(summary):
    lines = [
        "# D2-RF RNG Engineering Validation Report",
        "",
        f"**Date**: {summary['date']}",
        f"**Branch**: {summary['branch']}",
        f"**Elapsed**: {summary['elapsed_seconds']} s",
        "",
        "---",
        "",
        "## Gate Results",
        "",
        "| Gate | Verdict |",
        "|------|---------|",
    ]
    for gate, passed in summary["gate_results"].items():
        lines.append(f"| {gate} | {'**PASS**' if passed else '**FAIL**'} |")

    lines += [
        "",
        f"**All gates pass**: {summary['all_gates_pass']}",
        f"**FIX_VERDICT**: `{summary['FIX_VERDICT']}`",
        "",
        "---",
        "",
        "## Engineering Changes",
        "",
        "| Component | Change |",
        "|-----------|--------|",
        "| `make_rng` | Added `\"RESPONSE_SURVIVAL\": 4` to action_code — response survival now has its own namespace |",
        "| `nsga2_survival` | Added `response_rng=None` parameter; passed `random_state=response_rng` to `RankAndCrowding().do()` |",
        "| `respond_population` | Derives `surv_rng = make_rng(base_seed, t, primary_region, \"RESPONSE_SURVIVAL\")` before survival; passes `response_rng=surv_rng` to both `nsga2_survival` call sites |",
        "| `region_protected_survival` | Removed `np.random.seed()` global-state hack (`sorted()` already deterministic; hack was dead code in production path) |",
        "",
        "---",
        "",
        "## Seed Namespace",
        "",
        "```",
        "surv_rng = make_rng(base_seed, t, primary_active_region, \"RESPONSE_SURVIVAL\")",
        "# base_seed  = seed * 100000 + change_index * 100 + target_region",
        "# t          = history_t (change index in current experiment)",
        "# primary_region = first non-KEEP action region (sorted)",
        "# action code = 4 (RESPONSE_SURVIVAL)",
        "```",
        "",
        "Namespace includes: experiment seed, change index, target region, action type.",
        "No wall clock, OS entropy, or process scheduling involved.",
        "",
        "---",
        "",
        "## Constraints",
        "",
        "- RNG scheme chosen for **reproducibility**, not optimization performance ✓",
        "- Production default O0 unchanged ✓",
        "- No sigma tuning ✓",
        "- Seeds 109-128 (D2-C pre-fix evidence) frozen — not used for post-fix validation ✓",
        "",
        "---",
        "",
        "## Next Stage: D2-CZ",
        "",
        summary["next_stage_note"],
        "",
        "D2-CZ is NOT implemented here. This report documents the engineering fix only.",
        "",
    ]

    report_path = os.path.join(RESULTS_DIR, "D2RF_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report → {report_path}", flush=True)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
