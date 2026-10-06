"""
D2-B — Production-Path Equivalence Harness.

Verifies that the production oracle path with O3 (HARD_RESPONSE_ONLY) produces
the same child/survivor/FE results as calling region_protected_survival directly
on the same pre-response inputs.

Design:
    1. Build a minimal pre-response state using the controlled_predict problem.
       - Small population (POP_SIZE=30) and dimension (D=4) for fast tests.
       - Two history time-steps so PREDICT has a valid centroid sequence.
       - Region IDs assigned via partition + normalize (same as production).
    2. Diagnostic path: generate_children(PREDICT) + evaluate + region_protected_survival.
    3. Production path: respond_population(HARD_RESPONSE_ONLY).
    4. Compare survivors (set-equivalent), extra_fe, HARD structural invariant.

This harness does NOT re-run L2B seeds. It tests the production-path machinery
against the canonical core helper in isolation — proving D2-B equivalence.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import prdrl.experiments.E2P2_predict_noise_sweep as e2p2

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.core.survival import region_protected_survival
from prdrl.history.region_history import RegionHistory
from prdrl.oracle.oracle import (
    PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    PREDICT_SURVIVAL_UNRESTRICTED,
    generate_children,
    respond_population,
)
from prdrl.problems.controlled_predict import evaluate as cp_evaluate, get_bounds


# ============================================================
# Minimal scenario constants (fast, deterministic)
# ============================================================
POP_SIZE = 30
D = 4          # controlled_predict requires D >= 2
N_OBJ = 2
N_PARTITIONS = 3   # → 4 reference vectors (4 regions)
BUDGET = 4
SCENARIO = "PERSISTENT"


def _region_dynamics(n_bins):
    return e2p2.make_scenario_dynamics(SCENARIO, n_bins)


def _assign_regions(X, F, V):
    """Assign region IDs via normalize + partition (same as production oracle)."""
    F_m, _ = normalize(F, F)
    return np.asarray(partition(np.asarray(F_m), V), dtype=int)


def build_pre_response_state(seed=42, target_region=0):
    """Return a fully specified pre-response state for equivalence testing.

    Both the diagnostic path and the production path consume this identical state.

    Returns dict with keys:
        X, F_plus, region_ids, history, region_dynamics, V,
        t, xl, xu, target_region, n_r_before
    """
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)         # 4
    n_bins = R
    rd = _region_dynamics(n_bins)

    rng = np.random.default_rng(seed)
    X = rng.uniform(xl, xu, size=(POP_SIZE, D))

    t_hist = 10        # final history step
    t = t_hist + 1     # current (post-change) time step

    history = RegionHistory(n_regions=R)

    # Step t_hist - 1: build initial centroids
    F_t0 = cp_evaluate(X, t_hist - 1, rd)
    region_ids_t0 = _assign_regions(X, F_t0, V)
    history.update(t=t_hist - 1, X=X, F=F_t0, region_ids=region_ids_t0, xl=xl, xu=xu)

    # Step t_hist: perturb X slightly to create a non-trivial centroid motion
    rng2 = np.random.default_rng(seed + 9999)
    X_hist = X + rng2.uniform(-0.02, 0.02, size=X.shape)
    X_hist = np.clip(X_hist, xl, xu)
    F_t1 = cp_evaluate(X_hist, t_hist, rd)
    region_ids_t1 = _assign_regions(X_hist, F_t1, V)
    history.update(t=t_hist, X=X_hist, F=F_t1, region_ids=region_ids_t1, xl=xl, xu=xu)

    # Current state: X evaluated at t (post-change)
    F_plus = cp_evaluate(X, t, rd)
    region_ids = _assign_regions(X, F_plus, V)

    # Ensure target_region is not empty; if it is, pick the first non-empty region
    while int(np.sum(region_ids == target_region)) == 0:
        target_region = (target_region + 1) % R

    n_r_before = int(np.sum(region_ids == target_region))

    return {
        "X": X,
        "F_plus": F_plus,
        "region_ids": region_ids,
        "region_dynamics": rd,
        "V": V,
        "history": history,
        "t": t,
        "xl": xl,
        "xu": xu,
        "target_region": target_region,
        "n_r_before": n_r_before,
        "POP_SIZE": POP_SIZE,
    }


def run_diagnostic_path(state, sigma_predict=0.005):
    """Diagnostic path: generate_children + evaluate + region_protected_survival.

    This reproduces what the production path does internally, calling the helper
    directly — so we can verify they are equivalent.

    Returns dict: X_surv, F_surv, X_children, F_children, extra_fe, n_old_dropped_other
    """
    X = state["X"]
    F_plus = state["F_plus"]
    region_ids = state["region_ids"]
    rd = state["region_dynamics"]
    history = state["history"]
    t = state["t"]
    xl, xu = state["xl"], state["xu"]
    r = state["target_region"]
    n_r_before = state["n_r_before"]
    N = len(X)

    mask = region_ids == r
    X_region = X[mask]

    children = generate_children(
        action="PREDICT",
        region_id=r,
        X_region=X_region,
        history=history,
        t=t,
        budget=BUDGET,
        xl=xl,
        xu=xu,
        sigma_local=0.02,
        sigma_predict=sigma_predict,
        sigma_diversify=0.15,
        lambda_pred=1.0,
        base_seed=42,
    )

    F_children = cp_evaluate(children, t, rd)
    extra_fe = len(children)

    X_surv, F_surv, final_idx = region_protected_survival(
        X_pop=X,
        F_pop=F_plus,
        X_children=children,
        F_children=F_children,
        response_region_r=r,
        old_region_ids=region_ids,
        n_r_before=n_r_before,
        xl=xl,
        xu=xu,
    )

    # HARD invariant verification
    final_set = set(int(i) for i in final_idx if i < N)
    old_outside = [i for i in range(N) if region_ids[i] != r]
    n_old_dropped_other = sum(1 for i in old_outside if i not in final_set)

    return {
        "X_surv": X_surv,
        "F_surv": F_surv,
        "X_children": children,
        "F_children": F_children,
        "extra_fe": extra_fe,
        "n_old_dropped_other": n_old_dropped_other,
        "final_idx": final_idx,
    }


def run_production_path_o3(state, sigma_predict=0.005):
    """Production-path O3: respond_population with HARD_RESPONSE_ONLY."""
    X = state["X"]
    F_plus = state["F_plus"]
    region_ids = state["region_ids"]
    rd = state["region_dynamics"]
    V = state["V"]
    history = state["history"]
    t = state["t"]
    xl, xu = state["xl"], state["xu"]
    r = state["target_region"]

    actions = {r: "PREDICT"}

    result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions=actions,
        history=history,
        t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl,
        xu=xu,
        budget=BUDGET,
        sigma_local=0.02,
        sigma_predict=sigma_predict,
        sigma_diversify=0.15,
        lambda_pred=1.0,
        base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )

    return {
        "X_surv": result["X"],
        "F_surv": result["F"],
        "extra_fe": result["extra_fe"],
    }


def run_production_path_o0(state, sigma_predict=0.03):
    """Production-path O0 (default): respond_population with UNRESTRICTED."""
    X = state["X"]
    F_plus = state["F_plus"]
    region_ids = state["region_ids"]
    rd = state["region_dynamics"]
    V = state["V"]
    history = state["history"]
    t = state["t"]
    xl, xu = state["xl"], state["xu"]
    r = state["target_region"]

    actions = {r: "PREDICT"}

    result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions=actions,
        history=history,
        t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl,
        xu=xu,
        budget=BUDGET,
        sigma_local=0.02,
        sigma_predict=sigma_predict,
        sigma_diversify=0.15,
        lambda_pred=1.0,
        base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )

    return {
        "X_surv": result["X"],
        "F_surv": result["F"],
        "extra_fe": result["extra_fe"],
    }


def run_action_isolated_path(state, action, sigma_predict=0.03):
    """Run non-PREDICT action through production path O0 (unrestricted)."""
    X = state["X"]
    F_plus = state["F_plus"]
    region_ids = state["region_ids"]
    rd = state["region_dynamics"]
    V = state["V"]
    history = state["history"]
    t = state["t"]
    xl, xu = state["xl"], state["xu"]
    r = state["target_region"]

    actions = {r: action}

    result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        actions=actions,
        history=history,
        t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl,
        xu=xu,
        budget=BUDGET,
        sigma_local=0.02,
        sigma_predict=sigma_predict,
        sigma_diversify=0.15,
        lambda_pred=1.0,
        base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )
    return {"X_surv": result["X"], "F_surv": result["F"], "extra_fe": result["extra_fe"]}


def _sorted_rows(X):
    """Sort rows of X for set-equality comparison."""
    return X[np.lexsort(np.round(X, 12).T)]


def run_equivalence_case(seed, target_region=0):
    """Run one equivalence case. Returns dict with equivalence flags."""
    state = build_pre_response_state(seed=seed, target_region=target_region)

    diag = run_diagnostic_path(state, sigma_predict=0.005)
    prod = run_production_path_o3(state, sigma_predict=0.005)
    o0 = run_production_path_o0(state, sigma_predict=0.03)

    # Size invariant
    survivors_size_ok = (
        len(diag["X_surv"]) == state["POP_SIZE"] and
        len(prod["X_surv"]) == state["POP_SIZE"]
    )

    # Survivor set equivalence (order-independent)
    X_d_sorted = _sorted_rows(diag["X_surv"])
    X_p_sorted = _sorted_rows(prod["X_surv"])
    survivors_identical = bool(np.allclose(X_d_sorted, X_p_sorted, atol=1e-10))

    F_d_sorted = _sorted_rows(diag["F_surv"])
    F_p_sorted = _sorted_rows(prod["F_surv"])
    f_survivors_identical = bool(np.allclose(F_d_sorted, F_p_sorted, atol=1e-10))

    fe_match = (diag["extra_fe"] == prod["extra_fe"])
    hard_invariant = (diag["n_old_dropped_other"] == 0)

    # O0 must be structurally distinct (different sigma → different children → likely different survivors)
    # Note: in rare edge cases they could match by chance, but we record the fact, not assert.
    o0_same_as_o3 = bool(np.allclose(
        _sorted_rows(o0["X_surv"]),
        _sorted_rows(prod["X_surv"]),
        atol=1e-10,
    ))

    case_pass = (
        survivors_size_ok and
        survivors_identical and
        f_survivors_identical and
        fe_match and
        hard_invariant
    )

    return {
        "seed": seed,
        "target_region": int(state["target_region"]),
        "n_r_before": state["n_r_before"],
        "extra_fe_diag": diag["extra_fe"],
        "extra_fe_prod": prod["extra_fe"],
        "fe_match": fe_match,
        "survivors_size_ok": survivors_size_ok,
        "survivors_identical": survivors_identical,
        "f_survivors_identical": f_survivors_identical,
        "hard_invariant_diag": hard_invariant,
        "n_old_dropped_other": diag["n_old_dropped_other"],
        "o0_same_as_o3": o0_same_as_o3,
        "PASS": case_pass,
    }


def run_batch(seeds, target_regions=None):
    """Run equivalence cases for multiple seeds."""
    if target_regions is None:
        target_regions = [0] * len(seeds)
    return [
        run_equivalence_case(seed, tr)
        for seed, tr in zip(seeds, target_regions)
    ]


def summarize_batch(results):
    n = len(results)
    return {
        "n_total": n,
        "n_pass": sum(1 for r in results if r["PASS"]),
        "n_fe_match": sum(1 for r in results if r["fe_match"]),
        "n_survivors_identical": sum(1 for r in results if r["survivors_identical"]),
        "n_f_survivors_identical": sum(1 for r in results if r["f_survivors_identical"]),
        "n_hard_invariant": sum(1 for r in results if r["hard_invariant_diag"]),
        "n_o0_same_as_o3": sum(1 for r in results if r["o0_same_as_o3"]),
        "all_pass": all(r["PASS"] for r in results),
        "all_fe_match": all(r["fe_match"] for r in results),
        "all_identical": all(r["survivors_identical"] for r in results),
        "all_hard_invariant": all(r["hard_invariant_diag"] for r in results),
    }


if __name__ == "__main__":
    OUT_DIR = Path("results/D2_implementation_validation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = list(range(81, 101))  # 20 seeds (matches L2B frozen range)
    print(f"Running D2-B equivalence gate: {len(seeds)} cases...")
    results = run_batch(seeds)

    summary = summarize_batch(results)

    jsonl_path = OUT_DIR / "D2_equivalence_cases.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            line = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                    for k, v in r.items()}
            f.write(json.dumps(line) + "\n")

    summary_path = OUT_DIR / "D2_equivalence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nD2-B Equivalence Gate:")
    print(f"  Cases:                {summary['n_total']}")
    print(f"  ALL PASS:             {summary['all_pass']}")
    print(f"  Survivors identical:  {summary['all_identical']}")
    print(f"  FE match:             {summary['all_fe_match']}")
    print(f"  HARD invariant:       {summary['all_hard_invariant']}")
    print(f"  O0 same as O3:        {summary['n_o0_same_as_o3']}/{summary['n_total']}")

    status = "PASS" if summary["all_pass"] else "FAIL"
    print(f"\nD2_DIAGNOSTIC_EQUIVALENCE: {status}")
