"""
H1 — Regional Controller Hypothesis Review.

DIAGNOSTIC stage following E3-R Gate B = FAIL.

Determines which of three hypotheses explains the failure:
  A. GLOBAL_ACTION_DOMINANCE      — no regional oracle headroom
  B. STATE_INFORMATION_INSUFFICIENT — headroom exists but V2 can't identify it
  C. RULE_MAPPING_INADEQUATE      — headroom + V2 signal, rule mapping was wrong

Stage invariants:
  - RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
  - No controller rollout, no RULE_V2, no DQN, no reward training.
  - Production code UNCHANGED.
  - Carrier: GLOBAL_LOCAL (predeclared, not oracle-chosen).
  - Seeds fresh: 901-940. Block A: 901-920. Block B: 921-940.
  - No overlap with E3 701-780, E3-R 801-882.
  - DELTA_H1_ROG = 1e-5. MARGIN_EPS = 1e-8. BAL_ACC_MARGIN = 0.05.
  - quality_fn = -IGD (existing convention from E2/E2a).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.detectors.quality_loss import quality_loss
from prdrl.detectors.severity import severity
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import regional_oracle, ACTIONS, operator_heterogeneity_index
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)
from prdrl.rl.interface import (
    CURRENT_RESPONSE_BUDGET_VALUE,
    LOCAL_SPEC,
    PREDICT_SPEC,
    DIVERSIFY_SPEC,
    execute_response,
)

# ============================================================
# LOCKED CONFIG
# ============================================================
N_OBJ              = 2
D                  = 10
POP_SIZE           = 100
N_PARTITIONS       = 9
WARMUP_GENERATIONS = 30
GENERATIONS_PER_CHANGE = 10
BUDGET             = CURRENT_RESPONSE_BUDGET_VALUE   # 10
SIGMA_LOCAL        = LOCAL_SPEC.sigma                # 0.02
SIGMA_PREDICT      = PREDICT_SPEC.sigma              # 0.005
SIGMA_DIVERSIFY    = DIVERSIFY_SPEC.sigma            # 0.15
LAMBDA_PREDICT     = PREDICT_SPEC.lambda_P           # 1.0
THETA_PBI          = 5.0
KAPPA_SEV          = 1.0
Q_PREV_INIT        = 1.0

N_SCORED_CHANGES = 10
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

DIAG_SEEDS   = list(range(901, 941))   # 40 seeds
BLOCK_A_SEEDS = list(range(901, 921))  # 20 seeds
BLOCK_B_SEEDS = list(range(921, 941))  # 20 seeds

# Excluded seed ranges
E3_SEEDS = set(range(701, 781))
E3R_SEEDS = set(range(801, 883))

# Statistical constants (FROZEN — do not tune after results)
DELTA_H1_ROG  = 1e-5
MARGIN_EPS    = 1e-8
BAL_ACC_MARGIN = 0.05

BOOTSTRAP_RESAMPLES_BLOCK  = 20_000
BOOTSTRAP_RESAMPLES_POOLED = 30_000
BOOTSTRAP_SEED = 7777

CARRIER_CONTROLLER = "GLOBAL_LOCAL"   # predeclared neutral carrier
EPS_ROG = 1e-12

RESULTS_DIR = Path("results/H1_regional_controller_hypothesis_review")

RL_STATUS    = "NOT_STARTED"
DRL_TRAINING = "NOT_STARTED"


# ============================================================
# Scenario helpers (same as E3-R)
# ============================================================

def make_region_dynamics(scenario: str, R: int) -> List[str]:
    if scenario == "PERSISTENT":
        return ["persistent"] * R
    if scenario == "STATIONARY":
        return ["stationary"] * R
    if scenario == "MIXED":
        return make_region_dynamics_map(R)
    raise ValueError(f"Unknown scenario: {scenario}")


def compute_time(generation: int) -> float:
    return (1.0 / 10) * np.floor(generation / 10)


# ============================================================
# NSGA-II wrapper (same as E3-R)
# ============================================================

class _CProblem(Problem):
    def __init__(self, t, rd, xl, xu):
        super().__init__(n_var=len(xl), n_obj=N_OBJ, xl=xl, xu=xu)
        self.t = t; self.rd = rd

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = cp_evaluate(X, self.t, self.rd)


def evolve_nsga2(X_start, t, rd, n_gen, seed, xl, xu):
    problem = _CProblem(t, rd, xl, xu)
    algorithm = NSGA2(pop_size=len(X_start), sampling=X_start)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return np.asarray(result.pop.get("X"), dtype=float)


# ============================================================
# Warmup + burn-in (shared across arms — same as E3-R)
# ============================================================

def _warmup_and_burnin(seed, rd, xl, xu, V, R):
    rng = np.random.default_rng(seed)
    X = rng.uniform(xl, xu, size=(POP_SIZE, D))
    gen = 0
    t0 = compute_time(gen)
    X = evolve_nsga2(X, t0, rd, WARMUP_GENERATIONS, seed * 10000, xl, xu)
    gen += WARMUP_GENERATIONS
    h = RegionHistory(n_regions=R)
    h_t = -1
    F_m = cp_evaluate(X, t0, rd)
    gen += GENERATIONS_PER_CHANGE
    t1 = compute_time(gen)
    F_p = cp_evaluate(X, t1, rd)
    Ftm, _ = normalize(F_m, F_p)
    rids = np.asarray(partition(np.asarray(Ftm), V))
    h_t += 1
    h.update(t=h_t, X=X, F=F_m, region_ids=rids, xl=xl, xu=xu)
    X = evolve_nsga2(X, t1, rd, GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu)
    return X, h, h_t, t1, gen


# ============================================================
# Per-seed per-scenario oracle run
# ============================================================

def run_h1_oracle(
    seed: int,
    scenario: str,
    n_scored: int,
    xl,
    xu,
    V,
    R: int,
    pf_star,
    rd: List[str],
    X_init,
    h_init: RegionHistory,
    h_t_init: int,
    t_init: float,
    gen_init: int,
    block: str,
) -> Tuple[List[dict], List[dict]]:
    """Run oracle evaluation for one seed × scenario.

    For each scored change:
    1. Freeze pre-response state
    2. Compute V2 features
    3. Call regional_oracle — evaluates all 4 actions in isolation, event-isolated
    4. Record per-region and per-event oracle data
    5. Advance carrier (GLOBAL_LOCAL → NSGA-II) — NOT from oracle result

    Returns (oracle_region_rows, oracle_event_rows).
    """
    X = X_init.copy()
    h = copy.deepcopy(h_init)
    h_t = h_t_init
    t_cur = t_init
    gen = gen_init

    # Carrier severity state (GLOBAL_LOCAL)
    q_prev = Q_PREV_INIT
    sev_hist: Dict[int, Optional[float]] = {r: None for r in range(R)}

    def quality_fn(F):
        return -float(igd(F, pf_star))

    region_rows: List[dict] = []
    event_rows: List[dict] = []

    for k in range(n_scored):
        gen += GENERATIONS_PER_CHANGE
        t_new = compute_time(gen)

        F_minus = cp_evaluate(X, t_cur, rd)
        F_plus  = cp_evaluate(X, t_new, rd)
        Ftm_m, Ftm_p = normalize(F_minus, F_plus)
        rids = np.asarray(partition(np.asarray(Ftm_m), V))

        active = [r for r in range(R) if int(np.sum(rids == r)) > 0]
        n_act = len(active)

        # History update (pre-change)
        h_t += 1
        h.update(t=h_t, X=X, F=F_minus, region_ids=rids, xl=xl, xu=xu)

        # V2 features
        S, C, q_new = severity(Ftm_m, Ftm_p, rids, R, q_prev, kappa=KAPPA_SEV)
        losses = quality_loss(Ftm_m, Ftm_p, rids, V, THETA_PBI)
        v2 = {}
        for r in active:
            n_r = int(np.sum(rids == r))
            sr = float(C[r]) if not np.isnan(C[r]) else 0.0
            lr = float(losses[r]) if not np.isnan(losses[r]) else 0.0
            or_ = n_r / POP_SIZE
            prev_s = sev_hist[r]
            dr = (sr - prev_s) if prev_s is not None else 0.0
            v2[r] = {"severity_r": sr, "convergence_loss_r": lr,
                     "occupancy_r": or_, "delta_severity_r": dr}

        # ── Event-isolated oracle ──────────────────────────────
        eval_fn = lambda Xq, _t=t_new, _rd=rd: cp_evaluate(Xq, _t, _rd)
        ores = regional_oracle(
            X=X, F_plus=F_plus, region_ids=rids, V=V,
            history=h, t=h_t + 1,
            evaluate_fn=eval_fn, quality_fn=quality_fn,
            xl=xl, xu=xu, budget=BUDGET,
            sigma_local=SIGMA_LOCAL, sigma_predict=SIGMA_PREDICT,
            sigma_diversify=SIGMA_DIVERSIFY, lambda_pred=LAMBDA_PREDICT,
            base_seed=seed,
        )

        # Event-level quantities
        Q_keep = float(ores["Q_keep"])
        Q_reg  = float(ores["regional_Q"])
        ROG    = float(ores["ROG"])
        best_global_a = ores["best_global_action"]
        Q_best_global = float(ores["best_global_Q"])
        OHI = float(ores["OHI"])
        Q_predict_global = float(ores["global_results"]["PREDICT"]["Q"])
        ROG_vs_pred = (Q_reg - Q_predict_global) / (abs(Q_predict_global) + EPS_ROG)

        event_rows.append({
            "scenario": scenario, "seed": seed, "block": block, "k": k,
            "t_new": float(t_new), "n_active_regions": n_act,
            "Q_keep": Q_keep, "Q_best_global": Q_best_global,
            "best_global_action": best_global_a, "Q_regional": Q_reg,
            "ROG": ROG, "ROG_vs_GLOBAL_PREDICT": float(ROG_vs_pred),
            "OHI": OHI,
            "Q_GLOBAL_KEEP":      float(ores["global_results"]["KEEP"]["Q"]),
            "Q_GLOBAL_LOCAL":     float(ores["global_results"]["LOCAL"]["Q"]),
            "Q_GLOBAL_PREDICT":   Q_predict_global,
            "Q_GLOBAL_DIVERSIFY": float(ores["global_results"]["DIVERSIFY"]["Q"]),
        })

        # Per-region records
        for r in active:
            utils = ores["utilities"][r]    # {KEEP, LOCAL, PREDICT, DIVERSIFY}
            preferred = ores["preferred_action"][r]

            # Sort utilities to get best/second (KEEP wins strict ties at 0)
            sorted_u = sorted(utils.items(), key=lambda kv: -kv[1])
            best_u   = sorted_u[0][1]
            second_u = sorted_u[1][1] if len(sorted_u) > 1 else float("nan")
            second_a = sorted_u[1][0] if len(sorted_u) > 1 else None
            margin   = (best_u - second_u) if not np.isnan(second_u) else float("nan")

            feat = v2[r]
            region_rows.append({
                "scenario": scenario, "seed": seed, "block": block, "k": k,
                "region_id": r, "n_active_regions": n_act,
                # V2 features (X for classifier)
                "severity_r":           feat["severity_r"],
                "convergence_loss_r":   feat["convergence_loss_r"],
                "occupancy_r":          feat["occupancy_r"],
                "delta_severity_r":     feat["delta_severity_r"],
                # Oracle utilities
                "U_KEEP":      float(utils["KEEP"]),
                "U_LOCAL":     float(utils.get("LOCAL",    float("nan"))),
                "U_PREDICT":   float(utils.get("PREDICT",  float("nan"))),
                "U_DIVERSIFY": float(utils.get("DIVERSIFY", float("nan"))),
                # Oracle labels
                "oracle_best_action":   preferred,
                "oracle_second_action": second_a,
                "best_utility":         float(best_u),
                "second_utility":       float(second_u),
                "action_margin":        float(margin),
                # Event-level (repeated per region)
                "best_global_action":   best_global_a,
                "Q_best_global":        Q_best_global,
                "Q_keep":               Q_keep,
                "Q_regional":           Q_reg,
                "ROG":                  ROG,
                "ROG_vs_GLOBAL_PREDICT": float(ROG_vs_pred),
                "OHI":                  OHI,
            })

        # ── Advance carrier (GLOBAL_LOCAL) ─────────────────────
        carrier_actions = {r: "LOCAL" for r in active}
        carrier_res, _ = execute_response(
            actions=carrier_actions, X=X, F_plus=F_plus,
            region_ids=rids, V=V, history=h, t=h_t + 1,
            evaluate_fn=eval_fn, xl=xl, xu=xu,
            base_seed=seed, budget=BUDGET,
            sigma_local=SIGMA_LOCAL, sigma_predict=SIGMA_PREDICT,
            sigma_diversify=SIGMA_DIVERSIFY, lambda_pred=LAMBDA_PREDICT,
        )
        X_next = np.asarray(carrier_res["X"], dtype=float)
        evo_seed = seed * 10000 + k + 2
        X = evolve_nsga2(X_next, t_new, rd, GENERATIONS_PER_CHANGE, evo_seed, xl, xu)

        # Update carrier severity state
        q_prev = float(q_new)
        for r in active:
            if not np.isnan(C[r]):
                sev_hist[r] = float(C[r])
        t_cur = t_new

    return region_rows, event_rows


# ============================================================
# Run all seeds
# ============================================================

def run_all_seeds(
    seeds: List[int],
    n_scored: int,
    xl, xu, V, R, pf_star,
    is_smoke: bool = False,
) -> Tuple[List[dict], List[dict]]:
    all_region: List[dict] = []
    all_event:  List[dict] = []
    total = len(seeds) * len(SCENARIOS)
    done  = 0
    for seed in seeds:
        blk = "A" if seed in BLOCK_A_SEEDS else ("B" if seed in BLOCK_B_SEEDS else "SMOKE")
        for scenario in SCENARIOS:
            rd = make_region_dynamics(scenario, R)
            X_init, h_init, h_t_init, t_init, gen_init = _warmup_and_burnin(
                seed, rd, xl, xu, V, R
            )
            rr, er = run_h1_oracle(
                seed=seed, scenario=scenario, n_scored=n_scored,
                xl=xl, xu=xu, V=V, R=R, pf_star=pf_star,
                rd=rd, X_init=X_init, h_init=h_init,
                h_t_init=h_t_init, t_init=t_init, gen_init=gen_init,
                block=blk,
            )
            all_region.extend(rr)
            all_event.extend(er)
            done += 1
            if not is_smoke:
                print(f"    [{done}/{total}] seed={seed} scenario={scenario} "
                      f"n_region_rows={len(rr)}")
    return all_region, all_event


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(values: List[float], n_resamples: int, rng_seed: int):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(rng_seed)
    means = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_resamples)]
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return float(np.mean(arr)), lo, hi


def classify_rog(mean_val, ci_lo, ci_hi, delta=DELTA_H1_ROG):
    if mean_val > delta and ci_lo > 0:
        return "SUPPORTED"
    if mean_val > delta:
        return "DIRECTIONAL"
    if abs(mean_val) <= delta:
        return "NEUTRAL"
    if mean_val < -delta and ci_hi < 0:
        return "CONTRADICTED"
    return "DIRECTIONAL_NEGATIVE"


def per_seed_mean(event_rows, seeds, scenario, metric):
    result = {}
    for s in seeds:
        vals = [r[metric] for r in event_rows
                if r["seed"] == s and r["scenario"] == scenario]
        result[s] = float(np.mean(vals)) if vals else float("nan")
    return result


def compute_headroom_stat(event_rows, seeds, scenario, n_resamples, rng_seed):
    seed_rogs = per_seed_mean(event_rows, seeds, scenario, "ROG")
    valid = [v for v in seed_rogs.values() if not np.isnan(v)]
    if not valid:
        return {"scenario": scenario, "n_seeds": 0, "mean_ROG": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "classification": "UNKNOWN"}
    mean_r, lo, hi = bootstrap_ci(valid, n_resamples, rng_seed)
    cls = classify_rog(mean_r, lo, hi)
    return {"scenario": scenario, "n_seeds": len(valid),
            "mean_ROG": mean_r, "ci_lo": lo, "ci_hi": hi, "classification": cls}


def compute_headroom_stat_vs_predict(event_rows, seeds, scenario, n_resamples, rng_seed):
    seed_vals = per_seed_mean(event_rows, seeds, scenario, "ROG_vs_GLOBAL_PREDICT")
    valid = [v for v in seed_vals.values() if not np.isnan(v)]
    if not valid:
        return {"scenario": scenario, "n_seeds": 0, "mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "classification": "UNKNOWN"}
    mean_r, lo, hi = bootstrap_ci(valid, n_resamples, rng_seed)
    cls = classify_rog(mean_r, lo, hi)
    return {"scenario": scenario, "n_seeds": len(valid),
            "mean": mean_r, "ci_lo": lo, "ci_hi": hi, "classification": cls}


# ============================================================
# Headroom decision
# ============================================================

def decide_headroom(pooled_stats, block_a_stats, block_b_stats):
    # Per-scenario headroom
    results = {}
    for sc in SCENARIOS:
        p = pooled_stats[sc]["classification"]
        a = block_a_stats[sc]["classification"]
        b = block_b_stats[sc]["classification"]
        if p == "SUPPORTED" and a in ("SUPPORTED", "DIRECTIONAL") and b in ("SUPPORTED", "DIRECTIONAL"):
            h = "SUPPORTED"
        elif pooled_stats[sc]["mean_ROG"] > 0 and p in ("DIRECTIONAL", "SUPPORTED"):
            h = "PARTIAL"
        else:
            h = "ABSENT"
        results[sc] = h

    # Overall headroom
    counts = Counter(results.values())
    if counts.get("SUPPORTED", 0) >= 2:
        overall = "SUPPORTED"
    elif counts.get("SUPPORTED", 0) + counts.get("PARTIAL", 0) >= 2:
        overall = "PARTIAL"
    else:
        overall = "ABSENT"
    return overall, results


# ============================================================
# Global PREDICT dominance
# ============================================================

def compute_predict_dominance(region_rows, margin_eps=MARGIN_EPS):
    non_tie = [r for r in region_rows if abs(r["action_margin"]) > margin_eps]
    if not non_tie:
        predict_frac = float("nan")
    else:
        predict_frac = sum(1 for r in non_tie if r["oracle_best_action"] == "PREDICT") / len(non_tie)

    # ROG vs PREDICT classification (pooled over all seeds/scenarios)
    # Use event_rows for ROG_vs_PREDICT
    return predict_frac


def decide_predict_dominance(predict_frac, rog_vs_predict_cls):
    if np.isnan(predict_frac):
        return "UNKNOWN"
    if predict_frac >= 0.80 and rog_vs_predict_cls != "SUPPORTED":
        return "STRONG"
    if predict_frac >= 0.60:
        return "MODERATE"
    return "WEAK"


# ============================================================
# Classifier probes
# ============================================================

ACTION_LABEL_MAP = {"KEEP": 0, "LOCAL": 1, "PREDICT": 2, "DIVERSIFY": 3}
ACTION_FROM_INT  = {v: k for k, v in ACTION_LABEL_MAP.items()}

def build_Xy(region_rows, seeds, scenario=None):
    rows = region_rows
    if seeds is not None:
        rows = [r for r in rows if r["seed"] in seeds]
    if scenario is not None:
        rows = [r for r in rows if r["scenario"] == scenario]
    X = np.array([[r["severity_r"], r["convergence_loss_r"],
                   r["occupancy_r"], r["delta_severity_r"]] for r in rows], dtype=float)
    y = np.array([ACTION_LABEL_MAP[r["oracle_best_action"]] for r in rows], dtype=int)
    return X, y


def run_classifier_probes(region_rows, seeds_a, seeds_b, n_permutations=100):
    """Cross-block probes: A->B and B->A for each scenario and pooled."""
    results = {}
    for scenario in SCENARIOS + ["POOLED"]:
        sc_filter = None if scenario == "POOLED" else scenario

        X_a, y_a = build_Xy(region_rows, set(seeds_a), sc_filter)
        X_b, y_b = build_Xy(region_rows, set(seeds_b), sc_filter)

        if len(X_a) == 0 or len(X_b) == 0:
            results[scenario] = {"error": "no_data"}
            continue

        sc_results = {}
        for direction, (X_tr, y_tr, X_te, y_te, dir_seed) in [
            ("A_to_B", (X_a, y_a, X_b, y_b, 1)),
            ("B_to_A", (X_b, y_b, X_a, y_a, 2)),
        ]:
            if len(np.unique(y_tr)) < 2:
                sc_results[direction] = {"error": "single_class_train"}
                continue

            # Logistic regression
            lr = LogisticRegression(max_iter=500, random_state=0, solver="lbfgs",
                                    multi_class="multinomial")
            lr.fit(X_tr, y_tr)
            pred_lr = lr.predict(X_te)

            # Decision tree
            dt = DecisionTreeClassifier(max_depth=3, random_state=0)
            dt.fit(X_tr, y_tr)
            pred_dt = dt.predict(X_te)

            bal_lr = float(balanced_accuracy_score(y_te, pred_lr))
            f1_lr  = float(f1_score(y_te, pred_lr, average="macro", zero_division=0))
            bal_dt = float(balanced_accuracy_score(y_te, pred_dt))
            f1_dt  = float(f1_score(y_te, pred_dt, average="macro", zero_division=0))

            # Simple baselines
            majority_class = int(Counter(y_tr).most_common(1)[0][0])
            pred_majority = np.full(len(y_te), majority_class, dtype=int)
            bal_majority = float(balanced_accuracy_score(y_te, pred_majority))

            pred_predict_label = np.full(len(y_te), ACTION_LABEL_MAP["PREDICT"], dtype=int)
            bal_predict_baseline = float(balanced_accuracy_score(y_te, pred_predict_label))

            # Confusion matrices
            labels_present = sorted(set(y_te) | set(y_tr))
            cm_lr = confusion_matrix(y_te, pred_lr, labels=labels_present).tolist()
            cm_dt = confusion_matrix(y_te, pred_dt, labels=labels_present).tolist()

            # Permutation control (100 deterministic permutations)
            perm_bal_lr = []
            perm_bal_dt = []
            rng_perm = np.random.default_rng(BOOTSTRAP_SEED + dir_seed * 100)
            for pi in range(n_permutations):
                y_perm = rng_perm.permutation(y_tr)
                lr_p = LogisticRegression(max_iter=200, random_state=pi, solver="lbfgs",
                                          multi_class="multinomial")
                try:
                    lr_p.fit(X_tr, y_perm)
                    perm_bal_lr.append(float(balanced_accuracy_score(y_te, lr_p.predict(X_te))))
                except Exception:
                    perm_bal_lr.append(float("nan"))
                dt_p = DecisionTreeClassifier(max_depth=3, random_state=pi)
                try:
                    dt_p.fit(X_tr, y_perm)
                    perm_bal_dt.append(float(balanced_accuracy_score(y_te, dt_p.predict(X_te))))
                except Exception:
                    perm_bal_dt.append(float("nan"))

            perm_arr_lr = [v for v in perm_bal_lr if not np.isnan(v)]
            perm_arr_dt = [v for v in perm_bal_dt if not np.isnan(v)]
            pval_lr = float(np.mean([v >= bal_lr for v in perm_arr_lr])) if perm_arr_lr else float("nan")
            pval_dt = float(np.mean([v >= bal_dt for v in perm_arr_dt])) if perm_arr_dt else float("nan")
            perm_pass_lr = pval_lr < 0.05 if not np.isnan(pval_lr) else False
            perm_pass_dt = pval_dt < 0.05 if not np.isnan(pval_dt) else False

            sc_results[direction] = {
                "logistic_bal_acc":   bal_lr,
                "logistic_f1_macro":  f1_lr,
                "tree_bal_acc":       bal_dt,
                "tree_f1_macro":      f1_dt,
                "majority_baseline_bal_acc":       bal_majority,
                "global_predict_label_bal_acc":    bal_predict_baseline,
                "cm_logistic":        cm_lr,
                "cm_tree":            cm_dt,
                "n_train":            int(len(y_tr)),
                "n_test":             int(len(y_te)),
                "n_classes_train":    int(len(np.unique(y_tr))),
                "majority_class_label": ACTION_FROM_INT.get(majority_class, str(majority_class)),
                "perm_pval_logistic":  pval_lr,
                "perm_pval_tree":      pval_dt,
                "perm_pass_logistic":  perm_pass_lr,
                "perm_pass_tree":      perm_pass_dt,
                "perm_mean_bal_lr":    float(np.mean(perm_arr_lr)) if perm_arr_lr else float("nan"),
                "perm_mean_bal_dt":    float(np.mean(perm_arr_dt)) if perm_arr_dt else float("nan"),
            }

        results[scenario] = sc_results

    return results


def run_classifier_probes_non_near_tie(region_rows, seeds_a, seeds_b):
    """Repeat classifier probes on non-near-tie subset."""
    nnt = [r for r in region_rows if abs(r["action_margin"]) > MARGIN_EPS]
    return run_classifier_probes(nnt, seeds_a, seeds_b, n_permutations=100)


# ============================================================
# Separability decision
# ============================================================

def decide_separability(probes_all, probes_nnt):
    """Decide V2_ACTION_SEPARABILITY from cross-block probe results."""
    supported_count  = 0
    partial_count    = 0
    total_directions = 0
    strongest_base   = float("nan")

    for scenario in SCENARIOS + ["POOLED"]:
        for direction in ["A_to_B", "B_to_A"]:
            sc_dir = probes_all.get(scenario, {}).get(direction, {})
            if "error" in sc_dir:
                continue
            total_directions += 1
            bal_lr = sc_dir.get("logistic_bal_acc", float("nan"))
            bal_dt = sc_dir.get("tree_bal_acc", float("nan"))
            bal_maj = sc_dir.get("majority_baseline_bal_acc", float("nan"))
            bal_pred = sc_dir.get("global_predict_label_bal_acc", float("nan"))
            perm_lr = sc_dir.get("perm_pass_logistic", False)
            perm_dt = sc_dir.get("perm_pass_tree", False)
            f1_lr = sc_dir.get("logistic_f1_macro", float("nan"))
            f1_dt = sc_dir.get("tree_f1_macro", float("nan"))

            stronger_base = max(bal_maj, bal_pred) if not (np.isnan(bal_maj) or np.isnan(bal_pred)) else float("nan")
            if not np.isnan(stronger_base):
                strongest_base = max(stronger_base, strongest_base) if not np.isnan(strongest_base) else stronger_base

            # Check non-near-tie signal
            nnt_dir = probes_nnt.get(scenario, {}).get(direction, {})
            nnt_bal_lr = nnt_dir.get("logistic_bal_acc", float("nan"))
            nnt_perm = nnt_dir.get("perm_pass_logistic", False)

            margin_above = (not np.isnan(stronger_base) and not np.isnan(bal_lr) and
                            bal_lr >= stronger_base + BAL_ACC_MARGIN)
            f1_above_perm = (perm_lr or perm_dt)
            nnt_signal = (not np.isnan(nnt_bal_lr) and not np.isnan(stronger_base) and
                          nnt_bal_lr >= stronger_base + BAL_ACC_MARGIN)

            if margin_above and f1_above_perm and nnt_signal:
                supported_count += 1
            elif margin_above or f1_above_perm:
                partial_count += 1

    if total_directions == 0:
        return "ABSENT"
    if supported_count >= 4:  # majority of 8 (POOLED + 3 scenarios × 2 directions)
        return "SUPPORTED"
    if supported_count + partial_count >= 4:
        return "PARTIAL"
    return "ABSENT"


# ============================================================
# H1 outcome decision
# ============================================================

def decide_h1_outcome(headroom, separability, predict_dominance):
    if headroom == "ABSENT" or predict_dominance == "STRONG":
        outcome = "GLOBAL_ACTION_DOMINANCE"
        next_stage = "REGIONAL_CONTROL_REDESIGN_OR_STOP"
    elif headroom in ("SUPPORTED", "PARTIAL") and separability == "ABSENT":
        outcome = "STATE_INFORMATION_INSUFFICIENT"
        next_stage = "STATE_REPRESENTATION_HYPOTHESIS_REVIEW"
    elif headroom == "SUPPORTED" and separability == "SUPPORTED" and predict_dominance != "STRONG":
        outcome = "RULE_MAPPING_INADEQUATE"
        next_stage = "E4_LEARNED_CONTROLLER_PREPARATION"
    else:
        outcome = "INCONCLUSIVE"
        next_stage = "H1_REVIEW"
    return outcome, next_stage


# ============================================================
# Output writers
# ============================================================

def _jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _hash_file(path):
    sha = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_oracle_event_summary(event_rows, path):
    fns = ["scenario", "seed", "block", "k", "n_active_regions",
           "ROG", "ROG_vs_GLOBAL_PREDICT", "OHI", "best_global_action",
           "Q_keep", "Q_best_global", "Q_regional",
           "Q_GLOBAL_PREDICT", "Q_GLOBAL_LOCAL", "Q_GLOBAL_DIVERSIFY"]
    rows = [{fn: r.get(fn, "") for fn in fns} for r in event_rows]
    write_csv(rows, fns, path)


def write_seed_summary(event_rows, region_rows, seeds, path):
    fns = ["scenario", "seed", "block",
           "mean_ROG", "mean_ROG_vs_GLOBAL_PREDICT", "mean_OHI",
           "n_events", "n_region_records",
           "frac_heterogeneous_events", "frac_non_PREDICT_oracle_labels"]
    rows_out = []
    for seed in seeds:
        blk = "A" if seed in BLOCK_A_SEEDS else "B"
        for sc in SCENARIOS:
            ev = [r for r in event_rows if r["seed"] == seed and r["scenario"] == sc]
            rr = [r for r in region_rows if r["seed"] == seed and r["scenario"] == sc]
            if not ev:
                continue
            frac_het = sum(1 for e in ev if e["OHI"] > 0) / len(ev)
            frac_non_pred = (sum(1 for r in rr if r["oracle_best_action"] != "PREDICT") / len(rr)
                             if rr else float("nan"))
            rows_out.append({
                "scenario": sc, "seed": seed, "block": blk,
                "mean_ROG":              float(np.mean([e["ROG"] for e in ev])),
                "mean_ROG_vs_GLOBAL_PREDICT": float(np.mean([e["ROG_vs_GLOBAL_PREDICT"] for e in ev])),
                "mean_OHI":              float(np.mean([e["OHI"] for e in ev])),
                "n_events":              len(ev),
                "n_region_records":      len(rr),
                "frac_heterogeneous_events": frac_het,
                "frac_non_PREDICT_oracle_labels": frac_non_pred,
            })
    write_csv(rows_out, fns, path)


def write_oracle_headroom_csv(headroom_results, path):
    fns = ["block", "scenario", "n_seeds", "mean_ROG", "ci_lo", "ci_hi", "classification"]
    rows_out = []
    for blk_lbl, blk_data in headroom_results.items():
        for sc, d in blk_data.items():
            rows_out.append({"block": blk_lbl, "scenario": sc, **d})
    write_csv(rows_out, fns, path)


def write_action_distribution(region_rows, path):
    fns = ["scenario", "block", "action", "count", "fraction"]
    rows_out = []
    for sc in SCENARIOS:
        for blk in ["A", "B", "POOLED"]:
            if blk == "POOLED":
                sub = [r for r in region_rows if r["scenario"] == sc]
            else:
                sub = [r for r in region_rows if r["scenario"] == sc and r["block"] == blk]
            if not sub:
                continue
            total = len(sub)
            for act in ACTIONS:
                cnt = sum(1 for r in sub if r["oracle_best_action"] == act)
                rows_out.append({"scenario": sc, "block": blk, "action": act,
                                 "count": cnt, "fraction": round(cnt / total, 4)})
    write_csv(rows_out, fns, path)


def write_action_margin(region_rows, path):
    fns = ["scenario", "oracle_best_action", "n", "frac_near_tie",
           "median_margin", "q25_margin", "q75_margin"]
    rows_out = []
    for sc in SCENARIOS + ["POOLED"]:
        sub = region_rows if sc == "POOLED" else [r for r in region_rows if r["scenario"] == sc]
        for act in ACTIONS:
            act_sub = [r for r in sub if r["oracle_best_action"] == act]
            if not act_sub:
                continue
            margins = [r["action_margin"] for r in act_sub if not np.isnan(r["action_margin"])]
            if not margins:
                continue
            m_arr = np.array(margins)
            frac_nt = float(np.mean(m_arr <= MARGIN_EPS))
            rows_out.append({
                "scenario": sc, "oracle_best_action": act, "n": len(margins),
                "frac_near_tie": frac_nt,
                "median_margin": float(np.median(m_arr)),
                "q25_margin":    float(np.percentile(m_arr, 25)),
                "q75_margin":    float(np.percentile(m_arr, 75)),
            })
    write_csv(rows_out, fns, path)


def write_global_predict_audit(event_rows, region_rows, path):
    fns = ["scenario", "block",
           "frac_regional_oracle_is_PREDICT", "frac_events_best_global_is_PREDICT",
           "mean_ROG_vs_GLOBAL_PREDICT"]
    rows_out = []
    for sc in SCENARIOS:
        for blk in ["A", "B", "POOLED"]:
            if blk == "POOLED":
                ev_sub = [r for r in event_rows if r["scenario"] == sc]
                rr_sub = [r for r in region_rows if r["scenario"] == sc]
            else:
                ev_sub = [r for r in event_rows if r["scenario"] == sc and r["block"] == blk]
                rr_sub = [r for r in region_rows if r["scenario"] == sc and r["block"] == blk]
            if not ev_sub:
                continue
            frac_reg = (sum(1 for r in rr_sub if r["oracle_best_action"] == "PREDICT") / len(rr_sub)
                        if rr_sub else float("nan"))
            frac_ev  = sum(1 for e in ev_sub if e["best_global_action"] == "PREDICT") / len(ev_sub)
            mean_rog_pred = float(np.mean([e["ROG_vs_GLOBAL_PREDICT"] for e in ev_sub]))
            rows_out.append({
                "scenario": sc, "block": blk,
                "frac_regional_oracle_is_PREDICT": frac_reg,
                "frac_events_best_global_is_PREDICT": frac_ev,
                "mean_ROG_vs_GLOBAL_PREDICT": mean_rog_pred,
            })
    write_csv(rows_out, fns, path)


def write_v2_label_feature_summary(region_rows, path):
    fns = ["scenario", "oracle_best_action", "n", "frac_total",
           "med_severity", "iqr_severity",
           "med_convergence_loss", "iqr_convergence_loss",
           "med_occupancy", "iqr_occupancy",
           "med_delta_severity", "iqr_delta_severity"]
    rows_out = []
    for sc in SCENARIOS + ["POOLED"]:
        sub = region_rows if sc == "POOLED" else [r for r in region_rows if r["scenario"] == sc]
        total = len(sub)
        for act in ACTIONS:
            act_sub = [r for r in sub if r["oracle_best_action"] == act]
            if not act_sub:
                continue
            def q25_q75(arr): return np.percentile(arr, 25), np.percentile(arr, 75)
            s = np.array([r["severity_r"] for r in act_sub])
            l = np.array([r["convergence_loss_r"] for r in act_sub])
            o = np.array([r["occupancy_r"] for r in act_sub])
            d = np.array([r["delta_severity_r"] for r in act_sub])
            s25, s75 = q25_q75(s); l25, l75 = q25_q75(l)
            o25, o75 = q25_q75(o); d25, d75 = q25_q75(d)
            rows_out.append({
                "scenario": sc, "oracle_best_action": act,
                "n": len(act_sub), "frac_total": round(len(act_sub)/total, 4),
                "med_severity": float(np.median(s)), "iqr_severity": float(s75-s25),
                "med_convergence_loss": float(np.median(l)), "iqr_convergence_loss": float(l75-l25),
                "med_occupancy": float(np.median(o)), "iqr_occupancy": float(o75-o25),
                "med_delta_severity": float(np.median(d)), "iqr_delta_severity": float(d75-d25),
            })
    write_csv(rows_out, fns, path)


def write_mixed_within_event(region_rows, event_rows, path):
    """Per-event MIXED stats: unique non-near-tie actions, fractions."""
    fns = ["scenario", "seed", "block", "k", "n_active_regions",
           "n_unique_nnt_actions", "frac_PREDICT", "frac_KEEP",
           "frac_LOCAL", "frac_DIVERSIFY",
           "ROG", "ROG_vs_GLOBAL_PREDICT"]
    mixed_ev = [e for e in event_rows if e["scenario"] == "MIXED"]
    mixed_rr = [r for r in region_rows if r["scenario"] == "MIXED"]

    rows_out = []
    for ev in mixed_ev:
        key = (ev["seed"], ev["scenario"], ev["k"])
        ev_rr = [r for r in mixed_rr if r["seed"] == ev["seed"] and r["k"] == ev["k"]]
        nnt   = [r for r in ev_rr if abs(r["action_margin"]) > MARGIN_EPS]
        unique_nnt = len(set(r["oracle_best_action"] for r in nnt)) if nnt else 0
        all_labels = [r["oracle_best_action"] for r in ev_rr] if ev_rr else []
        total_l = len(all_labels) or 1
        rows_out.append({
            "scenario": "MIXED", "seed": ev["seed"], "block": ev["block"],
            "k": ev["k"], "n_active_regions": ev["n_active_regions"],
            "n_unique_nnt_actions": unique_nnt,
            "frac_PREDICT":   sum(1 for a in all_labels if a == "PREDICT") / total_l,
            "frac_KEEP":      sum(1 for a in all_labels if a == "KEEP")    / total_l,
            "frac_LOCAL":     sum(1 for a in all_labels if a == "LOCAL")   / total_l,
            "frac_DIVERSIFY": sum(1 for a in all_labels if a == "DIVERSIFY") / total_l,
            "ROG":                   ev["ROG"],
            "ROG_vs_GLOBAL_PREDICT": ev["ROG_vs_GLOBAL_PREDICT"],
        })
    write_csv(rows_out, fns, path)


def write_scenario_summary(headroom_results, predict_dom_per_sc, probes, path):
    summary = {}
    for sc in SCENARIOS:
        pooled = headroom_results.get("POOLED", {}).get(sc, {})
        summary[sc] = {
            "REGIONAL_ORACLE_HEADROOM": pooled.get("classification", "UNKNOWN"),
            "pooled_ROG_mean": pooled.get("mean_ROG", float("nan")),
            "pooled_ROG_CI":   [pooled.get("ci_lo", float("nan")), pooled.get("ci_hi", float("nan"))],
            "PREDICT_oracle_frac": predict_dom_per_sc.get(sc, float("nan")),
        }
    _jdump(summary, path)


def write_report(
    headroom_overall, headroom_per_sc, separability, predict_dominance,
    h1_outcome, next_stage, predict_frac,
    probes_all, pooled_headroom, path
):
    lines = [
        "# H1 — Regional Controller Hypothesis Review: Report",
        "",
        f"**Stage**: H1  **Date**: 2026-09-20  "
        f"**RL_STATUS**: {RL_STATUS}  **DRL_TRAINING**: {DRL_TRAINING}",
        f"**H1_OUTCOME**: {h1_outcome}",
        "",
        "## A. Why H1",
        "",
        "E3-R Gate B = FAIL on the correct V2 state. H1 diagnoses whether the failure "
        "was due to: (A) lack of regional oracle headroom, (B) V2 state insufficient "
        "to identify headroom, or (C) rule mapping inadequate.",
        "",
        "## B. Oracle Protocol",
        "",
        f"quality_fn = -IGD(F, PF*). Carrier = GLOBAL_LOCAL (neutral, predeclared). "
        f"Seeds: {DIAG_SEEDS[0]}..{DIAG_SEEDS[-1]} (Block A: {BLOCK_A_SEEDS[0]}-{BLOCK_A_SEEDS[-1]}, "
        f"Block B: {BLOCK_B_SEEDS[0]}-{BLOCK_B_SEEDS[-1]}). "
        f"MARGIN_EPS={MARGIN_EPS}. DELTA_H1_ROG={DELTA_H1_ROG}. BAL_ACC_MARGIN={BAL_ACC_MARGIN}.",
        "",
        "## C. Oracle Headroom (ROG)",
        "",
        f"REGIONAL_ORACLE_HEADROOM (overall) = {headroom_overall}",
        "",
    ]
    for sc in SCENARIOS:
        p = pooled_headroom.get(sc, {})
        lines.append(f"  {sc}: {p.get('classification','?')} "
                     f"mean={p.get('mean_ROG',float('nan')):.6f} "
                     f"CI=[{p.get('ci_lo',float('nan')):.6f},{p.get('ci_hi',float('nan')):.6f}]")
    lines += [
        "",
        "## D. Global PREDICT Dominance",
        "",
        f"GLOBAL_PREDICT_DOMINANCE = {predict_dominance}",
        f"Non-near-tie PREDICT fraction (pooled) = {predict_frac:.4f}" if not np.isnan(predict_frac) else "Non-near-tie PREDICT fraction = N/A",
        "",
        "## E. V2 Action Separability",
        "",
        f"V2_ACTION_SEPARABILITY = {separability}",
        "",
    ]
    for sc in SCENARIOS + ["POOLED"]:
        for direction in ["A_to_B", "B_to_A"]:
            d = probes_all.get(sc, {}).get(direction, {})
            if "error" not in d:
                lines.append(
                    f"  {sc} {direction}: LR_bal={d.get('logistic_bal_acc',float('nan')):.4f} "
                    f"DT_bal={d.get('tree_bal_acc',float('nan')):.4f} "
                    f"maj_base={d.get('majority_baseline_bal_acc',float('nan')):.4f} "
                    f"perm_pass_LR={d.get('perm_pass_logistic',False)}"
                )
    lines += [
        "",
        "## F. H1 Outcome",
        "",
        f"H1_OUTCOME = {h1_outcome}",
        f"NEXT_STAGE = {next_stage}",
        "",
        "## G. What This Does Not Prove",
        "",
        "This is a diagnostic stage only. Classifiers are probes, not deployed controllers. "
        "No RULE_V2, no DQN, no training. If H1_OUTCOME = RULE_MAPPING_INADEQUATE, "
        "a learned controller is scientifically motivated but not yet built.",
        "",
        "## H. Invariants",
        "",
        f"RL_STATUS = {RL_STATUS}. DRL_TRAINING = {DRL_TRAINING}. "
        "Production code unchanged. E3/E3-R verdicts not modified.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    pf_star = get_true_pf(n_points=500)

    print("H1 — Regional Controller Hypothesis Review")
    print(f"  D={D}, POP_SIZE={POP_SIZE}, R={R}")
    print(f"  Diagnostic seeds: {DIAG_SEEDS[0]}-{DIAG_SEEDS[-1]}")
    print(f"  Block A: {BLOCK_A_SEEDS[0]}-{BLOCK_A_SEEDS[-1]}, Block B: {BLOCK_B_SEEDS[0]}-{BLOCK_B_SEEDS[-1]}")
    print(f"  DELTA_H1_ROG={DELTA_H1_ROG}, MARGIN_EPS={MARGIN_EPS}, BAL_ACC_MARGIN={BAL_ACC_MARGIN}")
    print(f"  Carrier: {CARRIER_CONTROLLER}")

    # Seed sanity: no overlap
    assert not (set(DIAG_SEEDS) & E3_SEEDS),  "E3 seed overlap!"
    assert not (set(DIAG_SEEDS) & E3R_SEEDS), "E3-R seed overlap!"

    # ── H1-0: Pre-run manifest ───────────────────────────────────
    print("\n[H1-0] Pre-run manifest")
    key_files = [
        "src/prdrl/detectors/severity.py",
        "src/prdrl/detectors/quality_loss.py",
        "src/prdrl/rl/interface.py",
        "src/prdrl/oracle/oracle.py",
        "src/prdrl/operators/budget.py",
        "src/prdrl/operators/local.py",
        "src/prdrl/operators/predict.py",
        "src/prdrl/operators/diversify.py",
        "src/prdrl/operators/survival.py",
        "results/E3R_rule_v2_semantic_rerun/E3R_verdict.json",
        "results/R0_state_semantics_reconciliation/SM_verdict.json",
    ]
    manifest = {
        "stage": "H1_regional_controller_hypothesis_review",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "STATE_SCHEMA_VERSION": "V2",
        "STATE_FEATURES": ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"],
        "STATE_DIM": 4,
        "E3R_GATE_B": "FAIL",
        "DIAG_SEEDS": [DIAG_SEEDS[0], DIAG_SEEDS[-1]],
        "BLOCK_A": [BLOCK_A_SEEDS[0], BLOCK_A_SEEDS[-1]],
        "BLOCK_B": [BLOCK_B_SEEDS[0], BLOCK_B_SEEDS[-1]],
        "excluded_seeds": {"E3": "701-780", "E3R": "801-882"},
        "carrier": CARRIER_CONTROLLER,
        "quality_fn": "negative_IGD",
        "oracle_utility_convention": "U_{r,a} = Q(P^{r,a}) - Q(P^KEEP)",
        "oracle_tie_rule": "KEEP wins strict ties (KEEP preferred iff all others <= 0)",
        "statistical_unit": "seed",
        "DELTA_H1_ROG": DELTA_H1_ROG,
        "MARGIN_EPS": MARGIN_EPS,
        "BAL_ACC_MARGIN": BAL_ACC_MARGIN,
        "bootstrap": {"BLOCK": BOOTSTRAP_RESAMPLES_BLOCK, "POOLED": BOOTSTRAP_RESAMPLES_POOLED},
        "action_semantics": {
            "KEEP":      "FE=0",
            "LOCAL":     f"sigma_L={SIGMA_LOCAL}",
            "PREDICT":   f"sigma_P={SIGMA_PREDICT}, lambda_P={LAMBDA_PREDICT}, UNRESTRICTED survival",
            "DIVERSIFY": f"sigma_D={SIGMA_DIVERSIFY}",
        },
        "no_HARD": True, "no_B_r_a": True, "no_DRL": True, "no_RULE_V2": True,
        "production_code_unchanged": True,
        "e3r_verdict_immutable": True,
        "decision_matrix": {
            "GLOBAL_ACTION_DOMINANCE":       "headroom ABSENT or predict_dominance STRONG",
            "STATE_INFORMATION_INSUFFICIENT": "headroom SUPPORTED/PARTIAL, separability ABSENT",
            "RULE_MAPPING_INADEQUATE":        "headroom SUPPORTED, separability SUPPORTED, predict != STRONG",
            "INCONCLUSIVE":                  "conflicting evidence",
        },
        "headroom_classification": {
            "SUPPORTED":   "mean > DELTA_H1_ROG AND ci_lo > 0",
            "DIRECTIONAL": "mean > DELTA_H1_ROG, CI crosses 0",
            "NEUTRAL":     "abs(mean) <= DELTA_H1_ROG",
            "CONTRADICTED": "mean < -DELTA_H1_ROG AND ci_hi < 0",
        },
        "separability_classification": {
            "SUPPORTED": ">=4 of 8 cross-block directions: bal_acc >= stronger_base + BAL_ACC_MARGIN AND perm_pass AND nnt_signal",
            "PARTIAL":   ">=4 partial",
            "ABSENT":    "otherwise",
        },
        "predict_dominance": {
            "STRONG":   ">=80% non-near-tie labels PREDICT AND ROG_vs_PREDICT not SUPPORTED",
            "MODERATE": "60-80% PREDICT or mixed ROG_vs_PREDICT",
            "WEAK":     "<60% PREDICT AND headroom SUPPORTED vs PREDICT",
        },
        "files": {fp: _hash_file(fp) for fp in key_files},
    }
    _jdump(manifest, RESULTS_DIR / "H1_pre_run_manifest.json")
    print(f"  oracle.py: {manifest['files'].get('src/prdrl/oracle/oracle.py','?')[:16]}...")

    # ── Smoke test ───────────────────────────────────────────────
    print("\n[H1-smoke] Smoke (seeds 901-902, PERSISTENT, 2 changes)")
    smoke_seeds = [901, 902]
    s_rr, s_ev = run_all_seeds(smoke_seeds, 2, xl, xu, V, R, pf_star, is_smoke=True)
    exp_smoke_ev = 2 * 3 * 2   # seeds × scenarios × changes
    assert len(s_ev) == exp_smoke_ev, f"Smoke event count: {len(s_ev)} != {exp_smoke_ev}"
    print(f"  Smoke events: {len(s_ev)} (expected {exp_smoke_ev}) — PASS")

    # ── Full diagnostic run ──────────────────────────────────────
    print(f"\n[H1-main] Full run (seeds {DIAG_SEEDS[0]}-{DIAG_SEEDS[-1]})")
    t_run = time.time()
    all_region, all_event = run_all_seeds(
        DIAG_SEEDS, N_SCORED_CHANGES, xl, xu, V, R, pf_star, is_smoke=False
    )
    print(f"  Done in {time.time()-t_run:.1f}s")
    exp_events = len(DIAG_SEEDS) * len(SCENARIOS) * N_SCORED_CHANGES
    assert len(all_event) == exp_events, f"Event count: {len(all_event)} != {exp_events}"

    # ── Write core outputs ───────────────────────────────────────
    print("\n[H1-out] Writing output files")
    write_jsonl(all_region, RESULTS_DIR / "H1_oracle_region_records.jsonl")
    write_oracle_event_summary(all_event,  RESULTS_DIR / "H1_oracle_event_summary.csv")
    write_seed_summary(all_event, all_region, DIAG_SEEDS,
                       RESULTS_DIR / "H1_seed_summary.csv")
    write_action_distribution(all_region, RESULTS_DIR / "H1_action_distribution.csv")
    write_action_margin(all_region,        RESULTS_DIR / "H1_action_margin.csv")
    write_global_predict_audit(all_event, all_region,
                               RESULTS_DIR / "H1_global_predict_audit.csv")
    write_v2_label_feature_summary(all_region,
                                   RESULTS_DIR / "H1_v2_label_feature_summary.csv")
    write_mixed_within_event(all_region, all_event,
                             RESULTS_DIR / "H1_mixed_within_event.csv")

    # ── Statistics ───────────────────────────────────────────────
    print("\n[H1-stats] Bootstrap (paired seed inference)")
    headroom_results = {}
    headroom_vs_predict_results = {}
    for blk_lbl, seeds, n_res in [
        ("BLOCK_A", BLOCK_A_SEEDS, BOOTSTRAP_RESAMPLES_BLOCK),
        ("BLOCK_B", BLOCK_B_SEEDS, BOOTSTRAP_RESAMPLES_BLOCK + 1),
        ("POOLED",  DIAG_SEEDS,    BOOTSTRAP_RESAMPLES_POOLED),
    ]:
        headroom_results[blk_lbl] = {}
        headroom_vs_predict_results[blk_lbl] = {}
        for sc in SCENARIOS:
            headroom_results[blk_lbl][sc] = compute_headroom_stat(
                all_event, seeds, sc, n_res, BOOTSTRAP_SEED + hash(blk_lbl + sc) % 9999)
            headroom_vs_predict_results[blk_lbl][sc] = compute_headroom_stat_vs_predict(
                all_event, seeds, sc, n_res, BOOTSTRAP_SEED + hash("vsp" + blk_lbl + sc) % 9999)

    write_oracle_headroom_csv(headroom_results, RESULTS_DIR / "H1_oracle_headroom.csv")

    pooled_stats = headroom_results["POOLED"]
    block_a_stats = headroom_results["BLOCK_A"]
    block_b_stats = headroom_results["BLOCK_B"]
    headroom_overall, headroom_per_sc = decide_headroom(pooled_stats, block_a_stats, block_b_stats)

    for sc in SCENARIOS:
        p = pooled_stats[sc]
        vs = headroom_vs_predict_results["POOLED"][sc]
        print(f"  {sc}: ROG={p['mean_ROG']:.5f} CI=[{p['ci_lo']:.5f},{p['ci_hi']:.5f}] "
              f"{p['classification']} | ROG_vs_PREDICT={vs['mean']:.5f} {vs['classification']}")
    print(f"  Overall headroom: {headroom_overall}")

    # ── PREDICT dominance ────────────────────────────────────────
    predict_frac = compute_predict_dominance(all_region)
    pooled_rog_vs_pred_cls = headroom_vs_predict_results["POOLED"].get(
        "MIXED", {}).get("classification", "UNKNOWN")
    predict_dominance = decide_predict_dominance(predict_frac, pooled_rog_vs_pred_cls)

    predict_dom_per_sc = {}
    for sc in SCENARIOS:
        nnt = [r for r in all_region if r["scenario"] == sc and abs(r["action_margin"]) > MARGIN_EPS]
        predict_dom_per_sc[sc] = (
            sum(1 for r in nnt if r["oracle_best_action"] == "PREDICT") / len(nnt)
            if nnt else float("nan")
        )

    print(f"  GLOBAL_PREDICT_DOMINANCE = {predict_dominance} (frac={predict_frac:.4f})")

    # ── Classifier probes ────────────────────────────────────────
    print("\n[H1-probes] Classifier probes (cross-block)")
    probes_all = run_classifier_probes(all_region, BLOCK_A_SEEDS, BLOCK_B_SEEDS)
    probes_nnt = run_classifier_probes_non_near_tie(all_region, BLOCK_A_SEEDS, BLOCK_B_SEEDS)
    _jdump(probes_all, RESULTS_DIR / "H1_separability_crossblock.json")
    _jdump(probes_nnt, RESULTS_DIR / "H1_permutation_control.json")

    separability = decide_separability(probes_all, probes_nnt)
    print(f"  V2_ACTION_SEPARABILITY = {separability}")

    # Print cross-block results
    for sc in SCENARIOS + ["POOLED"]:
        for direction in ["A_to_B", "B_to_A"]:
            d = probes_all.get(sc, {}).get(direction, {})
            if "error" not in d:
                print(f"    {sc} {direction}: LR_bal={d['logistic_bal_acc']:.4f} "
                      f"DT_bal={d['tree_bal_acc']:.4f} "
                      f"maj={d['majority_baseline_bal_acc']:.4f} "
                      f"pred_base={d['global_predict_label_bal_acc']:.4f} "
                      f"perm_LR={d['perm_pass_logistic']}")

    # ── Decision + outcome ───────────────────────────────────────
    h1_outcome, next_stage = decide_h1_outcome(
        headroom_overall, separability, predict_dominance
    )

    # ── Write remaining outputs ───────────────────────────────────
    write_scenario_summary(headroom_results, predict_dom_per_sc, probes_all,
                           RESULTS_DIR / "H1_scenario_summary.json")

    decision_matrix = {
        "REGIONAL_ORACLE_HEADROOM": headroom_overall,
        "REGIONAL_ORACLE_HEADROOM_PER_SCENARIO": headroom_per_sc,
        "GLOBAL_PREDICT_DOMINANCE": predict_dominance,
        "NON_NEAR_TIE_PREDICT_FRAC": float(predict_frac),
        "V2_ACTION_SEPARABILITY": separability,
        "H1_OUTCOME": h1_outcome,
        "NEXT_STAGE": next_stage,
        "pooled_ROG": {sc: pooled_stats[sc] for sc in SCENARIOS},
        "pooled_ROG_vs_PREDICT": {sc: headroom_vs_predict_results["POOLED"][sc] for sc in SCENARIOS},
        "BAL_ACC_MARGIN_FROZEN": BAL_ACC_MARGIN,
        "MARGIN_EPS_FROZEN": MARGIN_EPS,
        "DELTA_H1_ROG_FROZEN": DELTA_H1_ROG,
    }
    _jdump(decision_matrix, RESULTS_DIR / "H1_decision_matrix.json")

    pooled_rog_mixed = pooled_stats.get("MIXED", {})
    rog_vs_pred_pooled_mixed = headroom_vs_predict_results["POOLED"].get("MIXED", {})

    # Permutation control summary
    perm_pass_any = any(
        probes_all.get(sc, {}).get(dir_, {}).get("perm_pass_logistic", False)
        or probes_all.get(sc, {}).get(dir_, {}).get("perm_pass_tree", False)
        for sc in SCENARIOS + ["POOLED"]
        for dir_ in ["A_to_B", "B_to_A"]
    )
    perm_status = "PASS" if perm_pass_any else "FAIL"

    # Crossblock summaries
    def _dir_val(probes, metric):
        vals = []
        for sc in ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = probes.get(sc, {}).get(direction, {})
                if "error" not in d and metric in d:
                    vals.append(d[metric])
        return vals

    lr_vals = _dir_val(probes_all, "logistic_bal_acc")
    dt_vals = _dir_val(probes_all, "tree_bal_acc")
    maj_vals = _dir_val(probes_all, "majority_baseline_bal_acc")
    pred_base_vals = _dir_val(probes_all, "global_predict_label_bal_acc")

    verdict = {
        "stage": "H1_regional_controller_hypothesis_review",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "STATE_SCHEMA_VERSION": "V2",
        "DIAG_SEEDS": f"{DIAG_SEEDS[0]}..{DIAG_SEEDS[-1]}",
        "quality_fn": "negative_IGD",
        "CARRIER": CARRIER_CONTROLLER,
        "DELTA_H1_ROG": DELTA_H1_ROG,
        "MARGIN_EPS": MARGIN_EPS,
        "BAL_ACC_MARGIN": BAL_ACC_MARGIN,
        "REGIONAL_ORACLE_HEADROOM": headroom_overall,
        "REGIONAL_ORACLE_HEADROOM_PER_SCENARIO": headroom_per_sc,
        "POOLED_ROG_MIXED_MEAN":  pooled_rog_mixed.get("mean_ROG", float("nan")),
        "POOLED_ROG_MIXED_CI":    [pooled_rog_mixed.get("ci_lo", float("nan")),
                                    pooled_rog_mixed.get("ci_hi", float("nan"))],
        "POOLED_ROG_VS_GLOBAL_PREDICT_MIXED": rog_vs_pred_pooled_mixed.get("mean", float("nan")),
        "NON_NEAR_TIE_PREDICT_FRAC": float(predict_frac),
        "GLOBAL_PREDICT_DOMINANCE": predict_dominance,
        "V2_ACTION_SEPARABILITY": separability,
        "PERMUTATION_CONTROL": perm_status,
        "CROSSBLOCK_LOGISTIC_BAL_ACC_A_TO_B": lr_vals[0] if len(lr_vals) > 0 else float("nan"),
        "CROSSBLOCK_LOGISTIC_BAL_ACC_B_TO_A": lr_vals[1] if len(lr_vals) > 1 else float("nan"),
        "CROSSBLOCK_TREE_BAL_ACC_A_TO_B": dt_vals[0] if len(dt_vals) > 0 else float("nan"),
        "CROSSBLOCK_TREE_BAL_ACC_B_TO_A": dt_vals[1] if len(dt_vals) > 1 else float("nan"),
        "MAJORITY_BASELINE_BAL_ACC": float(np.mean(maj_vals)) if maj_vals else float("nan"),
        "GLOBAL_PREDICT_LABEL_BASELINE_BAL_ACC": float(np.mean(pred_base_vals)) if pred_base_vals else float("nan"),
        "H1_OUTCOME": h1_outcome,
        "E3R_STATUS": "ADMISSIBLE_GATE_B_FAIL",
        "PRODUCTION_DEFAULT": "O1_F005",
        "NEXT_STAGE": next_stage,
        "HISTORICAL_E3_STATUS": "VALID_AS_CENTROID_BASELINE_NOT_DRL_GATE",
        "NO_CLASSIFIER_ROLLOUT": True,
        "NO_RULE_V2": True, "NO_DQN": True,
    }
    _jdump(verdict, RESULTS_DIR / "H1_verdict.json")

    write_report(
        headroom_overall, headroom_per_sc, separability, predict_dominance,
        h1_outcome, next_stage, predict_frac, probes_all, pooled_stats,
        RESULTS_DIR / "H1_report.md",
    )

    # ── Final terminal output ────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print("H1 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"REGIONAL_ORACLE_HEADROOM = {headroom_overall}")
    print(f"POOLED_ROG_MEAN = {pooled_rog_mixed.get('mean_ROG', float('nan')):.6f}")
    print(f"POOLED_ROG_CI = [{pooled_rog_mixed.get('ci_lo',float('nan')):.6f},{pooled_rog_mixed.get('ci_hi',float('nan')):.6f}]")
    print(f"POOLED_ROG_VS_GLOBAL_PREDICT = {rog_vs_pred_pooled_mixed.get('mean',float('nan')):.6f}")
    print(f"REGIONAL_ACTION_HETEROGENEITY = OHI>0 fraction by scenario printed above")
    print(f"NON_NEAR_TIE_PREDICT_FRACTION = {predict_frac:.4f}")
    print(f"GLOBAL_PREDICT_DOMINANCE = {predict_dominance}")
    print(f"V2_ACTION_SEPARABILITY = {separability}")
    a2b_lr = lr_vals[0] if len(lr_vals) > 0 else float("nan")
    b2a_lr = lr_vals[1] if len(lr_vals) > 1 else float("nan")
    a2b_dt = dt_vals[0] if len(dt_vals) > 0 else float("nan")
    b2a_dt = dt_vals[1] if len(dt_vals) > 1 else float("nan")
    print(f"CROSSBLOCK_LOGISTIC_BAL_ACC = A_TO_B:{a2b_lr:.4f}, B_TO_A:{b2a_lr:.4f}")
    print(f"CROSSBLOCK_TREE_BAL_ACC = A_TO_B:{a2b_dt:.4f}, B_TO_A:{b2a_dt:.4f}")
    print(f"MAJORITY_BASELINE = {float(np.mean(maj_vals)) if maj_vals else float('nan'):.4f}")
    print(f"GLOBAL_PREDICT_LABEL_BASELINE = {float(np.mean(pred_base_vals)) if pred_base_vals else float('nan'):.4f}")
    print(f"PERMUTATION_CONTROL = {perm_status}")
    for sc in SCENARIOS:
        print(f"{'PERSISTENT' if sc == 'PERSISTENT' else sc}_ORACLE_HEADROOM = {headroom_per_sc.get(sc,'?')}")
    print(f"H1_OUTCOME = {h1_outcome}")
    print(f"E3R_STATUS = ADMISSIBLE_GATE_B_FAIL")
    print(f"PRODUCTION_DEFAULT = O1_F005")
    print(f"RL_STATUS = {RL_STATUS}")
    print(f"DRL_TRAINING = {DRL_TRAINING}")
    print(f"FULL_REGRESSION = (run separately)")
    print(f"NEXT_STAGE = {next_stage}")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
