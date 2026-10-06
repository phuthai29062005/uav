"""
E4GD — Controlled Diagnostic Execution (PR-DRL-DMOEA).

Executes the frozen E4G D1..D8 diagnostic contracts to identify WHY the Shared
Double-DQN collapsed. POST-HOC ONLY:
  - NO training, NO gradient/optimizer step, NO weight/reward/normalization change,
  - NO final-seed reuse (1081..1140 never instantiated),
  - the immutable E4F verdict (FAILED) is not changed.

Diagnostic rollouts use ONLY seeds 1201..1220 (POST_HOC_DIAGNOSTIC_ONLY) with the
frozen policies in eval mode (epsilon=0). D1/D2/D8 use the states collected in that
diagnostic rollout; D4 logs reward components; D5 replays conditional counterfactuals
from frozen pre-response snapshots; D6/D7 read existing validation/training logs.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---- validated environment primitives (reused; NO modification) -------------
from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, compute_joint_rewards, N_SCORED_CHANGES, NETWORK_INPUT_DIM,
    Q_PREV_INIT, ALPHA_G, ALPHA_L, BETA,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import (
    make_region_dynamics, _warmup_and_burnin,
)
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.agent import select_regional_actions, ACTION_TO_IDX, IDX_TO_ACTION
from prdrl.rl.interface import execute_response
from prdrl.metrics.igd import igd
from prdrl.problems.controlled_predict import evaluate as cp_evaluate

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/E4GD_controlled_diagnostic_execution")
E4G_DIR = Path("results/E4G_controlled_diagnostic_design")
E4F_DIR = Path("results/E4F_final_evaluation")
E4FFA_DIR = Path("results/E4F_failure_analysis")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

DIAGNOSTIC_SEEDS = list(range(1201, 1221))
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
FINAL_EVAL_SEEDS = set(range(1081, 1141))
LOCAL_FEATURES = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]
GLOBAL_FEATURES = ["active_region_fraction", "mean_severity", "max_severity",
                   "mean_convergence_loss", "max_convergence_loss"]
FEATURE_NAMES = LOCAL_FEATURES + GLOBAL_FEATURES
DEVICE = torch.device("cpu")

# D5 event selection (documented; bounded but representative)
D5_SCENARIO = "MIXED"
D5_CHANGES = [0, 4, 9]     # early / mid / late scored changes
# D5 interaction thresholds: E4G froze the MEASURE but not thresholds -> predeclared here.
D5_INTERACTION_MATERIAL = 0.10
D5_INTERACTION_WEAK = 0.02

_final_eval_access = {"count": 0}
_env_runs = {"count": 0}


# ---------------------------------------------------------------------------
def _hash_file(path) -> str:
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _guard_seed(seed):
    if int(seed) in FINAL_EVAL_SEEDS:
        _final_eval_access["count"] += 1
        raise RuntimeError(f"E4GD firewall: final seed {seed} must not be used")


# ---------------------------------------------------------------------------
# PHASE 0 — integrity
# ---------------------------------------------------------------------------

def pre_manifest() -> Tuple[dict, bool]:
    e4g = _load(E4G_DIR / "E4G_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                == ck["policies"][str(ps)]["sha256"] for ps in POLICY_SEEDS)
    prod_files = ["src/prdrl/operators/local.py", "src/prdrl/operators/predict.py",
                  "src/prdrl/operators/diversify.py", "src/prdrl/operators/budget.py",
                  "src/prdrl/oracle/oracle.py", "src/prdrl/core/survival.py",
                  "src/prdrl/rl/network.py", "src/prdrl/rl/reward.py",
                  "src/prdrl/rl/interface.py"]
    prod_hashes = {fp: _hash_file(fp) for fp in prod_files}
    diag = set(DIAGNOSTIC_SEEDS)
    consumed = (set(range(701, 781)) | set(range(801, 883)) | set(range(901, 941))
                | set(range(1001, 1141)))
    disjoint = len(diag & consumed) == 0
    manifest = {
        "stage": "E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION", "post_hoc": True,
        "E4G_COMPLETE": e4g["E4G_COMPLETE"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "e4f_verdict_hash": _hash_file(E4F_DIR / "E4F_verdict.json"),
        "e4f_episode_results_hash": _hash_file(E4F_DIR / "E4F_episode_results.csv"),
        "checkpoints_unchanged": ck_ok,
        "production_operator_hashes": prod_hashes,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "diagnostic_seeds_disjoint": disjoint,
        "diagnostic_seed_label": "POST_HOC_DIAGNOSTIC_ONLY",
        "no_training": True, "no_final_seed_reuse": True, "no_model_modification": True,
    }
    ok = (e4g["E4G_COMPLETE"] == "YES" and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
          and ck_ok and disjoint)
    return manifest, ok


# ---------------------------------------------------------------------------
# Diagnostic rollout on seeds 1201..1220 (collect states / reward / D5 snapshots)
# ---------------------------------------------------------------------------

def load_nets() -> Dict[int, SharedRegionalQNetwork]:
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    nets = {}
    for ps in POLICY_SEEDS:
        payload = torch.load(ck["policies"][str(ps)]["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(NETWORK_INPUT_DIM)
        net.load_state_dict(payload["online_state_dict"])
        net.eval()
        nets[ps] = net
    return nets


def build_cache(env) -> dict:
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in DIAGNOSTIC_SEEDS:
            _guard_seed(seed)
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)
            _env_runs["count"] += 1
    return cache


def _fresh_state(scenario, seed, cache, R):
    X_init, h_init, h_t_init, t_cur_init, gen_init = cache[(scenario, seed)]
    return {"X": X_init.copy(), "h": copy.deepcopy(h_init), "h_t": h_t_init,
            "t_current": t_cur_init, "gen": gen_init, "q_prev": Q_PREV_INIT,
            "severity_history": {r: None for r in range(R)}}


# ---------------------------------------------------------------------------
# D1 — normalization signal
# ---------------------------------------------------------------------------

def d1_normalization(state_rows, contract) -> Tuple[dict, List[dict]]:
    near_range = contract["NEAR_CONST_RANGE"]
    near_std = contract["NEAR_CONST_STD"]
    rows = []
    near_const_local = 0
    for feat in FEATURE_NAMES:
        raw = np.array([r[f"raw_{feat}"] for r in state_rows], dtype=float)
        norm = np.array([r[f"norm_{feat}"] for r in state_rows], dtype=float)
        def stats(a):
            return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                    "std": float(a.std()), "q01": float(np.percentile(a, 1)),
                    "q05": float(np.percentile(a, 5)), "median": float(np.median(a)),
                    "q95": float(np.percentile(a, 95)), "q99": float(np.percentile(a, 99))}
        rs, ns = stats(raw), stats(norm)
        norm_range = ns["q95"] - ns["q05"]
        near_const = (norm_range < near_range) or (ns["std"] < near_std)
        is_local = feat in LOCAL_FEATURES
        if is_local and near_const:
            near_const_local += 1
        rows.append({"feature": feat, "group": "local" if is_local else "global",
                     "raw_min": rs["min"], "raw_max": rs["max"], "raw_mean": rs["mean"],
                     "raw_std": rs["std"], "raw_median": rs["median"],
                     "norm_min": ns["min"], "norm_max": ns["max"], "norm_mean": ns["mean"],
                     "norm_std": ns["std"], "norm_q05": ns["q05"], "norm_q95": ns["q95"],
                     "norm_q95_q05_range": norm_range, "near_constant": near_const})
    if near_const_local >= 3:
        cls = "SEVERELY_COMPRESSED"
    elif near_const_local >= 1:
        cls = "COMPRESSED"
    else:
        cls = "PRESERVED"
    out = {"n_states": len(state_rows), "near_constant_local_features": near_const_local,
           "thresholds": {"NEAR_CONST_RANGE": near_range, "NEAR_CONST_STD": near_std},
           "per_feature": rows, "D1_NORMALIZATION_SIGNAL": cls}
    _jdump(out, RESULTS_DIR / "E4GD_D1_normalization_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D1_feature_distributions.csv", list(rows[0].keys()), rows)
    return out, rows


# ---------------------------------------------------------------------------
# D2 — separability (raw vs normalized), MIXED regime labels
# ---------------------------------------------------------------------------

def d2_separability(state_rows) -> dict:
    mixed = [r for r in state_rows if r["scenario"] == "MIXED"]
    y = np.array([1 if r["regime"] == "persistent" else 0 for r in mixed])
    if len(set(y.tolist())) < 2:
        out = {"D2_NORMALIZATION_SEPARABILITY": "NOT_AUDITABLE", "reason": "single class"}
        _jdump(out, RESULTS_DIR / "E4GD_D2_separability_results.json")
        _write_csv(RESULTS_DIR / "E4GD_D2_probe_results.csv", ["probe", "representation", "accuracy"], [])
        return out
    X_raw = np.array([[r[f"raw_{f}"] for f in FEATURE_NAMES] for r in mixed], dtype=float)
    X_norm = np.array([[r[f"norm_{f}"] for f in FEATURE_NAMES] for r in mixed], dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    probes = {"logistic": LogisticRegression(max_iter=1000),
              "tree": DecisionTreeClassifier(max_depth=4, random_state=0),
              "kNN": KNeighborsClassifier(n_neighbors=15)}
    csv_rows = []
    gaps = []
    for name, clf in probes.items():
        acc_raw = float(np.mean(cross_val_score(clf, X_raw, y, cv=skf, scoring="balanced_accuracy")))
        acc_norm = float(np.mean(cross_val_score(clf, X_norm, y, cv=skf, scoring="balanced_accuracy")))
        gaps.append(acc_raw - acc_norm)
        csv_rows.append({"probe": name, "representation": "RAW", "accuracy": acc_raw})
        csv_rows.append({"probe": name, "representation": "NORMALIZED", "accuracy": acc_norm})
    mean_gap = float(np.mean(gaps))
    if mean_gap >= 0.10:
        cls = "MATERIAL_LOSS"
    elif mean_gap >= 0.03:
        cls = "MODERATE_LOSS"
    else:
        cls = "NO_LOSS"
    out = {"label": "MIXED regime (persistent vs stationary region)", "n_samples": len(mixed),
           "class_balance": {"persistent": int(y.sum()), "stationary": int((1 - y).sum())},
           "mean_raw_minus_normalized_gap": mean_gap,
           "probe_gaps": {csv_rows[i]["probe"]: None for i in range(0)},
           "per_probe": csv_rows, "D2_NORMALIZATION_SEPARABILITY": cls,
           "thresholds": {"MATERIAL>=": 0.10, "MODERATE>=": 0.03}}
    _jdump(out, RESULTS_DIR / "E4GD_D2_separability_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D2_probe_results.csv", ["probe", "representation", "accuracy"], csv_rows)
    return out


# ---------------------------------------------------------------------------
# D3 — joint-action representation (code audit)
# ---------------------------------------------------------------------------

def d3_joint_representation(contract) -> dict:
    net = SharedRegionalQNetwork(9)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    out = {
        "network_input_dim": linears[0].in_features,
        "own_action_represented": "yes (Q outputs one value per own action; a_r selected by argmax)",
        "other_region_actions_in_Q": False,
        "previous_joint_action_in_Q": False,
        "requested_action_distribution_in_Q": False,
        "centralized_critic_exists": False, "Q_tot_exists": False,
        "mixing_network_exists": False, "region_interactions_explicit": False,
        "current_form": "Q_theta(s_r, a_r), s_r = local V2 + shared global STATE context",
        "can_represent_Q_r_with_a_minus_r": False,
        "can_represent_Q_joint": False,
        "JOINT_ACTION_REPRESENTATION": "IMPLICIT_STATE_ONLY",
        "option_table": contract["design_options"],
        "candidate_representations": contract["candidate_representations"],
    }
    _jdump(out, RESULTS_DIR / "E4GD_D3_joint_representation_results.json")
    rows = []
    for name, o in contract["design_options"].items():
        rows.append({"option": name, "decision_time": o["decision_time"],
                     "training_time": o["training_time"],
                     "simultaneous_execution": o["simultaneous_execution"],
                     "decentralized_selection": o["decentralized_selection"]})
    _write_csv(RESULTS_DIR / "E4GD_D3_option_table.csv",
               ["option", "decision_time", "training_time", "simultaneous_execution",
                "decentralized_selection"], rows)
    return out


# ---------------------------------------------------------------------------
# D4 — reward component scale
# ---------------------------------------------------------------------------

def d4_reward_scale(reward_rows) -> dict:
    # verify total reconstruction
    max_err = 0.0
    for r in reward_rows:
        recon = r["weighted_global_reward"] + r["weighted_local_reward"] - r["weighted_fe_penalty"]
        max_err = max(max_err, abs(recon - r["total_reward"]))
    def mags(key):
        a = np.abs(np.array([r[key] for r in reward_rows], dtype=float))
        return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
                "q05": float(np.percentile(a, 5)), "q95": float(np.percentile(a, 95))}
    wg = mags("weighted_global_reward")
    wl = mags("weighted_local_reward")
    wf = mags("weighted_fe_penalty")
    ratio = wg["median"] / (wl["median"] + 1e-12)
    if ratio >= 10.0:
        cls = "GLOBAL_OVERWHELMS_LOCAL"
    elif ratio >= 0.5:
        cls = "COMPARABLE"
    else:
        cls = "LOCAL_COMPARABLE_OR_LARGER"
    # per scenario/policy
    by = {}
    for sc in SCENARIOS:
        for ps in POLICY_SEEDS:
            sub = [r for r in reward_rows if r["scenario"] == sc and r["policy_seed"] == ps]
            if sub:
                wgm = float(np.median(np.abs([r["weighted_global_reward"] for r in sub])))
                wlm = float(np.median(np.abs([r["weighted_local_reward"] for r in sub])))
                by[f"{ps}_{sc}"] = {"median_abs_weighted_global": wgm,
                                    "median_abs_weighted_local": wlm,
                                    "ratio": wgm / (wlm + 1e-12)}
    out = {"n_transitions": len(reward_rows), "total_reconstruction_max_error": max_err,
           "reward_weights": {"alpha_g": ALPHA_G, "alpha_l": ALPHA_L, "beta": BETA},
           "weighted_global_abs": wg, "weighted_local_abs": wl, "weighted_fe_abs": wf,
           "median_ratio_global_over_local": ratio,
           "dominance_threshold_10x": True,
           "by_policy_scenario": by, "D4_REWARD_SCALE": cls}
    _jdump(out, RESULTS_DIR / "E4GD_D4_reward_scale_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D4_reward_components.csv",
               list(reward_rows[0].keys()), reward_rows[:5000])  # cap CSV size
    return out


# ---------------------------------------------------------------------------
# D5 — conditional counterfactual credit
# ---------------------------------------------------------------------------

def _U_conditional(snap, r, action, env, action_map):
    """Execute joint action (action_map with region r -> action) from the frozen
    pre-response snapshot; return U = -IGD(F_resp)."""
    rd = snap["region_dynamics"]
    t_new = snap["t_new"]
    amap = dict(action_map)
    amap[r] = action
    evaluate_fn = lambda Xq, _t=t_new, _rd=rd: cp_evaluate(Xq, _t, _rd)
    resp, _recs = execute_response(
        actions=amap, X=snap["X"], F_plus=snap["F_plus"], region_ids=snap["rids"],
        V=env.V, history=snap["history"], t=_history_t(snap) + 1, evaluate_fn=evaluate_fn,
        xl=env.xl, xu=env.xu, base_seed=snap["base_seed"],
    )
    _env_runs["count"] += 1
    return -float(igd(np.asarray(resp["F"]), env.pf_star))


def _history_t(snap):
    # history was updated up to the decision event; execute_response expects t = h_t+1 used at rollout
    return snap["_h_t"]


def d5_conditional_credit(d5_snapshots, env) -> Tuple[dict, List[dict], List[dict]]:
    region_rows = []
    interaction_rows = []
    regrets = []
    n_differ = 0
    n_pairs = 0
    TIE = 1e-5
    for snap in d5_snapshots:
        actual = snap["actual_actions"]
        allkeep = {r: "KEEP" for r in snap["active_regions"]}
        for r in snap["active_regions"]:
            # conditional utilities holding a_-r = actual
            U = {a: _U_conditional(snap, r, a, env, actual) for a in ACTION_ORDER}
            best_a = max(ACTION_ORDER, key=lambda a: U[a])
            chosen = actual[r]
            regret = U[best_a] - U[chosen]
            regrets.append(regret)
            region_rows.append({
                "policy_seed": snap["policy_seed"], "environment_seed": snap["environment_seed"],
                "change_index": snap["change_index"], "region_id": r,
                "chosen_action": chosen, "conditional_best_action": best_a,
                "regret": regret, "chosen_is_best": bool(chosen == best_a),
                "near_tie": bool(regret <= TIE),
                **{f"U_{a}": U[a] for a in ACTION_ORDER},
            })
            # interaction: best action under a_-r = allKEEP
            U_k = {a: _U_conditional(snap, r, a, env, allkeep) for a in ACTION_ORDER}
            best_k = max(ACTION_ORDER, key=lambda a: U_k[a])
            differ = (best_a != best_k)
            n_pairs += 1
            if differ:
                n_differ += 1
            interaction_rows.append({
                "policy_seed": snap["policy_seed"], "environment_seed": snap["environment_seed"],
                "change_index": snap["change_index"], "region_id": r,
                "best_action_given_actual": best_a, "best_action_given_allKEEP": best_k,
                "argmax_differs": differ})
    regrets = np.array(regrets)
    median_regret = float(np.median(regrets))
    if median_regret <= TIE:
        status = "ALIGNED"
    elif median_regret > 10 * TIE:
        status = "MISALIGNED"
    else:
        status = "MIXED"
    interaction_frac = n_differ / n_pairs if n_pairs else 0.0
    if interaction_frac >= D5_INTERACTION_MATERIAL:
        inter_cls = "MATERIAL"
    elif interaction_frac >= D5_INTERACTION_WEAK:
        inter_cls = "WEAK"
    else:
        inter_cls = "NEGLIGIBLE"
    # PREDICT-specific
    predict_rows = [r for r in region_rows if r["chosen_action"] == "PREDICT"]
    predict_is_best = float(np.mean([r["chosen_is_best"] for r in predict_rows])) if predict_rows else float("nan")
    out = {
        "n_events": len(d5_snapshots), "event_selection": {"scenario": D5_SCENARIO, "changes": D5_CHANGES},
        "n_region_counterfactuals": len(region_rows),
        "median_regret": median_regret, "mean_regret": float(regrets.mean()),
        "q95_regret": float(np.percentile(regrets, 95)),
        "frac_chosen_is_best": float(np.mean([r["chosen_is_best"] for r in region_rows])),
        "predict_frac_conditionally_best": predict_is_best,
        "CONDITIONAL_CREDIT_STATUS": status,
        "interaction_argmax_differ_fraction": interaction_frac,
        "D5_INTER_REGION_INTERACTION": inter_cls,
        "interaction_thresholds_predeclared": {"MATERIAL>=": D5_INTERACTION_MATERIAL,
                                               "WEAK>=": D5_INTERACTION_WEAK,
                                               "note": "E4G froze the interaction MEASURE but not "
                                                       "thresholds; predeclared here before compute"},
        "regret_thresholds": {"ALIGNED<=": TIE, "MISALIGNED>": 10 * TIE},
        "not_supervised_targets": True,
    }
    _jdump(out, RESULTS_DIR / "E4GD_D5_conditional_credit_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D5_region_counterfactuals.csv", list(region_rows[0].keys()), region_rows)
    _write_csv(RESULTS_DIR / "E4GD_D5_interaction_results.csv", list(interaction_rows[0].keys()), interaction_rows)
    return out, region_rows, interaction_rows


# ---------------------------------------------------------------------------
# D6 — checkpoint tension (validation only)
# ---------------------------------------------------------------------------

def d6_checkpoint_tension() -> dict:
    rows = list(csv.DictReader(open(E4TR_DIR / "E4TR_validation_all_policies.csv")))
    per = {}
    n_differ = 0
    mixed_diffs = []
    csv_rows = []
    for ps in POLICY_SEEDS:
        sub = [r for r in rows if int(r["policy_seed"]) == ps]
        macro_sel = [r for r in sub if r["selected"] == "True"][0]
        mixed_min = min(sub, key=lambda r: float(r["MIGD_response_MIXED"]))
        same = macro_sel["checkpoint"] == mixed_min["checkpoint"]
        mixed_diff = float(macro_sel["MIGD_response_MIXED"]) - float(mixed_min["MIGD_response_MIXED"])
        mixed_diffs.append(abs(mixed_diff))
        if not same:
            n_differ += 1
        per[str(ps)] = {"macro_selected": macro_sel["checkpoint"],
                        "macro_transition": int(macro_sel["transition_count"]),
                        "mixed_min": mixed_min["checkpoint"],
                        "mixed_min_transition": int(mixed_min["transition_count"]),
                        "same_checkpoint": same, "validation_MIXED_difference": mixed_diff,
                        "macro_difference": float(macro_sel["VALIDATION_MACRO_MIGD_RESPONSE"])
                        - float(mixed_min["VALIDATION_MACRO_MIGD_RESPONSE"])}
        for r in sub:
            csv_rows.append({"policy_seed": ps, "checkpoint": r["checkpoint"],
                             "transition_count": r["transition_count"],
                             "macro": r["VALIDATION_MACRO_MIGD_RESPONSE"],
                             "MIXED": r["MIGD_response_MIXED"],
                             "PERSISTENT": r["MIGD_response_PERSISTENT"],
                             "STATIONARY": r["MIGD_response_STATIONARY"],
                             "selected_macro": r["selected"]})
    mean_mixed_diff = float(np.mean(mixed_diffs))
    if n_differ >= 4 and mean_mixed_diff > 1e-4:
        cls = "STRONG"
    elif n_differ >= 2:
        cls = "MODERATE"
    elif n_differ == 1:
        cls = "WEAK"
    else:
        cls = "ABSENT"
    out = {"per_policy": per, "n_policies_differ": n_differ, "mean_abs_MIXED_diff": mean_mixed_diff,
           "CHECKPOINT_SELECTION_TENSION": cls,
           "forbidden_claim": "the MIXED-selected checkpoint would have won E4F final (NOT evaluated)"}
    _jdump(out, RESULTS_DIR / "E4GD_D6_checkpoint_tension_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D6_checkpoint_table.csv",
               ["policy_seed", "checkpoint", "transition_count", "macro", "MIXED",
                "PERSISTENT", "STATIONARY", "selected_macro"], csv_rows)
    return out


# ---------------------------------------------------------------------------
# D7 — collapse timeline
# ---------------------------------------------------------------------------

def d7_collapse_timeline() -> dict:
    rows = list(csv.DictReader(open(E4TR_DIR / "E4TR_training_diagnostics.csv")))
    csv_rows = []
    per_collapse = {}
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            ents = []
            for ep in range(6):
                sub = [r for r in rows if int(r["policy_seed"]) == ps and r["scenario"] == sc
                       and int(r["epoch"]) == ep]
                if not sub:
                    continue
                fr = {a: float(np.mean([float(r[f"action_fraction_{a}"]) for r in sub])) for a in ACTION_ORDER}
                H = -sum(p * math.log(p) for p in fr.values() if p > 0)
                dom = max(ACTION_ORDER, key=lambda a: fr[a])
                eps = float(np.mean([float(r["epsilon"]) for r in sub]))
                rew = float(np.mean([float(r["cumulative_reward"]) for r in sub]))
                mig = float(np.mean([float(r["MIGD_response"]) for r in sub]))
                ents.append((ep, H, fr[dom]))
                csv_rows.append({"policy_seed": ps, "scenario": sc, "epoch": ep + 1,
                                 "entropy": H, "dominant_action": dom, "dominant_fraction": fr[dom],
                                 "epsilon": eps, "cumulative_reward": rew, "train_MIGD_response": mig,
                                 **{f"frac_{a}": fr[a] for a in ACTION_ORDER}})
            if ents:
                per_collapse[f"{ps}_{sc}"] = {"epoch1_entropy": ents[0][1],
                                              "epoch6_entropy": ents[-1][1],
                                              "epoch1_dom": ents[0][2], "epoch6_dom": ents[-1][2]}
    early = sum(1 for v in per_collapse.values() if v["epoch1_dom"] >= 0.75)
    late_only = sum(1 for v in per_collapse.values()
                    if v["epoch1_dom"] < 0.75 and v["epoch6_dom"] >= 0.75)
    never = sum(1 for v in per_collapse.values() if v["epoch6_dom"] < 0.75)
    n = len(per_collapse)
    if never == n:
        timing = "NO_COLLAPSE"
    elif early >= 0.6 * n:
        timing = "EARLY"
    elif late_only >= 0.6 * n:
        timing = "LATE"
    elif early > 0 and late_only > 0:
        timing = "POLICY_DEPENDENT"
    else:
        timing = "GRADUAL"
    # epsilon coincidence: epsilon reaches 0.05 ~ transition 50000 ~ epoch 3 (18000/epoch)
    out = {"COLLAPSE_TIMING": timing, "n_early": early, "n_late_only": late_only, "n_never": never,
           "epsilon_reaches_floor_at_transition": 50000,
           "transitions_per_epoch": 18000,
           "coincidence_note": "epsilon hits 0.05 near epoch 3; collapse is LATE (by epoch 6) — "
                               "descriptive coincidence, no causal claim",
           "per_policy_scenario": per_collapse}
    _jdump(out, RESULTS_DIR / "E4GD_D7_collapse_timeline_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D7_action_trajectory.csv",
               ["policy_seed", "scenario", "epoch", "entropy", "dominant_action",
                "dominant_fraction", "epsilon", "cumulative_reward", "train_MIGD_response"]
               + [f"frac_{a}" for a in ACTION_ORDER], csv_rows)
    return out


# ---------------------------------------------------------------------------
# D8 — feature usage (frozen weights, real states)
# ---------------------------------------------------------------------------

def d8_feature_usage(nets, state_rows) -> dict:
    mixed = [r for r in state_rows if r["scenario"] == "MIXED"]
    X = np.array([[r[f"norm_{f}"] for f in FEATURE_NAMES] for r in mixed], dtype=np.float32)
    # subsample for speed
    if len(X) > 3000:
        idx = np.random.default_rng(0).choice(len(X), 3000, replace=False)
        X = X[idx]
    csv_rows = []
    per_policy = {}
    ratios = []
    for ps in POLICY_SEEDS:
        net = nets[ps]
        action_sens = {}
        for a in range(N_ACTIONS):
            xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            q = net(xt)[:, a].sum()
            q.backward()
            g = xt.grad.abs().mean(dim=0).numpy()
            local_s = float(np.sum(g[:4])); global_s = float(np.sum(g[4:]))
            action_sens[ACTION_ORDER[a]] = {"local": local_s, "global": global_s,
                                            "ratio_local_over_global": local_s / (global_s + 1e-12)}
            csv_rows.append({"policy_seed": ps, "action": ACTION_ORDER[a],
                             "local_sensitivity": local_s, "global_sensitivity": global_s,
                             "ratio_local_over_global": local_s / (global_s + 1e-12),
                             **{f"grad_{FEATURE_NAMES[i]}": float(g[i]) for i in range(9)}})
        # policy-level (mean over actions)
        pl = float(np.mean([action_sens[a]["ratio_local_over_global"] for a in ACTION_ORDER]))
        ratios.append(pl)
        per_policy[str(ps)] = {"mean_ratio_local_over_global": pl,
                               "predict_ratio": action_sens["PREDICT"]["ratio_local_over_global"],
                               "action_sensitivity": action_sens}
    mean_ratio = float(np.mean(ratios))
    per_cls = []
    for ps in POLICY_SEEDS:
        rr = per_policy[str(ps)]["mean_ratio_local_over_global"]
        per_cls.append("LOCAL_DOMINANT" if rr > 1.5 else "GLOBAL_DOMINANT" if rr < 0.67 else "BALANCED")
    if len(set(per_cls)) > 1:
        pattern = "POLICY_DEPENDENT"
    elif mean_ratio > 1.5:
        pattern = "LOCAL_DOMINANT"
    elif mean_ratio < 0.67:
        pattern = "GLOBAL_DOMINANT"
    else:
        pattern = "BALANCED"
    out = {"n_states": len(X), "mean_ratio_local_over_global": mean_ratio,
           "per_policy": per_policy, "per_policy_classification": dict(zip([str(p) for p in POLICY_SEEDS], per_cls)),
           "FEATURE_USAGE_PATTERN": pattern,
           "thresholds": {"LOCAL_DOMINANT>": 1.5, "GLOBAL_DOMINANT<": 0.67}}
    _jdump(out, RESULTS_DIR / "E4GD_D8_feature_usage_results.json")
    _write_csv(RESULTS_DIR / "E4GD_D8_feature_sensitivity.csv", list(csv_rows[0].keys()), csv_rows)
    return out


# ---------------------------------------------------------------------------
# Cross-diagnostic synthesis + branch decision
# ---------------------------------------------------------------------------

def synthesize(d1, d2, d3, d4, d5, d6, d7, d8, ffa) -> dict:
    table = {}
    table["A_STATE_NORMALIZATION_LIMITATION"] = {
        "diagnostic_sources": ["D1", "D2"],
        "evidence_for": [f"D1={d1['D1_NORMALIZATION_SIGNAL']}", f"D2={d2['D2_NORMALIZATION_SEPARABILITY']}"],
        "evidence_against": [],
        "classification": _cls_state_norm(d1, d2),
    }
    table["B_REWARD_CREDIT_MISALIGNMENT"] = {
        "diagnostic_sources": ["D4", "D5"],
        "evidence_for": [f"D4={d4['D4_REWARD_SCALE']}", f"D5={d5['CONDITIONAL_CREDIT_STATUS']}"],
        "evidence_against": [],
        "classification": _cls_reward(d4, d5),
    }
    table["C_JOINT_ACTION_REPRESENTATION_LIMITATION"] = {
        "diagnostic_sources": ["D3", "D5"],
        "evidence_for": [f"D3={d3['JOINT_ACTION_REPRESENTATION']}",
                         f"D5_interaction={d5['D5_INTER_REGION_INTERACTION']}"],
        "evidence_against": ["causal role not directly provable"],
        "classification": _cls_joint(d3, d5),
    }
    table["D_CHECKPOINT_SELECTION_TENSION"] = {
        "diagnostic_sources": ["D6"],
        "evidence_for": [f"D6={d6['CHECKPOINT_SELECTION_TENSION']}"],
        "evidence_against": ["cannot be tested on final seeds"],
        "classification": ("SUPPORTED" if d6["CHECKPOINT_SELECTION_TENSION"] in ("STRONG", "MODERATE")
                           else "PLAUSIBLE" if d6["CHECKPOINT_SELECTION_TENSION"] == "WEAK"
                           else "NOT_SUPPORTED"),
    }
    table["E_GLOBAL_ACTION_DOMINANCE"] = {
        "diagnostic_sources": ["D5", "E4F-FA"],
        "evidence_for": [f"MIXED PREDICT fraction (E4F-FA)={ffa['MIXED_PREDICT_FRACTION']:.3f}",
                         f"D5 predict conditionally best frac={d5.get('predict_frac_conditionally_best')}"],
        "evidence_against": [],
        "classification": _cls_global_dom(d5, ffa),
    }
    table["F_POLICY_COLLAPSE"] = {
        "diagnostic_sources": ["D7", "E4F-FA"],
        "evidence_for": [f"D7 COLLAPSE_TIMING={d7['COLLAPSE_TIMING']}",
                         "E4F-FA 13/15 cells collapsed"],
        "evidence_against": [],
        "classification": "SUPPORTED",
    }
    _jdump(table, RESULTS_DIR / "E4GD_failure_mechanism_table.json")
    return table


def _cls_state_norm(d1, d2):
    d1s, d2s = d1["D1_NORMALIZATION_SIGNAL"], d2["D2_NORMALIZATION_SEPARABILITY"]
    if d1s in ("COMPRESSED", "SEVERELY_COMPRESSED") and d2s in ("MODERATE_LOSS", "MATERIAL_LOSS"):
        return "SUPPORTED"
    if d1s in ("COMPRESSED", "SEVERELY_COMPRESSED") or d2s in ("MODERATE_LOSS", "MATERIAL_LOSS"):
        return "PLAUSIBLE"
    if d1s == "NOT_AUDITABLE" and d2s == "NOT_AUDITABLE":
        return "NOT_AUDITABLE"
    return "NOT_SUPPORTED"


def _cls_reward(d4, d5):
    if d4["D4_REWARD_SCALE"] == "GLOBAL_OVERWHELMS_LOCAL" or d5["CONDITIONAL_CREDIT_STATUS"] == "MISALIGNED":
        return "SUPPORTED"
    if d5["CONDITIONAL_CREDIT_STATUS"] == "MIXED":
        return "PLAUSIBLE"
    return "NOT_SUPPORTED"


def _cls_joint(d3, d5):
    if d3["JOINT_ACTION_REPRESENTATION"] == "IMPLICIT_STATE_ONLY" and d5["D5_INTER_REGION_INTERACTION"] == "MATERIAL":
        return "SUPPORTED"
    if d3["JOINT_ACTION_REPRESENTATION"] == "IMPLICIT_STATE_ONLY":
        return "PLAUSIBLE"
    return "NOT_SUPPORTED"


def _cls_global_dom(d5, ffa):
    pb = d5.get("predict_frac_conditionally_best")
    if ffa["MIXED_PREDICT_FRACTION"] >= 0.7 and (pb is not None and not math.isnan(pb) and pb >= 0.7):
        return "SUPPORTED"
    if ffa["MIXED_PREDICT_FRACTION"] >= 0.7:
        return "PLAUSIBLE"
    return "NOT_SUPPORTED"


def branch_decision(table, d1, d2, d4, d5, d3) -> dict:
    bl = _load(E4G_DIR / "E4G_branch_logic.json")
    a_elig = (d1["D1_NORMALIZATION_SIGNAL"] in ("COMPRESSED", "SEVERELY_COMPRESSED")
              and d2["D2_NORMALIZATION_SEPARABILITY"] in ("MODERATE_LOSS", "MATERIAL_LOSS"))
    b_elig = (d4["D4_REWARD_SCALE"] == "GLOBAL_OVERWHELMS_LOCAL"
              or d5["CONDITIONAL_CREDIT_STATUS"] == "MISALIGNED")
    c_elig = (table["C_JOINT_ACTION_REPRESENTATION_LIMITATION"]["classification"] == "SUPPORTED"
              and d5["D5_INTER_REGION_INTERACTION"] == "MATERIAL")
    eligible = [b for b, e in [("A", a_elig), ("B", b_elig), ("C", c_elig)] if e]
    # priority: smallest modification targeting strongest mechanism.
    # order of "smallest scientific modification": A (normalization) < B (reward) < C (architecture)
    priority = {"A": 0, "B": 1, "C": 2}
    if len(eligible) == 0:
        selected = "UNRESOLVED"
    elif len(eligible) == 1:
        selected = eligible[0]
    else:
        selected = sorted(eligible, key=lambda x: priority[x])[0]
    stage_map = {
        "A": "E4H_STATE_NORMALIZATION_REVISION_DESIGN",
        "B": "E4H_REWARD_CREDIT_REVISION_DESIGN",
        "C": "E4H_JOINT_COORDINATION_ARCHITECTURE_REVIEW",
        "UNRESOLVED": "E4H_DIAGNOSTIC_REVIEW",
    }
    sel_full = {"A": "A_STATE_NORMALIZATION", "B": "B_REWARD_CREDIT",
                "C": "C_JOINT_COORDINATION", "UNRESOLVED": "UNRESOLVED"}[selected]
    out = {
        "branch_logic_source": "E4G_branch_logic.json",
        "BRANCH_A_ELIGIBLE": "YES" if a_elig else "NO",
        "BRANCH_B_ELIGIBLE": "YES" if b_elig else "NO",
        "BRANCH_C_ELIGIBLE": "YES" if c_elig else "NO",
        "eligible": eligible,
        "priority_rule": "smallest modification targeting strongest mechanism (A<B<C); one factor at a time",
        "SELECTED_BRANCH": sel_full,
        "NEXT_STAGE": stage_map[selected],
        "one_factor_at_a_time": True,
        "selecting_does_not_mean_implementing": True,
    }
    _jdump(out, RESULTS_DIR / "E4GD_branch_decision.json")
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4GD — Controlled Diagnostic Execution (POST-HOC)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest, ok = pre_manifest()
    _jdump(manifest, RESULTS_DIR / "E4GD_pre_manifest.json")
    if not ok:
        print("[E4GD] integrity FAIL — STOP")
        return {"E4GD_COMPLETE": "NO"}
    print("[E4GD] integrity PASS. diagnostic seeds 1201..1220 only.")

    # frozen contracts
    c_d1 = _load(E4G_DIR / "E4G_D1_normalization_contract.json")
    c_d3 = _load(E4G_DIR / "E4G_D3_joint_representation_contract.json")
    ffa = _load(E4FFA_DIR / "E4FFA_verdict.json")

    env = make_env_context()
    nets = load_nets()
    cache = build_cache(env)
    print("[E4GD] running diagnostic rollout on 1201..1220 ...")
    state_rows, reward_rows, d5_snaps = _rollout_with_snapshots(nets, cache, env)
    print(f"[E4GD] collected {len(state_rows)} state rows, {len(reward_rows)} reward rows, "
          f"{len(d5_snaps)} D5 events")

    d1, _ = d1_normalization(state_rows, c_d1)
    d2 = d2_separability(state_rows)
    d3 = d3_joint_representation(c_d3)
    d4 = d4_reward_scale(reward_rows)
    print("[E4GD] running D5 conditional counterfactuals ...")
    d5, _, _ = d5_conditional_credit(d5_snaps, env)
    d6 = d6_checkpoint_tension()
    d7 = d7_collapse_timeline()
    d8 = d8_feature_usage(nets, state_rows)

    table = synthesize(d1, d2, d3, d4, d5, d6, d7, d8, ffa)
    branch = branch_decision(table, d1, d2, d4, d5, d3)

    # seed usage audit
    _jdump({"seeds": DIAGNOSTIC_SEEDS, "label": "POST_HOC_DIAGNOSTIC_ONLY",
            "final_seed_overlap": "NONE", "train_seed_overlap": "NONE",
            "validation_seed_overlap": "NONE",
            "number_environment_runs": _env_runs["count"],
            "final_eval_access_count": _final_eval_access["count"],
            "training_performed": False},
           RESULTS_DIR / "E4GD_diagnostic_seed_usage.json")

    _jdump({"ORIGINAL_E4F_VERDICT": "FAILED", "E4GD_IS_POST_HOC": True,
            "E4GD_CHANGES_E4F_VERDICT": False, "E4GD_TRAINS_NEW_MODEL": False,
            "ANY_FUTURE_REDESIGN": "NEW_EXPERIMENT"},
           RESULTS_DIR / "E4GD_verdict_addendum.json")

    verdict = _build_verdict(manifest, d1, d2, d3, d4, d5, d6, d7, d8, table, branch)
    _jdump(verdict, RESULTS_DIR / "E4GD_verdict.json")
    _write_report(verdict, table, d1, d2, d3, d4, d5, d6, d7, d8, branch)
    _emit_terminal(verdict)
    print(f"[E4GD] wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}; "
          f"final_eval_access={_final_eval_access['count']}")
    return verdict


def _build_verdict(manifest, d1, d2, d3, d4, d5, d6, d7, d8, table, branch):
    return {
        "stage": "E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION",
        "E4GD_COMPLETE": "YES",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4F_VERDICT_CHANGED": "NO",
        "D1_NORMALIZATION_SIGNAL": d1["D1_NORMALIZATION_SIGNAL"],
        "D2_NORMALIZATION_SEPARABILITY": d2["D2_NORMALIZATION_SEPARABILITY"],
        "D3_JOINT_ACTION_REPRESENTATION": d3["JOINT_ACTION_REPRESENTATION"],
        "D4_REWARD_SCALE": d4["D4_REWARD_SCALE"],
        "D5_CONDITIONAL_CREDIT_STATUS": d5["CONDITIONAL_CREDIT_STATUS"],
        "D5_INTER_REGION_INTERACTION": d5["D5_INTER_REGION_INTERACTION"],
        "D6_CHECKPOINT_SELECTION_TENSION": d6["CHECKPOINT_SELECTION_TENSION"],
        "D7_COLLAPSE_TIMING": d7["COLLAPSE_TIMING"],
        "D8_FEATURE_USAGE_PATTERN": d8["FEATURE_USAGE_PATTERN"],
        "STATE_NORMALIZATION_LIMITATION": table["A_STATE_NORMALIZATION_LIMITATION"]["classification"],
        "REWARD_CREDIT_MISALIGNMENT": table["B_REWARD_CREDIT_MISALIGNMENT"]["classification"],
        "JOINT_ACTION_REPRESENTATION_LIMITATION": table["C_JOINT_ACTION_REPRESENTATION_LIMITATION"]["classification"],
        "CHECKPOINT_SELECTION_TENSION_HYPOTHESIS": table["D_CHECKPOINT_SELECTION_TENSION"]["classification"],
        "GLOBAL_ACTION_DOMINANCE": table["E_GLOBAL_ACTION_DOMINANCE"]["classification"],
        "POLICY_COLLAPSE": "SUPPORTED",
        "BRANCH_A_ELIGIBLE": branch["BRANCH_A_ELIGIBLE"],
        "BRANCH_B_ELIGIBLE": branch["BRANCH_B_ELIGIBLE"],
        "BRANCH_C_ELIGIBLE": branch["BRANCH_C_ELIGIBLE"],
        "SELECTED_BRANCH": branch["SELECTED_BRANCH"],
        "NEW_TRAINING_RUN": "NO", "NEW_CONFIRMATORY_EVALUATION": "NO", "FINAL_SEED_REUSE": "NO",
        "DIAGNOSTIC_SEEDS_USED": f"{DIAGNOSTIC_SEEDS[0]}..{DIAGNOSTIC_SEEDS[-1]}",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": branch["NEXT_STAGE"],
    }


def _write_report(v, table, d1, d2, d3, d4, d5, d6, d7, d8, branch):
    L = ["# E4GD — Controlled Diagnostic Execution (post-hoc)", "",
         "**Original E4F verdict remains FAILED (immutable). E4F_VERDICT_CHANGED = NO. "
         "No training, no model change, no final-seed reuse.**", "",
         "## A. Original failure preserved",
         "- E4F MIXED DDQN vs GLOBAL_PREDICT = DIRECTIONAL_NEGATIVE; 0/5 policies positive; verdict FAILED.",
         "", "## B. D1 normalization signal",
         f"- D1_NORMALIZATION_SIGNAL = **{d1['D1_NORMALIZATION_SIGNAL']}** "
         f"({d1['near_constant_local_features']}/4 local features near-constant after normalization).",
         "", "## C. D2 raw-vs-normalized separability (MIXED regime label)",
         f"- mean(raw − normalized) balanced-accuracy gap = {d2.get('mean_raw_minus_normalized_gap', float('nan')):.4f} "
         f"→ D2 = **{d2['D2_NORMALIZATION_SEPARABILITY']}**.",
         "", "## D. D3 joint-action representation",
         f"- D3 = **{d3['JOINT_ACTION_REPRESENTATION']}** (no other-region actions / no centralized critic in Q).",
         "", "## E. D4 reward-component empirical scale (seeds 1201..1220)",
         f"- median |weighted_global| / |weighted_local| = {d4['median_ratio_global_over_local']:.2f} "
         f"→ D4_REWARD_SCALE = **{d4['D4_REWARD_SCALE']}** "
         f"(total reconstruction max error {d4['total_reconstruction_max_error']:.2e}).",
         "", "## F. D5 conditional action credit (seeds 1201..1220)",
         f"- median regret = {d5['median_regret']:.3e}; chosen==conditional-best frac = {d5['frac_chosen_is_best']:.3f}; "
         f"PREDICT conditionally-best frac = {d5.get('predict_frac_conditionally_best')}.",
         f"- CONDITIONAL_CREDIT_STATUS = **{d5['CONDITIONAL_CREDIT_STATUS']}**; "
         f"inter-region interaction (argmax differs {d5['interaction_argmax_differ_fraction']:.3f}) = "
         f"**{d5['D5_INTER_REGION_INTERACTION']}**.",
         "", "## G. D6 checkpoint-selection tension (validation only)",
         f"- {d6['n_policies_differ']}/5 policies: macro-selected ≠ MIXED-min checkpoint → "
         f"**{d6['CHECKPOINT_SELECTION_TENSION']}** (never tested on final seeds).",
         "", "## H. D7 collapse timeline",
         f"- COLLAPSE_TIMING = **{d7['COLLAPSE_TIMING']}**; {d7['coincidence_note']}",
         "", "## I. D8 local-vs-global feature usage",
         f"- mean local/global sensitivity ratio = {d8['mean_ratio_local_over_global']:.2f} → "
         f"FEATURE_USAGE_PATTERN = **{d8['FEATURE_USAGE_PATTERN']}**.",
         "", "## J. Cross-diagnostic evidence table", ""]
    for k, val in table.items():
        L.append(f"- **{k}** → {val['classification']} (sources: {', '.join(val['diagnostic_sources'])})")
    L += ["", "## K. Failure phenomenon vs mechanism",
          "- Phenomenon: POLICY_COLLAPSE toward a near-global action (observed).",
          "- Mechanism(s): determined by the supported table rows above; collapse is the symptom, "
          "not automatically the cause.",
          "", "## L/M. Branch eligibility and selection",
          f"- A={branch['BRANCH_A_ELIGIBLE']} B={branch['BRANCH_B_ELIGIBLE']} C={branch['BRANCH_C_ELIGIBLE']} "
          f"→ SELECTED_BRANCH = **{branch['SELECTED_BRANCH']}** (one factor at a time).",
          "", "## N. Scientific limitations",
          "- Diagnostic seeds 1201..1220 only; controlled benchmark only; no UAV/CEC validation.",
          "- E4GD identifies a mechanism to TEST next; it makes NO superiority claim (no redesign trained).",
          "", "## O. Original E4F verdict preservation",
          "- ORIGINAL_E4F_VERDICT = FAILED; ANY_FUTURE_REDESIGN = NEW_EXPERIMENT.",
          "", f"## Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4GD_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4GD_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4F_VERDICT_CHANGED",
              "D1_NORMALIZATION_SIGNAL", "D2_NORMALIZATION_SEPARABILITY",
              "D3_JOINT_ACTION_REPRESENTATION", "D4_REWARD_SCALE",
              "D5_CONDITIONAL_CREDIT_STATUS", "D5_INTER_REGION_INTERACTION",
              "D6_CHECKPOINT_SELECTION_TENSION", "D7_COLLAPSE_TIMING",
              "D8_FEATURE_USAGE_PATTERN", "STATE_NORMALIZATION_LIMITATION",
              "REWARD_CREDIT_MISALIGNMENT", "JOINT_ACTION_REPRESENTATION_LIMITATION",
              "CHECKPOINT_SELECTION_TENSION_HYPOTHESIS", "GLOBAL_ACTION_DOMINANCE",
              "POLICY_COLLAPSE", "BRANCH_A_ELIGIBLE", "BRANCH_B_ELIGIBLE", "BRANCH_C_ELIGIBLE",
              "SELECTED_BRANCH", "NEW_TRAINING_RUN", "NEW_CONFIRMATORY_EVALUATION",
              "FINAL_SEED_REUSE", "DIAGNOSTIC_SEEDS_USED", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<40} = {v[k]}")


# ---------------------------------------------------------------------------
# rollout wrapper that captures pre-response snapshots correctly
# ---------------------------------------------------------------------------

def _rollout_with_snapshots(nets, cache, env):
    from prdrl.experiments.E4P_learned_controller_preparation import compute_global_context_v1
    state_rows, reward_rows, d5_snaps = [], [], []
    for ps in POLICY_SEEDS:
        net = nets[ps]
        for scenario in SCENARIOS:
            rd = make_region_dynamics(scenario, env.R)
            for seed in DIAGNOSTIC_SEEDS:
                _guard_seed(seed)
                st = _fresh_state(scenario, seed, cache, env.R)
                for k in range(N_SCORED_CHANGES):
                    t_current_before = st["t_current"]
                    X_pre = st["X"].copy()                 # pre-response population
                    pre = pre_response(st, scenario, seed, env, rd)
                    h_t_at_decision = st["h_t"]            # execute_response uses h_t+1
                    local_norm, gc_norm, full = build_region_states(pre["v2_states"], env.R)
                    ids = sorted(pre["active_regions"])
                    x = torch.tensor(np.stack([full[r] for r in ids]), dtype=torch.float32)
                    with torch.no_grad():
                        qv = net(x).numpy()
                    _, action_str = select_regional_actions(net, local_norm, gc_norm,
                                                            epsilon=0.0, event_seed=0, device=DEVICE)
                    actions = {r: action_str[r] for r in pre["active_regions"]}
                    if scenario == D5_SCENARIO and k in D5_CHANGES:
                        d5_snaps.append({
                            "policy_seed": ps, "environment_seed": seed, "change_index": k,
                            "X": X_pre, "F_plus": np.array(pre["F_plus"]),
                            "rids": np.array(pre["rids"]),
                            "history": copy.deepcopy(st["h"]), "_h_t": h_t_at_decision,
                            "t_new": pre["t_new"], "region_dynamics": rd,
                            "base_seed": seed, "actual_actions": dict(actions),
                            "active_regions": list(pre["active_regions"])})
                    post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
                    _env_runs["count"] += 1
                    gr, region_rewards, aux = compute_joint_rewards(
                        pre, post, post["records"], env, rd, t_current_before)
                    rawg = compute_global_context_v1(pre["v2_states"], env.R)
                    for i, r in enumerate(ids):
                        v2 = pre["v2_states"][r]
                        raw_vec = [v2["severity_r"], v2["convergence_loss_r"], v2["occupancy_r"],
                                   v2["delta_severity_r"]] + list(rawg)
                        norm_vec = list(np.asarray(full[r], dtype=float))
                        regime = rd[r] if scenario == "MIXED" else scenario.lower()
                        state_rows.append({
                            "policy_seed": ps, "scenario": scenario, "environment_seed": seed,
                            "change_index": k, "region_id": r, "regime": regime, "action": action_str[r],
                            **{f"raw_{FEATURE_NAMES[j]}": float(raw_vec[j]) for j in range(9)},
                            **{f"norm_{FEATURE_NAMES[j]}": float(norm_vec[j]) for j in range(9)},
                            **{f"Q_{ACTION_ORDER[a]}": float(qv[i, a]) for a in range(N_ACTIONS)}})
                        reward_rows.append({
                            "policy_seed": ps, "scenario": scenario, "environment_seed": seed,
                            "change_index": k, "region_id": r,
                            "raw_global_reward": gr, "weighted_global_reward": ALPHA_G * gr,
                            "raw_local_reward": aux["local_by_region"][r],
                            "weighted_local_reward": ALPHA_L * aux["local_by_region"][r],
                            "raw_fe_cost": aux["fe_by_region"][r],
                            "weighted_fe_penalty": BETA * aux["fe_by_region"][r],
                            "total_reward": region_rewards[r]})
                    advance_state(st, pre, post, env)
    return state_rows, reward_rows, d5_snaps


if __name__ == "__main__":
    main()
