"""
E4F — Final Evaluation Execution (PR-DRL-DMOEA).

FIRST stage allowed to instantiate the reserved final environment seeds
1081..1140. The complete protocol was frozen in E4E and is executed here EXACTLY:
no checkpoint reselection, no comparator reselection, no retuning, no learning.

Matrix: 3 scenarios x 60 environment seeds x 10 controller arms = 1800 episodes.
Arms (deterministic order): GLOBAL_KEEP, GLOBAL_LOCAL, GLOBAL_PREDICT,
GLOBAL_DIVERSIFY, RULE_V1_V2STATE, DDQN_2001..2005.

Inference: crossed two-axis bootstrap (frozen in src/prdrl/evaluation/), primary
= DDQN vs validation-selected GLOBAL_PREDICT on MIXED MIGD_response.

Whatever this run returns is preserved. No rescue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

# ---- validated environment primitives (reused unchanged from E4TR) ----------
import prdrl.experiments.E4TR_shared_double_dqn_training as E4TR
from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _assert_no_region_disappearance, N_SCORED_CHANGES,
    NETWORK_INPUT_DIM,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import (
    make_region_dynamics, _warmup_and_burnin,
)
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions, ACTION_TO_IDX
from prdrl.controllers.rule_based import choose_actions_v2

# ---- frozen inferential statistics (frozen in E4E) --------------------------
from prdrl.evaluation.crossed_bootstrap import (
    crossed_two_axis_bootstrap, crossed_two_axis_bootstrap_macro, derive_seed,
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, BOOTSTRAP_MASTER_SEED,
)
from prdrl.evaluation.effects import (
    scenario_mean_effect, per_policy_effects, relative_improvement, macro_effect,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect, policy_robustness, safety_gate, scientific_verdict,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE,
    HARMFUL, VALIDATED, PARTIAL, FAILED,
)

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/E4F_final_evaluation")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
FINAL_EVAL_SEEDS = list(range(1081, 1141))     # 60 — opened HERE
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]
FIXED_GLOBALS = ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]
RULE_CTRL = "RULE_V1_V2STATE"
DDQN_ARMS = [f"DDQN_{ps}" for ps in POLICY_SEEDS]
ARM_ORDER = FIXED_GLOBALS + [RULE_CTRL] + DDQN_ARMS   # 10 arms

PRESELECTED_COMPARATOR = {
    "PERSISTENT": "GLOBAL_PREDICT",
    "STATIONARY": "GLOBAL_DIVERSIFY",
    "MIXED": "GLOBAL_PREDICT",
}
MIXED_ROBUSTNESS_THRESHOLD = 4
DEVICE = torch.device("cpu")

_GLOBAL_ACTION = {
    "GLOBAL_KEEP": "KEEP", "GLOBAL_LOCAL": "LOCAL",
    "GLOBAL_PREDICT": "PREDICT", "GLOBAL_DIVERSIFY": "DIVERSIFY",
}


# ---------------------------------------------------------------------------
def _hash_file(path: str) -> str:
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


# ---------------------------------------------------------------------------
# PHASE E4F-0 — pre-execution integrity
# ---------------------------------------------------------------------------

def pre_execution_integrity() -> Tuple[dict, dict, bool]:
    e4e_verdict = _load(E4E_DIR / "E4E_verdict.json")
    ck_freeze = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    comp_freeze = _load(E4E_DIR / "E4E_comparator_freeze.json")

    e4e_ok = e4e_verdict["E4E_COMPLETE"] == "YES"

    ck_rows = {}
    ck_ok = True
    for ps in POLICY_SEEDS:
        info = ck_freeze["policies"][str(ps)]
        actual = _hash_file(info["checkpoint_path"])
        exists = Path(info["checkpoint_path"]).exists()
        match = exists and (actual == info["sha256"])
        ck_ok = ck_ok and match
        ck_rows[str(ps)] = {
            "checkpoint": info["checkpoint_filename"],
            "transition_count": info["transition_count"],
            "path": info["checkpoint_path"],
            "sha256_frozen": info["sha256"], "sha256_actual": actual,
            "exists": exists, "hash_match": match,
        }

    prod_files = ["src/prdrl/operators/local.py", "src/prdrl/operators/predict.py",
                  "src/prdrl/operators/diversify.py", "src/prdrl/operators/budget.py",
                  "src/prdrl/oracle/oracle.py", "src/prdrl/core/survival.py",
                  "src/prdrl/rl/network.py", "src/prdrl/rl/agent.py",
                  "src/prdrl/rl/interface.py"]
    prod_hashes = {fp: _hash_file(fp) for fp in prod_files}

    rule_config = _load("results/E3R_rule_v2_semantic_rerun/E3R_rule_v1_v2state.json")["thresholds"]

    # runtime spec must match frozen comparator set
    spec_ok = (comp_freeze["primary_comparator_by_scenario"] == PRESELECTED_COMPARATOR
               and ck_freeze["immutable_for_final_evaluation"] is True)

    manifest = {
        "stage": "E4F_FINAL_EVALUATION_EXECUTION",
        "E4E_COMPLETE": e4e_verdict["E4E_COMPLETE"],
        "e4e_ok": e4e_ok,
        "runtime_matches_frozen_spec": spec_ok,
        "production_operator_hashes": prod_hashes,
        "rule_thresholds": rule_config,
        "final_environment_seeds": {"start": 1081, "end": 1140, "n": 60},
        "scenarios": SCENARIOS, "arm_order": ARM_ORDER,
        "preselected_comparator": PRESELECTED_COMPARATOR,
        "execution_mode": {"epsilon": 0.0, "network_mode": "eval", "gradients": "disabled",
                           "replay": "disabled", "optimizer": "disabled",
                           "target_updates": "disabled", "learning": "none"},
    }
    integrity = {
        "E4E_COMPLETE": e4e_verdict["E4E_COMPLETE"],
        "CHECKPOINT_INTEGRITY": "PASS" if (ck_ok and e4e_ok and spec_ok) else "FAIL",
        "checkpoints": ck_rows,
        "runtime_matches_frozen_spec": spec_ok,
    }
    ok = ck_ok and e4e_ok and spec_ok
    return manifest, integrity, ok, rule_config


# ---------------------------------------------------------------------------
# Environment cache for final seeds (opened here; deterministic per scenario+seed)
# ---------------------------------------------------------------------------

def build_final_cache(env) -> dict:
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_EVAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)
    return cache


def _fresh_state(scenario, seed, cache, R):
    import copy
    X_init, h_init, h_t_init, t_cur_init, gen_init = cache[(scenario, seed)]
    return {"X": X_init.copy(), "h": copy.deepcopy(h_init), "h_t": h_t_init,
            "t_current": t_cur_init, "gen": gen_init, "q_prev": E4TR.Q_PREV_INIT,
            "severity_history": {r: None for r in range(R)}}


def load_checkpoints(ck_freeze) -> Dict[int, SharedRegionalQNetwork]:
    nets = {}
    for ps in POLICY_SEEDS:
        path = ck_freeze["policies"][str(ps)]["checkpoint_path"]
        payload = torch.load(path, weights_only=False)
        net = SharedRegionalQNetwork(NETWORK_INPUT_DIM)
        net.load_state_dict(payload["online_state_dict"])   # ONLINE weights only
        net.eval()                                          # no grad / no learning
        nets[ps] = net
    return nets


# ---------------------------------------------------------------------------
# Arm runners (episode-level + change-level). Reuse validated primitives.
# ---------------------------------------------------------------------------

def _run_arm(select_fn, scenario, seed, cache, env, controller, policy_seed):
    rd = make_region_dynamics(scenario, env.R)
    state = _fresh_state(scenario, seed, cache, env.R)
    igd_resp, igd_ep, fe_list = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    n_fallback = 0
    change_rows = []
    prev_active = None
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(state, scenario, seed, env, rd)
        if prev_active is not None:
            _assert_no_region_disappearance(prev_active, pre["active_regions"])
        local_norm, gc_norm, _full = build_region_states(pre["v2_states"], env.R)
        actions = select_fn(pre, local_norm, gc_norm)
        post = execute_and_recover(actions, pre, state, scenario, seed, env, rd, k + 2)
        fe_change = int(sum(rec.response_fe for rec in post["records"]))
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe_list.append(fe_change)
        reg = []
        for rec in post["records"]:
            act_counts[rec.requested_action] += 1
            if rec.fallback_used:
                n_fallback += 1
            reg.append({"region_id": rec.region_id, "requested": rec.requested_action,
                        "realized": rec.realized_operator, "fallback": bool(rec.fallback_used)})
        change_rows.append({
            "controller": controller, "policy_seed": policy_seed, "scenario": scenario,
            "environment_seed": seed, "change_index": k,
            "MIGD_response_change": post["igd_response"],
            "MIGD_epoch_change": post["igd_epoch"],
            "response_FE_change": fe_change,
            "regional": reg,
        })
        prev_active = pre["active_regions"]
        advance_state(state, pre, post, env)
    total_act = sum(act_counts.values()) or 1
    episode = {
        "controller": controller,
        "policy_seed": policy_seed if policy_seed is not None else "",
        "scenario": scenario, "environment_seed": seed,
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "response_FE": int(np.sum(fe_list)),
        "action_KEEP_fraction": act_counts["KEEP"] / total_act,
        "action_LOCAL_fraction": act_counts["LOCAL"] / total_act,
        "action_PREDICT_fraction": act_counts["PREDICT"] / total_act,
        "action_DIVERSIFY_fraction": act_counts["DIVERSIFY"] / total_act,
        "predict_fallback_fraction": n_fallback / total_act,
    }
    return episode, change_rows


def make_global_select(action_name):
    def select(pre, local_norm, gc_norm):
        return {r: action_name for r in pre["active_regions"]}
    return select


def make_rule_select(rule_config):
    def select(pre, local_norm, gc_norm):
        return choose_actions_v2(RULE_CTRL, pre["contexts"], rule_config)
    return select


def make_ddqn_select(net):
    def select(pre, local_norm, gc_norm):
        _, action_str = select_regional_actions(net, local_norm, gc_norm,
                                                epsilon=0.0, event_seed=0, device=DEVICE)
        return {r: action_str[r] for r in pre["active_regions"]}
    return select


# ---------------------------------------------------------------------------
# MATRIX EXECUTION (deterministic order, resumable)
# ---------------------------------------------------------------------------

def _episode_key(row):
    return (row["controller"], str(row["policy_seed"]), row["scenario"], int(row["environment_seed"]))


def run_matrix(env, cache, nets, rule_config) -> Tuple[List[dict], List[dict]]:
    progress_path = RESULTS_DIR / "E4F_progress.jsonl"
    completed: Dict[tuple, dict] = {}
    change_path = RESULTS_DIR / "E4F_change_results.jsonl"

    # resume: reload any completed episodes (deterministic identity by key)
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                completed[_episode_key(row)] = row

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # open change-results append (only write for freshly computed episodes)
    episodes: List[dict] = []
    n_done = 0
    total = len(SCENARIOS) * len(FINAL_EVAL_SEEDS) * len(ARM_ORDER)

    with open(progress_path, "a") as prog, open(change_path, "a") as chg:
        for scenario in SCENARIOS:
            for seed in FINAL_EVAL_SEEDS:
                for arm in ARM_ORDER:
                    if arm in FIXED_GLOBALS:
                        controller, policy_seed = arm, None
                        select_fn = make_global_select(_GLOBAL_ACTION[arm])
                    elif arm == RULE_CTRL:
                        controller, policy_seed = arm, None
                        select_fn = make_rule_select(rule_config)
                    else:  # DDQN_xxxx
                        ps = int(arm.split("_")[1])
                        controller, policy_seed = "DDQN", ps
                        select_fn = make_ddqn_select(nets[ps])

                    key = (controller, str(policy_seed) if policy_seed is not None else "",
                           scenario, seed)
                    if key in completed:
                        episodes.append(completed[key])
                        n_done += 1
                        continue

                    episode, change_rows = _run_arm(
                        select_fn, scenario, seed, cache, env, controller, policy_seed)
                    episodes.append(episode)
                    prog.write(json.dumps(episode) + "\n"); prog.flush()
                    for cr in change_rows:
                        chg.write(json.dumps(cr) + "\n")
                    chg.flush()
                    n_done += 1
                    if n_done % 180 == 0:
                        print(f"    [E4F] {n_done}/{total} episodes done")
    return episodes, n_done


# ---------------------------------------------------------------------------
# MATRIX COMPLETENESS
# ---------------------------------------------------------------------------

def check_matrix(episodes) -> dict:
    keys = [_episode_key(r) for r in episodes]
    uniq = set(keys)
    expected = set()
    for scenario in SCENARIOS:
        for seed in FINAL_EVAL_SEEDS:
            for arm in FIXED_GLOBALS + [RULE_CTRL]:
                expected.add((arm, "", scenario, seed))
            for ps in POLICY_SEEDS:
                expected.add(("DDQN", str(ps), scenario, seed))
    finite = all(np.isfinite(r["MIGD_response"]) and np.isfinite(r["MIGD_epoch"])
                 and np.isfinite(r["response_FE"]) for r in episodes)
    complete = (len(episodes) == 1800 and len(uniq) == len(keys) == 1800
                and uniq == expected and finite)
    return {
        "n_rows": len(episodes), "n_unique_keys": len(uniq),
        "no_duplicates": len(uniq) == len(keys),
        "all_expected_present": uniq == expected,
        "all_finite": finite,
        "FINAL_MATRIX_COMPLETE": "YES" if complete else "NO",
    }


# ---------------------------------------------------------------------------
# BUILD METRIC MATRICES (L: 5x60 per scenario; B: 60 per scenario per arm)
# ---------------------------------------------------------------------------

def build_arm_index(episodes):
    """idx[(controller,policy_seed_str,scenario,seed)] = row."""
    return {_episode_key(r): r for r in episodes}


def L_matrix(idx, scenario, endpoint):
    """Learned matrix (5 x 60) for the given endpoint."""
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_EVAL_SEEDS)))
    for i, ps in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_EVAL_SEEDS):
            M[i, j] = idx[("DDQN", str(ps), scenario, e)][endpoint]
    return M


def B_vector(idx, scenario, arm, endpoint):
    """Comparator/baseline vector (60,) for a fixed-global or rule arm."""
    v = np.zeros(len(FINAL_EVAL_SEEDS))
    for j, e in enumerate(FINAL_EVAL_SEEDS):
        v[j] = idx[(arm, "", scenario, e)][endpoint]
    return v


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def analyze(idx, integrity_pass) -> dict:
    out = {"scenario_effects": {}, "bootstrap": {}, "gates": {}, "descriptive": {}}

    # ---- descriptive: all fixed globals + rule + DDQN per scenario ----
    desc = {}
    for scenario in SCENARIOS:
        desc[scenario] = {}
        for arm in FIXED_GLOBALS + [RULE_CTRL]:
            resp = B_vector(idx, scenario, arm, "MIGD_response")
            ep = B_vector(idx, scenario, arm, "MIGD_epoch")
            fe = B_vector(idx, scenario, arm, "response_FE")
            desc[scenario][arm] = {"mean_MIGD_response": float(resp.mean()),
                                   "mean_MIGD_epoch": float(ep.mean()),
                                   "mean_response_FE": float(fe.mean())}
        Lr = L_matrix(idx, scenario, "MIGD_response")
        Le = L_matrix(idx, scenario, "MIGD_epoch")
        Lf = L_matrix(idx, scenario, "response_FE")
        desc[scenario]["DDQN_pooled"] = {"mean_MIGD_response": float(Lr.mean()),
                                         "mean_MIGD_epoch": float(Le.mean()),
                                         "mean_response_FE": float(Lf.mean())}
    out["descriptive"] = desc

    # ---- scenario effects vs preselected comparator (response + epoch + FE) ----
    def effect_block(scenario, comparator, endpoint):
        L = L_matrix(idx, scenario, endpoint)
        B = B_vector(idx, scenario, comparator, endpoint)
        if endpoint == "response_FE":
            # positive FE_saving favors DDQN (fewer evaluations)
            delta = float((B[None, :] - L).mean())
        else:
            delta = scenario_mean_effect(L, B)
        bs = crossed_two_axis_bootstrap(L, B, seed=derive_seed(endpoint, scenario, comparator))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])
        ri = relative_improvement(delta, float(B.mean()))
        return {"comparator": comparator, "endpoint": endpoint,
                "baseline_mean": float(B.mean()), "ddqn_mean": float(L.mean()),
                "delta": delta, "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": ri, "classification": cls,
                "per_policy": per_policy_effects(L, B)}

    scen_eff = {}
    for scenario in SCENARIOS:
        comp = PRESELECTED_COMPARATOR[scenario]
        scen_eff[scenario] = {
            "MIGD_response": effect_block(scenario, comp, "MIGD_response"),
            "MIGD_epoch": effect_block(scenario, comp, "MIGD_epoch"),
            "response_FE": effect_block(scenario, comp, "response_FE"),
        }
    out["scenario_effects"] = scen_eff

    # ---- PRIMARY: MIXED DDQN vs GLOBAL_PREDICT on MIGD_response ----
    mixed = scen_eff["MIXED"]["MIGD_response"]
    out["gates"]["MIXED_PRIMARY"] = mixed["classification"]
    out["gates"]["MIXED_DELTA_RESPONSE"] = mixed["delta"]
    out["gates"]["MIXED_DELTA_RESPONSE_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]
    out["gates"]["MIXED_RELATIVE_IMPROVEMENT"] = mixed["relative_improvement"]
    out["gates"]["MIXED_DDQN_MEAN"] = mixed["ddqn_mean"]
    out["gates"]["MIXED_COMPARATOR_MEAN"] = mixed["baseline_mean"]

    # ---- policy robustness on MIXED response ----
    pp = mixed["per_policy"]
    rob = policy_robustness(pp, threshold=MIXED_ROBUSTNESS_THRESHOLD)
    out["gates"]["MIXED_POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    out["gates"]["MIXED_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"
    out["gates"]["per_policy_delta_MIXED_response"] = pp

    # ---- DDQN vs RULE on MIXED response ----
    Lr = L_matrix(idx, "MIXED", "MIGD_response")
    Brule = B_vector(idx, "MIXED", RULE_CTRL, "MIGD_response")
    rule_delta = scenario_mean_effect(Lr, Brule)
    rule_bs = crossed_two_axis_bootstrap(Lr, Brule, seed=derive_seed("MIGD_response", "MIXED", RULE_CTRL))
    rule_cls = classify_effect(rule_bs["point_estimate"], rule_bs["ci_lo"], rule_bs["ci_hi"])
    out["gates"]["DDQN_VS_RULE_MIXED"] = rule_cls
    out["gates"]["DDQN_VS_RULE_MIXED_DELTA"] = rule_delta
    out["gates"]["DDQN_VS_RULE_MIXED_CI"] = [rule_bs["ci_lo"], rule_bs["ci_hi"]]

    # ---- safety: PERSISTENT, STATIONARY (response + epoch) ----
    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = scen_eff[scenario]["MIGD_response"]["classification"]
        elab = scen_eff[scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        out["gates"][f"{scenario}_RESPONSE_EFFECT"] = rlab
        out["gates"][f"{scenario}_EPOCH_EFFECT"] = elab
        out["gates"][f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    # ---- macro (response + epoch), shared draws across scenarios ----
    def macro_block(endpoint):
        Ls = {s: L_matrix(idx, s, endpoint) for s in SCENARIOS}
        Bs = {s: B_vector(idx, s, PRESELECTED_COMPARATOR[s], endpoint) for s in SCENARIOS}
        deltas = {s: (float((Bs[s][None, :] - Ls[s]).mean()) if endpoint == "response_FE"
                      else scenario_mean_effect(Ls[s], Bs[s])) for s in SCENARIOS}
        point = macro_effect(deltas)
        bs = crossed_two_axis_bootstrap_macro(Ls, Bs, SCENARIOS,
                                              seed=derive_seed("macro", endpoint))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])
        return {"delta": point, "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": cls, "scenario_deltas": deltas}

    macro_resp = macro_block("MIGD_response")
    macro_epoch = macro_block("MIGD_epoch")
    macro_fe = macro_block("response_FE")
    out["macro"] = {"MIGD_response": macro_resp, "MIGD_epoch": macro_epoch,
                    "response_FE": macro_fe}
    macro_sg = safety_gate(macro_resp["classification"], macro_epoch["classification"])
    out["gates"]["MACRO_RESPONSE_EFFECT"] = macro_resp["classification"]
    out["gates"]["MACRO_EPOCH_EFFECT"] = macro_epoch["classification"]
    out["gates"]["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"

    # ---- FE savings ----
    out["gates"]["MIXED_FE_SAVING"] = scen_eff["MIXED"]["response_FE"]["delta"]
    out["gates"]["PERSISTENT_FE_SAVING"] = scen_eff["PERSISTENT"]["response_FE"]["delta"]
    out["gates"]["STATIONARY_FE_SAVING"] = scen_eff["STATIONARY"]["response_FE"]["delta"]

    # ---- scientific verdict (frozen logic) ----
    v = scientific_verdict(
        mixed_primary_label=out["gates"]["MIXED_PRIMARY"],
        ddqn_vs_rule_mixed_label=out["gates"]["DDQN_VS_RULE_MIXED"],
        mixed_policy_robustness_pass=(out["gates"]["MIXED_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(out["gates"]["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(out["gates"]["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(out["gates"]["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=integrity_pass,
        protocol_integrity_pass=integrity_pass,
        no_training_during_final=True,
    )
    out["SCIENTIFIC_VERDICT"] = v["verdict"]
    out["verdict_reasons"] = v["reasons"]
    return out


# ---------------------------------------------------------------------------
# OUTPUT WRITERS
# ---------------------------------------------------------------------------

def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_outputs(episodes, idx, analysis, manifest, integrity, matrix_check, verdict_block):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # episode results
    ep_fields = ["controller", "policy_seed", "scenario", "environment_seed",
                 "MIGD_response", "MIGD_epoch", "response_FE",
                 "action_KEEP_fraction", "action_LOCAL_fraction",
                 "action_PREDICT_fraction", "action_DIVERSIFY_fraction",
                 "predict_fallback_fraction"]
    _write_csv(RESULTS_DIR / "E4F_episode_results.csv", ep_fields, episodes)

    # ddqn / baseline / rule splits
    _write_csv(RESULTS_DIR / "E4F_ddqn_results.csv", ep_fields,
               [r for r in episodes if r["controller"] == "DDQN"])
    _write_csv(RESULTS_DIR / "E4F_baseline_results.csv", ep_fields,
               [r for r in episodes if r["controller"] in FIXED_GLOBALS])
    _write_csv(RESULTS_DIR / "E4F_rule_results.csv", ep_fields,
               [r for r in episodes if r["controller"] == RULE_CTRL])

    # action distribution (policy seed x scenario)
    ad_rows = []
    for ps in POLICY_SEEDS:
        for scenario in SCENARIOS:
            rows = [idx[("DDQN", str(ps), scenario, e)] for e in FINAL_EVAL_SEEDS]
            ad_rows.append({
                "policy_seed": ps, "scenario": scenario,
                "KEEP": float(np.mean([r["action_KEEP_fraction"] for r in rows])),
                "LOCAL": float(np.mean([r["action_LOCAL_fraction"] for r in rows])),
                "PREDICT": float(np.mean([r["action_PREDICT_fraction"] for r in rows])),
                "DIVERSIFY": float(np.mean([r["action_DIVERSIFY_fraction"] for r in rows])),
                "predict_fallback": float(np.mean([r["predict_fallback_fraction"] for r in rows])),
            })
    _write_csv(RESULTS_DIR / "E4F_action_distribution.csv",
               ["policy_seed", "scenario", "KEEP", "LOCAL", "PREDICT", "DIVERSIFY",
                "predict_fallback"], ad_rows)

    # policy seed summary
    ps_rows = []
    for ps in POLICY_SEEDS:
        for scenario in SCENARIOS:
            rows = [idx[("DDQN", str(ps), scenario, e)] for e in FINAL_EVAL_SEEDS]
            comp = PRESELECTED_COMPARATOR[scenario]
            comp_resp = np.mean([idx[(comp, "", scenario, e)]["MIGD_response"] for e in FINAL_EVAL_SEEDS])
            ddqn_resp = np.mean([r["MIGD_response"] for r in rows])
            ps_rows.append({
                "policy_seed": ps, "scenario": scenario, "n_env": 60,
                "mean_MIGD_response": float(ddqn_resp),
                "mean_MIGD_epoch": float(np.mean([r["MIGD_epoch"] for r in rows])),
                "mean_response_FE": float(np.mean([r["response_FE"] for r in rows])),
                "effect_vs_preselected_global_response": float(comp_resp - ddqn_resp),
                "action_KEEP_fraction": float(np.mean([r["action_KEEP_fraction"] for r in rows])),
                "action_LOCAL_fraction": float(np.mean([r["action_LOCAL_fraction"] for r in rows])),
                "action_PREDICT_fraction": float(np.mean([r["action_PREDICT_fraction"] for r in rows])),
                "action_DIVERSIFY_fraction": float(np.mean([r["action_DIVERSIFY_fraction"] for r in rows])),
                "predict_fallback_fraction": float(np.mean([r["predict_fallback_fraction"] for r in rows])),
            })
    _write_csv(RESULTS_DIR / "E4F_policy_seed_summary.csv",
               ["policy_seed", "scenario", "n_env", "mean_MIGD_response", "mean_MIGD_epoch",
                "mean_response_FE", "effect_vs_preselected_global_response",
                "action_KEEP_fraction", "action_LOCAL_fraction", "action_PREDICT_fraction",
                "action_DIVERSIFY_fraction", "predict_fallback_fraction"], ps_rows)

    # scenario effects
    se_rows = []
    for scenario in SCENARIOS:
        for endpoint in ["MIGD_response", "MIGD_epoch"]:
            b = analysis["scenario_effects"][scenario][endpoint]
            se_rows.append({
                "scenario": scenario, "endpoint": endpoint, "comparator": b["comparator"],
                "baseline_mean": b["baseline_mean"], "ddqn_mean": b["ddqn_mean"],
                "delta_B_minus_DDQN": b["delta"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "relative_improvement": b["relative_improvement"],
                "classification": b["classification"],
            })
    _write_csv(RESULTS_DIR / "E4F_scenario_effects.csv",
               ["scenario", "endpoint", "comparator", "baseline_mean", "ddqn_mean",
                "delta_B_minus_DDQN", "ci_lo", "ci_hi", "relative_improvement",
                "classification"], se_rows)

    # macro effects
    mac_rows = []
    for endpoint in ["MIGD_response", "MIGD_epoch", "response_FE"]:
        m = analysis["macro"][endpoint]
        mac_rows.append({"endpoint": endpoint, "delta": m["delta"], "ci_lo": m["ci_lo"],
                         "ci_hi": m["ci_hi"], "classification": m["classification"]})
    _write_csv(RESULTS_DIR / "E4F_macro_effects.csv",
               ["endpoint", "delta", "ci_lo", "ci_hi", "classification"], mac_rows)

    # rule comparison
    g = analysis["gates"]
    _write_csv(RESULTS_DIR / "E4F_rule_comparison.csv",
               ["scenario", "endpoint", "delta_RULE_minus_DDQN", "ci_lo", "ci_hi", "classification"],
               [{"scenario": "MIXED", "endpoint": "MIGD_response",
                 "delta_RULE_minus_DDQN": g["DDQN_VS_RULE_MIXED_DELTA"],
                 "ci_lo": g["DDQN_VS_RULE_MIXED_CI"][0], "ci_hi": g["DDQN_VS_RULE_MIXED_CI"][1],
                 "classification": g["DDQN_VS_RULE_MIXED"]}])

    # fe effects
    fe_rows = []
    for scenario in SCENARIOS:
        b = analysis["scenario_effects"][scenario]["response_FE"]
        fe_rows.append({"scenario": scenario, "fe_saving": b["delta"],
                        "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                        "classification": b["classification"]})
    m = analysis["macro"]["response_FE"]
    fe_rows.append({"scenario": "MACRO", "fe_saving": m["delta"], "ci_lo": m["ci_lo"],
                    "ci_hi": m["ci_hi"], "classification": m["classification"]})
    _write_csv(RESULTS_DIR / "E4F_fe_effects.csv",
               ["scenario", "fe_saving", "ci_lo", "ci_hi", "classification"], fe_rows)

    # bootstrap results (full numeric)
    _jdump({
        "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL,
                     "CI_METHOD": CI_METHOD, "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
                     "design": "CROSSED_TWO_AXIS_BOOTSTRAP", "policy_axis": 5,
                     "environment_axis": 60},
        "scenario_effects": analysis["scenario_effects"],
        "macro": analysis["macro"],
    }, RESULTS_DIR / "E4F_bootstrap_results.json")

    # scientific gates
    _jdump(analysis["gates"], RESULTS_DIR / "E4F_scientific_gates.json")

    # manifests
    _jdump(manifest, RESULTS_DIR / "E4F_pre_execution_manifest.json")
    _jdump(integrity, RESULTS_DIR / "E4F_checkpoint_integrity.json")

    # verdict
    _jdump(verdict_block, RESULTS_DIR / "E4F_verdict.json")


def write_report(analysis, matrix_check, integrity, verdict_block):
    g = analysis["gates"]
    desc = analysis["descriptive"]
    L = []
    L.append("# E4F — Final Evaluation Execution")
    L.append("")
    L.append(f"**SCIENTIFIC_VERDICT = {analysis['SCIENTIFIC_VERDICT']}**")
    L.append("")
    L.append("## A. Final protocol integrity")
    L.append(f"- CHECKPOINT_INTEGRITY = {integrity['CHECKPOINT_INTEGRITY']}")
    L.append(f"- runtime matches frozen spec = {integrity['runtime_matches_frozen_spec']}")
    L.append(f"- FINAL_MATRIX_COMPLETE = {matrix_check['FINAL_MATRIX_COMPLETE']} "
             f"({matrix_check['n_rows']}/1800 rows, all finite = {matrix_check['all_finite']})")
    L.append("")
    L.append("## B. Frozen checkpoints")
    for ps in POLICY_SEEDS:
        c = integrity["checkpoints"][str(ps)]
        L.append(f"- {ps}: {c['checkpoint']} (transition {c['transition_count']}, hash_match={c['hash_match']})")
    L.append("")
    L.append("## D. Fixed-global final descriptive results (mean MIGD_response)")
    L.append("")
    L.append("| scenario | KEEP | LOCAL | PREDICT | DIVERSIFY | RULE | DDQN(pooled) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        d = desc[s]
        L.append(f"| {s} | {d['GLOBAL_KEEP']['mean_MIGD_response']:.6f} "
                 f"| {d['GLOBAL_LOCAL']['mean_MIGD_response']:.6f} "
                 f"| {d['GLOBAL_PREDICT']['mean_MIGD_response']:.6f} "
                 f"| {d['GLOBAL_DIVERSIFY']['mean_MIGD_response']:.6f} "
                 f"| {d['RULE_V1_V2STATE']['mean_MIGD_response']:.6f} "
                 f"| {d['DDQN_pooled']['mean_MIGD_response']:.6f} |")
    L.append("")
    L.append("## G. Primary MIXED effect (DDQN vs GLOBAL_PREDICT, MIGD_response)")
    L.append(f"- DDQN mean = {g['MIXED_DDQN_MEAN']:.6f}; comparator mean = {g['MIXED_COMPARATOR_MEAN']:.6f}")
    L.append(f"- Delta = {g['MIXED_DELTA_RESPONSE']:.6e}  CI {g['MIXED_DELTA_RESPONSE_CI']}")
    L.append(f"- Relative improvement = {g['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}%")
    L.append(f"- **MIXED_PRIMARY = {g['MIXED_PRIMARY']}**")
    L.append("")
    L.append("## I. MIXED policy robustness")
    L.append(f"- positive policies = {g['MIXED_POLICY_POSITIVE_COUNT']}/5; "
             f"MIXED_POLICY_ROBUSTNESS = {g['MIXED_POLICY_ROBUSTNESS']}")
    L.append(f"- per-policy delta (MIXED response): {[round(x,6) for x in g['per_policy_delta_MIXED_response']]}")
    L.append("")
    L.append("## J. DDQN vs RULE on MIXED")
    L.append(f"- Delta = {g['DDQN_VS_RULE_MIXED_DELTA']:.6e} CI {g['DDQN_VS_RULE_MIXED_CI']}; "
             f"DDQN_VS_RULE_MIXED = {g['DDQN_VS_RULE_MIXED']}")
    L.append("")
    L.append("## K/L. Safety")
    L.append(f"- PERSISTENT: response={g['PERSISTENT_RESPONSE_EFFECT']}, "
             f"epoch={g['PERSISTENT_EPOCH_EFFECT']} -> {g['PERSISTENT_SAFETY']}")
    L.append(f"- STATIONARY: response={g['STATIONARY_RESPONSE_EFFECT']}, "
             f"epoch={g['STATIONARY_EPOCH_EFFECT']} -> {g['STATIONARY_SAFETY']}")
    L.append("")
    L.append("## M. Macro")
    L.append(f"- response={g['MACRO_RESPONSE_EFFECT']}, epoch={g['MACRO_EPOCH_EFFECT']} "
             f"-> MACRO_SAFETY={g['MACRO_SAFETY']}")
    L.append("")
    L.append("## N. FE tradeoff (positive = DDQN cheaper)")
    L.append(f"- MIXED={g['MIXED_FE_SAVING']:.2f}, PERSISTENT={g['PERSISTENT_FE_SAVING']:.2f}, "
             f"STATIONARY={g['STATIONARY_FE_SAVING']:.2f}")
    L.append("")
    L.append("## Q. Scientific verdict")
    L.append(f"- **{analysis['SCIENTIFIC_VERDICT']}** — reasons: {analysis['verdict_reasons']}")
    L.append("")
    L.append("## R. Limitations")
    L.append("- Evidence is for the current controlled dynamic benchmark protocol only.")
    L.append("- Not yet UAV real-world validation; not yet CEC/DF full benchmark suite.")
    L.append("- No claim of global optimality; no claim sigma_P=0.005 is globally optimal;")
    L.append("  no claim the DDQN architecture is globally optimal; no true joint-oracle claim.")
    L.append("")
    L.append(f"## S. Next research stage: {verdict_block['NEXT_STAGE']}")
    (RESULTS_DIR / "E4F_report.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4F — Final Evaluation Execution")
    print("=" * 60)

    manifest, integrity, ok, rule_config = pre_execution_integrity()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _jdump(manifest, RESULTS_DIR / "E4F_pre_execution_manifest.json")
    _jdump(integrity, RESULTS_DIR / "E4F_checkpoint_integrity.json")
    if not ok:
        print("[E4F] PRE-EXECUTION INTEGRITY FAIL — STOP (no final seed opened)")
        _emit_stop("E4F_PROTOCOL_INTEGRITY_REVIEW")
        return {"E4F_COMPLETE": "NO"}
    print(f"[E4F] integrity PASS. Checkpoints verified. Opening final seeds 1081..1140.")

    env = make_env_context()
    ck_freeze = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    nets = load_checkpoints(ck_freeze)
    cache = build_final_cache(env)
    print(f"[E4F] env R={env.R}; cache built for 60 seeds x 3 scenarios.")

    episodes, n_done = run_matrix(env, cache, nets, rule_config)
    matrix_check = check_matrix(episodes)
    print(f"[E4F] matrix: {matrix_check['n_rows']}/1800, "
          f"COMPLETE={matrix_check['FINAL_MATRIX_COMPLETE']}")

    if matrix_check["FINAL_MATRIX_COMPLETE"] != "YES":
        verdict_block = {
            "stage": "E4F_FINAL_EVALUATION_EXECUTION", "E4F_COMPLETE": "NO",
            "FINAL_MATRIX_COMPLETE": "NO", "SCIENTIFIC_VERDICT": "NOT_COMPUTED",
            "NEXT_STAGE": "E4F_ENGINEERING_FAILURE_REVIEW",
        }
        _jdump(verdict_block, RESULTS_DIR / "E4F_verdict.json")
        _jdump(matrix_check, RESULTS_DIR / "E4F_scientific_gates.json")
        _emit_stop("E4F_ENGINEERING_FAILURE_REVIEW", matrix_check=matrix_check)
        return verdict_block

    idx = build_arm_index(episodes)
    analysis = analyze(idx, integrity_pass=(integrity["CHECKPOINT_INTEGRITY"] == "PASS"))

    verdict = analysis["SCIENTIFIC_VERDICT"]
    next_stage = {VALIDATED: "E5_ABLATION_AND_GENERALIZATION",
                  PARTIAL: "E4F_RESULT_DIAGNOSIS",
                  FAILED: "E4F_FAILURE_ANALYSIS"}[verdict]

    # post-run checkpoint immutability re-check
    post_hashes_ok = all(
        _hash_file(ck_freeze["policies"][str(ps)]["checkpoint_path"])
        == ck_freeze["policies"][str(ps)]["sha256"] for ps in POLICY_SEEDS)

    verdict_block = {
        "stage": "E4F_FINAL_EVALUATION_EXECUTION",
        "E4F_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES",
        "EPISODES_COMPLETE": f"{matrix_check['n_rows']}/1800",
        "CHECKPOINT_INTEGRITY": integrity["CHECKPOINT_INTEGRITY"],
        "FINAL_PROTOCOL_INTEGRITY": "PASS" if integrity["runtime_matches_frozen_spec"] else "FAIL",
        "checkpoint_files_unchanged_after_E4F": post_hashes_ok,
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_COMPARATOR": "GLOBAL_PREDICT",
        "gates": analysis["gates"],
        "SCIENTIFIC_VERDICT": verdict,
        "verdict_reasons": analysis["verdict_reasons"],
        "RL_STATUS": "FINAL_EVALUATED", "DRL_TRAINING": "COMPLETE",
        "FINAL_EVALUATION": "COMPLETE",
        "HYPERPARAMETER_RETUNING": "NO", "CHECKPOINT_RESELECTION": "NO",
        "FINAL_COMPARATOR_RESELECTION": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }

    write_outputs(episodes, idx, analysis, manifest, integrity, matrix_check, verdict_block)
    write_report(analysis, matrix_check, integrity, verdict_block)

    _emit_terminal(matrix_check, integrity, analysis, verdict_block)
    print(f"[E4F] total wall {time.time()-t0:.0f}s")
    return verdict_block


def _emit_stop(next_stage, matrix_check=None):
    print()
    print("E4F_COMPLETE                 = NO")
    if matrix_check:
        print(f"FINAL_MATRIX_COMPLETE        = {matrix_check['FINAL_MATRIX_COMPLETE']}")
        print(f"EPISODES_COMPLETE            = {matrix_check['n_rows']}/1800")
    print(f"SCIENTIFIC_VERDICT           = NOT_COMPUTED")
    print(f"NEXT_STAGE                   = {next_stage}")


def _emit_terminal(matrix_check, integrity, analysis, vb):
    g = analysis["gates"]
    ci = g["MIXED_DELTA_RESPONSE_CI"]
    print()
    print(f"E4F_COMPLETE                       = {vb['E4F_COMPLETE']}")
    print(f"FINAL_MATRIX_COMPLETE             = {matrix_check['FINAL_MATRIX_COMPLETE']}")
    print(f"EPISODES_COMPLETE                 = {matrix_check['n_rows']}/1800")
    print(f"CHECKPOINT_INTEGRITY              = {integrity['CHECKPOINT_INTEGRITY']}")
    print(f"FINAL_PROTOCOL_INTEGRITY          = {vb['FINAL_PROTOCOL_INTEGRITY']}")
    print(f"PRIMARY_SCENARIO                  = MIXED")
    print(f"PRIMARY_COMPARATOR                = GLOBAL_PREDICT")
    print(f"MIXED_DDQN_MEAN_MIGD_RESPONSE     = {g['MIXED_DDQN_MEAN']:.6f}")
    print(f"MIXED_GLOBAL_PREDICT_MEAN_MIGD_RESPONSE = {g['MIXED_COMPARATOR_MEAN']:.6f}")
    print(f"MIXED_DELTA_RESPONSE              = {g['MIXED_DELTA_RESPONSE']:.6e}")
    print(f"MIXED_DELTA_RESPONSE_CI           = [{ci[0]:.6e},{ci[1]:.6e}]")
    print(f"MIXED_RELATIVE_IMPROVEMENT        = {g['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}%")
    print(f"MIXED_PRIMARY                     = {g['MIXED_PRIMARY']}")
    print(f"MIXED_POLICY_POSITIVE_COUNT       = {g['MIXED_POLICY_POSITIVE_COUNT']}/5")
    print(f"MIXED_POLICY_ROBUSTNESS           = {g['MIXED_POLICY_ROBUSTNESS']}")
    print(f"DDQN_VS_RULE_MIXED                = {g['DDQN_VS_RULE_MIXED']}")
    print(f"PERSISTENT_RESPONSE_EFFECT        = {g['PERSISTENT_RESPONSE_EFFECT']}")
    print(f"PERSISTENT_EPOCH_EFFECT           = {g['PERSISTENT_EPOCH_EFFECT']}")
    print(f"PERSISTENT_SAFETY                 = {g['PERSISTENT_SAFETY']}")
    print(f"STATIONARY_RESPONSE_EFFECT        = {g['STATIONARY_RESPONSE_EFFECT']}")
    print(f"STATIONARY_EPOCH_EFFECT           = {g['STATIONARY_EPOCH_EFFECT']}")
    print(f"STATIONARY_SAFETY                 = {g['STATIONARY_SAFETY']}")
    print(f"MACRO_RESPONSE_EFFECT             = {g['MACRO_RESPONSE_EFFECT']}")
    print(f"MACRO_EPOCH_EFFECT                = {g['MACRO_EPOCH_EFFECT']}")
    print(f"MACRO_SAFETY                      = {g['MACRO_SAFETY']}")
    print(f"MIXED_FE_SAVING                   = {g['MIXED_FE_SAVING']:.2f}")
    print(f"PERSISTENT_FE_SAVING              = {g['PERSISTENT_FE_SAVING']:.2f}")
    print(f"STATIONARY_FE_SAVING             = {g['STATIONARY_FE_SAVING']:.2f}")
    print(f"SCIENTIFIC_VERDICT                = {vb['SCIENTIFIC_VERDICT']}")
    print(f"RL_STATUS                         = {vb['RL_STATUS']}")
    print(f"DRL_TRAINING                      = {vb['DRL_TRAINING']}")
    print(f"FINAL_EVALUATION                  = {vb['FINAL_EVALUATION']}")
    print(f"HYPERPARAMETER_RETUNING           = NO")
    print(f"CHECKPOINT_RESELECTION            = NO")
    print(f"FINAL_COMPARATOR_RESELECTION      = NO")
    print(f"FULL_REGRESSION                   = {vb['FULL_REGRESSION']}")
    print(f"NEXT_STAGE                        = {vb['NEXT_STAGE']}")


if __name__ == "__main__":
    main()
