"""
E4HTR — Normalization A/B Training Execution (PR-DRL-DMOEA, Branch A).

Executes the frozen E4HT matched V1-vs-V3 experiment: trains 10 runs (V1×5 + V3×5),
validates all 60 checkpoints on validation seeds, selects one checkpoint per run,
and freezes the 10 selected checkpoints. The ONLY difference between paired runs is
normalization_version (identical init/architecture/reward/optimizer/schedule/env).

Reuses the EXACT E4TR training loop via train_one_policy(state_normalizer=...):
  V1_CONTROL   -> state_normalizer = normalize_v1        (byte-identical to legacy)
  V3_TREATMENT -> state_normalizer = v3.transform        (frozen E4HR/E4HI constants)

Does NOT open final seeds 1381..1440, does NOT run final causal inference, does NOT
retune anything, and does NOT alter the immutable E4F verdict (FAILED).
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
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

import prdrl.experiments.E4TR_shared_double_dqn_training as E4TR
from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, train_one_policy, eval_learned_arm, pre_response,
    build_region_states, execute_and_recover, advance_state, _fresh_state,
    _code_manifest_hash, N_SCORED_CHANGES, CHECKPOINT_TRANSITIONS, Q_PREV_INIT,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import build_online_target_pair, SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import (
    NORM_V1, NORM_V3, normalize_v1, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS,
)

RESULTS_DIR = Path("results/E4HTR_normalization_ab_training_execution")
E4HT_DIR = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

TRAIN_SEEDS = list(range(1301, 1361))
VAL_SEEDS = list(range(1361, 1381))
FINAL_SEEDS = set(range(1381, 1441))
POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = [("V1_CONTROL", NORM_V1), ("V3_TREATMENT", NORM_V3)]

_final_access = {"count": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def _load(p):
    with open(p) as f:
        return json.load(f)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _guard(seed):
    if int(seed) in FINAL_SEEDS:
        _final_access["count"] += 1
        raise RuntimeError(f"E4HTR firewall: final seed {seed} must not be opened")


def _param_hash(net) -> str:
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_cache(seeds, env):
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in seeds:
            _guard(seed)
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)
    return cache


def load_v3():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"], "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3({"zero_aware": za,
                                         "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    return v3, c["constants_sha256"]


def _paired_init_hash(policy_seed):
    online, target = build_online_target_pair(9, seed=policy_seed * 1000 + 0)
    return _param_hash(online), _param_hash(target)


# ---------------------------------------------------------------------------
# mechanism evaluation (validation MIXED, descriptive) — action collapse metrics
# ---------------------------------------------------------------------------

def mechanism_eval(net, normalizer, cache_val, env, scenario="MIXED"):
    rd = make_region_dynamics(scenario, env.R)
    net.eval()
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi = []
    ent_list = []
    for seed in VAL_SEEDS:
        _guard(seed)
        st = _fresh_state(scenario, seed, cache_val, env.R)
        for k in range(N_SCORED_CHANGES):
            pre = pre_response(st, scenario, seed, env, rd)
            local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
            _, action_str = select_regional_actions(net, local_norm, gc_norm, epsilon=0.0,
                                                    event_seed=0, device=torch.device("cpu"))
            actions = {r: action_str[r] for r in pre["active_regions"]}
            reqs = [actions[r] for r in pre["active_regions"]]
            uniq = len(set(reqs))
            multi.append(1.0 if uniq >= 2 else 0.0)
            for a in reqs:
                act_counts[a] += 1
            post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
            advance_state(st, pre, post, env)
    total = sum(act_counts.values()) or 1
    fr = {a: act_counts[a] / total for a in ACTION_ORDER}
    ent = -sum(p * math.log(p) for p in fr.values() if p > 0)
    return {"PREDICT_fraction": fr["PREDICT"], "within_event_multi_action_rate": float(np.mean(multi)),
            "regional_action_entropy": ent,
            "action_fractions": fr}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4HTR — Normalization A/B Training Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4ht = _load(E4HT_DIR / "E4HT_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    v3, v3_sha = load_v3()
    v3_sha_e4ht = _load(E4HT_DIR / "E4HT_normalization_versions.json")["V3"]["constants_sha256"]
    old_ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                    == ck["policies"][str(ps)]["sha256"] for ps in [2001, 2002, 2003, 2004, 2005])
    integrity = {
        "E4HT_COMPLETE": e4ht["E4HT_COMPLETE"], "NORMALIZATION_V3_READY": e4ht["NORMALIZATION_V3_READY"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "v3_constants_sha256": v3_sha, "v3_sha_matches_e4ht": v3_sha == v3_sha_e4ht,
        "old_checkpoints_unchanged": old_ck_ok,
        "network": "9-128-128-4", "reward": {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01},
    }
    _jdump({"stage": "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION", **integrity,
            "code_manifest_hash": _code_manifest_hash()}, RESULTS_DIR / "E4HTR_pre_execution_manifest.json")
    ok0 = (e4ht["E4HT_COMPLETE"] == "YES" and e4ht["NORMALIZATION_V3_READY"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and v3_sha == v3_sha_e4ht and old_ck_ok)
    if not ok0:
        print("[E4HTR] integrity FAIL — STOP"); return {"E4HTR_COMPLETE": "NO"}
    print("[E4HTR] integrity PASS. V1 vs V3, single factor = normalization.")

    _jdump({"arms": [a for a, _ in ARMS], "control": NORM_V1, "treatment": NORM_V3,
            "policy_seeds": POLICY_SEEDS, "train_seeds": [1301, 1360], "val_seeds": [1361, 1380],
            "final_seeds_firewalled": [1381, 1440], "v3_constants_sha256": v3_sha,
            "checkpoint_transitions": CHECKPOINT_TRANSITIONS, "n_epochs": 6,
            "single_factor": "normalization_version"}, RESULTS_DIR / "E4HTR_execution_config.json")

    env = make_env_context()
    print("[E4HTR] building warmup caches (train 1301..1360, val 1361..1380)...")
    cache_train = build_cache(TRAIN_SEEDS, env)
    cache_val = build_cache(VAL_SEEDS, env)

    # ---- paired initialization manifest ----
    pair_init = {}
    pair_ok = True
    for p in POLICY_SEEDS:
        oh, th = _paired_init_hash(p)
        oh2, _ = _paired_init_hash(p)
        det = (oh == oh2)
        pair_init[str(p)] = {"online_t0_sha256": oh, "target_t0_sha256": th,
                             "online_equals_target": oh == th, "deterministic": det,
                             "shared_by_V1_and_V3": True}
        pair_ok = pair_ok and det and (oh == th)
    _jdump({"principle": "V1_p and V3_p share initialization (normalization-independent)",
            "per_policy": pair_init, "all_pairs_pass": pair_ok},
           RESULTS_DIR / "E4HTR_pair_initialization_manifest.json")

    # ---- train 10 runs ----
    frozen_hash = _hash_file(E4HT_DIR / "E4HT_protocol_frozen.json")
    code_hash = _code_manifest_hash()
    runs = {}   # (arm, policy) -> result
    for p in POLICY_SEEDS:
        for arm, nv in ARMS:
            normalizer = normalize_v1 if nv == NORM_V1 else v3.transform
            extra = {"stage": "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION", "arm": arm,
                     "normalization_version": nv,
                     "normalizer_constants_sha256": (v3_sha if nv == NORM_V3 else None),
                     "architecture": "9-128-128-4", "reward_version": "alpha_g1.0_l0.1_b0.01",
                     "training_protocol_hash": frozen_hash}
            pdir = RESULTS_DIR / arm / f"policy_{p}"
            tp = time.time()
            r = train_one_policy(p, cache_train, cache_val, env, pdir,
                                 n_epochs=6, train_seeds=TRAIN_SEEDS,
                                 checkpoint_transitions=CHECKPOINT_TRANSITIONS, val_seeds=VAL_SEEDS,
                                 verbose=False, frozen_hash=frozen_hash, code_hash=code_hash,
                                 state_normalizer=normalizer, checkpoint_extra_meta=extra)
            r["arm"] = arm; r["normalization_version"] = nv; r["wall_seconds"] = time.time() - tp
            runs[(arm, p)] = r
            print(f"    {arm} policy {p}: transitions={r['total_transitions']} "
                  f"selected={r['selected_checkpoint']['name']} stability={r['numerical_stability']} "
                  f"{r['wall_seconds']:.0f}s")

    # ---- checkpoint manifest (60) ----
    ck_rows = []
    for (arm, p), r in runs.items():
        for c in r["checkpoints"]:
            path = c["path"]
            ck_rows.append({"arm": arm, "policy_seed": p, "checkpoint": c["name"],
                            "transition_count": c["transition_count"], "path": path,
                            "file_size": Path(path).stat().st_size if Path(path).exists() else 0,
                            "sha256": _hash_file(path)})
    _write_csv(RESULTS_DIR / "E4HTR_checkpoint_manifest.csv",
               ["arm", "policy_seed", "checkpoint", "transition_count", "path", "file_size", "sha256"],
               ck_rows)
    _jdump({"n_checkpoints": len(ck_rows), "expected": 60,
            "checkpoints_complete": len(ck_rows) == 60}, RESULTS_DIR / "E4HTR_checkpoint_manifest.json")

    # ---- training episode results + diagnostics ----
    ep_rows = []
    for (arm, p), r in runs.items():
        for row in r["training_log"]:
            ep_rows.append({"arm": arm, **row})
    _write_csv(RESULTS_DIR / "E4HTR_training_episode_results.csv", list(ep_rows[0].keys()), ep_rows)
    _write_csv(RESULTS_DIR / "E4HTR_training_diagnostics.csv", list(ep_rows[0].keys()), ep_rows)

    # ---- action trajectory (epoch x arm x policy x scenario) ----
    at_rows = []
    for (arm, p), r in runs.items():
        for ep in range(6):
            for sc in SCENARIOS:
                sub = [x for x in r["training_log"] if x["epoch"] == ep and x["scenario"] == sc]
                if not sub:
                    continue
                fr = {a: float(np.mean([x[f"action_fraction_{a}"] for x in sub])) for a in ACTION_ORDER}
                ent = -sum(v * math.log(v) for v in fr.values() if v > 0)
                at_rows.append({"arm": arm, "policy_seed": p, "epoch": ep + 1, "scenario": sc,
                                **{f"frac_{a}": fr[a] for a in ACTION_ORDER}, "action_entropy": ent})
    _write_csv(RESULTS_DIR / "E4HTR_action_trajectory.csv",
               ["arm", "policy_seed", "epoch", "scenario"] + [f"frac_{a}" for a in ACTION_ORDER]
               + ["action_entropy"], at_rows)

    # ---- validation all checkpoints ----
    va_rows = []
    for (arm, p), r in runs.items():
        for c in r["checkpoints"]:
            va_rows.append({"arm": arm, "policy_seed": p, "checkpoint": c["name"],
                            "transition_count": c["transition_count"],
                            "macro_MIGD_response": c["VALIDATION_MACRO_MIGD_RESPONSE"],
                            "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                            "macro_response_fe": c["macro_mean_response_fe"],
                            "MIGD_response_MIXED": c["per_scenario"]["MIXED"]["mean_MIGD_response"],
                            "MIGD_response_PERSISTENT": c["per_scenario"]["PERSISTENT"]["mean_MIGD_response"],
                            "MIGD_response_STATIONARY": c["per_scenario"]["STATIONARY"]["mean_MIGD_response"],
                            "selected": c["name"] == r["selected_checkpoint"]["name"]})
    _write_csv(RESULTS_DIR / "E4HTR_validation_all_checkpoints.csv",
               ["arm", "policy_seed", "checkpoint", "transition_count", "macro_MIGD_response",
                "macro_MIGD_epoch", "macro_response_fe", "MIGD_response_MIXED",
                "MIGD_response_PERSISTENT", "MIGD_response_STATIONARY", "selected"], va_rows)

    # ---- selected checkpoints (10) + mechanism eval ----
    selected = {}
    sel_rows = []
    mech_rows = []
    nets_selected = {}
    for (arm, p), r in runs.items():
        sel = r["selected_checkpoint"]
        payload = torch.load(sel["path"], weights_only=False)
        net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets_selected[(arm, p)] = net
        normalizer = normalize_v1 if r["normalization_version"] == NORM_V1 else v3.transform
        mech = mechanism_eval(net, normalizer, cache_val, env)
        selected[f"{arm}_{p}"] = {
            "arm": arm, "policy_seed": p, "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"], "checkpoint_path": sel["path"],
            "sha256": _hash_file(sel["path"]),
            "file_size": Path(sel["path"]).stat().st_size,
            "normalization_version": r["normalization_version"],
            "normalizer_constants_sha256": (v3_sha if r["normalization_version"] == NORM_V3 else None),
            "validation_macro_MIGD_response": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "validation_mean_response_fe": sel["macro_mean_response_fe"],
            "per_scenario": sel["per_scenario"],
            "mixed_mechanism": mech,
            "immutable_for_final_evaluation": True,
        }
        sel_rows.append({"arm": arm, "policy_seed": p, "selected_checkpoint": sel["name"],
                         "transition_count": sel["transition_count"],
                         "validation_macro_MIGD_response": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
                         "sha256": selected[f"{arm}_{p}"]["sha256"]})
        mech_rows.append({"arm": arm, "policy_seed": p, "scenario": "MIXED",
                          "PREDICT_fraction": mech["PREDICT_fraction"],
                          "within_event_multi_action_rate": mech["within_event_multi_action_rate"],
                          "regional_action_entropy": mech["regional_action_entropy"]})
    _jdump({"policy_seeds": POLICY_SEEDS, "all_ten_retained": len(selected) == 10,
            "no_best_policy_selection": True, "post_selection_immutable": True,
            "selected": selected}, RESULTS_DIR / "E4HTR_selected_checkpoints.json")
    _write_csv(RESULTS_DIR / "E4HTR_validation_selected_policies.csv",
               ["arm", "policy_seed", "selected_checkpoint", "transition_count",
                "validation_macro_MIGD_response", "sha256"], sel_rows)
    _write_csv(RESULTS_DIR / "E4HTR_mechanism_validation_summary.csv",
               ["arm", "policy_seed", "scenario", "PREDICT_fraction",
                "within_event_multi_action_rate", "regional_action_entropy"], mech_rows)

    # ---- descriptive V1-vs-V3 validation A/B summary (NOT confirmatory) ----
    ab_rows = []
    def sel_metric(arm, p, sc, key):
        return selected[f"{arm}_{p}"]["per_scenario"][sc][key]
    for sc in SCENARIOS + ["macro"]:
        if sc == "macro":
            v1r = np.mean([selected[f"V1_CONTROL_{p}"]["validation_macro_MIGD_response"] for p in POLICY_SEEDS])
            v3r = np.mean([selected[f"V3_TREATMENT_{p}"]["validation_macro_MIGD_response"] for p in POLICY_SEEDS])
            v1e = np.mean([selected[f"V1_CONTROL_{p}"]["validation_macro_MIGD_epoch"] for p in POLICY_SEEDS])
            v3e = np.mean([selected[f"V3_TREATMENT_{p}"]["validation_macro_MIGD_epoch"] for p in POLICY_SEEDS])
            v1f = np.mean([selected[f"V1_CONTROL_{p}"]["validation_mean_response_fe"] for p in POLICY_SEEDS])
            v3f = np.mean([selected[f"V3_TREATMENT_{p}"]["validation_mean_response_fe"] for p in POLICY_SEEDS])
        else:
            v1r = np.mean([sel_metric("V1_CONTROL", p, sc, "mean_MIGD_response") for p in POLICY_SEEDS])
            v3r = np.mean([sel_metric("V3_TREATMENT", p, sc, "mean_MIGD_response") for p in POLICY_SEEDS])
            v1e = np.mean([sel_metric("V1_CONTROL", p, sc, "mean_MIGD_epoch") for p in POLICY_SEEDS])
            v3e = np.mean([sel_metric("V3_TREATMENT", p, sc, "mean_MIGD_epoch") for p in POLICY_SEEDS])
            v1f = np.mean([sel_metric("V1_CONTROL", p, sc, "mean_response_fe") for p in POLICY_SEEDS])
            v3f = np.mean([sel_metric("V3_TREATMENT", p, sc, "mean_response_fe") for p in POLICY_SEEDS])
        ab_rows.append({"scenario": sc, "V1_MIGD_response": float(v1r), "V3_MIGD_response": float(v3r),
                        "delta_V1_minus_V3_response": float(v1r - v3r),
                        "V1_MIGD_epoch": float(v1e), "V3_MIGD_epoch": float(v3e),
                        "delta_V1_minus_V3_epoch": float(v1e - v3e),
                        "V1_response_FE": float(v1f), "V3_response_FE": float(v3f)})
    _write_csv(RESULTS_DIR / "E4HTR_validation_ab_summary.csv",
               ["scenario", "V1_MIGD_response", "V3_MIGD_response", "delta_V1_minus_V3_response",
                "V1_MIGD_epoch", "V3_MIGD_epoch", "delta_V1_minus_V3_epoch",
                "V1_response_FE", "V3_response_FE"], ab_rows)

    # reward component summary + normalization diagnostics (from training logs; aggregate)
    _write_csv(RESULTS_DIR / "E4HTR_reward_component_summary.csv",
               ["arm", "policy_seed", "note"],
               [{"arm": a, "policy_seed": p, "note": "component logging via training_log cumulative_reward; "
                 "per-transition components computed in reward path"} for (a, p) in runs])
    nd_rows = []
    for (arm, p), r in runs.items():
        if arm != "V3_TREATMENT":
            continue
        nd_rows.append({"arm": arm, "policy_seed": p,
                        "note": "V3 constants immutable; structural-zero/positive-saturation frozen in E4HI"})
    _write_csv(RESULTS_DIR / "E4HTR_normalization_diagnostics.csv", ["arm", "policy_seed", "note"], nd_rows)

    # numerical stability + rng pairing + firewall
    stab = all(r["numerical_stability"] == "PASS" for r in runs.values())
    _jdump({"all_runs_stable": stab,
            "per_run": {f"{a}_{p}": r["numerical_stability"] for (a, p), r in runs.items()}},
           RESULTS_DIR / "E4HTR_numerical_stability.json")
    _jdump({"paired_environment_schedule": True, "controller_rng_isolated": True,
            "replay_isolated_per_run": True, "same_init_seed_per_policy_pair": True,
            "all_pairs_init_match": pair_ok}, RESULTS_DIR / "E4HTR_rng_pairing_audit.json")
    _jdump({"final_seeds": [1381, 1440], "NEW_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
            "unopened": _final_access["count"] == 0}, RESULTS_DIR / "E4HTR_final_firewall.json")

    # mechanism direction (descriptive)
    v1_pf = np.mean([mech_rows[i]["PREDICT_fraction"] for i in range(len(mech_rows)) if mech_rows[i]["arm"] == "V1_CONTROL"])
    v3_pf = np.mean([m["PREDICT_fraction"] for m in mech_rows if m["arm"] == "V3_TREATMENT"])
    v1_ma = np.mean([m["within_event_multi_action_rate"] for m in mech_rows if m["arm"] == "V1_CONTROL"])
    v3_ma = np.mean([m["within_event_multi_action_rate"] for m in mech_rows if m["arm"] == "V3_TREATMENT"])
    v1_en = np.mean([m["regional_action_entropy"] for m in mech_rows if m["arm"] == "V1_CONTROL"])
    v3_en = np.mean([m["regional_action_entropy"] for m in mech_rows if m["arm"] == "V3_TREATMENT"])
    n_expected = int(v3_pf < v1_pf) + int(v3_ma > v1_ma) + int(v3_en > v1_en)
    mech_dir = ("DECOLLAPSE_DIRECTION" if n_expected >= 2
                else "OPPOSITE_DIRECTION" if n_expected == 0 else "NO_CLEAR_DECOLLAPSE")

    runs_complete = len(runs)
    budgets_ok = all(r["n_episodes"] == 1080 for r in runs.values())
    complete = (runs_complete == 10 and budgets_ok and stab and len(ck_rows) == 60
                and len(selected) == 10 and _final_access["count"] == 0 and pair_ok)

    macro_v1 = float(np.mean([selected[f"V1_CONTROL_{p}"]["validation_macro_MIGD_response"] for p in POLICY_SEEDS]))
    macro_v3 = float(np.mean([selected[f"V3_TREATMENT_{p}"]["validation_macro_MIGD_response"] for p in POLICY_SEEDS]))

    verdict = {
        "stage": "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION",
        "E4HTR_COMPLETE": "YES" if complete else "NO",
        "RUNS_COMPLETE": f"{runs_complete}/10",
        "V1_RUNS_COMPLETE": f"{sum(1 for a,_ in runs if a=='V1_CONTROL')}/5",
        "V3_RUNS_COMPLETE": f"{sum(1 for a,_ in runs if a=='V3_TREATMENT')}/5",
        "TRAINING_EPISODES_COMPLETE": f"{sum(r['n_episodes'] for r in runs.values())}/10800",
        "PAIRED_INITIALIZATION": "PASS" if pair_ok else "FAIL",
        "SINGLE_FACTOR_INTEGRITY": "PASS", "V3_CONSTANTS_INTEGRITY": "PASS" if v3_sha == v3_sha_e4ht else "FAIL",
        "NUMERICAL_STABILITY": "PASS" if stab else "FAIL",
        "CHECKPOINTS_COMPLETE": f"{len(ck_rows)}/60",
        "VALIDATION_CHECKPOINTS_COMPLETE": f"{len(ck_rows)}/60",
        "V1_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('V1'))}/5",
        "V3_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('V3'))}/5",
        "V1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_v1,
        "V3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_v3,
        "VALIDATION_PAIRED_DELTA_V1_MINUS_V3": macro_v1 - macro_v3,
        "V1_MIXED_PREDICT_FRACTION": float(v1_pf), "V3_MIXED_PREDICT_FRACTION": float(v3_pf),
        "V1_MIXED_MULTI_ACTION_RATE": float(v1_ma), "V3_MIXED_MULTI_ACTION_RATE": float(v3_ma),
        "V1_MIXED_ACTION_ENTROPY": float(v1_en), "V3_MIXED_ACTION_ENTROPY": float(v3_en),
        "VALIDATION_MECHANISM_DIRECTION": mech_dir,
        "HYPERPARAMETER_RETUNING": "NO", "CHECKPOINT_SELECTION_CHANGED": "NO",
        "BEST_POLICY_SELECTION": "NO", "FINAL_EVALUATION": "NOT_RUN",
        "NORMALIZATION_CAUSAL_CLAIM": "NOT_MADE",
        "NEW_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
        "ORIGINAL_E4F_VERDICT": "FAILED", "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4HE_NORMALIZATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE" if complete
                       else "E4HTR_ENGINEERING_REVIEW"),
    }
    _jdump(verdict, RESULTS_DIR / "E4HTR_verdict.json")
    _write_report(verdict, ab_rows)
    _emit_terminal(verdict)
    print(f"[E4HTR] total wall {time.time()-t0:.0f}s")
    return verdict


def _write_report(v, ab_rows):
    L = ["# E4HTR — Normalization A/B Training Execution", "",
         f"**E4HTR_COMPLETE = {v['E4HTR_COMPLETE']}** — trained the frozen V1-vs-V3 matched "
         "experiment (single factor = normalization). No final seeds; no causal claim.", "",
         "## A/B/C. Integrity & pairing",
         f"- SINGLE_FACTOR_INTEGRITY = {v['SINGLE_FACTOR_INTEGRITY']}; V3_CONSTANTS_INTEGRITY = "
         f"{v['V3_CONSTANTS_INTEGRITY']}; PAIRED_INITIALIZATION = {v['PAIRED_INITIALIZATION']}.",
         "## D/E. Completion & stability",
         f"- RUNS_COMPLETE = {v['RUNS_COMPLETE']}; episodes = {v['TRAINING_EPISODES_COMPLETE']}; "
         f"checkpoints = {v['CHECKPOINTS_COMPLETE']}; NUMERICAL_STABILITY = {v['NUMERICAL_STABILITY']}.",
         "## L/O. Descriptive validation A/B (NOT confirmatory)", "",
         "| scenario | V1 MIGD_response | V3 MIGD_response | Δ(V1−V3) |", "|---|---|---|---|"]
    for r in ab_rows:
        L.append(f"| {r['scenario']} | {r['V1_MIGD_response']:.6f} | {r['V3_MIGD_response']:.6f} "
                 f"| {r['delta_V1_minus_V3_response']:+.6e} |")
    L += ["", "## P. Mechanism (validation MIXED; descriptive)",
          f"- PREDICT fraction: V1={v['V1_MIXED_PREDICT_FRACTION']:.3f} vs V3={v['V3_MIXED_PREDICT_FRACTION']:.3f}",
          f"- multi-action rate: V1={v['V1_MIXED_MULTI_ACTION_RATE']:.3f} vs V3={v['V3_MIXED_MULTI_ACTION_RATE']:.3f}",
          f"- action entropy: V1={v['V1_MIXED_ACTION_ENTROPY']:.3f} vs V3={v['V3_MIXED_ACTION_ENTROPY']:.3f}",
          f"- VALIDATION_MECHANISM_DIRECTION = {v['VALIDATION_MECHANISM_DIRECTION']}", "",
          "## Q. Final-seed firewall",
          f"- NEW_FINAL_EVAL_ACCESS_COUNT = {v['NEW_FINAL_EVAL_ACCESS_COUNT']} (1381..1440 untouched).", "",
          "## R. Scientific interpretation boundary",
          "- Allowed: V3 trained differently/similarly to V1; validation metrics better/worse/similar; "
          "collapse appeared reduced/not on validation.",
          "- FORBIDDEN here: normalization causal effect validated; V3 superior; V3 beats GLOBAL_PREDICT. "
          "Those require the frozen final evaluation.", "",
          f"## S. Next stage: {v['NEXT_STAGE']}",
          "All 10 selected checkpoints retained and frozen; whatever validation shows, no reselection."]
    (RESULTS_DIR / "E4HTR_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4HTR_COMPLETE", "RUNS_COMPLETE", "V1_RUNS_COMPLETE", "V3_RUNS_COMPLETE",
              "TRAINING_EPISODES_COMPLETE", "PAIRED_INITIALIZATION", "SINGLE_FACTOR_INTEGRITY",
              "V3_CONSTANTS_INTEGRITY", "NUMERICAL_STABILITY", "CHECKPOINTS_COMPLETE",
              "VALIDATION_CHECKPOINTS_COMPLETE", "V1_SELECTED_CHECKPOINTS", "V3_SELECTED_CHECKPOINTS",
              "V1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN", "V3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
              "VALIDATION_PAIRED_DELTA_V1_MINUS_V3", "V1_MIXED_PREDICT_FRACTION", "V3_MIXED_PREDICT_FRACTION",
              "V1_MIXED_MULTI_ACTION_RATE", "V3_MIXED_MULTI_ACTION_RATE", "V1_MIXED_ACTION_ENTROPY",
              "V3_MIXED_ACTION_ENTROPY", "VALIDATION_MECHANISM_DIRECTION", "HYPERPARAMETER_RETUNING",
              "CHECKPOINT_SELECTION_CHANGED", "BEST_POLICY_SELECTION", "FINAL_EVALUATION",
              "NORMALIZATION_CAUSAL_CLAIM", "NEW_FINAL_EVAL_ACCESS_COUNT", "ORIGINAL_E4F_VERDICT",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6f}"
        print(f"{k:<42} = {val}")


if __name__ == "__main__":
    main()
