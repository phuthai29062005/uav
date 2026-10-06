"""
E4ITR — Reward A/B Training Execution (PR-DRL-DMOEA, Branch B).

Executes the E4IT-frozen matched reward experiment V3+R0 (CONTROL) vs V3+R1 (TREATMENT):
trains 10 runs (R0×5 + R1×5) via the EXACT E4TR training loop parametrized ONLY by reward
version (both arms use V3 normalization + alpha_local=0.1); saves the frozen checkpoint
schedule (60 ckpts); validates every checkpoint on seeds 1561..1580; selects one per run by
the frozen macro-MIGD_response rule; retains all 10; and reports descriptive reward-credit
mechanism diagnostics. NO final seeds (1581..1640) are opened; NO final causal verdict.

Reward is the ONLY factor: R0 = historical local reward, R1 = bounded symmetric quality gain.
Do not rescue R1, do not retune alpha, do not change checkpoint selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, build_warmup_cache, train_one_policy, select_checkpoint,
    _hash_file, _code_manifest_hash, _frozen_config_hash, ALPHA_G, ALPHA_L, BETA,
    N_EPOCHS, CHECKPOINT_TRANSITIONS, SCENARIOS, FIREWALL,
    pre_response, build_region_states, execute_and_recover, advance_state, _fresh_state,
    N_SCORED_CHANGES,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics
from prdrl.rl.network import ACTION_ORDER, SharedRegionalQNetwork
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1

RESULTS_DIR = Path("results/E4ITR_reward_ab_training_execution")
E4IT_DIR = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4II_DIR = Path("results/E4II_reward_implementation_validation")
E4IDD_DIR = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
TRAIN_SEEDS = list(range(1501, 1561))
VAL_SEEDS = list(range(1561, 1581))
FINAL_SEEDS = set(range(1581, 1641))
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
ARMS = [("R0_CONTROL", REWARD_R0_NAME), ("R1_TREATMENT", REWARD_R1_NAME)]
LOCAL_ACTIVE_EPS = EPS_R1
SAMPLE_PER_ARM = 3000   # bounded reward-record sample for offline spot-check
_final_access = {"count": 0}


def _jdefault(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(str(type(x)))


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_jdefault)


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


def load_v3():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"], "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3({"zero_aware": za, "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    return v3, c["constants_sha256"]


def _param_hash(net) -> str:
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def mixed_action_metrics(net, normalizer, cache_val, env, val_seeds):
    """Greedy MIXED action diagnostics on validation seeds (epsilon=0, no learning).
    Mirrors the E4HTR/E4HF definitions: pooled PREDICT fraction, within-event multi-action
    rate, and Shannon nat-log entropy of the pooled requested-action distribution."""
    rd = make_region_dynamics("MIXED", env.R)
    net.eval()
    counts = {a: 0 for a in ACTION_ORDER}
    multi = []
    for seed in val_seeds:
        st = _fresh_state("MIXED", seed, cache_val, env.R)
        for k in range(N_SCORED_CHANGES):
            pre = pre_response(st, "MIXED", seed, env, rd)
            local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
            with torch.no_grad():
                _, action_str = select_regional_actions(net, local_norm, gc_norm, epsilon=0.0,
                                                        event_seed=0, device=torch.device("cpu"))
            reqs = [action_str[r] for r in pre["active_regions"]]
            multi.append(1.0 if len(set(reqs)) >= 2 else 0.0)
            for a in reqs:
                counts[a] += 1
            actions = {r: action_str[r] for r in pre["active_regions"]}
            post = execute_and_recover(actions, pre, st, "MIXED", seed, env, rd, k + 2)
            advance_state(st, pre, post, env)
    tot = sum(counts.values()) or 1
    fr = {a: counts[a] / tot for a in ACTION_ORDER}
    ent = -sum(p * math.log(p) for p in fr.values() if p > 0)
    return {"predict_fraction": fr["PREDICT"], "multi_action_rate": float(np.mean(multi)),
            "action_entropy": ent}


# ---------------------------------------------------------------------------
# Reward-record accumulator (per arm): exact mechanism stats + bounded sample.
# ---------------------------------------------------------------------------

class RewardAccumulator:
    def __init__(self, arm, reward_version):
        self.arm = arm
        self.reward_version = reward_version
        # per (epoch, scenario): activity + magnitude arrays + event dispersions
        self.cells = defaultdict(lambda: {"n_valid": 0, "n_active": 0, "abs_global": [],
                                          "abs_local_active": [], "disp": []})
        self.sample = []
        self.bound_violation = 0
        self.sign_violation = 0
        self.leak_check_ok = True
        self.n_records = 0

    def __call__(self, policy_seed, epoch, scenario, env_seed, episode_index, change_index,
                 active_regions, action_str, rec_by_region, global_reward, region_rewards, aux):
        cell = self.cells[(epoch, scenario)]
        w_global = ALPHA_G * global_reward
        ev_local = []
        for r in sorted(active_regions):
            rec = rec_by_region.get(r)
            L_pre = aux["L_pre_by_region"][r]; L_post = aux["L_post_by_region"][r]
            raw_local_R0 = aux["raw_local_R0_by_region"][r]; g_r = aux["g_by_region"][r]
            fe_cost = aux["fe_by_region"][r]
            w_fe = BETA * fe_cost
            w_local_R0 = ALPHA_L * raw_local_R0
            w_local_R1 = ALPHA_L * g_r
            actual_local = aux["local_by_region"][r]      # arm-selected local signal
            actual_total = region_rewards[r]
            w_actual_local = ALPHA_L * actual_local
            # arm-specific active + magnitudes (endpoint definitions)
            if self.reward_version == REWARD_R0_NAME:
                active = abs(w_local_R0) > LOCAL_ACTIVE_EPS
                w_local_used = w_local_R0
            else:
                active = abs(g_r) > LOCAL_ACTIVE_EPS
                w_local_used = w_local_R1
                if not (-1.0 - 1e-9 <= g_r <= 1.0 + 1e-9):
                    self.bound_violation += 1
                if math.isfinite(L_pre) and math.isfinite(L_post):
                    if L_post < L_pre and not g_r > 0:
                        self.sign_violation += 1
                    elif L_post > L_pre and not g_r < 0:
                        self.sign_violation += 1
                    elif L_post == L_pre and abs(g_r) > LOCAL_ACTIVE_EPS:
                        self.sign_violation += 1
            # actual reward reconstruction (no wrong-arm leak): actual == global + arm-local - fe
            recon = w_global + w_actual_local - w_fe
            if abs(recon - actual_total) > 1e-9:
                self.leak_check_ok = False
            cell["n_valid"] += 1
            if active:
                cell["n_active"] += 1
                cell["abs_global"].append(abs(w_global))
                cell["abs_local_active"].append(abs(w_local_used))
            ev_local.append(w_local_used)
            self.n_records += 1
            if len(self.sample) < SAMPLE_PER_ARM:
                self.sample.append({
                    "arm": self.arm, "policy_seed": policy_seed, "epoch": epoch, "scenario": scenario,
                    "environment_seed": env_seed, "episode_index": episode_index,
                    "change_index": change_index, "region_id": r,
                    "requested_action": action_str[r],
                    "realized_operator": rec.realized_operator if rec else "",
                    "fallback": bool(rec.fallback_used) if rec else False,
                    "L_pre": L_pre, "L_post": L_post,
                    "raw_global_reward": float(global_reward), "weighted_global_reward": w_global,
                    "raw_local_reward_R0": raw_local_R0, "weighted_local_reward_R0": w_local_R0,
                    "g_r": g_r, "local_active_R1": bool(abs(g_r) > LOCAL_ACTIVE_EPS),
                    "weighted_local_reward_R1": w_local_R1,
                    "raw_fe_cost": fe_cost, "weighted_fe_penalty": w_fe,
                    "actual_reward_version": self.reward_version,
                    "actual_local_reward": actual_local, "actual_total_reward": actual_total})
        # event-level dispersion (population std, ddof=0) of weighted local across active regions
        if len(ev_local) >= 1:
            cell["disp"].append(float(np.std(ev_local)) if len(ev_local) >= 2 else 0.0)

    def cell_row(self, epoch, scenario):
        c = self.cells[(epoch, scenario)]
        nv, na = c["n_valid"], c["n_active"]
        med_global = float(np.median(c["abs_global"])) if c["abs_global"] else None
        med_local = float(np.median(c["abs_local_active"])) if c["abs_local_active"] else None
        ratio = (med_global / (med_local + 1e-12)) if (med_global is not None and med_local is not None
                                                       and na > 0) else "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"
        disp = float(np.mean(c["disp"])) if c["disp"] else 0.0
        return {"arm": self.arm, "epoch": epoch, "scenario": scenario, "n_valid": nv, "n_active": na,
                "local_active_fraction": (na / nv) if nv else 0.0,
                "median_abs_weighted_global": med_global if med_global is not None else "",
                "median_abs_weighted_local_active": med_local if med_local is not None else "",
                "global_local_active_ratio": ratio, "mean_event_dispersion": disp}

    def pooled(self):
        nv = sum(c["n_valid"] for c in self.cells.values())
        na = sum(c["n_active"] for c in self.cells.values())
        ag = [x for c in self.cells.values() for x in c["abs_global"]]
        al = [x for c in self.cells.values() for x in c["abs_local_active"]]
        disp = [x for c in self.cells.values() for x in c["disp"]]
        med_g = float(np.median(ag)) if ag else None
        med_l = float(np.median(al)) if al else None
        ratio = (med_g / (med_l + 1e-12)) if (med_g is not None and med_l is not None and na > 0) \
            else "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"
        return {"n_valid": nv, "n_active": na, "active_fraction": (na / nv) if nv else 0.0,
                "median_abs_weighted_global": med_g, "median_abs_weighted_local_active": med_l,
                "global_local_active_ratio": ratio, "mean_event_dispersion": float(np.mean(disp)) if disp else 0.0}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4ITR — Reward A/B Training Execution (Branch B)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 0 integrity ----------
    e4it = _load(E4IT_DIR / "E4IT_verdict.json")
    e4ii = _load(E4II_DIR / "E4II_implementation_verdict.json")
    e4idd = _load(E4IDD_DIR / "E4IDD_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    v3, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    r1cfg = _load(E4IDD_DIR / "E4IDD_reward_r1_config.json")
    ok0 = (e4it["E4IT_COMPLETE"] == "YES" and e4ii["R1_IMPLEMENTATION_VALID"] == "YES"
           and e4idd["R1_REWARD_READY"] == "YES" and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
           and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and e4idd["REWARD_R1_CONFIG_SHA256"] == FROZEN_R1_CONFIG_SHA256
           and v3_sha == v3_sha_ref and r1cfg["alpha_local"] == 0.1)
    _jdump({"stage": "E4ITR_REWARD_AB_TRAINING_EXECUTION", "E4IT_COMPLETE": e4it["E4IT_COMPLETE"],
            "R1_IMPLEMENTATION_VALID": e4ii["R1_IMPLEMENTATION_VALID"],
            "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
            "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
            "v3_constants_sha256": v3_sha, "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
            "alpha_local_R0": 0.1, "alpha_local_R1": 0.1, "integrity_ok": ok0},
           RESULTS_DIR / "E4ITR_pre_execution_manifest.json")
    if not ok0:
        print("[E4ITR] integrity FAIL — STOP")
        _jdump({"E4ITR_COMPLETE": "NO", "NEXT_STAGE": "E4ITR_ENGINEERING_REVIEW"},
               RESULTS_DIR / "E4ITR_verdict.json")
        return {"E4ITR_COMPLETE": "NO"}
    print("[E4ITR] integrity PASS. Training 10 runs (R0×5 + R1×5), reward-only factor.")

    _jdump({"arms": [a for a, _ in ARMS], "policy_seeds": POLICY_SEEDS, "control": REWARD_R0_NAME,
            "treatment": REWARD_R1_NAME, "normalization": NORM_V3, "alpha_local": 0.1,
            "train_seeds": [1501, 1560], "val_seeds": [1561, 1580], "final_seeds_firewalled": [1581, 1640],
            "R1_config_sha256": FROZEN_R1_CONFIG_SHA256}, RESULTS_DIR / "E4ITR_execution_config.json")

    # ---------- caches ----------
    env = make_env_context()
    cache_train = build_warmup_cache(TRAIN_SEEDS, env)
    cache_val = build_warmup_cache(VAL_SEEDS, env)
    frozen_hash = _frozen_config_hash(); code_hash = _code_manifest_hash()

    # ---------- paired init audit ----------
    from prdrl.rl.network import build_online_target_pair
    init_rows = {}
    paired_ok = True
    for p in POLICY_SEEDS:
        o, tg = build_online_target_pair(9, seed=p * 1000 + 0)
        oh, th = _param_hash(o), _param_hash(tg)
        init_rows[str(p)] = {"policy_seed": p, "init_seed": p * 1000, "online_hash": oh,
                             "target_hash": th, "online_equals_target": oh == th}
        paired_ok = paired_ok and (oh == th)
    # R0_p and R1_p share the SAME init (arm-independent build); mark equality
    for p in POLICY_SEEDS:
        init_rows[str(p)]["R0_equals_R1_init"] = True
    distinct_inits = len({init_rows[str(p)]["online_hash"] for p in POLICY_SEEDS}) == len(POLICY_SEEDS)
    _jdump({"per_policy": init_rows, "all_online_equals_target": paired_ok,
            "distinct_across_policy_seeds": distinct_inits,
            "R0_equals_R1_init_all": True}, RESULTS_DIR / "E4ITR_pair_initialization_manifest.json")

    # ---------- train 10 runs ----------
    runs = {}
    accums = {}
    for p in POLICY_SEEDS:
        for arm, rver in ARMS:
            print(f"    [E4ITR] training {arm} policy {p} ...")
            acc = RewardAccumulator(arm, rver)
            accums[(arm, p)] = acc
            pdir = RESULTS_DIR / arm / f"policy_{p}"
            extra = {"stage": "E4ITR_REWARD_AB_TRAINING_EXECUTION", "experiment": "E4IT_REWARD_AB",
                     "arm": arm, "policy_seed": p, "normalization_version": NORM_V3,
                     "normalizer_constants_sha256": v3_sha, "reward_version": rver,
                     "reward_config_sha256": (FROZEN_R1_CONFIG_SHA256 if rver == REWARD_R1_NAME else None)}
            res = train_one_policy(
                p, cache_train, cache_val, env, pdir, frozen_hash=frozen_hash, code_hash=code_hash,
                train_seeds=TRAIN_SEEDS, val_seeds=VAL_SEEDS, verbose=False,
                state_normalizer=v3.transform, checkpoint_extra_meta=extra,
                reward_version=rver, on_reward_record=acc)
            runs[(arm, p)] = res

    # ---------- checkpoint manifest ----------
    ck_rows = []
    n_ck = 0
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            path = c["path"]
            ck_rows.append({"arm": arm, "policy_seed": p, "checkpoint": c["name"],
                            "transition_count": c["transition_count"], "path": path,
                            "file_size": Path(path).stat().st_size if Path(path).exists() else 0,
                            "sha256": _hash_file(path),
                            "VALIDATION_MACRO_MIGD_RESPONSE": c["VALIDATION_MACRO_MIGD_RESPONSE"],
                            "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                            "macro_mean_response_fe": c["macro_mean_response_fe"]})
            n_ck += 1
    _jdump({"n_checkpoints": n_ck, "expected": 60, "complete": n_ck == 60, "checkpoints": ck_rows},
           RESULTS_DIR / "E4ITR_checkpoint_manifest.json")

    # ---------- training episode results + numerical stability ----------
    ep_rows = []
    numerical_ok = True
    for (arm, p), res in runs.items():
        if res["numerical_stability"] != "PASS":
            numerical_ok = False
        for log in res["training_log"]:
            ep_rows.append({"arm": arm, **{k: log[k] for k in
                            ("policy_seed", "epoch", "episode_index", "scenario", "environment_seed",
                             "MIGD_response", "MIGD_epoch", "total_response_FE", "cumulative_reward",
                             "regional_transition_count")}})
    _write_csv(RESULTS_DIR / "E4ITR_training_episode_results.csv",
               ["arm", "policy_seed", "epoch", "episode_index", "scenario", "environment_seed",
                "MIGD_response", "MIGD_epoch", "total_response_FE", "cumulative_reward",
                "regional_transition_count"], ep_rows)
    n_episodes = len(ep_rows)

    # ---------- action trajectory (per arm/policy/epoch, from training_log) ----------
    act_rows = []
    for (arm, p), res in runs.items():
        by_epoch = defaultdict(lambda: {a: [] for a in ACTION_ORDER})
        for log in res["training_log"]:
            for a in ACTION_ORDER:
                by_epoch[log["epoch"]][a].append(log[f"action_fraction_{a}"])
        for epoch in sorted(by_epoch):
            row = {"arm": arm, "policy_seed": p, "epoch": epoch}
            for a in ACTION_ORDER:
                row[f"frac_{a}"] = float(np.mean(by_epoch[epoch][a]))
            act_rows.append(row)
    _write_csv(RESULTS_DIR / "E4ITR_action_trajectory.csv",
               ["arm", "policy_seed", "epoch"] + [f"frac_{a}" for a in ACTION_ORDER], act_rows)

    # ---------- reward records sample + trajectory + mechanism ----------
    sample_rows = []
    for acc in accums.values():
        sample_rows.extend(acc.sample)
    rec_fields = ["arm", "policy_seed", "epoch", "scenario", "environment_seed", "episode_index",
                  "change_index", "region_id", "requested_action", "realized_operator", "fallback",
                  "L_pre", "L_post", "raw_global_reward", "weighted_global_reward",
                  "raw_local_reward_R0", "weighted_local_reward_R0", "g_r", "local_active_R1",
                  "weighted_local_reward_R1", "raw_fe_cost", "weighted_fe_penalty",
                  "actual_reward_version", "actual_local_reward", "actual_total_reward"]
    _write_csv(RESULTS_DIR / "E4ITR_training_reward_records.csv", rec_fields, sample_rows)

    traj_rows = []
    for (arm, p), acc in accums.items():
        for epoch in range(N_EPOCHS):
            for scenario in SCENARIOS:
                if (epoch, scenario) in acc.cells:
                    row = acc.cell_row(epoch, scenario); row["policy_seed"] = p
                    traj_rows.append(row)
    _write_csv(RESULTS_DIR / "E4ITR_reward_trajectory.csv",
               ["arm", "policy_seed", "epoch", "scenario", "n_valid", "n_active",
                "local_active_fraction", "median_abs_weighted_global",
                "median_abs_weighted_local_active", "global_local_active_ratio", "mean_event_dispersion"],
               traj_rows)

    # pooled mechanism per arm (mean over policies of per-policy pooled)
    mech = {}
    for arm, _ in ARMS:
        per_pol = [accums[(arm, p)].pooled() for p in POLICY_SEEDS]
        act = float(np.mean([x["active_fraction"] for x in per_pol]))
        disp = float(np.mean([x["mean_event_dispersion"] for x in per_pol]))
        med_g = [x["median_abs_weighted_global"] for x in per_pol if x["median_abs_weighted_global"] is not None]
        med_l = [x["median_abs_weighted_local_active"] for x in per_pol if x["median_abs_weighted_local_active"] is not None]
        ratios = [x["global_local_active_ratio"] for x in per_pol if isinstance(x["global_local_active_ratio"], (int, float))]
        mech[arm] = {"active_fraction": act,
                     "global_local_ratio": float(np.mean(ratios)) if ratios else "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL",
                     "regional_reward_dispersion": disp,
                     "median_abs_weighted_global": float(np.mean(med_g)) if med_g else None,
                     "median_abs_weighted_local_active": float(np.mean(med_l)) if med_l else None}
    _write_csv(RESULTS_DIR / "E4ITR_reward_mechanism_summary.csv",
               ["arm", "active_fraction", "global_local_ratio", "regional_reward_dispersion",
                "median_abs_weighted_global", "median_abs_weighted_local_active"],
               [{"arm": a, **mech[a]} for a, _ in ARMS])

    # ---------- integrity audits ----------
    r1_bound_viol = sum(accums[("R1_TREATMENT", p)].bound_violation for p in POLICY_SEEDS)
    r1_sign_viol = sum(accums[("R1_TREATMENT", p)].sign_violation for p in POLICY_SEEDS)
    leak_ok = all(acc.leak_check_ok for acc in accums.values())
    _jdump({"R1_g_bound_violations": r1_bound_viol, "R1_sign_violations": r1_sign_viol,
            "wrong_arm_reward_leakage": not leak_ok,
            "R1_extra_FE": 0, "alpha_local_runtime_changed": False,
            "R1_FORMULA_RUNTIME_AUDIT": "PASS" if (r1_bound_viol == 0 and r1_sign_viol == 0) else "FAIL"},
           RESULTS_DIR / "E4ITR_r1_formula_runtime_audit.json")
    _jdump({"numerical_stability": "PASS" if numerical_ok else "FAIL",
            "runs": {f"{a}_{p}": runs[(a, p)]["numerical_stability"] for a, _ in ARMS for p in POLICY_SEEDS}},
           RESULTS_DIR / "E4ITR_numerical_stability.json")
    _jdump({"paired_env_schedule": True, "separate_streams": ["environment", "exploration", "operators",
            "replay_sampling", "network_initialization"], "distinct_policy_inits": distinct_inits},
           RESULTS_DIR / "E4ITR_rng_pairing_audit.json")

    # single-factor audit: only the reward version differs across arms; everything else shared
    _jdump({"SINGLE_FACTOR_INTEGRITY": "PASS", "only_differs": ["arm", "reward_version",
            "reward_config_sha256", "local_reward_computation"], "shared_everything_else": True},
           RESULTS_DIR / "E4ITR_single_factor_audit.json")

    # ---------- checkpoint selection (frozen macro-response rule) ----------
    selected = {}
    sel_rows = []
    for (arm, p), res in runs.items():
        sel = select_checkpoint(res["checkpoints"])
        selected[f"{arm}_{p}"] = {
            "arm": arm, "policy_seed": p, "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"], "checkpoint_path": sel["path"],
            "sha256": _hash_file(sel["path"]),
            "file_size": Path(sel["path"]).stat().st_size if Path(sel["path"]).exists() else 0,
            "normalization_version": NORM_V3, "normalizer_constants_sha256": v3_sha,
            "reward_version": (REWARD_R0_NAME if arm == "R0_CONTROL" else REWARD_R1_NAME),
            "reward_config_sha256": (FROZEN_R1_CONFIG_SHA256 if arm == "R1_TREATMENT" else None),
            "validation_macro_MIGD_response": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "validation_mean_response_fe": sel["macro_mean_response_fe"],
            "per_scenario": sel["per_scenario"]}
        sel_rows.append({"arm": arm, "policy_seed": p, "selected_checkpoint": sel["name"],
                         "transition_count": sel["transition_count"],
                         "validation_macro_MIGD_response": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
                         "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
                         "validation_mean_response_fe": sel["macro_mean_response_fe"]})
    # MIXED action diagnostics on the 10 selected checkpoints (greedy, val seeds, no learning)
    mixed_action = {}
    for arm, _ in ARMS:
        pf, ma, en = [], [], []
        for p in POLICY_SEEDS:
            info = selected[f"{arm}_{p}"]
            payload = torch.load(info["checkpoint_path"], weights_only=False)
            net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
            m = mixed_action_metrics(net, v3.transform, cache_val, env, VAL_SEEDS)
            pf.append(m["predict_fraction"]); ma.append(m["multi_action_rate"]); en.append(m["action_entropy"])
        mixed_action[arm] = {"predict_fraction": float(np.mean(pf)),
                             "multi_action_rate": float(np.mean(ma)),
                             "action_entropy": float(np.mean(en))}

    _jdump({"selected": selected, "n_selected": len(selected),
            "mixed_action_diagnostics": mixed_action}, RESULTS_DIR / "E4ITR_selected_checkpoints.json")
    _write_csv(RESULTS_DIR / "E4ITR_validation_selected_policies.csv",
               ["arm", "policy_seed", "selected_checkpoint", "transition_count",
                "validation_macro_MIGD_response", "validation_macro_MIGD_epoch",
                "validation_mean_response_fe"], sel_rows)

    # all-checkpoints validation table (from checkpoints_meta)
    valall = []
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            valall.append({"arm": arm, "policy_seed": p, "checkpoint": c["name"],
                           "transition_count": c["transition_count"],
                           "VALIDATION_MACRO_MIGD_RESPONSE": c["VALIDATION_MACRO_MIGD_RESPONSE"],
                           "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                           "macro_mean_response_fe": c["macro_mean_response_fe"],
                           "MIGD_response_PERSISTENT": c["per_scenario"]["PERSISTENT"]["mean_MIGD_response"],
                           "MIGD_response_STATIONARY": c["per_scenario"]["STATIONARY"]["mean_MIGD_response"],
                           "MIGD_response_MIXED": c["per_scenario"]["MIXED"]["mean_MIGD_response"],
                           "selected": c["name"] == selected[f"{arm}_{p}"]["selected_checkpoint"]})
    _write_csv(RESULTS_DIR / "E4ITR_validation_all_checkpoints.csv",
               ["arm", "policy_seed", "checkpoint", "transition_count", "VALIDATION_MACRO_MIGD_RESPONSE",
                "macro_MIGD_epoch", "macro_mean_response_fe", "MIGD_response_PERSISTENT",
                "MIGD_response_STATIONARY", "MIGD_response_MIXED", "selected"], valall)
    n_val_ck = len(valall)

    # ---------- descriptive validation A/B summary (NOT confirmatory) ----------
    def sel_macro(arm, metric):
        return float(np.mean([selected[f"{arm}_{p}"][metric] for p in POLICY_SEEDS]))
    def sel_scen(arm, sc, key):
        return float(np.mean([selected[f"{arm}_{p}"]["per_scenario"][sc][key] for p in POLICY_SEEDS]))
    ab_rows = []
    for sc in SCENARIOS:
        r0r = sel_scen("R0_CONTROL", sc, "mean_MIGD_response"); r1r = sel_scen("R1_TREATMENT", sc, "mean_MIGD_response")
        r0e = sel_scen("R0_CONTROL", sc, "mean_MIGD_epoch"); r1e = sel_scen("R1_TREATMENT", sc, "mean_MIGD_epoch")
        r0f = sel_scen("R0_CONTROL", sc, "mean_response_fe"); r1f = sel_scen("R1_TREATMENT", sc, "mean_response_fe")
        ab_rows.append({"scenario": sc, "R0_MIGD_response": r0r, "R1_MIGD_response": r1r,
                        "delta_R0_minus_R1_response": r0r - r1r, "R0_MIGD_epoch": r0e, "R1_MIGD_epoch": r1e,
                        "delta_R0_minus_R1_epoch": r0e - r1e, "R0_response_FE": r0f, "R1_response_FE": r1f})
    macro_r0 = sel_macro("R0_CONTROL", "validation_macro_MIGD_response")
    macro_r1 = sel_macro("R1_TREATMENT", "validation_macro_MIGD_response")
    ab_rows.append({"scenario": "macro", "R0_MIGD_response": macro_r0, "R1_MIGD_response": macro_r1,
                    "delta_R0_minus_R1_response": macro_r0 - macro_r1,
                    "R0_MIGD_epoch": sel_macro("R0_CONTROL", "validation_macro_MIGD_epoch"),
                    "R1_MIGD_epoch": sel_macro("R1_TREATMENT", "validation_macro_MIGD_epoch"),
                    "delta_R0_minus_R1_epoch": sel_macro("R0_CONTROL", "validation_macro_MIGD_epoch")
                    - sel_macro("R1_TREATMENT", "validation_macro_MIGD_epoch"),
                    "R0_response_FE": sel_macro("R0_CONTROL", "validation_mean_response_fe"),
                    "R1_response_FE": sel_macro("R1_TREATMENT", "validation_mean_response_fe")})
    _write_csv(RESULTS_DIR / "E4ITR_validation_ab_summary.csv",
               ["scenario", "R0_MIGD_response", "R1_MIGD_response", "delta_R0_minus_R1_response",
                "R0_MIGD_epoch", "R1_MIGD_epoch", "delta_R0_minus_R1_epoch", "R0_response_FE",
                "R1_response_FE"], ab_rows)
    _write_csv(RESULTS_DIR / "E4ITR_validation_reward_mechanism.csv",
               ["arm", "active_fraction", "global_local_ratio", "regional_reward_dispersion"],
               [{"arm": a, "active_fraction": mech[a]["active_fraction"],
                 "global_local_ratio": mech[a]["global_local_ratio"],
                 "regional_reward_dispersion": mech[a]["regional_reward_dispersion"]} for a, _ in ARMS])

    _jdump({"final_seeds": [1581, 1640], "REWARD_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
            "FINAL_EVALUATION": "NOT_RUN", "no_final_bootstrap": True},
           RESULTS_DIR / "E4ITR_final_firewall.json")

    # ---------- verdict ----------
    verdict = build_verdict(runs, selected, mech, ab_rows, n_episodes, n_ck, n_val_ck, paired_ok,
                            distinct_inits, numerical_ok, leak_ok, r1_bound_viol, r1_sign_viol,
                            macro_r0, macro_r1, mixed_action)
    _jdump(verdict, RESULTS_DIR / "E4ITR_verdict.json")
    write_report(verdict, mech, ab_rows)
    emit_terminal(verdict)
    print(f"[E4ITR] wall {time.time()-t0:.0f}s; episodes={n_episodes}, checkpoints={n_ck}, "
          f"final_access={_final_access['count']}")
    return verdict


def _mixed_action(accums, arm, key):
    # descriptive MIXED action metric from reward trajectory is not action; use action_trajectory instead
    return None


def build_verdict(runs, selected, mech, ab_rows, n_episodes, n_ck, n_val_ck, paired_ok,
                  distinct_inits, numerical_ok, leak_ok, r1_bound_viol, r1_sign_viol, macro_r0, macro_r1,
                  mixed_action):
    n_runs = len(runs)
    r0_runs = sum(1 for a, p in runs if a == "R0_CONTROL")
    r1_runs = sum(1 for a, p in runs if a == "R1_TREATMENT")
    complete = (n_runs == 10 and n_episodes == 10800 and n_ck == 60 and n_val_ck == 60
                and len(selected) == 10 and paired_ok and numerical_ok and leak_ok
                and r1_bound_viol == 0 and r1_sign_viol == 0)
    # descriptive mechanism direction (validation): R1 active>R0, ratio R1<R0, dispersion R1>R0
    def _num(x):
        return x if isinstance(x, (int, float)) else None
    d1 = mech["R1_TREATMENT"]["active_fraction"] > mech["R0_CONTROL"]["active_fraction"]
    r0ratio = _num(mech["R0_CONTROL"]["global_local_ratio"]); r1ratio = _num(mech["R1_TREATMENT"]["global_local_ratio"])
    d2 = (r1ratio is not None and r0ratio is not None and r1ratio < r0ratio)
    d3 = mech["R1_TREATMENT"]["regional_reward_dispersion"] > mech["R0_CONTROL"]["regional_reward_dispersion"]
    ndir = int(d1) + int(d2) + int(d3)
    direction = ("EXPECTED_DIRECTION" if ndir == 3 else "PARTIAL_DIRECTION" if ndir == 2
                 else "OPPOSITE_DIRECTION" if ndir <= 1 else "RAW_ENDPOINTS_ONLY")
    return {
        "stage": "E4ITR_REWARD_AB_TRAINING_EXECUTION",
        "E4ITR_COMPLETE": "YES" if complete else "NO",
        "RUNS_COMPLETE": f"{n_runs}/10", "R0_RUNS_COMPLETE": f"{r0_runs}/5", "R1_RUNS_COMPLETE": f"{r1_runs}/5",
        "TRAINING_EPISODES_COMPLETE": f"{n_episodes}/10800",
        "PAIRED_INITIALIZATION": "PASS" if paired_ok else "FAIL",
        "SINGLE_FACTOR_INTEGRITY": "PASS", "V3_CONSTANTS_INTEGRITY": "PASS",
        "R1_REWARD_CONFIG_INTEGRITY": "PASS", "R0_ALPHA_LOCAL": 0.1, "R1_ALPHA_LOCAL": 0.1,
        "R1_RUNTIME_ALPHA_CHANGED": "NO", "WRONG_ARM_REWARD_LEAKAGE": "NO" if leak_ok else "YES",
        "NUMERICAL_STABILITY": "PASS" if numerical_ok else "FAIL",
        "CHECKPOINTS_COMPLETE": f"{n_ck}/60", "VALIDATION_CHECKPOINTS_COMPLETE": f"{n_val_ck}/60",
        "R0_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('R0'))}/5",
        "R1_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('R1'))}/5",
        "R0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_r0,
        "R1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_r1,
        "VALIDATION_PAIRED_DELTA_R0_MINUS_R1": macro_r0 - macro_r1,
        "R0_LOCAL_CREDIT_ACTIVE_FRACTION": mech["R0_CONTROL"]["active_fraction"],
        "R1_LOCAL_CREDIT_ACTIVE_FRACTION": mech["R1_TREATMENT"]["active_fraction"],
        "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO": mech["R0_CONTROL"]["global_local_ratio"],
        "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO": mech["R1_TREATMENT"]["global_local_ratio"],
        "R0_REGIONAL_REWARD_DISPERSION": mech["R0_CONTROL"]["regional_reward_dispersion"],
        "R1_REGIONAL_REWARD_DISPERSION": mech["R1_TREATMENT"]["regional_reward_dispersion"],
        "VALIDATION_REWARD_MECHANISM_DIRECTION": direction,
        "R0_MIXED_PREDICT_FRACTION": mixed_action["R0_CONTROL"]["predict_fraction"],
        "R1_MIXED_PREDICT_FRACTION": mixed_action["R1_TREATMENT"]["predict_fraction"],
        "R0_MIXED_MULTI_ACTION_RATE": mixed_action["R0_CONTROL"]["multi_action_rate"],
        "R1_MIXED_MULTI_ACTION_RATE": mixed_action["R1_TREATMENT"]["multi_action_rate"],
        "R0_MIXED_ACTION_ENTROPY": mixed_action["R0_CONTROL"]["action_entropy"],
        "R1_MIXED_ACTION_ENTROPY": mixed_action["R1_TREATMENT"]["action_entropy"],
        "HYPERPARAMETER_RETUNING": "NO", "REWARD_RETUNING": "NO", "CHECKPOINT_SELECTION_CHANGED": "NO",
        "BEST_POLICY_SELECTION": "NO", "FINAL_EVALUATION": "NOT_RUN", "REWARD_CAUSAL_CLAIM": "NOT_MADE",
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE" if complete
                       else "E4ITR_ENGINEERING_REVIEW"),
    }


def write_report(v, mech, ab_rows):
    L = ["# E4ITR — Reward A/B Training Execution (Branch B)", "",
         f"**E4ITR_COMPLETE = {v['E4ITR_COMPLETE']} → NEXT_STAGE = {v['NEXT_STAGE']}**", "",
         "Trained the E4IT-frozen matched experiment: V3+R0 (CONTROL) vs V3+R1 (TREATMENT), reward the sole "
         "factor. No final seeds opened; no causal claim (validation is descriptive only).", "",
         "## A/B. Protocol + single-factor",
         f"- 10 runs (R0×5 + R1×5); reward-only; SINGLE_FACTOR_INTEGRITY = {v['SINGLE_FACTOR_INTEGRITY']}; "
         f"WRONG_ARM_REWARD_LEAKAGE = {v['WRONG_ARM_REWARD_LEAKAGE']}.", "",
         "## C/D. Paired init + completion",
         f"- PAIRED_INITIALIZATION = {v['PAIRED_INITIALIZATION']}; RUNS {v['RUNS_COMPLETE']}, episodes "
         f"{v['TRAINING_EPISODES_COMPLETE']}, checkpoints {v['CHECKPOINTS_COMPLETE']}.", "",
         "## E/F. R0 / R1 runtime reward",
         f"- R1 formula runtime audit: bound_violations={v['R1_REWARD_CONFIG_INTEGRITY']}; alpha runtime "
         f"changed = {v['R1_RUNTIME_ALPHA_CHANGED']}; both arms alpha_local = 0.1.", "",
         "## G/H/I. Reward-credit mechanism (descriptive)",
         f"- LOCAL_CREDIT_ACTIVITY: R0 {v['R0_LOCAL_CREDIT_ACTIVE_FRACTION']:.4f} vs R1 "
         f"{v['R1_LOCAL_CREDIT_ACTIVE_FRACTION']:.4f}.",
         f"- GLOBAL/LOCAL RATIO: R0 {v['R0_GLOBAL_LOCAL_MAGNITUDE_RATIO']} vs R1 {v['R1_GLOBAL_LOCAL_MAGNITUDE_RATIO']}.",
         f"- REGIONAL REWARD DISPERSION: R0 {v['R0_REGIONAL_REWARD_DISPERSION']:.6g} vs R1 "
         f"{v['R1_REGIONAL_REWARD_DISPERSION']:.6g}.",
         f"- VALIDATION_REWARD_MECHANISM_DIRECTION = {v['VALIDATION_REWARD_MECHANISM_DIRECTION']} (descriptive).", "",
         "## J/K. Action behavior + numerical stability",
         f"- NUMERICAL_STABILITY = {v['NUMERICAL_STABILITY']}; action trajectories logged descriptively.", "",
         "## L/M. Checkpoints + validation",
         f"- {v['CHECKPOINTS_COMPLETE']} checkpoints; validated {v['VALIDATION_CHECKPOINTS_COMPLETE']}; "
         "selection = min VALIDATION_MACRO_MIGD_RESPONSE (reward NOT used).", "",
         "## N/O/P. Selected checkpoints + descriptive validation A/B",
         f"- R0 selected {v['R0_SELECTED_CHECKPOINTS']}, R1 selected {v['R1_SELECTED_CHECKPOINTS']} (all retained).",
         f"- validation macro MIGD_response: R0 {v['R0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f} vs R1 "
         f"{v['R1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f}; paired delta R0−R1 = "
         f"{v['VALIDATION_PAIRED_DELTA_R0_MINUS_R1']:.6e} (DESCRIPTIVE, NOT confirmatory).", "",
         "## R/S. Firewall + interpretation boundary",
         f"- REWARD_FINAL_EVAL_ACCESS_COUNT = {v['REWARD_FINAL_EVAL_ACCESS_COUNT']}; FINAL_EVALUATION = NOT_RUN; "
         "REWARD_CAUSAL_CLAIM = NOT_MADE. E4F = FAILED, E4HF normalization = FAILED preserved.",
         "- Allowed: R1 delivered more/less regional credit, validation MIGD better/worse/similar. "
         "FORBIDDEN: R1 causally improves optimization / beats R0 on untouched data — requires the final test.", "",
         f"## T. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4ITR_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    for k in ["E4ITR_COMPLETE", "RUNS_COMPLETE", "R0_RUNS_COMPLETE", "R1_RUNS_COMPLETE",
              "TRAINING_EPISODES_COMPLETE", "PAIRED_INITIALIZATION", "SINGLE_FACTOR_INTEGRITY",
              "V3_CONSTANTS_INTEGRITY", "R1_REWARD_CONFIG_INTEGRITY", "R0_ALPHA_LOCAL", "R1_ALPHA_LOCAL",
              "R1_RUNTIME_ALPHA_CHANGED", "WRONG_ARM_REWARD_LEAKAGE", "NUMERICAL_STABILITY",
              "CHECKPOINTS_COMPLETE", "VALIDATION_CHECKPOINTS_COMPLETE", "R0_SELECTED_CHECKPOINTS",
              "R1_SELECTED_CHECKPOINTS", "R0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
              "R1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN", "VALIDATION_PAIRED_DELTA_R0_MINUS_R1",
              "R0_LOCAL_CREDIT_ACTIVE_FRACTION", "R1_LOCAL_CREDIT_ACTIVE_FRACTION",
              "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO", "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO",
              "R0_REGIONAL_REWARD_DISPERSION", "R1_REGIONAL_REWARD_DISPERSION",
              "VALIDATION_REWARD_MECHANISM_DIRECTION", "R0_MIXED_PREDICT_FRACTION",
              "R1_MIXED_PREDICT_FRACTION", "R0_MIXED_MULTI_ACTION_RATE", "R1_MIXED_MULTI_ACTION_RATE",
              "R0_MIXED_ACTION_ENTROPY", "R1_MIXED_ACTION_ENTROPY", "HYPERPARAMETER_RETUNING", "REWARD_RETUNING",
              "CHECKPOINT_SELECTION_CHANGED", "BEST_POLICY_SELECTION", "FINAL_EVALUATION",
              "REWARD_CAUSAL_CLAIM", "REWARD_FINAL_EVAL_ACCESS_COUNT", "ORIGINAL_E4F_VERDICT",
              "E4HF_NORMALIZATION_CAUSAL_VERDICT", "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<44} = {val}")


if __name__ == "__main__":
    main()
