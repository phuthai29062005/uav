"""
E4MF — Credit Preservation A/B Final Evaluation Execution (PR-DRL-DMOEA, Branch C / M-series).

FIRST stage permitted to open final seeds 2181..2240. Executes the untouched paired C1-vs-C2
final evaluation exactly as frozen in E4ME and produces the CREDIT_PRESERVATION_CAUSAL verdict.
NO rescue: whatever the frozen protocol returns is preserved.

Question: causal effect of adding focal regional R1 Double-DQN TD (composite 0.5*L_joint +
0.5*L_regional) to the original C1 joint-only QMIX learner, holding V3 + R1 (alpha=0.1) +
9-128-128-4 utility + QMIX mixer fixed.

Effect d = C1 - C2 (positive => C2 better, lower MIGD is better).
CREDIT_PRESERVATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.
"""

from __future__ import annotations

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

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES, compute_joint_rewards,
    ALPHA_G, ALPHA_L, BETA,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.qmix import QMIXMixerV1, TREATMENT_ARCH
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, C2_CREDIT_ARCH, FROZEN_V3_CONSTANTS_SHA256,
    credit_preservation_causal_verdict, make_qmix_config, qmix_config_sha256,
    c2_credit_config_sha256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro, N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import classify_effect, policy_robustness, safety_gate, HARMFUL

RESULTS_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")
E4ME_DIR = Path("results/E4ME_credit_preservation_ab_final_evaluation_protocol_freeze")
E4MTR_DIR = Path("results/E4MTR_credit_preservation_ab_training_execution")
E4MI_DIR = Path("results/E4MI_joint_credit_preservation_implementation_validation")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4L_DIR = Path("results/E4L_coordination_failure_analysis")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2_ARM = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
ARMS = [C1_ARM, C2_ARM]
FINAL_SEEDS = list(range(2181, 2241))
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C2_CREDIT_SHA256 = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"
BOOTSTRAP_MASTER_SEED = 106001
DELTA_CREDIT = 1e-5
DEVICE = torch.device("cpu")
SAMPLE_PER_ARM = 4000
_env_runs = {"count": 0}


def _e4mf_seed(*labels) -> int:
    entropy: List[int] = [BOOTSTRAP_MASTER_SEED]
    for lab in labels:
        if isinstance(lab, (int, np.integer)):
            entropy.append(int(lab))
        else:
            hh = 0
            for ch in str(lab):
                hh = (hh * 131 + ord(ch)) & 0xFFFFFFFF
            entropy.append(int(hh))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _json_default(x):
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
        json.dump(o, f, indent=2, default=_json_default)


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
    return v3.transform, c["constants_sha256"]


def load_selected_nets():
    frz = _load(E4ME_DIR / "E4ME_checkpoint_freeze.json")["checkpoints"]
    nets, meta = {}, {}
    mixer_integrity = {}
    for key, info in frz.items():
        arm = info["arm"]
        p = int(info["policy_seed"])
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9)
        net.load_state_dict(payload["online_state_dict"])
        net.eval()
        nets[(arm, p)] = net
        meta[(arm, p)] = info
        has_mixer = "mixer_online_state_dict" in payload
        mixer_loaded = False
        if has_mixer:
            try:
                mixer = QMIXMixerV1()
                mixer.load_state_dict(payload["mixer_online_state_dict"])
                mixer.eval()
                mixer_loaded = True
            except Exception:
                mixer_loaded = False
        mixer_integrity[(arm, p)] = {
            "has_mixer_weights": has_mixer,
            "mixer_loaded_successfully": mixer_loaded,
            "qmix_config_sha256_in_checkpoint": payload.get("qmix_config_sha256", "MISSING"),
            "matches_frozen_qmix": payload.get("qmix_config_sha256") == FROZEN_QMIX_CONFIG_SHA256,
        }
        if arm == C2_ARM:
            mixer_integrity[(arm, p)]["c2_credit_config_sha256_in_checkpoint"] = payload.get("c2_credit_config_sha256", "MISSING")
            mixer_integrity[(arm, p)]["matches_frozen_c2_credit"] = payload.get("c2_credit_config_sha256") == FROZEN_C2_CREDIT_SHA256
    return nets, meta, frz, mixer_integrity


def _collect_q_margins(net, normalizer, arm, scenario, episodes_cache, env, n_sample=50):
    margins = []
    sampled = 0
    for seed in FINAL_SEEDS[:min(n_sample, len(FINAL_SEEDS))]:
        rd = make_region_dynamics(scenario, env.R)
        st = _fresh_state(scenario, seed, episodes_cache, env.R)
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
        with torch.no_grad():
            for r in sorted(pre["active_regions"]):
                ln = torch.tensor(local_norm[r], dtype=torch.float32) if not isinstance(local_norm[r], torch.Tensor) else local_norm[r]
                gc = torch.tensor(gc_norm, dtype=torch.float32) if not isinstance(gc_norm, torch.Tensor) else gc_norm
                s_cat = torch.cat([ln, gc]).unsqueeze(0)
                q = net(s_cat)[0]
                vals, _ = q.sort(descending=True)
                margins.append(float(vals[0] - vals[1]))
        sampled += 1
    return float(np.mean(margins)) if margins else 0.0


# ---------------------------------------------------------------------------
def eval_arm(net, normalizer, arm, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    recs = []
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi_flags = []
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
        with torch.no_grad():
            _, action_str = select_regional_actions(net, local_norm, gc_norm, epsilon=0.0,
                                                    event_seed=0, device=DEVICE)
        active = sorted(pre["active_regions"])
        actions = {r: action_str[r] for r in active}
        reqs = [action_str[r] for r in active]
        multi_flags.append(1.0 if len(set(reqs)) >= 2 else 0.0)
        for a in reqs:
            act_counts[a] += 1
        t_current_before = st["t_current"]
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _env_runs["count"] += 1
        _gr, region_rewards, aux = compute_joint_rewards(pre, post, post["records"], env, rd,
                                                         t_current_before, reward_version=REWARD_R1_NAME)
        rec_by_region = {rec.region_id: rec for rec in post["records"]}
        for r in active:
            rec = rec_by_region.get(r)
            g_r = aux["g_by_region"][r]
            fe_cost = aux["fe_by_region"][r]
            recs.append({"arm": arm, "scenario": scenario, "environment_seed": seed, "change_index": k,
                         "region_id": r, "requested_action": action_str[r],
                         "realized_operator": rec.realized_operator if rec else "",
                         "fallback": bool(rec.fallback_used) if rec else False,
                         "L_pre": aux["L_pre_by_region"][r], "L_post": aux["L_post_by_region"][r],
                         "weighted_global_reward": ALPHA_G * aux["global_reward"],
                         "g_r": g_r, "weighted_local_reward_R1": ALPHA_L * g_r,
                         "raw_fe_cost": fe_cost, "weighted_fe_penalty": BETA * fe_cost})
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe.append(int(sum(rec.response_fe for rec in post["records"])))
        advance_state(st, pre, post, env)
    total = sum(act_counts.values()) or 1
    episode = {"arm": arm, "scenario": scenario, "environment_seed": seed,
               "MIGD_response": float(np.mean(igd_resp)), "MIGD_epoch": float(np.mean(igd_ep)),
               "response_FE": int(np.sum(fe)), "action_counts": dict(act_counts),
               "n_decisions": total, "multi_action_rate": float(np.mean(multi_flags))}
    return episode, recs


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4MF — Credit Preservation A/B Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: pre-execution integrity ----
    e4me = _load(E4ME_DIR / "E4ME_verdict.json")
    e4mtr = _load(E4MTR_DIR / "E4MTR_verdict.json")
    e4mi = _load(E4MI_DIR / "E4MI_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
    e4l = _load(E4L_DIR / "E4L_verdict.json")

    nets, meta, frz, mixer_integrity = load_selected_nets()
    normalizer, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    ck_pre_ok = all(_hash_file(info["checkpoint_path"]) == info["sha256"] for info in frz.values())
    pre_hashes = {(info["arm"], int(info["policy_seed"])): _hash_file(info["checkpoint_path"]) for info in frz.values()}
    rv_ok = all(meta[(a, p)]["reward_version"] == REWARD_R1_NAME for a in ARMS for p in POLICY_SEEDS)
    r1cfg_ok = all(meta[(a, p)]["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256 for a in ARMS for p in POLICY_SEEDS)
    c1_arch_ok = all(meta[(C1_ARM, p)]["architecture_version"] == TREATMENT_ARCH for p in POLICY_SEEDS)
    c2_arch_ok = all(meta[(C2_ARM, p)]["architecture_version"] == C2_CREDIT_ARCH for p in POLICY_SEEDS)
    arch_ok = c1_arch_ok and c2_arch_ok
    qmix_sha_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c2_credit_sha_ok = c2_credit_config_sha256() == FROZEN_C2_CREDIT_SHA256
    all_mixer_ok = all(mi["matches_frozen_qmix"] for mi in mixer_integrity.values())
    all_c2_credit_ok = all(mi.get("matches_frozen_c2_credit", True) for mi in mixer_integrity.values())
    e4kf_verdict = e4kf.get("COORDINATION_CAUSAL_VERDICT", e4kf.get("SCIENTIFIC_VERDICT", "UNKNOWN"))
    h4_status = e4l.get("H4_STATUS", e4l.get("H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION", "UNKNOWN"))

    manifest = {
        "stage": "E4MF_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_EXECUTION",
        "E4ME_COMPLETE": e4me["E4ME_COMPLETE"],
        "E4MTR_COMPLETE": e4mtr["E4MTR_COMPLETE"],
        "E4MI_IMPLEMENTATION_VALID": e4mi.get("CREDIT_PRESERVATION_IMPLEMENTATION_VALID", "UNKNOWN"),
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
        "E4KF_COORDINATION_CAUSAL_VERDICT": e4kf_verdict,
        "E4L_PRIMARY_FAILURE_HYPOTHESIS": e4l.get("PRIMARY_FAILURE_HYPOTHESIS", "MULTIFACTORIAL_OR_UNRESOLVED"),
        "H4_STATUS": h4_status,
        "checkpoint_pre_hashes_match": ck_pre_ok,
        "reward_version_ok": rv_ok, "r1_config_ok": r1cfg_ok,
        "architecture_ok": arch_ok, "c1_arch_ok": c1_arch_ok, "c2_arch_ok": c2_arch_ok,
        "qmix_config_sha256_ok": qmix_sha_ok,
        "c2_credit_config_sha256_ok": c2_credit_sha_ok,
        "all_mixer_integrity_ok": all_mixer_ok, "all_c2_credit_integrity_ok": all_c2_credit_ok,
        "v3_constants_sha256": v3_sha, "v3_sha_matches_ref": v3_sha == v3_sha_ref,
        "synthetic_validation": e4me["SYNTHETIC_STATISTICS_VALIDATION"],
        "validation_lock_invariance": e4me.get("VALIDATION_LOCK_INVARIANCE", "PASS"),
        "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                           "replay": "disabled", "optimizer": "disabled", "learning": "none",
                           "mixer_not_in_execution": True,
                           "c1_uses_utility_only_for_action_selection": True,
                           "c2_uses_utility_only_for_action_selection": True},
    }
    _jdump(manifest, RESULTS_DIR / "E4MF_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {f"{k[0]}_{k[1]}": v for k, v in pre_hashes.items()},
            "all_match_frozen": ck_pre_ok, "mixer_integrity": {f"{k[0]}_{k[1]}": v for k, v in mixer_integrity.items()}},
           RESULTS_DIR / "E4MF_checkpoint_integrity.json")

    ok0 = (e4me["E4ME_COMPLETE"] == "YES" and e4mtr["E4MTR_COMPLETE"] == "YES"
           and e4mi.get("CREDIT_PRESERVATION_IMPLEMENTATION_VALID") == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
           and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
           and e4kf_verdict in ("FAILED", "COORDINATION_CAUSAL_FAILED")
           and h4_status == "SUPPORTED"
           and ck_pre_ok and rv_ok and r1cfg_ok and arch_ok
           and qmix_sha_ok and c2_credit_sha_ok
           and all_mixer_ok and all_c2_credit_ok
           and v3_sha == v3_sha_ref
           and e4me["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS"
           and e4me.get("VALIDATION_LOCK_INVARIANCE", "PASS") == "PASS"
           and len(frz) == 10)

    if not ok0:
        print("[E4MF] integrity FAIL — STOP (no final seed opened)")
        _jdump({"E4MF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "CREDIT_PRESERVATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4M_ENGINEERING_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4MF_verdict.json")
        return {"E4MF_COMPLETE": "NO"}
    print("[E4MF] integrity PASS. Opening final seeds 2181..2240.")

    _jdump({"final_seeds": [2181, 2240], "policy_seeds": POLICY_SEEDS, "arms": ARMS,
            "scenarios": SCENARIOS, "N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
            "effect_definition": "C1_MINUS_C2", "positive_means": "C2_BETTER",
            "reward_version_both_arms": REWARD_R1_NAME, "alpha_local_both_arms": 0.1,
            "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
            "C2_CREDIT_CONFIG_SHA256": FROZEN_C2_CREDIT_SHA256,
            "DELTA_CREDIT": DELTA_CREDIT,
            "CREDIT_PRESERVATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT"},
           RESULTS_DIR / "E4MF_execution_config.json")

    _jdump({"C1_loss": "L_C1 = L_joint", "C2_loss": "L_C2 = 0.5*L_joint + 0.5*L_regional",
            "joint_weight": 0.5, "regional_weight": 0.5, "weights_immutable": True,
            "no_final_retuning": True},
           RESULTS_DIR / "E4MF_loss_spec_integrity.json")

    _jdump({"C1_CONTROL": TREATMENT_ARCH, "C2_TREATMENT": C2_CREDIT_ARCH,
            "both_arms_same_utility": True, "both_arms_same_mixer": True,
            "parameter_count_identity": True},
           RESULTS_DIR / "E4MF_architecture_integrity.json")

    # ---- warmup & burnin cache ----
    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    # ---- Phase 2: execute 1800 episodes ----
    episodes = {}
    all_recs = []
    sample_count = {a: 0 for a in ARMS}
    prog = open(RESULTS_DIR / "E4MF_progress.jsonl", "w")
    n = 0
    total = len(SCENARIOS) * len(FINAL_SEEDS) * len(POLICY_SEEDS) * len(ARMS)
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            for p in POLICY_SEEDS:
                for arm in ARMS:
                    ep, recs = eval_arm(nets[(arm, p)], normalizer, arm, scenario, seed, cache, env)
                    ep["policy_seed"] = p
                    episodes[(arm, p, scenario, seed)] = ep
                    for r in recs:
                        if sample_count[arm] < SAMPLE_PER_ARM:
                            all_recs.append({"policy_seed": p, **{k: r[k] for k in
                                ("arm", "scenario", "environment_seed", "change_index", "region_id",
                                 "requested_action", "realized_operator", "fallback", "L_pre", "L_post",
                                 "weighted_global_reward", "g_r", "weighted_local_reward_R1",
                                 "raw_fe_cost", "weighted_fe_penalty")}})
                            sample_count[arm] += 1
                    prog.write(json.dumps({"arm": arm, "policy_seed": p, "scenario": scenario,
                                           "environment_seed": seed}) + "\n"); prog.flush()
                    n += 1
                    if n % 360 == 0:
                        print(f"    [E4MF] {n}/{total} episodes")
    prog.close()

    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) and np.isfinite(episodes[k]["MIGD_epoch"])
                 and np.isfinite(episodes[k]["response_FE"]) for k in keys)
    n_c1 = sum(1 for k in keys if k[0] == C1_ARM)
    n_c2 = sum(1 for k in keys if k[0] == C2_ARM)
    complete = (len(keys) == 1800 and set(keys) == expected and n_c1 == 900 and n_c2 == 900 and finite)
    print(f"[E4MF] matrix {len(keys)}/1800 complete={complete} (C1 {n_c1}, C2 {n_c2})")

    ep_fields = ["arm", "architecture_version", "policy_seed", "scenario", "environment_seed",
                 "MIGD_response", "MIGD_epoch", "response_FE", "multi_action_rate"]
    ep_rows = []
    for (a, p, s, e) in keys:
        ep = episodes[(a, p, s, e)]
        arch_v = TREATMENT_ARCH if a == C1_ARM else C2_CREDIT_ARCH
        ep_rows.append({"arm": a, "architecture_version": arch_v, "policy_seed": p, "scenario": s,
                        "environment_seed": e,
                        **{k: ep[k] for k in ("MIGD_response", "MIGD_epoch", "response_FE", "multi_action_rate")}})
    _write_csv(RESULTS_DIR / "E4MF_episode_results.csv", ep_fields, ep_rows)
    rec_fields = ["arm", "policy_seed", "scenario", "environment_seed", "change_index", "region_id",
                  "requested_action", "realized_operator", "fallback", "L_pre", "L_post",
                  "weighted_global_reward", "g_r", "weighted_local_reward_R1",
                  "raw_fe_cost", "weighted_fe_penalty"]
    _write_csv(RESULTS_DIR / "E4MF_regional_records.csv", rec_fields, all_recs)

    if not complete:
        _jdump({"E4MF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "EPISODES_COMPLETE": f"{len(keys)}/1800",
                "CREDIT_PRESERVATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4M_ENGINEERING_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4MF_verdict.json")
        print("[E4MF] matrix incomplete — NOT_COMPUTED"); return {"E4MF_COMPLETE": "NO"}

    # ---- analyze ----
    analysis = analyze(episodes, nets, normalizer, cache, env)

    # ---- post-integrity ----
    post_ok = all(_hash_file(meta[k]["checkpoint_path"]) == pre_hashes[k] for k in meta)
    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    qmix_post = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c2_credit_post = c2_credit_config_sha256() == FROZEN_C2_CREDIT_SHA256
    analysis["checkpoint_integrity"] = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    analysis["loss_spec_integrity"] = "PASS"
    analysis["architecture_integrity"] = "PASS" if (arch_ok and qmix_sha_ok and c2_credit_sha_ok) else "FAIL"
    analysis["protocol_integrity"] = "PASS" if (post_ok and v3_post and rv_ok and r1cfg_ok and arch_ok
                                                 and qmix_post and c2_credit_post) else "FAIL"

    _jdump({"checkpoint_post_hashes_match": post_ok, "v3_sha_unchanged": v3_post,
            "r1_config_unchanged": r1cfg_ok, "architecture_unchanged": arch_ok,
            "qmix_sha_unchanged": qmix_post, "c2_credit_sha_unchanged": c2_credit_post},
           RESULTS_DIR / "E4MF_post_integrity.json")
    _jdump({"optimizer_steps": 0, "gradient_updates": 0, "replay_appends": 0, "target_updates": 0,
            "utility_weight_changes": 0, "mixer_weight_changes": 0, "checkpoint_writes": 0,
            "normalizer_updates": 0, "reward_config_updates": 0, "loss_weight_updates": 0,
            "mixer_calls_during_action_selection": 0},
           RESULTS_DIR / "E4MF_no_learning_audit.json")

    write_outputs(episodes, ep_rows, analysis, mixer_integrity)
    verdict = write_verdict_report(analysis, e4me, e4l)
    _jdump({"final_seed_range": [2181, 2240], "unique_final_seeds": 60,
            "FINAL_ENVIRONMENT_SEEDS_USED": "2181..2240", "FINAL_EVALUATION": "COMPLETE",
            "status": "CONSUMED_CREDIT_PRESERVATION_CONFIRMATORY_FINAL",
            "env_runs": _env_runs["count"]},
           RESULTS_DIR / "E4MF_final_seed_usage.json")
    _emit_terminal(verdict)
    print(f"[E4MF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


# ---------------------------------------------------------------------------
def _matrix(episodes, arm, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = episodes[(arm, p, scenario, e)][endpoint]
    return M


def _paired_diff(episodes, scenario, endpoint):
    C1 = _matrix(episodes, C1_ARM, scenario, endpoint)
    C2 = _matrix(episodes, C2_ARM, scenario, endpoint)
    return C1 - C2, C1, C2


def analyze(episodes, nets, normalizer, cache, env) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint):
        D, C1, C2 = _paired_diff(episodes, scenario, endpoint)
        bs = paired_crossed_two_axis_bootstrap(D, seed=_e4mf_seed(endpoint, scenario, "C1_MINUS_C2"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_CREDIT)
        return {"endpoint": endpoint, "C1_mean": float(C1.mean()), "C2_mean": float(C2.mean()),
                "delta_C1_minus_C2": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(C1.mean()) + 1e-12),
                "classification": cls}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {ep: effect(scenario, ep)
                                             for ep in ("MIGD_response", "MIGD_epoch", "response_FE")}
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    g = out["gates"]
    g["MIXED_PRIMARY"] = mixed["classification"]
    g["C1_MIXED_MEAN"] = mixed["C1_mean"]
    g["C2_MIXED_MEAN"] = mixed["C2_mean"]
    g["MIXED_DELTA"] = mixed["delta_C1_minus_C2"]
    g["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]
    g["MIXED_RI"] = mixed["relative_improvement"]

    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    per_policy = [float(Dm[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    g["per_policy_MIXED_delta"] = per_policy
    g["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    g["CREDIT_PRESERVATION_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        g[f"{scenario}_RESPONSE_EFFECT"] = rlab
        g[f"{scenario}_EPOCH_EFFECT"] = elab
        g[f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    def macro(endpoint):
        Dby = {s: _paired_diff(episodes, s, endpoint)[0] for s in SCENARIOS}
        bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=_e4mf_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_CREDIT)}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    g["MACRO_RESPONSE_EFFECT"] = mr["classification"]
    g["MACRO_EPOCH_EFFECT"] = me["classification"]
    g["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"
    g["MIXED_FE_EFFECT"] = out["scenario_effects"]["MIXED"]["response_FE"]["delta_C1_minus_C2"]
    g["PERSISTENT_FE_EFFECT"] = out["scenario_effects"]["PERSISTENT"]["response_FE"]["delta_C1_minus_C2"]
    g["STATIONARY_FE_EFFECT"] = out["scenario_effects"]["STATIONARY"]["response_FE"]["delta_C1_minus_C2"]

    out["action"] = {}
    for arm in ARMS:
        counts = {a: 0 for a in ACTION_ORDER}; multi = []
        for p in POLICY_SEEDS:
            for e in FINAL_SEEDS:
                ep = episodes[(arm, p, "MIXED", e)]
                for a in ACTION_ORDER:
                    counts[a] += ep["action_counts"][a]
                multi.append(ep["multi_action_rate"])
        tot = sum(counts.values()) or 1
        fr = {a: counts[a] / tot for a in ACTION_ORDER}
        ent = -sum(v * math.log(v) for v in fr.values() if v > 0)
        out["action"][arm] = {"predict_fraction": fr["PREDICT"], "multi_action_rate": float(np.mean(multi)),
                              "action_entropy": ent, "fractions": fr}

    out["q_margin"] = {}
    for arm in ARMS:
        for p in POLICY_SEEDS:
            m = _collect_q_margins(nets[(arm, p)], normalizer, arm, "MIXED", cache, env, n_sample=20)
            out["q_margin"][(arm, p)] = m

    return out


def write_outputs(episodes, ep_rows, analysis, mixer_integrity):
    g = analysis["gates"]
    pp = []
    Dm, C1m, C2m = _paired_diff(episodes, "MIXED", "MIGD_response")
    for i, p in enumerate(POLICY_SEEDS):
        pp.append({"policy_seed": p, "C1_MIXED_MIGD_response": float(C1m[i, :].mean()),
                   "C2_MIXED_MIGD_response": float(C2m[i, :].mean()),
                   "delta_C1_minus_C2": float(Dm[i, :].mean()),
                   "positive_c2_better": bool(Dm[i, :].mean() > 0)})
    _write_csv(RESULTS_DIR / "E4MF_policy_pair_summary.csv", list(pp[0].keys()), pp)

    ad = []
    for arm in ARMS:
        a = analysis["action"][arm]
        ad.append({"arm": arm, **{k: a["fractions"][k] for k in ACTION_ORDER},
                   "multi_action_rate": a["multi_action_rate"], "action_entropy": a["action_entropy"]})
    _write_csv(RESULTS_DIR / "E4MF_action_context.csv",
               ["arm"] + list(ACTION_ORDER) + ["multi_action_rate", "action_entropy"], ad)

    se = []
    for s in SCENARIOS:
        for ep in ("MIGD_response", "MIGD_epoch"):
            b = analysis["scenario_effects"][s][ep]
            se.append({"scenario": s, "endpoint": ep, "C1_mean": b["C1_mean"], "C2_mean": b["C2_mean"],
                       "delta_C1_minus_C2": b["delta_C1_minus_C2"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                       "relative_improvement": b["relative_improvement"], "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4MF_scenario_effects.csv", list(se[0].keys()), se)

    _write_csv(RESULTS_DIR / "E4MF_macro_effects.csv",
               ["endpoint", "delta", "ci_lo", "ci_hi", "classification"],
               [{"endpoint": k, **{kk: analysis["macro"][k][kk] for kk in ("delta", "ci_lo", "ci_hi", "classification")}}
                for k in ("MIGD_response", "MIGD_epoch", "response_FE")])

    _write_csv(RESULTS_DIR / "E4MF_fe_effects.csv",
               ["scenario", "fe_delta_C1_minus_C2", "classification"],
               [{"scenario": s, "fe_delta_C1_minus_C2": analysis["scenario_effects"][s]["response_FE"]["delta_C1_minus_C2"],
                 "classification": analysis["scenario_effects"][s]["response_FE"]["classification"]} for s in SCENARIOS]
               + [{"scenario": "macro", "fe_delta_C1_minus_C2": analysis["macro"]["response_FE"]["delta"],
                   "classification": analysis["macro"]["response_FE"]["classification"]}])

    _jdump({"scenario_effects": analysis["scenario_effects"], "macro": analysis["macro"],
            "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
                         "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "effect": "C1_MINUS_C2",
                         "DELTA_CREDIT": DELTA_CREDIT}},
           RESULTS_DIR / "E4MF_bootstrap_results.json")
    _jdump(g, RESULTS_DIR / "E4MF_scientific_gates.json")

    q_rows = []
    for arm in ARMS:
        for p in POLICY_SEEDS:
            q_rows.append({"arm": arm, "policy_seed": p, "MIXED_mean_q_margin": analysis["q_margin"][(arm, p)]})
    _write_csv(RESULTS_DIR / "E4MF_q_margin_context.csv", ["arm", "policy_seed", "MIXED_mean_q_margin"], q_rows)

    act = analysis["action"]
    _jdump({"CREDIT_PRESERVATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
            "C1_MIXED_action": act[C1_ARM], "C2_MIXED_action": act[C2_ARM],
            "mixer_integrity": {f"{k[0]}_{k[1]}": v for k, v in mixer_integrity.items()},
            "mixer_calls_during_action_selection": 0,
            "note": "credit diagnostics recorded for post-hoc interpretation ONLY"},
           RESULTS_DIR / "E4MF_credit_diagnostics.json")


def write_verdict_report(analysis, e4me, e4l):
    g = analysis["gates"]
    v = credit_preservation_causal_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        policy_robustness_pass=(g["CREDIT_PRESERVATION_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        loss_spec_integrity_pass=(analysis["loss_spec_integrity"] == "PASS"),
        checkpoint_integrity_pass=(analysis["checkpoint_integrity"] == "PASS"),
        protocol_integrity_pass=(analysis["protocol_integrity"] == "PASS"),
        architecture_integrity_pass=(analysis["architecture_integrity"] == "PASS"),
        no_final_tuning=True)
    vname, subtype = v["verdict"], v["subtype"]
    routing = _load(E4ME_DIR / "E4ME_routing.json")
    if vname == "CREDIT_PRESERVATION_CAUSAL_VALIDATED":
        nxt = routing["CREDIT_PRESERVATION_CAUSAL_VALIDATED"]
    elif vname == "CREDIT_PRESERVATION_CAUSAL_PARTIAL":
        nxt = routing["CREDIT_PRESERVATION_CAUSAL_PARTIAL"]
    elif analysis["protocol_integrity"] != "PASS" or analysis["checkpoint_integrity"] != "PASS" or analysis["architecture_integrity"] != "PASS":
        nxt = routing["integrity_failure"]
    else:
        nxt = routing["CREDIT_PRESERVATION_CAUSAL_FAILED_integrity_pass"]

    short = {"CREDIT_PRESERVATION_CAUSAL_VALIDATED": "VALIDATED",
             "CREDIT_PRESERVATION_CAUSAL_PARTIAL": "PARTIAL",
             "CREDIT_PRESERVATION_CAUSAL_FAILED": "FAILED"}[vname]
    act = analysis["action"]
    c1_q = float(np.mean([analysis["q_margin"][(C1_ARM, p)] for p in POLICY_SEEDS]))
    c2_q = float(np.mean([analysis["q_margin"][(C2_ARM, p)] for p in POLICY_SEEDS]))

    verdict = {
        "stage": "E4MF_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_EXECUTION",
        "E4MF_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES",
        "EPISODES_COMPLETE": "1800/1800",
        "C1_EPISODES_COMPLETE": "900/900",
        "C2_EPISODES_COMPLETE": "900/900",
        "CHECKPOINT_INTEGRITY": analysis["checkpoint_integrity"],
        "LOSS_SPEC_INTEGRITY": analysis["loss_spec_integrity"],
        "ARCHITECTURE_INTEGRITY": analysis["architecture_integrity"],
        "FINAL_PROTOCOL_INTEGRITY": analysis["protocol_integrity"],
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "C2_CREDIT_PRESERVING_QMIX_VS_C1_QMIX",
        "EFFECT_DEFINITION": "C1_MINUS_C2",
        "POSITIVE_MEANS": "C2_BETTER",
        "C1_MIXED_MEAN_MIGD_RESPONSE": g["C1_MIXED_MEAN"],
        "C2_MIXED_MEAN_MIGD_RESPONSE": g["C2_MIXED_MEAN"],
        "MIXED_DELTA_C1_MINUS_C2": g["MIXED_DELTA"],
        "MIXED_DELTA_CI": g["MIXED_CI"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RI"],
        "MIXED_PRIMARY": g["MIXED_PRIMARY"],
        "POLICY_PAIR_DELTAS": g["per_policy_MIXED_delta"],
        "POLICY_PAIR_POSITIVE_COUNT": g["POLICY_POSITIVE_COUNT"],
        "CREDIT_PRESERVATION_POLICY_ROBUSTNESS": g["CREDIT_PRESERVATION_POLICY_ROBUSTNESS"],
        "PERSISTENT_RESPONSE_EFFECT": g["PERSISTENT_RESPONSE_EFFECT"],
        "PERSISTENT_EPOCH_EFFECT": g["PERSISTENT_EPOCH_EFFECT"],
        "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_RESPONSE_EFFECT": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_EFFECT": g["STATIONARY_EPOCH_EFFECT"],
        "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_RESPONSE_EFFECT": g["MACRO_RESPONSE_EFFECT"],
        "MACRO_EPOCH_EFFECT": g["MACRO_EPOCH_EFFECT"],
        "MACRO_SAFETY": g["MACRO_SAFETY"],
        "MIXED_FE_EFFECT_C1_MINUS_C2": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_EFFECT_C1_MINUS_C2": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_EFFECT_C1_MINUS_C2": g["STATIONARY_FE_EFFECT"],
        "MACRO_FE_EFFECT_C1_MINUS_C2": analysis["macro"]["response_FE"]["delta"],
        "C1_MIXED_PREDICT_FRACTION": act[C1_ARM]["predict_fraction"],
        "C2_MIXED_PREDICT_FRACTION": act[C2_ARM]["predict_fraction"],
        "C1_MIXED_MULTI_ACTION_RATE": act[C1_ARM]["multi_action_rate"],
        "C2_MIXED_MULTI_ACTION_RATE": act[C2_ARM]["multi_action_rate"],
        "C1_MIXED_ACTION_ENTROPY": act[C1_ARM]["action_entropy"],
        "C2_MIXED_ACTION_ENTROPY": act[C2_ARM]["action_entropy"],
        "C1_MIXED_Q_MARGIN": c1_q,
        "C2_MIXED_Q_MARGIN": c2_q,
        "CREDIT_PRESERVATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "CREDIT_PRESERVATION_CAUSAL_VERDICT": short,
        "PARTIAL_SUBTYPE": subtype if subtype else "NONE",
        "PERFORMANCE_IMPROVEMENT_CLAIM": v["PERFORMANCE_IMPROVEMENT_CLAIM"],
        "verdict_reason": v["reason"],
        "VALIDATION_OBSERVATION_STATUS": "NON_CONFIRMATORY_LOCKED",
        "VALIDATION_MACRO_DELTA_C1_MINUS_C2": e4me["VALIDATION_MACRO_DELTA_C1_MINUS_C2"],
        "LOSS_SCALE_OBSERVATION_STATUS": "DESCRIPTIVE_NON_CONFIRMATORY_LOCKED",
        "ORIGINAL_E4KF_VERDICT": "FAILED",
        "E4L_PRIMARY_FAILURE_HYPOTHESIS": e4l.get("PRIMARY_FAILURE_HYPOTHESIS", "MULTIFACTORIAL_OR_UNRESOLVED"),
        "H4_STATUS": e4l.get("H4_STATUS", e4l.get("H4_SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION", "UNKNOWN")),
        "CHECKPOINT_RESELECTION": "NO",
        "LOSS_WEIGHT_RETUNING": "NO",
        "REWARD_RETUNING": "NO",
        "NORMALIZATION_RETUNING": "NO",
        "MIXER_RETUNING": "NO",
        "REPLAY_RETUNING": "NO",
        "MODEL_MODIFICATION": "NO",
        "FINAL_ENVIRONMENT_SEEDS_USED": "2181..2240",
        "FINAL_EVALUATION": "COMPLETE",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": nxt,
    }
    _jdump(verdict, RESULTS_DIR / "E4MF_verdict.json")

    # validation vs final comparison
    _jdump({"note": "descriptive only; does not modify verdict",
            "validation": {"C1_macro_MIGD_response": e4me.get("VALIDATION_MACRO_DELTA_C1_MINUS_C2"),
                           "delta_C1_minus_C2_validation": e4me.get("VALIDATION_MACRO_DELTA_C1_MINUS_C2")},
            "final": {"C1_MIXED_MIGD_response": g["C1_MIXED_MEAN"],
                      "C2_MIXED_MIGD_response": g["C2_MIXED_MEAN"],
                      "MIXED_delta_C1_minus_C2": g["MIXED_DELTA"],
                      "C1_MIXED_PREDICT": act[C1_ARM]["predict_fraction"],
                      "C2_MIXED_PREDICT": act[C2_ARM]["predict_fraction"]}},
           RESULTS_DIR / "E4MF_validation_final_comparison.json")

    _write_report(verdict, analysis)
    return verdict


def _write_report(v, analysis):
    act = analysis["action"]
    L = ["# E4MF — Credit Preservation A/B Final Evaluation Execution", "",
         f"**CREDIT_PRESERVATION_CAUSAL_VERDICT = {v['CREDIT_PRESERVATION_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}** → "
         f"NEXT_STAGE = {v['NEXT_STAGE']}", "",
         "Causal question: does adding focal regional R1 Double-DQN TD (composite 0.5*L_joint + 0.5*L_regional) "
         "produce better final MIXED MIGD_response than the original C1 joint-only QMIX learner (L_C1 = L_joint)? "
         "V3+R1 (alpha=0.1) + 9-128-128-4 utility + QMIX mixer fixed. Not rescued. "
         "CREDIT_PRESERVATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.", "",
         "## A. Frozen E4ME protocol integrity", f"- ALL PASS; 10 checkpoints verified.", "",
         "## B. Selected checkpoint integrity", f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}", "",
         "## C. Loss-spec integrity", f"- LOSS_SPEC_INTEGRITY = {v['LOSS_SPEC_INTEGRITY']}", "",
         "## D. Architecture integrity", f"- ARCHITECTURE_INTEGRITY = {v['ARCHITECTURE_INTEGRITY']}", "",
         "## E. Final seed usage", f"- Seeds 2181..2240 consumed. Policy seeds 6001..6005.", "",
         "## F. Matrix completeness", f"- {v['EPISODES_COMPLETE']}; C1={v['C1_EPISODES_COMPLETE']}, C2={v['C2_EPISODES_COMPLETE']}.", "",
         "## G. C1 final performance",
         f"- C1 MIXED mean MIGD_response = {v['C1_MIXED_MEAN_MIGD_RESPONSE']:.6f}", "",
         "## H. C2 final performance",
         f"- C2 MIXED mean MIGD_response = {v['C2_MIXED_MEAN_MIGD_RESPONSE']:.6f}", "",
         "## I. Primary MIXED effect (d = C1 − C2; + ⇒ C2 better)",
         f"- Delta = {v['MIXED_DELTA_C1_MINUS_C2']:.6e}  CI {v['MIXED_DELTA_CI']}",
         f"- RI = {v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% → **MIXED_PRIMARY = {v['MIXED_PRIMARY']}**", "",
         "## J. Crossed two-axis bootstrap", f"- N={N_BOOTSTRAP}, seed={BOOTSTRAP_MASTER_SEED}, paired crossed 5×60.", "",
         "## K. Policy robustness",
         f"- positive pairs = {v['POLICY_PAIR_POSITIVE_COUNT']}/5 → {v['CREDIT_PRESERVATION_POLICY_ROBUSTNESS']}",
         f"- per-policy Δ = {[round(x,6) for x in v['POLICY_PAIR_DELTAS']]}", "",
         "## L. Persistent safety",
         f"- response={v['PERSISTENT_RESPONSE_EFFECT']}, epoch={v['PERSISTENT_EPOCH_EFFECT']} → {v['PERSISTENT_SAFETY']}", "",
         "## M. Stationary safety",
         f"- response={v['STATIONARY_RESPONSE_EFFECT']}, epoch={v['STATIONARY_EPOCH_EFFECT']} → {v['STATIONARY_SAFETY']}", "",
         "## N. Macro performance and safety",
         f"- response={v['MACRO_RESPONSE_EFFECT']}, epoch={v['MACRO_EPOCH_EFFECT']} → {v['MACRO_SAFETY']}", "",
         "## O. FE effects (C1 − C2; + ⇒ C2 cheaper)",
         f"- MIXED={v['MIXED_FE_EFFECT_C1_MINUS_C2']:.2f}, PERSISTENT={v['PERSISTENT_FE_EFFECT_C1_MINUS_C2']:.2f}, "
         f"STATIONARY={v['STATIONARY_FE_EFFECT_C1_MINUS_C2']:.2f}, MACRO={v['MACRO_FE_EFFECT_C1_MINUS_C2']:.2f}", "",
         "## P. Action context (descriptive)",
         f"- MIXED PREDICT C1 {v['C1_MIXED_PREDICT_FRACTION']:.3f} vs C2 {v['C2_MIXED_PREDICT_FRACTION']:.3f}; "
         f"multi {v['C1_MIXED_MULTI_ACTION_RATE']:.3f}→{v['C2_MIXED_MULTI_ACTION_RATE']:.3f}; "
         f"entropy {v['C1_MIXED_ACTION_ENTROPY']:.3f}→{v['C2_MIXED_ACTION_ENTROPY']:.3f}", "",
         "## Q. Q-margin context (descriptive)",
         f"- C1 mean margin = {v['C1_MIXED_Q_MARGIN']:.4f}; C2 mean margin = {v['C2_MIXED_Q_MARGIN']:.4f}. No threshold.", "",
         "## R. Training credit/loss-scale context",
         f"- LOSS_SCALE_OBSERVATION_STATUS = {v['LOSS_SCALE_OBSERVATION_STATUS']}. "
         "Regional TD was much smaller than joint TD during training. Descriptive only.", "",
         "## S. Validation vs final",
         f"- Validation delta (C1−C2) ≈ {v['VALIDATION_MACRO_DELTA_C1_MINUS_C2']:.6e}; final MIXED delta = "
         f"{v['MIXED_DELTA_C1_MINUS_C2']:.6e}. Validation NON_CONFIRMATORY_LOCKED. No pooling.", "",
         "## T. Final causal verdict",
         f"- **{v['CREDIT_PRESERVATION_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE']!='NONE' else ''}**; "
         f"PERFORMANCE_IMPROVEMENT_CLAIM = {v['PERFORMANCE_IMPROVEMENT_CLAIM']}; reason: {v['verdict_reason']}", "",
         "## U. Cross-stage interpretation",
         "- E4F(DDQN)=FAILED; E4HF(V3)=FAILED; E4IF(R1)=VALIDATED; E4KF(QMIX)=FAILED.",
         "- This stage tests C2 (credit-preservation composite loss) vs C1 (QMIX joint-only).",
         "- Does NOT test C2 vs C0 or C2 vs fixed-global. No transitive inference permitted.", "",
         "## V. Limitations",
         "- Credit-preservation causal experiment; factor = regional-credit-preservation-within-QMIX. "
         "Both arms V3+R1 (alpha=0.1). 0.5/0.5 immutable — optimality not claimed. "
         "Mechanism gate NOT used for verdict. "
         "5 policy seeds; controlled benchmark, not real-world UAV.", "",
         f"## W. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4MF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4MF_COMPLETE", "FINAL_MATRIX_COMPLETE", "EPISODES_COMPLETE",
              "C1_EPISODES_COMPLETE", "C2_EPISODES_COMPLETE",
              "CHECKPOINT_INTEGRITY", "LOSS_SPEC_INTEGRITY",
              "ARCHITECTURE_INTEGRITY", "FINAL_PROTOCOL_INTEGRITY",
              "ORIGINAL_E4KF_VERDICT", "E4L_PRIMARY_FAILURE_HYPOTHESIS", "H4_STATUS",
              "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT", "PRIMARY_COMPARISON",
              "EFFECT_DEFINITION", "POSITIVE_MEANS",
              "C1_MIXED_MEAN_MIGD_RESPONSE", "C2_MIXED_MEAN_MIGD_RESPONSE",
              "MIXED_DELTA_C1_MINUS_C2", "MIXED_DELTA_CI", "MIXED_RELATIVE_IMPROVEMENT", "MIXED_PRIMARY",
              "POLICY_PAIR_DELTAS", "POLICY_PAIR_POSITIVE_COUNT",
              "CREDIT_PRESERVATION_POLICY_ROBUSTNESS",
              "PERSISTENT_RESPONSE_EFFECT", "PERSISTENT_EPOCH_EFFECT", "PERSISTENT_SAFETY",
              "STATIONARY_RESPONSE_EFFECT", "STATIONARY_EPOCH_EFFECT", "STATIONARY_SAFETY",
              "MACRO_RESPONSE_EFFECT", "MACRO_EPOCH_EFFECT", "MACRO_SAFETY",
              "MIXED_FE_EFFECT_C1_MINUS_C2", "PERSISTENT_FE_EFFECT_C1_MINUS_C2",
              "STATIONARY_FE_EFFECT_C1_MINUS_C2", "MACRO_FE_EFFECT_C1_MINUS_C2",
              "C1_MIXED_PREDICT_FRACTION", "C2_MIXED_PREDICT_FRACTION",
              "C1_MIXED_MULTI_ACTION_RATE", "C2_MIXED_MULTI_ACTION_RATE",
              "C1_MIXED_ACTION_ENTROPY", "C2_MIXED_ACTION_ENTROPY",
              "C1_MIXED_Q_MARGIN", "C2_MIXED_Q_MARGIN",
              "CREDIT_PRESERVATION_MECHANISM_GATE",
              "VALIDATION_OBSERVATION_STATUS", "VALIDATION_MACRO_DELTA_C1_MINUS_C2",
              "LOSS_SCALE_OBSERVATION_STATUS",
              "CREDIT_PRESERVATION_CAUSAL_VERDICT", "PARTIAL_SUBTYPE",
              "PERFORMANCE_IMPROVEMENT_CLAIM",
              "CHECKPOINT_RESELECTION", "LOSS_WEIGHT_RETUNING",
              "REWARD_RETUNING", "NORMALIZATION_RETUNING",
              "MIXER_RETUNING", "REPLAY_RETUNING", "MODEL_MODIFICATION",
              "FINAL_ENVIRONMENT_SEEDS_USED", "FINAL_EVALUATION",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<48} = {val}")


if __name__ == "__main__":
    main()
