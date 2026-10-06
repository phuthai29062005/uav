"""
E4KF — Joint Coordination A/B Final Evaluation Execution (PR-DRL-DMOEA, Branch C).

FIRST stage permitted to open the reserved final seeds 1981..2040. Executes the untouched
paired C0-vs-C1 final evaluation exactly as frozen in E4KE and produces the COORDINATION_CAUSAL
verdict. NO rescue: whatever the frozen protocol returns is preserved. No retraining, refit,
reselection, or model/reward/normalization/architecture change. E4F=FAILED, E4HF=FAILED,
E4IF=VALIDATED, E4JF=FAILED remain immutable.

Question: causal effect of replacing C0's independent shared DDQN with C1's QMIX monotonic
value decomposition, holding V3 normalization + R1 reward (alpha=0.1) + 9-128-128-4 utility
architecture fixed. Comparator = C0_INDEPENDENT_SHARED_DDQN. Effect d = C0 - C1 (positive
⇒ QMIX better).

COORDINATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.
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
    CONTROL_ARCH, FROZEN_V3_CONSTANTS_SHA256, coordination_causal_verdict,
    make_qmix_config, qmix_config_sha256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro, N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import classify_effect, policy_robustness, safety_gate, HARMFUL

RESULTS_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4KE_DIR = Path("results/E4KE_joint_coordination_ab_final_evaluation_protocol_freeze")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FINAL_SEEDS = list(range(1981, 2041))
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"]
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
BOOTSTRAP_MASTER_SEED = 86001
DELTA_COORD = 1e-5
DEVICE = torch.device("cpu")
SAMPLE_PER_ARM = 4000
_env_runs = {"count": 0}


def _e4kf_seed(*labels) -> int:
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
    frz = _load(E4KE_DIR / "E4KE_checkpoint_freeze.json")["checkpoints"]
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
        if arm == ARMS[1]:
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
            mixer_integrity[p] = {
                "has_mixer_weights": has_mixer,
                "mixer_loaded_successfully": mixer_loaded,
                "qmix_config_sha256_in_checkpoint": payload.get("qmix_config_sha256", "MISSING"),
                "matches_frozen": payload.get("qmix_config_sha256") == FROZEN_QMIX_CONFIG_SHA256,
            }
    return nets, meta, frz, mixer_integrity


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
    print("E4KF — Joint Coordination A/B Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4ke = _load(E4KE_DIR / "E4KE_verdict.json")
    e4ktr = _load(E4KTR_DIR / "E4KTR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    nets, meta, frz, mixer_integrity = load_selected_nets()
    normalizer, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    ck_pre_ok = all(_hash_file(info["checkpoint_path"]) == info["sha256"] for info in frz.values())
    pre_hashes = {(info["arm"], int(info["policy_seed"])): _hash_file(info["checkpoint_path"]) for info in frz.values()}
    rv_ok = all(meta[(a, p)]["reward_version"] == REWARD_R1_NAME for a in ARMS for p in POLICY_SEEDS)
    r1cfg_ok = all(meta[(a, p)]["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256 for a in ARMS for p in POLICY_SEEDS)
    arch_ok = (all(meta[(ARMS[0], p)]["architecture_version"] == CONTROL_ARCH for p in POLICY_SEEDS)
               and all(meta[(ARMS[1], p)]["architecture_version"] == TREATMENT_ARCH for p in POLICY_SEEDS))
    qmix_sha_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c1_mixer_ok = all(mi["mixer_loaded_successfully"] and mi["matches_frozen"] for mi in mixer_integrity.values())

    manifest = {"stage": "E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION",
                "E4KE_COMPLETE": e4ke["E4KE_COMPLETE"],
                "E4KTR_COMPLETE": e4ktr["E4KTR_COMPLETE"],
                "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
                "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
                "E4JF_GLOBAL_COMPARATOR_VERDICT": e4jf["GLOBAL_COMPARATOR_VERDICT"],
                "checkpoint_pre_hashes_match": ck_pre_ok, "reward_version_ok": rv_ok,
                "r1_config_ok": r1cfg_ok, "architecture_ok": arch_ok,
                "qmix_config_sha256_ok": qmix_sha_ok, "c1_mixer_integrity_ok": c1_mixer_ok,
                "v3_constants_sha256": v3_sha, "v3_sha_matches_ref": v3_sha == v3_sha_ref,
                "synthetic_validation": e4ke["SYNTHETIC_STATISTICS_VALIDATION"],
                "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                                   "replay": "disabled", "optimizer": "disabled", "learning": "none",
                                   "mixer_not_in_execution": True,
                                   "c0_uses_utility_only": True,
                                   "c1_uses_utility_only_for_action_selection": True}}
    _jdump(manifest, RESULTS_DIR / "E4KF_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {f"{k[0]}_{k[1]}": v for k, v in pre_hashes.items()},
            "all_match_frozen": ck_pre_ok, "c1_mixer_integrity": mixer_integrity},
           RESULTS_DIR / "E4KF_checkpoint_integrity.json")
    ok0 = (e4ke["E4KE_COMPLETE"] == "YES" and e4ktr["E4KTR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED" and e4jf["GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
           and ck_pre_ok and rv_ok and r1cfg_ok and arch_ok and qmix_sha_ok and c1_mixer_ok
           and v3_sha == v3_sha_ref and e4ke["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS")
    if not ok0:
        print("[E4KF] integrity FAIL — STOP (no final seed opened)")
        _jdump({"E4KF_COMPLETE": "NO", "COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4KF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4KF_verdict.json")
        return {"E4KF_COMPLETE": "NO"}
    print("[E4KF] integrity PASS. Opening final seeds 1981..2040.")
    _jdump({"final_seeds": [1981, 2040], "policy_seeds": POLICY_SEEDS, "arms": ARMS, "scenarios": SCENARIOS,
            "N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
            "reward_version_both_arms": REWARD_R1_NAME, "alpha_local_both_arms": 0.1,
            "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
            "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT"},
           RESULTS_DIR / "E4KF_execution_config.json")

    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    episodes = {}
    all_recs = []
    sample_count = {a: 0 for a in ARMS}
    prog = open(RESULTS_DIR / "E4KF_progress.jsonl", "w")
    n = 0
    total = len(SCENARIOS) * len(FINAL_SEEDS) * len(POLICY_SEEDS) * len(ARMS)
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            for p in POLICY_SEEDS:
                for arm in ARMS:
                    ep, recs = eval_arm(nets[(arm, p)], normalizer, arm, scenario, seed, cache, env)
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
                        print(f"    [E4KF] {n}/{total} episodes")
    prog.close()

    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) and np.isfinite(episodes[k]["MIGD_epoch"])
                 and np.isfinite(episodes[k]["response_FE"]) for k in keys)
    n_c0 = sum(1 for k in keys if k[0] == ARMS[0])
    n_c1 = sum(1 for k in keys if k[0] == ARMS[1])
    complete = (len(keys) == 1800 and set(keys) == expected and n_c0 == 900 and n_c1 == 900 and finite)
    print(f"[E4KF] matrix {len(keys)}/1800 complete={complete} (C0 {n_c0}, C1 {n_c1})")

    ep_fields = ["arm", "policy_seed", "scenario", "environment_seed", "MIGD_response", "MIGD_epoch",
                 "response_FE", "multi_action_rate"]
    ep_rows = [{"arm": a, "policy_seed": p, "scenario": s, "environment_seed": e,
                **{k: episodes[(a, p, s, e)][k] for k in ("MIGD_response", "MIGD_epoch", "response_FE",
                   "multi_action_rate")}} for (a, p, s, e) in keys]
    _write_csv(RESULTS_DIR / "E4KF_episode_results.csv", ep_fields, ep_rows)
    rec_fields = ["arm", "policy_seed", "scenario", "environment_seed", "change_index", "region_id",
                  "requested_action", "realized_operator", "fallback", "L_pre", "L_post",
                  "weighted_global_reward", "g_r", "weighted_local_reward_R1",
                  "raw_fe_cost", "weighted_fe_penalty"]
    _write_csv(RESULTS_DIR / "E4KF_regional_records.csv", rec_fields, all_recs)

    if not complete:
        _jdump({"E4KF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "EPISODES_COMPLETE": f"{len(keys)}/1800", "COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4KF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4KF_verdict.json")
        print("[E4KF] matrix incomplete — NOT_COMPUTED"); return {"E4KF_COMPLETE": "NO"}

    analysis = analyze(episodes)
    post_ok = all(_hash_file(meta[k]["checkpoint_path"]) == pre_hashes[k] for k in meta)
    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    analysis["checkpoint_integrity"] = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    analysis["version_integrity"] = "PASS" if (post_ok and v3_post and rv_ok and r1cfg_ok and arch_ok) else "FAIL"
    analysis["architecture_integrity"] = "PASS" if (arch_ok and qmix_sha_ok) else "FAIL"
    _jdump({"checkpoint_post_hashes_match": post_ok, "v3_sha_unchanged": v3_post,
            "r1_config_unchanged": r1cfg_ok, "architecture_unchanged": arch_ok,
            "qmix_sha_unchanged": qmix_sha_ok},
           RESULTS_DIR / "E4KF_post_integrity.json")
    _jdump({"optimizer_steps": 0, "gradient_updates": 0, "replay_appends": 0, "target_updates": 0,
            "checkpoint_writes": 0, "model_weight_changes": 0, "normalizer_updates": 0,
            "reward_config_updates": 0, "mixer_calls_during_action_selection": 0},
           RESULTS_DIR / "E4KF_no_learning_audit.json")

    write_outputs(episodes, ep_rows, analysis, mixer_integrity)
    verdict = write_verdict_report(analysis)
    _jdump({"final_seed_range": [1981, 2040], "unique_final_seeds": 60,
            "FINAL_ENVIRONMENT_SEEDS_USED": "1981..2040", "FINAL_EVALUATION": "COMPLETE",
            "status": "CONSUMED_FINAL_COORDINATION_AB", "env_runs": _env_runs["count"]},
           RESULTS_DIR / "E4KF_final_seed_usage.json")
    _emit_terminal(verdict)
    print(f"[E4KF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


# ---------------------------------------------------------------------------
def _matrix(episodes, arm, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = episodes[(arm, p, scenario, e)][endpoint]
    return M


def _paired_diff(episodes, scenario, endpoint):
    C0 = _matrix(episodes, ARMS[0], scenario, endpoint)
    C1 = _matrix(episodes, ARMS[1], scenario, endpoint)
    return C0 - C1, C0, C1


def analyze(episodes) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint):
        D, C0, C1 = _paired_diff(episodes, scenario, endpoint)
        bs = paired_crossed_two_axis_bootstrap(D, seed=_e4kf_seed(endpoint, scenario, "C0_MINUS_C1"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_COORD)
        return {"endpoint": endpoint, "C0_mean": float(C0.mean()), "C1_mean": float(C1.mean()),
                "delta_C0_minus_C1": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(C0.mean()) + 1e-12), "classification": cls}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {ep: effect(scenario, ep)
                                             for ep in ("MIGD_response", "MIGD_epoch", "response_FE")}
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    g = out["gates"]
    g["MIXED_PRIMARY"] = mixed["classification"]
    g["C0_MIXED_MEAN"] = mixed["C0_mean"]
    g["C1_MIXED_MEAN"] = mixed["C1_mean"]
    g["MIXED_DELTA"] = mixed["delta_C0_minus_C1"]
    g["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]
    g["MIXED_RI"] = mixed["relative_improvement"]

    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    per_policy = [float(Dm[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    g["per_policy_MIXED_delta"] = per_policy
    g["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    g["COORDINATION_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        g[f"{scenario}_RESPONSE_EFFECT"] = rlab
        g[f"{scenario}_EPOCH_EFFECT"] = elab
        g[f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    def macro(endpoint):
        Dby = {s: _paired_diff(episodes, s, endpoint)[0] for s in SCENARIOS}
        bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=_e4kf_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_COORD)}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    g["MACRO_RESPONSE_EFFECT"] = mr["classification"]
    g["MACRO_EPOCH_EFFECT"] = me["classification"]
    g["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"
    g["MIXED_FE_EFFECT"] = out["scenario_effects"]["MIXED"]["response_FE"]["delta_C0_minus_C1"]
    g["PERSISTENT_FE_EFFECT"] = out["scenario_effects"]["PERSISTENT"]["response_FE"]["delta_C0_minus_C1"]
    g["STATIONARY_FE_EFFECT"] = out["scenario_effects"]["STATIONARY"]["response_FE"]["delta_C0_minus_C1"]

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
    return out


def write_outputs(episodes, ep_rows, analysis, mixer_integrity):
    g = analysis["gates"]
    pp = []
    Dm, C0m, C1m = _paired_diff(episodes, "MIXED", "MIGD_response")
    for i, p in enumerate(POLICY_SEEDS):
        pp.append({"policy_seed": p, "C0_MIXED_MIGD_response": float(C0m[i, :].mean()),
                   "C1_MIXED_MIGD_response": float(C1m[i, :].mean()),
                   "delta_C0_minus_C1": float(Dm[i, :].mean()),
                   "positive_qmix_better": bool(Dm[i, :].mean() > 0)})
    _write_csv(RESULTS_DIR / "E4KF_policy_pair_summary.csv", list(pp[0].keys()), pp)

    ad = []
    for arm in ARMS:
        a = analysis["action"][arm]
        ad.append({"arm": arm, **{k: a["fractions"][k] for k in ACTION_ORDER},
                   "multi_action_rate": a["multi_action_rate"], "action_entropy": a["action_entropy"]})
    _write_csv(RESULTS_DIR / "E4KF_action_distribution.csv",
               ["arm"] + list(ACTION_ORDER) + ["multi_action_rate", "action_entropy"], ad)

    se = []
    for s in SCENARIOS:
        for ep in ("MIGD_response", "MIGD_epoch"):
            b = analysis["scenario_effects"][s][ep]
            se.append({"scenario": s, "endpoint": ep, "C0_mean": b["C0_mean"], "C1_mean": b["C1_mean"],
                       "delta_C0_minus_C1": b["delta_C0_minus_C1"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                       "relative_improvement": b["relative_improvement"], "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4KF_scenario_effects.csv", list(se[0].keys()), se)
    _write_csv(RESULTS_DIR / "E4KF_macro_effects.csv", ["endpoint", "delta", "ci_lo", "ci_hi", "classification"],
               [{"endpoint": k, **{kk: analysis["macro"][k][kk] for kk in ("delta", "ci_lo", "ci_hi", "classification")}}
                for k in ("MIGD_response", "MIGD_epoch", "response_FE")])
    _write_csv(RESULTS_DIR / "E4KF_fe_effects.csv", ["scenario", "fe_delta_C0_minus_C1", "classification"],
               [{"scenario": s, "fe_delta_C0_minus_C1": analysis["scenario_effects"][s]["response_FE"]["delta_C0_minus_C1"],
                 "classification": analysis["scenario_effects"][s]["response_FE"]["classification"]} for s in SCENARIOS]
               + [{"scenario": "macro", "fe_delta_C0_minus_C1": analysis["macro"]["response_FE"]["delta"],
                   "classification": analysis["macro"]["response_FE"]["classification"]}])
    _jdump({"scenario_effects": analysis["scenario_effects"], "macro": analysis["macro"],
            "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
                         "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "effect": "C0_MINUS_C1",
                         "DELTA_COORD": DELTA_COORD}},
           RESULTS_DIR / "E4KF_bootstrap_results.json")
    _jdump(g, RESULTS_DIR / "E4KF_scientific_gates.json")

    act = analysis["action"]
    _jdump({"COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
            "C0_MIXED_action": act[ARMS[0]], "C1_MIXED_action": act[ARMS[1]],
            "C1_mixer_integrity": mixer_integrity,
            "mixer_calls_during_action_selection": 0,
            "note": "coordination diagnostics recorded for post-hoc interpretation ONLY"},
           RESULTS_DIR / "E4KF_coordination_diagnostics.json")

    e4ktr_v = _load(E4KTR_DIR / "E4KTR_verdict.json")
    _jdump({"note": "descriptive only; does not modify verdict",
            "validation": {"C0_macro": e4ktr_v.get("C0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "C1_macro": e4ktr_v.get("C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "delta_C0_minus_C1": e4ktr_v.get("VALIDATION_PAIRED_DELTA_C0_MINUS_C1"),
                           "C0_MIXED_PREDICT_FRACTION": e4ktr_v.get("C0_MIXED_PREDICT_FRACTION"),
                           "C1_MIXED_PREDICT_FRACTION": e4ktr_v.get("C1_MIXED_PREDICT_FRACTION")},
            "final": {"C0_MIXED_MIGD_response": g["C0_MIXED_MEAN"], "C1_MIXED_MIGD_response": g["C1_MIXED_MEAN"],
                      "MIXED_delta_C0_minus_C1": g["MIXED_DELTA"],
                      "C0_MIXED_PREDICT": act[ARMS[0]]["predict_fraction"],
                      "C1_MIXED_PREDICT": act[ARMS[1]]["predict_fraction"]}},
           RESULTS_DIR / "E4KF_validation_final_comparison.json")


def write_verdict_report(analysis):
    g = analysis["gates"]
    v = coordination_causal_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        policy_robustness_pass=(g["COORDINATION_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(analysis["checkpoint_integrity"] == "PASS"),
        protocol_integrity_pass=(analysis["version_integrity"] == "PASS"),
        architecture_integrity_pass=(analysis["architecture_integrity"] == "PASS"),
        no_final_tuning=True)
    vname, subtype = v["verdict"], v["subtype"]
    if vname == "COORDINATION_CAUSAL_VALIDATED":
        nxt = "E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
    elif subtype == "PERFORMANCE_PARTIAL":
        nxt = "E4L_COORDINATION_RESULT_REVIEW"
    elif subtype == "PRACTICALLY_NEUTRAL":
        nxt = "E4L_COORDINATION_RESULT_REVIEW"
    else:
        nxt = "E4L_COORDINATION_FAILURE_ANALYSIS"
    short = {"COORDINATION_CAUSAL_VALIDATED": "VALIDATED", "COORDINATION_CAUSAL_PARTIAL": "PARTIAL",
             "COORDINATION_CAUSAL_FAILED": "FAILED"}[vname]
    act = analysis["action"]
    verdict = {
        "stage": "E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION", "E4KF_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES", "EPISODES_COMPLETE": "1800/1800",
        "CHECKPOINT_INTEGRITY": analysis["checkpoint_integrity"],
        "FINAL_PROTOCOL_INTEGRITY": "PASS",
        "ARCHITECTURE_INTEGRITY": analysis["architecture_integrity"],
        "REWARD_VERSION_CONFIG_INTEGRITY": analysis["version_integrity"],
        "NORMALIZATION_INTEGRITY": analysis["version_integrity"],
        "QMIX_CONFIG_SHA256_INTEGRITY": "PASS",
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "C1_QMIX_VS_C0_DDQN", "EFFECT_DEFINITION": "C0_MINUS_C1",
        "POSITIVE_MEANS": "QMIX_BETTER", "PRACTICAL_MARGIN": DELTA_COORD,
        "C0_MIXED_MEAN_MIGD_RESPONSE": g["C0_MIXED_MEAN"], "C1_MIXED_MEAN_MIGD_RESPONSE": g["C1_MIXED_MEAN"],
        "MIXED_DELTA_C0_MINUS_C1": g["MIXED_DELTA"], "MIXED_DELTA_CI": g["MIXED_CI"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RI"], "MIXED_PRIMARY": g["MIXED_PRIMARY"],
        "per_policy_MIXED_deltas": g["per_policy_MIXED_delta"],
        "POLICY_PAIR_POSITIVE_COUNT": g["POLICY_POSITIVE_COUNT"],
        "COORDINATION_POLICY_ROBUSTNESS": g["COORDINATION_POLICY_ROBUSTNESS"],
        "C0_MIXED_PREDICT_FRACTION": act[ARMS[0]]["predict_fraction"],
        "C1_MIXED_PREDICT_FRACTION": act[ARMS[1]]["predict_fraction"],
        "C0_MIXED_MULTI_ACTION_RATE": act[ARMS[0]]["multi_action_rate"],
        "C1_MIXED_MULTI_ACTION_RATE": act[ARMS[1]]["multi_action_rate"],
        "C0_MIXED_ACTION_ENTROPY": act[ARMS[0]]["action_entropy"],
        "C1_MIXED_ACTION_ENTROPY": act[ARMS[1]]["action_entropy"],
        "PERSISTENT_RESPONSE_EFFECT": g["PERSISTENT_RESPONSE_EFFECT"],
        "PERSISTENT_EPOCH_EFFECT": g["PERSISTENT_EPOCH_EFFECT"],
        "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_RESPONSE_EFFECT": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_EFFECT": g["STATIONARY_EPOCH_EFFECT"],
        "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_RESPONSE_EFFECT": g["MACRO_RESPONSE_EFFECT"],
        "MACRO_EPOCH_EFFECT": g["MACRO_EPOCH_EFFECT"],
        "MACRO_SAFETY": g["MACRO_SAFETY"],
        "MIXED_FE_EFFECT_C0_MINUS_C1": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_EFFECT_C0_MINUS_C1": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_EFFECT_C0_MINUS_C1": g["STATIONARY_FE_EFFECT"],
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "COORDINATION_CAUSAL_VERDICT": short, "PARTIAL_SUBTYPE": subtype if subtype else "NONE",
        "COORDINATION_IMPROVEMENT_CLAIM": v["COORDINATION_IMPROVEMENT_CLAIM"],
        "verdict_reason": v["reason"],
        "ALPHA_RETUNING": "NO", "REWARD_RETUNING": "NO", "ARCHITECTURE_RETUNING": "NO",
        "CHECKPOINT_RESELECTION": "NO", "MODEL_MODIFICATION": "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED", "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "FINAL_EVALUATION": "COMPLETE", "FINAL_ENVIRONMENT_SEEDS_USED": "1981..2040",
        "FULL_REGRESSION": "(run separately)", "NEXT_STAGE": nxt}
    _jdump(verdict, RESULTS_DIR / "E4KF_verdict.json")
    _write_report(verdict, analysis)
    return verdict


def _write_report(v, analysis):
    act = analysis["action"]
    L = ["# E4KF — Joint Coordination A/B Final Evaluation Execution", "",
         f"**COORDINATION_CAUSAL_VERDICT = {v['COORDINATION_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}** → NEXT_STAGE = {v['NEXT_STAGE']}", "",
         "Causal question: replacing C0's independent shared DDQN with C1's QMIX monotonic value decomposition, "
         "with V3+R1 (alpha=0.1) + 9-128-128-4 utility fixed. Comparator C0_INDEPENDENT_SHARED_DDQN. Not rescued. "
         "COORDINATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.", "",
         "## A/B/C. Integrity & completeness",
         f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}; ARCHITECTURE_INTEGRITY = "
         f"{v['ARCHITECTURE_INTEGRITY']}; REWARD_VERSION_CONFIG_INTEGRITY = "
         f"{v['REWARD_VERSION_CONFIG_INTEGRITY']}; NORMALIZATION_INTEGRITY = {v['NORMALIZATION_INTEGRITY']}; "
         f"matrix {v['EPISODES_COMPLETE']}.", "",
         "## D/E/F. Primary MIXED paired effect (d = C0 − C1; + ⇒ QMIX better)",
         f"- C0 mean = {v['C0_MIXED_MEAN_MIGD_RESPONSE']:.6f}; C1 mean = {v['C1_MIXED_MEAN_MIGD_RESPONSE']:.6f}",
         f"- Delta = {v['MIXED_DELTA_C0_MINUS_C1']:.6e}  CI {v['MIXED_DELTA_CI']}; RI = "
         f"{v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% → **MIXED_PRIMARY = {v['MIXED_PRIMARY']}**", "",
         "## H. Policy-pair robustness",
         f"- positive pairs = {v['POLICY_PAIR_POSITIVE_COUNT']}/5 → {v['COORDINATION_POLICY_ROBUSTNESS']}; per-policy Δ = "
         f"{[round(x,6) for x in v['per_policy_MIXED_deltas']]}", "",
         "## I. Action behavior (descriptive)",
         f"- MIXED PREDICT C0 {v['C0_MIXED_PREDICT_FRACTION']:.3f} vs C1 {v['C1_MIXED_PREDICT_FRACTION']:.3f}; "
         f"multi {v['C0_MIXED_MULTI_ACTION_RATE']:.3f}→{v['C1_MIXED_MULTI_ACTION_RATE']:.3f}; "
         f"entropy {v['C0_MIXED_ACTION_ENTROPY']:.3f}→{v['C1_MIXED_ACTION_ENTROPY']:.3f}.", "",
         "## J. Coordination mechanism gate",
         "- NOT_USED_FOR_SCIENTIFIC_VERDICT. Diagnostics (monotonicity, joint TD, action entropy) "
         "recorded for post-hoc interpretation ONLY. Action diversity cannot rescue bad quality. "
         "FE savings cannot rescue bad quality.", "",
         "## K/L/M. Safety",
         f"- PERSISTENT: response={v['PERSISTENT_RESPONSE_EFFECT']}, epoch={v['PERSISTENT_EPOCH_EFFECT']} → {v['PERSISTENT_SAFETY']}",
         f"- STATIONARY: response={v['STATIONARY_RESPONSE_EFFECT']}, epoch={v['STATIONARY_EPOCH_EFFECT']} → {v['STATIONARY_SAFETY']}",
         f"- MACRO: response={v['MACRO_RESPONSE_EFFECT']}, epoch={v['MACRO_EPOCH_EFFECT']} → {v['MACRO_SAFETY']}", "",
         "## N. FE (C0 − C1; + ⇒ C1/QMIX cheaper)",
         f"- MIXED={v['MIXED_FE_EFFECT_C0_MINUS_C1']:.2f}, PERSISTENT={v['PERSISTENT_FE_EFFECT_C0_MINUS_C1']:.2f}, "
         f"STATIONARY={v['STATIONARY_FE_EFFECT_C0_MINUS_C1']:.2f}", "",
         "## O. Coordination causal verdict",
         f"- **{v['COORDINATION_CAUSAL_VERDICT']}{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE']!='NONE' else ''}**; "
         f"COORDINATION_IMPROVEMENT_CLAIM = {v['COORDINATION_IMPROVEMENT_CLAIM']}; reason: {v['verdict_reason']}", "",
         "## P. Historical preservation",
         "- ORIGINAL_E4F_VERDICT = FAILED; E4HF normalization = FAILED; E4IF reward = VALIDATED; "
         "E4JF global-comparator = FAILED (unchanged).", "",
         "## Q. Limitations",
         "- Coordination-learner causal experiment; factor = joint-coordination-learner-specification. "
         "Both arms V3+R1 (alpha=0.1). QMIX diagnostics NOT used for verdict. "
         "Does NOT test superiority over GLOBAL_PREDICT or universal coordination optimality. "
         "5 policy seeds; controlled benchmark, not real-world UAV.", "",
         f"## R. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4KF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4KF_COMPLETE", "FINAL_MATRIX_COMPLETE", "EPISODES_COMPLETE", "ORIGINAL_E4F_VERDICT",
              "E4HF_NORMALIZATION_CAUSAL_VERDICT", "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
              "CHECKPOINT_INTEGRITY", "FINAL_PROTOCOL_INTEGRITY", "ARCHITECTURE_INTEGRITY",
              "REWARD_VERSION_CONFIG_INTEGRITY", "NORMALIZATION_INTEGRITY", "QMIX_CONFIG_SHA256_INTEGRITY",
              "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT", "PRIMARY_COMPARISON", "EFFECT_DEFINITION",
              "POSITIVE_MEANS", "PRACTICAL_MARGIN",
              "C0_MIXED_MEAN_MIGD_RESPONSE", "C1_MIXED_MEAN_MIGD_RESPONSE",
              "MIXED_DELTA_C0_MINUS_C1", "MIXED_DELTA_CI", "MIXED_RELATIVE_IMPROVEMENT", "MIXED_PRIMARY",
              "POLICY_PAIR_POSITIVE_COUNT", "COORDINATION_POLICY_ROBUSTNESS",
              "C0_MIXED_PREDICT_FRACTION", "C1_MIXED_PREDICT_FRACTION",
              "C0_MIXED_MULTI_ACTION_RATE", "C1_MIXED_MULTI_ACTION_RATE",
              "C0_MIXED_ACTION_ENTROPY", "C1_MIXED_ACTION_ENTROPY",
              "PERSISTENT_RESPONSE_EFFECT", "PERSISTENT_EPOCH_EFFECT", "PERSISTENT_SAFETY",
              "STATIONARY_RESPONSE_EFFECT", "STATIONARY_EPOCH_EFFECT", "STATIONARY_SAFETY",
              "MACRO_RESPONSE_EFFECT", "MACRO_EPOCH_EFFECT", "MACRO_SAFETY",
              "MIXED_FE_EFFECT_C0_MINUS_C1", "PERSISTENT_FE_EFFECT_C0_MINUS_C1", "STATIONARY_FE_EFFECT_C0_MINUS_C1",
              "COORDINATION_MECHANISM_GATE", "COORDINATION_CAUSAL_VERDICT", "PARTIAL_SUBTYPE",
              "COORDINATION_IMPROVEMENT_CLAIM",
              "ALPHA_RETUNING", "REWARD_RETUNING", "ARCHITECTURE_RETUNING",
              "CHECKPOINT_RESELECTION", "MODEL_MODIFICATION",
              "FINAL_EVALUATION", "FINAL_ENVIRONMENT_SEEDS_USED", "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<48} = {val}")


if __name__ == "__main__":
    main()
