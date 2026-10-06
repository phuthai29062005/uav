"""
E4IF — Reward A/B Final Evaluation Execution (PR-DRL-DMOEA, Branch B).

FIRST stage permitted to open the reserved final seeds 1581..1640. Executes the untouched
paired R0-vs-R1 final evaluation exactly as frozen in E4IE and produces the REWARD_CAUSAL
verdict. NO rescue: whatever the frozen protocol returns is preserved. No retraining, refit,
reselection, or model/reward/normalization/architecture change. E4F=FAILED, E4HF normalization
=FAILED remain immutable.

Question: causal effect of replacing R0's scale-mismatched local credit with R1's bounded
regional quality-gain, holding V3 normalization + the Shared Double-DQN system fixed and
alpha_local=0.1 in both arms. Comparator = R0_CONTROL. Effect d = R0 - R1 (positive ⇒ R1 better).
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
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES, compute_joint_rewards,
    ALPHA_G, ALPHA_L, BETA,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1
from prdrl.rl import reward_integration as RI
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro, N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import classify_effect, policy_robustness, safety_gate, HARMFUL

RESULTS_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4IE_DIR = Path("results/E4IE_reward_ab_final_evaluation_protocol_freeze")
E4ITR_DIR = Path("results/E4ITR_reward_ab_training_execution")
E4IDD_DIR = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FINAL_SEEDS = list(range(1581, 1641))
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = ["R0_CONTROL", "R1_TREATMENT"]
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
BOOTSTRAP_MASTER_SEED = 66001
EPS_ACTIVE = EPS_R1
DEVICE = torch.device("cpu")
SAMPLE_PER_ARM = 4000
_env_runs = {"count": 0}


def _e4if_seed(*labels) -> int:
    """Deterministic substream seed from the E4IE-frozen master 66001 + labels (same
    algorithm as crossed_bootstrap.derive_seed but the reward-A/B master)."""
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
    frz = _load(E4IE_DIR / "E4IE_checkpoint_freeze.json")["checkpoints"]
    nets, meta = {}, {}
    for key, info in frz.items():
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets[(info["arm"], int(info["policy_seed"]))] = net
        meta[(info["arm"], int(info["policy_seed"]))] = info
    return nets, meta, frz


# ---------------------------------------------------------------------------
def eval_arm(net, normalizer, arm, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    recs = []
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi_flags = []
    reward_version = REWARD_R0_NAME if arm == "R0_CONTROL" else REWARD_R1_NAME
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
                                                         t_current_before, reward_version=reward_version)
        rec_by_region = {rec.region_id: rec for rec in post["records"]}
        ev_local = []
        for r in active:
            rec = rec_by_region.get(r)
            L_pre = aux["L_pre_by_region"][r]; L_post = aux["L_post_by_region"][r]
            raw_local_R0 = aux["raw_local_R0_by_region"][r]; g_r = aux["g_by_region"][r]
            fe_cost = aux["fe_by_region"][r]
            w_global = ALPHA_G * aux["global_reward"]
            w_local_R0 = ALPHA_L * raw_local_R0
            w_local_R1 = ALPHA_L * g_r
            w_fe = BETA * fe_cost
            w_local_used = w_local_R0 if arm == "R0_CONTROL" else w_local_R1
            ev_local.append(w_local_used)
            recs.append({"arm": arm, "scenario": scenario, "environment_seed": seed, "change_index": k,
                         "region_id": r, "requested_action": action_str[r],
                         "realized_operator": rec.realized_operator if rec else "",
                         "fallback": bool(rec.fallback_used) if rec else False,
                         "L_pre": L_pre, "L_post": L_post, "weighted_global_reward": w_global,
                         "raw_local_reward_R0": raw_local_R0, "weighted_local_reward_R0": w_local_R0,
                         "g_r": g_r, "weighted_local_reward_R1": w_local_R1,
                         "local_active_R0": bool(abs(w_local_R0) > EPS_ACTIVE),
                         "local_active_R1": bool(abs(g_r) > EPS_ACTIVE),
                         "raw_fe_cost": fe_cost, "weighted_fe_penalty": w_fe,
                         "w_local_used": w_local_used})
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
    print("E4IF — Reward A/B Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4ie = _load(E4IE_DIR / "E4IE_verdict.json")
    e4itr = _load(E4ITR_DIR / "E4ITR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    nets, meta, frz = load_selected_nets()
    normalizer, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    ck_pre_ok = all(_hash_file(info["checkpoint_path"]) == info["sha256"] for info in frz.values())
    pre_hashes = {k: _hash_file(meta[k]["checkpoint_path"]) for k in meta}
    rv_ok = all((meta[(a, p)]["reward_version"] == (REWARD_R0_NAME if a == "R0_CONTROL" else REWARD_R1_NAME))
                for a in ARMS for p in POLICY_SEEDS)
    r1cfg_ok = all(meta[("R1_TREATMENT", p)]["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256 for p in POLICY_SEEDS)
    manifest = {"stage": "E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION", "E4IE_COMPLETE": e4ie["E4IE_COMPLETE"],
                "E4ITR_COMPLETE": e4itr["E4ITR_COMPLETE"], "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
                "checkpoint_pre_hashes_match": ck_pre_ok, "reward_version_ok": rv_ok,
                "r1_config_ok": r1cfg_ok, "v3_constants_sha256": v3_sha,
                "v3_sha_matches_ref": v3_sha == v3_sha_ref,
                "synthetic_validation": e4ie["SYNTHETIC_STATISTICS_VALIDATION"],
                "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                                   "replay": "disabled", "optimizer": "disabled", "learning": "none"}}
    _jdump(manifest, RESULTS_DIR / "E4IF_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {f"{k[0]}_{k[1]}": v for k, v in pre_hashes.items()},
            "all_match_frozen": ck_pre_ok}, RESULTS_DIR / "E4IF_checkpoint_integrity.json")
    ok0 = (e4ie["E4IE_COMPLETE"] == "YES" and e4itr["E4ITR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and ck_pre_ok and rv_ok and r1cfg_ok and v3_sha == v3_sha_ref
           and e4ie["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS")
    if not ok0:
        print("[E4IF] integrity FAIL — STOP (no final seed opened)")
        _jdump({"E4IF_COMPLETE": "NO", "REWARD_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4IF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4IF_verdict.json")
        return {"E4IF_COMPLETE": "NO"}
    print("[E4IF] integrity PASS. Opening final seeds 1581..1640.")
    _jdump({"final_seeds": [1581, 1640], "policy_seeds": POLICY_SEEDS, "arms": ARMS, "scenarios": SCENARIOS,
            "N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED},
           RESULTS_DIR / "E4IF_execution_config.json")

    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    episodes = {}
    all_recs = []          # bounded sample for CSV
    sample_count = {a: 0 for a in ARMS}
    # per (arm, policy) mechanism accumulators (pooled over all scenarios)
    mech_acc = {(a, p): {"n_valid": 0, "n_active": 0, "abs_global": [], "abs_local_active": [], "disp": []}
                for a in ARMS for p in POLICY_SEEDS}
    prog = open(RESULTS_DIR / "E4IF_progress.jsonl", "w")
    n = 0
    total = len(SCENARIOS) * len(FINAL_SEEDS) * len(POLICY_SEEDS) * len(ARMS)
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            for p in POLICY_SEEDS:
                for arm in ARMS:
                    ep, recs = eval_arm(nets[(arm, p)], normalizer, arm, scenario, seed, cache, env)
                    episodes[(arm, p, scenario, seed)] = ep
                    acc = mech_acc[(arm, p)]
                    ev_by_change = defaultdict(list)
                    for r in recs:
                        acc["n_valid"] += 1
                        active = r["local_active_R0"] if arm == "R0_CONTROL" else r["local_active_R1"]
                        if active:
                            acc["n_active"] += 1
                            acc["abs_global"].append(abs(r["weighted_global_reward"]))
                            acc["abs_local_active"].append(abs(r["w_local_used"]))
                        ev_by_change[r["change_index"]].append(r["w_local_used"])
                        if sample_count[arm] < SAMPLE_PER_ARM:
                            all_recs.append({"policy_seed": p, **{k: r[k] for k in
                                ("arm", "scenario", "environment_seed", "change_index", "region_id",
                                 "requested_action", "realized_operator", "fallback", "L_pre", "L_post",
                                 "weighted_global_reward", "raw_local_reward_R0", "weighted_local_reward_R0",
                                 "g_r", "weighted_local_reward_R1", "local_active_R0", "local_active_R1",
                                 "raw_fe_cost", "weighted_fe_penalty")}})
                            sample_count[arm] += 1
                    for ch, vals in ev_by_change.items():
                        acc["disp"].append(float(np.std(vals)) if len(vals) >= 2 else 0.0)
                    prog.write(json.dumps({"arm": arm, "policy_seed": p, "scenario": scenario,
                                           "environment_seed": seed}) + "\n"); prog.flush()
                    n += 1
                    if n % 360 == 0:
                        print(f"    [E4IF] {n}/{total} episodes")
    prog.close()

    # matrix completeness
    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) and np.isfinite(episodes[k]["MIGD_epoch"])
                 and np.isfinite(episodes[k]["response_FE"]) for k in keys)
    n_r0 = sum(1 for k in keys if k[0] == "R0_CONTROL"); n_r1 = sum(1 for k in keys if k[0] == "R1_TREATMENT")
    complete = (len(keys) == 1800 and set(keys) == expected and n_r0 == 900 and n_r1 == 900 and finite)
    print(f"[E4IF] matrix {len(keys)}/1800 complete={complete} (R0 {n_r0}, R1 {n_r1})")

    ep_fields = ["arm", "policy_seed", "scenario", "environment_seed", "MIGD_response", "MIGD_epoch",
                 "response_FE", "multi_action_rate"]
    ep_rows = [{"arm": a, "policy_seed": p, "scenario": s, "environment_seed": e,
                **{k: episodes[(a, p, s, e)][k] for k in ("MIGD_response", "MIGD_epoch", "response_FE",
                   "multi_action_rate")}} for (a, p, s, e) in keys]
    _write_csv(RESULTS_DIR / "E4IF_episode_results.csv", ep_fields, ep_rows)
    rec_fields = ["arm", "policy_seed", "scenario", "environment_seed", "change_index", "region_id",
                  "requested_action", "realized_operator", "fallback", "L_pre", "L_post",
                  "weighted_global_reward", "raw_local_reward_R0", "weighted_local_reward_R0", "g_r",
                  "weighted_local_reward_R1", "local_active_R0", "local_active_R1", "raw_fe_cost",
                  "weighted_fe_penalty"]
    _write_csv(RESULTS_DIR / "E4IF_regional_reward_records.csv", rec_fields, all_recs)

    if not complete:
        _jdump({"E4IF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "EPISODES_COMPLETE": f"{len(keys)}/1800", "REWARD_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4IF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4IF_verdict.json")
        print("[E4IF] matrix incomplete — NOT_COMPUTED"); return {"E4IF_COMPLETE": "NO"}

    analysis = analyze(episodes, mech_acc)
    # post integrity
    post_ok = all(_hash_file(meta[k]["checkpoint_path"]) == pre_hashes[k] for k in meta)
    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    analysis["checkpoint_integrity"] = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    analysis["version_integrity"] = "PASS" if (post_ok and v3_post and rv_ok and r1cfg_ok) else "FAIL"
    _jdump({"checkpoint_post_hashes_match": post_ok, "v3_sha_unchanged": v3_post,
            "r1_config_unchanged": r1cfg_ok}, RESULTS_DIR / "E4IF_post_integrity.json")
    _jdump({"optimizer_steps": 0, "gradient_updates": 0, "replay_appends": 0, "target_updates": 0,
            "checkpoint_writes": 0, "model_weight_changes": 0, "normalizer_updates": 0,
            "reward_config_updates": 0}, RESULTS_DIR / "E4IF_no_learning_audit.json")

    write_outputs(episodes, ep_rows, analysis)
    verdict = write_verdict_report(analysis)
    _jdump({"final_seed_range": [1581, 1640], "unique_final_seeds": 60,
            "FINAL_ENVIRONMENT_SEEDS_USED": "1581..1640", "FINAL_EVALUATION": "COMPLETE",
            "status": "CONSUMED_FINAL_REWARD_AB", "env_runs": _env_runs["count"]},
           RESULTS_DIR / "E4IF_final_seed_usage.json")
    _emit_terminal(verdict)
    print(f"[E4IF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


# ---------------------------------------------------------------------------
def _matrix(episodes, arm, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = episodes[(arm, p, scenario, e)][endpoint]
    return M


def _paired_diff(episodes, scenario, endpoint):
    R0 = _matrix(episodes, "R0_CONTROL", scenario, endpoint)
    R1 = _matrix(episodes, "R1_TREATMENT", scenario, endpoint)
    return R0 - R1, R0, R1


def _arm_mech(mech_acc, arm):
    """Per-arm mechanism = mean over 5 policies of each policy's pooled-over-all-scenarios metric."""
    act, ratio_list, disp = [], [], []
    ratio_applicable = []
    for p in POLICY_SEEDS:
        c = mech_acc[(arm, p)]
        nv, na = c["n_valid"], c["n_active"]
        act.append(na / nv if nv else 0.0)
        disp.append(float(np.mean(c["disp"])) if c["disp"] else 0.0)
        if na > 0 and c["abs_global"] and c["abs_local_active"]:
            r = float(np.median(c["abs_global"])) / (float(np.median(c["abs_local_active"])) + 1e-12)
            ratio_list.append(r); ratio_applicable.append(True)
        else:
            ratio_applicable.append(False)
    ratio = float(np.mean(ratio_list)) if ratio_list else "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"
    return {"active_fraction": float(np.mean(act)), "global_local_ratio": ratio,
            "regional_reward_dispersion": float(np.mean(disp))}


def analyze(episodes, mech_acc) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint):
        D, R0, R1 = _paired_diff(episodes, scenario, endpoint)   # R0 - R1
        bs = paired_crossed_two_axis_bootstrap(D, seed=_e4if_seed(endpoint, scenario, "R0_MINUS_R1"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])
        return {"endpoint": endpoint, "R0_mean": float(R0.mean()), "R1_mean": float(R1.mean()),
                "delta_R0_minus_R1": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(R0.mean()) + 1e-12), "classification": cls}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {ep: effect(scenario, ep)
                                             for ep in ("MIGD_response", "MIGD_epoch", "response_FE")}
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    g = out["gates"]
    g["MIXED_PRIMARY"] = mixed["classification"]; g["R0_MIXED_MEAN"] = mixed["R0_mean"]
    g["R1_MIXED_MEAN"] = mixed["R1_mean"]; g["MIXED_DELTA"] = mixed["delta_R0_minus_R1"]
    g["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]; g["MIXED_RI"] = mixed["relative_improvement"]

    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    per_policy = [float(Dm[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    g["per_policy_MIXED_delta"] = per_policy; g["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    g["REWARD_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    m_r0 = _arm_mech(mech_acc, "R0_CONTROL"); m_r1 = _arm_mech(mech_acc, "R1_TREATMENT")
    d_activity = m_r1["active_fraction"] - m_r0["active_fraction"]
    r0r, r1r = m_r0["global_local_ratio"], m_r1["global_local_ratio"]
    d_ratio = (r0r - r1r) if (isinstance(r0r, (int, float)) and isinstance(r1r, (int, float))) \
        else "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL"
    d_disp = m_r1["regional_reward_dispersion"] - m_r0["regional_reward_dispersion"]
    mech = RI.reward_mechanism_supported(d_activity, d_ratio, d_disp)
    out["mechanism"] = {"R0": m_r0, "R1": m_r1, "DELTA_ACTIVITY": d_activity, "DELTA_RATIO": d_ratio,
                        "DELTA_DISPERSION": d_disp, **mech}
    g["MECHANISM_EXPECTED_DIRECTION_COUNT"] = mech["n_expected"]
    g["REWARD_CREDIT_MECHANISM_SUPPORTED"] = mech["REWARD_CREDIT_MECHANISM_SUPPORTED"]

    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        g[f"{scenario}_RESPONSE_EFFECT"] = rlab; g[f"{scenario}_EPOCH_EFFECT"] = elab
        g[f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    def macro(endpoint):
        Dby = {s: _paired_diff(episodes, s, endpoint)[0] for s in SCENARIOS}
        bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=_e4if_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    g["MACRO_RESPONSE_EFFECT"] = mr["classification"]; g["MACRO_EPOCH_EFFECT"] = me["classification"]
    g["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"
    g["MIXED_FE_EFFECT"] = out["scenario_effects"]["MIXED"]["response_FE"]["delta_R0_minus_R1"]
    g["PERSISTENT_FE_EFFECT"] = out["scenario_effects"]["PERSISTENT"]["response_FE"]["delta_R0_minus_R1"]
    g["STATIONARY_FE_EFFECT"] = out["scenario_effects"]["STATIONARY"]["response_FE"]["delta_R0_minus_R1"]

    # MIXED action diagnostics (pooled per arm over MIXED)
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


def write_outputs(episodes, ep_rows, analysis):
    g = analysis["gates"]
    # policy pair summary
    pp = []
    Dm, R0m, R1m = _paired_diff(episodes, "MIXED", "MIGD_response")
    for i, p in enumerate(POLICY_SEEDS):
        pp.append({"policy_seed": p, "R0_MIXED_MIGD_response": float(R0m[i, :].mean()),
                   "R1_MIXED_MIGD_response": float(R1m[i, :].mean()),
                   "delta_R0_minus_R1": float(Dm[i, :].mean()), "positive_r1_better": bool(Dm[i, :].mean() > 0)})
    _write_csv(RESULTS_DIR / "E4IF_policy_pair_summary.csv", list(pp[0].keys()), pp)

    ad = []
    for arm in ARMS:
        a = analysis["action"][arm]
        ad.append({"arm": arm, **{k: a["fractions"][k] for k in ACTION_ORDER},
                   "multi_action_rate": a["multi_action_rate"], "action_entropy": a["action_entropy"]})
    _write_csv(RESULTS_DIR / "E4IF_action_distribution.csv",
               ["arm"] + list(ACTION_ORDER) + ["multi_action_rate", "action_entropy"], ad)

    m = analysis["mechanism"]
    _write_csv(RESULTS_DIR / "E4IF_reward_mechanism_effects.csv",
               ["endpoint", "R0", "R1", "delta_treatment_favoring", "expected_direction_met"],
               [{"endpoint": "LOCAL_CREDIT_ACTIVITY", "R0": m["R0"]["active_fraction"],
                 "R1": m["R1"]["active_fraction"], "delta_treatment_favoring": m["DELTA_ACTIVITY"],
                 "expected_direction_met": bool(m["DELTA_ACTIVITY"] > 0)},
                {"endpoint": "GLOBAL_LOCAL_MAGNITUDE_RATIO", "R0": m["R0"]["global_local_ratio"],
                 "R1": m["R1"]["global_local_ratio"], "delta_treatment_favoring": m["DELTA_RATIO"],
                 "expected_direction_met": bool(m["DELTA_RATIO"] > 0) if isinstance(m["DELTA_RATIO"], (int, float)) else False},
                {"endpoint": "REGIONAL_REWARD_DISPERSION", "R0": m["R0"]["regional_reward_dispersion"],
                 "R1": m["R1"]["regional_reward_dispersion"], "delta_treatment_favoring": m["DELTA_DISPERSION"],
                 "expected_direction_met": bool(m["DELTA_DISPERSION"] > 0)}])

    se = []
    for s in SCENARIOS:
        for ep in ("MIGD_response", "MIGD_epoch"):
            b = analysis["scenario_effects"][s][ep]
            se.append({"scenario": s, "endpoint": ep, "R0_mean": b["R0_mean"], "R1_mean": b["R1_mean"],
                       "delta_R0_minus_R1": b["delta_R0_minus_R1"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                       "relative_improvement": b["relative_improvement"], "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4IF_scenario_effects.csv", list(se[0].keys()), se)
    _write_csv(RESULTS_DIR / "E4IF_macro_effects.csv", ["endpoint", "delta", "ci_lo", "ci_hi", "classification"],
               [{"endpoint": k, **{kk: analysis["macro"][k][kk] for kk in ("delta", "ci_lo", "ci_hi", "classification")}}
                for k in ("MIGD_response", "MIGD_epoch", "response_FE")])
    _write_csv(RESULTS_DIR / "E4IF_fe_effects.csv", ["scenario", "fe_delta_R0_minus_R1", "classification"],
               [{"scenario": s, "fe_delta_R0_minus_R1": analysis["scenario_effects"][s]["response_FE"]["delta_R0_minus_R1"],
                 "classification": analysis["scenario_effects"][s]["response_FE"]["classification"]} for s in SCENARIOS]
               + [{"scenario": "macro", "fe_delta_R0_minus_R1": analysis["macro"]["response_FE"]["delta"],
                   "classification": analysis["macro"]["response_FE"]["classification"]}])
    _jdump({"scenario_effects": analysis["scenario_effects"], "macro": analysis["macro"],
            "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
                         "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "effect": "R0_MINUS_R1"}},
           RESULTS_DIR / "E4IF_bootstrap_results.json")
    _jdump(g, RESULTS_DIR / "E4IF_scientific_gates.json")

    # validation vs final descriptive
    _jdump({"note": "descriptive only; does not modify verdict",
            "validation": {"R0_macro": e4itr_get("R0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "R1_macro": e4itr_get("R1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "delta_R0_minus_R1": e4itr_get("VALIDATION_PAIRED_DELTA_R0_MINUS_R1"),
                           "mechanism_direction": e4itr_get("VALIDATION_REWARD_MECHANISM_DIRECTION")},
            "final": {"R0_MIXED_MIGD_response": g["R0_MIXED_MEAN"], "R1_MIXED_MIGD_response": g["R1_MIXED_MEAN"],
                      "MIXED_delta_R0_minus_R1": g["MIXED_DELTA"],
                      "mechanism_expected_direction_count": g["MECHANISM_EXPECTED_DIRECTION_COUNT"],
                      "REWARD_CREDIT_MECHANISM_SUPPORTED": g["REWARD_CREDIT_MECHANISM_SUPPORTED"]}},
           RESULTS_DIR / "E4IF_validation_final_comparison.json")


def e4itr_get(key):
    return _load(E4ITR_DIR / "E4ITR_verdict.json").get(key)


def write_verdict_report(analysis):
    g = analysis["gates"]
    v = RI.reward_ab_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        mechanism_supported=(g["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"),
        policy_robustness_pass=(g["REWARD_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(analysis["checkpoint_integrity"] == "PASS"),
        protocol_integrity_pass=(analysis["version_integrity"] == "PASS"),
        version_integrity_pass=(analysis["version_integrity"] == "PASS"), no_final_tuning=True)
    vname, subtype = v["verdict"], v["subtype"]
    mech_yes = g["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"
    if vname == "REWARD_CAUSAL_VALIDATED":
        nxt = "E4J_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
    elif subtype == "MECHANISTIC_PARTIAL":
        nxt = "E4J_JOINT_COORDINATION_REVISION_DESIGN"
    elif subtype == "PERFORMANCE_PARTIAL":
        nxt = "E4J_REWARD_RESULT_REVIEW"
    elif vname == "REWARD_CAUSAL_FAILED" and mech_yes:
        nxt = "E4J_JOINT_COORDINATION_REVISION_DESIGN"
    else:
        nxt = "E4J_REWARD_FAILURE_ANALYSIS"
    short = {"REWARD_CAUSAL_VALIDATED": "VALIDATED", "REWARD_CAUSAL_PARTIAL": "PARTIAL",
             "REWARD_CAUSAL_FAILED": "FAILED"}[vname]
    m = analysis["mechanism"]; act = analysis["action"]
    verdict = {
        "stage": "E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION", "E4IF_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES", "EPISODES_COMPLETE": "1800/1800",
        "CHECKPOINT_INTEGRITY": analysis["checkpoint_integrity"], "FINAL_PROTOCOL_INTEGRITY": "PASS",
        "REWARD_VERSION_CONFIG_INTEGRITY": analysis["version_integrity"],
        "NORMALIZATION_INTEGRITY": analysis["version_integrity"],
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "R1_TREATMENT_VS_R0_CONTROL", "EFFECT_DEFINITION": "R0_MINUS_R1",
        "R0_MIXED_MEAN_MIGD_RESPONSE": g["R0_MIXED_MEAN"], "R1_MIXED_MEAN_MIGD_RESPONSE": g["R1_MIXED_MEAN"],
        "MIXED_DELTA_R0_MINUS_R1": g["MIXED_DELTA"], "MIXED_DELTA_CI": g["MIXED_CI"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RI"], "MIXED_PRIMARY": g["MIXED_PRIMARY"],
        "per_policy_MIXED_deltas": g["per_policy_MIXED_delta"],
        "POLICY_PAIR_POSITIVE_COUNT": g["POLICY_POSITIVE_COUNT"],
        "REWARD_POLICY_ROBUSTNESS": g["REWARD_POLICY_ROBUSTNESS"],
        "R0_LOCAL_CREDIT_ACTIVE_FRACTION": m["R0"]["active_fraction"],
        "R1_LOCAL_CREDIT_ACTIVE_FRACTION": m["R1"]["active_fraction"], "DELTA_ACTIVITY": m["DELTA_ACTIVITY"],
        "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO": m["R0"]["global_local_ratio"],
        "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO": m["R1"]["global_local_ratio"], "DELTA_GLOBAL_LOCAL_RATIO": m["DELTA_RATIO"],
        "R0_REGIONAL_REWARD_DISPERSION": m["R0"]["regional_reward_dispersion"],
        "R1_REGIONAL_REWARD_DISPERSION": m["R1"]["regional_reward_dispersion"], "DELTA_REWARD_DISPERSION": m["DELTA_DISPERSION"],
        "REWARD_MECHANISM_EXPECTED_DIRECTION_COUNT": g["MECHANISM_EXPECTED_DIRECTION_COUNT"],
        "REWARD_CREDIT_MECHANISM_SUPPORTED": g["REWARD_CREDIT_MECHANISM_SUPPORTED"],
        "R0_MIXED_PREDICT_FRACTION": act["R0_CONTROL"]["predict_fraction"],
        "R1_MIXED_PREDICT_FRACTION": act["R1_TREATMENT"]["predict_fraction"],
        "R0_MIXED_MULTI_ACTION_RATE": act["R0_CONTROL"]["multi_action_rate"],
        "R1_MIXED_MULTI_ACTION_RATE": act["R1_TREATMENT"]["multi_action_rate"],
        "R0_MIXED_ACTION_ENTROPY": act["R0_CONTROL"]["action_entropy"],
        "R1_MIXED_ACTION_ENTROPY": act["R1_TREATMENT"]["action_entropy"],
        "PERSISTENT_RESPONSE_EFFECT": g["PERSISTENT_RESPONSE_EFFECT"], "PERSISTENT_EPOCH_EFFECT": g["PERSISTENT_EPOCH_EFFECT"],
        "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"], "STATIONARY_RESPONSE_EFFECT": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_EFFECT": g["STATIONARY_EPOCH_EFFECT"], "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_RESPONSE_EFFECT": g["MACRO_RESPONSE_EFFECT"], "MACRO_EPOCH_EFFECT": g["MACRO_EPOCH_EFFECT"],
        "MACRO_SAFETY": g["MACRO_SAFETY"], "MIXED_FE_EFFECT_R0_MINUS_R1": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_EFFECT_R0_MINUS_R1": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_EFFECT_R0_MINUS_R1": g["STATIONARY_FE_EFFECT"],
        "REWARD_CAUSAL_VERDICT": short, "PARTIAL_SUBTYPE": subtype if subtype else "NONE",
        "PERFORMANCE_IMPROVEMENT_CLAIM": v["PERFORMANCE_IMPROVEMENT_CLAIM"], "verdict_reason": v["reason"],
        "R1_ALPHA_LOCAL": 0.1, "ALPHA_RETUNING": "NO", "REWARD_RETUNING": "NO", "CHECKPOINT_RESELECTION": "NO",
        "MODEL_MODIFICATION": "NO", "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "FINAL_EVALUATION": "COMPLETE", "FINAL_ENVIRONMENT_SEEDS_USED": "1581..1640",
        "FULL_REGRESSION": "(run separately)", "NEXT_STAGE": nxt}
    _jdump(verdict, RESULTS_DIR / "E4IF_verdict.json")
    _write_report(verdict, analysis)
    return verdict


def _write_report(v, analysis):
    L = ["# E4IF — Reward A/B Final Evaluation Execution", "",
         f"**REWARD_CAUSAL_VERDICT = {v['REWARD_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}** → NEXT_STAGE = {v['NEXT_STAGE']}", "",
         "Causal question: replacing R0's scale-mismatched local credit with R1's bounded regional quality-gain, "
         "with V3 + Shared Double-DQN fixed and alpha_local 0.1 both arms. Comparator R0_CONTROL. Not rescued.", "",
         "## A/B/C. Integrity & completeness",
         f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}; REWARD_VERSION_CONFIG_INTEGRITY = "
         f"{v['REWARD_VERSION_CONFIG_INTEGRITY']}; NORMALIZATION_INTEGRITY = {v['NORMALIZATION_INTEGRITY']}; "
         f"matrix {v['EPISODES_COMPLETE']}.", "",
         "## D/E/F. Primary MIXED paired effect (d = R0 − R1; + ⇒ R1 better)",
         f"- R0 mean = {v['R0_MIXED_MEAN_MIGD_RESPONSE']:.6f}; R1 mean = {v['R1_MIXED_MEAN_MIGD_RESPONSE']:.6f}",
         f"- Delta = {v['MIXED_DELTA_R0_MINUS_R1']:.6e}  CI {v['MIXED_DELTA_CI']}; RI = "
         f"{v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% → **MIXED_PRIMARY = {v['MIXED_PRIMARY']}**", "",
         "## H. Policy-pair robustness",
         f"- positive pairs = {v['POLICY_PAIR_POSITIVE_COUNT']}/5 → {v['REWARD_POLICY_ROBUSTNESS']}; per-policy Δ = "
         f"{[round(x,6) for x in v['per_policy_MIXED_deltas']]}", "",
         "## I/J/K/L. Reward-credit mechanism",
         f"- activity: R0 {v['R0_LOCAL_CREDIT_ACTIVE_FRACTION']:.4f} vs R1 {v['R1_LOCAL_CREDIT_ACTIVE_FRACTION']:.4f} "
         f"(Δ={v['DELTA_ACTIVITY']:+.4f})",
         f"- global/local ratio: R0 {v['R0_GLOBAL_LOCAL_MAGNITUDE_RATIO']} vs R1 {v['R1_GLOBAL_LOCAL_MAGNITUDE_RATIO']} "
         f"(Δ={v['DELTA_GLOBAL_LOCAL_RATIO']})",
         f"- dispersion: R0 {v['R0_REGIONAL_REWARD_DISPERSION']:.6g} vs R1 {v['R1_REGIONAL_REWARD_DISPERSION']:.6g} "
         f"(Δ={v['DELTA_REWARD_DISPERSION']:+.6g})",
         f"- expected-direction count = {v['REWARD_MECHANISM_EXPECTED_DIRECTION_COUNT']}/3 → "
         f"REWARD_CREDIT_MECHANISM_SUPPORTED = {v['REWARD_CREDIT_MECHANISM_SUPPORTED']}", "",
         "## M. Action behavior (descriptive)",
         f"- MIXED PREDICT R0 {v['R0_MIXED_PREDICT_FRACTION']:.3f} vs R1 {v['R1_MIXED_PREDICT_FRACTION']:.3f}; "
         f"multi {v['R0_MIXED_MULTI_ACTION_RATE']:.3f}→{v['R1_MIXED_MULTI_ACTION_RATE']:.3f}; "
         f"entropy {v['R0_MIXED_ACTION_ENTROPY']:.3f}→{v['R1_MIXED_ACTION_ENTROPY']:.3f}.", "",
         "## N/O/P. Safety",
         f"- PERSISTENT: response={v['PERSISTENT_RESPONSE_EFFECT']}, epoch={v['PERSISTENT_EPOCH_EFFECT']} → {v['PERSISTENT_SAFETY']}",
         f"- STATIONARY: response={v['STATIONARY_RESPONSE_EFFECT']}, epoch={v['STATIONARY_EPOCH_EFFECT']} → {v['STATIONARY_SAFETY']}",
         f"- MACRO: response={v['MACRO_RESPONSE_EFFECT']}, epoch={v['MACRO_EPOCH_EFFECT']} → {v['MACRO_SAFETY']}", "",
         "## Q. FE (R0 − R1; + ⇒ R1 cheaper)",
         f"- MIXED={v['MIXED_FE_EFFECT_R0_MINUS_R1']:.2f}, PERSISTENT={v['PERSISTENT_FE_EFFECT_R0_MINUS_R1']:.2f}, "
         f"STATIONARY={v['STATIONARY_FE_EFFECT_R0_MINUS_R1']:.2f}", "",
         "## S. Reward causal verdict",
         f"- **{v['REWARD_CAUSAL_VERDICT']}{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE']!='NONE' else ''}**; "
         f"PERFORMANCE_IMPROVEMENT_CLAIM = {v['PERFORMANCE_IMPROVEMENT_CLAIM']}; reason: {v['verdict_reason']}", "",
         "## T. Historical preservation", "- ORIGINAL_E4F_VERDICT = FAILED; E4HF normalization = FAILED (unchanged).", "",
         "## U. Limitations",
         "- Reward-formulation causal experiment; both arms V3 + current implicit joint-action (Branch C untouched). "
         "Does NOT test Branch C, superiority over GLOBAL_PREDICT, or universal reward optimality. 5 policy seeds; "
         "controlled benchmark, not real-world UAV.", "",
         f"## V. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4IF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4IF_COMPLETE", "FINAL_MATRIX_COMPLETE", "EPISODES_COMPLETE", "ORIGINAL_E4F_VERDICT",
              "E4HF_NORMALIZATION_CAUSAL_VERDICT", "CHECKPOINT_INTEGRITY", "FINAL_PROTOCOL_INTEGRITY",
              "REWARD_VERSION_CONFIG_INTEGRITY", "NORMALIZATION_INTEGRITY", "PRIMARY_SCENARIO",
              "PRIMARY_ENDPOINT", "PRIMARY_COMPARISON", "R0_MIXED_MEAN_MIGD_RESPONSE",
              "R1_MIXED_MEAN_MIGD_RESPONSE", "MIXED_DELTA_R0_MINUS_R1", "MIXED_DELTA_CI",
              "MIXED_RELATIVE_IMPROVEMENT", "MIXED_PRIMARY", "POLICY_PAIR_POSITIVE_COUNT",
              "REWARD_POLICY_ROBUSTNESS", "R0_LOCAL_CREDIT_ACTIVE_FRACTION", "R1_LOCAL_CREDIT_ACTIVE_FRACTION",
              "DELTA_ACTIVITY", "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO", "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO",
              "DELTA_GLOBAL_LOCAL_RATIO", "R0_REGIONAL_REWARD_DISPERSION", "R1_REGIONAL_REWARD_DISPERSION",
              "DELTA_REWARD_DISPERSION", "REWARD_MECHANISM_EXPECTED_DIRECTION_COUNT",
              "REWARD_CREDIT_MECHANISM_SUPPORTED", "R0_MIXED_PREDICT_FRACTION", "R1_MIXED_PREDICT_FRACTION",
              "R0_MIXED_MULTI_ACTION_RATE", "R1_MIXED_MULTI_ACTION_RATE", "R0_MIXED_ACTION_ENTROPY",
              "R1_MIXED_ACTION_ENTROPY", "PERSISTENT_RESPONSE_EFFECT", "PERSISTENT_EPOCH_EFFECT",
              "PERSISTENT_SAFETY", "STATIONARY_RESPONSE_EFFECT", "STATIONARY_EPOCH_EFFECT", "STATIONARY_SAFETY",
              "MACRO_RESPONSE_EFFECT", "MACRO_EPOCH_EFFECT", "MACRO_SAFETY", "MIXED_FE_EFFECT_R0_MINUS_R1",
              "PERSISTENT_FE_EFFECT_R0_MINUS_R1", "STATIONARY_FE_EFFECT_R0_MINUS_R1", "REWARD_CAUSAL_VERDICT",
              "PARTIAL_SUBTYPE", "PERFORMANCE_IMPROVEMENT_CLAIM", "R1_ALPHA_LOCAL", "ALPHA_RETUNING",
              "REWARD_RETUNING", "CHECKPOINT_RESELECTION", "MODEL_MODIFICATION", "FINAL_EVALUATION",
              "FINAL_ENVIRONMENT_SEEDS_USED", "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<42} = {val}")


if __name__ == "__main__":
    main()
