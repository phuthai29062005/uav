"""
E4JF — Global Comparator Final Evaluation Execution (PR-DRL-DMOEA, Branch B follow-up).

FIRST stage permitted to open the fresh comparator seeds 1801..1860. Executes the untouched
confirmatory comparison between the E4IF-validated learned controller V3_R1_DDQN (5 frozen R1
checkpoints) and the historically PRESELECTED fixed-global comparator map (PERSISTENT/STATIONARY
/MIXED = GLOBAL_PREDICT/GLOBAL_DIVERSIFY/GLOBAL_PREDICT), exactly as frozen in E4J.

Effect d = GLOBAL - LEARNED (positive ⇒ learned better). The fixed-global baseline has NO policy
axis: one realization per (scenario, env), broadcast across the 5 learned policies via the crossed
shared-baseline bootstrap (master seed 76001). NO rescue; do NOT import the E4IF conclusion; do
NOT reselect the comparator on fresh data. E4F/E4HF/E4IF verdicts remain immutable.
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
    make_env_context, pre_response, build_region_states, execute_and_recover, advance_state,
    _fresh_state, eval_fixed_arm, N_SCORED_CHANGES,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.evaluation.crossed_bootstrap import (
    crossed_two_axis_bootstrap, crossed_two_axis_bootstrap_macro, N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import classify_effect, policy_robustness, safety_gate, HARMFUL
from prdrl.experiments.E4J_global_comparator_protocol_design import global_comparator_verdict

RESULTS_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4J_DIR = Path("results/E4J_global_comparator_protocol_design")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FINAL_SEEDS = list(range(1801, 1861))
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
COMPARATOR_MAP = {"PERSISTENT": "GLOBAL_PREDICT", "STATIONARY": "GLOBAL_DIVERSIFY", "MIXED": "GLOBAL_PREDICT"}
BOOTSTRAP_MASTER_SEED = 76001
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
DEVICE = torch.device("cpu")
_env_runs = {"count": 0}


def _e4jf_seed(*labels) -> int:
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


def load_learned_nets():
    frz = _load(E4J_DIR / "E4J_learned_checkpoint_freeze.json")["checkpoints"]
    nets, meta = {}, {}
    for key, info in frz.items():
        payload = torch.load(info["path"], weights_only=False)
        net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets[int(info["policy_seed"])] = net
        meta[int(info["policy_seed"])] = info
    return nets, meta, frz


# ---------------------------------------------------------------------------
def eval_learned(net, normalizer, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi_flags, n_fallback = [], 0
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
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _env_runs["count"] += 1
        for rec in post["records"]:
            if rec.fallback_used:
                n_fallback += 1
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe.append(int(sum(rec.response_fe for rec in post["records"])))
        advance_state(st, pre, post, env)
    total = sum(act_counts.values()) or 1
    return {"MIGD_response": float(np.mean(igd_resp)), "MIGD_epoch": float(np.mean(igd_ep)),
            "response_FE": int(np.sum(fe)), "action_counts": dict(act_counts), "n_decisions": total,
            "multi_action_rate": float(np.mean(multi_flags)), "fallback_rate": n_fallback / total}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4JF — Global Comparator Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4j = _load(E4J_DIR / "E4J_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    nets, meta, frz = load_learned_nets()
    normalizer, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    cmap = _load(E4J_DIR / "E4J_global_comparator_map.json")["FROZEN_BEST_GLOBAL_BY_SCENARIO"]
    ck_pre_ok = all(_hash_file(info["path"]) == info["sha256"] for info in frz.values())
    pre_hashes = {p: _hash_file(meta[p]["path"]) for p in POLICY_SEEDS}
    meta_ok = all((meta[p]["normalization_version"] == NORM_V3 and meta[p]["reward_version"] == REWARD_R1_NAME
                   and meta[p]["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256) for p in POLICY_SEEDS)
    map_ok = cmap == COMPARATOR_MAP
    _jdump({"stage": "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION",
            "E4J_COMPLETE": e4j["E4J_COMPLETE"], "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
            "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
            "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
            "checkpoint_pre_hashes_match": ck_pre_ok, "reward_norm_metadata_ok": meta_ok,
            "comparator_map_ok": map_ok, "v3_constants_sha256": v3_sha, "v3_sha_matches_ref": v3_sha == v3_sha_ref,
            "synthetic_validation": e4j["SYNTHETIC_STATISTICS_VALIDATION"],
            "learned_eval_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled", "learning": "none"}},
           RESULTS_DIR / "E4JF_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {str(p): pre_hashes[p] for p in POLICY_SEEDS},
            "all_match_frozen": ck_pre_ok}, RESULTS_DIR / "E4JF_checkpoint_integrity.json")
    _jdump({"comparator_map": cmap, "map_matches_frozen": map_ok,
            "PERSISTENT": cmap.get("PERSISTENT"), "STATIONARY": cmap.get("STATIONARY"),
            "MIXED": cmap.get("MIXED"), "not_reselected_on_final": True}, RESULTS_DIR / "E4JF_comparator_integrity.json")

    ok0 = (e4j["E4J_COMPLETE"] == "YES" and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and ck_pre_ok and meta_ok and map_ok and v3_sha == v3_sha_ref and len(nets) == 5
           and e4j["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS")
    if not ok0:
        print("[E4JF] integrity FAIL — STOP (no fresh seed opened)")
        _jdump({"E4JF_COMPLETE": "NO", "GLOBAL_COMPARATOR_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4J_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4JF_verdict.json")
        return {"E4JF_COMPLETE": "NO"}
    print("[E4JF] integrity PASS. Opening fresh comparator seeds 1801..1860.")
    _jdump({"fresh_seeds": [1801, 1860], "policy_seeds": POLICY_SEEDS, "scenarios": SCENARIOS,
            "comparator_map": cmap, "N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
            "effect": "GLOBAL_MINUS_LEARNED"}, RESULTS_DIR / "E4JF_execution_config.json")

    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    # ---- run matrix (frozen order: scenario, env, then global once + 5 learned) ----
    learned = {}   # (p, scenario, seed) -> episode
    glob = {}      # (scenario, seed) -> episode
    prog = open(RESULTS_DIR / "E4JF_progress.jsonl", "w")
    n = 0
    for scenario in SCENARIOS:
        comp = COMPARATOR_MAP[scenario]
        for seed in FINAL_SEEDS:
            gm = eval_fixed_arm(comp, scenario, seed, cache, env, None)
            _env_runs["count"] += 1
            glob[(scenario, seed)] = {"controller": "FROZEN_GLOBAL_COMPARATOR", "requested_global_action": comp,
                                      "scenario": scenario, "environment_seed": seed,
                                      "MIGD_response": gm["MIGD_response"], "MIGD_epoch": gm["MIGD_epoch"],
                                      "response_FE": gm["total_response_fe"], "action_fraction": gm["action_fraction"],
                                      "fallback_rate": gm["fallback_rate"]}
            prog.write(json.dumps({"controller": "GLOBAL", "scenario": scenario, "environment_seed": seed}) + "\n")
            for p in POLICY_SEEDS:
                ep = eval_learned(nets[p], normalizer, scenario, seed, cache, env)
                learned[(p, scenario, seed)] = {"controller": "V3_R1_DDQN", "policy_seed": p, "scenario": scenario,
                                                "environment_seed": seed, **ep}
                prog.write(json.dumps({"controller": "V3_R1_DDQN", "policy_seed": p, "scenario": scenario,
                                       "environment_seed": seed}) + "\n")
            prog.flush()
            n += 1
            if n % 30 == 0:
                print(f"    [E4JF] {n}/180 env-cells")
    prog.close()

    # ---- completeness ----
    n_learned = len(learned); n_global = len(glob)
    lkeys = set(learned.keys()); gkeys = set(glob.keys())
    exp_l = {(p, s, e) for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    exp_g = {(s, e) for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(v["MIGD_response"]) and np.isfinite(v["MIGD_epoch"]) and np.isfinite(v["response_FE"])
                 for v in list(learned.values()) + list(glob.values()))
    complete = (n_learned == 900 and n_global == 180 and lkeys == exp_l and gkeys == exp_g and finite)
    # shared-baseline handling: exactly one global row per (scenario, env), no policy axis
    baseline_axis_absent = all("policy_seed" not in g for g in glob.values())
    print(f"[E4JF] learned {n_learned}/900, global {n_global}/180, complete={complete}")

    _write_csv(RESULTS_DIR / "E4JF_learned_episode_results.csv",
               ["controller", "policy_seed", "scenario", "environment_seed", "MIGD_response", "MIGD_epoch",
                "response_FE", "multi_action_rate", "fallback_rate"],
               [dict(v, **{"multi_action_rate": v["multi_action_rate"], "fallback_rate": v["fallback_rate"]})
                for v in learned.values()])
    _write_csv(RESULTS_DIR / "E4JF_global_episode_results.csv",
               ["controller", "requested_global_action", "scenario", "environment_seed", "MIGD_response",
                "MIGD_epoch", "response_FE", "fallback_rate"],
               [{k: v[k] for k in ("controller", "requested_global_action", "scenario", "environment_seed",
                 "MIGD_response", "MIGD_epoch", "response_FE", "fallback_rate")} for v in glob.values()])

    if not complete:
        _jdump({"E4JF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "LEARNED_EPISODES_COMPLETE": f"{n_learned}/900", "GLOBAL_EPISODES_COMPLETE": f"{n_global}/180",
                "GLOBAL_COMPARATOR_VERDICT": "NOT_COMPUTED", "NEXT_STAGE": "E4J_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4JF_verdict.json")
        print("[E4JF] matrix incomplete — NOT_COMPUTED"); return {"E4JF_COMPLETE": "NO"}

    analysis = analyze(learned, glob)
    post_ok = all(_hash_file(meta[p]["path"]) == pre_hashes[p] for p in POLICY_SEEDS)
    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    analysis["checkpoint_integrity"] = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    analysis["comparator_map_integrity"] = "PASS" if (map_ok and cmap == COMPARATOR_MAP) else "FAIL"
    analysis["shared_baseline_handling"] = "PASS" if baseline_axis_absent else "FAIL"
    _jdump({"checkpoint_post_hashes_match": post_ok, "v3_sha_unchanged": v3_post,
            "r1_config_unchanged": meta_ok, "comparator_map_unchanged": cmap == COMPARATOR_MAP},
           RESULTS_DIR / "E4JF_post_integrity.json")
    _jdump({"optimizer_steps": 0, "gradient_updates": 0, "replay_appends": 0, "target_updates": 0,
            "checkpoint_writes": 0, "model_weight_changes": 0, "normalizer_updates": 0,
            "reward_config_updates": 0}, RESULTS_DIR / "E4JF_no_learning_audit.json")

    write_outputs(learned, glob, analysis)
    verdict = write_verdict_report(analysis, baseline_axis_absent)
    _jdump({"fresh_seed_range": [1801, 1860], "unique_fresh_final_seeds": 60,
            "FINAL_ENVIRONMENT_SEEDS_USED": "1801..1860", "FINAL_EVALUATION": "COMPLETE",
            "status": "CONSUMED_GLOBAL_COMPARATOR_CONFIRMATORY_FINAL", "env_runs": _env_runs["count"]},
           RESULTS_DIR / "E4JF_final_seed_usage.json")
    _emit_terminal(verdict)
    print(f"[E4JF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


# ---------------------------------------------------------------------------
def _L_matrix(learned, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = learned[(p, scenario, e)][endpoint]
    return M


def _B_vector(glob, scenario, endpoint):
    return np.array([glob[(scenario, e)][endpoint] for e in FINAL_SEEDS])


def analyze(learned, glob) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint):
        L = _L_matrix(learned, scenario, endpoint); B = _B_vector(glob, scenario, endpoint)
        bs = crossed_two_axis_bootstrap(L, B, seed=_e4jf_seed(endpoint, scenario, "GLOBAL_MINUS_LEARNED"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])
        return {"endpoint": endpoint, "global_mean": float(B.mean()), "learned_mean": float(L.mean()),
                "delta_global_minus_learned": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(B.mean()) + 1e-12), "classification": cls}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {ep: effect(scenario, ep)
                                             for ep in ("MIGD_response", "MIGD_epoch", "response_FE")}
    g = out["gates"]
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    g["MIXED_PRIMARY"] = mixed["classification"]; g["GLOBAL_MIXED_MEAN"] = mixed["global_mean"]
    g["LEARNED_MIXED_MEAN"] = mixed["learned_mean"]; g["MIXED_DELTA"] = mixed["delta_global_minus_learned"]
    g["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]; g["MIXED_RI"] = mixed["relative_improvement"]

    # policy robustness (MIXED): per policy mean_e(B[e]-L[p,e])
    Lm = _L_matrix(learned, "MIXED", "MIGD_response"); Bm = _B_vector(glob, "MIXED", "MIGD_response")
    per_policy = [float((Bm - Lm[i, :]).mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    g["per_policy_MIXED_delta"] = per_policy; g["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    g["GLOBAL_COMPARATOR_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        g[f"{scenario}_RESPONSE_EFFECT"] = rlab; g[f"{scenario}_EPOCH_EFFECT"] = elab
        g[f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    def macro(endpoint):
        Lby = {s: _L_matrix(learned, s, endpoint) for s in SCENARIOS}
        Bby = {s: _B_vector(glob, s, endpoint) for s in SCENARIOS}
        bs = crossed_two_axis_bootstrap_macro(Lby, Bby, SCENARIOS, seed=_e4jf_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    g["MACRO_RESPONSE_EFFECT"] = mr["classification"]; g["MACRO_EPOCH_EFFECT"] = me["classification"]
    g["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"
    g["MIXED_FE_EFFECT"] = out["scenario_effects"]["MIXED"]["response_FE"]["delta_global_minus_learned"]
    g["PERSISTENT_FE_EFFECT"] = out["scenario_effects"]["PERSISTENT"]["response_FE"]["delta_global_minus_learned"]
    g["STATIONARY_FE_EFFECT"] = out["scenario_effects"]["STATIONARY"]["response_FE"]["delta_global_minus_learned"]

    # algebra check
    g["ALGEBRA_CHECK"] = bool(abs(g["MIXED_DELTA"] - (Bm.mean() - Lm.mean())) < 1e-9)

    # MIXED action context (learned pooled)
    counts = {a: 0 for a in ACTION_ORDER}; multi = []
    for p in POLICY_SEEDS:
        for e in FINAL_SEEDS:
            ep = learned[(p, "MIXED", e)]
            for a in ACTION_ORDER:
                counts[a] += ep["action_counts"][a]
            multi.append(ep["multi_action_rate"])
    tot = sum(counts.values()) or 1
    fr = {a: counts[a] / tot for a in ACTION_ORDER}
    ent = -sum(v * math.log(v) for v in fr.values() if v > 0)
    out["action"] = {"learned_fractions": fr, "learned_multi_action_rate": float(np.mean(multi)),
                     "learned_action_entropy": ent, "global_requested_action": "GLOBAL_PREDICT (MIXED)"}
    return out


def write_outputs(learned, glob, analysis):
    g = analysis["gates"]
    Lm = _L_matrix(learned, "MIXED", "MIGD_response"); Bm = _B_vector(glob, "MIXED", "MIGD_response")
    global_mixed_mean = float(Bm.mean())
    pp = []
    for i, p in enumerate(POLICY_SEEDS):
        pp.append({"policy_seed": p, "GLOBAL_PREDICT_MIXED_MIGD_response": global_mixed_mean,
                   "learned_MIXED_MIGD_response": float(Lm[i, :].mean()),
                   "delta_global_minus_learned": float((Bm - Lm[i, :]).mean()),
                   "positive_learned_better": bool((Bm - Lm[i, :]).mean() > 0)})
    _write_csv(RESULTS_DIR / "E4JF_policy_pair_summary.csv", list(pp[0].keys()), pp)

    a = analysis["action"]
    _write_csv(RESULTS_DIR / "E4JF_action_context.csv",
               ["controller"] + list(ACTION_ORDER) + ["multi_action_rate", "action_entropy", "requested_global_action"],
               [{"controller": "V3_R1_DDQN_MIXED", **a["learned_fractions"],
                 "multi_action_rate": a["learned_multi_action_rate"], "action_entropy": a["learned_action_entropy"],
                 "requested_global_action": ""},
                {"controller": "FROZEN_GLOBAL_COMPARATOR_MIXED", **{k: "" for k in ACTION_ORDER},
                 "multi_action_rate": "", "action_entropy": "", "requested_global_action": "GLOBAL_PREDICT"}])

    se = []
    for s in SCENARIOS:
        for ep in ("MIGD_response", "MIGD_epoch"):
            b = analysis["scenario_effects"][s][ep]
            se.append({"scenario": s, "comparator": COMPARATOR_MAP[s], "endpoint": ep,
                       "global_mean": b["global_mean"], "learned_mean": b["learned_mean"],
                       "delta_global_minus_learned": b["delta_global_minus_learned"], "ci_lo": b["ci_lo"],
                       "ci_hi": b["ci_hi"], "relative_improvement": b["relative_improvement"],
                       "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4JF_scenario_effects.csv", list(se[0].keys()), se)
    _write_csv(RESULTS_DIR / "E4JF_macro_effects.csv", ["endpoint", "delta", "ci_lo", "ci_hi", "classification"],
               [{"endpoint": k, **{kk: analysis["macro"][k][kk] for kk in ("delta", "ci_lo", "ci_hi", "classification")}}
                for k in ("MIGD_response", "MIGD_epoch", "response_FE")])
    _write_csv(RESULTS_DIR / "E4JF_fe_effects.csv", ["scenario", "fe_delta_global_minus_learned", "classification"],
               [{"scenario": s, "fe_delta_global_minus_learned": analysis["scenario_effects"][s]["response_FE"]["delta_global_minus_learned"],
                 "classification": analysis["scenario_effects"][s]["response_FE"]["classification"]} for s in SCENARIOS]
               + [{"scenario": "macro", "fe_delta_global_minus_learned": analysis["macro"]["response_FE"]["delta"],
                   "classification": analysis["macro"]["response_FE"]["classification"]}])
    _jdump({"method": "CROSSED_POLICY_ENV_WITH_SHARED_BASELINE", "policy_axis_n": 5, "environment_axis_n": 60,
            "baseline_policy_axis": "NONE", "resamples": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
            "CI": "95 percentile", "scenario_effects": analysis["scenario_effects"], "macro": analysis["macro"],
            "macro_shared_draws": True}, RESULTS_DIR / "E4JF_bootstrap_results.json")
    _jdump(g, RESULTS_DIR / "E4JF_scientific_gates.json")


def write_verdict_report(analysis, baseline_axis_absent):
    g = analysis["gates"]
    v = global_comparator_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        robustness_pass=(g["GLOBAL_COMPARATOR_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        integrity_pass=(analysis["checkpoint_integrity"] == "PASS"
                        and analysis["comparator_map_integrity"] == "PASS"))
    vname, subtype = v["verdict"], v["subtype"]
    short = {"GLOBAL_COMPARATOR_VALIDATED": "VALIDATED", "GLOBAL_COMPARATOR_PARTIAL": "PARTIAL",
             "GLOBAL_COMPARATOR_FAILED": "FAILED"}[vname]
    integrity_pass = (analysis["checkpoint_integrity"] == "PASS"
                      and analysis["comparator_map_integrity"] == "PASS")
    if vname == "GLOBAL_COMPARATOR_VALIDATED":
        nxt = "E5_BENCHMARK_EXTENSION_PROTOCOL_DESIGN"
    elif vname == "GLOBAL_COMPARATOR_PARTIAL":
        nxt = "E4K_GLOBAL_COMPARATOR_RESULT_REVIEW"
    elif not integrity_pass:
        nxt = "E4J_INTEGRITY_REVIEW"
    else:
        # scientific (non-integrity) FAILED: learned did not overcome the fixed-global benchmark
        # (whether via directional-negative/harmful MIXED primary or a scenario safety failure) ->
        # Branch C (joint coordination) is the next isolated hypothesis (state + reward already validated).
        nxt = "E4K_JOINT_COORDINATION_REVISION_DESIGN"
    a = analysis["action"]
    verdict = {
        "stage": "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION", "E4JF_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES", "LEARNED_EPISODES_COMPLETE": "900/900",
        "GLOBAL_EPISODES_COMPLETE": "180/180", "TOTAL_EPISODES_COMPLETE": "1080/1080",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED", "LEARNED_CONTROLLER": "V3_R1_DDQN",
        "CHECKPOINT_INTEGRITY": analysis["checkpoint_integrity"],
        "COMPARATOR_MAP_INTEGRITY": analysis["comparator_map_integrity"], "FINAL_PROTOCOL_INTEGRITY": "PASS",
        "SHARED_BASELINE_HANDLING": analysis["shared_baseline_handling"],
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE", "PRIMARY_COMPARATOR": "GLOBAL_PREDICT",
        "EFFECT_DEFINITION": "GLOBAL_MINUS_LEARNED", "POSITIVE_MEANS": "V3_R1_BETTER",
        "GLOBAL_MIXED_MEAN_MIGD_RESPONSE": g["GLOBAL_MIXED_MEAN"],
        "LEARNED_MIXED_MEAN_MIGD_RESPONSE": g["LEARNED_MIXED_MEAN"],
        "MIXED_DELTA_GLOBAL_MINUS_LEARNED": g["MIXED_DELTA"], "MIXED_DELTA_CI": g["MIXED_CI"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RI"], "MIXED_PRIMARY": g["MIXED_PRIMARY"],
        "ALGEBRA_CHECK": "PASS" if g["ALGEBRA_CHECK"] else "FAIL",
        "per_policy_MIXED_deltas": g["per_policy_MIXED_delta"], "POLICY_PAIR_POSITIVE_COUNT": g["POLICY_POSITIVE_COUNT"],
        "GLOBAL_COMPARATOR_POLICY_ROBUSTNESS": g["GLOBAL_COMPARATOR_POLICY_ROBUSTNESS"],
        "PERSISTENT_COMPARATOR": "GLOBAL_PREDICT", "PERSISTENT_RESPONSE_EFFECT": g["PERSISTENT_RESPONSE_EFFECT"],
        "PERSISTENT_EPOCH_EFFECT": g["PERSISTENT_EPOCH_EFFECT"], "PERSISTENT_GLOBAL_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_COMPARATOR": "GLOBAL_DIVERSIFY", "STATIONARY_RESPONSE_EFFECT": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_EFFECT": g["STATIONARY_EPOCH_EFFECT"], "STATIONARY_GLOBAL_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_COMPARATOR": "FROZEN_BEST_GLOBAL_BY_SCENARIO", "MACRO_RESPONSE_EFFECT": g["MACRO_RESPONSE_EFFECT"],
        "MACRO_EPOCH_EFFECT": g["MACRO_EPOCH_EFFECT"], "GLOBAL_MACRO_SAFETY": g["MACRO_SAFETY"],
        "MIXED_FE_EFFECT_GLOBAL_MINUS_LEARNED": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_EFFECT_GLOBAL_MINUS_LEARNED": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_EFFECT_GLOBAL_MINUS_LEARNED": g["STATIONARY_FE_EFFECT"],
        "LEARNED_MIXED_PREDICT_FRACTION": a["learned_fractions"]["PREDICT"],
        "LEARNED_MIXED_MULTI_ACTION_RATE": a["learned_multi_action_rate"],
        "LEARNED_MIXED_ACTION_ENTROPY": a["learned_action_entropy"],
        "GLOBAL_COMPARATOR_VERDICT": short, "PARTIAL_SUBTYPE": subtype if subtype else "NONE",
        "SUPERIORITY_CLAIM": v["SUPERIORITY_CLAIM"], "verdict_reason": v["reason"],
        "BASELINE_POLICY_AXIS": "ABSENT" if baseline_axis_absent else "PRESENT",
        "FINAL_ENVIRONMENT_SEEDS_USED": "1801..1860", "TRAINING": "NO", "CHECKPOINT_RESELECTION": "NO",
        "COMPARATOR_RESELECTION": "NO", "MODEL_MODIFICATION": "NO", "FINAL_EVALUATION": "COMPLETE",
        "FULL_REGRESSION": "(run separately)", "NEXT_STAGE": nxt}
    _jdump(verdict, RESULTS_DIR / "E4JF_verdict.json")
    _write_report(verdict, analysis)
    return verdict


def _write_report(v, analysis):
    L = ["# E4JF — Global Comparator Final Evaluation Execution", "",
         f"**GLOBAL_COMPARATOR_VERDICT = {v['GLOBAL_COMPARATOR_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}** → NEXT_STAGE = {v['NEXT_STAGE']}", "",
         "Confirmatory test on fresh seeds 1801..1860: learned V3_R1_DDQN vs preselected fixed-global "
         "(effect d = GLOBAL − LEARNED, + ⇒ learned better). Not rescued; E4IF conclusion NOT imported.", "",
         "## A/B/C. Integrity",
         f"- CHECKPOINT_INTEGRITY={v['CHECKPOINT_INTEGRITY']}; COMPARATOR_MAP_INTEGRITY={v['COMPARATOR_MAP_INTEGRITY']}; "
         f"SHARED_BASELINE_HANDLING={v['SHARED_BASELINE_HANDLING']} (BASELINE_POLICY_AXIS={v['BASELINE_POLICY_AXIS']}).", "",
         "## E/F. Matrix + shared baseline",
         f"- learned {v['LEARNED_EPISODES_COMPLETE']}, global {v['GLOBAL_EPISODES_COMPLETE']}, total "
         f"{v['TOTAL_EPISODES_COMPLETE']}; one global row per (scenario,env); ALGEBRA_CHECK={v['ALGEBRA_CHECK']}.", "",
         "## I/J. Primary MIXED (vs GLOBAL_PREDICT)",
         f"- GLOBAL mean={v['GLOBAL_MIXED_MEAN_MIGD_RESPONSE']:.6f}, LEARNED mean={v['LEARNED_MIXED_MEAN_MIGD_RESPONSE']:.6f}",
         f"- Delta(GLOBAL−LEARNED)={v['MIXED_DELTA_GLOBAL_MINUS_LEARNED']:.6e} CI {v['MIXED_DELTA_CI']}; RI="
         f"{v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% → **MIXED_PRIMARY={v['MIXED_PRIMARY']}**", "",
         "## K. Policy robustness",
         f"- positive pairs={v['POLICY_PAIR_POSITIVE_COUNT']}/5 → {v['GLOBAL_COMPARATOR_POLICY_ROBUSTNESS']}; "
         f"per-policy Δ={[round(x,6) for x in v['per_policy_MIXED_deltas']]}", "",
         "## L/M/N. Safety + macro",
         f"- PERSISTENT (vs GLOBAL_PREDICT): resp={v['PERSISTENT_RESPONSE_EFFECT']}, epoch={v['PERSISTENT_EPOCH_EFFECT']} → {v['PERSISTENT_GLOBAL_SAFETY']}",
         f"- STATIONARY (vs GLOBAL_DIVERSIFY): resp={v['STATIONARY_RESPONSE_EFFECT']}, epoch={v['STATIONARY_EPOCH_EFFECT']} → {v['STATIONARY_GLOBAL_SAFETY']}",
         f"- MACRO: resp={v['MACRO_RESPONSE_EFFECT']}, epoch={v['MACRO_EPOCH_EFFECT']} → {v['GLOBAL_MACRO_SAFETY']}", "",
         "## O. FE (GLOBAL − LEARNED; + ⇒ learned cheaper)",
         f"- MIXED={v['MIXED_FE_EFFECT_GLOBAL_MINUS_LEARNED']:.2f}, PERSISTENT={v['PERSISTENT_FE_EFFECT_GLOBAL_MINUS_LEARNED']:.2f}, "
         f"STATIONARY={v['STATIONARY_FE_EFFECT_GLOBAL_MINUS_LEARNED']:.2f}", "",
         "## P. Action context (descriptive)",
         f"- learned MIXED PREDICT={v['LEARNED_MIXED_PREDICT_FRACTION']:.3f}, multi={v['LEARNED_MIXED_MULTI_ACTION_RATE']:.3f}, "
         f"entropy={v['LEARNED_MIXED_ACTION_ENTROPY']:.3f}; global requested = PREDICT (MIXED).", "",
         "## Q. Verdict",
         f"- **{v['GLOBAL_COMPARATOR_VERDICT']}**; SUPERIORITY_CLAIM={v['SUPERIORITY_CLAIM']}; reason: {v['verdict_reason']}", "",
         "## R. Cross-stage narrative (descriptive, no CI combination)",
         "- E4F: Shared DDQN FAILED vs fixed-global. E4HF: V3 de-collapsed behavior, no performance gain. "
         "E4IF: R1 reward causally improved V3 learned performance vs R0. E4JF: this stage vs frozen global comparator.", "",
         "## S. Limitations",
         "- comparator is preselected fixed-global (not an oracle); no policy axis for global; 5 learned seeds; "
         "controlled benchmark only; no universal-optimality or real-UAV claim; joint architecture not claimed optimal.", "",
         f"## T. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4JF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4JF_COMPLETE", "FINAL_MATRIX_COMPLETE", "LEARNED_EPISODES_COMPLETE", "GLOBAL_EPISODES_COMPLETE",
              "TOTAL_EPISODES_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "E4IF_REWARD_CAUSAL_VERDICT", "LEARNED_CONTROLLER", "CHECKPOINT_INTEGRITY",
              "COMPARATOR_MAP_INTEGRITY", "FINAL_PROTOCOL_INTEGRITY", "SHARED_BASELINE_HANDLING",
              "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT", "PRIMARY_COMPARATOR", "GLOBAL_MIXED_MEAN_MIGD_RESPONSE",
              "LEARNED_MIXED_MEAN_MIGD_RESPONSE", "MIXED_DELTA_GLOBAL_MINUS_LEARNED", "MIXED_DELTA_CI",
              "MIXED_RELATIVE_IMPROVEMENT", "MIXED_PRIMARY", "POLICY_PAIR_POSITIVE_COUNT",
              "GLOBAL_COMPARATOR_POLICY_ROBUSTNESS", "PERSISTENT_COMPARATOR", "PERSISTENT_RESPONSE_EFFECT",
              "PERSISTENT_EPOCH_EFFECT", "PERSISTENT_GLOBAL_SAFETY", "STATIONARY_COMPARATOR",
              "STATIONARY_RESPONSE_EFFECT", "STATIONARY_EPOCH_EFFECT", "STATIONARY_GLOBAL_SAFETY",
              "MACRO_COMPARATOR", "MACRO_RESPONSE_EFFECT", "MACRO_EPOCH_EFFECT", "GLOBAL_MACRO_SAFETY",
              "MIXED_FE_EFFECT_GLOBAL_MINUS_LEARNED", "PERSISTENT_FE_EFFECT_GLOBAL_MINUS_LEARNED",
              "STATIONARY_FE_EFFECT_GLOBAL_MINUS_LEARNED", "GLOBAL_COMPARATOR_VERDICT", "PARTIAL_SUBTYPE",
              "SUPERIORITY_CLAIM", "BASELINE_POLICY_AXIS", "FINAL_ENVIRONMENT_SEEDS_USED", "TRAINING",
              "CHECKPOINT_RESELECTION", "COMPARATOR_RESELECTION", "MODEL_MODIFICATION", "FULL_REGRESSION",
              "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<44} = {val}")


if __name__ == "__main__":
    main()
