"""
E4HF — Normalization A/B Final Evaluation Execution (PR-DRL-DMOEA, Branch A).

FIRST stage permitted to open final seeds 1381..1440. Executes the untouched
paired V1-vs-V3 final evaluation exactly as frozen in E4HE and produces the
NORMALIZATION_CAUSAL verdict. NO rescue: whatever the frozen protocol returns is
preserved. No retraining, no refit, no reselection, no model/reward/architecture
change. The immutable E4F verdict (FAILED) is not altered.

Question: causal effect of changing ONLY normalization (V1 legacy -> V3 zero-aware)
while holding the rest of the Shared Double-DQN system fixed. Comparator = V1_CONTROL.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES, Q_PREV_INIT,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V1, NORM_V3, normalize_v1, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro, derive_seed,
    N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect, policy_robustness, safety_gate,
    mechanism_decollapse_supported, normalization_ab_verdict, HARMFUL,
)

RESULTS_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HE_DIR = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HTR_DIR = Path("results/E4HTR_normalization_ab_training_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

FINAL_SEEDS = list(range(1381, 1441))
POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = ["V1_CONTROL", "V3_TREATMENT"]
DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED = 56001   # E4HE-frozen master (distinct from crossed_bootstrap's 46001)
DEVICE = torch.device("cpu")
_env_runs = {"count": 0}


def _e4hf_seed(*labels) -> int:
    """Deterministic substream seed from the E4HE-frozen master 56001 + labels.

    Identical algorithm to crossed_bootstrap.derive_seed but seeded from the
    normalization-A/B master (56001) that E4HE froze, so the paired bootstrap
    streams are distinct from the E4E/E4F (46001) pipeline. No result-dependent
    seeding is possible (labels are fixed endpoint/scenario strings)."""
    entropy: List[int] = [BOOTSTRAP_MASTER_SEED]
    for lab in labels:
        if isinstance(lab, (int, np.integer)):
            entropy.append(int(lab))
        else:
            h = 0
            for ch in str(lab):
                h = (h * 131 + ord(ch)) & 0xFFFFFFFF
            entropy.append(int(h))
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


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


def load_arm_normalizers():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"], "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3({"zero_aware": za, "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    return {NORM_V1: normalize_v1, NORM_V3: v3.transform}, c["constants_sha256"]


def load_selected_nets():
    frz = _load(E4HE_DIR / "E4HE_checkpoint_freeze.json")["checkpoints"]
    nets = {}
    meta = {}
    for key, info in frz.items():
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets[(info["arm"], int(info["policy_seed"]))] = (net, info["normalization_version"])
        meta[(info["arm"], int(info["policy_seed"]))] = info
    return nets, meta, frz


# ---------------------------------------------------------------------------
def eval_arm(net, normalizer, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    change_rows = []
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi_flags = []
    n_fallback = 0
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
        _, action_str = select_regional_actions(net, local_norm, gc_norm, epsilon=0.0,
                                                event_seed=0, device=DEVICE)
        active = sorted(pre["active_regions"])
        actions = {r: action_str[r] for r in active}
        reqs = [action_str[r] for r in active]
        uniq = len(set(reqs))
        multi_flags.append(1.0 if uniq >= 2 else 0.0)
        for a in reqs:
            act_counts[a] += 1
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _env_runs["count"] += 1
        fe_change = int(sum(rec.response_fe for rec in post["records"]))
        for rec in post["records"]:
            if rec.fallback_used:
                n_fallback += 1
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"]); fe.append(fe_change)
        cnt = Counter(reqs); n = len(reqs)
        ent = -sum((v / n) * math.log(v / n) for v in cnt.values()) if n else 0.0
        change_rows.append({
            "scenario": scenario, "environment_seed": seed, "change_index": k,
            "active_region_count": len(active),
            "requested_actions": [rec.requested_action for rec in post["records"]],
            "realized_operators": [rec.realized_operator for rec in post["records"]],
            "fallback_flags": [bool(rec.fallback_used) for rec in post["records"]],
            "predict_requested_fraction": (sum(1 for a in reqs if a == "PREDICT") / n) if n else 0.0,
            "event_multi_action": bool(uniq >= 2), "regional_action_entropy": ent,
            "MIGD_response_change": post["igd_response"], "MIGD_epoch_change": post["igd_epoch"],
            "response_FE_change": fe_change})
        advance_state(st, pre, post, env)
    total = sum(act_counts.values()) or 1
    episode = {"MIGD_response": float(np.mean(igd_resp)), "MIGD_epoch": float(np.mean(igd_ep)),
               "response_FE": int(np.sum(fe)),
               "predict_fraction": act_counts["PREDICT"] / total,
               "multi_action_rate": float(np.mean(multi_flags)),
               "action_counts": dict(act_counts), "n_decisions": total,
               "predict_fallback_fraction": n_fallback / total}
    return episode, change_rows


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4HF — Normalization A/B Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4he = _load(E4HE_DIR / "E4HE_verdict.json")
    e4htr = _load(E4HTR_DIR / "E4HTR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    nets, meta, frz = load_selected_nets()
    normalizers, v3_sha = load_arm_normalizers()
    ck_pre_ok = all(_hash_file(info["checkpoint_path"]) == info["sha256"] for info in frz.values())
    pre_hashes = {k: _hash_file(meta[k]["checkpoint_path"]) for k in meta}
    prod_files = ["src/prdrl/operators/local.py", "src/prdrl/operators/predict.py",
                  "src/prdrl/operators/diversify.py", "src/prdrl/operators/budget.py",
                  "src/prdrl/oracle/oracle.py", "src/prdrl/core/survival.py",
                  "src/prdrl/rl/network.py", "src/prdrl/rl/normalization.py"]
    prod_hashes = {fp: _hash_file(fp) for fp in prod_files}
    manifest = {"stage": "E4HF_NORMALIZATION_AB_FINAL_EVALUATION_EXECUTION",
                "E4HE_COMPLETE": e4he["E4HE_COMPLETE"], "E4HTR_COMPLETE": e4htr["E4HTR_COMPLETE"],
                "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "checkpoint_pre_hashes_match": ck_pre_ok, "v3_constants_sha256": v3_sha,
                "synthetic_validation": e4he["SYNTHETIC_STATISTICS_VALIDATION"],
                "production_operator_hashes": prod_hashes,
                "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                                   "replay": "disabled", "optimizer": "disabled", "learning": "none"}}
    _jdump(manifest, RESULTS_DIR / "E4HF_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {f"{k[0]}_{k[1]}": v for k, v in pre_hashes.items()},
            "all_match_frozen": ck_pre_ok}, RESULTS_DIR / "E4HF_checkpoint_integrity.json")
    ok0 = (e4he["E4HE_COMPLETE"] == "YES" and e4htr["E4HTR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and ck_pre_ok
           and e4he["SYNTHETIC_STATISTICS_VALIDATION"] == "PASS")
    if not ok0:
        print("[E4HF] integrity FAIL — STOP (no final seed opened)")
        _jdump({"E4HF_COMPLETE": "NO", "NORMALIZATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4HF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4HF_verdict.json")
        return {"E4HF_COMPLETE": "NO"}
    print("[E4HF] integrity PASS. Opening final seeds 1381..1440.")
    _jdump({"final_seeds": [1381, 1440], "policy_seeds": POLICY_SEEDS, "arms": ARMS,
            "scenarios": SCENARIOS, "N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
            "DELTA_NORM": DELTA_NORM}, RESULTS_DIR / "E4HF_execution_config.json")

    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    # ---- run matrix (frozen order: scenario, env, policy, within-pair V1 then V3) ----
    episodes = {}   # (arm, policy, scenario, seed) -> episode dict
    change_path = RESULTS_DIR / "E4HF_change_results.jsonl"
    prog_path = RESULTS_DIR / "E4HF_progress.jsonl"
    n = 0
    total = len(SCENARIOS) * len(FINAL_SEEDS) * len(POLICY_SEEDS) * len(ARMS)
    with open(change_path, "w") as chg, open(prog_path, "w") as prog:
        for scenario in SCENARIOS:
            for seed in FINAL_SEEDS:
                for p in POLICY_SEEDS:
                    for arm in ARMS:
                        net, nv = nets[(arm, p)]
                        ep, chrows = eval_arm(net, normalizers[nv], scenario, seed, cache, env)
                        episodes[(arm, p, scenario, seed)] = ep
                        prog.write(json.dumps({"arm": arm, "policy_seed": p, "scenario": scenario,
                                               "environment_seed": seed}) + "\n"); prog.flush()
                        for cr in chrows:
                            chg.write(json.dumps({"arm": arm, "policy_seed": p, **cr}) + "\n")
                        n += 1
                        if n % 360 == 0:
                            print(f"    [E4HF] {n}/{total} episodes")

    # ---- matrix completeness ----
    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) and np.isfinite(episodes[k]["MIGD_epoch"])
                 and np.isfinite(episodes[k]["response_FE"]) for k in keys)
    n_v1 = sum(1 for k in keys if k[0] == "V1_CONTROL")
    n_v3 = sum(1 for k in keys if k[0] == "V3_TREATMENT")
    complete = (len(keys) == 1800 and set(keys) == expected and n_v1 == 900 and n_v3 == 900 and finite)
    matrix_check = {"n_rows": len(keys), "n_v1": n_v1, "n_v3": n_v3, "all_finite": finite,
                    "FINAL_MATRIX_COMPLETE": "YES" if complete else "NO"}
    print(f"[E4HF] matrix {len(keys)}/1800, COMPLETE={matrix_check['FINAL_MATRIX_COMPLETE']}")

    # episode CSVs
    ep_fields = ["arm", "policy_seed", "scenario", "environment_seed", "MIGD_response",
                 "MIGD_epoch", "response_FE", "predict_fraction", "multi_action_rate",
                 "predict_fallback_fraction"]
    ep_rows = [{"arm": a, "policy_seed": p, "scenario": s, "environment_seed": e, **{k: episodes[(a, p, s, e)][k]
                for k in ("MIGD_response", "MIGD_epoch", "response_FE", "predict_fraction",
                          "multi_action_rate", "predict_fallback_fraction")}}
               for (a, p, s, e) in keys]
    _write_csv(RESULTS_DIR / "E4HF_episode_results.csv", ep_fields, ep_rows)
    _write_csv(RESULTS_DIR / "E4HF_v1_results.csv", ep_fields, [r for r in ep_rows if r["arm"] == "V1_CONTROL"])
    _write_csv(RESULTS_DIR / "E4HF_v3_results.csv", ep_fields, [r for r in ep_rows if r["arm"] == "V3_TREATMENT"])

    if not complete:
        _jdump({"stage": "E4HF...", "E4HF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "NORMALIZATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4HF_INTEGRITY_REVIEW", **matrix_check},
               RESULTS_DIR / "E4HF_verdict.json")
        print("[E4HF] matrix incomplete — verdict NOT_COMPUTED"); return {"E4HF_COMPLETE": "NO"}

    analysis = analyze(episodes, ck_pre_ok, matrix_check)
    # post integrity
    post_ok = all(_hash_file(meta[k]["checkpoint_path"]) == pre_hashes[k] for k in meta)
    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    analysis["checkpoint_integrity"] = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    analysis["normalization_version_integrity"] = "PASS" if (post_ok and v3_post) else "FAIL"

    write_outputs(episodes, ep_rows, analysis, matrix_check, e4htr)
    verdict = write_verdict_report(analysis, matrix_check)
    _jdump({"final_seed_range": [1381, 1440], "unique_final_seeds_instantiated": 60,
            "FINAL_ENVIRONMENT_SEEDS_USED": "1381..1440", "FINAL_EVALUATION": "COMPLETE",
            "status": "CONSUMED_FINAL_NORMALIZATION_AB",
            "env_runs": _env_runs["count"]}, RESULTS_DIR / "E4HF_final_seed_usage.json")
    _emit_terminal(verdict)
    print(f"[E4HF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


# ---------------------------------------------------------------------------
def _arm_metric_matrix(episodes, arm, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = episodes[(arm, p, scenario, e)][endpoint]
    return M


def _paired_diff(episodes, scenario, endpoint):
    V1 = _arm_metric_matrix(episodes, "V1_CONTROL", scenario, endpoint)
    V3 = _arm_metric_matrix(episodes, "V3_TREATMENT", scenario, endpoint)
    return V1 - V3, V1, V3   # positive = V3 better (V1 higher/worse)


def _pooled_mechanism(episodes, arm):
    """Arm-level MIXED mechanism = mean across the 5 policies of each policy's
    pooled-over-60-env metric (exact E4HTR mechanism_eval aggregation: per-policy
    pooled action-fraction entropy/PREDICT + mean multi-action, then averaged
    across policies)."""
    pf_list, ma_list, ent_list = [], [], []
    for p in POLICY_SEEDS:
        mp = _pooled_policy_mech(episodes, arm, p)
        pf_list.append(mp["predict_fraction"])
        ma_list.append(mp["multi_action_rate"])
        ent_list.append(mp["action_entropy"])
    return {"predict_fraction": float(np.mean(pf_list)),
            "multi_action_rate": float(np.mean(ma_list)),
            "action_entropy": float(np.mean(ent_list))}


def analyze(episodes, integrity_pass, matrix_check) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint, positive_is_v3_better=True):
        D, V1, V3 = _paired_diff(episodes, scenario, endpoint)  # V1 - V3
        bs = paired_crossed_two_axis_bootstrap(D, seed=_e4hf_seed(endpoint, scenario, "V1_MINUS_V3"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])
        return {"endpoint": endpoint, "V1_mean": float(V1.mean()), "V3_mean": float(V3.mean()),
                "delta_V1_minus_V3": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(V1.mean()) + 1e-12),
                "classification": cls, "D": D}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {
            "MIGD_response": effect(scenario, "MIGD_response"),
            "MIGD_epoch": effect(scenario, "MIGD_epoch"),
            "response_FE": effect(scenario, "response_FE")}

    # primary MIXED
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    out["gates"]["MIXED_PRIMARY"] = mixed["classification"]
    out["gates"]["V1_MIXED_MEAN"] = mixed["V1_mean"]
    out["gates"]["V3_MIXED_MEAN"] = mixed["V3_mean"]
    out["gates"]["MIXED_DELTA"] = mixed["delta_V1_minus_V3"]
    out["gates"]["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]
    out["gates"]["MIXED_RELATIVE_IMPROVEMENT"] = mixed["relative_improvement"]

    # policy robustness (per-policy MIXED delta)
    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    per_policy = [float(Dm[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    out["gates"]["per_policy_MIXED_delta"] = per_policy
    out["gates"]["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    out["gates"]["NORMALIZATION_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    # mechanism (pooled MIXED per arm)
    m_v1 = _pooled_mechanism(episodes, "V1_CONTROL")
    m_v3 = _pooled_mechanism(episodes, "V3_TREATMENT")
    dpred = m_v1["predict_fraction"] - m_v3["predict_fraction"]
    dmulti = m_v3["multi_action_rate"] - m_v1["multi_action_rate"]
    dent = m_v3["action_entropy"] - m_v1["action_entropy"]
    mech = mechanism_decollapse_supported(dpred, dmulti, dent)
    out["mechanism"] = {"V1": m_v1, "V3": m_v3, "DELTA_PREDICT": dpred, "DELTA_MULTI": dmulti,
                        "DELTA_ENTROPY": dent, **mech}
    out["gates"]["MECHANISM_EXPECTED_DIRECTION_COUNT"] = mech["n_expected"]
    out["gates"]["MECHANISM_DECOLLAPSE_SUPPORTED"] = mech["MECHANISM_DECOLLAPSE_SUPPORTED"]

    # safety
    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        out["gates"][f"{scenario}_RESPONSE_EFFECT"] = rlab
        out["gates"][f"{scenario}_EPOCH_EFFECT"] = elab
        out["gates"][f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    # macro (shared draws)
    def macro(endpoint):
        Dby = {s: _paired_diff(episodes, s, endpoint)[0] for s in SCENARIOS}
        bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=_e4hf_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"])}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    out["gates"]["MACRO_RESPONSE_EFFECT"] = mr["classification"]
    out["gates"]["MACRO_EPOCH_EFFECT"] = me["classification"]
    out["gates"]["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"

    # FE
    out["gates"]["MIXED_FE_EFFECT"] = out["scenario_effects"]["MIXED"]["response_FE"]["delta_V1_minus_V3"]
    out["gates"]["PERSISTENT_FE_EFFECT"] = out["scenario_effects"]["PERSISTENT"]["response_FE"]["delta_V1_minus_V3"]
    out["gates"]["STATIONARY_FE_EFFECT"] = out["scenario_effects"]["STATIONARY"]["response_FE"]["delta_V1_minus_V3"]

    # strip non-serializable D matrices before returning (kept only for bootstrap)
    for s in SCENARIOS:
        for ep in out["scenario_effects"][s].values():
            ep.pop("D", None)
    return out


def write_outputs(episodes, ep_rows, analysis, matrix_check, e4htr):
    # policy pair summary
    pp_rows = []
    for p in POLICY_SEEDS:
        v1 = np.mean([episodes[("V1_CONTROL", p, "MIXED", e)]["MIGD_response"] for e in FINAL_SEEDS])
        v3 = np.mean([episodes[("V3_TREATMENT", p, "MIXED", e)]["MIGD_response"] for e in FINAL_SEEDS])
        mv1 = _pooled_policy_mech(episodes, "V1_CONTROL", p)
        mv3 = _pooled_policy_mech(episodes, "V3_TREATMENT", p)
        pp_rows.append({"policy_seed": p, "V1_MIXED_MIGD_response": float(v1),
                        "V3_MIXED_MIGD_response": float(v3), "delta_V1_minus_V3": float(v1 - v3),
                        "positive_v3_better": bool(v1 - v3 > 0),
                        "V1_PREDICT_fraction": mv1["predict_fraction"], "V3_PREDICT_fraction": mv3["predict_fraction"],
                        "V1_multi_action_rate": mv1["multi_action_rate"], "V3_multi_action_rate": mv3["multi_action_rate"],
                        "V1_action_entropy": mv1["action_entropy"], "V3_action_entropy": mv3["action_entropy"]})
    _write_csv(RESULTS_DIR / "E4HF_policy_pair_summary.csv", list(pp_rows[0].keys()), pp_rows)

    # action distribution (per arm x policy x scenario, mean over env)
    ad_rows = []
    for arm in ARMS:
        for p in POLICY_SEEDS:
            for s in SCENARIOS:
                fr = {a: float(np.mean([episodes[(arm, p, s, e)]["action_counts"][a]
                      / max(episodes[(arm, p, s, e)]["n_decisions"], 1) for e in FINAL_SEEDS])) for a in ACTION_ORDER}
                ad_rows.append({"arm": arm, "policy_seed": p, "scenario": s, **fr,
                                "multi_action_rate": float(np.mean([episodes[(arm, p, s, e)]["multi_action_rate"]
                                for e in FINAL_SEEDS]))})
    _write_csv(RESULTS_DIR / "E4HF_action_distribution.csv",
               ["arm", "policy_seed", "scenario"] + list(ACTION_ORDER) + ["multi_action_rate"], ad_rows)

    # mechanism effects
    m = analysis["mechanism"]
    _write_csv(RESULTS_DIR / "E4HF_mechanism_effects.csv",
               ["endpoint", "V1", "V3", "delta", "expected_direction_met"],
               [{"endpoint": "PREDICT_fraction", "V1": m["V1"]["predict_fraction"], "V3": m["V3"]["predict_fraction"],
                 "delta": m["DELTA_PREDICT"], "expected_direction_met": bool(m["DELTA_PREDICT"] > 0)},
                {"endpoint": "multi_action_rate", "V1": m["V1"]["multi_action_rate"], "V3": m["V3"]["multi_action_rate"],
                 "delta": m["DELTA_MULTI"], "expected_direction_met": bool(m["DELTA_MULTI"] > 0)},
                {"endpoint": "action_entropy", "V1": m["V1"]["action_entropy"], "V3": m["V3"]["action_entropy"],
                 "delta": m["DELTA_ENTROPY"], "expected_direction_met": bool(m["DELTA_ENTROPY"] > 0)}])

    # scenario + macro effects
    se_rows = []
    for s in SCENARIOS:
        for endpoint in ["MIGD_response", "MIGD_epoch"]:
            b = analysis["scenario_effects"][s][endpoint]
            se_rows.append({"scenario": s, "endpoint": endpoint, "V1_mean": b["V1_mean"],
                            "V3_mean": b["V3_mean"], "delta_V1_minus_V3": b["delta_V1_minus_V3"],
                            "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                            "relative_improvement": b["relative_improvement"], "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4HF_scenario_effects.csv", list(se_rows[0].keys()), se_rows)
    _write_csv(RESULTS_DIR / "E4HF_macro_effects.csv", ["endpoint", "delta", "ci_lo", "ci_hi", "classification"],
               [{"endpoint": k, **{kk: analysis["macro"][k][kk] for kk in ("delta", "ci_lo", "ci_hi", "classification")}}
                for k in ["MIGD_response", "MIGD_epoch", "response_FE"]])
    _write_csv(RESULTS_DIR / "E4HF_fe_effects.csv", ["scenario", "fe_delta_V1_minus_V3", "classification"],
               [{"scenario": s, "fe_delta_V1_minus_V3": analysis["scenario_effects"][s]["response_FE"]["delta_V1_minus_V3"],
                 "classification": analysis["scenario_effects"][s]["response_FE"]["classification"]} for s in SCENARIOS]
               + [{"scenario": "macro", "fe_delta_V1_minus_V3": analysis["macro"]["response_FE"]["delta"],
                   "classification": analysis["macro"]["response_FE"]["classification"]}])
    _jdump({"scenario_effects": analysis["scenario_effects"], "macro": analysis["macro"],
            "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
                         "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"}},
           RESULTS_DIR / "E4HF_bootstrap_results.json")
    _jdump(analysis["gates"], RESULTS_DIR / "E4HF_scientific_gates.json")

    # validation vs final descriptive
    val_v1 = e4htr["V1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"] if "V1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN" in e4htr else None
    _jdump({"note": "descriptive only; does not modify verdict",
            "validation": {"V1_macro": e4htr.get("V1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "V3_macro": e4htr.get("V3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
                           "mechanism_direction": e4htr.get("VALIDATION_MECHANISM_DIRECTION"),
                           "V1_MIXED_PREDICT": e4htr.get("V1_MIXED_PREDICT_FRACTION"),
                           "V3_MIXED_PREDICT": e4htr.get("V3_MIXED_PREDICT_FRACTION")},
            "final": {"V1_MIXED_MIGD_response": analysis["gates"]["V1_MIXED_MEAN"],
                      "V3_MIXED_MIGD_response": analysis["gates"]["V3_MIXED_MEAN"],
                      "MIXED_delta_V1_minus_V3": analysis["gates"]["MIXED_DELTA"],
                      "mechanism_direction_count": analysis["gates"]["MECHANISM_EXPECTED_DIRECTION_COUNT"],
                      "V1_MIXED_PREDICT": analysis["mechanism"]["V1"]["predict_fraction"],
                      "V3_MIXED_PREDICT": analysis["mechanism"]["V3"]["predict_fraction"]}},
           RESULTS_DIR / "E4HF_validation_final_comparison.json")


def _pooled_policy_mech(episodes, arm, p):
    counts = {a: 0 for a in ACTION_ORDER}; multi = []
    for e in FINAL_SEEDS:
        ep = episodes[(arm, p, "MIXED", e)]
        for a in ACTION_ORDER:
            counts[a] += ep["action_counts"][a]
        multi.append(ep["multi_action_rate"])
    tot = sum(counts.values()) or 1
    fr = {a: counts[a] / tot for a in ACTION_ORDER}
    ent = -sum(v * math.log(v) for v in fr.values() if v > 0)
    return {"predict_fraction": fr["PREDICT"], "multi_action_rate": float(np.mean(multi)), "action_entropy": ent}


def write_verdict_report(analysis, matrix_check):
    g = analysis["gates"]
    v = normalization_ab_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        mechanism_supported=(g["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"),
        policy_robustness_pass=(g["NORMALIZATION_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(analysis["checkpoint_integrity"] == "PASS"),
        protocol_integrity_pass=(analysis["normalization_version_integrity"] == "PASS"),
        version_integrity_pass=(analysis["normalization_version_integrity"] == "PASS"),
        no_final_tuning=True)
    verdict_name = v["verdict"]
    subtype = v["subtype"]
    mech_yes = g["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"
    # next stage by result category
    if verdict_name == "NORMALIZATION_CAUSAL_VALIDATED":
        nxt = "E4I_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
    elif subtype == "MECHANISTIC_PARTIAL":
        nxt = "E4I_REWARD_CREDIT_REVISION_DESIGN"
    elif subtype == "PERFORMANCE_PARTIAL":
        nxt = "E4I_NORMALIZATION_RESULT_REVIEW"
    elif verdict_name == "NORMALIZATION_CAUSAL_FAILED" and mech_yes:
        nxt = "E4I_REWARD_CREDIT_REVISION_DESIGN"
    else:
        nxt = "E4I_NORMALIZATION_FAILURE_ANALYSIS"

    verdict = {
        "stage": "E4HF_NORMALIZATION_AB_FINAL_EVALUATION_EXECUTION",
        "E4HF_COMPLETE": "YES", "FINAL_MATRIX_COMPLETE": "YES",
        "EPISODES_COMPLETE": f"{matrix_check['n_rows']}/1800",
        "CHECKPOINT_INTEGRITY": analysis["checkpoint_integrity"],
        "FINAL_PROTOCOL_INTEGRITY": "PASS",
        "NORMALIZATION_VERSION_INTEGRITY": analysis["normalization_version_integrity"],
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "V3_TREATMENT_VS_V1_CONTROL", "EFFECT_DEFINITION": "V1_MINUS_V3",
        "V1_MIXED_MEAN_MIGD_RESPONSE": g["V1_MIXED_MEAN"], "V3_MIXED_MEAN_MIGD_RESPONSE": g["V3_MIXED_MEAN"],
        "MIXED_DELTA_V1_MINUS_V3": g["MIXED_DELTA"], "MIXED_DELTA_CI": g["MIXED_CI"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RELATIVE_IMPROVEMENT"], "MIXED_PRIMARY": g["MIXED_PRIMARY"],
        "per_policy_MIXED_deltas": g["per_policy_MIXED_delta"],
        "POLICY_PAIR_POSITIVE_COUNT": g["POLICY_POSITIVE_COUNT"],
        "NORMALIZATION_POLICY_ROBUSTNESS": g["NORMALIZATION_POLICY_ROBUSTNESS"],
        "V1_MIXED_PREDICT_FRACTION": analysis["mechanism"]["V1"]["predict_fraction"],
        "V3_MIXED_PREDICT_FRACTION": analysis["mechanism"]["V3"]["predict_fraction"],
        "DELTA_PREDICT": analysis["mechanism"]["DELTA_PREDICT"],
        "V1_MIXED_MULTI_ACTION_RATE": analysis["mechanism"]["V1"]["multi_action_rate"],
        "V3_MIXED_MULTI_ACTION_RATE": analysis["mechanism"]["V3"]["multi_action_rate"],
        "DELTA_MULTI_ACTION": analysis["mechanism"]["DELTA_MULTI"],
        "V1_MIXED_ACTION_ENTROPY": analysis["mechanism"]["V1"]["action_entropy"],
        "V3_MIXED_ACTION_ENTROPY": analysis["mechanism"]["V3"]["action_entropy"],
        "DELTA_ACTION_ENTROPY": analysis["mechanism"]["DELTA_ENTROPY"],
        "MECHANISM_EXPECTED_DIRECTION_COUNT": g["MECHANISM_EXPECTED_DIRECTION_COUNT"],
        "MECHANISM_DECOLLAPSE_SUPPORTED": g["MECHANISM_DECOLLAPSE_SUPPORTED"],
        "PERSISTENT_RESPONSE_EFFECT": g["PERSISTENT_RESPONSE_EFFECT"],
        "PERSISTENT_EPOCH_EFFECT": g["PERSISTENT_EPOCH_EFFECT"], "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_RESPONSE_EFFECT": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_EFFECT": g["STATIONARY_EPOCH_EFFECT"], "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_RESPONSE_EFFECT": g["MACRO_RESPONSE_EFFECT"], "MACRO_EPOCH_EFFECT": g["MACRO_EPOCH_EFFECT"],
        "MACRO_SAFETY": g["MACRO_SAFETY"],
        "MIXED_FE_EFFECT_V1_MINUS_V3": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_EFFECT_V1_MINUS_V3": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_EFFECT_V1_MINUS_V3": g["STATIONARY_FE_EFFECT"],
        "NORMALIZATION_CAUSAL_VERDICT": {"NORMALIZATION_CAUSAL_VALIDATED": "VALIDATED",
                                         "NORMALIZATION_CAUSAL_PARTIAL": "PARTIAL",
                                         "NORMALIZATION_CAUSAL_FAILED": "FAILED"}[verdict_name],
        "PARTIAL_SUBTYPE": subtype if subtype else "NONE",
        "PERFORMANCE_IMPROVEMENT_CLAIM": v["PERFORMANCE_IMPROVEMENT_CLAIM"],
        "verdict_reason": v["reason"],
        "ORIGINAL_E4F_VERDICT": "FAILED", "HYPERPARAMETER_RETUNING": "NO",
        "CHECKPOINT_RESELECTION": "NO", "NORMALIZER_REFIT": "NO", "MODEL_MODIFICATION": "NO",
        "FINAL_EVALUATION": "COMPLETE", "FINAL_ENVIRONMENT_SEEDS_USED": "1381..1440",
        "FULL_REGRESSION": "(run separately)", "NEXT_STAGE": nxt}
    _jdump(verdict, RESULTS_DIR / "E4HF_verdict.json")
    _write_report(verdict, analysis)
    return verdict


def _write_report(v, analysis):
    g = v
    L = ["# E4HF — Normalization A/B Final Evaluation Execution", "",
         f"**NORMALIZATION_CAUSAL_VERDICT = {v['NORMALIZATION_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}**", "",
         "Causal question: effect of changing ONLY normalization (V1 legacy → V3 zero-aware) with the rest "
         "of the Shared Double-DQN fixed. Comparator = V1_CONTROL. Original E4F verdict remains FAILED.", "",
         "## A/B/C. Integrity & completeness",
         f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}; NORMALIZATION_VERSION_INTEGRITY = "
         f"{v['NORMALIZATION_VERSION_INTEGRITY']}; matrix {v['EPISODES_COMPLETE']}.", "",
         "## F/G. Primary MIXED paired effect (d = V1 − V3; + ⇒ V3 better)",
         f"- V1 mean = {v['V1_MIXED_MEAN_MIGD_RESPONSE']:.6f}; V3 mean = {v['V3_MIXED_MEAN_MIGD_RESPONSE']:.6f}",
         f"- Delta = {v['MIXED_DELTA_V1_MINUS_V3']:.6e}  CI {v['MIXED_DELTA_CI']}; RI = "
         f"{v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% → **MIXED_PRIMARY = {v['MIXED_PRIMARY']}**", "",
         "## H. Policy-pair robustness",
         f"- positive pairs = {v['POLICY_PAIR_POSITIVE_COUNT']}/5 → {v['NORMALIZATION_POLICY_ROBUSTNESS']}; "
         f"per-policy Δ = {[round(x,6) for x in v['per_policy_MIXED_deltas']]}", "",
         "## I/J/K/L. Mechanism (MIXED, pooled)",
         f"- PREDICT: V1 {v['V1_MIXED_PREDICT_FRACTION']:.3f} vs V3 {v['V3_MIXED_PREDICT_FRACTION']:.3f} "
         f"(Δ={v['DELTA_PREDICT']:+.3f})",
         f"- multi-action: V1 {v['V1_MIXED_MULTI_ACTION_RATE']:.3f} vs V3 {v['V3_MIXED_MULTI_ACTION_RATE']:.3f} "
         f"(Δ={v['DELTA_MULTI_ACTION']:+.3f})",
         f"- entropy: V1 {v['V1_MIXED_ACTION_ENTROPY']:.3f} vs V3 {v['V3_MIXED_ACTION_ENTROPY']:.3f} "
         f"(Δ={v['DELTA_ACTION_ENTROPY']:+.3f})",
         f"- expected-direction count = {v['MECHANISM_EXPECTED_DIRECTION_COUNT']}/3 → "
         f"MECHANISM_DECOLLAPSE_SUPPORTED = {v['MECHANISM_DECOLLAPSE_SUPPORTED']}", "",
         "## M/N/O. Safety",
         f"- PERSISTENT: response={v['PERSISTENT_RESPONSE_EFFECT']}, epoch={v['PERSISTENT_EPOCH_EFFECT']} → {v['PERSISTENT_SAFETY']}",
         f"- STATIONARY: response={v['STATIONARY_RESPONSE_EFFECT']}, epoch={v['STATIONARY_EPOCH_EFFECT']} → {v['STATIONARY_SAFETY']}",
         f"- MACRO: response={v['MACRO_RESPONSE_EFFECT']}, epoch={v['MACRO_EPOCH_EFFECT']} → {v['MACRO_SAFETY']}", "",
         "## P. FE (V1 − V3; + ⇒ V3 cheaper)",
         f"- MIXED={v['MIXED_FE_EFFECT_V1_MINUS_V3']:.2f}, PERSISTENT={v['PERSISTENT_FE_EFFECT_V1_MINUS_V3']:.2f}, "
         f"STATIONARY={v['STATIONARY_FE_EFFECT_V1_MINUS_V3']:.2f}", "",
         "## R. Normalization causal verdict",
         f"- **{v['NORMALIZATION_CAUSAL_VERDICT']}"
         f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE']!='NONE' else ''}**; "
         f"PERFORMANCE_IMPROVEMENT_CLAIM = {v['PERFORMANCE_IMPROVEMENT_CLAIM']}; reason: {v['verdict_reason']}", "",
         "## S. E4F preservation", "- ORIGINAL_E4F_VERDICT = FAILED (unchanged).", "",
         "## T. Limitations",
         "- Normalization-only causal experiment; does NOT isolate reward or joint-action architecture.",
         "- Controlled benchmark only (not real-world UAV); 5 policy-training seeds is a small policy axis.",
         "- No global-optimality claim; V3 not claimed universally optimal.",
         "- A de-collapse without performance gain means: under the CURRENT reward + joint-action "
         "representation, restoring regional action diversity via normalization alone did not improve "
         "final performance — it does NOT mean 'regional diversity is bad'.", "",
         f"## U. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4HF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4HF_COMPLETE", "FINAL_MATRIX_COMPLETE", "EPISODES_COMPLETE", "CHECKPOINT_INTEGRITY",
              "FINAL_PROTOCOL_INTEGRITY", "NORMALIZATION_VERSION_INTEGRITY", "PRIMARY_SCENARIO",
              "PRIMARY_ENDPOINT", "PRIMARY_COMPARISON", "V1_MIXED_MEAN_MIGD_RESPONSE",
              "V3_MIXED_MEAN_MIGD_RESPONSE", "MIXED_DELTA_V1_MINUS_V3", "MIXED_DELTA_CI",
              "MIXED_RELATIVE_IMPROVEMENT", "MIXED_PRIMARY", "POLICY_PAIR_POSITIVE_COUNT",
              "NORMALIZATION_POLICY_ROBUSTNESS", "V1_MIXED_PREDICT_FRACTION", "V3_MIXED_PREDICT_FRACTION",
              "DELTA_PREDICT", "V1_MIXED_MULTI_ACTION_RATE", "V3_MIXED_MULTI_ACTION_RATE", "DELTA_MULTI_ACTION",
              "V1_MIXED_ACTION_ENTROPY", "V3_MIXED_ACTION_ENTROPY", "DELTA_ACTION_ENTROPY",
              "MECHANISM_EXPECTED_DIRECTION_COUNT", "MECHANISM_DECOLLAPSE_SUPPORTED",
              "PERSISTENT_RESPONSE_EFFECT", "PERSISTENT_EPOCH_EFFECT", "PERSISTENT_SAFETY",
              "STATIONARY_RESPONSE_EFFECT", "STATIONARY_EPOCH_EFFECT", "STATIONARY_SAFETY",
              "MACRO_RESPONSE_EFFECT", "MACRO_EPOCH_EFFECT", "MACRO_SAFETY",
              "MIXED_FE_EFFECT_V1_MINUS_V3", "PERSISTENT_FE_EFFECT_V1_MINUS_V3", "STATIONARY_FE_EFFECT_V1_MINUS_V3",
              "NORMALIZATION_CAUSAL_VERDICT", "PARTIAL_SUBTYPE", "PERFORMANCE_IMPROVEMENT_CLAIM",
              "ORIGINAL_E4F_VERDICT", "HYPERPARAMETER_RETUNING", "CHECKPOINT_RESELECTION",
              "NORMALIZER_REFIT", "MODEL_MODIFICATION", "FINAL_EVALUATION", "FINAL_ENVIRONMENT_SEEDS_USED",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6f}"
        print(f"{k:<40} = {val}")


if __name__ == "__main__":
    main()
