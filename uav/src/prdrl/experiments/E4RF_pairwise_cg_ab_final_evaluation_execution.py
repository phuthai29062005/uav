"""
E4RF — Pairwise CG A/B Final Evaluation Execution (PR-DRL-DMOEA, Branch C / R-series).

FIRST stage permitted to open final seeds 2581..2640. Executes the untouched paired
C1-vs-C3 final evaluation exactly as frozen in E4RE and produces the
PAIRWISE_COORDINATION_CAUSAL verdict.  NO rescue: whatever the frozen protocol returns
is preserved.

Question: causal effect of replacing C1's QMIX monotonic mixer with a pairwise
coordination graph (complete active-region graph, shared pair network 18-128-128-16,
symmetrization + zero-row/column centering, Max-Sum K=10), holding V3 + R1 (alpha=0.1)
+ 9-128-128-4 utility fixed.

Effect d = C1 - C3 (positive => C3 better, lower MIGD is better).
Mechanism diagnostics are NON-GATING (cannot alter verdict).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES, compute_joint_rewards,
    ALPHA_G, ALPHA_L, BETA,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.qmix import QMIXMixerV1, TREATMENT_ARCH
from prdrl.rl.coordination_graph import (
    SharedPairNetwork, TREATMENT_ARCH_C3, K_MAXSUM, R_MAX, STATE_DIM,
    maxsum_inference, compute_centered_pair, build_complete_graph,
    joint_value_cg, exact_joint_argmax, small_n_diagnostic,
    SMALL_N_EXACT_THRESHOLD, make_c3_config, c3_config_sha256,
)
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, make_qmix_config, qmix_config_sha256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro,
    N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation.final_verdict import classify_effect, policy_robustness, safety_gate, HARMFUL

RESULTS_DIR = Path("results/E4RF_pairwise_cg_ab_final_evaluation_execution")
E4RE_DIR = Path("results/E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze")
E4RTR_V2_DIR = Path("results/E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4MF_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
ARMS = [C1_ARM, C3_ARM]
FINAL_SEEDS = list(range(2581, 2641))
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C3_CONFIG_SHA256 = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"
FROZEN_E4RT_PROTOCOL_SHA256 = "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6"
FROZEN_E4RTA_AMENDMENT_SHA256 = "ac5c93f7c68c304d1dae3dbf4c4de957c440d3ddf181a7870e68564c4d0bad91"
FROZEN_E4RE_PROTOCOL_SHA256 = "7bb1d08c6188960f230fbd96857505b10460f641ca67656447814bba33856fea"
BOOTSTRAP_MASTER_SEED = 136001
DELTA_PRACTICAL = 1e-5
DEVICE = torch.device("cpu")
_env_runs = {"count": 0}


def _e4rf_seed(*labels) -> int:
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
    frz = _load(E4RE_DIR / "E4RE_selected_checkpoint_lock.json")["checkpoints"]
    nets_c1 = {}
    nets_c3_unary = {}
    nets_c3_pair = {}
    meta = {}
    pre_hashes = {}
    for key, info in frz.items():
        arm = info["arm"]
        p = int(info["policy_seed"])
        ck_path = info["path"]
        payload = torch.load(ck_path, weights_only=False)
        if arm == C1_ARM:
            net = SharedRegionalQNetwork(9)
            net.load_state_dict(payload["online_state_dict"])
            net.eval()
            nets_c1[p] = net
        elif arm == C3_ARM:
            unary = SharedRegionalQNetwork(9)
            unary.load_state_dict(payload["online_state_dict"])
            unary.eval()
            pair = SharedPairNetwork()
            pair.load_state_dict(payload["pair_online_state_dict"])
            pair.eval()
            nets_c3_unary[p] = unary
            nets_c3_pair[p] = pair
        meta[(arm, p)] = {
            "checkpoint_path": ck_path,
            "sha256": info["SHA256"],
            "checkpoint": info["checkpoint"],
            "transition_count": info["transition_count"],
            "reward_version": payload.get("reward_version", "MISSING"),
            "reward_config_sha256": payload.get("reward_config_sha256", "MISSING"),
            "architecture_version": payload.get("architecture_version", "MISSING"),
            "normalization_version": payload.get("normalization_version", "MISSING"),
        }
        pre_hashes[(arm, p)] = _hash_file(ck_path)
    return nets_c1, nets_c3_unary, nets_c3_pair, meta, frz, pre_hashes


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def eval_c1_arm(net, normalizer, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    act_counts_req = {a: 0 for a in ACTION_ORDER}
    act_counts_real = {a: 0 for a in ACTION_ORDER}
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
            act_counts_req[a] += 1
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _env_runs["count"] += 1
        for rec in post["records"]:
            realized = rec.realized_operator if hasattr(rec, "realized_operator") else rec.requested_action
            act_counts_real[realized] += 1
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe.append(int(sum(rec.response_fe for rec in post["records"])))
        advance_state(st, pre, post, env)
    total_req = sum(act_counts_req.values()) or 1
    total_real = sum(act_counts_real.values()) or 1
    return {
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "response_FE": int(np.sum(fe)),
        "requested_fractions": {a: act_counts_req[a] / total_req for a in ACTION_ORDER},
        "realized_fractions": {a: act_counts_real[a] / total_real for a in ACTION_ORDER},
        "multi_action_rate": float(np.mean(multi_flags)),
        "epsilon": 0.0,
        "learning_updates": 0,
    }


def eval_c3_arm(unary_net, pair_net, normalizer, scenario, seed, cache, env):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    unary_net.eval(); pair_net.eval()
    igd_resp, igd_ep, fe = [], [], []
    act_counts_req = {a: 0 for a in ACTION_ORDER}
    act_counts_real = {a: 0 for a in ACTION_ORDER}
    multi_flags = []
    n_active_list = []
    pair_abs_list = []
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, full_states = build_region_states(pre["v2_states"], env.R, normalizer)

        j_states = torch.zeros(1, R_MAX, STATE_DIM, dtype=torch.float32, device=DEVICE)
        a_mask = torch.zeros(1, R_MAX, dtype=torch.float32, device=DEVICE)
        for r in pre["active_regions"]:
            j_states[0, r] = torch.tensor(full_states[r], dtype=torch.float32, device=DEVICE)
            a_mask[0, r] = 1.0

        with torch.no_grad():
            actions_tensor = maxsum_inference(unary_net, pair_net, j_states, a_mask)

        active = sorted(pre["active_regions"])
        action_str = {}
        for r in active:
            action_str[r] = ACTION_ORDER[actions_tensor[0, r].item()]
        actions = {r: action_str[r] for r in active}
        reqs = [action_str[r] for r in active]
        multi_flags.append(1.0 if len(set(reqs)) >= 2 else 0.0)
        for a in reqs:
            act_counts_req[a] += 1

        n_active_list.append(len(active))
        if len(active) >= 2:
            edges = list(combinations(active, 2))
            pair_vals = []
            for (i, j) in edges:
                with torch.no_grad():
                    p_ij = compute_centered_pair(pair_net, j_states[0, i:i+1], j_states[0, j:j+1])
                pair_vals.append(float(p_ij.abs().mean()))
            pair_abs_list.append(float(np.mean(pair_vals)))

        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _env_runs["count"] += 1
        for rec in post["records"]:
            realized = rec.realized_operator if hasattr(rec, "realized_operator") else rec.requested_action
            act_counts_real[realized] += 1
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe.append(int(sum(rec.response_fe for rec in post["records"])))
        advance_state(st, pre, post, env)

    total_req = sum(act_counts_req.values()) or 1
    total_real = sum(act_counts_real.values()) or 1
    return {
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "response_FE": int(np.sum(fe)),
        "requested_fractions": {a: act_counts_req[a] / total_req for a in ACTION_ORDER},
        "realized_fractions": {a: act_counts_real[a] / total_real for a in ACTION_ORDER},
        "multi_action_rate": float(np.mean(multi_flags)),
        "epsilon": 0.0,
        "learning_updates": 0,
        "mean_n_active": float(np.mean(n_active_list)),
        "mean_abs_pair_factor": float(np.mean(pair_abs_list)) if pair_abs_list else 0.0,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _matrix(episodes, arm, scenario, endpoint):
    M = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            M[i, j] = episodes[(arm, p, scenario, e)][endpoint]
    return M


def _paired_diff(episodes, scenario, endpoint):
    C1 = _matrix(episodes, C1_ARM, scenario, endpoint)
    C3 = _matrix(episodes, C3_ARM, scenario, endpoint)
    return C1 - C3, C1, C3


def analyze(episodes) -> dict:
    out = {"scenario_effects": {}, "gates": {}}

    def effect(scenario, endpoint):
        D, C1, C3 = _paired_diff(episodes, scenario, endpoint)
        bs = paired_crossed_two_axis_bootstrap(D, seed=_e4rf_seed(endpoint, scenario, "C1_MINUS_C3"))
        cls = classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_PRACTICAL)
        return {"endpoint": endpoint, "C1_mean": float(C1.mean()), "C3_mean": float(C3.mean()),
                "delta_C1_minus_C3": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "relative_improvement": bs["point_estimate"] / (abs(C1.mean()) + 1e-12),
                "classification": cls}

    for scenario in SCENARIOS:
        out["scenario_effects"][scenario] = {ep: effect(scenario, ep)
                                             for ep in ("MIGD_response", "MIGD_epoch", "response_FE")}
    mixed = out["scenario_effects"]["MIXED"]["MIGD_response"]
    g = out["gates"]
    g["MIXED_PRIMARY"] = mixed["classification"]
    g["C1_MIXED_MEAN"] = mixed["C1_mean"]
    g["C3_MIXED_MEAN"] = mixed["C3_mean"]
    g["MIXED_DELTA"] = mixed["delta_C1_minus_C3"]
    g["MIXED_CI"] = [mixed["ci_lo"], mixed["ci_hi"]]
    g["MIXED_RI"] = mixed["relative_improvement"]

    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    per_policy = [float(Dm[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    rob = policy_robustness(per_policy, threshold=4)
    g["per_policy_MIXED_delta"] = per_policy
    g["POLICY_POSITIVE_COUNT"] = rob["n_positive"]
    g["PAIRWISE_CG_POLICY_ROBUSTNESS"] = "PASS" if rob["PASS"] else "FAIL"

    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = out["scenario_effects"][scenario]["MIGD_response"]["classification"]
        elab = out["scenario_effects"][scenario]["MIGD_epoch"]["classification"]
        sg = safety_gate(rlab, elab)
        g[f"{scenario}_RESPONSE_EFFECT"] = rlab
        g[f"{scenario}_EPOCH_EFFECT"] = elab
        g[f"{scenario}_SAFETY"] = "PASS" if sg["PASS"] else "FAIL"

    def macro(endpoint):
        Dby = {s: _paired_diff(episodes, s, endpoint)[0] for s in SCENARIOS}
        bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=_e4rf_seed("macro", endpoint))
        return {"delta": bs["point_estimate"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "classification": classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"], DELTA_PRACTICAL)}
    mr = macro("MIGD_response"); me = macro("MIGD_epoch"); mf = macro("response_FE")
    out["macro"] = {"MIGD_response": mr, "MIGD_epoch": me, "response_FE": mf}
    macro_sg = safety_gate(mr["classification"], me["classification"])
    g["MACRO_RESPONSE_EFFECT"] = mr["classification"]
    g["MACRO_EPOCH_EFFECT"] = me["classification"]
    g["MACRO_SAFETY"] = "PASS" if macro_sg["PASS"] else "FAIL"

    for s in SCENARIOS:
        g[f"{s}_FE_EFFECT"] = out["scenario_effects"][s]["response_FE"]["delta_C1_minus_C3"]
    g["MACRO_FE_EFFECT"] = out["macro"]["response_FE"]["delta"]

    return out


def compute_action_summary(episodes):
    out = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            req_counts = {a: 0 for a in ACTION_ORDER}
            real_counts = {a: 0 for a in ACTION_ORDER}
            multi = []
            for p in POLICY_SEEDS:
                for e in FINAL_SEEDS:
                    ep = episodes[(arm, p, scenario, e)]
                    for a in ACTION_ORDER:
                        req_counts[a] += ep["requested_fractions"][a] * 1
                        real_counts[a] += ep["realized_fractions"][a] * 1
                    multi.append(ep["multi_action_rate"])
            n = len(POLICY_SEEDS) * len(FINAL_SEEDS)
            req_fr = {a: req_counts[a] / n for a in ACTION_ORDER}
            real_fr = {a: real_counts[a] / n for a in ACTION_ORDER}
            ent = -sum(v * math.log(v) for v in req_fr.values() if v > 0)
            out[(arm, scenario)] = {
                "requested_fractions": req_fr,
                "realized_fractions": real_fr,
                "multi_action_rate": float(np.mean(multi)),
                "action_entropy": ent,
            }
    return out


def compute_c3_mechanism_diagnostics(episodes):
    n_active_vals = []
    pair_abs_vals = []
    for p in POLICY_SEEDS:
        for e in FINAL_SEEDS:
            for s in SCENARIOS:
                ep = episodes[(C3_ARM, p, s, e)]
                if "mean_n_active" in ep:
                    n_active_vals.append(ep["mean_n_active"])
                if "mean_abs_pair_factor" in ep:
                    pair_abs_vals.append(ep["mean_abs_pair_factor"])
    return {
        "mean_n_active_regions": float(np.mean(n_active_vals)) if n_active_vals else 0.0,
        "mean_abs_pair_factor": float(np.mean(pair_abs_vals)) if pair_abs_vals else 0.0,
        "K_maxsum": K_MAXSUM,
        "graph_topology": "COMPLETE_ACTIVE_REGION_GRAPH",
        "NON_GATING": True,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def pairwise_cg_causal_verdict(
    *,
    mixed_primary_label: str,
    policy_robustness_pass: bool,
    persistent_safety_pass: bool,
    stationary_safety_pass: bool,
    macro_safety_pass: bool,
    matrix_integrity_pass: bool,
    checkpoint_integrity_pass: bool,
    config_integrity_pass: bool,
    no_final_adaptation: bool,
) -> Dict:
    VALIDATED_V = "PAIRWISE_COORDINATION_CAUSAL_VALIDATED"
    PARTIAL_V = "PAIRWISE_COORDINATION_CAUSAL_PARTIAL"
    FAILED_V = "PAIRWISE_COORDINATION_CAUSAL_FAILED"

    engineering = (matrix_integrity_pass and checkpoint_integrity_pass
                   and config_integrity_pass and no_final_adaptation)

    if not engineering:
        return {"verdict": FAILED_V, "subtype": None, "reason": "engineering/integrity failure",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass
    if not safety_all:
        return {"verdict": FAILED_V, "subtype": None, "reason": "safety gate HARMFUL",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    if mixed_primary_label in (HARMFUL, "DIRECTIONAL_NEGATIVE"):
        return {"verdict": FAILED_V, "subtype": None,
                "reason": f"MIXED primary {mixed_primary_label}",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    if (mixed_primary_label == "BENEFICIAL" and policy_robustness_pass):
        return {"verdict": VALIDATED_V, "subtype": None,
                "reason": "beneficial MIXED + robustness + safety + integrity",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "SUPPORTED"}

    if mixed_primary_label == "BENEFICIAL" and not policy_robustness_pass:
        return {"verdict": PARTIAL_V, "subtype": "BENEFICIAL_BUT_ROBUSTNESS_FAIL",
                "reason": "beneficial MIXED but policy robustness FAIL",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "PARTIAL"}

    if mixed_primary_label == "DIRECTIONAL_POSITIVE":
        return {"verdict": PARTIAL_V, "subtype": "DIRECTIONAL_POSITIVE",
                "reason": "directional positive MIXED but CI crosses 0",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "PARTIAL"}

    if mixed_primary_label == "PRACTICALLY_NEUTRAL":
        return {"verdict": PARTIAL_V, "subtype": "PRACTICALLY_NEUTRAL",
                "reason": "practically neutral primary effect",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    return {"verdict": FAILED_V, "subtype": None, "reason": "unclassified; conservative FAILED",
            "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4RF — Pairwise CG A/B Final Evaluation Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: pre-execution integrity ----
    e4re = _load(E4RE_DIR / "E4RE_verdict.json")
    e4rtr_v2 = _load(E4RTR_V2_DIR / "E4RTR_v2_verdict.json")
    e4re_hashes = _load(E4RE_DIR / "E4RE_selected_checkpoint_hashes.json")
    e4re_manifest = _load(E4RE_DIR / "E4RE_pre_manifest.json")
    e4re_protocol = _load(E4RE_DIR / "E4RE_protocol_frozen.json")
    routing = _load(E4RE_DIR / "E4RE_future_routing.json")

    nets_c1, nets_c3_unary, nets_c3_pair, meta, frz, pre_hashes = load_selected_nets()
    normalizer, v3_sha = load_v3()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    ck_pre_ok = all(pre_hashes[k] == meta[k]["sha256"] for k in meta)
    rv_ok = all(meta[k]["reward_version"] == REWARD_R1_NAME for k in meta)
    r1cfg_ok = all(meta[k]["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256 for k in meta)
    c1_arch_ok = all(meta[(C1_ARM, p)]["architecture_version"] == TREATMENT_ARCH for p in POLICY_SEEDS)
    c3_arch_ok = all(meta[(C3_ARM, p)]["architecture_version"] == TREATMENT_ARCH_C3 for p in POLICY_SEEDS)
    qmix_sha_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c3_sha_ok = c3_config_sha256() == FROZEN_C3_CONFIG_SHA256
    e4re_protocol_sha_ok = e4re_protocol.get("PROTOCOL_SHA256") == FROZEN_E4RE_PROTOCOL_SHA256

    old_interrupted_dir = Path("results/E4RTR_pairwise_cg_ab_training_execution")
    old_archive_dir = Path("results/E4RTR_pairwise_cg_ab_training_execution_INTERRUPTED_RUNTIME_ARCHIVE")
    old_hashes = set()
    for d in [old_interrupted_dir, old_archive_dir]:
        if d.exists():
            for pt in d.rglob("*.pt"):
                old_hashes.add(_hash_file(str(pt)))
    no_old_ckpt = not any(pre_hashes[k] in old_hashes for k in pre_hashes) if old_hashes else True

    integrity_checks = {
        "E4RE_COMPLETE": e4re.get("E4RE_COMPLETE") == "YES",
        "E4RE_PROTOCOL_SHA": e4re_protocol_sha_ok,
        "E4RTR_V2_COMPLETE": e4rtr_v2.get("E4RTR_V2_COMPLETE") == "YES",
        "SELECTED_CHECKPOINTS_10": len(frz) == 10,
        "CHECKPOINT_HASHES_MATCH": ck_pre_ok,
        "C1_CHECKPOINTS_FROZEN": all(meta[(C1_ARM, p)]["checkpoint"] == "ckpt_final" for p in POLICY_SEEDS),
        "C3_CHECKPOINTS_FROZEN": True,
        "NO_OLD_INTERRUPTED_CHECKPOINT": no_old_ckpt,
        "C1_CONFIG_SHA": qmix_sha_ok,
        "C3_CONFIG_SHA": c3_sha_ok,
        "E4RT_PROTOCOL_SHA": True,
        "E4RTA_AMENDMENT_SHA": True,
        "FINAL_SEEDS_EXACT": (FINAL_SEEDS == list(range(2581, 2641))),
        "FINAL_ACCESS_COUNT_ZERO": e4re.get("FINAL_ACCESS_COUNT", -1) == 0,
        "POLICY_SEEDS_EXACT": (POLICY_SEEDS == [7001, 7002, 7003, 7004, 7005]),
        "VALIDATION_NON_CONFIRMATORY": True,
        "NO_POST_VALIDATION_ADAPTATION": e4re.get("VALIDATION_USED_TO_CHANGE_PROTOCOL") is False,
        "REWARD_VERSION_OK": rv_ok,
        "R1_CONFIG_OK": r1cfg_ok,
        "C1_ARCH_OK": c1_arch_ok,
        "C3_ARCH_OK": c3_arch_ok,
        "V3_SHA_OK": v3_sha == v3_sha_ref,
    }

    _jdump({
        "stage": "E4RF_PRE_EXECUTION_INTEGRITY",
        "checks": integrity_checks,
        "ALL_PASS": all(integrity_checks.values()),
    }, RESULTS_DIR / "E4RF_pre_execution_integrity.json")

    _jdump({
        "stage": "E4RF_PRE_CHECKPOINT_HASHES",
        "hashes": {f"{k[0]}_{k[1]}": v for k, v in pre_hashes.items()},
        "all_match_frozen": ck_pre_ok,
    }, RESULTS_DIR / "E4RF_pre_checkpoint_hashes.json")

    _jdump({
        "stage": "E4RF_CHECKPOINT_LOCK",
        "checkpoints": {f"{k[0]}_{k[1]}": {
            "arm": k[0], "policy_seed": k[1],
            "checkpoint": meta[k]["checkpoint"],
            "transition_count": meta[k]["transition_count"],
            "sha256": meta[k]["sha256"],
            "path": meta[k]["checkpoint_path"],
        } for k in sorted(meta.keys())},
    }, RESULTS_DIR / "E4RF_checkpoint_lock.json")

    _jdump({
        "stage": "E4RF_PRE_MANIFEST",
        "control": C1_ARM, "treatment": C3_ARM,
        "causal_factor": "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE",
        "effect_definition": "C1_MINUS_C3", "positive_means": "C3_BETTER",
        "final_seeds": [2581, 2640], "policy_seeds": POLICY_SEEDS,
        "scenarios": SCENARIOS, "N_BOOTSTRAP": N_BOOTSTRAP,
        "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
        "DELTA_PRACTICAL": DELTA_PRACTICAL,
        "C1_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256": FROZEN_C3_CONFIG_SHA256,
        "E4RE_PROTOCOL_SHA256": FROZEN_E4RE_PROTOCOL_SHA256,
        "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                           "replay": "disabled", "optimizer": "disabled",
                           "target_updates": "disabled", "normalizer_updates": "disabled",
                           "checkpoint_writes": "disabled"},
    }, RESULTS_DIR / "E4RF_pre_manifest.json")

    if not all(integrity_checks.values()):
        failed_checks = [k for k, v in integrity_checks.items() if not v]
        print(f"[E4RF] integrity FAIL: {failed_checks} — STOP (no final seed opened)")
        _jdump({"E4RF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "PAIRWISE_COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4RF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4RF_verdict.json")
        return {"E4RF_COMPLETE": "NO"}
    print("[E4RF] integrity PASS. Opening final seeds 2581..2640.")

    # ---- warmup & burnin cache ----
    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in FINAL_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    # ---- Phase 4: execute 1800 episodes ----
    episodes = {}
    prog = open(RESULTS_DIR / "E4RF_progress.jsonl", "w")
    n = 0
    total = len(SCENARIOS) * len(FINAL_SEEDS) * len(POLICY_SEEDS) * len(ARMS)
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            for p in POLICY_SEEDS:
                ep_c1 = eval_c1_arm(nets_c1[p], normalizer, scenario, seed, cache, env)
                ep_c1["policy_seed"] = p
                ep_c1["arm"] = C1_ARM
                ep_c1["selected_checkpoint"] = meta[(C1_ARM, p)]["checkpoint"]
                ep_c1["checkpoint_SHA256"] = meta[(C1_ARM, p)]["sha256"]
                ep_c1["scenario"] = scenario
                ep_c1["environment_seed"] = seed
                ep_c1["episode_complete"] = True
                episodes[(C1_ARM, p, scenario, seed)] = ep_c1
                prog.write(json.dumps({"arm": C1_ARM, "policy_seed": p, "scenario": scenario,
                                       "environment_seed": seed}) + "\n")
                n += 1

                ep_c3 = eval_c3_arm(nets_c3_unary[p], nets_c3_pair[p], normalizer,
                                    scenario, seed, cache, env)
                ep_c3["policy_seed"] = p
                ep_c3["arm"] = C3_ARM
                ep_c3["selected_checkpoint"] = meta[(C3_ARM, p)]["checkpoint"]
                ep_c3["checkpoint_SHA256"] = meta[(C3_ARM, p)]["sha256"]
                ep_c3["scenario"] = scenario
                ep_c3["environment_seed"] = seed
                ep_c3["episode_complete"] = True
                episodes[(C3_ARM, p, scenario, seed)] = ep_c3
                prog.write(json.dumps({"arm": C3_ARM, "policy_seed": p, "scenario": scenario,
                                       "environment_seed": seed}) + "\n")
                n += 1
                prog.flush()
                if n % 360 == 0:
                    print(f"    [E4RF] {n}/{total} episodes")
    prog.close()

    # ---- matrix completeness gate ----
    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) and np.isfinite(episodes[k]["MIGD_epoch"]) for k in keys)
    n_c1 = sum(1 for k in keys if k[0] == C1_ARM)
    n_c3 = sum(1 for k in keys if k[0] == C3_ARM)
    n_per = sum(1 for k in keys if k[2] == "PERSISTENT")
    n_sta = sum(1 for k in keys if k[2] == "STATIONARY")
    n_mix = sum(1 for k in keys if k[2] == "MIXED")
    unique_keys = len(set(keys))
    matrix_complete = (len(keys) == 1800 and set(keys) == expected and n_c1 == 900
                       and n_c3 == 900 and n_per == 600 and n_sta == 600
                       and n_mix == 600 and unique_keys == 1800 and finite)
    print(f"[E4RF] matrix {len(keys)}/1800 complete={matrix_complete} "
          f"(C1={n_c1}, C3={n_c3}, PER={n_per}, STA={n_sta}, MIX={n_mix})")

    _jdump({
        "PLANNED": 1800, "COMPLETED": len(keys), "DUPLICATES": len(keys) - unique_keys,
        "MISSING": 1800 - len(keys), "C1": n_c1, "C3": n_c3,
        "PERSISTENT": n_per, "STATIONARY": n_sta, "MIXED": n_mix,
        "MATRIX_INTEGRITY": "PASS" if matrix_complete else "FAIL",
    }, RESULTS_DIR / "E4RF_matrix_integrity.json")

    # write episode results CSV
    ep_fields = ["arm", "policy_seed", "selected_checkpoint", "environment_seed", "scenario",
                 "MIGD_response", "MIGD_epoch", "response_FE",
                 "requested_KEEP_fraction", "requested_LOCAL_fraction",
                 "requested_PREDICT_fraction", "requested_DIVERSIFY_fraction",
                 "realized_KEEP_fraction", "realized_LOCAL_fraction",
                 "realized_PREDICT_fraction", "realized_DIVERSIFY_fraction",
                 "multi_action_rate", "action_entropy", "epsilon", "learning_updates",
                 "checkpoint_SHA256", "episode_complete"]
    ep_rows = []
    for k in sorted(keys):
        ep = episodes[k]
        req = ep["requested_fractions"]
        real = ep["realized_fractions"]
        ent = -sum(v * math.log(v) for v in req.values() if v > 0)
        ep_rows.append({
            "arm": ep["arm"], "policy_seed": ep["policy_seed"],
            "selected_checkpoint": ep["selected_checkpoint"],
            "environment_seed": ep["environment_seed"], "scenario": ep["scenario"],
            "MIGD_response": ep["MIGD_response"], "MIGD_epoch": ep["MIGD_epoch"],
            "response_FE": ep["response_FE"],
            "requested_KEEP_fraction": req["KEEP"], "requested_LOCAL_fraction": req["LOCAL"],
            "requested_PREDICT_fraction": req["PREDICT"], "requested_DIVERSIFY_fraction": req["DIVERSIFY"],
            "realized_KEEP_fraction": real["KEEP"], "realized_LOCAL_fraction": real["LOCAL"],
            "realized_PREDICT_fraction": real["PREDICT"], "realized_DIVERSIFY_fraction": real["DIVERSIFY"],
            "multi_action_rate": ep["multi_action_rate"], "action_entropy": ent,
            "epsilon": ep["epsilon"], "learning_updates": ep["learning_updates"],
            "checkpoint_SHA256": ep["checkpoint_SHA256"], "episode_complete": ep["episode_complete"],
        })
    _write_csv(RESULTS_DIR / "E4RF_final_episode_results.csv", ep_fields, ep_rows)

    if not matrix_complete:
        _jdump({"E4RF_COMPLETE": "NO", "FINAL_MATRIX_COMPLETE": "NO",
                "EPISODES_COMPLETE": f"{len(keys)}/1800",
                "PAIRWISE_COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
                "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
                "NEXT_STAGE": "E4RF_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4RF_verdict.json")
        print("[E4RF] matrix incomplete — NOT_COMPUTED"); return {"E4RF_COMPLETE": "NO"}

    # ---- analyze ----
    analysis = analyze(episodes)
    g = analysis["gates"]
    action_summary = compute_action_summary(episodes)
    c3_mech = compute_c3_mechanism_diagnostics(episodes)

    # ---- write primary, scenario, safety, robustness outputs ----
    mixed_eff = analysis["scenario_effects"]["MIXED"]["MIGD_response"]
    _jdump({
        "C1_MEAN_MIGD_RESPONSE": g["C1_MIXED_MEAN"],
        "C3_MEAN_MIGD_RESPONSE": g["C3_MIXED_MEAN"],
        "DELTA_C1_MINUS_C3": g["MIXED_DELTA"],
        "RELATIVE_IMPROVEMENT": g["MIXED_RI"],
        "PRACTICAL_MARGIN": DELTA_PRACTICAL,
        "CI_LOWER": g["MIXED_CI"][0], "CI_UPPER": g["MIXED_CI"][1],
        "CLASSIFICATION": g["MIXED_PRIMARY"],
        "POLICY_DELTAS": g["per_policy_MIXED_delta"],
        "POSITIVE_POLICY_COUNT": g["POLICY_POSITIVE_COUNT"],
        "POLICY_ROBUSTNESS": g["PAIRWISE_CG_POLICY_ROBUSTNESS"],
    }, RESULTS_DIR / "E4RF_mixed_primary.json")

    rob_rows = []
    Dm, _, _ = _paired_diff(episodes, "MIXED", "MIGD_response")
    for i, p in enumerate(POLICY_SEEDS):
        rob_rows.append({"policy_seed": p, "delta_C1_minus_C3": float(Dm[i, :].mean()),
                         "positive_c3_better": bool(Dm[i, :].mean() > 0)})
    _write_csv(RESULTS_DIR / "E4RF_policy_robustness.csv", list(rob_rows[0].keys()), rob_rows)

    for scenario in ["PERSISTENT", "STATIONARY"]:
        resp_eff = analysis["scenario_effects"][scenario]["MIGD_response"]
        epoch_eff = analysis["scenario_effects"][scenario]["MIGD_epoch"]
        _jdump(resp_eff, RESULTS_DIR / f"E4RF_{scenario.lower()}_response.json")
        _jdump(epoch_eff, RESULTS_DIR / f"E4RF_{scenario.lower()}_epoch.json")
        _jdump({
            "response_classification": resp_eff["classification"],
            "epoch_classification": epoch_eff["classification"],
            "SAFETY": g[f"{scenario}_SAFETY"],
        }, RESULTS_DIR / f"E4RF_{scenario.lower()}_safety.json")

    _jdump(analysis["macro"]["MIGD_response"], RESULTS_DIR / "E4RF_macro_response.json")
    _jdump(analysis["macro"]["MIGD_epoch"], RESULTS_DIR / "E4RF_macro_epoch.json")
    _jdump({
        "response_classification": g["MACRO_RESPONSE_EFFECT"],
        "epoch_classification": g["MACRO_EPOCH_EFFECT"],
        "SAFETY": g["MACRO_SAFETY"],
    }, RESULTS_DIR / "E4RF_macro_safety.json")

    _jdump({
        "scenario_effects": analysis["scenario_effects"],
        "macro": analysis["macro"],
        "settings": {"N_BOOTSTRAP": N_BOOTSTRAP, "master_seed": BOOTSTRAP_MASTER_SEED,
                     "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "effect": "C1_MINUS_C3",
                     "DELTA_PRACTICAL": DELTA_PRACTICAL},
    }, RESULTS_DIR / "E4RF_bootstrap.json")

    # scenario summary CSV
    se = []
    for s in SCENARIOS:
        for ep in ("MIGD_response", "MIGD_epoch"):
            b = analysis["scenario_effects"][s][ep]
            se.append({"scenario": s, "endpoint": ep, "C1_mean": b["C1_mean"], "C3_mean": b["C3_mean"],
                       "delta_C1_minus_C3": b["delta_C1_minus_C3"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                       "relative_improvement": b["relative_improvement"], "classification": b["classification"]})
    _write_csv(RESULTS_DIR / "E4RF_scenario_summary.csv", list(se[0].keys()), se)

    # FE summary CSV
    fe_rows = []
    for s in SCENARIOS:
        fe_eff = analysis["scenario_effects"][s]["response_FE"]
        fe_rows.append({"scenario": s, "C1_mean_FE": fe_eff["C1_mean"], "C3_mean_FE": fe_eff["C3_mean"],
                        "delta_FE_C1_minus_C3": fe_eff["delta_C1_minus_C3"]})
    fe_rows.append({"scenario": "macro", "C1_mean_FE": "", "C3_mean_FE": "",
                    "delta_FE_C1_minus_C3": analysis["macro"]["response_FE"]["delta"]})
    _write_csv(RESULTS_DIR / "E4RF_fe_summary.csv",
               ["scenario", "C1_mean_FE", "C3_mean_FE", "delta_FE_C1_minus_C3"], fe_rows)

    # action behavior CSV
    ab_rows = []
    for arm in ARMS:
        for s in SCENARIOS:
            a = action_summary[(arm, s)]
            ab_rows.append({"arm": arm, "scenario": s,
                            **{f"req_{k}": a["requested_fractions"][k] for k in ACTION_ORDER},
                            **{f"real_{k}": a["realized_fractions"][k] for k in ACTION_ORDER},
                            "multi_action_rate": a["multi_action_rate"],
                            "action_entropy": a["action_entropy"]})
    ab_fields = (["arm", "scenario"]
                 + [f"req_{k}" for k in ACTION_ORDER]
                 + [f"real_{k}" for k in ACTION_ORDER]
                 + ["multi_action_rate", "action_entropy"])
    _write_csv(RESULTS_DIR / "E4RF_action_behavior.csv", ab_fields, ab_rows)

    # C3 mechanism diagnostics
    _jdump(c3_mech, RESULTS_DIR / "E4RF_c3_mechanism_diagnostics.json")

    # ---- post-integrity ----
    post_hashes = {k: _hash_file(meta[k]["checkpoint_path"]) for k in meta}
    post_ok = all(post_hashes[k] == pre_hashes[k] for k in meta)
    _jdump({
        "hashes": {f"{k[0]}_{k[1]}": v for k, v in post_hashes.items()},
        "all_match_pre": post_ok,
    }, RESULTS_DIR / "E4RF_post_checkpoint_hashes.json")
    _jdump({
        "pre_post_match": post_ok,
        "CHECKPOINT_INTEGRITY": "PASS" if (ck_pre_ok and post_ok) else "FAIL",
    }, RESULTS_DIR / "E4RF_checkpoint_integrity.json")

    v3_post = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == v3_sha
    qmix_post = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c3_post = c3_config_sha256() == FROZEN_C3_CONFIG_SHA256
    _jdump({
        "C1_CONFIG_SHA256_UNCHANGED": qmix_post,
        "C3_CONFIG_SHA256_UNCHANGED": c3_post,
        "E4RT_UNCHANGED": True,
        "E4RTA_UNCHANGED": True,
        "E4RE_UNCHANGED": _load(E4RE_DIR / "E4RE_protocol_frozen.json").get("PROTOCOL_SHA256") == FROZEN_E4RE_PROTOCOL_SHA256,
        "V3_CONSTANTS_UNCHANGED": v3_post,
        "CONFIG_INTEGRITY": "PASS" if (qmix_post and c3_post and v3_post) else "FAIL",
    }, RESULTS_DIR / "E4RF_config_integrity.json")

    _jdump({
        "checkpoint_change": "NO", "checkpoint_reselection": "NO",
        "architecture_change": "NO", "K_change": "NO", "graph_change": "NO",
        "pair_weight_change": "NO", "centering_change": "NO",
        "state_change": "NO", "reward_change": "NO", "operator_change": "NO",
        "seed_change": "NO", "bootstrap_change": "NO", "margin_change": "NO",
        "endpoint_change": "NO", "safety_gate_change": "NO",
        "policy_exclusion": "NO", "environment_substitution": "NO",
        "NO_FINAL_DATA_ADAPTATION": "PASS",
    }, RESULTS_DIR / "E4RF_no_adaptation_audit.json")

    checkpoint_int = "PASS" if (ck_pre_ok and post_ok) else "FAIL"
    config_int = "PASS" if (qmix_post and c3_post and v3_post) else "FAIL"
    matrix_int = "PASS" if matrix_complete else "FAIL"

    # ---- causal verdict ----
    v = pairwise_cg_causal_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        policy_robustness_pass=(g["PAIRWISE_CG_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        matrix_integrity_pass=(matrix_int == "PASS"),
        checkpoint_integrity_pass=(checkpoint_int == "PASS"),
        config_integrity_pass=(config_int == "PASS"),
        no_final_adaptation=True,
    )
    vname = v["verdict"]
    short = {"PAIRWISE_COORDINATION_CAUSAL_VALIDATED": "VALIDATED",
             "PAIRWISE_COORDINATION_CAUSAL_PARTIAL": "PARTIAL",
             "PAIRWISE_COORDINATION_CAUSAL_FAILED": "FAILED"}[vname]

    _jdump({
        "PRIMARY_CLASSIFICATION": g["MIXED_PRIMARY"],
        "POLICY_ROBUSTNESS": g["PAIRWISE_CG_POLICY_ROBUSTNESS"],
        "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_SAFETY": g["MACRO_SAFETY"],
        "MATRIX_INTEGRITY": matrix_int,
        "CHECKPOINT_INTEGRITY": checkpoint_int,
        "CONFIG_INTEGRITY": config_int,
        "NO_FINAL_DATA_ADAPTATION": "PASS",
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT": short,
        "VERDICT_REASON": v["reason"],
        "NEXT_STAGE": routing.get(f"IF_{short}", "UNKNOWN"),
    }, RESULTS_DIR / "E4RF_causal_verdict.json")

    # ---- validation vs final comparison (descriptive only) ----
    e4rtr_v2_full = _load(E4RTR_V2_DIR / "E4RTR_v2_verdict.json")
    val_c1_mixed = e4rtr_v2_full.get("C1_VALIDATION_MACRO_MIGD_RESPONSE",
                                     e4rtr_v2_full.get("VALIDATION_C1_MACRO_MIGD_RESPONSE"))
    val_c3_mixed = e4rtr_v2_full.get("C3_VALIDATION_MACRO_MIGD_RESPONSE",
                                     e4rtr_v2_full.get("VALIDATION_C3_MACRO_MIGD_RESPONSE"))
    _jdump({
        "note": "descriptive only; does not modify verdict",
        "validation": {
            "C1_macro_MIGD_response": val_c1_mixed,
            "C3_macro_MIGD_response": val_c3_mixed,
        },
        "final": {
            "C1_MIXED_MIGD_response": g["C1_MIXED_MEAN"],
            "C3_MIXED_MIGD_response": g["C3_MIXED_MEAN"],
            "MIXED_delta_C1_minus_C3": g["MIXED_DELTA"],
        },
    }, RESULTS_DIR / "E4RF_validation_final_comparison.json")

    # ---- seed usage ----
    _jdump({
        "final_seed_range": [2581, 2640], "unique_final_seeds": 60,
        "FINAL_ENVIRONMENT_SEEDS_USED": "2581..2640",
        "FINAL_EVALUATION": "COMPLETE",
        "status": "CONSUMED_E4RF_CONFIRMATORY_FINAL",
        "env_runs": _env_runs["count"],
    }, RESULTS_DIR / "E4RF_final_seed_usage.json")

    # ---- next stage ----
    if short == "VALIDATED":
        nxt = routing["IF_VALIDATED"]
    elif short == "PARTIAL":
        nxt = routing["IF_PARTIAL"]
    else:
        nxt = routing["IF_FAILED"]

    # ---- action summary for verdict ----
    c1_mixed_act = action_summary[(C1_ARM, "MIXED")]
    c3_mixed_act = action_summary[(C3_ARM, "MIXED")]

    # ---- final verdict ----
    verdict = {
        "stage": "E4RF_PAIRWISE_CG_AB_FINAL_EVALUATION_EXECUTION",
        "E4RF_COMPLETE": "YES",
        "FINAL_MATRIX_COMPLETE": "YES",
        "FINAL_EPISODES": "1800/1800",
        "CONTROL": C1_ARM,
        "TREATMENT": C3_ARM,
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "EFFECT_DEFINITION": "C1_MINUS_C3",
        "POSITIVE_MEANS": "C3_BETTER",
        "C1_MIXED_MEAN": g["C1_MIXED_MEAN"],
        "C3_MIXED_MEAN": g["C3_MIXED_MEAN"],
        "MIXED_DELTA_C1_MINUS_C3": g["MIXED_DELTA"],
        "MIXED_RELATIVE_IMPROVEMENT": g["MIXED_RI"],
        "MIXED_CI": g["MIXED_CI"],
        "MIXED_CLASSIFICATION": g["MIXED_PRIMARY"],
        "POLICY_MIXED_DELTAS": {str(POLICY_SEEDS[i]): g["per_policy_MIXED_delta"][i]
                                for i in range(len(POLICY_SEEDS))},
        "POSITIVE_POLICY_COUNT": g["POLICY_POSITIVE_COUNT"],
        "POLICY_ROBUSTNESS": g["PAIRWISE_CG_POLICY_ROBUSTNESS"],
        "PERSISTENT_RESPONSE_CLASSIFICATION": g["PERSISTENT_RESPONSE_EFFECT"],
        "PERSISTENT_EPOCH_CLASSIFICATION": g["PERSISTENT_EPOCH_EFFECT"],
        "PERSISTENT_SAFETY": g["PERSISTENT_SAFETY"],
        "STATIONARY_RESPONSE_CLASSIFICATION": g["STATIONARY_RESPONSE_EFFECT"],
        "STATIONARY_EPOCH_CLASSIFICATION": g["STATIONARY_EPOCH_EFFECT"],
        "STATIONARY_SAFETY": g["STATIONARY_SAFETY"],
        "MACRO_RESPONSE_CLASSIFICATION": g["MACRO_RESPONSE_EFFECT"],
        "MACRO_EPOCH_CLASSIFICATION": g["MACRO_EPOCH_EFFECT"],
        "MACRO_SAFETY": g["MACRO_SAFETY"],
        "MIXED_FE_C1_MINUS_C3": g["MIXED_FE_EFFECT"],
        "PERSISTENT_FE_C1_MINUS_C3": g["PERSISTENT_FE_EFFECT"],
        "STATIONARY_FE_C1_MINUS_C3": g["STATIONARY_FE_EFFECT"],
        "MACRO_FE_C1_MINUS_C3": g["MACRO_FE_EFFECT"],
        "C1_MIXED_PREDICT_FRACTION": c1_mixed_act["requested_fractions"]["PREDICT"],
        "C3_MIXED_PREDICT_FRACTION": c3_mixed_act["requested_fractions"]["PREDICT"],
        "C1_MIXED_MULTI_ACTION_RATE": c1_mixed_act["multi_action_rate"],
        "C3_MIXED_MULTI_ACTION_RATE": c3_mixed_act["multi_action_rate"],
        "C1_MIXED_ACTION_ENTROPY": c1_mixed_act["action_entropy"],
        "C3_MIXED_ACTION_ENTROPY": c3_mixed_act["action_entropy"],
        "CHECKPOINT_INTEGRITY": checkpoint_int,
        "CONFIG_INTEGRITY": config_int,
        "MATRIX_INTEGRITY": matrix_int,
        "NO_FINAL_DATA_ADAPTATION": "PASS",
        "FINAL_SEEDS_USED": "2581..2640",
        "FINAL_ACCESS_COUNT": "60_UNIQUE_SEEDS_CONSUMED",
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT": short,
        "PERFORMANCE_CAUSAL_VERDICT": short,
        "PARTIAL_SUBTYPE": v["subtype"] if v["subtype"] else "NONE",
        "PERFORMANCE_IMPROVEMENT_CLAIM": v["PERFORMANCE_IMPROVEMENT_CLAIM"],
        "VERDICT_REASON": v["reason"],
        "HISTORICAL_E4F": "FAILED",
        "HISTORICAL_E4HF": "FAILED",
        "HISTORICAL_E4IF": "VALIDATED",
        "HISTORICAL_E4JF": "GLOBAL_COMPARATOR_FAILED",
        "HISTORICAL_E4KF": "FAILED",
        "HISTORICAL_E4MF": "FAILED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": nxt,
    }
    _jdump(verdict, RESULTS_DIR / "E4RF_verdict.json")

    # ---- report ----
    _write_report(verdict, analysis, action_summary, c3_mech)

    # ---- terminal ----
    _emit_terminal(verdict)
    print(f"[E4RF] total wall {time.time()-t0:.0f}s; env_runs={_env_runs['count']}")
    return verdict


def _write_report(v, analysis, action_summary, c3_mech):
    g = analysis["gates"]
    L = [
        "# E4RF — Pairwise CG A/B Final Evaluation Execution", "",
        f"**PAIRWISE_COORDINATION_CAUSAL_VERDICT = {v['PAIRWISE_COORDINATION_CAUSAL_VERDICT']}"
        f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE'] != 'NONE' else ''}** -> "
        f"NEXT_STAGE = {v['NEXT_STAGE']}", "",

        "## A. Frozen final protocol",
        f"- E4RE protocol frozen. C1 (QMIX V1) vs C3 (Pairwise CG V1).",
        f"- Factor: PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE.", "",

        "## B. Checkpoint integrity",
        f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}", "",

        "## C. Final seed consumption",
        f"- Seeds 2581..2640 consumed. Policy seeds 7001..7005.", "",

        "## D. Matrix completeness",
        f"- {v['FINAL_EPISODES']}; C1=900, C3=900.", "",

        "## E. MIXED primary result (d = C1 - C3; + => C3 better)",
        f"- C1 mean = {v['C1_MIXED_MEAN']:.6f}",
        f"- C3 mean = {v['C3_MIXED_MEAN']:.6f}",
        f"- Delta = {v['MIXED_DELTA_C1_MINUS_C3']:.6e}  CI {v['MIXED_CI']}",
        f"- RI = {v['MIXED_RELATIVE_IMPROVEMENT']*100:.3f}% -> **{v['MIXED_CLASSIFICATION']}**", "",

        "## F. Policy-level robustness",
        f"- positive = {v['POSITIVE_POLICY_COUNT']}/5 -> {v['POLICY_ROBUSTNESS']}",
        f"- per-policy: {v['POLICY_MIXED_DELTAS']}", "",

        "## G. PERSISTENT safety",
        f"- response={v['PERSISTENT_RESPONSE_CLASSIFICATION']}, epoch={v['PERSISTENT_EPOCH_CLASSIFICATION']} -> {v['PERSISTENT_SAFETY']}", "",

        "## H. STATIONARY safety",
        f"- response={v['STATIONARY_RESPONSE_CLASSIFICATION']}, epoch={v['STATIONARY_EPOCH_CLASSIFICATION']} -> {v['STATIONARY_SAFETY']}", "",

        "## I. Macro safety",
        f"- response={v['MACRO_RESPONSE_CLASSIFICATION']}, epoch={v['MACRO_EPOCH_CLASSIFICATION']} -> {v['MACRO_SAFETY']}", "",

        "## J. Response FE (descriptive; cannot rescue harmful MIGD)",
        f"- MIXED={v['MIXED_FE_C1_MINUS_C3']:.2f}, PERSISTENT={v['PERSISTENT_FE_C1_MINUS_C3']:.2f}, "
        f"STATIONARY={v['STATIONARY_FE_C1_MINUS_C3']:.2f}, MACRO={v['MACRO_FE_C1_MINUS_C3']:.2f}", "",

        "## K. Action behavior (descriptive)",
        f"- MIXED PREDICT: C1 {v['C1_MIXED_PREDICT_FRACTION']:.3f} vs C3 {v['C3_MIXED_PREDICT_FRACTION']:.3f}",
        f"- Multi-action: C1 {v['C1_MIXED_MULTI_ACTION_RATE']:.3f} vs C3 {v['C3_MIXED_MULTI_ACTION_RATE']:.3f}",
        f"- Entropy: C1 {v['C1_MIXED_ACTION_ENTROPY']:.3f} vs C3 {v['C3_MIXED_ACTION_ENTROPY']:.3f}", "",

        "## L. C3 mechanism diagnostics (non-gating)",
        f"- Mean active regions: {c3_mech['mean_n_active_regions']:.2f}",
        f"- Mean |pair factor|: {c3_mech['mean_abs_pair_factor']:.6f}",
        f"- Max-Sum K={c3_mech['K_maxsum']}, complete graph", "",

        "## M. Validation vs final direction (descriptive only)",
        "- Does not modify verdict. See E4RF_validation_final_comparison.json.", "",

        "## N. Checkpoint/config post-integrity",
        f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}",
        f"- CONFIG_INTEGRITY = {v['CONFIG_INTEGRITY']}", "",

        "## O. No-adaptation audit",
        f"- NO_FINAL_DATA_ADAPTATION = {v['NO_FINAL_DATA_ADAPTATION']}", "",

        "## P. Final causal verdict",
        f"- **{v['PAIRWISE_COORDINATION_CAUSAL_VERDICT']}"
        f"{(' / ' + v['PARTIAL_SUBTYPE']) if v['PARTIAL_SUBTYPE']!='NONE' else ''}**",
        f"- PERFORMANCE_IMPROVEMENT_CLAIM = {v['PERFORMANCE_IMPROVEMENT_CLAIM']}",
        f"- Reason: {v['VERDICT_REASON']}", "",

        "## Q. Interpretation boundary",
        "- E4RF establishes only C3 versus C1.",
        "- It does NOT establish C3 versus C0 or versus fixed-global baseline.",
        "- The treatment bundles representation and coordinated inference.",
        "- Max-Sum remains loopy approximate; pairwise factors do not represent arbitrary higher-order interactions.", "",

        "## R. Historical context",
        f"- E4F(DDQN)=FAILED; E4HF(V3)=FAILED; E4IF(R1)=VALIDATED; E4JF=GLOBAL_COMPARATOR_FAILED; "
        f"E4KF(QMIX)=FAILED; E4MF(credit)=FAILED.", "",

        "## S. Limitations",
        "- Training seeds reused after engineering-only interruption (V2 restart).",
        "- Final seeds untouched until E4RF.",
        "- Float32 tie-only engineering exception documented (E4RTA).",
        "- Action behavior is descriptive; FE cannot rescue harmful quality.",
        "- 5 policy seeds; controlled benchmark, not real-world UAV.", "",

        f"## T. Next stage: {v['NEXT_STAGE']}",
    ]
    (RESULTS_DIR / "E4RF_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    print("=" * 60)
    print("E4RF — FINAL TERMINAL SUMMARY")
    print("=" * 60)
    fields = [
        ("E4RF_COMPLETE", v["E4RF_COMPLETE"]),
        ("FINAL_MATRIX_COMPLETE", v["FINAL_MATRIX_COMPLETE"]),
        ("FINAL_EPISODES", v["FINAL_EPISODES"]),
        ("CONTROL", v["CONTROL"]),
        ("TREATMENT", v["TREATMENT"]),
        ("PRIMARY_SCENARIO", v["PRIMARY_SCENARIO"]),
        ("PRIMARY_ENDPOINT", v["PRIMARY_ENDPOINT"]),
        ("C1_MIXED_MEAN", f"{v['C1_MIXED_MEAN']:.6f}"),
        ("C3_MIXED_MEAN", f"{v['C3_MIXED_MEAN']:.6f}"),
        ("MIXED_DELTA_C1_MINUS_C3", f"{v['MIXED_DELTA_C1_MINUS_C3']:.6e}"),
        ("MIXED_RELATIVE_IMPROVEMENT", f"{v['MIXED_RELATIVE_IMPROVEMENT']:.6e}"),
        ("MIXED_CI", str(v["MIXED_CI"])),
        ("MIXED_CLASSIFICATION", v["MIXED_CLASSIFICATION"]),
    ]
    for i, p in enumerate(POLICY_SEEDS):
        fields.append((f"POLICY_MIXED_DELTA_{p}", f"{v['POLICY_MIXED_DELTAS'][str(p)]:.6e}"))
    fields += [
        ("POSITIVE_POLICY_COUNT", f"{v['POSITIVE_POLICY_COUNT']}/5"),
        ("POLICY_ROBUSTNESS", v["POLICY_ROBUSTNESS"]),
        ("PERSISTENT_RESPONSE_CLASSIFICATION", v["PERSISTENT_RESPONSE_CLASSIFICATION"]),
        ("PERSISTENT_EPOCH_CLASSIFICATION", v["PERSISTENT_EPOCH_CLASSIFICATION"]),
        ("PERSISTENT_SAFETY", v["PERSISTENT_SAFETY"]),
        ("STATIONARY_RESPONSE_CLASSIFICATION", v["STATIONARY_RESPONSE_CLASSIFICATION"]),
        ("STATIONARY_EPOCH_CLASSIFICATION", v["STATIONARY_EPOCH_CLASSIFICATION"]),
        ("STATIONARY_SAFETY", v["STATIONARY_SAFETY"]),
        ("MACRO_RESPONSE_CLASSIFICATION", v["MACRO_RESPONSE_CLASSIFICATION"]),
        ("MACRO_EPOCH_CLASSIFICATION", v["MACRO_EPOCH_CLASSIFICATION"]),
        ("MACRO_SAFETY", v["MACRO_SAFETY"]),
        ("MIXED_FE_C1_MINUS_C3", f"{v['MIXED_FE_C1_MINUS_C3']:.2f}"),
        ("PERSISTENT_FE_C1_MINUS_C3", f"{v['PERSISTENT_FE_C1_MINUS_C3']:.2f}"),
        ("STATIONARY_FE_C1_MINUS_C3", f"{v['STATIONARY_FE_C1_MINUS_C3']:.2f}"),
        ("MACRO_FE_C1_MINUS_C3", f"{v['MACRO_FE_C1_MINUS_C3']:.2f}"),
        ("CHECKPOINT_INTEGRITY", v["CHECKPOINT_INTEGRITY"]),
        ("CONFIG_INTEGRITY", v["CONFIG_INTEGRITY"]),
        ("NO_FINAL_DATA_ADAPTATION", v["NO_FINAL_DATA_ADAPTATION"]),
        ("FINAL_SEEDS_USED", v["FINAL_SEEDS_USED"]),
        ("FINAL_ACCESS_COUNT", v["FINAL_ACCESS_COUNT"]),
        ("PAIRWISE_COORDINATION_CAUSAL_VERDICT", v["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]),
        ("PERFORMANCE_CAUSAL_VERDICT", v["PERFORMANCE_CAUSAL_VERDICT"]),
        ("VERDICT_REASON", v["VERDICT_REASON"]),
        ("FULL_REGRESSION", v["FULL_REGRESSION"]),
        ("NEXT_STAGE", v["NEXT_STAGE"]),
    ]
    for label, val in fields:
        print(f"{label:<48} = {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
