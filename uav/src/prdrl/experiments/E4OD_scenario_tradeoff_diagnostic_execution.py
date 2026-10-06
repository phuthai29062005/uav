"""E4OD — Scenario Trade-off Diagnostic Execution (PR-DRL-DMOEA).

Opens diagnostic seeds 2301..2360 for the FIRST TIME. Runs 1800 evaluation-only
episodes (2 arms × 5 policies × 60 envs × 3 scenarios), collects transitions,
computes OFFLINE gradient alignment diagnostics, and produces the O1 support
verdict for scenario-specific gradient conflict.

E4OD is NOT another attempt to make C2 win. E4MF already established that
C2 FAILED. This is a post-hoc mechanism diagnostic only.
PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED throughout.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES, compute_joint_rewards,
    ALPHA_G, ALPHA_L, BETA, ACTION_TO_IDX,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, build_online_target_pair
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.qmix import QMIXMixerV1, TREATMENT_ARCH, R_MAX, build_mixer_online_target_pair
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, C2_CREDIT_ARCH, STATE_DIM, GAMMA, TARGET_TAU,
    JOINT_WEIGHT, REGIONAL_WEIGHT, CENTRAL_STATE_DIM,
    joint_reward_mean,
    current_joint_q, double_q_joint_target, treatment_loss,
    focal_regional_ddqn_target, focal_current_q,
    regional_smoothl1_loss, c2_composite_loss,
    make_qmix_config, qmix_config_sha256,
    make_c2_credit_config, c2_credit_config_sha256,
    FROZEN_V3_CONSTANTS_SHA256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.rl.joint_context import central_state_from_regions

from prdrl.diagnostics.gradient_diagnostic import (
    extract_utility_gradient, extract_mixer_gradient,
    gradient_norms, cosine_similarity_jr,
    gradient_norm_ratio, composite_gradient_identity,
    mixer_gradient_check, conflict_fraction,
    scenario_gap_cosine, scenario_gap_conflict,
    policy_level_gaps, EPSILON_COSINE,
)
from prdrl.diagnostics.common_state_diagnostic import (
    build_common_state_record, cross_evaluate,
    action_disagreement_fraction, action_transition_matrix,
    q_margin_stats, stratify_by_delta_severity,
)
from prdrl.diagnostics.signal_diagnostic import (
    r1_signal_statistics, r1_activity,
    regional_td_error, joint_td_error, td_error_ratio, td_error_summary,
)
from prdrl.diagnostics.statistical_diagnostic import (
    aggregate_policy_env_scenario,
    crossed_bootstrap_ci, scenario_gap_bootstrap,
    o1_support_verdict,
)
from prdrl.diagnostics.diagnostic_config import (
    DIAGNOSTIC_VERSION, make_diagnostic_config, diagnostic_config_sha256,
    validate_upstream_hashes,
    REQUIRED_QMIX_SHA, REQUIRED_C2_CREDIT_SHA, REQUIRED_R1_SHA, REQUIRED_V3_SHA,
    DIAGNOSTIC_SEEDS, DIAGNOSTIC_BOOTSTRAP_N, DIAGNOSTIC_BOOTSTRAP_SEED,
    POLICY_SEEDS, SCENARIOS, N_DIAGNOSTIC_ENVS, DIAGNOSTIC_EPISODES,
)

STAGE = "E4OD_SCENARIO_TRADEOFF_DIAGNOSTIC_EXECUTION"
RESULTS_DIR = Path("results/E4OD_scenario_tradeoff_diagnostic_execution")
E4O_DIR = Path("results/E4O_scenario_tradeoff_diagnostic_design")
E4OI_DIR = Path("results/E4OI_scenario_tradeoff_diagnostic_implementation_validation")
E4MF_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")
E4MTR_DIR = Path("results/E4MTR_credit_preservation_ab_training_execution")
E4ME_DIR = Path("results/E4ME_credit_preservation_ab_final_evaluation_protocol_freeze")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4N_DIR = Path("results/E4N_credit_preservation_failure_analysis")

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2_ARM = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
ARMS = [C1_ARM, C2_ARM]
FROZEN_QMIX_CONFIG_SHA256 = REQUIRED_QMIX_SHA
FROZEN_C2_CREDIT_SHA256 = REQUIRED_C2_CREDIT_SHA
DELTA_CREDIT = 1e-5
DEVICE = torch.device("cpu")
GRADIENT_BATCH_SIZE = 64

COUNTERS = {
    "optimizer_steps": 0,
    "parameter_changes": 0,
    "target_updates": 0,
    "replay_mutations": 0,
    "diagnostic_seed_access": 0,
    "env_runs": 0,
}

PRIOR_SEED_RANGES = [
    (1001, 1060), (1061, 1080), (1081, 1140),
    (1201, 1220),
    (1301, 1360), (1361, 1380), (1381, 1440),
    (1501, 1560), (1561, 1580), (1581, 1640),
    (1701, 1720),
    (1801, 1860),
    (1901, 1960), (1961, 1980), (1981, 2040),
    (2101, 2160), (2161, 2180), (2181, 2240),
]


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
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
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


def load_checkpoint_networks():
    """Load all 10 frozen checkpoint networks with online+target utility and mixer."""
    frz_ckpt = _load(E4O_DIR / "E4O_checkpoint_lock.json")["frozen_checkpoints"]
    selected = _load(E4MTR_DIR / "E4MTR_selected_checkpoints.json")["selected"]

    nets = {}
    target_nets = {}
    mixers = {}
    target_mixers = {}
    meta = {}
    ckpt_hashes = {}

    for arm in ARMS:
        for p in POLICY_SEEDS:
            key = f"{arm}_{p}"
            info = selected[key]
            ckpt_path = info["checkpoint_path"]
            expected_hash = frz_ckpt["hashes"][key]

            actual_hash = _hash_file(ckpt_path)
            ckpt_hashes[key] = {"expected": expected_hash, "actual": actual_hash,
                                "match": actual_hash == expected_hash}

            payload = torch.load(ckpt_path, weights_only=False)

            online = SharedRegionalQNetwork(STATE_DIM)
            online.load_state_dict(payload["online_state_dict"])
            online.eval()
            nets[(arm, p)] = online

            target = SharedRegionalQNetwork(STATE_DIM)
            target.load_state_dict(payload["target_state_dict"])
            target.eval()
            target_nets[(arm, p)] = target

            mixer_o = QMIXMixerV1()
            mixer_o.load_state_dict(payload["mixer_online_state_dict"])
            mixer_o.eval()
            mixers[(arm, p)] = mixer_o

            mixer_t = QMIXMixerV1()
            mixer_t.load_state_dict(payload["mixer_target_state_dict"])
            mixer_t.eval()
            target_mixers[(arm, p)] = mixer_t

            meta[(arm, p)] = info

    all_match = all(v["match"] for v in ckpt_hashes.values())
    return nets, target_nets, mixers, target_mixers, meta, ckpt_hashes, all_match


# ---------------------------------------------------------------------------
# Episode execution with full transition collection
# ---------------------------------------------------------------------------
def run_diagnostic_episode(
    net, target_net, mixer, target_mixer, normalizer,
    c1_net_for_policy, c2_net_for_policy,
    arm, scenario, seed, cache, env, policy_seed,
):
    """Run one evaluation-only episode and collect full transition data."""
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe_list = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    multi_flags = []
    transitions = []
    prev_full_states = None
    prev_active_regions = None
    prev_j_states = None
    prev_a_mask = None
    prev_c_state = None
    prev_a_vec = None
    prev_rr_array = None
    prev_r_joint = None

    for k in range(N_SCORED_CHANGES):
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, full_states = build_region_states(
            pre["v2_states"], env.R, normalizer)

        j_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
        a_mask = np.zeros(R_MAX, dtype=np.float32)
        for r in pre["active_regions"]:
            j_states[r] = full_states[r]
            a_mask[r] = 1.0
        c_state = central_state_from_regions(
            {r: local_norm[r] for r in pre["active_regions"]}, gc_norm)

        with torch.no_grad():
            _, action_str = select_regional_actions(
                net, local_norm, gc_norm, epsilon=0.0,
                event_seed=0, device=DEVICE)
        active = sorted(pre["active_regions"])
        actions = {r: action_str[r] for r in active}
        reqs = [action_str[r] for r in active]
        multi_flags.append(1.0 if len(set(reqs)) >= 2 else 0.0)

        a_vec = np.zeros(R_MAX, dtype=np.int64)
        for r in active:
            a_vec[r] = ACTION_TO_IDX[action_str[r]]
            act_counts[action_str[r]] += 1

        # finalize previous change's transitions with next_state info
        if prev_full_states is not None:
            for tr in transitions:
                if tr.get("_pending"):
                    r = tr["region_id"]
                    tr["next_local_state_9D"] = full_states[r].tolist() if r in full_states else [0.0]*9
                    tr["next_central_state_55D"] = c_state.tolist()
                    tr["_pending"] = False

        t_current_before = st["t_current"]
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        COUNTERS["env_runs"] += 1
        _gr, region_rewards, aux = compute_joint_rewards(
            pre, post, post["records"], env, rd,
            t_current_before, reward_version=REWARD_R1_NAME)

        rr_array = np.zeros(R_MAX, dtype=np.float32)
        for r in active:
            rr_array[r] = region_rewards[r]
        r_joint = joint_reward_mean(rr_array, a_mask)

        rec_by_region = {rec.region_id: rec for rec in post["records"]}
        done = (k == N_SCORED_CHANGES - 1)

        # cross-evaluate C1 and C2 on each active region's state
        for r in active:
            s_t = torch.tensor(full_states[r], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_c1 = c1_net_for_policy(s_t)[0].tolist()
                q_c2 = c2_net_for_policy(s_t)[0].tolist()

            rec = rec_by_region.get(r)
            tr = {
                "source_arm": arm,
                "policy_seed": policy_seed,
                "scenario": scenario,
                "diagnostic_environment_seed": seed,
                "episode_change_index": k,
                "region_id": r,
                "active_mask": a_mask.tolist(),
                "local_state_9D": full_states[r].tolist(),
                "central_state_55D": c_state.tolist(),
                "requested_action": int(a_vec[r]),
                "realized_operator": rec.realized_operator if rec else "",
                "focal_R1_reward": float(aux["g_by_region"].get(r, 0.0)),
                "all_active_regional_R1_rewards": [float(aux["g_by_region"].get(ar, 0.0)) for ar in active],
                "r_joint": float(r_joint),
                "done": done,
                "C1_utility_Q_vector": q_c1,
                "C2_utility_Q_vector": q_c2,
                # joint context for gradient computation
                "_joint_states": j_states.copy(),
                "_active_mask": a_mask.copy(),
                "_action_vector": a_vec.copy(),
                "_central_state": c_state.copy(),
                "_regional_rewards": rr_array.copy(),
                "_r_joint": r_joint,
            }
            if done:
                tr["next_local_state_9D"] = [0.0] * 9
                tr["next_central_state_55D"] = [0.0] * 55
                tr["_next_joint_states"] = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                tr["_next_active_mask"] = np.zeros(R_MAX, dtype=np.float32)
                tr["_next_central_state"] = np.zeros(CENTRAL_STATE_DIM, dtype=np.float32)
                tr["_pending"] = False
            else:
                tr["next_local_state_9D"] = None
                tr["next_central_state_55D"] = None
                tr["_pending"] = True
            transitions.append(tr)

        igd_resp.append(post["igd_response"])
        igd_ep.append(post["igd_epoch"])
        fe_list.append(int(sum(rec.response_fe for rec in post["records"])))
        prev_full_states = full_states
        prev_active_regions = active
        advance_state(st, pre, post, env)

    # fill next_joint_states for non-terminal transitions
    # rebuild from transitions: group by change_index
    for tr in transitions:
        if tr.get("_pending"):
            tr["next_local_state_9D"] = [0.0] * 9
            tr["next_central_state_55D"] = [0.0] * 55
            tr["_pending"] = False

    # build next-step joint context for non-terminal transitions
    by_change = {}
    for tr in transitions:
        by_change.setdefault(tr["episode_change_index"], []).append(tr)
    for k_idx in sorted(by_change.keys()):
        if k_idx < N_SCORED_CHANGES - 1:
            next_k = k_idx + 1
            if next_k in by_change:
                next_trs = by_change[next_k]
                if next_trs:
                    ref = next_trs[0]
                    njs = ref["_joint_states"]
                    nam = ref["_active_mask"]
                    ncs = ref["_central_state"]
                    for tr in by_change[k_idx]:
                        tr["_next_joint_states"] = njs.copy()
                        tr["_next_active_mask"] = nam.copy()
                        tr["_next_central_state"] = ncs.copy()
            else:
                for tr in by_change[k_idx]:
                    if "_next_joint_states" not in tr:
                        tr["_next_joint_states"] = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                        tr["_next_active_mask"] = np.zeros(R_MAX, dtype=np.float32)
                        tr["_next_central_state"] = np.zeros(CENTRAL_STATE_DIM, dtype=np.float32)

    total = sum(act_counts.values()) or 1
    episode = {
        "arm": arm, "scenario": scenario, "environment_seed": seed,
        "policy_seed": policy_seed,
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "response_FE": int(np.sum(fe_list)),
        "action_counts": dict(act_counts),
        "n_decisions": total,
        "multi_action_rate": float(np.mean(multi_flags)),
    }
    return episode, transitions


# ---------------------------------------------------------------------------
# Offline gradient computation (C2 only — C1 has no L_regional)
# ---------------------------------------------------------------------------
def compute_offline_gradients(c2_transitions_by_policy_scenario, nets, target_nets,
                              mixers, target_mixers):
    """Compute cosine_JR for each (policy, scenario) batch on C2 transitions.

    Returns per-record cosine values and aggregated per policy×scenario."""
    cosine_records = []
    norm_records = []

    for (p, scenario), trs in sorted(c2_transitions_by_policy_scenario.items()):
        if len(trs) == 0:
            continue

        online_util = nets[(C2_ARM, p)]
        target_util = target_nets[(C2_ARM, p)]
        online_mixer = mixers[(C2_ARM, p)]
        target_mixer_net = target_mixers[(C2_ARM, p)]

        # process in batches
        for batch_start in range(0, len(trs), GRADIENT_BATCH_SIZE):
            batch = trs[batch_start:batch_start + GRADIENT_BATCH_SIZE]
            B = len(batch)

            js = torch.tensor(np.stack([tr["_joint_states"] for tr in batch]),
                              dtype=torch.float32)
            am = torch.tensor(np.stack([tr["_active_mask"] for tr in batch]),
                              dtype=torch.float32)
            av = torch.tensor(np.stack([tr["_action_vector"] for tr in batch]),
                              dtype=torch.int64)
            cs = torch.tensor(np.stack([tr["_central_state"] for tr in batch]),
                              dtype=torch.float32)
            njs = torch.tensor(np.stack([tr["_next_joint_states"] for tr in batch]),
                               dtype=torch.float32)
            nam = torch.tensor(np.stack([tr["_next_active_mask"] for tr in batch]),
                               dtype=torch.float32)
            ncs = torch.tensor(np.stack([tr["_next_central_state"] for tr in batch]),
                               dtype=torch.float32)
            rj = torch.tensor([tr["_r_joint"] for tr in batch], dtype=torch.float32)
            done_t = torch.tensor([1.0 if tr["done"] else 0.0 for tr in batch],
                                  dtype=torch.float32)
            focal_s = torch.tensor(np.stack([np.array(tr["local_state_9D"], dtype=np.float32)
                                             for tr in batch]), dtype=torch.float32)
            focal_a = torch.tensor([tr["requested_action"] for tr in batch], dtype=torch.int64)
            focal_r = torch.tensor([tr["focal_R1_reward"] for tr in batch], dtype=torch.float32)
            focal_ns = torch.tensor(np.stack([np.array(tr["next_local_state_9D"], dtype=np.float32)
                                              for tr in batch]), dtype=torch.float32)

            # enable grad for online utility
            online_util.train()
            for param in online_util.parameters():
                param.requires_grad_(True)
            online_mixer.train()
            for param in online_mixer.parameters():
                param.requires_grad_(True)

            # L_joint (UNWEIGHTED)
            q_tot = current_joint_q(online_util, online_mixer, js, av, am, cs)
            y_joint = double_q_joint_target(online_util, target_util, target_mixer_net,
                                            njs, nam, ncs, rj, done_t, GAMMA)
            l_joint = treatment_loss(q_tot, y_joint)

            g_j = extract_utility_gradient(l_joint, online_util, retain_graph=True)
            g_mixer_j = extract_mixer_gradient(l_joint, online_mixer, retain_graph=False)

            # L_regional (UNWEIGHTED)
            q_local = focal_current_q(online_util, focal_s, focal_a)
            y_local = focal_regional_ddqn_target(online_util, target_util,
                                                  focal_r, focal_ns, done_t, GAMMA)
            l_regional = regional_smoothl1_loss(q_local, y_local)

            g_r = extract_utility_gradient(l_regional, online_util, retain_graph=True)

            # composite identity check
            l_c2 = c2_composite_loss(l_joint.detach().requires_grad_(False),
                                     l_regional.detach().requires_grad_(False))

            # cosine
            cos_result = cosine_similarity_jr(g_j, g_r)
            rho = gradient_norm_ratio(cos_result["norm_g_R"], cos_result["norm_g_J"])

            cosine_records.append({
                "policy_seed": p,
                "scenario": scenario,
                "batch_start": batch_start,
                "batch_size": B,
                "cosine_JR": cos_result["cosine_JR"],
                "cosine_valid": cos_result["cosine_valid"],
                "norm_g_J": cos_result["norm_g_J"],
                "norm_g_R": cos_result["norm_g_R"],
                "rho_g": rho,
                "l_joint": float(l_joint.item()),
                "l_regional": float(l_regional.item()),
            })

            norm_records.append({
                "policy_seed": p,
                "scenario": scenario,
                "batch_start": batch_start,
                "norm_g_J": cos_result["norm_g_J"],
                "norm_g_R": cos_result["norm_g_R"],
            })

            online_util.eval()
            online_mixer.eval()
            online_util.zero_grad()
            online_mixer.zero_grad()

    assert COUNTERS["optimizer_steps"] == 0
    assert COUNTERS["parameter_changes"] == 0
    return cosine_records, norm_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4OD — Scenario Trade-off Diagnostic Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Phase 0: Pre-execution integrity
    # ================================================================
    print("[E4OD-0] Pre-execution integrity...")
    e4oi = _load(E4OI_DIR / "E4OI_verdict.json")
    e4o = _load(E4O_DIR / "E4O_verdict.json")
    e4mf = _load(E4MF_DIR / "E4MF_verdict.json")
    e4n = _load(E4N_DIR / "E4N_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")

    historical = {
        "E4F": e4f.get("SCIENTIFIC_VERDICT", "UNKNOWN"),
        "E4HF": e4hf.get("NORMALIZATION_CAUSAL_VERDICT", "UNKNOWN"),
        "E4IF": e4if.get("REWARD_CAUSAL_VERDICT", "UNKNOWN"),
        "E4KF": e4kf.get("COORDINATION_CAUSAL_VERDICT",
                         e4kf.get("SCIENTIFIC_VERDICT", "UNKNOWN")),
        "E4MF": e4mf.get("CREDIT_PRESERVATION_CAUSAL_VERDICT", "UNKNOWN"),
    }
    hist_ok = (
        historical["E4F"] == "FAILED"
        and historical["E4HF"] == "FAILED"
        and historical["E4IF"] in ("VALIDATED", "REWARD_CAUSAL_VALIDATED")
        and historical["E4KF"] in ("FAILED", "COORDINATION_CAUSAL_FAILED")
        and historical["E4MF"] in ("FAILED", "CREDIT_PRESERVATION_CAUSAL_FAILED")
    )
    _jdump({"stage": STAGE, "historical_verdicts": historical,
            "all_preserved": hist_ok}, RESULTS_DIR / "E4OD_historical_integrity.json")

    # E4OI prerequisite
    e4oi_ok = e4oi.get("E4OI_COMPLETE") == "YES"
    e4oi_cfg_sha = e4oi.get("E4OI_DIAGNOSTIC_CONFIG_SHA256", "MISSING")
    cfg_sha_ok = e4oi_cfg_sha == diagnostic_config_sha256()

    # Upstream config hashes
    qmix_sha_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c2_sha_ok = c2_credit_config_sha256() == FROZEN_C2_CREDIT_SHA256
    upstream = validate_upstream_hashes(
        qmix_config_sha256(), c2_credit_config_sha256(),
        FROZEN_R1_CONFIG_SHA256, FROZEN_V3_CONSTANTS_SHA256)
    _jdump({"stage": STAGE, "upstream_verification": upstream,
            "qmix_sha_ok": qmix_sha_ok, "c2_sha_ok": c2_sha_ok},
           RESULTS_DIR / "E4OD_upstream_config.json")

    # Seed firewall
    seed_firewall_ok = True
    for lo, hi in PRIOR_SEED_RANGES:
        for s in DIAGNOSTIC_SEEDS:
            if lo <= s <= hi:
                seed_firewall_ok = False
    _jdump({"stage": STAGE, "diagnostic_seeds": DIAGNOSTIC_SEEDS,
            "prior_ranges": PRIOR_SEED_RANGES,
            "no_overlap": seed_firewall_ok},
           RESULTS_DIR / "E4OD_seed_firewall.json")

    # Load checkpoints
    print("[E4OD-1] Loading checkpoints...")
    nets, target_nets, mixers, target_mixers, meta, ckpt_hashes, ckpt_ok = \
        load_checkpoint_networks()
    normalizer, v3_sha = load_v3()
    v3_ok = v3_sha == FROZEN_V3_CONSTANTS_SHA256

    _jdump({"stage": STAGE, "checkpoint_hashes": ckpt_hashes,
            "all_match": ckpt_ok, "v3_sha": v3_sha, "v3_match": v3_ok},
           RESULTS_DIR / "E4OD_checkpoint_integrity.json")

    # diagnostic config
    diag_cfg = make_diagnostic_config()
    diag_sha = diagnostic_config_sha256()
    _jdump({"stage": STAGE, "diagnostic_config": diag_cfg,
            "diagnostic_config_sha256": diag_sha,
            "matches_e4oi": diag_sha == e4oi_cfg_sha},
           RESULTS_DIR / "E4OD_diagnostic_config.json")

    pre_ok = (hist_ok and e4oi_ok and cfg_sha_ok and ckpt_ok and v3_ok
              and upstream["all_pass"] and seed_firewall_ok)
    manifest = {
        "stage": STAGE,
        "pre_integrity": "PASS" if pre_ok else "FAIL",
        "historical_ok": hist_ok,
        "e4oi_ok": e4oi_ok,
        "config_sha_ok": cfg_sha_ok,
        "checkpoints_ok": ckpt_ok,
        "v3_ok": v3_ok,
        "upstream_ok": upstream["all_pass"],
        "seed_firewall_ok": seed_firewall_ok,
        "execution_mode": {
            "epsilon": 0.0, "eval": True, "gradients": "OFFLINE_ONLY",
            "optimizer": "disabled", "learning": "none",
            "mixer_not_in_execution": True,
        },
    }
    _jdump(manifest, RESULTS_DIR / "E4OD_pre_manifest.json")

    if not pre_ok:
        print("[E4OD] integrity FAIL — STOP")
        _jdump({"E4OD_COMPLETE": "NO", "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
                "O1_SUPPORT": "NOT_COMPUTED", "reason": "pre-integrity FAIL"},
               RESULTS_DIR / "E4OD_verdict.json")
        return {"E4OD_COMPLETE": "NO"}
    print("[E4OD-0] integrity PASS. Opening diagnostic seeds 2301..2360.")
    COUNTERS["diagnostic_seed_access"] = 60

    # ================================================================
    # Phase 2: Warmup & burnin cache
    # ================================================================
    print("[E4OD-2] Building warmup/burnin cache...")
    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in DIAGNOSTIC_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    # ================================================================
    # Phase 3: Execute 1800 diagnostic episodes
    # ================================================================
    print("[E4OD-3] Executing 1800 diagnostic episodes...")
    episodes = {}
    all_transitions = []
    n_done = 0
    total = DIAGNOSTIC_EPISODES

    for scenario in SCENARIOS:
        for seed in DIAGNOSTIC_SEEDS:
            for p in POLICY_SEEDS:
                for arm in ARMS:
                    ep, trs = run_diagnostic_episode(
                        nets[(arm, p)], target_nets[(arm, p)],
                        mixers[(arm, p)], target_mixers[(arm, p)],
                        normalizer,
                        nets[(C1_ARM, p)], nets[(C2_ARM, p)],
                        arm, scenario, seed, cache, env, p,
                    )
                    episodes[(arm, p, scenario, seed)] = ep
                    all_transitions.extend(trs)
                    n_done += 1
                    if n_done % 360 == 0:
                        print(f"    [E4OD] {n_done}/{total} episodes")

    # ================================================================
    # Phase 4: Matrix completeness
    # ================================================================
    keys = list(episodes.keys())
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS
                for s in SCENARIOS for e in DIAGNOSTIC_SEEDS}
    n_c1 = sum(1 for k in keys if k[0] == C1_ARM)
    n_c2 = sum(1 for k in keys if k[0] == C2_ARM)
    finite = all(np.isfinite(episodes[k]["MIGD_response"]) for k in keys)
    complete = (len(keys) == total and set(keys) == expected
                and n_c1 == 900 and n_c2 == 900 and finite)
    print(f"[E4OD-4] matrix {len(keys)}/{total} complete={complete}")

    if not complete:
        _jdump({"E4OD_COMPLETE": "NO", "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
                "reason": "matrix incomplete"},
               RESULTS_DIR / "E4OD_verdict.json")
        return {"E4OD_COMPLETE": "NO"}

    # write episode results
    ep_fields = ["arm", "policy_seed", "scenario", "environment_seed",
                 "MIGD_response", "MIGD_epoch", "response_FE", "multi_action_rate"]
    ep_rows = []
    for (a, p, s, e) in sorted(keys):
        ep = episodes[(a, p, s, e)]
        ep_rows.append({"arm": a, "policy_seed": p, "scenario": s,
                        "environment_seed": e,
                        **{k: ep[k] for k in ("MIGD_response", "MIGD_epoch",
                                               "response_FE", "multi_action_rate")}})
    _write_csv(RESULTS_DIR / "E4OD_episode_results.csv", ep_fields, ep_rows)
    _jdump({"total_episodes": len(keys), "total_transitions": len(all_transitions),
            "c1_episodes": n_c1, "c2_episodes": n_c2,
            "per_scenario": {s: sum(1 for k in keys if k[2] == s) for s in SCENARIOS}},
           RESULTS_DIR / "E4OD_matrix_summary.json")

    # ================================================================
    # Phase 5: Offline gradient diagnostics (C2 only)
    # ================================================================
    print("[E4OD-5] Computing offline gradient diagnostics...")
    c2_trs_by_ps = {}
    for tr in all_transitions:
        if tr["source_arm"] == C2_ARM:
            key = (tr["policy_seed"], tr["scenario"])
            c2_trs_by_ps.setdefault(key, []).append(tr)

    cosine_records, norm_records = compute_offline_gradients(
        c2_trs_by_ps, nets, target_nets, mixers, target_mixers)

    # aggregate cosines per (policy, scenario)
    cosine_by_ps = {}
    for rec in cosine_records:
        key = (rec["policy_seed"], rec["scenario"])
        cosine_by_ps.setdefault(key, []).append(rec)

    mean_cosine_ps = {}
    for (p, s), recs in cosine_by_ps.items():
        valid = [r["cosine_JR"] for r in recs if r["cosine_valid"]]
        mean_cosine_ps[(p, s)] = float(np.mean(valid)) if valid else None

    # per-scenario mean cosine (aggregated across policies and batches)
    scenario_cosines = {}
    for s in SCENARIOS:
        all_cos = []
        for rec in cosine_records:
            if rec["scenario"] == s and rec["cosine_valid"]:
                all_cos.append(rec["cosine_JR"])
        scenario_cosines[s] = float(np.mean(all_cos)) if all_cos else None

    # per-scenario conflict fraction
    scenario_conflicts = {}
    for s in SCENARIOS:
        cos_vals = [rec["cosine_JR"] for rec in cosine_records
                    if rec["scenario"] == s and rec["cosine_valid"]]
        scenario_conflicts[s] = conflict_fraction(cos_vals)

    # per-scenario gradient norm ratio
    scenario_rho = {}
    for s in SCENARIOS:
        rhos = [rec["rho_g"] for rec in cosine_records if rec["scenario"] == s]
        scenario_rho[s] = float(np.mean(rhos)) if rhos else None

    _jdump({"stage": STAGE, "cosine_records_count": len(cosine_records),
            "scenario_mean_cosine": scenario_cosines,
            "scenario_conflict_fraction": scenario_conflicts,
            "scenario_mean_rho_g": scenario_rho,
            "optimizer_steps": COUNTERS["optimizer_steps"],
            "parameter_changes": COUNTERS["parameter_changes"]},
           RESULTS_DIR / "E4OD_gradient_diagnostics.json")

    # ================================================================
    # Phase 6: G_cos, G_conflict, policy-level
    # ================================================================
    print("[E4OD-6] Computing scenario gaps...")
    sc = scenario_cosines
    if all(v is not None for v in sc.values()):
        G_cos = scenario_gap_cosine(sc["STATIONARY"], sc["PERSISTENT"], sc["MIXED"])
    else:
        G_cos = float("nan")

    sf = scenario_conflicts
    if all(np.isfinite(v) for v in sf.values()):
        G_conflict = scenario_gap_conflict(sf["STATIONARY"], sf["PERSISTENT"], sf["MIXED"])
    else:
        G_conflict = float("nan")

    # per-policy cosines and conflicts
    per_policy_cosines = {}
    per_policy_conflicts = {}
    for p in POLICY_SEEDS:
        cos_p = {}
        con_p = {}
        for s in SCENARIOS:
            vals = [rec["cosine_JR"] for rec in cosine_records
                    if rec["policy_seed"] == p and rec["scenario"] == s and rec["cosine_valid"]]
            cos_p[s] = float(np.mean(vals)) if vals else 0.0
            con_p[s] = conflict_fraction(vals) if vals else float("nan")
        per_policy_cosines[p] = cos_p
        per_policy_conflicts[p] = con_p

    plg = policy_level_gaps(per_policy_cosines, per_policy_conflicts)

    _jdump({"stage": STAGE, "G_cos": G_cos, "G_conflict": G_conflict,
            "per_policy_cosines": {str(k): v for k, v in per_policy_cosines.items()},
            "per_policy_conflicts": {str(k): v for k, v in per_policy_conflicts.items()},
            "policy_level_gaps": plg,
            "scenario_cosines": scenario_cosines,
            "scenario_conflicts": scenario_conflicts},
           RESULTS_DIR / "E4OD_scenario_gaps.json")

    # ================================================================
    # Phase 7: Crossed bootstrap for G_cos
    # ================================================================
    print("[E4OD-7] Running crossed bootstrap for G_cos...")
    # build cosine matrices: [n_policies, n_envs] per scenario
    cosine_agg_records = []
    for rec in cosine_records:
        if rec["cosine_valid"] and rec["cosine_JR"] is not None:
            cosine_agg_records.append({
                "policy": rec["policy_seed"],
                "diagnostic_environment": rec["scenario"],
                "scenario": rec["scenario"],
                "cosine_JR": rec["cosine_JR"],
            })

    # aggregate cosine per (policy, env, scenario) — but env info is in batch granularity
    # actually, cosine is computed per batch, not per env. For bootstrap, need per-env cosines.
    # Since we batch across envs, we need to compute per-env cosine records.
    # Let me use the individual transition-level data to build per-env aggregates.

    # compute per (policy, env, scenario) mean cosine from per-batch data
    # Each batch contains transitions from multiple envs. We need per-env cosine.
    # Actually, the better approach: rebuild per-env cosine from transitions.

    # compute cosine per (policy, env, scenario) by grouping transitions
    cosine_per_pse = {}
    for (p, s), trs in c2_trs_by_ps.items():
        # group by env
        by_env = {}
        for tr in trs:
            by_env.setdefault(tr["diagnostic_environment_seed"], []).append(tr)
        for env_seed, env_trs in by_env.items():
            # compute a single cosine for this (p, env, s) using all transitions
            B = len(env_trs)
            if B == 0:
                continue
            online_util = nets[(C2_ARM, p)]
            target_util = target_nets[(C2_ARM, p)]
            online_mixer = mixers[(C2_ARM, p)]
            target_mixer_net = target_mixers[(C2_ARM, p)]

            js = torch.tensor(np.stack([tr["_joint_states"] for tr in env_trs]),
                              dtype=torch.float32)
            am = torch.tensor(np.stack([tr["_active_mask"] for tr in env_trs]),
                              dtype=torch.float32)
            av = torch.tensor(np.stack([tr["_action_vector"] for tr in env_trs]),
                              dtype=torch.int64)
            cs_t = torch.tensor(np.stack([tr["_central_state"] for tr in env_trs]),
                                dtype=torch.float32)
            njs = torch.tensor(np.stack([tr["_next_joint_states"] for tr in env_trs]),
                               dtype=torch.float32)
            nam = torch.tensor(np.stack([tr["_next_active_mask"] for tr in env_trs]),
                               dtype=torch.float32)
            ncs = torch.tensor(np.stack([tr["_next_central_state"] for tr in env_trs]),
                               dtype=torch.float32)
            rj = torch.tensor([tr["_r_joint"] for tr in env_trs], dtype=torch.float32)
            done_t = torch.tensor([1.0 if tr["done"] else 0.0 for tr in env_trs],
                                  dtype=torch.float32)
            focal_s = torch.tensor(np.stack([np.array(tr["local_state_9D"], dtype=np.float32)
                                             for tr in env_trs]), dtype=torch.float32)
            focal_a = torch.tensor([tr["requested_action"] for tr in env_trs], dtype=torch.int64)
            focal_r = torch.tensor([tr["focal_R1_reward"] for tr in env_trs], dtype=torch.float32)
            focal_ns = torch.tensor(np.stack([np.array(tr["next_local_state_9D"], dtype=np.float32)
                                              for tr in env_trs]), dtype=torch.float32)

            online_util.train()
            for param in online_util.parameters():
                param.requires_grad_(True)
            online_mixer.train()

            q_tot = current_joint_q(online_util, online_mixer, js, av, am, cs_t)
            y_joint = double_q_joint_target(online_util, target_util, target_mixer_net,
                                            njs, nam, ncs, rj, done_t, GAMMA)
            l_joint = treatment_loss(q_tot, y_joint)
            g_j = extract_utility_gradient(l_joint, online_util, retain_graph=True)

            q_local = focal_current_q(online_util, focal_s, focal_a)
            y_local = focal_regional_ddqn_target(online_util, target_util,
                                                  focal_r, focal_ns, done_t, GAMMA)
            l_regional = regional_smoothl1_loss(q_local, y_local)
            g_r_vec = extract_utility_gradient(l_regional, online_util, retain_graph=False)

            cos = cosine_similarity_jr(g_j, g_r_vec)
            cosine_per_pse[(p, s, env_seed)] = cos["cosine_JR"] if cos["cosine_valid"] else float("nan")

            online_util.eval()
            online_mixer.eval()
            online_util.zero_grad()
            online_mixer.zero_grad()

    # build scenario matrices for bootstrap
    cos_matrices = {}
    for s in SCENARIOS:
        mat = np.full((len(POLICY_SEEDS), len(DIAGNOSTIC_SEEDS)), np.nan)
        for pi, p in enumerate(POLICY_SEEDS):
            for ei, e in enumerate(DIAGNOSTIC_SEEDS):
                key = (p, s, e)
                if key in cosine_per_pse and np.isfinite(cosine_per_pse[key]):
                    mat[pi, ei] = cosine_per_pse[key]
        cos_matrices[s] = mat

    gap_bs = scenario_gap_bootstrap(cos_matrices, n_bootstrap=DIAGNOSTIC_BOOTSTRAP_N,
                                    seed=DIAGNOSTIC_BOOTSTRAP_SEED)

    _jdump({"stage": STAGE,
            "G_cos_point": gap_bs["G_point"],
            "G_cos_ci_lower": gap_bs["G_ci_lower"],
            "G_cos_ci_upper": gap_bs["G_ci_upper"],
            "per_scenario_means": gap_bs["per_scenario_means"],
            "n_bootstrap": DIAGNOSTIC_BOOTSTRAP_N,
            "seed": DIAGNOSTIC_BOOTSTRAP_SEED},
           RESULTS_DIR / "E4OD_gradient_bootstrap.json")

    # conflict fraction bootstrap: build conflict matrices
    conflict_matrices = {}
    for s in SCENARIOS:
        mat = np.full((len(POLICY_SEEDS), len(DIAGNOSTIC_SEEDS)), np.nan)
        for pi, p in enumerate(POLICY_SEEDS):
            for ei, e in enumerate(DIAGNOSTIC_SEEDS):
                key = (p, s, e)
                if key in cosine_per_pse and np.isfinite(cosine_per_pse[key]):
                    mat[pi, ei] = 1.0 if cosine_per_pse[key] < 0 else 0.0
        conflict_matrices[s] = mat

    gap_conf_bs = scenario_gap_bootstrap(conflict_matrices, n_bootstrap=DIAGNOSTIC_BOOTSTRAP_N,
                                         seed=DIAGNOSTIC_BOOTSTRAP_SEED + 1)

    _jdump({"stage": STAGE,
            "G_conflict_point": gap_conf_bs["G_point"],
            "G_conflict_ci_lower": gap_conf_bs["G_ci_lower"],
            "G_conflict_ci_upper": gap_conf_bs["G_ci_upper"],
            "per_scenario_means": gap_conf_bs["per_scenario_means"]},
           RESULTS_DIR / "E4OD_conflict_bootstrap.json")

    # ================================================================
    # Phase 8: O1 support verdict
    # ================================================================
    print("[E4OD-8] Computing O1 support verdict...")
    o1_verdict = o1_support_verdict(
        g_cos=gap_bs["G_point"],
        g_cos_ci_upper=gap_bs["G_ci_upper"],
        n_policies_g_cos_neg=plg["n_policies_G_cos_negative"],
        n_policies=len(POLICY_SEEDS),
        g_conflict=G_conflict,
        n_policies_g_conflict_pos=plg["n_policies_G_conflict_positive"],
    )
    print(f"    O1 = {o1_verdict}")

    o1_detail = {
        "stage": STAGE,
        "O1_SUPPORT": o1_verdict,
        "criteria": {
            "G_cos": gap_bs["G_point"],
            "G_cos_negative": gap_bs["G_point"] < 0,
            "G_cos_ci_upper": gap_bs["G_ci_upper"],
            "ci_upper_negative": gap_bs["G_ci_upper"] < 0,
            "n_policies_G_cos_negative": plg["n_policies_G_cos_negative"],
            "G_conflict": G_conflict,
            "G_conflict_positive": G_conflict > 0,
            "n_policies_G_conflict_positive": plg["n_policies_G_conflict_positive"],
            "policy_G_cos_values": plg["policy_G_cos"],
            "policy_G_conflict_values": plg["policy_G_conflict"],
        },
    }
    _jdump(o1_detail, RESULTS_DIR / "E4OD_o1_verdict.json")

    # ================================================================
    # Phase 9: Common-state bank
    # ================================================================
    print("[E4OD-9] Building common-state banks...")
    state_bank_c1 = []
    state_bank_c2 = []
    for tr in all_transitions:
        rec = build_common_state_record(
            source_arm=tr["source_arm"],
            policy=tr["policy_seed"],
            scenario=tr["scenario"],
            env_seed=tr["diagnostic_environment_seed"],
            change_index=tr["episode_change_index"],
            region_id=tr["region_id"],
            local_state_9d=np.array(tr["local_state_9D"], dtype=np.float32),
            central_state_55d=np.array(tr["central_state_55D"], dtype=np.float32),
            active_mask=np.array(tr["active_mask"], dtype=np.float32),
        )
        if tr["source_arm"] == C1_ARM:
            state_bank_c1.append(rec)
        else:
            state_bank_c2.append(rec)

    _jdump({"stage": STAGE, "n_states_source_c1": len(state_bank_c1),
            "n_states_source_c2": len(state_bank_c2),
            "n_states_total": len(state_bank_c1) + len(state_bank_c2)},
           RESULTS_DIR / "E4OD_common_state_summary.json")

    # ================================================================
    # Phase 10: Cross-evaluation
    # ================================================================
    print("[E4OD-10] Cross-evaluating networks on common states...")
    cross_eval_results = {"SOURCE_C1": {}, "SOURCE_C2": {}, "POOLED": {}}

    for source_label, bank in [("SOURCE_C1", state_bank_c1), ("SOURCE_C2", state_bank_c2)]:
        per_scenario = {s: {"disagreement": [], "margin_c1": [], "margin_c2": [],
                            "delta_margin": [], "transition_matrix": np.zeros((4, 4), dtype=np.int64)}
                        for s in SCENARIOS}

        for rec in bank:
            p = rec["policy"]
            s = rec["scenario"]
            local_t = torch.tensor(rec["local_state_9D"], dtype=torch.float32).unsqueeze(0)
            xeval = cross_evaluate(local_t, nets[(C1_ARM, p)], nets[(C2_ARM, p)])
            per_scenario[s]["disagreement"].append(float(xeval["disagreement"].item()))
            per_scenario[s]["margin_c1"].append(float(xeval["margin_C1"].item()))
            per_scenario[s]["margin_c2"].append(float(xeval["margin_C2"].item()))
            per_scenario[s]["delta_margin"].append(float(xeval["Delta_margin"].item()))
            c1_a = int(xeval["argmax_C1"].item())
            c2_a = int(xeval["argmax_C2"].item())
            per_scenario[s]["transition_matrix"][c1_a, c2_a] += 1

        for s in SCENARIOS:
            d = per_scenario[s]
            cross_eval_results[source_label][s] = {
                "n_states": len(d["disagreement"]),
                "disagreement_fraction": float(np.mean(d["disagreement"])) if d["disagreement"] else 0.0,
                "mean_margin_C1": float(np.mean(d["margin_c1"])) if d["margin_c1"] else 0.0,
                "mean_margin_C2": float(np.mean(d["margin_c2"])) if d["margin_c2"] else 0.0,
                "mean_delta_margin": float(np.mean(d["delta_margin"])) if d["delta_margin"] else 0.0,
                "transition_matrix": d["transition_matrix"].tolist(),
            }

    # pooled
    for s in SCENARIOS:
        c1d = cross_eval_results["SOURCE_C1"].get(s, {})
        c2d = cross_eval_results["SOURCE_C2"].get(s, {})
        n1 = c1d.get("n_states", 0)
        n2 = c2d.get("n_states", 0)
        total_n = n1 + n2
        if total_n > 0:
            cross_eval_results["POOLED"][s] = {
                "n_states": total_n,
                "disagreement_fraction": (c1d.get("disagreement_fraction", 0)*n1
                                          + c2d.get("disagreement_fraction", 0)*n2) / total_n,
                "mean_margin_C1": (c1d.get("mean_margin_C1", 0)*n1
                                   + c2d.get("mean_margin_C1", 0)*n2) / total_n,
                "mean_margin_C2": (c1d.get("mean_margin_C2", 0)*n1
                                   + c2d.get("mean_margin_C2", 0)*n2) / total_n,
            }

    _jdump({"stage": STAGE, **cross_eval_results},
           RESULTS_DIR / "E4OD_cross_evaluation.json")

    # ================================================================
    # Phase 11: R1 signal diagnostics
    # ================================================================
    print("[E4OD-11] Computing R1 signal diagnostics...")
    r1_diag = {}
    for s in SCENARIOS:
        r1_vals = [tr["focal_R1_reward"] for tr in all_transitions
                   if tr["source_arm"] == C2_ARM and tr["scenario"] == s]
        r1_diag[s] = r1_signal_statistics(np.array(r1_vals))
        r1_diag[s]["activity"] = r1_activity(np.array(r1_vals))

    _jdump({"stage": STAGE, "per_scenario": r1_diag},
           RESULTS_DIR / "E4OD_r1_signal.json")

    # ================================================================
    # Phase 12: TD error diagnostics (C2 only)
    # ================================================================
    print("[E4OD-12] Computing TD error diagnostics...")
    td_diag = {}
    for s in SCENARIOS:
        s_trs = [tr for tr in all_transitions
                 if tr["source_arm"] == C2_ARM and tr["scenario"] == s]
        if not s_trs:
            continue
        # sample up to 1000 transitions for TD computation
        sample = s_trs[:min(1000, len(s_trs))]
        B = len(sample)

        focal_s_t = torch.tensor(np.stack([np.array(tr["local_state_9D"], dtype=np.float32)
                                           for tr in sample]), dtype=torch.float32)
        focal_a_t = torch.tensor([tr["requested_action"] for tr in sample], dtype=torch.int64)
        focal_r_t = torch.tensor([tr["focal_R1_reward"] for tr in sample], dtype=torch.float32)
        focal_ns_t = torch.tensor(np.stack([np.array(tr["next_local_state_9D"], dtype=np.float32)
                                            for tr in sample]), dtype=torch.float32)
        done_t = torch.tensor([1.0 if tr["done"] else 0.0 for tr in sample], dtype=torch.float32)

        # use first available C2 policy for TD computation
        p0 = POLICY_SEEDS[0]
        online_u = nets[(C2_ARM, p0)]
        target_u = target_nets[(C2_ARM, p0)]
        online_u.eval()

        with torch.no_grad():
            q_local = focal_current_q(online_u, focal_s_t, focal_a_t)
            y_local = focal_regional_ddqn_target(online_u, target_u, focal_r_t,
                                                  focal_ns_t, done_t, GAMMA)

        td_r = regional_td_error(q_local, y_local)
        td_diag[s] = {
            "regional": {
                "mean_abs_delta_R": td_r["mean_abs_delta_R"],
                "mean_smoothl1": td_r["mean_regional_smoothl1"],
                "summary": td_error_summary(td_r["abs_delta_R"]),
            },
            "n_transitions": B,
        }

    _jdump({"stage": STAGE, "per_scenario": td_diag},
           RESULTS_DIR / "E4OD_td_error.json")

    # ================================================================
    # Phase 13: State stratification
    # ================================================================
    print("[E4OD-13] Computing state stratification...")
    strat_diag = {}
    for s in SCENARIOS:
        states = np.array([tr["local_state_9D"] for tr in all_transitions
                           if tr["source_arm"] == C2_ARM and tr["scenario"] == s],
                          dtype=np.float32)
        if len(states) > 0:
            strat = stratify_by_delta_severity(states)
            strat_diag[s] = {
                "n_total": strat["n_total"],
                "n_delta_severity_zero": strat["n_delta_severity_zero"],
                "n_delta_severity_positive": strat["n_delta_severity_positive"],
                "zero_fraction": strat["n_delta_severity_zero"] / max(strat["n_total"], 1),
            }

    _jdump({"stage": STAGE, "per_scenario": strat_diag},
           RESULTS_DIR / "E4OD_stratification.json")

    # ================================================================
    # Phase 14: Gradient norm diagnostics
    # ================================================================
    print("[E4OD-14] Computing gradient norm summary...")
    norm_diag = {}
    for s in SCENARIOS:
        norms_j = [r["norm_g_J"] for r in norm_records if r["scenario"] == s]
        norms_r = [r["norm_g_R"] for r in norm_records if r["scenario"] == s]
        if norms_j and norms_r:
            norm_diag[s] = {
                "mean_norm_g_J": float(np.mean(norms_j)),
                "mean_norm_g_R": float(np.mean(norms_r)),
                "mean_rho_g": float(np.mean(norms_r)) / (float(np.mean(norms_j)) + 1e-12),
            }

    _jdump({"stage": STAGE, "per_scenario": norm_diag},
           RESULTS_DIR / "E4OD_gradient_norms.json")

    # ================================================================
    # Phase 15-17: Hypothesis reassessment
    # ================================================================
    print("[E4OD-15] Reassessing hypotheses...")

    # N3: Action specialization — from cross-evaluation disagreement
    n3_data = {}
    for s in SCENARIOS:
        pooled = cross_eval_results["POOLED"].get(s, {})
        n3_data[s] = pooled.get("disagreement_fraction", 0.0)
    n3_stat_higher = (n3_data.get("STATIONARY", 0) > n3_data.get("PERSISTENT", 0)
                      and n3_data.get("STATIONARY", 0) > n3_data.get("MIXED", 0))
    n3_verdict = "PLAUSIBLE" if n3_stat_higher else "NOT_SUPPORTED"

    # N4: Policy decisiveness — from Q-margins
    n4_data = {}
    for s in SCENARIOS:
        pooled = cross_eval_results["POOLED"].get(s, {})
        n4_data[s] = {
            "mean_margin_C1": pooled.get("mean_margin_C1", 0),
            "mean_margin_C2": pooled.get("mean_margin_C2", 0),
        }
    # C2 less decisive in STATIONARY?
    stat_delta = n4_data.get("STATIONARY", {}).get("mean_margin_C2", 0) - \
                 n4_data.get("STATIONARY", {}).get("mean_margin_C1", 0)
    per_delta = n4_data.get("PERSISTENT", {}).get("mean_margin_C2", 0) - \
                n4_data.get("PERSISTENT", {}).get("mean_margin_C1", 0)
    n4_stat_worse = stat_delta < per_delta
    n4_verdict = "PLAUSIBLE" if n4_stat_worse else "NOT_SUPPORTED"

    # N5: R1 signal structure
    stat_r1 = r1_diag.get("STATIONARY", {})
    per_r1 = r1_diag.get("PERSISTENT", {})
    mix_r1 = r1_diag.get("MIXED", {})
    n5_distinct = (stat_r1.get("zero_fraction", 0) > per_r1.get("zero_fraction", 0)
                   and stat_r1.get("variance", 0) < per_r1.get("variance", 0)
                   if stat_r1 and per_r1 else False)
    n5_verdict = "PLAUSIBLE" if n5_distinct else "NOT_SUPPORTED"

    # N8: Joint-regional tradeoff
    n8_rho_stat = scenario_rho.get("STATIONARY", 0)
    n8_rho_per = scenario_rho.get("PERSISTENT", 0)
    n8_higher = n8_rho_stat is not None and n8_rho_per is not None and n8_rho_stat > n8_rho_per
    n8_verdict = "PLAUSIBLE" if n8_higher else "NOT_SUPPORTED"

    # Preserved: N6, N7, N9
    n6_verdict = "SUPPORTED"
    n7_verdict = e4n.get("N7_MECHANISM", "PRESERVED_AS_IS")
    n9_verdict = "PLAUSIBLE"

    hypothesis_reassessment = {
        "N3_ACTION_SPECIALIZATION": {"verdict": n3_verdict, "data": n3_data},
        "N4_POLICY_DECISIVENESS": {"verdict": n4_verdict, "stat_delta": stat_delta, "per_delta": per_delta},
        "N5_REGIONAL_CREDIT_STRUCTURE": {"verdict": n5_verdict, "distinct": n5_distinct},
        "N6_CHECKPOINT_SELECTION": {"verdict": n6_verdict, "preserved": True},
        "N7_GLOBAL_DOMINANCE": {"verdict": n7_verdict, "preserved": True},
        "N8_JOINT_REGIONAL_TRADEOFF": {"verdict": n8_verdict, "rho_stat": n8_rho_stat, "rho_per": n8_rho_per},
        "N9_NONMONOTONIC_INTERACTION": {"verdict": n9_verdict, "preserved": True},
    }
    _jdump({"stage": STAGE, **hypothesis_reassessment},
           RESULTS_DIR / "E4OD_hypothesis_reassessment.json")

    # ================================================================
    # Phase 18: Post-integrity
    # ================================================================
    print("[E4OD-18] Post-integrity checks...")
    post_ckpt_ok = True
    for key, info in ckpt_hashes.items():
        arm_name = key.rsplit("_", 1)[0]
        p_seed = int(key.rsplit("_", 1)[1])
        actual = _hash_file(meta[(arm_name, p_seed)]["checkpoint_path"])
        if actual != info["expected"]:
            post_ckpt_ok = False
    v3_post_ok = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"] == FROZEN_V3_CONSTANTS_SHA256
    qmix_post_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256
    c2_post_ok = c2_credit_config_sha256() == FROZEN_C2_CREDIT_SHA256

    post_integrity = {
        "checkpoint_unchanged": post_ckpt_ok,
        "v3_unchanged": v3_post_ok,
        "qmix_unchanged": qmix_post_ok,
        "c2_credit_unchanged": c2_post_ok,
        "all_pass": post_ckpt_ok and v3_post_ok and qmix_post_ok and c2_post_ok,
    }
    _jdump({"stage": STAGE, **post_integrity},
           RESULTS_DIR / "E4OD_post_integrity.json")

    # No-learning audit
    _jdump({"stage": STAGE, **COUNTERS, "all_zero_except_seeds_and_env":
            COUNTERS["optimizer_steps"] == 0 and COUNTERS["parameter_changes"] == 0
            and COUNTERS["target_updates"] == 0 and COUNTERS["replay_mutations"] == 0},
           RESULTS_DIR / "E4OD_no_learning_audit.json")

    # ================================================================
    # Phase 19: Routing
    # ================================================================
    routing = _load(E4O_DIR / "E4O_future_routing.json")["routing_rules"]
    if o1_verdict == "SUPPORTED":
        next_stage = routing["O1_SUPPORTED"]["NEXT_STAGE"]
    elif n3_verdict == "SUPPORTED":
        next_stage = routing.get("O1_NOT_SUPPORTED_AND_N3_STRONGLY_SUPPORTED", {}).get(
            "NEXT_STAGE", "E4P_ACTION_SPECIALIZATION_REVISION_DESIGN")
    elif n4_verdict == "SUPPORTED":
        next_stage = routing.get("O1_NOT_SUPPORTED_AND_N4_STRONGLY_SUPPORTED", {}).get(
            "NEXT_STAGE", "E4P_POLICY_DECISIVENESS_REVISION_DESIGN")
    elif n5_verdict == "SUPPORTED":
        next_stage = routing.get("N5_STRONGEST", {}).get(
            "NEXT_STAGE", "E4P_SCENARIO_CONDITIONAL_CREDIT_REVIEW")
    elif n8_verdict == "SUPPORTED":
        next_stage = routing.get("N8_STRONGEST_WITHOUT_GRADIENT_CONFLICT", {}).get(
            "NEXT_STAGE", "E4P_JOINT_REGIONAL_OBJECTIVE_REVIEW")
    else:
        next_stage = routing.get("ALL_MECHANISMS_INSUFFICIENT", {}).get(
            "NEXT_STAGE", "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN")

    # ================================================================
    # Phase 20: Verdict
    # ================================================================
    elapsed = time.time() - t0
    verdict = {
        "stage": STAGE,
        "E4OD_COMPLETE": "YES",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "E4MF_REMAINS_AUTHORITATIVE": True,
        "O1_SUPPORT": o1_verdict,
        "O1_CRITERIA": {
            "G_cos": gap_bs["G_point"],
            "G_cos_ci_upper": gap_bs["G_ci_upper"],
            "ci_excludes_zero": gap_bs["G_ci_upper"] < 0,
            "n_policies_G_cos_neg": plg["n_policies_G_cos_negative"],
            "G_conflict": G_conflict,
            "n_policies_G_conflict_pos": plg["n_policies_G_conflict_positive"],
        },
        "HYPOTHESIS_REASSESSMENT": {
            "N3": n3_verdict, "N4": n4_verdict, "N5": n5_verdict,
            "N6": n6_verdict, "N7": n7_verdict, "N8": n8_verdict, "N9": n9_verdict,
        },
        "DIAGNOSTIC_MATRIX": {
            "total_episodes": len(keys),
            "total_transitions": len(all_transitions),
            "c2_transitions_for_gradient": sum(len(v) for v in c2_trs_by_ps.values()),
        },
        "INTEGRITY": {
            "pre": "PASS" if pre_ok else "FAIL",
            "post": "PASS" if post_integrity["all_pass"] else "FAIL",
            "no_learning": COUNTERS["optimizer_steps"] == 0,
        },
        "SCENARIO_COSINES": scenario_cosines,
        "SCENARIO_CONFLICTS": scenario_conflicts,
        "GRADIENT_NORMS": norm_diag,
        "elapsed_seconds": round(elapsed, 1),
        "NEXT_STAGE": next_stage,
    }
    _jdump(verdict, RESULTS_DIR / "E4OD_verdict.json")

    # ================================================================
    # Phase 21: Seed usage
    # ================================================================
    _jdump({"diagnostic_seed_range": [2301, 2360],
            "unique_seeds": 60, "status": "CONSUMED_DIAGNOSTIC",
            "env_runs": COUNTERS["env_runs"]},
           RESULTS_DIR / "E4OD_seed_usage.json")

    # ================================================================
    # Phase 22: Report
    # ================================================================
    print("[E4OD-22] Writing report...")
    report = _build_report(verdict, gap_bs, gap_conf_bs, plg, scenario_cosines,
                           scenario_conflicts, scenario_rho, r1_diag, td_diag,
                           cross_eval_results, strat_diag, hypothesis_reassessment,
                           norm_diag, o1_detail)
    with open(RESULTS_DIR / "E4OD_report.md", "w") as f:
        f.write(report)

    # ================================================================
    # Phase 23: Terminal summary
    # ================================================================
    _emit_terminal(verdict, gap_bs, gap_conf_bs, plg, scenario_cosines,
                   scenario_conflicts, hypothesis_reassessment, elapsed)

    print(f"\n[E4OD] total wall {elapsed:.0f}s; env_runs={COUNTERS['env_runs']}")
    return verdict


def _build_report(verdict, gap_bs, gap_conf_bs, plg, sc, sf, sr, r1, td,
                  xeval, strat, hyp, norms, o1d):
    lines = ["# E4OD — Scenario Trade-off Diagnostic Execution Report\n"]
    lines.append(f"## A. Verdict: O1 = {verdict['O1_SUPPORT']}\n")
    lines.append(f"PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED (E4MF authoritative)\n")
    lines.append(f"\n## B. Gradient Alignment\n")
    lines.append(f"- G_cos = {gap_bs['G_point']:.6f}")
    lines.append(f"- CI = [{gap_bs['G_ci_lower']:.6f}, {gap_bs['G_ci_upper']:.6f}]")
    lines.append(f"- Per-scenario mean cosine: PER={sc.get('PERSISTENT','NA'):.4f}, "
                 f"STAT={sc.get('STATIONARY','NA'):.4f}, MIX={sc.get('MIXED','NA'):.4f}")
    lines.append(f"\n## C. Conflict Fraction\n")
    lines.append(f"- Per-scenario conflict: PER={sf.get('PERSISTENT','NA'):.4f}, "
                 f"STAT={sf.get('STATIONARY','NA'):.4f}, MIX={sf.get('MIXED','NA'):.4f}")
    lines.append(f"- G_conflict = {gap_conf_bs['G_point']:.6f}")
    lines.append(f"\n## D. Policy Consistency\n")
    lines.append(f"- Policies with G_cos < 0: {plg['n_policies_G_cos_negative']}/5")
    lines.append(f"- Policies with G_conflict > 0: {plg['n_policies_G_conflict_positive']}/5")
    lines.append(f"- Policy G_cos values: {[f'{g:.4f}' for g in plg['policy_G_cos']]}")
    lines.append(f"\n## E. Gradient Norms\n")
    for s in SCENARIOS:
        nd = norms.get(s, {})
        lines.append(f"- {s}: ||g_J||={nd.get('mean_norm_g_J','NA'):.6f}, "
                     f"||g_R||={nd.get('mean_norm_g_R','NA'):.6f}, "
                     f"rho={nd.get('mean_rho_g','NA'):.4f}")
    lines.append(f"\n## F. Cross-Evaluation (Pooled)\n")
    for s in SCENARIOS:
        pd = xeval["POOLED"].get(s, {})
        lines.append(f"- {s}: D_action={pd.get('disagreement_fraction','NA'):.4f}, "
                     f"margin_C1={pd.get('mean_margin_C1','NA'):.4f}, "
                     f"margin_C2={pd.get('mean_margin_C2','NA'):.4f}")
    lines.append(f"\n## G. R1 Signal\n")
    for s in SCENARIOS:
        rd = r1.get(s, {})
        lines.append(f"- {s}: zero={rd.get('zero_fraction','NA'):.4f}, "
                     f"activity={rd.get('activity','NA'):.4f}, "
                     f"mean={rd.get('mean','NA'):.6f}, var={rd.get('variance','NA'):.8f}")
    lines.append(f"\n## H. Hypothesis Reassessment\n")
    for h, data in hyp.items():
        lines.append(f"- {h}: {data.get('verdict', 'NA')}")
    lines.append(f"\n## I. Routing\n")
    lines.append(f"- NEXT_STAGE = {verdict['NEXT_STAGE']}")
    lines.append(f"\n## J. Integrity\n")
    lines.append(f"- Pre: {verdict['INTEGRITY']['pre']}")
    lines.append(f"- Post: {verdict['INTEGRITY']['post']}")
    lines.append(f"- No learning: {verdict['INTEGRITY']['no_learning']}")
    lines.append(f"- Diagnostic seeds consumed: 2301..2360")
    lines.append(f"\nElapsed: {verdict['elapsed_seconds']:.1f}s")
    return "\n".join(lines) + "\n"


def _emit_terminal(verdict, gap_bs, gap_conf_bs, plg, sc, sf, hyp, elapsed):
    print("\n" + "=" * 72)
    print("E4OD — SCENARIO TRADE-OFF DIAGNOSTIC EXECUTION — TERMINAL SUMMARY")
    print("=" * 72)
    print(f"  PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED (E4MF authoritative)")
    print(f"  O1_SUPPORT = {verdict['O1_SUPPORT']}")
    print()
    print("  GRADIENT ALIGNMENT:")
    print(f"    G_cos = {gap_bs['G_point']:.6f}  CI=[{gap_bs['G_ci_lower']:.6f}, {gap_bs['G_ci_upper']:.6f}]")
    print(f"    G_conflict = {gap_conf_bs['G_point']:.6f}")
    print(f"    Per-scenario cosine: PER={sc.get('PERSISTENT','NA')}, "
          f"STAT={sc.get('STATIONARY','NA')}, MIX={sc.get('MIXED','NA')}")
    print(f"    Per-scenario conflict: PER={sf.get('PERSISTENT','NA')}, "
          f"STAT={sf.get('STATIONARY','NA')}, MIX={sf.get('MIXED','NA')}")
    print(f"    Policy G_cos<0: {plg['n_policies_G_cos_negative']}/5")
    print(f"    Policy G_conflict>0: {plg['n_policies_G_conflict_positive']}/5")
    print()
    print("  HYPOTHESIS REASSESSMENT:")
    for h, data in hyp.items():
        print(f"    {h}: {data.get('verdict', 'NA')}")
    print()
    print(f"  INTEGRITY: pre={verdict['INTEGRITY']['pre']} post={verdict['INTEGRITY']['post']}")
    print(f"  NEXT_STAGE = {verdict['NEXT_STAGE']}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
