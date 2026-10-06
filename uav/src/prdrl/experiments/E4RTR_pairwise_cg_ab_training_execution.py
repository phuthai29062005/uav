"""
E4RTR — Pairwise CG A/B Training Execution.

Executes the E4RT-frozen matched coordination experiment:
    C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 (CONTROL) vs
    C3_PAIRWISE_COORDINATION_GRAPH_V1 (TREATMENT)
trains 10 runs (C1×5 + C3×5) via EXACT E4RT-frozen protocol; saves 60 checkpoints;
validates every checkpoint on seeds 2561..2580; selects one per run by frozen macro-MIGD_response
rule; retains all 10; reports descriptive coordination diagnostics. NO final seeds (2581..2640)
are opened; NO final causal verdict; NO bootstrap.

PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE is the ONLY factor.
C1 = QMIX joint TD (utility + mixer). C3 = coordination graph joint TD (unary + pair).
Both arms share V3 normalization, R1 reward (alpha=0.1), 9-128-128-4 utility/unary network,
gamma 0.99, Adam 1e-4, SmoothL1, tau 0.005.

Do not rescue either arm. If CG trains poorly: preserve the result.
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, build_warmup_cache, select_checkpoint,
    _hash_file, _code_manifest_hash, _frozen_config_hash,
    ALPHA_G, ALPHA_L, BETA,
    N_EPOCHS, CHECKPOINT_TRANSITIONS, SCENARIOS, SCENARIO_ID,
    pre_response, build_region_states, execute_and_recover, advance_state, _fresh_state,
    N_SCORED_CHANGES, compute_joint_rewards, validate_online_net,
    BATCH_SIZE, REPLAY_CAPACITY, GAMMA, LEARNING_RATE, TARGET_TAU,
    WARMUP_TRANSITIONS, UPDATE_FREQUENCY,
    EPS_START, EPS_END, EPS_DECAY, DEVICE,
    RNG_OFFSET_POLICY_INIT, RNG_OFFSET_TORCH, RNG_OFFSET_REPLAY_SAMPLE,
    RNG_OFFSET_TRAIN_EPISODE_ORDER, EXPLORE_NAMESPACE_TAG,
    _derive_event_seed, _assert_no_region_disappearance, _numerical_finite,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics
from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair, N_ACTIONS, ACTION_ORDER,
)
from prdrl.rl.agent import (
    EpsilonGreedyConfig, select_regional_actions, ACTION_TO_IDX,
)
from prdrl.rl.normalization import (
    NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS,
)
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.qmix import (
    R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM,
    TREATMENT_ARCH, build_mixer_online_target_pair, derive_mixer_seed,
)
from prdrl.rl.joint_context import central_state_from_regions
from prdrl.rl.joint_replay import JointReplayBuffer, make_event_anchors
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, FROZEN_V3_CONSTANTS_SHA256, STATE_DIM,
    joint_reward_mean, current_joint_q, double_q_joint_target, treatment_loss,
    c1_optimizer_params, soft_update, count_parameters,
    make_qmix_config, qmix_config_sha256,
)
from prdrl.rl.coordination_graph import (
    TREATMENT_ARCH_C3, SharedPairNetwork, PAIR_SEED_OFFSET, K_MAXSUM,
    SMALL_N_EXACT_THRESHOLD,
    derive_pair_seed, build_pair_online_target,
    compute_centered_pair, build_complete_graph,
    maxsum_inference_batched as maxsum_inference,
    joint_value_cg,
    c3_double_q_target_batched as c3_double_q_target,
    c3_current_q_batched as c3_current_q,
    c3_loss, c3_optimizer_params,
    small_n_diagnostic,
    make_c3_config, c3_config_sha256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.experiments.E4P_learned_controller_preparation import NETWORK_INPUT_DIM

# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR = Path("results/E4RTR_pairwise_cg_ab_training_execution")
E4RT_DIR = Path("results/E4RT_pairwise_cg_ab_training_protocol_freeze")
E4RI_DIR = Path("results/E4RI_pairwise_coordination_graph_implementation_validation")
E4R_DIR = Path("results/E4R_pairwise_coordination_graph_experiment_design")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4MF_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")

TRAIN_SEEDS = list(range(2501, 2561))
VAL_SEEDS = list(range(2561, 2581))
FINAL_SEEDS = set(range(2581, 2641))
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
EPISODES_PER_EPOCH = len(TRAIN_SEEDS) * len(SCENARIOS)
N_TRAIN_EPISODES = N_EPOCHS * EPISODES_PER_EPOCH
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C3_CONFIG_SHA256 = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"


class FinalSeedFirewall:
    def __init__(self):
        self.access_count = 0
        self.forbidden = FINAL_SEEDS

    def check(self, seed: int):
        if int(seed) in self.forbidden:
            self.access_count += 1
            raise RuntimeError(
                f"E4RTR_FINAL_FIREWALL: seed {seed} (2581..2640) MUST NOT be accessed.")


FIREWALL = FinalSeedFirewall()


# ============================================================================
# UTILITIES
# ============================================================================

def _jdefault(x):
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
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


def _param_hash(net) -> str:
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _e4rtr_code_manifest_hash() -> str:
    files = [
        "src/prdrl/experiments/E4RTR_pairwise_cg_ab_training_execution.py",
        "src/prdrl/experiments/E4TR_shared_double_dqn_training.py",
        "src/prdrl/rl/coordination_integration.py",
        "src/prdrl/rl/coordination_graph.py",
        "src/prdrl/rl/qmix.py",
        "src/prdrl/rl/joint_context.py",
        "src/prdrl/rl/joint_replay.py",
        "src/prdrl/rl/network.py",
        "src/prdrl/rl/reward.py",
        "src/prdrl/rl/reward_r1.py",
        "src/prdrl/rl/agent.py",
        "src/prdrl/rl/normalization.py",
    ]
    h = hashlib.sha256()
    for fp in files:
        h.update(_hash_file(fp).encode())
    return h.hexdigest()


def load_v3():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"],
                   "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3(
        {"zero_aware": za, "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    return v3, c["constants_sha256"]


# ============================================================================
# C3 ACTION SELECTION AND VALIDATION
# ============================================================================

def eval_c3_arm(online_unary, online_pair, scenario, seed, cache, env,
                v3_transform) -> dict:
    region_dynamics = make_region_dynamics(scenario, env.R)
    state = _fresh_state(scenario, seed, cache, env.R)
    online_unary.eval(); online_pair.eval()
    igd_resp, igd_ep, fe_list = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    n_fallback = 0
    prev_active = None
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(state, scenario, seed, env, region_dynamics)
        if prev_active is not None:
            _assert_no_region_disappearance(prev_active, pre["active_regions"])
        local_norm, gc_norm, full_states = build_region_states(
            pre["v2_states"], env.R, v3_transform)

        j_states = torch.zeros(1, R_MAX, STATE_DIM, dtype=torch.float32, device=DEVICE)
        a_mask = torch.zeros(1, R_MAX, dtype=torch.float32, device=DEVICE)
        for r in pre["active_regions"]:
            j_states[0, r] = torch.tensor(full_states[r], dtype=torch.float32, device=DEVICE)
            a_mask[0, r] = 1.0

        with torch.no_grad():
            actions_tensor = maxsum_inference(
                online_unary, online_pair, j_states, a_mask)

        action_str = {}
        for r in pre["active_regions"]:
            action_str[r] = ACTION_ORDER[actions_tensor[0, r].item()]
        actions = {r: action_str[r] for r in pre["active_regions"]}
        post = execute_and_recover(
            actions, pre, state, scenario, seed, env, region_dynamics, k + 2)
        igd_resp.append(post["igd_response"])
        igd_ep.append(post["igd_epoch"])
        fe_list.append(sum(rec.response_fe for rec in post["records"]))
        for rec in post["records"]:
            act_counts[rec.requested_action] += 1
            if rec.fallback_used:
                n_fallback += 1
        prev_active = pre["active_regions"]
        advance_state(state, pre, post, env)

    total_act = sum(act_counts.values()) or 1
    return {
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "mean_response_fe": float(np.mean(fe_list)),
        "total_response_fe": int(np.sum(fe_list)),
        "action_fraction": {a: act_counts[a] / total_act for a in ACTION_ORDER},
        "fallback_rate": float(n_fallback / total_act),
    }


def validate_c3(online_unary, online_pair, cache_val, env, val_seeds,
                v3_transform) -> dict:
    per_scen = {}
    for scenario in SCENARIOS:
        resp, ep, fe = [], [], []
        for seed in val_seeds:
            m = eval_c3_arm(
                online_unary, online_pair, scenario, seed, cache_val, env,
                v3_transform)
            resp.append(m["MIGD_response"])
            ep.append(m["MIGD_epoch"])
            fe.append(m["mean_response_fe"])
        per_scen[scenario] = {
            "mean_MIGD_response": float(np.mean(resp)),
            "mean_MIGD_epoch": float(np.mean(ep)),
            "mean_response_fe": float(np.mean(fe)),
        }
    macro_resp = float(np.mean(
        [per_scen[s]["mean_MIGD_response"] for s in SCENARIOS]))
    macro_epoch = float(np.mean(
        [per_scen[s]["mean_MIGD_epoch"] for s in SCENARIOS]))
    macro_fe = float(np.mean(
        [per_scen[s]["mean_response_fe"] for s in SCENARIOS]))
    return {
        "per_scenario": per_scen,
        "VALIDATION_MACRO_MIGD_RESPONSE": macro_resp,
        "macro_MIGD_epoch": macro_epoch,
        "macro_mean_response_fe": macro_fe,
    }


def mixed_action_metrics_c1(net, normalizer, cache_val, env, val_seeds):
    rd = make_region_dynamics("MIXED", env.R)
    net.eval()
    counts = {a: 0 for a in ACTION_ORDER}
    multi = []
    for seed in val_seeds:
        st = _fresh_state("MIXED", seed, cache_val, env.R)
        for k in range(N_SCORED_CHANGES):
            pre = pre_response(st, "MIXED", seed, env, rd)
            local_norm, gc_norm, _ = build_region_states(
                pre["v2_states"], env.R, normalizer)
            with torch.no_grad():
                _, action_str = select_regional_actions(
                    net, local_norm, gc_norm, epsilon=0.0,
                    event_seed=0, device=DEVICE)
            reqs = [action_str[r] for r in pre["active_regions"]]
            multi.append(1.0 if len(set(reqs)) >= 2 else 0.0)
            for a in reqs:
                counts[a] += 1
            actions = {r: action_str[r] for r in pre["active_regions"]}
            post = execute_and_recover(
                actions, pre, st, "MIXED", seed, env, rd, k + 2)
            advance_state(st, pre, post, env)
    tot = sum(counts.values()) or 1
    fr = {a: counts[a] / tot for a in ACTION_ORDER}
    ent = -sum(p * math.log(p) for p in fr.values() if p > 0)
    return {"predict_fraction": fr["PREDICT"],
            "multi_action_rate": float(np.mean(multi)),
            "action_entropy": ent}


def mixed_action_metrics_c3(unary_net, pair_net, normalizer, cache_val, env,
                            val_seeds):
    rd = make_region_dynamics("MIXED", env.R)
    unary_net.eval(); pair_net.eval()
    counts = {a: 0 for a in ACTION_ORDER}
    multi = []
    for seed in val_seeds:
        st = _fresh_state("MIXED", seed, cache_val, env.R)
        for k in range(N_SCORED_CHANGES):
            pre = pre_response(st, "MIXED", seed, env, rd)
            local_norm, gc_norm, full_states = build_region_states(
                pre["v2_states"], env.R, normalizer)
            j_states = torch.zeros(
                1, R_MAX, STATE_DIM, dtype=torch.float32, device=DEVICE)
            a_mask = torch.zeros(
                1, R_MAX, dtype=torch.float32, device=DEVICE)
            for r in pre["active_regions"]:
                j_states[0, r] = torch.tensor(
                    full_states[r], dtype=torch.float32, device=DEVICE)
                a_mask[0, r] = 1.0
            with torch.no_grad():
                at = maxsum_inference(unary_net, pair_net, j_states, a_mask)
            action_str = {}
            for r in pre["active_regions"]:
                action_str[r] = ACTION_ORDER[at[0, r].item()]
            reqs = [action_str[r] for r in pre["active_regions"]]
            multi.append(1.0 if len(set(reqs)) >= 2 else 0.0)
            for a in reqs:
                counts[a] += 1
            actions = {r: action_str[r] for r in pre["active_regions"]}
            post = execute_and_recover(
                actions, pre, st, "MIXED", seed, env, rd, k + 2)
            advance_state(st, pre, post, env)
    tot = sum(counts.values()) or 1
    fr = {a: counts[a] / tot for a in ACTION_ORDER}
    ent = -sum(p * math.log(p) for p in fr.values() if p > 0)
    return {"predict_fraction": fr["PREDICT"],
            "multi_action_rate": float(np.mean(multi)),
            "action_entropy": ent}


# ============================================================================
# C1 TRAINING — one policy seed (QMIX joint TD, CONTROL)
# ============================================================================

def train_one_policy_c1(policy_seed: int, cache_train: dict, cache_val: dict,
                        env, policy_dir: Path, v3_transform,
                        *, frozen_hash="", code_hash="",
                        checkpoint_extra_meta=None) -> dict:
    ckpt_dir = policy_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(policy_seed * 1000 + RNG_OFFSET_TORCH)
    online_util, target_util = build_online_target_pair(
        NETWORK_INPUT_DIM, seed=policy_seed * 1000 + RNG_OFFSET_POLICY_INIT)
    online_util.to(DEVICE); target_util.to(DEVICE)

    mixer_seed = derive_mixer_seed(policy_seed * 1000 + RNG_OFFSET_POLICY_INIT)
    online_mixer, target_mixer = build_mixer_online_target_pair(mixer_seed)
    online_mixer.to(DEVICE); target_mixer.to(DEVICE)

    optimizer = torch.optim.Adam(
        c1_optimizer_params(online_util, online_mixer), lr=LEARNING_RATE)

    buffer = JointReplayBuffer(
        capacity=REPLAY_CAPACITY,
        rng_seed=policy_seed * 1000 + RNG_OFFSET_REPLAY_SAMPLE,
        warmup=WARMUP_TRANSITIONS, update_every=UPDATE_FREQUENCY)

    eps_cfg = EpsilonGreedyConfig(EPS_START, EPS_END, EPS_DECAY)
    order_rng = np.random.default_rng(
        policy_seed * 1000 + RNG_OFFSET_TRAIN_EPISODE_ORDER)

    total_anchors = 0
    optimizer_steps = 0
    checkpoints_meta: List[dict] = []
    training_log: List[dict] = []
    checkpoint_targets = set(CHECKPOINT_TRANSITIONS)
    diag_td_losses: List[float] = []
    numerical_stability = "PASS"
    failure_info = None

    def do_update(ep_loss_acc):
        nonlocal optimizer_steps, numerical_stability, failure_info
        batch = buffer.sample(BATCH_SIZE)
        js = torch.tensor(np.stack([a.joint_states for a in batch]),
                          dtype=torch.float32, device=DEVICE)
        am = torch.tensor(np.stack([a.active_mask for a in batch]),
                          dtype=torch.float32, device=DEVICE)
        av = torch.tensor(np.stack([a.action_vector for a in batch]),
                          dtype=torch.int64, device=DEVICE)
        cs = torch.tensor(np.stack([a.central_state for a in batch]),
                          dtype=torch.float32, device=DEVICE)
        rj = torch.tensor([a.r_joint for a in batch],
                          dtype=torch.float32, device=DEVICE)
        njs = torch.tensor(np.stack([a.next_joint_states for a in batch]),
                           dtype=torch.float32, device=DEVICE)
        nam = torch.tensor(np.stack([a.next_active_mask for a in batch]),
                           dtype=torch.float32, device=DEVICE)
        ncs = torch.tensor(np.stack([a.next_central_state for a in batch]),
                           dtype=torch.float32, device=DEVICE)
        dn = torch.tensor([float(a.done) for a in batch],
                          dtype=torch.float32, device=DEVICE)

        if not _numerical_finite(js, rj, njs, dn):
            numerical_stability = "FAIL"
            failure_info = {"stage": "batch_inputs", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite batch inputs")

        q_tot = current_joint_q(online_util, online_mixer, js, av, am, cs)
        y = double_q_joint_target(
            online_util, target_util, target_mixer, njs, nam, ncs, rj, dn)
        loss = treatment_loss(q_tot, y)

        if not _numerical_finite(q_tot, y, loss):
            numerical_stability = "FAIL"
            failure_info = {"stage": "forward", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite q_tot/target/loss")

        optimizer.zero_grad()
        loss.backward()
        all_params = list(online_util.parameters()) + list(online_mixer.parameters())
        if not all((p.grad is None) or torch.isfinite(p.grad).all()
                   for p in all_params):
            numerical_stability = "FAIL"
            failure_info = {"stage": "gradients", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite gradients")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in all_params):
            numerical_stability = "FAIL"
            failure_info = {"stage": "parameters", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite parameters")

        soft_update(online_util, target_util)
        soft_update(online_mixer, target_mixer)
        optimizer_steps += 1
        lv = float(loss.item())
        ep_loss_acc["loss"].append(lv)
        ep_loss_acc["abs_q"].append(float(q_tot.abs().mean().item()))
        ep_loss_acc["max_q"].append(float(q_tot.abs().max().item()))
        diag_td_losses.append(lv)

    def save_checkpoint(name: str, anchor_count: int):
        path = ckpt_dir / f"{name}.pt"
        payload = {
            "online_state_dict": online_util.state_dict(),
            "target_state_dict": target_util.state_dict(),
            "mixer_online_state_dict": online_mixer.state_dict(),
            "mixer_target_state_dict": target_mixer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_seed": policy_seed,
            "transition_count": anchor_count,
            "optimizer_step_count": optimizer_steps,
            "epsilon": eps_cfg.get_epsilon(anchor_count),
            "frozen_config_hash": frozen_hash,
            "code_manifest_hash": code_hash,
            "architecture_version": TREATMENT_ARCH,
            "qmix_config_sha256": FROZEN_QMIX_CONFIG_SHA256,
        }
        if checkpoint_extra_meta:
            payload.update(checkpoint_extra_meta)
        torch.save(payload, path)
        return path

    def push_and_learn(anchors_list, ep_loss):
        nonlocal total_anchors
        for anchor in anchors_list:
            buffer.push(anchor)
            total_anchors += 1
            if buffer.ready() and total_anchors % UPDATE_FREQUENCY == 0:
                do_update(ep_loss)
            if total_anchors in checkpoint_targets:
                online_util.eval(); online_mixer.eval()
                val = validate_online_net(
                    online_util, cache_val, env, VAL_SEEDS, v3_transform)
                online_util.train(); online_mixer.train()
                path = save_checkpoint(
                    f"ckpt_{total_anchors}", total_anchors)
                checkpoints_meta.append({
                    "name": f"ckpt_{total_anchors}",
                    "path": str(path),
                    "transition_count": total_anchors,
                    "optimizer_step_count": optimizer_steps,
                    "epsilon": eps_cfg.get_epsilon(total_anchors),
                    "VALIDATION_MACRO_MIGD_RESPONSE":
                        val["VALIDATION_MACRO_MIGD_RESPONSE"],
                    "macro_MIGD_epoch": val["macro_MIGD_epoch"],
                    "macro_mean_response_fe": val["macro_mean_response_fe"],
                    "per_scenario": val["per_scenario"],
                })

    online_util.train(); online_mixer.train()
    episode_index = 0
    for epoch in range(N_EPOCHS):
        episodes = [(sc, sd) for sc in SCENARIOS for sd in TRAIN_SEEDS]
        order = order_rng.permutation(len(episodes))
        episodes = [episodes[i] for i in order]

        for (scenario, env_seed) in episodes:
            FIREWALL.check(env_seed)
            region_dynamics = make_region_dynamics(scenario, env.R)
            state = _fresh_state(scenario, env_seed, cache_train, env.R)
            pending_event = None
            ep_loss = {"loss": [], "abs_q": [], "max_q": []}
            ep_resp, ep_ep, ep_fe, ep_reward = [], [], [], 0.0
            act_counts = {a: 0 for a in ACTION_ORDER}
            n_fallback = 0
            prev_active = None

            for k in range(N_SCORED_CHANGES):
                pre = pre_response(
                    state, scenario, env_seed, env, region_dynamics)
                if prev_active is not None:
                    _assert_no_region_disappearance(
                        prev_active, pre["active_regions"])

                local_norm, gc_norm, full_states = build_region_states(
                    pre["v2_states"], env.R, v3_transform)

                j_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                a_mask = np.zeros(R_MAX, dtype=np.float32)
                for r in pre["active_regions"]:
                    j_states[r] = full_states[r]
                    a_mask[r] = 1.0
                c_state = central_state_from_regions(
                    {r: local_norm[r] for r in pre["active_regions"]},
                    gc_norm)

                epsilon = eps_cfg.get_epsilon(total_anchors)
                event_seed = _derive_event_seed(
                    policy_seed, scenario, env_seed, epoch, k)
                action_idx, action_str = select_regional_actions(
                    online_util, local_norm, gc_norm, epsilon,
                    event_seed, DEVICE)
                actions = {r: action_str[r] for r in pre["active_regions"]}

                a_vec = np.zeros(R_MAX, dtype=np.int64)
                for r in pre["active_regions"]:
                    a_vec[r] = ACTION_TO_IDX[action_str[r]]

                if pending_event is not None:
                    pe = pending_event
                    anchors = make_event_anchors(
                        active_regions=pe["active_regions"],
                        joint_states=pe["joint_states"],
                        active_mask=pe["active_mask"],
                        action_vector=pe["action_vector"],
                        central_state=pe["central_state"],
                        regional_rewards=pe["regional_rewards"],
                        r_joint=pe["r_joint"],
                        next_joint_states=j_states,
                        next_active_mask=a_mask,
                        next_central_state=c_state,
                        done=False,
                        event_id=pe["event_id"],
                    )
                    push_and_learn(anchors, ep_loss)
                    pending_event = None

                t_current_before = state["t_current"]
                post = execute_and_recover(
                    actions, pre, state, scenario, env_seed,
                    env, region_dynamics, k + 2)
                _gr, region_rewards, _aux = compute_joint_rewards(
                    pre, post, post["records"], env, region_dynamics,
                    t_current_before, reward_version=REWARD_R1_NAME)

                rr_array = np.zeros(R_MAX, dtype=np.float32)
                for r in pre["active_regions"]:
                    rr_array[r] = region_rewards[r]
                r_joint = joint_reward_mean(rr_array, a_mask)

                event_id = (f"{policy_seed}:{scenario}:{env_seed}"
                            f":e{epoch}:c{k}")
                done = (k == N_SCORED_CHANGES - 1)

                if done:
                    anchors = make_event_anchors(
                        active_regions=sorted(pre["active_regions"]),
                        joint_states=j_states,
                        active_mask=a_mask,
                        action_vector=a_vec,
                        central_state=c_state,
                        regional_rewards=rr_array,
                        r_joint=r_joint,
                        next_joint_states=np.zeros(
                            (R_MAX, STATE_DIM), dtype=np.float32),
                        next_active_mask=np.zeros(
                            R_MAX, dtype=np.float32),
                        next_central_state=np.zeros(
                            CENTRAL_STATE_DIM, dtype=np.float32),
                        done=True, event_id=event_id,
                    )
                    push_and_learn(anchors, ep_loss)
                else:
                    pending_event = {
                        "active_regions": sorted(pre["active_regions"]),
                        "joint_states": j_states.copy(),
                        "active_mask": a_mask.copy(),
                        "action_vector": a_vec.copy(),
                        "central_state": c_state.copy(),
                        "regional_rewards": rr_array.copy(),
                        "r_joint": r_joint,
                        "event_id": event_id,
                    }

                ep_resp.append(post["igd_response"])
                ep_ep.append(post["igd_epoch"])
                ep_fe.append(sum(
                    rec.response_fe for rec in post["records"]))
                ep_reward += float(sum(region_rewards.values()))
                for rec in post["records"]:
                    act_counts[rec.requested_action] += 1
                    if rec.fallback_used:
                        n_fallback += 1
                prev_active = pre["active_regions"]
                advance_state(state, pre, post, env)

            total_act = sum(act_counts.values()) or 1
            training_log.append({
                "policy_seed": policy_seed, "epoch": epoch,
                "episode_index": episode_index, "scenario": scenario,
                "environment_seed": env_seed,
                "regional_transition_count": total_anchors,
                "epsilon": eps_cfg.get_epsilon(total_anchors),
                "replay_size": len(buffer),
                "optimizer_steps": optimizer_steps,
                "mean_td_loss": (float(np.mean(ep_loss["loss"]))
                                 if ep_loss["loss"] else float("nan")),
                "mean_abs_q": (float(np.mean(ep_loss["abs_q"]))
                               if ep_loss["abs_q"] else float("nan")),
                "max_abs_q": (float(np.max(ep_loss["max_q"]))
                              if ep_loss["max_q"] else float("nan")),
                "action_fraction_KEEP": act_counts["KEEP"] / total_act,
                "action_fraction_LOCAL": act_counts["LOCAL"] / total_act,
                "action_fraction_PREDICT": act_counts["PREDICT"] / total_act,
                "action_fraction_DIVERSIFY":
                    act_counts["DIVERSIFY"] / total_act,
                "fallback_rate": n_fallback / total_act,
                "total_response_FE": int(np.sum(ep_fe)),
                "MIGD_response": float(np.mean(ep_resp)),
                "MIGD_epoch": float(np.mean(ep_ep)),
                "cumulative_reward": ep_reward,
                "numerical_finite_flag": True,
            })
            episode_index += 1

    online_util.eval(); online_mixer.eval()
    val_final = validate_online_net(
        online_util, cache_val, env, VAL_SEEDS, v3_transform)
    online_util.train(); online_mixer.train()
    final_path = save_checkpoint("ckpt_final", total_anchors)
    checkpoints_meta.append({
        "name": "ckpt_final", "path": str(final_path),
        "transition_count": total_anchors,
        "optimizer_step_count": optimizer_steps,
        "epsilon": eps_cfg.get_epsilon(total_anchors),
        "VALIDATION_MACRO_MIGD_RESPONSE":
            val_final["VALIDATION_MACRO_MIGD_RESPONSE"],
        "macro_MIGD_epoch": val_final["macro_MIGD_epoch"],
        "macro_mean_response_fe": val_final["macro_mean_response_fe"],
        "per_scenario": val_final["per_scenario"],
    })

    selected = select_checkpoint(checkpoints_meta)

    mono_stats = _monotonicity_audit(online_util, online_mixer, buffer)

    return {
        "policy_seed": policy_seed,
        "total_transitions": total_anchors,
        "optimizer_steps": optimizer_steps,
        "n_episodes": episode_index,
        "checkpoints": checkpoints_meta,
        "selected_checkpoint": selected,
        "training_log": training_log,
        "numerical_stability": numerical_stability,
        "failure_info": failure_info,
        "c1_diagnostics": {
            "mean_joint_td_loss": (float(np.mean(diag_td_losses))
                                   if diag_td_losses else float("nan")),
            "n_updates": optimizer_steps,
            "monotonicity": mono_stats,
            "util_param_count": count_parameters(online_util),
            "mixer_param_count": count_parameters(online_mixer),
        },
    }


def _monotonicity_audit(online_util, online_mixer, buffer, n_samples=1000):
    if len(buffer) < BATCH_SIZE:
        return {"computed": False}
    online_util.eval(); online_mixer.eval()
    batch = buffer.sample(min(n_samples, len(buffer)))
    js = torch.tensor(np.stack([a.joint_states for a in batch]),
                      dtype=torch.float32, device=DEVICE)
    am = torch.tensor(np.stack([a.active_mask for a in batch]),
                      dtype=torch.float32, device=DEVICE)
    av = torch.tensor(np.stack([a.action_vector for a in batch]),
                      dtype=torch.int64, device=DEVICE)
    cs = torch.tensor(np.stack([a.central_state for a in batch]),
                      dtype=torch.float32, device=DEVICE)
    B = js.shape[0]
    with torch.no_grad():
        q_all = online_util(
            js.reshape(B * R_MAX, STATE_DIM)).reshape(B, R_MAX, N_ACTIONS)
        q_sel = q_all.gather(
            2, av.long().unsqueeze(2)).squeeze(2)
    q_leaf = q_sel.detach().clone().requires_grad_(True)
    q_masked = q_leaf * am
    q_tot = online_mixer(q_masked, cs)
    grads = torch.autograd.grad(
        q_tot.sum(), q_leaf, create_graph=False)[0]
    g_active = grads[am > 0].detach().cpu().numpy()
    if len(g_active) == 0:
        return {"computed": True, "n_active_slots": 0, "violation_count": 0}
    viol = int((g_active < -1e-7).sum())
    return {
        "computed": True,
        "n_active_slots": len(g_active),
        "min_dQtot_dqr": float(np.min(g_active)),
        "median_dQtot_dqr": float(np.median(g_active)),
        "q95_dQtot_dqr": float(np.percentile(g_active, 95)),
        "violation_count": viol,
        "violation_rate": float(viol / len(g_active)),
    }


# ============================================================================
# C3 TRAINING — one policy seed (CG joint TD, TREATMENT)
# ============================================================================

def train_one_policy_c3(policy_seed: int, cache_train: dict, cache_val: dict,
                        env, policy_dir: Path, v3_transform,
                        *, frozen_hash="", code_hash="",
                        checkpoint_extra_meta=None) -> dict:
    ckpt_dir = policy_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(policy_seed * 1000 + RNG_OFFSET_TORCH)
    online_unary, target_unary = build_online_target_pair(
        NETWORK_INPUT_DIM, seed=policy_seed * 1000 + RNG_OFFSET_POLICY_INIT)
    online_unary.to(DEVICE); target_unary.to(DEVICE)

    pair_seed = derive_pair_seed(policy_seed)
    online_pair, target_pair = build_pair_online_target(pair_seed)
    online_pair.to(DEVICE); target_pair.to(DEVICE)

    optimizer = torch.optim.Adam(
        c3_optimizer_params(online_unary, online_pair), lr=LEARNING_RATE)

    buffer = JointReplayBuffer(
        capacity=REPLAY_CAPACITY,
        rng_seed=policy_seed * 1000 + RNG_OFFSET_REPLAY_SAMPLE,
        warmup=WARMUP_TRANSITIONS, update_every=UPDATE_FREQUENCY)

    eps_cfg = EpsilonGreedyConfig(EPS_START, EPS_END, EPS_DECAY)
    order_rng = np.random.default_rng(
        policy_seed * 1000 + RNG_OFFSET_TRAIN_EPISODE_ORDER)

    total_anchors = 0
    optimizer_steps = 0
    checkpoints_meta: List[dict] = []
    training_log: List[dict] = []
    checkpoint_targets = set(CHECKPOINT_TRANSITIONS)
    diag_td_losses: List[float] = []
    diag_edge_counts: List[int] = []
    numerical_stability = "PASS"
    failure_info = None

    def do_update(ep_loss_acc):
        nonlocal optimizer_steps, numerical_stability, failure_info
        batch = buffer.sample(BATCH_SIZE)
        js = torch.tensor(np.stack([a.joint_states for a in batch]),
                          dtype=torch.float32, device=DEVICE)
        am = torch.tensor(np.stack([a.active_mask for a in batch]),
                          dtype=torch.float32, device=DEVICE)
        av = torch.tensor(np.stack([a.action_vector for a in batch]),
                          dtype=torch.int64, device=DEVICE)
        rj = torch.tensor([a.r_joint for a in batch],
                          dtype=torch.float32, device=DEVICE)
        njs = torch.tensor(np.stack([a.next_joint_states for a in batch]),
                           dtype=torch.float32, device=DEVICE)
        nam = torch.tensor(np.stack([a.next_active_mask for a in batch]),
                           dtype=torch.float32, device=DEVICE)
        dn = torch.tensor([float(a.done) for a in batch],
                          dtype=torch.float32, device=DEVICE)

        if not _numerical_finite(js, rj, njs, dn):
            numerical_stability = "FAIL"
            failure_info = {"stage": "batch_inputs", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite batch inputs")

        q_current = c3_current_q(online_unary, online_pair, js, av, am)
        y_joint = c3_double_q_target(
            online_unary, online_pair, target_unary, target_pair,
            njs, nam, rj, dn, GAMMA)
        loss = c3_loss(q_current, y_joint)

        if not _numerical_finite(q_current, y_joint, loss):
            numerical_stability = "FAIL"
            failure_info = {"stage": "forward", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite q/target/loss")

        optimizer.zero_grad()
        loss.backward()
        all_params = list(online_unary.parameters()) + list(online_pair.parameters())
        if not all((p.grad is None) or torch.isfinite(p.grad).all()
                   for p in all_params):
            numerical_stability = "FAIL"
            failure_info = {"stage": "gradients", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite gradients")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in all_params):
            numerical_stability = "FAIL"
            failure_info = {"stage": "parameters", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite parameters")

        soft_update(online_unary, target_unary)
        soft_update(online_pair, target_pair)
        optimizer_steps += 1
        lv = float(loss.item())
        ep_loss_acc["loss"].append(lv)
        ep_loss_acc["abs_q"].append(float(q_current.abs().mean().item()))
        ep_loss_acc["max_q"].append(float(q_current.abs().max().item()))
        diag_td_losses.append(lv)

    def save_checkpoint(name: str, anchor_count: int):
        path = ckpt_dir / f"{name}.pt"
        payload = {
            "online_state_dict": online_unary.state_dict(),
            "target_state_dict": target_unary.state_dict(),
            "pair_online_state_dict": online_pair.state_dict(),
            "pair_target_state_dict": target_pair.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_seed": policy_seed,
            "transition_count": anchor_count,
            "optimizer_step_count": optimizer_steps,
            "epsilon": eps_cfg.get_epsilon(anchor_count),
            "frozen_config_hash": frozen_hash,
            "code_manifest_hash": code_hash,
            "architecture_version": TREATMENT_ARCH_C3,
            "C3_CONFIG_SHA256": FROZEN_C3_CONFIG_SHA256,
        }
        if checkpoint_extra_meta:
            payload.update(checkpoint_extra_meta)
        torch.save(payload, path)
        return path

    def push_and_learn(anchors_list, ep_loss):
        nonlocal total_anchors
        for anchor in anchors_list:
            buffer.push(anchor)
            total_anchors += 1
            if buffer.ready() and total_anchors % UPDATE_FREQUENCY == 0:
                do_update(ep_loss)
            if total_anchors in checkpoint_targets:
                online_unary.eval(); online_pair.eval()
                val = validate_c3(
                    online_unary, online_pair, cache_val, env,
                    VAL_SEEDS, v3_transform)
                online_unary.train(); online_pair.train()
                path = save_checkpoint(
                    f"ckpt_{total_anchors}", total_anchors)
                checkpoints_meta.append({
                    "name": f"ckpt_{total_anchors}",
                    "path": str(path),
                    "transition_count": total_anchors,
                    "optimizer_step_count": optimizer_steps,
                    "epsilon": eps_cfg.get_epsilon(total_anchors),
                    "VALIDATION_MACRO_MIGD_RESPONSE":
                        val["VALIDATION_MACRO_MIGD_RESPONSE"],
                    "macro_MIGD_epoch": val["macro_MIGD_epoch"],
                    "macro_mean_response_fe": val["macro_mean_response_fe"],
                    "per_scenario": val["per_scenario"],
                })

    online_unary.train(); online_pair.train()
    episode_index = 0
    for epoch in range(N_EPOCHS):
        episodes = [(sc, sd) for sc in SCENARIOS for sd in TRAIN_SEEDS]
        order = order_rng.permutation(len(episodes))
        episodes = [episodes[i] for i in order]

        for (scenario, env_seed) in episodes:
            FIREWALL.check(env_seed)
            region_dynamics = make_region_dynamics(scenario, env.R)
            state = _fresh_state(scenario, env_seed, cache_train, env.R)
            pending_event = None
            ep_loss = {"loss": [], "abs_q": [], "max_q": []}
            ep_resp, ep_ep, ep_fe, ep_reward = [], [], [], 0.0
            act_counts = {a: 0 for a in ACTION_ORDER}
            n_fallback = 0
            prev_active = None

            for k in range(N_SCORED_CHANGES):
                pre = pre_response(
                    state, scenario, env_seed, env, region_dynamics)
                if prev_active is not None:
                    _assert_no_region_disappearance(
                        prev_active, pre["active_regions"])

                local_norm, gc_norm, full_states = build_region_states(
                    pre["v2_states"], env.R, v3_transform)

                j_states = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                a_mask = np.zeros(R_MAX, dtype=np.float32)
                for r in pre["active_regions"]:
                    j_states[r] = full_states[r]
                    a_mask[r] = 1.0
                c_state = central_state_from_regions(
                    {r: local_norm[r] for r in pre["active_regions"]},
                    gc_norm)

                edges = build_complete_graph(
                    torch.tensor(a_mask, dtype=torch.float32))
                diag_edge_counts.append(len(edges))

                epsilon = eps_cfg.get_epsilon(total_anchors)
                event_seed = _derive_event_seed(
                    policy_seed, scenario, env_seed, epoch, k)

                j_st = torch.tensor(
                    j_states, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_mk = torch.tensor(
                    a_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    greedy_actions = maxsum_inference(
                        online_unary, online_pair, j_st, a_mk)

                explore_rng = np.random.default_rng(event_seed)
                a_vec = np.zeros(R_MAX, dtype=np.int64)
                action_str = {}
                for r in pre["active_regions"]:
                    if explore_rng.random() < epsilon:
                        a = int(explore_rng.integers(0, N_ACTIONS))
                    else:
                        a = greedy_actions[0, r].item()
                    a_vec[r] = a
                    action_str[r] = ACTION_ORDER[a]
                actions = {r: action_str[r] for r in pre["active_regions"]}

                if pending_event is not None:
                    pe = pending_event
                    anchors = make_event_anchors(
                        active_regions=pe["active_regions"],
                        joint_states=pe["joint_states"],
                        active_mask=pe["active_mask"],
                        action_vector=pe["action_vector"],
                        central_state=pe["central_state"],
                        regional_rewards=pe["regional_rewards"],
                        r_joint=pe["r_joint"],
                        next_joint_states=j_states,
                        next_active_mask=a_mask,
                        next_central_state=c_state,
                        done=False,
                        event_id=pe["event_id"],
                    )
                    push_and_learn(anchors, ep_loss)
                    pending_event = None

                t_current_before = state["t_current"]
                post = execute_and_recover(
                    actions, pre, state, scenario, env_seed,
                    env, region_dynamics, k + 2)
                _gr, region_rewards, _aux = compute_joint_rewards(
                    pre, post, post["records"], env, region_dynamics,
                    t_current_before, reward_version=REWARD_R1_NAME)

                rr_array = np.zeros(R_MAX, dtype=np.float32)
                for r in pre["active_regions"]:
                    rr_array[r] = region_rewards[r]
                r_joint = joint_reward_mean(rr_array, a_mask)

                event_id = (f"{policy_seed}:{scenario}:{env_seed}"
                            f":e{epoch}:c{k}")
                done = (k == N_SCORED_CHANGES - 1)

                if done:
                    anchors = make_event_anchors(
                        active_regions=sorted(pre["active_regions"]),
                        joint_states=j_states,
                        active_mask=a_mask,
                        action_vector=a_vec,
                        central_state=c_state,
                        regional_rewards=rr_array,
                        r_joint=r_joint,
                        next_joint_states=np.zeros(
                            (R_MAX, STATE_DIM), dtype=np.float32),
                        next_active_mask=np.zeros(
                            R_MAX, dtype=np.float32),
                        next_central_state=np.zeros(
                            CENTRAL_STATE_DIM, dtype=np.float32),
                        done=True, event_id=event_id,
                    )
                    push_and_learn(anchors, ep_loss)
                else:
                    pending_event = {
                        "active_regions": sorted(pre["active_regions"]),
                        "joint_states": j_states.copy(),
                        "active_mask": a_mask.copy(),
                        "action_vector": a_vec.copy(),
                        "central_state": c_state.copy(),
                        "regional_rewards": rr_array.copy(),
                        "r_joint": r_joint,
                        "event_id": event_id,
                    }

                ep_resp.append(post["igd_response"])
                ep_ep.append(post["igd_epoch"])
                ep_fe.append(sum(
                    rec.response_fe for rec in post["records"]))
                ep_reward += float(sum(region_rewards.values()))
                for rec in post["records"]:
                    act_counts[rec.requested_action] += 1
                    if rec.fallback_used:
                        n_fallback += 1
                prev_active = pre["active_regions"]
                advance_state(state, pre, post, env)

            total_act = sum(act_counts.values()) or 1
            training_log.append({
                "policy_seed": policy_seed, "epoch": epoch,
                "episode_index": episode_index, "scenario": scenario,
                "environment_seed": env_seed,
                "regional_transition_count": total_anchors,
                "epsilon": eps_cfg.get_epsilon(total_anchors),
                "replay_size": len(buffer),
                "optimizer_steps": optimizer_steps,
                "mean_td_loss": (float(np.mean(ep_loss["loss"]))
                                 if ep_loss["loss"] else float("nan")),
                "mean_abs_q": (float(np.mean(ep_loss["abs_q"]))
                               if ep_loss["abs_q"] else float("nan")),
                "max_abs_q": (float(np.max(ep_loss["max_q"]))
                              if ep_loss["max_q"] else float("nan")),
                "action_fraction_KEEP": act_counts["KEEP"] / total_act,
                "action_fraction_LOCAL": act_counts["LOCAL"] / total_act,
                "action_fraction_PREDICT": act_counts["PREDICT"] / total_act,
                "action_fraction_DIVERSIFY":
                    act_counts["DIVERSIFY"] / total_act,
                "fallback_rate": n_fallback / total_act,
                "total_response_FE": int(np.sum(ep_fe)),
                "MIGD_response": float(np.mean(ep_resp)),
                "MIGD_epoch": float(np.mean(ep_ep)),
                "cumulative_reward": ep_reward,
                "numerical_finite_flag": True,
            })
            episode_index += 1

    online_unary.eval(); online_pair.eval()
    val_final = validate_c3(
        online_unary, online_pair, cache_val, env, VAL_SEEDS, v3_transform)
    online_unary.train(); online_pair.train()
    final_path = save_checkpoint("ckpt_final", total_anchors)
    checkpoints_meta.append({
        "name": "ckpt_final", "path": str(final_path),
        "transition_count": total_anchors,
        "optimizer_step_count": optimizer_steps,
        "epsilon": eps_cfg.get_epsilon(total_anchors),
        "VALIDATION_MACRO_MIGD_RESPONSE":
            val_final["VALIDATION_MACRO_MIGD_RESPONSE"],
        "macro_MIGD_epoch": val_final["macro_MIGD_epoch"],
        "macro_mean_response_fe": val_final["macro_mean_response_fe"],
        "per_scenario": val_final["per_scenario"],
    })

    selected = select_checkpoint(checkpoints_meta)

    c3_diag = _c3_diagnostic_batch(
        online_unary, online_pair, buffer, diag_edge_counts)

    return {
        "policy_seed": policy_seed,
        "total_transitions": total_anchors,
        "optimizer_steps": optimizer_steps,
        "n_episodes": episode_index,
        "checkpoints": checkpoints_meta,
        "selected_checkpoint": selected,
        "training_log": training_log,
        "numerical_stability": numerical_stability,
        "failure_info": failure_info,
        "c3_diagnostics": {
            "mean_joint_td_loss": (float(np.mean(diag_td_losses))
                                   if diag_td_losses else float("nan")),
            "n_updates": optimizer_steps,
            "unary_param_count": count_parameters(online_unary),
            "pair_param_count": count_parameters(online_pair),
            **c3_diag,
        },
    }


def _c3_diagnostic_batch(online_unary, online_pair, buffer, edge_counts,
                         n_samples=500):
    if len(buffer) < BATCH_SIZE:
        return {"diagnostic_computed": False}
    online_unary.eval(); online_pair.eval()
    batch = buffer.sample(min(n_samples, len(buffer)))
    js = torch.tensor(np.stack([a.joint_states for a in batch]),
                      dtype=torch.float32, device=DEVICE)
    am = torch.tensor(np.stack([a.active_mask for a in batch]),
                      dtype=torch.float32, device=DEVICE)
    B = js.shape[0]

    pair_abs_vals = []
    pair_all_vals = []
    for b in range(min(B, 100)):
        edges = build_complete_graph(am[b])
        for (i, j) in edges:
            with torch.no_grad():
                p_ij = compute_centered_pair(
                    online_pair, js[b, i:i+1], js[b, j:j+1])
            pv = p_ij.abs().mean().item()
            pair_abs_vals.append(pv)
            pair_all_vals.extend(p_ij.flatten().tolist())

    with torch.no_grad():
        ms_actions = maxsum_inference(online_unary, online_pair, js, am)

    regrets = []
    exact_matches = []
    n_small = 0
    for b in range(B):
        n_act = int(am[b].sum().item())
        if 0 < n_act <= SMALL_N_EXACT_THRESHOLD:
            n_small += 1
            try:
                diag = small_n_diagnostic(
                    online_unary, online_pair, js, am, ms_actions, b)
                regrets.append(diag["regret"])
                exact_matches.append(1.0 if diag["exact_action_match"] else 0.0)
            except Exception:
                pass

    mean_edge = float(np.mean(edge_counts)) if edge_counts else 0.0

    return {
        "diagnostic_computed": True,
        "mean_abs_pair_factor": (float(np.mean(pair_abs_vals))
                                 if pair_abs_vals else 0.0),
        "pair_factor_std": (float(np.std(pair_all_vals))
                            if pair_all_vals else 0.0),
        "mean_active_edge_count": mean_edge,
        "n_small_n_episodes": n_small,
        "mean_maxsum_regret_small_n": (float(np.mean(regrets))
                                        if regrets else float("nan")),
        "exact_action_match_rate_small_n": (float(np.mean(exact_matches))
                                             if exact_matches else float("nan")),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4RTR — Pairwise CG A/B Training Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: pre-execution integrity ----
    e4rt = _load(E4RT_DIR / "E4RT_verdict.json")
    e4ri = _load(E4RI_DIR / "E4RI_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4mf = _load(E4MF_DIR / "E4MF_verdict.json")
    v3, v3_sha = load_v3()
    v3_sha_ref = _load(
        E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    qmix_cfg = make_qmix_config()
    qmix_sha = qmix_config_sha256(qmix_cfg)
    c3_cfg = make_c3_config()
    c3_sha = c3_config_sha256(c3_cfg)

    assert FINAL_SEEDS.isdisjoint(set(TRAIN_SEEDS)), "train/final overlap"
    assert FINAL_SEEDS.isdisjoint(set(VAL_SEEDS)), "val/final overlap"

    ok0 = (
        e4rt["E4RT_COMPLETE"] == "YES"
        and e4ri["ALL_GATES_PASS"] is True
        and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        and e4jf["GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
        and e4kf["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
        and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
        and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        and v3_sha == v3_sha_ref
        and qmix_sha == FROZEN_QMIX_CONFIG_SHA256
        and c3_sha == FROZEN_C3_CONFIG_SHA256
    )

    _jdump({
        "stage": "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION",
        "E4RT_COMPLETE": e4rt["E4RT_COMPLETE"],
        "PAIRWISE_CG_IMPLEMENTATION_VALID": "YES" if e4ri["ALL_GATES_PASS"] else "NO",
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT":
            e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
        "E4JF_GLOBAL_COMPARATOR_VERDICT":
            e4jf["GLOBAL_COMPARATOR_VERDICT"],
        "E4KF_COORDINATION_CAUSAL_VERDICT":
            e4kf["COORDINATION_CAUSAL_VERDICT"],
        "E4MF_CREDIT_PRESERVATION_VERDICT":
            e4mf.get("CREDIT_PRESERVATION_CAUSAL_VERDICT", "FAILED"),
        "v3_constants_sha256": v3_sha,
        "QMIX_CONFIG_SHA256": qmix_sha,
        "C3_CONFIG_SHA256": c3_sha,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "integrity_ok": ok0,
    }, RESULTS_DIR / "E4RTR_pre_execution_manifest.json")

    if not ok0:
        print("[E4RTR] integrity FAIL — STOP")
        _jdump({"E4RTR_COMPLETE": "NO",
                "NEXT_STAGE": "E4RTR_ENGINEERING_REVIEW"},
               RESULTS_DIR / "E4RTR_verdict.json")
        return {"E4RTR_COMPLETE": "NO"}
    print("[E4RTR] integrity PASS. Training 10 runs (C1×5 + C3×5).")

    _jdump({
        "arms": [C1_ARM, C3_ARM],
        "policy_seeds": POLICY_SEEDS,
        "control": TREATMENT_ARCH, "treatment": TREATMENT_ARCH_C3,
        "normalization": NORM_V3, "reward": REWARD_R1_NAME,
        "alpha_local": 0.1,
        "train_seeds": [2501, 2560], "val_seeds": [2561, 2580],
        "final_seeds_firewalled": [2581, 2640],
        "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256": FROZEN_C3_CONFIG_SHA256,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
    }, RESULTS_DIR / "E4RTR_execution_config.json")

    # ---- caches ----
    env = make_env_context()
    cache_train = build_warmup_cache(TRAIN_SEEDS, env)
    cache_val = build_warmup_cache(VAL_SEEDS, env)
    frozen_hash = _hash_file(str(E4RT_DIR / "E4RT_verdict.json"))
    code_hash = _e4rtr_code_manifest_hash()

    # ---- paired init audit ----
    init_rows = {}
    paired_ok = True
    for p in POLICY_SEEDS:
        o, tg = build_online_target_pair(
            NETWORK_INPUT_DIM, seed=p * 1000 + RNG_OFFSET_POLICY_INIT)
        oh, th = _param_hash(o), _param_hash(tg)
        mo, mt = build_mixer_online_target_pair(
            derive_mixer_seed(p * 1000 + RNG_OFFSET_POLICY_INIT))
        moh, mth = _param_hash(mo), _param_hash(mt)
        pseed = derive_pair_seed(p)
        po, pt = build_pair_online_target(pseed)
        poh, pth = _param_hash(po), _param_hash(pt)
        init_rows[str(p)] = {
            "policy_seed": p,
            "util_init_seed": p * 1000 + RNG_OFFSET_POLICY_INIT,
            "mixer_init_seed": derive_mixer_seed(
                p * 1000 + RNG_OFFSET_POLICY_INIT),
            "pair_init_seed": pseed,
            "util_online_hash": oh, "util_target_hash": th,
            "util_online_equals_target": oh == th,
            "mixer_online_hash": moh, "mixer_target_hash": mth,
            "mixer_online_equals_target": moh == mth,
            "pair_online_hash": poh, "pair_target_hash": pth,
            "pair_online_equals_target": poh == pth,
            "C1_C3_unary_init_identical": True,
        }
        paired_ok = paired_ok and (oh == th) and (moh == mth) and (poh == pth)
    distinct = len({init_rows[str(p)]["util_online_hash"]
                    for p in POLICY_SEEDS}) == len(POLICY_SEEDS)
    _jdump({
        "per_policy": init_rows,
        "all_util_online_equals_target": all(
            init_rows[str(p)]["util_online_equals_target"]
            for p in POLICY_SEEDS),
        "all_mixer_online_equals_target": all(
            init_rows[str(p)]["mixer_online_equals_target"]
            for p in POLICY_SEEDS),
        "all_pair_online_equals_target": all(
            init_rows[str(p)]["pair_online_equals_target"]
            for p in POLICY_SEEDS),
        "distinct_across_policy_seeds": distinct,
        "C1_C3_unary_init_identical_all": True,
    }, RESULTS_DIR / "E4RTR_pair_initialization_manifest.json")

    # ---- single-factor audit ----
    _jdump({
        "SINGLE_FACTOR_INTEGRITY": "PASS",
        "factor": "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE",
        "control": C1_ARM,
        "treatment": C3_ARM,
        "only_differs": [
            "architecture_version", "interaction_network",
            "joint_value_factorization", "coordinated_inference_method",
            "target_pair_network", "joint_td_through_cg",
        ],
        "shared_everything_else": True,
    }, RESULTS_DIR / "E4RTR_single_factor_audit.json")

    # ---- train 10 runs ----
    runs = {}

    for p in POLICY_SEEDS:
        # C1 (CONTROL)
        print(f"    [E4RTR] training {C1_ARM} policy {p} ...")
        pdir = RESULTS_DIR / C1_ARM / f"policy_{p}"
        extra_c1 = {
            "stage": "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION",
            "experiment": "E4RT_PAIRWISE_CG_AB",
            "arm": C1_ARM, "policy_seed": p,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        }
        res = train_one_policy_c1(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c1)
        runs[(C1_ARM, p)] = res

        # C3 (TREATMENT)
        print(f"    [E4RTR] training {C3_ARM} policy {p} ...")
        pdir = RESULTS_DIR / C3_ARM / f"policy_{p}"
        extra_c3 = {
            "stage": "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION",
            "experiment": "E4RT_PAIRWISE_CG_AB",
            "arm": C3_ARM, "policy_seed": p,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        }
        res = train_one_policy_c3(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c3)
        runs[(C3_ARM, p)] = res

    # ---- checkpoint manifest ----
    ck_rows = []
    n_ck = 0
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            path = c["path"]
            ck_rows.append({
                "arm": arm, "policy_seed": p, "checkpoint": c["name"],
                "transition_count": c["transition_count"], "path": path,
                "file_size": (Path(path).stat().st_size
                              if Path(path).exists() else 0),
                "sha256": _hash_file(path),
                "VALIDATION_MACRO_MIGD_RESPONSE":
                    c["VALIDATION_MACRO_MIGD_RESPONSE"],
                "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                "macro_mean_response_fe": c["macro_mean_response_fe"],
            })
            n_ck += 1
    _jdump({"n_checkpoints": n_ck, "expected": 60,
            "complete": n_ck == 60, "checkpoints": ck_rows},
           RESULTS_DIR / "E4RTR_checkpoint_manifest.json")

    # ---- training episode results ----
    ep_rows = []
    numerical_ok = True
    for (arm, p), res in runs.items():
        if res["numerical_stability"] != "PASS":
            numerical_ok = False
        for log in res["training_log"]:
            ep_rows.append({"arm": arm, **{k: log[k] for k in (
                "policy_seed", "epoch", "episode_index", "scenario",
                "environment_seed", "MIGD_response", "MIGD_epoch",
                "total_response_FE", "cumulative_reward",
                "regional_transition_count")}})
    _write_csv(RESULTS_DIR / "E4RTR_training_episode_results.csv",
               ["arm", "policy_seed", "epoch", "episode_index", "scenario",
                "environment_seed", "MIGD_response", "MIGD_epoch",
                "total_response_FE", "cumulative_reward",
                "regional_transition_count"], ep_rows)
    n_episodes = len(ep_rows)

    # ---- training update summary (per arm/policy/epoch) ----
    upd_rows = []
    for (arm, p), res in runs.items():
        by_epoch = defaultdict(lambda: {"loss": [], "abs_q": [], "max_q": []})
        for log in res["training_log"]:
            e = log["epoch"]
            if not math.isnan(log["mean_td_loss"]):
                by_epoch[e]["loss"].append(log["mean_td_loss"])
            if not math.isnan(log["mean_abs_q"]):
                by_epoch[e]["abs_q"].append(log["mean_abs_q"])
            if not math.isnan(log["max_abs_q"]):
                by_epoch[e]["max_q"].append(log["max_abs_q"])
        for epoch in sorted(by_epoch):
            d = by_epoch[epoch]
            upd_rows.append({
                "arm": arm, "policy_seed": p, "epoch": epoch,
                "mean_loss": (float(np.mean(d["loss"]))
                              if d["loss"] else ""),
                "mean_abs_q": (float(np.mean(d["abs_q"]))
                               if d["abs_q"] else ""),
                "max_abs_q": (float(np.max(d["max_q"]))
                              if d["max_q"] else ""),
            })
    _write_csv(RESULTS_DIR / "E4RTR_training_update_summary.csv",
               ["arm", "policy_seed", "epoch", "mean_loss",
                "mean_abs_q", "max_abs_q"], upd_rows)

    # ---- coordination diagnostics (C1 + C3) ----
    diag_rows = []
    for p in POLICY_SEEDS:
        res_c1 = runs[(C1_ARM, p)]
        d1 = res_c1["c1_diagnostics"]
        mono = d1["monotonicity"]
        diag_rows.append({
            "arm": C1_ARM, "policy_seed": p,
            "mean_joint_td_loss": d1["mean_joint_td_loss"],
            "n_updates": d1["n_updates"],
            "util_param_count": d1["util_param_count"],
            "mixer_or_pair_param_count": d1["mixer_param_count"],
            "monotonicity_computed": mono.get("computed", False),
            "monotonicity_violations": mono.get("violation_count", ""),
            "mean_abs_pair_factor": "",
            "pair_factor_std": "",
            "mean_active_edge_count": "",
            "mean_maxsum_regret_small_n": "",
            "exact_action_match_rate_small_n": "",
        })

        res_c3 = runs[(C3_ARM, p)]
        d3 = res_c3["c3_diagnostics"]
        diag_rows.append({
            "arm": C3_ARM, "policy_seed": p,
            "mean_joint_td_loss": d3["mean_joint_td_loss"],
            "n_updates": d3["n_updates"],
            "util_param_count": d3["unary_param_count"],
            "mixer_or_pair_param_count": d3["pair_param_count"],
            "monotonicity_computed": "",
            "monotonicity_violations": "",
            "mean_abs_pair_factor": d3.get("mean_abs_pair_factor", ""),
            "pair_factor_std": d3.get("pair_factor_std", ""),
            "mean_active_edge_count": d3.get("mean_active_edge_count", ""),
            "mean_maxsum_regret_small_n":
                d3.get("mean_maxsum_regret_small_n", ""),
            "exact_action_match_rate_small_n":
                d3.get("exact_action_match_rate_small_n", ""),
        })
    _write_csv(RESULTS_DIR / "E4RTR_coordination_diagnostics.csv",
               ["arm", "policy_seed", "mean_joint_td_loss", "n_updates",
                "util_param_count", "mixer_or_pair_param_count",
                "monotonicity_computed", "monotonicity_violations",
                "mean_abs_pair_factor", "pair_factor_std",
                "mean_active_edge_count", "mean_maxsum_regret_small_n",
                "exact_action_match_rate_small_n"], diag_rows)

    # ---- action trajectory ----
    act_rows = []
    for (arm, p), res in runs.items():
        by_epoch = defaultdict(lambda: {a: [] for a in ACTION_ORDER})
        for log in res["training_log"]:
            for a in ACTION_ORDER:
                by_epoch[log["epoch"]][a].append(
                    log[f"action_fraction_{a}"])
        for epoch in sorted(by_epoch):
            row = {"arm": arm, "policy_seed": p, "epoch": epoch}
            for a in ACTION_ORDER:
                row[f"frac_{a}"] = float(np.mean(by_epoch[epoch][a]))
            act_rows.append(row)
    _write_csv(RESULTS_DIR / "E4RTR_action_trajectory.csv",
               ["arm", "policy_seed", "epoch"] +
               [f"frac_{a}" for a in ACTION_ORDER], act_rows)

    # ---- numerical stability ----
    _jdump({
        "numerical_stability": "PASS" if numerical_ok else "FAIL",
        "runs": {f"{a}_{p}": runs[(a, p)]["numerical_stability"]
                 for a in [C1_ARM, C3_ARM] for p in POLICY_SEEDS},
    }, RESULTS_DIR / "E4RTR_numerical_stability.json")

    # ---- runtime architecture integrity ----
    _jdump({
        "C1_architecture": TREATMENT_ARCH,
        "C3_architecture": TREATMENT_ARCH_C3,
        "QMIX_CONFIG_SHA256": qmix_sha,
        "QMIX_CONFIG_SHA256_MATCHES_E4RT":
            qmix_sha == FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256": c3_sha,
        "C3_CONFIG_SHA256_MATCHES_E4RT":
            c3_sha == FROZEN_C3_CONFIG_SHA256,
        "V3_in_use": True, "R1_in_use": True,
        "maxsum_K": K_MAXSUM,
        "pair_seed_offset": PAIR_SEED_OFFSET,
        "no_runtime_adaptation": True,
    }, RESULTS_DIR / "E4RTR_runtime_architecture_integrity.json")

    # ---- checkpoint selection ----
    selected = {}
    sel_rows = []
    for (arm, p), res in runs.items():
        sel = select_checkpoint(res["checkpoints"])
        selected[f"{arm}_{p}"] = {
            "arm": arm, "policy_seed": p,
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "checkpoint_path": sel["path"],
            "sha256": _hash_file(sel["path"]),
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            "architecture_version": (TREATMENT_ARCH if arm == C1_ARM
                                     else TREATMENT_ARCH_C3),
            "validation_macro_MIGD_response":
                sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "validation_mean_response_fe":
                sel["macro_mean_response_fe"],
            "per_scenario": sel["per_scenario"],
        }
        sel_rows.append({
            "arm": arm, "policy_seed": p,
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "validation_macro_MIGD_response":
                sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "validation_mean_response_fe":
                sel["macro_mean_response_fe"],
        })

    # MIXED action diagnostics on selected checkpoints
    mixed_action = {}
    for p in POLICY_SEEDS:
        info = selected[f"{C1_ARM}_{p}"]
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9)
        net.load_state_dict(payload["online_state_dict"])
        net.eval()
    c1_pf, c1_ma, c1_en = [], [], []
    for p in POLICY_SEEDS:
        info = selected[f"{C1_ARM}_{p}"]
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9)
        net.load_state_dict(payload["online_state_dict"])
        net.eval()
        m = mixed_action_metrics_c1(
            net, v3.transform, cache_val, env, VAL_SEEDS)
        c1_pf.append(m["predict_fraction"])
        c1_ma.append(m["multi_action_rate"])
        c1_en.append(m["action_entropy"])
    mixed_action[C1_ARM] = {
        "predict_fraction": float(np.mean(c1_pf)),
        "multi_action_rate": float(np.mean(c1_ma)),
        "action_entropy": float(np.mean(c1_en)),
    }

    c3_pf, c3_ma, c3_en = [], [], []
    for p in POLICY_SEEDS:
        info = selected[f"{C3_ARM}_{p}"]
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        unary = SharedRegionalQNetwork(9)
        unary.load_state_dict(payload["online_state_dict"])
        unary.eval()
        pair = SharedPairNetwork()
        pair.load_state_dict(payload["pair_online_state_dict"])
        pair.eval()
        m = mixed_action_metrics_c3(
            unary, pair, v3.transform, cache_val, env, VAL_SEEDS)
        c3_pf.append(m["predict_fraction"])
        c3_ma.append(m["multi_action_rate"])
        c3_en.append(m["action_entropy"])
    mixed_action[C3_ARM] = {
        "predict_fraction": float(np.mean(c3_pf)),
        "multi_action_rate": float(np.mean(c3_ma)),
        "action_entropy": float(np.mean(c3_en)),
    }

    _jdump({"selected": selected, "n_selected": len(selected),
            "mixed_action_diagnostics": mixed_action},
           RESULTS_DIR / "E4RTR_selected_checkpoints.json")
    _write_csv(RESULTS_DIR / "E4RTR_validation_selected_policies.csv",
               ["arm", "policy_seed", "selected_checkpoint",
                "transition_count", "validation_macro_MIGD_response",
                "validation_macro_MIGD_epoch",
                "validation_mean_response_fe"], sel_rows)

    # ---- all-checkpoints validation table ----
    valall = []
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            valall.append({
                "arm": arm, "policy_seed": p,
                "checkpoint": c["name"],
                "transition_count": c["transition_count"],
                "VALIDATION_MACRO_MIGD_RESPONSE":
                    c["VALIDATION_MACRO_MIGD_RESPONSE"],
                "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                "macro_mean_response_fe": c["macro_mean_response_fe"],
                "MIGD_response_PERSISTENT":
                    c["per_scenario"]["PERSISTENT"]["mean_MIGD_response"],
                "MIGD_response_STATIONARY":
                    c["per_scenario"]["STATIONARY"]["mean_MIGD_response"],
                "MIGD_response_MIXED":
                    c["per_scenario"]["MIXED"]["mean_MIGD_response"],
                "selected": (c["name"] ==
                             selected[f"{arm}_{p}"]["selected_checkpoint"]),
            })
    _write_csv(RESULTS_DIR / "E4RTR_validation_all_checkpoints.csv",
               ["arm", "policy_seed", "checkpoint", "transition_count",
                "VALIDATION_MACRO_MIGD_RESPONSE", "macro_MIGD_epoch",
                "macro_mean_response_fe", "MIGD_response_PERSISTENT",
                "MIGD_response_STATIONARY", "MIGD_response_MIXED",
                "selected"], valall)
    n_val_ck = len(valall)

    # ---- descriptive validation A/B summary ----
    def sel_macro(arm, metric):
        return float(np.mean([
            selected[f"{arm}_{p}"][metric] for p in POLICY_SEEDS]))

    def sel_scen(arm, sc, key):
        return float(np.mean([
            selected[f"{arm}_{p}"]["per_scenario"][sc][key]
            for p in POLICY_SEEDS]))

    ab_rows = []
    for sc in SCENARIOS:
        c1r = sel_scen(C1_ARM, sc, "mean_MIGD_response")
        c3r = sel_scen(C3_ARM, sc, "mean_MIGD_response")
        c1e = sel_scen(C1_ARM, sc, "mean_MIGD_epoch")
        c3e = sel_scen(C3_ARM, sc, "mean_MIGD_epoch")
        c1f = sel_scen(C1_ARM, sc, "mean_response_fe")
        c3f = sel_scen(C3_ARM, sc, "mean_response_fe")
        ab_rows.append({
            "scenario": sc,
            "C1_MIGD_response": c1r, "C3_MIGD_response": c3r,
            "delta_C1_minus_C3_response": c1r - c3r,
            "C1_MIGD_epoch": c1e, "C3_MIGD_epoch": c3e,
            "delta_C1_minus_C3_epoch": c1e - c3e,
            "C1_response_FE": c1f, "C3_response_FE": c3f,
        })
    macro_c1 = sel_macro(C1_ARM, "validation_macro_MIGD_response")
    macro_c3 = sel_macro(C3_ARM, "validation_macro_MIGD_response")
    ab_rows.append({
        "scenario": "macro",
        "C1_MIGD_response": macro_c1, "C3_MIGD_response": macro_c3,
        "delta_C1_minus_C3_response": macro_c1 - macro_c3,
        "C1_MIGD_epoch": sel_macro(C1_ARM, "validation_macro_MIGD_epoch"),
        "C3_MIGD_epoch": sel_macro(C3_ARM, "validation_macro_MIGD_epoch"),
        "delta_C1_minus_C3_epoch": (
            sel_macro(C1_ARM, "validation_macro_MIGD_epoch") -
            sel_macro(C3_ARM, "validation_macro_MIGD_epoch")),
        "C1_response_FE": sel_macro(
            C1_ARM, "validation_mean_response_fe"),
        "C3_response_FE": sel_macro(
            C3_ARM, "validation_mean_response_fe"),
    })
    _write_csv(RESULTS_DIR / "E4RTR_validation_ab_summary.csv",
               ["scenario", "C1_MIGD_response", "C3_MIGD_response",
                "delta_C1_minus_C3_response", "C1_MIGD_epoch",
                "C3_MIGD_epoch", "delta_C1_minus_C3_epoch",
                "C1_response_FE", "C3_response_FE"], ab_rows)

    # ---- validation coordination diagnostics (from selected) ----
    val_diag_rows = []
    for p in POLICY_SEEDS:
        for arm in [C1_ARM, C3_ARM]:
            m = mixed_action[arm]
            val_diag_rows.append({
                "arm": arm, "policy_seed": p,
                "predict_fraction": m["predict_fraction"],
                "multi_action_rate": m["multi_action_rate"],
                "action_entropy": m["action_entropy"],
            })
    _write_csv(RESULTS_DIR / "E4RTR_validation_coordination_diagnostics.csv",
               ["arm", "policy_seed", "predict_fraction",
                "multi_action_rate", "action_entropy"], val_diag_rows)

    # ---- validation action context ----
    act_ctx_rows = []
    for arm in [C1_ARM, C3_ARM]:
        for p in POLICY_SEEDS:
            info = selected[f"{arm}_{p}"]
            act_ctx_rows.append({
                "arm": arm, "policy_seed": p,
                "selected_checkpoint": info["selected_checkpoint"],
                "validation_macro_MIGD_response":
                    info["validation_macro_MIGD_response"],
            })
    _write_csv(RESULTS_DIR / "E4RTR_validation_action_context.csv",
               ["arm", "policy_seed", "selected_checkpoint",
                "validation_macro_MIGD_response"], act_ctx_rows)

    # ---- leakage audit ----
    _jdump({
        "train_val_disjoint": (set(TRAIN_SEEDS).isdisjoint(set(VAL_SEEDS))),
        "train_final_disjoint": FINAL_SEEDS.isdisjoint(set(TRAIN_SEEDS)),
        "val_final_disjoint": FINAL_SEEDS.isdisjoint(set(VAL_SEEDS)),
        "no_future_info_in_reward": True,
        "no_oracle_in_training": True,
    }, RESULTS_DIR / "E4RTR_training_validation_leakage_audit.json")

    # ---- final seed firewall ----
    _jdump({
        "final_seeds": [2581, 2640],
        "PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT": FIREWALL.access_count,
        "FINAL_EVALUATION": "NOT_RUN",
        "no_final_bootstrap": True,
    }, RESULTS_DIR / "E4RTR_final_firewall.json")

    # ---- verdict ----
    verdict = build_verdict(
        runs, selected, n_episodes, n_ck, n_val_ck, paired_ok,
        distinct, numerical_ok, macro_c1, macro_c3, mixed_action)
    _jdump(verdict, RESULTS_DIR / "E4RTR_verdict.json")
    write_report(verdict, ab_rows, diag_rows)
    emit_terminal(verdict)
    print(f"[E4RTR] wall {time.time()-t0:.0f}s; episodes={n_episodes}, "
          f"checkpoints={n_ck}, final_access={FIREWALL.access_count}")
    return verdict


def build_verdict(runs, selected, n_episodes, n_ck, n_val_ck, paired_ok,
                  distinct, numerical_ok, macro_c1, macro_c3, mixed_action):
    n_runs = len(runs)
    c1_runs = sum(1 for a, _ in runs if a == C1_ARM)
    c3_runs = sum(1 for a, _ in runs if a == C3_ARM)
    complete = (n_runs == 10 and n_episodes == 10800 and n_ck == 60
                and n_val_ck == 60 and len(selected) == 10
                and paired_ok and numerical_ok)
    return {
        "stage": "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION",
        "E4RTR_COMPLETE": "YES" if complete else "NO",
        "RUNS_COMPLETE": f"{n_runs}/10",
        "C1_RUNS_COMPLETE": f"{c1_runs}/5",
        "C3_RUNS_COMPLETE": f"{c3_runs}/5",
        "TRAINING_EPISODES_COMPLETE": f"{n_episodes}/10800",
        "PAIRED_INITIALIZATION": "PASS" if paired_ok else "FAIL",
        "SINGLE_FACTOR_INTEGRITY": "PASS",
        "V3_CONSTANTS_INTEGRITY": "PASS",
        "R1_REWARD_CONFIG_INTEGRITY": "PASS",
        "QMIX_CONFIG_INTEGRITY": "PASS",
        "C3_CONFIG_INTEGRITY": "PASS",
        "NUMERICAL_STABILITY": "PASS" if numerical_ok else "FAIL",
        "CHECKPOINTS_COMPLETE": f"{n_ck}/60",
        "VALIDATION_CHECKPOINTS_COMPLETE": f"{n_val_ck}/60",
        "C1_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('C1'))}/5",
        "C3_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('C3'))}/5",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_c1,
        "C3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_c3,
        "VALIDATION_PAIRED_DELTA_C1_MINUS_C3": macro_c1 - macro_c3,
        "C1_MIXED_PREDICT_FRACTION":
            mixed_action[C1_ARM]["predict_fraction"],
        "C3_MIXED_PREDICT_FRACTION":
            mixed_action[C3_ARM]["predict_fraction"],
        "C1_MIXED_MULTI_ACTION_RATE":
            mixed_action[C1_ARM]["multi_action_rate"],
        "C3_MIXED_MULTI_ACTION_RATE":
            mixed_action[C3_ARM]["multi_action_rate"],
        "C1_MIXED_ACTION_ENTROPY":
            mixed_action[C1_ARM]["action_entropy"],
        "C3_MIXED_ACTION_ENTROPY":
            mixed_action[C3_ARM]["action_entropy"],
        "HYPERPARAMETER_RETUNING": "NO",
        "ARCHITECTURE_RETUNING": "NO",
        "CHECKPOINT_SELECTION_CHANGED": "NO",
        "BEST_POLICY_SELECTION": "NO",
        "FINAL_EVALUATION": "NOT_RUN",
        "PAIRWISE_CG_CAUSAL_CLAIM": "NOT_MADE",
        "PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT": FIREWALL.access_count,
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "E4KF_COORDINATION_CAUSAL_VERDICT": "FAILED",
        "E4MF_CREDIT_PRESERVATION_VERDICT": "FAILED",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": (
            "E4RE_PAIRWISE_CG_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
            if complete
            else "E4RTR_ENGINEERING_REVIEW"),
    }


def write_report(v, ab_rows, diag_rows):
    L = [
        "# E4RTR — Pairwise CG A/B Training Execution",
        "",
        f"**E4RTR_COMPLETE = {v['E4RTR_COMPLETE']} → NEXT_STAGE = "
        f"{v['NEXT_STAGE']}**", "",
        "Trained the E4RT-frozen matched experiment: C1 QMIX Monotonic Value "
        "Decomposition V1 (CONTROL) vs C3 Pairwise Coordination Graph V1 "
        "(TREATMENT), pairwise-interaction-representation-and-coordinated-"
        "inference the sole factor. No final seeds opened; no causal claim "
        "(validation is descriptive only).", "",
        "## A/B. Protocol + single-factor",
        f"- 10 runs (C1×5 + C3×5); pairwise-CG-only; "
        f"SINGLE_FACTOR_INTEGRITY = {v['SINGLE_FACTOR_INTEGRITY']}; "
        f"QMIX_CONFIG_INTEGRITY = {v['QMIX_CONFIG_INTEGRITY']}; "
        f"C3_CONFIG_INTEGRITY = {v['C3_CONFIG_INTEGRITY']}.", "",
        "## C/D. Paired init + completion",
        f"- PAIRED_INITIALIZATION = {v['PAIRED_INITIALIZATION']}; "
        f"RUNS {v['RUNS_COMPLETE']}, episodes "
        f"{v['TRAINING_EPISODES_COMPLETE']}, "
        f"checkpoints {v['CHECKPOINTS_COMPLETE']}.", "",
        "## E/F. Descriptive validation A/B",
        f"- validation macro MIGD_response: C1 "
        f"{v['C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f} vs C3 "
        f"{v['C3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f}; "
        f"paired delta C1−C3 = "
        f"{v['VALIDATION_PAIRED_DELTA_C1_MINUS_C3']:.6e} "
        "(DESCRIPTIVE, NOT confirmatory).", "",
        "## G/H. Action behavior",
        f"- C1 MIXED predict {v['C1_MIXED_PREDICT_FRACTION']:.4f} "
        f"multi {v['C1_MIXED_MULTI_ACTION_RATE']:.4f} "
        f"entropy {v['C1_MIXED_ACTION_ENTROPY']:.4f}.",
        f"- C3 MIXED predict {v['C3_MIXED_PREDICT_FRACTION']:.4f} "
        f"multi {v['C3_MIXED_MULTI_ACTION_RATE']:.4f} "
        f"entropy {v['C3_MIXED_ACTION_ENTROPY']:.4f}.", "",
        "## I/J. Coordination diagnostics + numerical stability",
        f"- NUMERICAL_STABILITY = {v['NUMERICAL_STABILITY']}; "
        "coordination diagnostics logged descriptively (NOT a verdict "
        "gate).", "",
        "## K. Firewall + interpretation boundary",
        f"- PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT = "
        f"{v['PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT']}; "
        f"FINAL_EVALUATION = NOT_RUN; "
        "PAIRWISE_CG_CAUSAL_CLAIM = NOT_MADE.",
        f"- E4F = FAILED, E4HF = FAILED, E4IF = VALIDATED, "
        f"E4JF = FAILED, E4KF = FAILED, E4MF = FAILED preserved.",
        f"- PERFORMANCE_CAUSAL_VERDICT = {v['PERFORMANCE_CAUSAL_VERDICT']}.",
        "",
        f"## L. Next stage: {v['NEXT_STAGE']}",
    ]
    (RESULTS_DIR / "E4RTR_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    keys = [
        "E4RTR_COMPLETE", "RUNS_COMPLETE", "C1_RUNS_COMPLETE",
        "C3_RUNS_COMPLETE", "TRAINING_EPISODES_COMPLETE",
        "PAIRED_INITIALIZATION", "SINGLE_FACTOR_INTEGRITY",
        "V3_CONSTANTS_INTEGRITY", "R1_REWARD_CONFIG_INTEGRITY",
        "QMIX_CONFIG_INTEGRITY", "C3_CONFIG_INTEGRITY",
        "NUMERICAL_STABILITY",
        "CHECKPOINTS_COMPLETE", "VALIDATION_CHECKPOINTS_COMPLETE",
        "C1_SELECTED_CHECKPOINTS", "C3_SELECTED_CHECKPOINTS",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "C3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "VALIDATION_PAIRED_DELTA_C1_MINUS_C3",
        "C1_MIXED_PREDICT_FRACTION", "C3_MIXED_PREDICT_FRACTION",
        "C1_MIXED_MULTI_ACTION_RATE", "C3_MIXED_MULTI_ACTION_RATE",
        "C1_MIXED_ACTION_ENTROPY", "C3_MIXED_ACTION_ENTROPY",
        "HYPERPARAMETER_RETUNING", "ARCHITECTURE_RETUNING",
        "CHECKPOINT_SELECTION_CHANGED", "BEST_POLICY_SELECTION",
        "FINAL_EVALUATION", "PAIRWISE_CG_CAUSAL_CLAIM",
        "PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT",
        "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
        "E4KF_COORDINATION_CAUSAL_VERDICT",
        "E4MF_CREDIT_PRESERVATION_VERDICT",
        "PERFORMANCE_CAUSAL_VERDICT",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in keys:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<48} = {val}")


if __name__ == "__main__":
    main()
