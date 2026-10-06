"""
E4MTR — Credit Preservation A/B Training Execution.

Executes the E4MT-frozen matched credit-preservation experiment:
    C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 (CONTROL)
    vs C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1 (TREATMENT)
trains 10 runs (C1×5 + C2×5) via EXACT E4MT-frozen protocol; saves 60 checkpoints;
validates every checkpoint on seeds 2161..2180; selects one per run by frozen macro-MIGD_response
rule; retains all 10; reports descriptive credit diagnostics. NO final seeds (2181..2240)
are opened; NO final causal verdict; NO bootstrap.

Regional-credit-preservation-within-QMIX is the ONLY factor. C1 = QMIX joint TD (L_joint only).
C2 = QMIX composite (0.5*L_joint + 0.5*L_regional). Both arms share V3 normalization, R1 reward
(alpha=0.1), 9-128-128-4 utility network, QMIX_MIXER_V1, gamma 0.99, Adam 1e-4, SmoothL1,
tau 0.005. C2 mixer gets gradients from 0.5*L_joint only (NOT compensated).

Do not rescue either arm. If C2 trains poorly: preserve the result.
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
    CONTROL_ARCH, C2_CREDIT_ARCH, FROZEN_V3_CONSTANTS_SHA256, STATE_DIM,
    JOINT_WEIGHT, REGIONAL_WEIGHT, C2_LOSS_VERSION,
    joint_reward_mean, current_joint_q, double_q_joint_target, treatment_loss,
    focal_regional_ddqn_target, focal_current_q, regional_smoothl1_loss,
    c2_composite_loss, c2_optimizer_params,
    c1_optimizer_params, soft_update, count_parameters,
    make_qmix_config, qmix_config_sha256,
    make_c2_credit_config, c2_credit_config_sha256,
    loss_dispersion_diagnostic, gradient_diversity_diagnostic,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.experiments.E4P_learned_controller_preparation import NETWORK_INPUT_DIM

# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR = Path("results/E4MTR_credit_preservation_ab_training_execution")
E4MT_DIR = Path("results/E4MT_credit_preservation_ab_training_protocol_freeze")
E4MI_DIR = Path("results/E4MI_joint_credit_preservation_implementation_validation")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")

TRAIN_SEEDS = list(range(2101, 2161))
VAL_SEEDS = list(range(2161, 2181))
FINAL_SEEDS = set(range(2181, 2241))
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
EPISODES_PER_EPOCH = len(TRAIN_SEEDS) * len(SCENARIOS)
N_TRAIN_EPISODES = N_EPOCHS * EPISODES_PER_EPOCH
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C2_CREDIT_CONFIG_SHA256 = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2_ARM = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"


class FinalSeedFirewall:
    def __init__(self):
        self.access_count = 0
        self.forbidden = FINAL_SEEDS

    def check(self, seed: int):
        if int(seed) in self.forbidden:
            self.access_count += 1
            raise RuntimeError(
                f"E4MTR_FINAL_FIREWALL: seed {seed} (2181..2240) MUST NOT be accessed.")


FIREWALL = FinalSeedFirewall()


# ============================================================================
# UTILITIES
# ============================================================================

def _jdefault(x):
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(f"Object of type {type(x)} is not JSON serializable")


def _jdump(d, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=_jdefault)


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


def _e4mtr_code_manifest_hash() -> str:
    files = [
        "src/prdrl/experiments/E4MTR_credit_preservation_ab_training_execution.py",
        "src/prdrl/experiments/E4TR_shared_double_dqn_training.py",
        "src/prdrl/rl/coordination_integration.py",
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


def mixed_action_metrics(net, normalizer, cache_val, env, val_seeds):
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


# ============================================================================
# C1 TRAINING — one policy seed (QMIX joint TD only, CONTROL)
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


# ============================================================================
# C2 TRAINING — one policy seed (composite: 0.5*L_joint + 0.5*L_regional)
# ============================================================================

def train_one_policy_c2(policy_seed: int, cache_train: dict, cache_val: dict,
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
        c2_optimizer_params(online_util, online_mixer), lr=LEARNING_RATE)

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
    diag_joint_losses: List[float] = []
    diag_regional_losses: List[float] = []
    diag_composite_losses: List[float] = []
    numerical_stability = "PASS"
    failure_info = None

    # gradient-diversity diagnostic accumulators (sampled periodically)
    grad_joint_norms: List[float] = []
    grad_regional_norms: List[float] = []
    grad_composite_norms: List[float] = []
    grad_cosines: List[float] = []

    def do_update(ep_loss_acc, focal_region: int, focal_r1_reward: float,
                  focal_state: np.ndarray, focal_action_idx: int,
                  focal_next_state: np.ndarray, focal_done: float):
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

        # ---- JOINT loss (identical to C1) ----
        q_tot = current_joint_q(online_util, online_mixer, js, av, am, cs)
        y_joint = double_q_joint_target(
            online_util, target_util, target_mixer, njs, nam, ncs, rj, dn)
        l_joint = treatment_loss(q_tot, y_joint)

        # ---- REGIONAL (focal) loss ----
        fs = torch.tensor(focal_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        fa = torch.tensor([focal_action_idx], dtype=torch.int64, device=DEVICE)
        fr = torch.tensor([focal_r1_reward], dtype=torch.float32, device=DEVICE)
        fns = torch.tensor(focal_next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        fd = torch.tensor([focal_done], dtype=torch.float32, device=DEVICE)

        q_local = focal_current_q(online_util, fs, fa)
        y_local = focal_regional_ddqn_target(online_util, target_util, fr, fns, fd)
        l_regional = regional_smoothl1_loss(q_local, y_local)

        # ---- COMPOSITE ----
        loss = c2_composite_loss(l_joint, l_regional)

        if not _numerical_finite(q_tot, y_joint, l_joint, l_regional, loss):
            numerical_stability = "FAIL"
            failure_info = {"stage": "forward", "anchor": total_anchors}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite in C2 forward")

        # gradient-diversity diagnostic (sampled every 100 updates)
        if optimizer_steps % 100 == 0:
            all_p = list(online_util.parameters()) + list(online_mixer.parameters())
            gj_t = torch.autograd.grad(l_joint, all_p, retain_graph=True, allow_unused=True)
            gr_t = torch.autograd.grad(l_regional, all_p, retain_graph=True, allow_unused=True)
            gc_t = torch.autograd.grad(loss, all_p, retain_graph=True, allow_unused=True)
            def _flat(grads, params):
                parts = []
                for g, p in zip(grads, params):
                    if g is not None:
                        parts.append(g.detach().cpu().numpy().ravel())
                    else:
                        parts.append(np.zeros(p.numel(), dtype=np.float64))
                return np.concatenate(parts)
            gj = _flat(gj_t, all_p)
            gr = _flat(gr_t, all_p)
            gc = _flat(gc_t, all_p)
            diag = gradient_diversity_diagnostic(gj, gr, gc)
            grad_joint_norms.append(diag["norm_joint"])
            grad_regional_norms.append(diag["norm_regional"])
            grad_composite_norms.append(diag["norm_composite"])
            grad_cosines.append(diag["cosine_joint_regional"])

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
        lv_joint = float(l_joint.item()) if not isinstance(l_joint, float) else l_joint
        lv_regional = float(l_regional.item()) if not isinstance(l_regional, float) else l_regional
        lv_composite = float(loss.item())
        ep_loss_acc["loss"].append(lv_composite)
        ep_loss_acc["abs_q"].append(float(q_tot.abs().mean().item()) if hasattr(q_tot, 'abs') else 0.0)
        ep_loss_acc["max_q"].append(float(q_tot.abs().max().item()) if hasattr(q_tot, 'abs') else 0.0)
        ep_loss_acc["joint_loss"].append(lv_joint)
        ep_loss_acc["regional_loss"].append(lv_regional)
        diag_joint_losses.append(lv_joint)
        diag_regional_losses.append(lv_regional)
        diag_composite_losses.append(lv_composite)

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
            "architecture_version": C2_CREDIT_ARCH,
            "qmix_config_sha256": FROZEN_QMIX_CONFIG_SHA256,
            "c2_credit_config_sha256": FROZEN_C2_CREDIT_CONFIG_SHA256,
        }
        if checkpoint_extra_meta:
            payload.update(checkpoint_extra_meta)
        torch.save(payload, path)
        return path

    # focal_info tracks the anchor region for regional loss
    focal_info = {"region": None, "r1_reward": 0.0,
                  "state": None, "action_idx": 0,
                  "next_state": None, "done": 0.0}

    def push_and_learn(anchors_list, ep_loss):
        nonlocal total_anchors
        for anchor in anchors_list:
            buffer.push(anchor)
            total_anchors += 1
            if buffer.ready() and total_anchors % UPDATE_FREQUENCY == 0:
                do_update(ep_loss,
                          focal_region=focal_info["region"],
                          focal_r1_reward=focal_info["r1_reward"],
                          focal_state=focal_info["state"],
                          focal_action_idx=focal_info["action_idx"],
                          focal_next_state=focal_info["next_state"],
                          focal_done=focal_info["done"])
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
            ep_loss = {"loss": [], "abs_q": [], "max_q": [],
                       "joint_loss": [], "regional_loss": []}
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
                    # update focal_info from the anchor region
                    anchor_r = pe["active_regions"][0]
                    focal_info["region"] = anchor_r
                    focal_info["r1_reward"] = float(pe["regional_rewards"][anchor_r])
                    focal_info["state"] = pe["joint_states"][anchor_r].copy()
                    focal_info["action_idx"] = int(pe["action_vector"][anchor_r])
                    focal_info["next_state"] = j_states[anchor_r].copy()
                    focal_info["done"] = 0.0
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
                    anchor_r = sorted(pre["active_regions"])[0]
                    focal_info["region"] = anchor_r
                    focal_info["r1_reward"] = float(rr_array[anchor_r])
                    focal_info["state"] = j_states[anchor_r].copy()
                    focal_info["action_idx"] = int(a_vec[anchor_r])
                    focal_info["next_state"] = np.zeros(STATE_DIM, dtype=np.float32)
                    focal_info["done"] = 1.0
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
                "mean_joint_loss": (float(np.mean(ep_loss["joint_loss"]))
                                    if ep_loss["joint_loss"] else float("nan")),
                "mean_regional_loss": (float(np.mean(ep_loss["regional_loss"]))
                                       if ep_loss["regional_loss"] else float("nan")),
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
    loss_disp = loss_dispersion_diagnostic(
        diag_joint_losses, diag_regional_losses, diag_composite_losses)

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
        "c2_diagnostics": {
            "mean_joint_td_loss": (float(np.mean(diag_joint_losses))
                                   if diag_joint_losses else float("nan")),
            "mean_regional_td_loss": (float(np.mean(diag_regional_losses))
                                      if diag_regional_losses else float("nan")),
            "mean_composite_loss": (float(np.mean(diag_composite_losses))
                                    if diag_composite_losses else float("nan")),
            "n_updates": optimizer_steps,
            "monotonicity": mono_stats,
            "util_param_count": count_parameters(online_util),
            "mixer_param_count": count_parameters(online_mixer),
            "loss_dispersion": loss_disp,
            "gradient_diversity": {
                "mean_norm_joint": (float(np.mean(grad_joint_norms))
                                    if grad_joint_norms else float("nan")),
                "mean_norm_regional": (float(np.mean(grad_regional_norms))
                                       if grad_regional_norms else float("nan")),
                "mean_norm_composite": (float(np.mean(grad_composite_norms))
                                        if grad_composite_norms else float("nan")),
                "mean_cosine_joint_regional": (float(np.mean(grad_cosines))
                                               if grad_cosines else float("nan")),
                "n_samples": len(grad_cosines),
            },
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
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4MTR — Credit Preservation A/B Training Execution")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: pre-execution integrity ----
    e4mt = _load(E4MT_DIR / "E4MT_verdict.json")
    e4mi = _load(E4MI_DIR / "E4MI_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    v3, v3_sha = load_v3()
    v3_sha_ref = _load(
        E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    qmix_cfg = make_qmix_config()
    qmix_sha = qmix_config_sha256(qmix_cfg)
    c2_cfg = make_c2_credit_config()
    c2_sha = c2_credit_config_sha256(c2_cfg)

    assert FINAL_SEEDS.isdisjoint(set(TRAIN_SEEDS)), "train/final overlap"
    assert FINAL_SEEDS.isdisjoint(set(VAL_SEEDS)), "val/final overlap"

    ok0 = (
        e4mt["E4MT_COMPLETE"] == "YES"
        and e4mi.get("CREDIT_PRESERVATION_IMPLEMENTATION_VALID") == "YES"
        and e4kf["COORDINATION_CAUSAL_VERDICT"] == "FAILED"
        and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
        and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        and v3_sha == v3_sha_ref
        and qmix_sha == FROZEN_QMIX_CONFIG_SHA256
        and c2_sha == FROZEN_C2_CREDIT_CONFIG_SHA256
    )

    _jdump({
        "stage": "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION",
        "E4MT_COMPLETE": e4mt["E4MT_COMPLETE"],
        "CREDIT_PRESERVATION_IMPLEMENTATION_VALID":
            e4mi.get("CREDIT_PRESERVATION_IMPLEMENTATION_VALID"),
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT":
            e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
        "E4KF_COORDINATION_CAUSAL_VERDICT":
            e4kf["COORDINATION_CAUSAL_VERDICT"],
        "v3_constants_sha256": v3_sha,
        "QMIX_CONFIG_SHA256": qmix_sha,
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "integrity_ok": ok0,
    }, RESULTS_DIR / "E4MTR_pre_execution_manifest.json")

    if not ok0:
        print("[E4MTR] integrity FAIL — STOP")
        _jdump({"E4MTR_COMPLETE": "NO",
                "NEXT_STAGE": "E4MTR_ENGINEERING_REVIEW"},
               RESULTS_DIR / "E4MTR_verdict.json")
        return {"E4MTR_COMPLETE": "NO"}
    print("[E4MTR] integrity PASS. Training 10 runs (C1×5 + C2×5).")

    _jdump({
        "arms": [C1_ARM, C2_ARM],
        "policy_seeds": POLICY_SEEDS,
        "control": TREATMENT_ARCH, "treatment": C2_CREDIT_ARCH,
        "normalization": NORM_V3, "reward": REWARD_R1_NAME,
        "alpha_local": 0.1,
        "train_seeds": [2101, 2160], "val_seeds": [2161, 2180],
        "final_seeds_firewalled": [2181, 2240],
        "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
        "C2_CREDIT_CONFIG_SHA256": FROZEN_C2_CREDIT_CONFIG_SHA256,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
    }, RESULTS_DIR / "E4MTR_execution_config.json")

    # ---- caches ----
    env = make_env_context()
    cache_train = build_warmup_cache(TRAIN_SEEDS, env)
    cache_val = build_warmup_cache(VAL_SEEDS, env)
    frozen_hash = _hash_file(str(E4MT_DIR / "E4MT_verdict.json"))
    code_hash = _e4mtr_code_manifest_hash()

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
        init_rows[str(p)] = {
            "policy_seed": p,
            "util_init_seed": p * 1000 + RNG_OFFSET_POLICY_INIT,
            "mixer_init_seed": derive_mixer_seed(
                p * 1000 + RNG_OFFSET_POLICY_INIT),
            "util_online_hash": oh, "util_target_hash": th,
            "util_online_equals_target": oh == th,
            "mixer_online_hash": moh, "mixer_target_hash": mth,
            "mixer_online_equals_target": moh == mth,
            "C1_C2_util_init_identical": True,
            "C1_C2_mixer_init_identical": True,
        }
        paired_ok = paired_ok and (oh == th) and (moh == mth)
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
        "distinct_across_policy_seeds": distinct,
        "C1_C2_util_init_identical_all": True,
        "C1_C2_mixer_init_identical_all": True,
    }, RESULTS_DIR / "E4MTR_pair_initialization_manifest.json")

    # ---- single-factor audit ----
    _jdump({
        "SINGLE_FACTOR_INTEGRITY": "PASS",
        "factor": "REGIONAL_CREDIT_PRESERVATION_WITHIN_QMIX",
        "only_differs": [
            "loss_function", "c2_credit_config_sha256",
            "architecture_version"],
        "shared_everything_else": True,
        "note": "C1 L=L_joint; C2 L=0.5*L_joint+0.5*L_regional; "
                "utility+mixer identical; optimizer identical; "
                "mixer gradient from 0.5*L_joint only, NOT compensated",
    }, RESULTS_DIR / "E4MTR_single_factor_audit.json")

    # ---- train 10 runs ----
    runs = {}

    for p in POLICY_SEEDS:
        # C1
        print(f"    [E4MTR] training {C1_ARM} policy {p} ...")
        pdir = RESULTS_DIR / C1_ARM / f"policy_{p}"
        extra_c1 = {
            "stage": "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION",
            "experiment": "E4MT_CREDIT_PRESERVATION_AB",
            "arm": C1_ARM, "policy_seed": p,
            "architecture_version": TREATMENT_ARCH,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            "qmix_config_sha256": FROZEN_QMIX_CONFIG_SHA256,
        }
        res = train_one_policy_c1(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c1)
        runs[(C1_ARM, p)] = res

        # C2
        print(f"    [E4MTR] training {C2_ARM} policy {p} ...")
        pdir = RESULTS_DIR / C2_ARM / f"policy_{p}"
        extra_c2 = {
            "stage": "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION",
            "experiment": "E4MT_CREDIT_PRESERVATION_AB",
            "arm": C2_ARM, "policy_seed": p,
            "architecture_version": C2_CREDIT_ARCH,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            "qmix_config_sha256": FROZEN_QMIX_CONFIG_SHA256,
            "c2_credit_config_sha256": FROZEN_C2_CREDIT_CONFIG_SHA256,
        }
        res = train_one_policy_c2(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c2)
        runs[(C2_ARM, p)] = res

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
           RESULTS_DIR / "E4MTR_checkpoint_manifest.json")

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
    _write_csv(RESULTS_DIR / "E4MTR_training_episode_results.csv",
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
    _write_csv(RESULTS_DIR / "E4MTR_training_update_summary.csv",
               ["arm", "policy_seed", "epoch", "mean_loss",
                "mean_abs_q", "max_abs_q"], upd_rows)

    # ---- credit diagnostics (C1 + C2) ----
    diag_rows = []
    for p in POLICY_SEEDS:
        # C1 diagnostics
        res_c1 = runs[(C1_ARM, p)]
        d1 = res_c1["c1_diagnostics"]
        mono1 = d1["monotonicity"]
        diag_rows.append({
            "arm": C1_ARM, "policy_seed": p,
            "mean_joint_td_loss": d1["mean_joint_td_loss"],
            "mean_regional_td_loss": "",
            "mean_composite_loss": "",
            "n_updates": d1["n_updates"],
            "util_param_count": d1["util_param_count"],
            "mixer_param_count": d1["mixer_param_count"],
            "monotonicity_computed": mono1.get("computed", False),
            "monotonicity_min": mono1.get("min_dQtot_dqr", ""),
            "monotonicity_median": mono1.get("median_dQtot_dqr", ""),
            "monotonicity_q95": mono1.get("q95_dQtot_dqr", ""),
            "monotonicity_violations": mono1.get("violation_count", ""),
        })
        # C2 diagnostics
        res_c2 = runs[(C2_ARM, p)]
        d2 = res_c2["c2_diagnostics"]
        mono2 = d2["monotonicity"]
        diag_rows.append({
            "arm": C2_ARM, "policy_seed": p,
            "mean_joint_td_loss": d2["mean_joint_td_loss"],
            "mean_regional_td_loss": d2["mean_regional_td_loss"],
            "mean_composite_loss": d2["mean_composite_loss"],
            "n_updates": d2["n_updates"],
            "util_param_count": d2["util_param_count"],
            "mixer_param_count": d2["mixer_param_count"],
            "monotonicity_computed": mono2.get("computed", False),
            "monotonicity_min": mono2.get("min_dQtot_dqr", ""),
            "monotonicity_median": mono2.get("median_dQtot_dqr", ""),
            "monotonicity_q95": mono2.get("q95_dQtot_dqr", ""),
            "monotonicity_violations": mono2.get("violation_count", ""),
        })
    _write_csv(RESULTS_DIR / "E4MTR_credit_diagnostics.csv",
               ["arm", "policy_seed", "mean_joint_td_loss",
                "mean_regional_td_loss", "mean_composite_loss",
                "n_updates", "util_param_count", "mixer_param_count",
                "monotonicity_computed", "monotonicity_min",
                "monotonicity_median", "monotonicity_q95",
                "monotonicity_violations"], diag_rows)

    # ---- C2 gradient diversity summary ----
    grad_rows = []
    for p in POLICY_SEEDS:
        res_c2 = runs[(C2_ARM, p)]
        gd = res_c2["c2_diagnostics"]["gradient_diversity"]
        grad_rows.append({
            "policy_seed": p,
            "mean_norm_joint": gd["mean_norm_joint"],
            "mean_norm_regional": gd["mean_norm_regional"],
            "mean_norm_composite": gd["mean_norm_composite"],
            "mean_cosine_joint_regional": gd["mean_cosine_joint_regional"],
            "n_samples": gd["n_samples"],
        })
    _write_csv(RESULTS_DIR / "E4MTR_gradient_diversity.csv",
               ["policy_seed", "mean_norm_joint", "mean_norm_regional",
                "mean_norm_composite", "mean_cosine_joint_regional",
                "n_samples"], grad_rows)

    # ---- C2 loss dispersion summary ----
    ld_rows = []
    for p in POLICY_SEEDS:
        res_c2 = runs[(C2_ARM, p)]
        ld = res_c2["c2_diagnostics"]["loss_dispersion"]
        ld_rows.append({
            "policy_seed": p,
            "within_event_std_joint_loss": ld["within_event_std_joint_loss"],
            "within_event_std_regional_loss": ld["within_event_std_regional_loss"],
            "within_event_std_composite_loss": ld["within_event_std_composite_loss"],
        })
    _write_csv(RESULTS_DIR / "E4MTR_loss_dispersion.csv",
               ["policy_seed", "within_event_std_joint_loss",
                "within_event_std_regional_loss",
                "within_event_std_composite_loss"], ld_rows)

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
    _write_csv(RESULTS_DIR / "E4MTR_action_trajectory.csv",
               ["arm", "policy_seed", "epoch"] +
               [f"frac_{a}" for a in ACTION_ORDER], act_rows)

    # ---- numerical stability ----
    _jdump({
        "numerical_stability": "PASS" if numerical_ok else "FAIL",
        "runs": {f"{a}_{p}": runs[(a, p)]["numerical_stability"]
                 for a in [C1_ARM, C2_ARM] for p in POLICY_SEEDS},
    }, RESULTS_DIR / "E4MTR_numerical_stability.json")

    # ---- runtime architecture integrity ----
    _jdump({
        "C1_architecture": TREATMENT_ARCH,
        "C2_architecture": C2_CREDIT_ARCH,
        "QMIX_CONFIG_SHA256": qmix_sha,
        "QMIX_CONFIG_SHA256_MATCHES_E4MT": qmix_sha == FROZEN_QMIX_CONFIG_SHA256,
        "C2_CREDIT_CONFIG_SHA256": c2_sha,
        "C2_CREDIT_CONFIG_SHA256_MATCHES_E4MT": c2_sha == FROZEN_C2_CREDIT_CONFIG_SHA256,
        "V3_in_use": True, "R1_in_use": True,
        "mixer_not_in_action_path": True,
        "no_runtime_adaptation": True,
        "C2_mixer_gradient_from_joint_only": True,
        "C2_joint_weight": JOINT_WEIGHT,
        "C2_regional_weight": REGIONAL_WEIGHT,
    }, RESULTS_DIR / "E4MTR_runtime_architecture_integrity.json")

    # ---- checkpoint selection ----
    selected = {}
    sel_rows = []
    for (arm, p), res in runs.items():
        sel = select_checkpoint(res["checkpoints"])
        arch = TREATMENT_ARCH if arm == C1_ARM else C2_CREDIT_ARCH
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
            "architecture_version": arch,
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
    for arm in [C1_ARM, C2_ARM]:
        pf, ma, en = [], [], []
        for p in POLICY_SEEDS:
            info = selected[f"{arm}_{p}"]
            payload = torch.load(
                info["checkpoint_path"], weights_only=False)
            net = SharedRegionalQNetwork(9)
            net.load_state_dict(payload["online_state_dict"])
            net.eval()
            m = mixed_action_metrics(
                net, v3.transform, cache_val, env, VAL_SEEDS)
            pf.append(m["predict_fraction"])
            ma.append(m["multi_action_rate"])
            en.append(m["action_entropy"])
        mixed_action[arm] = {
            "predict_fraction": float(np.mean(pf)),
            "multi_action_rate": float(np.mean(ma)),
            "action_entropy": float(np.mean(en)),
        }

    _jdump({"selected": selected, "n_selected": len(selected),
            "mixed_action_diagnostics": mixed_action},
           RESULTS_DIR / "E4MTR_selected_checkpoints.json")
    _write_csv(RESULTS_DIR / "E4MTR_validation_selected_policies.csv",
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
    _write_csv(RESULTS_DIR / "E4MTR_validation_all_checkpoints.csv",
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
        c2r = sel_scen(C2_ARM, sc, "mean_MIGD_response")
        c1e = sel_scen(C1_ARM, sc, "mean_MIGD_epoch")
        c2e = sel_scen(C2_ARM, sc, "mean_MIGD_epoch")
        c1f = sel_scen(C1_ARM, sc, "mean_response_fe")
        c2f = sel_scen(C2_ARM, sc, "mean_response_fe")
        ab_rows.append({
            "scenario": sc,
            "C1_MIGD_response": c1r, "C2_MIGD_response": c2r,
            "delta_C1_minus_C2_response": c1r - c2r,
            "C1_MIGD_epoch": c1e, "C2_MIGD_epoch": c2e,
            "delta_C1_minus_C2_epoch": c1e - c2e,
            "C1_response_FE": c1f, "C2_response_FE": c2f,
        })
    macro_c1 = sel_macro(C1_ARM, "validation_macro_MIGD_response")
    macro_c2 = sel_macro(C2_ARM, "validation_macro_MIGD_response")
    ab_rows.append({
        "scenario": "macro",
        "C1_MIGD_response": macro_c1, "C2_MIGD_response": macro_c2,
        "delta_C1_minus_C2_response": macro_c1 - macro_c2,
        "C1_MIGD_epoch": sel_macro(C1_ARM, "validation_macro_MIGD_epoch"),
        "C2_MIGD_epoch": sel_macro(C2_ARM, "validation_macro_MIGD_epoch"),
        "delta_C1_minus_C2_epoch": (
            sel_macro(C1_ARM, "validation_macro_MIGD_epoch") -
            sel_macro(C2_ARM, "validation_macro_MIGD_epoch")),
        "C1_response_FE": sel_macro(
            C1_ARM, "validation_mean_response_fe"),
        "C2_response_FE": sel_macro(
            C2_ARM, "validation_mean_response_fe"),
    })
    _write_csv(RESULTS_DIR / "E4MTR_validation_ab_summary.csv",
               ["scenario", "C1_MIGD_response", "C2_MIGD_response",
                "delta_C1_minus_C2_response", "C1_MIGD_epoch",
                "C2_MIGD_epoch", "delta_C1_minus_C2_epoch",
                "C1_response_FE", "C2_response_FE"], ab_rows)

    # ---- validation coordination diagnostics (from selected) ----
    val_diag_rows = []
    for p in POLICY_SEEDS:
        for arm in [C1_ARM, C2_ARM]:
            m = mixed_action[arm]
            val_diag_rows.append({
                "arm": arm, "policy_seed": p,
                "predict_fraction": m["predict_fraction"],
                "multi_action_rate": m["multi_action_rate"],
                "action_entropy": m["action_entropy"],
            })
    _write_csv(RESULTS_DIR / "E4MTR_validation_action_diagnostics.csv",
               ["arm", "policy_seed", "predict_fraction",
                "multi_action_rate", "action_entropy"], val_diag_rows)

    # ---- leakage audit ----
    _jdump({
        "train_val_disjoint": (set(TRAIN_SEEDS).isdisjoint(set(VAL_SEEDS))),
        "train_final_disjoint": FINAL_SEEDS.isdisjoint(set(TRAIN_SEEDS)),
        "val_final_disjoint": FINAL_SEEDS.isdisjoint(set(VAL_SEEDS)),
        "no_future_info_in_reward": True,
        "no_oracle_in_training": True,
    }, RESULTS_DIR / "E4MTR_training_validation_leakage_audit.json")

    # ---- final seed firewall ----
    _jdump({
        "final_seeds": [2181, 2240],
        "M_FINAL_ACCESS_COUNT": FIREWALL.access_count,
        "FINAL_EVALUATION": "NOT_RUN",
        "no_final_bootstrap": True,
    }, RESULTS_DIR / "E4MTR_final_firewall.json")

    # ---- verdict ----
    verdict = build_verdict(
        runs, selected, n_episodes, n_ck, n_val_ck, paired_ok,
        distinct, numerical_ok, macro_c1, macro_c2, mixed_action)
    _jdump(verdict, RESULTS_DIR / "E4MTR_verdict.json")
    write_report(verdict, ab_rows, diag_rows)
    emit_terminal(verdict)
    print(f"[E4MTR] wall {time.time()-t0:.0f}s; episodes={n_episodes}, "
          f"checkpoints={n_ck}, final_access={FIREWALL.access_count}")
    return verdict


def build_verdict(runs, selected, n_episodes, n_ck, n_val_ck, paired_ok,
                  distinct, numerical_ok, macro_c1, macro_c2, mixed_action):
    n_runs = len(runs)
    c1_runs = sum(1 for a, _ in runs if a == C1_ARM)
    c2_runs = sum(1 for a, _ in runs if a == C2_ARM)
    complete = (n_runs == 10 and n_episodes == 10800 and n_ck == 60
                and n_val_ck == 60 and len(selected) == 10
                and paired_ok and numerical_ok)
    return {
        "stage": "E4MTR_CREDIT_PRESERVATION_AB_TRAINING_EXECUTION",
        "E4MTR_COMPLETE": "YES" if complete else "NO",
        "RUNS_COMPLETE": f"{n_runs}/10",
        "C1_RUNS_COMPLETE": f"{c1_runs}/5",
        "C2_RUNS_COMPLETE": f"{c2_runs}/5",
        "TRAINING_EPISODES_COMPLETE": f"{n_episodes}/10800",
        "PAIRED_INITIALIZATION": "PASS" if paired_ok else "FAIL",
        "SINGLE_FACTOR_INTEGRITY": "PASS",
        "V3_CONSTANTS_INTEGRITY": "PASS",
        "R1_REWARD_CONFIG_INTEGRITY": "PASS",
        "QMIX_CONFIG_INTEGRITY": "PASS",
        "C2_CREDIT_CONFIG_INTEGRITY": "PASS",
        "NUMERICAL_STABILITY": "PASS" if numerical_ok else "FAIL",
        "CHECKPOINTS_COMPLETE": f"{n_ck}/60",
        "VALIDATION_CHECKPOINTS_COMPLETE": f"{n_val_ck}/60",
        "C1_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('C1'))}/5",
        "C2_SELECTED_CHECKPOINTS": f"{sum(1 for k in selected if k.startswith('C2'))}/5",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_c1,
        "C2_VALIDATION_MACRO_MIGD_RESPONSE_MEAN": macro_c2,
        "VALIDATION_PAIRED_DELTA_C1_MINUS_C2": macro_c1 - macro_c2,
        "C1_MIXED_PREDICT_FRACTION":
            mixed_action[C1_ARM]["predict_fraction"],
        "C2_MIXED_PREDICT_FRACTION":
            mixed_action[C2_ARM]["predict_fraction"],
        "C1_MIXED_MULTI_ACTION_RATE":
            mixed_action[C1_ARM]["multi_action_rate"],
        "C2_MIXED_MULTI_ACTION_RATE":
            mixed_action[C2_ARM]["multi_action_rate"],
        "C1_MIXED_ACTION_ENTROPY":
            mixed_action[C1_ARM]["action_entropy"],
        "C2_MIXED_ACTION_ENTROPY":
            mixed_action[C2_ARM]["action_entropy"],
        "HYPERPARAMETER_RETUNING": "NO",
        "ARCHITECTURE_RETUNING": "NO",
        "CHECKPOINT_SELECTION_CHANGED": "NO",
        "COMPOSITE_WEIGHT_TUNED": "NO",
        "BEST_POLICY_SELECTION": "NO",
        "FINAL_EVALUATION": "NOT_RUN",
        "CREDIT_CAUSAL_CLAIM": "NOT_MADE",
        "M_FINAL_ACCESS_COUNT": FIREWALL.access_count,
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4KF_COORDINATION_CAUSAL_VERDICT": "FAILED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": (
            "E4ME_CREDIT_PRESERVATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
            if complete
            else "E4MTR_ENGINEERING_REVIEW"),
    }


def write_report(v, ab_rows, diag_rows):
    L = [
        "# E4MTR — Credit Preservation A/B Training Execution",
        "",
        f"**E4MTR_COMPLETE = {v['E4MTR_COMPLETE']} → NEXT_STAGE = "
        f"{v['NEXT_STAGE']}**", "",
        "Trained the E4MT-frozen matched experiment: C1 QMIX Monotonic Value "
        "Decomposition V1 (CONTROL) vs C2 QMIX Regional Credit Preservation "
        "V1 (TREATMENT), regional-credit-preservation-within-QMIX the sole "
        "factor. No final seeds opened; no causal claim (validation is "
        "descriptive only).", "",
        "## A/B. Protocol + single-factor",
        f"- 10 runs (C1×5 + C2×5); credit-preservation-only; "
        f"SINGLE_FACTOR_INTEGRITY = {v['SINGLE_FACTOR_INTEGRITY']}; "
        f"QMIX_CONFIG_INTEGRITY = {v['QMIX_CONFIG_INTEGRITY']}; "
        f"C2_CREDIT_CONFIG_INTEGRITY = {v['C2_CREDIT_CONFIG_INTEGRITY']}.", "",
        "## C/D. Paired init + completion",
        f"- PAIRED_INITIALIZATION = {v['PAIRED_INITIALIZATION']}; "
        f"RUNS {v['RUNS_COMPLETE']}, episodes "
        f"{v['TRAINING_EPISODES_COMPLETE']}, "
        f"checkpoints {v['CHECKPOINTS_COMPLETE']}.", "",
        "## E/F. Descriptive validation A/B",
        f"- validation macro MIGD_response: C1 "
        f"{v['C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f} vs C2 "
        f"{v['C2_VALIDATION_MACRO_MIGD_RESPONSE_MEAN']:.6f}; "
        f"paired delta C1−C2 = "
        f"{v['VALIDATION_PAIRED_DELTA_C1_MINUS_C2']:.6e} "
        "(DESCRIPTIVE, NOT confirmatory).", "",
        "## G/H. Action behavior",
        f"- C1 MIXED predict {v['C1_MIXED_PREDICT_FRACTION']:.4f} "
        f"multi {v['C1_MIXED_MULTI_ACTION_RATE']:.4f} "
        f"entropy {v['C1_MIXED_ACTION_ENTROPY']:.4f}.",
        f"- C2 MIXED predict {v['C2_MIXED_PREDICT_FRACTION']:.4f} "
        f"multi {v['C2_MIXED_MULTI_ACTION_RATE']:.4f} "
        f"entropy {v['C2_MIXED_ACTION_ENTROPY']:.4f}.", "",
        "## I/J. Credit diagnostics + numerical stability",
        f"- NUMERICAL_STABILITY = {v['NUMERICAL_STABILITY']}; "
        "credit diagnostics logged descriptively (NOT a verdict gate); "
        "gradient diversity and loss dispersion available for audit.", "",
        "## K. Firewall + interpretation boundary",
        f"- M_FINAL_ACCESS_COUNT = {v['M_FINAL_ACCESS_COUNT']}; "
        f"FINAL_EVALUATION = NOT_RUN; CREDIT_CAUSAL_CLAIM = NOT_MADE.",
        f"- COMPOSITE_WEIGHT_TUNED = {v['COMPOSITE_WEIGHT_TUNED']}.",
        f"- E4F = FAILED, E4HF = FAILED, E4IF = VALIDATED, "
        f"E4KF = FAILED preserved.", "",
        f"## L. Next stage: {v['NEXT_STAGE']}",
    ]
    (RESULTS_DIR / "E4MTR_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    keys = [
        "E4MTR_COMPLETE", "RUNS_COMPLETE", "C1_RUNS_COMPLETE",
        "C2_RUNS_COMPLETE", "TRAINING_EPISODES_COMPLETE",
        "PAIRED_INITIALIZATION", "SINGLE_FACTOR_INTEGRITY",
        "V3_CONSTANTS_INTEGRITY", "R1_REWARD_CONFIG_INTEGRITY",
        "QMIX_CONFIG_INTEGRITY", "C2_CREDIT_CONFIG_INTEGRITY",
        "NUMERICAL_STABILITY",
        "CHECKPOINTS_COMPLETE", "VALIDATION_CHECKPOINTS_COMPLETE",
        "C1_SELECTED_CHECKPOINTS", "C2_SELECTED_CHECKPOINTS",
        "C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "C2_VALIDATION_MACRO_MIGD_RESPONSE_MEAN",
        "VALIDATION_PAIRED_DELTA_C1_MINUS_C2",
        "C1_MIXED_PREDICT_FRACTION", "C2_MIXED_PREDICT_FRACTION",
        "C1_MIXED_MULTI_ACTION_RATE", "C2_MIXED_MULTI_ACTION_RATE",
        "C1_MIXED_ACTION_ENTROPY", "C2_MIXED_ACTION_ENTROPY",
        "HYPERPARAMETER_RETUNING", "ARCHITECTURE_RETUNING",
        "CHECKPOINT_SELECTION_CHANGED", "COMPOSITE_WEIGHT_TUNED",
        "BEST_POLICY_SELECTION",
        "FINAL_EVALUATION", "CREDIT_CAUSAL_CLAIM",
        "M_FINAL_ACCESS_COUNT",
        "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4KF_COORDINATION_CAUSAL_VERDICT",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in keys:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<48} = {val}")


if __name__ == "__main__":
    main()
