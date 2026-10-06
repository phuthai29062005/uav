"""
Branch-C coordination integration (frozen by E4K).

Dispatches the two matched architecture arms without changing historical behavior:

    C0_INDEPENDENT_SHARED_DDQN  — the existing V3+R1 regional Double-DQN (utility only)
    C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 — same utility + QMIX mixer, joint Double-Q TD,
        active-region mean R1 team reward

Provides: joint reward aggregation (mean, never sum), the Double-Q joint target, current joint Q,
the treatment SmoothL1 loss, the exact C0 control target/loss (byte-identical to E4TR do_update),
optimizer parameter partitions, the immutable QMIX config + its sha256, and checkpoint version
guards. No training and no environment execution live here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from prdrl.rl.network import SharedRegionalQNetwork, double_dqn_target, N_ACTIONS
from prdrl.rl.qmix import (
    QMIXMixerV1, R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM, NONNEGATIVE_TRANSFORM, TREATMENT_ARCH,
    MIXER_SEED_OFFSET,
)
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256
from prdrl.evaluation.final_verdict import (
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)

CONTROL_ARCH = "INDEPENDENT_SHARED_DDQN"
C2_CREDIT_ARCH = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"
FROZEN_V3_CONSTANTS_SHA256 = "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"
STATE_DIM = 9
GAMMA = 0.99
TARGET_TAU = 0.005
INACTIVE_ACTION_SENTINEL = 0   # KEEP index — placeholder action for inactive slots (contributes 0)
JOINT_WEIGHT = 0.5
REGIONAL_WEIGHT = 0.5
C2_LOSS_VERSION = "EQUAL_COMPONENT_MEAN_V1"


class CoordinationConfigError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Joint reward aggregation (ACTIVE_REGION_MEAN_OF_R1) — never sum
# ---------------------------------------------------------------------------
def joint_reward_mean(regional_rewards: Sequence[float], active_mask: Sequence[float]) -> float:
    """r_joint = (1/|A|) * sum_{r in A} r_r^R1, over active regions only. |A|==0 -> 0.0."""
    rr = np.asarray(regional_rewards, dtype=np.float64)
    m = np.asarray(active_mask, dtype=np.float64)
    n = float(m.sum())
    if n <= 0:
        return 0.0
    return float((rr * m).sum() / n)


# ---------------------------------------------------------------------------
# C1 treatment: current joint Q and Double-Q joint target
# ---------------------------------------------------------------------------
def current_joint_q(online_util: SharedRegionalQNetwork, online_mixer: QMIXMixerV1,
                    joint_states: torch.Tensor, action_vector: torch.Tensor,
                    active_mask: torch.Tensor, central_state: torch.Tensor) -> torch.Tensor:
    """Q_tot_online(requested joint action, current central state). Inactive regions -> q=0.
    joint_states [B,R,9], action_vector [B,R] (requested policy action), active_mask [B,R],
    central_state [B,55] -> [B,1]."""
    B, R, _ = joint_states.shape
    q_all = online_util(joint_states.reshape(B * R, STATE_DIM)).reshape(B, R, N_ACTIONS)
    q_sel = q_all.gather(2, action_vector.long().unsqueeze(2)).squeeze(2)   # [B,R]
    q_sel = q_sel * active_mask.to(q_sel.dtype)                             # inactive -> 0
    return online_mixer(q_sel, central_state)


def double_q_joint_target(online_util: SharedRegionalQNetwork, target_util: SharedRegionalQNetwork,
                          target_mixer: QMIXMixerV1, next_joint_states: torch.Tensor,
                          next_active_mask: torch.Tensor, next_central_state: torch.Tensor,
                          r_joint: torch.Tensor, done: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """Double-Q joint target: a*_r = argmax_a Q_online(s'_r,a); q'_r = Q_target(s'_r,a*_r);
    inactive -> 0; Q_tot_target = mixer_target(q', c'); y = r_joint + gamma*(1-done)*Q_tot_target.
    Returns [B,1]. No gradient flows (target path)."""
    with torch.no_grad():
        B, R, _ = next_joint_states.shape
        flat = next_joint_states.reshape(B * R, STATE_DIM)
        a_star = online_util(flat).argmax(dim=1, keepdim=True)              # ONLINE selects
        q_next = target_util(flat).gather(1, a_star).squeeze(1).reshape(B, R)  # TARGET evaluates
        q_next = q_next * next_active_mask.to(q_next.dtype)                 # inactive -> 0
        q_tot_next = target_mixer(q_next, next_central_state)               # [B,1]
        r_joint = r_joint.reshape(B, 1).to(q_tot_next.dtype)
        done = done.reshape(B, 1).to(q_tot_next.dtype)
        y = r_joint + gamma * (1.0 - done) * q_tot_next
    return y


def treatment_loss(q_tot_current: torch.Tensor, y_joint: torch.Tensor) -> torch.Tensor:
    """C1 loss = SmoothL1(Q_tot_current, y_joint). No auxiliary/regularizer terms."""
    return nn.SmoothL1Loss()(q_tot_current, y_joint)


# ---------------------------------------------------------------------------
# C0 control: exact existing V3+R1 Independent Shared DDQN target/loss
# ---------------------------------------------------------------------------
def control_ddqn_target_and_loss(online: SharedRegionalQNetwork, target: SharedRegionalQNetwork,
                                 s: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                                 ns: torch.Tensor, d: torch.Tensor, gamma: float = GAMMA):
    """Byte-identical to E4TR do_update: y=double_dqn_target(...); q=online(s).gather(a);
    loss=SmoothL1(q,y). Returns (y, q, loss)."""
    y = double_dqn_target(r, ns, d, online, target, gamma)
    q = online(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
    loss = nn.SmoothL1Loss()(q, y)
    return y, q, loss


# ---------------------------------------------------------------------------
# Optimizer parameter partitions
# ---------------------------------------------------------------------------
def c0_optimizer_params(online_util: SharedRegionalQNetwork) -> List[nn.Parameter]:
    return list(online_util.parameters())


def c1_optimizer_params(online_util: SharedRegionalQNetwork, online_mixer: QMIXMixerV1) -> List[nn.Parameter]:
    return list(online_util.parameters()) + list(online_mixer.parameters())


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# Soft target update (utility and mixer) at tau=0.005
# ---------------------------------------------------------------------------
def soft_update(online: nn.Module, target: nn.Module, tau: float = TARGET_TAU) -> None:
    with torch.no_grad():
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.copy_((1.0 - tau) * p_t.data + tau * p_o.data)


# ---------------------------------------------------------------------------
# Immutable QMIX config + sha256; checkpoint version guards
# ---------------------------------------------------------------------------
def make_qmix_config() -> Dict:
    """Immutable QMIX config built ONLY from frozen E4K design fields."""
    return {
        "architecture_version": TREATMENT_ARCH,
        "control_architecture_version": CONTROL_ARCH,
        "R_MAX": R_MAX,
        "central_state_dim": CENTRAL_STATE_DIM,
        "utility_architecture": "9-128-128-4",
        "mixer_hidden_dim": MIX_HIDDEN_DIM,
        "W1_hypernetwork": "Linear(central_state_dim, R_MAX*mixer_hidden_dim) -> reshape[R_MAX,mixer_hidden_dim]",
        "W2_hypernetwork": "Linear(central_state_dim, mixer_hidden_dim) -> reshape[mixer_hidden_dim,1]",
        "b1_hypernetwork": "Linear(central_state_dim, mixer_hidden_dim)",
        "b2_hypernetwork": "Linear(central_state_dim, 1)",
        "nonnegative_transform": NONNEGATIVE_TRANSFORM,
        "activation": "ELU",
        "tau": TARGET_TAU,
        "mixer_seed_offset": MIXER_SEED_OFFSET,
        "joint_reward_aggregation": "ACTIVE_REGION_MEAN_OF_R1",
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "replay_capacity": 50000,
        "warmup": 1000,
        "update_every": 4,
        "batch_size": 128,
        "gamma": GAMMA,
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "normalization_version": NORM_V3,
        "normalizer_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
    }


def qmix_config_sha256(config: Dict = None) -> str:
    cfg = make_qmix_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_c1_checkpoint_meta(meta: Dict, expected_config_sha: str = None) -> None:
    """Hard error on a wrong/missing/incompatible C1 metadata combination."""
    want = expected_config_sha if expected_config_sha is not None else qmix_config_sha256()
    if meta.get("architecture_version") != TREATMENT_ARCH:
        raise CoordinationConfigError(
            f"C1 architecture_version {meta.get('architecture_version')} != {TREATMENT_ARCH}")
    if meta.get("qmix_config_sha256") != want:
        raise CoordinationConfigError(
            f"C1 qmix_config_sha256 {meta.get('qmix_config_sha256')} != {want}")
    if meta.get("reward_version") != REWARD_R1_NAME:
        raise CoordinationConfigError("C1 reward_version must be R1")
    if meta.get("reward_config_sha256") != FROZEN_R1_CONFIG_SHA256:
        raise CoordinationConfigError("C1 reward_config_sha256 mismatch")
    if meta.get("normalization_version") != NORM_V3:
        raise CoordinationConfigError("C1 normalization_version must be V3")
    if meta.get("normalizer_constants_sha256") != FROZEN_V3_CONSTANTS_SHA256:
        raise CoordinationConfigError("C1 V3 constants hash mismatch")


def validate_c0_checkpoint_meta(meta: Dict) -> None:
    """C0 must stay INDEPENDENT_SHARED_DDQN and carry no treatment mixer weights."""
    if meta.get("architecture_version") != CONTROL_ARCH:
        raise CoordinationConfigError(
            f"C0 architecture_version {meta.get('architecture_version')} != {CONTROL_ARCH}")
    if meta.get("mixer_state_dict") is not None or meta.get("has_mixer"):
        raise CoordinationConfigError("C0 checkpoint must not contain a mixer")


# ---------------------------------------------------------------------------
# Coordination causal verdict (frozen by E4KT/E4KE; NO mechanism gate)
# ---------------------------------------------------------------------------
def coordination_causal_verdict(
    *,
    mixed_primary_label: str,
    policy_robustness_pass: bool,
    persistent_safety_pass: bool,
    stationary_safety_pass: bool,
    macro_safety_pass: bool,
    checkpoint_integrity_pass: bool,
    protocol_integrity_pass: bool,
    architecture_integrity_pass: bool,
    no_final_tuning: bool,
) -> Dict:
    """Frozen E4KT/E4KE success logic for the C0-vs-C1 COORDINATION CAUSALITY question.

    COORDINATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.
    """
    V = "COORDINATION_CAUSAL_VALIDATED"
    P = "COORDINATION_CAUSAL_PARTIAL"
    F = "COORDINATION_CAUSAL_FAILED"
    engineering = (checkpoint_integrity_pass and protocol_integrity_pass
                   and architecture_integrity_pass and no_final_tuning)
    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass

    if not engineering:
        return {"verdict": F, "subtype": None, "reason": "engineering/integrity failure",
                "COORDINATION_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if not safety_all:
        return {"verdict": F, "subtype": None, "reason": "safety gate HARMFUL",
                "COORDINATION_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE):
        return {"verdict": F, "subtype": None,
                "reason": f"MIXED primary {mixed_primary_label}",
                "COORDINATION_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == PRACTICALLY_NEUTRAL:
        return {"verdict": P, "subtype": "PRACTICALLY_NEUTRAL",
                "reason": "MIXED performance neutral; safe + engineering ok",
                "COORDINATION_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == BENEFICIAL and policy_robustness_pass:
        return {"verdict": V, "subtype": None,
                "reason": "beneficial MIXED + robustness + safety + integrity",
                "COORDINATION_IMPROVEMENT_CLAIM": "SUPPORTED"}
    if mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE):
        return {"verdict": P, "subtype": "PERFORMANCE_PARTIAL",
                "reason": "positive MIXED but robustness unmet or CI crosses 0",
                "COORDINATION_IMPROVEMENT_CLAIM": "PARTIAL"}
    return {"verdict": F, "subtype": None, "reason": "unclassified; conservative FAILED",
            "COORDINATION_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}


# ---------------------------------------------------------------------------
# C2 credit preservation: focal regional Double-DQN target + composite loss
# ---------------------------------------------------------------------------
def focal_regional_ddqn_target(online_util: SharedRegionalQNetwork, target_util: SharedRegionalQNetwork,
                               focal_r1_reward: torch.Tensor, focal_next_state: torch.Tensor,
                               done: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """Focal regional Double-DQN target — identical to C0 regional semantics.
    a*_local = argmax_a Q_online(s'_r, a); y_local = r + gamma*(1-done)*Q_target(s'_r, a*_local).
    focal_r1_reward [B], focal_next_state [B,9], done [B] -> y_local [B]."""
    return double_dqn_target(focal_r1_reward, focal_next_state, done, online_util, target_util, gamma)


def focal_current_q(online_util: SharedRegionalQNetwork, focal_state: torch.Tensor,
                    focal_action: torch.Tensor) -> torch.Tensor:
    """q_local = Q_online(focal_state, focal_requested_action). [B,9],[B] -> [B]."""
    q_all = online_util(focal_state)
    return q_all.gather(1, focal_action.long().unsqueeze(1)).squeeze(1)


def regional_smoothl1_loss(q_local: torch.Tensor, y_local: torch.Tensor) -> torch.Tensor:
    """L_regional = SmoothL1(q_local, y_local). Same convention as C0/C1 joint."""
    return nn.SmoothL1Loss()(q_local, y_local)


def c2_composite_loss(l_joint: torch.Tensor, l_regional: torch.Tensor) -> torch.Tensor:
    """L_C2 = 0.5 * L_joint + 0.5 * L_regional. Immutable equal-component mean."""
    return JOINT_WEIGHT * l_joint + REGIONAL_WEIGHT * l_regional


def c2_optimizer_params(online_util: SharedRegionalQNetwork, online_mixer: QMIXMixerV1) -> List[nn.Parameter]:
    """C2 optimizer parameter set — identical to C1 (online utility + online mixer)."""
    return c1_optimizer_params(online_util, online_mixer)


# ---------------------------------------------------------------------------
# C2 credit config + sha256 (distinct from QMIX architecture config)
# ---------------------------------------------------------------------------
def make_c2_credit_config() -> Dict:
    """Immutable C2 credit-preservation training specification."""
    return {
        "architecture_version": C2_CREDIT_ARCH,
        "base_qmix_config_sha256": qmix_config_sha256(),
        "joint_loss_version": "JOINT_QMIX_SMOOTHL1",
        "regional_loss_version": "FOCAL_R1_DOUBLE_DQN_SMOOTHL1",
        "local_target_version": "DOUBLE_DQN_TARGET",
        "joint_weight": JOINT_WEIGHT,
        "regional_weight": REGIONAL_WEIGHT,
        "combination": "EQUAL_COMPONENT_MEAN",
        "loss_version": C2_LOSS_VERSION,
        "replay_sampling_unit": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "normalization_version": NORM_V3,
        "normalizer_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "gamma": GAMMA,
        "tau": TARGET_TAU,
    }


def c2_credit_config_sha256(config: Dict = None) -> str:
    cfg = make_c2_credit_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# C2 checkpoint version guard
# ---------------------------------------------------------------------------
def validate_c2_checkpoint_meta(meta: Dict, expected_credit_sha: str = None) -> None:
    """Hard error on wrong/missing/incompatible C2 metadata."""
    want_credit = expected_credit_sha if expected_credit_sha is not None else c2_credit_config_sha256()
    if meta.get("architecture_version") != C2_CREDIT_ARCH:
        raise CoordinationConfigError(
            f"C2 architecture_version {meta.get('architecture_version')} != {C2_CREDIT_ARCH}")
    if meta.get("c2_credit_config_sha256") != want_credit:
        raise CoordinationConfigError(
            f"C2 credit config hash {meta.get('c2_credit_config_sha256')} != {want_credit}")
    want_qmix = qmix_config_sha256()
    if meta.get("qmix_config_sha256") != want_qmix:
        raise CoordinationConfigError(
            f"C2 base qmix_config_sha256 {meta.get('qmix_config_sha256')} != {want_qmix}")
    if meta.get("reward_version") != REWARD_R1_NAME:
        raise CoordinationConfigError("C2 reward_version must be R1")
    if meta.get("reward_config_sha256") != FROZEN_R1_CONFIG_SHA256:
        raise CoordinationConfigError("C2 reward_config_sha256 mismatch")
    if meta.get("normalization_version") != NORM_V3:
        raise CoordinationConfigError("C2 normalization_version must be V3")
    if meta.get("normalizer_constants_sha256") != FROZEN_V3_CONSTANTS_SHA256:
        raise CoordinationConfigError("C2 V3 constants hash mismatch")


# ---------------------------------------------------------------------------
# C2 loss-dispersion diagnostic (descriptive, no scientific threshold)
# ---------------------------------------------------------------------------
def loss_dispersion_diagnostic(joint_losses: Sequence[float], regional_losses: Sequence[float],
                               composite_losses: Sequence[float]) -> Dict:
    jl = np.asarray(joint_losses, dtype=np.float64)
    rl = np.asarray(regional_losses, dtype=np.float64)
    cl = np.asarray(composite_losses, dtype=np.float64)
    return {
        "within_event_std_joint_loss": float(np.std(jl, ddof=0)),
        "within_event_std_regional_loss": float(np.std(rl, ddof=0)),
        "within_event_std_composite_loss": float(np.std(cl, ddof=0)),
    }


# ---------------------------------------------------------------------------
# C2 gradient-diversity diagnostic (descriptive, no scientific threshold)
# ---------------------------------------------------------------------------
def gradient_diversity_diagnostic(g_joint_flat: np.ndarray, g_regional_flat: np.ndarray,
                                  g_composite_flat: np.ndarray) -> Dict:
    gj = np.asarray(g_joint_flat, dtype=np.float64).ravel()
    gr = np.asarray(g_regional_flat, dtype=np.float64).ravel()
    gc = np.asarray(g_composite_flat, dtype=np.float64).ravel()
    norm_j = float(np.linalg.norm(gj))
    norm_r = float(np.linalg.norm(gr))
    norm_c = float(np.linalg.norm(gc))
    cos_jr = 0.0
    if norm_j > 0 and norm_r > 0:
        cos_jr = float(np.dot(gj, gr) / (norm_j * norm_r))
    return {
        "norm_joint": norm_j, "norm_regional": norm_r, "norm_composite": norm_c,
        "cosine_joint_regional": cos_jr,
    }


# ---------------------------------------------------------------------------
# Credit preservation causal verdict (frozen by E4MT/E4ME; NO mechanism gate)
# ---------------------------------------------------------------------------
def credit_preservation_causal_verdict(
    *,
    mixed_primary_label: str,
    policy_robustness_pass: bool,
    persistent_safety_pass: bool,
    stationary_safety_pass: bool,
    macro_safety_pass: bool,
    loss_spec_integrity_pass: bool,
    checkpoint_integrity_pass: bool,
    protocol_integrity_pass: bool,
    architecture_integrity_pass: bool,
    no_final_tuning: bool,
) -> Dict:
    V = "CREDIT_PRESERVATION_CAUSAL_VALIDATED"
    P = "CREDIT_PRESERVATION_CAUSAL_PARTIAL"
    F = "CREDIT_PRESERVATION_CAUSAL_FAILED"
    engineering = (loss_spec_integrity_pass and checkpoint_integrity_pass
                   and protocol_integrity_pass and architecture_integrity_pass
                   and no_final_tuning)
    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass

    if not engineering:
        return {"verdict": F, "subtype": None, "reason": "engineering/integrity failure",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if not safety_all:
        return {"verdict": F, "subtype": None, "reason": "safety gate HARMFUL",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE):
        return {"verdict": F, "subtype": None,
                "reason": f"MIXED primary {mixed_primary_label}",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == PRACTICALLY_NEUTRAL:
        return {"verdict": P, "subtype": "PRACTICALLY_NEUTRAL",
                "reason": "MIXED performance neutral; safe + engineering ok",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == BENEFICIAL and policy_robustness_pass:
        return {"verdict": V, "subtype": None,
                "reason": "beneficial MIXED + robustness + safety + integrity",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "SUPPORTED"}
    if mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE):
        return {"verdict": P, "subtype": "PERFORMANCE_PARTIAL",
                "reason": "positive MIXED but robustness unmet or CI crosses 0",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "PARTIAL"}
    return {"verdict": F, "subtype": None, "reason": "unclassified; conservative FAILED",
            "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
