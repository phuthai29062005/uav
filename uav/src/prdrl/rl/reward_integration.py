"""
Reward integration layer for the future V3+R0 vs V3+R1 experiment (added in E4II).

E4II ENGINEERING BOUNDARY: this module is integration/plumbing only. It does NOT change
the mathematical R1 formula (reward_r1.regional_quality_gain_r1, byte-stable, hash frozen in
the E4IDD config), R0 (reward.py, byte-stable), alpha_local (0.1, frozen), the global term,
the FE penalty, the state, V3, the network, or the DDQN algorithm.

It provides: reward-version dispatch, invalid-input validation for the R1 path, frozen R1
config loading + hash verification, checkpoint reward-version guards, and single-factor arm
config builders — so the future training pipeline can select R0 or R1 as the SOLE factor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional

from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward, compute_fe_cost, compute_regional_reward,
    ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT, BUDGET_PER_REGION,
)
from prdrl.rl.reward_r1 import (
    REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1, ALPHA_G_R1, BETA_R1, regional_quality_gain_r1,
)
from prdrl.rl.normalization import NORM_V3

SUPPORTED_REWARD_VERSIONS = {REWARD_R0_NAME, REWARD_R1_NAME}

# Frozen R1 reward configuration (from E4IDD). alpha_local is NOT recomputed at runtime.
FROZEN_R1_ALPHA_LOCAL = 0.1
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
E4IDD_R1_CONFIG_PATH = Path("results/E4IDD_reward_scale_diagnostic_execution/E4IDD_reward_r1_config.json")

LOCAL_ACTIVE_EPS = EPS_R1   # 1e-12


class RewardConfigError(ValueError):
    """Hard error for invalid reward configuration / inputs (never silently absorbed)."""


# ---------------------------------------------------------------------------
# Input validation (R1 path). NaN is the frozen degenerate-region convention (-> g_r=0);
# infinities and negative losses are scientifically impossible -> hard error.
# ---------------------------------------------------------------------------

def validate_regional_losses(L_pre, L_post) -> None:
    for name, v in (("L_pre", L_pre), ("L_post", L_post)):
        if v is None:
            continue
        fv = float(v)
        if math.isinf(fv):
            raise RewardConfigError(f"{name} is infinite ({fv}); regional loss must be finite >= 0")
        if (not math.isnan(fv)) and fv < 0.0:
            raise RewardConfigError(f"{name} is negative ({fv}); regional loss must be >= 0")
        # NaN passes: it is the frozen E4I degenerate-region signal (g_r -> 0).


def local_active(g_r: float) -> bool:
    return abs(float(g_r)) > LOCAL_ACTIVE_EPS


# ---------------------------------------------------------------------------
# Reward version dispatch (single-factor: only the local term differs).
# ---------------------------------------------------------------------------

def compute_local_reward(reward_version: str, L_pre, L_post) -> float:
    """R0 raw local = pbi_before - pbi_after; R1 local = g_r (bounded symmetric gain)."""
    if reward_version == REWARD_R0_NAME:
        return float(compute_regional_pbi_reward(L_pre, L_post))
    if reward_version == REWARD_R1_NAME:
        validate_regional_losses(L_pre, L_post)
        return float(regional_quality_gain_r1(L_pre, L_post))
    raise RewardConfigError(f"unsupported reward version {reward_version!r}")


def compute_regional_total(reward_version: str, *, global_reward: float, L_pre, L_post,
                           fe_cost: float, alpha_local: float) -> Dict[str, float]:
    """Total regional reward with global term and FE penalty IDENTICAL across versions.
    Only the local term (and its version) differ."""
    if reward_version not in SUPPORTED_REWARD_VERSIONS:
        raise RewardConfigError(f"unsupported reward version {reward_version!r}")
    if reward_version == REWARD_R1_NAME and abs(alpha_local - FROZEN_R1_ALPHA_LOCAL) > 0:
        raise RewardConfigError(f"R1 alpha_local must be exactly {FROZEN_R1_ALPHA_LOCAL}, got {alpha_local}")
    raw_local = compute_local_reward(reward_version, L_pre, L_post)
    w_global = ALPHA_G_DEFAULT * float(global_reward)
    w_local = float(alpha_local) * raw_local
    w_fe = BETA_DEFAULT * float(fe_cost)
    total = w_global + w_local - w_fe
    return {"raw_global_reward": float(global_reward), "weighted_global_reward": w_global,
            "raw_local_reward": raw_local, "weighted_local_reward": w_local,
            "weighted_fe_penalty": w_fe, "total_reward": total}


# ---------------------------------------------------------------------------
# Frozen R1 config loading + hash verification.
# ---------------------------------------------------------------------------

def _config_sha256(cfg: dict) -> str:
    c = dict(cfg)
    c.pop("REWARD_R1_CONFIG_SHA256", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def load_r1_config(path: Path = E4IDD_R1_CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    stored = cfg.get("REWARD_R1_CONFIG_SHA256")
    recomputed = _config_sha256(cfg)
    if stored != recomputed:
        raise RewardConfigError(f"R1 config hash mismatch: stored {stored} != recomputed {recomputed}")
    if stored != FROZEN_R1_CONFIG_SHA256:
        raise RewardConfigError(f"R1 config hash {stored} != frozen {FROZEN_R1_CONFIG_SHA256}")
    if cfg["reward_version"] != REWARD_R1_NAME:
        raise RewardConfigError("R1 config reward_version mismatch")
    if float(cfg["alpha_local"]) != FROZEN_R1_ALPHA_LOCAL:
        raise RewardConfigError("R1 config alpha_local mismatch")
    return cfg


def verify_r1_config_hash(cfg: dict) -> bool:
    return _config_sha256(cfg) == cfg.get("REWARD_R1_CONFIG_SHA256") == FROZEN_R1_CONFIG_SHA256


# ---------------------------------------------------------------------------
# Checkpoint reward-version guard.
# ---------------------------------------------------------------------------

def validate_checkpoint_reward_version(meta: dict, expected_version: str,
                                       expected_config_sha: Optional[str] = None) -> bool:
    v = meta.get("reward_version")
    if v is None:
        raise RewardConfigError("checkpoint has no reward_version; cannot be used as a future "
                                "reward-experiment arm (historical checkpoints are never relabeled)")
    if v not in SUPPORTED_REWARD_VERSIONS:
        raise RewardConfigError(f"unsupported reward version {v!r}")
    if v != expected_version:
        raise RewardConfigError(f"reward version mismatch: checkpoint {v} != expected {expected_version}")
    if expected_version == REWARD_R1_NAME:
        sha = meta.get("reward_config_sha256")
        want = expected_config_sha if expected_config_sha is not None else FROZEN_R1_CONFIG_SHA256
        if sha != want:
            raise RewardConfigError(f"R1 checkpoint config hash mismatch: {sha} != {want}")
    return True


# ---------------------------------------------------------------------------
# Single-factor arm config builders.
# ---------------------------------------------------------------------------

_SHARED_SCIENTIFIC_CONFIG = {
    "normalization": NORM_V3, "state_dim": 9, "network": "9-128-128-4", "algorithm": "SHARED_DOUBLE_DQN",
    "gamma": 0.99, "learning_rate": 1e-4, "optimizer": "Adam", "loss": "SmoothL1", "batch_size": 128,
    "replay_capacity": 50000, "warmup_transitions": 1000, "update_frequency": 4, "grad_steps": 1,
    "target_tau": 0.005, "grad_clip": None, "epsilon_schedule": "1.0->0.05 over 50000 transitions",
    "n_epochs": 6, "episodes_per_policy": 1080,
    "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
                  "predict_fallback": "LOCAL", "sigma_D": 0.15, "budget": "10 PER_ACTIVE_REGION",
                  "survival": "unrestricted NSGA-II"},
    "alpha_global": 1.0, "beta": 0.01,
    "global_reward": "normalized_IGD_improvement_of_joint_population",
    "fe_penalty": "response_fe/(10*n_active)",
    "checkpoint_selection": "VALIDATION_MACRO_MIGD_RESPONSE",
    "train_seeds": [1501, 1560], "val_seeds": [1561, 1580], "final_seeds": [1581, 1640],
    "policy_seeds": [4001, 4002, 4003, 4004, 4005], "paired_initialization": True,
    "joint_representation": "IMPLICIT_STATE_ONLY",
}


def build_arm_config(arm: str) -> dict:
    cfg = dict(_SHARED_SCIENTIFIC_CONFIG)
    if arm == "R0_CONTROL":
        cfg.update({"arm": "R0_CONTROL", "reward_version": REWARD_R0_NAME, "alpha_local": ALPHA_L_DEFAULT})
    elif arm == "R1_TREATMENT":
        cfg.update({"arm": "R1_TREATMENT", "reward_version": REWARD_R1_NAME,
                    "alpha_local": FROZEN_R1_ALPHA_LOCAL, "reward_config_sha256": FROZEN_R1_CONFIG_SHA256})
    else:
        raise RewardConfigError(f"unknown arm {arm!r}")
    return cfg


def arm_config_scientific_diff(a: dict, b: dict) -> Dict[str, list]:
    """Fields that differ after excluding intrinsic reward-identity fields. Empty => single-factor."""
    ignore = {"arm", "reward_version", "reward_config_sha256", "alpha_local"}
    keys = (set(a) | set(b)) - ignore
    diff = [k for k in sorted(keys) if a.get(k) != b.get(k)]
    return {"differing_fields": diff, "single_factor": len(diff) == 0}


def reward_mechanism_supported(delta_activity: float, delta_ratio: float,
                               delta_dispersion: float) -> Dict:
    """Reward-credit mechanism gate (Branch B). Treatment-favoring directions:
      delta_activity   = R1_active_fraction - R0_active_fraction  (>0 expected)
      delta_ratio      = R0_global_local_ratio - R1_global_local_ratio (>0 expected: R1 less overwhelmed)
      delta_dispersion = R1_regional_dispersion - R0_regional_dispersion (>0 expected)
    SUPPORTED iff >= 2 of 3 point-estimate effects are in the expected (>0) direction.
    A NOT_APPLICABLE ratio delta (no active local signal) counts as NOT in the expected direction."""
    da = int(delta_activity > 0) if isinstance(delta_activity, (int, float)) else 0
    dr = int(delta_ratio > 0) if isinstance(delta_ratio, (int, float)) else 0
    dd = int(delta_dispersion > 0) if isinstance(delta_dispersion, (int, float)) else 0
    n = da + dr + dd
    return {"delta_activity": delta_activity, "delta_ratio": delta_ratio,
            "delta_dispersion": delta_dispersion, "n_expected": n,
            "REWARD_CREDIT_MECHANISM_SUPPORTED": "YES" if n >= 2 else "NO"}


def reward_ab_verdict(**kwargs) -> Dict:
    """Reward R0-vs-R1 causal verdict. Structurally IDENTICAL to the frozen normalization-A/B
    decision logic (final_verdict.normalization_ab_verdict); only the labels are relabeled to
    REWARD_CAUSAL_* so the reward experiment reuses the exact, already-tested gate semantics
    (mechanism never rescues harmful; beneficial-without-mechanism = PARTIAL; neutral+mechanism =
    MECHANISTIC_PARTIAL with PERFORMANCE_IMPROVEMENT_CLAIM=NOT_SUPPORTED)."""
    from prdrl.evaluation.final_verdict import normalization_ab_verdict
    v = normalization_ab_verdict(**kwargs)
    relabel = {"NORMALIZATION_CAUSAL_VALIDATED": "REWARD_CAUSAL_VALIDATED",
               "NORMALIZATION_CAUSAL_PARTIAL": "REWARD_CAUSAL_PARTIAL",
               "NORMALIZATION_CAUSAL_FAILED": "REWARD_CAUSAL_FAILED"}
    return {"verdict": relabel[v["verdict"]], "subtype": v["subtype"], "reason": v["reason"],
            "PERFORMANCE_IMPROVEMENT_CLAIM": v["PERFORMANCE_IMPROVEMENT_CLAIM"]}
