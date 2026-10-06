"""Diagnostic configuration, version guards, and config hash for E4OI."""
from __future__ import annotations

import hashlib
import json
from typing import Dict


DIAGNOSTIC_VERSION = "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_V1"

REQUIRED_QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
REQUIRED_C2_CREDIT_SHA = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"
REQUIRED_R1_SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
REQUIRED_V3_SHA = "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"

DIAGNOSTIC_SEEDS = list(range(2301, 2361))
DIAGNOSTIC_BOOTSTRAP_N = 50000
DIAGNOSTIC_BOOTSTRAP_SEED = 116001
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
N_DIAGNOSTIC_ENVS = 60
DIAGNOSTIC_EPISODES = 1800
EPSILON_EVAL = 0.0

STATE_DIM = 9
CENTRAL_STATE_DIM = 55
R_MAX = 10
N_ACTIONS = 4
EPSILON_COSINE = 1e-12
JOINT_WEIGHT = 0.5
REGIONAL_WEIGHT = 0.5
GAMMA = 0.99
TARGET_TAU = 0.005

GRADIENT_MODE = "OFFLINE_ONLY"
OPTIMIZER_STEPS = 0
PARAMETER_CHANGES = 0


def make_diagnostic_config() -> Dict:
    """Immutable E4OI diagnostic configuration."""
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "qmix_config_sha256": REQUIRED_QMIX_SHA,
        "c2_credit_config_sha256": REQUIRED_C2_CREDIT_SHA,
        "r1_config_sha256": REQUIRED_R1_SHA,
        "v3_constants_sha256": REQUIRED_V3_SHA,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "policy_seeds": POLICY_SEEDS,
        "scenarios": SCENARIOS,
        "n_diagnostic_envs": N_DIAGNOSTIC_ENVS,
        "total_episodes": DIAGNOSTIC_EPISODES,
        "epsilon_eval": EPSILON_EVAL,
        "bootstrap_N": DIAGNOSTIC_BOOTSTRAP_N,
        "bootstrap_seed": DIAGNOSTIC_BOOTSTRAP_SEED,
        "state_dim": STATE_DIM,
        "central_state_dim": CENTRAL_STATE_DIM,
        "r_max": R_MAX,
        "n_actions": N_ACTIONS,
        "epsilon_cosine": EPSILON_COSINE,
        "joint_weight": JOINT_WEIGHT,
        "regional_weight": REGIONAL_WEIGHT,
        "gamma": GAMMA,
        "target_tau": TARGET_TAU,
        "gradient_mode": GRADIENT_MODE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "parameter_changes": PARAMETER_CHANGES,
    }


def diagnostic_config_sha256(config: Dict = None) -> str:
    cfg = make_diagnostic_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_upstream_hashes(qmix_sha: str, c2_sha: str, r1_sha: str, v3_sha: str) -> Dict:
    """Verify all upstream config hashes match frozen requirements."""
    checks = {
        "qmix": qmix_sha == REQUIRED_QMIX_SHA,
        "c2_credit": c2_sha == REQUIRED_C2_CREDIT_SHA,
        "r1_reward": r1_sha == REQUIRED_R1_SHA,
        "v3_normalization": v3_sha == REQUIRED_V3_SHA,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
    }
