"""E4PI nonmonotonic interaction diagnostic configuration and version guard."""
from __future__ import annotations

import hashlib
import json
from typing import Dict

DIAGNOSTIC_VERSION = "E4PI_NONMONOTONIC_DIAGNOSTIC_V1"

PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
EPSILON_RANK = 1e-12
N_RANDOM_TAPES = 4
PRIMARY_BACKGROUND = "KEEP_BACKGROUND"
ROBUSTNESS_BACKGROUND = "SOURCE_POLICY_BACKGROUND"
SOURCE_ARM_WEIGHT = 0.5
SEED_RANGE = list(range(2401, 2421))
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
BOOTSTRAP_POLICY_AXIS = 5
BOOTSTRAP_ENV_AXIS = 20
BOOTSTRAP_N = 50000
BOOTSTRAP_SEED = 126001
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

OPERATOR_CONSTANTS = {
    "KEEP": {"FE": 0},
    "LOCAL": {"sigma_L": 0.02, "B_PER_ACTIVE_REGION": 10},
    "PREDICT": {"sigma_P": 0.005, "lambda_P": 1.0, "B_PER_ACTIVE_REGION": 10,
                "fallback": "LOCAL"},
    "DIVERSIFY": {"sigma_D": 0.15, "B_PER_ACTIVE_REGION": 10},
}
SURVIVAL = "UNRESTRICTED_STANDARD_NSGA_II"


def make_nonmonotonic_config() -> Dict:
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "primary_actions": PRIMARY_ACTIONS,
        "secondary_actions": SECONDARY_ACTIONS,
        "epsilon_rank": EPSILON_RANK,
        "n_random_tapes": N_RANDOM_TAPES,
        "primary_background": PRIMARY_BACKGROUND,
        "robustness_background": ROBUSTNESS_BACKGROUND,
        "source_arm_weight": SOURCE_ARM_WEIGHT,
        "seed_range": SEED_RANGE,
        "policy_seeds": POLICY_SEEDS,
        "bootstrap_policy_axis": BOOTSTRAP_POLICY_AXIS,
        "bootstrap_env_axis": BOOTSTRAP_ENV_AXIS,
        "bootstrap_N": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scenarios": SCENARIOS,
        "operator_constants": OPERATOR_CONSTANTS,
        "survival": SURVIVAL,
    }


def nonmonotonic_config_sha256(config: Dict = None) -> str:
    cfg = make_nonmonotonic_config() if config is None else config
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_config(config: Dict) -> Dict:
    ref = make_nonmonotonic_config()
    errors = []
    if config.get("primary_actions") != ref["primary_actions"]:
        errors.append("wrong_primary_actions")
    if config.get("secondary_actions") != ref["secondary_actions"]:
        errors.append("wrong_secondary_actions")
    if config.get("n_random_tapes") != ref["n_random_tapes"]:
        errors.append("wrong_random_tape_count")
    if config.get("epsilon_rank") != ref["epsilon_rank"]:
        errors.append("wrong_epsilon_rank")
    if config.get("seed_range") != ref["seed_range"]:
        errors.append("wrong_seed_range")
    if config.get("policy_seeds") != ref["policy_seeds"]:
        errors.append("wrong_policy_seeds")
    if config.get("bootstrap_N") != ref["bootstrap_N"]:
        errors.append("wrong_bootstrap_N")
    if config.get("bootstrap_seed") != ref["bootstrap_seed"]:
        errors.append("wrong_bootstrap_seed")
    return {"valid": len(errors) == 0, "errors": errors}
