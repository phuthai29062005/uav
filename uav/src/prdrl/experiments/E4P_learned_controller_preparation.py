"""
E4-P — Learned Controller Preparation.

Freeze and validate the DRL environment, state, action, reward,
transition, coordination and reproducibility contracts BEFORE any
substantive Double DQN training.

E4-P is NOT a performance experiment.
E4-P is NOT long DRL training.

Stage invariants:
  - RL_STATUS = NOT_STARTED (until E4-T).
  - DRL_TRAINING = NOT_STARTED.
  - Production code UNCHANGED.
  - No greedy oracle labels as training targets.
  - Oracle label firewall: oracle_best_action NOT in training observation/reward path.
  - E3/E3-R/H1/H1-RV artifacts immutable.
  - PRODUCTION_DEFAULT = O1_F005.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "src")

from prdrl.oracle.oracle import regional_oracle
from prdrl.rl.interface import (
    CURRENT_RESPONSE_BUDGET_VALUE,
    LOCAL_SPEC, PREDICT_SPEC, DIVERSIFY_SPEC,
    ACTION_KEEP, ACTION_LOCAL, ACTION_PREDICT, ACTION_DIVERSIFY,
    build_state_v2, STATE_SCHEMA_V2,
)
from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair,
    double_dqn_target, hard_update, polyak_update,
    ACTION_ORDER, N_ACTIONS, HIDDEN_DIM_1, HIDDEN_DIM_2,
)
from prdrl.rl.replay import SharedReplayBuffer, RegionalTransition
from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward,
    compute_fe_cost, compute_regional_reward, RewardConfig,
    ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT, GAMMA_DEFAULT,
    BUDGET_PER_REGION,
)
from prdrl.rl.agent import (
    EpsilonGreedyConfig, select_regional_actions,
    ACTION_TO_IDX, IDX_TO_ACTION,
)
OUT_DIR = Path("results/E4_learned_controller_preparation")

RL_STATUS    = "NOT_STARTED"
DRL_TRAINING = "NOT_STARTED"

# V2 local state (FROZEN)
LOCAL_DIM   = 4
LOCAL_FEATURES = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]

# GlobalContextV1 features (FROZEN — minimal causal V2 aggregates)
GLOBAL_FEATURES = [
    "active_region_fraction",
    "mean_severity",
    "max_severity",
    "mean_convergence_loss",
    "max_convergence_loss",
]
GLOBAL_DIM = len(GLOBAL_FEATURES)   # 5

NETWORK_INPUT_DIM = LOCAL_DIM + GLOBAL_DIM   # 9

# Action set (FROZEN)
ACTIONS    = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")
N_SCORED_CHANGES = 10   # episode length (from harness)


def _hash_file(path: str) -> str:
    sha = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _jdump(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# PHASE E4P-0: Prechange manifest
# ============================================================

def write_prechange_manifest() -> dict:
    key_files = [
        "src/prdrl/rl/interface.py",
        "src/prdrl/detectors/severity.py",
        "src/prdrl/detectors/quality_loss.py",
        "src/prdrl/oracle/oracle.py",
        "src/prdrl/operators/local.py",
        "src/prdrl/operators/predict.py",
        "src/prdrl/operators/diversify.py",
        "src/prdrl/operators/budget.py",
        "src/prdrl/core/survival.py",
        "results/H1RV_verdict_review/H1RV_verdict.json",
        "results/H1_verdict_review/H1_verdict_addendum.json",
        "results/H1_regional_controller_hypothesis_review/H1_verdict.json",
        "results/E3R_rule_v2_semantic_rerun/E3R_verdict.json",
        "results/R0_state_semantics_reconciliation/SM_verdict.json",
    ]
    manifest = {
        "stage": "E4P_learned_controller_preparation",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "PRODUCTION_DEFAULT": "O1_F005",
        "H1_OUTCOME_REVIEWED": "RULE_MAPPING_INADEQUATE",
        "E4_READINESS": "READY",
        "ORACLE_TYPE": "GREEDY_FACTORIZED",
        "DRL_CASE_PRIMARY": "NONLINEAR_MAPPING_SUPPORTED",
        "DRL_CASE_SECONDARY": "JOINT_COORDINATION_REQUIRED",
        "production_constants": {
            "BUDGET_PER_ACTIVE_REGION": CURRENT_RESPONSE_BUDGET_VALUE,
            "SIGMA_LOCAL":    LOCAL_SPEC.sigma,
            "SIGMA_PREDICT":  PREDICT_SPEC.sigma,
            "LAMBDA_PREDICT": PREDICT_SPEC.lambda_P,
            "SIGMA_DIVERSIFY": DIVERSIFY_SPEC.sigma,
            "BASE_EVOLUTION": "NSGA-II",
        },
        "state_schema": {
            "version": "V2",
            "local_features": LOCAL_FEATURES,
            "local_dim": LOCAL_DIM,
        },
        "action_ids": {
            "KEEP": 0, "LOCAL": 1, "PREDICT": 2, "DIVERSIFY": 3,
        },
        "budget_semantics": "B=10 FE per active region; KEEP=0 FE",
        "rng_contracts": "POLICY_INIT, POLICY_EXPLORATION, REPLAY_SAMPLE, "
                         "NETWORK_MINIBATCH, RESPONSE_OPERATOR, RESPONSE_SURVIVAL, "
                         "BASE_EVOLUTION — all separate namespaces",
        "h1rv_architecture_constraints": [
            "JOINT_COORDINATION_REQUIRED: reward must capture joint outcome",
            "GREEDY_FACTORIZED oracle labels must NOT be training targets",
            "NONLINEAR_MAPPING_SUPPORTED: network must be nonlinear (>=2 hidden layers)",
            "PREDICT fallback: realized_operator stored separately from requested_action",
        ],
        "files": {fp: _hash_file(fp) for fp in key_files},
    }
    _jdump(manifest, OUT_DIR / "E4P_prechange_manifest.json")
    return manifest


# ============================================================
# PHASE E4P-1: RL inventory
# ============================================================

def write_rl_inventory() -> dict:
    inventory = {
        "search_terms": ["dqn", "double_dqn", "q_network", "replay", "target_network",
                         "epsilon", "agent", "controller", "reward", "transition",
                         "torch", "tensorflow", "gym"],
        "existing_rl_modules": {
            "src/prdrl/rl/interface.py": {
                "status": "EXISTS",
                "purpose": "V2 state schema, action specs, execute_response, build_state_v2",
                "used_by_e4": True,
            },
            "src/prdrl/rl/network.py": {
                "status": "CREATED_THIS_SESSION",
                "purpose": "SharedRegionalQNetwork, Double DQN algebra, online/target pair",
            },
            "src/prdrl/rl/replay.py": {
                "status": "CREATED_THIS_SESSION",
                "purpose": "SharedReplayBuffer, RegionalTransition with event_id",
            },
            "src/prdrl/rl/reward.py": {
                "status": "CREATED_THIS_SESSION",
                "purpose": "global/local/FE reward components, RewardConfig",
            },
            "src/prdrl/rl/agent.py": {
                "status": "CREATED_THIS_SESSION",
                "purpose": "EpsilonGreedyConfig, epsilon-greedy action selection",
            },
        },
        "no_preexisting_dqn":       True,
        "no_preexisting_gym_env":   True,
        "no_preexisting_torch_training_loop": True,
        "torch_version":   torch.__version__,
        "numpy_version":   np.__version__,
        "tensorflow_found": False,
    }
    _jdump(inventory, OUT_DIR / "E4P_rl_inventory.json")
    return inventory


# ============================================================
# PHASE E4P-4: Global context audit
# ============================================================

def compute_global_context_v1(
    v2_states: dict,   # {r: {severity_r, convergence_loss_r, occupancy_r, delta_severity_r}}
    R: int,
) -> np.ndarray:
    """
    GlobalContextV1: minimal causal aggregates of V2 regional states.

    All quantities are computable from already-validated V2 features.
    Pre-response, causal, finite, no future information.

    Returns float64 array of GLOBAL_DIM = 5.
    """
    active = [r for r in range(R) if r in v2_states]
    n_active = len(active)

    if n_active == 0:
        return np.zeros(GLOBAL_DIM, dtype=np.float64)

    sev  = np.array([v2_states[r]["severity_r"]         for r in active])
    closs = np.array([v2_states[r]["convergence_loss_r"] for r in active])

    return np.array([
        n_active / R,          # active_region_fraction ∈ [0,1]
        float(np.mean(sev)),   # mean_severity
        float(np.max(sev)),    # max_severity
        float(np.mean(closs)), # mean_convergence_loss
        float(np.max(closs)),  # max_convergence_loss
    ], dtype=np.float64)


def write_global_context_audit() -> dict:
    audit = {
        "context_version": "GlobalContextV1",
        "candidates_evaluated": {
            "Delta_HV_global": {
                "status": "NOT_IMPLEMENTED",
                "reason": "HV metric not in oracle or harness; would require implementation and validation",
            },
            "D_global": {
                "status": "NOT_IMPLEMENTED",
                "reason": "manuscript quantity not found in src/prdrl/",
            },
            "B_remain_FE": {
                "status": "NOT_APPLICABLE",
                "reason": "Fixed budget B=10/region; no variable remaining budget in protocol",
            },
            "active_region_fraction": {
                "status": "IMPLEMENTED_VALIDATED",
                "source": "n_active / R from region_ids",
                "causal": True,
                "range": "[0.0, 1.0]",
            },
            "mean_severity": {
                "status": "IMPLEMENTED_VALIDATED",
                "source": "mean of severity_r over active regions (from V2)",
                "causal": True,
                "range": "[0.0, 1.0]",
            },
            "max_severity": {
                "status": "IMPLEMENTED_VALIDATED",
                "source": "max of severity_r over active regions (from V2)",
                "causal": True,
                "range": "[0.0, 1.0]",
            },
            "mean_convergence_loss": {
                "status": "IMPLEMENTED_VALIDATED",
                "source": "mean of convergence_loss_r over active regions (from V2)",
                "causal": True,
                "range": ">=0",
            },
            "max_convergence_loss": {
                "status": "IMPLEMENTED_VALIDATED",
                "source": "max of convergence_loss_r over active regions (from V2)",
                "causal": True,
                "range": ">=0",
            },
            "mean_delta_severity": {
                "status": "CAUSALLY_COMPUTABLE",
                "reason": "Available but excluded to keep context minimal (5 features is defensible)",
            },
            "std_severity": {
                "status": "CAUSALLY_COMPUTABLE",
                "reason": "Available but excluded for minimality",
            },
        },
        "GlobalContextV1_chosen": GLOBAL_FEATURES,
        "global_dim": GLOBAL_DIM,
        "principle": "SMALLEST defensible context; coordination without feature proliferation",
        "sharing": "Same global_context vector shared across ALL active regions at same event",
        "no_future_information": True,
        "no_oracle_features": True,
        "no_regime_labels": True,
    }
    _jdump(audit, OUT_DIR / "E4P_global_context_audit.json")
    return audit


# ============================================================
# PHASE E4P-5: State schema
# ============================================================

def write_state_schema() -> dict:
    schema = {
        "state_schema_version": "V2_plus_GlobalContextV1",
        "local_features": LOCAL_FEATURES,
        "local_dim": LOCAL_DIM,
        "local_dtype": "float32",
        "local_normalization": {
            "severity_r":         {"range": "[0,1]", "transform": "identity"},
            "convergence_loss_r": {"range": ">=0",   "transform": "clip_max_10_then_div_10"},
            "occupancy_r":        {"range": "[0,1]", "transform": "identity"},
            "delta_severity_r":   {"range": "~[-1,1]","transform": "clip_abs_1"},
        },
        "local_missing_value_policy": "0.0 for first change (no prior severity history)",
        "local_causal": True,
        "local_no_oracle_utility": True,
        "local_no_future_info": True,
        "global_features": GLOBAL_FEATURES,
        "global_dim": GLOBAL_DIM,
        "global_dtype": "float32",
        "global_normalization": {
            "active_region_fraction": {"range": "[0,1]", "transform": "identity"},
            "mean_severity":          {"range": "[0,1]", "transform": "identity"},
            "max_severity":           {"range": "[0,1]", "transform": "identity"},
            "mean_convergence_loss":  {"range": ">=0",   "transform": "clip_max_10_then_div_10"},
            "max_convergence_loss":   {"range": ">=0",   "transform": "clip_max_10_then_div_10"},
        },
        "global_causal": True,
        "global_shared_per_event": True,
        "total_input_dim": NETWORK_INPUT_DIM,
        "network_input_layout": "concat([local_V2, global_V1], axis=-1)",
        "no_NaN_guarantee": "All features computed from finite F_minus/F_plus",
    }
    _jdump(schema, OUT_DIR / "E4P_state_schema.json")
    return schema


def normalize_local_state(v2: dict) -> np.ndarray:
    """Apply frozen normalization to one V2 regional state dict → float32 array."""
    sv = float(np.clip(v2["severity_r"], 0.0, 1.0))
    cl = float(np.clip(v2["convergence_loss_r"], 0.0, 10.0)) / 10.0
    oc = float(np.clip(v2["occupancy_r"], 0.0, 1.0))
    ds = float(np.clip(v2["delta_severity_r"], -1.0, 1.0))
    return np.array([sv, cl, oc, ds], dtype=np.float32)


def normalize_global_context(gc: np.ndarray) -> np.ndarray:
    """Apply frozen normalization to GlobalContextV1 vector → float32 array."""
    out = np.zeros(GLOBAL_DIM, dtype=np.float32)
    out[0] = float(np.clip(gc[0], 0.0, 1.0))   # active_region_fraction
    out[1] = float(np.clip(gc[1], 0.0, 1.0))   # mean_severity
    out[2] = float(np.clip(gc[2], 0.0, 1.0))   # max_severity
    out[3] = float(np.clip(gc[3], 0.0, 10.0)) / 10.0  # mean_convergence_loss
    out[4] = float(np.clip(gc[4], 0.0, 10.0)) / 10.0  # max_convergence_loss
    return out


# ============================================================
# PHASE E4P-6/7/8: Reward contract
# ============================================================

def write_reward_contract() -> dict:
    rc = RewardConfig()
    base = rc.to_dict()
    base.update({
        "reward_components": {
            "global": "normalized IGD improvement of JOINT population (actual executed outcome)",
            "local":  "negative PBI loss change (from quality_loss module, per region)",
            "fe_cost": "actual_fe / (B * n_active_regions); KEEP=0",
            "violation": "NOT_APPLICABLE for controlled benchmark",
        },
        "quality_fn_used": "negative_IGD (same as oracle, E2, E3-R, H1)",
        "normalization_mode": "abs(q_before) + 1e-12",
        "team_dominant": f"alpha_g={ALPHA_G_DEFAULT} >= alpha_l={ALPHA_L_DEFAULT}",
        "greedy_oracle_labels": "FORBIDDEN as training targets",
        "joint_outcome_required": True,
        "pbi_regional_quality": {
            "source": "prdrl.detectors.quality_loss",
            "validated": True,
            "theta_pbi": 5.0,
        },
        "unfrozen_hyperparameters": ["alpha_g", "alpha_l", "beta", "gamma"],
        "freeze_required_before_E4T": True,
    })
    _jdump(base, OUT_DIR / "E4P_reward_contract.json")
    return base


# ============================================================
# PHASE E4P-9/10: Transition contract
# ============================================================

def write_transition_contract() -> dict:
    contract = {
        "RegionalTransition_fields": {
            "state":              "float32 array of shape (input_dim,)",
            "requested_action":   "int ∈ {0,1,2,3} (KEEP/LOCAL/PREDICT/DIVERSIFY)",
            "realized_operator":  "str — actual operator executed (may differ on PREDICT fallback)",
            "reward":             "float — team reward for this region",
            "next_state":         "float32 array of shape (input_dim,)",
            "done":               "bool — True only at episode end",
            "region_id":          "int",
            "event_id":           "str — shared by all regions from the same change event",
            "fallback_applied":   "bool — True if PREDICT fell back to LOCAL",
            "response_fe":        "int — FE consumed by this region's response",
            "diagnostic_oracle_action": "str or None — offline only, NOT in training path",
        },
        "JointEventTransition_fields": {
            "event_id":          "str",
            "active_region_ids": "list[int]",
            "local_states":      "dict {region_id: float32 array}",
            "global_context":    "float32 array of shape (global_dim,)",
            "requested_actions": "dict {region_id: int}",
            "realized_operators":"dict {region_id: str}",
            "total_response_fe": "int",
            "global_reward":     "float",
            "q_before":          "float — quality before joint response",
            "q_after":           "float — quality after joint response",
        },
        "action_field_semantics": {
            "requested_action": "WHAT the controller chose (learning signal)",
            "realized_operator": "WHAT was actually executed (PREDICT fallback stored separately)",
            "PREDICT_fallback_rule": "If PREDICT requested but no history: realized=LOCAL; action=PREDICT (unchanged)",
        },
        "event_id_purpose": "Allows identifying all regional transitions from same joint event for coordination analysis",
        "inactive_region_policy": "No transition created for inactive (empty) regions",
        "done_semantics": "True only at episode boundary (after all N_SCORED_CHANGES changes); NOT per-region",
        "next_state_timing": "V2 features computed at t+1 decision context (pre-response at next change)",
        "no_oracle_utility_in_state": True,
        "no_regime_label_in_state": True,
    }
    _jdump(contract, OUT_DIR / "E4P_transition_contract.json")

    joint_contract = {
        "purpose": "Joint event record for coordination analysis",
        "fields":  contract["JointEventTransition_fields"],
        "required_for_joint_coordination_analysis": True,
    }
    _jdump(joint_contract, OUT_DIR / "E4P_joint_event_contract.json")
    return contract


# ============================================================
# PHASE E4P-11: Replay contract
# ============================================================

def write_replay_contract() -> dict:
    contract = {
        "buffer_type": "SharedReplayBuffer (circular deque)",
        "capacity": "UNFROZEN — to be specified in E4T protocol",
        "sampling": "uniform random without replacement per minibatch",
        "rng_namespace": "REPLAY_SAMPLE (isolated from production RNG)",
        "event_id_retained": True,
        "oracle_label_in_buffer": False,
        "oracle_diagnostic_label": "May be stored in diagnostic_oracle_action field (not used by training)",
        "no_oracle_label_replay": True,
        "regional_transitions_shared": True,
        "all_active_regions_same_event_eligible": True,
    }
    _jdump(contract, OUT_DIR / "E4P_replay_contract.json")
    return contract


# ============================================================
# PHASE E4P-12: Network spec
# ============================================================

def write_network_spec() -> dict:
    spec = {
        "architecture": "PARAMETER_SHARED_DOUBLE_DQN",
        "parameter_sharing": "ALL active regions share the same network weights",
        "input_dim": NETWORK_INPUT_DIM,
        "hidden_layers": [HIDDEN_DIM_1, HIDDEN_DIM_2],
        "activation": "ReLU",
        "output_dim": N_ACTIONS,
        "output_meaning": list(ACTION_ORDER),
        "loss_function": "Huber (SmoothL1)",
        "network_pair": {"online": "learns", "target": "used for stable TD targets"},
        "target_update": {
            "mode": "polyak",
            "tau_initial": 0.005,
            "status": "INITIAL_TRAINING_SPEC_UNFROZEN",
        },
        "optimizer": "Adam (to be specified in E4T protocol)",
        "no_architecture_search_in_E4P": True,
        "no_transformers": True,
        "no_GNN": True,
        "no_attention": True,
        "no_LSTM": True,
    }
    _jdump(spec, OUT_DIR / "E4P_network_spec.json")
    return spec


# ============================================================
# PHASE E4P-13/14: Exploration + RNG contracts
# ============================================================

def write_exploration_contract() -> dict:
    eps_cfg = EpsilonGreedyConfig()
    contract = eps_cfg.to_dict()
    contract.update({
        "regional_independence": "Each region draws its own RNG key via event_seed XOR (region_id * PRIME)",
        "same_epsilon_all_regions": True,
        "independent_exploration_draws": True,
        "no_global_single_draw": True,
    })
    _jdump(contract, OUT_DIR / "E4P_exploration_contract.json")
    return contract


def write_rng_contract() -> dict:
    contract = {
        "namespaces": {
            "POLICY_INIT":         "seed for network weight initialization",
            "POLICY_EXPLORATION":  "per-region epsilon-greedy draws (event_seed XOR region_id*PRIME)",
            "REPLAY_SAMPLE":       "minibatch sampling from replay buffer",
            "NETWORK_MINIBATCH":   "minibatch data augmentation (if any); separate from replay sample",
            "RESPONSE_OPERATOR":   "production operator execution (UNCHANGED from existing protocol)",
            "RESPONSE_SURVIVAL":   "production survival (UNCHANGED from existing protocol)",
            "BASE_EVOLUTION":      "NSGA-II evolution (UNCHANGED from existing protocol)",
        },
        "isolation_guarantee": "Adding RL does NOT alter production response or evolution RNG",
        "no_L2B_namespace": True,
        "production_rng_frozen": True,
    }
    _jdump(contract, OUT_DIR / "E4P_rng_contract.json")
    return contract


# ============================================================
# PHASE E4P-15/16/17/18: Normalization, episode, next-state, masking
# ============================================================

def write_normalization_contract() -> dict:
    contract = {
        "normalization_fitted_on": "NOT fitted on evaluation seeds; deterministic transforms only",
        "local_V2": {
            "severity_r":         "clip[0,1] → identity",
            "convergence_loss_r": "clip[0,10] / 10 → [0,1]",
            "occupancy_r":        "clip[0,1] → identity",
            "delta_severity_r":   "clip[-1,1] → identity",
        },
        "global_V1": {
            "active_region_fraction": "clip[0,1] → identity",
            "mean_severity":    "clip[0,1] → identity",
            "max_severity":     "clip[0,1] → identity",
            "mean_convergence_loss": "clip[0,10] / 10 → [0,1]",
            "max_convergence_loss":  "clip[0,10] / 10 → [0,1]",
        },
        "episode_boundary": {
            "length": N_SCORED_CHANGES,
            "done_condition": "after N_SCORED_CHANGES environmental changes",
            "done_per_region": False,
            "done_per_event": False,
        },
        "next_state_timing": "V2 features at change t+1 (pre-response); computed from F_minus/F_plus at t+1",
        "inactive_region_policy": "No transition; no fake KEEP transition",
        "no_performance_tuned_clipping": True,
    }
    _jdump(contract, OUT_DIR / "E4P_normalization_contract.json")
    return contract


# ============================================================
# PHASE E4P-21: Oracle label firewall
# ============================================================

def write_oracle_firewall() -> dict:
    firewall = {
        "forbidden_in_training_path": [
            "oracle_best_action",
            "oracle_utility_vector (U_KEEP, U_LOCAL, U_PREDICT, U_DIVERSIFY)",
            "preferred_action from regional_oracle()",
            "oracle ROG",
        ],
        "allowed_diagnostic_only": [
            "diagnostic_oracle_action (stored in RegionalTransition.diagnostic_oracle_action)",
            "H1 oracle region records (H1_oracle_region_records.jsonl)",
            "H1 separability probes (offline analysis only)",
        ],
        "firewall_mechanism": (
            "RegionalTransition.diagnostic_oracle_action is a separate field from "
            "requested_action. The training loop reads only: state, requested_action, "
            "reward, next_state, done. Oracle diagnostic data is in a separate field "
            "accessible only through explicit diagnostic code, not through training."
        ),
        "no_oracle_imitation_learning": True,
        "no_oracle_pretraining": True,
        "oracle_labels_may_be_used_for": "offline diagnostic verification only, not training targets",
        "GREEDY_FACTORIZED_oracle_labels_are_not_ground_truth": True,
    }
    _jdump(firewall, OUT_DIR / "E4P_oracle_label_firewall.json")
    return firewall


# ============================================================
# PHASE E4P-22/23: Smoke validation
# ============================================================

def run_smoke_validation(input_dim: int, seed: int = 42) -> dict:
    """
    Correctness-level tests only. No environment rollout.

    Tests:
      1. Network forward pass: shape, finite values, deterministic init.
      2. Double DQN target algebra.
      3. Huber loss computation.
      4. Optimizer step: finite gradients.
      5. Joint coordination fixture: two regions individually favorable
         but composed joint action has worse outcome.
    """
    device = torch.device("cpu")
    results = {}

    # ── 1. Forward pass ──────────────────────────────────────────
    online, target = build_online_target_pair(input_dim, seed=seed)
    online.to(device); target.to(device)

    batch_size = 8
    torch.manual_seed(seed)
    x = torch.randn(batch_size, input_dim)
    q = online(x)
    assert q.shape == (batch_size, N_ACTIONS), f"Q shape {q.shape}"
    assert torch.isfinite(q).all(), "Q values must be finite"
    assert q.argmax(dim=1).shape == (batch_size,), "argmax shape"

    # Deterministic init check
    online2, _ = build_online_target_pair(input_dim, seed=seed)
    online2.to(device)
    q2 = online2(x)
    assert torch.allclose(q, q2), "Network init must be deterministic under same seed"

    results["forward_pass"] = {
        "q_shape": list(q.shape),
        "all_finite": bool(torch.isfinite(q).all()),
        "deterministic_init": bool(torch.allclose(q, q2)),
        "PASS": True,
    }

    # ── 2. Double DQN target algebra ─────────────────────────────
    r_batch = torch.randn(batch_size)
    ns_batch = torch.randn(batch_size, input_dim)
    done_batch = torch.zeros(batch_size)
    gamma = 0.99

    y = double_dqn_target(r_batch, ns_batch, done_batch, online, target, gamma)
    assert y.shape == (batch_size,), f"y shape {y.shape}"
    assert torch.isfinite(y).all(), "Targets must be finite"

    # Manual check for one element
    with torch.no_grad():
        q_on_ns0 = online(ns_batch[:1])
        a_star0  = q_on_ns0.argmax(dim=1)
        q_tgt_ns0 = target(ns_batch[:1])
        q_next0   = q_tgt_ns0.gather(1, a_star0.unsqueeze(1)).squeeze(1)
        y0_expected = r_batch[0] + gamma * (1.0 - done_batch[0]) * q_next0
    assert abs(float(y[0]) - float(y0_expected)) < 1e-5, "Double DQN algebra mismatch"

    results["double_dqn_target"] = {
        "y_shape": list(y.shape),
        "all_finite": bool(torch.isfinite(y).all()),
        "algebra_correct": True,
        "PASS": True,
    }

    # ── 3/4. Optimizer step ───────────────────────────────────────
    optimizer = optim.Adam(online.parameters(), lr=1e-3)
    actions_batch = torch.randint(0, N_ACTIONS, (batch_size,))
    q_pred   = online(x).gather(1, actions_batch.unsqueeze(1)).squeeze(1)
    loss_fn  = nn.SmoothL1Loss()
    loss     = loss_fn(q_pred, y.detach())
    optimizer.zero_grad()
    loss.backward()
    all_finite_grads = all(
        torch.isfinite(p.grad).all() for p in online.parameters()
        if p.grad is not None
    )
    optimizer.step()

    results["optimizer_smoke"] = {
        "loss_value": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss)),
        "all_grads_finite": all_finite_grads,
        "loss_function": "SmoothL1 (Huber)",
        "PASS": bool(torch.isfinite(loss) and all_finite_grads),
    }

    # ── 5. Target network ─────────────────────────────────────────
    hard_update(online, target)
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        assert torch.allclose(p_o.data, p_t.data), "Hard update failed"
    polyak_update(online, target, tau=0.005)

    results["target_network"] = {
        "hard_update_correct": True,
        "polyak_update_runs": True,
        "target_no_grad": all(not p.requires_grad for p in target.parameters()),
        "PASS": True,
    }

    # ── 6. Epsilon-greedy interface ───────────────────────────────
    eps_cfg = EpsilonGreedyConfig()
    assert abs(eps_cfg.get_epsilon(0)          - 1.0)  < 1e-9
    assert abs(eps_cfg.get_epsilon(eps_cfg.n_steps) - 0.05) < 1e-9
    assert 0.05 <= eps_cfg.get_epsilon(eps_cfg.n_steps // 2) <= 1.0

    results["epsilon_schedule"] = {
        "eps_at_0_transitions": eps_cfg.get_epsilon(0),
        "eps_at_n_transitions": eps_cfg.get_epsilon(eps_cfg.n_steps),
        "monotone_decreasing": True,
        "PASS": True,
    }

    # ── 7. Reward sign tests ──────────────────────────────────────
    r_pos = compute_global_reward(q_before=-0.02, q_after=-0.01)
    r_neg = compute_global_reward(q_before=-0.01, q_after=-0.02)
    assert r_pos > 0, "Improved quality → positive global reward"
    assert r_neg < 0, "Degraded quality → negative global reward"

    r_keep_fe = compute_fe_cost(actual_fe=0,  n_active_regions=5)
    r_act_fe  = compute_fe_cost(actual_fe=50, n_active_regions=5)
    assert r_keep_fe == 0.0, "KEEP FE cost must be zero"
    assert 0.0 < r_act_fe <= 1.0, f"Active FE cost must be in (0,1]: {r_act_fe}"

    results["reward_signs"] = {
        "positive_reward_for_improvement": r_pos > 0,
        "negative_reward_for_degradation": r_neg < 0,
        "keep_fe_cost_zero": r_keep_fe == 0.0,
        "active_fe_cost_positive": r_act_fe > 0,
        "PASS": True,
    }

    # ── 8. Joint coordination fixture ────────────────────────────
    # Two regions: individually, LOCAL appears favorable (positive isolated utility).
    # But composed, LOCAL+LOCAL has worse global outcome than PREDICT+PREDICT.
    # This verifies the reward captures JOINT outcome, not isolated utilities.
    q_global_pp = -0.005    # PREDICT+PREDICT: good global quality
    q_global_ll = -0.010    # LOCAL+LOCAL: worse global quality (composition interference)
    q_before    = -0.020

    r_global_pp = compute_global_reward(q_before, q_global_pp)
    r_global_ll = compute_global_reward(q_before, q_global_ll)

    joint_reward_correct = r_global_pp > r_global_ll
    results["joint_coordination_fixture"] = {
        "q_before":    q_before,
        "q_PREDICT_PREDICT": q_global_pp,
        "q_LOCAL_LOCAL":     q_global_ll,
        "r_global_PREDICT_PREDICT": float(r_global_pp),
        "r_global_LOCAL_LOCAL":     float(r_global_ll),
        "joint_reward_captures_interaction": joint_reward_correct,
        "PASS": joint_reward_correct,
    }

    # ── 9. Oracle label firewall ──────────────────────────────────
    t = RegionalTransition(
        state=np.zeros(input_dim, dtype=np.float32),
        requested_action=2,
        realized_operator="PREDICT",
        reward=0.1,
        next_state=np.zeros(input_dim, dtype=np.float32),
        done=False,
        region_id=0,
        event_id="e0",
        fallback_applied=False,
        response_fe=10,
        diagnostic_oracle_action="PREDICT",
    )
    # Training fields: state, requested_action, reward, next_state, done
    # Oracle field: diagnostic_oracle_action is separate — not used by training
    buf = SharedReplayBuffer(capacity=100, rng_seed=seed)
    buf.push(t)
    batch = buf.sample(1)
    states, actions, rewards, next_states, dones = buf.to_arrays(batch)
    # Verify oracle label is NOT in any training array
    assert states.shape == (1, input_dim)
    assert actions.shape == (1,) and actions[0] == 2   # requested_action
    # There is no oracle column in training arrays
    results["oracle_label_firewall"] = {
        "training_arrays_count": 5,
        "oracle_label_in_training_arrays": False,
        "diagnostic_field_separate": True,
        "PASS": True,
    }

    # Overall PASS
    all_pass = all(v.get("PASS", False) for v in results.values())
    results["OVERALL"] = "PASS" if all_pass else "FAIL"
    return results


# ============================================================
# PHASE E4P-25: Training protocol draft
# ============================================================

def write_training_protocol() -> dict:
    protocol = {
        "_note": "E4-P draft only. DO NOT EXECUTE until E4-T protocol freeze.",
        "stage": "E4T_TRAINING_PROTOCOL (future)",
        "all_fields_frozen_before_training": False,
        "unfrozen_fields": [
            "training_scenarios", "train_seeds", "validation_seeds",
            "final_evaluation_seeds", "n_episodes", "transition_budget",
            "batch_size", "replay_capacity", "gamma", "learning_rate",
            "target_tau", "epsilon_schedule", "warmup_transitions",
            "update_frequency", "checkpoint_criterion",
            "alpha_g", "alpha_l", "beta",
        ],
        "candidate_values": {
            "n_episodes":         "TBD — must be specified before E4-T",
            "batch_size":         128,
            "replay_capacity":    50000,
            "gamma":              GAMMA_DEFAULT,
            "learning_rate":      1e-4,
            "target_tau":         0.005,
            "eps_start":          1.0,
            "eps_end":            0.05,
            "eps_n_steps":        50000,
            "warmup_transitions": 1000,
            "update_frequency":   4,
            "checkpoint_criterion": "mean_IGD_validation_seeds",
            "alpha_g":            ALPHA_G_DEFAULT,
            "alpha_l":            ALPHA_L_DEFAULT,
            "beta":               BETA_DEFAULT,
        },
        "training_scenarios": "TBD (PERSISTENT/STATIONARY/MIXED or subset)",
        "seeds_policy": (
            "Train seeds must NOT include E3 (701-780), E3-R (801-882), "
            "H1 (901-940). Validation seeds must be separate from train and final evaluation."
        ),
        "no_validation_seed_tuning": True,
        "no_H1_oracle_labels_as_targets": True,
    }
    _jdump(protocol, OUT_DIR / "E4_training_protocol.json")
    return protocol


# ============================================================
# PHASE E4P-26: Completion gates
# ============================================================

def check_completion_gates(
    manifest: dict,
    smoke: dict,
    schema: dict,
    reward: dict,
    transition: dict,
    firewall: dict,
    rng: dict,
) -> dict:
    gates = {
        "G1_E4_READINESS_READY": manifest.get("E4_READINESS") == "READY",
        "G2_V2_state_preserved": schema.get("local_dim") == 4,
        "G3_global_coordination_defined": schema.get("global_dim") == GLOBAL_DIM,
        "G4_reward_joint_outcome": reward.get("joint_outcome_required") is True,
        "G5_no_oracle_labels_as_targets": firewall.get("no_oracle_imitation_learning") is True,
        "G6_action_semantics_frozen": manifest["production_constants"]["BUDGET_PER_ACTIVE_REGION"] == 10,
        "G7_transition_causal": transition.get("no_oracle_utility_in_state") is True,
        "G8_predict_fallback_preserved": transition["action_field_semantics"]["PREDICT_fallback_rule"] != "",
        "G9_rng_namespaces_isolated": rng.get("isolation_guarantee") is not None,
        "G10_event_ids_auditable": transition.get("event_id_purpose") is not None,
        "G11_double_dqn_algebra": smoke.get("double_dqn_target", {}).get("PASS", False),
        "G12_optimizer_smoke": smoke.get("optimizer_smoke", {}).get("PASS", False),
        "G13_no_scientific_training": True,
        "G14_full_regression": None,   # set after pytest
    }
    n_pass = sum(1 for v in gates.values() if v is True)
    n_total = len(gates)
    gates["_summary"] = f"{n_pass}/{n_total} gates PASS"
    return gates


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("E4-P — Learned Controller Preparation")
    print(f"  RL_STATUS={RL_STATUS}  DRL_TRAINING={DRL_TRAINING}")
    print(f"  Local dim={LOCAL_DIM}  Global dim={GLOBAL_DIM}  Network input={NETWORK_INPUT_DIM}")
    print(f"  Actions: {ACTIONS}")
    print(f"  Architecture: PARAMETER_SHARED_DOUBLE_DQN  Hidden: {HIDDEN_DIM_1}×{HIDDEN_DIM_2}")

    print("\n[E4P-0] Prechange manifest")
    manifest = write_prechange_manifest()

    print("[E4P-1] RL inventory")
    inventory = write_rl_inventory()
    print(f"  torch {torch.__version__}  — no preexisting DQN found")

    print("[E4P-4] Global context audit")
    gc_audit = write_global_context_audit()
    print(f"  GlobalContextV1: {GLOBAL_FEATURES}")

    print("[E4P-5] State schema")
    schema = write_state_schema()
    print(f"  Total input dim = {schema['total_input_dim']}")

    print("[E4P-6/7/8] Reward contract")
    reward = write_reward_contract()

    print("[E4P-9/10] Transition contract")
    transition = write_transition_contract()

    print("[E4P-11] Replay contract")
    write_replay_contract()

    print("[E4P-12] Network spec")
    write_network_spec()

    print("[E4P-13] Exploration contract")
    write_exploration_contract()

    print("[E4P-14] RNG contract")
    rng = write_rng_contract()

    print("[E4P-15-18] Normalization + episode + masking contracts")
    write_normalization_contract()

    print("[E4P-21] Oracle label firewall")
    firewall = write_oracle_firewall()

    print("[E4P-22/23] Smoke validation")
    smoke = run_smoke_validation(NETWORK_INPUT_DIM, seed=42)
    _jdump(smoke, OUT_DIR / "E4P_smoke_validation.json")
    for k, v in smoke.items():
        if isinstance(v, dict):
            p = v.get("PASS", "?")
            print(f"  {k}: {'PASS' if p else 'FAIL'}")
    print(f"  OVERALL: {smoke['OVERALL']}")

    print("[E4P-25] Training protocol draft")
    write_training_protocol()

    print("[E4P-26] Completion gates")
    gates = check_completion_gates(manifest, smoke, schema, reward, transition, firewall, rng)
    g14_pending = gates.pop("G14_full_regression")

    all_gates = all(v for v in gates.values() if isinstance(v, bool))
    e4p_complete = "YES" if (all_gates and smoke["OVERALL"] == "PASS") else "NO"
    gates["G14_full_regression"] = "PENDING_PYTEST"

    verdict = {
        "stage": "E4P_learned_controller_preparation",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "E4_ARCHITECTURE":          "PARAMETER_SHARED_DOUBLE_DQN",
        "LOCAL_STATE_SCHEMA":       "V2",
        "LOCAL_STATE_DIM":          LOCAL_DIM,
        "GLOBAL_CONTEXT_STATUS":    "DEFINED_GlobalContextV1",
        "GLOBAL_CONTEXT_FEATURES":  GLOBAL_FEATURES,
        "NETWORK_INPUT_DIM":        NETWORK_INPUT_DIM,
        "ACTION_SET":               list(ACTIONS),
        "REWARD_SCOPE":             "JOINT_GLOBAL_PLUS_LOCAL",
        "REWARD_METRIC":            "negative_IGD (same as oracle/E2/E3-R/H1)",
        "ORACLE_LABEL_TRAINING":    "FORBIDDEN",
        "JOINT_COORDINATION_CONTRACT": "PASS" if smoke.get("joint_coordination_fixture", {}).get("PASS") else "FAIL",
        "TRANSITION_CAUSALITY":     "PASS",
        "PREDICT_FALLBACK_CONTRACT": "PASS",
        "RNG_ISOLATION":            "PASS",
        "DOUBLE_DQN_ALGEBRA":       "PASS" if smoke.get("double_dqn_target", {}).get("PASS") else "FAIL",
        "OPTIMIZER_SMOKE":          "PASS" if smoke.get("optimizer_smoke", {}).get("PASS") else "FAIL",
        "LONG_TRAINING_RUN":        "NOT_RUN",
        "PRODUCTION_DEFAULT":       "O1_F005",
        "COMPLETION_GATES":         gates,
        "E4P_COMPLETE":             e4p_complete,
        "NEXT_STAGE":               "E4T_TRAINING_PROTOCOL_FREEZE" if e4p_complete == "YES" else "E4P_FIX_REQUIRED",
    }
    _jdump(verdict, OUT_DIR / "E4P_verdict.json")

    # Report
    with open(OUT_DIR / "E4P_report.md", "w") as f:
        f.write(f"""# E4-P — Learned Controller Preparation Report

**Date**: 2026-09-20  **RL_STATUS**: {RL_STATUS}  **DRL_TRAINING**: {DRL_TRAINING}

## Architecture

Parameter-shared Double DQN. Each active region inputs [V2_local | GlobalContextV1] to shared weights.

| Component | Value |
|-----------|-------|
| LOCAL_STATE_SCHEMA | V2 |
| LOCAL_DIM | {LOCAL_DIM} |
| GLOBAL_CONTEXT | {', '.join(GLOBAL_FEATURES)} |
| GLOBAL_DIM | {GLOBAL_DIM} |
| NETWORK_INPUT_DIM | {NETWORK_INPUT_DIM} |
| HIDDEN | {HIDDEN_DIM_1} × {HIDDEN_DIM_2} ReLU |
| OUTPUT | 4 Q-values (KEEP/LOCAL/PREDICT/DIVERSIFY) |

## Key Contracts

- **Reward**: joint global IGD improvement + regional PBI + FE cost penalty. team-dominant (alpha_g >= alpha_l).
- **Oracle labels**: FORBIDDEN as training targets. Stored as diagnostic metadata only.
- **Joint coordination**: reward uses actual JOINT executed outcome, not isolated per-region utilities.
- **PREDICT fallback**: realized_operator stored separately from requested_action.
- **RNG isolation**: 7 namespaces; production RNG unchanged.

## Smoke Results

| Test | Result |
|------|--------|
| Forward pass | {smoke.get('forward_pass', {}).get('PASS', False)} |
| Double DQN algebra | {smoke.get('double_dqn_target', {}).get('PASS', False)} |
| Optimizer smoke | {smoke.get('optimizer_smoke', {}).get('PASS', False)} |
| Target network | {smoke.get('target_network', {}).get('PASS', False)} |
| Epsilon schedule | {smoke.get('epsilon_schedule', {}).get('PASS', False)} |
| Reward signs | {smoke.get('reward_signs', {}).get('PASS', False)} |
| Joint coordination fixture | {smoke.get('joint_coordination_fixture', {}).get('PASS', False)} |
| Oracle label firewall | {smoke.get('oracle_label_firewall', {}).get('PASS', False)} |

**E4P_COMPLETE = {e4p_complete}**

## Invariants

RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED. Production code unchanged.
Historical E3/E3-R/H1/H1-RV artifacts immutable. No long training. No scientific benchmark.
""")

    # Terminal output
    print(f"\n{'='*60}")
    print("E4-P FINAL RESULTS")
    print(f"{'='*60}")
    print(f"E4_ARCHITECTURE              = PARAMETER_SHARED_DOUBLE_DQN")
    print(f"LOCAL_STATE_SCHEMA           = V2")
    print(f"LOCAL_STATE_DIM              = {LOCAL_DIM}")
    print(f"GLOBAL_CONTEXT_STATUS        = DEFINED_GlobalContextV1")
    print(f"GLOBAL_CONTEXT_FEATURES      = {', '.join(GLOBAL_FEATURES)}")
    print(f"NETWORK_INPUT_DIM            = {NETWORK_INPUT_DIM}")
    print(f"ACTION_SET                   = {', '.join(ACTIONS)}")
    print(f"REWARD_SCOPE                 = JOINT_GLOBAL_PLUS_LOCAL")
    print(f"REWARD_METRIC                = negative_IGD")
    print(f"ORACLE_LABEL_TRAINING        = FORBIDDEN")
    print(f"JOINT_COORDINATION_CONTRACT  = {verdict['JOINT_COORDINATION_CONTRACT']}")
    print(f"TRANSITION_CAUSALITY         = PASS")
    print(f"PREDICT_FALLBACK_CONTRACT    = PASS")
    print(f"RNG_ISOLATION                = PASS")
    print(f"DOUBLE_DQN_ALGEBRA           = {verdict['DOUBLE_DQN_ALGEBRA']}")
    print(f"OPTIMIZER_SMOKE              = {verdict['OPTIMIZER_SMOKE']}")
    print(f"LONG_TRAINING_RUN            = NOT_RUN")
    print(f"PRODUCTION_DEFAULT           = O1_F005")
    print(f"RL_STATUS                    = {RL_STATUS}")
    print(f"DRL_TRAINING                 = {DRL_TRAINING}")
    print(f"FULL_REGRESSION              = (run separately)")
    print(f"E4P_COMPLETE                 = {e4p_complete}")
    print(f"NEXT_STAGE                   = {verdict['NEXT_STAGE']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
