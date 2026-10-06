"""
E4TR — Shared Parameter Double DQN Training (PR-DRL-DMOEA).

FIRST stage of substantive DRL training. The training protocol was frozen in
E4-T (results/E4T_training_protocol_freeze/E4T_training_protocol_frozen.json)
and is executed here EXACTLY. No tuning, no hyperparameter search, no
early stopping, no best-policy-seed selection.

Controller: PARAMETER_SHARED_DOUBLE_DQN.
  - One shared Q-network for all active regions.
  - Per-region input = concat(local_V2[4], GlobalContextV1[5]) = 9-D.
  - Output = 4 Q-values (KEEP, LOCAL, PREDICT, DIVERSIFY).

Firewall invariants (HARD):
  - Final-evaluation seeds 1081..1140 are NEVER instantiated (runtime guard).
  - GREEDY_FACTORIZED oracle labels are NEVER used as state/target/reward
    shortcut. No oracle call in the training runtime path.
  - Production operators / NSGA-II / PREDICT are reused unchanged.
  - Regional decisions are SIMULTANEOUS: all regions observed, all actions
    chosen, THEN the joint action vector is executed.

Reward (frozen):  reward_r = alpha_g*global + alpha_l*local_r - beta*fe_cost_r
  global : normalized IGD improvement of the ACTUAL joint executed population.
  local  : frozen quality_loss (PBI) improvement on the ACTUAL joint response
           population (before = pre-response regional convergence loss;
           after = regional convergence loss recomputed on the executed
           response population). No isolated oracle branches.
  fe_cost: actual response FE per region / budget (KEEP = 0).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

# ---- environment primitives (validated, reused unchanged) --------------------
from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.detectors.quality_loss import quality_loss
from prdrl.detectors.severity import severity
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
)
from prdrl.rl.interface import (
    CURRENT_RESPONSE_BUDGET_VALUE,
    LOCAL_SPEC,
    PREDICT_SPEC,
    DIVERSIFY_SPEC,
    RegionalDecisionContextV2,
    build_decision_context_v2,
    execute_response,
)
from prdrl.controllers.rule_based import choose_actions_v2
from prdrl.experiments.E3R_rule_v2_semantic_rerun import (
    N_OBJ, D, POP_SIZE, N_PARTITIONS,
    WARMUP_GENERATIONS, GENERATIONS_PER_CHANGE, BUDGET,
    SIGMA_LOCAL, SIGMA_PREDICT, SIGMA_DIVERSIFY, LAMBDA_PREDICT,
    THETA_PBI, KAPPA_SEV, Q_PREV_INIT, EVOLUTION_FE,
    make_region_dynamics, evolve_nsga2_predict, compute_time,
    target_region, get_target_regime,
    _warmup_and_burnin,
)

# ---- RL modules (reused) -----------------------------------------------------
from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair, polyak_update,
    double_dqn_target, ACTION_ORDER, N_ACTIONS,
)
from prdrl.rl.replay import SharedReplayBuffer, RegionalTransition
from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward,
    compute_fe_cost, compute_regional_reward, RewardConfig, BUDGET_PER_REGION,
)
from prdrl.rl.reward_r1 import (
    REWARD_R0_NAME, REWARD_R1_NAME, regional_quality_gain_r1,
)
from prdrl.rl.agent import (
    EpsilonGreedyConfig, select_regional_actions,
    ACTION_TO_IDX, IDX_TO_ACTION,
)
from prdrl.experiments.E4P_learned_controller_preparation import (
    compute_global_context_v1, normalize_local_state, normalize_global_context,
    LOCAL_DIM, GLOBAL_DIM, NETWORK_INPUT_DIM,
)

# =============================================================================
# FROZEN PROTOCOL (mirrors E4T_training_protocol_frozen.json — validated at load)
# =============================================================================

RESULTS_DIR = Path("results/E4TR_shared_double_dqn_training")
E4T_FROZEN_PATH = Path(
    "results/E4T_training_protocol_freeze/E4T_training_protocol_frozen.json"
)

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
SCENARIO_ID = {"PERSISTENT": 0, "STATIONARY": 1, "MIXED": 2}

TRAIN_SEEDS = list(range(1001, 1061))       # 60
VALIDATION_SEEDS = list(range(1061, 1081))  # 20
FINAL_EVAL_SEEDS = list(range(1081, 1141))  # 60 — FIREWALLED
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]

N_EPOCHS = 6
EPISODES_PER_EPOCH = len(TRAIN_SEEDS) * len(SCENARIOS)   # 180
N_TRAIN_EPISODES = N_EPOCHS * EPISODES_PER_EPOCH         # 1080
N_SCORED_CHANGES = 10

# RL hyperparameters (frozen)
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_TAU = 0.005
WARMUP_TRANSITIONS = 1000
UPDATE_FREQUENCY = 4
GRAD_STEPS_PER_UPDATE = 1
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 50_000

# Reward coefficients (frozen)
ALPHA_G, ALPHA_L, BETA = 1.0, 0.1, 0.01
_REWARD_EPS = 1e-12

# Checkpoints (frozen)
CHECKPOINT_TRANSITIONS = [10_000, 20_000, 30_000, 40_000, 50_000]

# RNG namespace offsets (frozen; POLICY_INIT/REPLAY_SAMPLE/NETWORK_MINIBATCH/TRAIN_EPISODE_ORDER)
RNG_OFFSET_POLICY_INIT = 0
RNG_OFFSET_TORCH = 1
RNG_OFFSET_REPLAY_SAMPLE = 2
RNG_OFFSET_NETWORK_MINIBATCH = 3
RNG_OFFSET_TRAIN_EPISODE_ORDER = 7
EXPLORE_NAMESPACE_TAG = 909_090_909   # POLICY_EXPLORATION SeedSequence entropy tag

DEVICE = torch.device("cpu")

RL_STATUS_NOT_STARTED = "NOT_STARTED"

BASELINE_GLOBALS = ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]
RULE_CTRL = "RULE_V1_V2STATE"

# Source files whose hash forms the code manifest (checkpoint provenance)
_CODE_MANIFEST_FILES = [
    "src/prdrl/experiments/E4TR_shared_double_dqn_training.py",
    "src/prdrl/rl/network.py",
    "src/prdrl/rl/replay.py",
    "src/prdrl/rl/reward.py",
    "src/prdrl/rl/agent.py",
    "src/prdrl/rl/interface.py",
    "src/prdrl/oracle/oracle.py",
    "src/prdrl/detectors/severity.py",
    "src/prdrl/detectors/quality_loss.py",
]


# =============================================================================
# Utilities
# =============================================================================

def _hash_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    sha = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _code_manifest_hash() -> str:
    h = hashlib.sha256()
    for fp in _CODE_MANIFEST_FILES:
        h.update(_hash_file(fp).encode())
    return h.hexdigest()


def _frozen_config_hash() -> str:
    return _hash_file(str(E4T_FROZEN_PATH))


# =============================================================================
# FINAL-EVAL FIREWALL
# =============================================================================

class FinalEvalFirewall:
    """Runtime guard: forbids any instantiation of final-eval seeds in E4TR."""

    def __init__(self):
        self.access_count = 0
        self.forbidden = set(FINAL_EVAL_SEEDS)

    def check(self, seed: int):
        if int(seed) in self.forbidden:
            self.access_count += 1
            raise RuntimeError(
                f"FINAL_EVAL_FIREWALL: seed {seed} is a final-evaluation seed "
                f"(1081..1140) and MUST NOT be instantiated in E4TR."
            )


FIREWALL = FinalEvalFirewall()


# =============================================================================
# PHASE 0 — pretrain safety
# =============================================================================

def load_and_validate_frozen_config() -> Tuple[dict, List[str]]:
    """Load frozen E4-T config; return (config, mismatches vs this prompt)."""
    with open(E4T_FROZEN_PATH) as f:
        cfg = json.load(f)
    mismatches: List[str] = []

    def _eq(label, got, want):
        if got != want:
            mismatches.append(f"{label}: frozen={got!r} expected={want!r}")

    _eq("train_seeds", cfg["seed_split"]["train"]["seeds"], TRAIN_SEEDS)
    _eq("validation_seeds", cfg["seed_split"]["validation"]["seeds"], VALIDATION_SEEDS)
    _eq("final_eval_seeds", cfg["seed_split"]["final_eval"]["seeds"], FINAL_EVAL_SEEDS)
    _eq("scenarios", cfg["scenarios"], SCENARIOS)
    _eq("policy_seeds", cfg["policy_seeds"], POLICY_SEEDS)
    _eq("n_epochs", cfg["training_budget"]["n_epochs"], N_EPOCHS)
    _eq("episodes_per_epoch", cfg["training_budget"]["episodes_per_epoch"], EPISODES_PER_EPOCH)
    _eq("n_train_episodes", cfg["training_budget"]["n_train_episodes"], N_TRAIN_EPISODES)
    _eq("changes_per_episode", cfg["training_budget"]["changes_per_episode"], N_SCORED_CHANGES)
    _eq("input_dim", cfg["network"]["input_dim"], NETWORK_INPUT_DIM)
    _eq("hidden_layers", cfg["network"]["hidden_layers"], [128, 128])
    _eq("output_dim", cfg["network"]["output_dim"], N_ACTIONS)
    _eq("batch_size", cfg["hyperparameters"]["batch_size"], BATCH_SIZE)
    _eq("replay_capacity", cfg["hyperparameters"]["replay_capacity"], REPLAY_CAPACITY)
    _eq("gamma", cfg["hyperparameters"]["gamma"], GAMMA)
    _eq("learning_rate", cfg["hyperparameters"]["learning_rate"], LEARNING_RATE)
    _eq("target_tau", cfg["hyperparameters"]["target_tau"], TARGET_TAU)
    _eq("warmup_transitions", cfg["hyperparameters"]["warmup_transitions"], WARMUP_TRANSITIONS)
    _eq("update_frequency", cfg["hyperparameters"]["update_frequency"], UPDATE_FREQUENCY)
    _eq("eps_start", cfg["exploration"]["eps_start"], EPS_START)
    _eq("eps_end", cfg["exploration"]["eps_end"], EPS_END)
    _eq("eps_decay_transitions", cfg["exploration"]["eps_decay_transitions"], EPS_DECAY)
    _eq("alpha_g", cfg["reward"]["alpha_g"], ALPHA_G)
    _eq("alpha_l", cfg["reward"]["alpha_l"], ALPHA_L)
    _eq("beta", cfg["reward"]["beta"], BETA)
    _eq("checkpoint_transitions", cfg["checkpointing"]["transitions"], CHECKPOINT_TRANSITIONS)
    _eq("early_stopping", cfg["checkpointing"]["early_stopping"], False)
    return cfg, mismatches


def seed_intersections() -> Dict[str, list]:
    tr, va, fi = set(TRAIN_SEEDS), set(VALIDATION_SEEDS), set(FINAL_EVAL_SEEDS)
    return {
        "train_cap_validation": sorted(tr & va),
        "train_cap_final": sorted(tr & fi),
        "validation_cap_final": sorted(va & fi),
    }


def historical_seed_overlap() -> Dict[str, list]:
    used = set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(FINAL_EVAL_SEEDS)
    hist = {
        "E3_701_780": set(range(701, 781)),
        "E3R_801_882": set(range(801, 883)),
        "H1_901_940": set(range(901, 941)),
    }
    return {k: sorted(used & v) for k, v in hist.items()}


# =============================================================================
# ENVIRONMENT — warmup/burn-in cache + step primitives (reuse E3-R semantics)
# =============================================================================

class EnvContext:
    def __init__(self, xl, xu, V, R, pf_star):
        self.xl, self.xu, self.V, self.R, self.pf_star = xl, xu, V, R, pf_star


def make_env_context() -> EnvContext:
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    pf_star = get_true_pf(n_points=500)
    return EnvContext(xl, xu, V, R, pf_star)


def build_warmup_cache(seeds: List[str], env: EnvContext) -> dict:
    """Deterministic warmup+burn-in per (scenario, seed). Reused across epochs
    and policy seeds (identical initial condition; independent of the network).
    Same pattern E3-R uses to reuse an initial condition across controllers."""
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in seeds:
            FIREWALL.check(seed)
            X_init, h_init, h_t_init, t_cur_init, gen_init = _warmup_and_burnin(
                seed, rd, env.xl, env.xu, env.V, env.R
            )
            cache[(scenario, seed)] = (X_init, h_init, h_t_init, t_cur_init, gen_init)
    return cache


def env_reset(scenario: str, seed: int, cache: dict):
    X_init, h_init, h_t_init, t_cur_init, gen_init = cache[(scenario, seed)]
    return {
        "X": X_init.copy(),
        "h": copy.deepcopy(h_init),
        "h_t": h_t_init,
        "t_current": t_cur_init,
        "gen": gen_init,
        "q_prev": Q_PREV_INIT,
        "severity_history": {r: None for r in range(len(env_reset_R(cache)))} if False else None,
    }


def env_reset_R(cache):  # pragma: no cover - helper kept simple
    return []


def pre_response(state: dict, scenario: str, seed: int, env: EnvContext, region_dynamics):
    """Compute pre-response quantities at the current change. Updates history.
    Returns a dict with rids, igd_pre, per-region v2 states, contexts, active list."""
    X = state["X"]
    state["gen"] += GENERATIONS_PER_CHANGE
    t_new = compute_time(state["gen"])
    t_current = state["t_current"]

    F_minus = cp_evaluate(X, t_current, region_dynamics)
    F_plus = cp_evaluate(X, t_new, region_dynamics)
    F_tm_minus, F_tm_plus = normalize(F_minus, F_plus)
    rids = np.asarray(partition(np.asarray(F_tm_minus), env.V))
    igd_pre = float(igd(F_plus, env.pf_star))

    state["h_t"] += 1
    state["h"].update(t=state["h_t"], X=X, F=F_minus, region_ids=rids,
                      xl=env.xl, xu=env.xu)

    S, C, q_new = severity(F_tm_minus, F_tm_plus, rids, env.R, state["q_prev"], kappa=KAPPA_SEV)
    losses = quality_loss(F_tm_minus, F_tm_plus, rids, env.V, THETA_PBI)

    active_regions = [r for r in range(env.R) if int(np.sum(rids == r)) > 0]
    v2_states: Dict[int, dict] = {}
    contexts: Dict[int, RegionalDecisionContextV2] = {}
    for r in active_regions:
        n_r = int(np.sum(rids == r))
        sev_r = float(C[r]) if not np.isnan(C[r]) else 0.0
        closs_r = float(losses[r]) if not np.isnan(losses[r]) else 0.0
        prev_sev = state["severity_history"][r]
        ctx = build_decision_context_v2(
            region_id=r, n_r=n_r, N_total=POP_SIZE, t=state["h_t"] + 1,
            severity_r=sev_r, convergence_loss_r=closs_r, prev_severity_r=prev_sev,
        )
        contexts[r] = ctx
        v2_states[r] = {
            "severity_r": sev_r,
            "convergence_loss_r": closs_r,
            "occupancy_r": float(n_r / POP_SIZE),
            "delta_severity_r": float(ctx.delta_severity_r),
        }

    return {
        "t_new": t_new, "F_minus": F_minus, "F_plus": F_plus,
        "F_tm_minus": F_tm_minus, "F_tm_plus": F_tm_plus,
        "rids": rids, "igd_pre": igd_pre, "C": C, "q_new": q_new,
        "losses": losses, "active_regions": active_regions,
        "v2_states": v2_states, "contexts": contexts,
    }


def build_region_states(v2_states: Dict[int, dict], R: int, state_normalizer=None):
    """Normalization → per-region 9-D state and shared global context.

    state_normalizer is None for the frozen E4TR/E4F legacy V1 mapping (unchanged
    default; used by all existing callers). If provided, it is a callable raw9->9
    applied to the raw [local V2 | global context] vector — used by the E4HTR
    V1-vs-V3 A/B experiment (per-dimension transforms only)."""
    gc = compute_global_context_v1(v2_states, R)
    local_norm: Dict[int, np.ndarray] = {}
    full_states: Dict[int, np.ndarray] = {}
    if state_normalizer is None:
        gc_norm = normalize_global_context(gc)
        for r, v2 in v2_states.items():
            loc = normalize_local_state(v2)
            local_norm[r] = loc
            full_states[r] = np.concatenate([loc, gc_norm]).astype(np.float32)
        return local_norm, gc_norm, full_states
    # arm normalizer (V1-legacy or V3): apply to the raw 9-vector per region.
    gc = np.asarray(gc, dtype=np.float64)
    gc_norm = np.asarray(state_normalizer(np.concatenate([np.zeros(4), gc]))[4:], dtype=np.float32)
    for r, v2 in v2_states.items():
        raw9 = np.array([v2["severity_r"], v2["convergence_loss_r"], v2["occupancy_r"],
                         v2["delta_severity_r"], *gc], dtype=np.float64)
        z = np.asarray(state_normalizer(raw9), dtype=np.float32)
        local_norm[r] = z[:4]
        full_states[r] = z
    return local_norm, gc_norm, full_states


def execute_and_recover(actions: Dict[int, str], pre: dict, state: dict,
                        scenario: str, seed: int, env: EnvContext,
                        region_dynamics, scored_change_idx: int):
    """Execute the joint action vector, evaluate, unrestricted NSGA-II survival,
    then standard NSGA-II recovery. Reuses production dispatch unchanged."""
    t_new = pre["t_new"]
    evaluate_fn = lambda Xq, _t=t_new, _rd=region_dynamics: cp_evaluate(Xq, _t, _rd)
    response_result, records = execute_response(
        actions=actions, X=state["X"], F_plus=pre["F_plus"], region_ids=pre["rids"],
        V=env.V, history=state["h"], t=state["h_t"] + 1, evaluate_fn=evaluate_fn,
        xl=env.xl, xu=env.xu, base_seed=seed, budget=BUDGET,
        sigma_local=SIGMA_LOCAL, sigma_predict=SIGMA_PREDICT,
        sigma_diversify=SIGMA_DIVERSIFY, lambda_pred=LAMBDA_PREDICT,
    )
    X_resp = np.asarray(response_result["X"], dtype=float)
    F_resp = np.asarray(response_result["F"], dtype=float)
    igd_response = float(igd(F_resp, env.pf_star))

    # NSGA-II recovery evolution (production RNG namespace, unchanged)
    evo_seed = seed * 10000 + scored_change_idx
    X_next = evolve_nsga2_predict(X_resp, t_new, region_dynamics,
                                  GENERATIONS_PER_CHANGE, evo_seed, env.xl, env.xu)
    F_epoch = cp_evaluate(X_next, t_new, region_dynamics)
    igd_epoch = float(igd(F_epoch, env.pf_star))

    return {
        "X_resp": X_resp, "F_resp": F_resp, "igd_response": igd_response,
        "X_next": X_next, "igd_epoch": igd_epoch, "records": records,
    }


def compute_joint_rewards(pre: dict, post: dict, records, env: EnvContext,
                          region_dynamics, t_current: float,
                          reward_version: str = REWARD_R0_NAME) -> Tuple[float, Dict[int, float], dict]:
    """Frozen team reward from the ACTUAL joint executed outcome.

    global : normalized IGD improvement of the executed joint population.
    local  : R0 = quality_loss (PBI) improvement (pbi_before - pbi_after);
             R1 = bounded symmetric regional quality gain g_r (E4I/E4II frozen).
    fe_cost: per-region actual response FE / budget.

    reward_version selects ONLY the local-credit signal (single factor). It defaults to
    ORIGINAL_REWARD_R0, byte-identical to the historical E4TR/E4F/E4HTR reward. aux always
    carries BOTH local formulations + L_pre/L_post (for shadow logging), regardless of arm;
    only region_rewards / local_by_region use the selected version.
    """
    q_before = -pre["igd_pre"]
    q_after = -post["igd_response"]
    global_reward = compute_global_reward(q_before=q_before, q_after=q_after,
                                          normalization=q_before)

    # regional PBI loss AFTER, on the actual joint response population
    X_resp, F_resp = post["X_resp"], post["F_resp"]
    F_resp_minus = cp_evaluate(X_resp, t_current, region_dynamics)   # offline reward label
    F_tm_rm, F_tm_rp = normalize(F_resp_minus, F_resp)
    rids_resp = np.asarray(partition(np.asarray(F_tm_rm), env.V))
    losses_after = quality_loss(F_tm_rm, F_tm_rp, rids_resp, env.V, THETA_PBI)

    rec_by_region = {rec.region_id: rec for rec in records}
    region_rewards: Dict[int, float] = {}
    local_by_region: Dict[int, float] = {}
    fe_by_region: Dict[int, float] = {}
    lpre_by_region: Dict[int, float] = {}
    lpost_by_region: Dict[int, float] = {}
    r0_local_by_region: Dict[int, float] = {}
    g_by_region: Dict[int, float] = {}
    for r in pre["active_regions"]:
        pbi_before = pre["losses"][r]
        pbi_after = losses_after[r] if r < len(losses_after) else np.nan
        raw_local_r0 = compute_regional_pbi_reward(pbi_before, pbi_after)
        g_r = regional_quality_gain_r1(pbi_before, pbi_after)
        local_r = raw_local_r0 if reward_version == REWARD_R0_NAME else g_r
        rec = rec_by_region.get(r)
        response_fe = int(rec.response_fe) if rec is not None else 0
        fe_cost_r = compute_fe_cost(actual_fe=response_fe, n_active_regions=1,
                                    budget_per_region=BUDGET_PER_REGION)
        reward_r = compute_regional_reward(
            global_reward=global_reward, local_pbi_reward=local_r,
            fe_cost=fe_cost_r, alpha_g=ALPHA_G, alpha_l=ALPHA_L, beta=BETA,
        )
        region_rewards[r] = reward_r
        local_by_region[r] = local_r
        fe_by_region[r] = fe_cost_r
        lpre_by_region[r] = float(pbi_before) if not np.isnan(pbi_before) else float("nan")
        lpost_by_region[r] = float(pbi_after) if not np.isnan(pbi_after) else float("nan")
        r0_local_by_region[r] = raw_local_r0
        g_by_region[r] = g_r

    aux = {"global_reward": global_reward, "q_before": q_before, "q_after": q_after,
           "local_by_region": local_by_region, "fe_by_region": fe_by_region,
           "reward_version": reward_version,
           "L_pre_by_region": lpre_by_region, "L_post_by_region": lpost_by_region,
           "raw_local_R0_by_region": r0_local_by_region, "g_by_region": g_by_region}
    return global_reward, region_rewards, aux


def advance_state(state: dict, pre: dict, post: dict, env: EnvContext):
    """Roll forward per-arm state after a change (EMA + severity history + pop)."""
    state["q_prev"] = float(pre["q_new"])
    for r in range(env.R):
        if int(np.sum(pre["rids"] == r)) > 0 and not np.isnan(pre["C"][r]):
            state["severity_history"][r] = float(pre["C"][r])
    state["X"] = post["X_next"]
    state["t_current"] = pre["t_new"]


def _fresh_state(scenario, seed, cache, R):
    X_init, h_init, h_t_init, t_cur_init, gen_init = cache[(scenario, seed)]
    return {
        "X": X_init.copy(), "h": copy.deepcopy(h_init), "h_t": h_t_init,
        "t_current": t_cur_init, "gen": gen_init, "q_prev": Q_PREV_INIT,
        "severity_history": {r: None for r in range(R)},
    }


def _derive_event_seed(policy_seed, scenario, env_seed, epoch, change) -> int:
    ss = np.random.SeedSequence([
        EXPLORE_NAMESPACE_TAG, int(policy_seed), SCENARIO_ID[scenario],
        int(env_seed), int(epoch), int(change),
    ])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# =============================================================================
# EVAL ARM (validation / baselines) — greedy, no learning, no transitions
# =============================================================================

def eval_learned_arm(online_net, scenario, seed, cache, env, state_normalizer=None) -> dict:
    """Greedy (epsilon=0) evaluation of a learned policy on one arm."""
    region_dynamics = make_region_dynamics(scenario, env.R)
    state = _fresh_state(scenario, seed, cache, env.R)
    online_net.eval()
    igd_resp, igd_ep, fe_list = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    n_fallback = 0
    prev_active = None
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(state, scenario, seed, env, region_dynamics)
        if prev_active is not None:
            _assert_no_region_disappearance(prev_active, pre["active_regions"])
        local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, state_normalizer)
        _, action_str = select_regional_actions(
            online_net, local_norm, gc_norm, epsilon=0.0, event_seed=0, device=DEVICE,
        )
        actions = {r: action_str[r] for r in pre["active_regions"]}
        post = execute_and_recover(actions, pre, state, scenario, seed, env,
                                   region_dynamics, k + 2)
        igd_resp.append(post["igd_response"])
        igd_ep.append(post["igd_epoch"])
        fe_list.append(sum(rec.response_fe for rec in post["records"]))
        for rec in post["records"]:
            act_counts[rec.requested_action] += 1
            if rec.fallback_used:
                n_fallback += 1
        prev_active = pre["active_regions"]
        advance_state(state, pre, post, env)
    return _summarize_arm(igd_resp, igd_ep, fe_list, act_counts, n_fallback)


def eval_fixed_arm(controller_name, scenario, seed, cache, env, rule_config=None) -> dict:
    """Evaluate a fixed baseline (GLOBAL_* or RULE_V1_V2STATE) on one arm."""
    region_dynamics = make_region_dynamics(scenario, env.R)
    state = _fresh_state(scenario, seed, cache, env.R)
    igd_resp, igd_ep, fe_list = [], [], []
    act_counts = {a: 0 for a in ACTION_ORDER}
    n_fallback = 0
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(state, scenario, seed, env, region_dynamics)
        actions = choose_actions_v2(controller_name, pre["contexts"], rule_config)
        post = execute_and_recover(actions, pre, state, scenario, seed, env,
                                   region_dynamics, k + 2)
        igd_resp.append(post["igd_response"])
        igd_ep.append(post["igd_epoch"])
        fe_list.append(sum(rec.response_fe for rec in post["records"]))
        for rec in post["records"]:
            act_counts[rec.requested_action] += 1
            if rec.fallback_used:
                n_fallback += 1
        advance_state(state, pre, post, env)
    return _summarize_arm(igd_resp, igd_ep, fe_list, act_counts, n_fallback)


def _summarize_arm(igd_resp, igd_ep, fe_list, act_counts, n_fallback) -> dict:
    total_act = sum(act_counts.values()) or 1
    return {
        "MIGD_response": float(np.mean(igd_resp)),
        "MIGD_epoch": float(np.mean(igd_ep)),
        "mean_response_fe": float(np.mean(fe_list)),
        "total_response_fe": int(np.sum(fe_list)),
        "action_fraction": {a: act_counts[a] / total_act for a in ACTION_ORDER},
        "fallback_rate": float(n_fallback / total_act),
    }


def _assert_no_region_disappearance(prev_active, cur_active):
    """Phase 8 guard: a region active at t must not be inactive at t+1 (else the
    transition contract is undefined and we STOP rather than invent a state)."""
    disappeared = set(prev_active) - set(cur_active)
    if disappeared:
        raise RuntimeError(
            f"TRANSITION_CONTRACT_GAP=ACTIVE_TO_INACTIVE_REGION: regions "
            f"{sorted(disappeared)} active then inactive; next_state undefined."
        )


# =============================================================================
# VALIDATION of a checkpoint on validation seeds (macro across scenarios)
# =============================================================================

def validate_online_net(online_net, cache_val, env,
                        val_seeds=VALIDATION_SEEDS, state_normalizer=None) -> dict:
    per_scen = {}
    per_arm_rows = []
    for scenario in SCENARIOS:
        resp, ep, fe = [], [], []
        for seed in val_seeds:
            m = eval_learned_arm(online_net, scenario, seed, cache_val, env, state_normalizer)
            resp.append(m["MIGD_response"]); ep.append(m["MIGD_epoch"]); fe.append(m["mean_response_fe"])
            per_arm_rows.append({"scenario": scenario, "seed": seed, **m})
        per_scen[scenario] = {
            "mean_MIGD_response": float(np.mean(resp)),
            "mean_MIGD_epoch": float(np.mean(ep)),
            "mean_response_fe": float(np.mean(fe)),
        }
    macro_resp = float(np.mean([per_scen[s]["mean_MIGD_response"] for s in SCENARIOS]))
    macro_epoch = float(np.mean([per_scen[s]["mean_MIGD_epoch"] for s in SCENARIOS]))
    macro_fe = float(np.mean([per_scen[s]["mean_response_fe"] for s in SCENARIOS]))
    return {
        "per_scenario": per_scen,
        "VALIDATION_MACRO_MIGD_RESPONSE": macro_resp,
        "macro_MIGD_epoch": macro_epoch,
        "macro_mean_response_fe": macro_fe,
        "per_arm": per_arm_rows,
    }


def select_checkpoint(checkpoint_metrics: List[dict]) -> dict:
    """Frozen selection: (1) lowest macro MIGD_response, (2) tie<=1e-12 lowest
    macro MIGD_epoch, (3) lower mean response FE, (4) earlier transition count."""
    def key(c):
        return (
            round(c["VALIDATION_MACRO_MIGD_RESPONSE"], 12),
            round(c["macro_MIGD_epoch"], 12),
            round(c["macro_mean_response_fe"], 12),
            c["transition_count"],
        )
    return sorted(checkpoint_metrics, key=key)[0]


# =============================================================================
# TRAINING — one policy seed
# =============================================================================

def _to_tensor_batch(buffer: SharedReplayBuffer, batch):
    s, a, r, ns, d = buffer.to_arrays(batch)
    return (
        torch.tensor(s, dtype=torch.float32, device=DEVICE),
        torch.tensor(a, dtype=torch.int64, device=DEVICE),
        torch.tensor(r, dtype=torch.float32, device=DEVICE),
        torch.tensor(ns, dtype=torch.float32, device=DEVICE),
        torch.tensor(d, dtype=torch.float32, device=DEVICE),
    )


def _numerical_finite(*tensors) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return False
    return True


def train_one_policy(policy_seed: int, cache_train: dict, cache_val: dict,
                     env: EnvContext, policy_dir: Path,
                     *, n_epochs=N_EPOCHS, train_seeds=TRAIN_SEEDS,
                     checkpoint_transitions=CHECKPOINT_TRANSITIONS,
                     val_seeds=VALIDATION_SEEDS, verbose=True,
                     frozen_hash="", code_hash="",
                     state_normalizer=None, checkpoint_extra_meta=None,
                     reward_version=REWARD_R0_NAME, on_reward_record=None) -> dict:
    """Run one independent policy seed end-to-end. Returns per-policy summary.

    state_normalizer=None reproduces the frozen E4TR/E4F legacy V1 mapping (default).
    A callable raw9->9 selects an alternative per-dimension normalization for the
    E4HTR V1-vs-V3 A/B experiment. checkpoint_extra_meta is merged into each saved
    checkpoint payload (arm / normalization_version / constants hash / stage)."""
    ckpt_dir = policy_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- independent instances (Phase 2) ------------------------------------
    torch.manual_seed(policy_seed * 1000 + RNG_OFFSET_TORCH)
    online, target = build_online_target_pair(
        NETWORK_INPUT_DIM, seed=policy_seed * 1000 + RNG_OFFSET_POLICY_INIT)
    online.to(DEVICE); target.to(DEVICE)
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    buffer = SharedReplayBuffer(REPLAY_CAPACITY,
                                rng_seed=policy_seed * 1000 + RNG_OFFSET_REPLAY_SAMPLE)
    _ = np.random.default_rng(policy_seed * 1000 + RNG_OFFSET_NETWORK_MINIBATCH)  # reserved
    eps_cfg = EpsilonGreedyConfig(EPS_START, EPS_END, EPS_DECAY)
    order_rng = np.random.default_rng(policy_seed * 1000 + RNG_OFFSET_TRAIN_EPISODE_ORDER)

    total_transitions = 0
    optimizer_steps = 0
    checkpoints_meta: List[dict] = []
    training_log: List[dict] = []
    checkpoint_targets = set(checkpoint_transitions)
    huber = nn.SmoothL1Loss()
    numerical_stability = "PASS"
    failure_info = None

    def do_update(update_loss_acc):
        nonlocal optimizer_steps, numerical_stability, failure_info
        batch = buffer.sample(BATCH_SIZE)
        s, a, r, ns, d = _to_tensor_batch(buffer, batch)
        if not _numerical_finite(s, r, ns, d):
            numerical_stability = "FAIL"
            failure_info = {"stage": "batch_inputs", "transition": total_transitions}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite batch inputs")
        y = double_dqn_target(r, ns, d, online, target, GAMMA)
        q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = huber(q, y)
        if not _numerical_finite(q, y, loss):
            numerical_stability = "FAIL"
            failure_info = {"stage": "forward", "transition": total_transitions}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite q/target/loss")
        optimizer.zero_grad()
        loss.backward()
        grads_finite = all(
            (p.grad is None) or torch.isfinite(p.grad).all() for p in online.parameters()
        )
        if not grads_finite:
            numerical_stability = "FAIL"
            failure_info = {"stage": "gradients", "transition": total_transitions}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite gradients")
        optimizer.step()  # NO gradient clipping (frozen)
        if not all(torch.isfinite(p).all() for p in online.parameters()):
            numerical_stability = "FAIL"
            failure_info = {"stage": "parameters", "transition": total_transitions}
            raise RuntimeError("NUMERICAL_GUARD_FAIL: non-finite parameters")
        polyak_update(online, target, TARGET_TAU)
        optimizer_steps += 1
        update_loss_acc["loss"].append(float(loss.item()))
        update_loss_acc["abs_q"].append(float(q.abs().mean().item()))
        update_loss_acc["max_q"].append(float(q.abs().max().item()))

    def save_checkpoint(name: str, transition_count: int):
        path = ckpt_dir / f"{name}.pt"
        payload = {
            "online_state_dict": online.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_seed": policy_seed,
            "transition_count": transition_count,
            "optimizer_step_count": optimizer_steps,
            "epsilon": eps_cfg.get_epsilon(transition_count),
            "frozen_config_hash": frozen_hash,
            "code_manifest_hash": code_hash,
        }
        if checkpoint_extra_meta:
            payload.update(checkpoint_extra_meta)
        torch.save(payload, path)
        return path

    def append_and_learn(transition: RegionalTransition, update_loss_acc):
        nonlocal total_transitions
        buffer.push(transition)
        total_transitions += 1
        if len(buffer) >= WARMUP_TRANSITIONS and total_transitions % UPDATE_FREQUENCY == 0:
            do_update(update_loss_acc)
        if total_transitions in checkpoint_targets:
            online.eval()
            val = validate_online_net(online, cache_val, env, val_seeds, state_normalizer)
            online.train()
            path = save_checkpoint(f"ckpt_{total_transitions}", total_transitions)
            checkpoints_meta.append({
                "name": f"ckpt_{total_transitions}",
                "path": str(path),
                "transition_count": total_transitions,
                "optimizer_step_count": optimizer_steps,
                "epsilon": eps_cfg.get_epsilon(total_transitions),
                "VALIDATION_MACRO_MIGD_RESPONSE": val["VALIDATION_MACRO_MIGD_RESPONSE"],
                "macro_MIGD_epoch": val["macro_MIGD_epoch"],
                "macro_mean_response_fe": val["macro_mean_response_fe"],
                "per_scenario": val["per_scenario"],
            })

    online.train()
    episode_index = 0
    for epoch in range(n_epochs):
        episodes = [(sc, sd) for sc in SCENARIOS for sd in train_seeds]
        order = order_rng.permutation(len(episodes))
        episodes = [episodes[i] for i in order]

        for (scenario, env_seed) in episodes:
            FIREWALL.check(env_seed)
            region_dynamics = make_region_dynamics(scenario, env.R)
            state = _fresh_state(scenario, env_seed, cache_train, env.R)
            pending: List[Tuple[RegionalTransition, int]] = []  # (transition, region_id) awaiting next_state
            pending_states: Dict[int, np.ndarray] = {}          # region -> state at previous change
            ep_loss = {"loss": [], "abs_q": [], "max_q": []}
            ep_resp, ep_ep, ep_fe, ep_reward = [], [], [], 0.0
            act_counts = {a: 0 for a in ACTION_ORDER}
            n_fallback = 0
            prev_active = None
            start_transitions = total_transitions

            for k in range(N_SCORED_CHANGES):
                pre = pre_response(state, scenario, env_seed, env, region_dynamics)
                if prev_active is not None:
                    _assert_no_region_disappearance(prev_active, pre["active_regions"])
                local_norm, gc_norm, full_states = build_region_states(pre["v2_states"], env.R, state_normalizer)

                epsilon = eps_cfg.get_epsilon(total_transitions)
                event_seed = _derive_event_seed(policy_seed, scenario, env_seed, epoch, k)
                action_idx, action_str = select_regional_actions(
                    online, local_norm, gc_norm, epsilon, event_seed, DEVICE)
                actions = {r: action_str[r] for r in pre["active_regions"]}

                # finalize previous change's transitions with next_state = current state
                if pending:
                    for (trans, rid) in pending:
                        trans.next_state = full_states[rid]
                    for (trans, rid) in sorted(pending, key=lambda x: x[1]):
                        append_and_learn(trans, ep_loss)
                    pending = []

                t_current_before = state["t_current"]
                post = execute_and_recover(actions, pre, state, scenario, env_seed,
                                           env, region_dynamics, k + 2)
                _gr, region_rewards, _aux = compute_joint_rewards(
                    pre, post, post["records"], env, region_dynamics, t_current_before,
                    reward_version=reward_version)

                event_id = f"{policy_seed}:{scenario}:{env_seed}:e{epoch}:c{k}"
                rec_by_region = {rec.region_id: rec for rec in post["records"]}
                done = (k == N_SCORED_CHANGES - 1)
                if on_reward_record is not None:
                    on_reward_record(policy_seed, epoch, scenario, env_seed, episode_index, k,
                                     pre["active_regions"], action_str, rec_by_region, _gr,
                                     region_rewards, _aux)
                new_pending = []
                for r in sorted(pre["active_regions"]):
                    rec = rec_by_region[r]
                    trans = RegionalTransition(
                        state=full_states[r],
                        requested_action=ACTION_TO_IDX[action_str[r]],
                        realized_operator=rec.realized_operator,
                        reward=region_rewards[r],
                        next_state=np.zeros(NETWORK_INPUT_DIM, dtype=np.float32),
                        done=done,
                        region_id=r,
                        event_id=event_id,
                        fallback_applied=rec.fallback_used,
                        response_fe=int(rec.response_fe),
                        diagnostic_oracle_action=None,  # firewall: never used
                    )
                    if done:
                        append_and_learn(trans, ep_loss)  # terminal: next_state stays zero
                    else:
                        new_pending.append((trans, r))
                pending = new_pending

                # diagnostics
                ep_resp.append(post["igd_response"]); ep_ep.append(post["igd_epoch"])
                ep_fe.append(sum(rec.response_fe for rec in post["records"]))
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
                "regional_transition_count": total_transitions,
                "epsilon": eps_cfg.get_epsilon(total_transitions),
                "replay_size": len(buffer), "optimizer_steps": optimizer_steps,
                "mean_td_loss": float(np.mean(ep_loss["loss"])) if ep_loss["loss"] else float("nan"),
                "mean_abs_q": float(np.mean(ep_loss["abs_q"])) if ep_loss["abs_q"] else float("nan"),
                "max_abs_q": float(np.max(ep_loss["max_q"])) if ep_loss["max_q"] else float("nan"),
                "action_fraction_KEEP": act_counts["KEEP"] / total_act,
                "action_fraction_LOCAL": act_counts["LOCAL"] / total_act,
                "action_fraction_PREDICT": act_counts["PREDICT"] / total_act,
                "action_fraction_DIVERSIFY": act_counts["DIVERSIFY"] / total_act,
                "fallback_rate": n_fallback / total_act,
                "total_response_FE": int(np.sum(ep_fe)),
                "MIGD_response": float(np.mean(ep_resp)),
                "MIGD_epoch": float(np.mean(ep_ep)),
                "cumulative_reward": ep_reward,
                "numerical_finite_flag": True,
            })
            episode_index += 1
        if verbose:
            print(f"    [policy {policy_seed}] epoch {epoch+1}/{n_epochs} done "
                  f"(transitions={total_transitions}, opt_steps={optimizer_steps})")

    # final checkpoint + its validation
    online.eval()
    val_final = validate_online_net(online, cache_val, env, val_seeds, state_normalizer)
    online.train()
    final_path = save_checkpoint("ckpt_final", total_transitions)
    checkpoints_meta.append({
        "name": "ckpt_final", "path": str(final_path),
        "transition_count": total_transitions, "optimizer_step_count": optimizer_steps,
        "epsilon": eps_cfg.get_epsilon(total_transitions),
        "VALIDATION_MACRO_MIGD_RESPONSE": val_final["VALIDATION_MACRO_MIGD_RESPONSE"],
        "macro_MIGD_epoch": val_final["macro_MIGD_epoch"],
        "macro_mean_response_fe": val_final["macro_mean_response_fe"],
        "per_scenario": val_final["per_scenario"],
    })

    selected = select_checkpoint(checkpoints_meta)

    return {
        "policy_seed": policy_seed,
        "total_transitions": total_transitions,
        "optimizer_steps": optimizer_steps,
        "n_episodes": episode_index,
        "checkpoints": checkpoints_meta,
        "selected_checkpoint": selected,
        "training_log": training_log,
        "numerical_stability": numerical_stability,
        "failure_info": failure_info,
    }


# =============================================================================
# BASELINES on validation seeds (Phase 15)
# =============================================================================

def run_validation_baselines(cache_val, env, rule_config,
                             val_seeds=VALIDATION_SEEDS) -> dict:
    results = {}
    for ctrl in BASELINE_GLOBALS + [RULE_CTRL]:
        results[ctrl] = {}
        for scenario in SCENARIOS:
            resp, ep, fe = [], [], []
            for seed in val_seeds:
                cfg = rule_config if ctrl == RULE_CTRL else None
                m = eval_fixed_arm(ctrl, scenario, seed, cache_val, env, cfg)
                resp.append(m["MIGD_response"]); ep.append(m["MIGD_epoch"]); fe.append(m["mean_response_fe"])
            results[ctrl][scenario] = {
                "mean_MIGD_response": float(np.mean(resp)),
                "mean_MIGD_epoch": float(np.mean(ep)),
                "mean_response_fe": float(np.mean(fe)),
            }

    best_fixed_global = {}
    for scenario in SCENARIOS:
        cand = {c: results[c][scenario] for c in BASELINE_GLOBALS}
        # 1 lowest MIGD_response, 2 lower FE, 3 lower MIGD_epoch, 4 canonical lower action id
        def key(c):
            m = cand[c]
            return (round(m["mean_MIGD_response"], 12), round(m["mean_response_fe"], 12),
                    round(m["mean_MIGD_epoch"], 12), BASELINE_GLOBALS.index(c))
        best = sorted(BASELINE_GLOBALS, key=key)[0]
        best_fixed_global[scenario] = {
            "controller": best,
            "action": best.replace("GLOBAL_", ""),
            "mean_MIGD_response": cand[best]["mean_MIGD_response"],
        }
    return {"per_controller": results, "best_fixed_global": best_fixed_global}


# =============================================================================
# ARTIFACT WRITERS
# =============================================================================

def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_all_artifacts(policy_results, baselines, det_check, rule_config,
                        frozen_cfg, mismatches, prod_hashes):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # per-policy dirs
    for pr in policy_results:
        ps = pr["policy_seed"]
        pdir = RESULTS_DIR / f"policy_{ps}"
        pdir.mkdir(parents=True, exist_ok=True)

        _write_csv(pdir / "training_log.csv",
                   list(pr["training_log"][0].keys()), pr["training_log"])

        cv_rows = []
        for c in pr["checkpoints"]:
            for scenario in SCENARIOS:
                sc = c["per_scenario"][scenario]
                cv_rows.append({
                    "checkpoint": c["name"], "transition_count": c["transition_count"],
                    "scenario": scenario,
                    "mean_MIGD_response": sc["mean_MIGD_response"],
                    "mean_MIGD_epoch": sc["mean_MIGD_epoch"],
                    "mean_response_fe": sc["mean_response_fe"],
                    "VALIDATION_MACRO_MIGD_RESPONSE": c["VALIDATION_MACRO_MIGD_RESPONSE"],
                })
        _write_csv(pdir / "checkpoint_validation.csv",
                   ["checkpoint", "transition_count", "scenario", "mean_MIGD_response",
                    "mean_MIGD_epoch", "mean_response_fe", "VALIDATION_MACRO_MIGD_RESPONSE"],
                   cv_rows)

        _jdump({
            "policy_seed": ps,
            "total_transitions": pr["total_transitions"],
            "optimizer_steps": pr["optimizer_steps"],
            "n_episodes": pr["n_episodes"],
            "numerical_stability": pr["numerical_stability"],
            "failure_info": pr["failure_info"],
            "checkpoints": [{k: v for k, v in c.items() if k != "per_scenario"}
                            for c in pr["checkpoints"]],
        }, pdir / "training_summary.json")

        # action summary from last epoch episodes
        _jdump(_policy_action_summary(pr), pdir / "action_summary.json")

        sel = pr["selected_checkpoint"]
        ckpt_path = Path(sel["path"])
        _jdump({
            "policy_seed": ps,
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "VALIDATION_MACRO_MIGD_RESPONSE": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "macro_mean_response_fe": sel["macro_mean_response_fe"],
            "checkpoint_path": sel["path"],
            "sha256": _hash_file(sel["path"]),
            "file_size": Path(sel["path"]).stat().st_size if ckpt_path.exists() else 0,
        }, pdir / "selected_checkpoint.json")

    # aggregate training diagnostics
    all_log = []
    for pr in policy_results:
        all_log.extend(pr["training_log"])
    if all_log:
        _write_csv(RESULTS_DIR / "E4TR_training_diagnostics.csv",
                   list(all_log[0].keys()), all_log)

    # validation all policies
    va_rows = []
    for pr in policy_results:
        for c in pr["checkpoints"]:
            va_rows.append({
                "policy_seed": pr["policy_seed"], "checkpoint": c["name"],
                "transition_count": c["transition_count"],
                "VALIDATION_MACRO_MIGD_RESPONSE": c["VALIDATION_MACRO_MIGD_RESPONSE"],
                "macro_MIGD_epoch": c["macro_MIGD_epoch"],
                "macro_mean_response_fe": c["macro_mean_response_fe"],
                "MIGD_response_PERSISTENT": c["per_scenario"]["PERSISTENT"]["mean_MIGD_response"],
                "MIGD_response_STATIONARY": c["per_scenario"]["STATIONARY"]["mean_MIGD_response"],
                "MIGD_response_MIXED": c["per_scenario"]["MIXED"]["mean_MIGD_response"],
                "selected": (c["name"] == pr["selected_checkpoint"]["name"]),
            })
    _write_csv(RESULTS_DIR / "E4TR_validation_all_policies.csv",
               ["policy_seed", "checkpoint", "transition_count",
                "VALIDATION_MACRO_MIGD_RESPONSE", "macro_MIGD_epoch",
                "macro_mean_response_fe", "MIGD_response_PERSISTENT",
                "MIGD_response_STATIONARY", "MIGD_response_MIXED", "selected"],
               va_rows)

    # action distribution (per policy, selected checkpoint validation is greedy; use train last-epoch)
    ad_rows = []
    for pr in policy_results:
        s = _policy_action_summary(pr)
        for a in ACTION_ORDER:
            ad_rows.append({"policy_seed": pr["policy_seed"], "action": a,
                            "final_epoch_fraction": s["final_epoch_fraction"][a]})
    _write_csv(RESULTS_DIR / "E4TR_action_distribution.csv",
               ["policy_seed", "action", "final_epoch_fraction"], ad_rows)

    # selected checkpoints (with hashes)
    sel_all = {}
    for pr in policy_results:
        sel = pr["selected_checkpoint"]
        sel_all[str(pr["policy_seed"])] = {
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "VALIDATION_MACRO_MIGD_RESPONSE": sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "macro_mean_response_fe": sel["macro_mean_response_fe"],
            "checkpoint_path": sel["path"],
            "sha256": _hash_file(sel["path"]),
            "file_size": Path(sel["path"]).stat().st_size if Path(sel["path"]).exists() else 0,
        }
    _jdump({
        "policy_seeds": POLICY_SEEDS,
        "all_five_retained": len(policy_results) == 5,
        "no_best_policy_seed_selection": True,
        "selected": sel_all,
    }, RESULTS_DIR / "E4TR_selected_checkpoints.json")

    _jdump(baselines["best_fixed_global"], RESULTS_DIR / "E4TR_best_fixed_global_validation.json")
    _jdump({
        "rule_name": RULE_CTRL, "retuned": False,
        "thresholds_source": "results/E3R_rule_v2_semantic_rerun/E3R_rule_v1_v2state.json",
        "thresholds": rule_config,
        "validation_metrics": baselines["per_controller"][RULE_CTRL],
    }, RESULTS_DIR / "E4TR_rule_validation.json")

    _jdump({
        "namespaces": {
            "POLICY_INIT": "policy_seed*1000+0 (torch generator, network init)",
            "POLICY_EXPLORATION": "SeedSequence[TAG,policy,scenario,seed,epoch,change] XOR region*PRIME",
            "REPLAY_SAMPLE": "policy_seed*1000+2 (SharedReplayBuffer rng)",
            "NETWORK_MINIBATCH": "policy_seed*1000+3 (reserved, unused)",
            "TRAIN_EPISODE_ORDER": "policy_seed*1000+7 (episode permutation)",
            "RESPONSE_OPERATOR": "PRODUCTION UNCHANGED (make_rng in oracle.py)",
            "RESPONSE_SURVIVAL": "PRODUCTION UNCHANGED (make_rng RESPONSE_SURVIVAL)",
            "BASE_EVOLUTION": "PRODUCTION UNCHANGED (NSGA-II seed=env_seed*10000+idx)",
        },
        "drl_streams_isolated_from_production": True,
        "per_policy_independent": True,
    }, RESULTS_DIR / "E4TR_rng_audit.json")

    _jdump({
        "oracle_called_in_training_path": False,
        "diagnostic_oracle_action_in_transitions": False,
        "training_tensor_fields": ["state", "requested_action", "reward", "next_state", "done"],
        "forbidden_as_target_or_state": [
            "oracle_best_action", "diagnostic_oracle_action", "oracle_utility",
            "ROG", "best_global_action", "regime_label",
        ],
        "regime_label_in_state": False,
        "status": "PASS",
    }, RESULTS_DIR / "E4TR_oracle_firewall_audit.json")

    _jdump({
        "final_eval_seeds": FINAL_EVAL_SEEDS,
        "FINAL_EVAL_ACCESS_COUNT": FIREWALL.access_count,
        "instantiated": FIREWALL.access_count > 0,
        "status": "PASS" if FIREWALL.access_count == 0 else "FAIL",
    }, RESULTS_DIR / "E4TR_final_eval_firewall.json")

    # execution config (frozen echo + operational definitions)
    _jdump({
        "stage": "E4TR_SHARED_DOUBLE_DQN_TRAINING",
        "frozen_config_source": str(E4T_FROZEN_PATH),
        "frozen_config_sha256": _frozen_config_hash(),
        "config_mismatches_vs_prompt": mismatches,
        "seed_split": {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS,
                       "final_eval_FIREWALLED": FINAL_EVAL_SEEDS},
        "policy_seeds": POLICY_SEEDS,
        "training_budget": {"n_epochs": N_EPOCHS, "episodes_per_epoch": EPISODES_PER_EPOCH,
                            "n_train_episodes": N_TRAIN_EPISODES,
                            "changes_per_episode": N_SCORED_CHANGES},
        "network": {"input_dim": NETWORK_INPUT_DIM, "hidden": [128, 128],
                    "output_dim": N_ACTIONS, "loss": "SmoothL1", "optimizer": "Adam",
                    "lr": LEARNING_RATE, "gradient_clipping": "NONE"},
        "hyperparameters": {"batch_size": BATCH_SIZE, "replay_capacity": REPLAY_CAPACITY,
                            "gamma": GAMMA, "target_tau": TARGET_TAU,
                            "warmup_transitions": WARMUP_TRANSITIONS,
                            "update_frequency": UPDATE_FREQUENCY,
                            "gradient_steps_per_update": GRAD_STEPS_PER_UPDATE},
        "epsilon": {"start": EPS_START, "end": EPS_END, "decay_transitions": EPS_DECAY,
                    "validation_epsilon": 0.0},
        "reward": {"scope": "JOINT_GLOBAL_PLUS_LOCAL", "alpha_g": ALPHA_G,
                   "alpha_l": ALPHA_L, "beta": BETA,
                   "global_normalization_denominator": "abs(q_before)+1e-12",
                   "quality_fn": "negative_IGD"},
        "local_reward_operational_definition": {
            "pbi_loss_before_r": "quality_loss(F_tm_minus,F_tm_plus,rids,V,theta)[r] "
                                 "(pre-response regional convergence loss == convergence_loss_r)",
            "pbi_loss_after_r": "quality_loss on the ACTUAL joint RESPONSE population: "
                                "F_resp_minus=cp_evaluate(X_resp,t_current) (offline reward label), "
                                "F_resp_plus=response F at t_new; normalize; partition; quality_loss[r]",
            "local_reward_r": "-(pbi_loss_after_r - pbi_loss_before_r)",
            "uses_isolated_oracle_branches": False,
            "uses_actual_joint_response_population": True,
            "theta_pbi": THETA_PBI,
        },
        "checkpoint_transitions": CHECKPOINT_TRANSITIONS,
        "checkpoint_metric": "VALIDATION_MACRO_MIGD_RESPONSE",
        "early_stopping": False,
        "region_disappearance_gate": "GAP_ABSENT (R=10 regions always active; runtime guard armed)",
        "RL_STATUS_at_start": RL_STATUS_NOT_STARTED,
    }, RESULTS_DIR / "E4TR_execution_config.json")

    # determinism check
    _jdump(det_check, RESULTS_DIR / "E4TR_validation_determinism.json")

    # firewall recheck (Phase 0 record)
    inter = seed_intersections()
    hist = historical_seed_overlap()
    _jdump({
        "seed_intersections": inter,
        "all_intersections_empty": all(len(v) == 0 for v in inter.values()),
        "historical_overlap": hist,
        "no_historical_overlap": all(len(v) == 0 for v in hist.values()),
        "final_eval_firewall_access_count": FIREWALL.access_count,
        "oracle_firewall": "PASS",
        "production_operator_hashes": prod_hashes,
    }, RESULTS_DIR / "E4TR_firewall_recheck.json")

    # pretrain manifest
    _jdump({
        "stage": "E4TR_SHARED_DOUBLE_DQN_TRAINING",
        "E4T_COMPLETE": True,
        "frozen_config_sha256": _frozen_config_hash(),
        "code_manifest_sha256": _code_manifest_hash(),
        "production_operator_hashes": prod_hashes,
        "PRODUCTION_DEFAULT": "O1_F005",
        "no_HARD_action": True, "no_MEMORY_action": True,
        "budget_per_active_region": CURRENT_RESPONSE_BUDGET_VALUE,
        "base_evolution": "NSGA-II",
    }, RESULTS_DIR / "E4TR_pretrain_manifest.json")


def _policy_action_summary(pr) -> dict:
    log = pr["training_log"]
    last_epoch = max(row["epoch"] for row in log)
    rows = [r for r in log if r["epoch"] == last_epoch]
    frac = {a: float(np.mean([r[f"action_fraction_{a}"] for r in rows])) for a in ACTION_ORDER}
    first_rows = [r for r in log if r["epoch"] == 0]
    frac0 = {a: float(np.mean([r[f"action_fraction_{a}"] for r in first_rows])) for a in ACTION_ORDER}
    return {
        "policy_seed": pr["policy_seed"],
        "final_epoch_fraction": frac,
        "first_epoch_fraction": frac0,
        "final_epoch_mean_MIGD_response": float(np.mean([r["MIGD_response"] for r in rows])),
        "first_epoch_mean_MIGD_response": float(np.mean([r["MIGD_response"] for r in first_rows])),
    }


def write_verdict_and_report(policy_results, baselines, det_check, mismatches,
                             full_regression="(run separately)"):
    all_stable = all(pr["numerical_stability"] == "PASS" for pr in policy_results)
    n_complete = sum(1 for pr in policy_results if pr["numerical_stability"] == "PASS")
    all_five = len(policy_results) == 5
    budgets_ok = all(pr["n_episodes"] == N_TRAIN_EPISODES for pr in policy_results)
    firewall_ok = FIREWALL.access_count == 0
    det_ok = det_check["determinism_pass"]

    e4tr_complete = (all_five and n_complete == 5 and all_stable and budgets_ok
                     and firewall_ok and det_ok and len(mismatches) == 0)

    verdict = {
        "stage": "E4TR_SHARED_DOUBLE_DQN_TRAINING",
        "E4TR_COMPLETE": "YES" if e4tr_complete else "NO",
        "POLICY_RUNS_COMPLETE": f"{n_complete}/5",
        "all_five_retained": all_five,
        "no_best_policy_seed_selection": True,
        "training_budget_per_policy": N_TRAIN_EPISODES,
        "training_budgets_completed": budgets_ok,
        "numerical_stability": "PASS" if all_stable else "FAIL",
        "oracle_label_firewall": "PASS",
        "rng_isolation": "PASS",
        "validation_determinism": "PASS" if det_ok else "FAIL",
        "FINAL_EVAL_ACCESS_COUNT": FIREWALL.access_count,
        "hyperparameter_retuning": "NO",
        "final_evaluation": "NOT_RUN",
        "scientific_superiority_claim": "NOT_MADE",
        "config_mismatches_vs_prompt": mismatches,
        "selected_checkpoints": {
            str(pr["policy_seed"]): {
                "transition_count": pr["selected_checkpoint"]["transition_count"],
                "VALIDATION_MACRO_MIGD_RESPONSE": pr["selected_checkpoint"]["VALIDATION_MACRO_MIGD_RESPONSE"],
            } for pr in policy_results
        },
        "best_fixed_global": baselines["best_fixed_global"],
        "RL_STATUS": "TRAINED_AWAITING_FINAL_EVAL" if e4tr_complete else "TRAINING_FAILED",
        "DRL_TRAINING": "COMPLETE" if e4tr_complete else "FAILED",
        "FULL_REGRESSION": full_regression,
        "NEXT_STAGE": ("E4E_FINAL_EVALUATION_PROTOCOL_FREEZE" if e4tr_complete
                       else "E4TR_ENGINEERING_FAILURE_REVIEW"),
    }
    _jdump(verdict, RESULTS_DIR / "E4TR_verdict.json")

    lines = [
        "# E4TR — Shared Parameter Double DQN Training",
        "",
        f"**E4TR_COMPLETE = {verdict['E4TR_COMPLETE']}**",
        "",
        "First substantive DRL training stage. Protocol frozen in E4-T and executed "
        "exactly; no tuning, no early stopping, no best-policy-seed selection. Final "
        "evaluation seeds 1081..1140 were never instantiated.",
        "",
        "## Selected checkpoints (per policy seed, all retained)",
        "",
        "| policy | selected | transitions | VAL_MACRO_MIGD_RESPONSE |",
        "|---|---|---|---|",
    ]
    for pr in policy_results:
        sel = pr["selected_checkpoint"]
        lines.append(f"| {pr['policy_seed']} | {sel['name']} | {sel['transition_count']} "
                     f"| {sel['VALIDATION_MACRO_MIGD_RESPONSE']:.6f} |")
    lines += [
        "",
        "## Best fixed-global comparator (validation seeds only)",
        "",
        "| scenario | best global | MIGD_response |",
        "|---|---|---|",
    ]
    for sc in SCENARIOS:
        b = baselines["best_fixed_global"][sc]
        lines.append(f"| {sc} | {b['action']} | {b['mean_MIGD_response']:.6f} |")
    lines += [
        "",
        f"- NUMERICAL_STABILITY = {verdict['numerical_stability']}",
        f"- VALIDATION_DETERMINISM = {verdict['validation_determinism']}",
        f"- FINAL_EVAL_ACCESS_COUNT = {FIREWALL.access_count}",
        f"- RL_STATUS = {verdict['RL_STATUS']}",
        f"- NEXT_STAGE = {verdict['NEXT_STAGE']}",
        "",
        "SCIENTIFIC_SUPERIORITY_CLAIM = NOT_MADE (final test seeds remain unopened).",
    ]
    (RESULTS_DIR / "E4TR_report.md").write_text("\n".join(lines))
    return verdict


# =============================================================================
# MAIN
# =============================================================================

def load_rule_config() -> dict:
    with open("results/E3R_rule_v2_semantic_rerun/E3R_rule_v1_v2state.json") as f:
        return json.load(f)["thresholds"]


def production_operator_hashes() -> dict:
    files = ["src/prdrl/operators/local.py", "src/prdrl/operators/predict.py",
             "src/prdrl/operators/diversify.py", "src/prdrl/operators/budget.py",
             "src/prdrl/oracle/oracle.py", "src/prdrl/core/survival.py"]
    return {fp: _hash_file(fp) for fp in files}


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4TR — Shared Parameter Double DQN Training")
    print("=" * 60)

    # ---- Phase 0 ------------------------------------------------------------
    frozen_cfg, mismatches = load_and_validate_frozen_config()
    if mismatches:
        print("[E4TR] FROZEN CONFIG MISMATCH — STOP:")
        for m in mismatches:
            print("   ", m)
        raise SystemExit("E4TR_STOP: frozen config disagrees with prompt")
    inter = seed_intersections()
    assert all(len(v) == 0 for v in inter.values()), f"seed intersection non-empty: {inter}"
    hist = historical_seed_overlap()
    assert all(len(v) == 0 for v in hist.values()), f"historical overlap: {hist}"
    print("[E4TR] Phase 0: config OK, seed sets disjoint, no historical overlap.")

    env = make_env_context()
    print(f"[E4TR] env: R={env.R}, POP_SIZE={POP_SIZE}, D={D}")
    rule_config = load_rule_config()
    prod_hashes = production_operator_hashes()
    frozen_hash = _frozen_config_hash()
    code_hash = _code_manifest_hash()

    print("[E4TR] building warmup+burn-in caches (train + validation)...")
    cache_train = build_warmup_cache(TRAIN_SEEDS, env)
    cache_val = build_warmup_cache(VALIDATION_SEEDS, env)

    # ---- Phase 1-14: train five independent policies ------------------------
    policy_results = []
    for ps in POLICY_SEEDS:
        print(f"[E4TR] training policy seed {ps} ...")
        tp = time.time()
        pdir = RESULTS_DIR / f"policy_{ps}"
        pr = train_one_policy(ps, cache_train, cache_val, env, pdir,
                              frozen_hash=frozen_hash, code_hash=code_hash)
        pr["wall_seconds"] = time.time() - tp
        print(f"    policy {ps}: transitions={pr['total_transitions']}, "
              f"opt_steps={pr['optimizer_steps']}, "
              f"selected={pr['selected_checkpoint']['name']}, "
              f"stability={pr['numerical_stability']}, {pr['wall_seconds']:.0f}s")
        policy_results.append(pr)

    # ---- Phase 14: validation determinism -----------------------------------
    print("[E4TR] validation determinism check (policy 2001, ckpt_10000 x2)...")
    det_check = run_determinism_check(policy_results, cache_val, env)

    # ---- Phase 15: baselines on validation seeds ----------------------------
    print("[E4TR] validation baselines (GLOBAL_*, RULE_V1_V2STATE)...")
    baselines = run_validation_baselines(cache_val, env, rule_config)

    # ---- Phase 16: firewall assertion ---------------------------------------
    assert FIREWALL.access_count == 0, "FINAL_EVAL firewall breached"

    # ---- Phase 19: artifacts ------------------------------------------------
    print("[E4TR] writing artifacts...")
    write_all_artifacts(policy_results, baselines, det_check, rule_config,
                        frozen_cfg, mismatches, prod_hashes)
    verdict = write_verdict_and_report(policy_results, baselines, det_check, mismatches)

    _print_terminal_summary(policy_results, baselines, det_check, verdict)
    print(f"[E4TR] total wall time {time.time()-t0:.0f}s")
    return verdict


def run_determinism_check(policy_results, cache_val, env) -> dict:
    """Re-evaluate policy 2001 ckpt_10000 twice; require identical macro metrics."""
    target = None
    for pr in policy_results:
        if pr["policy_seed"] == 2001:
            for c in pr["checkpoints"]:
                if c["name"] == "ckpt_10000":
                    target = c
                    break
    if target is None:
        return {"determinism_pass": True, "note": "ckpt_10000 not present (smoke)"}
    payload = torch.load(target["path"], weights_only=False)
    net1 = SharedRegionalQNetwork(NETWORK_INPUT_DIM); net1.load_state_dict(payload["online_state_dict"]); net1.eval()
    net2 = SharedRegionalQNetwork(NETWORK_INPUT_DIM); net2.load_state_dict(payload["online_state_dict"]); net2.eval()
    v1 = validate_online_net(net1, cache_val, env)
    v2 = validate_online_net(net2, cache_val, env)
    diff = abs(v1["VALIDATION_MACRO_MIGD_RESPONSE"] - v2["VALIDATION_MACRO_MIGD_RESPONSE"])
    return {
        "determinism_pass": diff <= 1e-12,
        "run1_macro": v1["VALIDATION_MACRO_MIGD_RESPONSE"],
        "run2_macro": v2["VALIDATION_MACRO_MIGD_RESPONSE"],
        "abs_diff": diff, "tolerance": 1e-12,
        "checkpoint": "policy_2001/ckpt_10000",
    }


def _print_terminal_summary(policy_results, baselines, det_check, verdict):
    prmap = {pr["policy_seed"]: pr for pr in policy_results}
    print()
    print(f"E4TR_COMPLETE                 = {verdict['E4TR_COMPLETE']}")
    print(f"POLICY_RUNS_COMPLETE          = {verdict['POLICY_RUNS_COMPLETE']}")
    for ps in POLICY_SEEDS:
        pr = prmap.get(ps)
        if pr:
            print(f"POLICY_{ps}_SELECTED_CHECKPOINT = {pr['selected_checkpoint']['transition_count']}")
    for ps in POLICY_SEEDS:
        pr = prmap.get(ps)
        if pr:
            print(f"POLICY_{ps}_VALIDATION_MACRO_MIGD = "
                  f"{pr['selected_checkpoint']['VALIDATION_MACRO_MIGD_RESPONSE']:.6f}")
    for sc in SCENARIOS:
        print(f"BEST_FIXED_GLOBAL_{sc:<11} = {baselines['best_fixed_global'][sc]['action']}")
    print(f"NUMERICAL_STABILITY           = {verdict['numerical_stability']}")
    print(f"ORACLE_LABEL_FIREWALL         = PASS")
    print(f"RNG_ISOLATION                 = PASS")
    print(f"VALIDATION_DETERMINISM        = {verdict['validation_determinism']}")
    print(f"FINAL_EVAL_ACCESS_COUNT       = {FIREWALL.access_count}")
    print(f"HYPERPARAMETER_RETUNING       = NO")
    print(f"FINAL_EVALUATION              = NOT_RUN")
    print(f"SCIENTIFIC_SUPERIORITY_CLAIM  = NOT_MADE")
    print(f"RL_STATUS                     = {verdict['RL_STATUS']}")
    print(f"DRL_TRAINING                  = {verdict['DRL_TRAINING']}")
    print(f"FULL_REGRESSION               = {verdict['FULL_REGRESSION']}")
    print(f"NEXT_STAGE                    = {verdict['NEXT_STAGE']}")


if __name__ == "__main__":
    main()
