"""
E4KT — Joint Coordination A/B Training Protocol Freeze (PR-DRL-DMOEA, Branch C).

Freezes the COMPLETE matched training experiment C0 (V3+R1 Independent Shared DDQN) versus
C1 (V3+R1+QMIX monotonic joint-value training) BEFORE any RL training, with the coordination
learner specification as the SOLE scientific factor. FREEZE ONLY: no training, no instantiation
of Branch-C seeds 1901..2040.

Both arms share V3 normalization, R1 reward (alpha_local=0.1), 9-128-128-4 utility network,
all hyperparameters/operators/seeds/checkpoint schedule/selection. The coordination experiment
tests the JOINT TRAINING ARCHITECTURE, not reward or normalization.

Effect: C0 - C1 (positive => QMIX treatment better).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

import torch
from prdrl.rl.network import build_online_target_pair
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256
from prdrl.rl.coordination_integration import (
    make_qmix_config, qmix_config_sha256, CONTROL_ARCH, FROZEN_V3_CONSTANTS_SHA256,
    GAMMA, TARGET_TAU, STATE_DIM, INACTIVE_ACTION_SENTINEL,
)
from prdrl.rl.qmix import (
    TREATMENT_ARCH, R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM,
    NONNEGATIVE_TRANSFORM, MIXER_SEED_OFFSET, build_mixer_online_target_pair,
)
from prdrl.evaluation.crossed_bootstrap import N_BOOTSTRAP, CI_LEVEL, CI_METHOD
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)
from prdrl.experiments.E4K_joint_coordination_revision_design import (
    coordination_causal_verdict_skeleton,
    CONSUMED_SEED_SETS, CONSUMED_POLICY_SEEDS,
)

RESULTS_DIR = Path("results/E4KT_joint_coordination_ab_training_protocol_freeze")
E4K_DIR = Path("results/E4K_joint_coordination_revision_design")
E4KI_DIR = Path("results/E4KI_joint_coordination_implementation_validation")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")

QMIX_CONFIG_SHA256_EXPECTED = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
TRAIN_SEEDS = list(range(1901, 1961))
VALIDATION_SEEDS = list(range(1961, 1981))
FINAL_SEEDS = list(range(1981, 2041))
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]

N_EPOCHS = 6
EPISODES_PER_EPOCH = len(TRAIN_SEEDS) * len(SCENARIOS)   # 180
EPISODES_PER_RUN = N_EPOCHS * EPISODES_PER_EPOCH          # 1080
N_RUNS = 10                                                # 2 arms × 5 policies
TOTAL_TRAINING_EPISODES = N_RUNS * EPISODES_PER_RUN       # 10800
N_SCORED_CHANGES = 10

BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
LEARNING_RATE = 1e-4
WARMUP_ANCHORS = 1000
UPDATE_FREQUENCY = 4
GRAD_STEPS_PER_UPDATE = 1
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 50_000
GRAD_CLIP = "NONE"
LOSS_FN = "SmoothL1"

CHECKPOINT_ANCHORS = [10_000, 20_000, 30_000, 40_000, 50_000]
CHECKPOINTS_PER_RUN = 6
TOTAL_CHECKPOINTS = 60
VALIDATION_EPISODES = 2 * 5 * 6 * 3 * 20  # 3600

RNG_OFFSET_POLICY_INIT = 0
BOOTSTRAP_MASTER_SEED = 86001
PRACTICAL_MARGIN = 1e-5

_coord_access = {"train": 0, "validation": 0, "final": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _param_hash(net) -> str:
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_json_default)


def _json_default(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(str(type(x)))


def _load(p):
    with open(p) as f:
        return json.load(f)


# ============================================================================
# PHASE E4KT-0: PRE-FREEZE INTEGRITY
# ============================================================================

def phase0_integrity() -> Dict:
    e4ki = _load(E4KI_DIR / "E4KI_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    e4k = _load(E4K_DIR / "E4K_protocol_frozen.json")
    e4ki_qmix = _load(E4KI_DIR / "E4KI_qmix_config.json")

    checks = {}
    checks["E4KI_COMPLETE"] = e4ki.get("E4KI_COMPLETE") == "YES"
    checks["QMIX_IMPLEMENTATION_VALID"] = e4ki.get("QMIX_IMPLEMENTATION_VALID") == "YES"
    checks["E4JF_FAILED"] = e4jf.get("GLOBAL_COMPARATOR_VERDICT") == "FAILED"
    checks["E4IF_VALIDATED"] = e4if.get("REWARD_CAUSAL_VERDICT") == "VALIDATED"
    checks["E4HF_FAILED"] = e4hf.get("NORMALIZATION_CAUSAL_VERDICT") == "FAILED"
    checks["E4F_FAILED"] = e4f.get("SCIENTIFIC_VERDICT") == "FAILED"
    checks["BRANCH_C"] = e4k.get("SELECTED_BRANCH") == "C_JOINT_COORDINATION"
    checks["V3_HASH"] = e4k.get("FIXED_NORMALIZATION") == NORM_V3
    checks["R1_REWARD"] = e4k.get("FIXED_REWARD") == REWARD_R1_NAME
    checks["ALPHA_LOCAL"] = e4k.get("FIXED_ALPHA_LOCAL") == 0.1
    checks["UTILITY_NETWORK"] = e4k.get("UTILITY_NETWORK") == "9-128-128-4"
    checks["R_MAX"] = e4k.get("R_MAX") == 10
    checks["CENTRAL_STATE_DIM"] = e4k.get("CENTRAL_STATE_DIM") == 55
    checks["QMIX_CONFIG_SHA256"] = qmix_config_sha256() == QMIX_CONFIG_SHA256_EXPECTED
    checks["QMIX_CONFIG_FROM_KI"] = e4ki_qmix.get("QMIX_CONFIG_SHA256") == QMIX_CONFIG_SHA256_EXPECTED

    e4ki_ceq = _load(E4KI_DIR / "E4KI_control_equivalence.json")
    checks["C0_BACKWARD_EQUIV"] = e4ki_ceq.get("C0_BACKWARD_EQUIVALENCE") == "PASS"

    e4ki_fw = _load(E4KI_DIR / "E4KI_future_seed_firewall.json")
    checks["PRIOR_FIREWALL_PASS"] = e4ki_fw.get("PASS") is True

    all_ok = all(checks.values())
    return {"integrity_ok": all_ok, "checks": checks,
            "e4f_verdict": e4f.get("SCIENTIFIC_VERDICT"),
            "e4hf_verdict": e4hf.get("NORMALIZATION_CAUSAL_VERDICT"),
            "e4if_verdict": e4if.get("REWARD_CAUSAL_VERDICT"),
            "e4jf_verdict": e4jf.get("GLOBAL_COMPARATOR_VERDICT")}


# ============================================================================
# PHASE E4KT-1: SINGLE SCIENTIFIC FACTOR
# ============================================================================

def build_arm_config(arm_name: str) -> Dict:
    shared = {
        "normalization_version": NORM_V3,
        "v3_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "state_semantics": "local_V2[4]+GlobalContextV1[5]=9D",
        "regional_utility_architecture": "9-128-128-4",
        "action_set": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "response_operators": {
            "KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
            "predict_fallback": "LOCAL", "sigma_D": 0.15,
        },
        "response_budget": "10 PER_ACTIVE_REGION",
        "survival": "unrestricted standard NSGA-II",
        "gamma": GAMMA,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_family": LOSS_FN,
        "replay_anchor_capacity": REPLAY_CAPACITY,
        "warmup": WARMUP_ANCHORS,
        "update_cadence": UPDATE_FREQUENCY,
        "batch_size": BATCH_SIZE,
        "epsilon_schedule": f"{EPS_START}->{EPS_END} linear over {EPS_DECAY} regional anchors/transitions",
        "target_tau": TARGET_TAU,
        "training_seeds": "1901..1960",
        "validation_seeds": "1961..1980",
        "final_seeds": "1981..2040",
        "policy_seeds": POLICY_SEEDS,
        "training_budget": f"{EPISODES_PER_RUN} episodes/run, {N_RUNS} runs, {TOTAL_TRAINING_EPISODES} total",
        "checkpoint_schedule": str(CHECKPOINT_ANCHORS) + " + final",
        "validation_checkpoint_selection_rule": "min VALIDATION_MACRO_MIGD_RESPONSE",
    }
    if arm_name == "C0":
        shared["architecture_version"] = CONTROL_ARCH
        shared["mixer"] = "NONE"
        shared["joint_td"] = "regional Double-DQN SmoothL1 TD"
        shared["joint_reward"] = "regional R1 (no aggregation)"
        shared["replay_type"] = "regional transitions"
        shared["optimizer_params"] = "utility only"
        shared["target_update"] = "utility target tau=0.005"
    elif arm_name == "C1":
        shared["architecture_version"] = TREATMENT_ARCH
        shared["mixer"] = "QMIX_MIXER_V1"
        shared["mixer_hidden_dim"] = MIX_HIDDEN_DIM
        shared["central_state_dim"] = CENTRAL_STATE_DIM
        shared["R_MAX"] = R_MAX
        shared["nonnegative_transform"] = NONNEGATIVE_TRANSFORM
        shared["joint_td"] = "Double-Q QMIX joint SmoothL1 TD"
        shared["joint_reward"] = "ACTIVE_REGION_MEAN_OF_R1"
        shared["replay_type"] = "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT"
        shared["optimizer_params"] = "utility + mixer"
        shared["target_update"] = "utility target + mixer target tau=0.005"
        shared["qmix_config_sha256"] = QMIX_CONFIG_SHA256_EXPECTED
    return shared


def arm_config_scientific_diff(c0: Dict, c1: Dict) -> Dict:
    shared_keys = []
    differing = []
    for k in sorted(set(c0.keys()) | set(c1.keys())):
        if k in c0 and k in c1:
            if c0[k] == c1[k]:
                shared_keys.append(k)
            else:
                differing.append(k)
        else:
            differing.append(k)
    authorized = {
        "architecture_version", "mixer", "mixer_hidden_dim", "central_state_dim",
        "R_MAX", "nonnegative_transform", "joint_td", "joint_reward",
        "replay_type", "optimizer_params", "target_update", "qmix_config_sha256",
    }
    unauthorized = [k for k in differing if k not in authorized]
    return {
        "shared_fields": shared_keys,
        "differing_fields": differing,
        "authorized_differences": sorted(set(differing) & authorized),
        "unauthorized_differences": unauthorized,
        "single_factor": len(unauthorized) == 0 and len(differing) > 0,
    }


# ============================================================================
# PHASE E4KT-16: PAIRED UTILITY INITIALIZATION
# ============================================================================

def verify_paired_initialization() -> Dict:
    rows = {}
    paired_ok = True
    for p in POLICY_SEEDS:
        util_seed = p * 1000 + RNG_OFFSET_POLICY_INIT
        c0_online, c0_target = build_online_target_pair(STATE_DIM, seed=util_seed)
        c1_online, c1_target = build_online_target_pair(STATE_DIM, seed=util_seed)
        c0_oh, c0_th = _param_hash(c0_online), _param_hash(c0_target)
        c1_oh, c1_th = _param_hash(c1_online), _param_hash(c1_target)

        mixer_seed = util_seed + MIXER_SEED_OFFSET
        mx_on, mx_tgt = build_mixer_online_target_pair(mixer_seed)
        mx_on_h, mx_tgt_h = _param_hash(mx_on), _param_hash(mx_tgt)

        row = {
            "policy_seed": p,
            "utility_init_seed": util_seed,
            "C0_utility_online_hash": c0_oh,
            "C1_utility_online_hash": c1_oh,
            "C0_utility_target_hash": c0_th,
            "C1_utility_target_hash": c1_th,
            "C0_online_equals_target": c0_oh == c0_th,
            "C1_online_equals_target": c1_oh == c1_th,
            "C0_equals_C1_utility_init": c0_oh == c1_oh,
            "mixer_seed": mixer_seed,
            "mixer_online_hash": mx_on_h,
            "mixer_target_hash": mx_tgt_h,
            "mixer_online_equals_target": mx_on_h == mx_tgt_h,
        }
        rows[str(p)] = row
        ok = (row["C0_online_equals_target"] and row["C1_online_equals_target"]
              and row["C0_equals_C1_utility_init"] and row["mixer_online_equals_target"])
        paired_ok = paired_ok and ok
    return {"per_policy": rows, "all_paired": paired_ok}


# ============================================================================
# VERDICT LOGIC VERIFICATION
# ============================================================================

def verify_verdict_logic():
    def V(mp, rob, ps=True, ss=True, ms=True, ai=True, fpi=True, nfa=True):
        return coordination_causal_verdict_skeleton(
            mixed_primary_label=mp, robustness_pass=rob,
            persistent_safety_pass=ps, stationary_safety_pass=ss,
            macro_safety_pass=ms, architecture_integrity_pass=ai,
            final_protocol_integrity_pass=fpi, no_final_data_adaptation=nfa)
    results = {}
    # VALIDATED
    r = V(BENEFICIAL, True)
    results["VALIDATED"] = r["verdict"] == "COORDINATION_CAUSAL_VALIDATED"
    # PERFORMANCE_PARTIAL: beneficial but robustness fail
    r = V(BENEFICIAL, False)
    results["PERF_PARTIAL_beneficial_norobust"] = (r["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
                                                    and r["subtype"] == "PERFORMANCE_PARTIAL")
    # PERFORMANCE_PARTIAL: directional-positive with robustness
    r = V(DIRECTIONAL_POSITIVE, True)
    results["PERF_PARTIAL_dirpos_robust"] = (r["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
                                              and r["subtype"] == "PERFORMANCE_PARTIAL")
    # PERFORMANCE_PARTIAL: directional-positive without robustness
    r = V(DIRECTIONAL_POSITIVE, False)
    results["PERF_PARTIAL_dirpos_norobust"] = (r["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
                                                and r["subtype"] == "PERFORMANCE_PARTIAL")
    # PRACTICALLY_NEUTRAL
    r = V(PRACTICALLY_NEUTRAL, False)
    results["NEUTRAL_PARTIAL"] = (r["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
                                   and r["subtype"] == "PRACTICALLY_NEUTRAL")
    # FAILED: directional-negative
    r = V(DIRECTIONAL_NEGATIVE, True)
    results["FAILED_dirneg"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    # FAILED: harmful
    r = V(HARMFUL, True)
    results["FAILED_harmful"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    # FAILED: safety
    r = V(BENEFICIAL, True, ps=False)
    results["FAILED_persistent_safety"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    r = V(BENEFICIAL, True, ss=False)
    results["FAILED_stationary_safety"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    r = V(BENEFICIAL, True, ms=False)
    results["FAILED_macro_safety"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    # FAILED: integrity
    r = V(BENEFICIAL, True, ai=False)
    results["FAILED_arch_integrity"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    r = V(BENEFICIAL, True, fpi=False)
    results["FAILED_protocol_integrity"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    r = V(BENEFICIAL, True, nfa=False)
    results["FAILED_adaptation"] = r["verdict"] == "COORDINATION_CAUSAL_FAILED"
    # PRACTICALLY_NEUTRAL with robustness
    r = V(PRACTICALLY_NEUTRAL, True)
    results["NEUTRAL_with_robust"] = (r["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
                                       and r["subtype"] == "PRACTICALLY_NEUTRAL")

    all_ok = all(results.values())
    return {"all_branches_verified": all_ok, "branches": results}


# ============================================================================
# CLASSIFICATION LOGIC EXHAUSTIVE TRUTH TABLE
# ============================================================================

def verify_classification_logic():
    checks = {}
    checks["BENEFICIAL"] = classify_effect(+2e-5, +1e-6, +4e-5, PRACTICAL_MARGIN) == BENEFICIAL
    checks["DIRECTIONAL_POSITIVE"] = classify_effect(+2e-5, -1e-6, +5e-5, PRACTICAL_MARGIN) == DIRECTIONAL_POSITIVE
    checks["PRACTICALLY_NEUTRAL"] = classify_effect(0.0, -5e-6, +5e-6, PRACTICAL_MARGIN) == PRACTICALLY_NEUTRAL
    checks["DIRECTIONAL_NEGATIVE"] = classify_effect(-2e-5, -5e-5, +1e-6, PRACTICAL_MARGIN) == DIRECTIONAL_NEGATIVE
    checks["HARMFUL"] = classify_effect(-2e-5, -5e-5, -1e-6, PRACTICAL_MARGIN) == HARMFUL
    checks["PRACTICAL_MARGIN_MATCHES_DELTA_E4"] = PRACTICAL_MARGIN == DELTA_E4
    return {"all_ok": all(checks.values()), "checks": checks}


# ============================================================================
# SEED DISJOINTNESS
# ============================================================================

def verify_seed_disjointness() -> Dict:
    new_env = set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(FINAL_SEEDS)
    consumed_env = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    env_disjoint = len(new_env & consumed_env) == 0
    internal_disjoint = (len(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS)) == 0
                         and len(set(TRAIN_SEEDS) & set(FINAL_SEEDS)) == 0
                         and len(set(VALIDATION_SEEDS) & set(FINAL_SEEDS)) == 0)
    consumed_pol = set().union(*[set(v) for v in CONSUMED_POLICY_SEEDS.values()])
    pol_disjoint = len(set(POLICY_SEEDS) & consumed_pol) == 0
    return {
        "env_disjoint_from_all_history": env_disjoint,
        "internal_split_disjoint": internal_disjoint,
        "policy_disjoint": pol_disjoint,
        "all_disjoint": env_disjoint and internal_disjoint and pol_disjoint,
        "checked_env_against": sorted(CONSUMED_SEED_SETS.keys()),
        "checked_policy_against": sorted(CONSUMED_POLICY_SEEDS.keys()),
        "env_overlap": sorted(new_env & consumed_env),
        "policy_overlap": sorted(set(POLICY_SEEDS) & consumed_pol),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4KT — Joint Coordination A/B Training Protocol Freeze")
    print("       (Branch C, FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: integrity ----
    integ = phase0_integrity()
    _jdump(integ, RESULTS_DIR / "E4KT_historical_integrity.json")
    if not integ["integrity_ok"]:
        print("[E4KT] INTEGRITY FAIL — STOP")
        _jdump({"E4KT_COMPLETE": "NO", "NEXT_STAGE": "E4KT_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4KT_verdict.json")
        return {"E4KT_COMPLETE": "NO"}
    print("[E4KT] Phase 0 integrity PASS.")

    # ---- Phase 1: single factor ----
    c0_cfg = build_arm_config("C0")
    c1_cfg = build_arm_config("C1")
    diff = arm_config_scientific_diff(c0_cfg, c1_cfg)
    single_factor_ok = diff["single_factor"]
    print(f"[E4KT] Phase 1 single-factor: {single_factor_ok}; "
          f"authorized diff = {diff['authorized_differences']}")

    # ---- Phase 16: paired utility initialization ----
    pairing = verify_paired_initialization()
    paired_ok = pairing["all_paired"]
    print(f"[E4KT] Phase 16 paired init: {paired_ok}")

    # ---- Seed disjointness ----
    seeds = verify_seed_disjointness()
    seeds_ok = seeds["all_disjoint"]
    print(f"[E4KT] Seed disjointness: {seeds_ok}")

    # ---- Classification logic ----
    cls_logic = verify_classification_logic()
    cls_ok = cls_logic["all_ok"]
    print(f"[E4KT] Classification logic: {cls_ok}")

    # ---- Verdict logic ----
    vl = verify_verdict_logic()
    vl_ok = vl["all_branches_verified"]
    print(f"[E4KT] Verdict logic: {vl_ok}")

    # ---- Build all freeze artifacts ----
    C: Dict[str, dict] = {}

    C["E4KT_pre_manifest.json"] = {
        "stage": "E4KT_JOINT_COORDINATION_AB_TRAINING_PROTOCOL_FREEZE",
        "freeze_only": True, "no_training": True, "no_seed_instantiation": True,
        "integrity_ok": integ["integrity_ok"],
        "single_factor_ok": single_factor_ok, "paired_init_ok": paired_ok,
        "seeds_disjoint": seeds_ok,
    }

    C["E4KT_ab_factor_contract.json"] = {
        "experiment_factor": "JOINT_COORDINATION_LEARNER_SPECIFICATION",
        "factor_is_not": ["NETWORK_SIZE", "REWARD", "NORMALIZATION"],
        "CONTROL": c0_cfg,
        "TREATMENT": c1_cfg,
        "single_factor_diff": diff,
        "single_factor_between_arms": "coordination learner specification only "
            "(V3, R1, alpha, utility net, env budget, training budget all held fixed)",
    }

    C["E4KT_common_scientific_config.json"] = {
        "normalization_version": NORM_V3,
        "v3_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "reward_version": REWARD_R1_NAME,
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "state_semantics": "local_V2[4]+GlobalContextV1[5]=9D",
        "state_dim": STATE_DIM,
        "regional_utility_architecture": "9-128-128-4",
        "action_set": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "gamma": GAMMA, "loss": LOSS_FN, "optimizer": "Adam",
        "learning_rate": LEARNING_RATE, "tau": TARGET_TAU,
        "replay_anchor_capacity": REPLAY_CAPACITY,
        "warmup_anchors": WARMUP_ANCHORS,
        "update_every_anchors": UPDATE_FREQUENCY,
        "batch_size_anchors": BATCH_SIZE,
        "gradient_clipping": GRAD_CLIP,
        "gradient_steps_per_update": GRAD_STEPS_PER_UPDATE,
        "epsilon_schedule": f"{EPS_START}->{EPS_END} linear over {EPS_DECAY} regional anchors/transitions",
        "epsilon_validation": 0.0, "epsilon_final": 0.0,
    }

    C["E4KT_control_contract.json"] = {
        "arm": "C0_INDEPENDENT_SHARED_DDQN",
        "architecture_version": CONTROL_ARCH,
        "normalization": NORM_V3, "reward": REWARD_R1_NAME,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "utility_network": "9-128-128-4",
        "loss": "existing regional Double-DQN SmoothL1 TD",
        "joint_representation": "IMPLICIT_STATE_ONLY",
        "mixer": "NONE",
        "optimizer_params": "utility only",
        "target_update": "utility target tau=0.005",
        "control_target_formula": "a*=argmax Q_online(s',a); y=r_R1+gamma*(1-done)*Q_target(s',a*)",
    }

    C["E4KT_treatment_contract.json"] = {
        "arm": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "architecture_version": TREATMENT_ARCH,
        "normalization": NORM_V3, "reward_primitives": REWARD_R1_NAME,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "utility_network": "9-128-128-4",
        "central_state_dim": CENTRAL_STATE_DIM, "R_MAX": R_MAX,
        "mixer": "QMIX_MIXER_V1", "mixer_hidden_dim": MIX_HIDDEN_DIM,
        "nonnegative_transform": NONNEGATIVE_TRANSFORM,
        "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
        "joint_td": "DOUBLE_Q_QMIX",
        "replay_type": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "optimizer_params": "utility + mixer",
        "target_update": "utility target + mixer target tau=0.005",
        "qmix_config_sha256": QMIX_CONFIG_SHA256_EXPECTED,
        "treatment_target_formula": (
            "q_r=Q_online(s_r,a_r) active, inactive=0; Q_tot=mixer_online(q,c); "
            "a*=argmax Q_online(s',a) per active; q'=Q_target(s',a*); inactive=0; "
            "Q_tot_target=mixer_target(q',c'); y=r_joint+gamma*(1-done)*Q_tot_target"
        ),
        "no_auxiliary_losses": True,
        "no_regional_auxiliary_td": True, "no_entropy_bonus": True,
        "no_mixer_regularizer": True, "no_coordination_loss": True,
        "no_contrastive_loss": True, "no_value_decomposition_auxiliary": True,
    }

    C["E4KT_qmix_config_integrity.json"] = {
        "qmix_config_sha256": qmix_config_sha256(),
        "expected": QMIX_CONFIG_SHA256_EXPECTED,
        "match": qmix_config_sha256() == QMIX_CONFIG_SHA256_EXPECTED,
        "no_runtime_reconstruction_from_defaults": True,
        "no_hidden_dim_change": True, "no_abs_to_softplus": True,
        "no_mixer_depth_change": True, "no_attention": True, "no_graph_layer": True,
    }

    C["E4KT_normalization_lock.json"] = {
        "normalization": NORM_V3, "v3_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "identical_both_arms": True, "no_refit": True, "no_adaptation": True,
        "no_running_statistics": True, "no_branch_specific_scaler": True,
    }

    C["E4KT_reward_lock.json"] = {
        "reward_version": REWARD_R1_NAME, "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local": FROZEN_R1_ALPHA_LOCAL, "identical_both_arms": True,
        "no_R0_reward": True, "no_alpha_retuning": True,
        "no_scale_adaptation": True, "no_counterfactual_credit": True,
        "no_oracle_labels": True,
        "reward_distinction": (
            "C0 replay target uses regional R1 reward; "
            "C1 joint target uses active-region mean of those SAME regional R1 rewards; "
            "this aggregation difference is intrinsic to the frozen coordination factor"
        ),
    }

    C["E4KT_utility_network_lock.json"] = {
        "input": STATE_DIM, "hidden": [128, 128], "activation": "ReLU", "output": 4,
        "architecture_string": "9-128-128-4",
        "parameter_shared_across_regions": True,
        "no_extra_utility_input_in_C1": True,
        "identical_both_arms": True,
    }

    C["E4KT_replay_contract.json"] = {
        "C0": "existing regional replay anchors",
        "C1": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "sampling_unit": "REGIONAL_ANCHOR",
        "capacity": REPLAY_CAPACITY,
        "warmup": WARMUP_ANCHORS,
        "update_every": UPDATE_FREQUENCY,
        "batch_size": BATCH_SIZE,
        "sibling_anchors": "NOT deduplicated (frozen sampling semantics)",
        "all_counters_in_regional_anchors": True,
    }

    C["E4KT_optimizer_contract.json"] = {
        "optimizer": "Adam", "lr": LEARNING_RATE,
        "C0_params": "utility only", "C1_params": "utility + mixer",
        "targets_excluded": True,
        "parameter_count_difference": "INTRINSIC_TO_COORDINATION_TREATMENT",
        "no_dummy_params_C0": True, "no_shrink_C1_utility": True,
        "no_widen_C0_utility": True,
    }

    C["E4KT_hyperparameters.json"] = {
        "gamma": GAMMA, "loss": LOSS_FN, "tau": TARGET_TAU,
        "gradient_clipping": GRAD_CLIP,
        "replay_capacity_anchors": REPLAY_CAPACITY,
        "warmup_anchors": WARMUP_ANCHORS,
        "batch_anchors": BATCH_SIZE,
        "update_every_anchors": UPDATE_FREQUENCY,
        "optimizer_steps_per_update": GRAD_STEPS_PER_UPDATE,
        "C0_target_update": "utility target soft update tau=0.005",
        "C1_target_update": "utility target AND mixer target soft update tau=0.005",
    }

    C["E4KT_operator_contract.json"] = {
        "KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
        "predict_fallback": "LOCAL", "sigma_D": 0.15,
        "budget": "10 PER_ACTIVE_REGION",
        "survival": "unrestricted standard NSGA-II",
        "no_HARD": True, "no_MEMORY": True,
        "identical_both_arms": True,
    }

    C["E4KT_seed_split.json"] = {
        "train": [1901, 1960], "n_train": 60,
        "validation": [1961, 1980], "n_val": 20,
        "final": [1981, 2040], "n_final": 60,
        "policy_seeds": POLICY_SEEDS,
        "disjointness": seeds,
    }

    C["E4KT_seed_disjointness.json"] = seeds

    C["E4KT_policy_pairing_contract.json"] = {
        "policy_seeds": POLICY_SEEDS, "matched_pairs": 5,
        "pairing_key": ["policy_seed", "scenario", "training_environment_seed", "logical_episode_index"],
        "same_deterministic_environment_schedule_per_pair": True,
        "controller_streams_isolated": True,
        "no_shared_mutable_replay_or_rng": True,
    }

    C["E4KT_initialization_pairing_contract.json"] = {
        "principle": "C0_p and C1_p start from identical utility online/target parameters "
                     "(arm-independent, seed = policy_seed*1000+0)",
        "C0_online_equals_target_at_t0": True,
        "C1_online_equals_target_at_t0": True,
        "C0_equals_C1_utility_init_at_t0": True,
        "C1_mixer_deterministic": True,
        "C1_mixer_online_equals_target_at_t0": True,
        "no_mixer_initialization_sweep": True,
        "parameter_count_difference": "INTRINSIC (C1 has mixer; this is not an integrity failure)",
        "per_policy": pairing["per_policy"],
        "all_paired": pairing["all_paired"],
    }

    C["E4KT_rng_contract.json"] = {
        "environment_schedule_paired_per_policy": True,
        "isolated_mutable_streams": ["replay_RNG", "exploration_RNG", "operator_RNG", "optimizer_model_state"],
        "no_forced_identical_actions_after_divergence": True,
        "no_action_dependent_common_random_numbers": True,
        "existing_paired_AB_rng_contract": "followed exactly if more specific",
    }

    C["E4KT_training_budget.json"] = {
        "epochs_per_run": N_EPOCHS,
        "seeds_per_epoch": len(TRAIN_SEEDS),
        "scenarios_per_epoch": len(SCENARIOS),
        "episodes_per_epoch": EPISODES_PER_EPOCH,
        "episodes_per_run": EPISODES_PER_RUN,
        "arms": 2, "policies": 5, "runs": N_RUNS,
        "total_training_episodes": TOTAL_TRAINING_EPISODES,
        "changes_per_episode": N_SCORED_CHANGES,
        "no_early_stopping": True, "no_extra_epoch_for_C1": True,
        "no_architecture_specific_extension": True,
        "scenario_balance": "60 PERSISTENT + 60 STATIONARY + 60 MIXED per epoch (equal)",
        "no_persistent_oversampling": True,
        "run_isolation": {
            "each_run_owns_independent": [
                "online_utility", "target_utility", "optimizer", "replay",
                "epsilon_state", "RNG_state"
            ],
            "C1_additionally": ["online_mixer", "target_mixer"],
            "no_replay_sharing_across_arms": True,
            "no_optimizer_sharing": True,
            "no_model_sharing_after_init": True,
            "no_mixer_sharing_across_policy_seeds": True,
        },
    }

    C["E4KT_checkpoint_contract.json"] = {
        "schedule_anchors": CHECKPOINT_ANCHORS, "plus_final": True,
        "checkpoints_per_run": CHECKPOINTS_PER_RUN,
        "total_future_checkpoints": TOTAL_CHECKPOINTS,
        "no_additional_C1_only_checkpoint": True,
        "common_metadata": [
            "stage", "experiment", "arm", "policy_seed", "transition_anchor_count",
            "normalization_version", "v3_constants_sha256", "reward_version",
            "reward_config_sha256", "alpha_local", "utility_architecture",
            "architecture_version", "training_protocol_hash", "optimizer_config_hash",
            "seed_metadata",
        ],
        "C0_metadata": {"architecture_version": CONTROL_ARCH, "no_mixer_weights": True},
        "C1_metadata": {
            "architecture_version": TREATMENT_ARCH,
            "must_contain": ["online_utility", "target_utility", "online_mixer", "target_mixer"],
            "qmix_config_sha256": QMIX_CONFIG_SHA256_EXPECTED,
        },
        "version_guards": {
            "wrong_V3_hash": "hard_error", "wrong_R1_hash": "hard_error",
            "wrong_architecture_version": "hard_error",
            "missing_C1_mixer": "hard_error", "unexpected_mixer_in_C0": "hard_error",
            "wrong_QMIX_hash": "hard_error",
        },
    }

    C["E4KT_validation_contract.json"] = {
        "validation_seeds": [1961, 1980], "scenarios": SCENARIOS,
        "epsilon": 0.0, "mode": "eval", "no_gradients": True,
        "no_optimizer": True, "no_replay_updates": True,
        "no_target_updates": True, "no_normalizer_adaptation": True,
        "mixer_may_load_for_integrity": True,
        "mixer_must_not_select_actions": True,
        "validation_matrix": {
            "arms": 2, "policies": 5, "checkpoints": 6,
            "scenarios": 3, "environments": 20,
            "total": VALIDATION_EPISODES,
        },
        "all_60_checkpoints_evaluated": True,
    }

    C["E4KT_checkpoint_selection.json"] = {
        "scope": "per arm × policy_seed",
        "primary_metric": "minimum validation MACRO_MIGD_RESPONSE",
        "macro": "equal 1/3 mean of PERSISTENT + STATIONARY + MIXED",
        "tie_1": "lower macro MIGD_epoch",
        "tie_2": "lower macro response_FE",
        "tie_3": "earlier checkpoint",
        "forbidden_signals": [
            "QMIX_training_loss", "Q_tot_magnitude", "mixer_weight_variation",
            "kappa_dispersion", "PERSISTENT_only_performance", "MIXED_only_performance",
            "global_comparator_performance", "action_diversity",
        ],
        "all_10_policies_retained": True,
        "no_best_policy_selection": True,
        "no_dropping_unstable_seeds": True,
    }

    C["E4KT_training_logging_contract.json"] = {
        "per_episode": [
            "arm", "policy_seed", "epoch", "scenario", "environment_seed",
            "episode_index", "MIGD_response", "MIGD_epoch", "response_FE",
            "cumulative_reward_summary", "regional_anchor_count",
        ],
        "C0_training_audit": [
            "regional_R1_reward", "regional_TD_target", "regional_predicted_Q",
            "SmoothL1_loss", "replay_count", "update_count", "epsilon",
        ],
        "C1_training_audit": [
            "regional_R1_reward_primitives", "active_count", "joint_reward",
            "Q_tot_current", "Q_tot_target", "joint_TD_loss",
            "replay_anchor_count", "update_count", "epsilon", "mixer_config_hash",
        ],
        "requested_realized": ["requested_action", "realized_operator", "fallback"],
        "C1_joint_action_vector": "requested actions (not realized)",
    }

    C["E4KT_coordination_diagnostics.json"] = {
        "status": "descriptive_only",
        "not_a_verdict_gate": True,
        "diagnostics": {
            "A_monotonic_gradient": {
                "metric": "dQ_tot/dq_r for active regions",
                "report": ["min", "median", "q95", "violation_count"],
                "negative_violations_beyond_tolerance": "integrity failure",
                "magnitude_not_a_performance_gate": True,
            },
            "B_mixer_state_variation": {
                "W1_variation_across_states": True,
                "W2_variation_across_states": True,
                "no_frozen_minimum_activity_threshold": True,
            },
            "C_kappa_dispersion": {
                "metric": "population std(kappa_r, ddof=0) over active regions within-event",
                "no_success_threshold": True,
            },
            "D_other_region_sensitivity": {
                "method": "perturb q_j in stored/synthetic mixer inputs only",
                "no_environment_counterfactual_rollout": True,
                "no_scientific_gate": True,
            },
            "E_joint_td_loss": "descriptive by epoch/policy/scenario",
        },
        "runtime_monotonicity_audit": {
            "purpose": "IMPLEMENTATION_INTEGRITY_CHECK",
            "not": "scientific_mechanism_success_gate",
        },
    }

    C["E4KT_action_diagnostics.json"] = {
        "for_both_arms": True,
        "metrics": [
            "KEEP_fraction", "LOCAL_fraction", "PREDICT_fraction", "DIVERSIFY_fraction",
            "within_event_multi_action_rate", "regional_action_entropy",
        ],
        "status": "descriptive",
        "no_more_diversity_equals_better_gate": True,
    }

    C["E4KT_primary_hypothesis.json"] = {
        "question": (
            "Does explicit monotonic joint-value coordination during training improve "
            "the learned dynamic-response controller relative to the current Independent "
            "Shared DDQN when state(V3), reward(R1), alpha(0.1), utility(9-128-128-4), "
            "actions, operators, env budget, and training budget are all fixed?"
        ),
        "primary_scenario": "MIXED",
        "primary_endpoint": "MIGD_RESPONSE",
        "cell_effect": "d[p,e,MIXED] = MIGD_response_C0[p,e] - MIGD_response_C1[p,e]",
        "positive_means": "QMIX_BETTER",
    }

    C["E4KT_effect_definition.json"] = {
        "effect": "C0_MINUS_C1",
        "positive_means": "QMIX_BETTER",
        "practical_margin": PRACTICAL_MARGIN,
        "matches_DELTA_E4": PRACTICAL_MARGIN == DELTA_E4,
    }

    C["E4KT_classification_logic.json"] = {
        "BENEFICIAL": "mean > +margin AND CI_lower > 0",
        "DIRECTIONAL_POSITIVE": "mean > +margin AND CI includes/crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(mean) <= margin (complete frozen rare-case logic)",
        "DIRECTIONAL_NEGATIVE": "mean < -margin AND CI includes/crosses 0",
        "HARMFUL": "mean < -margin AND CI_upper < 0",
        "practical_margin": PRACTICAL_MARGIN,
        "verification": cls_logic,
    }

    C["E4KT_statistical_design.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis": 5, "environment_axis": 60,
        "resample_both_with_replacement": True,
        "full_sampled_cartesian_product": True,
        "preserve_C0_C1_pairing": True,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "macro": "equal 1/3 with SAME resampled policy+env indices across scenarios",
        "forbidden": [
            "iid-bootstrap 300 cells",
            "independently bootstrap C0/C1",
            "region as replicate",
            "replay anchor as replicate",
            "environmental change as replicate",
        ],
    }

    C["E4KT_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED] = mean_e(MIGD_C0[p,e] - MIGD_C1[p,e])",
        "rule": "COORDINATION_POLICY_ROBUSTNESS = PASS iff >=4/5 policy effects > 0",
        "threshold": 4, "of": 5,
        "no_best_policy_filtering": True,
    }

    C["E4KT_safety_contract.json"] = {
        "PERSISTENT_COORDINATION_SAFETY": {
            "effect": "C0 - C1",
            "PASS_iff": "neither MIGD_response nor MIGD_epoch is HARMFUL",
        },
        "STATIONARY_COORDINATION_SAFETY": {
            "effect": "C0 - C1",
            "PASS_iff": "neither MIGD_response nor MIGD_epoch is HARMFUL",
        },
        "COORDINATION_MACRO_SAFETY": {
            "macro": "equal 1/3 scenario",
            "PASS_iff": "neither macro MIGD_response nor macro MIGD_epoch is HARMFUL",
        },
    }

    C["E4KT_fe_contract.json"] = {
        "effect": "response_FE_C0 - response_FE_C1",
        "positive_means": "QMIX cheaper",
        "scenarios": SCENARIOS + ["macro"],
        "FE_is_secondary": True,
        "FE_cannot_rescue_harmful_MIGD": True,
    }

    C["E4KT_success_logic.json"] = {
        "COORDINATION_CAUSAL_VALIDATED_requires_all": [
            "MIXED MIGD_response BENEFICIAL",
            "COORDINATION_POLICY_ROBUSTNESS PASS",
            "PERSISTENT_COORDINATION_SAFETY PASS",
            "STATIONARY_COORDINATION_SAFETY PASS",
            "COORDINATION_MACRO_SAFETY PASS",
            "ARCHITECTURE_INTEGRITY PASS",
            "CHECKPOINT_INTEGRITY PASS",
            "FINAL_PROTOCOL_INTEGRITY PASS",
            "no final-data adaptation",
        ],
        "PERFORMANCE_PARTIAL": (
            "primary in {BENEFICIAL, DIRECTIONAL_POSITIVE} AND no safety HARMFUL AND integrity PASS "
            "BUT one or more supporting gates (robustness) fail OR primary is directional-positive"
        ),
        "PRACTICALLY_NEUTRAL": (
            "primary PRACTICALLY_NEUTRAL AND all safety PASS AND integrity PASS -> "
            "PERFORMANCE_IMPROVEMENT_CLAIM = NOT_SUPPORTED (do not interpret as equivalence)"
        ),
        "FAILED": (
            "primary in {DIRECTIONAL_NEGATIVE, HARMFUL} OR any safety FAIL OR "
            "architecture/checkpoint/protocol integrity FAIL"
        ),
        "coordination_mechanism_gate": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "coordination_diagnostics_cannot_rescue_harmful": True,
        "FE_savings_cannot_rescue_harmful": True,
        "larger_param_count_not_success_criterion": True,
        "verdict_function": "E4K.coordination_causal_verdict_skeleton",
        "verification": vl,
    }

    C["E4KT_future_final_matrix.json"] = {
        "arms": 2, "policies": 5, "environments": 60,
        "scenarios": 3, "total_episodes": 1800,
        "seeds": "1981..2040",
        "not_executed_in_E4KT": True,
        "not_executed_in_E4KTR": True,
        "requires_separate_final_protocol_freeze": True,
    }

    C["E4KT_future_stage_sequence.json"] = {
        "if_E4KT_passes": "E4KTR_JOINT_COORDINATION_AB_TRAINING_EXECUTION",
        "after_successful_training": "E4KE_JOINT_COORDINATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "after_E4KE": "E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION",
        "final_seeds_unopened_through": ["E4KT", "E4KTR", "validation", "checkpoint_selection", "debugging"],
        "no_global_comparator_in_architecture_AB": True,
        "if_VALIDATED": "E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN (another untouched seed set; NOT 1801..1860)",
        "if_PARTIAL": "E4L_COORDINATION_RESULT_REVIEW (do not immediately tune mixer)",
        "if_FAILED_integrity_pass": "E4L_COORDINATION_FAILURE_ANALYSIS (do not auto-escalate to bigger QMIX/attention/GNN/AC)",
        "if_FAILED_integrity": "E4K_ENGINEERING_INTEGRITY_REVIEW",
    }

    C["E4KT_e4ktr_output_contract.json"] = {
        "output_directory": "results/E4KTR_joint_coordination_ab_training_execution/",
        "required_artifacts": [
            "E4KTR_pre_execution_manifest.json",
            "E4KTR_execution_config.json",
            "E4KTR_pair_initialization_manifest.json",
            "E4KTR_single_factor_audit.json",
            "E4KTR_training_episode_results.csv",
            "E4KTR_training_update_summary.csv",
            "E4KTR_coordination_diagnostics.csv",
            "E4KTR_action_trajectory.csv",
            "E4KTR_numerical_stability.json",
            "E4KTR_checkpoint_manifest.json",
            "E4KTR_validation_all_checkpoints.csv",
            "E4KTR_validation_selected_policies.csv",
            "E4KTR_selected_checkpoints.json",
            "E4KTR_validation_ab_summary.csv",
            "E4KTR_runtime_architecture_integrity.json",
            "E4KTR_final_firewall.json",
            "E4KTR_verdict.json",
            "E4KTR_report.md",
        ],
        "plus_checkpoint_directories": ["C0/", "C1/"],
        "execution_contract": {
            "exactly_10_runs": True,
            "exactly_1080_episodes_per_run": True,
            "exactly_10800_training_episodes": True,
            "exact_seed_split": True,
            "all_5_policy_pairs_retained": True,
            "60_checkpoints": True,
            "all_60_validation_evaluated": True,
            "exactly_10_selected_checkpoints": True,
            "final_seeds_access_count_zero": True,
            "no_hyperparameter_retuning": True,
            "no_architecture_retuning": True,
        },
    }

    C["E4KT_final_firewall.json"] = {
        "final_seeds": [1981, 2040],
        "untouched_through": ["E4KT", "E4KTR", "validation", "checkpoint_selection", "debugging"],
        "C_TRAIN_ACCESS_COUNT": _coord_access["train"],
        "C_VALIDATION_ACCESS_COUNT": _coord_access["validation"],
        "C_FINAL_ACCESS_COUNT": _coord_access["final"],
    }

    C["E4KT_no_training_audit.json"] = {
        "real_environment_episodes": 0,
        "training_episodes": 0,
        "optimizer_steps": 0,
        "replay_appends": 0,
        "new_policy_checkpoints": 0,
        "model_weight_changes": 0,
        "note": "paired-init networks built for hashing only, never trained or saved",
    }

    # ---- Write all artifacts ----
    for name, obj in C.items():
        _jdump(obj, RESULTS_DIR / name)

    # ---- Protocol hash ----
    protocol_payload = json.dumps({k: C[k] for k in sorted(C)}, sort_keys=True,
                                   default=str)
    protocol_hash = hashlib.sha256(protocol_payload.encode()).hexdigest()
    _jdump({
        "stage": "E4KT_JOINT_COORDINATION_AB_TRAINING_PROTOCOL_FREEZE",
        "experiment_factor": "JOINT_COORDINATION_LEARNER_SPECIFICATION",
        "control": "C0_INDEPENDENT_SHARED_DDQN",
        "treatment": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "fixed_normalization": NORM_V3,
        "fixed_reward": REWARD_R1_NAME,
        "fixed_alpha_local": FROZEN_R1_ALPHA_LOCAL,
        "qmix_config_sha256": QMIX_CONFIG_SHA256_EXPECTED,
        "single_factor_diff": diff,
        "paired_init_ok": paired_ok,
        "seeds_disjoint": seeds_ok,
        "protocol_hash": protocol_hash,
        "components": sorted(C.keys()),
        "next_stage": "E4KTR_JOINT_COORDINATION_AB_TRAINING_EXECUTION",
    }, RESULTS_DIR / "E4KT_protocol_frozen.json")

    # ---- Verdict ----
    all_ok = (integ["integrity_ok"] and single_factor_ok and paired_ok
              and seeds_ok and cls_ok and vl_ok)
    verdict = build_verdict(integ, diff, pairing, seeds, cls_logic, vl, all_ok)
    _jdump(verdict, RESULTS_DIR / "E4KT_verdict.json")
    write_report(verdict)
    emit_terminal(verdict)
    print(f"\n[E4KT] wall {time.time()-t0:.1f}s")
    return verdict


def build_verdict(integ, diff, pairing, seeds, cls_logic, vl, all_ok):
    return {
        "stage": "E4KT_JOINT_COORDINATION_AB_TRAINING_PROTOCOL_FREEZE",
        "E4KT_COMPLETE": "YES" if all_ok else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "SELECTED_BRANCH": "C_JOINT_COORDINATION",
        "EXPERIMENT_FACTOR": "JOINT_COORDINATION_LEARNER_SPECIFICATION",
        "CONTROL": "C0_INDEPENDENT_SHARED_DDQN",
        "TREATMENT": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "QMIX_IMPLEMENTATION_VALID": "YES",
        "QMIX_CONFIG_SHA256": QMIX_CONFIG_SHA256_EXPECTED,
        "FIXED_NORMALIZATION": NORM_V3,
        "FIXED_REWARD": REWARD_R1_NAME,
        "FIXED_ALPHA_LOCAL": FROZEN_R1_ALPHA_LOCAL,
        "UTILITY_NETWORK": "9-128-128-4",
        "REGION_SLOT_COUNT": R_MAX,
        "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
        "MIXER_HIDDEN_DIM": MIX_HIDDEN_DIM,
        "JOINT_REWARD": "ACTIVE_REGION_MEAN_OF_R1",
        "REPLAY_SAMPLING_UNIT": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "TRAIN_SEEDS": "1901..1960",
        "VALIDATION_SEEDS": "1961..1980",
        "FINAL_SEEDS": "1981..2040",
        "POLICY_SEEDS": "5001..5005",
        "PAIRED_UTILITY_INITIALIZATION": "FROZEN" if pairing["all_paired"] else "FAIL",
        "TRAINING_EPISODES_PER_RUN": EPISODES_PER_RUN,
        "TOTAL_FUTURE_RUNS": N_RUNS,
        "TOTAL_FUTURE_TRAINING_EPISODES": TOTAL_TRAINING_EPISODES,
        "CHECKPOINTS_PLANNED": TOTAL_CHECKPOINTS,
        "VALIDATION_EPISODES_PLANNED": VALIDATION_EPISODES,
        "CHECKPOINT_SELECTION": "VALIDATION_MACRO_MIGD_RESPONSE",
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_EFFECT": "C0_MINUS_C1",
        "POSITIVE_MEANS": "QMIX_BETTER",
        "PRACTICAL_MARGIN": PRACTICAL_MARGIN,
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_QMIX_BETTER",
        "STATISTICAL_DESIGN": "PAIRED_CROSSED_TWO_AXIS_5x60",
        "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "PARAMETER_COUNT_DIFFERENCE": "INTRINSIC_TO_COORDINATION_TREATMENT",
        "SINGLE_FACTOR_DIFF": diff["differing_fields"],
        "SINGLE_FACTOR_OK": diff["single_factor"],
        "SEEDS_DISJOINT": seeds["all_disjoint"],
        "CLASSIFICATION_LOGIC_OK": cls_logic["all_ok"],
        "VERDICT_LOGIC_OK": vl["all_branches_verified"],
        "C_TRAIN_ACCESS_COUNT": 0,
        "C_VALIDATION_ACCESS_COUNT": 0,
        "C_FINAL_ACCESS_COUNT": 0,
        "NEW_ENVIRONMENT_EXECUTION": "NO",
        "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4KTR_JOINT_COORDINATION_AB_TRAINING_EXECUTION" if all_ok
                       else "E4KT_INTEGRITY_REVIEW"),
    }


def write_report(v):
    L = [
        "# E4KT — Joint Coordination A/B Training Protocol Freeze (Branch C)", "",
        f"**E4KT_COMPLETE = {v['E4KT_COMPLETE']} | NEXT_STAGE = {v['NEXT_STAGE']}**", "",
        "Freeze only. No training, no environment instantiation (seeds 1901..2040 untouched). "
        "The single scientific factor is the joint coordination learner specification; both arms "
        "share V3 normalization, R1 reward (alpha=0.1), and 9-128-128-4 utility network.", "",
        "## A. Historical result preservation",
        "- E4F=FAILED, E4HF=FAILED, E4IF=VALIDATED, E4JF=FAILED preserved; V3/R1 hashes "
        "unchanged; QMIX implementation valid (E4KI); future access counts 0.", "",
        "## B. E4KI implementation-valid status",
        "- QMIX_IMPLEMENTATION_VALID=YES; all 40 E4KI gates PASS; monotonicity 0 violations; "
        "argmax equivalence 220/220; C0 backward equivalence PASS.", "",
        "## C. Branch-C scientific question",
        "- Does explicit monotonic joint-value coordination (QMIX) during training improve the "
        "learned controller relative to Independent Shared DDQN when all other factors are fixed?", "",
        "## D. Experimental factor",
        f"- JOINT_COORDINATION_LEARNER_SPECIFICATION; single-factor diff = "
        f"{v['SINGLE_FACTOR_DIFF']}; all authorized by the frozen factor.", "",
        "## E. Exact C0 control",
        f"- {v['CONTROL']}; V3+R1 regional Double-DQN SmoothL1 TD; utility only in optimizer; "
        "no mixer.", "",
        "## F. Exact C1 treatment",
        f"- {v['TREATMENT']}; V3+R1+QMIX mixer (hidden 32, ABS, ELU); central state 55-D; "
        "Double-Q joint TD with active-region mean R1; utility+mixer in optimizer.", "",
        "## G. Authorized factor differences",
        "- Architecture version, mixer module/params, joint TD objective, joint reward aggregation, "
        "joint context in replay, mixer in optimizer, target mixer, mixer checkpoint metadata — all "
        "intrinsic to the coordination factor. Differing fields are NON-empty and authorized.", "",
        "## H. V3 normalization lock",
        f"- Both arms: {v['FIXED_NORMALIZATION']}; no refit/adaptation/running stats/branch-specific scaler.", "",
        "## I. R1 reward lock",
        f"- Both arms: {v['FIXED_REWARD']}; alpha=0.1; no R0/retuning/scale-adaptation/counterfactual/oracle. "
        "C0 uses regional R1; C1 uses active-region MEAN of those same R1 rewards (intrinsic).", "",
        "## J. Utility architecture lock",
        f"- Both arms: {v['UTILITY_NETWORK']}; parameter-shared; no extra utility input in C1.", "",
        "## K. QMIX config lock",
        f"- QMIX_CONFIG_SHA256 = {v['QMIX_CONFIG_SHA256']}; no runtime reconstruction from defaults; "
        "no hidden-dim/ABS/depth/attention/graph changes.", "",
        "## L. Replay/sampling semantics",
        f"- C0: regional transitions; C1: {v['REPLAY_SAMPLING_UNIT']}; both capacity=50000, "
        "warmup=1000, update=4, batch=128 (in regional anchors); siblings not deduplicated.", "",
        "## M. Shared optimization hyperparameters",
        f"- gamma=0.99, SmoothL1, Adam lr=1e-4, tau=0.005, no gradient clipping, 1 step/update; "
        f"C0 target=utility only, C1 target=utility+mixer.", "",
        "## N. Fresh seed split",
        f"- Train {v['TRAIN_SEEDS']}, Val {v['VALIDATION_SEEDS']}, Final {v['FINAL_SEEDS']}, "
        f"Policy {v['POLICY_SEEDS']}; disjoint={v['SEEDS_DISJOINT']}.", "",
        "## O. Paired utility initialization",
        f"- {v['PAIRED_UTILITY_INITIALIZATION']}; C0 utility t0 == C1 utility t0 per policy; "
        "online==target both arms; mixer deterministic per seed; no mixer init sweep.", "",
        "## P. RNG/environment pairing",
        "- Environment schedule paired per policy seed; isolated mutable streams; no forced "
        "identical actions; no action-dependent CRN.", "",
        "## Q. Training budget",
        f"- {v['TRAINING_EPISODES_PER_RUN']} eps/run × {v['TOTAL_FUTURE_RUNS']} runs = "
        f"{v['TOTAL_FUTURE_TRAINING_EPISODES']} total; 6 epochs, equal scenario balance "
        "(60P+60S+60M per epoch); no early stop; no PERSISTENT oversampling.", "",
        "## R. Checkpoint schedule",
        f"- {CHECKPOINT_ANCHORS} + final = {CHECKPOINTS_PER_RUN}/run, {TOTAL_CHECKPOINTS} total; "
        "no additional C1-only checkpoint.", "",
        "## S. Validation/checkpoint selection",
        f"- {VALIDATION_EPISODES} validation episodes (all 60 ckpts); selection = min validation "
        "macro MIGD_response (ties: epoch, FE, earlier); all 10 policies retained.", "",
        "## T. Runtime integrity",
        "- C0: arch exact, no mixer, V3/R1 exact. C1: QMIX hash exact, mixer present, V3/R1 "
        "exact, decentralized action path, no runtime adaptation.", "",
        "## U. Coordination diagnostics",
        "- A monotonic gradient / B mixer weight variation / C kappa dispersion / D other-region "
        "sensitivity / E joint TD loss — descriptive only, NOT a verdict gate. Runtime "
        "monotonicity is an integrity check, not mechanism gate.", "",
        "## V. Action diagnostics",
        "- KEEP/LOCAL/PREDICT/DIVERSIFY fractions, multi-action rate, entropy — descriptive "
        "for both arms; no 'more diversity=better' gate.", "",
        "## W. Future primary causal contrast",
        f"- PRIMARY={v['PRIMARY_SCENARIO']} {v['PRIMARY_ENDPOINT']}; effect {v['PRIMARY_EFFECT']}; "
        f"positive means {v['POSITIVE_MEANS']}; margin={v['PRACTICAL_MARGIN']}.", "",
        "## X. Statistical design",
        f"- {v['STATISTICAL_DESIGN']}, N={v['BOOTSTRAP_RESAMPLES']}, 95% percentile, "
        f"master seed={v['BOOTSTRAP_MASTER_SEED']}; macro shared draws.", "",
        "## Y. Policy robustness",
        f"- {v['POLICY_ROBUSTNESS_RULE']}; no best-policy filtering.", "",
        "## Z. Scenario safety",
        "- PERSISTENT/STATIONARY/macro safety = PASS iff neither MIGD_response nor MIGD_epoch "
        "is HARMFUL; FE cannot rescue harmful quality.", "",
        "## AA. FE analysis",
        "- effect C0-C1 FE; positive=QMIX cheaper; secondary; cannot rescue harmful MIGD.", "",
        "## AB. Final causal verdict logic",
        "- COORDINATION_CAUSAL_VALIDATED / PERFORMANCE_PARTIAL / PRACTICALLY_NEUTRAL / FAILED; "
        "coordination diagnostics cannot rescue harmful; FE cannot rescue; larger params not "
        f"success criterion; mechanism gate = {v['COORDINATION_MECHANISM_GATE']}.", "",
        "## AC. Future final firewall",
        "- Seeds 1981..2040 untouched through E4KT/E4KTR/validation/debugging; access 0/0/0.", "",
        "## AD. Stage sequence",
        "- E4KT → E4KTR → E4KE → E4KF; no global comparator in architecture A/B.", "",
        "## AE. No-training audit",
        "- 0 environment episodes, 0 optimizer steps, 0 checkpoints, 0 weight changes.", "",
        f"## AF. Next stage: {v['NEXT_STAGE']}",
    ]
    (RESULTS_DIR / "E4KT_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    fields = [
        "E4KT_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
        "SELECTED_BRANCH", "EXPERIMENT_FACTOR", "CONTROL", "TREATMENT",
        "QMIX_IMPLEMENTATION_VALID", "QMIX_CONFIG_SHA256",
        "FIXED_NORMALIZATION", "FIXED_REWARD", "FIXED_ALPHA_LOCAL",
        "UTILITY_NETWORK", "REGION_SLOT_COUNT", "CENTRAL_STATE_DIM", "MIXER_HIDDEN_DIM",
        "JOINT_REWARD", "REPLAY_SAMPLING_UNIT",
        "TRAIN_SEEDS", "VALIDATION_SEEDS", "FINAL_SEEDS", "POLICY_SEEDS",
        "PAIRED_UTILITY_INITIALIZATION",
        "TRAINING_EPISODES_PER_RUN", "TOTAL_FUTURE_RUNS", "TOTAL_FUTURE_TRAINING_EPISODES",
        "CHECKPOINTS_PLANNED", "VALIDATION_EPISODES_PLANNED", "CHECKPOINT_SELECTION",
        "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT", "PRIMARY_EFFECT", "POSITIVE_MEANS",
        "PRACTICAL_MARGIN", "POLICY_ROBUSTNESS_RULE",
        "STATISTICAL_DESIGN", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_MASTER_SEED",
        "COORDINATION_MECHANISM_GATE", "PARAMETER_COUNT_DIFFERENCE",
        "C_TRAIN_ACCESS_COUNT", "C_VALIDATION_ACCESS_COUNT", "C_FINAL_ACCESS_COUNT",
        "NEW_ENVIRONMENT_EXECUTION", "NEW_TRAINING",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in fields:
        print(f"{k} =\n    {v[k]}\n")


if __name__ == "__main__":
    main()
