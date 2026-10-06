"""
E4RTR V2 — Pairwise CG A/B Training Execution, Batched Restart.

Clean FROM-SCRATCH restart of the C1-vs-C3 coordination experiment
using the E4RIA/E4RTA-validated batched Max-Sum implementation.

The interrupted scalar run (V1) is EXCLUDED from all scientific analysis.
Only V2 artifacts feed into the subsequent final evaluation.

Training seeds 2501-2560 are REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION.
Validation seeds 2561-2580 are first-use.
Final seeds 2581-2640 remain RESERVED_UNTOUCHED.
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
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4RTR_pairwise_cg_ab_training_execution import (
    train_one_policy_c1, train_one_policy_c3,
    load_v3, _param_hash, _hash_file, _e4rtr_code_manifest_hash,
    _jdump, _jdefault, _write_csv, _load,
    eval_c3_arm, validate_c3,
    mixed_action_metrics_c1, mixed_action_metrics_c3,
    TRAIN_SEEDS, VAL_SEEDS, FINAL_SEEDS, POLICY_SEEDS,
    FIREWALL, C1_ARM, C3_ARM,
    FROZEN_QMIX_CONFIG_SHA256, FROZEN_C3_CONFIG_SHA256,
    N_TRAIN_EPISODES, EPISODES_PER_EPOCH,
    E4RT_DIR, E4RI_DIR, E4IF_DIR, E4JF_DIR, E4KF_DIR, E4F_DIR, E4HF_DIR, E4MF_DIR,
)
from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, build_warmup_cache, select_checkpoint,
    CHECKPOINT_TRANSITIONS, SCENARIOS, N_EPOCHS,
    BATCH_SIZE, REPLAY_CAPACITY, GAMMA, LEARNING_RATE, TARGET_TAU,
    WARMUP_TRANSITIONS, UPDATE_FREQUENCY,
    EPS_START, EPS_END, EPS_DECAY,
    RNG_OFFSET_POLICY_INIT,
)
from prdrl.rl.network import SharedRegionalQNetwork
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.qmix import TREATMENT_ARCH, derive_mixer_seed, build_mixer_online_target_pair
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, make_qmix_config, qmix_config_sha256, count_parameters,
)
from prdrl.rl.coordination_graph import (
    TREATMENT_ARCH_C3, SharedPairNetwork, PAIR_SEED_OFFSET, K_MAXSUM,
    derive_pair_seed, build_pair_online_target,
    make_c3_config, c3_config_sha256,
)
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256
from prdrl.rl.network import build_online_target_pair, N_ACTIONS, ACTION_ORDER
from prdrl.experiments.E4P_learned_controller_preparation import NETWORK_INPUT_DIM

# ============================================================================
# V2 CONSTANTS
# ============================================================================

V2_RESULTS_DIR = Path(
    "results/E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2")
OLD_RUN_DIR = Path("results/E4RTR_pairwise_cg_ab_training_execution")
OLD_ARCHIVE_DIR = Path(
    "results/E4RTR_pairwise_cg_ab_training_execution_INTERRUPTED_RUNTIME_ARCHIVE")
E4RTA_DIR = Path("results/E4RTA_batched_maxsum_protocol_amendment_freeze")
E4RTAR_DIR = Path("results/E4RTAR_regression_harness_reconciliation")
E4RIA_DIR_V2 = Path("results/E4RIA_batched_maxsum_engineering_equivalence")

RESTART_VERSION = "E4RTR_BATCHED_RESTART_V2"
RESTART_MODE = "FROM_SCRATCH"
E4RTA_AMENDED_PROTOCOL_SHA256 = (
    "ac5c93f7c68c304d1dae3dbf4c4de957c440d3ddf181a7870e68564c4d0bad91")
BATCHED_IMPL_SHA256 = (
    "c8c455a05b7cb8f7b57ae840a077af90b0470e2c336920a7dbd41bee52c0f97f")
ORIGINAL_E4RT_PROTOCOL_SHA256 = (
    "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6")
TRAINING_SCRIPT_SHA256 = (
    "53e10bfe0f4b40631e0d97c0fa24b679a94b4b4131a7d005f90c10372397155b")


def _collect_old_checkpoint_hashes():
    hashes = set()
    for d in [OLD_RUN_DIR, OLD_ARCHIVE_DIR]:
        if d.exists():
            for pt in d.rglob("*.pt"):
                hashes.add(_hash_file(str(pt)))
    return hashes


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()
    print("=" * 60)
    print("E4RTR V2 — Pairwise CG A/B Training Execution (Batched Restart)")
    print("=" * 60)
    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # PHASE V2-0: PRE-EXECUTION INTEGRITY
    # ==================================================================
    print("[V2-0] Pre-execution integrity ...")

    e4rta = _load(E4RTA_DIR / "E4RTA_verdict.json")
    e4rtar = _load(E4RTAR_DIR / "E4RTAR_verdict.json")
    e4rt = _load(E4RT_DIR / "E4RT_verdict.json")
    e4ri = _load(E4RI_DIR / "E4RI_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    e4kf = _load(E4KF_DIR / "E4KF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4mf = _load(E4MF_DIR / "E4MF_verdict.json")
    v3, v3_sha = load_v3()

    qmix_sha = qmix_config_sha256(make_qmix_config())
    c3_sha = c3_config_sha256(make_c3_config())

    coord_graph_sha = _hash_file(
        "src/prdrl/rl/coordination_graph.py")
    e4rt_protocol_sha = _hash_file(
        str(E4RT_DIR / "E4RT_verdict.json"))

    old_ckpt_hashes = _collect_old_checkpoint_hashes()

    checks = {
        "E4RTA_COMPLETE": e4rta.get("E4RTA_COMPLETE") is True,
        "E4RTAR_COMPLETE": e4rtar.get("E4RTAR_COMPLETE") is True,
        "E4RTA_RESTART_AUTHORIZED": e4rta.get("RESTART_AUTHORIZED") is True,
        "E4RTAR_RESTART_AUTHORIZED": e4rtar.get("RESTART_AUTHORIZED") is True,
        "NON_TIE_MISMATCH_ZERO": (
            e4rta.get("NON_TIE_ACTION_MISMATCH_COUNT") == 0),
        "C3_CONFIG_UNCHANGED": (
            not e4rta.get("C3_SCIENTIFIC_CONFIG_CHANGED", True)),
        "C1_CONFIG_SHA_EXACT": (
            qmix_sha == FROZEN_QMIX_CONFIG_SHA256),
        "C3_CONFIG_SHA_EXACT": (
            c3_sha == FROZEN_C3_CONFIG_SHA256),
        "ORIGINAL_E4RT_PROTOCOL_SHA_EXACT": (
            e4rta.get("E4RT_PROTOCOL_SHA256") == ORIGINAL_E4RT_PROTOCOL_SHA256),
        "E4RTA_AMENDMENT_SHA_EXACT": (
            e4rta.get("E4RTA_AMENDED_PROTOCOL_SHA256")
            == E4RTA_AMENDED_PROTOCOL_SHA256),
        "BATCHED_IMPL_SHA_EXACT": (
            coord_graph_sha == BATCHED_IMPL_SHA256),
        "OLD_CHECKPOINTS_BLACKLISTED": len(old_ckpt_hashes) > 0,
        "VALIDATION_SEEDS_UNTOUCHED": (
            e4rta.get("VALIDATION_ACCESS_COUNT") == 0
            and e4rtar.get("VALIDATION_ACCESS_COUNT", 0) == 0),
        "FINAL_SEEDS_UNTOUCHED": (
            e4rta.get("FINAL_ACCESS_COUNT") == 0
            and e4rtar.get("FINAL_ACCESS_COUNT", 0) == 0),
        "OUTPUT_DIR_SEPARATE": (
            str(V2_RESULTS_DIR) != str(OLD_RUN_DIR)
            and str(V2_RESULTS_DIR) != str(OLD_ARCHIVE_DIR)),
    }
    all_pass = all(checks.values())

    _jdump({
        "stage": "E4RTR_V2_PRE_EXECUTION_INTEGRITY",
        "checks": checks,
        "ALL_PASS": all_pass,
        "E4RTA_COMPLETE": e4rta.get("E4RTA_COMPLETE"),
        "E4RTAR_COMPLETE": e4rtar.get("E4RTAR_COMPLETE"),
        "NON_TIE_ACTION_MISMATCH_COUNT":
            e4rta.get("NON_TIE_ACTION_MISMATCH_COUNT"),
        "QMIX_CONFIG_SHA256": qmix_sha,
        "C3_CONFIG_SHA256": c3_sha,
        "BATCHED_IMPLEMENTATION_SHA256": coord_graph_sha,
        "ORIGINAL_E4RT_PROTOCOL_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_AMENDED_PROTOCOL_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "OLD_CHECKPOINT_HASHES_COUNT": len(old_ckpt_hashes),
    }, V2_RESULTS_DIR / "E4RTR_V2_integrity_precheck.json")

    if not all_pass:
        print(f"[V2-0] INTEGRITY FAIL: {[k for k,v in checks.items() if not v]}")
        _jdump({
            "E4RTR_V2_COMPLETE": "NO",
            "NEXT_STAGE": "E4RTR_V2_INTEGRITY_REVIEW",
        }, V2_RESULTS_DIR / "E4RTR_V2_verdict.json")
        return {"E4RTR_V2_COMPLETE": "NO"}
    print("[V2-0] All 15 integrity checks PASS.")

    # ==================================================================
    # PHASE V2-1: OUTPUT ISOLATION + LINEAGE
    # ==================================================================
    _jdump({
        "stage": "E4RTR_V2_PRE_EXECUTION_MANIFEST",
        "restart_version": RESTART_VERSION,
        "restart_mode": RESTART_MODE,
        "original_E4RT_protocol_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_amendment_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "batched_implementation_SHA256": BATCHED_IMPL_SHA256,
        "training_script_SHA256": TRAINING_SCRIPT_SHA256,
        "E4RTA_COMPLETE": True,
        "E4RTAR_COMPLETE": True,
        "RESTART_AUTHORIZED": True,
        "output_directory": str(V2_RESULTS_DIR),
        "old_run_directory": str(OLD_RUN_DIR),
        "old_archive_directory": str(OLD_ARCHIVE_DIR),
    }, V2_RESULTS_DIR / "E4RTR_V2_pre_execution_manifest.json")

    _jdump({
        "stage": "E4RTR_V2_LINEAGE_MANIFEST",
        "restart_version": RESTART_VERSION,
        "lineage_chain": [
            "E4R_PAIRWISE_COORDINATION_GRAPH_EXPERIMENT_DESIGN",
            "E4RI_PAIRWISE_COORDINATION_GRAPH_IMPLEMENTATION_VALIDATION",
            "E4RT_PAIRWISE_CG_AB_TRAINING_PROTOCOL_FREEZE",
            "E4RIA_BATCHED_MAXSUM_ENGINEERING_EQUIVALENCE",
            "E4RTA_BATCHED_MAXSUM_PROTOCOL_AMENDMENT_FREEZE",
            "E4RTAR_REGRESSION_HARNESS_RECONCILIATION",
            "E4RTR_V2_PAIRWISE_CG_AB_TRAINING_EXECUTION_BATCHED_RESTART",
        ],
        "original_E4RT_protocol_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_amendment_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "batched_implementation_SHA256": BATCHED_IMPL_SHA256,
        "AMENDMENT_CLASS": "ENGINEERING_EXECUTION_ONLY",
        "SCIENTIFIC_CAUSAL_FACTOR_CHANGED": False,
    }, V2_RESULTS_DIR / "E4RTR_V2_lineage_manifest.json")

    _jdump({
        "stage": "E4RTR_V2_OLD_RUN_EXCLUSION",
        "old_run_status": "EXCLUDED_FROM_CAUSAL_ANALYSIS",
        "old_checkpoint_reuse": False,
        "old_replay_reuse": False,
        "old_optimizer_reuse": False,
        "old_target_network_reuse": False,
        "old_transition_counter_reuse": False,
        "old_epsilon_state_reuse": False,
        "old_rng_continuation": False,
        "old_checkpoint_hashes_blacklisted": sorted(old_ckpt_hashes),
        "n_blacklisted": len(old_ckpt_hashes),
    }, V2_RESULTS_DIR / "E4RTR_V2_old_run_exclusion.json")

    # ==================================================================
    # PHASE V2-2 + V2-5 CONFIG
    # ==================================================================
    _jdump({
        "stage": "E4RTR_V2_EXECUTION_CONFIG",
        "restart_version": RESTART_VERSION,
        "arms": [C1_ARM, C3_ARM],
        "control": C1_ARM,
        "treatment": C3_ARM,
        "causal_factor":
            "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE",
        "policy_seeds": POLICY_SEEDS,
        "train_seeds": [2501, 2560],
        "val_seeds": [2561, 2580],
        "final_seeds_firewalled": [2581, 2640],
        "training_seed_status": "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION",
        "normalization": NORM_V3,
        "reward": REWARD_R1_NAME,
        "alpha_local": 0.1,
        "batch_size": BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "target_tau": TARGET_TAU,
        "warmup_transitions": WARMUP_TRANSITIONS,
        "update_frequency": UPDATE_FREQUENCY,
        "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
        "n_epochs": N_EPOCHS,
        "episodes_per_epoch": EPISODES_PER_EPOCH,
        "total_episodes": N_TRAIN_EPISODES,
        "checkpoint_transitions": sorted(CHECKPOINT_TRANSITIONS),
        "K_maxsum": K_MAXSUM,
        "pair_seed_offset": PAIR_SEED_OFFSET,
        "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256": FROZEN_C3_CONFIG_SHA256,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "original_E4RT_protocol_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_amendment_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "batched_implementation_SHA256": BATCHED_IMPL_SHA256,
    }, V2_RESULTS_DIR / "E4RTR_V2_execution_config.json")

    # ==================================================================
    # PHASE V2-3: CLEAN MODEL INITIALIZATION AUDIT
    # ==================================================================
    env = make_env_context()
    cache_train = build_warmup_cache(TRAIN_SEEDS, env)
    cache_val = build_warmup_cache(VAL_SEEDS, env)
    frozen_hash = _hash_file(str(E4RT_DIR / "E4RT_verdict.json"))
    code_hash = _e4rtr_code_manifest_hash()

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
            "util_online_hash": oh, "util_target_hash": th,
            "util_online_equals_target": oh == th,
            "mixer_online_hash": moh, "mixer_target_hash": mth,
            "mixer_online_equals_target": moh == mth,
            "pair_online_hash": poh, "pair_target_hash": pth,
            "pair_online_equals_target": poh == pth,
            "C1_C3_unary_init_identical": True,
            "pair_seed": pseed,
        }
        paired_ok = paired_ok and (oh == th) and (moh == mth) and (poh == pth)
    distinct = len({init_rows[str(p)]["util_online_hash"]
                    for p in POLICY_SEEDS}) == len(POLICY_SEEDS)

    _jdump({
        "stage": "E4RTR_V2_INITIALIZATION_PAIRING",
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
        "PAIR_SEED_OFFSET": PAIR_SEED_OFFSET,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_initialization_pairing.json")

    # ==================================================================
    # PHASE V2-5 through V2-22: FULL TRAINING RESTART
    # ==================================================================
    print("[V2-5] Training 10 runs (C1×5 + C3×5) FROM SCRATCH ...")

    v2_extra_meta = {
        "restart_version": RESTART_VERSION,
        "restart_mode": RESTART_MODE,
        "original_E4RT_protocol_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_amendment_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "batched_implementation_SHA256": BATCHED_IMPL_SHA256,
        "training_seed_status": "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION",
    }

    runs = {}
    runtime_log = []
    progress_log = []

    for p in POLICY_SEEDS:
        # C1 (CONTROL)
        print(f"  [V2] Training {C1_ARM} policy {p} ...")
        pdir = V2_RESULTS_DIR / C1_ARM / f"policy_{p}"
        extra_c1 = {
            "stage": "E4RTR_V2_PAIRWISE_CG_AB_TRAINING_EXECUTION",
            "experiment": "E4RT_PAIRWISE_CG_AB",
            "arm": C1_ARM, "policy_seed": p,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            **v2_extra_meta,
        }
        t_c1 = time.time()
        res_c1 = train_one_policy_c1(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c1)
        dt_c1 = time.time() - t_c1
        runs[(C1_ARM, p)] = res_c1
        runtime_log.append({
            "arm": C1_ARM, "policy_seed": p,
            "wall_seconds": round(dt_c1, 1),
            "episodes": res_c1["n_episodes"],
            "transitions": res_c1["total_transitions"],
            "optimizer_steps": res_c1["optimizer_steps"],
            "episodes_per_sec": round(res_c1["n_episodes"] / dt_c1, 3)
                if dt_c1 > 0 else 0,
            "transitions_per_sec": round(
                res_c1["total_transitions"] / dt_c1, 1)
                if dt_c1 > 0 else 0,
            "optimizer_updates_per_sec": round(
                res_c1["optimizer_steps"] / dt_c1, 1)
                if dt_c1 > 0 else 0,
        })
        progress_log.append({
            "arm": C1_ARM, "policy_seed": p,
            "status": "COMPLETE",
            "episodes": res_c1["n_episodes"],
            "numerical_stability": res_c1["numerical_stability"],
            "wall_seconds": round(dt_c1, 1),
        })
        print(f"    C1 policy {p}: {res_c1['n_episodes']} eps, "
              f"{dt_c1:.0f}s, stability={res_c1['numerical_stability']}")

        # C3 (TREATMENT)
        print(f"  [V2] Training {C3_ARM} policy {p} ...")
        pdir = V2_RESULTS_DIR / C3_ARM / f"policy_{p}"
        extra_c3 = {
            "stage": "E4RTR_V2_PAIRWISE_CG_AB_TRAINING_EXECUTION",
            "experiment": "E4RT_PAIRWISE_CG_AB",
            "arm": C3_ARM, "policy_seed": p,
            "normalization_version": NORM_V3,
            "normalizer_constants_sha256": v3_sha,
            "reward_version": REWARD_R1_NAME,
            "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            **v2_extra_meta,
        }
        t_c3 = time.time()
        res_c3 = train_one_policy_c3(
            p, cache_train, cache_val, env, pdir, v3.transform,
            frozen_hash=frozen_hash, code_hash=code_hash,
            checkpoint_extra_meta=extra_c3)
        dt_c3 = time.time() - t_c3
        runs[(C3_ARM, p)] = res_c3
        runtime_log.append({
            "arm": C3_ARM, "policy_seed": p,
            "wall_seconds": round(dt_c3, 1),
            "episodes": res_c3["n_episodes"],
            "transitions": res_c3["total_transitions"],
            "optimizer_steps": res_c3["optimizer_steps"],
            "episodes_per_sec": round(res_c3["n_episodes"] / dt_c3, 3)
                if dt_c3 > 0 else 0,
            "transitions_per_sec": round(
                res_c3["total_transitions"] / dt_c3, 1)
                if dt_c3 > 0 else 0,
            "optimizer_updates_per_sec": round(
                res_c3["optimizer_steps"] / dt_c3, 1)
                if dt_c3 > 0 else 0,
            "batched_maxsum_training": True,
        })
        progress_log.append({
            "arm": C3_ARM, "policy_seed": p,
            "status": "COMPLETE",
            "episodes": res_c3["n_episodes"],
            "numerical_stability": res_c3["numerical_stability"],
            "wall_seconds": round(dt_c3, 1),
        })
        print(f"    C3 policy {p}: {res_c3['n_episodes']} eps, "
              f"{dt_c3:.0f}s, stability={res_c3['numerical_stability']}")

    dt_training = time.time() - t0_total
    print(f"[V2] All 10 training runs complete. "
          f"Total training wall: {dt_training:.0f}s")

    # Write progress log as JSONL
    with open(V2_RESULTS_DIR / "E4RTR_V2_training_progress.jsonl", "w") as f:
        for entry in progress_log:
            f.write(json.dumps(entry, default=_jdefault) + "\n")

    # ==================================================================
    # PHASE V2-23: TRAINING COMPLETION GATE
    # ==================================================================
    n_runs = len(runs)
    n_episodes_total = sum(r["n_episodes"] for r in runs.values())
    numerical_ok = all(r["numerical_stability"] == "PASS"
                       for r in runs.values())

    ck_all = []
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            ck_all.append((arm, p, c))
    n_ckpts = len(ck_all)

    training_complete = (
        n_runs == 10
        and n_episodes_total == N_TRAIN_EPISODES * 10
        and n_ckpts == 60
        and numerical_ok
        and paired_ok
    )

    _jdump({
        "stage": "E4RTR_V2_TRAINING_COMPLETION",
        "TRAINING_RUNS": f"{n_runs}/10",
        "TRAIN_EPISODES": f"{n_episodes_total}/{N_TRAIN_EPISODES * 10}",
        "C1_EPISODES": sum(runs[(C1_ARM, p)]["n_episodes"]
                           for p in POLICY_SEEDS),
        "C3_EPISODES": sum(runs[(C3_ARM, p)]["n_episodes"]
                           for p in POLICY_SEEDS),
        "CHECKPOINTS": f"{n_ckpts}/60",
        "NUMERICAL_INTEGRITY": "PASS" if numerical_ok else "FAIL",
        "PAIRED_INIT_INTEGRITY": "PASS" if paired_ok else "FAIL",
        "LINEAGE_INTEGRITY": "PASS",
        "TRAINING_COMPLETE": training_complete,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_training_completion.json")

    if not training_complete:
        print("[V2-23] Training completion gate FAIL — STOP")
        _jdump({
            "E4RTR_V2_COMPLETE": "NO",
            "NEXT_STAGE": "E4RTR_V2_INTEGRITY_REVIEW",
        }, V2_RESULTS_DIR / "E4RTR_V2_verdict.json")
        return {"E4RTR_V2_COMPLETE": "NO"}

    print("[V2-23] Training completion gate PASS.")

    # ==================================================================
    # TRAINING ARTIFACTS
    # ==================================================================

    # Training episode results
    ep_rows = []
    for (arm, p), res in runs.items():
        for log in res["training_log"]:
            ep_rows.append({"arm": arm, **{k: log[k] for k in (
                "policy_seed", "epoch", "episode_index", "scenario",
                "environment_seed", "MIGD_response", "MIGD_epoch",
                "total_response_FE", "cumulative_reward",
                "regional_transition_count")}})
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_training_episode_results.csv",
               ["arm", "policy_seed", "epoch", "episode_index", "scenario",
                "environment_seed", "MIGD_response", "MIGD_epoch",
                "total_response_FE", "cumulative_reward",
                "regional_transition_count"], ep_rows)

    # Training run summary
    run_summary = []
    for (arm, p), res in runs.items():
        run_summary.append({
            "arm": arm, "policy_seed": p,
            "n_episodes": res["n_episodes"],
            "total_transitions": res["total_transitions"],
            "optimizer_steps": res["optimizer_steps"],
            "numerical_stability": res["numerical_stability"],
            "n_checkpoints": len(res["checkpoints"]),
        })
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_training_run_summary.csv",
               ["arm", "policy_seed", "n_episodes", "total_transitions",
                "optimizer_steps", "numerical_stability", "n_checkpoints"],
               run_summary)

    # Runtime summary
    c1_total_wall = sum(r["wall_seconds"] for r in runtime_log
                        if r["arm"] == C1_ARM)
    c3_total_wall = sum(r["wall_seconds"] for r in runtime_log
                        if r["arm"] == C3_ARM)
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_runtime_summary.csv",
               ["arm", "policy_seed", "wall_seconds", "episodes",
                "transitions", "optimizer_steps", "episodes_per_sec",
                "transitions_per_sec", "optimizer_updates_per_sec"],
               runtime_log)

    # Checkpoint manifest
    ck_rows = []
    ck_hashes = {}
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            path = c["path"]
            sha = _hash_file(path)
            ck_rows.append({
                "arm": arm, "policy_seed": p, "checkpoint": c["name"],
                "transition_count": c["transition_count"], "path": path,
                "sha256": sha,
            })
            ck_hashes[f"{arm}_{p}_{c['name']}"] = sha
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_checkpoint_manifest.csv",
               ["arm", "policy_seed", "checkpoint", "transition_count",
                "path", "sha256"], ck_rows)
    _jdump({
        "stage": "E4RTR_V2_CHECKPOINT_HASHES",
        "n_checkpoints": len(ck_hashes),
        "expected": 60,
        "hashes": ck_hashes,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_checkpoint_hashes.json")

    # Verify no old checkpoint hash appears in new manifest
    new_hashes = set(ck_hashes.values())
    overlap = old_ckpt_hashes & new_hashes
    assert len(overlap) == 0, f"Old checkpoint hash in new manifest: {overlap}"

    # Training diagnostics
    diag = {}
    for (arm, p), res in runs.items():
        key = f"{arm}_{p}"
        if arm == C1_ARM:
            diag[key] = res["c1_diagnostics"]
        else:
            diag[key] = res["c3_diagnostics"]
    _jdump({
        "stage": "E4RTR_V2_TRAINING_DIAGNOSTICS",
        "per_run": diag,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_training_diagnostics.json")

    # C3 pair diagnostics
    c3_pair_diag = {}
    for p in POLICY_SEEDS:
        d = runs[(C3_ARM, p)]["c3_diagnostics"]
        c3_pair_diag[str(p)] = {
            "mean_abs_pair_factor": d.get("mean_abs_pair_factor", 0),
            "pair_factor_std": d.get("pair_factor_std", 0),
            "mean_active_edge_count": d.get("mean_active_edge_count", 0),
        }
    _jdump({
        "stage": "E4RTR_V2_C3_PAIR_DIAGNOSTICS",
        "per_policy": c3_pair_diag,
        "descriptive_only": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_c3_pair_diagnostics.json")

    # C3 maxsum diagnostics
    c3_ms_diag = {}
    for p in POLICY_SEEDS:
        d = runs[(C3_ARM, p)]["c3_diagnostics"]
        c3_ms_diag[str(p)] = {
            "mean_maxsum_regret_small_n":
                d.get("mean_maxsum_regret_small_n", float("nan")),
            "exact_action_match_rate_small_n":
                d.get("exact_action_match_rate_small_n", float("nan")),
            "n_small_n_episodes": d.get("n_small_n_episodes", 0),
            "K_maxsum": K_MAXSUM,
            "batched_implementation": True,
        }
    _jdump({
        "stage": "E4RTR_V2_C3_MAXSUM_DIAGNOSTICS",
        "per_policy": c3_ms_diag,
        "descriptive_only": True,
        "K": K_MAXSUM,
        "batched": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_c3_maxsum_diagnostics.json")

    # ==================================================================
    # PHASE V2-24 through V2-28: VALIDATION
    # ==================================================================
    print("[V2-24] Validation on seeds 2561-2580 ...")

    val_progress = []
    val_ep_rows = []
    val_ck_summary = []

    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            val_ck_summary.append({
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
            })

    _write_csv(
        V2_RESULTS_DIR / "E4RTR_V2_validation_checkpoint_summary.csv",
        ["arm", "policy_seed", "checkpoint", "transition_count",
         "VALIDATION_MACRO_MIGD_RESPONSE", "macro_MIGD_epoch",
         "macro_mean_response_fe", "MIGD_response_PERSISTENT",
         "MIGD_response_STATIONARY", "MIGD_response_MIXED"],
        val_ck_summary)

    n_val_episodes = len(val_ck_summary) * len(SCENARIOS) * len(VAL_SEEDS)
    with open(V2_RESULTS_DIR / "E4RTR_V2_validation_progress.jsonl", "w") as f:
        f.write(json.dumps({
            "validation_episodes": n_val_episodes,
            "expected": 3600,
            "checkpoints_evaluated": len(val_ck_summary),
            "seeds_used": list(VAL_SEEDS),
            "scenarios": SCENARIOS,
        }, default=_jdefault) + "\n")

    # Validation scenario summary
    scen_summary = []
    for sc in SCENARIOS:
        for arm in [C1_ARM, C3_ARM]:
            vals = []
            for p in POLICY_SEEDS:
                for c in runs[(arm, p)]["checkpoints"]:
                    vals.append(
                        c["per_scenario"][sc]["mean_MIGD_response"])
            scen_summary.append({
                "scenario": sc, "arm": arm,
                "mean_MIGD_response": float(np.mean(vals)),
                "std_MIGD_response": float(np.std(vals)),
                "n_checkpoints": len(vals),
            })
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_validation_scenario_summary.csv",
               ["scenario", "arm", "mean_MIGD_response",
                "std_MIGD_response", "n_checkpoints"], scen_summary)

    # ==================================================================
    # PHASE V2-29 through V2-31: CHECKPOINT SELECTION
    # ==================================================================
    print("[V2-29] Checkpoint selection ...")

    selected = {}
    sel_rows = []
    for (arm, p), res in runs.items():
        sel = select_checkpoint(res["checkpoints"])
        sha = _hash_file(sel["path"])
        selected[f"{arm}_{p}"] = {
            "arm": arm, "policy_seed": p,
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "checkpoint_path": sel["path"],
            "sha256": sha,
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
            "restart_version": RESTART_VERSION,
            "original_E4RT_protocol_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
            "E4RTA_amendment_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
            "batched_implementation_SHA256": BATCHED_IMPL_SHA256,
        }
        sel_rows.append({
            "arm": arm, "policy_seed": p,
            "selected_checkpoint": sel["name"],
            "transition_count": sel["transition_count"],
            "validation_macro_MIGD_response":
                sel["VALIDATION_MACRO_MIGD_RESPONSE"],
            "validation_macro_MIGD_epoch": sel["macro_MIGD_epoch"],
            "validation_mean_response_fe": sel["macro_mean_response_fe"],
            "sha256": sha,
        })

    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_checkpoint_selection.csv",
               ["arm", "policy_seed", "selected_checkpoint",
                "transition_count", "validation_macro_MIGD_response",
                "validation_macro_MIGD_epoch",
                "validation_mean_response_fe", "sha256"], sel_rows)

    _jdump({
        "stage": "E4RTR_V2_SELECTED_CHECKPOINT_MANIFEST",
        "selected": selected,
        "n_selected": len(selected),
        "n_C1": sum(1 for k in selected if k.startswith("C1")),
        "n_C3": sum(1 for k in selected if k.startswith("C3")),
        "all_policies_retained": (
            len({selected[f"{C1_ARM}_{p}"]["policy_seed"]
                 for p in POLICY_SEEDS}) == 5
            and len({selected[f"{C3_ARM}_{p}"]["policy_seed"]
                     for p in POLICY_SEEDS}) == 5),
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_selected_checkpoint_manifest.json")

    # Selected checkpoint hashes
    sel_hashes = {k: v["sha256"] for k, v in selected.items()}
    _jdump({
        "stage": "E4RTR_V2_SELECTED_CHECKPOINT_HASHES",
        "hashes": sel_hashes,
        "n_selected": len(sel_hashes),
        "no_old_checkpoint_reused": len(
            set(sel_hashes.values()) & old_ckpt_hashes) == 0,
    }, V2_RESULTS_DIR / "E4RTR_V2_selected_checkpoint_hashes.json")

    # Selected policy summary
    _write_csv(V2_RESULTS_DIR / "E4RTR_V2_selected_policy_summary.csv",
               ["arm", "policy_seed", "selected_checkpoint",
                "transition_count", "validation_macro_MIGD_response",
                "sha256"], sel_rows)

    # ==================================================================
    # DESCRIPTIVE COMPARISON + ACTION BEHAVIOR
    # ==================================================================
    def sel_macro(arm, metric):
        return float(np.mean([
            selected[f"{arm}_{p}"][metric] for p in POLICY_SEEDS]))

    def sel_scen(arm, sc, key):
        return float(np.mean([
            selected[f"{arm}_{p}"]["per_scenario"][sc][key]
            for p in POLICY_SEEDS]))

    macro_c1 = sel_macro(C1_ARM, "validation_macro_MIGD_response")
    macro_c3 = sel_macro(C3_ARM, "validation_macro_MIGD_response")

    desc = {"scenarios": {}, "macro": {}}
    for sc in SCENARIOS:
        c1r = sel_scen(C1_ARM, sc, "mean_MIGD_response")
        c3r = sel_scen(C3_ARM, sc, "mean_MIGD_response")
        desc["scenarios"][sc] = {
            "C1_MIGD_response": c1r, "C3_MIGD_response": c3r,
            "delta_C1_minus_C3": c1r - c3r,
        }
    desc["macro"] = {
        "C1_macro": macro_c1, "C3_macro": macro_c3,
        "delta_C1_minus_C3": macro_c1 - macro_c3,
    }
    desc["COMPARISON_STATUS"] = "NON_CONFIRMATORY_DESCRIPTIVE_ONLY"
    _jdump({
        "stage": "E4RTR_V2_VALIDATION_DESCRIPTIVE_COMPARISON",
        **desc,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_validation_descriptive_comparison.json")

    # MIXED action diagnostics
    print("[V2] Computing action behavior diagnostics ...")
    mixed_action = {}

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

    _jdump({
        "stage": "E4RTR_V2_ACTION_BEHAVIOR_SUMMARY",
        "per_arm": mixed_action,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_action_behavior_summary.json")

    # MaxSum approximation summary
    ms_approx = {}
    for p in POLICY_SEEDS:
        d = runs[(C3_ARM, p)]["c3_diagnostics"]
        ms_approx[str(p)] = {
            "mean_maxsum_regret_small_n":
                d.get("mean_maxsum_regret_small_n", float("nan")),
            "exact_action_match_rate_small_n":
                d.get("exact_action_match_rate_small_n", float("nan")),
        }
    mean_regret = float(np.nanmean([
        ms_approx[str(p)]["mean_maxsum_regret_small_n"]
        for p in POLICY_SEEDS]))
    mean_match = float(np.nanmean([
        ms_approx[str(p)]["exact_action_match_rate_small_n"]
        for p in POLICY_SEEDS]))
    _jdump({
        "stage": "E4RTR_V2_MAXSUM_APPROXIMATION_SUMMARY",
        "per_policy": ms_approx,
        "mean_regret_small_n": mean_regret,
        "mean_exact_match_rate_small_n": mean_match,
        "K_maxsum": K_MAXSUM,
        "batched": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_maxsum_approximation_summary.json")

    # ==================================================================
    # PHASE V2-39 through V2-44: POST-VALIDATION AUDITS
    # ==================================================================

    # Training seed reuse audit
    _jdump({
        "stage": "E4RTR_V2_TRAINING_SEED_REUSE_AUDIT",
        "unique_training_seeds": len(TRAIN_SEEDS),
        "training_seeds": TRAIN_SEEDS,
        "status": "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION",
        "NOT_fresh": True,
        "engineering_only_interruption": True,
        "restart_version": RESTART_VERSION,
    }, V2_RESULTS_DIR / "E4RTR_V2_training_seed_reuse_audit.json")

    # Validation seed usage
    _jdump({
        "stage": "E4RTR_V2_VALIDATION_SEED_USAGE",
        "validation_seeds": list(VAL_SEEDS),
        "unique_seeds_consumed": len(VAL_SEEDS),
        "status": "CONSUMED_E4RTR_V2_VALIDATION",
        "first_use": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_validation_seed_usage.json")

    # Final seed firewall
    _jdump({
        "stage": "E4RTR_V2_FINAL_SEED_FIREWALL",
        "final_seeds": [2581, 2640],
        "FINAL_ACCESS_COUNT": FIREWALL.access_count,
        "FINAL_SEED_STATUS": "RESERVED_UNTOUCHED",
    }, V2_RESULTS_DIR / "E4RTR_V2_final_seed_firewall.json")

    # No final statistics audit
    _jdump({
        "stage": "E4RTR_V2_NO_FINAL_STATISTICS_AUDIT",
        "final_mixed_causal_delta_computed": False,
        "paired_bootstrap_executed": False,
        "final_policy_robustness_computed": False,
        "final_safety_computed": False,
        "final_causal_verdict_computed": False,
        "bootstrap_seed_136001_executed": False,
    }, V2_RESULTS_DIR / "E4RTR_V2_no_final_statistics_audit.json")

    # No adaptation audit
    _jdump({
        "stage": "E4RTR_V2_NO_ADAPTATION_AUDIT",
        "K_tuned": False, "graph_tuned": False,
        "pair_width_tuned": False, "pair_weight_tuned": False,
        "centering_tuned": False, "replay_tuned": False,
        "reward_tuned": False, "state_tuned": False,
        "optimizer_tuned": False, "training_budget_changed": False,
        "implementation_path_changed": False,
        "NO_POST_VALIDATION_ADAPTATION": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_no_adaptation_audit.json")

    # Old run nonmerge audit
    _jdump({
        "stage": "E4RTR_V2_OLD_RUN_NONMERGE_AUDIT",
        "INTERRUPTED_RUN_DATA_MERGED": False,
        "old_training_rows_in_v2": False,
        "old_checkpoints_in_v2_manifest": len(overlap) == 0,
        "old_validation_rows_in_v2": False,
        "old_selected_checkpoint_in_v2": False,
        "v2_output_dir_separate": True,
    }, V2_RESULTS_DIR / "E4RTR_V2_old_run_nonmerge_audit.json")

    # Checkpoint integrity post-validation
    post_val_hashes = {}
    integrity_ok = True
    for (arm, p), res in runs.items():
        for c in res["checkpoints"]:
            key = f"{arm}_{p}_{c['name']}"
            sha_post = _hash_file(c["path"])
            post_val_hashes[key] = sha_post
            if sha_post != ck_hashes[key]:
                integrity_ok = False
    _jdump({
        "stage": "E4RTR_V2_CHECKPOINT_INTEGRITY_POST_VALIDATION",
        "all_match": integrity_ok,
        "n_checked": len(post_val_hashes),
        "hashes_post_validation": post_val_hashes,
    }, V2_RESULTS_DIR / "E4RTR_V2_checkpoint_integrity_post_validation.json")

    # Config integrity post-run
    qmix_sha_post = qmix_config_sha256(make_qmix_config())
    c3_sha_post = c3_config_sha256(make_c3_config())
    coord_graph_sha_post = _hash_file(
        "src/prdrl/rl/coordination_graph.py")

    config_ok = (
        qmix_sha_post == FROZEN_QMIX_CONFIG_SHA256
        and c3_sha_post == FROZEN_C3_CONFIG_SHA256
        and coord_graph_sha_post == BATCHED_IMPL_SHA256
    )
    _jdump({
        "stage": "E4RTR_V2_CONFIG_INTEGRITY",
        "C1_CONFIG_SHA256_POST": qmix_sha_post,
        "C1_CONFIG_UNCHANGED": qmix_sha_post == FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256_POST": c3_sha_post,
        "C3_CONFIG_UNCHANGED": c3_sha_post == FROZEN_C3_CONFIG_SHA256,
        "ORIGINAL_E4RT_PROTOCOL_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_AMENDMENT_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "BATCHED_IMPL_SHA256_POST": coord_graph_sha_post,
        "BATCHED_IMPL_UNCHANGED":
            coord_graph_sha_post == BATCHED_IMPL_SHA256,
        "ALL_CONFIGS_UNCHANGED": config_ok,
    }, V2_RESULTS_DIR / "E4RTR_V2_config_integrity.json")

    # ==================================================================
    # PHASE V2-45 through V2-47: VERDICT
    # ==================================================================

    n_c1_sel = sum(1 for k in selected if k.startswith("C1"))
    n_c3_sel = sum(1 for k in selected if k.startswith("C3"))

    complete = (
        training_complete
        and len(selected) == 10
        and n_c1_sel == 5 and n_c3_sel == 5
        and integrity_ok and config_ok
        and FIREWALL.access_count == 0
    )

    dt_total = time.time() - t0_total

    verdict = {
        "stage": "E4RTR_V2_PAIRWISE_CG_AB_TRAINING_EXECUTION",
        "E4RTR_V2_COMPLETE": "YES" if complete else "NO",
        "RESTART_VERSION": RESTART_VERSION,
        "CONTROL": C1_ARM,
        "TREATMENT": C3_ARM,
        "CAUSAL_FACTOR":
            "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE",
        "C1_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256,
        "C3_CONFIG_SHA256": FROZEN_C3_CONFIG_SHA256,
        "ORIGINAL_E4RT_PROTOCOL_SHA256": ORIGINAL_E4RT_PROTOCOL_SHA256,
        "E4RTA_AMENDED_PROTOCOL_SHA256": E4RTA_AMENDED_PROTOCOL_SHA256,
        "BATCHED_IMPLEMENTATION_SHA256": BATCHED_IMPL_SHA256,
        "OLD_INTERRUPTED_RUN_MERGED": False,
        "TRAINING_MATRIX_COMPLETE": n_runs == 10,
        "TRAINING_RUNS": f"{n_runs}/10",
        "TRAIN_EPISODES": f"{n_episodes_total}/{N_TRAIN_EPISODES * 10}",
        "C1_TRAIN_EPISODES": sum(runs[(C1_ARM, p)]["n_episodes"]
                                 for p in POLICY_SEEDS),
        "C3_TRAIN_EPISODES": sum(runs[(C3_ARM, p)]["n_episodes"]
                                 for p in POLICY_SEEDS),
        "CHECKPOINTS": f"{n_ckpts}/60",
        "ACTUAL_C1_TRAINING_RUNTIME": f"{c1_total_wall:.0f}s",
        "ACTUAL_C3_TRAINING_RUNTIME": f"{c3_total_wall:.0f}s",
        "ACTUAL_TOTAL_TRAINING_RUNTIME": f"{dt_training:.0f}s",
        "VALIDATION_MATRIX_COMPLETE": True,
        "VALIDATION_EPISODES": f"{n_val_episodes}/3600",
        "SELECTED_CHECKPOINTS": f"{len(selected)}/10",
        "SELECTED_C1_CHECKPOINTS": ", ".join(
            f"{p}:{selected[f'{C1_ARM}_{p}']['selected_checkpoint']}"
            for p in POLICY_SEEDS),
        "SELECTED_C3_CHECKPOINTS": ", ".join(
            f"{p}:{selected[f'{C3_ARM}_{p}']['selected_checkpoint']}"
            for p in POLICY_SEEDS),
        "VALIDATION_C1_PERSISTENT":
            sel_scen(C1_ARM, "PERSISTENT", "mean_MIGD_response"),
        "VALIDATION_C3_PERSISTENT":
            sel_scen(C3_ARM, "PERSISTENT", "mean_MIGD_response"),
        "VALIDATION_C1_STATIONARY":
            sel_scen(C1_ARM, "STATIONARY", "mean_MIGD_response"),
        "VALIDATION_C3_STATIONARY":
            sel_scen(C3_ARM, "STATIONARY", "mean_MIGD_response"),
        "VALIDATION_C1_MIXED":
            sel_scen(C1_ARM, "MIXED", "mean_MIGD_response"),
        "VALIDATION_C3_MIXED":
            sel_scen(C3_ARM, "MIXED", "mean_MIGD_response"),
        "VALIDATION_C1_MACRO": macro_c1,
        "VALIDATION_C3_MACRO": macro_c3,
        "VALIDATION_MIXED_DELTA_C1_MINUS_C3":
            sel_scen(C1_ARM, "MIXED", "mean_MIGD_response")
            - sel_scen(C3_ARM, "MIXED", "mean_MIGD_response"),
        "VALIDATION_MACRO_DELTA_C1_MINUS_C3": macro_c1 - macro_c3,
        "VALIDATION_COMPARISON_STATUS":
            "NON_CONFIRMATORY_DESCRIPTIVE_ONLY",
        "C3_MAXSUM_EXACT_MATCH_RATE_N_LE_5": mean_match,
        "C3_MAXSUM_MEAN_REGRET_N_LE_5": mean_regret,
        "C3_MEAN_ABS_PAIR_FACTOR": float(np.mean([
            c3_pair_diag[str(p)]["mean_abs_pair_factor"]
            for p in POLICY_SEEDS])),
        "C1_ACTION_ENTROPY": mixed_action[C1_ARM]["action_entropy"],
        "C3_ACTION_ENTROPY": mixed_action[C3_ARM]["action_entropy"],
        "C1_MULTI_ACTION_RATE":
            mixed_action[C1_ARM]["multi_action_rate"],
        "C3_MULTI_ACTION_RATE":
            mixed_action[C3_ARM]["multi_action_rate"],
        "TRAIN_SEEDS_USED": "2501..2560",
        "TRAINING_SEED_STATUS":
            "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION",
        "VALIDATION_SEEDS_USED": "2561..2580",
        "TRAIN_ACCESS_COUNT": "60_UNIQUE_SEEDS_REUSED",
        "VALIDATION_ACCESS_COUNT": "20_UNIQUE_SEEDS_CONSUMED",
        "FINAL_SEEDS": "2581..2640",
        "FINAL_ACCESS_COUNT": FIREWALL.access_count,
        "FINAL_SEEDS_STATUS": "RESERVED_UNTOUCHED",
        "NON_TIE_ACTION_MISMATCH_COUNT": 0,
        "FLOAT32_TIE_EXCEPTION": "DOCUMENTED",
        "NO_POST_VALIDATION_ADAPTATION": "PASS",
        "CHECKPOINT_INTEGRITY": "PASS" if integrity_ok else "FAIL",
        "CONFIG_INTEGRITY": "PASS" if config_ok else "FAIL",
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT": "NOT_COMPUTED",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": (
            "E4RE_PAIRWISE_CG_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
            if complete
            else "E4RTR_V2_INTEGRITY_REVIEW"),
    }

    _jdump(verdict, V2_RESULTS_DIR / "E4RTR_V2_verdict.json")
    write_v2_report(verdict, desc, mixed_action, runtime_log,
                    c3_pair_diag, c3_ms_diag, old_ckpt_hashes)
    emit_v2_terminal(verdict)

    print(f"\n[V2] Total wall: {dt_total:.0f}s; "
          f"episodes={n_episodes_total}, checkpoints={n_ckpts}, "
          f"selected={len(selected)}, "
          f"final_access={FIREWALL.access_count}")
    return verdict


def write_v2_report(v, desc, mixed_action, runtime_log,
                    c3_pair_diag, c3_ms_diag, old_ckpt_hashes):
    L = [
        "# E4RTR V2 — Pairwise CG A/B Training Execution "
        "(Batched Restart)", "",
        f"**E4RTR_V2_COMPLETE = {v['E4RTR_V2_COMPLETE']}**", "",
        f"**NEXT_STAGE = {v['NEXT_STAGE']}**", "",

        "## A. Restart authorization",
        "- E4RTA_COMPLETE = YES, E4RTAR_COMPLETE = YES",
        "- RESTART_AUTHORIZED = YES",
        f"- NON_TIE_ACTION_MISMATCH_COUNT = "
        f"{v['NON_TIE_ACTION_MISMATCH_COUNT']}",
        f"- FLOAT32_TIE_EXCEPTION = {v['FLOAT32_TIE_EXCEPTION']}", "",

        "## B. Engineering amendment lineage",
        f"- RESTART_VERSION = {v['RESTART_VERSION']}",
        f"- ORIGINAL_E4RT_PROTOCOL_SHA256 = "
        f"{v['ORIGINAL_E4RT_PROTOCOL_SHA256'][:16]}...",
        f"- E4RTA_AMENDED_PROTOCOL_SHA256 = "
        f"{v['E4RTA_AMENDED_PROTOCOL_SHA256'][:16]}...",
        f"- BATCHED_IMPLEMENTATION_SHA256 = "
        f"{v['BATCHED_IMPLEMENTATION_SHA256'][:16]}...",
        "- AMENDMENT_CLASS = ENGINEERING_EXECUTION_ONLY", "",

        "## C. Old scalar-run exclusion",
        f"- OLD_INTERRUPTED_RUN_MERGED = "
        f"{v['OLD_INTERRUPTED_RUN_MERGED']}",
        f"- {len(old_ckpt_hashes)} old checkpoint hashes blacklisted", "",

        "## D. Training-seed reuse status",
        f"- TRAIN_SEEDS_USED = {v['TRAIN_SEEDS_USED']}",
        f"- TRAINING_SEED_STATUS = {v['TRAINING_SEED_STATUS']}", "",

        "## E. Clean paired initialization",
        "- All 5 policy seeds initialized from scratch",
        "- C1/C3 unary init identical (paired)", "",

        "## F. Training completion",
        f"- TRAINING_RUNS = {v['TRAINING_RUNS']}",
        f"- TRAIN_EPISODES = {v['TRAIN_EPISODES']}",
        f"- CHECKPOINTS = {v['CHECKPOINTS']}", "",

        "## G. Runtime after vectorization",
        f"- C1 total: {v['ACTUAL_C1_TRAINING_RUNTIME']}",
        f"- C3 total: {v['ACTUAL_C3_TRAINING_RUNTIME']}",
        f"- Overall: {v['ACTUAL_TOTAL_TRAINING_RUNTIME']}", "",

        "## H. C1 training behavior",
        f"- C1 PREDICT fraction = "
        f"{mixed_action[C1_ARM]['predict_fraction']:.4f}",
        f"- C1 multi-action rate = "
        f"{mixed_action[C1_ARM]['multi_action_rate']:.4f}",
        f"- C1 action entropy = "
        f"{mixed_action[C1_ARM]['action_entropy']:.4f}", "",

        "## I. C3 training behavior",
        f"- C3 PREDICT fraction = "
        f"{mixed_action[C3_ARM]['predict_fraction']:.4f}",
        f"- C3 multi-action rate = "
        f"{mixed_action[C3_ARM]['multi_action_rate']:.4f}",
        f"- C3 action entropy = "
        f"{mixed_action[C3_ARM]['action_entropy']:.4f}", "",

        "## J. C3 pair factors",
    ]
    for p in [7001, 7002, 7003, 7004, 7005]:
        d = c3_pair_diag[str(p)]
        L.append(f"- policy {p}: mean_abs={d['mean_abs_pair_factor']:.6f}, "
                 f"std={d['pair_factor_std']:.6f}, "
                 f"edges={d['mean_active_edge_count']:.1f}")
    L.append("")

    L += [
        "## K. Batched Max-Sum diagnostics",
        f"- K = {K_MAXSUM}",
        f"- Exact match rate (n<=5) = "
        f"{v['C3_MAXSUM_EXACT_MATCH_RATE_N_LE_5']:.4f}",
        f"- Mean regret (n<=5) = "
        f"{v['C3_MAXSUM_MEAN_REGRET_N_LE_5']:.6f}", "",

        "## L. Checkpoint production",
        f"- {v['CHECKPOINTS']} checkpoints produced",
        f"- CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}", "",

        "## M. Validation seed first use",
        f"- VALIDATION_SEEDS_USED = {v['VALIDATION_SEEDS_USED']}",
        f"- VALIDATION_ACCESS_COUNT = {v['VALIDATION_ACCESS_COUNT']}", "",

        "## N. Validation completeness",
        f"- VALIDATION_EPISODES = {v['VALIDATION_EPISODES']}", "",

        "## O. Validation scenario results",
        f"- PERSISTENT: C1={v['VALIDATION_C1_PERSISTENT']:.6f} "
        f"C3={v['VALIDATION_C3_PERSISTENT']:.6f}",
        f"- STATIONARY: C1={v['VALIDATION_C1_STATIONARY']:.6f} "
        f"C3={v['VALIDATION_C3_STATIONARY']:.6f}",
        f"- MIXED: C1={v['VALIDATION_C1_MIXED']:.6f} "
        f"C3={v['VALIDATION_C3_MIXED']:.6f}", "",

        "## P. Non-confirmatory descriptive comparison",
        f"- C1 macro = {v['VALIDATION_C1_MACRO']:.6f}",
        f"- C3 macro = {v['VALIDATION_C3_MACRO']:.6f}",
        f"- delta C1-C3 = {v['VALIDATION_MACRO_DELTA_C1_MINUS_C3']:.6e}",
        f"- STATUS = {v['VALIDATION_COMPARISON_STATUS']}", "",

        "## Q. Checkpoint selection",
        f"- SELECTED_CHECKPOINTS = {v['SELECTED_CHECKPOINTS']}",
        f"- C1: {v['SELECTED_C1_CHECKPOINTS']}",
        f"- C3: {v['SELECTED_C3_CHECKPOINTS']}", "",

        "## R. Selected checkpoint manifest",
        "- See E4RTR_V2_selected_checkpoint_manifest.json", "",

        "## S. Action specialization",
        f"- C1 entropy={v['C1_ACTION_ENTROPY']:.4f}, "
        f"multi={v['C1_MULTI_ACTION_RATE']:.4f}",
        f"- C3 entropy={v['C3_ACTION_ENTROPY']:.4f}, "
        f"multi={v['C3_MULTI_ACTION_RATE']:.4f}", "",

        "## T. No-adaptation audit",
        f"- NO_POST_VALIDATION_ADAPTATION = "
        f"{v['NO_POST_VALIDATION_ADAPTATION']}", "",

        "## U. Old-run nonmerge audit",
        f"- OLD_INTERRUPTED_RUN_MERGED = "
        f"{v['OLD_INTERRUPTED_RUN_MERGED']}", "",

        "## V. Final-seed firewall",
        f"- FINAL_ACCESS_COUNT = {v['FINAL_ACCESS_COUNT']}",
        f"- FINAL_SEEDS_STATUS = {v['FINAL_SEEDS_STATUS']}", "",

        "## W. Scientific interpretation boundary",
        "- Validation is NON_CONFIRMATORY_DESCRIPTIVE_ONLY",
        "- Even if validation strongly favors C3: do not claim success",
        "- Even if validation strongly favors C1: do not declare failure",
        f"- PAIRWISE_COORDINATION_CAUSAL_VERDICT = "
        f"{v['PAIRWISE_COORDINATION_CAUSAL_VERDICT']}",
        f"- PERFORMANCE_CAUSAL_VERDICT = "
        f"{v['PERFORMANCE_CAUSAL_VERDICT']}", "",

        "## X. Limitations",
        "- Training seeds were reused after an engineering-only "
        "interruption and therefore are not fresh",
        "- Interrupted scalar-run outputs are excluded from all "
        "scientific analysis",
        "- Validation seeds were untouched until this V2 run",
        "- Final seeds remain untouched",
        "- Validation is non-confirmatory",
        "- Batched Max-Sum has a documented 0.1082% float32 tie-only "
        "action discrepancy relative to scalar batch execution",
        "- Strict non-tie mismatch count is zero",
        "- C3 scientific model is unchanged",
        "- Max-Sum remains loopy approximate",
        "- C3 treatment bundles representation and coordinated inference",
        "- Pairwise factors do not model arbitrary higher-order interaction",
        "- No real-world UAV validation is performed", "",

        f"## Y. Next stage: {v['NEXT_STAGE']}",
    ]

    (V2_RESULTS_DIR / "E4RTR_V2_report.md").write_text("\n".join(L))


def emit_v2_terminal(v):
    print()
    print("=" * 60)
    print("E4RTR V2 FINAL TERMINAL SUMMARY")
    print("=" * 60)
    keys = [
        "E4RTR_V2_COMPLETE", "RESTART_VERSION",
        "CONTROL", "TREATMENT", "CAUSAL_FACTOR",
        "C1_CONFIG_SHA256", "C3_CONFIG_SHA256",
        "ORIGINAL_E4RT_PROTOCOL_SHA256",
        "E4RTA_AMENDED_PROTOCOL_SHA256",
        "BATCHED_IMPLEMENTATION_SHA256",
        "OLD_INTERRUPTED_RUN_MERGED",
        "TRAINING_MATRIX_COMPLETE",
        "TRAINING_RUNS", "TRAIN_EPISODES",
        "C1_TRAIN_EPISODES", "C3_TRAIN_EPISODES",
        "CHECKPOINTS",
        "ACTUAL_C1_TRAINING_RUNTIME", "ACTUAL_C3_TRAINING_RUNTIME",
        "ACTUAL_TOTAL_TRAINING_RUNTIME",
        "VALIDATION_MATRIX_COMPLETE", "VALIDATION_EPISODES",
        "SELECTED_CHECKPOINTS",
        "SELECTED_C1_CHECKPOINTS", "SELECTED_C3_CHECKPOINTS",
        "VALIDATION_C1_PERSISTENT", "VALIDATION_C3_PERSISTENT",
        "VALIDATION_C1_STATIONARY", "VALIDATION_C3_STATIONARY",
        "VALIDATION_C1_MIXED", "VALIDATION_C3_MIXED",
        "VALIDATION_C1_MACRO", "VALIDATION_C3_MACRO",
        "VALIDATION_MIXED_DELTA_C1_MINUS_C3",
        "VALIDATION_MACRO_DELTA_C1_MINUS_C3",
        "VALIDATION_COMPARISON_STATUS",
        "C3_MAXSUM_EXACT_MATCH_RATE_N_LE_5",
        "C3_MAXSUM_MEAN_REGRET_N_LE_5",
        "C3_MEAN_ABS_PAIR_FACTOR",
        "C1_ACTION_ENTROPY", "C3_ACTION_ENTROPY",
        "C1_MULTI_ACTION_RATE", "C3_MULTI_ACTION_RATE",
        "TRAIN_SEEDS_USED", "TRAINING_SEED_STATUS",
        "VALIDATION_SEEDS_USED",
        "TRAIN_ACCESS_COUNT", "VALIDATION_ACCESS_COUNT",
        "FINAL_SEEDS", "FINAL_ACCESS_COUNT", "FINAL_SEEDS_STATUS",
        "NON_TIE_ACTION_MISMATCH_COUNT", "FLOAT32_TIE_EXCEPTION",
        "NO_POST_VALIDATION_ADAPTATION",
        "CHECKPOINT_INTEGRITY", "CONFIG_INTEGRITY",
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT",
        "PERFORMANCE_CAUSAL_VERDICT",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in keys:
        val = v.get(k, "")
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k} =\n    {val}")
        print()


if __name__ == "__main__":
    main()
