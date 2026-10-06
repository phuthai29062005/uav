"""
E4IT — Reward A/B Training Protocol Freeze (PR-DRL-DMOEA, Branch B).

Freezes the COMPLETE matched training experiment V3+R0 (CONTROL) vs V3+R1 (TREATMENT) BEFORE
any RL training, with reward specification as the SOLE scientific factor. FREEZE ONLY: no
training, no instantiation of reward seeds 1501..1640. May build freeze/infrastructure only
(arm/metadata schema, paired-init loader, logging schema, firewall counters).

Both arms share corrected V3 state, 9-dim state, 9-128-128-4 Shared Double DQN, all
hyperparameters/operators/seeds/checkpoint schedule/selection, and alpha_local=0.1. The reward
experiment tests the local-credit FORMULATION (R0 raw scale-mismatched difference vs R1 bounded
symmetric quality gain) — NOT a larger coefficient. E4I/E4II/E4IDD contracts are authoritative.
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

import torch.nn as nn
from prdrl.rl.network import build_online_target_pair
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward import ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1
from prdrl.rl import reward_integration as RI
from prdrl.evaluation.crossed_bootstrap import N_BOOTSTRAP, CI_LEVEL, CI_METHOD
from prdrl.evaluation.final_verdict import (
    DELTA_E4, BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)

RESULTS_DIR = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4I_DIR = Path("results/E4I_reward_credit_revision_design")
E4IDD_DIR = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4II_DIR = Path("results/E4II_reward_implementation_validation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
REWARD_BOOTSTRAP_MASTER_SEED = 66001
PRACTICAL_MARGIN = 1e-5

TRAIN_SEEDS = list(range(1501, 1561))       # 60
VALIDATION_SEEDS = list(range(1561, 1581))  # 20
FINAL_SEEDS = list(range(1581, 1641))       # 60
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

CONSUMED_SEED_SETS = {
    "E3_701_780": range(701, 781), "E3R_801_882": range(801, 883), "H1_901_940": range(901, 941),
    "E4_train_1001_1060": range(1001, 1061), "E4_val_1061_1080": range(1061, 1081),
    "E4_final_1081_1140": range(1081, 1141), "E4GD_diag_1201_1220": range(1201, 1221),
    "E4H_train_1301_1360": range(1301, 1361), "E4H_val_1361_1380": range(1361, 1381),
    "E4H_final_1381_1440": range(1381, 1441), "E4IDD_diag_1701_1720": range(1701, 1721),
}
CONSUMED_POLICY_SEEDS = {"E4TR": [2001, 2002, 2003, 2004, 2005], "E4HTR": [3001, 3002, 3003, 3004, 3005]}

RNG_OFFSET_POLICY_INIT = 0
_reward_access = {"train": 0, "validation": 0, "final": 0}


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
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4IT — Reward A/B Training Protocol Freeze (Branch B, FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 0 integrity ----------
    e4ii = _load(E4II_DIR / "E4II_implementation_verdict.json")
    e4idd = _load(E4IDD_DIR / "E4IDD_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    src_hashes = {"reward.py": _hash_file("src/prdrl/rl/reward.py"),
                  "reward_r1.py": _hash_file("src/prdrl/rl/reward_r1.py"),
                  "reward_integration.py": _hash_file("src/prdrl/rl/reward_integration.py"),
                  "network.py": _hash_file("src/prdrl/rl/network.py"),
                  "normalization.py": _hash_file("src/prdrl/rl/normalization.py"),
                  "E4IDD_reward_r1_config.json": _hash_file(E4IDD_DIR / "E4IDD_reward_r1_config.json")}
    r1cfg = _load(E4IDD_DIR / "E4IDD_reward_r1_config.json")
    guard_ok = _load(E4II_DIR / "E4II_reward_version_guard.json")["REWARD_VERSION_GUARD"] == "PASS"

    hist = {"E4II_COMPLETE": e4ii["E4II_COMPLETE"], "R1_IMPLEMENTATION_VALID": e4ii["R1_IMPLEMENTATION_VALID"],
            "R1_REWARD_READY": e4idd["R1_REWARD_READY"], "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
            "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
            "SELECTED_BRANCH": "B_REWARD_CREDIT", "REWARD_R1_CONFIG_SHA256": e4idd["REWARD_R1_CONFIG_SHA256"],
            "ALPHA_FINAL": e4idd["ALPHA_FINAL"], "v3_constants_sha256": v3_sha,
            "reward_version_guard": "PASS" if guard_ok else "FAIL", "source_hashes": src_hashes,
            "reward_r1_hash_matches_config": src_hashes["reward_r1.py"] == r1cfg["R1_formula_hash"],
            "v3_sha_matches_config": v3_sha == r1cfg["v3_constants_sha256"]}
    _jdump(hist, RESULTS_DIR / "E4IT_historical_integrity.json")

    # ---------- paired initialization (build-only, no training) ----------
    init_rows = {}
    paired_ok = True
    for p in POLICY_SEEDS:
        online, target = build_online_target_pair(9, seed=p * 1000 + RNG_OFFSET_POLICY_INIT)
        oh, th = _param_hash(online), _param_hash(target)
        # arm-independent: R0_p and R1_p rebuild identically from the same seed
        online2, _ = build_online_target_pair(9, seed=p * 1000 + RNG_OFFSET_POLICY_INIT)
        oh2 = _param_hash(online2)
        row = {"policy_seed": p, "R0_online_hash": oh, "R1_online_hash": oh2,
               "target_hash": th, "online_equals_target": oh == th,
               "R0_equals_R1_init": oh == oh2}
        init_rows[str(p)] = row
        paired_ok = paired_ok and row["online_equals_target"] and row["R0_equals_R1_init"]

    integ_ok = (e4ii["E4II_COMPLETE"] == "YES" and e4ii["R1_IMPLEMENTATION_VALID"] == "YES"
                and e4idd["R1_REWARD_READY"] == "YES" and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
                and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
                and e4idd["REWARD_R1_CONFIG_SHA256"] == FROZEN_R1_CONFIG_SHA256
                and e4idd["ALPHA_FINAL"] == 0.1 and guard_ok
                and hist["reward_r1_hash_matches_config"] and hist["v3_sha_matches_config"] and paired_ok)
    _jdump({"stage": "E4IT_REWARD_AB_TRAINING_PROTOCOL_FREEZE", "freeze_only": True,
            "no_training": True, "no_reward_seed_instantiation": True, "integrity_ok": integ_ok,
            "source_hashes": src_hashes}, RESULTS_DIR / "E4IT_pre_manifest.json")
    if not integ_ok:
        print("[E4IT] integrity FAIL — STOP")
        _jdump({"E4IT_COMPLETE": "NO", "NEXT_STAGE": "E4IT_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4IT_verdict.json")
        return {"E4IT_COMPLETE": "NO"}
    print("[E4IT] integrity PASS. Paired init verified (online==target, R0==R1 per policy).")

    # ---------- seed disjointness ----------
    new_env = set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(FINAL_SEEDS)
    consumed = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    env_disjoint = len(new_env & consumed) == 0
    pol_disjoint = len(set(POLICY_SEEDS) & set().union(*[set(v) for v in CONSUMED_POLICY_SEEDS.values()])) == 0
    internal_disjoint = (len(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS)) == 0
                         and len(set(TRAIN_SEEDS) & set(FINAL_SEEDS)) == 0
                         and len(set(VALIDATION_SEEDS) & set(FINAL_SEEDS)) == 0)

    # ---------- freeze contracts ----------
    C: Dict[str, dict] = {}
    arm0 = RI.build_arm_config("R0_CONTROL")
    arm1 = RI.build_arm_config("R1_TREATMENT")
    diff = RI.arm_config_scientific_diff(arm0, arm1)

    C["E4IT_ab_factor_contract.json"] = {
        "experiment_factor": "REWARD_ONLY", "CONTROL": "R0_CONTROL", "TREATMENT": "R1_TREATMENT",
        "control_reward": REWARD_R0_NAME, "treatment_reward": REWARD_R1_NAME,
        "only_scientific_difference": "reward specification",
        "invariants": ["state_features", "state_dim", "normalization", "network", "parameter_sharing",
                       "joint_action_representation", "optimizer", "learning_rate", "gamma", "tau",
                       "replay_capacity", "batch_size", "warmup", "update_frequency", "epsilon_schedule",
                       "training_budget", "response_operators", "response_budget", "survival",
                       "checkpoint_schedule", "checkpoint_selection_rule"],
        "single_factor_diff": diff}
    C["E4IT_state_contract.json"] = {
        "local_v2": ["severity", "convergence_loss", "occupancy", "delta_severity"],
        "global_context_v1": ["active_region_fraction", "mean_severity", "max_severity",
                              "mean_convergence_loss", "max_convergence_loss"],
        "state_dim": 9, "identical_both_arms": True,
        "forbidden_inputs": ["reward_as_state", "previous_reward", "region_id", "action_history",
                             "joint_action_input"]}
    C["E4IT_normalization_contract.json"] = {
        "normalization": NORM_V3, "v3_constants_sha256": v3_sha, "identical_both_arms": True,
        "no_refit": True, "no_running_statistics": True, "no_adaptation": True,
        "no_online_quantile_update": True}
    C["E4IT_reward_versions.json"] = {
        "CONTROL": {"arm": "R0_CONTROL", "reward_version": REWARD_R0_NAME,
                    "alpha_global": ALPHA_G_DEFAULT, "alpha_local": ALPHA_L_DEFAULT, "beta": BETA_DEFAULT,
                    "source": "src/prdrl/rl/reward.py (historical, unchanged)"},
        "TREATMENT": {"arm": "R1_TREATMENT", "reward_version": REWARD_R1_NAME, "alpha_global": 1.0,
                      "alpha_local": 0.1, "beta": 0.01, "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
                      "source": "src/prdrl/rl/reward_r1.py (frozen)"},
        "no_third_candidate": True, "no_R1.1": True, "no_adjusted_alpha_variant": True,
        "no_counterfactual_variant": True}
    C["E4IT_r1_config_integrity.json"] = {
        "reward_config_sha256": FROZEN_R1_CONFIG_SHA256, "verified": RI.verify_r1_config_hash(r1cfg),
        "alpha_local": 0.1, "no_runtime_alpha_derivation": True, "no_scale_refit": True,
        "no_new_diagnostic": True}
    C["E4IT_network_contract.json"] = {
        "network": "9-128-128-4", "layers": "9->Linear(128)->ReLU->Linear(128)->ReLU->Linear(4)",
        "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"], "parameter_shared_regional_ddqn": True,
        "identical_both_arms": True,
        "double_dqn_target": "a*=argmax_a Q_online(s',a); y=r+gamma*Q_target(s',a*) (nonterminal)",
        "joint_representation": "IMPLICIT_STATE_ONLY", "no_centralized_critic": True, "no_QMIX": True,
        "no_VDN": True, "no_autoregressive": True, "no_action_communication": True,
        "no_other_region_action_input": True, "branch_C_unmodified": True}
    C["E4IT_hyperparameters.json"] = {
        "gamma": 0.99, "optimizer": "Adam", "learning_rate": 1e-4, "loss": "SmoothL1", "batch_size": 128,
        "replay_capacity": 50000, "warmup_regional_transitions": 1000,
        "update_every_appended_transitions": 4, "gradient_steps_per_update": 1, "tau": 0.005,
        "gradient_clipping": None, "epsilon_train": "1.0->0.05 linear over 50000 regional transitions",
        "epsilon_validation": 0.0, "epsilon_final": 0.0, "identical_both_arms": True}
    C["E4IT_operator_contract.json"] = {
        "KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
        "predict_insufficient_history_fallback": "LOCAL", "sigma_D": 0.15,
        "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted standard NSGA-II",
        "identical_both_arms": True}
    C["E4IT_seed_split.json"] = {
        "train": [1501, 1560], "validation": [1561, 1580], "final": [1581, 1640],
        "n_train": 60, "n_val": 20, "n_final": 60, "policy_seeds": POLICY_SEEDS,
        "env_disjoint_from_all_history": env_disjoint, "policy_disjoint": pol_disjoint,
        "internal_split_disjoint": internal_disjoint, "diagnostic_1701_1720_excluded": True,
        "checked_against": list(CONSUMED_SEED_SETS.keys())}
    C["E4IT_policy_pairing_contract.json"] = {
        "policy_seeds": POLICY_SEEDS, "matched_pairs": 5,
        "pairing_key_for_env_schedule": ["policy_seed", "scenario", "training_environment_seed",
                                         "logical_episode_index"],
        "same_deterministic_environment_schedule_per_pair": True,
        "controller_streams_isolated": True, "no_shared_mutable_replay_or_rng": True}
    C["E4IT_initialization_pairing_contract.json"] = {
        "principle": "R0_p and R1_p start from identical online/target parameters (arm-independent, "
                     "seed = policy_seed*1000+0)", "online_equals_target_at_t0": True,
        "R0_equals_R1_init_at_t0": True, "per_policy": init_rows, "all_paired": paired_ok}
    C["E4IT_rng_contract.json"] = {
        "distinct_logical_streams": ["environment", "exploration", "operators", "replay_sampling",
                                     "network_initialization"],
        "environment_schedule_paired": True,
        "no_action_dependent_common_random_numbers_altering_production": True}
    C["E4IT_training_budget.json"] = {
        "epochs_per_run": 6, "seeds_per_epoch": 60, "scenarios_per_epoch": 3, "episodes_per_epoch": 180,
        "episodes_per_run": 1080, "runs": 10, "arms": 2, "policies": 5,
        "total_training_episodes": 10800, "changes_per_episode": 10, "no_early_stopping": True,
        "no_extra_epoch": True, "no_outcome_dependent_extension": True,
        "scenario_balance": "equal PERSISTENT/STATIONARY/MIXED per epoch; no MIXED oversampling",
        "run_isolation": ["online", "target", "optimizer", "replay", "epsilon_state", "controller_rng"],
        "no_replay_sharing_across_arms_or_policies": True}
    C["E4IT_logging_contract.json"] = {
        "identity": ["arm", "policy_seed", "epoch", "scenario", "environment_seed", "episode",
                     "change_index", "region_id"],
        "action": ["requested_action", "realized_operator", "fallback"],
        "quality": ["L_pre", "L_post"], "global": ["raw_global_reward", "weighted_global_reward"],
        "R0_local": ["raw_local_reward_R0", "weighted_local_reward_R0"],
        "R1_local": ["g_r", "local_active_R1", "weighted_local_reward_R1"],
        "fe": ["raw_fe_cost", "weighted_fe_penalty"], "actual": ["actual_reward_version", "actual_total_reward"],
        "reconstruction_assertion": {"R0": "actual_total = weighted_global + weighted_local_R0 - weighted_FE",
                                     "R1": "actual_total = weighted_global + weighted_local_R1 - weighted_FE"},
        "shadow_logging": {"status": "OPTIONAL", "zero_extra_FE": True, "no_action_influence": True,
                           "no_cross_arm_replay_reward_influence": True,
                           "replay_uses_arm_specific_reward_only": True},
        "action_diagnostics": ["KEEP_fraction", "LOCAL_fraction", "PREDICT_fraction", "DIVERSIFY_fraction",
                               "within_event_multi_action_rate", "action_entropy"],
        "action_diagnostics_not_primary_mechanism": True,
        "reward_trajectory_diagnostics": ["local_active_fraction", "median_abs_weighted_global",
                                          "median_abs_weighted_local_active", "global_local_active_ratio",
                                          "event_regional_local_reward_dispersion", "total_reward_distribution"]}
    C["E4IT_checkpoint_contract.json"] = {
        "schedule_transitions": [10000, 20000, 30000, 40000, 50000], "plus_final": True,
        "checkpoints_per_run": 6, "total_future_checkpoints": 60,
        "metadata_required": ["experiment=E4IT_REWARD_AB", "arm", "policy_seed", "transition_count",
                              "normalization_version", "v3_constants_sha256", "reward_version",
                              "reward_config_sha256 (R1)", "architecture", "optimizer_config_hash",
                              "protocol_hash"],
        "version_guard": {"R0": REWARD_R0_NAME, "R1": REWARD_R1_NAME,
                          "R1_requires_config_sha": FROZEN_R1_CONFIG_SHA256, "mismatch": "hard_error"}}
    C["E4IT_validation_contract.json"] = {
        "val_seeds": [1561, 1580], "scenarios": SCENARIOS, "epsilon": 0.0, "eval": True,
        "no_gradients": True, "no_replay_updates": True, "no_optimizer": True, "no_target_updates": True,
        "validation_matrix": {"arms": 2, "policies": 5, "checkpoints": 6, "scenarios": 3, "environments": 20,
                              "total_validation_episodes": 3600}, "every_checkpoint_evaluated": True}
    C["E4IT_primary_hypothesis.json"] = {
        "question": "With V3 normalization fixed in both arms, does replacing R0's local credit "
                    "formulation with R1 improve final MIXED response quality?",
        "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "cell_effect": "d[p,e,MIXED] = MIGD_response_R0[p,e] - MIGD_response_R1[p,e]",
        "positive_means": "R1_better", "secondary": ["MIGD_epoch", "response_FE"],
        "secondary_effect": "R0 - R1", "positive_FE_means": "R1_fewer_FE",
        "fe_cannot_rescue_harmful_migd": True, "DELTA_NORM": PRACTICAL_MARGIN,
        "checkpoint_selection_not_affected_by_reward": True}
    C["E4IT_mechanism_endpoints.json"] = {
        "primary_mechanism": "region-specific credit differentiation (NOT action diversity; V3 fixed that)",
        "endpoint_1_LOCAL_CREDIT_ACTIVITY": {
            "R0": "nonzero fraction of R0 local reward (E4I LOCAL_REWARD_NONZERO_FRACTION semantics)",
            "R1": "fraction with |g_r| > 1e-12", "expected": "R1 active fraction > R0 active fraction"},
        "endpoint_2_GLOBAL_TO_LOCAL_WEIGHTED_MAGNITUDE_RATIO": {
            "definition": "median|weighted_global| / (median|weighted_local| + eps) on matched active subset",
            "expected": "R1 ratio < R0 ratio",
            "zero_active_handling": "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL (no divide by zero; ratio 0 not 'better')"},
        "endpoint_3_REGIONAL_REWARD_DISPERSION_WITHIN_EVENT": {
            "metric": "population std (ddof=0) of weighted local reward across valid active regions per event, "
                      "then mean event-level dispersion", "single_active_region_event": 0.0,
            "no_valid_active_region_event": "excluded", "expected": "R1 > R0", "no_switch_to_MAD": True},
        "REWARD_CREDIT_MECHANISM_SUPPORTED": "YES iff >=2 of 3 endpoints in expected direction",
        "significance_not_required": True,
        "mechanism_data": "training/validation reward logs (descriptive) + final-evaluation trajectories "
                          "(reward primitives available without learning)"}
    C["E4IT_statistical_design.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "policy_axis": 5, "environment_axis": 60,
        "resample_both_with_replacement": True, "full_sampled_cartesian_product": True,
        "preserve_R0_R1_pairing": True, "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL,
        "CI_METHOD": CI_METHOD, "BOOTSTRAP_MASTER_SEED": REWARD_BOOTSTRAP_MASTER_SEED,
        "effect": "R0 - R1", "positive_favors": "R1",
        "macro": "equal 1/3 with SAME resampled policy+env indices reused across scenarios",
        "practical_margin": PRACTICAL_MARGIN}
    C["E4IT_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED] = mean_e(MIGD_R0[p,e] - MIGD_R1[p,e])",
        "rule": "REWARD_POLICY_ROBUSTNESS = PASS iff >=4 of 5 policy pairs > 0", "threshold": 4}
    C["E4IT_safety_contract.json"] = {
        "scenarios": ["PERSISTENT", "STATIONARY"], "endpoints": ["MIGD_response", "MIGD_epoch"],
        "safety_pass": "neither endpoint HARMFUL",
        "macro_safety": "neither macro MIGD_response nor macro MIGD_epoch HARMFUL",
        "fe_cannot_rescue_harmful": True}
    C["E4IT_success_logic.json"] = {
        "REWARD_CAUSAL_VALIDATED_requires_all": ["MIXED MIGD_response BENEFICIAL",
            "REWARD_CREDIT_MECHANISM_SUPPORTED", "REWARD_POLICY_ROBUSTNESS PASS",
            "PERSISTENT/STATIONARY/MACRO safety PASS", "checkpoint integrity PASS",
            "reward-version/config integrity PASS", "final protocol integrity PASS", "no final-data tuning"],
        "PERFORMANCE_PARTIAL": "MIXED in {BENEFICIAL, DIRECTIONAL_POSITIVE}, no safety HARMFUL, but a "
                               "support gate (mechanism/robustness) fails",
        "MECHANISTIC_PARTIAL": "MIXED PRACTICALLY_NEUTRAL AND mechanism supported AND all safety PASS AND "
                               "integrity PASS -> PERFORMANCE_IMPROVEMENT_CLAIM=NOT_SUPPORTED",
        "FAILED": "MIXED in {DIRECTIONAL_NEGATIVE, HARMFUL} OR any safety FAIL OR integrity fail",
        "mechanism_cannot_rescue_harmful": True,
        "classification_labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL,
                                  DIRECTIONAL_NEGATIVE, HARMFUL], "DELTA_NORM": DELTA_E4,
        "source": "reward_integration.reward_ab_verdict (== final_verdict.normalization_ab_verdict logic, "
                  "relabeled REWARD_CAUSAL_*)",
        "validation_may_not_produce_validated": True,
        "scope": "reward/credit causality under V3 + implicit joint-action; NOT joint-arch adequacy, "
                 "global optimality, or superiority over GLOBAL_PREDICT"}
    C["E4IT_final_firewall.json"] = {
        "final_seeds": [1581, 1640], "untouched_through": ["E4IT", "E4ITR", "validation", "debugging",
                                                           "checkpoint_selection"],
        "REWARD_TRAIN_ACCESS_COUNT": _reward_access["train"],
        "REWARD_VALIDATION_ACCESS_COUNT": _reward_access["validation"],
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_access["final"],
        "future_stages": ["E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
                          "E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION"]}

    for name, obj in C.items():
        _jdump(obj, RESULTS_DIR / name)

    # verify success logic branches (using the frozen reward_ab_verdict)
    _verify_success_logic()

    protocol_hash = hashlib.sha256(json.dumps({k: C[k] for k in sorted(C)}, sort_keys=True,
                                              default=str).encode()).hexdigest()
    _jdump({"stage": "E4IT_REWARD_AB_TRAINING_PROTOCOL_FREEZE", "experiment_factor": "REWARD_ONLY",
            "control": "V3_PLUS_R0", "treatment": "V3_PLUS_R1_ALPHA_0P1", "fixed_normalization": NORM_V3,
            "single_factor_diff": diff, "paired_init_ok": paired_ok, "protocol_hash": protocol_hash,
            "components": list(C.keys()), "next_stage": "E4ITR_REWARD_AB_TRAINING_EXECUTION"},
           RESULTS_DIR / "E4IT_protocol_frozen.json")

    _jdump({"real_environment_episodes": 0, "training_episodes": 0, "optimizer_steps": 0,
            "replay_training_appends": 0, "new_policy_checkpoints": 0, "model_weight_changes": 0,
            "note": "paired-init networks were built for hashing only, never trained or saved"},
           RESULTS_DIR / "E4IT_no_training_audit.json")

    verdict = build_verdict(hist, diff, paired_ok, env_disjoint, pol_disjoint, internal_disjoint,
                            integ_ok)
    _jdump(verdict, RESULTS_DIR / "E4IT_verdict.json")
    write_report(verdict, hist)
    emit_terminal(verdict)
    print(f"[E4IT] wall {time.time()-t0:.1f}s")
    return verdict


def _verify_success_logic():
    def V(mp, mech, rob):
        return RI.reward_ab_verdict(mixed_primary_label=mp, mechanism_supported=mech,
                                    policy_robustness_pass=rob, persistent_safety_pass=True,
                                    stationary_safety_pass=True, macro_safety_pass=True,
                                    checkpoint_integrity_pass=True, protocol_integrity_pass=True,
                                    version_integrity_pass=True, no_final_tuning=True)
    assert V(BENEFICIAL, True, True)["verdict"] == "REWARD_CAUSAL_VALIDATED"
    assert V(BENEFICIAL, False, True)["subtype"] == "PERFORMANCE_PARTIAL"
    assert V(PRACTICALLY_NEUTRAL, True, False)["subtype"] == "MECHANISTIC_PARTIAL"
    assert V(DIRECTIONAL_NEGATIVE, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"
    assert V(HARMFUL, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"
    assert V(PRACTICALLY_NEUTRAL, False, True)["verdict"] == "REWARD_CAUSAL_FAILED"


def build_verdict(hist, diff, paired_ok, env_disjoint, pol_disjoint, internal_disjoint, integ_ok):
    return {
        "stage": "E4IT_REWARD_AB_TRAINING_PROTOCOL_FREEZE",
        "E4IT_COMPLETE": "YES" if integ_ok else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "SELECTED_BRANCH": "B_REWARD_CREDIT", "EXPERIMENT_FACTOR": "REWARD_ONLY",
        "FIXED_NORMALIZATION": NORM_V3, "CONTROL": "V3_PLUS_ORIGINAL_REWARD_R0",
        "TREATMENT": "V3_PLUS_REWARD_R1_REGIONAL_QUALITY_GAIN", "R0_ALPHA_LOCAL": 0.1, "R1_ALPHA_LOCAL": 0.1,
        "R1_REWARD_CONFIG_SHA256": FROZEN_R1_CONFIG_SHA256, "STATE_DIM": 9, "NETWORK": "9-128-128-4",
        "JOINT_ARCHITECTURE_CHANGED": "NO", "SINGLE_FACTOR_DIFF": diff["differing_fields"],
        "SINGLE_FACTOR_OK": diff["single_factor"],
        "TRAIN_SEEDS": "1501..1560", "VALIDATION_SEEDS": "1561..1580", "FINAL_SEEDS": "1581..1640",
        "POLICY_SEEDS": "4001..4005", "SEEDS_DISJOINT": env_disjoint and pol_disjoint and internal_disjoint,
        "PAIRED_INITIALIZATION": "FROZEN" if paired_ok else "FAIL",
        "TRAINING_EPISODES_PER_RUN": 1080, "TOTAL_FUTURE_RUNS": 10, "TOTAL_FUTURE_TRAINING_EPISODES": 10800,
        "CHECKPOINT_SELECTION": "VALIDATION_MACRO_MIGD_RESPONSE", "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE", "PRIMARY_EFFECT": "R0_MINUS_R1", "POSITIVE_MEANS": "R1_BETTER",
        "PRACTICAL_MARGIN": PRACTICAL_MARGIN,
        "REWARD_MECHANISM_ENDPOINTS": "LOCAL_CREDIT_ACTIVITY,GLOBAL_LOCAL_MAGNITUDE_RATIO,REGIONAL_REWARD_DISPERSION",
        "REWARD_MECHANISM_GATE": "AT_LEAST_2_OF_3_EXPECTED_DIRECTIONS",
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_R1_BETTER",
        "BOOTSTRAP": "PAIRED_CROSSED_TWO_AXIS_5x60", "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": REWARD_BOOTSTRAP_MASTER_SEED,
        "REWARD_TRAIN_ACCESS_COUNT": 0, "REWARD_VALIDATION_ACCESS_COUNT": 0,
        "REWARD_FINAL_EVAL_ACCESS_COUNT": 0, "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4ITR_REWARD_AB_TRAINING_EXECUTION" if integ_ok else "E4IT_INTEGRITY_REVIEW"),
    }


def write_report(v, hist):
    L = ["# E4IT — Reward A/B Training Protocol Freeze (Branch B)", "",
         f"**E4IT_COMPLETE = {v['E4IT_COMPLETE']} → NEXT_STAGE = {v['NEXT_STAGE']}**", "",
         "Freeze only. No training, no reward-seed instantiation (1501..1640 untouched). The single "
         "scientific factor is the local-credit reward FORMULATION; both arms share corrected V3 state and "
         "alpha_local=0.1.", "",
         "## A. Historical result preservation",
         f"- E4F = FAILED; E4HF normalization = FAILED; E4II R1_IMPLEMENTATION_VALID = YES; V3 constants + "
         "reward_r1.py byte-stable; reward-version guard PASS.", "",
         "## B. Why branch B is now tested",
         "- E4HF: V3 fixed the collapse mechanism but not performance. E4I/E4IDD: R0 local reward inert by "
         "scale; R1 bounded gain active + material at alpha=0.1. Now test reward FORMULATION causally.", "",
         "## C/D. Control, treatment, single-factor",
         f"- CONTROL V3+R0 (alpha_local 0.1); TREATMENT V3+R1 (alpha_local 0.1, config {FROZEN_R1_CONFIG_SHA256}). "
         f"Arm-config scientific diff = {v['SINGLE_FACTOR_DIFF']} (reward only).", "",
         "## E/F/G/H. V3 lock, R0/R1 rewards, equal alpha",
         "- Both arms V3 (no refit). R0 historical unchanged; R1 = weighted_global + 0.1*g_r - weighted_FE. "
         "alpha_local_R0 = alpha_local_R1 = 0.1 → tests FORMULATION, not a larger coefficient.", "",
         "## I. State/network/algorithm lock",
         "- state_dim 9, 9-128-128-4 Shared Double DQN, IMPLICIT_STATE_ONLY (no CTDE/QMIX/VDN/autoregressive); "
         "reward never enters the state.", "",
         "## J/K. Fresh seed split + pairing/initialization",
         f"- train 1501..1560, val 1561..1580, final 1581..1640, policy 4001..4005; disjoint = "
         f"{v['SEEDS_DISJOINT']}. Paired init FROZEN: per policy online==target and R0==R1 init "
         f"({v['PAIRED_INITIALIZATION']}).", "",
         "## L. Training budget",
         f"- 6 epochs × 180 episodes = 1080/run; 10 runs (2 arms × 5 policies) = 10800 episodes; 10 changes "
         "each; equal scenario balance; run isolation; no replay sharing; no early stop.", "",
         "## M/N. Reward logging + credit-mechanism endpoints",
         "- log L_pre/L_post/global/R0-local/R1-local/FE/actual; reconstruction assertions per arm; optional "
         "shadow logging cannot affect replay. Mechanism (≥2/3): (1) local-credit activity R1>R0, "
         "(2) global/local weighted ratio R1<R0, (3) regional reward dispersion (std ddof=0) R1>R0; "
         "zero-active ⇒ NOT_APPLICABLE. Action diversity is descriptive only.", "",
         "## O. Validation + checkpoint selection",
         "- eval every checkpoint on 1561..1580 (3600 episodes); selection = min VALIDATION_MACRO_MIGD_RESPONSE "
         "(ties: macro epoch, response FE, earlier ckpt); reward magnitude NOT used; all 5 policies retained.", "",
         "## P/Q/R. Statistics, robustness, safety",
         f"- paired crossed two-axis bootstrap 5×60, N=50000, 95% percentile, master seed "
         f"{REWARD_BOOTSTRAP_MASTER_SEED}, effect R0−R1; robustness ≥4/5; PERSISTENT/STATIONARY/macro safety "
         "= not HARMFUL; FE cannot rescue.", "",
         "## S. Final scientific verdict logic",
         "- REWARD_CAUSAL_VALIDATED / PERFORMANCE_PARTIAL / MECHANISTIC_PARTIAL / FAILED "
         "(reward_integration.reward_ab_verdict; mechanism never rescues harmful; validation may not produce "
         "VALIDATED — untouched final seeds decide).", "",
         "## T/U. Final-seed firewall + no-training audit",
         "- 1581..1640 untouched through E4IT/E4ITR; access counts 0/0/0; no environment episode, optimizer "
         "step, replay append, or checkpoint (paired-init nets built for hashing only).", "",
         f"## V. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4IT_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    for k in ["E4IT_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "SELECTED_BRANCH", "EXPERIMENT_FACTOR", "FIXED_NORMALIZATION", "CONTROL", "TREATMENT",
              "R0_ALPHA_LOCAL", "R1_ALPHA_LOCAL", "R1_REWARD_CONFIG_SHA256", "STATE_DIM", "NETWORK",
              "JOINT_ARCHITECTURE_CHANGED", "TRAIN_SEEDS", "VALIDATION_SEEDS", "FINAL_SEEDS",
              "POLICY_SEEDS", "PAIRED_INITIALIZATION", "TRAINING_EPISODES_PER_RUN", "TOTAL_FUTURE_RUNS",
              "TOTAL_FUTURE_TRAINING_EPISODES", "CHECKPOINT_SELECTION", "PRIMARY_SCENARIO",
              "PRIMARY_ENDPOINT", "PRIMARY_EFFECT", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
              "REWARD_MECHANISM_ENDPOINTS", "REWARD_MECHANISM_GATE", "POLICY_ROBUSTNESS_RULE",
              "BOOTSTRAP", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_MASTER_SEED", "REWARD_TRAIN_ACCESS_COUNT",
              "REWARD_VALIDATION_ACCESS_COUNT", "REWARD_FINAL_EVAL_ACCESS_COUNT", "NEW_TRAINING",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<36} = {v[k]}")


if __name__ == "__main__":
    main()
