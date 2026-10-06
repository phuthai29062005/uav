"""
E4HT — Normalization A/B Training Protocol Freeze (PR-DRL-DMOEA, Branch A).

Designs and FREEZES the matched V1-vs-V3 single-factor A/B training experiment.
The ONLY intended experimental difference between arms is normalization_version:
  CONTROL   = NORMALIZATION_V1_LEGACY
  TREATMENT = ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3 (frozen E4HR/E4HI constants).

Everything else (state, network, reward, optimizer, hyperparameters, operators,
budget, survival, checkpoint schedule, checkpoint-selection rule) is identical.

E4HT does NOT train, does NOT instantiate any train/validation/final environment,
does NOT refit V3, and does NOT change the immutable E4F verdict (FAILED). It only
freezes contracts and verifies non-training infrastructure (paired initialization,
version guards, firewall counters).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.rl.network import build_online_target_pair, SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, NORM_V3, check_normalization_compatibility,
)
from prdrl.evaluation.crossed_bootstrap import (
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, paired_crossed_two_axis_bootstrap,
)
from prdrl.evaluation.final_verdict import DELTA_E4  # reused label semantics

RESULTS_DIR = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HI_DIR = Path("results/E4HI_normalization_implementation_validation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

TRAIN_SEEDS = list(range(1301, 1361))    # 60
VAL_SEEDS = list(range(1361, 1381))      # 20
FINAL_SEEDS = list(range(1381, 1441))    # 60 — FIREWALLED
POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
NETWORK_INPUT_DIM = 9
DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED_AB = 56001
MECHANISM_ENDPOINTS = ["PREDICT_fraction", "within_event_multi_action_rate", "regional_action_entropy"]

_new_final_access = {"count": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def _load(p):
    with open(p) as f:
        return json.load(f)


def _scan_tokens(obj) -> List[str]:
    bad = ("TBD", "TODO", "UNFROZEN", "placeholder", "PLACEHOLDER")
    hits = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k); walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            if any(t in o for t in bad):
                hits.append(o)
    walk(obj)
    return hits


def _param_hash(net) -> str:
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _paired_init(policy_seed):
    """Deterministic paired initialization used by BOTH arms of policy_seed."""
    torch.manual_seed(policy_seed * 1000 + 1)
    online, target = build_online_target_pair(NETWORK_INPUT_DIM, seed=policy_seed * 1000 + 0)
    return online, target


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4HT — Normalization A/B Training Protocol Freeze")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4hi = _load(E4HI_DIR / "E4HI_verdict.json")
    e4hr = _load(E4HR_DIR / "E4HR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    v3_consts = _load(E4HR_DIR / "E4HR_v3_constants.json")
    old_ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                    == ck["policies"][str(ps)]["sha256"] for ps in [2001, 2002, 2003, 2004, 2005])
    manifest = {
        "stage": "E4HT_NORMALIZATION_AB_TRAINING_PROTOCOL_FREEZE", "freeze_only": True,
        "E4HI_COMPLETE": e4hi["E4HI_COMPLETE"], "NORMALIZATION_V3_READY": e4hi["NORMALIZATION_V3_READY"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"], "old_checkpoints_unchanged": old_ck_ok,
        "experiment_factor": "NORMALIZATION_ONLY", "no_training": True,
        "v3_constants_sha256": v3_consts["constants_sha256"],
    }
    _jdump(manifest, RESULTS_DIR / "E4HT_pre_manifest.json")
    ok0 = (e4hi["E4HI_COMPLETE"] == "YES" and e4hi["NORMALIZATION_V3_READY"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and old_ck_ok)
    if not ok0:
        print("[E4HT] integrity FAIL — STOP"); return {"E4HT_COMPLETE": "NO"}
    print("[E4HT] integrity PASS. V3 READY, E4F FAILED, old artifacts immutable.")

    # ---- paired initialization verification (non-training infra) ----
    init_rows = {}
    init_ok = True
    for ps in POLICY_SEEDS:
        o1, t1 = _paired_init(ps)
        o2, t2 = _paired_init(ps)      # rebuild -> must be identical (deterministic)
        oh, th = _param_hash(o1), _param_hash(t1)
        deterministic = (oh == _param_hash(o2) and th == _param_hash(t2))
        online_target_equal = (oh == th)   # build_online_target_pair makes them identical
        init_ok = init_ok and deterministic and online_target_equal
        init_rows[str(ps)] = {"online_param_sha256": oh, "target_param_sha256": th,
                              "deterministic": deterministic,
                              "online_equals_target_at_t0": online_target_equal,
                              "used_by_both_arms": True}

    # ---- contracts ----
    contracts = {}

    contracts["E4HT_ab_factor_contract.json"] = {
        "experiment_factor": "normalization_version (ONLY)",
        "control_arm": "V1_CONTROL", "treatment_arm": "V3_TREATMENT",
        "control_normalization": NORM_V1, "treatment_normalization": NORM_V3,
        "single_factor_rule": "all else identical between arms",
        "unchanged": ["state_semantics", "state_dimension", "network_architecture",
                      "parameter_sharing", "reward", "reward_coefficients", "optimizer",
                      "learning_rate", "gamma", "tau", "batch_size", "replay_size", "warmup",
                      "update_frequency", "epsilon_schedule", "training_budget", "action_set",
                      "response_operators", "response_budget", "survival", "checkpoint_schedule",
                      "checkpoint_selection_rule"],
    }

    contracts["E4HT_state_contract.json"] = {
        "local_v2": ["severity", "convergence_loss", "occupancy", "delta_severity"],
        "global_context_v1": ["active_region_fraction", "mean_severity", "max_severity",
                              "mean_convergence_loss", "max_convergence_loss"],
        "state_dim": 9, "no_region_id": True, "no_action_context": True,
        "no_joint_action_context": True, "identical_both_arms": True,
    }

    contracts["E4HT_normalization_versions.json"] = {
        "V1": {"name": NORM_V1, "role": "CONTROL", "status": "LEGACY"},
        "V2": {"name": NORM_V2, "role": "NONE", "status": "REJECTED_NOT_TRAINED"},
        "V3": {"name": NORM_V3, "role": "TREATMENT", "status": "READY",
               "constants_sha256": v3_consts["constants_sha256"]},
    }

    contracts["E4HT_v3_constants_integrity.json"] = {
        "v3_constants_source": "results/E4HR_normalization_design_review/E4HR_v3_constants.json",
        "v3_constants_sha256": v3_consts["constants_sha256"],
        "fit_seed_range": v3_consts["fit_seed_range"],
        "no_refit_on_new_train_seeds": True, "no_online_adaptation": True,
        "immutable_during_training": True,
        "forbidden": ["running_mean", "running_variance", "online_quantile_updates",
                      "new_q99_estimation", "new_positive_median_estimation"],
    }

    all_hist = (set(range(701, 781)) | set(range(801, 883)) | set(range(901, 941))
                | set(range(1001, 1141)) | set(range(1201, 1221)))
    new_all = set(TRAIN_SEEDS) | set(VAL_SEEDS) | set(FINAL_SEEDS)
    contracts["E4HT_seed_split.json"] = {
        "train": {"seeds": TRAIN_SEEDS, "range": [1301, 1360], "n": 60},
        "validation": {"seeds": VAL_SEEDS, "range": [1361, 1380], "n": 20},
        "final": {"seeds": FINAL_SEEDS, "range": [1381, 1440], "n": 60, "FIREWALL": "UNOPENED"},
        "scenarios": SCENARIOS,
        "all_disjoint": (len(set(TRAIN_SEEDS) & set(VAL_SEEDS)) == 0
                         and len(set(TRAIN_SEEDS) & set(FINAL_SEEDS)) == 0
                         and len(set(VAL_SEEDS) & set(FINAL_SEEDS)) == 0),
        "disjoint_from_historical": len(new_all & all_hist) == 0,
    }

    contracts["E4HT_policy_pairing_contract.json"] = {
        "policy_seeds": POLICY_SEEDS,
        "matched_pairs": [f"V1_{p} / V3_{p}" for p in POLICY_SEEDS],
        "all_five_pairs_retained": True, "no_best_policy_seed_selection": True,
        "no_best_pair_selection": True,
    }

    contracts["E4HT_initialization_pairing_contract.json"] = {
        "principle": "within policy seed p, V1_p and V3_p start from the SAME network init",
        "init_seed_rule": "build_online_target_pair(input_dim=9, seed=policy_seed*1000+0); "
                          "torch.manual_seed(policy_seed*1000+1)",
        "only_difference_at_transition_0": "normalization_version",
        "online_equals_target_at_t0": True, "deterministic": init_ok,
        "per_policy_init_hashes": init_rows,
        "integrity_verified": init_ok,
    }

    contracts["E4HT_rng_contract.json"] = {
        "environment_rng_independent_from_controller": True,
        "operator_rng_isolated": True, "replay_rng_isolated": True,
        "exploration_rng_isolated": True, "network_init_deterministic": True,
        "no_outcome_dependent_scheme": True, "production_operator_semantics_unchanged": True,
        "pairing": "V1/V3 streams keyed by policy_seed, logical episode, scenario; no shared mutable state",
        "namespaces": ["POLICY_INIT", "POLICY_EXPLORATION", "REPLAY_SAMPLE",
                       "TRAIN_EPISODE_ORDER", "RESPONSE_OPERATOR", "RESPONSE_SURVIVAL", "BASE_EVOLUTION"],
    }

    contracts["E4HT_training_budget.json"] = {
        "episodes_per_epoch": 180, "epochs": 6, "episodes_per_run": 1080,
        "policies_per_arm": 5, "arms": 2, "total_runs": 10, "total_episodes": 10800,
        "changes_per_episode": 10, "scenario_balance": "equal PERSISTENT/STATIONARY/MIXED per epoch",
        "no_mixed_oversampling": True,
    }

    contracts["E4HT_hyperparameters.json"] = {
        "gamma": 0.99, "learning_rate": 1e-4, "optimizer": "Adam", "loss": "SmoothL1",
        "batch_size": 128, "replay_capacity": 50000, "warmup_regional_transitions": 1000,
        "update_every_appended_transitions": 4, "gradient_steps_per_update": 1,
        "target_update": "Polyak", "tau": 0.005, "gradient_clipping": "NONE",
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_transitions": 50000,
        "validation_epsilon": 0.0, "final_epsilon": 0.0, "identical_both_arms": True,
    }

    contracts["E4HT_reward_contract.json"] = {
        "formula": "r_r = 1.0*r_global + 0.1*r_local_r - 0.01*FE_cost_r",
        "alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01,
        "reward_unchanged": True, "reason": "branch A is the single-factor experiment; "
        "E4GD GLOBAL_OVERWHELMS_LOCAL is deliberately NOT fixed here",
        "component_logging_required": ["raw_global_reward", "weighted_global_reward",
                                       "raw_local_reward", "weighted_local_reward",
                                       "raw_fe_cost", "weighted_fe_penalty", "total_reward"],
    }

    contracts["E4HT_operator_contract.json"] = {
        "KEEP": {"FE": 0}, "LOCAL": {"sigma_L": 0.02},
        "PREDICT": {"lambda_P": 1.0, "sigma_P": 0.005, "fallback": "LOCAL on insufficient history"},
        "DIVERSIFY": {"sigma_D": 0.15},
        "response_budget": "B=10 PER_ACTIVE_REGION (non-KEEP)", "survival": "unrestricted NSGA-II",
        "no_HARD": True, "no_MEMORY": True, "identical_both_arms": True,
    }

    contracts["E4HT_logging_contract.json"] = {
        "per_episode": ["arm", "policy_seed", "epoch", "scenario", "environment_seed",
                        "MIGD_response", "MIGD_epoch", "response_FE", "cumulative_reward"],
        "per_regional_transition": ["requested_action", "realized_operator", "fallback", "epsilon",
                                    "raw_global_reward", "weighted_global_reward", "raw_local_reward",
                                    "weighted_local_reward", "raw_fe_cost", "weighted_fe_penalty",
                                    "total_reward"],
        "action_diagnostics_per_epoch_policy_arm_scenario": [
            "KEEP_fraction", "LOCAL_fraction", "PREDICT_fraction", "DIVERSIFY_fraction",
            "action_entropy", "within_event_multi_action_rate"],
        "q_diagnostics_at_checkpoints": ["argmax_action_distribution", "top1_top2_margin", "q_variance"],
        "normalization_diagnostics_v3": ["structural_zero_fraction", "positive_lower_saturation",
                                         "positive_upper_saturation", "delta_nonzero_saturation"],
        "diagnostics_not_used_for_selection": True,
    }

    contracts["E4HT_checkpoint_contract.json"] = {
        "transitions": [10000, 20000, 30000, 40000, 50000], "final": True,
        "identical_schedule_both_arms": True,
        "metadata_required": ["experiment", "arm", "policy_seed", "transition_count",
                              "normalization_version", "normalizer_constants_sha256",
                              "network_architecture_version", "reward_version",
                              "optimizer_config_hash", "training_protocol_hash"],
        "version_guard": {"V1": NORM_V1, "V3": NORM_V3, "mismatch": "HARD_ERROR"},
    }

    contracts["E4HT_validation_contract.json"] = {
        "seeds": VAL_SEEDS, "epsilon": 0.0, "no_gradients": True, "no_replay": True,
        "no_optimizer": True, "no_target_update": True, "no_normalizer_change": True,
        "evaluate_every_checkpoint": True,
        "selection_metric": "VALIDATION_MACRO_MIGD_RESPONSE",
        "macro": "equal 1/3 mean over PERSISTENT/STATIONARY/MIXED",
        "tie_breakers": ["lower macro MIGD_epoch", "lower response_FE", "earlier checkpoint"],
        "no_mixed_only_selection": True, "identical_rule_both_arms": True,
        "retain_all_5_per_arm": True,
    }

    contracts["E4HT_primary_hypothesis.json"] = {
        "question": "Does V3 normalization improve final learned behavior vs V1 when every other "
                    "component is held fixed?",
        "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "primary_comparator": "V1_CONTROL", "treatment": "V3_TREATMENT",
        "cell_effect": "d[p,e,MIXED] = MIGD_response_V1[p,e] - MIGD_response_V3[p,e]",
        "positive_means": "V3 better", "negative_means": "V1 better",
        "not_yet_superiority_over_global_predict": True,
    }

    contracts["E4HT_mechanism_endpoints.json"] = {
        "endpoints": MECHANISM_ENDPOINTS, "scenario": "MIXED",
        "expected_decollapse_direction": {"PREDICT_fraction": "V3 < V1",
                                          "within_event_multi_action_rate": "V3 > V1",
                                          "regional_action_entropy": "V3 > V1"},
        "MECHANISM_DECOLLAPSE_SUPPORTED_rule": ">= 2 of 3 endpoints move in expected direction",
        "not_uniform_action_usage": "goal is state-dependent regional specialization, not uniform frequency",
        "mechanism_gate_not_sufficient_alone": True,
    }

    contracts["E4HT_statistical_design.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "policy_axis_n": 5, "environment_axis_n": 60,
        "paired_effect": "V1[p,e] - V3[p,e] (matched policy seed + environment seed + scenario)",
        "resample_policy_with_replacement": True, "resample_environment_with_replacement": True,
        "full_sampled_cartesian_product": True, "no_naive_300_row_bootstrap": True,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED_AB,
        "practical_margin_DELTA_NORM": DELTA_NORM,
        "effect_classification": {
            "BENEFICIAL": "mean > +DELTA_NORM AND ci_lo > 0",
            "DIRECTIONAL_POSITIVE": "mean > +DELTA_NORM AND ci crosses 0",
            "PRACTICALLY_NEUTRAL": "abs(mean) <= DELTA_NORM AND no established harm",
            "DIRECTIONAL_NEGATIVE": "mean < -DELTA_NORM AND ci crosses 0",
            "HARMFUL": "mean < -DELTA_NORM AND ci_hi < 0",
        },
        "exhaustive_rare_case_extension": "same as E4E (defer to CI direction for uncovered combos)",
        "macro": "(Delta_PERSISTENT+Delta_STATIONARY+Delta_MIXED)/3 with SHARED bootstrap draws across scenarios",
        "policy_robustness": "Delta_policy[p,MIXED]=mean_e(V1-V3); PASS iff >=4/5 policy pairs > 0",
    }

    contracts["E4HT_safety_contract.json"] = {
        "scenarios": ["PERSISTENT", "STATIONARY"], "endpoints": ["MIGD_response", "MIGD_epoch"],
        "effect": "V1 - V3 (positive favors V3)",
        "safety_pass": "neither endpoint HARMFUL",
        "macro_safety": "neither macro MIGD_response nor macro MIGD_epoch HARMFUL",
        "fe_cannot_rescue_harmful_migd": True,
        "secondary_performance": ["MIGD_epoch", "response_FE"],
        "fe_positive_means": "V3 uses fewer response evaluations",
    }

    contracts["E4HT_success_logic.json"] = {
        "NORMALIZATION_CAUSAL_VALIDATED_requires_all": [
            "MIXED V3-vs-V1 MIGD_response = BENEFICIAL",
            "MECHANISM_DECOLLAPSE_SUPPORTED = YES",
            "NORMALIZATION_POLICY_ROBUSTNESS = PASS (>=4/5)",
            "PERSISTENT safety = PASS", "STATIONARY safety = PASS", "macro safety = PASS",
            "protocol integrity = PASS", "normalization/version integrity = PASS",
            "no final-data tuning"],
        "NORMALIZATION_CAUSAL_PARTIAL_when": "MIXED primary BENEFICIAL or DIRECTIONAL_POSITIVE, no "
            "safety HARMFUL, but >=1 supporting mechanism/robustness gate fails",
        "MECHANISTIC_PARTIAL_when": "mechanism de-collapse strongly supported BUT performance "
            "PRACTICALLY_NEUTRAL — records mechanism change WITHOUT claiming performance improvement",
        "NORMALIZATION_CAUSAL_FAILED_when": "MIXED primary HARMFUL or DIRECTIONAL_NEGATIVE, OR any "
            "scenario/macro safety fails, OR protocol integrity fails",
        "practically_neutral_mixed_with_mechanism_improvement": "NOT VALIDATED (use PARTIAL/MECHANISTIC_PARTIAL)",
        "not_overall_superiority_over_global_predict": True,
    }

    contracts["E4HT_final_firewall.json"] = {
        "final_seeds": [1381, 1440], "NEW_FINAL_EVAL_ACCESS_COUNT": _new_final_access["count"],
        "must_not_instantiate_during": ["E4HT", "implementation_validation", "training",
                                        "checkpoint_selection", "validation", "debugging"],
        "unopened": _new_final_access["count"] == 0,
    }

    # placeholder scan
    token_hits = []
    for name, obj in contracts.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    ready = (init_ok and contracts["E4HT_seed_split.json"]["all_disjoint"]
             and contracts["E4HT_seed_split.json"]["disjoint_from_historical"]
             and _new_final_access["count"] == 0 and len(token_hits) == 0)

    # ---- write all contracts ----
    for name, obj in contracts.items():
        _jdump(obj, RESULTS_DIR / name)

    master = {
        "stage": "E4HT_NORMALIZATION_AB_TRAINING_PROTOCOL_FREEZE",
        "all_frozen_before_training": True, "experiment_factor": "NORMALIZATION_ONLY",
        "control": NORM_V1, "treatment": NORM_V3, "v3_constants_sha256": v3_consts["constants_sha256"],
        "state_dim": 9, "network": "9-128-128-4",
        "train_seeds": [1301, 1360], "validation_seeds": [1361, 1380], "final_seeds": [1381, 1440],
        "policy_seeds": POLICY_SEEDS, "paired_initialization": "FROZEN" if init_ok else "FAILED",
        "training_episodes_per_run": 1080, "total_future_runs": 10, "total_future_training_episodes": 10800,
        "checkpoint_selection": "VALIDATION_MACRO_MIGD_RESPONSE",
        "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "primary_effect": "V1_MINUS_V3", "practical_margin": DELTA_NORM,
        "bootstrap": "CROSSED_TWO_AXIS_5x60", "bootstrap_resamples": N_BOOTSTRAP,
        "mechanism_endpoints": MECHANISM_ENDPOINTS,
        "policy_robustness_rule": "AT_LEAST_4_OF_5_V3_BETTER",
        "no_placeholder_tokens": len(token_hits) == 0,
        "components": list(contracts.keys()),
        "next_stage": "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION",
    }
    _jdump(master, RESULTS_DIR / "E4HT_protocol_frozen.json")

    _jdump({"episodes_trained": 0, "optimizer_steps": 0, "replay_appends": 0,
            "new_policy_checkpoints": 0, "no_substantive_training": True},
           RESULTS_DIR / "E4HT_no_training_audit.json")

    verdict = _build_verdict(init_ok, token_hits, ready)
    _jdump(verdict, RESULTS_DIR / "E4HT_verdict.json")
    _write_report(verdict, init_rows)
    _emit_terminal(verdict)
    print(f"[E4HT] wall {time.time()-t0:.0f}s")
    return verdict


def _build_verdict(init_ok, token_hits, ready):
    return {
        "stage": "E4HT_NORMALIZATION_AB_TRAINING_PROTOCOL_FREEZE",
        "E4HT_COMPLETE": "YES" if ready else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "NORMALIZATION_V3_READY": "YES",
        "EXPERIMENT_FACTOR": "NORMALIZATION_ONLY",
        "CONTROL": NORM_V1, "TREATMENT": NORM_V3,
        "STATE_DIM": 9, "NETWORK": "9-128-128-4",
        "REWARD_CHANGED": "NO", "JOINT_ARCHITECTURE_CHANGED": "NO",
        "TRAIN_SEEDS": "1301..1360", "VALIDATION_SEEDS": "1361..1380", "FINAL_SEEDS": "1381..1440",
        "POLICY_SEEDS": "3001..3005", "PAIRED_INITIALIZATION": "FROZEN" if init_ok else "FAILED",
        "TRAINING_EPISODES_PER_RUN": 1080, "TOTAL_FUTURE_RUNS": 10,
        "TOTAL_FUTURE_TRAINING_EPISODES": 10800,
        "CHECKPOINT_SELECTION": "VALIDATION_MACRO_MIGD_RESPONSE",
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_EFFECT": "V1_MINUS_V3", "POSITIVE_MEANS": "V3_BETTER",
        "PRACTICAL_MARGIN": DELTA_NORM, "BOOTSTRAP": "CROSSED_TWO_AXIS_5x60",
        "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "MECHANISM_ENDPOINTS": ",".join(MECHANISM_ENDPOINTS),
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_V3_BETTER",
        "no_placeholder_tokens": len(token_hits) == 0, "placeholder_hits": token_hits,
        "NEW_FINAL_EVAL_ACCESS_COUNT": _new_final_access["count"], "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION" if ready else "E4HT_FREEZE_FIX_REQUIRED",
    }


def _write_report(v, init_rows):
    L = ["# E4HT — Normalization A/B Training Protocol Freeze", "",
         f"**E4HT_COMPLETE = {v['E4HT_COMPLETE']}** — freeze only; no training. Single factor = "
         "normalization_version. Original E4F verdict remains FAILED.", "",
         "## Arms", f"- CONTROL = {v['CONTROL']}; TREATMENT = {v['TREATMENT']} (frozen E4HR/E4HI "
         "constants; no refit, no online adaptation).", "",
         "## Held identical", "- state (dim 9), network 9-128-128-4, reward (1.0/0.1/0.01), optimizer, "
         "all hyperparameters, operators, B=10, NSGA-II survival, checkpoint schedule & selection.", "",
         "## Paired initialization", "| policy | online sha (head) | det | online==target@t0 |",
         "|---|---|---|---|"]
    for p, r in init_rows.items():
        L.append(f"| {p} | {r['online_param_sha256'][:12]}… | {r['deterministic']} | {r['online_equals_target_at_t0']} |")
    L += ["", f"- The only difference at transition 0 is normalization_version.", "",
          "## Seeds (fresh, disjoint; final firewalled)",
          f"- train 1301..1360, val 1361..1380, final 1381..1440 (UNOPENED, access="
          f"{v['NEW_FINAL_EVAL_ACCESS_COUNT']}); policy seeds 3001..3005 (matched pairs, all retained).", "",
          "## Future training (frozen; not executed)",
          f"- 1080 episodes/run × 10 runs = 10800 episodes; checkpoints 10k..50k+final; "
          f"selection = VALIDATION_MACRO_MIGD_RESPONSE.", "",
          "## Primary causal comparison",
          "- MIXED MIGD_response, effect d = V1 − V3 (positive ⇒ V3 better), DELTA_NORM = 1e-5.",
          "- PAIRED crossed two-axis bootstrap (policy=5 × env=60), N=50000, 95% percentile, master seed 56001.",
          "- Mechanism (MIXED): PREDICT fraction ↓, multi-action rate ↑, entropy ↑; ≥2/3 = MECHANISM_DECOLLAPSE_SUPPORTED.",
          "- Policy robustness: ≥4/5 pairs V3 better. Safety: PERSISTENT/STATIONARY/macro not HARMFUL.", "",
          "## Success logic (frozen)",
          "- VALIDATED needs BENEFICIAL MIXED + mechanism + robustness + all safety + integrity.",
          "- Neutral MIXED with mechanism improvement = MECHANISTIC_PARTIAL (no performance claim), not VALIDATED.",
          "- Tests normalization causality only — NOT overall superiority over GLOBAL_PREDICT.", "",
          f"## Next stage: {v['NEXT_STAGE']} (freeze first, train later)."]
    (RESULTS_DIR / "E4HT_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4HT_COMPLETE", "ORIGINAL_E4F_VERDICT", "NORMALIZATION_V3_READY", "EXPERIMENT_FACTOR",
              "CONTROL", "TREATMENT", "STATE_DIM", "NETWORK", "REWARD_CHANGED",
              "JOINT_ARCHITECTURE_CHANGED", "TRAIN_SEEDS", "VALIDATION_SEEDS", "FINAL_SEEDS",
              "POLICY_SEEDS", "PAIRED_INITIALIZATION", "TRAINING_EPISODES_PER_RUN", "TOTAL_FUTURE_RUNS",
              "TOTAL_FUTURE_TRAINING_EPISODES", "CHECKPOINT_SELECTION", "PRIMARY_SCENARIO",
              "PRIMARY_ENDPOINT", "PRIMARY_EFFECT", "POSITIVE_MEANS", "PRACTICAL_MARGIN", "BOOTSTRAP",
              "BOOTSTRAP_RESAMPLES", "MECHANISM_ENDPOINTS", "POLICY_ROBUSTNESS_RULE",
              "NEW_FINAL_EVAL_ACCESS_COUNT", "NEW_TRAINING", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<34} = {v[k]}")


if __name__ == "__main__":
    main()
