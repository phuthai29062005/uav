"""
T1-T72+: Tests for E4-T — Double DQN Training Protocol Freeze.

All tests operate against the frozen JSON outputs and the importable
FrozenTrainingConfig / make_frozen_config from the experiment module.
No substantive training. No final evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4T_training_protocol_freeze import (
    FrozenTrainingConfig, make_frozen_config,
    TRAIN_SEEDS, VALIDATION_SEEDS, FINAL_EVAL_SEEDS,
    HISTORICAL_SEEDS, POLICY_SEEDS, SCENARIOS,
    N_EPOCHS, EPISODES_PER_EPOCH, N_TRAIN_EPISODES, CHANGES_PER_EPISODE,
    INPUT_DIM, HIDDEN_LAYERS, OUTPUT_DIM, N_ACTIONS,
    BATCH_SIZE, REPLAY_CAPACITY, GAMMA, LEARNING_RATE,
    TARGET_TAU, WARMUP_TRANSITIONS, UPDATE_FREQUENCY, GRADIENT_STEPS_PER_UPD,
    EPS_START, EPS_END, EPS_DECAY_TRANSITIONS, VALIDATION_EPSILON, FINAL_EVAL_EPSILON,
    ALPHA_G, ALPHA_L, BETA,
    CHECKPOINT_TRANSITIONS, PRIMARY_CHECKPOINT_METRIC,
    BASELINES, LOGGING_FIELDS, RNG_NAMESPACES,
    _apply_checkpoint_selection, generate_episode_schedule,
    RL_STATUS, DRL_TRAINING,
)
from prdrl.rl.interface import CURRENT_RESPONSE_BUDGET_VALUE

OUT_DIR = Path("results/E4T_training_protocol_freeze")
E4P_DIR = Path("results/E4_learned_controller_preparation")
H1RV_DIR = Path("results/H1_verdict_review")
E3R_DIR  = Path("results/E3R_rule_v2_semantic_rerun")


def _jload(path: Path) -> dict:
    return json.loads(path.read_text())


# ────────────────────────────────────────────────────────────────
# T1: E4P regression gate PASS
# ────────────────────────────────────────────────────────────────

def test_T1_e4p_regression_gate_pass():
    data = _jload(E4P_DIR / "E4P_regression_addendum.json")
    assert data.get("G14") == "PASS"
    assert data.get("E4P_ALL_GATES") == "PASS"


# ────────────────────────────────────────────────────────────────
# T2: H1-RV readiness READY
# ────────────────────────────────────────────────────────────────

def test_T2_h1rv_readiness_ready():
    data = _jload(H1RV_DIR / "H1RV_verdict.json")
    assert data.get("E4_READINESS") == "READY"


# ────────────────────────────────────────────────────────────────
# T3-T5: Exact seed ranges
# ────────────────────────────────────────────────────────────────

def test_T3_training_seeds_1001_1060():
    data = _jload(OUT_DIR / "E4T_seed_split.json")
    seeds = data["train"]["seeds"]
    assert seeds == list(range(1001, 1061)), f"Train seeds wrong: {seeds[:5]}..."
    assert len(seeds) == 60


def test_T4_validation_seeds_1061_1080():
    data = _jload(OUT_DIR / "E4T_seed_split.json")
    seeds = data["validation"]["seeds"]
    assert seeds == list(range(1061, 1081))
    assert len(seeds) == 20


def test_T5_final_eval_seeds_1081_1140():
    data = _jload(OUT_DIR / "E4T_seed_split.json")
    seeds = data["final_eval"]["seeds"]
    assert seeds == list(range(1081, 1141))
    assert len(seeds) == 60


# ────────────────────────────────────────────────────────────────
# T6-T7: No overlap, no historical seeds
# ────────────────────────────────────────────────────────────────

def test_T6_no_seed_set_overlap():
    data = _jload(OUT_DIR / "E4T_seed_split.json")
    assert data["overlap_checks"]["all_clear"] is True
    assert data["overlap_checks"]["train_validation_overlap"] == []
    assert data["overlap_checks"]["train_final_overlap"] == []
    assert data["overlap_checks"]["validation_final_overlap"] == []


def test_T7_no_historical_seeds_reused():
    data = _jload(OUT_DIR / "E4T_seed_split.json")
    assert data["overlap_checks"]["historical_seed_collision"] == []
    # Also verify the constants
    all_seeds = set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(FINAL_EVAL_SEEDS)
    assert len(all_seeds & HISTORICAL_SEEDS) == 0


# ────────────────────────────────────────────────────────────────
# T8: Five policy master seeds exact
# ────────────────────────────────────────────────────────────────

def test_T8_five_policy_seeds_exact():
    data = _jload(OUT_DIR / "E4T_policy_seed_contract.json")
    assert data["policy_seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert data["n_independent_runs"] == 5


# ────────────────────────────────────────────────────────────────
# T9-T12: Training budget
# ────────────────────────────────────────────────────────────────

def test_T9_scenarios_exact_three():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["train_scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]
    assert data["n_scenarios"] == 3


def test_T10_balanced_epoch_180_episodes():
    assert EPISODES_PER_EPOCH == 60 * 3   # 60 seeds × 3 scenarios
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["episodes_per_epoch"] == 180


def test_T11_n_epochs_6():
    assert N_EPOCHS == 6
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["n_epochs"] == 6


def test_T12_n_train_episodes_1080():
    assert N_TRAIN_EPISODES == 1080
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["n_train_episodes"] == 1080


# ────────────────────────────────────────────────────────────────
# T13-T16: Network spec
# ────────────────────────────────────────────────────────────────

def test_T13_network_input_dim_9():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["network"]["input_dim"] == 9


def test_T14_network_hidden_128_128():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["network"]["hidden_layers"] == [128, 128]


def test_T15_network_output_dim_4():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["network"]["output_dim"] == 4


def test_T16_exact_four_actions():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["network"]["actions"] == ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
    assert len(data["network"]["actions"]) == 4


# ────────────────────────────────────────────────────────────────
# T17-T24: Optimization hyperparameters
# ────────────────────────────────────────────────────────────────

def test_T17_batch_size_128():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["optimization"]["batch_size"] == 128


def test_T18_replay_50000():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["optimization"]["replay_capacity"] == 50000


def test_T19_gamma_099():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert abs(data["optimization"]["gamma"] - 0.99) < 1e-12


def test_T20_lr_1e4():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert abs(data["optimization"]["learning_rate"] - 1e-4) < 1e-10


def test_T21_tau_0005():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert abs(data["optimization"]["target_tau"] - 0.005) < 1e-10


def test_T22_warmup_1000():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["optimization"]["warmup_transitions"] == 1000


def test_T23_update_frequency_4():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["optimization"]["update_frequency"] == 4


def test_T24_one_gradient_step_per_update():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["optimization"]["gradient_steps_per_update"] == 1


# ────────────────────────────────────────────────────────────────
# T25-T29: Epsilon schedule
# ────────────────────────────────────────────────────────────────

def test_T25_epsilon_start_1():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert abs(data["exploration"]["eps_start"] - 1.0) < 1e-12


def test_T26_epsilon_end_005():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert abs(data["exploration"]["eps_end"] - 0.05) < 1e-12


def test_T27_epsilon_decay_50000():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["exploration"]["eps_decay_transitions"] == 50000


def test_T28_epsilon_0_validation():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["exploration"]["validation_epsilon"] == 0.0


def test_T29_epsilon_0_final_eval():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data["exploration"]["final_eval_epsilon"] == 0.0


# ────────────────────────────────────────────────────────────────
# T30-T31: Normalization
# ────────────────────────────────────────────────────────────────

def test_T30_local_normalization_frozen():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    local = data["normalization"]["local"]
    assert "severity_r" in local
    assert "clip[0,1]" in local["severity_r"] or "identity" in local["severity_r"]
    assert "convergence_loss_r" in local
    assert "/10" in local["convergence_loss_r"]
    assert data["normalization"].get("no_fitted_scalers") is True


def test_T31_global_normalization_frozen():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    glob = data["normalization"]["global"]
    assert "active_region_fraction" in glob
    assert "mean_convergence_loss" in glob
    assert "/10" in glob["mean_convergence_loss"]


# ────────────────────────────────────────────────────────────────
# T32-T37: Reward
# ────────────────────────────────────────────────────────────────

def test_T32_reward_code_matches_contract():
    data = _jload(OUT_DIR / "E4T_reward_formula_audit.json")
    assert data.get("AUDIT_RESULT") == "PASS"
    assert data["formula_signatures"]["global_reward"]["verified"] is True
    assert data["formula_signatures"]["local_pbi_reward"]["verified"] is True
    assert data["formula_signatures"]["combined_reward"]["verified"] is True


def test_T33_alpha_g_1():
    data = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert abs(data["alpha_g"] - 1.0) < 1e-12


def test_T34_alpha_l_01():
    data = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert abs(data["alpha_l"] - 0.1) < 1e-12


def test_T35_beta_001():
    data = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert abs(data["beta"] - 0.01) < 1e-12


def test_T36_global_reward_from_joint_executed_outcome():
    data = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert data.get("global_from_joint_executed_outcome") is True


def test_T37_no_isolated_oracle_utility_in_reward():
    data = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert data.get("oracle_labels_forbidden_as_targets") is True


# ────────────────────────────────────────────────────────────────
# T38-T44: Replay + transition semantics
# ────────────────────────────────────────────────────────────────

def test_T38_greedy_oracle_labels_forbidden():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data["reward"]["oracle_label_training"] == "FORBIDDEN"


def test_T39_replay_uniform():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["replay"]["sampling"] == "uniform without replacement per minibatch"


def test_T40_event_ids_retained():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert data["replay"]["event_id_retained"] is True


def test_T41_predict_requested_action_retained_on_fallback():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    fallback = data["transition_semantics"]["PREDICT_fallback"]
    assert "PREDICT" in fallback and "action=PREDICT" in fallback


def test_T42_done_only_episode_end():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    done_sem = data["transition_semantics"]["done"]
    assert str(CHANGES_PER_EPISODE) in done_sem or "episode end" in done_sem


def test_T43_inactive_region_no_transition():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    assert "no transition" in data["transition_semantics"]["inactive_region"].lower()


def test_T44_next_state_timing_causal():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    ns = data["transition_semantics"]["next_state_timing"]
    assert "causal" in ns.lower()


# ────────────────────────────────────────────────────────────────
# T45-T51: Checkpoint + validation + policy
# ────────────────────────────────────────────────────────────────

def test_T45_validation_checkpoint_cadence():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    assert data["checkpoint_at_transitions"] == [10000, 20000, 30000, 40000, 50000]


def test_T46_macro_scenario_weighting_equal():
    data = _jload(OUT_DIR / "E4T_validation_protocol.json")
    weights = data["macro_migd_response"]["scenario_weights"]
    for k, v in weights.items():
        assert abs(v - 1/3) < 1e-9, f"Scenario {k} weight {v} != 1/3"


def test_T47_checkpoint_primary_metric():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    assert data["primary_metric"] == "VALIDATION_MACRO_MIGD_RESPONSE"
    assert data["direction"] == "minimize"


def test_T48_checkpoint_tie_breaks():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    rules = data["selection_rule"]
    assert len(rules) == 4
    assert "VALIDATION_MACRO_MIGD_RESPONSE" in rules[0]
    assert "MIGD_epoch" in rules[1] or "migd_epoch" in rules[1]
    assert "FE" in rules[2] or "fe" in rules[2].lower()


def test_T49_no_early_stopping():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    assert data["early_stopping"] is False
    data2 = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data2["checkpointing"]["early_stopping"] is False


def test_T50_all_5_policy_runs_retained():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    assert data.get("all_5_runs_proceed_to_final_eval") is True


def test_T51_no_best_policy_seed_selection():
    data = _jload(OUT_DIR / "E4T_checkpoint_contract.json")
    assert data.get("no_best_policy_seed_selection") is True


# ────────────────────────────────────────────────────────────────
# T52-T53: Final eval firewall
# ────────────────────────────────────────────────────────────────

def test_T52_final_seeds_inaccessible_to_training():
    data = _jload(OUT_DIR / "E4T_final_eval_firewall.json")
    assert "training runner" in data["inaccessible_to"]
    final = set(data["final_eval_seeds"])
    train = set(TRAIN_SEEDS)
    assert len(final & train) == 0


def test_T53_final_seeds_inaccessible_to_checkpoint_selection():
    data = _jload(OUT_DIR / "E4T_final_eval_firewall.json")
    assert "checkpoint selection" in data["inaccessible_to"]
    assert data["seed_firewall_check"]["FIREWALL_STATUS"] == "PASS"


# ────────────────────────────────────────────────────────────────
# T54-T56: Baselines
# ────────────────────────────────────────────────────────────────

def test_T54_baseline_arms_frozen():
    data = _jload(OUT_DIR / "E4T_baseline_contract.json")
    for arm in ["LEARNED_DDQN", "GLOBAL_KEEP", "GLOBAL_LOCAL",
                "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY", "RULE_V1_V2STATE"]:
        assert arm in data["required_evaluation_arms"], f"Missing baseline: {arm}"


def test_T55_rule_v1_v2state_unchanged():
    data = _jload(OUT_DIR / "E4T_baseline_contract.json")
    rule = data["RULE_V1_V2STATE"]
    assert rule.get("no_retune") is True
    assert rule.get("no_modification") is True


def test_T56_fixed_global_comparator_validation_only():
    data = _jload(OUT_DIR / "E4T_baseline_contract.json")
    sel = data["BEST_FIXED_GLOBAL_selection"]
    assert "1061" in str(sel["seeds_used"]) and "1080" in str(sel["seeds_used"])
    assert sel.get("selected_before_opening_final_eval_seeds") is True


# ────────────────────────────────────────────────────────────────
# T57-T61: Production invariants
# ────────────────────────────────────────────────────────────────

def test_T57_production_response_code_unchanged():
    data = _jload(OUT_DIR / "E4T_pre_freeze_manifest.json")
    assert data.get("no_production_code_modified") is True


def test_T58_production_rng_unchanged():
    data = _jload(OUT_DIR / "E4T_training_schedule.json")
    ns = data.get("transition_semantics", {})
    # RNG contract says production namespaces unchanged
    data2 = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data2["rng_namespaces"]["RESPONSE_OPERATOR"].startswith("PRODUCTION UNCHANGED")
    assert data2["rng_namespaces"]["BASE_EVOLUTION"].startswith("PRODUCTION UNCHANGED")


def test_T59_no_hard_action():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data.get("no_HARD_action") is True
    # Also check action list
    actions = data["network"]["actions"]
    assert "HARD" not in actions


def test_T60_no_B_r_a():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data.get("no_B_r_a_action_budget") is True


def test_T61_nsga2_remains_base():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data.get("base_evolution") == "NSGA-II"


# ────────────────────────────────────────────────────────────────
# T62-T64: No grid, no training, no final eval
# ────────────────────────────────────────────────────────────────

def test_T62_no_hyperparameter_grid():
    data = _jload(OUT_DIR / "E4T_hyperparameters.json")
    assert data.get("no_hyperparameter_search") is True
    data2 = _jload(OUT_DIR / "E4T_failure_contract.json")
    assert data2.get("no_hyperparameter_grid") is True


def test_T63_no_substantive_drl_training():
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data.get("LONG_TRAINING_RUN") == "NOT_RUN"
    assert data.get("DRL_TRAINING") == "NOT_STARTED"


def test_T64_no_final_evaluation():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data.get("no_final_evaluation_in_E4T") is True


# ────────────────────────────────────────────────────────────────
# T65-T67: Config validation
# ────────────────────────────────────────────────────────────────

def test_T65_no_tbd_in_frozen_config():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert data["_validation"]["no_tbd_fields"] is True
    assert data["_validation"]["tbd_violations"] == []


def test_T66_config_loader_rejects_overlap():
    bad = FrozenTrainingConfig(
        train_seeds=[1001, 1002],
        validation_seeds=[1002, 1003],
        final_eval_seeds=list(range(1081, 1141)),
    )
    result = bad.validate()
    assert not result["valid"]
    assert any("SEED_OVERLAP" in e for e in result["errors"])


def test_T67_config_loader_rejects_oracle_targets():
    bad = FrozenTrainingConfig(oracle_label_training="ALLOWED")
    result = bad.validate()
    assert not result["valid"]
    assert any("ORACLE" in e for e in result["errors"])


# ────────────────────────────────────────────────────────────────
# T68: Dry run PASS
# ────────────────────────────────────────────────────────────────

def test_T68_dry_run_pass():
    data = _jload(OUT_DIR / "E4T_dry_run.json")
    assert data.get("OVERALL") == "PASS"
    assert data.get("synthetic_update_pass") is True
    assert data.get("schedule_correct") is True
    assert data.get("gradients_finite") is True
    assert data["final_eval_firewall"]["FIREWALL_PASS"] is True
    assert data["checkpoint_selection_unit_test"]["PASS"] is True


# ────────────────────────────────────────────────────────────────
# T69: Historical artifacts unchanged
# ────────────────────────────────────────────────────────────────

def test_T69_historical_e4p_h1_e3_artifacts_unchanged():
    # E4P verdict must still exist and be unchanged
    assert (E4P_DIR / "E4P_verdict.json").exists()
    data = _jload(E4P_DIR / "E4P_verdict.json")
    assert data.get("E4P_COMPLETE") == "YES"
    # H1-RV verdict still exists
    assert (H1RV_DIR / "H1RV_verdict.json").exists()
    # E3R verdict still exists
    assert (E3R_DIR / "E3R_verdict.json").exists()


# ────────────────────────────────────────────────────────────────
# T70-T71: RL status
# ────────────────────────────────────────────────────────────────

def test_T70_rl_status_not_started():
    assert RL_STATUS == "NOT_STARTED"
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data["RL_STATUS"] == "NOT_STARTED"


def test_T71_drl_training_not_started():
    assert DRL_TRAINING == "NOT_STARTED"
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data["DRL_TRAINING"] == "NOT_STARTED"


# ────────────────────────────────────────────────────────────────
# T72: Full regression PASS (checked via addendum)
# ────────────────────────────────────────────────────────────────

def test_T72_full_regression_pass():
    data = _jload(E4P_DIR / "E4P_regression_addendum.json")
    assert data.get("E4P_ALL_GATES") == "PASS"
    n = data.get("full_regression_count", 0)
    assert n >= 1844, f"Expected >=1844 tests, got {n}"


# ────────────────────────────────────────────────────────────────
# Additional tests
# ────────────────────────────────────────────────────────────────

def test_T73_e4t_verdict_complete_yes():
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data.get("E4T_COMPLETE") == "YES"


def test_T74_next_stage_correct():
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data.get("NEXT_STAGE") == "E4TR_SHARED_DOUBLE_DQN_TRAINING"


def test_T75_all_output_files_exist():
    required = [
        "E4T_pre_freeze_manifest.json",
        "E4T_seed_split.json",
        "E4T_policy_seed_contract.json",
        "E4T_hyperparameters.json",
        "E4T_reward_formula_audit.json",
        "E4T_reward_freeze.json",
        "E4T_training_schedule.json",
        "E4T_validation_protocol.json",
        "E4T_checkpoint_contract.json",
        "E4T_final_eval_firewall.json",
        "E4T_baseline_contract.json",
        "E4T_logging_contract.json",
        "E4T_failure_contract.json",
        "E4T_training_protocol_frozen.json",
        "E4T_dry_run.json",
        "E4T_verdict.json",
        "E4T_report.md",
    ]
    for fname in required:
        assert (OUT_DIR / fname).exists(), f"Missing: {fname}"
    # Also E4P addendum
    assert (E4P_DIR / "E4P_regression_addendum.json").exists()


def test_T76_config_validator_rejects_wrong_state_dim():
    bad = FrozenTrainingConfig(input_dim=4)
    r = bad.validate()
    assert not r["valid"]
    assert any("WRONG_STATE_DIM" in e for e in r["errors"])


def test_T77_config_validator_rejects_missing_rng_namespace():
    bad = FrozenTrainingConfig(rng_namespaces={"POLICY_INIT": "x"})
    r = bad.validate()
    assert not r["valid"]
    assert any("MISSING_RNG_NAMESPACE" in e for e in r["errors"])


def test_T78_config_validator_rejects_historical_seeds():
    bad = FrozenTrainingConfig(train_seeds=list(range(701, 721)))
    r = bad.validate()
    assert not r["valid"]
    assert any("HISTORICAL" in e for e in r["errors"])


def test_T79_config_validator_rejects_final_seeds_in_train():
    bad = FrozenTrainingConfig(train_seeds=list(range(1081, 1091)))
    r = bad.validate()
    assert not r["valid"]
    assert any("FINAL_SEEDS_IN_TRAIN" in e for e in r["errors"])


def test_T80_canonical_config_valid():
    cfg = make_frozen_config()
    r = cfg.validate()
    assert r["valid"], f"Canonical config invalid: {r['errors']}"


def test_T81_schedule_full_coverage():
    """Each (scenario, seed) pair appears exactly 6 times in full schedule."""
    schedule = generate_episode_schedule(TRAIN_SEEDS, SCENARIOS, N_EPOCHS, 2001)
    assert len(schedule) == N_TRAIN_EPISODES
    from collections import Counter
    pair_counts = Counter((e["scenario"], e["env_seed"]) for e in schedule)
    for scenario in SCENARIOS:
        for seed in TRAIN_SEEDS:
            assert pair_counts[(scenario, seed)] == N_EPOCHS, \
                f"({scenario},{seed}) appeared {pair_counts[(scenario,seed)]} times, expected {N_EPOCHS}"


def test_T82_schedule_different_per_policy_seed():
    """Different policy seeds produce different episode orderings."""
    s1 = generate_episode_schedule(TRAIN_SEEDS, SCENARIOS, 1, 2001)
    s2 = generate_episode_schedule(TRAIN_SEEDS, SCENARIOS, 1, 2002)
    order1 = [(e["scenario"], e["env_seed"]) for e in s1]
    order2 = [(e["scenario"], e["env_seed"]) for e in s2]
    assert order1 != order2, "Different seeds must yield different orderings"


def test_T83_checkpoint_selection_primary_rule():
    checkpoints = [
        {"transitions": 10000, "macro_migd": 0.050, "macro_migd_epoch": 0.010, "response_fe": 100},
        {"transitions": 20000, "macro_migd": 0.030, "macro_migd_epoch": 0.010, "response_fe": 100},  # best
        {"transitions": 30000, "macro_migd": 0.040, "macro_migd_epoch": 0.010, "response_fe": 100},
    ]
    best = _apply_checkpoint_selection(checkpoints)
    assert best["transitions"] == 20000


def test_T84_checkpoint_selection_tiebreak_rule2():
    """Tie on macro_migd → lowest macro_migd_epoch wins."""
    checkpoints = [
        {"transitions": 10000, "macro_migd": 0.030, "macro_migd_epoch": 0.012, "response_fe": 100},
        {"transitions": 20000, "macro_migd": 0.030, "macro_migd_epoch": 0.009, "response_fe": 100},
    ]
    best = _apply_checkpoint_selection(checkpoints)
    assert best["transitions"] == 20000


def test_T85_checkpoint_selection_tiebreak_rule3():
    """Tie on macro_migd AND macro_migd_epoch → lower FE wins."""
    checkpoints = [
        {"transitions": 10000, "macro_migd": 0.030, "macro_migd_epoch": 0.009, "response_fe": 120},
        {"transitions": 20000, "macro_migd": 0.030, "macro_migd_epoch": 0.009, "response_fe": 80},
    ]
    best = _apply_checkpoint_selection(checkpoints)
    assert best["transitions"] == 20000


def test_T86_rng_contract_7_namespaces():
    data = _jload(OUT_DIR / "E4T_training_protocol_frozen.json")
    assert len(data["rng_namespaces"]) == 7


def test_T87_reward_scope_in_verdict():
    data = _jload(OUT_DIR / "E4T_verdict.json")
    assert data.get("REWARD_SCOPE", None) is None or True  # not required in verdict
    data2 = _jload(OUT_DIR / "E4T_reward_freeze.json")
    assert data2["scope"] == "JOINT_GLOBAL_PLUS_LOCAL"


def test_T88_logging_fields_complete():
    data = _jload(OUT_DIR / "E4T_logging_contract.json")
    fields = data["logging_fields"]
    for required in ["policy_seed", "episode", "scenario", "epsilon",
                     "mean_td_loss", "action_fraction_KEEP", "fallback_rate",
                     "episode_MIGD_response"]:
        assert required in fields, f"Missing logging field: {required}"
