"""
E4HT — Normalization A/B Training Protocol Freeze: contract-freeze + unit tests.

Verify the V1-vs-V3 single-factor A/B protocol is frozen correctly (identical
everything except normalization_version), paired initialization holds, the paired
crossed bootstrap is available, and nothing trained / no final seed opened.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4HT_normalization_ab_training_protocol_freeze as E4HT
from prdrl.rl.normalization import NORM_V1, NORM_V2, NORM_V3
from prdrl.evaluation.crossed_bootstrap import paired_crossed_two_axis_bootstrap

RES = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HI = Path("results/E4HI_normalization_implementation_validation")
E4HR = Path("results/E4HR_normalization_design_review")
E4F = Path("results/E4F_final_evaluation")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4HT_verdict.json").exists():
        pytest.skip("E4HT not run yet")


# ---- prerequisites ----

def test_T1_e4hi_complete():
    assert _load(E4HI / "E4HI_verdict.json")["E4HI_COMPLETE"] == "YES"


def test_T2_v3_ready():
    assert _load(E4HI / "E4HI_verdict.json")["NORMALIZATION_V3_READY"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


# ---- unit: paired init + paired bootstrap ----

def test_T21_T22_paired_init_deterministic_and_arm_independent():
    o1, t1 = E4HT._paired_init(3001)
    o2, t2 = E4HT._paired_init(3001)
    assert E4HT._param_hash(o1) == E4HT._param_hash(o2)   # deterministic (both arms)
    assert E4HT._param_hash(t1) == E4HT._param_hash(t2)
    assert E4HT._param_hash(o1) == E4HT._param_hash(t1)   # online==target at t0
    # different policy seed -> different init
    o3, _ = E4HT._paired_init(3002)
    assert E4HT._param_hash(o3) != E4HT._param_hash(o1)


def test_T40_T41_paired_bootstrap_available():
    D = np.random.default_rng(0).normal(0.0005, 0.001, size=(5, 60))
    out = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=56001)
    assert out["method"] == "paired_crossed_two_axis"
    assert out["n_policy"] == 5 and out["n_env"] == 60
    assert abs(out["point_estimate"] - float(D.mean())) < 1e-12
    # deterministic
    out2 = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=56001)
    assert out["ci_lo"] == out2["ci_lo"] and out["ci_hi"] == out2["ci_hi"]


# ---- artifact contract tests ----

def test_T4_T5_T6_two_arms_exact():
    _require()
    d = _load(RES / "E4HT_ab_factor_contract.json")
    assert d["control_normalization"] == NORM_V1
    assert d["treatment_normalization"] == NORM_V3
    assert d["experiment_factor"].startswith("normalization_version")


def test_T7_T8_state_identical():
    _require()
    d = _load(RES / "E4HT_state_contract.json")
    assert d["state_dim"] == 9
    assert d["no_region_id"] and d["no_joint_action_context"]
    assert d["identical_both_arms"] is True


def test_T9_network_equal():
    _require()
    assert _load(RES / "E4HT_protocol_frozen.json")["network"] == "9-128-128-4"


def test_T10_reward_equal_unchanged():
    _require()
    d = _load(RES / "E4HT_reward_contract.json")
    assert (d["alpha_g"], d["alpha_l"], d["beta"]) == (1.0, 0.1, 0.01)
    assert d["reward_unchanged"] is True
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in \
        Path("src/prdrl/rl/reward.py").read_text()


def test_T11_T12_hyperparameters_equal():
    _require()
    d = _load(RES / "E4HT_hyperparameters.json")
    assert d["gamma"] == 0.99 and d["learning_rate"] == 1e-4 and d["tau"] == 0.005
    assert d["batch_size"] == 128 and d["replay_capacity"] == 50000
    assert d["warmup_regional_transitions"] == 1000 and d["update_every_appended_transitions"] == 4
    assert d["identical_both_arms"] is True


def test_T13_T14_T15_actions_operators_budget():
    _require()
    d = _load(RES / "E4HT_operator_contract.json")
    assert d["LOCAL"]["sigma_L"] == 0.02 and d["PREDICT"]["sigma_P"] == 0.005
    assert d["DIVERSIFY"]["sigma_D"] == 0.15 and d["KEEP"]["FE"] == 0
    assert "B=10 PER_ACTIVE_REGION" in d["response_budget"]
    assert d["no_HARD"] and d["no_MEMORY"]


def test_T16_T17_T18_T19_T20_seed_split():
    _require()
    d = _load(RES / "E4HT_seed_split.json")
    assert d["train"]["seeds"] == list(range(1301, 1361))
    assert d["validation"]["seeds"] == list(range(1361, 1381))
    assert d["final"]["seeds"] == list(range(1381, 1441))
    assert d["all_disjoint"] is True and d["disjoint_from_historical"] is True
    p = _load(RES / "E4HT_policy_pairing_contract.json")
    assert p["policy_seeds"] == [3001, 3002, 3003, 3004, 3005]


def test_T22b_init_pairing_contract():
    _require()
    d = _load(RES / "E4HT_initialization_pairing_contract.json")
    assert d["integrity_verified"] is True
    assert d["online_equals_target_at_t0"] is True
    for ps in ["3001", "3002", "3003", "3004", "3005"]:
        assert d["per_policy_init_hashes"][ps]["deterministic"] is True


def test_T23_T24_replay_isolation():
    _require()
    d = _load(RES / "E4HT_rng_contract.json")
    assert d["replay_rng_isolated"] is True
    # each run owns its replay (documented in budget/rng contracts)


def test_T25_T26_T27_T28_training_budget():
    _require()
    d = _load(RES / "E4HT_training_budget.json")
    assert d["episodes_per_run"] == 1080
    assert d["total_runs"] == 10
    assert d["total_episodes"] == 10800
    assert "equal" in d["scenario_balance"]


def test_T29_epsilon_schedule():
    _require()
    d = _load(RES / "E4HT_hyperparameters.json")
    assert d["epsilon_start"] == 1.0 and d["epsilon_end"] == 0.05
    assert d["epsilon_decay_transitions"] == 50000 and d["validation_epsilon"] == 0.0


def test_T30_T31_T32_T33_T34_T35_checkpoint_validation():
    _require()
    cc = _load(RES / "E4HT_checkpoint_contract.json")
    assert cc["transitions"] == [10000, 20000, 30000, 40000, 50000]
    vc = _load(RES / "E4HT_validation_contract.json")
    assert vc["epsilon"] == 0.0
    assert vc["selection_metric"] == "VALIDATION_MACRO_MIGD_RESPONSE"
    assert vc["tie_breakers"] == ["lower macro MIGD_epoch", "lower response_FE", "earlier checkpoint"]
    assert vc["no_mixed_only_selection"] is True
    assert vc["retain_all_5_per_arm"] is True


def test_T36_T37_T38_T39_primary():
    _require()
    d = _load(RES / "E4HT_primary_hypothesis.json")
    assert d["primary_scenario"] == "MIXED"
    assert d["primary_endpoint"] == "MIGD_response"
    assert "V1" in d["cell_effect"] and "V3" in d["cell_effect"]
    assert d["positive_means"] == "V3 better"
    sd = _load(RES / "E4HT_statistical_design.json")
    assert sd["practical_margin_DELTA_NORM"] == 1e-5


def test_T40b_T41b_bootstrap_frozen():
    _require()
    sd = _load(RES / "E4HT_statistical_design.json")
    assert sd["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
    assert sd["N_BOOTSTRAP"] == 50000
    assert sd["BOOTSTRAP_MASTER_SEED"] == 56001
    assert sd["no_naive_300_row_bootstrap"] is True


def test_T42_policy_robustness_rule():
    _require()
    sd = _load(RES / "E4HT_statistical_design.json")
    assert ">=4/5" in sd["policy_robustness"]


def test_T43_T44_mechanism_endpoints():
    _require()
    d = _load(RES / "E4HT_mechanism_endpoints.json")
    assert d["endpoints"] == ["PREDICT_fraction", "within_event_multi_action_rate", "regional_action_entropy"]
    assert ">= 2 of 3" in d["MECHANISM_DECOLLAPSE_SUPPORTED_rule"]


def test_T45_T46_T47_T48_safety():
    _require()
    d = _load(RES / "E4HT_safety_contract.json")
    assert d["scenarios"] == ["PERSISTENT", "STATIONARY"]
    assert d["safety_pass"] == "neither endpoint HARMFUL"
    assert d["fe_cannot_rescue_harmful_migd"] is True
    assert "SHARED bootstrap draws" in _load(RES / "E4HT_statistical_design.json")["macro"]


def test_T49_T50_reward_logging_no_change():
    _require()
    d = _load(RES / "E4HT_reward_contract.json")
    for f in ["raw_global_reward", "weighted_global_reward", "raw_local_reward",
              "weighted_local_reward", "raw_fe_cost", "weighted_fe_penalty", "total_reward"]:
        assert f in d["component_logging_required"]


def test_T51_no_joint_change():
    _require()
    v = _load(RES / "E4HT_verdict.json")
    assert v["JOINT_ARCHITECTURE_CHANGED"] == "NO"


def test_T52_T53_no_online_adaptation_v3_hash():
    _require()
    d = _load(RES / "E4HT_v3_constants_integrity.json")
    assert d["no_online_adaptation"] is True
    assert d["no_refit_on_new_train_seeds"] is True
    assert d["v3_constants_sha256"] == _load(E4HR / "E4HR_v3_constants.json")["constants_sha256"]


def test_T54_T55_final_firewall():
    _require()
    d = _load(RES / "E4HT_final_firewall.json")
    assert d["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert d["unopened"] is True


def test_T56_T57_T58_T59_no_training():
    _require()
    d = _load(RES / "E4HT_no_training_audit.json")
    assert d["episodes_trained"] == 0
    assert d["optimizer_steps"] == 0
    assert d["replay_appends"] == 0
    assert d["new_policy_checkpoints"] == 0


def test_T60_success_logic_exact():
    _require()
    d = _load(RES / "E4HT_success_logic.json")
    assert "MIXED V3-vs-V1 MIGD_response = BENEFICIAL" in d["NORMALIZATION_CAUSAL_VALIDATED_requires_all"]
    assert "MECHANISTIC_PARTIAL_when" in d
    assert d["not_overall_superiority_over_global_predict"] is True


def test_checkpoint_version_guard():
    _require()
    cc = _load(RES / "E4HT_checkpoint_contract.json")
    assert cc["version_guard"]["V1"] == NORM_V1
    assert cc["version_guard"]["V3"] == NORM_V3
    assert cc["version_guard"]["mismatch"] == "HARD_ERROR"


def test_verdict_complete_and_next():
    _require()
    v = _load(RES / "E4HT_verdict.json")
    assert v["E4HT_COMPLETE"] == "YES"
    assert v["no_placeholder_tokens"] is True
    assert v["PAIRED_INITIALIZATION"] == "FROZEN"
    assert v["NEXT_STAGE"] == "E4HTR_NORMALIZATION_AB_TRAINING_EXECUTION"


def test_v2_not_an_arm():
    _require()
    nv = _load(RES / "E4HT_normalization_versions.json")
    assert nv["V2"]["role"] == "NONE"
    assert nv["V2"]["status"] == "REJECTED_NOT_TRAINED"
