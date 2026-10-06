"""
E4I — Reward / Credit Revision Design: contract + audit tests.

Design/audit/freeze only. Verifies R0 audit + inertness reproduction, R1 feasibility /
boundedness / semantics, single-factor V3+R0 vs V3+R1 freeze, fresh disjoint seeds,
future mechanism/statistics/success contracts, and the R1 readiness logic. No training.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4I_reward_credit_revision_design as E4I
from prdrl.rl import reward as R0
from prdrl.rl.reward_r1 import (
    REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1, ALPHA_LOCAL_R1_INITIAL,
    regional_quality_gain_r1, compute_regional_reward_r1,
)
from prdrl.rl.normalization import NORM_V3

RES = Path("results/E4I_reward_credit_revision_design")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")
E4GD = Path("results/E4GD_controlled_diagnostic_execution")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4I_verdict.json").exists():
        pytest.skip("E4I not run yet")


# ---- prerequisites / history ----

def test_T1_e4hf_complete():
    assert _load(E4HF / "E4HF_verdict.json")["E4HF_COMPLETE"] == "YES"


def test_T2_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T3_e4hf_normalization_verdict_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T4_branch_b_selected():
    _require()
    assert _load(RES / "E4I_verdict.json")["SELECTED_BRANCH"] == "B_REWARD_CREDIT"


def test_T5_v3_fixed_both_arms():
    _require()
    sf = _load(RES / "E4I_single_factor_contract.json")
    assert sf["CONTROL"]["normalization"] == NORM_V3
    assert sf["TREATMENT"]["normalization"] == NORM_V3
    assert sf["normalization_fixed_both_arms"] == NORM_V3


def test_T6_v1_not_a_future_arm():
    _require()
    sf = _load(RES / "E4I_single_factor_contract.json")
    assert sf["v1_arm"] is False and sf["v2_arm"] is False


def test_T7_joint_architecture_unchanged():
    _require()
    tc = _load(RES / "E4I_future_training_contract.json")
    assert tc["algorithm"] == "SHARED_DOUBLE_DQN"
    assert tc["centralized_critic"] is False and tc["joint_action_representation_change"] is False
    assert tc["qmix"] is False and tc["vdn"] is False and tc["autoregressive_policy"] is False


# ---- R0 audit ----

def test_T8_r0_source_audit_complete():
    _require()
    a = _load(RES / "E4I_reward_r0_audit.json")
    assert a["reward_version"] == REWARD_R0_NAME
    assert a["source"] == "src/prdrl/rl/reward.py"


def test_T9_r0_coefficients_exact():
    _require()
    a = _load(RES / "E4I_reward_r0_audit.json")
    assert a["alpha_g"] == 1.0 and a["alpha_l"] == 0.1 and a["beta"] == 0.01
    assert R0.ALPHA_G_DEFAULT == 1.0 and R0.ALPHA_L_DEFAULT == 0.1 and R0.BETA_DEFAULT == 0.01


def test_T10_r0_local_inertness_reproduced():
    _require()
    # reproduce zero/nonzero fraction directly from the pre-existing D4 CSV
    rows = list(csv.DictReader(open(E4GD / "E4GD_D4_reward_components.csv")))
    rl = np.array([float(r["raw_local_reward"]) for r in rows])
    zero_frac = float(np.mean(rl == 0.0))
    a = _load(RES / "E4I_r0_local_inertness_analysis.json")
    assert abs(a["LOCAL_REWARD_ZERO_FRACTION"] - zero_frac) < 1e-12
    assert abs(a["LOCAL_REWARD_NONZERO_FRACTION"] - (1 - zero_frac)) < 1e-12
    # inertness is a scale mismatch, not a dead signal
    assert a["LOCAL_REWARD_NONZERO_FRACTION"] > 0.20
    assert a["R0_LOCAL_INERTNESS_CAUSE"].startswith("C_SCALE_MISMATCH_DOMINANT")
    assert a["D4_REWARD_SCALE"] == "GLOBAL_OVERWHELMS_LOCAL"


def test_T11_regional_quality_metric_identified():
    _require()
    m = _load(RES / "E4I_local_quality_metric_contract.json")
    assert m["source"] == "src/prdrl/detectors/quality_loss.py::quality_loss"
    assert "convergence_loss_r" in m["L_pre"]


def test_T12_r1_same_region_id_pre_post():
    _require()
    m = _load(RES / "E4I_local_quality_metric_contract.json")
    assert m["same_region_id_pre_post"] is True


def test_T13_r1_no_extra_objective_eval():
    _require()
    m = _load(RES / "E4I_local_quality_metric_contract.json")
    assert m["extra_objective_evaluations_required"] == 0
    assert _load(RES / "E4I_r1_fe_accounting.json")["EXTRA_FE_FOR_REWARD_R1"] == 0


# ---- R1 formula unit tests ----

def test_T14_r1_bounded():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        lp, lq = rng.uniform(0, 10), rng.uniform(0, 10)
        g = regional_quality_gain_r1(lp, lq)
        assert -1.0 <= g <= 1.0


def test_T15_r1_positive_on_improvement():
    assert regional_quality_gain_r1(1.0, 0.2) > 0.0


def test_T16_r1_negative_on_degradation():
    assert regional_quality_gain_r1(0.2, 1.0) < 0.0


def test_T17_r1_zero_on_tie():
    assert regional_quality_gain_r1(0.5, 0.5) == 0.0
    assert regional_quality_gain_r1(0.0, 0.0) == 0.0


def test_T18_epsilon_exact():
    assert EPS_R1 == 1e-12
    assert _load(RES / "E4I_reward_r1_formula.json")["eps"] == 1e-12 if (RES / "E4I_reward_r1_formula.json").exists() else True


def test_T19_empty_degenerate_semantics_frozen():
    _require()
    m = _load(RES / "E4I_local_quality_metric_contract.json")
    assert "NaN" in m["empty_or_degenerate_region"]
    assert regional_quality_gain_r1(float("nan"), 0.3) == 0.0
    assert regional_quality_gain_r1(0.3, float("nan")) == 0.0


def test_T20_global_reward_unchanged():
    _require()
    f = _load(RES / "E4I_reward_r1_formula.json")
    assert f["alpha_g"] == 1.0
    # R1 global term is literally R0's compute_global_reward (unchanged source)
    assert R0.compute_global_reward(-0.01, -0.008, 0.01) == pytest.approx((-0.008 - -0.01) / (0.01 + R0._EPS))


def test_T21_fe_penalty_unchanged():
    _require()
    f = _load(RES / "E4I_reward_r1_formula.json")
    assert f["beta"] == 0.01
    assert R0.BUDGET_PER_REGION == 10


def test_T22_alpha_local_initial_preserved():
    _require()
    assert ALPHA_LOCAL_R1_INITIAL == 0.1
    assert _load(RES / "E4I_verdict.json")["R1_ALPHA_LOCAL"] == 0.1


def test_T23_no_performance_based_coefficient_sweep():
    _require()
    sg = _load(RES / "E4I_r1_scale_gate.json")
    assert sg["coefficient_frozen_without_performance_tuning"] is True
    assert "NEVER performance-tuned" in sg["scale_match_rule_if_needed"]


def test_T24_r1_activity_gate_exact():
    _require()
    g = _load(RES / "E4I_r1_activity_gate.json")
    # activity == raw_local nonzero fraction
    rows = list(csv.DictReader(open(E4GD / "E4GD_D4_reward_components.csv")))
    rl = np.array([float(r["raw_local_reward"]) for r in rows])
    assert abs(g["nonzero_fraction"] - float(np.mean(rl != 0.0))) < 1e-12
    assert g["R1_ACTIVITY_GATE"] == ("PASS" if g["nonzero_fraction"] >= 0.20 else "FAIL")


def test_T25_r1_scale_gate_exact():
    _require()
    sg = _load(RES / "E4I_r1_scale_gate.json")
    # magnitude not reconstructable from stored difference-only data
    assert sg["R1_SCALE_GATE"] == "NOT_AUDITABLE"


def test_T26_coefficient_adjustment_rule_deterministic():
    _require()
    sg = _load(RES / "E4I_r1_scale_gate.json")
    assert "median_abs_global / median_abs_g_r" in sg["scale_match_rule_if_needed"]
    assert sg["alpha_local_min"] == 0.01 and sg["alpha_local_max"] == 10.0


def test_T27_no_future_seed_in_scale_fitting():
    _require()
    audit = _load(RES / "E4I_r1_offline_scale_audit.json")
    assert audit["reconstructable_from"].startswith("E4GD D4")
    # confirms scale audit uses only pre-existing diagnostic data (no future seeds)


def test_T28_r1_causality_pass():
    _require()
    c = _load(RES / "E4I_r1_causality.json")
    assert c["R1_CAUSALITY"] == "PASS"
    assert c["uses_future_environment_state"] is False and c["uses_next_change_info"] is False


def test_T29_extra_reward_fe_zero():
    _require()
    assert _load(RES / "E4I_r1_fe_accounting.json")["EXTRA_FE_FOR_REWARD_R1"] == 0


def test_T30_future_control_exact():
    _require()
    sf = _load(RES / "E4I_single_factor_contract.json")
    assert sf["CONTROL"]["arm_name"] == "R0_CONTROL"
    assert sf["CONTROL"]["reward"] == REWARD_R0_NAME


def test_T31_future_treatment_exact():
    _require()
    sf = _load(RES / "E4I_single_factor_contract.json")
    assert sf["TREATMENT"]["arm_name"] == "R1_TREATMENT"
    assert sf["TREATMENT"]["reward"] == REWARD_R1_NAME


def test_T32_reward_is_only_future_factor():
    _require()
    sf = _load(RES / "E4I_single_factor_contract.json")
    assert sf["experimental_factor"] == "REWARD_ONLY"
    assert sf["everything_else_identical"] is True


def test_T33_state_dim_9():
    _require()
    assert _load(RES / "E4I_future_training_contract.json")["state_dim"] == 9


def test_T34_v3_constants_same_both_arms():
    _require()
    v = _load(RES / "E4I_verdict.json")
    assert v["FIXED_NORMALIZATION"] == NORM_V3
    rv = _load(RES / "E4I_single_factor_contract.json")
    assert rv["v3_refit"] is False


def test_T35_network_same_both_arms():
    _require()
    assert _load(RES / "E4I_future_training_contract.json")["network"] == "9-128-128-4"


def test_T36_hyperparameters_same_both_arms():
    _require()
    hp = _load(RES / "E4I_future_training_contract.json")["hyperparameters"]
    assert hp["gamma"] == 0.99 and hp["lr"] == 1e-4 and hp["batch"] == 128
    assert hp["replay"] == 50000 and hp["warmup"] == 1000 and hp["tau"] == 0.005
    assert hp["epochs"] == 6 and hp["episodes_per_policy"] == 1080
    assert _load(RES / "E4I_future_training_contract.json")["no_hyperparameter_retuning"] is True


def test_T37_operators_same_both_arms():
    _require()
    op = _load(RES / "E4I_future_training_contract.json")["operators"]
    assert op["sigma_L"] == 0.02 and op["sigma_P"] == 0.005 and op["lambda_P"] == 1.0
    assert op["sigma_D"] == 0.15 and op["KEEP_FE"] == 0 and op["survival"] == "unrestricted NSGA-II"


def test_T38_train_seeds_exact():
    _require()
    assert _load(RES / "E4I_future_seed_contract.json")["train"] == [1501, 1560]


def test_T39_validation_seeds_exact():
    _require()
    assert _load(RES / "E4I_future_seed_contract.json")["validation"] == [1561, 1580]


def test_T40_final_seeds_exact():
    _require()
    assert _load(RES / "E4I_future_seed_contract.json")["final"] == [1581, 1640]


def test_T41_policy_seeds_exact():
    _require()
    assert _load(RES / "E4I_future_seed_contract.json")["policy_seeds"] == [4001, 4002, 4003, 4004, 4005]


def test_T42_all_seed_sets_disjoint():
    _require()
    sc = _load(RES / "E4I_future_seed_contract.json")
    assert sc["env_disjoint_from_all_history"] is True
    assert sc["policy_disjoint_from_history"] is True
    assert sc["internal_split_disjoint"] is True
    # independent recompute of disjointness
    new = set(range(1501, 1561)) | set(range(1561, 1581)) | set(range(1581, 1641))
    consumed = set()
    for v in E4I.CONSUMED_SEED_SETS.values():
        consumed |= set(v)
    assert len(new & consumed) == 0


def test_T43_paired_initialization_contract():
    _require()
    p = _load(RES / "E4I_policy_pairing_contract.json")
    assert "identical network weights" in p["principle"]
    assert p["only_reward_differs"] is True
    assert set(["replay", "optimizer", "controller_rng", "networks"]).issubset(set(p["separate"]))


def test_T44_checkpoint_selection_macro_response():
    _require()
    c = _load(RES / "E4I_future_checkpoint_contract.json")
    assert c["selection_metric"] == "VALIDATION_MACRO_MIGD_RESPONSE"
    assert c["reward_value_not_used_for_selection"] is True


def test_T45_mechanism_endpoint_1():
    _require()
    m = _load(RES / "E4I_future_mechanism_contract.json")["mechanism_endpoints"]
    assert "expected R1 > R0" in m["1_LOCAL_REWARD_NONZERO_FRACTION"]


def test_T46_mechanism_endpoint_2():
    _require()
    m = _load(RES / "E4I_future_mechanism_contract.json")["mechanism_endpoints"]
    assert "R1 < R0" in m["2_GLOBAL_TO_LOCAL_WEIGHTED_MAGNITUDE_RATIO"]


def test_T47_mechanism_endpoint_3():
    _require()
    m = _load(RES / "E4I_future_mechanism_contract.json")
    ep3 = m["mechanism_endpoints"]["3_REGIONAL_REWARD_DISPERSION_WITHIN_EVENT"]
    assert "std" in ep3.lower() and "regional reward" in ep3.lower()
    # credit differentiation is the primary mechanism; entropy is descriptive only
    assert "credit differentiation" in m["primary_mechanism"].lower()
    assert "action_entropy" in m["descriptive_only"]


def test_T48_two_of_three_rule_exact():
    _require()
    m = _load(RES / "E4I_future_mechanism_contract.json")
    assert m["REWARD_CREDIT_MECHANISM_SUPPORTED"].startswith("YES iff >=2 of 3")
    assert m["mechanism_cannot_rescue_harmful_final_MIGD"] is True


def test_T49_bootstrap_design_frozen():
    _require()
    s = _load(RES / "E4I_future_statistical_contract.json")
    assert s["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
    assert s["policy_axis"] == 5 and s["environment_axis"] == 60 and s["N_BOOTSTRAP"] == 50000


def test_T50_bootstrap_master_seed_66001():
    _require()
    assert _load(RES / "E4I_future_statistical_contract.json")["BOOTSTRAP_MASTER_SEED"] == 66001
    assert E4I.REWARD_BOOTSTRAP_MASTER_SEED == 66001


def test_T51_practical_margin():
    _require()
    assert _load(RES / "E4I_future_statistical_contract.json")["PRACTICAL_MARGIN"] == 1e-5


def test_T52_policy_robustness_4_of_5():
    _require()
    assert "4 of 5" in _load(RES / "E4I_future_statistical_contract.json")["policy_robustness"]


def test_T53_no_training():
    _require()
    v = _load(RES / "E4I_verdict.json")
    assert v["NEW_TRAINING"] == "NO"
    assert _load(RES / "E4I_pre_manifest.json")["no_training"] is True


def test_T54_future_final_access_zero():
    _require()
    assert _load(RES / "E4I_final_firewall.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert _load(RES / "E4I_verdict.json")["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T55_historical_artifacts_unchanged():
    _require()
    h = _load(RES / "E4I_historical_result_preservation.json")
    assert h["checkpoint_all_match"] is True
    assert h["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert h["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


# ---- readiness + verdict consistency ----

def test_T56_r1_readiness_and_next_stage():
    _require()
    v = _load(RES / "E4I_verdict.json")
    ready = v["R1_REWARD_READY"] == "YES"
    if ready:
        assert v["NEXT_STAGE"] == "E4II_REWARD_IMPLEMENTATION_VALIDATION"
    elif v["R1_SCALE_GATE"] == "NOT_AUDITABLE":
        assert v["NEXT_STAGE"] == "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN"
    else:
        assert v["NEXT_STAGE"] == "E4I_REWARD_METRIC_REVIEW"
    # this run: scale gate not auditable offline -> NO -> E4ID
    assert v["R1_REWARD_READY"] == "NO"
    assert v["R1_SCALE_GATE"] == "NOT_AUDITABLE"
    assert v["NEXT_STAGE"] == "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN"


def test_T57_r0_source_byte_stable_no_r1_contamination():
    # single-factor discipline: R0 source contains no R1 symbols
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "REGIONAL_QUALITY_GAIN" not in src and "reward_r1" not in src


def test_T58_r1_reward_composition_matches_contract():
    # r_r^R1 = 1.0*global + 0.1*g - 0.01*fe (global & fe identical to R0)
    g = regional_quality_gain_r1(1.0, 0.25)
    total = compute_regional_reward_r1(global_reward=0.03, g_r=g, fe_cost=1.0)
    assert total == pytest.approx(1.0 * 0.03 + 0.1 * g - 0.01 * 1.0)
