"""Tests for E4OI — Scenario Trade-off Diagnostic Implementation Validation.

104 tests covering all diagnostic modules, synthetic validation,
E4O contract compliance, seed firewalls, and historical integrity.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prdrl.rl.network import SharedRegionalQNetwork, N_ACTIONS, build_online_target_pair
from prdrl.rl.qmix import QMIXMixerV1, R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM
from prdrl.rl.coordination_integration import (
    STATE_DIM, GAMMA, TARGET_TAU, JOINT_WEIGHT, REGIONAL_WEIGHT,
    current_joint_q, double_q_joint_target, treatment_loss,
    focal_regional_ddqn_target, focal_current_q,
    regional_smoothl1_loss, c2_composite_loss,
    make_qmix_config, qmix_config_sha256,
    make_c2_credit_config, c2_credit_config_sha256,
)
from prdrl.diagnostics.gradient_diagnostic import (
    extract_utility_gradient, extract_mixer_gradient,
    gradient_norms, cosine_similarity_jr,
    gradient_norm_ratio, composite_gradient_identity,
    mixer_gradient_check, conflict_fraction,
    scenario_gap_cosine, scenario_gap_conflict,
    policy_level_gaps, EPSILON_COSINE,
)
from prdrl.diagnostics.common_state_diagnostic import (
    build_common_state_record, cross_evaluate,
    action_disagreement_fraction, action_transition_matrix,
    q_margin_stats, stratify_by_delta_severity, _q_margin,
)
from prdrl.diagnostics.signal_diagnostic import (
    r1_signal_statistics, r1_activity,
    regional_td_error, joint_td_error, td_error_ratio, td_error_summary,
)
from prdrl.diagnostics.statistical_diagnostic import (
    aggregate_policy_env_scenario,
    crossed_bootstrap_ci, scenario_gap_bootstrap,
    o1_support_verdict,
)
from prdrl.diagnostics.diagnostic_config import (
    DIAGNOSTIC_VERSION, make_diagnostic_config, diagnostic_config_sha256,
    validate_upstream_hashes,
    REQUIRED_QMIX_SHA, REQUIRED_C2_CREDIT_SHA, REQUIRED_R1_SHA, REQUIRED_V3_SHA,
    DIAGNOSTIC_SEEDS, DIAGNOSTIC_BOOTSTRAP_N, DIAGNOSTIC_BOOTSTRAP_SEED,
    POLICY_SEEDS, SCENARIOS, N_DIAGNOSTIC_ENVS, DIAGNOSTIC_EPISODES,
)

RESULTS_DIR = Path("results/E4OI_scenario_tradeoff_diagnostic_implementation_validation")
E4O_DIR = Path("results/E4O_scenario_tradeoff_diagnostic_design")
SYNTH_SEED = 99001


def _load(path):
    with open(path) as f:
        return json.load(f)


def _synth_networks(seed=SYNTH_SEED):
    torch.manual_seed(seed)
    c1 = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    torch.manual_seed(seed + 1)
    c2 = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    torch.manual_seed(seed + 2)
    mixer = QMIXMixerV1()
    return c1, c2, mixer


# ==========================================================================
# T1-T5: E4OI experiment ran and output files exist
# ==========================================================================

class TestT01OutputExists:
    def test_verdict_exists(self):
        assert (RESULTS_DIR / "E4OI_verdict.json").is_file()

    def test_verdict_complete(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["E4OI_COMPLETE"] == "YES"

    def test_diagnostic_config_exists(self):
        assert (RESULTS_DIR / "E4OI_diagnostic_config.json").is_file()

    def test_report_exists(self):
        assert (RESULTS_DIR / "E4OI_report.md").is_file()

    def test_pre_manifest_pass(self):
        v = _load(RESULTS_DIR / "E4OI_pre_manifest.json")
        assert v["PASS"] is True


# ==========================================================================
# T6-T10: Historical integrity
# ==========================================================================

class TestT06Historical:
    def test_historical_pass(self):
        v = _load(RESULTS_DIR / "E4OI_historical_integrity.json")
        assert v["PASS"] is True

    def test_checkpoint_count_10(self):
        v = _load(RESULTS_DIR / "E4OI_historical_integrity.json")
        assert v["checkpoint_count"] == 10

    def test_qmix_sha_match(self):
        v = _load(RESULTS_DIR / "E4OI_historical_integrity.json")
        assert v["sha_match_qmix"] is True

    def test_c2_sha_match(self):
        v = _load(RESULTS_DIR / "E4OI_historical_integrity.json")
        assert v["sha_match_c2"] is True

    def test_e4o_historical_pass(self):
        v = _load(RESULTS_DIR / "E4OI_historical_integrity.json")
        assert v["E4O_historical_all_pass"] is True


# ==========================================================================
# T11-T14: Upstream config hashes
# ==========================================================================

class TestT11UpstreamConfig:
    def test_all_pass(self):
        v = _load(RESULTS_DIR / "E4OI_upstream_config_verification.json")
        assert v["all_pass"] is True

    def test_qmix_live_sha(self):
        assert qmix_config_sha256() == REQUIRED_QMIX_SHA

    def test_c2_live_sha(self):
        assert c2_credit_config_sha256() == REQUIRED_C2_CREDIT_SHA

    def test_validate_upstream(self):
        result = validate_upstream_hashes(REQUIRED_QMIX_SHA, REQUIRED_C2_CREDIT_SHA,
                                          REQUIRED_R1_SHA, REQUIRED_V3_SHA)
        assert result["all_pass"]


# ==========================================================================
# T15-T18: Diagnostic config
# ==========================================================================

class TestT15DiagConfig:
    def test_version(self):
        assert DIAGNOSTIC_VERSION == "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_V1"

    def test_sha_deterministic(self):
        s1 = diagnostic_config_sha256()
        s2 = diagnostic_config_sha256()
        assert s1 == s2

    def test_config_fields(self):
        cfg = make_diagnostic_config()
        for key in ["diagnostic_version", "qmix_config_sha256", "bootstrap_N",
                     "bootstrap_seed", "state_dim", "gradient_mode"]:
            assert key in cfg

    def test_sha_matches_verdict(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["E4OI_DIAGNOSTIC_CONFIG_SHA256"] == diagnostic_config_sha256()


# ==========================================================================
# T19-T22: Seed firewall
# ==========================================================================

class TestT19SeedFirewall:
    def test_firewall_pass(self):
        v = _load(RESULTS_DIR / "E4OI_seed_firewall.json")
        assert v["PASS"] is True

    def test_disjoint(self):
        v = _load(RESULTS_DIR / "E4OI_seed_firewall.json")
        assert v["disjoint"] is True

    def test_access_count_zero(self):
        v = _load(RESULTS_DIR / "E4OI_seed_firewall.json")
        assert v["O_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_seeds_range(self):
        assert DIAGNOSTIC_SEEDS == list(range(2301, 2361))


# ==========================================================================
# T23-T28: Gradient extraction
# ==========================================================================

class TestT23GradientExtraction:
    def test_g_J_pass(self):
        v = _load(RESULTS_DIR / "E4OI_g_J_extraction.json")
        assert v["PASS"] is True

    def test_g_R_pass(self):
        v = _load(RESULTS_DIR / "E4OI_g_R_extraction.json")
        assert v["PASS"] is True

    def test_g_J_nonzero(self):
        v = _load(RESULTS_DIR / "E4OI_g_J_extraction.json")
        assert v["g_J_nonzero"] is True

    def test_g_R_nonzero(self):
        v = _load(RESULTS_DIR / "E4OI_g_R_extraction.json")
        assert v["g_R_nonzero"] is True

    def test_g_J_g_R_same_dim(self):
        v = _load(RESULTS_DIR / "E4OI_g_R_extraction.json")
        assert v["g_R_same_dim_as_g_J"] is True

    def test_extract_returns_flat(self):
        torch.manual_seed(SYNTH_SEED)
        net = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
        x = torch.randn(4, STATE_DIM)
        y = torch.randn(4, N_ACTIONS)
        loss = nn.MSELoss()(net(x), y)
        g = extract_utility_gradient(loss, net)
        assert g.dim() == 1
        total_params = sum(p.numel() for p in net.parameters())
        assert g.numel() == total_params


# ==========================================================================
# T29-T35: Cosine similarity
# ==========================================================================

class TestT29Cosine:
    def test_aligned(self):
        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.tensor([2.0, 0.0, 2.0])
        r = cosine_similarity_jr(a, b)
        assert r["cosine_valid"]
        assert abs(r["cosine_JR"] - 1.0) < 1e-6

    def test_conflict(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([-1.0, 0.0, 0.0])
        r = cosine_similarity_jr(a, b)
        assert r["cosine_valid"]
        assert abs(r["cosine_JR"] - (-1.0)) < 1e-6

    def test_orthogonal(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        r = cosine_similarity_jr(a, b)
        assert r["cosine_valid"]
        assert abs(r["cosine_JR"]) < 1e-6

    def test_zero_invalid(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        r = cosine_similarity_jr(a, b)
        assert not r["cosine_valid"]
        assert r["cosine_JR"] is None

    def test_both_zero(self):
        a = torch.zeros(3)
        b = torch.zeros(3)
        r = cosine_similarity_jr(a, b)
        assert not r["cosine_valid"]
        assert r["cosine_JR"] is None
        assert r["reason"] == "zero_gradient"

    def test_cosine_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_cosine_verification.json")
        assert v["PASS"] is True

    def test_norm_norms_correct(self):
        g = torch.tensor([3.0, 4.0])
        assert abs(gradient_norms(g) - 5.0) < 1e-6


# ==========================================================================
# T36-T38: Gradient norm ratio
# ==========================================================================

class TestT36NormRatio:
    def test_ratio_half(self):
        assert abs(gradient_norm_ratio(3.0, 6.0) - 0.5) < 1e-6

    def test_ratio_zero(self):
        assert gradient_norm_ratio(0.0, 0.0) < 1e-6

    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_norm_ratio_verification.json")
        assert v["PASS"] is True


# ==========================================================================
# T39-T42: Composite gradient identity
# ==========================================================================

class TestT39CompositeIdentity:
    def test_identity_holds(self):
        v = _load(RESULTS_DIR / "E4OI_composite_gradient_identity.json")
        assert v["identity_holds"] is True

    def test_identity_max_diff_small(self):
        v = _load(RESULTS_DIR / "E4OI_composite_gradient_identity.json")
        assert v["max_abs_diff"] < 1e-5

    def test_synthetic_identity(self):
        g_j = torch.randn(100)
        g_r = torch.randn(100)
        g_c2 = 0.5 * g_j + 0.5 * g_r
        result = composite_gradient_identity(g_j, g_r, g_c2)
        assert result["identity_holds"]

    def test_identity_fails_on_wrong(self):
        g_j = torch.randn(100)
        g_r = torch.randn(100)
        g_wrong = torch.randn(100)
        result = composite_gradient_identity(g_j, g_r, g_wrong)
        assert not result["identity_holds"]


# ==========================================================================
# T43-T45: Mixer gradient
# ==========================================================================

class TestT43MixerGradient:
    def test_no_regional_path(self):
        v = _load(RESULTS_DIR / "E4OI_mixer_gradient_check.json")
        assert v["no_regional_path"] is True

    def test_mixer_check_synthetic(self):
        g_j = torch.randn(50)
        g_c2 = 0.5 * g_j
        result = mixer_gradient_check(g_j, g_c2)
        assert result["no_regional_path"]

    def test_mixer_check_fails_on_wrong(self):
        g_j = torch.randn(50)
        g_wrong = torch.randn(50)
        result = mixer_gradient_check(g_j, g_wrong)
        assert not result["no_regional_path"]


# ==========================================================================
# T46-T49: Common-state evaluation
# ==========================================================================

class TestT46CommonState:
    def test_cross_eval_pass(self):
        v = _load(RESULTS_DIR / "E4OI_common_state_evaluation.json")
        assert v["PASS"] is True

    def test_cross_eval_shapes(self):
        c1, c2, _ = _synth_networks()
        states = torch.randn(10, STATE_DIM)
        result = cross_evaluate(states, c1, c2)
        assert result["Q_C1"].shape == (10, N_ACTIONS)
        assert result["argmax_C1"].shape == (10,)
        assert result["margin_C1"].shape == (10,)

    def test_single_state(self):
        c1, c2, _ = _synth_networks()
        state = torch.randn(STATE_DIM)
        result = cross_evaluate(state, c1, c2)
        assert result["Q_C1"].shape == (1, N_ACTIONS)

    def test_schema_pass(self):
        v = _load(RESULTS_DIR / "E4OI_common_state_schema.json")
        assert v["PASS"] is True


# ==========================================================================
# T50-T53: Action disagreement
# ==========================================================================

class TestT50ActionDisagreement:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_action_disagreement.json")
        assert v["PASS"] is True

    def test_known_d_action(self):
        c1 = torch.tensor([0, 1, 2, 3])
        c2 = torch.tensor([1, 1, 3, 3])
        assert abs(action_disagreement_fraction(c1, c2) - 0.5) < 1e-6

    def test_transition_matrix_shape(self):
        c1 = torch.tensor([0, 1, 2, 3])
        c2 = torch.tensor([1, 1, 3, 3])
        tm = action_transition_matrix(c1, c2)
        assert tm.shape == (4, 4)
        assert tm.sum() == 4

    def test_all_agree(self):
        c1 = torch.tensor([0, 1, 2, 3])
        assert abs(action_disagreement_fraction(c1, c1) - 0.0) < 1e-6


# ==========================================================================
# T54-T57: Q-margin
# ==========================================================================

class TestT54QMargin:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_q_margin_verification.json")
        assert v["PASS"] is True

    def test_known_margins(self):
        q = torch.tensor([[1.0, 0.5, 0.2, 0.1]])
        m = _q_margin(q)
        assert abs(m.item() - 0.5) < 1e-6

    def test_equal_q(self):
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        m = _q_margin(q)
        assert abs(m.item()) < 1e-6

    def test_margin_stats_pass(self):
        v = _load(RESULTS_DIR / "E4OI_q_margin_stats.json")
        assert v["PASS"] is True


# ==========================================================================
# T58-T60: Scenario gap formulas
# ==========================================================================

class TestT58ScenarioGap:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_scenario_gap_verification.json")
        assert v["PASS"] is True

    def test_g_cos_formula(self):
        g = scenario_gap_cosine(-0.3, 0.5, 0.2)
        assert abs(g - (-0.3 - 0.5 * (0.5 + 0.2))) < 1e-10

    def test_g_conflict_formula(self):
        g = scenario_gap_conflict(0.6, 0.2, 0.3)
        assert abs(g - (0.6 - 0.5 * (0.2 + 0.3))) < 1e-10


# ==========================================================================
# T61-T64: Aggregation
# ==========================================================================

class TestT61Aggregation:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_aggregation_verification.json")
        assert v["PASS"] is True

    def test_cell_mean(self):
        records = [
            {"policy": 1, "diagnostic_environment": 1, "scenario": "A", "val": 2.0},
            {"policy": 1, "diagnostic_environment": 1, "scenario": "A", "val": 4.0},
        ]
        agg = aggregate_policy_env_scenario(records, "val")
        assert abs(agg["cell_means"][(1, 1, "A")] - 3.0) < 1e-10

    def test_missing_metric(self):
        records = [
            {"policy": 1, "diagnostic_environment": 1, "scenario": "A", "val": None},
        ]
        agg = aggregate_policy_env_scenario(records, "val")
        assert len(agg["cell_means"]) == 0

    def test_scenario_matrices(self):
        records = []
        for p in [1, 2]:
            for e in [10, 20]:
                for s in ["PERSISTENT", "STATIONARY", "MIXED"]:
                    records.append({"policy": p, "diagnostic_environment": e,
                                    "scenario": s, "val": 1.0})
        agg = aggregate_policy_env_scenario(records, "val")
        assert "PERSISTENT" in agg["scenario_matrices"]
        assert agg["scenario_matrices"]["PERSISTENT"].shape == (2, 2)


# ==========================================================================
# T65-T68: Crossed bootstrap
# ==========================================================================

class TestT65Bootstrap:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_bootstrap_verification.json")
        assert v["PASS"] is True

    def test_ci_contains_mean(self):
        mat = np.array([[0.1, 0.2], [0.3, 0.4]])
        boot = crossed_bootstrap_ci(mat, n_bootstrap=5000, seed=42)
        assert boot["ci_lower"] < boot["point"] < boot["ci_upper"]

    def test_deterministic(self):
        mat = np.array([[0.1, 0.2], [0.3, 0.4]])
        b1 = crossed_bootstrap_ci(mat, n_bootstrap=5000, seed=42)
        b2 = crossed_bootstrap_ci(mat, n_bootstrap=5000, seed=42)
        assert b1["ci_lower"] == b2["ci_lower"]
        assert b1["ci_upper"] == b2["ci_upper"]

    def test_gap_bootstrap_pass(self):
        v = _load(RESULTS_DIR / "E4OI_gap_bootstrap_verification.json")
        assert v["PASS"] is True


# ==========================================================================
# T69-T74: O1 truth table
# ==========================================================================

class TestT69O1TruthTable:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_o1_truth_table.json")
        assert v["PASS"] is True

    def test_supported(self):
        assert o1_support_verdict(-0.5, -0.1, 5, 5, 0.3, 5) == "SUPPORTED"

    def test_plausible_ci_crosses(self):
        assert o1_support_verdict(-0.5, 0.05, 5, 5, 0.3, 5) == "PLAUSIBLE"

    def test_plausible_few_policies(self):
        assert o1_support_verdict(-0.5, -0.1, 3, 5, 0.3, 5) == "PLAUSIBLE"

    def test_not_supported(self):
        assert o1_support_verdict(0.1, 0.3, 1, 5, -0.1, 1) == "NOT_SUPPORTED"

    def test_plausible_mixed(self):
        assert o1_support_verdict(-0.1, 0.1, 3, 5, -0.05, 2) == "PLAUSIBLE"


# ==========================================================================
# T75-T78: R1 signal
# ==========================================================================

class TestT75R1Signal:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_r1_signal_verification.json")
        assert v["PASS"] is True

    def test_all_zero(self):
        stats = r1_signal_statistics(np.zeros(10))
        assert stats["zero_fraction"] == 1.0
        assert stats["positive_fraction"] == 0.0

    def test_activity(self):
        r = np.array([0.0, 0.1, 0.0, -0.2, 0.0])
        assert abs(r1_activity(r) - 0.4) < 1e-10

    def test_empty(self):
        stats = r1_signal_statistics(np.array([]))
        assert stats["n"] == 0


# ==========================================================================
# T79-T82: TD error
# ==========================================================================

class TestT79TDError:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_td_error_verification.json")
        assert v["PASS"] is True

    def test_regional_td(self):
        q = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.5, 1.5])
        r = regional_td_error(q, y)
        assert torch.allclose(r["abs_delta_R"], torch.tensor([0.5, 0.5]))

    def test_joint_td(self):
        q = torch.tensor([1.0])
        y = torch.tensor([2.0])
        r = joint_td_error(q, y)
        assert abs(r["mean_abs_delta_J"] - 1.0) < 1e-6

    def test_ratio(self):
        r = td_error_ratio(torch.tensor([0.5]), torch.tensor([1.0]))
        assert abs(r.item() - 0.5) < 1e-6


# ==========================================================================
# T83-T85: State stratification
# ==========================================================================

class TestT83Stratification:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_stratification_verification.json")
        assert v["PASS"] is True

    def test_all_zero(self):
        states = np.zeros((5, 9))
        s = stratify_by_delta_severity(states)
        assert s["n_delta_severity_zero"] == 5
        assert s["n_delta_severity_positive"] == 0

    def test_all_positive(self):
        states = np.ones((5, 9))
        s = stratify_by_delta_severity(states)
        assert s["n_delta_severity_zero"] == 0
        assert s["n_delta_severity_positive"] == 5


# ==========================================================================
# T86-T88: Conflict fraction
# ==========================================================================

class TestT86ConflictFraction:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_conflict_fraction_verification.json")
        assert v["PASS"] is True

    def test_all_positive(self):
        assert abs(conflict_fraction([0.5, 0.3, 0.1]) - 0.0) < 1e-10

    def test_all_negative(self):
        assert abs(conflict_fraction([-0.5, -0.3, -0.1]) - 1.0) < 1e-10


# ==========================================================================
# T89-T91: Policy-level gaps
# ==========================================================================

class TestT89PolicyGaps:
    def test_result_pass(self):
        v = _load(RESULTS_DIR / "E4OI_policy_level_gaps.json")
        assert v["PASS"] is True

    def test_all_negative(self):
        v = _load(RESULTS_DIR / "E4OI_policy_level_gaps.json")
        assert v["n_policies_G_cos_negative"] == 5

    def test_all_positive_conflict(self):
        v = _load(RESULTS_DIR / "E4OI_policy_level_gaps.json")
        assert v["n_policies_G_conflict_positive"] == 5


# ==========================================================================
# T92-T94: No mutation
# ==========================================================================

class TestT92NoMutation:
    def test_all_zero(self):
        v = _load(RESULTS_DIR / "E4OI_no_mutation.json")
        assert v["all_zero"] is True

    def test_no_optimizer_steps(self):
        v = _load(RESULTS_DIR / "E4OI_no_mutation.json")
        assert v["optimizer_steps"] == 0

    def test_no_parameter_changes(self):
        v = _load(RESULTS_DIR / "E4OI_no_mutation.json")
        assert v["parameter_changes"] == 0


# ==========================================================================
# T95-T97: Version guards + no performance verdict
# ==========================================================================

class TestT95VersionGuards:
    def test_version_pass(self):
        v = _load(RESULTS_DIR / "E4OI_version_guards.json")
        assert v["PASS"] is True

    def test_no_performance_verdict(self):
        v = _load(RESULTS_DIR / "E4OI_no_performance_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"
        assert v["E4MF_REMAINS_AUTHORITATIVE"] is True

    def test_verdict_no_perf(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"


# ==========================================================================
# T98-T100: Matrix + diagnostic seeds
# ==========================================================================

class TestT98Matrix:
    def test_matrix_pass(self):
        v = _load(RESULTS_DIR / "E4OI_matrix_verification.json")
        assert v["PASS"] is True

    def test_total_1800(self):
        assert 2 * 5 * 60 * 3 == DIAGNOSTIC_EPISODES

    def test_seeds_60(self):
        assert len(DIAGNOSTIC_SEEDS) == 60


# ==========================================================================
# T101-T104: Final verdict + next stage
# ==========================================================================

class TestT101FinalVerdict:
    def test_complete(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["E4OI_COMPLETE"] == "YES"

    def test_next_stage(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["NEXT_STAGE"] == "E4OD_SCENARIO_TRADEOFF_DIAGNOSTIC_EXECUTION"

    def test_access_count(self):
        v = _load(RESULTS_DIR / "E4OI_verdict.json")
        assert v["O_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_full_regression(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
             "--tb=no", "--no-header",
             "--ignore=tests/test_e4oi_scenario_tradeoff_diagnostic_implementation_validation.py",
             "--ignore=tests/test_e4mi_joint_credit_preservation_implementation_validation.py",
             "--ignore=tests/test_e4mt_credit_preservation_ab_training_protocol_freeze.py",
             "--ignore=tests/test_e4od_scenario_tradeoff_diagnostic_execution.py",
             "--ignore=tests/test_e4odr_e4od_report_artifact_reconciliation.py",
             "--ignore=tests/test_e4p_nonmonotonic_interaction_diagnostic_design.py",
             "--ignore=tests/test_e4pi_nonmonotonic_interaction_diagnostic_implementation_validation.py",
             "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4rtr_v2_pairwise_cg_ab_training_execution.py",
             "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
            timeout=1200,
        )
        assert result.returncode == 0, f"Regression failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
