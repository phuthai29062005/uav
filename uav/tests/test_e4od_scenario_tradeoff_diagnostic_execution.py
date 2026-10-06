"""Tests for E4OD — Scenario Trade-off Diagnostic Execution.

76+ tests covering: output existence, historical integrity, upstream config,
checkpoint integrity, seed firewall, diagnostic matrix, gradient diagnostics,
scenario gaps, bootstrap, O1 verdict, common-state evaluation, R1 signal,
TD error, stratification, hypothesis reassessment, no-learning audit,
post-integrity, routing, and scientific constraints.

NOTE: This test file does NOT recursively invoke the complete pytest suite.
Full regression should be run externally.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "src")

RESULTS_DIR = Path("results/E4OD_scenario_tradeoff_diagnostic_execution")
E4O_DIR = Path("results/E4O_scenario_tradeoff_diagnostic_design")
E4OI_DIR = Path("results/E4OI_scenario_tradeoff_diagnostic_implementation_validation")
E4MF_DIR = Path("results/E4MF_credit_preservation_ab_final_evaluation_execution")
E4N_DIR = Path("results/E4N_credit_preservation_failure_analysis")

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
DIAGNOSTIC_SEEDS = list(range(2301, 2361))

FROZEN_QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C2_CREDIT_SHA = "56f43ade44533410fad091933e44db9f2c869ea0a56ddd98a7cd6dec824d5dd8"
FROZEN_R1_SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
FROZEN_V3_SHA = "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"


def _load(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# T1-T5: Output existence
# ---------------------------------------------------------------------------
class TestT01OutputExistence:
    REQUIRED_FILES = [
        "E4OD_verdict.json",
        "E4OD_pre_manifest.json",
        "E4OD_historical_integrity.json",
        "E4OD_upstream_config.json",
        "E4OD_seed_firewall.json",
        "E4OD_checkpoint_integrity.json",
        "E4OD_diagnostic_config.json",
        "E4OD_episode_results.csv",
        "E4OD_matrix_summary.json",
        "E4OD_gradient_diagnostics.json",
        "E4OD_scenario_gaps.json",
        "E4OD_gradient_bootstrap.json",
        "E4OD_conflict_bootstrap.json",
        "E4OD_o1_verdict.json",
        "E4OD_common_state_summary.json",
        "E4OD_cross_evaluation.json",
        "E4OD_r1_signal.json",
        "E4OD_td_error.json",
        "E4OD_stratification.json",
        "E4OD_gradient_norms.json",
        "E4OD_hypothesis_reassessment.json",
        "E4OD_post_integrity.json",
        "E4OD_no_learning_audit.json",
        "E4OD_seed_usage.json",
        "E4OD_report.md",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_file_exists(self, fname):
        assert (RESULTS_DIR / fname).exists(), f"Missing: {fname}"

    def test_results_dir_exists(self):
        assert RESULTS_DIR.exists()

    def test_verdict_is_json(self):
        v = _load(RESULTS_DIR / "E4OD_verdict.json")
        assert isinstance(v, dict)

    def test_at_least_25_files(self):
        count = len([f for f in RESULTS_DIR.iterdir() if f.is_file()])
        assert count >= 25, f"Only {count} output files"


# ---------------------------------------------------------------------------
# T6-T10: Historical integrity
# ---------------------------------------------------------------------------
class TestT06HistoricalIntegrity:
    @pytest.fixture(scope="class")
    def hist(self):
        return _load(RESULTS_DIR / "E4OD_historical_integrity.json")

    def test_e4f_failed(self, hist):
        assert hist["historical_verdicts"]["E4F"] == "FAILED"

    def test_e4hf_failed(self, hist):
        assert hist["historical_verdicts"]["E4HF"] == "FAILED"

    def test_e4if_validated(self, hist):
        assert hist["historical_verdicts"]["E4IF"] in ("VALIDATED", "REWARD_CAUSAL_VALIDATED")

    def test_e4kf_failed(self, hist):
        assert hist["historical_verdicts"]["E4KF"] in ("FAILED", "COORDINATION_CAUSAL_FAILED")

    def test_e4mf_failed(self, hist):
        assert hist["historical_verdicts"]["E4MF"] in ("FAILED", "CREDIT_PRESERVATION_CAUSAL_FAILED")

    def test_all_preserved(self, hist):
        assert hist["all_preserved"] is True


# ---------------------------------------------------------------------------
# T11-T14: Upstream config hashes
# ---------------------------------------------------------------------------
class TestT11UpstreamConfig:
    @pytest.fixture(scope="class")
    def cfg(self):
        return _load(RESULTS_DIR / "E4OD_upstream_config.json")

    def test_upstream_all_pass(self, cfg):
        assert cfg["upstream_verification"]["all_pass"] is True

    def test_qmix_sha(self, cfg):
        assert cfg["qmix_sha_ok"] is True

    def test_c2_sha(self, cfg):
        assert cfg["c2_sha_ok"] is True

    def test_diagnostic_config_matches_e4oi(self):
        dcfg = _load(RESULTS_DIR / "E4OD_diagnostic_config.json")
        assert dcfg["matches_e4oi"] is True


# ---------------------------------------------------------------------------
# T15-T18: Checkpoint integrity
# ---------------------------------------------------------------------------
class TestT15CheckpointIntegrity:
    @pytest.fixture(scope="class")
    def ckpt(self):
        return _load(RESULTS_DIR / "E4OD_checkpoint_integrity.json")

    def test_all_match(self, ckpt):
        assert ckpt["all_match"] is True

    def test_v3_match(self, ckpt):
        assert ckpt["v3_match"] is True

    def test_10_checkpoints(self, ckpt):
        assert len(ckpt["checkpoint_hashes"]) == 10

    def test_each_hash_matches(self, ckpt):
        for key, info in ckpt["checkpoint_hashes"].items():
            assert info["match"] is True, f"Checkpoint {key} hash mismatch"


# ---------------------------------------------------------------------------
# T19-T22: Seed firewall
# ---------------------------------------------------------------------------
class TestT19SeedFirewall:
    @pytest.fixture(scope="class")
    def fw(self):
        return _load(RESULTS_DIR / "E4OD_seed_firewall.json")

    def test_no_overlap(self, fw):
        assert fw["no_overlap"] is True

    def test_diagnostic_seeds_correct(self, fw):
        assert fw["diagnostic_seeds"] == DIAGNOSTIC_SEEDS

    def test_seeds_range(self, fw):
        seeds = fw["diagnostic_seeds"]
        assert seeds[0] == 2301
        assert seeds[-1] == 2360
        assert len(seeds) == 60

    def test_seed_usage(self):
        usage = _load(RESULTS_DIR / "E4OD_seed_usage.json")
        assert usage["unique_seeds"] == 60
        assert usage["status"] == "CONSUMED_DIAGNOSTIC"


# ---------------------------------------------------------------------------
# T23-T28: Diagnostic matrix
# ---------------------------------------------------------------------------
class TestT23DiagnosticMatrix:
    @pytest.fixture(scope="class")
    def mat(self):
        return _load(RESULTS_DIR / "E4OD_matrix_summary.json")

    def test_1800_episodes(self, mat):
        assert mat["total_episodes"] == 1800

    def test_c1_900(self, mat):
        assert mat["c1_episodes"] == 900

    def test_c2_900(self, mat):
        assert mat["c2_episodes"] == 900

    def test_per_scenario_600(self, mat):
        for s in SCENARIOS:
            assert mat["per_scenario"][s] == 600

    def test_positive_transitions(self, mat):
        assert mat["total_transitions"] > 0

    def test_csv_exists(self):
        p = RESULTS_DIR / "E4OD_episode_results.csv"
        assert p.exists()
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) >= 1801  # header + 1800


# ---------------------------------------------------------------------------
# T29-T35: Gradient diagnostics
# ---------------------------------------------------------------------------
class TestT29GradientDiagnostics:
    @pytest.fixture(scope="class")
    def gd(self):
        return _load(RESULTS_DIR / "E4OD_gradient_diagnostics.json")

    def test_cosine_records_positive(self, gd):
        assert gd["cosine_records_count"] > 0

    def test_optimizer_steps_zero(self, gd):
        assert gd["optimizer_steps"] == 0

    def test_parameter_changes_zero(self, gd):
        assert gd["parameter_changes"] == 0

    def test_all_scenarios_have_cosine(self, gd):
        for s in SCENARIOS:
            assert s in gd["scenario_mean_cosine"]
            assert gd["scenario_mean_cosine"][s] is not None

    def test_all_scenarios_have_conflict(self, gd):
        for s in SCENARIOS:
            assert s in gd["scenario_conflict_fraction"]
            val = gd["scenario_conflict_fraction"][s]
            assert 0.0 <= val <= 1.0

    def test_cosine_in_range(self, gd):
        for s in SCENARIOS:
            c = gd["scenario_mean_cosine"][s]
            assert -1.0 <= c <= 1.0, f"{s}: cosine={c}"

    def test_rho_g_finite(self, gd):
        for s in SCENARIOS:
            r = gd["scenario_mean_rho_g"][s]
            assert r is not None and math.isfinite(r)


# ---------------------------------------------------------------------------
# T36-T40: Scenario gaps
# ---------------------------------------------------------------------------
class TestT36ScenarioGaps:
    @pytest.fixture(scope="class")
    def sg(self):
        return _load(RESULTS_DIR / "E4OD_scenario_gaps.json")

    def test_g_cos_finite(self, sg):
        assert math.isfinite(sg["G_cos"])

    def test_g_conflict_finite(self, sg):
        assert math.isfinite(sg["G_conflict"])

    def test_policy_g_cos_5_values(self, sg):
        assert len(sg["policy_level_gaps"]["policy_G_cos"]) == 5

    def test_policy_g_conflict_5_values(self, sg):
        assert len(sg["policy_level_gaps"]["policy_G_conflict"]) == 5

    def test_g_cos_formula(self, sg):
        sc = sg["scenario_cosines"]
        expected = sc["STATIONARY"] - 0.5 * (sc["PERSISTENT"] + sc["MIXED"])
        assert abs(sg["G_cos"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# T41-T45: Bootstrap
# ---------------------------------------------------------------------------
class TestT41Bootstrap:
    @pytest.fixture(scope="class")
    def gbs(self):
        return _load(RESULTS_DIR / "E4OD_gradient_bootstrap.json")

    @pytest.fixture(scope="class")
    def cbs(self):
        return _load(RESULTS_DIR / "E4OD_conflict_bootstrap.json")

    def test_bootstrap_n(self, gbs):
        assert gbs["n_bootstrap"] == 50000

    def test_bootstrap_seed(self, gbs):
        assert gbs["seed"] == 116001

    def test_ci_contains_point(self, gbs):
        assert gbs["G_cos_ci_lower"] <= gbs["G_cos_point"] <= gbs["G_cos_ci_upper"]

    def test_conflict_ci_contains_point(self, cbs):
        assert cbs["G_conflict_ci_lower"] <= cbs["G_conflict_point"] <= cbs["G_conflict_ci_upper"]

    def test_per_scenario_means_3(self, gbs):
        assert len(gbs["per_scenario_means"]) == 3


# ---------------------------------------------------------------------------
# T46-T52: O1 verdict
# ---------------------------------------------------------------------------
class TestT46O1Verdict:
    @pytest.fixture(scope="class")
    def o1(self):
        return _load(RESULTS_DIR / "E4OD_o1_verdict.json")

    def test_o1_is_valid(self, o1):
        assert o1["O1_SUPPORT"] in ("SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED", "NOT_AUDITABLE")

    def test_criteria_present(self, o1):
        c = o1["criteria"]
        assert "G_cos" in c
        assert "G_cos_ci_upper" in c
        assert "n_policies_G_cos_negative" in c
        assert "G_conflict" in c
        assert "n_policies_G_conflict_positive" in c

    def test_policy_g_cos_values_present(self, o1):
        assert len(o1["criteria"]["policy_G_cos_values"]) == 5

    def test_policy_g_conflict_values_present(self, o1):
        assert len(o1["criteria"]["policy_G_conflict_values"]) == 5

    def test_o1_logic_consistency(self, o1):
        c = o1["criteria"]
        if o1["O1_SUPPORT"] == "SUPPORTED":
            assert c["G_cos_negative"] is True
            assert c["ci_upper_negative"] is True
            assert c["n_policies_G_cos_negative"] >= 4
            assert c["G_conflict_positive"] is True
            assert c["n_policies_G_conflict_positive"] >= 4

    def test_verdict_matches_o1(self):
        v = _load(RESULTS_DIR / "E4OD_verdict.json")
        o1 = _load(RESULTS_DIR / "E4OD_o1_verdict.json")
        assert v["O1_SUPPORT"] == o1["O1_SUPPORT"]

    def test_o1_not_supported_at_least_one_criterion_fails(self, o1):
        if o1["O1_SUPPORT"] == "NOT_SUPPORTED":
            c = o1["criteria"]
            all_met = (
                c["G_cos_negative"]
                and c["ci_upper_negative"]
                and c["n_policies_G_cos_negative"] >= 4
                and c["G_conflict_positive"]
                and c["n_policies_G_conflict_positive"] >= 4
            )
            assert not all_met, "All criteria pass but verdict is NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# T53-T56: Common-state evaluation
# ---------------------------------------------------------------------------
class TestT53CommonState:
    @pytest.fixture(scope="class")
    def cs(self):
        return _load(RESULTS_DIR / "E4OD_common_state_summary.json")

    @pytest.fixture(scope="class")
    def xeval(self):
        return _load(RESULTS_DIR / "E4OD_cross_evaluation.json")

    def test_both_sources_populated(self, cs):
        assert cs["n_states_source_c1"] > 0
        assert cs["n_states_source_c2"] > 0

    def test_total_equals_sum(self, cs):
        assert cs["n_states_total"] == cs["n_states_source_c1"] + cs["n_states_source_c2"]

    def test_cross_eval_has_sources(self, xeval):
        assert "SOURCE_C1" in xeval
        assert "SOURCE_C2" in xeval
        assert "POOLED" in xeval

    def test_disagreement_in_range(self, xeval):
        for source in ["SOURCE_C1", "SOURCE_C2", "POOLED"]:
            for s in SCENARIOS:
                if s in xeval[source]:
                    d = xeval[source][s].get("disagreement_fraction", 0)
                    assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# T57-T60: R1 signal
# ---------------------------------------------------------------------------
class TestT57R1Signal:
    @pytest.fixture(scope="class")
    def r1(self):
        return _load(RESULTS_DIR / "E4OD_r1_signal.json")

    def test_all_scenarios(self, r1):
        for s in SCENARIOS:
            assert s in r1["per_scenario"]

    def test_fractions_sum_to_one(self, r1):
        for s in SCENARIOS:
            d = r1["per_scenario"][s]
            total = d["zero_fraction"] + d["positive_fraction"] + d["negative_fraction"]
            assert abs(total - 1.0) < 1e-6, f"{s}: fractions sum to {total}"

    def test_activity_in_range(self, r1):
        for s in SCENARIOS:
            a = r1["per_scenario"][s]["activity"]
            assert 0.0 <= a <= 1.0

    def test_n_positive(self, r1):
        for s in SCENARIOS:
            assert r1["per_scenario"][s]["n"] > 0


# ---------------------------------------------------------------------------
# T61-T64: TD error
# ---------------------------------------------------------------------------
class TestT61TDError:
    @pytest.fixture(scope="class")
    def td(self):
        return _load(RESULTS_DIR / "E4OD_td_error.json")

    def test_all_scenarios(self, td):
        for s in SCENARIOS:
            assert s in td["per_scenario"]

    def test_regional_abs_delta_positive(self, td):
        for s in SCENARIOS:
            r = td["per_scenario"][s]["regional"]
            assert r["mean_abs_delta_R"] >= 0

    def test_smoothl1_nonnegative(self, td):
        for s in SCENARIOS:
            r = td["per_scenario"][s]["regional"]
            assert r["mean_smoothl1"] >= 0

    def test_n_transitions_positive(self, td):
        for s in SCENARIOS:
            assert td["per_scenario"][s]["n_transitions"] > 0


# ---------------------------------------------------------------------------
# T65-T67: Stratification
# ---------------------------------------------------------------------------
class TestT65Stratification:
    @pytest.fixture(scope="class")
    def strat(self):
        return _load(RESULTS_DIR / "E4OD_stratification.json")

    def test_all_scenarios(self, strat):
        for s in SCENARIOS:
            assert s in strat["per_scenario"]

    def test_counts_consistent(self, strat):
        for s in SCENARIOS:
            d = strat["per_scenario"][s]
            assert d["n_delta_severity_zero"] + d["n_delta_severity_positive"] == d["n_total"]

    def test_zero_fraction_in_range(self, strat):
        for s in SCENARIOS:
            f = strat["per_scenario"][s]["zero_fraction"]
            assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# T68-T72: Hypothesis reassessment
# ---------------------------------------------------------------------------
class TestT68HypothesisReassessment:
    @pytest.fixture(scope="class")
    def hyp(self):
        return _load(RESULTS_DIR / "E4OD_hypothesis_reassessment.json")

    def test_n3_valid(self, hyp):
        assert hyp["N3_ACTION_SPECIALIZATION_SHIFT"]["verdict"] in (
            "SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED")

    def test_n4_valid(self, hyp):
        assert hyp["N4_Q_MARGIN_POLICY_DECISIVENESS_SHIFT"]["verdict"] in (
            "SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED")

    def test_n5_valid(self, hyp):
        assert hyp["N5_REGIONAL_CREDIT_OVERFIT_TO_DYNAMIC_SCENARIOS"]["verdict"] in (
            "SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED")

    def test_n6_preserved(self, hyp):
        assert hyp["N6_CHECKPOINT_SELECTION_SCENARIO_TENSION"]["preserved"] is True
        assert hyp["N6_CHECKPOINT_SELECTION_SCENARIO_TENSION"]["verdict"] == "SUPPORTED"

    def test_n9_preserved(self, hyp):
        assert hyp["N9_MONOTONIC_QMIX_LIMITATION_REMAINS"]["preserved"] is True


# ---------------------------------------------------------------------------
# T73-T76: No-learning audit + post-integrity
# ---------------------------------------------------------------------------
class TestT73NoLearning:
    @pytest.fixture(scope="class")
    def audit(self):
        return _load(RESULTS_DIR / "E4OD_no_learning_audit.json")

    @pytest.fixture(scope="class")
    def post(self):
        return _load(RESULTS_DIR / "E4OD_post_integrity.json")

    def test_optimizer_steps_zero(self, audit):
        assert audit["optimizer_steps"] == 0

    def test_parameter_changes_zero(self, audit):
        assert audit["parameter_changes"] == 0

    def test_target_updates_zero(self, audit):
        assert audit["target_updates"] == 0

    def test_replay_mutations_zero(self, audit):
        assert audit["replay_mutations"] == 0

    def test_post_integrity_pass(self, post):
        assert post["all_pass"] is True

    def test_checkpoint_unchanged(self, post):
        assert post["checkpoint_unchanged"] is True

    def test_v3_unchanged(self, post):
        assert post["v3_unchanged"] is True


# ---------------------------------------------------------------------------
# T77-T80: Performance verdict + scientific constraints
# ---------------------------------------------------------------------------
class TestT77ScientificConstraints:
    @pytest.fixture(scope="class")
    def verdict(self):
        return _load(RESULTS_DIR / "E4OD_verdict.json")

    def test_performance_verdict_not_computed(self, verdict):
        assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_e4mf_authoritative(self, verdict):
        assert verdict["E4MF_REMAINS_AUTHORITATIVE"] is True

    def test_e4od_complete(self, verdict):
        assert verdict["E4OD_COMPLETE"] == "YES"

    def test_no_learning(self, verdict):
        assert verdict["INTEGRITY"]["no_learning"] is True


# ---------------------------------------------------------------------------
# T81-T85: Verdict structure and routing
# ---------------------------------------------------------------------------
class TestT81VerdictAndRouting:
    @pytest.fixture(scope="class")
    def verdict(self):
        return _load(RESULTS_DIR / "E4OD_verdict.json")

    def test_o1_support_in_verdict(self, verdict):
        assert "O1_SUPPORT" in verdict
        assert verdict["O1_SUPPORT"] in ("SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED", "NOT_AUDITABLE")

    def test_next_stage_present(self, verdict):
        assert "NEXT_STAGE" in verdict
        assert verdict["NEXT_STAGE"].startswith("E4P_")

    def test_routing_valid(self, verdict):
        o1 = verdict["O1_SUPPORT"]
        ns = verdict["NEXT_STAGE"]
        routing = _load(E4O_DIR / "E4O_future_routing.json")["routing_rules"]
        if o1 == "SUPPORTED":
            assert ns == routing["O1_SUPPORTED"]["NEXT_STAGE"]

    def test_elapsed_reasonable(self, verdict):
        assert 10 < verdict["elapsed_seconds"] < 7200

    def test_hypothesis_reassessment_in_verdict(self, verdict):
        hr = verdict["HYPOTHESIS_REASSESSMENT"]
        for h in ["N3", "N4", "N5", "N6", "N7", "N8", "N9"]:
            assert h in hr


# ---------------------------------------------------------------------------
# T86-T88: Gradient norms
# ---------------------------------------------------------------------------
class TestT86GradientNorms:
    @pytest.fixture(scope="class")
    def norms(self):
        return _load(RESULTS_DIR / "E4OD_gradient_norms.json")

    def test_all_scenarios(self, norms):
        for s in SCENARIOS:
            assert s in norms["per_scenario"]

    def test_norms_positive(self, norms):
        for s in SCENARIOS:
            d = norms["per_scenario"][s]
            assert d["mean_norm_g_J"] > 0
            assert d["mean_norm_g_R"] >= 0

    def test_rho_g_nonnegative(self, norms):
        for s in SCENARIOS:
            assert norms["per_scenario"][s]["mean_rho_g"] >= 0


# ---------------------------------------------------------------------------
# T89-T91: Pre-manifest
# ---------------------------------------------------------------------------
class TestT89PreManifest:
    @pytest.fixture(scope="class")
    def man(self):
        return _load(RESULTS_DIR / "E4OD_pre_manifest.json")

    def test_pre_integrity_pass(self, man):
        assert man["pre_integrity"] == "PASS"

    def test_epsilon_zero(self, man):
        assert man["execution_mode"]["epsilon"] == 0.0

    def test_gradient_mode_offline(self, man):
        assert man["execution_mode"]["gradients"] == "OFFLINE_ONLY"


# ---------------------------------------------------------------------------
# T92-T94: Report
# ---------------------------------------------------------------------------
class TestT92Report:
    def test_report_exists(self):
        assert (RESULTS_DIR / "E4OD_report.md").exists()

    def test_report_nonempty(self):
        text = (RESULTS_DIR / "E4OD_report.md").read_text()
        assert len(text) > 100

    def test_report_contains_o1(self):
        text = (RESULTS_DIR / "E4OD_report.md").read_text()
        assert "O1" in text
        assert "NOT_COMPUTED" in text
