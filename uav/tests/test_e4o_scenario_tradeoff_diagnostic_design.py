"""Tests for E4O — Scenario Trade-off Diagnostic Design."""

import json
import os
import pytest

OUT_DIR = "results/E4O_scenario_tradeoff_diagnostic_design"
E4N_DIR = "results/E4N_credit_preservation_failure_analysis"


def load(fn):
    with open(os.path.join(OUT_DIR, fn)) as f:
        return json.load(f)


def load_e4n(fn):
    with open(os.path.join(E4N_DIR, fn)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return load("E4O_verdict.json")


@pytest.fixture(scope="module")
def integrity():
    return load("E4O_historical_integrity.json")


@pytest.fixture(scope="module")
def gradient_contract():
    return load("E4O_gradient_alignment_contract.json")


@pytest.fixture(scope="module")
def norm_contract():
    return load("E4O_gradient_norm_contract.json")


@pytest.fixture(scope="module")
def conflict_contract():
    return load("E4O_gradient_conflict_contract.json")


@pytest.fixture(scope="module")
def gap_contract():
    return load("E4O_primary_scenario_gap.json")


@pytest.fixture(scope="module")
def support_logic():
    return load("E4O_hypothesis_support_logic.json")


@pytest.fixture(scope="module")
def r1_contract():
    return load("E4O_r1_signal_contract.json")


@pytest.fixture(scope="module")
def td_contract():
    return load("E4O_td_error_contract.json")


@pytest.fixture(scope="module")
def state_bank():
    return load("E4O_common_state_bank_contract.json")


@pytest.fixture(scope="module")
def action_contract():
    return load("E4O_action_specialization_contract.json")


@pytest.fixture(scope="module")
def q_margin():
    return load("E4O_q_margin_contract.json")


@pytest.fixture(scope="module")
def stratification():
    return load("E4O_state_stratification_contract.json")


@pytest.fixture(scope="module")
def tradeoff():
    return load("E4O_joint_regional_tradeoff_contract.json")


@pytest.fixture(scope="module")
def mixer():
    return load("E4O_mixer_gradient_contract.json")


@pytest.fixture(scope="module")
def mono():
    return load("E4O_monotonicity_boundary.json")


@pytest.fixture(scope="module")
def seeds():
    return load("E4O_diagnostic_seed_contract.json")


@pytest.fixture(scope="module")
def disjoint():
    return load("E4O_seed_disjointness.json")


@pytest.fixture(scope="module")
def matrix():
    return load("E4O_diagnostic_matrix.json")


@pytest.fixture(scope="module")
def logging_schema():
    return load("E4O_transition_logging_schema.json")


@pytest.fixture(scope="module")
def gradient_offline():
    return load("E4O_offline_gradient_contract.json")


@pytest.fixture(scope="module")
def stats():
    return load("E4O_statistical_design.json")


@pytest.fixture(scope="module")
def routing():
    return load("E4O_future_routing.json")


@pytest.fixture(scope="module")
def ckpt():
    return load("E4O_checkpoint_lock.json")


@pytest.fixture(scope="module")
def firewall():
    return load("E4O_future_firewall.json")


@pytest.fixture(scope="module")
def no_perf():
    return load("E4O_no_performance_verdict.json")


@pytest.fixture(scope="module")
def e4oi():
    return load("E4O_e4oi_validation_contract.json")


@pytest.fixture(scope="module")
def hypothesis():
    return load("E4O_hypothesis_contract.json")


@pytest.fixture(scope="module")
def inferential():
    return load("E4O_inferential_unit_contract.json")


# ===== T1-T13: Historical integrity =====

class TestHistoricalIntegrity:
    def test_t01_e4n_complete(self, integrity):
        assert integrity["checks"]["C01_E4N_COMPLETE"] is True

    def test_t02_e4mf_failed(self, verdict):
        assert verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"
        assert verdict["E4MF_VERDICT_CHANGED"] == "NO"

    def test_t03_mixed_directional_positive(self, integrity):
        assert integrity["checks"]["C03_MIXED_DIRECTIONAL_POSITIVE"] is True

    def test_t04_persistent_pass(self, integrity):
        assert integrity["checks"]["C05_PERSISTENT_SAFETY_PASS"] is True

    def test_t05_stationary_fail(self, integrity):
        assert integrity["checks"]["C06_STATIONARY_SAFETY_FAIL"] is True

    def test_t06_persistent_5_of_5(self, verdict):
        assert verdict["PERSISTENT_POLICY_PATTERN"] == "C2_BETTER_5_OF_5"

    def test_t07_stationary_5_of_5_harm(self, verdict):
        assert verdict["STATIONARY_POLICY_PATTERN"] == "C2_WORSE_5_OF_5"

    def test_t08_n1_not_auditable(self, verdict):
        assert verdict["N1_PREVIOUS_STATUS"] == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"

    def test_t09_n2_not_supported(self, verdict):
        assert verdict["N2_STATUS"] == "NOT_SUPPORTED"

    def test_t10_n6_supported(self, verdict):
        assert verdict["N6_STATUS"] == "SUPPORTED"

    def test_t11_n7_not_supported(self, integrity):
        assert integrity["checks"]["C14_N7_NOT_SUPPORTED"] is True

    def test_t12_n10_supported(self, integrity):
        assert integrity["checks"]["C10_N10_SUPPORTED"] is True

    def test_t13_h4_revision_not_sufficient(self, integrity):
        assert integrity["checks"]["C17_H4_REVISION_NOT_SUFFICIENT"] is True


# ===== T14-T15: Checkpoint lock =====

class TestCheckpointLock:
    def test_t14_ten_frozen_checkpoints(self, ckpt):
        assert ckpt["frozen_checkpoints"]["count"] == 10
        assert len(ckpt["frozen_checkpoints"]["hashes"]) == 10

    def test_t15_no_reselection(self, ckpt):
        assert ckpt["no_checkpoint_reselection"] is True
        assert ckpt["no_checkpoint_redesign"] is True
        assert ckpt["no_alternate_epoch"] is True


# ===== T16-T22: Gradient formulas =====

class TestGradientFormulas:
    def test_t16_g_j_definition(self, gradient_contract):
        defn = gradient_contract["definition"]
        assert "UNWEIGHTED L_joint" in defn["g_J"]
        assert "online utility parameters" in defn["g_J"]
        assert defn["coefficients_excluded"] is True

    def test_t17_g_r_definition(self, gradient_contract):
        defn = gradient_contract["definition"]
        assert "UNWEIGHTED L_regional" in defn["g_R"]
        assert "online utility parameters" in defn["g_R"]

    def test_t18_cosine_formula(self, gradient_contract):
        cos = gradient_contract["cosine_JR"]
        assert "dot(g_J, g_R)" in cos["formula"]
        assert "||g_J|| * ||g_R||" in cos["formula"]
        assert cos["epsilon"] == 1e-12

    def test_t19_gradient_ratio(self, norm_contract):
        rho = norm_contract["rho_g"]
        assert "||g_R||" in rho["formula"]
        assert "||g_J||" in rho["formula"]
        assert rho["epsilon"] == 1e-12

    def test_t20_conflict_fraction(self, conflict_contract):
        cf = conflict_contract["conflict_fraction"]
        assert "cosine_JR < 0" in cf["formula"]
        assert "policy" in cf["aggregation"]
        assert "environment" in cf["aggregation"]
        assert "scenario" in cf["aggregation"]

    def test_t21_g_cos(self, gap_contract):
        g = gap_contract["G_cos"]
        assert "STATIONARY" in g["formula"]
        assert "PERSISTENT" in g["formula"]
        assert "MIXED" in g["formula"]
        assert "0.5" in g["formula"]

    def test_t22_g_conflict(self, gap_contract):
        g = gap_contract["G_conflict"]
        assert "STATIONARY" in g["formula"]
        assert "PERSISTENT" in g["formula"]
        assert "MIXED" in g["formula"]


# ===== T23-T24: O1 support logic =====

class TestO1SupportLogic:
    def test_t23_o1_support_logic_exact(self, support_logic):
        o1 = support_logic["O1_SCENARIO_SPECIFIC_GRADIENT_ALIGNMENT"]
        reqs = o1["SUPPORTED"]["requires_ALL"]
        assert len(reqs) == 5
        assert any("G_cos < 0" in r for r in reqs)
        assert any("CI" in r and "G_cos" in r for r in reqs)
        assert any("4/5" in r and "G_cos" in r for r in reqs)
        assert any("G_conflict > 0" in r for r in reqs)
        assert any("4/5" in r and "G_conflict" in r for r in reqs)
        assert support_logic["frozen_BEFORE_new_data"] is True

    def test_t24_no_causal_from_o1(self, support_logic):
        caveat = support_logic["important_caveat"]
        assert "does NOT mean" in caveat
        assert "caused" in caveat.lower() or "causal" in caveat.lower()


# ===== T25-T27: R1 and TD contracts =====

class TestR1AndTD:
    def test_t25_r1_activity_frozen(self, r1_contract):
        assert "R1_activity" in r1_contract
        assert r1_contract["per_scenario_records"]["zero_fraction"] is not None
        assert r1_contract["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]

    def test_t26_regional_td_error_frozen(self, td_contract):
        rt = td_contract["regional_td_error"]
        assert "y_local - q_local" in rt["delta_R"]
        assert rt["per_scenario"] is True

    def test_t27_joint_td_error_frozen(self, td_contract):
        jt = td_contract["joint_td_error"]
        assert "y_joint - Q_tot" in jt["delta_J"]


# ===== T28-T31: Common-state, action, Q-margin =====

class TestCommonState:
    def test_t28_both_source_arms(self, state_bank):
        assert "SOURCE_C1" in state_bank["state_bank"]["sources"]
        assert "SOURCE_C2" in state_bank["state_bank"]["sources"]

    def test_t29_same_state_evaluation(self, state_bank):
        ce = state_bank["cross_evaluation"]
        assert "C1 utility network" in ce["for_each_stored_state"][0]
        assert "C2 utility network" in ce["for_each_stored_state"][1]

    def test_t30_action_disagreement(self, action_contract):
        d = action_contract["D_action"]
        assert "argmax_C1 != argmax_C2" in d["formula"]
        assert d["per_scenario"] is True
        tm = action_contract["transition_matrix"]
        assert tm["shape"] == "4x4"
        assert tm["actions"] == ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]

    def test_t31_q_margin(self, q_margin):
        assert "top1(Q_C1) - top2(Q_C1)" in q_margin["margin_C1"]
        assert "top1(Q_C2) - top2(Q_C2)" in q_margin["margin_C2"]
        assert "margin_C2 - margin_C1" in q_margin["Delta_margin"]
        assert q_margin["no_success_threshold"] is True


# ===== T32-T34: State, composite, mixer =====

class TestStateAndMixer:
    def test_t32_delta_severity_stratum(self, stratification):
        ms = stratification["minimum_strata"]
        assert "delta_severity == 0" in ms["delta_severity_zero"]
        assert "delta_severity > 0" in ms["delta_severity_positive"]

    def test_t33_composite_gradient(self, tradeoff):
        assert "g_C2 = 0.5*g_J + 0.5*g_R" in tradeoff["per_batch_records"]["||g_C2||"]

    def test_t34_mixer_regional_absent(self, mixer):
        assert mixer["no_regional_path"] is True
        assert "0.5*L_joint ONLY" in mixer["mixer_gradient_source"]


# ===== T35: Monotonicity =====

class TestMonotonicity:
    def test_t35_monotonicity_causal_excluded(self, mono):
        assert mono["O_MONOTONICITY_CAUSAL_TEST"] == "NOT_INCLUDED"


# ===== T36-T40: Seeds and matrix =====

class TestSeedsAndMatrix:
    def test_t36_diagnostic_seeds(self, seeds):
        assert seeds["diagnostic_seeds"] == "2301..2360"
        assert seeds["n_seeds"] == 60
        assert seeds["diagnostic_seed_list"] == list(range(2301, 2361))

    def test_t37_seed_disjointness(self, disjoint):
        assert disjoint["disjoint"] is True
        assert disjoint["SEED_DISJOINTNESS_PASS"] is True
        assert disjoint["overlap"] == []

    def test_t38_diagnostic_access_zero(self, firewall):
        assert firewall["O_DIAGNOSTIC_ACCESS_COUNT"] == 0
        assert firewall["no_seed_instantiated"] is True

    def test_t39_matrix_1800(self, matrix):
        assert matrix["total_diagnostic_episodes"] == 1800
        assert matrix["matches_expected"] is True

    def test_t40_epsilon_zero(self, matrix):
        assert matrix["epsilon"] == 0.0
        assert matrix["evaluation_only"] is True
        assert matrix["no_learning"] is True


# ===== T41-T46: Schema, gradient, aggregation =====

class TestSchemaAndAggregation:
    def test_t41_transition_schema_complete(self, logging_schema):
        fields = logging_schema["per_dynamic_change_event"]
        required = [
            "source_arm", "policy_seed", "scenario",
            "diagnostic_environment_seed", "episode_change_index",
            "region_id", "active_mask", "local_state_9D",
            "central_state_55D", "requested_action", "realized_operator",
            "focal_R1_reward", "all_active_regional_R1_rewards",
            "r_joint", "next_local_state_9D", "next_central_state_55D",
            "done", "C1_utility_Q_vector", "C2_utility_Q_vector",
        ]
        for field in required:
            assert field in fields, f"Missing field: {field}"

    def test_t42_gradient_offline_only(self, gradient_offline):
        assert gradient_offline["gradient_is_diagnostic_only"] is True

    def test_t43_optimizer_steps_forbidden(self, gradient_offline):
        nm = gradient_offline["no_model_mutation"]
        assert nm["optimizer_steps"] == 0

    def test_t44_parameter_mutation_forbidden(self, gradient_offline):
        nm = gradient_offline["no_model_mutation"]
        assert nm["parameter_changes"] == 0
        assert nm["target_updates"] == 0
        assert nm["replay_mutation"] == 0

    def test_t45_inferential_aggregation(self, inferential):
        assert inferential["policy_axis"] == 5
        assert inferential["environment_axis"] == 60

    def test_t46_no_anchor_pseudoreplication(self, inferential):
        assert inferential["no_anchor_pseudoreplication"] is True


# ===== T47-T51: Bootstrap and primary =====

class TestBootstrapAndPrimary:
    def test_t47_bootstrap_axes(self, stats):
        b = stats["bootstrap"]
        assert b["policy_axis"] == 5
        assert b["environment_axis"] == 60

    def test_t48_bootstrap_n(self, stats):
        assert stats["bootstrap"]["N"] == 50000

    def test_t49_bootstrap_seed(self, stats):
        assert stats["bootstrap"]["seed"] == 116001

    def test_t50_primary_is_gradient_gap(self, gap_contract):
        assert gap_contract["primary_diagnostic_contrast"] == "G_cos"

    def test_t51_migd_not_primary(self, gap_contract):
        assert gap_contract["no_MIGD_primary"] is True


# ===== T52: No performance verdict =====

class TestNoPerformanceVerdict:
    def test_t52_no_performance_verdict(self, no_perf):
        assert no_perf["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"
        assert no_perf["E4MF_REMAINS_AUTHORITATIVE"] is True


# ===== T53: N5 diagnostic =====

class TestN5:
    def test_t53_n5_diagnostic_frozen(self, r1_contract):
        assert r1_contract["N5_addressable"] is True
        assert "SUPPORTED" in r1_contract["N5_support_rule"]


# ===== T54-T59: Routing =====

class TestRouting:
    def test_t54_o1_routing(self, routing):
        r = routing["routing_rules"]
        assert r["O1_SUPPORTED"]["NEXT_STAGE"] == "E4P_GRADIENT_CONFLICT_REVISION_DESIGN"

    def test_t55_action_routing(self, routing):
        r = routing["routing_rules"]
        assert r["O1_NOT_SUPPORTED_AND_N3_STRONGLY_SUPPORTED"]["NEXT_STAGE"] == "E4P_ACTION_SPECIALIZATION_REVISION_DESIGN"

    def test_t56_q_margin_routing(self, routing):
        r = routing["routing_rules"]
        assert r["O1_NOT_SUPPORTED_AND_N4_STRONGLY_SUPPORTED"]["NEXT_STAGE"] == "E4P_POLICY_DECISIVENESS_REVISION_DESIGN"

    def test_t57_credit_structure_routing(self, routing):
        r = routing["routing_rules"]
        assert r["N5_STRONGEST"]["NEXT_STAGE"] == "E4P_SCENARIO_CONDITIONAL_CREDIT_REVIEW"

    def test_t58_n8_routing(self, routing):
        r = routing["routing_rules"]
        assert r["N8_STRONGEST_WITHOUT_GRADIENT_CONFLICT"]["NEXT_STAGE"] == "E4P_JOINT_REGIONAL_OBJECTIVE_REVIEW"

    def test_t59_unresolved_routing(self, routing):
        r = routing["routing_rules"]
        assert r["ALL_MECHANISMS_INSUFFICIENT"]["NEXT_STAGE"] == "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"


# ===== T60: E4OI next stage =====

class TestE4OI:
    def test_t60_e4oi_next_stage(self, e4oi):
        assert e4oi["next_immediate_stage"] == "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"
        assert "E4O" in e4oi["do_not_open_seeds_during"]
        assert "E4OI" in e4oi["do_not_open_seeds_during"]


# ===== T61-T63: No training, no env, artifacts =====

class TestNoExecution:
    def test_t61_no_training(self, verdict):
        assert verdict["NEW_TRAINING"] == "NO"

    def test_t62_no_environment_execution(self, verdict):
        assert verdict["NEW_ENVIRONMENT_EXECUTION"] == "NO"
        assert verdict["O_DIAGNOSTIC_ACCESS_COUNT"] == 0

    def test_t63_historical_artifacts_unchanged(self):
        e4n_verdict = load_e4n("E4N_verdict.json")
        assert e4n_verdict["E4N_COMPLETE"] == "YES"
        assert e4n_verdict["ORIGINAL_E4MF_VERDICT"] == "FAILED"
        assert e4n_verdict["NEXT_STAGE"] == "E4O_SCENARIO_TRADEOFF_DIAGNOSTIC_DESIGN"


# ===== T64: Full regression placeholder =====

class TestRegression:
    def test_t64_e4o_complete(self, verdict):
        assert verdict["E4O_COMPLETE"] == "YES"
        assert verdict["NEXT_STAGE"] == "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"


# ===== Additional completeness tests =====

class TestAdditional:
    def test_t65_all_output_files_exist(self):
        expected = [
            "E4O_pre_manifest.json",
            "E4O_historical_integrity.json",
            "E4O_e4n_evidence_lock.json",
            "E4O_diagnostic_question.json",
            "E4O_hypothesis_contract.json",
            "E4O_gradient_alignment_contract.json",
            "E4O_gradient_norm_contract.json",
            "E4O_gradient_conflict_contract.json",
            "E4O_primary_scenario_gap.json",
            "E4O_r1_signal_contract.json",
            "E4O_td_error_contract.json",
            "E4O_common_state_bank_contract.json",
            "E4O_action_specialization_contract.json",
            "E4O_q_margin_contract.json",
            "E4O_state_stratification_contract.json",
            "E4O_joint_regional_tradeoff_contract.json",
            "E4O_mixer_gradient_contract.json",
            "E4O_monotonicity_boundary.json",
            "E4O_checkpoint_lock.json",
            "E4O_diagnostic_seed_contract.json",
            "E4O_seed_disjointness.json",
            "E4O_diagnostic_matrix.json",
            "E4O_transition_logging_schema.json",
            "E4O_offline_gradient_contract.json",
            "E4O_inferential_unit_contract.json",
            "E4O_statistical_design.json",
            "E4O_hypothesis_support_logic.json",
            "E4O_no_performance_verdict.json",
            "E4O_future_routing.json",
            "E4O_e4oi_validation_contract.json",
            "E4O_future_firewall.json",
            "E4O_protocol_frozen.json",
            "E4O_verdict.json",
            "E4O_report.md",
        ]
        for fn in expected:
            path = os.path.join(OUT_DIR, fn)
            assert os.path.isfile(path), f"Missing: {fn}"

    def test_t66_all_historical_integrity_pass(self, integrity):
        assert integrity["all_pass"] is True
        assert integrity["total_checks"] == 17
        assert integrity["passed_checks"] == 17

    def test_t67_e4kf_historical_correct(self, integrity):
        assert integrity["checks"]["C08_E4KF_PERSISTENT_SAFETY_FAIL"] is True
        assert integrity["checks"]["C09_E4KF_STATIONARY_SAFETY_PASS"] is True

    def test_t68_protocol_frozen_all(self):
        pf = load("E4O_protocol_frozen.json")
        assert pf["all_contracts_frozen"] is True
        for key, val in pf.items():
            if key not in ("stage", "all_contracts_frozen"):
                assert val == "FROZEN", f"{key} not frozen"

    def test_t69_diagnostic_question_no_rescue(self):
        dq = load("E4O_diagnostic_question.json")
        assert dq["E4O_CANNOT_RESCUE_E4MF"] is True
        assert dq["NOT_A_PERFORMANCE_TEST"] is True

    def test_t70_hypothesis_contract_updates(self, hypothesis):
        assert "N1" in hypothesis["hypothesis_may_update"]
        assert "N3" in hypothesis["hypothesis_may_update"]
        assert "N4" in hypothesis["hypothesis_may_update"]
        assert "N5" in hypothesis["hypothesis_may_update"]
        assert "N8" in hypothesis["hypothesis_may_update"]
        assert hypothesis["hypothesis_may_NOT_alter"] == "E4MF_VERDICT"

    def test_t71_checkpoint_hashes_preserved(self):
        ckpt = load("E4O_checkpoint_lock.json")
        hashes = ckpt["frozen_checkpoints"]["hashes"]
        assert len(hashes) == 10
        for key in hashes:
            assert len(hashes[key]) == 64

    def test_t72_seed_disjointness_all_ranges(self, disjoint):
        ranges = disjoint["prior_seed_ranges"]
        assert len(ranges) >= 18

    def test_t73_report_exists(self):
        path = os.path.join(OUT_DIR, "E4O_report.md")
        assert os.path.isfile(path)
        with open(path) as f:
            content = f.read()
        for section in ["## A.", "## B.", "## C.", "## D.", "## E.", "## F.",
                        "## G.", "## H.", "## I.", "## J.", "## K.", "## L.",
                        "## M.", "## N.", "## O.", "## P.", "## Q.", "## R.",
                        "## S.", "## T.", "## U.", "## V.", "## W.", "## X.",
                        "## Y."]:
            assert section in content, f"Missing section {section}"

    def test_t74_e4oi_15_validation_requirements(self, e4oi):
        reqs = e4oi["E4OI_validation_requirements"]
        assert len(reqs) == 15
        for i in range(1, 16):
            key = f"V{i:02d}"
            matching = [k for k in reqs if k.startswith(key)]
            assert len(matching) == 1, f"Missing validation {key}"
