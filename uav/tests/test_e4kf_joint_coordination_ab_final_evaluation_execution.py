"""
Tests for E4KF — Joint Coordination A/B Final Evaluation Execution.

Validates saved E4KF artifacts without re-running any experiment. Covers:
  T1-T10   prerequisites and historical integrity
  T11-T25  checkpoint integrity and C1 mixer integrity
  T26-T40  execution contracts (mode, matrix, pairing, reward, normalization)
  T41-T55  statistical design and primary effect
  T56-T65  classification, robustness, safety
  T66-T80  verdict logic, coordination mechanism gate, action diagnostics
  T81-T88  outputs, comparison, seed accounting
  T89-T97  post-integrity, no-learning, routing, no-rescue, historical preservation
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4KE_DIR = Path("results/E4KE_joint_coordination_ab_final_evaluation_protocol_freeze")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")

ARMS = ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"]
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


def _load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load(DIR / "E4KF_verdict.json")


@pytest.fixture(scope="module")
def manifest():
    return _load(DIR / "E4KF_pre_execution_manifest.json")


@pytest.fixture(scope="module")
def ck_integrity():
    return _load(DIR / "E4KF_checkpoint_integrity.json")


@pytest.fixture(scope="module")
def gates():
    return _load(DIR / "E4KF_scientific_gates.json")


@pytest.fixture(scope="module")
def bootstrap():
    return _load(DIR / "E4KF_bootstrap_results.json")


@pytest.fixture(scope="module")
def coord_diag():
    return _load(DIR / "E4KF_coordination_diagnostics.json")


@pytest.fixture(scope="module")
def seed_usage():
    return _load(DIR / "E4KF_final_seed_usage.json")


@pytest.fixture(scope="module")
def no_learning():
    return _load(DIR / "E4KF_no_learning_audit.json")


@pytest.fixture(scope="module")
def post_integrity():
    return _load(DIR / "E4KF_post_integrity.json")


@pytest.fixture(scope="module")
def val_comparison():
    return _load(DIR / "E4KF_validation_final_comparison.json")


# ===== T1-T10: prerequisites and historical integrity =====

class TestPrerequisites:
    def test_t1_e4kf_complete(self, verdict):
        assert verdict["E4KF_COMPLETE"] == "YES"

    def test_t2_final_matrix_complete(self, verdict):
        assert verdict["FINAL_MATRIX_COMPLETE"] == "YES"
        assert verdict["EPISODES_COMPLETE"] == "1800/1800"

    def test_t3_e4f_verdict_preserved(self, verdict):
        assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"

    def test_t4_e4hf_verdict_preserved(self, verdict):
        assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"

    def test_t5_e4if_verdict_preserved(self, verdict):
        assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"

    def test_t6_e4jf_verdict_preserved(self, verdict):
        assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_t7_manifest_prerequisites(self, manifest):
        assert manifest["E4KE_COMPLETE"] == "YES"
        assert manifest["E4KTR_COMPLETE"] == "YES"
        assert manifest["ORIGINAL_E4F_VERDICT"] == "FAILED"
        assert manifest["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        assert manifest["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        assert manifest["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_t8_manifest_integrity_checks(self, manifest):
        assert manifest["checkpoint_pre_hashes_match"] is True
        assert manifest["reward_version_ok"] is True
        assert manifest["r1_config_ok"] is True
        assert manifest["architecture_ok"] is True
        assert manifest["qmix_config_sha256_ok"] is True
        assert manifest["v3_sha_matches_ref"] is True

    def test_t9_synthetic_validation_pass(self, manifest):
        assert manifest["synthetic_validation"] == "PASS"

    def test_t10_c1_mixer_integrity(self, manifest):
        assert manifest["c1_mixer_integrity_ok"] is True


# ===== T11-T25: checkpoint and mixer integrity =====

class TestCheckpointIntegrity:
    def test_t11_checkpoint_integrity_pass(self, verdict):
        assert verdict["CHECKPOINT_INTEGRITY"] == "PASS"

    def test_t12_pre_hashes_match(self, ck_integrity):
        assert ck_integrity["all_match_frozen"] is True

    def test_t13_all_ten_checkpoints(self, ck_integrity):
        assert len(ck_integrity["checkpoint_pre_hashes"]) == 10

    def test_t14_c0_checkpoints_present(self, ck_integrity):
        c0_keys = [k for k in ck_integrity["checkpoint_pre_hashes"] if "C0_" in k]
        assert len(c0_keys) == 5

    def test_t15_c1_checkpoints_present(self, ck_integrity):
        c1_keys = [k for k in ck_integrity["checkpoint_pre_hashes"] if "C1_" in k]
        assert len(c1_keys) == 5

    def test_t16_c1_mixer_loaded(self, ck_integrity):
        for p, mi in ck_integrity["c1_mixer_integrity"].items():
            assert mi["has_mixer_weights"] is True, f"C1 policy {p} missing mixer weights"
            assert mi["mixer_loaded_successfully"] is True, f"C1 policy {p} mixer load failed"

    def test_t17_c1_mixer_sha_match(self, ck_integrity):
        for p, mi in ck_integrity["c1_mixer_integrity"].items():
            assert mi["matches_frozen"] is True, f"C1 policy {p} QMIX config SHA mismatch"

    def test_t18_architecture_integrity(self, verdict):
        assert verdict["ARCHITECTURE_INTEGRITY"] == "PASS"

    def test_t19_qmix_config_integrity(self, verdict):
        assert verdict["QMIX_CONFIG_SHA256_INTEGRITY"] == "PASS"

    def test_t20_reward_version_integrity(self, verdict):
        assert verdict["REWARD_VERSION_CONFIG_INTEGRITY"] == "PASS"

    def test_t21_normalization_integrity(self, verdict):
        assert verdict["NORMALIZATION_INTEGRITY"] == "PASS"

    def test_t22_post_integrity(self, post_integrity):
        assert post_integrity["checkpoint_post_hashes_match"] is True
        assert post_integrity["v3_sha_unchanged"] is True
        assert post_integrity["r1_config_unchanged"] is True
        assert post_integrity["architecture_unchanged"] is True

    def test_t23_qmix_sha_post_unchanged(self, post_integrity):
        assert post_integrity["qmix_sha_unchanged"] is True

    def test_t24_pre_matches_e4ke(self, ck_integrity):
        e4ke_ck = _load(E4KE_DIR / "E4KE_checkpoint_freeze.json")["checkpoints"]
        for key, info in e4ke_ck.items():
            expected_hash = info["sha256"]
            arm = info["arm"]
            p = info["policy_seed"]
            actual_key = f"{arm}_{p}"
            assert actual_key in ck_integrity["checkpoint_pre_hashes"]
            assert ck_integrity["checkpoint_pre_hashes"][actual_key] == expected_hash

    def test_t25_final_protocol_integrity(self, verdict):
        assert verdict["FINAL_PROTOCOL_INTEGRITY"] == "PASS"


# ===== T26-T40: execution contracts =====

class TestExecutionContracts:
    def test_t26_execution_mode(self, manifest):
        em = manifest["execution_mode"]
        assert em["epsilon"] == 0.0
        assert em["eval"] is True
        assert em["gradients"] == "disabled"
        assert em["mixer_not_in_execution"] is True
        assert em["c0_uses_utility_only"] is True
        assert em["c1_uses_utility_only_for_action_selection"] is True

    def test_t27_execution_config(self):
        cfg = _load(DIR / "E4KF_execution_config.json")
        assert cfg["final_seeds"] == [1981, 2040]
        assert cfg["policy_seeds"] == POLICY_SEEDS
        assert cfg["arms"] == ARMS
        assert cfg["scenarios"] == SCENARIOS
        assert cfg["reward_version_both_arms"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"
        assert cfg["alpha_local_both_arms"] == 0.1

    def test_t28_coordination_mechanism_gate_in_config(self):
        cfg = _load(DIR / "E4KF_execution_config.json")
        assert cfg["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"

    def test_t29_episode_csv_exists(self):
        assert (DIR / "E4KF_episode_results.csv").exists()

    def test_t30_episode_csv_row_count(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1800

    def test_t31_episode_csv_arms_balanced(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        c0 = sum(1 for r in rows if r["arm"] == ARMS[0])
        c1 = sum(1 for r in rows if r["arm"] == ARMS[1])
        assert c0 == 900
        assert c1 == 900

    def test_t32_episode_csv_all_seeds(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        seeds = {int(r["environment_seed"]) for r in rows}
        assert seeds == set(range(1981, 2041))

    def test_t33_episode_csv_all_policies(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        policies = {int(r["policy_seed"]) for r in rows}
        assert policies == {5001, 5002, 5003, 5004, 5005}

    def test_t34_episode_csv_all_scenarios(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        scenarios = {r["scenario"] for r in rows}
        assert scenarios == set(SCENARIOS)

    def test_t35_episode_csv_finite_values(self):
        import csv
        with open(DIR / "E4KF_episode_results.csv") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert math.isfinite(float(r["MIGD_response"]))
            assert math.isfinite(float(r["MIGD_epoch"]))
            assert math.isfinite(float(r["response_FE"]))

    def test_t36_effect_definition(self, verdict):
        assert verdict["EFFECT_DEFINITION"] == "C0_MINUS_C1"
        assert verdict["POSITIVE_MEANS"] == "QMIX_BETTER"

    def test_t37_primary_scenario(self, verdict):
        assert verdict["PRIMARY_SCENARIO"] == "MIXED"
        assert verdict["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"

    def test_t38_primary_comparison(self, verdict):
        assert verdict["PRIMARY_COMPARISON"] == "C1_QMIX_VS_C0_DDQN"

    def test_t39_practical_margin(self, verdict):
        assert verdict["PRACTICAL_MARGIN"] == 1e-5

    def test_t40_both_arms_r1_reward(self, manifest):
        assert manifest["reward_version_ok"] is True
        assert manifest["r1_config_ok"] is True


# ===== T41-T55: statistical design and primary effect =====

class TestStatisticalDesign:
    def test_t41_bootstrap_design(self, bootstrap):
        assert bootstrap["settings"]["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
        assert bootstrap["settings"]["N_BOOTSTRAP"] == 50000
        assert bootstrap["settings"]["master_seed"] == 86001
        assert bootstrap["settings"]["effect"] == "C0_MINUS_C1"

    def test_t42_delta_coord_in_bootstrap(self, bootstrap):
        assert bootstrap["settings"]["DELTA_COORD"] == 1e-5

    def test_t43_mixed_primary_exists(self, gates):
        assert "MIXED_PRIMARY" in gates
        assert gates["MIXED_PRIMARY"] in [
            "BENEFICIAL", "DIRECTIONAL_POSITIVE", "PRACTICALLY_NEUTRAL",
            "DIRECTIONAL_NEGATIVE", "HARMFUL"]

    def test_t44_mixed_delta_finite(self, verdict):
        assert math.isfinite(verdict["MIXED_DELTA_C0_MINUS_C1"])

    def test_t45_mixed_ci_finite(self, verdict):
        ci = verdict["MIXED_DELTA_CI"]
        assert len(ci) == 2
        assert math.isfinite(ci[0]) and math.isfinite(ci[1])
        assert ci[0] <= ci[1]

    def test_t46_c0_c1_means_finite(self, verdict):
        assert math.isfinite(verdict["C0_MIXED_MEAN_MIGD_RESPONSE"])
        assert math.isfinite(verdict["C1_MIXED_MEAN_MIGD_RESPONSE"])

    def test_t47_mixed_ri_finite(self, verdict):
        assert math.isfinite(verdict["MIXED_RELATIVE_IMPROVEMENT"])

    def test_t48_scenario_effects_all_present(self, bootstrap):
        for s in SCENARIOS:
            assert s in bootstrap["scenario_effects"]
            for ep in ("MIGD_response", "MIGD_epoch", "response_FE"):
                assert ep in bootstrap["scenario_effects"][s]

    def test_t49_macro_effects_present(self, bootstrap):
        for ep in ("MIGD_response", "MIGD_epoch", "response_FE"):
            assert ep in bootstrap["macro"]

    def test_t50_scenario_effects_csv_exists(self):
        assert (DIR / "E4KF_scenario_effects.csv").exists()

    def test_t51_macro_effects_csv_exists(self):
        assert (DIR / "E4KF_macro_effects.csv").exists()

    def test_t52_per_policy_deltas(self, verdict):
        deltas = verdict["per_policy_MIXED_deltas"]
        assert len(deltas) == 5
        assert all(math.isfinite(d) for d in deltas)

    def test_t53_delta_consistent_with_means(self, verdict):
        c0 = verdict["C0_MIXED_MEAN_MIGD_RESPONSE"]
        c1 = verdict["C1_MIXED_MEAN_MIGD_RESPONSE"]
        delta = verdict["MIXED_DELTA_C0_MINUS_C1"]
        assert abs((c0 - c1) - delta) < 1e-8

    def test_t54_classification_matches_effect(self, verdict):
        from prdrl.evaluation.final_verdict import classify_effect
        delta = verdict["MIXED_DELTA_C0_MINUS_C1"]
        ci = verdict["MIXED_DELTA_CI"]
        expected = classify_effect(delta, ci[0], ci[1], 1e-5)
        assert verdict["MIXED_PRIMARY"] == expected

    def test_t55_policy_pair_summary_csv(self):
        import csv
        with open(DIR / "E4KF_policy_pair_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        seeds = {int(r["policy_seed"]) for r in rows}
        assert seeds == {5001, 5002, 5003, 5004, 5005}


# ===== T56-T65: classification, robustness, safety =====

class TestClassificationAndSafety:
    def test_t56_robustness_reported(self, verdict):
        assert "COORDINATION_POLICY_ROBUSTNESS" in verdict
        assert verdict["COORDINATION_POLICY_ROBUSTNESS"] in ("PASS", "FAIL")

    def test_t57_robustness_count(self, verdict):
        count = verdict["POLICY_PAIR_POSITIVE_COUNT"]
        assert 0 <= count <= 5

    def test_t58_robustness_consistency(self, verdict):
        count = verdict["POLICY_PAIR_POSITIVE_COUNT"]
        rob = verdict["COORDINATION_POLICY_ROBUSTNESS"]
        assert (rob == "PASS") == (count >= 4)

    def test_t59_persistent_safety(self, verdict):
        assert verdict["PERSISTENT_SAFETY"] in ("PASS", "FAIL")

    def test_t60_stationary_safety(self, verdict):
        assert verdict["STATIONARY_SAFETY"] in ("PASS", "FAIL")

    def test_t61_macro_safety(self, verdict):
        assert verdict["MACRO_SAFETY"] in ("PASS", "FAIL")

    def test_t62_persistent_effects_present(self, verdict):
        assert "PERSISTENT_RESPONSE_EFFECT" in verdict
        assert "PERSISTENT_EPOCH_EFFECT" in verdict

    def test_t63_stationary_effects_present(self, verdict):
        assert "STATIONARY_RESPONSE_EFFECT" in verdict
        assert "STATIONARY_EPOCH_EFFECT" in verdict

    def test_t64_macro_effects_present(self, verdict):
        assert "MACRO_RESPONSE_EFFECT" in verdict
        assert "MACRO_EPOCH_EFFECT" in verdict

    def test_t65_fe_effects_present(self, verdict):
        assert "MIXED_FE_EFFECT_C0_MINUS_C1" in verdict
        assert "PERSISTENT_FE_EFFECT_C0_MINUS_C1" in verdict
        assert "STATIONARY_FE_EFFECT_C0_MINUS_C1" in verdict


# ===== T66-T80: verdict logic and coordination diagnostics =====

class TestVerdictAndDiagnostics:
    def test_t66_verdict_present(self, verdict):
        assert verdict["COORDINATION_CAUSAL_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")

    def test_t67_improvement_claim(self, verdict):
        assert verdict["COORDINATION_IMPROVEMENT_CLAIM"] in ("SUPPORTED", "PARTIAL", "NOT_SUPPORTED")

    def test_t68_verdict_reason(self, verdict):
        assert "verdict_reason" in verdict
        assert len(verdict["verdict_reason"]) > 0

    def test_t69_partial_subtype(self, verdict):
        v = verdict["COORDINATION_CAUSAL_VERDICT"]
        sub = verdict["PARTIAL_SUBTYPE"]
        if v == "PARTIAL":
            assert sub in ("PERFORMANCE_PARTIAL", "PRACTICALLY_NEUTRAL")
        elif v == "VALIDATED":
            assert sub == "NONE"
        elif v == "FAILED":
            assert sub == "NONE"

    def test_t70_coordination_mechanism_gate(self, verdict):
        assert verdict["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"

    def test_t71_coordination_diagnostics_not_in_verdict_logic(self, coord_diag):
        assert coord_diag["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"
        assert coord_diag["mixer_calls_during_action_selection"] == 0

    def test_t72_action_diagnostics_present(self, coord_diag):
        for arm in ARMS:
            key = f"{'C0' if 'C0' in arm else 'C1'}_MIXED_action"
            assert key in coord_diag

    def test_t73_action_distribution_csv(self):
        import csv
        with open(DIR / "E4KF_action_distribution.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        arms_found = {r["arm"] for r in rows}
        assert arms_found == set(ARMS)

    def test_t74_c0_predict_fraction(self, verdict):
        assert 0 <= verdict["C0_MIXED_PREDICT_FRACTION"] <= 1

    def test_t75_c1_predict_fraction(self, verdict):
        assert 0 <= verdict["C1_MIXED_PREDICT_FRACTION"] <= 1

    def test_t76_multi_action_rates(self, verdict):
        assert 0 <= verdict["C0_MIXED_MULTI_ACTION_RATE"] <= 1
        assert 0 <= verdict["C1_MIXED_MULTI_ACTION_RATE"] <= 1

    def test_t77_action_entropy(self, verdict):
        assert verdict["C0_MIXED_ACTION_ENTROPY"] >= 0
        assert verdict["C1_MIXED_ACTION_ENTROPY"] >= 0

    def test_t78_verdict_function_validation(self):
        from prdrl.rl.coordination_integration import coordination_causal_verdict
        v = coordination_causal_verdict(
            mixed_primary_label="BENEFICIAL", policy_robustness_pass=True,
            persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
            checkpoint_integrity_pass=True, protocol_integrity_pass=True,
            architecture_integrity_pass=True, no_final_tuning=True)
        assert v["verdict"] == "COORDINATION_CAUSAL_VALIDATED"
        v2 = coordination_causal_verdict(
            mixed_primary_label="HARMFUL", policy_robustness_pass=True,
            persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
            checkpoint_integrity_pass=True, protocol_integrity_pass=True,
            architecture_integrity_pass=True, no_final_tuning=True)
        assert v2["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t79_diagnostics_cannot_rescue(self, coord_diag):
        assert "coordination diagnostics recorded for post-hoc interpretation ONLY" in coord_diag["note"]

    def test_t80_fe_effects_csv(self):
        assert (DIR / "E4KF_fe_effects.csv").exists()


# ===== T81-T88: outputs, comparison, seed accounting =====

class TestOutputs:
    def test_t81_required_outputs_present(self):
        e4ke_schema = _load(E4KE_DIR / "E4KE_output_schema.json")
        for fname in e4ke_schema["e4kf_required_outputs"]:
            assert (DIR / fname).exists(), f"missing required output: {fname}"

    def test_t82_report_exists(self):
        assert (DIR / "E4KF_report.md").exists()
        text = (DIR / "E4KF_report.md").read_text()
        assert "COORDINATION_CAUSAL_VERDICT" in text
        assert "NOT_USED_FOR_SCIENTIFIC_VERDICT" in text

    def test_t83_validation_final_comparison(self, val_comparison):
        assert val_comparison["note"] == "descriptive only; does not modify verdict"
        assert "validation" in val_comparison
        assert "final" in val_comparison

    def test_t84_validation_comparison_has_both(self, val_comparison):
        assert val_comparison["final"]["C0_MIXED_MIGD_response"] is not None
        assert val_comparison["final"]["C1_MIXED_MIGD_response"] is not None

    def test_t85_final_seed_usage(self, seed_usage):
        assert seed_usage["final_seed_range"] == [1981, 2040]
        assert seed_usage["unique_final_seeds"] == 60
        assert seed_usage["FINAL_EVALUATION"] == "COMPLETE"

    def test_t86_env_runs_positive(self, seed_usage):
        assert seed_usage["env_runs"] > 0

    def test_t87_bootstrap_results_json(self, bootstrap):
        assert "scenario_effects" in bootstrap
        assert "macro" in bootstrap
        assert "settings" in bootstrap

    def test_t88_scientific_gates_json(self, gates):
        assert "MIXED_PRIMARY" in gates
        assert "COORDINATION_POLICY_ROBUSTNESS" in gates
        assert "PERSISTENT_SAFETY" in gates
        assert "STATIONARY_SAFETY" in gates
        assert "MACRO_SAFETY" in gates


# ===== T89-T97: post-integrity, no-learning, routing, no-rescue =====

class TestPostIntegrity:
    def test_t89_no_learning_audit(self, no_learning):
        assert no_learning["optimizer_steps"] == 0
        assert no_learning["gradient_updates"] == 0
        assert no_learning["replay_appends"] == 0
        assert no_learning["target_updates"] == 0
        assert no_learning["checkpoint_writes"] == 0
        assert no_learning["model_weight_changes"] == 0
        assert no_learning["normalizer_updates"] == 0
        assert no_learning["reward_config_updates"] == 0
        assert no_learning["mixer_calls_during_action_selection"] == 0

    def test_t90_no_retuning(self, verdict):
        assert verdict["ALPHA_RETUNING"] == "NO"
        assert verdict["REWARD_RETUNING"] == "NO"
        assert verdict["ARCHITECTURE_RETUNING"] == "NO"
        assert verdict["CHECKPOINT_RESELECTION"] == "NO"
        assert verdict["MODEL_MODIFICATION"] == "NO"

    def test_t91_final_evaluation_complete(self, verdict):
        assert verdict["FINAL_EVALUATION"] == "COMPLETE"
        assert verdict["FINAL_ENVIRONMENT_SEEDS_USED"] == "1981..2040"

    def test_t92_routing_validated(self, verdict):
        v = verdict["COORDINATION_CAUSAL_VERDICT"]
        nxt = verdict["NEXT_STAGE"]
        if v == "VALIDATED":
            assert nxt == "E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
        elif v == "PARTIAL":
            assert nxt == "E4L_COORDINATION_RESULT_REVIEW"
        elif v == "FAILED":
            assert nxt == "E4L_COORDINATION_FAILURE_ANALYSIS"

    def test_t93_next_stage_present(self, verdict):
        assert "NEXT_STAGE" in verdict
        assert verdict["NEXT_STAGE"].startswith("E4L_")

    def test_t94_historical_verdicts_unchanged(self, verdict):
        assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"
        assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_t95_progress_log_exists(self):
        p = DIR / "E4KF_progress.jsonl"
        assert p.exists()
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == 1800

    def test_t96_safety_gate_consistency(self, verdict):
        from prdrl.evaluation.final_verdict import safety_gate
        for s in ["PERSISTENT", "STATIONARY"]:
            rlab = verdict[f"{s}_RESPONSE_EFFECT"]
            elab = verdict[f"{s}_EPOCH_EFFECT"]
            sg = safety_gate(rlab, elab)
            expected = "PASS" if sg["PASS"] else "FAIL"
            assert verdict[f"{s}_SAFETY"] == expected

    def test_t97_regional_records_csv_exists(self):
        assert (DIR / "E4KF_regional_records.csv").exists()
