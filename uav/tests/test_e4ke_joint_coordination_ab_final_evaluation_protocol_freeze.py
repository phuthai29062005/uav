"""
Tests for E4KE — Joint Coordination A/B Final Evaluation Protocol Freeze.

Validates saved E4KE artifacts without re-running any experiment. Covers:
  T1-T10   prerequisites and historical integrity
  T11-T25  checkpoint integrity (hash, metadata, arms)
  T26-T40  contracts (architecture, reward, normalization, QMIX config)
  T41-T55  statistical design, bootstrap, effect, classification, margin
  T56-T65  safety, robustness, FE, macro
  T66-T80  verdict logic via coordination_causal_verdict
  T81-T88  synthetic validation
  T89-T93  firewall, routing, no-retuning, final access
"""

import json
from pathlib import Path

import numpy as np
import pytest

DIR = Path("results/E4KE_joint_coordination_ab_final_evaluation_protocol_freeze")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")
E4KT_DIR = Path("results/E4KT_joint_coordination_ab_training_protocol_freeze")


def _load(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def verdict():
    return _load(DIR / "E4KE_verdict.json")


@pytest.fixture(scope="module")
def manifest():
    return _load(DIR / "E4KE_pre_freeze_manifest.json")


@pytest.fixture(scope="module")
def synth():
    return _load(DIR / "E4KE_synthetic_validation.json")


@pytest.fixture(scope="module")
def ckpt_freeze():
    return _load(DIR / "E4KE_checkpoint_freeze.json")


@pytest.fixture(scope="module")
def protocol():
    return _load(DIR / "E4KE_protocol_frozen.json")


@pytest.fixture(scope="module")
def firewall():
    return _load(DIR / "E4KE_final_firewall.json")


@pytest.fixture(scope="module")
def routing():
    return _load(DIR / "E4KE_routing.json")


@pytest.fixture(scope="module")
def success_logic():
    return _load(DIR / "E4KE_success_logic.json")


@pytest.fixture(scope="module")
def val_lock():
    return _load(DIR / "E4KE_validation_observation_lock.json")


# ===== T1-T10: prerequisites and historical integrity =====

class TestPrerequisites:
    def test_t1_e4ke_complete(self, verdict):
        assert verdict["E4KE_COMPLETE"] == "YES"

    def test_t2_e4kt_complete(self, verdict):
        assert verdict["E4KT_COMPLETE"] == "YES"

    def test_t3_e4ktr_complete(self, verdict):
        assert verdict["E4KTR_COMPLETE"] == "YES"

    def test_t4_e4f_verdict_preserved(self, verdict):
        assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"

    def test_t5_e4hf_verdict_preserved(self, verdict):
        assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"

    def test_t6_e4if_verdict_preserved(self, verdict):
        assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"

    def test_t7_e4jf_verdict_preserved(self, verdict):
        assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_t8_selected_branch(self, verdict):
        assert verdict["SELECTED_BRANCH"] == "C_JOINT_COORDINATION"

    def test_t9_historical_integrity_artifact(self):
        h = _load(DIR / "E4KE_historical_integrity.json")
        assert h["E4F"] == "FAILED"
        assert h["E4HF_normalization"] == "FAILED"
        assert h["E4IF_reward"] == "VALIDATED"
        assert h["E4JF_global_comparator"] == "FAILED"
        assert h["QMIX_IMPLEMENTATION_VALID"] == "YES"
        assert h["no_post_e4ktr_training"] is True

    def test_t10_manifest_integrity(self, manifest):
        assert manifest["freeze_only"] is True
        assert manifest["checkpoint_integrity"] == "PASS"
        assert manifest["qmix_config_sha256_match"] is True


# ===== T11-T25: checkpoint integrity =====

class TestCheckpointIntegrity:
    def test_t11_checkpoint_integrity_pass(self, verdict):
        assert verdict["CHECKPOINT_INTEGRITY"] == "PASS"

    def test_t12_c0_count(self, verdict):
        assert verdict["C0_SELECTED_CHECKPOINTS"] == "5/5"

    def test_t13_c1_count(self, verdict):
        assert verdict["C1_SELECTED_CHECKPOINTS"] == "5/5"

    def test_t14_total_checkpoints(self, ckpt_freeze):
        assert ckpt_freeze["n_checkpoints"] == 10
        assert ckpt_freeze["n_c0"] == 5
        assert ckpt_freeze["n_c1"] == 5

    def test_t15_all_ten_advance(self, ckpt_freeze):
        assert ckpt_freeze["all_ten_advance"] is True
        assert ckpt_freeze["no_reselection"] is True
        assert ckpt_freeze["no_weight_averaging"] is True
        assert ckpt_freeze["no_ensemble"] is True
        assert ckpt_freeze["no_additional_training"] is True

    def test_t16_all_hashes_match(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert info["hash_match"] is True, f"hash mismatch: {key}"

    def test_t17_all_normalization_ok(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert info["normalization_metadata_ok"] is True, f"norm mismatch: {key}"

    def test_t18_all_reward_ok(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert info["reward_metadata_ok"] is True, f"reward mismatch: {key}"

    def test_t19_all_architecture_ok(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert info["architecture_metadata_ok"] is True, f"arch mismatch: {key}"

    def test_t20_c0_architecture_versions(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            if info["arm"] == "C0_INDEPENDENT_SHARED_DDQN":
                assert info["architecture_version"] == "INDEPENDENT_SHARED_DDQN"

    def test_t21_c1_architecture_versions(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            if info["arm"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1":
                assert info["architecture_version"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"

    def test_t22_policy_seeds_covered(self, ckpt_freeze):
        c0_seeds = {info["policy_seed"] for key, info in ckpt_freeze["checkpoints"].items()
                    if info["arm"] == "C0_INDEPENDENT_SHARED_DDQN"}
        c1_seeds = {info["policy_seed"] for key, info in ckpt_freeze["checkpoints"].items()
                    if info["arm"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"}
        assert c0_seeds == {5001, 5002, 5003, 5004, 5005}
        assert c1_seeds == {5001, 5002, 5003, 5004, 5005}

    def test_t23_checkpoint_files_exist(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert Path(info["checkpoint_path"]).exists(), f"missing: {info['checkpoint_path']}"

    def test_t24_e4ktr_checkpoints_match(self, ckpt_freeze):
        e4ktr_sel = _load(E4KTR_DIR / "E4KTR_selected_checkpoints.json")["selected"]
        for key in e4ktr_sel:
            assert key in ckpt_freeze["checkpoints"]
            assert ckpt_freeze["checkpoints"][key]["sha256_recomputed"] == e4ktr_sel[key]["sha256"]

    def test_t25_all_reward_r1(self, ckpt_freeze):
        for key, info in ckpt_freeze["checkpoints"].items():
            assert info["reward_version"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"


# ===== T26-T40: contract artifacts =====

class TestContracts:
    def test_t26_architecture_integrity(self):
        a = _load(DIR / "E4KE_architecture_integrity.json")
        assert a["C0_CONTROL"] == "INDEPENDENT_SHARED_DDQN"
        assert a["C1_TREATMENT"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
        assert a["QMIX_CONFIG_SHA256"] == "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
        assert a["qmix_sha_verified"] is True
        assert a["c0_has_mixer"] is False
        assert a["c1_has_mixer"] is True
        assert a["both_arms_same_utility"] is True

    def test_t27_qmix_config_sha_integrity(self, verdict):
        assert verdict["QMIX_CONFIG_SHA256_INTEGRITY"] == "PASS"

    def test_t28_reward_version_integrity(self):
        r = _load(DIR / "E4KE_reward_version_integrity.json")
        assert r["both_arms_reward"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"
        assert r["R1_config_sha256"] == "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
        assert r["alpha_local_both_arms"] == 0.1

    def test_t29_normalization_integrity(self):
        n = _load(DIR / "E4KE_normalization_integrity.json")
        assert n["normalization"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"
        assert n["identical_both_arms"] is True
        assert n["no_refit"] is True

    def test_t30_final_seed_contract(self):
        s = _load(DIR / "E4KE_final_seed_contract.json")
        assert s["range"] == [1981, 2040]
        assert s["n"] == 60
        assert s["never_instantiated"] is True
        assert s["COORDINATION_FINAL_EVAL_ACCESS_COUNT"] == 0

    def test_t31_final_matrix_contract(self):
        m = _load(DIR / "E4KE_final_matrix_contract.json")
        assert m["total_episodes"] == 1800
        assert m["C0_rows"] == 900
        assert m["C1_rows"] == 900
        assert m["FINAL_MATRIX_COMPLETE_required"] is True

    def test_t32_pairing_contract(self):
        p = _load(DIR / "E4KE_pairing_contract.json")
        assert p["pair_by"] == ["policy_seed", "environment_seed", "scenario"]

    def test_t33_endpoint_contract(self):
        e = _load(DIR / "E4KE_endpoint_contract.json")
        assert e["primary_endpoint"] == "MIGD_response"
        assert e["primary_scenario"] == "MIXED"

    def test_t34_effect_definition(self):
        e = _load(DIR / "E4KE_effect_definition.json")
        assert e["positive_means"] == "QMIX_BETTER"
        assert "C0" in e["cell_effect"] and "C1" in e["cell_effect"]

    def test_t35_practical_margin(self):
        p = _load(DIR / "E4KE_practical_margin.json")
        assert p["DELTA_COORD"] == 1e-5
        assert p["matches_DELTA_E4"] is True

    def test_t36_execution_order(self):
        e = _load(DIR / "E4KE_execution_order.json")
        assert e["scenario_order"] == ["PERSISTENT", "STATIONARY", "MIXED"]
        assert e["no_early_stop"] is True

    def test_t37_final_execution_contract(self):
        f = _load(DIR / "E4KE_final_execution_contract.json")
        assert f["epsilon"] == 0.0
        assert f["network_mode"] == "eval"
        assert f["gradients"] == "disabled"
        assert f["mixer_not_in_execution"] is True
        assert f["model_updates"] == "forbidden"

    def test_t38_output_schema(self):
        o = _load(DIR / "E4KE_output_schema.json")
        assert "E4KF_verdict.json" in o["e4kf_required_outputs"]
        assert "E4KF_episode_results.csv" in o["e4kf_required_outputs"]

    def test_t39_coordination_mechanism_contract(self):
        c = _load(DIR / "E4KE_coordination_mechanism_contract.json")
        assert c["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"
        assert c["coordination_diagnostics_cannot_rescue_harmful"] is True
        assert c["FE_savings_cannot_rescue_harmful"] is True
        assert c["action_diversity_cannot_rescue_harmful"] is True

    def test_t40_success_logic_structure(self, success_logic):
        assert success_logic["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"
        assert "BENEFICIAL" in str(success_logic["COORDINATION_CAUSAL_VALIDATED_requires_all"])
        assert success_logic["coordination_diagnostics_cannot_rescue_harmful"] is True
        assert success_logic["FE_savings_cannot_rescue_harmful"] is True


# ===== T41-T55: statistical design =====

class TestStatisticalDesign:
    def test_t41_bootstrap_design(self):
        b = _load(DIR / "E4KE_paired_bootstrap_contract.json")
        assert b["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"
        assert b["policy_axis_n"] == 5
        assert b["environment_axis_n"] == 60
        assert b["N_BOOTSTRAP"] == 50000
        assert b["CI_LEVEL"] == 0.95
        assert b["CI_METHOD"] == "percentile"
        assert b["BOOTSTRAP_MASTER_SEED"] == 86001

    def test_t42_bootstrap_preserves_pairing(self):
        b = _load(DIR / "E4KE_paired_bootstrap_contract.json")
        assert b["preserve_C0_C1_pairing"] is True

    def test_t43_forbidden_methods(self):
        b = _load(DIR / "E4KE_paired_bootstrap_contract.json")
        assert "independent_arm_bootstrap" in b["forbidden"]

    def test_t44_macro_shared_draws(self):
        m = _load(DIR / "E4KE_macro_contract.json")
        assert m["shared_bootstrap_draws_across_scenarios"] is True
        assert abs(m["scenario_weight"] - 1/3) < 1e-12

    def test_t45_classification_logic(self):
        c = _load(DIR / "E4KE_classification_logic.json")
        assert c["DELTA_COORD"] == 1e-5
        assert "BENEFICIAL" in c["labels"]
        assert "HARMFUL" in c["labels"]

    def test_t46_classification_function_consistency(self):
        from prdrl.evaluation.final_verdict import classify_effect
        assert classify_effect(0.001, 0.0005, 0.0015, 1e-5) == "BENEFICIAL"
        assert classify_effect(0.001, -0.0001, 0.0015, 1e-5) == "DIRECTIONAL_POSITIVE"
        assert classify_effect(0.000005, -0.0001, 0.0001, 1e-5) == "PRACTICALLY_NEUTRAL"
        assert classify_effect(-0.001, -0.0015, 0.0001, 1e-5) == "DIRECTIONAL_NEGATIVE"
        assert classify_effect(-0.001, -0.0015, -0.0005, 1e-5) == "HARMFUL"

    def test_t47_effect_direction_in_verdict(self, verdict):
        assert verdict["EFFECT_DEFINITION"] == "C0_MINUS_C1"
        assert verdict["POSITIVE_MEANS"] == "QMIX_BETTER"

    def test_t48_protocol_frozen_artifact(self, protocol):
        assert protocol["all_frozen_before_final"] is True
        assert protocol["effect_definition"] == "C0_MINUS_C1"
        assert protocol["positive_means"] == "QMIX_BETTER"
        assert protocol["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"

    def test_t49_policy_robustness_contract(self):
        p = _load(DIR / "E4KE_policy_robustness.json")
        assert p["threshold"] == 4
        assert ">=4" in p["rule"]

    def test_t50_verdict_primary_scenario(self, verdict):
        assert verdict["PRIMARY_SCENARIO"] == "MIXED"
        assert verdict["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"

    def test_t51_bootstrap_master_seed(self, verdict):
        assert verdict["BOOTSTRAP_MASTER_SEED"] == 86001

    def test_t52_policy_axis(self, verdict):
        assert verdict["POLICY_AXIS_N"] == 5
        assert verdict["ENVIRONMENT_AXIS_N"] == 60

    def test_t53_bootstrap_resamples(self, verdict):
        assert verdict["BOOTSTRAP_RESAMPLES"] == 50000

    def test_t54_final_matrix_planned(self, verdict):
        assert verdict["FINAL_MATRIX_PLANNED"] == "1800_EPISODES"

    def test_t55_no_placeholder_tokens(self, verdict):
        assert verdict["no_placeholder_tokens"] is True


# ===== T56-T65: safety, robustness, FE =====

class TestSafetyAndRobustness:
    def test_t56_safety_contract(self):
        s = _load(DIR / "E4KE_safety_contract.json")
        assert s["scenarios"] == ["PERSISTENT", "STATIONARY"]
        assert s["improvement_not_required"] is True
        assert s["fe_cannot_rescue_harmful_migd"] is True
        assert s["diagnostics_cannot_rescue_harmful_migd"] is True

    def test_t57_fe_contract(self):
        f = _load(DIR / "E4KE_fe_contract.json")
        assert f["secondary"] is True
        assert f["cannot_rescue_harmful_quality"] is True

    def test_t58_safety_gate_function(self):
        from prdrl.evaluation.final_verdict import safety_gate
        assert safety_gate("BENEFICIAL", "DIRECTIONAL_POSITIVE")["PASS"] is True
        assert safety_gate("HARMFUL", "BENEFICIAL")["PASS"] is False
        assert safety_gate("BENEFICIAL", "HARMFUL")["PASS"] is False
        assert safety_gate("HARMFUL", "HARMFUL")["PASS"] is False

    def test_t59_policy_robustness_function(self):
        from prdrl.evaluation.final_verdict import policy_robustness
        assert policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)["PASS"] is True
        assert policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)["PASS"] is False
        assert policy_robustness([0.1, 0.1, 0.1, 0.1, 0.1], threshold=4)["PASS"] is True

    def test_t60_macro_safety_in_contract(self):
        s = _load(DIR / "E4KE_safety_contract.json")
        assert "macro_safety" in s

    def test_t61_policy_robustness_rule_in_verdict(self, verdict):
        assert verdict["POLICY_ROBUSTNESS_RULE"] == "AT_LEAST_4_OF_5_QMIX_BETTER"

    def test_t62_coordination_mechanism_not_used(self, verdict):
        assert verdict["COORDINATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"

    def test_t63_reward_integrity_in_verdict(self, verdict):
        assert verdict["REWARD_VERSION_CONFIG_INTEGRITY"] == "PASS"

    def test_t64_normalization_integrity_in_verdict(self, verdict):
        assert verdict["NORMALIZATION_INTEGRITY"] == "PASS"

    def test_t65_architecture_integrity_in_verdict(self, verdict):
        assert verdict["ARCHITECTURE_INTEGRITY"] == "PASS"


# ===== T66-T80: verdict logic =====

class TestVerdictLogic:
    @pytest.fixture(scope="class")
    def cv(self):
        from prdrl.rl.coordination_integration import coordination_causal_verdict
        return coordination_causal_verdict

    def _call(self, cv, mp, rob, ps=True, ss=True, ms=True, eng=True):
        return cv(mixed_primary_label=mp, policy_robustness_pass=rob,
                  persistent_safety_pass=ps, stationary_safety_pass=ss, macro_safety_pass=ms,
                  checkpoint_integrity_pass=eng, protocol_integrity_pass=eng,
                  architecture_integrity_pass=eng, no_final_tuning=eng)

    def test_t66_validated(self, cv):
        v = self._call(cv, "BENEFICIAL", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_VALIDATED"
        assert v["COORDINATION_IMPROVEMENT_CLAIM"] == "SUPPORTED"

    def test_t67_perf_partial_beneficial_norobust(self, cv):
        v = self._call(cv, "BENEFICIAL", False)
        assert v["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
        assert v["subtype"] == "PERFORMANCE_PARTIAL"

    def test_t68_perf_partial_dirpos_robust(self, cv):
        v = self._call(cv, "DIRECTIONAL_POSITIVE", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
        assert v["subtype"] == "PERFORMANCE_PARTIAL"

    def test_t69_perf_partial_dirpos_norobust(self, cv):
        v = self._call(cv, "DIRECTIONAL_POSITIVE", False)
        assert v["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
        assert v["subtype"] == "PERFORMANCE_PARTIAL"

    def test_t70_neutral_partial(self, cv):
        v = self._call(cv, "PRACTICALLY_NEUTRAL", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
        assert v["subtype"] == "PRACTICALLY_NEUTRAL"
        assert v["COORDINATION_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"

    def test_t71_neutral_without_robust(self, cv):
        v = self._call(cv, "PRACTICALLY_NEUTRAL", False)
        assert v["verdict"] == "COORDINATION_CAUSAL_PARTIAL"
        assert v["subtype"] == "PRACTICALLY_NEUTRAL"

    def test_t72_failed_dirneg(self, cv):
        v = self._call(cv, "DIRECTIONAL_NEGATIVE", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t73_failed_harmful(self, cv):
        v = self._call(cv, "HARMFUL", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t74_failed_persistent_safety(self, cv):
        v = self._call(cv, "BENEFICIAL", True, ps=False)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t75_failed_stationary_safety(self, cv):
        v = self._call(cv, "BENEFICIAL", True, ss=False)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t76_failed_macro_safety(self, cv):
        v = self._call(cv, "BENEFICIAL", True, ms=False)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t77_failed_engineering(self, cv):
        v = self._call(cv, "BENEFICIAL", True, eng=False)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t78_failed_no_final_tuning(self, cv):
        v = cv(mixed_primary_label="BENEFICIAL", policy_robustness_pass=True,
               persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
               checkpoint_integrity_pass=True, protocol_integrity_pass=True,
               architecture_integrity_pass=True, no_final_tuning=False)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t79_diagnostics_cannot_rescue_harmful(self, cv):
        v = self._call(cv, "HARMFUL", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"

    def test_t80_fe_cannot_rescue_harmful(self, cv):
        v = self._call(cv, "HARMFUL", True)
        assert v["verdict"] == "COORDINATION_CAUSAL_FAILED"


# ===== T81-T88: synthetic validation =====

class TestSyntheticValidation:
    def test_t81_all_pass(self, synth):
        assert synth["ALL_PASS"] is True

    def test_t82_n_checks(self, synth):
        assert synth["n_checks"] == 28

    def test_t83_no_final_seed_instantiated(self, synth):
        assert synth["no_final_seed_instantiated"] is True

    def test_t84_each_check_passes(self, synth):
        for name, check in synth["checks"].items():
            assert check["PASS"] is True, f"synthetic check failed: {name}"

    def test_t85_bootstrap_checks(self, synth):
        assert synth["checks"]["4_policy_axis_widens_ci"]["PASS"] is True
        assert synth["checks"]["5_env_axis_widens_ci"]["PASS"] is True
        assert synth["checks"]["7_paired_narrower_than_independent"]["PASS"] is True

    def test_t86_verdict_branch_checks(self, synth):
        assert synth["checks"]["11_validated"]["PASS"] is True
        assert synth["checks"]["17_failed_dirneg"]["PASS"] is True
        assert synth["checks"]["18_failed_harmful"]["PASS"] is True

    def test_t87_no_mechanism_gate_check(self, synth):
        assert synth["checks"]["26_no_mechanism_gate_in_validated"]["PASS"] is True

    def test_t88_qmix_config_sha_check(self, synth):
        assert synth["checks"]["28_qmix_config_sha_deterministic"]["PASS"] is True


# ===== T89-T93: firewall, routing, no-retuning =====

class TestFirewallAndRouting:
    def test_t89_final_access_zero(self, verdict, firewall):
        assert verdict["COORDINATION_FINAL_EVAL_ACCESS_COUNT"] == 0
        assert firewall["COORDINATION_FINAL_EVAL_ACCESS_COUNT"] == 0

    def test_t90_final_evaluation_not_run(self, verdict):
        assert verdict["FINAL_EVALUATION"] == "NOT_RUN"
        assert verdict["COORDINATION_CAUSAL_CLAIM"] == "NOT_MADE"
        assert verdict["NEW_TRAINING"] == "NO"

    def test_t91_routing(self, routing):
        assert routing["COORDINATION_CAUSAL_VALIDATED"] == "E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
        assert routing["COORDINATION_CAUSAL_PARTIAL"] == "E4L_COORDINATION_RESULT_REVIEW"
        assert routing["COORDINATION_CAUSAL_FAILED"] == "E4L_COORDINATION_FAILURE_ANALYSIS"
        assert routing["E4JF_historical_failure_cannot_force_E4KF_failure"] is True
        assert routing["E4KTR_validation_cannot_force_E4KF_failure"] is True

    def test_t92_validation_observation_locked(self, val_lock):
        assert val_lock["status"] == "NON_CONFIRMATORY_LOCKED"
        assert val_lock["validation_macro_delta_C0_minus_C1"] is not None
        assert "primary_endpoint" in val_lock["must_not_alter"]
        assert "architecture" in val_lock["must_not_alter"]

    def test_t93_next_stage(self, verdict):
        assert verdict["NEXT_STAGE"] == "E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION"
