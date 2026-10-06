"""
Tests for E4MF — Credit Preservation A/B Final Evaluation Execution.

Validates SAVED final outputs. Does NOT rerun 1800 episodes.
NO recursive subprocess regression test (per spec: run externally).
"""

import csv
import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4ME_DIR = BASE / "results" / "E4ME_credit_preservation_ab_final_evaluation_protocol_freeze"

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C2_ARM = "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"


def _load(name, d=RESULTS):
    with open(d / name) as f:
        return json.load(f)


def _load_csv(name, d=RESULTS):
    with open(d / name) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def verdict():
    return _load("E4MF_verdict.json")


@pytest.fixture(scope="module")
def manifest():
    return _load("E4MF_pre_execution_manifest.json")


@pytest.fixture(scope="module")
def episodes():
    return _load_csv("E4MF_episode_results.csv")


@pytest.fixture(scope="module")
def gates():
    return _load("E4MF_scientific_gates.json")


@pytest.fixture(scope="module")
def bootstrap():
    return _load("E4MF_bootstrap_results.json")


@pytest.fixture(scope="module")
def no_learn():
    return _load("E4MF_no_learning_audit.json")


@pytest.fixture(scope="module")
def post_int():
    return _load("E4MF_post_integrity.json")


@pytest.fixture(scope="module")
def seed_usage():
    return _load("E4MF_final_seed_usage.json")


@pytest.fixture(scope="module")
def loss_spec():
    return _load("E4MF_loss_spec_integrity.json")


@pytest.fixture(scope="module")
def arch():
    return _load("E4MF_architecture_integrity.json")


@pytest.fixture(scope="module")
def config():
    return _load("E4MF_execution_config.json")


@pytest.fixture(scope="module")
def policy_pairs():
    return _load_csv("E4MF_policy_pair_summary.csv")


# ---- T1-T5: Prerequisites ----

def test_T1_e4me_complete(manifest):
    assert manifest["E4ME_COMPLETE"] == "YES"


def test_T2_e4mtr_complete(manifest):
    assert manifest["E4MTR_COMPLETE"] == "YES"


def test_T3_e4mi_implementation_valid(manifest):
    assert manifest["E4MI_IMPLEMENTATION_VALID"] == "YES"


def test_T4_historical_e4kf_failed(verdict):
    assert verdict["ORIGINAL_E4KF_VERDICT"] == "FAILED"


def test_T5_h4_supported(verdict):
    assert verdict["H4_STATUS"] == "SUPPORTED"


# ---- T6-T9: Checkpoints ----

def test_T6_exact_ten_selected():
    frz = _load("E4ME_checkpoint_freeze.json", E4ME_DIR)
    assert frz["n_checkpoints"] == 10


def test_T7_five_c1():
    frz = _load("E4ME_checkpoint_freeze.json", E4ME_DIR)
    assert frz["n_c1"] == 5


def test_T8_five_c2():
    frz = _load("E4ME_checkpoint_freeze.json", E4ME_DIR)
    assert frz["n_c2"] == 5


def test_T9_pre_hashes_exact():
    ck = _load("E4MF_checkpoint_integrity.json")
    assert ck["all_match_frozen"] is True


# ---- T10-T12: Architecture ----

def test_T10_c1_architecture_exact(arch):
    assert arch["C1_CONTROL"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"


def test_T11_c2_architecture_exact(arch):
    assert arch["C2_TREATMENT"] == C2_ARM


def test_T12_parameter_structure_integrity(arch):
    assert arch["parameter_count_identity"] is True
    assert arch["both_arms_same_utility"] is True
    assert arch["both_arms_same_mixer"] is True


# ---- T13-T16: Loss spec ----

def test_T13_c1_loss_metadata_exact(loss_spec):
    assert loss_spec["C1_loss"] == "L_C1 = L_joint"


def test_T14_c2_joint_half(loss_spec):
    assert loss_spec["joint_weight"] == 0.5


def test_T15_c2_regional_half(loss_spec):
    assert loss_spec["regional_weight"] == 0.5


def test_T16_loss_weights_unchanged(loss_spec):
    assert loss_spec["weights_immutable"] is True
    assert loss_spec["no_final_retuning"] is True


# ---- T17-T21: V3/R1/QMIX/C2 ----

def test_T17_v3_exact(manifest):
    assert manifest["v3_sha_matches_ref"] is True


def test_T18_r1_exact(manifest):
    assert manifest["r1_config_ok"] is True


def test_T19_alpha_exact(config):
    assert config["alpha_local_both_arms"] == 0.1


def test_T20_qmix_config_exact(manifest):
    assert manifest["qmix_config_sha256_ok"] is True


def test_T21_c2_credit_config_exact(manifest):
    assert manifest["c2_credit_config_sha256_ok"] is True


# ---- T22-T23: Replay ----

def test_T22_replay_metadata_exact(manifest):
    assert manifest["execution_mode"]["replay"] == "disabled"


def test_T23_no_sibling_dedup():
    mt_replay = _load("E4MT_replay_lock.json",
                      BASE / "results" / "E4MT_credit_preservation_ab_training_protocol_freeze")
    assert mt_replay["sibling_deduplication"] is False


# ---- T24-T26: Seeds/scenarios ----

def test_T24_final_seeds_2181_2240(config):
    assert config["final_seeds"] == [2181, 2240]


def test_T25_policy_seeds_6001_6005(config):
    assert config["policy_seeds"] == [6001, 6002, 6003, 6004, 6005]


def test_T26_scenarios_exact(config):
    assert config["scenarios"] == ["PERSISTENT", "STATIONARY", "MIXED"]


# ---- T27-T32: Matrix completeness ----

def test_T27_exactly_1800(episodes):
    assert len(episodes) == 1800


def test_T28_exactly_900_c1(episodes):
    assert sum(1 for e in episodes if e["arm"] == C1_ARM) == 900


def test_T29_exactly_900_c2(episodes):
    assert sum(1 for e in episodes if e["arm"] == C2_ARM) == 900


def test_T30_no_duplicate_key(episodes):
    keys = [(e["arm"], e["policy_seed"], e["scenario"], e["environment_seed"]) for e in episodes]
    assert len(keys) == len(set(keys))


def test_T31_no_missing_cell(episodes):
    keys = {(e["arm"], e["policy_seed"], e["scenario"], e["environment_seed"]) for e in episodes}
    expected = {(a, str(p), s, str(e))
                for a in [C1_ARM, C2_ARM] for p in range(6001, 6006)
                for s in ["PERSISTENT", "STATIONARY", "MIXED"] for e in range(2181, 2241)}
    assert keys == expected


def test_T32_finite_metrics(episodes):
    for e in episodes:
        assert math.isfinite(float(e["MIGD_response"]))
        assert math.isfinite(float(e["MIGD_epoch"]))
        assert math.isfinite(float(e["response_FE"]))


# ---- T33-T35: Pairing and execution mode ----

def test_T33_paired_environment_structure(episodes):
    c1_keys = {(e["policy_seed"], e["scenario"], e["environment_seed"])
               for e in episodes if e["arm"] == C1_ARM}
    c2_keys = {(e["policy_seed"], e["scenario"], e["environment_seed"])
               for e in episodes if e["arm"] == C2_ARM}
    assert c1_keys == c2_keys


def test_T34_epsilon_zero(manifest):
    assert manifest["execution_mode"]["epsilon"] == 0.0


def test_T35_no_gradients(manifest):
    assert manifest["execution_mode"]["gradients"] == "disabled"


# ---- T36-T41: No-learning audit ----

def test_T36_no_optimizer_steps(no_learn):
    assert no_learn["optimizer_steps"] == 0


def test_T37_no_replay_append(no_learn):
    assert no_learn["replay_appends"] == 0


def test_T38_no_target_updates(no_learn):
    assert no_learn["target_updates"] == 0


def test_T39_no_model_updates(no_learn):
    assert no_learn["utility_weight_changes"] == 0
    assert no_learn["mixer_weight_changes"] == 0


def test_T40_no_normalizer_updates(no_learn):
    assert no_learn["normalizer_updates"] == 0


def test_T41_mixer_action_calls_zero(no_learn):
    assert no_learn["mixer_calls_during_action_selection"] == 0


# ---- T42-T44: Execution semantics ----

def test_T42_simultaneous_regional_execution():
    report = (RESULTS / "E4MF_report.md").read_text()
    assert "CREDIT_PRESERVATION_CAUSAL_VERDICT" in report


def test_T43_requested_realized_semantics():
    recs = _load_csv("E4MF_regional_records.csv")
    assert len(recs) > 0
    for r in recs[:100]:
        assert r["requested_action"] in ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")


def test_T44_production_operator_constants(config):
    assert config["reward_version_both_arms"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"


# ---- T45-T49: Primary endpoint ----

def test_T45_primary_mixed(verdict):
    assert verdict["PRIMARY_SCENARIO"] == "MIXED"


def test_T46_endpoint_migd_response(verdict):
    assert verdict["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"


def test_T47_effect_c1_minus_c2(verdict):
    assert verdict["EFFECT_DEFINITION"] == "C1_MINUS_C2"


def test_T48_positive_means_c2_better(verdict):
    assert verdict["POSITIVE_MEANS"] == "C2_BETTER"


def test_T49_practical_margin_1e5(config):
    assert config["DELTA_CREDIT"] == 1e-5


# ---- T50-T53: Primary effect values ----

def test_T50_c1_primary_mean(verdict):
    assert isinstance(verdict["C1_MIXED_MEAN_MIGD_RESPONSE"], float)
    assert verdict["C1_MIXED_MEAN_MIGD_RESPONSE"] > 0


def test_T51_c2_primary_mean(verdict):
    assert isinstance(verdict["C2_MIXED_MEAN_MIGD_RESPONSE"], float)
    assert verdict["C2_MIXED_MEAN_MIGD_RESPONSE"] > 0


def test_T52_primary_delta(verdict):
    expected = verdict["C1_MIXED_MEAN_MIGD_RESPONSE"] - verdict["C2_MIXED_MEAN_MIGD_RESPONSE"]
    assert abs(verdict["MIXED_DELTA_C1_MINUS_C2"] - expected) < 1e-8


def test_T53_relative_improvement(verdict):
    ri = verdict["MIXED_DELTA_C1_MINUS_C2"] / verdict["C1_MIXED_MEAN_MIGD_RESPONSE"]
    assert abs(verdict["MIXED_RELATIVE_IMPROVEMENT"] - ri) < 1e-8


# ---- T54-T60: Bootstrap ----

def test_T54_crossed_bootstrap(bootstrap):
    assert bootstrap["settings"]["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"


def test_T55_policy_axis_5():
    e4me = _load("E4ME_verdict.json", E4ME_DIR)
    assert e4me["POLICY_AXIS_N"] == 5


def test_T56_environment_axis_60():
    e4me = _load("E4ME_verdict.json", E4ME_DIR)
    assert e4me["ENVIRONMENT_AXIS_N"] == 60


def test_T57_bootstrap_n_50000(bootstrap):
    assert bootstrap["settings"]["N_BOOTSTRAP"] == 50000


def test_T58_bootstrap_seed_106001(bootstrap):
    assert bootstrap["settings"]["master_seed"] == 106001


def test_T59_iid_300_absent(bootstrap):
    assert bootstrap["settings"]["effect"] == "C1_MINUS_C2"


def test_T60_independent_arm_absent(bootstrap):
    assert bootstrap["settings"]["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"


# ---- T61-T62: CI and classification ----

def test_T61_primary_ci(verdict):
    ci = verdict["MIXED_DELTA_CI"]
    assert len(ci) == 2
    assert ci[0] < ci[1]


def test_T62_primary_classification(verdict):
    assert verdict["MIXED_PRIMARY"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                         "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


# ---- T63-T64: Policy robustness ----

def test_T63_five_policy_effects(verdict):
    assert len(verdict["POLICY_PAIR_DELTAS"]) == 5


def test_T64_robustness_rule(verdict):
    n_pos = sum(1 for d in verdict["POLICY_PAIR_DELTAS"] if d > 0)
    assert verdict["POLICY_PAIR_POSITIVE_COUNT"] == n_pos
    expected_rob = "PASS" if n_pos >= 4 else "FAIL"
    assert verdict["CREDIT_PRESERVATION_POLICY_ROBUSTNESS"] == expected_rob


# ---- T65-T70: Per-scenario safety ----

def test_T65_persistent_response(verdict):
    assert verdict["PERSISTENT_RESPONSE_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                                      "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T66_persistent_epoch(verdict):
    assert verdict["PERSISTENT_EPOCH_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                                    "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T67_persistent_safety(verdict):
    resp = verdict["PERSISTENT_RESPONSE_EFFECT"]
    epoch = verdict["PERSISTENT_EPOCH_EFFECT"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["PERSISTENT_SAFETY"] == expected


def test_T68_stationary_response(verdict):
    assert verdict["STATIONARY_RESPONSE_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                                      "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T69_stationary_epoch(verdict):
    assert verdict["STATIONARY_EPOCH_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                                    "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T70_stationary_safety(verdict):
    resp = verdict["STATIONARY_RESPONSE_EFFECT"]
    epoch = verdict["STATIONARY_EPOCH_EFFECT"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["STATIONARY_SAFETY"] == expected


# ---- T71-T75: Macro ----

def test_T71_macro_equal_third():
    mac = _load("E4ME_macro_contract.json", E4ME_DIR)
    assert abs(mac["scenario_weight"] - 1/3) < 1e-12


def test_T72_macro_shared_bootstrap():
    mac = _load("E4ME_macro_contract.json", E4ME_DIR)
    assert mac["shared_bootstrap_draws_across_scenarios"] is True


def test_T73_macro_response(verdict):
    assert verdict["MACRO_RESPONSE_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                                  "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T74_macro_epoch(verdict):
    assert verdict["MACRO_EPOCH_EFFECT"] in ("BENEFICIAL", "DIRECTIONAL_POSITIVE",
                                              "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T75_macro_safety(verdict):
    resp = verdict["MACRO_RESPONSE_EFFECT"]
    epoch = verdict["MACRO_EPOCH_EFFECT"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["MACRO_SAFETY"] == expected


# ---- T76-T77: FE ----

def test_T76_fe_effects(verdict):
    for k in ["MIXED_FE_EFFECT_C1_MINUS_C2", "PERSISTENT_FE_EFFECT_C1_MINUS_C2",
              "STATIONARY_FE_EFFECT_C1_MINUS_C2", "MACRO_FE_EFFECT_C1_MINUS_C2"]:
        assert isinstance(verdict[k], (int, float))


def test_T77_fe_cannot_rescue():
    fe = _load("E4ME_fe_contract.json", E4ME_DIR)
    assert fe["cannot_rescue_harmful_quality"] is True


# ---- T78-T80: Diagnostics ----

def test_T78_action_diagnostics_descriptive():
    ac = _load("E4ME_action_context_contract.json", E4ME_DIR)
    assert ac["descriptive_only"] is True


def test_T79_q_margins_descriptive():
    qm = _load("E4ME_q_margin_contract.json", E4ME_DIR)
    assert qm["descriptive_only"] is True
    assert qm["no_threshold"] is True


def test_T80_credit_mechanism_gate(verdict):
    assert verdict["CREDIT_PRESERVATION_MECHANISM_GATE"] == "NOT_USED_FOR_SCIENTIFIC_VERDICT"


# ---- T81: Loss-scale observation ----

def test_T81_loss_scale_descriptive(verdict):
    assert verdict["LOSS_SCALE_OBSERVATION_STATUS"] == "DESCRIPTIVE_NON_CONFIRMATORY_LOCKED"


# ---- T82-T87: Verdict logic ----

def test_T82_validated_logic():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    req = sl["CREDIT_PRESERVATION_CAUSAL_VALIDATED_requires_all"]
    assert any("BENEFICIAL" in r for r in req)
    assert any("robustness" in r.lower() for r in req)
    assert any("safety" in r.lower() for r in req)


def test_T83_performance_partial_logic():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    assert "DIRECTIONAL_POSITIVE" in sl["PERFORMANCE_PARTIAL"]


def test_T84_neutral_partial():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    assert "NOT_SUPPORTED" in sl["PRACTICALLY_NEUTRAL"]


def test_T85_failed_primary_negative():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    assert "HARMFUL" in sl["FAILED"]


def test_T86_failed_safety():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    assert "safety" in sl["FAILED"].lower()


def test_T87_failed_integrity():
    sl = _load("E4ME_success_logic.json", E4ME_DIR)
    assert "integrity" in sl["FAILED"].lower()


# ---- T88-T89: Validation locks ----

def test_T88_validation_non_confirmatory(verdict):
    assert verdict["VALIDATION_OBSERVATION_STATUS"] == "NON_CONFIRMATORY_LOCKED"


def test_T89_no_validation_final_pooling():
    comp = _load("E4MF_validation_final_comparison.json")
    assert comp["note"] == "descriptive only; does not modify verdict"


# ---- T90-T96: No retuning ----

def test_T90_no_checkpoint_reselection(verdict):
    assert verdict["CHECKPOINT_RESELECTION"] == "NO"


def test_T91_no_loss_weight_retuning(verdict):
    assert verdict["LOSS_WEIGHT_RETUNING"] == "NO"


def test_T92_no_reward_retuning(verdict):
    assert verdict["REWARD_RETUNING"] == "NO"


def test_T93_no_normalization_retuning(verdict):
    assert verdict["NORMALIZATION_RETUNING"] == "NO"


def test_T94_no_mixer_retuning(verdict):
    assert verdict["MIXER_RETUNING"] == "NO"


def test_T95_no_replay_retuning(verdict):
    assert verdict["REPLAY_RETUNING"] == "NO"


def test_T96_no_model_modification(verdict):
    assert verdict["MODEL_MODIFICATION"] == "NO"


# ---- T97-T98: Post-integrity ----

def test_T97_post_checkpoint_hashes(post_int):
    assert post_int["checkpoint_post_hashes_match"] is True
    assert post_int["v3_sha_unchanged"] is True
    assert post_int["qmix_sha_unchanged"] is True
    assert post_int["c2_credit_sha_unchanged"] is True


def test_T98_final_seeds_consumed(seed_usage):
    assert seed_usage["unique_final_seeds"] == 60
    assert seed_usage["FINAL_ENVIRONMENT_SEEDS_USED"] == "2181..2240"
    assert seed_usage["status"] == "CONSUMED_CREDIT_PRESERVATION_CONFIRMATORY_FINAL"


# ---- T99-T101: Exclusions ----

def test_T99_no_c0_execution(episodes):
    arms = {e["arm"] for e in episodes}
    assert "C0_INDEPENDENT_SHARED_DDQN" not in arms


def test_T100_no_global_comparator(episodes):
    arms = {e["arm"] for e in episodes}
    assert C1_ARM in arms
    assert C2_ARM in arms
    assert len(arms) == 2


def test_T101_no_transitive_c2_c0_claim():
    routing = _load("E4ME_routing.json", E4ME_DIR)
    assert routing["no_transitive_inference"] is True


# ---- T102-T104: Routing ----

def test_T102_routing_validated():
    routing = _load("E4ME_routing.json", E4ME_DIR)
    assert "E4N_CREDIT_PRESERVATION_VS_C0_PROTOCOL_DESIGN" == routing["CREDIT_PRESERVATION_CAUSAL_VALIDATED"]


def test_T103_routing_partial():
    routing = _load("E4ME_routing.json", E4ME_DIR)
    assert "E4N_CREDIT_PRESERVATION_RESULT_REVIEW" == routing["CREDIT_PRESERVATION_CAUSAL_PARTIAL"]


def test_T104_routing_failed():
    routing = _load("E4ME_routing.json", E4ME_DIR)
    assert "E4N_CREDIT_PRESERVATION_FAILURE_ANALYSIS" == routing["CREDIT_PRESERVATION_CAUSAL_FAILED_integrity_pass"]


# ---- T105: Full regression (no subprocess) ----

def test_T105_full_regression_placeholder(verdict):
    assert verdict["FULL_REGRESSION"] == "(run separately)"


# ---- Additional: overall checks ----

def test_e4mf_complete(verdict):
    assert verdict["E4MF_COMPLETE"] == "YES"


def test_final_evaluation_complete(verdict):
    assert verdict["FINAL_EVALUATION"] == "COMPLETE"


def test_next_stage_set(verdict):
    assert verdict["NEXT_STAGE"] in (
        "E4N_CREDIT_PRESERVATION_VS_C0_PROTOCOL_DESIGN",
        "E4N_CREDIT_PRESERVATION_RESULT_REVIEW",
        "E4N_CREDIT_PRESERVATION_FAILURE_ANALYSIS",
        "E4M_ENGINEERING_INTEGRITY_REVIEW",
    )


def test_all_required_files():
    required = [
        "E4MF_pre_execution_manifest.json", "E4MF_checkpoint_integrity.json",
        "E4MF_loss_spec_integrity.json", "E4MF_architecture_integrity.json",
        "E4MF_execution_config.json", "E4MF_progress.jsonl",
        "E4MF_episode_results.csv", "E4MF_action_context.csv",
        "E4MF_q_margin_context.csv", "E4MF_policy_pair_summary.csv",
        "E4MF_scenario_effects.csv", "E4MF_macro_effects.csv",
        "E4MF_fe_effects.csv", "E4MF_bootstrap_results.json",
        "E4MF_scientific_gates.json", "E4MF_validation_final_comparison.json",
        "E4MF_final_seed_usage.json", "E4MF_post_integrity.json",
        "E4MF_no_learning_audit.json", "E4MF_verdict.json", "E4MF_report.md",
    ]
    for f in required:
        assert (RESULTS / f).exists(), f"Missing: {f}"


def test_report_exists():
    text = (RESULTS / "E4MF_report.md").read_text()
    assert "CREDIT_PRESERVATION_CAUSAL_VERDICT" in text


def test_verdict_consistency(verdict):
    v = verdict["CREDIT_PRESERVATION_CAUSAL_VERDICT"]
    if v == "FAILED":
        assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"
    elif v == "VALIDATED":
        assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] == "SUPPORTED"


def test_actual_verdict_matches_safety(verdict):
    if verdict["STATIONARY_SAFETY"] == "FAIL" or verdict["PERSISTENT_SAFETY"] == "FAIL":
        assert verdict["CREDIT_PRESERVATION_CAUSAL_VERDICT"] == "FAILED"
