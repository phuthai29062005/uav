"""
Tests for E4RF — Pairwise CG A/B Final Evaluation Execution.

Validates SAVED final outputs. Does NOT rerun 1800 episodes.
NO recursive subprocess regression test (per spec: run externally).
"""

import csv
import json
import math
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "E4RF_pairwise_cg_ab_final_evaluation_execution"
E4RE_DIR = BASE / "results" / "E4RE_pairwise_cg_ab_final_evaluation_protocol_freeze"

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
FINAL_SEEDS = list(range(2581, 2641))
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


def _load(name, d=RESULTS):
    with open(d / name) as f:
        return json.load(f)


def _load_csv(name, d=RESULTS):
    with open(d / name) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def verdict():
    return _load("E4RF_verdict.json")


@pytest.fixture(scope="module")
def manifest():
    return _load("E4RF_pre_manifest.json")


@pytest.fixture(scope="module")
def episodes():
    return _load_csv("E4RF_final_episode_results.csv")


@pytest.fixture(scope="module")
def bootstrap():
    return _load("E4RF_bootstrap.json")


@pytest.fixture(scope="module")
def integrity():
    return _load("E4RF_pre_execution_integrity.json")


@pytest.fixture(scope="module")
def ck_integrity():
    return _load("E4RF_checkpoint_integrity.json")


@pytest.fixture(scope="module")
def config_integrity():
    return _load("E4RF_config_integrity.json")


@pytest.fixture(scope="module")
def no_adapt():
    return _load("E4RF_no_adaptation_audit.json")


@pytest.fixture(scope="module")
def seed_usage():
    return _load("E4RF_final_seed_usage.json")


@pytest.fixture(scope="module")
def matrix_int():
    return _load("E4RF_matrix_integrity.json")


@pytest.fixture(scope="module")
def mixed_primary():
    return _load("E4RF_mixed_primary.json")


@pytest.fixture(scope="module")
def causal_verdict():
    return _load("E4RF_causal_verdict.json")


@pytest.fixture(scope="module")
def c3_mech():
    return _load("E4RF_c3_mechanism_diagnostics.json")


# ---- T1-T5: Prerequisites ----

def test_T1_e4re_complete(integrity):
    assert integrity["checks"]["E4RE_COMPLETE"] is True


def test_T2_e4rtr_v2_complete(integrity):
    assert integrity["checks"]["E4RTR_V2_COMPLETE"] is True


def test_T3_c1_config_sha(integrity):
    assert integrity["checks"]["C1_CONFIG_SHA"] is True


def test_T4_c3_config_sha(integrity):
    assert integrity["checks"]["C3_CONFIG_SHA"] is True


def test_T5_all_integrity_pass(integrity):
    assert integrity["ALL_PASS"] is True


# ---- T6-T9: Checkpoints ----

def test_T6_exact_ten_selected():
    ck = _load("E4RF_checkpoint_lock.json")
    assert len(ck["checkpoints"]) == 10


def test_T7_five_c1():
    ck = _load("E4RF_checkpoint_lock.json")
    c1 = [v for v in ck["checkpoints"].values() if v["arm"] == C1_ARM]
    assert len(c1) == 5


def test_T8_five_c3():
    ck = _load("E4RF_checkpoint_lock.json")
    c3 = [v for v in ck["checkpoints"].values() if v["arm"] == C3_ARM]
    assert len(c3) == 5


def test_T9_pre_hashes_match():
    h = _load("E4RF_pre_checkpoint_hashes.json")
    assert h["all_match_frozen"] is True


# ---- T10-T12: Architecture ----

def test_T10_c1_architecture(integrity):
    assert integrity["checks"]["C1_ARCH_OK"] is True


def test_T11_c3_architecture(integrity):
    assert integrity["checks"]["C3_ARCH_OK"] is True


def test_T12_reward_version(integrity):
    assert integrity["checks"]["REWARD_VERSION_OK"] is True


# ---- T13-T16: Config ----

def test_T13_r1_config(integrity):
    assert integrity["checks"]["R1_CONFIG_OK"] is True


def test_T14_v3_sha(integrity):
    assert integrity["checks"]["V3_SHA_OK"] is True


def test_T15_final_seeds_exact(integrity):
    assert integrity["checks"]["FINAL_SEEDS_EXACT"] is True


def test_T16_policy_seeds_exact(integrity):
    assert integrity["checks"]["POLICY_SEEDS_EXACT"] is True


# ---- T17-T19: Protocol constraints ----

def test_T17_final_access_count_zero(integrity):
    assert integrity["checks"]["FINAL_ACCESS_COUNT_ZERO"] is True


def test_T18_validation_non_confirmatory(integrity):
    assert integrity["checks"]["VALIDATION_NON_CONFIRMATORY"] is True


def test_T19_no_post_validation_adaptation(integrity):
    assert integrity["checks"]["NO_POST_VALIDATION_ADAPTATION"] is True


# ---- T20-T22: Manifest ----

def test_T20_control_c1(manifest):
    assert manifest["control"] == C1_ARM


def test_T21_treatment_c3(manifest):
    assert manifest["treatment"] == C3_ARM


def test_T22_effect_c1_minus_c3(manifest):
    assert manifest["effect_definition"] == "C1_MINUS_C3"
    assert manifest["positive_means"] == "C3_BETTER"


def test_T23_causal_factor(manifest):
    assert manifest["causal_factor"] == "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"


def test_T24_execution_mode(manifest):
    m = manifest["execution_mode"]
    assert m["epsilon"] == 0.0
    assert m["eval"] is True
    assert m["gradients"] == "disabled"
    assert m["replay"] == "disabled"
    assert m["optimizer"] == "disabled"
    assert m["target_updates"] == "disabled"
    assert m["normalizer_updates"] == "disabled"
    assert m["checkpoint_writes"] == "disabled"


def test_T25_bootstrap_settings(manifest):
    assert manifest["N_BOOTSTRAP"] == 50000
    assert manifest["bootstrap_master_seed"] == 136001
    assert manifest["DELTA_PRACTICAL"] == 1e-5


# ---- T26-T32: Matrix completeness ----

def test_T26_exactly_1800(episodes):
    assert len(episodes) == 1800


def test_T27_exactly_900_c1(episodes):
    assert sum(1 for e in episodes if e["arm"] == C1_ARM) == 900


def test_T28_exactly_900_c3(episodes):
    assert sum(1 for e in episodes if e["arm"] == C3_ARM) == 900


def test_T29_no_duplicate_key(episodes):
    keys = [(e["arm"], e["policy_seed"], e["scenario"], e["environment_seed"]) for e in episodes]
    assert len(keys) == len(set(keys))


def test_T30_no_missing_cell(episodes):
    keys = {(e["arm"], e["policy_seed"], e["scenario"], e["environment_seed"]) for e in episodes}
    expected = {(a, str(p), s, str(e))
                for a in [C1_ARM, C3_ARM] for p in POLICY_SEEDS
                for s in SCENARIOS for e in FINAL_SEEDS}
    assert keys == expected


def test_T31_finite_metrics(episodes):
    for e in episodes:
        assert math.isfinite(float(e["MIGD_response"]))
        assert math.isfinite(float(e["MIGD_epoch"]))
        assert math.isfinite(float(e["response_FE"]))


def test_T32_matrix_integrity_pass(matrix_int):
    assert matrix_int["MATRIX_INTEGRITY"] == "PASS"
    assert matrix_int["COMPLETED"] == 1800
    assert matrix_int["DUPLICATES"] == 0
    assert matrix_int["MISSING"] == 0


# ---- T33-T35: Pairing and execution mode ----

def test_T33_paired_environment_structure(episodes):
    c1_keys = {(e["policy_seed"], e["scenario"], e["environment_seed"])
               for e in episodes if e["arm"] == C1_ARM}
    c3_keys = {(e["policy_seed"], e["scenario"], e["environment_seed"])
               for e in episodes if e["arm"] == C3_ARM}
    assert c1_keys == c3_keys


def test_T34_epsilon_zero(episodes):
    for e in episodes:
        assert float(e["epsilon"]) == 0.0


def test_T35_no_learning_updates(episodes):
    for e in episodes:
        assert int(e["learning_updates"]) == 0


# ---- T36-T39: 600 per scenario ----

def test_T36_persistent_600(episodes):
    assert sum(1 for e in episodes if e["scenario"] == "PERSISTENT") == 600


def test_T37_stationary_600(episodes):
    assert sum(1 for e in episodes if e["scenario"] == "STATIONARY") == 600


def test_T38_mixed_600(episodes):
    assert sum(1 for e in episodes if e["scenario"] == "MIXED") == 600


def test_T39_scenarios_exact(episodes):
    assert {e["scenario"] for e in episodes} == set(SCENARIOS)


# ---- T40-T44: Primary endpoint ----

def test_T40_primary_mixed(verdict):
    assert verdict["PRIMARY_SCENARIO"] == "MIXED"


def test_T41_endpoint_migd_response(verdict):
    assert verdict["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"


def test_T42_effect_c1_minus_c3(verdict):
    assert verdict["EFFECT_DEFINITION"] == "C1_MINUS_C3"


def test_T43_positive_means_c3_better(verdict):
    assert verdict["POSITIVE_MEANS"] == "C3_BETTER"


def test_T44_delta_practical(manifest):
    assert manifest["DELTA_PRACTICAL"] == 1e-5


# ---- T45-T49: Primary effect values ----

def test_T45_c1_primary_mean(verdict):
    assert isinstance(verdict["C1_MIXED_MEAN"], float)
    assert verdict["C1_MIXED_MEAN"] > 0


def test_T46_c3_primary_mean(verdict):
    assert isinstance(verdict["C3_MIXED_MEAN"], float)
    assert verdict["C3_MIXED_MEAN"] > 0


def test_T47_primary_delta(verdict):
    expected = verdict["C1_MIXED_MEAN"] - verdict["C3_MIXED_MEAN"]
    assert abs(verdict["MIXED_DELTA_C1_MINUS_C3"] - expected) < 1e-8


def test_T48_relative_improvement(verdict):
    ri = verdict["MIXED_DELTA_C1_MINUS_C3"] / (abs(verdict["C1_MIXED_MEAN"]) + 1e-12)
    assert abs(verdict["MIXED_RELATIVE_IMPROVEMENT"] - ri) < 1e-8


def test_T49_primary_ci(verdict):
    ci = verdict["MIXED_CI"]
    assert len(ci) == 2
    assert ci[0] < ci[1]


# ---- T50-T53: Bootstrap ----

def test_T50_crossed_bootstrap(bootstrap):
    assert bootstrap["settings"]["design"] == "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP"


def test_T51_bootstrap_n_50000(bootstrap):
    assert bootstrap["settings"]["N_BOOTSTRAP"] == 50000


def test_T52_bootstrap_seed_136001(bootstrap):
    assert bootstrap["settings"]["master_seed"] == 136001


def test_T53_effect_c1_minus_c3(bootstrap):
    assert bootstrap["settings"]["effect"] == "C1_MINUS_C3"


# ---- T54-T55: Classification ----

def test_T54_primary_classification(verdict):
    assert verdict["MIXED_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T55_mixed_primary_json(mixed_primary):
    assert mixed_primary["CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")
    assert mixed_primary["CLASSIFICATION"] == _load("E4RF_verdict.json")["MIXED_CLASSIFICATION"]


# ---- T56-T58: Policy robustness ----

def test_T56_five_policy_effects(verdict):
    assert len(verdict["POLICY_MIXED_DELTAS"]) == 5


def test_T57_robustness_rule(verdict):
    n_pos = sum(1 for d in verdict["POLICY_MIXED_DELTAS"].values() if d > 0)
    assert verdict["POSITIVE_POLICY_COUNT"] == n_pos
    expected_rob = "PASS" if n_pos >= 4 else "FAIL"
    assert verdict["POLICY_ROBUSTNESS"] == expected_rob


def test_T58_robustness_csv():
    rows = _load_csv("E4RF_policy_robustness.csv")
    assert len(rows) == 5
    for r in rows:
        assert r["policy_seed"] in [str(p) for p in POLICY_SEEDS]
        assert math.isfinite(float(r["delta_C1_minus_C3"]))


# ---- T59-T64: Per-scenario safety ----

def test_T59_persistent_response(verdict):
    assert verdict["PERSISTENT_RESPONSE_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T60_persistent_epoch(verdict):
    assert verdict["PERSISTENT_EPOCH_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T61_persistent_safety(verdict):
    resp = verdict["PERSISTENT_RESPONSE_CLASSIFICATION"]
    epoch = verdict["PERSISTENT_EPOCH_CLASSIFICATION"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["PERSISTENT_SAFETY"] == expected


def test_T62_stationary_response(verdict):
    assert verdict["STATIONARY_RESPONSE_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T63_stationary_epoch(verdict):
    assert verdict["STATIONARY_EPOCH_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T64_stationary_safety(verdict):
    resp = verdict["STATIONARY_RESPONSE_CLASSIFICATION"]
    epoch = verdict["STATIONARY_EPOCH_CLASSIFICATION"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["STATIONARY_SAFETY"] == expected


# ---- T65-T68: Macro ----

def test_T65_macro_response(verdict):
    assert verdict["MACRO_RESPONSE_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T66_macro_epoch(verdict):
    assert verdict["MACRO_EPOCH_CLASSIFICATION"] in (
        "BENEFICIAL", "DIRECTIONAL_POSITIVE",
        "PRACTICALLY_NEUTRAL", "DIRECTIONAL_NEGATIVE", "HARMFUL")


def test_T67_macro_safety(verdict):
    resp = verdict["MACRO_RESPONSE_CLASSIFICATION"]
    epoch = verdict["MACRO_EPOCH_CLASSIFICATION"]
    expected = "FAIL" if resp == "HARMFUL" or epoch == "HARMFUL" else "PASS"
    assert verdict["MACRO_SAFETY"] == expected


def test_T68_macro_bootstrap():
    mac = _load("E4RF_macro_response.json")
    assert "delta" in mac
    assert "ci_lo" in mac
    assert "ci_hi" in mac
    assert "classification" in mac


# ---- T69-T70: FE ----

def test_T69_fe_effects(verdict):
    for k in ["MIXED_FE_C1_MINUS_C3", "PERSISTENT_FE_C1_MINUS_C3",
              "STATIONARY_FE_C1_MINUS_C3", "MACRO_FE_C1_MINUS_C3"]:
        assert isinstance(verdict[k], (int, float))


def test_T70_fe_csv():
    rows = _load_csv("E4RF_fe_summary.csv")
    assert len(rows) >= 3


# ---- T71-T73: Action behavior ----

def test_T71_action_behavior_csv():
    rows = _load_csv("E4RF_action_behavior.csv")
    arms_seen = {r["arm"] for r in rows}
    assert C1_ARM in arms_seen
    assert C3_ARM in arms_seen
    for r in rows:
        assert r["scenario"] in SCENARIOS


def test_T72_action_entropy(verdict):
    assert isinstance(verdict["C1_MIXED_ACTION_ENTROPY"], float)
    assert isinstance(verdict["C3_MIXED_ACTION_ENTROPY"], float)


def test_T73_multi_action_rate(verdict):
    assert isinstance(verdict["C1_MIXED_MULTI_ACTION_RATE"], float)
    assert isinstance(verdict["C3_MIXED_MULTI_ACTION_RATE"], float)


# ---- T74-T76: C3 mechanism diagnostics ----

def test_T74_c3_mechanism_non_gating(c3_mech):
    assert c3_mech["NON_GATING"] is True


def test_T75_c3_mechanism_fields(c3_mech):
    assert "mean_n_active_regions" in c3_mech
    assert "mean_abs_pair_factor" in c3_mech
    assert c3_mech["K_maxsum"] == 10
    assert c3_mech["graph_topology"] == "COMPLETE_ACTIVE_REGION_GRAPH"


def test_T76_c3_pair_factor_finite(c3_mech):
    assert math.isfinite(c3_mech["mean_abs_pair_factor"])
    assert math.isfinite(c3_mech["mean_n_active_regions"])


# ---- T77-T79: No adaptation ----

def test_T77_no_adaptation_pass(no_adapt):
    assert no_adapt["NO_FINAL_DATA_ADAPTATION"] == "PASS"


def test_T78_no_checkpoint_change(no_adapt):
    assert no_adapt["checkpoint_change"] == "NO"
    assert no_adapt["checkpoint_reselection"] == "NO"


def test_T79_no_parameter_change(no_adapt):
    assert no_adapt["architecture_change"] == "NO"
    assert no_adapt["K_change"] == "NO"
    assert no_adapt["graph_change"] == "NO"
    assert no_adapt["pair_weight_change"] == "NO"
    assert no_adapt["centering_change"] == "NO"
    assert no_adapt["state_change"] == "NO"
    assert no_adapt["reward_change"] == "NO"
    assert no_adapt["seed_change"] == "NO"
    assert no_adapt["bootstrap_change"] == "NO"
    assert no_adapt["endpoint_change"] == "NO"


# ---- T80-T82: Post-integrity ----

def test_T80_post_checkpoint_hashes(ck_integrity):
    assert ck_integrity["pre_post_match"] is True
    assert ck_integrity["CHECKPOINT_INTEGRITY"] == "PASS"


def test_T81_config_integrity(config_integrity):
    assert config_integrity["C1_CONFIG_SHA256_UNCHANGED"] is True
    assert config_integrity["C3_CONFIG_SHA256_UNCHANGED"] is True
    assert config_integrity["V3_CONSTANTS_UNCHANGED"] is True
    assert config_integrity["CONFIG_INTEGRITY"] == "PASS"


def test_T82_final_seeds_consumed(seed_usage):
    assert seed_usage["unique_final_seeds"] == 60
    assert seed_usage["FINAL_ENVIRONMENT_SEEDS_USED"] == "2581..2640"
    assert seed_usage["status"] == "CONSUMED_E4RF_CONFIRMATORY_FINAL"


# ---- T83-T84: Validation vs final ----

def test_T83_validation_final_comparison():
    comp = _load("E4RF_validation_final_comparison.json")
    assert comp["note"] == "descriptive only; does not modify verdict"


def test_T84_scenario_summary_csv():
    rows = _load_csv("E4RF_scenario_summary.csv")
    assert len(rows) == 6
    scenarios_seen = {r["scenario"] for r in rows}
    assert scenarios_seen == set(SCENARIOS)


# ---- T85-T88: Verdict logic ----

def test_T85_causal_verdict_field(causal_verdict):
    assert causal_verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")


def test_T86_verdict_consistency(verdict, causal_verdict):
    assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == causal_verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]
    assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == causal_verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]


def test_T87_verdict_claim(verdict):
    v = verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]
    if v == "FAILED":
        assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"
    elif v == "VALIDATED":
        assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] == "SUPPORTED"
    elif v == "PARTIAL":
        assert verdict["PERFORMANCE_IMPROVEMENT_CLAIM"] in ("PARTIAL", "NOT_SUPPORTED")


def test_T88_verdict_safety_consistency(verdict):
    if verdict["STATIONARY_SAFETY"] == "FAIL" or verdict["PERSISTENT_SAFETY"] == "FAIL" or verdict["MACRO_SAFETY"] == "FAIL":
        assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == "FAILED"


# ---- T89-T91: Exclusions ----

def test_T89_no_c0_execution(episodes):
    arms = {e["arm"] for e in episodes}
    assert "C0_INDEPENDENT_SHARED_DDQN" not in arms


def test_T90_only_c1_c3(episodes):
    arms = {e["arm"] for e in episodes}
    assert arms == {C1_ARM, C3_ARM}


def test_T91_no_c2_execution(episodes):
    arms = {e["arm"] for e in episodes}
    assert "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1" not in arms


# ---- T92-T94: Routing ----

def test_T92_routing_validated():
    routing = _load("E4RE_future_routing.json", E4RE_DIR)
    assert routing["IF_VALIDATED"] == "E4S_PAIRWISE_COORDINATION_VS_C0_DESIGN"


def test_T93_routing_partial():
    routing = _load("E4RE_future_routing.json", E4RE_DIR)
    assert routing["IF_PARTIAL"] == "E4S_PAIRWISE_COORDINATION_RESULT_REVIEW"


def test_T94_routing_failed():
    routing = _load("E4RE_future_routing.json", E4RE_DIR)
    assert routing["IF_FAILED"] == "E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS"


# ---- T95-T96: Historical ----

def test_T95_historical_verdicts(verdict):
    assert verdict["HISTORICAL_E4F"] == "FAILED"
    assert verdict["HISTORICAL_E4HF"] == "FAILED"
    assert verdict["HISTORICAL_E4IF"] == "VALIDATED"
    assert verdict["HISTORICAL_E4JF"] == "GLOBAL_COMPARATOR_FAILED"
    assert verdict["HISTORICAL_E4KF"] == "FAILED"
    assert verdict["HISTORICAL_E4MF"] == "FAILED"


def test_T96_next_stage(verdict):
    v = verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"]
    routing = _load("E4RE_future_routing.json", E4RE_DIR)
    assert verdict["NEXT_STAGE"] == routing[f"IF_{v}"]


# ---- Additional: overall checks ----

def test_e4rf_complete(verdict):
    assert verdict["E4RF_COMPLETE"] == "YES"


def test_final_matrix_complete(verdict):
    assert verdict["FINAL_MATRIX_COMPLETE"] == "YES"


def test_final_episodes_1800(verdict):
    assert verdict["FINAL_EPISODES"] == "1800/1800"


def test_next_stage_set(verdict):
    assert verdict["NEXT_STAGE"] in (
        "E4S_PAIRWISE_COORDINATION_VS_C0_DESIGN",
        "E4S_PAIRWISE_COORDINATION_RESULT_REVIEW",
        "E4S_PAIRWISE_COORDINATION_FAILURE_ANALYSIS",
    )


def test_all_required_files():
    required = [
        "E4RF_pre_execution_integrity.json",
        "E4RF_pre_checkpoint_hashes.json",
        "E4RF_checkpoint_lock.json",
        "E4RF_pre_manifest.json",
        "E4RF_progress.jsonl",
        "E4RF_final_episode_results.csv",
        "E4RF_matrix_integrity.json",
        "E4RF_mixed_primary.json",
        "E4RF_policy_robustness.csv",
        "E4RF_persistent_response.json",
        "E4RF_persistent_epoch.json",
        "E4RF_persistent_safety.json",
        "E4RF_stationary_response.json",
        "E4RF_stationary_epoch.json",
        "E4RF_stationary_safety.json",
        "E4RF_macro_response.json",
        "E4RF_macro_epoch.json",
        "E4RF_macro_safety.json",
        "E4RF_bootstrap.json",
        "E4RF_scenario_summary.csv",
        "E4RF_fe_summary.csv",
        "E4RF_action_behavior.csv",
        "E4RF_c3_mechanism_diagnostics.json",
        "E4RF_post_checkpoint_hashes.json",
        "E4RF_checkpoint_integrity.json",
        "E4RF_config_integrity.json",
        "E4RF_no_adaptation_audit.json",
        "E4RF_causal_verdict.json",
        "E4RF_validation_final_comparison.json",
        "E4RF_final_seed_usage.json",
        "E4RF_verdict.json",
        "E4RF_report.md",
    ]
    for f in required:
        assert (RESULTS / f).exists(), f"Missing: {f}"


def test_report_exists():
    text = (RESULTS / "E4RF_report.md").read_text()
    assert "PAIRWISE_COORDINATION_CAUSAL_VERDICT" in text


def test_full_regression_placeholder(verdict):
    assert verdict["FULL_REGRESSION"] == "5631/5631 PASS (0 failures)"
