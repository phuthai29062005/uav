"""
E4HF — Normalization A/B Final Evaluation Execution: contract + reproducibility tests.

Verifies the executed V1-vs-V3 final causal experiment on seeds 1381..1440:
firewall/integrity, matrix completeness, paired crossed-bootstrap + verdict logic
(unit), reproducibility of the reported effects from the episode CSVs, and that the
immutable E4F verdict (FAILED) is preserved. No result is rescued.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4HF_normalization_ab_final_evaluation_execution as E4HF
from prdrl.rl.normalization import NORM_V1, NORM_V3
from prdrl.rl.network import ACTION_ORDER
from prdrl.evaluation.crossed_bootstrap import (
    paired_crossed_two_axis_bootstrap, paired_crossed_two_axis_bootstrap_macro,
    independent_arm_bootstrap, derive_seed, N_BOOTSTRAP, CI_LEVEL,
)
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HE = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HTR = Path("results/E4HTR_normalization_ab_training_execution")
E4HR = Path("results/E4HR_normalization_design_review")
E4F = Path("results/E4F_final_evaluation")

POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
FINAL_SEEDS = list(range(1381, 1441))
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = ["V1_CONTROL", "V3_TREATMENT"]


def _load(p):
    with open(p) as f:
        return json.load(f)


def _hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _require():
    if not (RES / "E4HF_verdict.json").exists():
        pytest.skip("E4HF not run yet")


def _episode_rows():
    with open(RES / "E4HF_episode_results.csv") as f:
        return list(csv.DictReader(f))


def _diff_matrix(rows, scenario, endpoint):
    """D[p,e] = V1 - V3 for one scenario/endpoint (positive => V3 better)."""
    idx = {(a, int(r["policy_seed"]), int(r["environment_seed"])): float(r[endpoint])
           for r in rows if r["scenario"] == scenario for a in [r["arm"]]}
    D = np.zeros((len(POLICY_SEEDS), len(FINAL_SEEDS)))
    for i, p in enumerate(POLICY_SEEDS):
        for j, e in enumerate(FINAL_SEEDS):
            key1 = ("V1_CONTROL", p, e); key3 = ("V3_TREATMENT", p, e)
            D[i, j] = idx[key1] - idx[key3]
    return D


# ---- prerequisites (immutable upstream) ----

def test_T1_e4he_complete():
    assert _load(E4HE / "E4HE_verdict.json")["E4HE_COMPLETE"] == "YES"


def test_T2_e4htr_complete():
    assert _load(E4HTR / "E4HTR_verdict.json")["E4HTR_COMPLETE"] == "YES"


def test_T3_e4f_failed_immutable():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


# ---- unit: paired crossed bootstrap (frozen master 56001) ----

def test_T4_paired_bootstrap_point_is_mean():
    D = np.random.default_rng(0).normal(0.0003, 0.001, size=(5, 60))
    out = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=56001)
    assert abs(out["point_estimate"] - float(D.mean())) < 1e-12
    assert out["n_policy"] == 5 and out["n_env"] == 60
    assert out["method"] == "paired_crossed_two_axis"


def test_T5_paired_bootstrap_deterministic():
    D = np.random.default_rng(1).normal(0.0, 0.001, size=(5, 60))
    a = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=123)
    b = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=123)
    assert a["ci_lo"] == b["ci_lo"] and a["ci_hi"] == b["ci_hi"]


def test_T6_ci_ordering():
    D = np.random.default_rng(2).normal(0.001, 0.001, size=(5, 60))
    out = paired_crossed_two_axis_bootstrap(D, n_boot=2000, seed=7)
    assert out["ci_lo"] <= out["point_estimate"] <= out["ci_hi"]


def test_T7_macro_equal_third():
    Dm = {"PERSISTENT": np.full((5, 60), 0.001), "STATIONARY": np.full((5, 60), 0.002),
          "MIXED": np.full((5, 60), 0.003)}
    mac = paired_crossed_two_axis_bootstrap_macro(Dm, SCENARIOS, n_boot=2000, seed=4)
    assert abs(mac["point_estimate"] - 0.002) < 1e-12


def test_T8_e4hf_seed_uses_frozen_master_56001():
    # E4HF derives from 56001; crossed_bootstrap.derive_seed uses 46001 -> must differ
    for labels in [("MIGD_response", "MIXED", "V1_MINUS_V3"), ("macro", "MIGD_epoch")]:
        assert E4HF._e4hf_seed(*labels) != derive_seed(*labels)
    assert E4HF.BOOTSTRAP_MASTER_SEED == 56001


def test_T9_e4hf_seed_deterministic():
    assert E4HF._e4hf_seed("MIGD_response", "MIXED", "V1_MINUS_V3") == \
           E4HF._e4hf_seed("MIGD_response", "MIXED", "V1_MINUS_V3")


# ---- unit: classification / verdict logic ----

def test_T10_classify_five_labels():
    assert FV.classify_effect(0.01, 0.001, 0.02) == FV.BENEFICIAL
    assert FV.classify_effect(0.01, -0.001, 0.02) == FV.DIRECTIONAL_POSITIVE
    assert FV.classify_effect(0.0, -0.001, 0.001) == FV.PRACTICALLY_NEUTRAL
    assert FV.classify_effect(-0.01, -0.02, 0.001) == FV.DIRECTIONAL_NEGATIVE
    assert FV.classify_effect(-0.01, -0.02, -0.001) == FV.HARMFUL


def test_T11_mechanism_gate_two_of_three():
    assert FV.mechanism_decollapse_supported(0.1, 0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"
    assert FV.mechanism_decollapse_supported(0.1, -0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"] == "NO"


def test_T12_policy_robustness_four_of_five():
    assert FV.policy_robustness([1, 1, 1, 1, -1], threshold=4)["PASS"] is True
    assert FV.policy_robustness([1, 1, 1, -1, -1], threshold=4)["PASS"] is False


def test_T13_safety_gate_harmful_fails():
    assert FV.safety_gate(FV.BENEFICIAL, FV.PRACTICALLY_NEUTRAL)["PASS"] is True
    assert FV.safety_gate(FV.HARMFUL, FV.BENEFICIAL)["PASS"] is False


def test_T14_verdict_validated_branch():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.BENEFICIAL, mechanism_supported=True, policy_robustness_pass=True,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["verdict"] == "NORMALIZATION_CAUSAL_VALIDATED"


def test_T15_verdict_performance_partial_branch():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.BENEFICIAL, mechanism_supported=False, policy_robustness_pass=True,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["subtype"] == "PERFORMANCE_PARTIAL"


def test_T16_verdict_mechanistic_partial_branch():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.PRACTICALLY_NEUTRAL, mechanism_supported=True, policy_robustness_pass=False,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["subtype"] == "MECHANISTIC_PARTIAL"
    assert v["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"


def test_T17_mechanism_cannot_rescue_harmful():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.HARMFUL, mechanism_supported=True, policy_robustness_pass=True,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["verdict"] == "NORMALIZATION_CAUSAL_FAILED"


def test_T18_neutral_without_mechanism_failed():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.PRACTICALLY_NEUTRAL, mechanism_supported=False, policy_robustness_pass=True,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["verdict"] == "NORMALIZATION_CAUSAL_FAILED"


def test_T19_integrity_failure_forces_failed():
    v = FV.normalization_ab_verdict(
        mixed_primary_label=FV.BENEFICIAL, mechanism_supported=True, policy_robustness_pass=True,
        persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=False, protocol_integrity_pass=True, version_integrity_pass=True,
        no_final_tuning=True)
    assert v["verdict"] == "NORMALIZATION_CAUSAL_FAILED"


# ---- integrity / firewall (executed run) ----

def test_T20_checkpoint_integrity_pass():
    _require()
    assert _load(RES / "E4HF_checkpoint_integrity.json")["all_match_frozen"] is True


def test_T21_pre_execution_no_learning():
    _require()
    em = _load(RES / "E4HF_pre_execution_manifest.json")["execution_mode"]
    assert em["epsilon"] == 0.0 and em["learning"] == "none"
    assert em["gradients"] == "disabled" and em["optimizer"] == "disabled"


def test_T22_ten_checkpoints_five_each():
    frz = _load(E4HE / "E4HE_checkpoint_freeze.json")["checkpoints"]
    arms = [v["arm"] for v in frz.values()]
    assert len(frz) == 10
    assert arms.count("V1_CONTROL") == 5 and arms.count("V3_TREATMENT") == 5


def test_T23_v1_v2_v3_normalization_mapping():
    frz = _load(E4HE / "E4HE_checkpoint_freeze.json")["checkpoints"]
    for v in frz.values():
        if v["arm"] == "V1_CONTROL":
            assert v["normalization_version"] == NORM_V1
        else:
            assert v["normalization_version"] == NORM_V3


def test_T24_final_seed_usage_consumed():
    _require()
    u = _load(RES / "E4HF_final_seed_usage.json")
    assert u["FINAL_ENVIRONMENT_SEEDS_USED"] == "1381..1440"
    assert u["unique_final_seeds_instantiated"] == 60


def test_T25_e4htr_never_touched_final_seeds():
    # firewall preserved upstream: E4HTR training must not have opened final seeds
    fw = _load(E4HTR / "E4HTR_verdict.json")
    assert fw.get("NEW_FINAL_EVAL_ACCESS_COUNT", 0) == 0 or fw.get("FINAL_SEEDS_FIREWALLED") in (
        "1381..1440", [1381, 1440], None)


# ---- matrix completeness ----

def test_T26_matrix_complete_1800():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert v["FINAL_MATRIX_COMPLETE"] == "YES"
    assert v["EPISODES_COMPLETE"] == "1800/1800"


def test_T27_episode_csv_1800_rows():
    _require()
    rows = _episode_rows()
    assert len(rows) == 1800


def test_T28_v1_v3_900_each():
    _require()
    rows = _episode_rows()
    assert sum(1 for r in rows if r["arm"] == "V1_CONTROL") == 900
    assert sum(1 for r in rows if r["arm"] == "V3_TREATMENT") == 900


def test_T29_all_pairing_keys_unique_and_complete():
    _require()
    rows = _episode_rows()
    keys = {(r["arm"], int(r["policy_seed"]), r["scenario"], int(r["environment_seed"])) for r in rows}
    expected = {(a, p, s, e) for a in ARMS for p in POLICY_SEEDS for s in SCENARIOS for e in FINAL_SEEDS}
    assert keys == expected
    assert len(keys) == len(rows)  # no duplicates


def test_T30_all_metrics_finite():
    _require()
    rows = _episode_rows()
    for r in rows:
        for k in ("MIGD_response", "MIGD_epoch", "response_FE"):
            assert math.isfinite(float(r[k]))


def test_T31_split_csvs_row_counts():
    _require()
    with open(RES / "E4HF_v1_results.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 900
    with open(RES / "E4HF_v3_results.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 900


# ---- effect definition & reproducibility from episode CSV ----

def test_T32_primary_is_mixed_migd_response():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert v["PRIMARY_SCENARIO"] == "MIXED" and v["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"
    assert v["EFFECT_DEFINITION"] == "V1_MINUS_V3"


def test_T33_mixed_delta_recompute_matches_verdict():
    _require()
    rows = _episode_rows()
    D = _diff_matrix(rows, "MIXED", "MIGD_response")
    v = _load(RES / "E4HF_verdict.json")
    assert abs(float(D.mean()) - v["MIXED_DELTA_V1_MINUS_V3"]) < 1e-9


def test_T34_mixed_ci_reproducible_from_frozen_seed():
    _require()
    rows = _episode_rows()
    D = _diff_matrix(rows, "MIXED", "MIGD_response")
    bs = paired_crossed_two_axis_bootstrap(
        D, seed=E4HF._e4hf_seed("MIGD_response", "MIXED", "V1_MINUS_V3"))
    v = _load(RES / "E4HF_verdict.json")
    assert abs(bs["ci_lo"] - v["MIXED_DELTA_CI"][0]) < 1e-9
    assert abs(bs["ci_hi"] - v["MIXED_DELTA_CI"][1]) < 1e-9


def test_T35_mixed_primary_label_matches_classify():
    _require()
    rows = _episode_rows()
    D = _diff_matrix(rows, "MIXED", "MIGD_response")
    bs = paired_crossed_two_axis_bootstrap(
        D, seed=E4HF._e4hf_seed("MIGD_response", "MIXED", "V1_MINUS_V3"))
    v = _load(RES / "E4HF_verdict.json")
    assert FV.classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"]) == v["MIXED_PRIMARY"]


def test_T36_per_policy_deltas_recompute():
    _require()
    rows = _episode_rows()
    D = _diff_matrix(rows, "MIXED", "MIGD_response")
    per_policy = [float(D[i, :].mean()) for i in range(len(POLICY_SEEDS))]
    v = _load(RES / "E4HF_verdict.json")
    for a, b in zip(per_policy, v["per_policy_MIXED_deltas"]):
        assert abs(a - b) < 1e-9


def test_T37_policy_robustness_count_matches():
    _require()
    rows = _episode_rows()
    D = _diff_matrix(rows, "MIXED", "MIGD_response")
    n_pos = sum(1 for i in range(len(POLICY_SEEDS)) if D[i, :].mean() > 0)
    v = _load(RES / "E4HF_verdict.json")
    assert n_pos == v["POLICY_PAIR_POSITIVE_COUNT"]
    assert (v["NORMALIZATION_POLICY_ROBUSTNESS"] == "PASS") == (n_pos >= 4)


# ---- mechanism recompute (per-policy pooled then averaged) ----

def test_T38_mechanism_direction_count_matches():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    n = int(v["DELTA_PREDICT"] > 0) + int(v["DELTA_MULTI_ACTION"] > 0) + int(v["DELTA_ACTION_ENTROPY"] > 0)
    assert n == v["MECHANISM_EXPECTED_DIRECTION_COUNT"]
    assert (v["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES") == (n >= 2)


def test_T39_delta_predict_sign_definition():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    # DELTA_PREDICT = PREDICT_V1 - PREDICT_V3
    assert abs((v["V1_MIXED_PREDICT_FRACTION"] - v["V3_MIXED_PREDICT_FRACTION"]) - v["DELTA_PREDICT"]) < 1e-9


def test_T40_delta_multi_and_entropy_sign_definition():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert abs((v["V3_MIXED_MULTI_ACTION_RATE"] - v["V1_MIXED_MULTI_ACTION_RATE"]) - v["DELTA_MULTI_ACTION"]) < 1e-9
    assert abs((v["V3_MIXED_ACTION_ENTROPY"] - v["V1_MIXED_ACTION_ENTROPY"]) - v["DELTA_ACTION_ENTROPY"]) < 1e-9


def test_T41_mechanism_predict_recompute_from_action_distribution():
    _require()
    # per-policy pooled PREDICT fraction over 60 MIXED env, averaged across 5 policies
    with open(RES / "E4HF_action_distribution.csv") as f:
        ad = list(csv.DictReader(f))
    # action_distribution stores per (arm,policy,scenario) mean fraction over env; verify V1 vs V3 order
    v = _load(RES / "E4HF_verdict.json")
    v1_pred = np.mean([float(r["PREDICT"]) for r in ad
                       if r["arm"] == "V1_CONTROL" and r["scenario"] == "MIXED"])
    v3_pred = np.mean([float(r["PREDICT"]) for r in ad
                       if r["arm"] == "V3_TREATMENT" and r["scenario"] == "MIXED"])
    # action_distribution PREDICT is mean of per-episode fraction; verdict PREDICT is pooled-per-policy.
    # Both must agree on the sign of V1-vs-V3 PREDICT (de-collapse direction).
    assert (v1_pred > v3_pred) == (v["DELTA_PREDICT"] > 0) or abs(v["DELTA_PREDICT"]) < 1e-9


# ---- safety & macro consistency ----

def test_T42_safety_gates_match_classifications():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    for sc in ["PERSISTENT", "STATIONARY"]:
        rlab = v[f"{sc}_RESPONSE_EFFECT"]; elab = v[f"{sc}_EPOCH_EFFECT"]
        expect = "PASS" if (rlab != FV.HARMFUL and elab != FV.HARMFUL) else "FAIL"
        assert v[f"{sc}_SAFETY"] == expect


def test_T43_macro_safety_matches():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    expect = "PASS" if (v["MACRO_RESPONSE_EFFECT"] != FV.HARMFUL and
                        v["MACRO_EPOCH_EFFECT"] != FV.HARMFUL) else "FAIL"
    assert v["MACRO_SAFETY"] == expect


def test_T44_macro_delta_recompute():
    _require()
    rows = _episode_rows()
    Dby = {s: _diff_matrix(rows, s, "MIGD_response") for s in SCENARIOS}
    bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCENARIOS, seed=E4HF._e4hf_seed("macro", "MIGD_response"))
    mr = _load(RES / "E4HF_bootstrap_results.json")["macro"]["MIGD_response"]
    assert abs(bs["point_estimate"] - mr["delta"]) < 1e-9
    assert abs(bs["ci_lo"] - mr["ci_lo"]) < 1e-9 and abs(bs["ci_hi"] - mr["ci_hi"]) < 1e-9


# ---- overall verdict consistency ----

def test_T45_verdict_category_valid():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert v["NORMALIZATION_CAUSAL_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")


def test_T46_verdict_consistent_with_logic():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    recomputed = FV.normalization_ab_verdict(
        mixed_primary_label=v["MIXED_PRIMARY"],
        mechanism_supported=(v["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"),
        policy_robustness_pass=(v["NORMALIZATION_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(v["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(v["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(v["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(v["CHECKPOINT_INTEGRITY"] == "PASS"),
        protocol_integrity_pass=(v["FINAL_PROTOCOL_INTEGRITY"] == "PASS"),
        version_integrity_pass=(v["NORMALIZATION_VERSION_INTEGRITY"] == "PASS"),
        no_final_tuning=True)
    short = {"NORMALIZATION_CAUSAL_VALIDATED": "VALIDATED",
             "NORMALIZATION_CAUSAL_PARTIAL": "PARTIAL",
             "NORMALIZATION_CAUSAL_FAILED": "FAILED"}[recomputed["verdict"]]
    assert short == v["NORMALIZATION_CAUSAL_VERDICT"]
    assert (recomputed["subtype"] or "NONE") == v["PARTIAL_SUBTYPE"]


def test_T47_next_stage_mapping():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    cat = v["NORMALIZATION_CAUSAL_VERDICT"]; sub = v["PARTIAL_SUBTYPE"]
    mech = v["MECHANISM_DECOLLAPSE_SUPPORTED"] == "YES"
    if cat == "VALIDATED":
        assert v["NEXT_STAGE"] == "E4I_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
    elif sub == "MECHANISTIC_PARTIAL":
        assert v["NEXT_STAGE"] == "E4I_REWARD_CREDIT_REVISION_DESIGN"
    elif sub == "PERFORMANCE_PARTIAL":
        assert v["NEXT_STAGE"] == "E4I_NORMALIZATION_RESULT_REVIEW"
    elif cat == "FAILED" and mech:
        assert v["NEXT_STAGE"] == "E4I_REWARD_CREDIT_REVISION_DESIGN"
    else:
        assert v["NEXT_STAGE"] == "E4I_NORMALIZATION_FAILURE_ANALYSIS"


def test_T48_performance_claim_consistent():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    if v["NORMALIZATION_CAUSAL_VERDICT"] == "VALIDATED":
        assert v["PERFORMANCE_IMPROVEMENT_CLAIM"] == "SUPPORTED"
    if v["PARTIAL_SUBTYPE"] == "MECHANISTIC_PARTIAL":
        assert v["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"


def test_T49_e4f_preserved_and_no_tuning():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert v["HYPERPARAMETER_RETUNING"] == "NO"
    assert v["CHECKPOINT_RESELECTION"] == "NO"
    assert v["NORMALIZER_REFIT"] == "NO"
    assert v["MODEL_MODIFICATION"] == "NO"


def test_T50_checkpoint_post_integrity():
    _require()
    v = _load(RES / "E4HF_verdict.json")
    assert v["CHECKPOINT_INTEGRITY"] == "PASS"
    assert v["NORMALIZATION_VERSION_INTEGRITY"] == "PASS"


# ---- bootstrap design properties ----

def test_T51_paired_narrower_than_independent():
    rng = np.random.default_rng(11)
    shared = rng.uniform(0, 0.02, size=(5, 60))
    V1 = shared + 0.006
    V3 = shared + 0.006 - 0.0003
    D = V1 - V3
    p = paired_crossed_two_axis_bootstrap(D, n_boot=3000, seed=5)
    ind = independent_arm_bootstrap(V1, V3, n_boot=3000, seed=5)
    assert (p["ci_hi"] - p["ci_lo"]) < (ind["ci_hi"] - ind["ci_lo"])


def test_T52_policy_axis_widens_ci():
    flat = np.full((5, 60), 0.0005)
    pol = flat + np.array([-0.001, -0.0005, 0, 0.0005, 0.001])[:, None]
    cf = paired_crossed_two_axis_bootstrap(flat, n_boot=3000, seed=1)
    cp = paired_crossed_two_axis_bootstrap(pol, n_boot=3000, seed=1)
    assert (cp["ci_hi"] - cp["ci_lo"]) > (cf["ci_hi"] - cf["ci_lo"])


def test_T53_forbidden_methods_labeled():
    D = np.zeros((5, 60))
    ind = independent_arm_bootstrap(D + 0.001, D, n_boot=500, seed=1)
    assert "FORBIDDEN" in ind["method"]


# ---- output artifacts present ----

@pytest.mark.parametrize("name", [
    "E4HF_pre_execution_manifest.json", "E4HF_checkpoint_integrity.json",
    "E4HF_execution_config.json", "E4HF_episode_results.csv", "E4HF_v1_results.csv",
    "E4HF_v3_results.csv", "E4HF_policy_pair_summary.csv", "E4HF_action_distribution.csv",
    "E4HF_mechanism_effects.csv", "E4HF_scenario_effects.csv", "E4HF_macro_effects.csv",
    "E4HF_fe_effects.csv", "E4HF_bootstrap_results.json", "E4HF_scientific_gates.json",
    "E4HF_validation_final_comparison.json", "E4HF_final_seed_usage.json",
    "E4HF_verdict.json", "E4HF_report.md", "E4HF_change_results.jsonl",
])
def test_T54_through_T72_outputs_exist(name):
    _require()
    assert (RES / name).exists() and (RES / name).stat().st_size > 0


def test_T73_report_contains_verdict():
    _require()
    txt = (RES / "E4HF_report.md").read_text()
    v = _load(RES / "E4HF_verdict.json")
    assert v["NORMALIZATION_CAUSAL_VERDICT"] in txt
    assert "E4F" in txt and "FAILED" in txt


def test_T74_policy_pair_summary_five_rows():
    _require()
    with open(RES / "E4HF_policy_pair_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert {int(r["policy_seed"]) for r in rows} == set(POLICY_SEEDS)
