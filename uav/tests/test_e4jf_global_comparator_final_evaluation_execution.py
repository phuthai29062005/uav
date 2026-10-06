"""
E4JF — Global Comparator Final Evaluation Execution: artifact + reproducibility tests.

Validates the executed V3+R1-vs-fixed-global comparison on seeds 1801..1860 WITHOUT rerunning
the 1080 episodes: integrity, matrix completeness, shared-baseline handling, primary MIXED
effect reproduced from the episode CSVs via the frozen crossed shared-baseline bootstrap
(seed 76001), robustness/safety/macro/verdict consistency, and preservation of E4F/E4HF/E4IF.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4JF_global_comparator_final_evaluation_execution as E4JF
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.evaluation.crossed_bootstrap import crossed_two_axis_bootstrap
from prdrl.evaluation import final_verdict as FV
from prdrl.experiments.E4J_global_comparator_protocol_design import global_comparator_verdict

RES = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4J = Path("results/E4J_global_comparator_protocol_design")
E4IF = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
POLICY = [4001, 4002, 4003, 4004, 4005]
SEEDS = list(range(1801, 1861))
SCEN = ["PERSISTENT", "STATIONARY", "MIXED"]
CMAP = {"PERSISTENT": "GLOBAL_PREDICT", "STATIONARY": "GLOBAL_DIVERSIFY", "MIXED": "GLOBAL_PREDICT"}


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4JF_verdict.json").exists():
        pytest.skip("E4JF not run yet")


def _learned():
    with open(RES / "E4JF_learned_episode_results.csv") as f:
        return list(csv.DictReader(f))


def _global():
    with open(RES / "E4JF_global_episode_results.csv") as f:
        return list(csv.DictReader(f))


def _Lmat(rows, scenario, endpoint):
    idx = {(int(r["policy_seed"]), int(r["environment_seed"])): float(r[endpoint])
           for r in rows if r["scenario"] == scenario}
    return np.array([[idx[(p, e)] for e in SEEDS] for p in POLICY])


def _Bvec(rows, scenario, endpoint):
    idx = {int(r["environment_seed"]): float(r[endpoint]) for r in rows if r["scenario"] == scenario}
    return np.array([idx[e] for e in SEEDS])


# ---- prerequisites / integrity ----

def test_T1_e4j_complete():
    assert _load(E4J / "E4J_verdict.json")["E4J_COMPLETE"] == "YES"


def test_T2_e4if_validated():
    assert _load(E4IF / "E4IF_verdict.json")["REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T3_T4_T5_learned_checkpoints():
    _require()
    ci = _load(RES / "E4JF_checkpoint_integrity.json")
    assert ci["all_match_frozen"] is True and len(ci["checkpoint_pre_hashes"]) == 5


def test_T6_policy_seeds():
    _require()
    assert sorted(int(k) for k in _load(RES / "E4JF_checkpoint_integrity.json")["checkpoint_pre_hashes"]) == POLICY


def test_T7_T8_T9_v3_r1_alpha():
    frz = _load(E4J / "E4J_learned_checkpoint_freeze.json")["checkpoints"]
    for v in frz.values():
        assert v["normalization_version"] == NORM_V3 and v["reward_version"] == REWARD_R1_NAME


def test_T10_final_seeds():
    _require()
    assert _load(RES / "E4JF_final_seed_usage.json")["FINAL_ENVIRONMENT_SEEDS_USED"] == "1801..1860"


def test_T12_T13_T14_T15_comparator_map():
    _require()
    ci = _load(RES / "E4JF_comparator_integrity.json")
    assert ci["MIXED"] == "GLOBAL_PREDICT" and ci["PERSISTENT"] == "GLOBAL_PREDICT"
    assert ci["STATIONARY"] == "GLOBAL_DIVERSIFY" and ci["map_matches_frozen"] is True


def test_T22_T23_T24_matrix_counts():
    _require()
    assert len(_learned()) == 900 and len(_global()) == 180


def test_T25_T26_unique_keys():
    _require()
    lk = {(int(r["policy_seed"]), r["scenario"], int(r["environment_seed"])) for r in _learned()}
    gk = {(r["scenario"], int(r["environment_seed"])) for r in _global()}
    assert len(lk) == 900 and len(gk) == 180


def test_T27_T28_no_missing_cells():
    _require()
    lk = {(int(r["policy_seed"]), r["scenario"], int(r["environment_seed"])) for r in _learned()}
    assert lk == {(p, s, e) for p in POLICY for s in SCEN for e in SEEDS}
    gk = {(r["scenario"], int(r["environment_seed"])) for r in _global()}
    assert gk == {(s, e) for s in SCEN for e in SEEDS}


def test_T29_T30_baseline_no_policy_axis():
    _require()
    cols = next(csv.reader(open(RES / "E4JF_global_episode_results.csv")))
    assert "policy_seed" not in cols
    # exactly one global row per (scenario, env)
    gk = [(r["scenario"], r["environment_seed"]) for r in _global()]
    assert len(gk) == len(set(gk)) == 180
    assert _load(RES / "E4JF_verdict.json")["BASELINE_POLICY_AXIS"] == "ABSENT"


def test_T39_T40_effect_definition():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert v["EFFECT_DEFINITION"] == "GLOBAL_MINUS_LEARNED" and v["POSITIVE_MEANS"] == "V3_R1_BETTER"


def test_T41_practical_margin():
    assert FV.DELTA_E4 == 1e-5


def test_T42_T43_T44_means_and_algebra():
    _require()
    L = _Lmat(_learned(), "MIXED", "MIGD_response"); B = _Bvec(_global(), "MIXED", "MIGD_response")
    v = _load(RES / "E4JF_verdict.json")
    assert abs(float(B.mean()) - v["GLOBAL_MIXED_MEAN_MIGD_RESPONSE"]) < 1e-9
    assert abs(float(L.mean()) - v["LEARNED_MIXED_MEAN_MIGD_RESPONSE"]) < 1e-9
    # algebra: mean(B-L over p,e) == mean_e B - mean_pe L
    assert abs(float((B[None, :] - L).mean()) - (B.mean() - L.mean())) < 1e-9
    assert v["ALGEBRA_CHECK"] == "PASS"


def test_T45_T52_T53_T54_bootstrap_reproduced():
    _require()
    L = _Lmat(_learned(), "MIXED", "MIGD_response"); B = _Bvec(_global(), "MIXED", "MIGD_response")
    bs = crossed_two_axis_bootstrap(L, B, seed=E4JF._e4jf_seed("MIGD_response", "MIXED", "GLOBAL_MINUS_LEARNED"))
    v = _load(RES / "E4JF_verdict.json")
    assert abs(bs["point_estimate"] - v["MIXED_DELTA_GLOBAL_MINUS_LEARNED"]) < 1e-9
    assert abs(bs["ci_lo"] - v["MIXED_DELTA_CI"][0]) < 1e-9 and abs(bs["ci_hi"] - v["MIXED_DELTA_CI"][1]) < 1e-9
    assert bs["n_policy"] == 5 and bs["n_env"] == 60
    assert E4JF.BOOTSTRAP_MASTER_SEED == 76001


def test_T55_mixed_primary_classify():
    _require()
    L = _Lmat(_learned(), "MIXED", "MIGD_response"); B = _Bvec(_global(), "MIXED", "MIGD_response")
    bs = crossed_two_axis_bootstrap(L, B, seed=E4JF._e4jf_seed("MIGD_response", "MIXED", "GLOBAL_MINUS_LEARNED"))
    v = _load(RES / "E4JF_verdict.json")
    assert FV.classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"]) == v["MIXED_PRIMARY"]


def test_T56_T57_policy_robustness():
    _require()
    L = _Lmat(_learned(), "MIXED", "MIGD_response"); B = _Bvec(_global(), "MIXED", "MIGD_response")
    per = [float((B - L[i, :]).mean()) for i in range(len(POLICY))]
    n_pos = sum(1 for x in per if x > 0)
    v = _load(RES / "E4JF_verdict.json")
    assert len(v["per_policy_MIXED_deltas"]) == 5 and n_pos == v["POLICY_PAIR_POSITIVE_COUNT"]
    assert (v["GLOBAL_COMPARATOR_POLICY_ROBUSTNESS"] == "PASS") == (n_pos >= 4)


def test_T58_T61_safety():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    for sc in ("PERSISTENT", "STATIONARY"):
        rlab = v[f"{sc}_RESPONSE_EFFECT"]; elab = v[f"{sc}_EPOCH_EFFECT"]
        expect = "PASS" if (rlab != FV.HARMFUL and elab != FV.HARMFUL) else "FAIL"
        assert v[f"{sc}_GLOBAL_SAFETY"] == expect


def test_T62_T63_T64_T65_macro():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert v["MACRO_COMPARATOR"] == "FROZEN_BEST_GLOBAL_BY_SCENARIO"
    expect = "PASS" if (v["MACRO_RESPONSE_EFFECT"] != FV.HARMFUL and v["MACRO_EPOCH_EFFECT"] != FV.HARMFUL) else "FAIL"
    assert v["GLOBAL_MACRO_SAFETY"] == expect


def test_T66_T67_fe():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert "MIXED_FE_EFFECT_GLOBAL_MINUS_LEARNED" in v  # secondary; cannot rescue (verdict is quality-driven)


def test_T68_no_reward_mechanism_regate():
    _require()
    # E4JF verdict has no reward-mechanism gate fields; success logic is comparator-only
    assert "REWARD_CREDIT_MECHANISM_SUPPORTED" not in _load(RES / "E4JF_verdict.json")


def test_T69_action_diversity_descriptive():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert isinstance(v["LEARNED_MIXED_PREDICT_FRACTION"], (int, float))


def test_T70_T71_T72_T73_verdict_logic():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    recomputed = global_comparator_verdict(
        mixed_primary_label=v["MIXED_PRIMARY"],
        robustness_pass=(v["GLOBAL_COMPARATOR_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(v["PERSISTENT_GLOBAL_SAFETY"] == "PASS"),
        stationary_safety_pass=(v["STATIONARY_GLOBAL_SAFETY"] == "PASS"),
        macro_safety_pass=(v["GLOBAL_MACRO_SAFETY"] == "PASS"),
        integrity_pass=(v["CHECKPOINT_INTEGRITY"] == "PASS" and v["COMPARATOR_MAP_INTEGRITY"] == "PASS"))
    short = {"GLOBAL_COMPARATOR_VALIDATED": "VALIDATED", "GLOBAL_COMPARATOR_PARTIAL": "PARTIAL",
             "GLOBAL_COMPARATOR_FAILED": "FAILED"}[recomputed["verdict"]]
    assert short == v["GLOBAL_COMPARATOR_VERDICT"]
    assert v["SUPERIORITY_CLAIM"] == recomputed["SUPERIORITY_CLAIM"]


def test_T74_T75_T76_T77_T78_T79_no_rescue():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert v["COMPARATOR_RESELECTION"] == "NO" and v["CHECKPOINT_RESELECTION"] == "NO"
    assert v["MODEL_MODIFICATION"] == "NO" and v["TRAINING"] == "NO"


def test_T80_T81_T82_T83_post_integrity():
    _require()
    pi = _load(RES / "E4JF_post_integrity.json")
    assert pi["checkpoint_post_hashes_match"] is True and pi["v3_sha_unchanged"] is True
    assert pi["comparator_map_unchanged"] is True


def test_T84_T85_T86_historical_preserved():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED" and v["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert v["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T87_final_seeds_consumed():
    _require()
    assert _load(RES / "E4JF_final_seed_usage.json")["status"] == "CONSUMED_GLOBAL_COMPARATOR_CONFIRMATORY_FINAL"


def test_T89_no_learning_audit():
    _require()
    a = _load(RES / "E4JF_no_learning_audit.json")
    for k in a:
        assert a[k] == 0


def test_T90_shared_baseline_handling_and_next_stage():
    _require()
    v = _load(RES / "E4JF_verdict.json")
    assert v["SHARED_BASELINE_HANDLING"] == "PASS"
    cat = v["GLOBAL_COMPARATOR_VERDICT"]
    integrity = v["CHECKPOINT_INTEGRITY"] == "PASS" and v["COMPARATOR_MAP_INTEGRITY"] == "PASS"
    if cat == "VALIDATED":
        assert v["NEXT_STAGE"] == "E5_BENCHMARK_EXTENSION_PROTOCOL_DESIGN"
    elif cat == "PARTIAL":
        assert v["NEXT_STAGE"] == "E4K_GLOBAL_COMPARATOR_RESULT_REVIEW"
    elif not integrity:
        assert v["NEXT_STAGE"] == "E4J_INTEGRITY_REVIEW"
    else:
        assert v["NEXT_STAGE"] == "E4K_JOINT_COORDINATION_REVISION_DESIGN"


def test_T91_scenario_effects_csv_comparator_labels():
    _require()
    with open(RES / "E4JF_scenario_effects.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["comparator"] == CMAP[r["scenario"]]


def test_T92_all_metrics_finite():
    _require()
    for r in _learned() + _global():
        for k in ("MIGD_response", "MIGD_epoch", "response_FE"):
            assert math.isfinite(float(r[k]))
