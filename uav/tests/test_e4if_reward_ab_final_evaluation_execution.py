"""
E4IF — Reward A/B Final Evaluation Execution: artifact + reproducibility tests.

Validates the executed R0-vs-R1 final causal experiment on seeds 1581..1640 WITHOUT rerunning
the 1800 episodes: firewall/integrity, matrix completeness, primary MIXED effect reproduced
from the episode CSV via the frozen paired bootstrap (seed 66001), mechanism direction/gate,
safety/macro/verdict consistency, and preservation of the immutable E4F/E4HF verdicts.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4IF_reward_ab_final_evaluation_execution as E4IF
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME
from prdrl.rl.normalization import NORM_V3
from prdrl.rl import reward_integration as RI
from prdrl.evaluation.crossed_bootstrap import paired_crossed_two_axis_bootstrap
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4IE = Path("results/E4IE_reward_ab_final_evaluation_protocol_freeze")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
POLICY = [4001, 4002, 4003, 4004, 4005]
SEEDS = list(range(1581, 1641))
SCEN = ["PERSISTENT", "STATIONARY", "MIXED"]
ARMS = ["R0_CONTROL", "R1_TREATMENT"]
SHA = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4IF_verdict.json").exists():
        pytest.skip("E4IF not run yet")


def _episodes():
    with open(RES / "E4IF_episode_results.csv") as f:
        return list(csv.DictReader(f))


def _diff(rows, scenario, endpoint):
    idx = {(r["arm"], int(r["policy_seed"]), int(r["environment_seed"])): float(r[endpoint])
           for r in rows if r["scenario"] == scenario}
    D = np.zeros((len(POLICY), len(SEEDS)))
    for i, p in enumerate(POLICY):
        for j, e in enumerate(SEEDS):
            D[i, j] = idx[("R0_CONTROL", p, e)] - idx[("R1_TREATMENT", p, e)]
    return D


# ---- prerequisites / integrity ----

def test_T1_e4ie_complete():
    assert _load(E4IE / "E4IE_verdict.json")["E4IE_COMPLETE"] == "YES"


def test_T2_T3_frozen_ten_checkpoints_prehash():
    _require()
    ci = _load(RES / "E4IF_checkpoint_integrity.json")
    assert ci["all_match_frozen"] is True and len(ci["checkpoint_pre_hashes"]) == 10


def test_T4_T5_five_each():
    frz = _load(E4IE / "E4IE_checkpoint_freeze.json")["checkpoints"]
    arms = [v["arm"] for v in frz.values()]
    assert arms.count("R0_CONTROL") == 5 and arms.count("R1_TREATMENT") == 5


def test_T6_T7_T8_reward_metadata():
    frz = _load(E4IE / "E4IE_checkpoint_freeze.json")["checkpoints"]
    for v in frz.values():
        if v["arm"] == "R0_CONTROL":
            assert v["reward_version"] == REWARD_R0_NAME
        else:
            assert v["reward_version"] == REWARD_R1_NAME and v["reward_config_sha256"] == SHA


def test_T9_T10_T11_alpha_and_v3():
    ri = _load(E4IE / "E4IE_reward_version_integrity.json")
    assert ri["alpha_local_R0"] == 0.1 and ri["alpha_local_R1"] == 0.1
    assert _load(E4IE / "E4IE_normalization_integrity.json")["normalization"] == NORM_V3


def test_T12_T13_T14_final_seeds_scenarios_policies():
    _require()
    u = _load(RES / "E4IF_final_seed_usage.json")
    assert u["FINAL_ENVIRONMENT_SEEDS_USED"] == "1581..1640" and u["unique_final_seeds"] == 60
    rows = _episodes()
    assert {r["scenario"] for r in rows} == set(SCEN)
    assert {int(r["policy_seed"]) for r in rows} == set(POLICY)


def test_T15_T16_T17_matrix_counts():
    _require()
    rows = _episodes()
    assert len(rows) == 1800
    assert sum(1 for r in rows if r["arm"] == "R0_CONTROL") == 900
    assert sum(1 for r in rows if r["arm"] == "R1_TREATMENT") == 900


def test_T18_T19_no_duplicate_no_missing():
    _require()
    rows = _episodes()
    keys = {(r["arm"], int(r["policy_seed"]), r["scenario"], int(r["environment_seed"])) for r in rows}
    expected = {(a, p, s, e) for a in ARMS for p in POLICY for s in SCEN for e in SEEDS}
    assert keys == expected and len(keys) == len(rows)


def test_T20_all_migd_finite():
    _require()
    for r in _episodes():
        for k in ("MIGD_response", "MIGD_epoch", "response_FE"):
            assert math.isfinite(float(r[k]))


def test_T22_T27_execution_mode():
    _require()
    em = _load(RES / "E4IF_pre_execution_manifest.json")["execution_mode"]
    assert em["epsilon"] == 0.0 and em["gradients"] == "disabled" and em["learning"] == "none"
    a = _load(RES / "E4IF_no_learning_audit.json")
    for k in a:
        assert a[k] == 0


def test_T30_T31_effect_definition():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    assert v["EFFECT_DEFINITION"] == "R0_MINUS_R1" and v["PRIMARY_COMPARISON"] == "R1_TREATMENT_VS_R0_CONTROL"


def test_T32_practical_margin():
    assert FV.DELTA_E4 == 1e-5


def test_T33_T37_mixed_delta_and_ci_reproduced():
    _require()
    rows = _episodes()
    D = _diff(rows, "MIXED", "MIGD_response")
    v = _load(RES / "E4IF_verdict.json")
    assert abs(float(D.mean()) - v["MIXED_DELTA_R0_MINUS_R1"]) < 1e-9
    bs = paired_crossed_two_axis_bootstrap(D, seed=E4IF._e4if_seed("MIGD_response", "MIXED", "R0_MINUS_R1"))
    assert abs(bs["ci_lo"] - v["MIXED_DELTA_CI"][0]) < 1e-9 and abs(bs["ci_hi"] - v["MIXED_DELTA_CI"][1]) < 1e-9
    assert v["BOOTSTRAP_MASTER_SEED"] if "BOOTSTRAP_MASTER_SEED" in v else True
    assert E4IF.BOOTSTRAP_MASTER_SEED == 66001


def test_T40_mixed_primary_matches_classify():
    _require()
    rows = _episodes()
    D = _diff(rows, "MIXED", "MIGD_response")
    bs = paired_crossed_two_axis_bootstrap(D, seed=E4IF._e4if_seed("MIGD_response", "MIXED", "R0_MINUS_R1"))
    v = _load(RES / "E4IF_verdict.json")
    assert FV.classify_effect(bs["point_estimate"], bs["ci_lo"], bs["ci_hi"]) == v["MIXED_PRIMARY"]


def test_T41_T42_policy_robustness():
    _require()
    rows = _episodes()
    D = _diff(rows, "MIXED", "MIGD_response")
    per = [float(D[i, :].mean()) for i in range(len(POLICY))]
    n_pos = sum(1 for x in per if x > 0)
    v = _load(RES / "E4IF_verdict.json")
    assert len(v["per_policy_MIXED_deltas"]) == 5
    assert n_pos == v["POLICY_PAIR_POSITIVE_COUNT"]
    assert (v["REWARD_POLICY_ROBUSTNESS"] == "PASS") == (n_pos >= 4)


def test_T45_T48_T53_mechanism_directions():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    assert (v["R1_LOCAL_CREDIT_ACTIVE_FRACTION"] - v["R0_LOCAL_CREDIT_ACTIVE_FRACTION"]) == pytest.approx(v["DELTA_ACTIVITY"], abs=1e-9)
    if isinstance(v["DELTA_GLOBAL_LOCAL_RATIO"], (int, float)):
        assert (v["R0_GLOBAL_LOCAL_MAGNITUDE_RATIO"] - v["R1_GLOBAL_LOCAL_MAGNITUDE_RATIO"]) == pytest.approx(v["DELTA_GLOBAL_LOCAL_RATIO"], abs=1e-6)
    assert (v["R1_REGIONAL_REWARD_DISPERSION"] - v["R0_REGIONAL_REWARD_DISPERSION"]) == pytest.approx(v["DELTA_REWARD_DISPERSION"], abs=1e-12)


def test_T54_mechanism_gate_count():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    da = int(v["DELTA_ACTIVITY"] > 0)
    dr = int(v["DELTA_GLOBAL_LOCAL_RATIO"] > 0) if isinstance(v["DELTA_GLOBAL_LOCAL_RATIO"], (int, float)) else 0
    dd = int(v["DELTA_REWARD_DISPERSION"] > 0)
    assert (da + dr + dd) == v["REWARD_MECHANISM_EXPECTED_DIRECTION_COUNT"]
    assert (v["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES") == ((da + dr + dd) >= 2)


def test_T52_dispersion_ddof0_contract():
    # frozen contract states ddof=0; verify recorded in E4IE mechanism contract
    m = _load(E4IE / "E4IE_reward_mechanism_contract.json")
    assert "ddof=0" in m["endpoint_3"]["metric"]


def test_T55_action_diversity_not_gate():
    _require()
    # action metrics present but NOT part of mechanism gate
    v = _load(RES / "E4IF_verdict.json")
    assert isinstance(v["R1_MIXED_PREDICT_FRACTION"], (int, float))
    assert v["REWARD_CREDIT_MECHANISM_SUPPORTED"] in ("YES", "NO")


def test_T56_T61_safety_gates():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    for sc in ("PERSISTENT", "STATIONARY"):
        rlab = v[f"{sc}_RESPONSE_EFFECT"]; elab = v[f"{sc}_EPOCH_EFFECT"]
        expect = "PASS" if (rlab != FV.HARMFUL and elab != FV.HARMFUL) else "FAIL"
        assert v[f"{sc}_SAFETY"] == expect


def test_T62_T63_T64_macro():
    _require()
    rows = _episodes()
    Dby = {s: _diff(rows, s, "MIGD_response") for s in SCEN}
    from prdrl.evaluation.crossed_bootstrap import paired_crossed_two_axis_bootstrap_macro
    bs = paired_crossed_two_axis_bootstrap_macro(Dby, SCEN, seed=E4IF._e4if_seed("macro", "MIGD_response"))
    mr = _load(RES / "E4IF_bootstrap_results.json")["macro"]["MIGD_response"]
    assert abs(bs["point_estimate"] - mr["delta"]) < 1e-9
    v = _load(RES / "E4IF_verdict.json")
    expect = "PASS" if (v["MACRO_RESPONSE_EFFECT"] != FV.HARMFUL and v["MACRO_EPOCH_EFFECT"] != FV.HARMFUL) else "FAIL"
    assert v["MACRO_SAFETY"] == expect


def test_T65_T66_fe_secondary():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    # FE effects present; verdict is quality-driven (FE cannot rescue) — verdict logic ignores FE
    assert "MIXED_FE_EFFECT_R0_MINUS_R1" in v


def test_T67_T71_verdict_logic_consistent():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    recomputed = RI.reward_ab_verdict(
        mixed_primary_label=v["MIXED_PRIMARY"],
        mechanism_supported=(v["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"),
        policy_robustness_pass=(v["REWARD_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(v["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(v["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(v["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(v["CHECKPOINT_INTEGRITY"] == "PASS"),
        protocol_integrity_pass=(v["REWARD_VERSION_CONFIG_INTEGRITY"] == "PASS"),
        version_integrity_pass=(v["NORMALIZATION_INTEGRITY"] == "PASS"), no_final_tuning=True)
    short = {"REWARD_CAUSAL_VALIDATED": "VALIDATED", "REWARD_CAUSAL_PARTIAL": "PARTIAL",
             "REWARD_CAUSAL_FAILED": "FAILED"}[recomputed["verdict"]]
    assert short == v["REWARD_CAUSAL_VERDICT"]
    assert (recomputed["subtype"] or "NONE") == v["PARTIAL_SUBTYPE"]
    assert recomputed["PERFORMANCE_IMPROVEMENT_CLAIM"] == v["PERFORMANCE_IMPROVEMENT_CLAIM"]


def test_T72_T78_no_rescue_flags():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    assert v["ALPHA_RETUNING"] == "NO" and v["REWARD_RETUNING"] == "NO"
    assert v["CHECKPOINT_RESELECTION"] == "NO" and v["MODEL_MODIFICATION"] == "NO"
    assert v["R1_ALPHA_LOCAL"] == 0.1


def test_T79_validation_comparison_descriptive():
    _require()
    vc = _load(RES / "E4IF_validation_final_comparison.json")
    assert "does not modify verdict" in vc["note"]


def test_T80_T81_T82_post_integrity():
    _require()
    pi = _load(RES / "E4IF_post_integrity.json")
    assert pi["checkpoint_post_hashes_match"] is True and pi["v3_sha_unchanged"] is True
    assert pi["r1_config_unchanged"] is True
    v = _load(RES / "E4IF_verdict.json")
    assert v["CHECKPOINT_INTEGRITY"] == "PASS" and v["NORMALIZATION_INTEGRITY"] == "PASS"


def test_T83_T84_historical_preserved():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED" and v["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T85_final_seeds_consumed():
    _require()
    assert _load(RES / "E4IF_final_seed_usage.json")["status"] == "CONSUMED_FINAL_REWARD_AB"


def test_T86_verdict_category_and_next_stage():
    _require()
    v = _load(RES / "E4IF_verdict.json")
    assert v["REWARD_CAUSAL_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")
    cat = v["REWARD_CAUSAL_VERDICT"]; sub = v["PARTIAL_SUBTYPE"]
    mech = v["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES"
    if cat == "VALIDATED":
        assert v["NEXT_STAGE"] == "E4J_GLOBAL_COMPARATOR_PROTOCOL_DESIGN"
    elif sub == "MECHANISTIC_PARTIAL":
        assert v["NEXT_STAGE"] == "E4J_JOINT_COORDINATION_REVISION_DESIGN"
    elif sub == "PERFORMANCE_PARTIAL":
        assert v["NEXT_STAGE"] == "E4J_REWARD_RESULT_REVIEW"
    elif cat == "FAILED" and mech:
        assert v["NEXT_STAGE"] == "E4J_JOINT_COORDINATION_REVISION_DESIGN"
    else:
        assert v["NEXT_STAGE"] == "E4J_REWARD_FAILURE_ANALYSIS"


def test_T87_mechanism_effects_csv_matches_verdict():
    _require()
    with open(RES / "E4IF_reward_mechanism_effects.csv") as f:
        rows = {r["endpoint"]: r for r in csv.DictReader(f)}
    v = _load(RES / "E4IF_verdict.json")
    assert float(rows["LOCAL_CREDIT_ACTIVITY"]["R1"]) == pytest.approx(v["R1_LOCAL_CREDIT_ACTIVE_FRACTION"], abs=1e-9)
    assert float(rows["REGIONAL_REWARD_DISPERSION"]["R1"]) == pytest.approx(v["R1_REGIONAL_REWARD_DISPERSION"], abs=1e-9)


def test_T88_regional_records_reconstruct():
    _require()
    with open(RES / "E4IF_regional_reward_records.csv") as f:
        rows = list(csv.DictReader(f))[:2000]
    from prdrl.rl.reward_r1 import regional_quality_gain_r1
    for r in rows:
        g = regional_quality_gain_r1(float(r["L_pre"]), float(r["L_post"]))
        assert abs(g - float(r["g_r"])) <= 1e-12
        assert abs(float(r["weighted_local_reward_R1"]) - 0.1 * g) <= 1e-12
