"""
E4J — Global Comparator Protocol Design: contract + unit tests.

No fresh comparator environment opened. Verifies the frozen V3+R1-vs-fixed-global protocol:
historically-verified comparator map, frozen learned checkpoints, fresh disjoint seeds
1801..1860, crossed shared-baseline bootstrap (seed 76001), robustness/safety/success logic,
and the 24-check synthetic validation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4J_global_comparator_protocol_design as E4J
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.evaluation.crossed_bootstrap import crossed_two_axis_bootstrap, naive_row_bootstrap
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4J_global_comparator_protocol_design")
E4IF = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4E = Path("results/E4E_final_evaluation_protocol_freeze")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4J_verdict.json").exists():
        pytest.skip("E4J not run yet")


def test_T1_e4if_complete():
    assert _load(E4IF / "E4IF_verdict.json")["E4IF_COMPLETE"] == "YES"


def test_T2_e4if_validated():
    assert _load(E4IF / "E4IF_verdict.json")["REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T4_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T5_T6_T7_five_learned_checkpoints():
    _require()
    ck = _load(RES / "E4J_learned_checkpoint_freeze.json")
    assert ck["n_checkpoints"] == 5 and ck["no_reselection"] is True and ck["no_retrain"] is True
    for v in ck["checkpoints"].values():
        assert v["hash_match"] is True


def test_T8_T9_T10_v3_r1_metadata():
    _require()
    for v in _load(RES / "E4J_learned_checkpoint_freeze.json")["checkpoints"].values():
        assert v["normalization_version"] == NORM_V3 and v["reward_version"] == REWARD_R1_NAME
        assert v["metadata_ok"] is True


def test_T11_T12_historical_evidence():
    _require()
    ev = _load(RES / "E4J_historical_global_comparator_evidence.json")
    assert ev["GLOBAL_COMPARATOR_MAP_VERIFIED"] is True and ev["agree"] is True
    assert ev["not_selected_on_new_final_seeds"] is True
    # independently confirm from E4E artifact
    e4e = _load(E4E / "E4E_comparator_freeze.json")["primary_comparator_by_scenario"]
    assert ev["E4E_map"] == e4e


def test_T13_T14_T15_comparator_map():
    _require()
    m = _load(RES / "E4J_global_comparator_map.json")["FROZEN_BEST_GLOBAL_BY_SCENARIO"]
    assert m["MIXED"] == "GLOBAL_PREDICT" and m["PERSISTENT"] == "GLOBAL_PREDICT"
    assert m["STATIONARY"] == "GLOBAL_DIVERSIFY"


def test_T16_no_reselection_on_new_data():
    _require()
    assert _load(RES / "E4J_global_comparator_map.json")["no_reselection_on_final"] is True


def test_T17_T18_T19_T20_operator_semantics():
    _require()
    op = _load(RES / "E4J_global_operator_contract.json")
    assert op["GLOBAL_PREDICT"]["sigma_P"] == 0.005 and op["GLOBAL_PREDICT"]["lambda_P"] == 1.0
    assert op["GLOBAL_PREDICT"]["fallback"] == "LOCAL" and op["GLOBAL_DIVERSIFY"]["sigma_D"] == 0.15


def test_T21_T22_T23_budget_survival():
    _require()
    op = _load(RES / "E4J_global_operator_contract.json")
    assert op["budget"] == "10 PER_ACTIVE_REGION" and "NSGA-II" in op["survival"]
    assert op["no_HARD"] is True


def test_T24_T25_fresh_seeds_disjoint():
    _require()
    sc = _load(RES / "E4J_fresh_seed_contract.json")
    assert sc["fresh_seeds"] == [1801, 1860] and sc["label"] == "GLOBAL_COMPARATOR_CONFIRMATORY_FINAL_ONLY"
    dj = _load(RES / "E4J_seed_disjointness.json")
    assert dj["disjoint_from_all_prior"] is True
    fresh = set(range(1801, 1861))
    consumed = set()
    for v in E4J.CONSUMED_SEED_SETS.values():
        consumed |= set(v)
    assert len(fresh & consumed) == 0


def test_T26_final_access_zero():
    _require()
    assert _load(RES / "E4J_final_firewall.json")["GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT"] == 0
    assert _load(RES / "E4J_verdict.json")["GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT"] == 0


def test_T27_T28_no_new_train_val_split():
    _require()
    sc = _load(RES / "E4J_fresh_seed_contract.json")
    assert sc["no_train_split"] is True and sc["no_validation_split"] is True


def test_T29_T30_T31_matrix():
    _require()
    m = _load(RES / "E4J_matrix_contract.json")
    assert m["learned"]["episodes"] == 900 and m["global"]["episodes"] == 180
    assert m["total_episodes"] == 1080
    assert m["global"]["one_realization_per_env_scenario"] is True
    assert m["no_baseline_duplication_as_independent_observations"] is True


def test_T32_T33_T34_T35_endpoint_effect():
    _require()
    v = _load(RES / "E4J_verdict.json")
    assert v["PRIMARY_SCENARIO"] == "MIXED" and v["PRIMARY_ENDPOINT"] == "MIGD_RESPONSE"
    assert v["EFFECT_DEFINITION"] == "GLOBAL_MINUS_LEARNED" and v["POSITIVE_MEANS"] == "V3_R1_BETTER"


def test_T36_practical_margin():
    _require()
    assert _load(RES / "E4J_practical_margin.json")["DELTA_NORM"] == 1e-5


def test_T37_T38_T39_T40_bootstrap_design():
    _require()
    b = _load(RES / "E4J_bootstrap_contract.json")
    assert b["design"] == "CROSSED_POLICY_ENV_BOOTSTRAP_WITH_SHARED_ENV_BASELINE"
    assert b["policy_axis_n"] == 5 and b["environment_axis_n"] == 60
    assert "(n_env,)" in b["baseline_shape"]


def test_T41_T42_bootstrap_settings():
    _require()
    b = _load(RES / "E4J_bootstrap_contract.json")
    assert b["N_BOOTSTRAP"] == 50000 and b["BOOTSTRAP_MASTER_SEED"] == 76001
    assert E4J.BOOTSTRAP_MASTER_SEED == 76001


def test_T43_T44_forbidden_bootstraps():
    _require()
    b = _load(RES / "E4J_bootstrap_contract.json")
    assert any("iid" in f for f in b["forbidden"]) and any("independent" in f for f in b["forbidden"])


def test_T45_policy_robustness():
    _require()
    r = _load(RES / "E4J_policy_robustness.json")
    assert r["threshold"] == 4 and ">=4 of 5" in r["rule"]


def test_T46_T47_scenario_safety_comparators():
    _require()
    s = _load(RES / "E4J_scenario_safety.json")
    assert s["PERSISTENT_comparator"] == "GLOBAL_PREDICT" and s["STATIONARY_comparator"] == "GLOBAL_DIVERSIFY"


def test_T48_T49_T50_macro():
    _require()
    m = _load(RES / "E4J_macro_contract.json")
    assert m["comparator_by_scenario"]["MIXED"] == "GLOBAL_PREDICT"
    assert m["scenario_weight"] == 1/3 and m["shared_bootstrap_draws_across_scenarios"] is True


def test_T51_T52_fe():
    _require()
    fe = _load(RES / "E4J_fe_contract.json")
    assert fe["secondary"] is True and fe["cannot_rescue_harmful_quality"] is True


def test_T53_no_reward_mechanism_regate():
    _require()
    sl = _load(RES / "E4J_success_logic.json")
    assert sl["no_new_reward_mechanism_gate"] is True
    assert sl["reward_causality_already_confirmed_by_E4IF"] is True


def test_T54_action_diversity_descriptive():
    _require()
    ac = _load(RES / "E4J_action_context_contract.json")
    assert ac["action_diversity_not_scored_as_superiority"] is True


def test_T55_validated_logic():
    v = E4J.global_comparator_verdict(FV.BENEFICIAL, True, True, True, True, True)
    assert v["verdict"] == "GLOBAL_COMPARATOR_VALIDATED" and v["SUPERIORITY_CLAIM"] == "SUPPORTED"


def test_T56_partial_logic():
    v = E4J.global_comparator_verdict(FV.BENEFICIAL, False, True, True, True, True)
    assert v["verdict"] == "GLOBAL_COMPARATOR_PARTIAL" and v["subtype"] == "PERFORMANCE_PARTIAL"


def test_T57_failed_logic():
    for lab in (FV.DIRECTIONAL_NEGATIVE, FV.HARMFUL):
        assert E4J.global_comparator_verdict(lab, True, True, True, True, True)["verdict"] == "GLOBAL_COMPARATOR_FAILED"
    assert E4J.global_comparator_verdict(FV.BENEFICIAL, True, False, True, True, True)["verdict"] == "GLOBAL_COMPARATOR_FAILED"
    assert E4J.global_comparator_verdict(FV.BENEFICIAL, True, True, True, True, False)["verdict"] == "GLOBAL_COMPARATOR_FAILED"


def test_T58_neutral_logic():
    v = E4J.global_comparator_verdict(FV.PRACTICALLY_NEUTRAL, True, True, True, True, True)
    assert v["verdict"] == "GLOBAL_COMPARATOR_PARTIAL" and v["subtype"] == "PRACTICALLY_NEUTRAL"
    assert v["SUPERIORITY_CLAIM"] == "NOT_SUPPORTED"


@pytest.mark.parametrize("key", [f"{i}_" for i in range(1, 25)])
def test_T59_through_T68_synthetic_checks(key):
    _require()
    synth = _load(RES / "E4J_synthetic_validation.json")
    matches = [k for k in synth["checks"] if k.startswith(key)]
    assert matches, f"missing synthetic check {key}"
    for k in matches:
        assert synth["checks"][k]["PASS"] is True


def test_T68b_synthetic_all_pass():
    _require()
    synth = _load(RES / "E4J_synthetic_validation.json")
    assert synth["ALL_PASS"] is True and synth["n_checks"] == 24


def test_T64_shared_baseline_unit():
    # crossed bootstrap: shared baseline B[e] (n_env,) not a policy axis; point == grand mean
    rng = np.random.default_rng(0)
    L = rng.uniform(0.006, 0.008, (5, 60)); B = L.mean(axis=0)
    c = crossed_two_axis_bootstrap(L, B, n_boot=1000, seed=76001)
    assert abs(c["point_estimate"] - float((B[None, :] - L).mean())) < 1e-12


def test_T43_naive_forbidden_labelled():
    nb = naive_row_bootstrap(np.full((5, 60), 0.007), np.full(60, 0.0072), n_boot=500, seed=1)
    assert "FORBIDDEN" in nb["method"]


def test_T69_T70_T71_no_execution():
    _require()
    v = _load(RES / "E4J_verdict.json")
    assert v["NEW_TRAINING"] == "NO" and v["NEW_ENVIRONMENT_EXECUTION"] == "NO"
    assert _load(RES / "E4J_pre_manifest.json")["no_new_environment_execution"] is True


def test_T72_historical_final_seeds_not_reused():
    _require()
    dj = _load(RES / "E4J_seed_disjointness.json")
    assert "1081..1140" in dj["explicitly_not_reused"] and "1581..1640" in dj["explicitly_not_reused"]


def test_T73_e4jf_schema_and_next_stage():
    _require()
    sch = _load(RES / "E4J_output_schema.json")
    assert len(sch["e4jf_required_outputs"]) == 18
    assert _load(RES / "E4J_verdict.json")["NEXT_STAGE"] == "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION"


def test_T74_next_stage_routing_frozen():
    _require()
    nl = _load(RES / "E4J_next_stage_logic.json")
    assert nl["if_VALIDATED"] == "E5_BENCHMARK_EXTENSION_PROTOCOL_DESIGN"
    assert nl["if_FAILED_negative_or_harmful"] == "E4K_JOINT_COORDINATION_REVISION_DESIGN"
    assert nl["no_joint_architecture_change_in_E4J"] is True


def test_T75_e4j_complete():
    _require()
    assert _load(RES / "E4J_verdict.json")["E4J_COMPLETE"] == "YES"
