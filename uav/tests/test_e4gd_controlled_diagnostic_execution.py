"""
E4GD — Controlled Diagnostic Execution: integrity / reproducibility / contract tests.

Verify E4GD changed nothing (no training/weights/reward/final-seed reuse), executed
the frozen D1..D8 contracts, and that recorded classifications are consistent with
the frozen logic. Never assert a particular scientific outcome beyond what the
frozen thresholds imply from the recorded numbers.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4GD_controlled_diagnostic_execution as E4GD

RES = Path("results/E4GD_controlled_diagnostic_execution")
E4G = Path("results/E4G_controlled_diagnostic_design")
E4F = Path("results/E4F_final_evaluation")
E4E = Path("results/E4E_final_evaluation_protocol_freeze")
E4TR = Path("results/E4TR_shared_double_dqn_training")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _hash(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _require():
    if not (RES / "E4GD_verdict.json").exists():
        pytest.skip("E4GD not run yet")


# ---- integrity ----

def test_T1_e4g_complete():
    assert _load(E4G / "E4G_verdict.json")["E4G_COMPLETE"] == "YES"


def test_T2_e4f_verdict_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T3_e4f_verdict_hash_unchanged():
    _require()
    man = _load(RES / "E4GD_pre_manifest.json")
    assert man["e4f_verdict_hash"] == _hash(E4F / "E4F_verdict.json")


def test_T4_checkpoint_hashes_unchanged():
    _require()
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in ["2001", "2002", "2003", "2004", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_T5_production_hashes_unchanged():
    _require()
    man = _load(RES / "E4GD_pre_manifest.json")
    for fp, h in man["production_operator_hashes"].items():
        assert h == _hash(fp)


def test_T6_T7_T8_T9_no_training_no_model_mod():
    _require()
    man = _load(RES / "E4GD_pre_manifest.json")
    assert man["no_training"] is True
    assert man["no_model_modification"] is True
    add = _load(RES / "E4GD_verdict_addendum.json")
    assert add["E4GD_TRAINS_NEW_MODEL"] is False
    usage = _load(RES / "E4GD_diagnostic_seed_usage.json")
    assert usage["training_performed"] is False


def test_T10_T11_diagnostic_seeds():
    _require()
    usage = _load(RES / "E4GD_diagnostic_seed_usage.json")
    assert usage["seeds"] == list(range(1201, 1221))
    assert usage["label"] == "POST_HOC_DIAGNOSTIC_ONLY"


def test_T12_no_final_seeds_used():
    _require()
    usage = _load(RES / "E4GD_diagnostic_seed_usage.json")
    assert usage["final_seed_overlap"] == "NONE"
    assert usage["final_eval_access_count"] == 0
    # no final seed appears in reward components CSV env-seed column
    seeds = set()
    with open(RES / "E4GD_D4_reward_components.csv") as f:
        for r in csv.DictReader(f):
            seeds.add(int(r["environment_seed"]))
    assert seeds.issubset(set(range(1201, 1221)))


def test_T13_d1_thresholds_frozen():
    _require()
    d1 = _load(RES / "E4GD_D1_normalization_results.json")
    c = _load(E4G / "E4G_D1_normalization_contract.json")
    assert d1["thresholds"]["NEAR_CONST_RANGE"] == c["NEAR_CONST_RANGE"]
    assert d1["thresholds"]["NEAR_CONST_STD"] == c["NEAR_CONST_STD"]
    assert d1["D1_NORMALIZATION_SIGNAL"] in ("PRESERVED", "COMPRESSED", "SEVERELY_COMPRESSED", "NOT_AUDITABLE")


def test_T14_d1_stats_reproducible():
    _require()
    rows = list(csv.DictReader(open(RES / "E4GD_D1_feature_distributions.csv")))
    # classification consistent with near_constant count
    d1 = _load(RES / "E4GD_D1_normalization_results.json")
    n_local_nc = sum(1 for r in rows if r["group"] == "local" and r["near_constant"] == "True")
    assert n_local_nc == d1["near_constant_local_features"]
    expect = ("SEVERELY_COMPRESSED" if n_local_nc >= 3 else
              "COMPRESSED" if n_local_nc >= 1 else "PRESERVED")
    assert d1["D1_NORMALIZATION_SIGNAL"] == expect


def test_T15_T16_d2_identical_folds_probes_diagnostic():
    _require()
    d2 = _load(RES / "E4GD_D2_separability_results.json")
    if d2["D2_NORMALIZATION_SEPARABILITY"] != "NOT_AUDITABLE":
        rows = list(csv.DictReader(open(RES / "E4GD_D2_probe_results.csv")))
        probes = {r["probe"] for r in rows}
        reps = {r["representation"] for r in rows}
        assert probes == {"logistic", "tree", "kNN"}
        assert reps == {"RAW", "NORMALIZED"}


def test_T17_T18_d3_code_audit_and_circularity():
    _require()
    d3 = _load(RES / "E4GD_D3_joint_representation_results.json")
    assert d3["network_input_dim"] == 9
    assert d3["other_region_actions_in_Q"] is False
    assert d3["centralized_critic_exists"] is False
    assert d3["JOINT_ACTION_REPRESENTATION"] == "IMPLICIT_STATE_ONLY"
    # circularity: autoregressive breaks simultaneity in the frozen option table
    assert d3["option_table"]["AUTOREGRESSIVE_JOINT_POLICY"]["simultaneous_execution"] is False


def test_T19_d4_reward_weights_unchanged():
    _require()
    d4 = _load(RES / "E4GD_D4_reward_scale_results.json")
    assert d4["reward_weights"] == {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01}


def test_T20_T21_T22_d4_components_and_reconstruction():
    _require()
    d4 = _load(RES / "E4GD_D4_reward_scale_results.json")
    assert d4["total_reconstruction_max_error"] < 1e-9
    # spot-check reconstruction from the CSV
    rows = list(csv.DictReader(open(RES / "E4GD_D4_reward_components.csv")))
    for r in rows[:200]:
        recon = (float(r["weighted_global_reward"]) + float(r["weighted_local_reward"])
                 - float(r["weighted_fe_penalty"]))
        assert abs(recon - float(r["total_reward"])) < 1e-9


def test_T23_T24_T25_d5_counterfactual_structure():
    _require()
    rows = list(csv.DictReader(open(RES / "E4GD_D5_region_counterfactuals.csv")))
    assert len(rows) > 0
    for r in rows[:50]:
        # all four conditional utilities present; regret = max_U - U_chosen
        us = {a: float(r[f"U_{a}"]) for a in ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]}
        best = max(us.values())
        chosen_u = us[r["chosen_action"]]
        assert abs(float(r["regret"]) - (best - chosen_u)) < 1e-9
        assert r["conditional_best_action"] == max(us, key=us.get)
    # events start from selected MIXED changes only
    d5 = _load(RES / "E4GD_D5_conditional_credit_results.json")
    assert d5["event_selection"]["scenario"] == "MIXED"
    assert set(d5["event_selection"]["changes"]).issubset({0, 4, 9})


def test_T26_d5_no_supervised_export():
    _require()
    d5 = _load(RES / "E4GD_D5_conditional_credit_results.json")
    assert d5["not_supervised_targets"] is True


def test_T27_T28_d6_validation_only():
    _require()
    d6 = _load(RES / "E4GD_D6_checkpoint_tension_results.json")
    assert "would have won E4F" in d6["forbidden_claim"]
    # checkpoint table only lists validation transition counts (10000..50000, final)
    rows = list(csv.DictReader(open(RES / "E4GD_D6_checkpoint_table.csv")))
    tc = {r["transition_count"] for r in rows}
    assert tc <= {"10000", "20000", "30000", "40000", "50000", "108000"}


def test_T29_d7_historical_logs_only():
    _require()
    rows = list(csv.DictReader(open(RES / "E4GD_D7_action_trajectory.csv")))
    epochs = {int(r["epoch"]) for r in rows}
    assert epochs <= {1, 2, 3, 4, 5, 6}


def test_T30_T31_d8_weights_unchanged_groups_exact():
    _require()
    # weights unchanged after autograd sensitivity
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]
    d8 = _load(RES / "E4GD_D8_feature_usage_results.json")
    assert d8["FEATURE_USAGE_PATTERN"] in ("LOCAL_DOMINANT", "GLOBAL_DOMINANT",
                                           "BALANCED", "POLICY_DEPENDENT")


def test_T32_T33_T34_T35_branch_logic_matches_frozen():
    _require()
    bd = _load(RES / "E4GD_branch_decision.json")
    d1 = _load(RES / "E4GD_D1_normalization_results.json")["D1_NORMALIZATION_SIGNAL"]
    d2 = _load(RES / "E4GD_D2_separability_results.json")["D2_NORMALIZATION_SEPARABILITY"]
    d4 = _load(RES / "E4GD_D4_reward_scale_results.json")["D4_REWARD_SCALE"]
    d5 = _load(RES / "E4GD_D5_conditional_credit_results.json")
    # A eligibility exact
    a = (d1 in ("COMPRESSED", "SEVERELY_COMPRESSED") and d2 in ("MODERATE_LOSS", "MATERIAL_LOSS"))
    assert (bd["BRANCH_A_ELIGIBLE"] == "YES") == a
    # B eligibility exact
    b = (d4 == "GLOBAL_OVERWHELMS_LOCAL" or d5["CONDITIONAL_CREDIT_STATUS"] == "MISALIGNED")
    assert (bd["BRANCH_B_ELIGIBLE"] == "YES") == b
    # C eligibility exact
    d3 = _load(RES / "E4GD_D3_joint_representation_results.json")["JOINT_ACTION_REPRESENTATION"]
    c = (d3 == "IMPLICIT_STATE_ONLY" and d5["D5_INTER_REGION_INTERACTION"] == "MATERIAL")
    # C requires the mechanism SUPPORTED which equals d3 implicit + interaction material
    table = _load(RES / "E4GD_failure_mechanism_table.json")
    c_sup = table["C_JOINT_ACTION_REPRESENTATION_LIMITATION"]["classification"] == "SUPPORTED"
    assert (bd["BRANCH_C_ELIGIBLE"] == "YES") == (c_sup and d5["D5_INTER_REGION_INTERACTION"] == "MATERIAL")


def test_T36_one_factor_at_a_time():
    _require()
    bd = _load(RES / "E4GD_branch_decision.json")
    assert bd["one_factor_at_a_time"] is True
    assert bd["SELECTED_BRANCH"] in ("A_STATE_NORMALIZATION", "B_REWARD_CREDIT",
                                     "C_JOINT_COORDINATION", "UNRESOLVED")
    # selection follows priority A<B<C among eligible
    if len(bd["eligible"]) >= 1:
        order = {"A": 0, "B": 1, "C": 2}
        best = sorted(bd["eligible"], key=lambda x: order[x])[0]
        assert bd["SELECTED_BRANCH"].startswith(best)


def test_T37_no_superiority_claim():
    _require()
    v = _load(RES / "E4GD_verdict.json")
    assert v["NEW_CONFIRMATORY_EVALUATION"] == "NO"
    assert v["NEW_TRAINING_RUN"] == "NO"


def test_T38_e4f_verdict_unchanged():
    _require()
    v = _load(RES / "E4GD_verdict.json")
    assert v["E4F_VERDICT_CHANGED"] == "NO"
    assert v["ORIGINAL_E4F_VERDICT"] == "FAILED"
    add = _load(RES / "E4GD_verdict_addendum.json")
    assert add["E4GD_CHANGES_E4F_VERDICT"] is False
    assert add["ANY_FUTURE_REDESIGN"] == "NEW_EXPERIMENT"


def test_next_stage_matches_branch():
    _require()
    v = _load(RES / "E4GD_verdict.json")
    m = {"A_STATE_NORMALIZATION": "E4H_STATE_NORMALIZATION_REVISION_DESIGN",
         "B_REWARD_CREDIT": "E4H_REWARD_CREDIT_REVISION_DESIGN",
         "C_JOINT_COORDINATION": "E4H_JOINT_COORDINATION_ARCHITECTURE_REVIEW",
         "UNRESOLVED": "E4H_DIAGNOSTIC_REVIEW"}
    assert v["NEXT_STAGE"] == m[v["SELECTED_BRANCH"]]


def test_policy_collapse_supported():
    _require()
    assert _load(RES / "E4GD_verdict.json")["POLICY_COLLAPSE"] == "SUPPORTED"


def test_evidence_table_all_six_mechanisms():
    _require()
    table = _load(RES / "E4GD_failure_mechanism_table.json")
    letters = {k.split("_")[0] for k in table}
    assert letters == set("ABCDEF")
    for v in table.values():
        assert v["classification"] in ("SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED", "NOT_AUDITABLE")
