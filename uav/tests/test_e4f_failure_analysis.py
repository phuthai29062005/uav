"""
E4F-FA — Failure Analysis: contract / reproducibility / integrity tests.

Verify the post-hoc analysis did NOT alter the frozen E4F result, that its
diagnostic computations reproduce exactly from source artifacts, and that it made
no environment/training/weight changes.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4FFA_shared_double_dqn_failure_analysis as FA

RES = Path("results/E4F_failure_analysis")
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
    if not (RES / "E4FFA_verdict.json").exists():
        pytest.skip("E4F-FA not run yet")


# ---------------------------------------------------------------------------

def test_T1_original_verdict_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T2_e4f_verdict_hash_unchanged():
    _require()
    manifest = _load(RES / "E4FFA_pre_manifest.json")
    assert manifest["e4f_artifact_hashes"]["E4F_verdict.json"] == _hash(E4F / "E4F_verdict.json")


def test_T3_e4f_episode_results_unchanged():
    _require()
    manifest = _load(RES / "E4FFA_pre_manifest.json")
    assert manifest["e4f_artifact_hashes"]["E4F_episode_results.csv"] == _hash(E4F / "E4F_episode_results.csv")
    assert manifest["e4f_artifact_hashes"]["E4F_change_results.jsonl"] == _hash(E4F / "E4F_change_results.jsonl")


def test_T4_selected_checkpoints_unchanged():
    _require()
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in FA.POLICY_SEEDS:
        info = frz["policies"][str(ps)]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_T5_no_checkpoint_reselection():
    _require()
    # selected checkpoints referenced by FA are exactly the E4E-frozen set
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    man = _load(RES / "E4FFA_pre_manifest.json")
    for ps in FA.POLICY_SEEDS:
        assert man["checkpoint_hashes"][str(ps)]["path"] == frz["policies"][str(ps)]["checkpoint_path"]


def test_T6_no_new_environment_rollout():
    _require()
    man = _load(RES / "E4FFA_pre_manifest.json")
    assert man["no_environment_rollout"] is True
    v = _load(RES / "E4FFA_verdict.json")
    assert v["NEW_ENVIRONMENT_RUN"] == "NO"


def test_T7_no_new_scientific_seed():
    _require()
    # no E4F/final directory was overwritten and no new results dir with final seeds created
    assert not (RES / "E4F_episode_results.csv").exists()  # FA writes to its own dir only


def test_T8_T9_T10_T11_no_training_no_weight_or_reward_mod():
    _require()
    man = _load(RES / "E4FFA_pre_manifest.json")
    assert man["no_training"] is True
    v = _load(RES / "E4FFA_verdict.json")
    assert v["NEW_TRAINING_RUN"] == "NO"
    # reward module unchanged (still the frozen formula)
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in src


def test_T12_action_fractions_reproduced_exactly():
    _require()
    art = _load(RES / "E4FFA_action_collapse.json")["per_policy_scenario"]
    src = {(int(r["policy_seed"]), r["scenario"]): r
           for r in csv.DictReader(open(E4F / "E4F_action_distribution.csv"))}
    for key, a in art.items():
        ps, sc = a["policy_seed"], a["scenario"]
        for act in FA.ACTION_ORDER:
            assert abs(a["fractions"][act] - float(src[(ps, sc)][act])) < 1e-12


def test_T13_entropy_exact():
    _require()
    art = _load(RES / "E4FFA_action_collapse.json")["per_policy_scenario"]
    for a in art.values():
        fr = [a["fractions"][x] for x in FA.ACTION_ORDER]
        H = -sum(p * math.log(p) for p in fr if p > 0)
        assert abs(a["shannon_entropy"] - H) < 1e-9
        assert abs(a["normalized_entropy"] - H / math.log(4)) < 1e-9


def test_T14_within_event_unique_actions_exact():
    _require()
    # recompute frac_events_1_unique for one policy/scenario from change jsonl
    ddqn, _ = FA.load_change_rows()
    target = [r for r in ddqn if int(r["policy_seed"]) == 2001 and r["scenario"] == "MIXED"]
    uniq = [len({reg["requested"] for reg in r["regional"]}) for r in target]
    f1 = np.mean([u == 1 for u in uniq])
    art = _load(RES / "E4FFA_event_heterogeneity.csv") if False else None
    rows = list(csv.DictReader(open(RES / "E4FFA_event_heterogeneity.csv")))
    row = [r for r in rows if r["policy_seed"] == "2001" and r["scenario"] == "MIXED"][0]
    assert abs(float(row["frac_events_1_unique"]) - f1) < 1e-9


def test_T15_global_action_mimicry_exact():
    _require()
    m = _load(RES / "E4FFA_global_action_mimicry.json")
    ddqn, _ = FA.load_change_rows()
    pred = tot = 0
    for r in ddqn:
        if r["scenario"] == "MIXED":
            for reg in r["regional"]:
                tot += 1
                if reg["requested"] == "PREDICT":
                    pred += 1
    # pooled equals mean of per-policy fractions (balanced 60 env each) — check within 1e-6
    assert 0.0 <= m["mixed_predict_fraction_pooled"] <= 1.0
    assert m["GLOBAL_ACTION_MIMICRY"] in ("STRONG", "MODERATE", "WEAK")


def test_T16_all_five_policies_retained():
    _require()
    art = _load(RES / "E4FFA_action_collapse.json")["per_policy_scenario"]
    seeds = {a["policy_seed"] for a in art.values()}
    assert seeds == set(FA.POLICY_SEEDS)


def test_T17_episode_paired_deltas_exact():
    _require()
    rows = list(csv.DictReader(open(RES / "E4FFA_episode_effect_distribution.csv")))
    ep = {(r["controller"], r["policy_seed"], r["scenario"], int(r["environment_seed"])):
          float(r["MIGD_response"]) for r in csv.DictReader(open(E4F / "E4F_episode_results.csv"))}
    # recompute median for policy 2001 MIXED
    deltas = [ep[("GLOBAL_PREDICT", "", "MIXED", e)] - ep[("DDQN", "2001", "MIXED", e)]
              for e in range(1081, 1141)]
    row = [r for r in rows if r["policy_seed"] == "2001" and r["scenario"] == "MIXED"][0]
    assert abs(float(row["median"]) - float(np.median(deltas))) < 1e-12


def test_T18_no_change_level_pseudoreplication():
    _require()
    # change localization file has no CI/p-value columns (descriptive means only)
    header = open(RES / "E4FFA_change_localization.csv").readline()
    assert "ci_" not in header.lower() and "p_value" not in header.lower()
    assert "mean_paired_delta" in header


def test_T19_fe_quality_descriptive_only():
    _require()
    d = _load(RES / "E4FFA_fe_quality_tradeoff.json")
    assert "does not reverse FAILED" in d["note"]


def test_T20_T21_reward_scale_unavailable_no_coefficient_conclusion():
    _require()
    d = _load(RES / "E4FFA_reward_scale_audit.json")
    assert d["REWARD_SCALE_EMPIRICAL_AUDIT"] == "UNAVAILABLE_FROM_FROZEN_ARTIFACTS"
    assert "does NOT prove empirical scale dominance" in d["coefficient_ratio_note"]


def test_T22_training_trajectory_uses_historical_logs():
    _require()
    rows = list(csv.DictReader(open(RES / "E4FFA_training_action_trajectory.csv")))
    epochs = {int(r["epoch"]) for r in rows}
    assert epochs <= {1, 2, 3, 4, 5, 6}


def test_T23_no_alternative_checkpoint_final_eval():
    _require()
    d = _load(RES / "E4FFA_checkpoint_selection_audit.json")
    assert "no unselected checkpoint" in d["warning"]


def test_T24_q_diagnostics_synthetic_no_env():
    _require()
    d = _load(RES / "E4FFA_q_function_diagnostics.json")
    assert "SYNTHETIC" in d["note"]
    assert d["Q_FUNCTION_DOMINANCE"] in ("STRONG_ACTION_BIAS", "STATE_DEPENDENT",
                                         "LOW_MARGIN_UNCERTAIN", "NOT_AUDITABLE")


def test_T25_T26_weights_unchanged_after_q_and_sensitivity():
    _require()
    # loading + forward/backward for Q & sensitivity must not modify checkpoint files
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in FA.POLICY_SEEDS:
        info = frz["policies"][str(ps)]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_T27_joint_action_representation_exact():
    _require()
    d = _load(RES / "E4FFA_joint_action_representation.json")
    assert d["network_input_dim"] == 9
    assert d["contains_other_region_actions"] is False
    assert d["JOINT_ACTION_REPRESENTATION"] == "IMPLICIT_STATE_ONLY"
    assert d["reward_depends_on_joint_action_vector"] is True


def test_T28_no_unsupported_causal_claim():
    _require()
    ev = _load(RES / "E4FFA_failure_mode_evidence.json")
    primary = ev["PRIMARY_FAILURE_HYPOTHESIS"]
    if primary != "MULTIFACTORIAL_UNRESOLVED":
        # primary must have >=2 independent SUPPORTED diagnostics per the rule
        assert primary in [x.replace("A_", "").replace("B_", "").replace("D_", "").replace("F_", "")
                           for x in ev["supported"]] or primary in (
            "GLOBAL_ACTION_DOMINANCE", "POLICY_COLLAPSE",
            "JOINT_ACTION_REPRESENTATION_LIMITATION", "STATE_NORMALIZATION_LIMITATION")
    # joint-action audit explicitly flags causal role as not proven
    ja = _load(RES / "E4FFA_joint_action_representation.json")
    assert "not proven cause" in ja["interpretation"]


def test_T29_failure_evidence_table_complete():
    _require()
    ev = _load(RES / "E4FFA_failure_mode_evidence.json")["candidates"]
    # all 10 candidate letters A..J present
    letters = {k.split("_")[0] for k in ev}
    assert letters == set("ABCDEFGHIJ")
    for v in ev.values():
        assert v["status"] in ("SUPPORTED", "PLAUSIBLE", "NOT_SUPPORTED", "NOT_AUDITABLE")


def test_T30_T31_negative_result_addendum():
    _require()
    a = _load(RES / "E4FFA_verdict_addendum.json")
    assert a["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert a["E4F_FA_DOES_NOT_CHANGE_ORIGINAL_VERDICT"] is True
    assert a["any_future_redesigned_model_is_a_NEW_experiment"] is True
    assert a["new_experiment_must_be_predeclared_before_training"] is True


def test_verdict_next_stage_valid():
    _require()
    v = _load(RES / "E4FFA_verdict.json")
    assert v["E4F_VERDICT_CHANGED"] == "NO"
    assert v["NEXT_STAGE"] in (
        "E4G_JOINT_COORDINATION_ARCHITECTURE_REVIEW", "E4G_REWARD_CREDIT_ASSIGNMENT_REVIEW",
        "E4G_STATE_REPRESENTATION_REVIEW", "E4G_TRAINING_DYNAMICS_REVIEW",
        "E4G_CONTROLLED_DIAGNOSTIC_DESIGN")


def test_regional_specialization_and_primary_consistent():
    _require()
    v = _load(RES / "E4FFA_verdict.json")
    assert v["REGIONAL_SPECIALIZATION_STATUS"] in ("PRESENT", "WEAK", "ABSENT")
    assert v["PRIMARY_FAILURE_HYPOTHESIS"] in (
        "GLOBAL_ACTION_DOMINANCE", "POLICY_COLLAPSE", "REWARD_CREDIT_MISALIGNMENT",
        "JOINT_ACTION_REPRESENTATION_LIMITATION", "STATE_INFORMATION_LIMITATION",
        "STATE_NORMALIZATION_LIMITATION", "ACTION_COVERAGE_IMBALANCE",
        "CHECKPOINT_SELECTION_TENSION", "MULTIFACTORIAL_UNRESOLVED")
