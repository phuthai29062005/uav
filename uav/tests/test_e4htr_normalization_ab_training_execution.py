"""
E4HTR — Normalization A/B Training Execution: contract / integrity / artifact tests.

Do NOT retrain the 10800 episodes. Validate artifacts, single-factor integrity,
paired initialization, checkpoint completeness, selection, and firewalls. Never
assert a particular scientific outcome (no causal claim at this stage).
"""

import csv
import json
from pathlib import Path

import pytest

import prdrl.experiments.E4HTR_normalization_ab_training_execution as E4HTR
from prdrl.rl.normalization import NORM_V1, NORM_V3

RES = Path("results/E4HTR_normalization_ab_training_execution")
E4HT = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HR = Path("results/E4HR_normalization_design_review")
E4F = Path("results/E4F_final_evaluation")
E4E = Path("results/E4E_final_evaluation_protocol_freeze")


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
    if not (RES / "E4HTR_verdict.json").exists():
        pytest.skip("E4HTR run not present yet")


# ---- prerequisites / unit ----

def test_T1_e4ht_complete():
    assert _load(E4HT / "E4HT_verdict.json")["E4HT_COMPLETE"] == "YES"


def test_T2_v3_ready():
    assert _load(E4HT / "E4HT_verdict.json")["NORMALIZATION_V3_READY"] == "YES"


def test_T3_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T8_T9_T10_paired_init_unit():
    o1, t1 = E4HTR._paired_init_hash(3001)
    o1b, _ = E4HTR._paired_init_hash(3001)
    assert o1 == o1b and o1 == t1                 # deterministic, online==target
    o2, _ = E4HTR._paired_init_hash(3002)
    assert o2 != o1                               # different seed -> different init


def test_T11_T12_arms_exact():
    assert E4HTR.ARMS == [("V1_CONTROL", NORM_V1), ("V3_TREATMENT", NORM_V3)]


def test_T4_policy_seeds():
    assert E4HTR.POLICY_SEEDS == [3001, 3002, 3003, 3004, 3005]


# ---- artifact tests ----

def test_T5_T6_T7_runs_and_episodes():
    _require()
    v = _load(RES / "E4HTR_verdict.json")
    assert v["RUNS_COMPLETE"] == "10/10"
    assert v["V1_RUNS_COMPLETE"] == "5/5"
    assert v["V3_RUNS_COMPLETE"] == "5/5"
    assert v["TRAINING_EPISODES_COMPLETE"] == "10800/10800"


def test_T8b_paired_init_manifest():
    _require()
    d = _load(RES / "E4HTR_pair_initialization_manifest.json")
    assert d["all_pairs_pass"] is True
    for p in ["3001", "3002", "3003", "3004", "3005"]:
        assert d["per_policy"][p]["online_equals_target"] is True


def test_T13_T14_v3_constants_integrity():
    _require()
    v = _load(RES / "E4HTR_verdict.json")
    assert v["V3_CONSTANTS_INTEGRITY"] == "PASS"
    man = _load(RES / "E4HTR_pre_execution_manifest.json")
    assert man["v3_constants_sha256"] == _load(E4HR / "E4HR_v3_constants.json")["constants_sha256"]


def test_T15_T16_T17_no_arch_reward_change():
    _require()
    man = _load(RES / "E4HTR_pre_execution_manifest.json")
    assert man["network"] == "9-128-128-4"
    assert man["reward"] == {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01}
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in \
        Path("src/prdrl/rl/reward.py").read_text()


def test_T26_T27_T28_sixty_checkpoints():
    _require()
    d = _load(RES / "E4HTR_checkpoint_manifest.json")
    assert d["n_checkpoints"] == 60
    assert d["checkpoints_complete"] is True
    rows = list(csv.DictReader(open(RES / "E4HTR_checkpoint_manifest.csv")))
    assert len(rows) == 60
    # 6 checkpoints per run, 10 runs
    per_run = {}
    for r in rows:
        per_run.setdefault((r["arm"], r["policy_seed"]), 0)
        per_run[(r["arm"], r["policy_seed"])] += 1
    assert len(per_run) == 10 and all(c == 6 for c in per_run.values())


def test_T28b_checkpoint_metadata_version_guard():
    _require()
    import torch
    sel = _load(RES / "E4HTR_selected_checkpoints.json")["selected"]
    for key, info in sel.items():
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        if info["arm"] == "V1_CONTROL":
            assert payload["normalization_version"] == NORM_V1
        else:
            assert payload["normalization_version"] == NORM_V3
            assert payload["normalizer_constants_sha256"] == info["normalizer_constants_sha256"]


def test_T35_numerical_stability():
    _require()
    assert _load(RES / "E4HTR_numerical_stability.json")["all_runs_stable"] is True


def test_T36_T37_validation_all_checkpoints():
    _require()
    rows = list(csv.DictReader(open(RES / "E4HTR_validation_all_checkpoints.csv")))
    assert len(rows) == 60   # every checkpoint validated


def test_T43_T44_T45_selection_macro():
    _require()
    # selected checkpoint per run is the min macro among that run's checkpoints
    rows = list(csv.DictReader(open(RES / "E4HTR_validation_all_checkpoints.csv")))
    by_run = {}
    for r in rows:
        by_run.setdefault((r["arm"], r["policy_seed"]), []).append(r)
    for key, rs in by_run.items():
        selected = [r for r in rs if r["selected"] == "True"]
        assert len(selected) == 1
        min_macro = min(float(r["macro_MIGD_response"]) for r in rs)
        assert abs(float(selected[0]["macro_MIGD_response"]) - min_macro) <= 1e-12


def test_T46_T47_T48_ten_selected_all_retained():
    _require()
    d = _load(RES / "E4HTR_selected_checkpoints.json")
    assert d["all_ten_retained"] is True
    assert d["no_best_policy_selection"] is True
    v = _load(RES / "E4HTR_verdict.json")
    assert v["V1_SELECTED_CHECKPOINTS"] == "5/5"
    assert v["V3_SELECTED_CHECKPOINTS"] == "5/5"


def test_T50_validation_non_confirmatory():
    _require()
    v = _load(RES / "E4HTR_verdict.json")
    assert v["NORMALIZATION_CAUSAL_CLAIM"] == "NOT_MADE"
    assert v["FINAL_EVALUATION"] == "NOT_RUN"


def test_T51_e4f_unchanged():
    _require()
    assert _load(RES / "E4HTR_verdict.json")["ORIGINAL_E4F_VERDICT"] == "FAILED"
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T52_T53_final_firewall():
    _require()
    d = _load(RES / "E4HTR_final_firewall.json")
    assert d["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0
    assert d["unopened"] is True
    # runtime access counter is the authoritative firewall (a substring scan of the
    # validation CSV would false-positive on final-seed digits inside MIGD floats;
    # validation used only val seeds 1361..1380, which has no environment_seed column).
    v = _load(RES / "E4HTR_verdict.json")
    assert v["NEW_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T49_T54_no_retuning_no_final_verdict():
    _require()
    v = _load(RES / "E4HTR_verdict.json")
    assert v["HYPERPARAMETER_RETUNING"] == "NO"
    assert v["CHECKPOINT_SELECTION_CHANGED"] == "NO"
    assert v["BEST_POLICY_SELECTION"] == "NO"


def test_T55_T56_selected_hashes_and_immutable():
    _require()
    d = _load(RES / "E4HTR_selected_checkpoints.json")
    assert d["post_selection_immutable"] is True
    for key, info in d["selected"].items():
        assert Path(info["checkpoint_path"]).exists()
        assert info["sha256"] == _hash(info["checkpoint_path"])
        assert info["immutable_for_final_evaluation"] is True


def test_old_checkpoints_unchanged():
    _require()
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in ["2001", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_mechanism_summary_present():
    _require()
    rows = list(csv.DictReader(open(RES / "E4HTR_mechanism_validation_summary.csv")))
    arms = {r["arm"] for r in rows}
    assert arms == {"V1_CONTROL", "V3_TREATMENT"}
    assert len(rows) == 10   # 5 policies x 2 arms, MIXED
    for r in rows:
        for k in ["PREDICT_fraction", "within_event_multi_action_rate", "regional_action_entropy"]:
            import math
            assert math.isfinite(float(r[k]))


def test_ab_summary_scenarios():
    _require()
    rows = list(csv.DictReader(open(RES / "E4HTR_validation_ab_summary.csv")))
    scen = {r["scenario"] for r in rows}
    assert scen == {"PERSISTENT", "STATIONARY", "MIXED", "macro"}


def test_verdict_complete_next_stage():
    _require()
    v = _load(RES / "E4HTR_verdict.json")
    if v["E4HTR_COMPLETE"] == "YES":
        assert v["NEXT_STAGE"] == "E4HE_NORMALIZATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
    assert v["VALIDATION_MECHANISM_DIRECTION"] in (
        "DECOLLAPSE_DIRECTION", "NO_CLEAR_DECOLLAPSE", "OPPOSITE_DIRECTION")
