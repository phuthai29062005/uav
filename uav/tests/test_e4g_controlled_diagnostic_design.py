"""
E4G — Controlled Diagnostic Design: contract-freeze tests.

Verify the diagnostic contracts are frozen correctly and that E4G changed
nothing (no training, no weights, no reward, no final-seed reuse, verdict intact).
"""

import json
from pathlib import Path

import pytest

RES = Path("results/E4G_controlled_diagnostic_design")
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
    if not (RES / "E4G_verdict.json").exists():
        pytest.skip("E4G not run yet")


def test_T1_original_verdict_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T2_e4f_artifacts_unchanged():
    _require()
    man = _load(RES / "E4G_pre_manifest.json")
    assert man["e4f_verdict_hash"] == _hash(E4F / "E4F_verdict.json")
    assert man["e4f_episode_results_hash"] == _hash(E4F / "E4F_episode_results.csv")


def test_T3_no_training():
    _require()
    v = _load(RES / "E4G_verdict.json")
    assert v["NEW_TRAINING"] == "NO"
    assert _load(RES / "E4G_pre_manifest.json")["no_training"] is True


def test_T4_no_network_weights_changed():
    _require()
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in ["2001", "2002", "2003", "2004", "2005"]:
        info = frz["policies"][ps]
        assert _hash(info["checkpoint_path"]) == info["sha256"]
    assert _load(RES / "E4G_verdict.json")["MODEL_MODIFICATION"] == "NO"


def test_T5_no_final_seed_reuse():
    _require()
    v = _load(RES / "E4G_verdict.json")
    assert v["FINAL_SEED_REUSE"] == "NO"
    sc = _load(RES / "E4G_diagnostic_seed_contract.json")
    final = set(range(1081, 1141))
    assert set(sc["diagnostic_seeds"]).isdisjoint(final)


def test_T6_diagnostic_seeds_exact():
    _require()
    sc = _load(RES / "E4G_diagnostic_seed_contract.json")
    assert sc["diagnostic_seeds"] == list(range(1201, 1221))
    assert sc["n"] == 20


def test_T7_diagnostic_seeds_post_hoc_labelled():
    _require()
    sc = _load(RES / "E4G_diagnostic_seed_contract.json")
    assert sc["label"] == "POST_HOC_DIAGNOSTIC_ONLY"
    assert sc["never_use_as_confirmatory_test_seeds"] is True
    assert sc["disjoint_from_consumed_prdrl_seeds"] is True
    assert sc["overlaps"] == {}


def test_T8_d1_frozen():
    _require()
    d = _load(RES / "E4G_D1_normalization_contract.json")
    assert d["diagnostic"] == "D1_STATE_NORMALIZATION"
    assert "SEVERELY_COMPRESSED" in d["classification_rule"]
    assert d["run_environment"] is False


def test_T9_d2_frozen():
    _require()
    d = _load(RES / "E4G_D2_separability_contract.json")
    assert d["no_neural_retraining"] is True
    assert "MATERIAL_LOSS" in d["classification_rule"]
    assert d["not_a_controller"] is True


def test_T10_d3_options_explicit():
    _require()
    d = _load(RES / "E4G_D3_joint_representation_contract.json")
    opts = d["design_options"]
    for name in ["CURRENT_SHARED_Q", "CTDE_CENTRALIZED_CRITIC", "VALUE_DECOMPOSITION",
                 "AUTOREGRESSIVE_JOINT_POLICY"]:
        assert name in opts
        assert "decision_time" in opts[name]
        assert "training_time" in opts[name]
        assert "simultaneous_execution" in opts[name]
        assert "decentralized_selection" in opts[name]
    assert d["winner_selected"] is False


def test_T11_simultaneous_circularity_addressed():
    _require()
    d = _load(RES / "E4G_D3_joint_representation_contract.json")
    assert d["circularity_addressed"] is True
    j1 = d["candidate_representations"]["J1_LOCAL_PLUS_OTHER_ACTION_SUMMARY"]
    assert "circularity_warning" in j1
    # autoregressive breaks simultaneity
    assert d["design_options"]["AUTOREGRESSIVE_JOINT_POLICY"]["simultaneous_execution"] is False


def test_T12_d4_logs_raw_and_weighted():
    _require()
    d = _load(RES / "E4G_D4_reward_logging_contract.json")
    logs = d["log_per_regional_transition"]
    for f in ["raw_global_reward", "weighted_global_reward", "raw_local_reward",
              "weighted_local_reward", "raw_fe_cost", "weighted_fe_penalty"]:
        assert f in logs


def test_T13_no_reward_coefficient_change():
    _require()
    d = _load(RES / "E4G_D4_reward_logging_contract.json")
    assert d["reward_weights_frozen"] == {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01}
    assert "reward_weights" in d["must_not_change"]
    # reward.py formula unchanged
    src = Path("src/prdrl/rl/reward.py").read_text()
    assert "alpha_g * global_reward + alpha_l * local_pbi_reward - beta * fe_cost" in src


def test_T14_d5_conditional_counterfactual_exact():
    _require()
    d = _load(RES / "E4G_D5_conditional_credit_contract.json")
    assert d["chosen_action_regret"] == \
        "max_a U_conditional(r,a|a_-r) - U_conditional(r,a_chosen|a_-r)"
    assert "MISALIGNED" in d["classification_rule"]


def test_T15_d5_no_supervised_targets():
    _require()
    d = _load(RES / "E4G_D5_conditional_credit_contract.json")
    assert d["not_training"] is True
    assert d["not_a_supervision_oracle"] is True
    assert d["not_a_new_reward"] is True


def test_T16_d6_validation_only():
    _require()
    d = _load(RES / "E4G_D6_checkpoint_tension_contract.json")
    assert d["validation_only"] is True
    assert "final seeds" in d["forbidden"]


def test_T17_no_final_checkpoint_reselection():
    _require()
    d = _load(RES / "E4G_D6_checkpoint_tension_contract.json")
    assert "would have won the final" in d["must_never_claim"]


def test_T18_d7_historical_logs_only():
    _require()
    d = _load(RES / "E4G_D7_collapse_timeline_contract.json")
    assert d["run_environment"] is False
    assert "training_diagnostics.csv" in d["data_source"]


def test_T19_d8_frozen_weights_only():
    _require()
    d = _load(RES / "E4G_D8_feature_usage_contract.json")
    assert d["frozen_weights_only"] is True
    assert d["no_weight_modification"] is True
    assert d["run_environment"] is False


def test_T20_branch_eligibility_exact():
    _require()
    b = _load(RES / "E4G_branch_logic.json")["branches"]
    assert "COMPRESSED" in b["BRANCH_A_STATE_NORMALIZATION_REVISION"]["eligible_iff"]
    assert "MISALIGNED" in b["BRANCH_B_REWARD_CREDIT_REVISION"]["eligible_iff"]
    assert "other-region actions" in b["BRANCH_C_JOINT_COORDINATION_ARCHITECTURE_REVISION"]["eligible_iff"]


def test_T21_one_factor_at_a_time_frozen():
    _require()
    b = _load(RES / "E4G_branch_logic.json")
    assert b["one_factor_at_a_time"] is True
    assert set(b["forbidden_simultaneous_change"]) == {"state", "reward", "architecture", "optimizer"}
    assert b["branch_selected_in_E4G"] is False


def test_T22_no_hyperparameter_search():
    _require()
    b = _load(RES / "E4G_branch_logic.json")
    assert b["no_hyperparameter_search"] is True


def test_no_placeholder_tokens_and_complete():
    _require()
    v = _load(RES / "E4G_verdict.json")
    assert v["E4G_COMPLETE"] == "YES"
    assert v["no_placeholder_tokens"] is True
    assert v["placeholder_hits"] == []
    assert v["NEXT_STAGE"] == "E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION"


def test_all_eight_diagnostics_frozen():
    _require()
    v = _load(RES / "E4G_verdict.json")
    for k in ["D1_NORMALIZATION_DIAGNOSTIC", "D2_SEPARABILITY_DIAGNOSTIC",
              "D3_JOINT_COORDINATION_DIAGNOSTIC", "D4_REWARD_SCALE_DIAGNOSTIC",
              "D5_CONDITIONAL_CREDIT_DIAGNOSTIC", "D6_CHECKPOINT_TENSION_DIAGNOSTIC",
              "D7_COLLAPSE_TIMELINE_DIAGNOSTIC", "D8_FEATURE_USAGE_DIAGNOSTIC"]:
        assert v[k] == "FROZEN"
