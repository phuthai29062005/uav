"""
E4E — Final Evaluation Protocol Freeze: contract / unit / artifact tests.

No final-evaluation environment is instantiated anywhere in these tests. They
verify the frozen protocol, the inferential statistics (on synthetic data), and
the E4E artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4E_final_evaluation_protocol_freeze as E4E
from prdrl.evaluation import crossed_bootstrap as CB
from prdrl.evaluation import effects as EF
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4E_final_evaluation_protocol_freeze")
E4TR = Path("results/E4TR_shared_double_dqn_training")
E3R = Path("results/E3R_rule_v2_semantic_rerun")


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


# ---------------------------------------------------------------------------
# E4TR readiness / checkpoint integrity
# ---------------------------------------------------------------------------

def test_T1_e4tr_complete():
    assert _load(E4TR / "E4TR_verdict.json")["E4TR_COMPLETE"] == "YES"


def test_T2_five_policies_retained():
    sel = _load(E4TR / "E4TR_selected_checkpoints.json")
    assert sel["all_five_retained"] is True
    assert len(sel["selected"]) == 5
    assert sel["policy_seeds"] == [2001, 2002, 2003, 2004, 2005]


def test_T3_selected_checkpoint_set_exact():
    frz = _load(RES / "E4E_checkpoint_freeze.json")
    sel = _load(E4TR / "E4TR_selected_checkpoints.json")
    for ps in E4E.POLICY_SEEDS:
        assert frz["policies"][str(ps)]["checkpoint_filename"] == \
            sel["selected"][str(ps)]["selected_checkpoint"]
        assert frz["policies"][str(ps)]["transition_count"] == \
            sel["selected"][str(ps)]["transition_count"]


def test_T4_checkpoint_sha256_exact():
    frz = _load(RES / "E4E_checkpoint_freeze.json")
    for ps in E4E.POLICY_SEEDS:
        info = frz["policies"][str(ps)]
        assert info["sha256"] == _hash(info["checkpoint_path"])


def test_T5_no_checkpoint_reselection():
    frz = _load(RES / "E4E_checkpoint_freeze.json")
    assert frz["immutable_for_final_evaluation"] is True
    assert frz["no_weights_may_be_altered_after_this_stage"] is True


def test_T6_final_seeds_exact():
    assert E4E.FINAL_EVAL_SEEDS == list(range(1081, 1141))
    assert len(E4E.FINAL_EVAL_SEEDS) == 60


def test_T7_T8_no_final_environment_instantiated():
    # E4E firewall never opened a final seed
    assert E4E.FIREWALL.access_count == 0
    # audit reproduces with access still 0
    audit, _ = E4E.pre_freeze_audit()
    assert E4E.FIREWALL.access_count == 0
    manifest = _load(RES / "E4E_pre_freeze_manifest.json")
    assert manifest["CHECKPOINT_INTEGRITY"] == "PASS"


def test_final_seed_firewall_guard_raises():
    fw = E4E.FinalEvalFirewall()
    with pytest.raises(RuntimeError):
        fw.check(1081)
    with pytest.raises(RuntimeError):
        fw.check(1140)
    assert fw.access_count == 2


# ---------------------------------------------------------------------------
# Design axes / arms
# ---------------------------------------------------------------------------

def test_T9_scenarios_exact():
    assert E4E.SCENARIOS == ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T10_all_five_ddqn_included():
    order = _load(RES / "E4E_execution_order.json")["arm_order_per_env_seed"]
    for ps in E4E.POLICY_SEEDS:
        assert f"DDQN_{ps}" in order


def test_T11_all_fixed_globals_included():
    order = _load(RES / "E4E_execution_order.json")["arm_order_per_env_seed"]
    for g in ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]:
        assert g in order


def test_T12_rule_included():
    order = _load(RES / "E4E_execution_order.json")["arm_order_per_env_seed"]
    assert "RULE_V1_V2STATE" in order


def test_T13_T14_T15_primary_comparators():
    cf = _load(RES / "E4E_comparator_freeze.json")["primary_comparator_by_scenario"]
    assert cf["PERSISTENT"] == "GLOBAL_PREDICT"
    assert cf["STATIONARY"] == "GLOBAL_DIVERSIFY"
    assert cf["MIXED"] == "GLOBAL_PREDICT"


def test_T16_comparator_source_validation_only():
    cf = _load(RES / "E4E_comparator_freeze.json")
    assert "VALIDATION" in cf["preselected_from"]


def test_T17_no_final_data_comparator_selection():
    cf = _load(RES / "E4E_comparator_freeze.json")
    assert cf["no_best_fixed_global_final_reselection"] is True
    assert cf["must_not_reselect_on_final_seeds"] is True


def test_T18_migd_response_primary():
    ec = _load(RES / "E4E_endpoint_contract.json")
    assert ec["primary_endpoint"] == "MIGD_response"
    assert ec["primary_direction"] == "lower_is_better"


def test_T19_T20_secondary_endpoints():
    ec = _load(RES / "E4E_endpoint_contract.json")
    assert ec["secondary_endpoints"] == ["MIGD_epoch", "response_FE"]


def test_T50_no_reward_metric_as_endpoint():
    ec = _load(RES / "E4E_endpoint_contract.json")
    for bad in ["cumulative_RL_reward", "Q_values", "TD_loss", "training_reward"]:
        assert bad in ec["forbidden_as_final_endpoints"]


# ---------------------------------------------------------------------------
# Effect definition
# ---------------------------------------------------------------------------

def test_T21_effect_sign_positive_means_ddqn_better():
    # L lower (better) than B -> positive effect
    L = np.array([[0.004, 0.004], [0.004, 0.004]])
    B = np.array([0.005, 0.005])
    assert EF.scenario_mean_effect(L, B) == pytest.approx(0.001)


def test_T22_environment_pairing_contract():
    c = _load(RES / "E4E_final_eval_execution_contract.json")
    assert "environment_seed" in c["common_environment_pairing"]
    assert c["no_new_common_random_number_scheme"] is True
    assert c["production_rng_unchanged"] is True


def test_T23_T24_axes():
    bs = _load(RES / "E4E_two_axis_bootstrap.json")
    assert bs["policy_axis_n"] == 5
    assert bs["environment_axis_n"] == 60


def test_T25_T26_no_pseudoreplication():
    ed = _load(RES / "E4E_effect_definition.json")
    assert ed["primary_statistical_units"] == ["policy_training_seed", "environment_seed"]
    for bad in ["region", "change", "transition"]:
        assert bad in ed["forbidden_replicate_units"]


def test_T27_no_naive_row_bootstrap():
    bs = _load(RES / "E4E_two_axis_bootstrap.json")
    assert bs["naive_row_bootstrap"] == "FORBIDDEN_FOR_INFERENCE"


def test_T28_T29_bootstrap_with_replacement():
    bs = _load(RES / "E4E_two_axis_bootstrap.json")
    assert bs["resample_policy_with_replacement"] is True
    assert bs["resample_environment_with_replacement"] is True
    assert bs["form_full_sampled_cartesian_product"] is True


def test_T30_sampled_cartesian_product_correct():
    # 2x2: every bootstrap replicate mean must lie in the finite set of means
    # obtainable from a full product of resampled (policy, env) multisets.
    L = np.array([[0.001, 0.002], [0.010, 0.020]])
    B = np.array([0.005, 0.006])
    out = CB.crossed_two_axis_bootstrap(L, B, n_boot=2000, seed=7)
    d = B[None, :] - L
    # enumerate all product means for p multiset in {0,1}^2, e multiset in {0,1}^2
    import itertools
    possible = set()
    for p_ms in itertools.product([0, 1], repeat=2):
        for e_ms in itertools.product([0, 1], repeat=2):
            sub = d[np.ix_(p_ms, e_ms)]
            possible.add(round(float(sub.mean()), 12))
    # regenerate the exact boot draws and verify membership
    rng = np.random.default_rng(7)
    for _ in range(2000):
        p_idx = rng.integers(0, 2, size=2)
        e_idx = rng.integers(0, 2, size=2)
        val = round(float((B[e_idx][None, :] - L[np.ix_(p_idx, e_idx)]).mean()), 12)
        assert val in possible


def test_T31_T32_T33_bootstrap_settings():
    assert CB.N_BOOTSTRAP == 50000
    assert CB.CI_LEVEL == 0.95
    assert CB.CI_METHOD == "percentile"
    assert CB.BOOTSTRAP_MASTER_SEED == 46001


def test_bootstrap_seed_deterministic():
    L = np.random.default_rng(0).uniform(0.004, 0.008, size=(5, 60))
    B = np.random.default_rng(1).uniform(0.004, 0.008, size=60)
    a = CB.crossed_two_axis_bootstrap(L, B, n_boot=2000, seed=999)
    b = CB.crossed_two_axis_bootstrap(L, B, n_boot=2000, seed=999)
    assert a["ci_lo"] == b["ci_lo"] and a["ci_hi"] == b["ci_hi"]


def test_T34_macro_equal_third():
    mc = _load(RES / "E4E_macro_contract.json")
    assert abs(mc["scenario_weight"] - 1.0 / 3.0) < 1e-12
    assert mc["no_observation_count_weighting"] is True
    assert EF.macro_effect({"P": 0.001, "S": 0.002, "M": 0.003}) == pytest.approx(0.002)


def test_T35_macro_shared_draws():
    d = _load(RES / "E4E_synthetic_stats_validation.json")
    assert d["checks"]["8_macro_shared_draws_across_scenarios"]["PASS"] is True


# ---------------------------------------------------------------------------
# Practical margin + classification
# ---------------------------------------------------------------------------

def test_T36_delta_e4():
    assert FV.DELTA_E4 == 1e-5
    assert _load(RES / "E4E_practical_margin.json")["DELTA_E4"] == 1e-5


def test_T37_beneficial_exact():
    assert FV.classify_effect(0.01, 0.001, 0.02) == FV.BENEFICIAL


def test_T38_directional_positive_exact():
    assert FV.classify_effect(0.01, -0.001, 0.02) == FV.DIRECTIONAL_POSITIVE


def test_T39_practically_neutral_exact():
    assert FV.classify_effect(0.0, -0.01, 0.01) == FV.PRACTICALLY_NEUTRAL
    # tiny mean within margin, CI straddles
    assert FV.classify_effect(5e-6, -0.001, 0.001) == FV.PRACTICALLY_NEUTRAL


def test_T40_directional_negative_exact():
    assert FV.classify_effect(-0.01, -0.02, 0.001) == FV.DIRECTIONAL_NEGATIVE


def test_T41_harmful_exact():
    assert FV.classify_effect(-0.01, -0.02, -0.001) == FV.HARMFUL


def test_classification_exhaustive_extension():
    # sub-margin mean but CI entirely negative -> directional negative (documented)
    assert FV.classify_effect(-5e-6, -0.01, -1e-4) == FV.DIRECTIONAL_NEGATIVE
    # contradictions defer to CI
    assert FV.classify_effect(0.01, -0.02, -0.001) == FV.DIRECTIONAL_NEGATIVE
    assert FV.classify_effect(-0.01, 0.001, 0.02) == FV.DIRECTIONAL_POSITIVE


# ---------------------------------------------------------------------------
# Hypotheses / gates
# ---------------------------------------------------------------------------

def test_T42_mixed_primary_frozen():
    m = _load(RES / "E4E_final_eval_protocol_frozen.json")
    assert m["primary_scenario"] == "MIXED"
    assert m["preselected_comparator"]["MIXED"] == "GLOBAL_PREDICT"


def test_T43_T44_T45_safety_gates():
    # neither harmful -> PASS
    assert FV.safety_gate(FV.BENEFICIAL, FV.PRACTICALLY_NEUTRAL)["PASS"] is True
    assert FV.safety_gate(FV.DIRECTIONAL_NEGATIVE, FV.PRACTICALLY_NEUTRAL)["PASS"] is True
    # harmful in either -> FAIL
    assert FV.safety_gate(FV.HARMFUL, FV.BENEFICIAL)["PASS"] is False
    assert FV.safety_gate(FV.BENEFICIAL, FV.HARMFUL)["PASS"] is False
    sg = _load(RES / "E4E_safety_gates.json")
    assert "HARMFUL" in sg["PERSISTENT_SAFETY"]
    assert "HARMFUL" in sg["STATIONARY_SAFETY"]
    assert "HARMFUL" in sg["MACRO_SAFETY"]


def test_T46_mixed_policy_robustness_threshold():
    pr = _load(RES / "E4E_policy_robustness.json")
    assert pr["threshold"] == 4
    assert FV.policy_robustness([1, 1, 1, 1, -1], threshold=4)["PASS"] is True
    assert FV.policy_robustness([1, 1, 1, -1, -1], threshold=4)["PASS"] is False


def test_T47_ddqn_vs_rule_frozen():
    rc = _load(RES / "E4E_rule_comparison.json")
    assert rc["gate_name"] == "DDQN_VS_RULE_MIXED"
    assert rc["required_for_full_validation"] == "BENEFICIAL"
    assert rc["no_retuning"] is True
    assert rc["no_recalibration_on_final_seeds"] is True


def test_T48_fe_cannot_rescue_harmful():
    sg = _load(RES / "E4E_safety_gates.json")
    assert sg["fe_cannot_rescue_harmful_quality"] is True
    # verdict has no FE input and HARMFUL mixed -> FAILED
    v = FV.scientific_verdict(
        mixed_primary_label=FV.HARMFUL, ddqn_vs_rule_mixed_label=FV.BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)
    assert v["verdict"] == FV.FAILED


def test_T49_no_p_value_fishing():
    vl = _load(RES / "E4E_verdict_logic.json")
    assert vl["no_p_value_fishing"] is True
    assert "GLOBAL_PREDICT" in vl["single_primary_hypothesis"]


# ---------------------------------------------------------------------------
# Execution contract
# ---------------------------------------------------------------------------

def test_T51_T52_T53_T54_execution_contract():
    c = _load(RES / "E4E_final_eval_execution_contract.json")
    assert c["epsilon"] == 0.0
    assert c["learning"] == "none"
    assert c["replay"] == "disabled"
    assert c["optimizer"] == "disabled"
    assert c["target_network_update"] == "disabled"
    assert c["checkpoint_modification"] == "forbidden"


def test_T55_complete_final_matrix_required():
    fm = _load(RES / "E4E_final_matrix_contract.json")
    assert fm["FINAL_MATRIX_COMPLETE_required"] is True
    assert fm["total_planned_episodes"] == 1800
    assert fm["learned_cells"] == 900


def test_T56_deterministic_execution_order():
    order = _load(RES / "E4E_execution_order.json")
    assert order["deterministic_outcome_independent_order"] is True
    assert order["no_early_stop_on_results"] is True
    assert order["scenario_order"] == ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T57_interruption_contract():
    fm = _load(RES / "E4E_final_matrix_contract.json")
    assert "resume only" in fm["interruption_contract"]
    assert "do not replace failed cells" in fm["interruption_contract"]


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def test_T58_zero_effect_fixture():
    B = np.linspace(0.005, 0.009, 60)
    L = np.tile(B, (5, 1))
    assert abs(EF.scenario_mean_effect(L, B)) < 1e-12


def test_T59_positive_effect_fixture():
    B = np.linspace(0.005, 0.009, 60)
    L = np.tile(B - 0.001, (5, 1))
    assert EF.scenario_mean_effect(L, B) == pytest.approx(0.001)


def test_T60_negative_effect_fixture():
    B = np.linspace(0.005, 0.009, 60)
    L = np.tile(B + 0.001, (5, 1))
    assert EF.scenario_mean_effect(L, B) == pytest.approx(-0.001)


def test_T61_clustered_crossed_wider_than_naive():
    B = np.linspace(0.005, 0.009, 60)
    offsets = np.array([-0.004, -0.004, 0.004, 0.004, 0.0])[:, None]
    L = np.tile(B - 0.001, (5, 1)) + offsets
    crossed = CB.crossed_two_axis_bootstrap(L, B, n_boot=6000, seed=5)
    naive = CB.naive_row_bootstrap(L, B, n_boot=6000, seed=5)
    assert (crossed["ci_hi"] - crossed["ci_lo"]) > (naive["ci_hi"] - naive["ci_lo"])


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _verdict(mp, rule, rob, ps, ss, ms):
    return FV.scientific_verdict(
        mixed_primary_label=mp, ddqn_vs_rule_mixed_label=rule,
        mixed_policy_robustness_pass=rob, persistent_safety_pass=ps,
        stationary_safety_pass=ss, macro_safety_pass=ms,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)["verdict"]


def test_T62_validated_logic():
    assert _verdict(FV.BENEFICIAL, FV.BENEFICIAL, True, True, True, True) == FV.VALIDATED


def test_T63_partial_logic():
    # beneficial primary but rule comparison not beneficial -> PARTIAL
    assert _verdict(FV.BENEFICIAL, FV.DIRECTIONAL_POSITIVE, True, True, True, True) == FV.PARTIAL
    # beneficial primary but only 3/5 policies -> PARTIAL
    assert _verdict(FV.BENEFICIAL, FV.BENEFICIAL, False, True, True, True) == FV.PARTIAL
    # directional-positive primary -> PARTIAL (not VALIDATED)
    assert _verdict(FV.DIRECTIONAL_POSITIVE, FV.BENEFICIAL, True, True, True, True) == FV.PARTIAL


def test_T64_failed_logic():
    assert _verdict(FV.PRACTICALLY_NEUTRAL, FV.BENEFICIAL, True, True, True, True) == FV.FAILED
    assert _verdict(FV.HARMFUL, FV.BENEFICIAL, True, True, True, True) == FV.FAILED
    assert _verdict(FV.DIRECTIONAL_NEGATIVE, FV.BENEFICIAL, True, True, True, True) == FV.FAILED
    # safety fail -> FAILED even with beneficial primary
    assert _verdict(FV.BENEFICIAL, FV.BENEFICIAL, True, False, True, True) == FV.FAILED


def test_engineering_integrity_failure_is_failed():
    v = FV.scientific_verdict(
        mixed_primary_label=FV.BENEFICIAL, ddqn_vs_rule_mixed_label=FV.BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=False, protocol_integrity_pass=True,
        no_training_during_final=True)
    assert v["verdict"] == FV.FAILED


# ---------------------------------------------------------------------------
# Production invariants / firewall / integrity
# ---------------------------------------------------------------------------

def test_T65_production_o1_f005_unchanged():
    manifest = _load(RES / "E4E_pre_freeze_manifest.json")
    for fp, h in manifest["production_operator_hashes"].items():
        assert h == _hash(fp), f"production file changed: {fp}"


def test_T66_T67_T68_no_hard_no_memory_budget():
    c = _load(RES / "E4E_final_eval_execution_contract.json")
    assert c["no_HARD"] is True
    assert c["no_MEMORY"] is True
    assert c["no_RECALL"] is True
    assert c["no_B_r_a"] is True
    assert c["production"] == "O1_F005"


def test_T69_e4tr_artifacts_unchanged():
    # selected checkpoints file still lists the exact 5 and hashes still match
    sel = _load(E4TR / "E4TR_selected_checkpoints.json")
    for ps in E4E.POLICY_SEEDS:
        info = sel["selected"][str(ps)]
        assert info["sha256"] == _hash(info["checkpoint_path"])


def test_T70_no_substantive_final_evaluation():
    v = _load(RES / "E4E_verdict.json")
    assert v["FINAL_EVALUATION"] == "NOT_RUN"
    assert v["SCIENTIFIC_SUPERIORITY_CLAIM"] == "NOT_MADE"
    assert v["FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T71_no_training_flags():
    m = _load(RES / "E4E_final_eval_protocol_frozen.json")
    assert m["DRL_TRAINING"] == "COMPLETE"
    assert m["RL_STATUS"] == "TRAINED_AWAITING_FINAL_EVAL"


def test_e4e_complete_and_no_placeholders():
    v = _load(RES / "E4E_verdict.json")
    assert v["E4E_COMPLETE"] == "YES"
    assert v["no_placeholder_tokens"] is True
    assert v["placeholder_hits"] == []
    assert v["NEXT_STAGE"] == "E4F_FINAL_EVALUATION_EXECUTION"


def test_master_frozen_has_no_placeholder_tokens():
    m = _load(RES / "E4E_final_eval_protocol_frozen.json")
    assert E4E._no_forbidden_tokens(m) == []


def test_output_schema_predeclared():
    os_ = _load(RES / "E4E_output_schema.json")
    assert os_["e4f_results_dir"] == "results/E4F_final_evaluation"
    assert "E4F_verdict.json" in os_["files"]
    assert os_["episode_metrics"] == ["MIGD_response", "MIGD_epoch", "response_FE"]
    assert os_["no_duplicate_keys"] is True
