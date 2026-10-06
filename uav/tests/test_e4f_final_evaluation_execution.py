"""
E4F — Final Evaluation Execution: contract / unit / artifact tests.

These tests DO NOT rerun the 1800 scientific episodes. They validate artifacts,
integrity, and that the recorded classifications/verdict are CONSISTENT with the
frozen logic. They never assert a particular scientific outcome — whatever E4F
returned must be preserved.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import prdrl.experiments.E4F_final_evaluation_execution as E4F
from prdrl.evaluation.crossed_bootstrap import (
    crossed_two_axis_bootstrap, crossed_two_axis_bootstrap_macro,
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, BOOTSTRAP_MASTER_SEED,
)
from prdrl.evaluation import final_verdict as FV

RES = Path("results/E4F_final_evaluation")
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


def _require_run():
    if not (RES / "E4F_verdict.json").exists():
        pytest.skip("E4F run artifacts not present yet")


def _episodes():
    with open(RES / "E4F_episode_results.csv") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# CONTRACT / UNIT (no artifacts)
# ---------------------------------------------------------------------------

def test_T1_e4e_complete():
    assert _load(E4E / "E4E_verdict.json")["E4E_COMPLETE"] == "YES"


def test_T3_exact_policy_seeds():
    assert E4F.POLICY_SEEDS == [2001, 2002, 2003, 2004, 2005]


def test_T4_exact_final_seeds():
    assert E4F.FINAL_EVAL_SEEDS == list(range(1081, 1141))
    assert len(E4F.FINAL_EVAL_SEEDS) == 60


def test_T5_exact_scenarios():
    assert E4F.SCENARIOS == ["PERSISTENT", "STATIONARY", "MIXED"]


def test_T6_T7_exact_ten_arms_order():
    assert E4F.ARM_ORDER == ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT",
                             "GLOBAL_DIVERSIFY", "RULE_V1_V2STATE",
                             "DDQN_2001", "DDQN_2002", "DDQN_2003", "DDQN_2004", "DDQN_2005"]
    assert len(E4F.ARM_ORDER) == 10


def test_T26_T27_T28_comparators():
    assert E4F.PRESELECTED_COMPARATOR["PERSISTENT"] == "GLOBAL_PREDICT"
    assert E4F.PRESELECTED_COMPARATOR["STATIONARY"] == "GLOBAL_DIVERSIFY"
    assert E4F.PRESELECTED_COMPARATOR["MIXED"] == "GLOBAL_PREDICT"


def test_T22_four_actions():
    from prdrl.rl.network import ACTION_ORDER
    assert ACTION_ORDER == ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")


def test_T35_bootstrap_settings():
    assert N_BOOTSTRAP == 50000
    assert CI_LEVEL == 0.95
    assert CI_METHOD == "percentile"
    assert BOOTSTRAP_MASTER_SEED == 46001


def test_T36_practical_margin():
    assert FV.DELTA_E4 == 1e-5


def test_T42_macro_shared_draws_unit():
    # identical scenarios -> macro CI equals single-scenario CI with same seed
    L = np.random.default_rng(0).uniform(0.004, 0.008, size=(5, 60))
    B = np.random.default_rng(1).uniform(0.004, 0.008, size=60)
    macro = crossed_two_axis_bootstrap_macro(
        {s: L for s in E4F.SCENARIOS}, {s: B for s in E4F.SCENARIOS},
        E4F.SCENARIOS, n_boot=3000, seed=555)
    single = crossed_two_axis_bootstrap(L, B, n_boot=3000, seed=555)
    assert abs(macro["ci_lo"] - single["ci_lo"]) < 1e-12
    assert abs(macro["ci_hi"] - single["ci_hi"]) < 1e-12


def test_T50_T51_T52_verdict_logic_unit():
    def v(mp, rule, rob, ps, ss, ms):
        return FV.scientific_verdict(
            mixed_primary_label=mp, ddqn_vs_rule_mixed_label=rule,
            mixed_policy_robustness_pass=rob, persistent_safety_pass=ps,
            stationary_safety_pass=ss, macro_safety_pass=ms,
            checkpoint_integrity_pass=True, protocol_integrity_pass=True,
            no_training_during_final=True)["verdict"]
    assert v(FV.BENEFICIAL, FV.BENEFICIAL, True, True, True, True) == FV.VALIDATED
    assert v(FV.BENEFICIAL, FV.DIRECTIONAL_POSITIVE, True, True, True, True) == FV.PARTIAL
    assert v(FV.PRACTICALLY_NEUTRAL, FV.BENEFICIAL, True, True, True, True) == FV.FAILED
    assert v(FV.HARMFUL, FV.BENEFICIAL, True, True, True, True) == FV.FAILED


def test_T44_fe_cannot_rescue_harmful():
    # verdict function has no FE input; harmful mixed -> FAILED
    v = FV.scientific_verdict(
        mixed_primary_label=FV.HARMFUL, ddqn_vs_rule_mixed_label=FV.BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)
    assert v["verdict"] == FV.FAILED


def test_T31_effect_sign_positive_favors_ddqn():
    from prdrl.evaluation.effects import scenario_mean_effect
    L = np.full((5, 60), 0.004)
    B = np.full(60, 0.005)
    assert scenario_mean_effect(L, B) == pytest.approx(0.001)  # DDQN lower -> positive


# ---------------------------------------------------------------------------
# ARTIFACT TESTS (require run)
# ---------------------------------------------------------------------------

def test_T2_checkpoint_hashes_exact():
    _require_run()
    integ = _load(RES / "E4F_checkpoint_integrity.json")
    assert integ["CHECKPOINT_INTEGRITY"] == "PASS"
    for ps in E4F.POLICY_SEEDS:
        c = integ["checkpoints"][str(ps)]
        assert c["hash_match"] is True
        assert c["sha256_actual"] == c["sha256_frozen"]


def test_T8_1800_episode_rows():
    _require_run()
    assert len(_episodes()) == 1800


def test_T9_T10_row_split():
    _require_run()
    eps = _episodes()
    ddqn = [r for r in eps if r["controller"] == "DDQN"]
    base = [r for r in eps if r["controller"] in E4F.FIXED_GLOBALS]
    rule = [r for r in eps if r["controller"] == "RULE_V1_V2STATE"]
    assert len(ddqn) == 900
    assert len(base) + len(rule) == 900
    assert len(rule) == 180  # 60 seeds x 3 scenarios


def test_T11_no_duplicate_keys():
    _require_run()
    eps = _episodes()
    keys = [(r["controller"], r["policy_seed"], r["scenario"], r["environment_seed"]) for r in eps]
    assert len(keys) == len(set(keys)) == 1800


def test_T12_matrix_complete():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    assert v["FINAL_MATRIX_COMPLETE"] == "YES"
    assert v["EPISODES_COMPLETE"] == "1800/1800"


def test_T13_all_metrics_finite():
    _require_run()
    for r in _episodes():
        for m in ["MIGD_response", "MIGD_epoch", "response_FE"]:
            assert np.isfinite(float(r[m]))


def test_T14_T15_T16_T17_T18_execution_mode():
    _require_run()
    m = _load(RES / "E4F_pre_execution_manifest.json")["execution_mode"]
    assert m["epsilon"] == 0.0
    assert m["gradients"] == "disabled"
    assert m["replay"] == "disabled"
    assert m["optimizer"] == "disabled"
    assert m["target_updates"] == "disabled"
    assert m["learning"] == "none"


def test_T19_T58_checkpoint_files_unchanged():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    assert v["checkpoint_files_unchanged_after_E4F"] is True
    frz = _load(E4E / "E4E_checkpoint_freeze.json")
    for ps in E4F.POLICY_SEEDS:
        info = frz["policies"][str(ps)]
        assert _hash(info["checkpoint_path"]) == info["sha256"]


def test_T20_production_o1_f005():
    _require_run()
    m = _load(RES / "E4F_pre_execution_manifest.json")
    # production operator hashes recorded and unchanged
    for fp, h in m["production_operator_hashes"].items():
        assert h == _hash(fp)


def test_T23_predict_fallback_semantics():
    _require_run()
    # scan change-level rows: whenever a region falls back, realized=LOCAL but requested=PREDICT
    found_predict = False
    with open(RES / "E4F_change_results.jsonl") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for reg in row["regional"]:
                if reg["requested"] == "PREDICT":
                    found_predict = True
                    if reg["fallback"]:
                        assert reg["realized"] == "LOCAL"
                    else:
                        assert reg["realized"] == "PREDICT"
            if i > 5000:
                break
    assert found_predict


def test_T30_primary_endpoint():
    _require_run()
    ec = _load(E4E / "E4E_endpoint_contract.json")
    assert ec["primary_endpoint"] == "MIGD_response"


def test_T32_matrix_5x60_per_scenario():
    _require_run()
    eps = _episodes()
    for scenario in E4F.SCENARIOS:
        ddqn = [r for r in eps if r["controller"] == "DDQN" and r["scenario"] == scenario]
        assert len(ddqn) == 300  # 5 policies x 60 env
        pols = {r["policy_seed"] for r in ddqn}
        assert pols == {str(p) for p in E4F.POLICY_SEEDS}


def test_T33_T34_crossed_not_naive():
    _require_run()
    bs = _load(RES / "E4F_bootstrap_results.json")
    assert bs["settings"]["design"] == "CROSSED_TWO_AXIS_BOOTSTRAP"
    assert bs["settings"]["policy_axis"] == 5
    assert bs["settings"]["environment_axis"] == 60
    assert bs["settings"]["N_BOOTSTRAP"] == 50000


def test_T37_mixed_primary_classification_consistent():
    _require_run()
    g = _load(RES / "E4F_scientific_gates.json")
    ci = g["MIXED_DELTA_RESPONSE_CI"]
    recomputed = FV.classify_effect(g["MIXED_DELTA_RESPONSE"], ci[0], ci[1])
    assert recomputed == g["MIXED_PRIMARY"]


def test_T38_policy_robustness_count_consistent():
    _require_run()
    g = _load(RES / "E4F_scientific_gates.json")
    pp = g["per_policy_delta_MIXED_response"]
    assert len(pp) == 5
    n_pos = sum(1 for d in pp if d > 0)
    assert n_pos == g["MIXED_POLICY_POSITIVE_COUNT"]
    expect = "PASS" if n_pos >= 4 else "FAIL"
    assert g["MIXED_POLICY_ROBUSTNESS"] == expect


def test_T39_ddqn_vs_rule_consistent():
    _require_run()
    g = _load(RES / "E4F_scientific_gates.json")
    ci = g["DDQN_VS_RULE_MIXED_CI"]
    recomputed = FV.classify_effect(g["DDQN_VS_RULE_MIXED_DELTA"], ci[0], ci[1])
    assert recomputed == g["DDQN_VS_RULE_MIXED"]


def test_T40_T41_safety_consistent():
    _require_run()
    g = _load(RES / "E4F_scientific_gates.json")
    for scenario in ["PERSISTENT", "STATIONARY"]:
        rlab = g[f"{scenario}_RESPONSE_EFFECT"]
        elab = g[f"{scenario}_EPOCH_EFFECT"]
        expect = FV.safety_gate(rlab, elab)["PASS"]
        assert (g[f"{scenario}_SAFETY"] == "PASS") == expect


def test_T43_macro_safety_consistent():
    _require_run()
    g = _load(RES / "E4F_scientific_gates.json")
    expect = FV.safety_gate(g["MACRO_RESPONSE_EFFECT"], g["MACRO_EPOCH_EFFECT"])["PASS"]
    assert (g["MACRO_SAFETY"] == "PASS") == expect


def test_T45_policy_summary_all_five():
    _require_run()
    with open(RES / "E4F_policy_seed_summary.csv") as f:
        rows = list(csv.DictReader(f))
    seeds = {r["policy_seed"] for r in rows}
    assert seeds == {str(p) for p in E4F.POLICY_SEEDS}
    assert len(rows) == 15  # 5 policies x 3 scenarios


def test_T46_action_fractions_sum_to_one():
    _require_run()
    with open(RES / "E4F_action_distribution.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        s = sum(float(r[a]) for a in ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"])
        assert abs(s - 1.0) < 1e-6


def test_T49_scientific_verdict_consistent():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    g = v["gates"]
    recomputed = FV.scientific_verdict(
        mixed_primary_label=g["MIXED_PRIMARY"],
        ddqn_vs_rule_mixed_label=g["DDQN_VS_RULE_MIXED"],
        mixed_policy_robustness_pass=(g["MIXED_POLICY_ROBUSTNESS"] == "PASS"),
        persistent_safety_pass=(g["PERSISTENT_SAFETY"] == "PASS"),
        stationary_safety_pass=(g["STATIONARY_SAFETY"] == "PASS"),
        macro_safety_pass=(g["MACRO_SAFETY"] == "PASS"),
        checkpoint_integrity_pass=(v["CHECKPOINT_INTEGRITY"] == "PASS"),
        protocol_integrity_pass=(v["FINAL_PROTOCOL_INTEGRITY"] == "PASS"),
        no_training_during_final=True)["verdict"]
    assert recomputed == v["SCIENTIFIC_VERDICT"]
    assert v["SCIENTIFIC_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")


def test_T29_T54_T55_T56_T57_no_reselection_or_retuning():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    assert v["HYPERPARAMETER_RETUNING"] == "NO"
    assert v["CHECKPOINT_RESELECTION"] == "NO"
    assert v["FINAL_COMPARATOR_RESELECTION"] == "NO"


def test_T48_no_reduced_analysis():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    # verdict only computed when matrix complete
    if v["FINAL_MATRIX_COMPLETE"] != "YES":
        assert v["SCIENTIFIC_VERDICT"] == "NOT_COMPUTED"
    else:
        assert v["SCIENTIFIC_VERDICT"] in ("VALIDATED", "PARTIAL", "FAILED")


def test_final_status_fields():
    _require_run()
    v = _load(RES / "E4F_verdict.json")
    if v["E4F_COMPLETE"] == "YES":
        assert v["RL_STATUS"] == "FINAL_EVALUATED"
        assert v["DRL_TRAINING"] == "COMPLETE"
        assert v["FINAL_EVALUATION"] == "COMPLETE"


def test_scenario_effects_csv_shape():
    _require_run()
    with open(RES / "E4F_scenario_effects.csv") as f:
        rows = list(csv.DictReader(f))
    # 3 scenarios x 2 endpoints (response, epoch)
    assert len(rows) == 6
    combos = {(r["scenario"], r["endpoint"]) for r in rows}
    assert ("MIXED", "MIGD_response") in combos
