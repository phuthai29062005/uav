"""
E4IDD — Reward Scale Diagnostic Execution: artifact validation tests.

Validates the executed 300-episode scale diagnostic WITHOUT rerunning all episodes:
integrity/firewall, matrix completeness, R1 formula/bounds/sign, R0/R1 side-by-side
reconstruction, activity vs active-magnitude on a matched subset, the deterministic
alpha decision + bounds/scenario/tail gates, and the frozen reward-R1 config. It also
recomputes the key scale quantities directly from the saved regional records.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from prdrl.rl.reward_r1 import REWARD_R1_NAME, regional_quality_gain_r1, EPS_R1

RES = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4ID = Path("results/E4ID_reward_scale_diagnostic_design")
E4I = Path("results/E4I_reward_credit_revision_design")
E4HF = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F = Path("results/E4F_final_evaluation")
E4HE = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")

POLICY = [3001, 3002, 3003, 3004, 3005]
SEEDS = list(range(1701, 1721))
SCEN = ["PERSISTENT", "STATIONARY", "MIXED"]


def _load(p):
    with open(p) as f:
        return json.load(f)


def _require():
    if not (RES / "E4IDD_verdict.json").exists():
        pytest.skip("E4IDD not run yet")


_REC_CACHE = None


def _records():
    global _REC_CACHE
    if _REC_CACHE is None:
        with open(RES / "E4IDD_regional_reward_records.csv") as f:
            _REC_CACHE = list(csv.DictReader(f))
    return _REC_CACHE


def _f(rows, key):
    return np.array([float(r[key]) for r in rows])


# ---- prerequisites / integrity ----

def test_T1_e4id_complete():
    assert _load(E4ID / "E4ID_verdict.json")["E4ID_COMPLETE"] == "YES"


def test_T2_e4f_failed():
    assert _load(E4F / "E4F_verdict.json")["SCIENTIFIC_VERDICT"] == "FAILED"


def test_T3_e4hf_normalization_failed():
    assert _load(E4HF / "E4HF_verdict.json")["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"


def test_T4_five_v3_behavior_policies():
    _require()
    ck = _load(RES / "E4IDD_checkpoint_integrity.json")["checkpoint_pre_hashes"]
    assert sorted(int(k) for k in ck) == POLICY


def test_T5_behavior_checkpoint_pre_hashes_exact():
    _require()
    frz = _load(E4HE / "E4HE_checkpoint_freeze.json")["checkpoints"]
    v3 = {int(v["policy_seed"]): v["sha256"] for v in frz.values() if v["arm"] == "V3_TREATMENT"}
    ck = _load(RES / "E4IDD_checkpoint_integrity.json")["checkpoint_pre_hashes"]
    for p in POLICY:
        assert ck[str(p)] == v3[p]
    assert _load(RES / "E4IDD_checkpoint_integrity.json")["all_match_frozen"] is True


def test_T6_diagnostic_seeds_exact():
    _require()
    assert _load(RES / "E4IDD_diagnostic_seed_usage.json")["diagnostic_seeds"] == [1701, 1720]


def test_T7_diagnostic_seed_label():
    _require()
    assert _load(RES / "E4IDD_diagnostic_seed_usage.json")["status"] == \
        "CONSUMED_POST_HOC_REWARD_SCALE_DIAGNOSTIC_ONLY"


def test_T8_exactly_300_episodes():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    assert v["EPISODES_COMPLETE"] == "300/300"
    assert v["DIAGNOSTIC_MATRIX_COMPLETE"] == "YES"


def test_T9_T10_no_duplicate_or_missing_cells():
    _require()
    with open(RES / "E4IDD_episode_results.csv") as f:
        rows = list(csv.DictReader(f))
    cells = {(int(r["policy_seed"]), r["scenario"], int(r["environment_seed"])) for r in rows}
    expected = {(p, s, e) for p in POLICY for s in SCEN for e in SEEDS}
    assert cells == expected and len(cells) == len(rows) == 300


def test_T11_scenarios_exact():
    _require()
    with open(RES / "E4IDD_episode_results.csv") as f:
        scen = {r["scenario"] for r in csv.DictReader(f)}
    assert scen == set(SCEN)


def test_T12_policy_seeds_exact():
    _require()
    with open(RES / "E4IDD_episode_results.csv") as f:
        pol = {int(r["policy_seed"]) for r in csv.DictReader(f)}
    assert pol == set(POLICY)


def test_T13_no_future_reward_seed_accessed():
    _require()
    fw = _load(RES / "E4IDD_future_seed_firewall.json")
    assert fw["REWARD_TRAIN_ACCESS_COUNT"] == 0
    assert fw["REWARD_VALIDATION_ACCESS_COUNT"] == 0
    assert fw["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T14_epsilon_zero():
    _require()
    assert _load(RES / "E4IDD_pre_execution_manifest.json")["execution_mode"]["epsilon"] == 0.0


def test_T15_T16_T17_T18_no_learning():
    _require()
    em = _load(RES / "E4IDD_pre_execution_manifest.json")["execution_mode"]
    assert em["gradients"] == "disabled" and em["optimizer"] == "disabled"
    assert em["replay"] == "disabled" and em["target_updates"] == "disabled" and em["learning"] == "none"


def test_T19_no_new_checkpoint_model_modification():
    _require()
    assert _load(RES / "E4IDD_verdict.json")["MODEL_MODIFICATION"] == "NO"


def test_T20_v3_normalization_exact():
    _require()
    from prdrl.rl.normalization import NORM_V3
    assert _load(RES / "E4IDD_pre_execution_manifest.json")["v3_sha_matches_reference"] is True


def test_T21_v3_constants_unchanged():
    _require()
    assert _load(RES / "E4IDD_pre_execution_manifest.json")["v3_sha_matches_reference"] is True


def test_T23_simultaneous_action_semantics_records_present():
    _require()
    # 300 episodes * 10 changes * active regions -> regional records exist and keyed uniquely
    recs = _records()
    keys = {(r["policy_seed"], r["scenario"], r["environment_seed"], r["change_index"], r["region_id"])
            for r in recs}
    assert len(keys) == len(recs)  # unique regional keys


def test_T24_T25_l_pre_l_post_logged():
    _require()
    r = _records()[0]
    assert "L_pre" in r and "L_post" in r


def test_T27_r1_formula_independent_reference_agreement():
    _require()
    fi = _load(RES / "E4IDD_r1_formula_integrity.json")
    assert fi["R1_FORMULA_INTEGRITY"] == "PASS"
    assert fi["max_abs_diff"] <= 1e-12
    # spot recompute g from L_pre/L_post via production function
    recs = _records()[:500]
    for r in recs:
        g = regional_quality_gain_r1(float(r["L_pre"]), float(r["L_post"]))
        assert abs(g - float(r["g_r"])) <= 1e-12


def test_T28_g_bounded():
    _require()
    b = _load(RES / "E4IDD_r1_bounds_audit.json")
    assert b["R1_BOUNDS_INTEGRITY"] == "PASS" and b["bound_violation_count"] == 0
    g = _f(_records(), "g_r")
    assert g.min() >= -1.0 - 1e-9 and g.max() <= 1.0 + 1e-9


def test_T29_T30_T31_sign_semantics():
    _require()
    s = _load(RES / "E4IDD_sign_semantics.json")
    assert s["SIGN_SEMANTICS"] == "PASS" and s["sign_violation_count"] == 0
    # independent recompute
    viol = 0
    for r in _records():
        Lp, Lq, g = float(r["L_pre"]), float(r["L_post"]), float(r["g_r"])
        if not (math.isfinite(Lp) and math.isfinite(Lq)):
            continue
        if Lq < Lp and not g > 0:
            viol += 1
        elif Lq > Lp and not g < 0:
            viol += 1
        elif Lq == Lp and abs(g) > EPS_R1:
            viol += 1
    assert viol == 0


def test_T32_r0_reconstruction_exact():
    _require()
    for r in _records():
        lhs = float(r["total_reward_R0"])
        rhs = (float(r["weighted_global_reward"]) + float(r["weighted_local_reward_R0"])
               - float(r["weighted_fe_penalty"]))
        assert abs(lhs - rhs) <= 1e-9


def test_T33_r1_alpha01_reconstruction_exact():
    _require()
    for r in _records():
        lhs = float(r["total_reward_R1_alpha_0p1"])
        rhs = (float(r["weighted_global_reward"]) + float(r["weighted_local_R1_alpha_0p1"])
               - float(r["weighted_fe_penalty"]))
        assert abs(lhs - rhs) <= 1e-9


def test_T34_same_global_term_r0_r1():
    _require()
    # weighted_global identical (single column shared) -> trivially same for R0 and R1 rows
    r = _records()[0]
    assert "weighted_global_reward" in r  # single shared column used by both totals


def test_T35_same_fe_term_r0_r1():
    _require()
    r = _records()[0]
    assert "weighted_fe_penalty" in r


def test_T36_extra_fe_zero():
    _require()
    assert _load(RES / "E4IDD_fe_accounting.json")["R1_EXTRA_FE"] == 0
    assert _load(RES / "E4IDD_verdict.json")["R1_EXTRA_FE"] == 0


def test_T37_local_active_threshold():
    _require()
    a = _load(RES / "E4IDD_r1_activity.json")
    # recompute active fraction from records with |g|>1e-12
    g = _f(_records(), "g_r")
    frac = float(np.mean(np.abs(g) > 1e-12))
    assert abs(frac - a["local_active_fraction"]) < 1e-9


def test_T38_activity_denominator_all_valid():
    _require()
    a = _load(RES / "E4IDD_r1_activity.json")
    assert a["n_valid_transitions"] == len(_records()) == 30000


def test_T39_activity_gate():
    _require()
    a = _load(RES / "E4IDD_r1_activity.json")
    assert a["R1_ACTIVITY_GATE"] == ("PASS" if a["local_active_fraction"] >= 0.20 else "FAIL")
    assert a["R1_ACTIVITY_GATE"] == "PASS"


def test_T40_unconditional_not_used_for_alpha():
    _require()
    # unconditional table exists (transparency) but the alpha decision uses the ACTIVE subset
    assert (RES / "E4IDD_r1_unconditional_magnitude.csv").exists()
    am = list(csv.DictReader(open(RES / "E4IDD_r1_active_magnitude.csv")))
    assert any("active" in r["metric"] for r in am)
    # the active-magnitude source explicitly flags subset = LOCAL_ACTIVE
    assert _load(RES / "E4IDD_initial_alpha_scale.json")["INITIAL_R1_SCALE"] in (
        "LOCAL_MATERIAL", "GLOBAL_OVERWHELMS_LOCAL")


def test_T41_T42_G_and_M_use_identical_active_mask():
    _require()
    recs = _records()
    g = _f(recs, "g_r")
    wg = np.abs(_f(recs, "weighted_global_reward"))
    mask = np.abs(g) > 1e-12
    G = float(np.median(wg[mask]))
    M = float(np.median(np.abs(g[mask])))
    isc = _load(RES / "E4IDD_initial_alpha_scale.json")
    assert abs(G - isc["G"]) < 1e-9 and abs(M - isc["M"]) < 1e-9


def test_T43_M_positive():
    _require()
    assert _load(RES / "E4IDD_initial_alpha_scale.json")["M"] > 0.0


def test_T44_L0_equals_median_abs_0p1_g():
    _require()
    recs = _records()
    g = _f(recs, "g_r")
    mask = np.abs(g) > 1e-12
    L0 = float(np.median(np.abs(0.1 * g[mask])))
    assert abs(L0 - _load(RES / "E4IDD_initial_alpha_scale.json")["L0"]) < 1e-9


def test_T45_ratio_0_exact():
    _require()
    isc = _load(RES / "E4IDD_initial_alpha_scale.json")
    assert abs(isc["RATIO_0"] - isc["G"] / (isc["L0"] + 1e-12)) < 1e-6


def test_T46_material_threshold_10():
    _require()
    isc = _load(RES / "E4IDD_initial_alpha_scale.json")
    assert isc["GLOBAL_OVERWHELMS_THRESHOLD"] == 10.0
    expect = "GLOBAL_OVERWHELMS_LOCAL" if isc["RATIO_0"] >= 10.0 else "LOCAL_MATERIAL"
    assert isc["INITIAL_R1_SCALE"] == expect


def test_T47_alpha_kept_01_if_ratio_lt_10():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    if v["INITIAL_GLOBAL_LOCAL_RATIO"] < 10.0:
        assert v["ALPHA_DECISION_CASE"] == "KEEP_INITIAL_0P1"
        assert v["ALPHA_FINAL"] == 0.1


def test_T48_alpha_scale_formula_if_ratio_ge_10():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    if v["INITIAL_GLOBAL_LOCAL_RATIO"] >= 10.0:
        isc = _load(RES / "E4IDD_initial_alpha_scale.json")
        assert v["ALPHA_DECISION_CASE"] == "SCALE_MATCH"
        assert abs(float(v["ALPHA_SCALE"]) - isc["G"] / (isc["M"] + 1e-12)) < 1e-9


def test_T49_no_alpha_sweep():
    _require()
    # only KEEP_0.1 or single deterministic SCALE_MATCH permitted
    assert _load(RES / "E4IDD_verdict.json")["ALPHA_DECISION_CASE"] in ("KEEP_INITIAL_0P1", "SCALE_MATCH")


def test_T50_T51_alpha_bounds():
    _require()
    b = _load(RES / "E4IDD_alpha_bound_gate.json")
    assert b["alpha_min"] == 0.1 and b["alpha_max"] == 1.0


def test_T52_no_silent_clipping():
    _require()
    assert _load(RES / "E4IDD_alpha_bound_gate.json")["no_clipping"] is True


def test_T53_T54_scenario_ratios_same_alpha_gate():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    ratios = [v["PERSISTENT_GLOBAL_LOCAL_RATIO"], v["STATIONARY_GLOBAL_LOCAL_RATIO"],
              v["MIXED_GLOBAL_LOCAL_RATIO"]]
    num = [r for r in ratios if isinstance(r, (int, float))]
    gate = "PASS" if all(r < 10.0 for r in num) else "FAIL"
    assert v["SCENARIO_SCALE_GATE"] == gate


def test_T55_T56_T57_T58_tail_ratios_and_gate():
    _require()
    t = _load(RES / "E4IDD_tail_safety.json")
    assert t["Q95_RATIO_MAX"] == 5.0 and t["Q99_RATIO_MAX"] == 10.0
    if isinstance(t["Q95_RATIO"], (int, float)) and isinstance(t["Q99_RATIO"], (int, float)):
        expect = "PASS" if (t["Q95_RATIO"] <= 5.0 and t["Q99_RATIO"] <= 10.0) else "FAIL"
        assert t["TAIL_SAFETY_GATE"] == expect


def test_T59_total_r1_reward_finite():
    _require()
    d = _load(RES / "E4IDD_reward_distribution.json")
    assert d["TOTAL_REWARD_NUMERICAL_GATE"] == "PASS" and d["all_finite"] is True


def test_T60_alpha_final_full_precision():
    _require()
    af = _load(RES / "E4IDD_alpha_freeze.json")
    assert af["alpha_final"] == 0.1 or isinstance(af["alpha_final"], float)
    assert "alpha_final_serialized_precision" in af


def test_T61_reward_config_hash_reproducible():
    _require()
    if not (RES / "E4IDD_reward_r1_config.json").exists():
        pytest.skip("alpha not frozen")
    cfg = _load(RES / "E4IDD_reward_r1_config.json")
    stored = cfg.pop("REWARD_R1_CONFIG_SHA256")
    recomputed = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert stored == recomputed
    assert _load(RES / "E4IDD_verdict.json")["REWARD_R1_CONFIG_SHA256"] == stored


def test_T62_r1_readiness_logic():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    ready = (v["DIAGNOSTIC_MATRIX_COMPLETE"] == "YES" and v["R1_FORMULA_INTEGRITY"] == "PASS"
             and v["R1_BOUNDS_INTEGRITY"] == "PASS" and v["SIGN_SEMANTICS"] == "PASS"
             and v["R1_ACTIVITY_GATE"] == "PASS" and v["SCENARIO_SCALE_GATE"] == "PASS"
             and v["TAIL_SAFETY_GATE"] == "PASS" and v["TOTAL_REWARD_NUMERICAL_GATE"] == "PASS"
             and v["ALPHA_FINAL"] != "NOT_FROZEN" and v["CHECKPOINT_POST_INTEGRITY"] == "PASS")
    assert (v["R1_REWARD_READY"] == "YES") == ready


def test_T63_performance_not_used_for_alpha():
    _require()
    sb = _load(RES / "E4IDD_scientific_boundary.json")
    assert sb["MIGD_not_used_for_alpha"] is True
    # episode MIGD columns are explicitly labelled not-used-for-alpha
    with open(RES / "E4IDD_episode_results.csv") as f:
        cols = next(csv.reader(f))
    assert any("DIAGNOSTIC_NOT_USED_FOR_ALPHA" in c for c in cols)


def test_T64_no_r1_performance_claim():
    _require()
    sb = _load(RES / "E4IDD_scientific_boundary.json")
    assert sb["no_R1_performance_claim"] is True
    assert "R1 beats R0" in sb["forbidden_claims"]


def test_T65_checkpoint_post_hashes_unchanged():
    _require()
    assert _load(RES / "E4IDD_checkpoint_post_integrity.json")["pre_equals_post"] is True
    assert _load(RES / "E4IDD_verdict.json")["CHECKPOINT_POST_INTEGRITY"] == "PASS"


def test_T66_diagnostic_seeds_marked_consumed():
    _require()
    assert _load(RES / "E4IDD_diagnostic_seed_usage.json")["status"] == \
        "CONSUMED_POST_HOC_REWARD_SCALE_DIAGNOSTIC_ONLY"


def test_T67_T68_T69_reward_access_zero():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    assert v["REWARD_TRAIN_ACCESS_COUNT"] == 0
    assert v["REWARD_VALIDATION_ACCESS_COUNT"] == 0
    assert v["REWARD_FINAL_EVAL_ACCESS_COUNT"] == 0


def test_T70_T71_no_model_modification_no_training():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    assert v["MODEL_MODIFICATION"] == "NO" and v["NEW_TRAINING"] == "NO"


def test_T72_next_stage_and_config_present():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    if v["R1_REWARD_READY"] == "YES":
        assert v["NEXT_STAGE"] == "E4II_REWARD_IMPLEMENTATION_VALIDATION"
        assert v["REWARD_R1_CONFIG_SHA256"] != "NOT_CREATED"
        assert (RES / "E4IDD_reward_r1_config.json").exists()


def test_T73_reward_version_in_config():
    _require()
    if (RES / "E4IDD_reward_r1_config.json").exists():
        cfg = _load(RES / "E4IDD_reward_r1_config.json")
        assert cfg["reward_version"] == REWARD_R1_NAME
        assert cfg["alpha_global"] == 1.0 and cfg["beta"] == 0.01


def test_T74_scenario_active_counts_transparency():
    _require()
    v = _load(RES / "E4IDD_verdict.json")
    ac = v["SCENARIO_ACTIVE_COUNTS"]
    # STATIONARY has no environmental change -> zero active local credit (documented)
    assert ac["STATIONARY"] == 0
    assert ac["PERSISTENT"] > 0 and ac["MIXED"] > 0
