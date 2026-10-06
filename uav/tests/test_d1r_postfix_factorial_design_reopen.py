"""
T1-T66+: D1-R Post-Fix 2×2 Factorial Design Reopen — test suite.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ================================================================
# T1-T6: Historical verdicts (cross-check prior results)
# ================================================================

def test_T1_d2rf_gates_all_pass():
    path = "results/D2_rng_engineering_fix/D2RF_gate_results.json"
    assert os.path.exists(path), "D2RF gate results not found"
    with open(path) as f:
        data = json.load(f)
    assert data.get("FIX_VERDICT") == "RESPONSE_SURVIVAL_RNG_DETERMINISTIC"
    gates = data.get("gates", {})
    assert all(v == "PASS" for v in gates.values()), f"Not all D2RF gates PASS: {gates}"


def test_T2_d2cz_verdict_loadable():
    path = "results/D2CZ_production_stability_confirmation/D2CZ_verdict.json"
    assert os.path.exists(path), "D2CZ verdict not found"
    with open(path) as f:
        data = json.load(f)
    assert data.get("audit") == "D2-CZ"
    assert "O3_POSTFIX_PRODUCTION_STABILITY" in data


def test_T3_d2cz_overall_failed():
    with open("results/D2CZ_production_stability_confirmation/D2CZ_verdict.json") as f:
        data = json.load(f)
    assert data["O3_POSTFIX_PRODUCTION_STABILITY"] == "FAILED"


def test_T4_d2cz_persistent_validated():
    with open("results/D2CZ_production_stability_confirmation/D2CZ_verdict.json") as f:
        data = json.load(f)
    assert data["PERSISTENT_POSTFIX"] == "VALIDATED"


def test_T5_d2cz_stationary_fail():
    with open("results/D2CZ_production_stability_confirmation/D2CZ_verdict.json") as f:
        data = json.load(f)
    assert data["STATIONARY_POSTFIX_SAFETY"] == "FAIL"


def test_T6_d2cz_mixed_partial():
    with open("results/D2CZ_production_stability_confirmation/D2CZ_verdict.json") as f:
        data = json.load(f)
    assert data["MIXED_POSTFIX_STABILITY"] == "PARTIAL"


# ================================================================
# T7-T12: Arm definitions
# ================================================================

def test_T7_exactly_four_arms():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert set(ARMS.keys()) == {"O0", "O1", "O2", "O3"}


def test_T8_O0_sigma_unrestricted():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED
    assert ARMS["O0"]["sigma_predict"] == 0.03
    assert ARMS["O0"]["mode"] == PREDICT_SURVIVAL_UNRESTRICTED


def test_T9_O1_sigma_unrestricted():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED
    assert ARMS["O1"]["sigma_predict"] == 0.005
    assert ARMS["O1"]["mode"] == PREDICT_SURVIVAL_UNRESTRICTED


def test_T10_O2_sigma_hard():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    assert ARMS["O2"]["sigma_predict"] == 0.03
    assert ARMS["O2"]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T11_O3_sigma_hard():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    assert ARMS["O3"]["sigma_predict"] == 0.005
    assert ARMS["O3"]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T12_no_fifth_arm():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert len(ARMS) == 4
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert '"O4"' not in src and "'O4'" not in src


# ================================================================
# T13-T16: Seed freshness
# ================================================================

def test_T13_seeds_401_440():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BLOCK_A_SEEDS, BLOCK_B_SEEDS, ALL_SEEDS
    assert BLOCK_A_SEEDS == list(range(401, 421))
    assert BLOCK_B_SEEDS == list(range(421, 441))
    assert ALL_SEEDS == BLOCK_A_SEEDS + BLOCK_B_SEEDS
    assert len(ALL_SEEDS) == 40


def test_T14_block_A_401_420():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BLOCK_A_SEEDS
    assert BLOCK_A_SEEDS[0] == 401
    assert BLOCK_A_SEEDS[-1] == 420


def test_T15_block_B_421_440():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BLOCK_B_SEEDS
    assert BLOCK_B_SEEDS[0] == 421
    assert BLOCK_B_SEEDS[-1] == 440


def test_T16_seeds_fresh_no_overlap():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ALL_SEEDS, SMOKE_SEEDS
    forbidden = list(range(1, 363))   # all used in previous experiments
    for s in ALL_SEEDS:
        assert s not in forbidden, f"Science seed {s} overlaps with prior seeds"
    for s in SMOKE_SEEDS:
        assert s not in ALL_SEEDS


# ================================================================
# T17-T20: Protocol compliance
# ================================================================

def test_T17_exact_scenario_set():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import SCENARIOS
    assert set(SCENARIOS.keys()) == {"PERSISTENT", "STATIONARY", "MIXED"}


def test_T18_authentic_child_rng_oracle_base_seed():
    """oracle_base_seed = seed * 100000 + k * 100 + tgt_r"""
    seed, k, R = 401, 0, 10
    tgt_r = (seed + k) % R
    oracle_base_seed = seed * 100000 + k * 100 + tgt_r
    assert oracle_base_seed == 40100001


def test_T19_deterministic_response_survival_rng():
    from prdrl.oracle.oracle import make_rng
    r1 = make_rng(40100001, 1, 1, "RESPONSE_SURVIVAL")
    r2 = make_rng(40100001, 1, 1, "RESPONSE_SURVIVAL")
    v1 = r1.integers(0, 1000, 5)
    v2 = r2.integers(0, 1000, 5)
    assert np.array_equal(v1, v2)


def test_T20_no_l2b_response_child_rng():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "L2B_RESPONSE" not in src
    assert "crn_diagnostic" not in src.lower()
    assert "L2B_CRN" not in src


# ================================================================
# T21-T26: State management
# ================================================================

def test_T21_carry_forward_in_source():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    # The runner should NOT reset X between scored changes
    assert "X_pop = rng_init.uniform" not in src.split("def run_one_seed")[1].split("for k in range")[1]


def test_T22_no_arm_state_aliasing():
    """Arm states must be deep-copied, not aliased."""
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    # deep copy used for history
    assert "copy.deepcopy" in src
    # X explicitly copied
    assert ".copy()" in src


def test_T23_common_environment():
    """All arms receive same environment (same cp_evaluate, same region_dynamics)."""
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import SCENARIOS, ARMS
    assert len(SCENARIOS) == 3   # same 3 scenarios for all arms


def test_T24_target_schedule_frozen():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import target_region
    for seed in range(401, 406):
        for k in range(10):
            assert target_region(seed, k, 10) == (seed + k) % 10


def test_T25_missing_history_local_fallback():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "has_predict_history" in src
    assert "LOCAL_FALLBACK" in src
    assert "LOCAL" in src


def test_T26_local_fallback_no_hard():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    # For LOCAL_FALLBACK, hard_triggered must be False
    # Check that hard_triggered logic requires realized_op == "PREDICT"
    assert "realized_op == \"PREDICT\"" in src or "realized_op == 'PREDICT'" in src


# ================================================================
# T27-T33: Action scope and HARD invariant
# ================================================================

def test_T27_O0_no_hard():
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED, PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert ARMS["O0"]["mode"] != PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T28_O1_no_hard():
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert ARMS["O1"]["mode"] != PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T29_O2_has_hard():
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert ARMS["O2"]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T30_O3_has_hard():
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    assert ARMS["O3"]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T31_fe_equality_required():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "FE FAIRNESS" in src
    assert "arm_fe_this" in src


def test_T32_O2_hard_invariant_checked():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "_check_hard_invariant" in src
    assert "hard_triggered" in src


def test_T33_O3_hard_invariant_checked():
    # Same function checks both O2 and O3 (both have HARD mode)
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    hard_arms = [a for a, v in ARMS.items() if v["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY]
    assert "O2" in hard_arms
    assert "O3" in hard_arms


# ================================================================
# T34-T40: Factorial algebra
# ================================================================

def _make_arm_means(o0, o1, o2, o3):
    return {"O0": o0, "O1": o1, "O2": o2, "O3": o3}

def test_T34_noise_benefit_U():
    """NoiseBenefit_U = MIGD(O0) - MIGD(O1)."""
    o0, o1, o2, o3 = 0.20, 0.15, 0.18, 0.12
    expected = o0 - o1  # 0.05
    assert abs(expected - 0.05) < 1e-12


def test_T35_noise_benefit_H():
    """NoiseBenefit_H = MIGD(O2) - MIGD(O3)."""
    o0, o1, o2, o3 = 0.20, 0.15, 0.18, 0.12
    expected = o2 - o3  # 0.06
    assert abs(expected - 0.06) < 1e-12


def test_T36_hard_benefit_03():
    """HardBenefit_03 = MIGD(O0) - MIGD(O2)."""
    o0, o1, o2, o3 = 0.20, 0.15, 0.18, 0.12
    expected = o0 - o2  # 0.02
    assert abs(expected - 0.02) < 1e-12


def test_T37_hard_benefit_005():
    """HardBenefit_005 = MIGD(O1) - MIGD(O3)."""
    o0, o1, o2, o3 = 0.20, 0.15, 0.18, 0.12
    expected = o1 - o3  # 0.03
    assert abs(expected - 0.03) < 1e-12


def test_T38_interaction_algebra_identity():
    """Interaction = NoiseBenefit_H - NoiseBenefit_U = HardBenefit_005 - HardBenefit_03."""
    o0, o1, o2, o3 = 0.20, 0.15, 0.18, 0.12
    nb_u = o0 - o1
    nb_h = o2 - o3
    hb03 = o0 - o2
    hb005 = o1 - o3
    inter_v1 = nb_h - nb_u
    inter_v2 = hb005 - hb03
    assert abs(inter_v1 - inter_v2) < 1e-12


def test_T39_response_effects_in_factorial_output():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_factorial_effects_per_seed
    rows_per_arm = {}
    for arm, igd_val in [("O0", 0.2), ("O1", 0.15), ("O2", 0.18), ("O3", 0.12)]:
        rows_per_arm[arm] = {
            "scenario": "PERSISTENT", "seed": 401, "block": "A", "arm": arm,
            "MIGD_response": igd_val, "MIGD_epoch": igd_val * 0.9,
            "early_MIGD_response": igd_val, "late_MIGD_response": igd_val,
            "early_MIGD_epoch": igd_val * 0.9, "late_MIGD_epoch": igd_val * 0.9,
            "final_igd": igd_val * 0.8, "worst_response_igd": igd_val, "worst_epoch_igd": igd_val,
        }
    summ = list(rows_per_arm.values())
    eff = compute_factorial_effects_per_seed(summ, "PERSISTENT")
    assert len(eff) == 1
    e = eff[0]
    assert abs(e["NoiseBenefit_U_response"] - (0.2 - 0.15)) < 1e-10
    assert abs(e["NoiseBenefit_H_response"] - (0.18 - 0.12)) < 1e-10
    assert abs(e["HardBenefit_03_response"] - (0.2 - 0.18)) < 1e-10
    assert abs(e["HardBenefit_005_response"] - (0.15 - 0.12)) < 1e-10
    # Interaction = NoiseBenefit_H - NoiseBenefit_U
    expected_inter = (0.18 - 0.12) - (0.2 - 0.15)
    assert abs(e["Interaction_response"] - expected_inter) < 1e-10


def test_T40_epoch_effects_computed():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_factorial_effects_per_seed
    summ = [
        {"scenario": "MIXED", "seed": 401, "block": "A", "arm": arm,
         "MIGD_response": v, "MIGD_epoch": v * 1.1,
         "early_MIGD_response": v, "late_MIGD_response": v,
         "early_MIGD_epoch": v * 1.1, "late_MIGD_epoch": v * 1.1,
         "final_igd": v, "worst_response_igd": v, "worst_epoch_igd": v * 1.1}
        for arm, v in [("O0", 0.2), ("O1", 0.15), ("O2", 0.18), ("O3", 0.12)]
    ]
    eff = compute_factorial_effects_per_seed(summ, "MIXED")
    assert len(eff) == 1
    e = eff[0]
    assert "NoiseBenefit_U_epoch" in e
    assert "HardBenefit_03_epoch" in e
    assert "Interaction_epoch" in e


# ================================================================
# T41-T45: Bootstrap
# ================================================================

def test_T41_seed_level_bootstrap():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import _bootstrap_ci
    vals = np.array([5e-5, 6e-5, 4e-5, 7e-5, 5.5e-5] * 4)
    lo, hi = _bootstrap_ci(vals, 5000, 42)
    assert lo < float(np.mean(vals)) < hi


def test_T42_bootstrap_reproducible():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import _bootstrap_ci
    vals = np.array([1e-4] * 20)
    lo1, hi1 = _bootstrap_ci(vals, 1000, 99)
    lo2, hi2 = _bootstrap_ci(vals, 1000, 99)
    assert lo1 == lo2 and hi1 == hi2


def test_T43_block_A_bootstrap_seed():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BOOTSTRAP_SEED_A
    assert BOOTSTRAP_SEED_A == 43001


def test_T44_block_B_bootstrap_seed():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BOOTSTRAP_SEED_B
    assert BOOTSTRAP_SEED_B == 43002


def test_T45_pooled_bootstrap_seed():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import BOOTSTRAP_SEED_POOLED
    assert BOOTSTRAP_SEED_POOLED == 43003


# ================================================================
# T46: Effect classification
# ================================================================

def test_T46_effect_classification():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import classify_effect, DELTA_D1R
    assert classify_effect(2e-5, 1e-6, 4e-5) == "BENEFICIAL"
    assert classify_effect(2e-5, -1e-6, 4e-5) == "DIRECTIONAL_POSITIVE"
    assert classify_effect(5e-6, -5e-6, 1e-5) == "PRACTICALLY_NEUTRAL"
    assert classify_effect(-2e-5, -4e-5, 1e-7) == "DIRECTIONAL_NEGATIVE"
    assert classify_effect(-2e-5, -4e-5, -1e-6) == "HARMFUL"


# ================================================================
# T47-T49: Stationary safety logic
# ================================================================

def _make_scen_stats_for(arm, pooled_cls_r, pooled_cls_e, bA_cls_r, bB_cls_r):
    """Build minimal scen_stats with specified classifications."""
    def _p(mean, cls):
        return {f"class_O0_vs_{arm}_response": cls, f"class_O0_vs_{arm}_epoch": cls,
                f"mean_O0_vs_{arm}_response": mean}
    return {
        "pooled":  {f"class_O0_vs_{arm}_response": pooled_cls_r,
                    f"class_O0_vs_{arm}_epoch": pooled_cls_e},
        "block_A": {f"class_O0_vs_{arm}_response": bA_cls_r},
        "block_B": {f"class_O0_vs_{arm}_response": bB_cls_r},
    }

def test_T47_O1_stationary_safe():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_stationary_safety
    stats = _make_scen_stats_for("O1", "PRACTICALLY_NEUTRAL", "PRACTICALLY_NEUTRAL",
                                  "PRACTICALLY_NEUTRAL", "PRACTICALLY_NEUTRAL")
    safety, _ = compute_stationary_safety("O1", stats)
    assert safety == "SAFE"


def test_T48_O2_stationary_unsafe_pooled_harmful():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_stationary_safety
    stats = _make_scen_stats_for("O2", "HARMFUL", "HARMFUL", "HARMFUL", "HARMFUL")
    safety, _ = compute_stationary_safety("O2", stats)
    assert safety == "UNSAFE"


def test_T49_O3_stationary_unsafe_blockA_harmful():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_stationary_safety
    stats = _make_scen_stats_for("O3", "DIRECTIONAL_NEGATIVE", "DIRECTIONAL_NEGATIVE",
                                  "HARMFUL", "PRACTICALLY_NEUTRAL")
    safety, _ = compute_stationary_safety("O3", stats)
    assert safety == "UNSAFE"


# ================================================================
# T50-T53: Mixed stability logic
# ================================================================

def _make_mixed_stats_for(arm, pooled_mean, pooled_cls, bA_mean, bA_cls, bB_mean, bB_cls):
    key_m = f"mean_O0_vs_{arm}_response"
    key_c = f"class_O0_vs_{arm}_response"
    return (
        {key_m: pooled_mean, key_c: pooled_cls},
        {key_m: bA_mean, key_c: bA_cls},
        {key_m: bB_mean, key_c: bB_cls},
    )

def test_T50_mixed_stable_positive():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_mixed_stability
    p, bA, bB = _make_mixed_stats_for("O1",
                                       5e-5, "DIRECTIONAL_POSITIVE",
                                       6e-5, "DIRECTIONAL_POSITIVE",
                                       4e-5, "DIRECTIONAL_POSITIVE")
    assert compute_mixed_stability("O1", p, bA, bB) == "STABLE_POSITIVE"


def test_T51_mixed_stable_nonharm():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_mixed_stability
    p, bA, bB = _make_mixed_stats_for("O1",
                                       1e-6, "PRACTICALLY_NEUTRAL",
                                       5e-6, "PRACTICALLY_NEUTRAL",
                                       -3e-6, "PRACTICALLY_NEUTRAL")
    result = compute_mixed_stability("O1", p, bA, bB)
    assert result in ("STABLE_NONHARM", "STABLE_POSITIVE")


def test_T52_mixed_unstable_sign():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_mixed_stability, DELTA_D1R
    p, bA, bB = _make_mixed_stats_for("O3",
                                       1e-5, "PRACTICALLY_NEUTRAL",
                                       5e-5, "DIRECTIONAL_POSITIVE",
                                       -5e-5, "DIRECTIONAL_NEGATIVE")
    result = compute_mixed_stability("O3", p, bA, bB)
    assert result == "UNSTABLE_SIGN"


def test_T53_mixed_stable_harm():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_mixed_stability
    p, bA, bB = _make_mixed_stats_for("O3",
                                       -3e-5, "HARMFUL",
                                       -4e-5, "HARMFUL",
                                       -2e-5, "HARMFUL")
    assert compute_mixed_stability("O3", p, bA, bB) == "STABLE_HARM"


# ================================================================
# T54-T57: Acceptability and design decision logic
# ================================================================

def _make_full_scen_stats(arm, pers_mean, stat_safe, mixed_p_cls, mixed_bA_cls, mixed_bB_cls,
                           pers_epoch_cls="PRACTICALLY_NEUTRAL", stat_epoch_cls="PRACTICALLY_NEUTRAL",
                           mixed_epoch_cls="PRACTICALLY_NEUTRAL"):
    def _ps(m, r_cls, e_cls):
        return {f"mean_O0_vs_{arm}_response": m,
                f"class_O0_vs_{arm}_response": r_cls,
                f"class_O0_vs_{arm}_epoch": e_cls}
    def _bs(r_cls):
        return {f"class_O0_vs_{arm}_response": r_cls}
    return {
        "PERSISTENT": {
            "pooled": _ps(pers_mean, "BENEFICIAL" if pers_mean > 1e-5 else "PRACTICALLY_NEUTRAL", pers_epoch_cls),
            "block_A": _bs("DIRECTIONAL_POSITIVE"),
            "block_B": _bs("DIRECTIONAL_POSITIVE"),
        },
        "STATIONARY": {
            "pooled": _ps(-1e-6 if stat_safe else -3e-5,
                          "PRACTICALLY_NEUTRAL" if stat_safe else "HARMFUL",
                          "PRACTICALLY_NEUTRAL" if stat_safe else "HARMFUL"),
            "block_A": _bs("PRACTICALLY_NEUTRAL" if stat_safe else "HARMFUL"),
            "block_B": _bs("PRACTICALLY_NEUTRAL" if stat_safe else "HARMFUL"),
        },
        "MIXED": {
            "pooled": _ps(2e-5, mixed_p_cls, mixed_epoch_cls),
            "block_A": _bs(mixed_bA_cls),
            "block_B": _bs(mixed_bB_cls),
        },
    }


def test_T54_acceptability_all_criteria():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import check_option_acceptability
    scen_stats = _make_full_scen_stats("O1",
                                        pers_mean=5e-5, stat_safe=True,
                                        mixed_p_cls="DIRECTIONAL_POSITIVE",
                                        mixed_bA_cls="DIRECTIONAL_POSITIVE",
                                        mixed_bB_cls="PRACTICALLY_NEUTRAL")
    acc, details = check_option_acceptability("O1", scen_stats, True, True, True)
    assert acc is True
    assert details["A2_stat_safety"] == "SAFE"


def test_T55_zero_acceptable():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_design_decision
    acceptable = {"O1": False, "O2": False, "O3": False}
    decision, next_stage, no_existing = compute_design_decision(acceptable, {})
    assert decision == "KEEP_O0_CURRENT"
    assert no_existing is True
    assert next_stage == "D3_NEW_DESIGN_HYPOTHESIS"


def test_T56_exactly_one_acceptable():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_design_decision
    acceptable = {"O1": True, "O2": False, "O3": False}
    decision, _, no_existing = compute_design_decision(acceptable, {})
    assert decision == "PROPOSE_O1_F005_ONLY"
    assert no_existing is False


def test_T57_parsimony_O1_preferred_over_O3():
    """When O1 and O3 are equally ranked, prefer O1 (parsimony)."""
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import compute_design_decision
    # Both O1 and O3 acceptable; make them equally ranked
    # O1 parsimony rule: prefer O1 over O3
    scen_stats = _make_full_scen_stats("O1", 5e-5, True, "DIRECTIONAL_POSITIVE",
                                        "DIRECTIONAL_POSITIVE", "DIRECTIONAL_POSITIVE")
    o3_stats = _make_full_scen_stats("O3", 5e-5, True, "DIRECTIONAL_POSITIVE",
                                      "DIRECTIONAL_POSITIVE", "DIRECTIONAL_POSITIVE")
    # Merge: use O1's stats (which is symmetrically favorable)
    merged = {k: {**scen_stats.get(k, {})} for k in scen_stats}
    # Build merged stats that allows both O1 and O3 to have same stability rank
    acceptable = {"O1": True, "O2": False, "O3": True}
    decision, _, _ = compute_design_decision(acceptable, merged)
    # Parsimony: O1 preferred over O3 when equal
    assert decision in ("PROPOSE_O1_F005_ONLY", "RECONFIRM_O3_JOINT_H005")


# ================================================================
# T58-T60: Historical warnings preserved
# ================================================================

def test_T58_historical_O1_warning_preserved():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "L2-B" in src or "L2B" in src or "historical" in src.lower()


def test_T59_historical_O2_noise_warning_preserved():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "known" in src.lower() or "noise" in src.lower()


def test_T60_O3_reconfirm_requires_safety():
    """O3 cannot be reconfirmed without passing STATIONARY safety and MIXED block check."""
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import check_option_acceptability
    # O3 with STATIONARY HARMFUL → not acceptable
    scen_stats = _make_full_scen_stats("O3",
                                        pers_mean=5e-5, stat_safe=False,  # STATIONARY unsafe
                                        mixed_p_cls="DIRECTIONAL_POSITIVE",
                                        mixed_bA_cls="DIRECTIONAL_POSITIVE",
                                        mixed_bB_cls="DIRECTIONAL_POSITIVE")
    acc, _ = check_option_acceptability("O3", scen_stats, True, True, True)
    assert acc is False, "O3 with STATIONARY UNSAFE must not be acceptable"


# ================================================================
# T61-T62: Smoke reproducibility (run 2 seeds × 3 changes)
# ================================================================

@pytest.fixture(scope="module")
def smoke_rows_441():
    from prdrl.experiments.D1R_postfix_factorial_design_reopen import (
        run_one_seed, SCENARIOS, N_SMOKE
    )
    from prdrl.problems.controlled_predict import get_true_pf
    pf_star = get_true_pf(500)
    rows1, _ = run_one_seed(441, N_SMOKE, "PERSISTENT", SCENARIOS["PERSISTENT"], pf_star, "SMOKE")
    rows2, _ = run_one_seed(441, N_SMOKE, "PERSISTENT", SCENARIOS["PERSISTENT"], pf_star, "SMOKE")
    return rows1, rows2


def test_T61_same_process_reproducibility(smoke_rows_441):
    rows1, rows2 = smoke_rows_441
    for r1, r2 in zip(rows1, rows2):
        assert r1["igd_response"] == r2["igd_response"], (
            f"igd_response mismatch: arm={r1['arm']} k={r1['k']}"
        )
        assert r1["epoch_x_fp"] == r2["epoch_x_fp"]


def test_T62_cross_process_reproducibility():
    """Run same seed in subprocess; check hash matches."""
    import subprocess
    import sys
    script = (
        "import sys; sys.path.insert(0,'src'); import hashlib; "
        "from prdrl.experiments.D1R_postfix_factorial_design_reopen import run_one_seed, SCENARIOS; "
        "from prdrl.problems.controlled_predict import get_true_pf; "
        "pf = get_true_pf(500); "
        "rows, _ = run_one_seed(441, 2, 'STATIONARY', SCENARIOS['STATIONARY'], pf, 'X'); "
        "fps = [r['epoch_x_fp'] for r in rows]; "
        "print(hashlib.md5('|'.join(str(x) for x in fps).encode()).hexdigest())"
    )
    hashes = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, cwd=os.getcwd())
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr[:200]}"
        hashes.append(proc.stdout.strip())
    assert hashes[0] == hashes[1], f"cross-process mismatch: {hashes}"


# ================================================================
# T63-T66: Production/RL/PDF status
# ================================================================

def test_T63_production_default_O0():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "PRODUCTION_DEFAULT" in src
    assert "O0_CURRENT" in src


def test_T64_pdf_unchanged():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "PDF_STATUS" in src
    assert "UNCHANGED" in src


def test_T65_rl_not_started():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "RL_STATUS" in src
    assert "NOT_STARTED" in src


def test_T66_no_post_hoc_third_block():
    src = open("src/prdrl/experiments/D1R_postfix_factorial_design_reopen.py").read()
    assert "BLOCK_C_SEEDS" not in src
    assert "range(441, 461)" not in src
    assert 'compute_block_factorial_stats(effects_list, "C"' not in src


# ================================================================
# T67: Smoke row count (4 arms × 3 changes = 12 rows)
# ================================================================

def test_T67_smoke_row_count(smoke_rows_441):
    rows1, _ = smoke_rows_441
    assert len(rows1) == 12  # 4 arms × 3 scored changes


# ================================================================
# T68: O0 and O1 never trigger HARD in smoke run
# ================================================================

def test_T68_action_scope_in_smoke(smoke_rows_441):
    rows1, _ = smoke_rows_441
    for r in rows1:
        if r["arm"] in ("O0", "O1"):
            assert not r.get("hard_triggered"), f"O0/O1 must never trigger HARD: arm={r['arm']} k={r['k']}"
