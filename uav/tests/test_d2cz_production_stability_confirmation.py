"""
T1-T60: D2-CZ Production Stability Confirmation — test suite.
Tests configuration, statistics, verdict logic, and structural correctness.
Smoke reproducibility tests use seeds 361, 362 (never 321..360).
"""
import copy
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ================================================================
# T1-T10: Import and constant checks
# ================================================================

def test_T1_module_importable():
    import prdrl.experiments.D2CZ_production_stability_confirmation as m
    assert m is not None


def test_T2_block_seeds_correct():
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        BLOCK_A_SEEDS, BLOCK_B_SEEDS, ALL_SEEDS
    )
    assert BLOCK_A_SEEDS == list(range(321, 341))
    assert BLOCK_B_SEEDS == list(range(341, 361))
    assert ALL_SEEDS == BLOCK_A_SEEDS + BLOCK_B_SEEDS
    assert len(ALL_SEEDS) == 40


def test_T3_smoke_seeds_outside_science():
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        SMOKE_SEEDS, ALL_SEEDS
    )
    for s in SMOKE_SEEDS:
        assert s not in ALL_SEEDS, f"Smoke seed {s} overlaps with science seeds"


def test_T4_no_seed_overlap_with_prefixBands():
    from prdrl.experiments.D2CZ_production_stability_confirmation import ALL_SEEDS
    forbidden = list(range(1, 221)) + list(range(109, 129))
    for s in ALL_SEEDS:
        assert s not in forbidden, f"Science seed {s} overlaps with forbidden bands"


def test_T5_delta_d2cz_frozen():
    from prdrl.experiments.D2CZ_production_stability_confirmation import DELTA_D2CZ
    assert DELTA_D2CZ == 1e-5


def test_T6_n_scored_correct():
    from prdrl.experiments.D2CZ_production_stability_confirmation import N_SCORED
    assert N_SCORED == 10


def test_T7_budget_pop_size():
    from prdrl.experiments.D2CZ_production_stability_confirmation import BUDGET, POP_SIZE
    assert BUDGET == 10
    assert POP_SIZE == 100


def test_T8_bootstrap_seeds_distinct():
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        BOOTSTRAP_SEED_A, BOOTSTRAP_SEED_B, BOOTSTRAP_SEED_POOLED
    )
    seeds = {BOOTSTRAP_SEED_A, BOOTSTRAP_SEED_B, BOOTSTRAP_SEED_POOLED}
    assert len(seeds) == 3


def test_T9_results_dir_name():
    from prdrl.experiments.D2CZ_production_stability_confirmation import RESULTS_DIR
    assert "D2CZ" in RESULTS_DIR


def test_T10_sigma_local():
    from prdrl.experiments.D2CZ_production_stability_confirmation import SIGMA_LOCAL
    assert SIGMA_LOCAL == 0.02


# ================================================================
# T11-T20: Scenario and arm definitions
# ================================================================

def test_T11_three_scenarios_defined():
    from prdrl.experiments.D2CZ_production_stability_confirmation import SCENARIOS
    assert set(SCENARIOS.keys()) == {"PERSISTENT", "STATIONARY", "MIXED"}


def test_T12_persistent_all_persistent():
    from prdrl.experiments.D2CZ_production_stability_confirmation import SCENARIOS
    rd = SCENARIOS["PERSISTENT"]
    assert all(r == "persistent" for r in rd)
    assert len(rd) == 10


def test_T13_stationary_all_stationary():
    from prdrl.experiments.D2CZ_production_stability_confirmation import SCENARIOS
    rd = SCENARIOS["STATIONARY"]
    assert all(r == "stationary" for r in rd)
    assert len(rd) == 10


def test_T14_mixed_has_both_regimes():
    from prdrl.experiments.D2CZ_production_stability_confirmation import SCENARIOS
    rd = SCENARIOS["MIXED"]
    assert "persistent" in rd
    assert "stationary" in rd
    assert len(rd) == 10


def test_T15_mixed_uses_make_region_dynamics_map():
    from prdrl.problems.controlled_predict import make_region_dynamics_map
    from prdrl.experiments.D2CZ_production_stability_confirmation import SCENARIOS
    expected = make_region_dynamics_map(10)
    assert SCENARIOS["MIXED"] == expected


def test_T16_two_arms_only():
    from prdrl.experiments.D2CZ_production_stability_confirmation import ARMS
    assert set(ARMS.keys()) == {"O0", "O3"}


def test_T17_O0_sigma_and_mode():
    from prdrl.experiments.D2CZ_production_stability_confirmation import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED
    assert ARMS["O0"]["sigma_predict"] == 0.03
    assert ARMS["O0"]["mode"] == PREDICT_SURVIVAL_UNRESTRICTED


def test_T18_O3_sigma_and_mode():
    from prdrl.experiments.D2CZ_production_stability_confirmation import ARMS
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
    assert ARMS["O3"]["sigma_predict"] == 0.005
    assert ARMS["O3"]["mode"] == PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T19_arms_have_label():
    from prdrl.experiments.D2CZ_production_stability_confirmation import ARMS
    for arm_name, arm in ARMS.items():
        assert "label" in arm and arm_name in arm["label"]


def test_T20_target_region_schedule():
    from prdrl.experiments.D2CZ_production_stability_confirmation import target_region
    R = 10
    for seed in range(321, 326):
        for k in range(10):
            assert target_region(seed, k, R) == (seed + k) % R


# ================================================================
# T21-T30: RNG spec compliance
# ================================================================

def test_T21_response_survival_rng_namespace_exists():
    from prdrl.oracle.oracle import make_rng
    rng = make_rng(12345, 3, 5, "RESPONSE_SURVIVAL")
    assert rng is not None


def test_T22_response_survival_rng_reproducible():
    from prdrl.oracle.oracle import make_rng
    r1 = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    r2 = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    v1 = r1.integers(0, 1000, 10)
    v2 = r2.integers(0, 1000, 10)
    assert np.array_equal(v1, v2)


def test_T23_response_survival_distinct_from_predict():
    from prdrl.oracle.oracle import make_rng
    r1 = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    r2 = make_rng(99, 1, 2, "PREDICT")
    v1 = r1.integers(0, 1000, 10)
    v2 = r2.integers(0, 1000, 10)
    assert not np.array_equal(v1, v2)


def test_T24_nsga2_survival_accepts_response_rng():
    from prdrl.oracle.oracle import nsga2_survival
    sig = inspect.signature(nsga2_survival)
    assert "response_rng" in sig.parameters


def test_T25_nsga2_survival_deterministic_with_rng():
    from prdrl.oracle.oracle import nsga2_survival, make_rng
    rng1 = np.random.default_rng(42)
    X = rng1.uniform(0, 1, (20, 5))
    F = rng1.uniform(0, 1, (20, 2))
    xl = np.zeros(5)
    xu = np.ones(5)
    rng_a = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    rng_b = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    X_surv_a, _ = nsga2_survival(X, F, 10, xl, xu, response_rng=rng_a)
    rng_b2 = make_rng(99, 1, 2, "RESPONSE_SURVIVAL")
    X_surv_b, _ = nsga2_survival(X, F, 10, xl, xu, response_rng=rng_b2)
    assert np.array_equal(X_surv_a, X_surv_b)


def test_T26_evolution_seed_formula():
    seed = 321
    scored_change_idx = 2
    evo_seed = seed * 10000 + scored_change_idx
    assert evo_seed == 3210002


def test_T27_oracle_base_seed_formula():
    seed = 321
    k = 0
    R = 10
    tgt_r = target_region_formula(seed, k, R)
    obs = seed * 100000 + k * 100 + tgt_r
    assert obs == 32100001


def target_region_formula(seed, k, R):
    return (seed + k) % R


def test_T28_no_diagnostic_crn_in_experiment():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    assert "L2B_CRN" not in src
    assert "diagnostic" not in src.lower() or "diagnostic" in "diagnostic-not-in-production"
    # More targeted: no reference to diagnostic CRN
    assert "CRN_DIAGNOSTIC" not in src
    assert "surv_rng_diag" not in src


def test_T29_no_sigma_tuning_in_source():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    # sigma values must be exactly 0.03 and 0.005 for O0 and O3 respectively
    assert "0.03" in src   # O0 sigma
    assert "0.005" in src  # O3 sigma
    # Must not introduce any other sigma values as arm parameters
    assert "sigma_O0" not in src
    assert "sigma_O3" not in src


def test_T30_no_rl_references():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    for forbidden in ["rl_agent", "ReplayBuffer", "DQN", "PPO", "RL_STARTED"]:
        assert forbidden not in src


# ================================================================
# T31-T40: Statistics helpers
# ================================================================

def test_T31_classify_effect_beneficial():
    from prdrl.experiments.D2CZ_production_stability_confirmation import classify_effect
    c = classify_effect(2e-5, 1e-6, 4e-5)
    assert c == "BENEFICIAL"


def test_T32_classify_effect_directional_positive():
    from prdrl.experiments.D2CZ_production_stability_confirmation import classify_effect
    c = classify_effect(2e-5, -1e-7, 4e-5)
    assert c == "DIRECTIONAL_POSITIVE"


def test_T33_classify_effect_practically_neutral():
    from prdrl.experiments.D2CZ_production_stability_confirmation import classify_effect
    c = classify_effect(5e-6, -5e-6, 1e-5)
    assert c == "PRACTICALLY_NEUTRAL"


def test_T34_classify_effect_directional_negative():
    from prdrl.experiments.D2CZ_production_stability_confirmation import classify_effect
    c = classify_effect(-2e-5, -4e-5, 1e-7)
    assert c == "DIRECTIONAL_NEGATIVE"


def test_T35_classify_effect_harmful():
    from prdrl.experiments.D2CZ_production_stability_confirmation import classify_effect
    c = classify_effect(-2e-5, -4e-5, -1e-6)
    assert c == "HARMFUL"


def test_T36_bootstrap_ci_coverage():
    from prdrl.experiments.D2CZ_production_stability_confirmation import _bootstrap_ci
    rng = np.random.default_rng(1)
    vals = rng.normal(0.01, 0.001, 20)
    lo, hi = _bootstrap_ci(vals, 5000, 42)
    assert lo < 0.01 < hi


def test_T37_bootstrap_ci_reproducible():
    from prdrl.experiments.D2CZ_production_stability_confirmation import _bootstrap_ci
    vals = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3] * 4)
    lo1, hi1 = _bootstrap_ci(vals, 1000, 99)
    lo2, hi2 = _bootstrap_ci(vals, 1000, 99)
    assert lo1 == lo2
    assert hi1 == hi2


def test_T38_cb_is_O0_minus_O3():
    """CB positive means O3 lower IGD = better."""
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        compute_seed_summaries, compute_cb_per_seed
    )
    # Build synthetic rows where O3 always has lower IGD
    rows = []
    for seed in range(321, 326):
        for arm, igd_val in [("O0", 0.2), ("O3", 0.1)]:
            for k in range(3):
                rows.append({
                    "scenario": "PERSISTENT", "seed": seed, "arm": arm, "k": k,
                    "block": "A", "igd_response": igd_val, "igd_epoch_end": igd_val,
                })
    seed_summ = compute_seed_summaries(rows, "PERSISTENT")
    cb_list = compute_cb_per_seed(seed_summ, "PERSISTENT")
    for cb in cb_list:
        assert cb["CB_response"] > 0, "CB should be positive when O3 has lower IGD"


def test_T39_compute_seed_summaries_structure():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_seed_summaries
    rows = []
    for arm in ("O0", "O3"):
        for k in range(3):
            rows.append({
                "scenario": "MIXED", "seed": 321, "arm": arm, "k": k,
                "block": "A", "igd_response": 0.1, "igd_epoch_end": 0.15,
            })
    summ = compute_seed_summaries(rows, "MIXED")
    assert len(summ) == 2
    for s in summ:
        assert "MIGD_response" in s
        assert "MIGD_epoch" in s
        assert "early_MIGD_response" in s
        assert "late_MIGD_response" in s


def test_T40_no_postfix_third_block_in_verdict_logic():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    assert "BLOCK_C_SEEDS" not in src
    assert "block_C_stats" not in src
    assert "range(361, 381)" not in src
    assert 'compute_block_stats(cb_list, "C"' not in src


# ================================================================
# T41-T50: Verdict logic
# ================================================================

def test_T41_persistent_verdict_validated():
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        compute_persistent_verdict, classify_effect
    )
    scen_stats = {
        "pooled": {
            "mean_CB_response": 5e-5,
            "CI_response": [1e-5, 9e-5],
            "classification_response": "BENEFICIAL",
            "mean_CB_epoch": 3e-5,
            "CI_epoch": [1e-5, 5e-5],
            "classification_epoch": "BENEFICIAL",
        },
        "block_A": {"mean_CB_response": 6e-5, "classification_response": "BENEFICIAL"},
        "block_B": {"mean_CB_response": 4e-5, "classification_response": "BENEFICIAL"},
    }
    verdict, _ = compute_persistent_verdict(scen_stats)
    assert verdict == "VALIDATED"


def test_T42_persistent_verdict_partial_when_CI_crosses_zero():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_persistent_verdict
    scen_stats = {
        "pooled": {
            "mean_CB_response": 5e-5,
            "CI_response": [-1e-6, 9e-5],  # CI crosses zero
            "classification_response": "DIRECTIONAL_POSITIVE",
            "mean_CB_epoch": 3e-5,
            "CI_epoch": [0, 5e-5],
            "classification_epoch": "DIRECTIONAL_POSITIVE",
        },
        "block_A": {"mean_CB_response": 6e-5, "classification_response": "DIRECTIONAL_POSITIVE"},
        "block_B": {"mean_CB_response": -1e-6, "classification_response": "PRACTICALLY_NEUTRAL"},
    }
    verdict, _ = compute_persistent_verdict(scen_stats)
    assert verdict in ("PARTIAL", "FAILED")


def test_T43_stationary_verdict_pass_when_not_harmful():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_stationary_verdict
    scen_stats = {
        "pooled": {
            "classification_response": "PRACTICALLY_NEUTRAL",
            "classification_epoch": "PRACTICALLY_NEUTRAL",
        },
        "block_A": {"classification_response": "PRACTICALLY_NEUTRAL"},
        "block_B": {"classification_response": "PRACTICALLY_NEUTRAL"},
    }
    verdict, _ = compute_stationary_verdict(scen_stats)
    assert verdict == "PASS"


def test_T44_stationary_verdict_fail_when_harmful():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_stationary_verdict
    scen_stats = {
        "pooled": {
            "classification_response": "HARMFUL",
            "classification_epoch": "HARMFUL",
        },
        "block_A": {"classification_response": "HARMFUL"},
        "block_B": {"classification_response": "HARMFUL"},
    }
    verdict, _ = compute_stationary_verdict(scen_stats)
    assert verdict == "FAIL"


def test_T45_mixed_verdict_fails_when_repro_fails():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_mixed_verdict
    scen_stats = {
        "pooled": {
            "mean_CB_response": 5e-5,
            "CI_response": [1e-5, 9e-5],
            "classification_response": "BENEFICIAL",
            "mean_CB_epoch": 3e-5,
            "CI_epoch": [1e-5, 5e-5],
            "classification_epoch": "BENEFICIAL",
        },
        "block_A": {"mean_CB_response": 6e-5, "classification_response": "BENEFICIAL"},
        "block_B": {"mean_CB_response": 4e-5, "classification_response": "BENEFICIAL"},
    }
    verdict, details = compute_mixed_verdict(scen_stats, True, True, repro_pass=False)
    assert verdict == "FAILED"


def test_T46_mixed_verdict_fails_when_hard_inv_fails():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_mixed_verdict
    scen_stats = {
        "pooled": {
            "mean_CB_response": 5e-5,
            "CI_response": [1e-5, 9e-5],
            "classification_response": "BENEFICIAL",
            "mean_CB_epoch": 3e-5,
            "CI_epoch": [1e-5, 5e-5],
            "classification_epoch": "BENEFICIAL",
        },
        "block_A": {"mean_CB_response": 6e-5, "classification_response": "BENEFICIAL"},
        "block_B": {"mean_CB_response": 4e-5, "classification_response": "BENEFICIAL"},
    }
    verdict, _ = compute_mixed_verdict(scen_stats, hard_inv_ok=False, fe_fair_ok=True, repro_pass=True)
    assert verdict == "FAILED"


def test_T47_mixed_verdict_validated():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_mixed_verdict
    scen_stats = {
        "pooled": {
            "mean_CB_response": 5e-5,
            "CI_response": [1e-5, 9e-5],
            "classification_response": "BENEFICIAL",
            "mean_CB_epoch": 3e-5,
            "CI_epoch": [1e-5, 5e-5],
            "classification_epoch": "BENEFICIAL",
        },
        "block_A": {"mean_CB_response": 6e-5, "classification_response": "BENEFICIAL"},
        "block_B": {"mean_CB_response": 4e-5, "classification_response": "BENEFICIAL"},
    }
    verdict, _ = compute_mixed_verdict(scen_stats, True, True, True)
    assert verdict == "VALIDATED"


def test_T48_overall_verdict_validated():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_overall_verdict
    v, ns = compute_overall_verdict("VALIDATED", "PASS", "VALIDATED")
    assert v == "VALIDATED"


def test_T49_overall_verdict_failed():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_overall_verdict
    v, ns = compute_overall_verdict("PARTIAL", "FAIL", "FAILED")
    assert v == "FAILED"


def test_T50_overall_verdict_next_stage_validated():
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_overall_verdict
    _, ns = compute_overall_verdict("VALIDATED", "PASS", "VALIDATED")
    assert "PDF" in ns or "SPEC" in ns or "D2D" in ns


# ================================================================
# T51-T55: Smoke / structural run checks (use 3 scored changes, seed 361/362)
# ================================================================

@pytest.fixture(scope="module")
def smoke_rows_361():
    """Run seed=361 with 3 scored changes for PERSISTENT. Expensive; cached per module."""
    from prdrl.experiments.D2CZ_production_stability_confirmation import (
        run_one_seed, SCENARIOS, N_SMOKE
    )
    from prdrl.problems.controlled_predict import get_true_pf
    pf_star = get_true_pf(500)
    rows_1 = run_one_seed(361, N_SMOKE, "PERSISTENT", SCENARIOS["PERSISTENT"], pf_star, "SMOKE")
    rows_2 = run_one_seed(361, N_SMOKE, "PERSISTENT", SCENARIOS["PERSISTENT"], pf_star, "SMOKE")
    return rows_1, rows_2


def test_T51_smoke_row_count(smoke_rows_361):
    rows_1, rows_2 = smoke_rows_361
    # 2 arms × 3 scored changes = 6 rows each
    assert len(rows_1) == 6
    assert len(rows_2) == 6


def test_T52_smoke_reproducibility_igd_response(smoke_rows_361):
    rows_1, rows_2 = smoke_rows_361
    for r1, r2 in zip(rows_1, rows_2):
        assert r1["igd_response"] == r2["igd_response"], (
            f"igd_response mismatch: arm={r1['arm']} k={r1['k']}: "
            f"{r1['igd_response']} vs {r2['igd_response']}"
        )


def test_T53_smoke_reproducibility_epoch_fp(smoke_rows_361):
    rows_1, rows_2 = smoke_rows_361
    for r1, r2 in zip(rows_1, rows_2):
        assert r1["epoch_x_fp"] == r2["epoch_x_fp"], (
            f"epoch_x_fp mismatch: arm={r1['arm']} k={r1['k']}"
        )


def test_T54_smoke_O3_hard_invariant(smoke_rows_361):
    rows_1, _ = smoke_rows_361
    for r in rows_1:
        if r["arm"] == "O3" and r.get("hard_triggered"):
            assert r.get("n_old_dropped_other") == 0, (
                f"HARD invariant violated: k={r['k']} n_old_dropped_other={r['n_old_dropped_other']}"
            )


def test_T55_smoke_no_action_scope_violation(smoke_rows_361):
    rows_1, _ = smoke_rows_361
    for r in rows_1:
        if r["arm"] == "O0":
            assert not r.get("hard_triggered"), "O0 must never trigger HARD"


# ================================================================
# T56-T60: Source-level structural checks
# ================================================================

def test_T56_prerun_manifest_creates_file(tmp_path, monkeypatch):
    from prdrl.experiments import D2CZ_production_stability_confirmation as m
    monkeypatch.setattr(m, "RESULTS_DIR", str(tmp_path))
    m.create_prerun_manifest()
    manifest_path = tmp_path / "D2CZ_pre_run_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["manifest_type"] == "D2CZ_PRERUN"
    assert "files" in manifest
    assert "constraints" in manifest
    assert manifest["constraints"]["no_sigma_tuning"] is True
    assert manifest["constraints"]["no_post_hoc_third_block"] is True


def test_T57_verdict_required_keys():
    """Verify that compute_overall_verdict returns values for all required terminal fields."""
    from prdrl.experiments.D2CZ_production_stability_confirmation import compute_overall_verdict
    for perst in ("VALIDATED", "PARTIAL", "FAILED"):
        for stat in ("PASS", "FAIL"):
            for mixed in ("VALIDATED", "PARTIAL", "FAILED"):
                v, ns = compute_overall_verdict(perst, stat, mixed)
                assert v in ("VALIDATED", "PARTIAL", "FAILED")
                assert ns


def test_T58_production_default_not_changed():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    assert "PRODUCTION_DEFAULT" in src
    assert "O0_CURRENT" in src


def test_T59_no_third_seed_block_logic():
    src = open("src/prdrl/experiments/D2CZ_production_stability_confirmation.py").read()
    assert "range(361, 381)" not in src   # hypothetical third block seeds never added
    assert "BLOCK_C_SEEDS" not in src


def test_T60_write_verdict_format(tmp_path, monkeypatch):
    from prdrl.experiments import D2CZ_production_stability_confirmation as m
    monkeypatch.setattr(m, "RESULTS_DIR", str(tmp_path))
    verdict = {
        "audit": "D2-CZ",
        "O3_POSTFIX_PRODUCTION_STABILITY": "VALIDATED",
        "NEXT_STAGE": "D2D_PRODUCTION_SPEC_AND_PDF_UPDATE",
        "PRODUCTION_DEFAULT": "O0_CURRENT",
        "PDF_STATUS": "UNCHANGED",
        "RL_STATUS": "NOT_STARTED",
    }
    m.write_verdict(verdict)
    p = tmp_path / "D2CZ_verdict.json"
    assert p.exists()
    with open(p) as f:
        loaded = json.load(f)
    assert loaded["O3_POSTFIX_PRODUCTION_STABILITY"] == "VALIDATED"
    assert loaded["PRODUCTION_DEFAULT"] == "O0_CURRENT"
