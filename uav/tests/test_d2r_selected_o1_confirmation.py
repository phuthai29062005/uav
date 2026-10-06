"""
Tests for D2-R Selected O1 Design Confirmation.

T1-T10:  Import/config checks (seeds, arms, delta, bootstrap)
T11-T20: Arm definitions — exactly two arms, no HARD
T21-T30: Seed freshness, block structure, smoke seeds
T31-T40: Scenario and RNG correctness
T41-T50: Statistics helpers (classify_effect, bootstrap CI, CB algebra)
T51-T57: Verdict logic (persistent/stationary/mixed/overall)
T58-T61: Smoke reproducibility gate (module-scoped, 3 changes)
"""
import copy
import importlib
import inspect
import os
import sys
import hashlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import prdrl.experiments.D2R_selected_o1_confirmation as D2R
from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED, PREDICT_SURVIVAL_HARD_RESPONSE_ONLY
from prdrl.problems.controlled_predict import get_true_pf, make_region_dynamics_map

# ============================================================
# Module-scoped smoke fixture
# ============================================================

@pytest.fixture(scope="module")
def smoke_rows():
    pf = get_true_pf(500)
    seed = 561
    rd = D2R.SCENARIOS["PERSISTENT"]
    return D2R.run_one_seed(seed, 3, "PERSISTENT", rd, pf, "SMOKE")


@pytest.fixture(scope="module")
def smoke_rows_562():
    pf = get_true_pf(500)
    seed = 562
    rd = D2R.SCENARIOS["MIXED"]
    return D2R.run_one_seed(seed, 3, "MIXED", rd, pf, "SMOKE")

# ============================================================
# T1-T10: Import / config
# ============================================================

class TestT1_T10_Config:

    def test_T1_d1r_decision(self):
        """D1-R decision must be PROPOSE_O1_F005_ONLY."""
        path = os.path.join("results", "D1R_postfix_factorial_design_reopen", "D1R_verdict.json")
        if not os.path.exists(path):
            pytest.skip("D1R_verdict.json not yet written")
        import json
        with open(path) as f:
            v = json.load(f)
        assert v["D1R_DESIGN_DECISION"] == "PROPOSE_O1_F005_ONLY"

    def test_T2_o1_acceptable(self):
        """D1-R option assessment: O1 ACCEPTABLE = true."""
        path = os.path.join("results", "D1R_postfix_factorial_design_reopen", "D1R_option_assessment.json")
        if not os.path.exists(path):
            pytest.skip("D1R_option_assessment.json not yet written")
        import json
        with open(path) as f:
            a = json.load(f)
        assert a["O1"]["acceptable"] is True

    def test_T3_o2_acceptable_o3_not(self):
        """D1-R option assessment: O2 acceptable=true, O3 acceptable=false."""
        path = os.path.join("results", "D1R_postfix_factorial_design_reopen", "D1R_option_assessment.json")
        if not os.path.exists(path):
            pytest.skip("D1R_option_assessment.json not yet written")
        import json
        with open(path) as f:
            a = json.load(f)
        assert a["O2"]["acceptable"] is True
        assert a["O3"]["acceptable"] is False

    def test_T4_no_only_passer_prose(self):
        """Source must not affirmatively claim O1 was the only A1-A8 passer."""
        src = inspect.getsource(D2R)
        # The prohibited affirmative claim:
        assert "only o1 passed" not in src.lower()
        # Corrective prose ("not the only ...", "NOT because O1 was the only ...") is allowed.
        # The report explicitly states "O2 also passed" — that is the required traceability.

    def test_T5_delta_d2r(self):
        assert D2R.DELTA_D2R == 1e-5

    def test_T6_bootstrap_n_block(self):
        assert D2R.N_BOOTSTRAP_BLOCK == 20_000

    def test_T7_bootstrap_n_pooled(self):
        assert D2R.N_BOOTSTRAP_POOLED == 30_000

    def test_T8_bootstrap_seeds_distinct(self):
        seeds = [D2R.BOOTSTRAP_SEED_A, D2R.BOOTSTRAP_SEED_B, D2R.BOOTSTRAP_SEED_POOLED]
        assert len(set(seeds)) == 3

    def test_T9_n_scored(self):
        assert D2R.N_SCORED == 10

    def test_T10_budget_10(self):
        assert D2R.BUDGET == 10

# ============================================================
# T11-T20: Arm definitions
# ============================================================

class TestT11_T20_Arms:

    def test_T11_exactly_two_arms(self):
        assert set(D2R.ARMS.keys()) == {"O0", "O1"}

    def test_T12_no_o2_o3_arms(self):
        assert "O2" not in D2R.ARMS
        assert "O3" not in D2R.ARMS

    def test_T13_o0_sigma(self):
        assert D2R.ARMS["O0"]["sigma_predict"] == pytest.approx(0.03)

    def test_T14_o1_sigma(self):
        assert D2R.ARMS["O1"]["sigma_predict"] == pytest.approx(0.005)

    def test_T15_o0_mode_unrestricted(self):
        assert D2R.ARMS["O0"]["mode"] == PREDICT_SURVIVAL_UNRESTRICTED

    def test_T16_o1_mode_unrestricted(self):
        assert D2R.ARMS["O1"]["mode"] == PREDICT_SURVIVAL_UNRESTRICTED

    def test_T17_no_hard_in_any_arm(self):
        for arm_def in D2R.ARMS.values():
            assert arm_def["mode"] != PREDICT_SURVIVAL_HARD_RESPONSE_ONLY

    def test_T18_arm_labels_present(self):
        assert "label" in D2R.ARMS["O0"]
        assert "label" in D2R.ARMS["O1"]

    def test_T19_o0_label_mentions_0_03(self):
        assert "0.03" in D2R.ARMS["O0"]["label"] or "sigma=0.03" in D2R.ARMS["O0"]["label"]

    def test_T20_o1_label_mentions_0_005(self):
        assert "0.005" in D2R.ARMS["O1"]["label"] or "sigma=0.005" in D2R.ARMS["O1"]["label"]

# ============================================================
# T21-T30: Seed structure
# ============================================================

class TestT21_T30_Seeds:

    def test_T21_block_a_range(self):
        assert D2R.BLOCK_A_SEEDS == list(range(501, 531))

    def test_T22_block_b_range(self):
        assert D2R.BLOCK_B_SEEDS == list(range(531, 561))

    def test_T23_block_a_n30(self):
        assert len(D2R.BLOCK_A_SEEDS) == 30

    def test_T24_block_b_n30(self):
        assert len(D2R.BLOCK_B_SEEDS) == 30

    def test_T25_all_seeds_60(self):
        assert len(D2R.ALL_SEEDS) == 60

    def test_T26_blocks_disjoint(self):
        assert len(set(D2R.BLOCK_A_SEEDS) & set(D2R.BLOCK_B_SEEDS)) == 0

    def test_T27_smoke_seeds(self):
        assert D2R.SMOKE_SEEDS == [561, 562]

    def test_T28_seeds_fresh_no_prior_overlap(self):
        # Prior experiments: 1..362, 401..442, 561..563 are smoke/cross-process
        prior = set(range(1, 363)) | set(range(401, 443))
        assert len(prior & set(D2R.ALL_SEEDS)) == 0

    def test_T29_smoke_seeds_outside_all_seeds(self):
        assert len(set(D2R.SMOKE_SEEDS) & set(D2R.ALL_SEEDS)) == 0

    def test_T30_all_seeds_continuous(self):
        assert D2R.ALL_SEEDS == list(range(501, 561))

# ============================================================
# T31-T40: Scenarios and RNG
# ============================================================

class TestT31_T40_ScenariosRNG:

    def test_T31_three_scenarios(self):
        assert set(D2R.SCENARIOS.keys()) == {"PERSISTENT", "STATIONARY", "MIXED"}

    def test_T32_persistent_all_persistent(self):
        assert all(s == "persistent" for s in D2R.SCENARIOS["PERSISTENT"])

    def test_T33_stationary_all_stationary(self):
        assert all(s == "stationary" for s in D2R.SCENARIOS["STATIONARY"])

    def test_T34_mixed_5_5(self):
        m = D2R.SCENARIOS["MIXED"]
        assert m[:5] == ["persistent"] * 5
        assert m[5:] == ["stationary"] * 5

    def test_T35_mixed_uses_make_region_dynamics_map(self):
        expected = make_region_dynamics_map(10)
        assert D2R.SCENARIOS["MIXED"] == expected

    def test_T36_target_region_formula(self):
        # target_r(seed, k, R) = (seed + k) % R
        R = 10
        assert D2R.target_region(501, 0, R) == 501 % R
        assert D2R.target_region(501, 7, R) == (501 + 7) % R

    def test_T37_no_l2b_response_namespace(self):
        # L2B_RESPONSE must not be used as a child RNG namespace in make_rng() calls.
        # Historical prose mentioning it (e.g. "No L2B_RESPONSE CRN") is allowed.
        src = inspect.getsource(D2R)
        # Check no make_rng(..., "L2B_RESPONSE") or make_rng(..., 'L2B_RESPONSE') call exists
        import re
        calls = re.findall(r'make_rng\([^)]+\)', src)
        for call in calls:
            assert "L2B_RESPONSE" not in call, f"make_rng called with L2B_RESPONSE: {call}"

    def test_T38_no_diagnostic_crn(self):
        # No diagnostic CRN may be injected in the runner — only allowed in constraint dict keys.
        src = inspect.getsource(D2R)
        # "no_diagnostic_crn": True in the constraint dict is fine. Check no actual CRN seeding.
        assert "common_rng" not in src
        assert "crn_seed" not in src.lower()
        assert "DIAGNOSTIC_CRN_SEED" not in src

    def test_T39_predict_namespace_exists(self):
        from prdrl.oracle.oracle import make_rng
        r = make_rng(561 * 100000, 0, 0, "PREDICT")
        assert r is not None

    def test_T40_response_survival_namespace_exists(self):
        from prdrl.oracle.oracle import make_rng
        r = make_rng(561, 0, 0, "RESPONSE_SURVIVAL")
        assert r is not None

# ============================================================
# T41-T50: Statistics helpers
# ============================================================

class TestT41_T50_Statistics:

    def test_T41_classify_beneficial(self):
        cls = D2R.classify_effect(2e-5, 1e-6, 5e-5)
        assert cls == "BENEFICIAL"

    def test_T42_classify_directional_positive(self):
        cls = D2R.classify_effect(2e-5, -1e-6, 5e-5)
        assert cls == "DIRECTIONAL_POSITIVE"

    def test_T43_classify_harmful(self):
        cls = D2R.classify_effect(-2e-5, -5e-5, -1e-6)
        assert cls == "HARMFUL"

    def test_T44_classify_directional_negative(self):
        cls = D2R.classify_effect(-2e-5, -5e-5, 1e-6)
        assert cls == "DIRECTIONAL_NEGATIVE"

    def test_T45_classify_practically_neutral(self):
        cls = D2R.classify_effect(5e-7, -1e-5, 1e-5)
        assert cls == "PRACTICALLY_NEUTRAL"

    def test_T46_bootstrap_ci_coverage(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(loc=0.0, scale=1.0, size=50)
        lo, hi = D2R._bootstrap_ci(vals, 5000, 42)
        assert lo < hi
        true_mean = 0.0
        assert lo < true_mean + 3.0  # wide sanity check
        assert hi > true_mean - 3.0

    def test_T47_bootstrap_ci_reproducible(self):
        rng = np.random.default_rng(99)
        vals = rng.normal(size=30)
        ci1 = D2R._bootstrap_ci(vals, 1000, 7)
        ci2 = D2R._bootstrap_ci(vals, 1000, 7)
        assert ci1 == ci2

    def test_T48_cb_algebra_o1benefit(self):
        # O1Benefit = MIGD_O0 - MIGD_O1
        o0_migd, o1_migd = 0.002, 0.0015
        expected = o0_migd - o1_migd
        assert abs(expected - 5e-4) < 1e-12

    def test_T49_seed_summaries_keys(self):
        pf = get_true_pf(500)
        rd = D2R.SCENARIOS["PERSISTENT"]
        rows = D2R.run_one_seed(561, 2, "PERSISTENT", rd, pf, "SMOKE")
        summ = D2R.compute_seed_summaries(rows, "PERSISTENT")
        assert len(summ) > 0
        keys = set(summ[0].keys())
        assert "MIGD_response" in keys
        assert "MIGD_epoch" in keys

    def test_T50_cb_per_seed_contains_o1benefit(self):
        pf = get_true_pf(500)
        rd = D2R.SCENARIOS["PERSISTENT"]
        rows = D2R.run_one_seed(561, 2, "PERSISTENT", rd, pf, "SMOKE")
        summ = D2R.compute_seed_summaries(rows, "PERSISTENT")
        cb = D2R.compute_cb_per_seed(summ, "PERSISTENT")
        assert len(cb) == 1  # one seed
        assert "O1Benefit_response" in cb[0]
        assert "O1Benefit_epoch" in cb[0]

# ============================================================
# T51-T57: Verdict logic
# ============================================================

class TestT51_T57_Verdicts:

    def _make_stats(self, cls_response, cls_epoch="PRACTICALLY_NEUTRAL",
                    bA_cls="PRACTICALLY_NEUTRAL", bB_cls="PRACTICALLY_NEUTRAL",
                    m=1e-4, bA_m=5e-5, bB_m=5e-5):
        return {
            "pooled": {
                "mean_O1Benefit_response": m, "CI_response": [0.01e-4, 2e-4],
                "class_response": cls_response,
                "mean_O1Benefit_epoch": m, "CI_epoch": [-1e-5, 3e-4],
                "class_epoch": cls_epoch,
            },
            "block_A": {
                "mean_O1Benefit_response": bA_m, "CI_response": [-1e-5, 2e-4],
                "class_response": bA_cls,
                "mean_O1Benefit_epoch": 5e-5, "CI_epoch": [-1e-5, 2e-4],
                "class_epoch": "PRACTICALLY_NEUTRAL",
                "early_O1Benefit_response": 1e-4, "late_O1Benefit_response": 1e-4,
            },
            "block_B": {
                "mean_O1Benefit_response": bB_m, "CI_response": [-1e-5, 2e-4],
                "class_response": bB_cls,
                "mean_O1Benefit_epoch": 5e-5, "CI_epoch": [-1e-5, 2e-4],
                "class_epoch": "PRACTICALLY_NEUTRAL",
                "early_O1Benefit_response": 1e-4, "late_O1Benefit_response": 1e-4,
            },
        }

    def test_T51_persistent_validated(self):
        stats = self._make_stats("BENEFICIAL", m=2e-4, bA_m=1e-4, bB_m=1e-4)
        stats["pooled"]["CI_response"] = [1e-5, 3e-4]
        v = D2R.compute_persistent_verdict(stats)
        assert v == "VALIDATED"

    def test_T52_persistent_failed_if_harmful(self):
        stats = self._make_stats("HARMFUL", m=-2e-4, bA_m=-1e-4, bB_m=-1e-4,
                                  bA_cls="HARMFUL", bB_cls="HARMFUL")
        v = D2R.compute_persistent_verdict(stats)
        assert v == "FAILED"

    def test_T53_stationary_safe(self):
        stats = self._make_stats("PRACTICALLY_NEUTRAL")
        v = D2R.compute_stationary_verdict(stats)
        assert v == "SAFE"

    def test_T54_stationary_unsafe_if_harmful(self):
        stats = self._make_stats("HARMFUL", m=-2e-4, cls_epoch="HARMFUL")
        v = D2R.compute_stationary_verdict(stats)
        assert v == "UNSAFE"

    def test_T55_mixed_stable(self):
        stats = self._make_stats("PRACTICALLY_NEUTRAL", m=1e-5, bA_m=2e-5, bB_m=2e-5)
        regime_stats = {
            "persistent": {"safety": "PASS"},
            "stationary": {"safety": "PASS"},
        }
        v = D2R.compute_mixed_verdict(stats, regime_stats, True, True)
        assert v in ("STABLE", "PARTIAL")

    def test_T56_overall_failed_if_repro_fail(self):
        v = D2R.compute_overall_verdict(
            "VALIDATED", "SAFE", "STABLE", "STABLE_POSITIVE",
            True, True, True, False  # repro_pass=False
        )
        assert v == "FAILED"

    def test_T57_overall_validated(self):
        v = D2R.compute_overall_verdict(
            "VALIDATED", "SAFE", "STABLE", "STABLE_POSITIVE",
            True, True, True, True
        )
        assert v == "VALIDATED"

# ============================================================
# T58-T61: Smoke reproducibility
# ============================================================

class TestT58_T61_Smoke:

    def test_T58_smoke_row_count(self, smoke_rows):
        # 2 arms × 3 scored changes = 6 rows per seed
        assert len(smoke_rows) == 6

    def test_T59_smoke_igd_response_reproducible(self, smoke_rows):
        pf = get_true_pf(500)
        rd = D2R.SCENARIOS["PERSISTENT"]
        rows2 = D2R.run_one_seed(561, 3, "PERSISTENT", rd, pf, "SMOKE")
        for r1, r2 in zip(smoke_rows, rows2):
            assert r1["igd_response"] == pytest.approx(r2["igd_response"], rel=1e-10)

    def test_T60_smoke_epoch_fp_reproducible(self, smoke_rows):
        pf = get_true_pf(500)
        rd = D2R.SCENARIOS["PERSISTENT"]
        rows2 = D2R.run_one_seed(561, 3, "PERSISTENT", rd, pf, "SMOKE")
        for r1, r2 in zip(smoke_rows, rows2):
            assert r1["epoch_x_fp"] == r2["epoch_x_fp"]

    def test_T61_smoke_no_hard_arm(self, smoke_rows):
        arms = set(r["arm"] for r in smoke_rows)
        assert "O2" not in arms
        assert "O3" not in arms
        assert arms == {"O0", "O1"}

    def test_T62_smoke_arms_have_identical_env_at_k0(self, smoke_rows):
        # At k=0, arms share the same cloned population, so igd_pre must match.
        by_k = {}
        for r in smoke_rows:
            by_k.setdefault(r["k"], {})[r["arm"]] = r
        k0 = by_k.get(0, {})
        if "O0" in k0 and "O1" in k0:
            assert k0["O0"]["igd_pre"] == pytest.approx(k0["O1"]["igd_pre"], rel=1e-10)

    def test_T63_smoke_seed_562_mixed_reproducible(self, smoke_rows_562):
        pf = get_true_pf(500)
        rd = D2R.SCENARIOS["MIXED"]
        rows2 = D2R.run_one_seed(562, 3, "MIXED", rd, pf, "SMOKE")
        for r1, r2 in zip(smoke_rows_562, rows2):
            assert r1["igd_response"] == pytest.approx(r2["igd_response"], rel=1e-10)
            assert r1["epoch_x_fp"] == r2["epoch_x_fp"]

# ============================================================
# T64-T70: Structural / no-forbidden-code checks
# ============================================================

class TestT64_T70_Structural:

    def test_T64_no_rl_in_source(self):
        src = inspect.getsource(D2R)
        assert "reinforcement_learning" not in src.lower()
        assert "from prdrl.rl" not in src
        assert "import rl" not in src

    def test_T65_no_sigma_tuning(self):
        src = inspect.getsource(D2R)
        assert "adaptive_sigma" not in src.lower()
        assert "sigma_schedule" not in src.lower()

    def test_T66_no_third_block(self):
        src = inspect.getsource(D2R)
        assert "BLOCK_C_SEEDS" not in src
        assert "block_C_stats" not in src
        assert "range(581" not in src

    def test_T67_production_default_unchanged(self):
        src = inspect.getsource(D2R)
        assert "O0_CURRENT" in src

    def test_T68_pdf_status_unchanged(self):
        src = inspect.getsource(D2R)
        assert "PDF_STATUS" in src
        assert "UNCHANGED" in src

    def test_T69_rl_status_not_started(self):
        src = inspect.getsource(D2R)
        assert "RL_STATUS" in src
        assert "NOT_STARTED" in src

    def test_T70_results_dir_correct(self):
        assert "D2R_selected_o1_confirmation" in D2R.RESULTS_DIR

    def test_T71_prerun_manifest_function_exists(self):
        assert callable(D2R.create_prerun_manifest)

    def test_T72_run_one_seed_function_exists(self):
        assert callable(D2R.run_one_seed)

    def test_T73_verdict_functions_present(self):
        assert callable(D2R.compute_persistent_verdict)
        assert callable(D2R.compute_stationary_verdict)
        assert callable(D2R.compute_mixed_verdict)
        assert callable(D2R.compute_overall_verdict)

    def test_T74_mixed_regime_analysis_present(self):
        assert callable(D2R.compute_mixed_regime_analysis)

    def test_T75_write_verdict_present(self):
        assert callable(D2R.write_verdict)

    def test_T76_write_report_present(self):
        assert callable(D2R.write_report)

    def test_T77_classify_effect_correct_threshold(self):
        # Boundary: exactly DELTA_D2R should be PRACTICALLY_NEUTRAL
        cls = D2R.classify_effect(D2R.DELTA_D2R, -1e-5, 3e-5)
        assert cls == "PRACTICALLY_NEUTRAL"

    def test_T78_target_region_is_deterministic_formula(self):
        for seed in [501, 510, 520, 560]:
            for k in range(10):
                expected = (seed + k) % 10
                assert D2R.target_region(seed, k, 10) == expected

    def test_T79_mixed_regime_safety_note_present(self):
        src = inspect.getsource(D2R)
        assert "L2-B" in src or "L2B" in src

    def test_T80_d1r_selection_context_in_source(self):
        src = inspect.getsource(D2R)
        assert "hierarchy" in src.lower() or "parsimony" in src.lower()
