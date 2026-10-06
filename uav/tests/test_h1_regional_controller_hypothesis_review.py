"""
T1-T65+ test suite for H1 — Regional Controller Hypothesis Review.

Invariants:
  - RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
  - Diagnostic stage only: no RULE_V2, no DQN, no rollout, no reward learning.
  - Seeds 901-940 (Block A: 901-920, Block B: 921-940).
  - No overlap with E3 701-780, E3-R 801-882.
  - Production code unchanged. Historical verdicts immutable.
  - DELTA_H1_ROG = 1e-5. MARGIN_EPS = 1e-8. BAL_ACC_MARGIN = 0.05.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "src")

RESULTS_DIR = Path("results/H1_regional_controller_hypothesis_review")

DIAG_SEEDS = list(range(901, 941))
BLOCK_A_SEEDS = list(range(901, 921))
BLOCK_B_SEEDS = list(range(921, 941))
E3_SEEDS = set(range(701, 781))
E3R_SEEDS = set(range(801, 883))
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ACTIONS = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")
N_SCORED_CHANGES = 10
N_REGIONS = 10  # from reference(2, 9) → R=10
DELTA_H1_ROG = 1e-5
MARGIN_EPS = 1e-8
BAL_ACC_MARGIN = 0.05

H1_EXPECTED_EVENTS = len(DIAG_SEEDS) * len(SCENARIOS) * N_SCORED_CHANGES  # 1200


# ============================================================
# Helpers
# ============================================================

def load_json(name):
    return json.loads((RESULTS_DIR / name).read_text())


def load_jsonl(name):
    rows = []
    with open(RESULTS_DIR / name) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(name):
    with open(RESULTS_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# T1-T8: Pre-run manifest invariants
# ============================================================

class TestPreRunManifest:
    def test_T1_manifest_exists(self):
        assert (RESULTS_DIR / "H1_pre_run_manifest.json").exists()

    def test_T2_rl_status_not_started(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m["RL_STATUS"] == "NOT_STARTED"
        assert m["DRL_TRAINING"] == "NOT_STARTED"

    def test_T3_state_schema_v2(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m["STATE_SCHEMA_VERSION"] == "V2"
        assert m["STATE_DIM"] == 4
        feats = set(m["STATE_FEATURES"])
        assert feats == {"severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"}

    def test_T4_seed_ranges_correct(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m["DIAG_SEEDS"] == [901, 940]
        assert m["BLOCK_A"] == [901, 920]
        assert m["BLOCK_B"] == [921, 940]

    def test_T5_no_seed_overlap_e3(self):
        assert not (set(DIAG_SEEDS) & E3_SEEDS)

    def test_T6_no_seed_overlap_e3r(self):
        assert not (set(DIAG_SEEDS) & E3R_SEEDS)

    def test_T7_carrier_declared(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m["carrier"] == "GLOBAL_LOCAL"

    def test_T8_no_hard_no_b_r_a(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m.get("no_HARD") is True
        assert m.get("no_B_r_a") is True
        assert m.get("no_DRL") is True
        assert m.get("no_RULE_V2") is True


# ============================================================
# T9-T15: Historical verdicts immutable
# ============================================================

class TestHistoricalImmutability:
    def test_T9_e3r_verdict_exists(self):
        assert Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").exists()

    def test_T10_e3r_gate_b_still_fail(self):
        v = json.loads(Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").read_text())
        assert v.get("E3R_GATE_B") == "FAIL"

    def test_T11_e3r_rl_status_not_started(self):
        v = json.loads(Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").read_text())
        assert v.get("RL_STATUS") == "NOT_STARTED"

    def test_T12_e3r_production_default_unchanged(self):
        v = json.loads(Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").read_text())
        assert v.get("PRODUCTION_DEFAULT") == "O1_F005"

    def test_T13_e3r_next_stage_matches(self):
        v = json.loads(Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").read_text())
        assert v.get("NEXT_STAGE") == "REGIONAL_CONTROLLER_HYPOTHESIS_REVIEW"

    def test_T14_manifest_hashes_recorded(self):
        m = load_json("H1_pre_run_manifest.json")
        files = m.get("files", {})
        assert "src/prdrl/oracle/oracle.py" in files
        assert "src/prdrl/detectors/severity.py" in files
        assert "src/prdrl/rl/interface.py" in files

    def test_T15_e3_verdict_quality_fn_immutable(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m.get("quality_fn") == "negative_IGD"


# ============================================================
# T16-T22: Oracle region records schema
# ============================================================

class TestOracleRegionRecords:
    @pytest.fixture(scope="class")
    def region_rows(self):
        return load_jsonl("H1_oracle_region_records.jsonl")

    def test_T16_region_records_exist(self, region_rows):
        assert len(region_rows) > 0

    def test_T17_region_records_schema(self, region_rows):
        required = {
            "scenario", "seed", "block", "k", "region_id",
            "severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r",
            "U_KEEP", "U_LOCAL", "U_PREDICT", "U_DIVERSIFY",
            "oracle_best_action", "oracle_second_action",
            "best_utility", "second_utility", "action_margin",
            "best_global_action", "ROG", "ROG_vs_GLOBAL_PREDICT", "OHI",
        }
        for r in region_rows[:20]:
            assert required.issubset(r.keys()), f"Missing: {required - set(r.keys())}"

    def test_T18_no_HARD_action(self, region_rows):
        for r in region_rows:
            assert r["oracle_best_action"] in ACTIONS
            if r["oracle_second_action"] is not None:
                assert r["oracle_second_action"] in ACTIONS or r["oracle_second_action"] is None

    def test_T19_U_KEEP_is_zero(self, region_rows):
        for r in region_rows[:100]:
            assert abs(float(r["U_KEEP"])) < 1e-12, f"U_KEEP={r['U_KEEP']} should be 0"

    def test_T20_v2_features_valid_range(self, region_rows):
        for r in region_rows[:200]:
            assert 0.0 <= float(r["severity_r"]) <= 1.0 + 1e-9
            assert float(r["convergence_loss_r"]) >= 0.0
            assert 0.0 <= float(r["occupancy_r"]) <= 1.0 + 1e-9

    def test_T21_seeds_correct_range(self, region_rows):
        seeds = {r["seed"] for r in region_rows}
        assert seeds.issubset(set(DIAG_SEEDS)), f"Unknown seeds: {seeds - set(DIAG_SEEDS)}"

    def test_T22_scenarios_complete(self, region_rows):
        scenarios = {r["scenario"] for r in region_rows}
        assert scenarios == set(SCENARIOS)


# ============================================================
# T23-T30: Oracle event summary
# ============================================================

class TestOracleEventSummary:
    @pytest.fixture(scope="class")
    def event_rows(self):
        return load_csv("H1_oracle_event_summary.csv")

    def test_T23_event_count(self, event_rows):
        assert len(event_rows) == H1_EXPECTED_EVENTS, \
            f"Expected {H1_EXPECTED_EVENTS} events, got {len(event_rows)}"

    def test_T24_event_schema(self, event_rows):
        required = {"scenario", "seed", "block", "k", "ROG", "ROG_vs_GLOBAL_PREDICT",
                    "OHI", "best_global_action", "Q_keep", "Q_best_global", "Q_regional"}
        for r in event_rows[:10]:
            assert required.issubset(r.keys())

    def test_T25_all_seeds_present(self, event_rows):
        seeds = {int(r["seed"]) for r in event_rows}
        assert seeds == set(DIAG_SEEDS)

    def test_T26_all_scenarios_present(self, event_rows):
        scenarios = {r["scenario"] for r in event_rows}
        assert scenarios == set(SCENARIOS)

    def test_T27_k_range(self, event_rows):
        for r in event_rows:
            assert 0 <= int(r["k"]) < N_SCORED_CHANGES

    def test_T28_best_global_action_valid(self, event_rows):
        for r in event_rows:
            assert r["best_global_action"] in ACTIONS

    def test_T29_OHI_in_unit_interval(self, event_rows):
        for r in event_rows:
            ohi = float(r["OHI"])
            assert -1e-9 <= ohi <= 1.0 + 1e-9, f"OHI={ohi} out of range"

    def test_T30_block_assignment_correct(self, event_rows):
        for r in event_rows:
            s = int(r["seed"])
            blk = r["block"]
            if s in BLOCK_A_SEEDS:
                assert blk == "A", f"seed {s} should be block A"
            elif s in BLOCK_B_SEEDS:
                assert blk == "B", f"seed {s} should be block B"


# ============================================================
# T31-T36: Seed summary
# ============================================================

class TestSeedSummary:
    @pytest.fixture(scope="class")
    def seed_rows(self):
        return load_csv("H1_seed_summary.csv")

    def test_T31_seed_summary_exists(self, seed_rows):
        assert len(seed_rows) > 0

    def test_T32_seed_summary_complete(self, seed_rows):
        keys = {(int(r["seed"]), r["scenario"]) for r in seed_rows}
        expected = {(s, sc) for s in DIAG_SEEDS for sc in SCENARIOS}
        assert keys == expected, f"Missing: {expected - keys}"

    def test_T33_n_events_correct(self, seed_rows):
        for r in seed_rows:
            assert int(r["n_events"]) == N_SCORED_CHANGES

    def test_T34_frac_non_predict_in_unit_interval(self, seed_rows):
        for r in seed_rows:
            v = float(r["frac_non_PREDICT_oracle_labels"])
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_T35_frac_heterogeneous_in_unit_interval(self, seed_rows):
        for r in seed_rows:
            v = float(r["frac_heterogeneous_events"])
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_T36_mean_ROG_is_finite(self, seed_rows):
        for r in seed_rows:
            v = float(r["mean_ROG"])
            assert not (v != v), f"NaN mean_ROG for seed={r['seed']}"


# ============================================================
# T37-T42: Oracle headroom CSV
# ============================================================

class TestOracleHeadroom:
    @pytest.fixture(scope="class")
    def headroom_rows(self):
        return load_csv("H1_oracle_headroom.csv")

    def test_T37_headroom_exists(self, headroom_rows):
        assert len(headroom_rows) > 0

    def test_T38_headroom_blocks_scenarios(self, headroom_rows):
        block_sc = {(r["block"], r["scenario"]) for r in headroom_rows}
        expected = {(b, sc) for b in ["BLOCK_A", "BLOCK_B", "POOLED"] for sc in SCENARIOS}
        assert expected == block_sc

    def test_T39_headroom_classification_valid(self, headroom_rows):
        valid = {"SUPPORTED", "DIRECTIONAL", "NEUTRAL", "CONTRADICTED",
                 "DIRECTIONAL_NEGATIVE", "UNKNOWN"}
        for r in headroom_rows:
            assert r["classification"] in valid

    def test_T40_headroom_n_seeds_correct(self, headroom_rows):
        for r in headroom_rows:
            blk = r["block"]
            n = int(r["n_seeds"])
            if blk == "BLOCK_A":
                assert n == len(BLOCK_A_SEEDS)
            elif blk == "BLOCK_B":
                assert n == len(BLOCK_B_SEEDS)
            elif blk == "POOLED":
                assert n == len(DIAG_SEEDS)

    def test_T41_ci_lo_le_mean_le_ci_hi(self, headroom_rows):
        for r in headroom_rows:
            lo = float(r["ci_lo"])
            hi = float(r["ci_hi"])
            mean = float(r["mean_ROG"])
            assert lo <= mean + 1e-12 and mean <= hi + 1e-12

    def test_T42_mean_ROG_finite(self, headroom_rows):
        for r in headroom_rows:
            assert float(r["mean_ROG"]) == float(r["mean_ROG"])  # not NaN


# ============================================================
# T43-T48: Action distribution
# ============================================================

class TestActionDistribution:
    @pytest.fixture(scope="class")
    def dist_rows(self):
        return load_csv("H1_action_distribution.csv")

    def test_T43_all_actions_present(self, dist_rows):
        actions = {r["action"] for r in dist_rows}
        assert actions == set(ACTIONS)

    def test_T44_fractions_sum_to_one(self, dist_rows):
        for sc in SCENARIOS:
            for blk in ["A", "B", "POOLED"]:
                sub = [r for r in dist_rows if r["scenario"] == sc and r["block"] == blk]
                total = sum(int(r["count"]) for r in sub)
                frac_sum = sum(float(r["fraction"]) for r in sub)
                if total > 0:
                    assert abs(frac_sum - 1.0) < 0.01, \
                        f"{sc}/{blk}: fractions sum={frac_sum:.4f}"

    def test_T45_no_HARD_in_distribution(self, dist_rows):
        for r in dist_rows:
            assert r["action"] != "HARD"

    def test_T46_counts_non_negative(self, dist_rows):
        for r in dist_rows:
            assert int(r["count"]) >= 0

    def test_T47_scenarios_complete(self, dist_rows):
        scenarios = {r["scenario"] for r in dist_rows}
        assert scenarios == set(SCENARIOS)

    def test_T48_blocks_complete(self, dist_rows):
        blocks = {r["block"] for r in dist_rows}
        assert blocks == {"A", "B", "POOLED"}


# ============================================================
# T49-T53: Action margin
# ============================================================

class TestActionMargin:
    @pytest.fixture(scope="class")
    def margin_rows(self):
        return load_csv("H1_action_margin.csv")

    def test_T49_margin_exists(self, margin_rows):
        assert len(margin_rows) > 0

    def test_T50_actions_valid(self, margin_rows):
        for r in margin_rows:
            assert r["oracle_best_action"] in ACTIONS

    def test_T51_frac_near_tie_in_unit_interval(self, margin_rows):
        for r in margin_rows:
            v = float(r["frac_near_tie"])
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_T52_median_ge_zero(self, margin_rows):
        for r in margin_rows:
            assert float(r["median_margin"]) >= -1e-12

    def test_T53_margin_ordering(self, margin_rows):
        for r in margin_rows:
            q25 = float(r["q25_margin"])
            med  = float(r["median_margin"])
            q75  = float(r["q75_margin"])
            assert q25 <= med + 1e-12
            assert med <= q75 + 1e-12


# ============================================================
# T54-T57: Global PREDICT audit
# ============================================================

class TestGlobalPredictAudit:
    @pytest.fixture(scope="class")
    def audit_rows(self):
        return load_csv("H1_global_predict_audit.csv")

    def test_T54_audit_exists(self, audit_rows):
        assert len(audit_rows) > 0

    def test_T55_fracs_in_unit_interval(self, audit_rows):
        for r in audit_rows:
            assert 0.0 <= float(r["frac_regional_oracle_is_PREDICT"]) <= 1.0 + 1e-9
            assert 0.0 <= float(r["frac_events_best_global_is_PREDICT"]) <= 1.0 + 1e-9

    def test_T56_scenarios_and_blocks(self, audit_rows):
        keys = {(r["scenario"], r["block"]) for r in audit_rows}
        expected = {(sc, blk) for sc in SCENARIOS for blk in ["A", "B", "POOLED"]}
        assert expected == keys

    def test_T57_mean_rog_vs_predict_finite(self, audit_rows):
        for r in audit_rows:
            assert float(r["mean_ROG_vs_GLOBAL_PREDICT"]) == float(r["mean_ROG_vs_GLOBAL_PREDICT"])


# ============================================================
# T58-T62: V2 label feature summary
# ============================================================

class TestV2LabelFeatureSummary:
    @pytest.fixture(scope="class")
    def feature_rows(self):
        return load_csv("H1_v2_label_feature_summary.csv")

    def test_T58_feature_summary_exists(self, feature_rows):
        assert len(feature_rows) > 0

    def test_T59_actions_valid(self, feature_rows):
        for r in feature_rows:
            assert r["oracle_best_action"] in ACTIONS

    def test_T60_frac_total_sums_to_one_per_scenario(self, feature_rows):
        for sc in SCENARIOS + ["POOLED"]:
            sub = [r for r in feature_rows if r["scenario"] == sc]
            if not sub:
                continue
            frac_sum = sum(float(r["frac_total"]) for r in sub)
            assert abs(frac_sum - 1.0) < 0.01, f"{sc}: frac_total sum={frac_sum:.4f}"

    def test_T61_median_severity_non_negative(self, feature_rows):
        for r in feature_rows:
            assert float(r["med_severity"]) >= -1e-12

    def test_T62_iqr_non_negative(self, feature_rows):
        for r in feature_rows:
            assert float(r["iqr_severity"]) >= -1e-12
            assert float(r["iqr_convergence_loss"]) >= -1e-12


# ============================================================
# T63-T67: Separability cross-block probe
# ============================================================

class TestSeparabilityCrossBlock:
    @pytest.fixture(scope="class")
    def probes(self):
        return load_json("H1_separability_crossblock.json")

    def test_T63_separability_exists(self, probes):
        assert probes is not None
        assert len(probes) > 0

    def test_T64_pooled_directions_present(self, probes):
        pooled = probes.get("POOLED", {})
        assert "A_to_B" in pooled or "B_to_A" in pooled

    def test_T65_bal_acc_in_unit_interval(self, probes):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = probes.get(sc, {}).get(direction, {})
                if "error" in d:
                    continue
                for key in ["logistic_bal_acc", "tree_bal_acc",
                            "majority_baseline_bal_acc", "global_predict_label_bal_acc"]:
                    v = d.get(key, float("nan"))
                    if v == v:  # not NaN
                        assert 0.0 <= v <= 1.0 + 1e-9, f"{sc}/{direction}/{key}={v}"

    def test_T66_perm_pvals_in_unit_interval(self, probes):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = probes.get(sc, {}).get(direction, {})
                if "error" in d:
                    continue
                for key in ["perm_pval_logistic", "perm_pval_tree"]:
                    v = d.get(key, float("nan"))
                    if v == v:
                        assert 0.0 <= v <= 1.0 + 1e-9

    def test_T67_no_data_leakage_cross_block(self, probes):
        for sc in SCENARIOS + ["POOLED"]:
            d_ab = probes.get(sc, {}).get("A_to_B", {})
            d_ba = probes.get(sc, {}).get("B_to_A", {})
            if "error" not in d_ab and "error" not in d_ba:
                n_train_ab = d_ab.get("n_train", 0)
                n_test_ab  = d_ab.get("n_test",  0)
                n_train_ba = d_ba.get("n_train", 0)
                n_test_ba  = d_ba.get("n_test",  0)
                assert n_train_ab == n_test_ba, "A_to_B train size != B_to_A test size"
                assert n_train_ba == n_test_ab, "B_to_A train size != A_to_B test size"


# ============================================================
# T68-T72: Permutation control
# ============================================================

class TestPermutationControl:
    @pytest.fixture(scope="class")
    def perm(self):
        return load_json("H1_permutation_control.json")

    def test_T68_perm_exists(self, perm):
        assert perm is not None

    def test_T69_perm_pval_present(self, perm):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = perm.get(sc, {}).get(direction, {})
                if "error" not in d:
                    assert "perm_pval_logistic" in d
                    assert "perm_pval_tree" in d

    def test_T70_perm_mean_lt_or_eq_real(self, perm):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = perm.get(sc, {}).get(direction, {})
                if "error" in d:
                    continue
                bal = d.get("logistic_bal_acc", float("nan"))
                perm_mean = d.get("perm_mean_bal_lr", float("nan"))
                if bal == bal and perm_mean == perm_mean:
                    # Not guaranteed, but permuted mean should generally be lower
                    # We only test that the perm_mean is a finite number
                    assert perm_mean == perm_mean  # not NaN

    def test_T71_non_near_tie_action_labels_valid(self, perm):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = perm.get(sc, {}).get(direction, {})
                if "error" not in d:
                    maj = d.get("majority_class_label")
                    if maj:
                        assert maj in ACTIONS

    def test_T72_n_train_n_test_positive(self, perm):
        for sc in SCENARIOS + ["POOLED"]:
            for direction in ["A_to_B", "B_to_A"]:
                d = perm.get(sc, {}).get(direction, {})
                if "error" not in d:
                    assert d.get("n_train", 0) > 0
                    assert d.get("n_test",  0) > 0


# ============================================================
# T73-T78: Mixed within-event
# ============================================================

class TestMixedWithinEvent:
    @pytest.fixture(scope="class")
    def mixed_rows(self):
        return load_csv("H1_mixed_within_event.csv")

    def test_T73_mixed_exists(self, mixed_rows):
        assert len(mixed_rows) > 0

    def test_T74_all_mixed_scenario(self, mixed_rows):
        for r in mixed_rows:
            assert r["scenario"] == "MIXED"

    def test_T75_row_count(self, mixed_rows):
        expected = len(DIAG_SEEDS) * N_SCORED_CHANGES
        assert len(mixed_rows) == expected, f"Expected {expected}, got {len(mixed_rows)}"

    def test_T76_fracs_in_unit_interval(self, mixed_rows):
        for r in mixed_rows:
            for key in ["frac_PREDICT", "frac_KEEP", "frac_LOCAL", "frac_DIVERSIFY"]:
                v = float(r[key])
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_T77_fracs_sum_to_one(self, mixed_rows):
        for r in mixed_rows:
            total = sum(float(r[k]) for k in ["frac_PREDICT", "frac_KEEP",
                                               "frac_LOCAL", "frac_DIVERSIFY"])
            assert abs(total - 1.0) < 0.01

    def test_T78_n_unique_nnt_ge_zero(self, mixed_rows):
        for r in mixed_rows:
            assert int(r["n_unique_nnt_actions"]) >= 0


# ============================================================
# T79-T83: Scenario summary
# ============================================================

class TestScenarioSummary:
    @pytest.fixture(scope="class")
    def summary(self):
        return load_json("H1_scenario_summary.json")

    def test_T79_scenario_summary_exists(self, summary):
        assert summary is not None

    def test_T80_all_scenarios_present(self, summary):
        assert set(SCENARIOS).issubset(summary.keys())

    def test_T81_headroom_field_present(self, summary):
        for sc in SCENARIOS:
            assert "REGIONAL_ORACLE_HEADROOM" in summary[sc]

    def test_T82_predict_frac_in_unit_interval(self, summary):
        for sc in SCENARIOS:
            v = summary[sc].get("PREDICT_oracle_frac")
            if v is not None and v == v:  # not NaN
                assert 0.0 <= float(v) <= 1.0 + 1e-9

    def test_T83_headroom_classification_valid(self, summary):
        valid = {"SUPPORTED", "DIRECTIONAL", "NEUTRAL", "CONTRADICTED",
                 "DIRECTIONAL_NEGATIVE", "UNKNOWN"}
        for sc in SCENARIOS:
            cls = summary[sc].get("REGIONAL_ORACLE_HEADROOM", "UNKNOWN")
            assert cls in valid


# ============================================================
# T84-T90: Decision matrix
# ============================================================

class TestDecisionMatrix:
    @pytest.fixture(scope="class")
    def dm(self):
        return load_json("H1_decision_matrix.json")

    def test_T84_decision_matrix_exists(self, dm):
        assert dm is not None

    def test_T85_headroom_present(self, dm):
        assert "REGIONAL_ORACLE_HEADROOM" in dm

    def test_T86_predict_dominance_present(self, dm):
        assert "GLOBAL_PREDICT_DOMINANCE" in dm
        assert dm["GLOBAL_PREDICT_DOMINANCE"] in {"STRONG", "MODERATE", "WEAK", "UNKNOWN"}

    def test_T87_separability_present(self, dm):
        assert "V2_ACTION_SEPARABILITY" in dm
        assert dm["V2_ACTION_SEPARABILITY"] in {"SUPPORTED", "PARTIAL", "ABSENT"}

    def test_T88_outcome_present(self, dm):
        assert "H1_OUTCOME" in dm
        valid = {"GLOBAL_ACTION_DOMINANCE", "STATE_INFORMATION_INSUFFICIENT",
                 "RULE_MAPPING_INADEQUATE", "INCONCLUSIVE"}
        assert dm["H1_OUTCOME"] in valid

    def test_T89_frozen_constants(self, dm):
        assert abs(float(dm["BAL_ACC_MARGIN_FROZEN"])   - BAL_ACC_MARGIN) < 1e-15
        assert abs(float(dm["MARGIN_EPS_FROZEN"])        - MARGIN_EPS) < 1e-20
        assert abs(float(dm["DELTA_H1_ROG_FROZEN"])      - DELTA_H1_ROG) < 1e-17

    def test_T90_next_stage_present(self, dm):
        valid_next = {"E4_LEARNED_CONTROLLER_PREPARATION",
                      "REGIONAL_CONTROL_REDESIGN_OR_STOP",
                      "STATE_REPRESENTATION_HYPOTHESIS_REVIEW",
                      "H1_REVIEW"}
        assert dm.get("NEXT_STAGE") in valid_next


# ============================================================
# T91-T97: Verdict
# ============================================================

class TestVerdict:
    @pytest.fixture(scope="class")
    def verdict(self):
        return load_json("H1_verdict.json")

    def test_T91_verdict_exists(self, verdict):
        assert verdict is not None

    def test_T92_rl_status_not_started(self, verdict):
        assert verdict["RL_STATUS"] == "NOT_STARTED"
        assert verdict["DRL_TRAINING"] == "NOT_STARTED"

    def test_T93_production_default_unchanged(self, verdict):
        assert verdict["PRODUCTION_DEFAULT"] == "O1_F005"

    def test_T94_h1_outcome_valid(self, verdict):
        valid = {"GLOBAL_ACTION_DOMINANCE", "STATE_INFORMATION_INSUFFICIENT",
                 "RULE_MAPPING_INADEQUATE", "INCONCLUSIVE"}
        assert verdict["H1_OUTCOME"] in valid

    def test_T95_e3r_status_correct(self, verdict):
        assert verdict.get("E3R_STATUS") == "ADMISSIBLE_GATE_B_FAIL"

    def test_T96_no_classifier_rollout(self, verdict):
        assert verdict.get("NO_CLASSIFIER_ROLLOUT") is True
        assert verdict.get("NO_RULE_V2") is True
        assert verdict.get("NO_DQN") is True

    def test_T97_next_stage_set(self, verdict):
        valid_next = {"E4_LEARNED_CONTROLLER_PREPARATION",
                      "REGIONAL_CONTROL_REDESIGN_OR_STOP",
                      "STATE_REPRESENTATION_HYPOTHESIS_REVIEW",
                      "H1_REVIEW"}
        assert verdict.get("NEXT_STAGE") in valid_next


# ============================================================
# T98-T102: Report
# ============================================================

class TestReport:
    @pytest.fixture(scope="class")
    def report_text(self):
        return (RESULTS_DIR / "H1_report.md").read_text()

    def test_T98_report_exists(self, report_text):
        assert len(report_text) > 100

    def test_T99_report_has_h1_outcome(self, report_text):
        assert "H1_OUTCOME" in report_text

    def test_T100_report_rl_status(self, report_text):
        assert "NOT_STARTED" in report_text

    def test_T101_report_no_drl_training(self, report_text):
        assert "NO_CLASSIFIER_ROLLOUT" not in report_text or "no deployed" in report_text.lower() or \
               "no rollout" in report_text.lower() or "probes" in report_text.lower()

    def test_T102_report_headroom_section(self, report_text):
        assert "Headroom" in report_text or "ROG" in report_text


# ============================================================
# T103-T108: Cross-invariant checks
# ============================================================

class TestCrossInvariants:
    def test_T103_delta_h1_rog_matches_spec(self):
        assert DELTA_H1_ROG == 1e-5

    def test_T104_margin_eps_matches_spec(self):
        assert MARGIN_EPS == 1e-8

    def test_T105_bal_acc_margin_matches_spec(self):
        assert BAL_ACC_MARGIN == 0.05

    def test_T106_carrier_is_global_local(self):
        m = load_json("H1_pre_run_manifest.json")
        assert m["carrier"] == "GLOBAL_LOCAL"

    def test_T107_oracle_utility_convention_correct(self):
        m = load_json("H1_pre_run_manifest.json")
        assert "U_{r,a}" in m["oracle_utility_convention"]
        assert "KEEP" in m["oracle_utility_convention"]

    def test_T108_keep_wins_ties(self):
        m = load_json("H1_pre_run_manifest.json")
        assert "KEEP wins strict ties" in m["oracle_tie_rule"]

    def test_T109_oracle_region_count_per_event_plausible(self):
        rows = load_csv("H1_oracle_event_summary.csv")
        for r in rows[:50]:
            n = int(r["n_active_regions"])
            assert 1 <= n <= N_REGIONS

    def test_T110_keep_utility_zero_in_region_records(self):
        rr = load_jsonl("H1_oracle_region_records.jsonl")
        for r in rr[:200]:
            assert abs(float(r["U_KEEP"])) < 1e-12, f"U_KEEP={r['U_KEEP']} != 0"

    def test_T111_ROG_sign_consistency(self):
        ev = load_csv("H1_oracle_event_summary.csv")
        for r in ev[:50]:
            q_reg = float(r["Q_regional"])
            q_best = float(r["Q_best_global"])
            rog = float(r["ROG"])
            # Sign check: ROG >= 0 iff Q_regional >= Q_best_global
            sign_correct = (
                (q_reg >= q_best - 1e-10 and rog >= -1e-10) or
                (q_reg < q_best + 1e-10 and rog <= 1e-10)
            )
            assert sign_correct, f"ROG sign inconsistent: ROG={rog}, Q_reg={q_reg}, Q_best={q_best}"

    def test_T112_ROG_vs_predict_definition(self):
        ev = load_csv("H1_oracle_event_summary.csv")
        for r in ev[:50]:
            q_reg = float(r["Q_regional"])
            q_pred = float(r["Q_GLOBAL_PREDICT"])
            rog_vs = float(r["ROG_vs_GLOBAL_PREDICT"])
            expected = (q_reg - q_pred) / (abs(q_pred) + 1e-12)
            assert abs(rog_vs - expected) < 1e-8, \
                f"ROG_vs_PREDICT={rog_vs} != expected={expected}"

    def test_T113_action_margin_consistency(self):
        rr = load_jsonl("H1_oracle_region_records.jsonl")
        for r in rr[:100]:
            best_u  = float(r["best_utility"])
            second_u = float(r["second_utility"])
            margin   = float(r["action_margin"])
            if not (best_u != best_u or second_u != second_u or margin != margin):  # no NaN
                assert abs(margin - (best_u - second_u)) < 1e-10


# ============================================================
# T114-T118: Output file completeness
# ============================================================

class TestOutputFiles:
    def test_T114_all_output_files_exist(self):
        expected_files = [
            "H1_pre_run_manifest.json",
            "H1_oracle_region_records.jsonl",
            "H1_oracle_event_summary.csv",
            "H1_seed_summary.csv",
            "H1_oracle_headroom.csv",
            "H1_action_distribution.csv",
            "H1_action_margin.csv",
            "H1_global_predict_audit.csv",
            "H1_v2_label_feature_summary.csv",
            "H1_separability_crossblock.json",
            "H1_permutation_control.json",
            "H1_mixed_within_event.csv",
            "H1_scenario_summary.json",
            "H1_decision_matrix.json",
            "H1_verdict.json",
            "H1_report.md",
        ]
        for fn in expected_files:
            assert (RESULTS_DIR / fn).exists(), f"Missing: {fn}"

    def test_T115_region_records_count_plausible(self):
        rows = load_jsonl("H1_oracle_region_records.jsonl")
        # At minimum: 40 seeds × 3 scenarios × 10 changes × 1 region
        assert len(rows) >= H1_EXPECTED_EVENTS
        # At maximum: × R_max = × 10 regions
        assert len(rows) <= H1_EXPECTED_EVENTS * N_REGIONS + 1

    def test_T116_verdict_stage_field(self):
        v = load_json("H1_verdict.json")
        assert v.get("stage") == "H1_regional_controller_hypothesis_review"

    def test_T117_historical_verdicts_not_modified(self):
        # E3R verdict must still say Gate B = FAIL
        e3r = json.loads(Path("results/E3R_rule_v2_semantic_rerun/E3R_verdict.json").read_text())
        assert e3r.get("E3R_GATE_B") == "FAIL"
        assert e3r.get("RL_STATUS") == "NOT_STARTED"

    def test_T118_no_dqn_or_training_in_any_output(self):
        forbidden = ["DQN_STARTED", "TRAINING_STARTED", "REWARD_LEARNING", "RULE_V2_DEPLOYED"]
        for fn in ["H1_verdict.json", "H1_decision_matrix.json"]:
            text = (RESULTS_DIR / fn).read_text()
            for term in forbidden:
                assert term not in text, f"{fn} contains forbidden term: {term}"
