"""E4RTR: Pairwise CG A/B Training Execution — 120+ tests."""

import csv
import json
import math
import os
import hashlib
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "results")
E4RTR = os.path.join(BASE, "E4RTR_pairwise_cg_ab_training_execution")
E4RT = os.path.join(BASE, "E4RT_pairwise_cg_ab_training_protocol_freeze")
E4RI = os.path.join(BASE, "E4RI_pairwise_coordination_graph_implementation_validation")

C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
C3_SHA = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"


def load(directory, name):
    with open(os.path.join(directory, name)) as f:
        return json.load(f)


def load_csv(directory, name):
    with open(os.path.join(directory, name)) as f:
        return list(csv.DictReader(f))


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def verdict():
    return load(E4RTR, "E4RTR_verdict.json")

@pytest.fixture(scope="module")
def pre_manifest():
    return load(E4RTR, "E4RTR_pre_execution_manifest.json")

@pytest.fixture(scope="module")
def exec_config():
    return load(E4RTR, "E4RTR_execution_config.json")

@pytest.fixture(scope="module")
def init_manifest():
    return load(E4RTR, "E4RTR_pair_initialization_manifest.json")

@pytest.fixture(scope="module")
def single_factor():
    return load(E4RTR, "E4RTR_single_factor_audit.json")

@pytest.fixture(scope="module")
def ckpt_manifest():
    return load(E4RTR, "E4RTR_checkpoint_manifest.json")

@pytest.fixture(scope="module")
def selected():
    return load(E4RTR, "E4RTR_selected_checkpoints.json")

@pytest.fixture(scope="module")
def numerical():
    return load(E4RTR, "E4RTR_numerical_stability.json")

@pytest.fixture(scope="module")
def arch_integrity():
    return load(E4RTR, "E4RTR_runtime_architecture_integrity.json")

@pytest.fixture(scope="module")
def firewall():
    return load(E4RTR, "E4RTR_final_firewall.json")

@pytest.fixture(scope="module")
def leakage():
    return load(E4RTR, "E4RTR_training_validation_leakage_audit.json")

@pytest.fixture(scope="module")
def ep_csv():
    return load_csv(E4RTR, "E4RTR_training_episode_results.csv")

@pytest.fixture(scope="module")
def ab_csv():
    return load_csv(E4RTR, "E4RTR_validation_ab_summary.csv")

@pytest.fixture(scope="module")
def diag_csv():
    return load_csv(E4RTR, "E4RTR_coordination_diagnostics.csv")


# ── 1. Verdict tests ──────────────────────────────────────

class TestVerdict:
    def test_v01_complete(self, verdict):
        assert verdict["E4RTR_COMPLETE"] == "YES"

    def test_v02_runs(self, verdict):
        assert verdict["RUNS_COMPLETE"] == "10/10"

    def test_v03_c1_runs(self, verdict):
        assert verdict["C1_RUNS_COMPLETE"] == "5/5"

    def test_v04_c3_runs(self, verdict):
        assert verdict["C3_RUNS_COMPLETE"] == "5/5"

    def test_v05_episodes(self, verdict):
        assert verdict["TRAINING_EPISODES_COMPLETE"] == "10800/10800"

    def test_v06_paired_init(self, verdict):
        assert verdict["PAIRED_INITIALIZATION"] == "PASS"

    def test_v07_single_factor(self, verdict):
        assert verdict["SINGLE_FACTOR_INTEGRITY"] == "PASS"

    def test_v08_v3_integrity(self, verdict):
        assert verdict["V3_CONSTANTS_INTEGRITY"] == "PASS"

    def test_v09_r1_integrity(self, verdict):
        assert verdict["R1_REWARD_CONFIG_INTEGRITY"] == "PASS"

    def test_v10_qmix_integrity(self, verdict):
        assert verdict["QMIX_CONFIG_INTEGRITY"] == "PASS"

    def test_v11_c3_integrity(self, verdict):
        assert verdict["C3_CONFIG_INTEGRITY"] == "PASS"

    def test_v12_numerical(self, verdict):
        assert verdict["NUMERICAL_STABILITY"] == "PASS"

    def test_v13_checkpoints(self, verdict):
        assert verdict["CHECKPOINTS_COMPLETE"] == "60/60"

    def test_v14_val_checkpoints(self, verdict):
        assert verdict["VALIDATION_CHECKPOINTS_COMPLETE"] == "60/60"

    def test_v15_c1_selected(self, verdict):
        assert verdict["C1_SELECTED_CHECKPOINTS"] == "5/5"

    def test_v16_c3_selected(self, verdict):
        assert verdict["C3_SELECTED_CHECKPOINTS"] == "5/5"

    def test_v17_no_retuning(self, verdict):
        assert verdict["HYPERPARAMETER_RETUNING"] == "NO"
        assert verdict["ARCHITECTURE_RETUNING"] == "NO"

    def test_v18_no_ckpt_change(self, verdict):
        assert verdict["CHECKPOINT_SELECTION_CHANGED"] == "NO"
        assert verdict["BEST_POLICY_SELECTION"] == "NO"

    def test_v19_final_not_run(self, verdict):
        assert verdict["FINAL_EVALUATION"] == "NOT_RUN"

    def test_v20_no_causal_claim(self, verdict):
        assert verdict["PAIRWISE_CG_CAUSAL_CLAIM"] == "NOT_MADE"

    def test_v21_final_access_zero(self, verdict):
        assert verdict["PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT"] == 0

    def test_v22_historical_e4f(self, verdict):
        assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"

    def test_v23_historical_e4hf(self, verdict):
        assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"

    def test_v24_historical_e4if(self, verdict):
        assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"

    def test_v25_historical_e4jf(self, verdict):
        assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_v26_historical_e4kf(self, verdict):
        assert verdict["E4KF_COORDINATION_CAUSAL_VERDICT"] == "FAILED"

    def test_v27_historical_e4mf(self, verdict):
        assert verdict["E4MF_CREDIT_PRESERVATION_VERDICT"] == "FAILED"

    def test_v28_perf_verdict(self, verdict):
        assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_v29_next_stage(self, verdict):
        assert verdict["NEXT_STAGE"] == \
            "E4RE_PAIRWISE_CG_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"

    def test_v30_c1_val_is_float(self, verdict):
        assert isinstance(verdict["C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"], float)

    def test_v31_c3_val_is_float(self, verdict):
        assert isinstance(verdict["C3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"], float)

    def test_v32_delta_sign_coherent(self, verdict):
        d = verdict["VALIDATION_PAIRED_DELTA_C1_MINUS_C3"]
        c1 = verdict["C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"]
        c3 = verdict["C3_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"]
        assert abs(d - (c1 - c3)) < 1e-12


# ── 2. Pre-execution manifest ────────────────────────────

class TestPreManifest:
    def test_pm01_integrity_ok(self, pre_manifest):
        assert pre_manifest["integrity_ok"] is True

    def test_pm02_e4rt_complete(self, pre_manifest):
        assert pre_manifest["E4RT_COMPLETE"] == "YES"

    def test_pm03_cg_valid(self, pre_manifest):
        assert pre_manifest["PAIRWISE_CG_IMPLEMENTATION_VALID"] == "YES"

    def test_pm04_e4if(self, pre_manifest):
        assert pre_manifest["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"

    def test_pm05_e4jf(self, pre_manifest):
        assert pre_manifest["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"

    def test_pm06_e4kf(self, pre_manifest):
        assert pre_manifest["E4KF_COORDINATION_CAUSAL_VERDICT"] == "FAILED"

    def test_pm07_qmix_sha(self, pre_manifest):
        assert pre_manifest["QMIX_CONFIG_SHA256"] == QMIX_SHA

    def test_pm08_c3_sha(self, pre_manifest):
        assert pre_manifest["C3_CONFIG_SHA256"] == C3_SHA


# ── 3. Execution config ─────────────────────────────────

class TestExecConfig:
    def test_ec01_arms(self, exec_config):
        assert C1_ARM in exec_config["arms"]
        assert C3_ARM in exec_config["arms"]

    def test_ec02_seeds(self, exec_config):
        assert exec_config["train_seeds"] == [2501, 2560]
        assert exec_config["val_seeds"] == [2561, 2580]
        assert exec_config["final_seeds_firewalled"] == [2581, 2640]

    def test_ec03_policy_seeds(self, exec_config):
        assert exec_config["policy_seeds"] == POLICY_SEEDS

    def test_ec04_shas(self, exec_config):
        assert exec_config["QMIX_CONFIG_SHA256"] == QMIX_SHA
        assert exec_config["C3_CONFIG_SHA256"] == C3_SHA


# ── 4. Paired initialization ────────────────────────────

class TestPairedInit:
    def test_pi01_all_util_equal(self, init_manifest):
        assert init_manifest["all_util_online_equals_target"] is True

    def test_pi02_all_mixer_equal(self, init_manifest):
        assert init_manifest["all_mixer_online_equals_target"] is True

    def test_pi03_all_pair_equal(self, init_manifest):
        assert init_manifest["all_pair_online_equals_target"] is True

    def test_pi04_distinct_seeds(self, init_manifest):
        assert init_manifest["distinct_across_policy_seeds"] is True

    def test_pi05_c1_c3_unary_identical(self, init_manifest):
        assert init_manifest["C1_C3_unary_init_identical_all"] is True

    def test_pi06_per_policy_has_pair_seed(self, init_manifest):
        for p in POLICY_SEEDS:
            row = init_manifest["per_policy"][str(p)]
            assert "pair_init_seed" in row
            assert row["pair_init_seed"] == p + 1


# ── 5. Single-factor audit ──────────────────────────────

class TestSingleFactor:
    def test_sf01_pass(self, single_factor):
        assert single_factor["SINGLE_FACTOR_INTEGRITY"] == "PASS"

    def test_sf02_factor(self, single_factor):
        assert single_factor["factor"] == \
            "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"

    def test_sf03_arms(self, single_factor):
        assert single_factor["control"] == C1_ARM
        assert single_factor["treatment"] == C3_ARM


# ── 6. Checkpoint manifest ──────────────────────────────

class TestCheckpoints:
    def test_ck01_count(self, ckpt_manifest):
        assert ckpt_manifest["n_checkpoints"] == 60

    def test_ck02_complete(self, ckpt_manifest):
        assert ckpt_manifest["complete"] is True

    def test_ck03_per_arm(self, ckpt_manifest):
        c1_count = sum(1 for c in ckpt_manifest["checkpoints"]
                       if c["arm"] == C1_ARM)
        c3_count = sum(1 for c in ckpt_manifest["checkpoints"]
                       if c["arm"] == C3_ARM)
        assert c1_count == 30
        assert c3_count == 30

    def test_ck04_six_per_run(self, ckpt_manifest):
        for arm in [C1_ARM, C3_ARM]:
            for p in POLICY_SEEDS:
                count = sum(1 for c in ckpt_manifest["checkpoints"]
                            if c["arm"] == arm and c["policy_seed"] == p)
                assert count == 6, f"{arm} policy {p} has {count} checkpoints"

    def test_ck05_all_have_sha(self, ckpt_manifest):
        for c in ckpt_manifest["checkpoints"]:
            assert len(c["sha256"]) == 64

    def test_ck06_all_have_val(self, ckpt_manifest):
        for c in ckpt_manifest["checkpoints"]:
            assert isinstance(c["VALIDATION_MACRO_MIGD_RESPONSE"], float)


# ── 7. Training episodes ────────────────────────────────

class TestTrainingEpisodes:
    def test_ep01_total(self, ep_csv):
        assert len(ep_csv) == 10800

    def test_ep02_c1_count(self, ep_csv):
        c1 = [r for r in ep_csv if r["arm"] == C1_ARM]
        assert len(c1) == 5400

    def test_ep03_c3_count(self, ep_csv):
        c3 = [r for r in ep_csv if r["arm"] == C3_ARM]
        assert len(c3) == 5400

    def test_ep04_scenarios(self, ep_csv):
        scenarios = set(r["scenario"] for r in ep_csv)
        assert scenarios == {"PERSISTENT", "STATIONARY", "MIXED"}

    def test_ep05_seeds_in_range(self, ep_csv):
        for r in ep_csv:
            seed = int(r["environment_seed"])
            assert 2501 <= seed <= 2560

    def test_ep06_no_final_seeds(self, ep_csv):
        for r in ep_csv:
            seed = int(r["environment_seed"])
            assert seed < 2581 or seed > 2640

    def test_ep07_migd_finite(self, ep_csv):
        for r in ep_csv:
            assert not math.isnan(float(r["MIGD_response"]))


# ── 8. Numerical stability ──────────────────────────────

class TestNumerical:
    def test_ns01_overall(self, numerical):
        assert numerical["numerical_stability"] == "PASS"

    def test_ns02_all_runs(self, numerical):
        for key, val in numerical["runs"].items():
            assert val == "PASS", f"{key} = {val}"


# ── 9. Architecture integrity ────────────────────────────

class TestArchIntegrity:
    def test_ai01_c1_arch(self, arch_integrity):
        assert arch_integrity["C1_architecture"] == \
            "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"

    def test_ai02_c3_arch(self, arch_integrity):
        assert arch_integrity["C3_architecture"] == \
            "C3_PAIRWISE_COORDINATION_GRAPH_V1"

    def test_ai03_qmix_sha(self, arch_integrity):
        assert arch_integrity["QMIX_CONFIG_SHA256_MATCHES_E4RT"] is True

    def test_ai04_c3_sha(self, arch_integrity):
        assert arch_integrity["C3_CONFIG_SHA256_MATCHES_E4RT"] is True

    def test_ai05_maxsum_k(self, arch_integrity):
        assert arch_integrity["maxsum_K"] == 10

    def test_ai06_pair_offset(self, arch_integrity):
        assert arch_integrity["pair_seed_offset"] == 1

    def test_ai07_no_adaptation(self, arch_integrity):
        assert arch_integrity["no_runtime_adaptation"] is True


# ── 10. Final seed firewall ──────────────────────────────

class TestFirewall:
    def test_fw01_access_zero(self, firewall):
        assert firewall["PAIRWISE_CG_FINAL_EVAL_ACCESS_COUNT"] == 0

    def test_fw02_final_not_run(self, firewall):
        assert firewall["FINAL_EVALUATION"] == "NOT_RUN"

    def test_fw03_no_bootstrap(self, firewall):
        assert firewall["no_final_bootstrap"] is True

    def test_fw04_seeds(self, firewall):
        assert firewall["final_seeds"] == [2581, 2640]


# ── 11. Leakage audit ───────────────────────────────────

class TestLeakage:
    def test_lk01_train_val_disjoint(self, leakage):
        assert leakage["train_val_disjoint"] is True

    def test_lk02_train_final_disjoint(self, leakage):
        assert leakage["train_final_disjoint"] is True

    def test_lk03_val_final_disjoint(self, leakage):
        assert leakage["val_final_disjoint"] is True


# ── 12. Selected checkpoints ────────────────────────────

class TestSelected:
    def test_sel01_count(self, selected):
        assert selected["n_selected"] == 10

    def test_sel02_c1_five(self, selected):
        c1 = [k for k in selected["selected"] if k.startswith("C1")]
        assert len(c1) == 5

    def test_sel03_c3_five(self, selected):
        c3 = [k for k in selected["selected"] if k.startswith("C3")]
        assert len(c3) == 5

    def test_sel04_each_has_sha(self, selected):
        for key, info in selected["selected"].items():
            assert len(info["sha256"]) == 64

    def test_sel05_each_has_val_migd(self, selected):
        for key, info in selected["selected"].items():
            assert isinstance(
                info["validation_macro_MIGD_response"], float)

    def test_sel06_mixed_action_has_both_arms(self, selected):
        assert C1_ARM in selected["mixed_action_diagnostics"]
        assert C3_ARM in selected["mixed_action_diagnostics"]

    def test_sel07_mixed_predict_finite(self, selected):
        for arm in [C1_ARM, C3_ARM]:
            m = selected["mixed_action_diagnostics"][arm]
            assert isinstance(m["predict_fraction"], float)
            assert 0 <= m["predict_fraction"] <= 1

    def test_sel08_mixed_multi_finite(self, selected):
        for arm in [C1_ARM, C3_ARM]:
            m = selected["mixed_action_diagnostics"][arm]
            assert isinstance(m["multi_action_rate"], float)
            assert 0 <= m["multi_action_rate"] <= 1


# ── 13. Validation A/B summary ──────────────────────────

class TestValidationAB:
    def test_ab01_rows(self, ab_csv):
        assert len(ab_csv) == 4

    def test_ab02_scenarios(self, ab_csv):
        scens = [r["scenario"] for r in ab_csv]
        assert "PERSISTENT" in scens
        assert "STATIONARY" in scens
        assert "MIXED" in scens
        assert "macro" in scens

    def test_ab03_delta_coherent(self, ab_csv):
        for r in ab_csv:
            c1 = float(r["C1_MIGD_response"])
            c3 = float(r["C3_MIGD_response"])
            d = float(r["delta_C1_minus_C3_response"])
            assert abs(d - (c1 - c3)) < 1e-10


# ── 14. Coordination diagnostics ─────────────────────────

class TestDiagnostics:
    def test_dg01_row_count(self, diag_csv):
        assert len(diag_csv) == 10

    def test_dg02_c1_rows(self, diag_csv):
        c1 = [r for r in diag_csv if r["arm"] == C1_ARM]
        assert len(c1) == 5

    def test_dg03_c3_rows(self, diag_csv):
        c3 = [r for r in diag_csv if r["arm"] == C3_ARM]
        assert len(c3) == 5

    def test_dg04_c1_has_monotonicity(self, diag_csv):
        c1 = [r for r in diag_csv if r["arm"] == C1_ARM]
        for r in c1:
            assert r["monotonicity_computed"] != ""

    def test_dg05_c3_has_pair_factor(self, diag_csv):
        c3 = [r for r in diag_csv if r["arm"] == C3_ARM]
        for r in c3:
            assert r["mean_abs_pair_factor"] != ""

    def test_dg06_c3_has_edge_count(self, diag_csv):
        c3 = [r for r in diag_csv if r["arm"] == C3_ARM]
        for r in c3:
            assert r["mean_active_edge_count"] != ""


# ── 15. Report file ─────────────────────────────────────

class TestReport:
    def test_rp01_exists(self):
        assert os.path.isfile(os.path.join(E4RTR, "E4RTR_report.md"))

    def test_rp02_mentions_c1_c3(self):
        txt = open(os.path.join(E4RTR, "E4RTR_report.md")).read()
        assert "C1" in txt
        assert "C3" in txt

    def test_rp03_mentions_not_made(self):
        txt = open(os.path.join(E4RTR, "E4RTR_report.md")).read()
        assert "NOT_MADE" in txt


# ── 16. Output file completeness ─────────────────────────

class TestOutputFiles:
    expected_json = [
        "E4RTR_pre_execution_manifest.json",
        "E4RTR_execution_config.json",
        "E4RTR_pair_initialization_manifest.json",
        "E4RTR_single_factor_audit.json",
        "E4RTR_checkpoint_manifest.json",
        "E4RTR_numerical_stability.json",
        "E4RTR_runtime_architecture_integrity.json",
        "E4RTR_selected_checkpoints.json",
        "E4RTR_training_validation_leakage_audit.json",
        "E4RTR_final_firewall.json",
        "E4RTR_verdict.json",
    ]
    expected_csv = [
        "E4RTR_training_episode_results.csv",
        "E4RTR_training_update_summary.csv",
        "E4RTR_coordination_diagnostics.csv",
        "E4RTR_action_trajectory.csv",
        "E4RTR_validation_selected_policies.csv",
        "E4RTR_validation_all_checkpoints.csv",
        "E4RTR_validation_ab_summary.csv",
        "E4RTR_validation_coordination_diagnostics.csv",
        "E4RTR_validation_action_context.csv",
    ]

    @pytest.mark.parametrize("name", expected_json)
    def test_json_exists(self, name):
        path = os.path.join(E4RTR, name)
        assert os.path.isfile(path), f"missing {name}"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    @pytest.mark.parametrize("name", expected_csv)
    def test_csv_exists(self, name):
        path = os.path.join(E4RTR, name)
        assert os.path.isfile(path), f"missing {name}"
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0


# ── 17. Checkpoint files on disk ─────────────────────────

class TestCheckpointFiles:
    @pytest.mark.parametrize("arm", [C1_ARM, C3_ARM])
    def test_cf01_dirs_exist(self, arm):
        for p in POLICY_SEEDS:
            d = os.path.join(E4RTR, arm, f"policy_{p}", "checkpoints")
            assert os.path.isdir(d), f"missing {d}"

    def test_cf02_c1_checkpoint_has_mixer(self, ckpt_manifest):
        c1_paths = [c["path"] for c in ckpt_manifest["checkpoints"]
                    if c["arm"] == C1_ARM]
        import torch
        for path in c1_paths[:1]:
            payload = torch.load(path, weights_only=False, map_location="cpu")
            assert "mixer_online_state_dict" in payload

    def test_cf03_c3_checkpoint_has_pair(self, ckpt_manifest):
        c3_paths = [c["path"] for c in ckpt_manifest["checkpoints"]
                    if c["arm"] == C3_ARM]
        import torch
        for path in c3_paths[:1]:
            payload = torch.load(path, weights_only=False, map_location="cpu")
            assert "pair_online_state_dict" in payload
            assert "pair_target_state_dict" in payload

    def test_cf04_c3_checkpoint_has_c3_sha(self, ckpt_manifest):
        c3_paths = [c["path"] for c in ckpt_manifest["checkpoints"]
                    if c["arm"] == C3_ARM]
        import torch
        for path in c3_paths[:1]:
            payload = torch.load(path, weights_only=False, map_location="cpu")
            assert payload["C3_CONFIG_SHA256"] == C3_SHA


# ── 18. Seed disjointness ───────────────────────────────

class TestSeedDisjointness:
    def test_sd01_train_val(self):
        train = set(range(2501, 2561))
        val = set(range(2561, 2581))
        assert train.isdisjoint(val)

    def test_sd02_train_final(self):
        train = set(range(2501, 2561))
        final = set(range(2581, 2641))
        assert train.isdisjoint(final)

    def test_sd03_val_final(self):
        val = set(range(2561, 2581))
        final = set(range(2581, 2641))
        assert val.isdisjoint(final)


# ── 19. Validation checkpoint table ──────────────────────

class TestValAllCheckpoints:
    def test_va01_row_count(self):
        rows = load_csv(E4RTR, "E4RTR_validation_all_checkpoints.csv")
        assert len(rows) == 60

    def test_va02_exactly_10_selected(self):
        rows = load_csv(E4RTR, "E4RTR_validation_all_checkpoints.csv")
        sel = [r for r in rows if r["selected"] == "True"]
        assert len(sel) == 10


# ── 20. Action trajectory ───────────────────────────────

class TestActionTrajectory:
    def test_at01_nonempty(self):
        rows = load_csv(E4RTR, "E4RTR_action_trajectory.csv")
        assert len(rows) > 0

    def test_at02_has_both_arms(self):
        rows = load_csv(E4RTR, "E4RTR_action_trajectory.csv")
        arms = set(r["arm"] for r in rows)
        assert C1_ARM in arms
        assert C3_ARM in arms

    def test_at03_fracs_sum_one(self):
        rows = load_csv(E4RTR, "E4RTR_action_trajectory.csv")
        for r in rows:
            total = sum(float(r[f"frac_{a}"]) for a in
                        ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"])
            assert abs(total - 1.0) < 0.01


# ── 21. Training update summary ─────────────────────────

class TestTrainUpdateSummary:
    def test_us01_nonempty(self):
        rows = load_csv(E4RTR, "E4RTR_training_update_summary.csv")
        assert len(rows) > 0

    def test_us02_has_epochs(self):
        rows = load_csv(E4RTR, "E4RTR_training_update_summary.csv")
        epochs = set(int(r["epoch"]) for r in rows)
        assert 0 in epochs
        assert 5 in epochs
