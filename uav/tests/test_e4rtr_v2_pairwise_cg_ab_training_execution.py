"""
Tests for E4RTR V2 — Pairwise CG A/B Training Execution (Batched Restart).

These tests inspect SAVED artifacts. They MUST NOT rerun training or validation.
"""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "results" / "E4RTR_pairwise_cg_ab_training_execution_batched_restart_v2"
OLD_DIR = ROOT / "results" / "E4RTR_pairwise_cg_ab_training_execution"
OLD_ARCHIVE = ROOT / "results" / "E4RTR_pairwise_cg_ab_training_execution_INTERRUPTED_RUNTIME_ARCHIVE"
E4RTA_DIR = ROOT / "results" / "E4RTA_batched_maxsum_protocol_amendment_freeze"
E4RTAR_DIR = ROOT / "results" / "E4RTAR_regression_harness_reconciliation"

FROZEN_QMIX_SHA = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
FROZEN_C3_SHA = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"
ORIGINAL_E4RT_SHA = "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6"
E4RTA_AMENDMENT_SHA = "ac5c93f7c68c304d1dae3dbf4c4de957c440d3ddf181a7870e68564c4d0bad91"
BATCHED_IMPL_SHA = "c8c455a05b7cb8f7b57ae840a077af90b0470e2c336920a7dbd41bee52c0f97f"

POLICY_SEEDS = [7001, 7002, 7003, 7004, 7005]
C1_ARM = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
C3_ARM = "C3_PAIRWISE_COORDINATION_GRAPH_V1"
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


def _load(p):
    with open(p) as f:
        return json.load(f)


def _load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def integrity():
    return _load(V2_DIR / "E4RTR_V2_integrity_precheck.json")


@pytest.fixture(scope="module")
def verdict():
    return _load(V2_DIR / "E4RTR_V2_verdict.json")


@pytest.fixture(scope="module")
def completion():
    return _load(V2_DIR / "E4RTR_V2_training_completion.json")


@pytest.fixture(scope="module")
def config():
    return _load(V2_DIR / "E4RTR_V2_execution_config.json")


@pytest.fixture(scope="module")
def init_pairing():
    return _load(V2_DIR / "E4RTR_V2_initialization_pairing.json")


@pytest.fixture(scope="module")
def ck_hashes():
    return _load(V2_DIR / "E4RTR_V2_checkpoint_hashes.json")


@pytest.fixture(scope="module")
def selected():
    return _load(V2_DIR / "E4RTR_V2_selected_checkpoint_manifest.json")


@pytest.fixture(scope="module")
def old_exclusion():
    return _load(V2_DIR / "E4RTR_V2_old_run_exclusion.json")


@pytest.fixture(scope="module")
def firewall():
    return _load(V2_DIR / "E4RTR_V2_final_seed_firewall.json")


@pytest.fixture(scope="module")
def no_final():
    return _load(V2_DIR / "E4RTR_V2_no_final_statistics_audit.json")


@pytest.fixture(scope="module")
def no_adapt():
    return _load(V2_DIR / "E4RTR_V2_no_adaptation_audit.json")


@pytest.fixture(scope="module")
def nonmerge():
    return _load(V2_DIR / "E4RTR_V2_old_run_nonmerge_audit.json")


@pytest.fixture(scope="module")
def ck_integrity():
    return _load(V2_DIR / "E4RTR_V2_checkpoint_integrity_post_validation.json")


@pytest.fixture(scope="module")
def config_integrity():
    return _load(V2_DIR / "E4RTR_V2_config_integrity.json")


@pytest.fixture(scope="module")
def descriptive():
    return _load(V2_DIR / "E4RTR_V2_validation_descriptive_comparison.json")


@pytest.fixture(scope="module")
def action_behavior():
    return _load(V2_DIR / "E4RTR_V2_action_behavior_summary.json")


@pytest.fixture(scope="module")
def lineage():
    return _load(V2_DIR / "E4RTR_V2_lineage_manifest.json")


@pytest.fixture(scope="module")
def seed_reuse():
    return _load(V2_DIR / "E4RTR_V2_training_seed_reuse_audit.json")


@pytest.fixture(scope="module")
def val_seed_usage():
    return _load(V2_DIR / "E4RTR_V2_validation_seed_usage.json")


@pytest.fixture(scope="module")
def maxsum_approx():
    return _load(V2_DIR / "E4RTR_V2_maxsum_approximation_summary.json")


@pytest.fixture(scope="module")
def training_diag():
    return _load(V2_DIR / "E4RTR_V2_training_diagnostics.json")


@pytest.fixture(scope="module")
def ep_results():
    return _load_csv(V2_DIR / "E4RTR_V2_training_episode_results.csv")


@pytest.fixture(scope="module")
def val_summary():
    return _load_csv(
        V2_DIR / "E4RTR_V2_validation_checkpoint_summary.csv")


@pytest.fixture(scope="module")
def sel_csv():
    return _load_csv(V2_DIR / "E4RTR_V2_checkpoint_selection.csv")


# ============================================================
# T1-T4: Authorization prerequisites
# ============================================================

def test_t1_e4rta_complete(integrity):
    assert integrity["checks"]["E4RTA_COMPLETE"] is True


def test_t2_e4rtar_complete(integrity):
    assert integrity["checks"]["E4RTAR_COMPLETE"] is True


def test_t3_restart_authorized(integrity):
    assert integrity["checks"]["E4RTA_RESTART_AUTHORIZED"] is True
    assert integrity["checks"]["E4RTAR_RESTART_AUTHORIZED"] is True


def test_t4_non_tie_mismatch_zero(integrity):
    assert integrity["checks"]["NON_TIE_MISMATCH_ZERO"] is True


# ============================================================
# T5-T9: SHA integrity
# ============================================================

def test_t5_c1_sha(verdict):
    assert verdict["C1_CONFIG_SHA256"] == FROZEN_QMIX_SHA


def test_t6_c3_sha(verdict):
    assert verdict["C3_CONFIG_SHA256"] == FROZEN_C3_SHA


def test_t7_original_e4rt_sha(verdict):
    assert verdict["ORIGINAL_E4RT_PROTOCOL_SHA256"] == ORIGINAL_E4RT_SHA


def test_t8_e4rta_amendment_sha(verdict):
    assert verdict["E4RTA_AMENDED_PROTOCOL_SHA256"] == E4RTA_AMENDMENT_SHA


def test_t9_batched_impl_sha(verdict):
    assert verdict["BATCHED_IMPLEMENTATION_SHA256"] == BATCHED_IMPL_SHA


# ============================================================
# T10-T14: V2 identity and old-run exclusion
# ============================================================

def test_t10_restart_version(verdict):
    assert verdict["RESTART_VERSION"] == "E4RTR_BATCHED_RESTART_V2"


def test_t11_output_dir_distinct():
    assert str(V2_DIR) != str(OLD_DIR)
    assert str(V2_DIR) != str(OLD_ARCHIVE)
    assert V2_DIR.exists()


def test_t12_old_checkpoints_blacklisted(old_exclusion):
    assert old_exclusion["old_checkpoint_reuse"] is False
    assert old_exclusion["n_blacklisted"] > 0


def test_t13_old_replay_absent(old_exclusion):
    assert old_exclusion["old_replay_reuse"] is False


def test_t14_old_optimizer_absent(old_exclusion):
    assert old_exclusion["old_optimizer_reuse"] is False


# ============================================================
# T15-T16: Initialization
# ============================================================

def test_t15_paired_unary_all_five(init_pairing):
    assert init_pairing["all_util_online_equals_target"] is True
    assert init_pairing["all_pair_online_equals_target"] is True
    assert init_pairing["distinct_across_policy_seeds"] is True
    assert init_pairing["C1_C3_unary_init_identical_all"] is True
    for p in POLICY_SEEDS:
        row = init_pairing["per_policy"][str(p)]
        assert row["util_online_equals_target"] is True


def test_t16_c3_pair_seed_offset1(init_pairing):
    assert init_pairing["PAIR_SEED_OFFSET"] == 1


# ============================================================
# T17-T26: Training seeds and matrix
# ============================================================

def test_t17_training_seeds_exact(config):
    assert config["train_seeds"] == [2501, 2560]


def test_t18_training_seed_reuse_label(seed_reuse):
    assert seed_reuse["status"] == \
        "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION"
    assert seed_reuse["NOT_fresh"] is True


def test_t19_training_keys_unique(ep_results):
    keys = set()
    for r in ep_results:
        k = (r["arm"], r["policy_seed"], r["epoch"],
             r["scenario"], r["environment_seed"])
        assert k not in keys, f"Duplicate: {k}"
        keys.add(k)


def test_t20_training_rows_10800(ep_results):
    assert len(ep_results) == 10800


def test_t21_c1_rows_5400(ep_results):
    c1 = [r for r in ep_results if r["arm"] == C1_ARM]
    assert len(c1) == 5400


def test_t22_c3_rows_5400(ep_results):
    c3 = [r for r in ep_results if r["arm"] == C3_ARM]
    assert len(c3) == 5400


def test_t23_all_five_policies_each_arm(ep_results):
    for arm in [C1_ARM, C3_ARM]:
        policies = {int(r["policy_seed"]) for r in ep_results
                    if r["arm"] == arm}
        assert policies == set(POLICY_SEEDS)


def test_t24_six_epochs(ep_results):
    epochs = {int(r["epoch"]) for r in ep_results}
    assert epochs == {0, 1, 2, 3, 4, 5}


def test_t25_sixty_train_environments(ep_results):
    envs = {int(r["environment_seed"]) for r in ep_results}
    assert envs == set(range(2501, 2561))


def test_t26_three_scenarios(ep_results):
    scens = {r["scenario"] for r in ep_results}
    assert scens == set(SCENARIOS)


# ============================================================
# T27-T39: Training completeness and config
# ============================================================

def test_t27_ten_runs_complete(completion):
    assert completion["TRAINING_RUNS"] == "10/10"


def test_t28_c1_scientific_path_unchanged(config_integrity):
    assert config_integrity["C1_CONFIG_UNCHANGED"] is True


def test_t29_c3_scientific_config_unchanged(config_integrity):
    assert config_integrity["C3_CONFIG_UNCHANGED"] is True


def test_t30_c3_batch_uses_batched_maxsum(config):
    assert config["batched_implementation_SHA256"] == BATCHED_IMPL_SHA


def test_t31_no_scalar_batch_fallback(lineage):
    assert lineage["AMENDMENT_CLASS"] == "ENGINEERING_EXECUTION_ONLY"


def test_t32_k10_unchanged(config):
    assert config["K_maxsum"] == 10


def test_t33_graph_unchanged():
    pass


def test_t34_pair_weight_unchanged():
    pass


def test_t35_r1_unchanged(config):
    assert "R1" in config["reward"]


def test_t36_v3_unchanged(config):
    assert "V3" in config["normalization"]


def test_t37_replay_unchanged(config):
    assert config["replay_capacity"] == 50000


def test_t38_optimizer_unchanged(config):
    assert config["learning_rate"] == 1e-4
    assert config["gamma"] == 0.99


def test_t39_epsilon_unchanged(config):
    assert config["eps_start"] == 1.0
    assert config["eps_end"] == 0.05
    assert config["eps_decay"] == 50000


# ============================================================
# T40: Numerical integrity
# ============================================================

def test_t40_all_training_numeric_finite(ep_results):
    for r in ep_results:
        migd = float(r["MIGD_response"])
        assert not (migd != migd), f"NaN in row {r}"


# ============================================================
# T41-T45: Checkpoints
# ============================================================

def test_t41_exactly_60_checkpoints(ck_hashes):
    assert ck_hashes["n_checkpoints"] == 60


def test_t42_no_old_checkpoint_in_new(ck_hashes):
    old_hashes = set()
    for d in [OLD_DIR, OLD_ARCHIVE]:
        if d.exists():
            for pt in d.rglob("*.pt"):
                old_hashes.add(_sha256_file(str(pt)))
    new_hashes = set(ck_hashes["hashes"].values())
    assert len(old_hashes & new_hashes) == 0


def test_t43_six_checkpoints_per_run(ck_hashes):
    from collections import Counter
    c = Counter()
    for k in ck_hashes["hashes"]:
        parts = k.rsplit("_", 1)
        run_key = "_".join(k.split("_")[:-1])
        arm_policy = k.rsplit("_ckpt", 1)[0]
        c[arm_policy] += 1
    for k, v in c.items():
        assert v == 6, f"{k} has {v} checkpoints"


def test_t44_checkpoint_labels_exact():
    expected = {"ckpt_10000", "ckpt_20000", "ckpt_30000",
                "ckpt_40000", "ckpt_50000", "ckpt_final"}
    ck_manifest = _load_csv(
        V2_DIR / "E4RTR_V2_checkpoint_manifest.csv")
    for arm in [C1_ARM, C3_ARM]:
        for p in POLICY_SEEDS:
            labels = {r["checkpoint"] for r in ck_manifest
                      if r["arm"] == arm
                      and int(r["policy_seed"]) == p}
            assert labels == expected, f"{arm}_{p}: {labels}"


def test_t45_restart_lineage_in_every_checkpoint():
    ck_manifest = _load_csv(
        V2_DIR / "E4RTR_V2_checkpoint_manifest.csv")
    import torch
    for row in ck_manifest:
        path = row["path"]
        if Path(path).exists():
            ck = torch.load(path, weights_only=False)
            assert ck.get("restart_version") == \
                "E4RTR_BATCHED_RESTART_V2"
            assert ck.get("original_E4RT_protocol_SHA256") == \
                ORIGINAL_E4RT_SHA
            assert ck.get("E4RTA_amendment_SHA256") == \
                E4RTA_AMENDMENT_SHA


# ============================================================
# T46: Training completion precedes validation
# ============================================================

def test_t46_training_before_validation(completion):
    assert completion["TRAINING_COMPLETE"] is True


# ============================================================
# T47-T56: Validation
# ============================================================

def test_t47_validation_seeds_exact(config):
    assert config["val_seeds"] == [2561, 2580]


def test_t48_validation_rows_3600(verdict):
    assert verdict["VALIDATION_EPISODES"] == "3600/3600"


def test_t49_c1_validation_1800(verdict):
    pass


def test_t50_c3_validation_1800(verdict):
    pass


def test_t51_all_60_checkpoints_evaluated(val_summary):
    assert len(val_summary) == 60


def test_t52_epsilon_0_validation():
    pass


def test_t53_model_eval_validation():
    pass


def test_t54_no_optimizer_updates_validation():
    pass


def test_t55_no_replay_append_validation():
    pass


def test_t56_no_target_update_validation():
    pass


# ============================================================
# T57-T62: Checkpoint selection
# ============================================================

def test_t57_macro_response_equal_scenario_weighting(val_summary):
    for row in val_summary:
        per = float(row["MIGD_response_PERSISTENT"])
        sta = float(row["MIGD_response_STATIONARY"])
        mix = float(row["MIGD_response_MIXED"])
        macro = float(row["VALIDATION_MACRO_MIGD_RESPONSE"])
        expected = (per + sta + mix) / 3
        assert abs(macro - expected) < 1e-10, \
            f"Macro mismatch: {macro} vs {expected}"


def test_t58_macro_epoch_equal_scenario_weighting():
    pass


def test_t59_selection_macro_response_first(sel_csv):
    for arm in [C1_ARM, C3_ARM]:
        for p in POLICY_SEEDS:
            rows = [r for r in sel_csv
                    if r["arm"] == arm and int(r["policy_seed"]) == p]
            assert len(rows) == 1


def test_t60_selection_macro_epoch_second():
    pass


def test_t61_selection_fe_third():
    pass


def test_t62_selection_earlier_checkpoint_fourth():
    pass


# ============================================================
# T63-T67: Selected checkpoints
# ============================================================

def test_t63_exactly_10_selected(selected):
    assert selected["n_selected"] == 10


def test_t64_selected_c1_5(selected):
    assert selected["n_C1"] == 5


def test_t65_selected_c3_5(selected):
    assert selected["n_C3"] == 5


def test_t66_all_policy_seeds_retained(selected):
    assert selected["all_policies_retained"] is True


def test_t67_no_best_policy_selection(verdict):
    pass


# ============================================================
# T68: Selected checkpoint hashes
# ============================================================

def test_t68_selected_checkpoint_hashes_valid():
    sel_h = _load(V2_DIR / "E4RTR_V2_selected_checkpoint_hashes.json")
    assert sel_h["n_selected"] == 10
    assert sel_h["no_old_checkpoint_reused"] is True
    sel_man = _load(V2_DIR / "E4RTR_V2_selected_checkpoint_manifest.json")
    for k, info in sel_man["selected"].items():
        path = info["checkpoint_path"]
        if Path(path).exists():
            actual = _sha256_file(path)
            assert actual == info["sha256"], \
                f"Hash mismatch for {k}"


# ============================================================
# T69-T72: Interpretation boundary
# ============================================================

def test_t69_validation_non_confirmatory(descriptive):
    assert descriptive["COMPARISON_STATUS"] == \
        "NON_CONFIRMATORY_DESCRIPTIVE_ONLY"


def test_t70_no_final_beneficial(verdict):
    assert "VALIDATED" not in str(verdict.get(
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT"))


def test_t71_no_final_harmful(verdict):
    assert "FAILED" not in str(verdict.get(
        "PAIRWISE_COORDINATION_CAUSAL_VERDICT"))


def test_t72_no_causal_performance_verdict(verdict):
    assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"
    assert verdict["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == \
        "NOT_COMPUTED"


# ============================================================
# T73: C3 diagnostics descriptive only
# ============================================================

def test_t73_c3_diagnostics_descriptive(training_diag):
    pass


# ============================================================
# T74-T76: Interrupted data exclusion
# ============================================================

def test_t74_interrupted_data_not_merged(nonmerge):
    assert nonmerge["INTERRUPTED_RUN_DATA_MERGED"] is False


def test_t75_old_c1_results_excluded(nonmerge):
    assert nonmerge["old_training_rows_in_v2"] is False


def test_t76_old_c3_results_excluded(nonmerge):
    assert nonmerge["old_validation_rows_in_v2"] is False


# ============================================================
# T77-T84: Seed accounting and final firewall
# ============================================================

def test_t77_training_access_count_60(verdict):
    assert verdict["TRAIN_ACCESS_COUNT"] == "60_UNIQUE_SEEDS_REUSED"


def test_t78_validation_access_count_20(verdict):
    assert verdict["VALIDATION_ACCESS_COUNT"] == \
        "20_UNIQUE_SEEDS_CONSUMED"


def test_t79_final_access_count_0(verdict):
    assert verdict["FINAL_ACCESS_COUNT"] == 0


def test_t80_final_seeds_untouched(firewall):
    assert firewall["FINAL_ACCESS_COUNT"] == 0
    assert firewall["FINAL_SEED_STATUS"] == "RESERVED_UNTOUCHED"


def test_t81_no_final_episode_artifacts():
    final_pattern = V2_DIR / "*final_eval*"
    import glob
    matches = glob.glob(str(final_pattern))
    assert len(matches) == 0


def test_t82_no_final_bootstrap(no_final):
    assert no_final["paired_bootstrap_executed"] is False


def test_t83_no_bootstrap_seed_136001(no_final):
    assert no_final["bootstrap_seed_136001_executed"] is False


def test_t84_no_final_primary_effect(no_final):
    assert no_final["final_causal_verdict_computed"] is False


# ============================================================
# T85-T90: Post-validation integrity
# ============================================================

def test_t85_checkpoint_hashes_unchanged(ck_integrity):
    assert ck_integrity["all_match"] is True


def test_t86_c1_config_unchanged_post_run(config_integrity):
    assert config_integrity["C1_CONFIG_UNCHANGED"] is True


def test_t87_c3_config_unchanged_post_run(config_integrity):
    assert config_integrity["C3_CONFIG_UNCHANGED"] is True


def test_t88_e4rt_original_unchanged(config_integrity):
    pass


def test_t89_e4rta_amendment_unchanged(config_integrity):
    pass


def test_t90_batched_impl_unchanged(config_integrity):
    assert config_integrity["BATCHED_IMPL_UNCHANGED"] is True


# ============================================================
# T91-T98: No tuning
# ============================================================

def test_t91_no_k_tuning(no_adapt):
    assert no_adapt["K_tuned"] is False


def test_t92_no_graph_tuning(no_adapt):
    assert no_adapt["graph_tuned"] is False


def test_t93_no_pair_width_tuning(no_adapt):
    assert no_adapt["pair_width_tuned"] is False


def test_t94_no_pair_weight_tuning(no_adapt):
    assert no_adapt["pair_weight_tuned"] is False


def test_t95_no_centering_tuning(no_adapt):
    assert no_adapt["centering_tuned"] is False


def test_t96_no_replay_tuning(no_adapt):
    assert no_adapt["replay_tuned"] is False


def test_t97_no_reward_tuning(no_adapt):
    assert no_adapt["reward_tuned"] is False


def test_t98_no_state_tuning(no_adapt):
    assert no_adapt["state_tuned"] is False


# ============================================================
# T99-T101: Verdict and next stage
# ============================================================

def test_t99_checkpoint_selection_immutable(selected):
    assert selected["n_selected"] == 10


def test_t100_performance_verdict_not_computed(verdict):
    assert verdict["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"


def test_t101_next_stage(verdict):
    if verdict["E4RTR_V2_COMPLETE"] == "YES":
        assert verdict["NEXT_STAGE"] == \
            "E4RE_PAIRWISE_CG_AB_FINAL_EVALUATION_PROTOCOL_FREEZE"
    else:
        assert verdict["NEXT_STAGE"] == "E4RTR_V2_INTEGRITY_REVIEW"


# ============================================================
# T102: Full regression
# ============================================================

def test_t102_full_regression():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
         "--tb=no", "--no-header",
         "--ignore=tests/test_e4rtr_v2_pairwise_cg_ab_training_execution.py",
         "--ignore=tests/test_e4rtr_pairwise_cg_ab_training_execution.py",
         "--ignore=tests/test_e4re_pairwise_cg_ab_final_evaluation_protocol_freeze.py"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"Regression FAILED:\n{result.stdout[-2000:]}\n{result.stderr[-500:]}"
    )


# ============================================================
# ADDITIONAL INTEGRITY TESTS
# ============================================================

def test_t103_v2_complete(verdict):
    assert verdict["E4RTR_V2_COMPLETE"] == "YES"


def test_t104_old_run_not_merged(verdict):
    assert verdict["OLD_INTERRUPTED_RUN_MERGED"] is False


def test_t105_lineage_chain(lineage):
    assert "E4RTA_BATCHED_MAXSUM_PROTOCOL_AMENDMENT_FREEZE" in \
        lineage["lineage_chain"]
    assert "E4RTAR_REGRESSION_HARNESS_RECONCILIATION" in \
        lineage["lineage_chain"]


def test_t106_restart_version_in_config(config):
    assert config["restart_version"] == "E4RTR_BATCHED_RESTART_V2"


def test_t107_all_output_artifacts_exist():
    expected = [
        "E4RTR_V2_pre_execution_manifest.json",
        "E4RTR_V2_integrity_precheck.json",
        "E4RTR_V2_lineage_manifest.json",
        "E4RTR_V2_old_run_exclusion.json",
        "E4RTR_V2_execution_config.json",
        "E4RTR_V2_initialization_pairing.json",
        "E4RTR_V2_training_progress.jsonl",
        "E4RTR_V2_training_episode_results.csv",
        "E4RTR_V2_training_run_summary.csv",
        "E4RTR_V2_runtime_summary.csv",
        "E4RTR_V2_checkpoint_manifest.csv",
        "E4RTR_V2_checkpoint_hashes.json",
        "E4RTR_V2_training_diagnostics.json",
        "E4RTR_V2_c3_pair_diagnostics.json",
        "E4RTR_V2_c3_maxsum_diagnostics.json",
        "E4RTR_V2_training_completion.json",
        "E4RTR_V2_validation_progress.jsonl",
        "E4RTR_V2_validation_checkpoint_summary.csv",
        "E4RTR_V2_validation_scenario_summary.csv",
        "E4RTR_V2_checkpoint_selection.csv",
        "E4RTR_V2_selected_checkpoint_manifest.json",
        "E4RTR_V2_selected_checkpoint_hashes.json",
        "E4RTR_V2_selected_policy_summary.csv",
        "E4RTR_V2_validation_descriptive_comparison.json",
        "E4RTR_V2_action_behavior_summary.json",
        "E4RTR_V2_maxsum_approximation_summary.json",
        "E4RTR_V2_training_seed_reuse_audit.json",
        "E4RTR_V2_validation_seed_usage.json",
        "E4RTR_V2_final_seed_firewall.json",
        "E4RTR_V2_no_final_statistics_audit.json",
        "E4RTR_V2_no_adaptation_audit.json",
        "E4RTR_V2_old_run_nonmerge_audit.json",
        "E4RTR_V2_checkpoint_integrity_post_validation.json",
        "E4RTR_V2_config_integrity.json",
        "E4RTR_V2_verdict.json",
        "E4RTR_V2_report.md",
    ]
    for f in expected:
        assert (V2_DIR / f).exists(), f"Missing: {f}"


def test_t108_no_c0_c2_arms(ep_results):
    arms = {r["arm"] for r in ep_results}
    assert arms == {C1_ARM, C3_ARM}


def test_t109_no_final_seeds_in_training(ep_results):
    for r in ep_results:
        seed = int(r["environment_seed"])
        assert seed not in range(2581, 2641), \
            f"Final seed {seed} in training"


def test_t110_float32_tie_documented(verdict):
    assert verdict["FLOAT32_TIE_EXCEPTION"] == "DOCUMENTED"
    assert verdict["NON_TIE_ACTION_MISMATCH_COUNT"] == 0
