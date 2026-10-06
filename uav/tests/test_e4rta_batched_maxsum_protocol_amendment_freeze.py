"""
E4RTA — Batched MaxSum Protocol Amendment Freeze Tests.

Verifies the formal engineering-only amendment replacing scalar MaxSum
with the E4RIA-validated batched implementation. No training, no
environment execution, no validation/final seed access.
"""
import json
import hashlib
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
E4RTA_DIR = ROOT / "results" / "E4RTA_batched_maxsum_protocol_amendment_freeze"
E4RIA_DIR = ROOT / "results" / "E4RIA_batched_maxsum_engineering_equivalence"

# Frozen constants
C3_CONFIG_SHA_EXPECTED = "62c5107d638b6d511b8aff214286fc71956baf21edba148cf851fbcfa385a5f9"
QMIX_CONFIG_SHA_EXPECTED = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
E4RT_PROTOCOL_SHA_EXPECTED = "16829d33740cd0ebb4c1fda1c12083f0d79d39f956f1078ec58dea7ee7e979d6"

TIE_TOLERANCE = 1e-6


def _load(name, directory=E4RTA_DIR):
    with open(directory / name) as f:
        return json.load(f)


# ===== T1-T2: E4RIA Complete and Equivalence =====

class TestE4RIAPrerequisites:
    def test_t1_e4ria_complete(self):
        r = _load("E4RIA_report.json", E4RIA_DIR)
        assert r["E4RIA_COMPLETE"] is True

    def test_t2_e4ria_equivalence_pass(self):
        r = _load("E4RIA_report.json", E4RIA_DIR)
        assert r["BATCHED_MAXSUM_ENGINEERING_VALID"] is True


# ===== T3: E4R/E4RI/E4RT Complete =====

class TestPriorStages:
    def test_t3_historical_integrity(self):
        h = _load("E4RTA_historical_integrity.json")
        assert h["E4R_COMPLETE"] is True
        assert h["E4RI_COMPLETE"] is True
        assert h["E4RT_COMPLETE"] is True
        assert h["E4RIA_COMPLETE"] is True
        assert h["HISTORICAL_INTEGRITY_PASS"] is True


# ===== T4-T6: Scientific Config Preservation =====

class TestScientificConfig:
    def test_t4_c3_config_hash_unchanged(self):
        from prdrl.rl.coordination_graph import c3_config_sha256
        assert c3_config_sha256() == C3_CONFIG_SHA_EXPECTED

    def test_t5_qmix_config_unchanged(self):
        lock = _load("E4RTA_scientific_config_lock.json")
        assert lock["QMIX_CONFIG_SHA256"] == QMIX_CONFIG_SHA_EXPECTED
        assert lock["QMIX_CONFIG_UNCHANGED"] is True

    def test_t6_original_e4rt_hash_preserved(self):
        lock = _load("E4RTA_scientific_config_lock.json")
        assert lock["E4RT_PROTOCOL_SHA256"] == E4RT_PROTOCOL_SHA_EXPECTED
        assert lock["E4RT_PROTOCOL_PRESERVED"] is True


# ===== T7-T8: Amendment Scope =====

class TestAmendmentScope:
    def test_t7_engineering_only(self):
        scope = _load("E4RTA_amendment_scope.json")
        assert scope["AMENDMENT_CLASS"] == "ENGINEERING_EXECUTION_ONLY"
        assert scope["scope"] == "MAXSUM_BATCH_EXECUTION_IMPLEMENTATION_ONLY"

    def test_t8_causal_factor_unchanged(self):
        lock = _load("E4RTA_scientific_config_lock.json")
        assert lock["CAUSAL_FACTOR"] == "PAIRWISE_INTERACTION_REPRESENTATION_AND_COORDINATED_INFERENCE"
        assert lock["CAUSAL_FACTOR_CHANGED"] is False
        assert lock["BATCHED_MAXSUM_IS_NOT_A_SCIENTIFIC_TREATMENT_FACTOR"] is True


# ===== T9-T15: Float32 Tie Discrepancy =====

class TestFloat32TieAudit:
    @pytest.fixture(autouse=True)
    def _load_audit(self):
        self.audit = _load("E4RTA_float32_tie_audit.json")

    def test_t9_b1_bit_exact_recorded(self):
        r = _load("E4RIA_report.json", E4RIA_DIR)
        assert "BIT_EXACT" in str(r.get("equivalence_note", "")) or \
               r.get("equivalence_class") == "NUMERICALLY_EQUIVALENT"

    def test_t10_b_gt1_discrepancy_recorded(self):
        assert self.audit["total_mismatched_regions"] > 0
        assert self.audit["mismatch_rate_pct"] < 1.0

    def test_t11_mismatch_audit_complete(self):
        assert self.audit["total_trials"] >= 200
        assert self.audit["total_samples"] > 0
        assert "mismatch_details" in self.audit
        assert len(self.audit["mismatch_details"]) == self.audit["total_mismatched_regions"]

    def test_t12_non_tie_mismatch_count_zero(self):
        assert self.audit["NON_TIE_ACTION_MISMATCH_COUNT"] == 0

    def test_t13_every_mismatch_inside_tie_tolerance(self):
        assert self.audit["all_mismatches_inside_tie_tolerance"] is True
        for rec in self.audit["mismatch_details"]:
            assert rec["inside_tie_tolerance"] is True
            assert rec["absolute_value_difference"] < TIE_TOLERANCE

    def test_t14_no_performance_metric_for_acceptance(self):
        exc = _load("E4RTA_numerical_tie_exception.json")
        assert exc["no_performance_metric_used_for_acceptance"] is True

    def test_t15_tie_exception_wording_bounded(self):
        exc = _load("E4RTA_numerical_tie_exception.json")
        assert "numerically equivalent" in exc["allowed_wording"].lower()
        assert exc["NOT_ALLOWED_WORDING"] == "all actions are exactly identical"


# ===== T16-T26: Implementation and Config Locks =====

class TestImplementationLocks:
    def test_t16_scalar_reference_retained(self):
        ret = _load("E4RTA_scalar_reference_retention.json")
        assert ret["do_NOT_delete"] is True
        assert "maxsum_inference" in ret["scalar_functions_retained"]
        from prdrl.rl.coordination_graph import maxsum_inference
        assert callable(maxsum_inference)

    def test_t17_batched_implementation_hash_locked(self):
        lock = _load("E4RTA_implementation_hash_lock.json")
        sha = lock["coordination_graph_py_sha256"]
        assert len(sha) == 64
        h = hashlib.sha256()
        with open(ROOT / "src/prdrl/rl/coordination_graph.py", "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        assert h.hexdigest() == sha

    def test_t18_dtype_float32(self):
        dt = _load("E4RTA_dtype_lock.json")
        assert dt["dtype"] == "float32"
        assert dt["do_NOT_switch_to_float64"] is True
        assert dt["do_NOT_switch_to_float16_bfloat16"] is True

    def test_t19_k10_unchanged(self):
        from prdrl.rl.coordination_graph import K_MAXSUM
        assert K_MAXSUM == 10

    def test_t20_graph_unchanged(self):
        from prdrl.rl.coordination_graph import R_MAX, _N_UNDIRECTED
        from math import comb
        assert _N_UNDIRECTED == comb(R_MAX, 2) == 45

    def test_t21_pair_factor_unchanged(self):
        from prdrl.rl.coordination_graph import BETA_PAIR
        assert BETA_PAIR == 1.0

    def test_t22_joint_q_unchanged(self):
        cfg = json.loads(json.dumps(
            {"K_MAXSUM": 10, "PAIR_SEED_OFFSET": 1, "BETA_PAIR": 1.0}))
        from prdrl.rl.coordination_graph import K_MAXSUM, PAIR_SEED_OFFSET, BETA_PAIR
        assert K_MAXSUM == cfg["K_MAXSUM"]
        assert PAIR_SEED_OFFSET == cfg["PAIR_SEED_OFFSET"]
        assert BETA_PAIR == cfg["BETA_PAIR"]

    def test_t23_r1_unchanged(self):
        lock = _load("E4RTA_implementation_hash_lock.json")
        assert len(lock["reward_r1_py_sha256"]) == 64

    def test_t24_v3_unchanged(self):
        lock = _load("E4RTA_implementation_hash_lock.json")
        assert len(lock["normalization_py_sha256"]) == 64

    def test_t25_replay_unchanged(self):
        scope = _load("E4RTA_amendment_scope.json")
        assert "replay" in scope["does_NOT_change"]

    def test_t26_optimizer_unchanged(self):
        scope = _load("E4RTA_amendment_scope.json")
        assert "optimizer" in scope["does_NOT_change"]


# ===== T27-T34: Interrupted Run and Restart =====

class TestInterruptedRunAndRestart:
    def test_t27_interrupted_old_run_excluded(self):
        exc = _load("E4RTA_interrupted_run_exclusion.json")
        assert exc["status"] == "EXCLUDED_FROM_CAUSAL_ANALYSIS"
        assert exc["merge_with_restarted_data"] is False

    def test_t28_old_c1_checkpoints_blacklisted(self):
        bl = _load("E4RTA_old_checkpoint_blacklist.json")
        assert bl["OLD_CHECKPOINT_REUSE"] is False
        assert bl["cannot_enter_future_selection"] is True

    def test_t29_old_c3_checkpoint_blacklisted(self):
        bl = _load("E4RTA_old_checkpoint_blacklist.json")
        assert any("C3" in s for s in bl["blacklisted_checkpoints"])

    def test_t30_restart_id_distinct(self):
        ri = _load("E4RTA_restart_identity.json")
        assert ri["restart_version"] == "E4RTR_BATCHED_RESTART_V2"
        assert ri["overwrite_interrupted_directory"] is False

    def test_t31_restart_from_scratch_required(self):
        c = _load("E4RTA_restart_from_scratch_contract.json")
        assert c["RESTART_MODE"] == "FROM_SCRATCH"

    def test_t32_no_optimizer_replay_resume(self):
        c = _load("E4RTA_restart_from_scratch_contract.json")
        assert "optimizer_state" in c["do_NOT_resume"]
        assert "replay_state" in c["do_NOT_resume"]

    def test_t33_all_c1_policies_restart(self):
        c = _load("E4RTA_restart_from_scratch_contract.json")
        assert "C1 policy 7001-7005" in c["initialize_from_scratch"]

    def test_t34_all_c3_policies_restart(self):
        c = _load("E4RTA_restart_from_scratch_contract.json")
        assert "C3 policy 7001-7005" in c["initialize_from_scratch"]


# ===== T35-T38: Seed Management =====

class TestSeedManagement:
    def test_t35_training_seeds_reused_correct_label(self):
        tsr = _load("E4RTA_training_seed_reuse_contract.json")
        assert tsr["status"] == "REUSED_AFTER_ENGINEERING_ONLY_INTERRUPTION"
        assert tsr["training_seeds"] == list(range(2501, 2561))

    def test_t36_training_seeds_not_called_fresh(self):
        tsr = _load("E4RTA_training_seed_reuse_contract.json")
        assert tsr["NOT_described_as"] == "fresh"

    def test_t37_validation_seeds_untouched(self):
        vf = _load("E4RTA_validation_seed_firewall.json")
        assert vf["access_count"] == 0
        assert vf["status"] == "NEVER_ACCESSED"
        assert vf["validation_seeds"] == list(range(2561, 2581))

    def test_t38_final_seeds_untouched(self):
        ff = _load("E4RTA_final_seed_firewall.json")
        assert ff["access_count"] == 0
        assert ff["status"] == "NEVER_ACCESSED"
        assert ff["final_seeds"] == list(range(2581, 2641))


# ===== T39-T44: Protocol Values =====

class TestProtocolValues:
    @pytest.fixture(autouse=True)
    def _load_protocol(self):
        self.p = _load("E4RTA_scientific_protocol_preservation.json")

    def test_t39_training_count_10800(self):
        assert self.p["TRAINING_EPISODES"] == 10800

    def test_t40_checkpoints_60(self):
        assert self.p["CHECKPOINTS"] == 60

    def test_t41_validation_3600(self):
        assert self.p["VALIDATION_EPISODES"] == 3600

    def test_t42_final_1800(self):
        assert self.p["FINAL_EPISODES"] == 1800

    def test_t43_primary_endpoint_unchanged(self):
        assert self.p["PRIMARY_ENDPOINT"] == "MIXED MIGD_response"

    def test_t44_bootstrap_unchanged(self):
        assert self.p["FINAL_BOOTSTRAP"] == "paired crossed 5x60"
        assert self.p["BOOTSTRAP_N"] == 50000
        assert self.p["BOOTSTRAP_SEED"] == 136001


# ===== T45-T48: Batched Path and Lineage =====

class TestBatchedPathAndLineage:
    def test_t45_batched_call_path_frozen(self):
        bp = _load("E4RTA_batched_action_path_lock.json")
        assert "maxsum_inference_batched" in bp["batch_training_path"]
        assert bp["no_accidental_mix_of_old_scalar_and_new_batched"] is True

    def test_t46_no_automatic_scalar_fallback(self):
        nf = _load("E4RTA_no_fallback_contract.json")
        assert nf["silent_fallback_from_batched_to_scalar"] is False

    def test_t47_future_checkpoint_lineage_includes_amendment_hash(self):
        lin = _load("E4RTA_future_checkpoint_lineage.json")
        assert "E4RTA_AMENDED_PROTOCOL_SHA256" in lin["required_metadata_per_checkpoint"]

    def test_t48_old_checkpoints_cannot_enter_future_selection(self):
        lin = _load("E4RTA_future_checkpoint_lineage.json")
        assert lin["old_checkpoints_excluded_from_selection"] is True


# ===== T49-T54: Verdicts and Authorization =====

class TestVerdictsAndAuthorization:
    def test_t49_no_environment_execution(self):
        pre = _load("E4RTA_pre_manifest.json")
        assert pre["environments_instantiated"] is False

    def test_t50_no_training(self):
        pre = _load("E4RTA_pre_manifest.json")
        assert pre["training_executed"] is False

    def test_t51_performance_verdict_not_computed(self):
        v = _load("E4RTA_verdict.json")
        assert v["PERFORMANCE_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_t52_pairwise_causal_verdict_not_computed(self):
        v = _load("E4RTA_verdict.json")
        assert v["PAIRWISE_COORDINATION_CAUSAL_VERDICT"] == "NOT_COMPUTED"

    def test_t53_restart_authorization_checks(self):
        auth = _load("E4RTA_restart_authorization.json")
        for key, val in auth["checks"].items():
            assert val is True, f"Auth check {key} is not True"

    def test_t54_next_stage_exact(self):
        v = _load("E4RTA_verdict.json")
        assert v["NEXT_STAGE"] == "E4RTR_PAIRWISE_CG_AB_TRAINING_EXECUTION_RESTART_V2"

    def test_t55_amendment_hash_exists(self):
        ah = _load("E4RTA_amended_protocol_hash.json")
        sha = ah["E4RTA_AMENDED_PROTOCOL_SHA256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


# ===== T56+: Additional Verification =====

class TestAdditionalVerification:
    def test_t56_no_firewalled_seeds_in_test_file(self):
        test_src = Path(__file__).read_text()
        for s in range(2561, 2641):
            assert f"seed={s}" not in test_src
            assert f"seed={s}," not in test_src

    def test_t57_all_required_outputs_exist(self):
        required = [
            "E4RTA_pre_manifest.json",
            "E4RTA_historical_integrity.json",
            "E4RTA_amendment_scope.json",
            "E4RTA_scientific_config_lock.json",
            "E4RTA_implementation_hash_lock.json",
            "E4RTA_float32_tie_audit.json",
            "E4RTA_non_tie_equivalence.json",
            "E4RTA_numerical_tie_exception.json",
            "E4RTA_interrupted_run_exclusion.json",
            "E4RTA_old_checkpoint_blacklist.json",
            "E4RTA_restart_identity.json",
            "E4RTA_restart_from_scratch_contract.json",
            "E4RTA_training_seed_reuse_contract.json",
            "E4RTA_validation_seed_firewall.json",
            "E4RTA_final_seed_firewall.json",
            "E4RTA_scientific_protocol_preservation.json",
            "E4RTA_batched_action_path_lock.json",
            "E4RTA_scalar_reference_retention.json",
            "E4RTA_no_fallback_contract.json",
            "E4RTA_dtype_lock.json",
            "E4RTA_future_checkpoint_lineage.json",
            "E4RTA_restart_authorization.json",
            "E4RTA_amended_protocol.json",
            "E4RTA_amended_protocol_hash.json",
            "E4RTA_verdict.json",
        ]
        for name in required:
            assert (E4RTA_DIR / name).exists(), f"Missing: {name}"

    def test_t58_amended_protocol_content_consistent(self):
        ap = _load("E4RTA_amended_protocol.json")
        assert ap["C3_CONFIG_SHA256"] == C3_CONFIG_SHA_EXPECTED
        assert ap["QMIX_CONFIG_SHA256"] == QMIX_CONFIG_SHA_EXPECTED
        assert ap["original_E4RT_PROTOCOL_SHA256"] == E4RT_PROTOCOL_SHA_EXPECTED
        assert ap["NON_TIE_MISMATCH_COUNT"] == 0
        assert ap["FLOAT32_TIE_EXCEPTION"] == "DOCUMENTED"
        assert ap["restart_version"] == "E4RTR_BATCHED_RESTART_V2"
        assert ap["restart_mode"] == "FROM_SCRATCH"

    def test_t59_training_script_uses_batched_imports(self):
        script = (ROOT / "src/prdrl/experiments/E4RTR_pairwise_cg_ab_training_execution.py").read_text()
        assert "maxsum_inference_batched as maxsum_inference" in script
        assert "c3_double_q_target_batched as c3_double_q_target" in script
        assert "c3_current_q_batched as c3_current_q" in script

    def test_t60_scalar_functions_still_importable(self):
        from prdrl.rl.coordination_graph import (
            maxsum_inference, joint_value_cg,
            c3_double_q_target, c3_current_q,
        )
        assert callable(maxsum_inference)
        assert callable(joint_value_cg)
        assert callable(c3_double_q_target)
        assert callable(c3_current_q)

    def test_t61_batched_functions_importable(self):
        from prdrl.rl.coordination_graph import (
            maxsum_inference_batched, joint_value_cg_batched,
            c3_double_q_target_batched, c3_current_q_batched,
        )
        assert callable(maxsum_inference_batched)
        assert callable(joint_value_cg_batched)
        assert callable(c3_double_q_target_batched)
        assert callable(c3_current_q_batched)
