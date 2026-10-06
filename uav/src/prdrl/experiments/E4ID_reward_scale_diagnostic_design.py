"""
E4ID — Reward Scale Diagnostic Design (PR-DRL-DMOEA, Branch B).

DESIGN / FREEZE ONLY. Does NOT execute diagnostic environments, does NOT train, does NOT
modify any network, and does NOT choose alpha_local. It freezes the exact E4IDD diagnostic
experiment that will (later) MEASURE the empirical scale of the R1 regional-credit signal
g_r and DETERMINISTICALLY choose/freeze alpha_local for the future V3+R0 vs V3+R1 study.

Core scientific discipline (the reason this stage exists): R1 has a structural point mass
at zero (E4I nonzero fraction ~0.4954), so unconditional median|g_r| can be 0 even when the
nonzero signal is meaningful. E4ID therefore separates ACTIVITY (how often g_r != 0) from
ACTIVE-SIGNAL MAGNITUDE (how big g_r is when nonzero), and requires global vs local scale to
be compared on the SAME LOCAL_ACTIVE transition subset.

E4I is authoritative. Historical results remain immutable: E4F=FAILED, E4HF normalization
verdict=FAILED, E4I=COMPLETE, R1_REWARD_READY=NO. E4ID does NOT make R1 ready.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "src")

from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1

RESULTS_DIR = Path("results/E4ID_reward_scale_diagnostic_design")
E4I_DIR = Path("results/E4I_reward_credit_revision_design")
E4HE_DIR = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

# ---- diagnostic (scale) seeds — POST-HOC, never train/val/confirmatory ----
DIAG_SEEDS = list(range(1701, 1721))          # 20
DIAG_SEED_LABEL = "POST_HOC_REWARD_SCALE_DIAGNOSTIC_ONLY"
BEHAVIOR_POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]   # frozen V3_TREATMENT policies
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DIAG_EPISODES = len(BEHAVIOR_POLICY_SEEDS) * len(DIAG_SEEDS) * len(SCENARIOS)   # 300

# ---- frozen scale-analysis constants ----
EPS_ACTIVE = EPS_R1                            # 1e-12 (reuse E4I tolerance)
ACTIVITY_MIN_FRACTION = 0.20
GLOBAL_OVERWHELMS_THRESHOLD = 10.0            # reused E4GD criterion
INITIAL_ALPHA_LOCAL = 0.1
ALPHA_LOCAL_MIN = 0.1                          # E4ID-frozen operative bounds (narrow E4I's preliminary [0.01,10])
ALPHA_LOCAL_MAX = 1.0
SCENARIO_RATIO_MAX = 10.0
Q95_RATIO_MAX = 5.0
Q99_RATIO_MAX = 10.0

# ---- future reward experiment (frozen in E4I; UNTOUCHED here) ----
REWARD_TRAIN_SEEDS = list(range(1501, 1561))
REWARD_VAL_SEEDS = list(range(1561, 1581))
REWARD_FINAL_SEEDS = list(range(1581, 1641))
REWARD_POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
REWARD_BOOTSTRAP_MASTER_SEED = 66001

# ---- all historically consumed/reserved environment seeds (for disjointness) ----
CONSUMED_SEED_SETS = {
    "E3_701_780": range(701, 781), "E3R_801_882": range(801, 883), "H1_901_940": range(901, 941),
    "E4_train_1001_1060": range(1001, 1061), "E4_val_1061_1080": range(1061, 1081),
    "E4_final_1081_1140": range(1081, 1141), "E4GD_diag_1201_1220": range(1201, 1221),
    "E4H_train_1301_1360": range(1301, 1361), "E4H_val_1361_1380": range(1361, 1381),
    "E4H_final_1381_1440": range(1381, 1441),
    "REWARD_train_1501_1560": range(1501, 1561), "REWARD_val_1561_1580": range(1561, 1581),
    "REWARD_final_1581_1640": range(1581, 1641),
}

_access = {"REWARD_TRAIN": 0, "REWARD_VALIDATION": 0, "REWARD_FINAL": 0, "DIAG_ENV": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4ID — Reward Scale Diagnostic Design (Branch B, DESIGN/FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Phase E4ID-0.5 historical integrity ----------
    e4i = _load(E4I_DIR / "E4I_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    r1_formula = _load(E4I_DIR / "E4I_reward_r1_formula.json")
    metric = _load(E4I_DIR / "E4I_local_quality_metric_contract.json")
    frz = _load(E4HE_DIR / "E4HE_checkpoint_freeze.json")["checkpoints"]
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    v3_ck = {int(v["policy_seed"]): v for v in frz.values() if v["arm"] == "V3_TREATMENT"}
    ck_rows = {p: {"policy_seed": p, "arm": "V3_TREATMENT",
                   "normalization_version": v3_ck[p]["normalization_version"],
                   "checkpoint_path": v3_ck[p]["checkpoint_path"], "frozen_sha256": v3_ck[p]["sha256"],
                   "recomputed_sha256": _hash_file(v3_ck[p]["checkpoint_path"]),
                   "match": _hash_file(v3_ck[p]["checkpoint_path"]) == v3_ck[p]["sha256"]}
               for p in BEHAVIOR_POLICY_SEEDS}
    ck_all_match = all(r["match"] for r in ck_rows.values())
    v3_all = all(ck_rows[p]["normalization_version"] == NORM_V3 for p in BEHAVIOR_POLICY_SEEDS)

    hist = {
        "E4I_COMPLETE": e4i["E4I_COMPLETE"], "E4I_R1_REWARD_READY": e4i["R1_REWARD_READY"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "SELECTED_BRANCH": "B_REWARD_CREDIT",
        "R1_REWARD": REWARD_R1_NAME, "FIXED_NORMALIZATION": NORM_V3, "v3_constants_sha256": v3_sha,
        "R1_formula_frozen": r1_formula["g_r_formula"], "R1_eps": r1_formula["eps"],
        "behavior_checkpoints": ck_rows, "checkpoint_all_match": ck_all_match,
        "v3_normalization_all": v3_all, "no_policy_reselection": True, "no_retrain": True,
        "historical_artifact_hashes": {
            "E4I_verdict.json": _hash_file(E4I_DIR / "E4I_verdict.json"),
            "E4I_reward_r1_formula.json": _hash_file(E4I_DIR / "E4I_reward_r1_formula.json"),
            "E4HF_verdict.json": _hash_file(E4HF_DIR / "E4HF_verdict.json"),
            "E4F_verdict.json": _hash_file(E4F_DIR / "E4F_verdict.json"),
            "reward.py": _hash_file("src/prdrl/rl/reward.py"),
            "reward_r1.py": _hash_file("src/prdrl/rl/reward_r1.py"),
            "quality_loss.py": _hash_file("src/prdrl/detectors/quality_loss.py"),
        },
    }
    _jdump(hist, RESULTS_DIR / "E4ID_historical_integrity.json")

    # ---------- seed disjointness ----------
    diag = set(DIAG_SEEDS)
    consumed = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    diag_disjoint = len(diag & consumed) == 0
    future_reward = set(REWARD_TRAIN_SEEDS) | set(REWARD_VAL_SEEDS) | set(REWARD_FINAL_SEEDS)
    diag_vs_future = len(diag & future_reward) == 0

    integrity_ok = (e4i["E4I_COMPLETE"] == "YES" and e4i["R1_REWARD_READY"] == "NO"
                    and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
                    and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
                    and ck_all_match and v3_all and diag_disjoint and diag_vs_future
                    and len(DIAG_SEEDS) == 20 and DIAG_EPISODES == 300)

    _jdump({"stage": "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN", "design_only": True,
            "no_diagnostic_execution": True, "no_training": True, "no_network_modification": True,
            "no_alpha_selection_here": True, "e4i_authoritative": True,
            "integrity_ok": integrity_ok}, RESULTS_DIR / "E4ID_pre_manifest.json")
    if not integrity_ok:
        print("[E4ID] integrity/disjointness FAIL — STOP")
        _jdump({"E4ID_COMPLETE": "NO", "NEXT_STAGE": "E4ID_PROTOCOL_REVIEW"},
               RESULTS_DIR / "E4ID_verdict.json")
        return {"E4ID_COMPLETE": "NO"}
    print(f"[E4ID] integrity PASS. Diagnostic seeds {DIAG_SEEDS[0]}..{DIAG_SEEDS[-1]} reserved; "
          f"5 V3 policies hash-match; future reward seeds untouched.")

    # ---------- freeze contracts ----------
    C: Dict[str, dict] = {}

    C["E4ID_behavior_policy_contract.json"] = {
        "behavior_policies": [f"V3_{p}" for p in BEHAVIOR_POLICY_SEEDS],
        "policy_seeds": BEHAVIOR_POLICY_SEEDS, "arm": "V3_TREATMENT",
        "source": "E4HTR/E4HF selected V3 checkpoints (E4HE_checkpoint_freeze.json)",
        "checkpoints": ck_rows, "not_retrained": True, "not_fine_tuned": True,
        "not_reselected": True, "role": "fixed behavior policy for reward-scale logging only",
        "why": "representative trajectories under corrected V3 state, behavior fixed, so R1 can "
               "be computed side-by-side without a training feedback loop."}

    C["E4ID_diagnostic_seed_contract.json"] = {
        "diagnostic_seeds": [1701, 1720], "n": 20, "label": DIAG_SEED_LABEL,
        "may_be_used_only_for": "E4IDD reward-scale diagnostic",
        "never_training": True, "never_validation": True, "never_confirmatory_final": True,
        "disjoint_from_all_history": diag_disjoint, "disjoint_from_future_reward_1501_1640": diag_vs_future,
        "checked_against": list(CONSUMED_SEED_SETS.keys()),
        "no_old_final_reuse": {"forbid_1081_1140": True, "forbid_1381_1440": True}}

    C["E4ID_diagnostic_matrix_contract.json"] = {
        "behavior_policies": 5, "diagnostic_environment_seeds": 20, "scenarios": SCENARIOS,
        "total_diagnostic_episodes": DIAG_EPISODES, "changes_per_episode": 10,
        "no_additional_policy_arms": True, "no_r0_vs_r1_behavioral_comparison": True,
        "behavior_policy_fixed": True,
        "R0_and_R1_computed_on_same_trajectory": True,
        "per_transition": ["one_actual_action", "one_actual_response", "one_pre_population",
                           "one_post_population"]}

    C["E4ID_execution_mode_contract.json"] = {
        "network": "selected V3 online network only", "epsilon": 0.0, "eval": True,
        "gradients": "disabled", "optimizer": "disabled", "replay": "disabled",
        "target_updates": "disabled", "learning": "none",
        "checkpoint_hashes_unchanged_pre_post": True,
        "normalization": NORM_V3, "v3_refit": False, "online_scaling": False,
        "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
                      "predict_fallback": "LOCAL", "sigma_D": 0.15, "budget": "10 PER_ACTIVE_REGION",
                      "survival": "unrestricted NSGA-II", "no_HARD": True, "no_MEMORY": True},
        "R1_influences_actions": False, "R1_is_observational_logging_only": True}

    C["E4ID_reward_logging_contract.json"] = {
        "identity": ["policy_seed", "scenario", "diagnostic_environment_seed", "episode_index",
                     "change_index", "region_id"],
        "action": ["requested_action", "realized_operator", "fallback"],
        "quality": ["L_pre", "L_post", "active_status", "valid_or_degenerate"],
        "R0": ["raw_global_reward", "weighted_global_reward", "raw_local_reward_R0",
               "weighted_local_reward_R0", "raw_fe_cost", "weighted_fe_penalty", "total_reward_R0"],
        "R1_candidate": ["g_r", "local_active", "weighted_local_R1_alpha_0p1", "total_reward_R1_alpha_0p1"],
        "identical_for_R0_and_R1": ["global reward", "weighted global reward", "FE cost", "FE penalty"],
        "only_local_term_differs": True,
        "weighted_local_R1_alpha_0p1": "0.1 * g_r",
        "total_reward_R1_alpha_0p1": "weighted_global_reward + 0.1*g_r - weighted_fe_penalty",
        "L_pre_L_post_source": metric["source"], "same_region_id_pre_post": True}

    C["E4ID_active_local_definition.json"] = {
        "LOCAL_ACTIVE": "|g_r| > EPS_ACTIVE", "EPS_ACTIVE": EPS_ACTIVE,
        "reused_from_E4I_tolerance": True,
        "LOCAL_ACTIVE_FRACTION": "count(LOCAL_ACTIVE)/count(valid regional transitions)",
        "MEDIAN_ABS_G_ACTIVE": "median(|g_r| over LOCAL_ACTIVE only)",
        "structural_zero_note": "g_r has a point mass at 0 (~0.4954 nonzero); do NOT use "
                                "unconditional median|g_r| as the scale estimator."}

    C["E4ID_activity_gate.json"] = {
        "rule": "R1_ACTIVITY_GATE = PASS iff LOCAL_ACTIVE_FRACTION >= 0.20 (pooled)",
        "ACTIVITY_MIN_FRACTION": ACTIVITY_MIN_FRACTION,
        "report_scenario_specific": True, "e4i_historical_proxy_nonzero": 0.4954,
        "do_not_force_proxy": True}

    C["E4ID_scale_metric_contract.json"] = {
        "MATCHING_RULE": "global and local magnitudes computed on the SAME LOCAL_ACTIVE subset",
        "MEDIAN_ABS_GLOBAL_ACTIVE": "median(|weighted_global_reward| over LOCAL_ACTIVE)",
        "MEDIAN_ABS_LOCAL_ACTIVE(alpha)": "median(|alpha*g_r| over LOCAL_ACTIVE)",
        "GLOBAL_LOCAL_ACTIVE_RATIO(alpha)": "MEDIAN_ABS_GLOBAL_ACTIVE / (MEDIAN_ABS_LOCAL_ACTIVE(alpha)+1e-12)",
        "forbidden": "compare global median (all transitions) vs local median (nonzero only)",
        "transparency_only": ["median|g_r| over all", "median|weighted local| over all", "zero_fraction"],
        "transparency_not_used_for_alpha": True}

    C["E4ID_initial_alpha_contract.json"] = {
        "INITIAL_ALPHA_LOCAL": INITIAL_ALPHA_LOCAL,
        "G": "median(|weighted_global| on LOCAL_ACTIVE)", "M": "median(|g_r| on LOCAL_ACTIVE)",
        "L0": "0.1 * M", "RATIO_0": "G / (L0 + 1e-12)",
        "GLOBAL_OVERWHELMS_THRESHOLD": GLOBAL_OVERWHELMS_THRESHOLD,
        "classification": "RATIO_0>=10 -> GLOBAL_OVERWHELMS_LOCAL else LOCAL_MATERIAL",
        "if_material": "FINAL_R1_ALPHA_LOCAL = 0.1 (no adjustment, no further optimization)"}

    C["E4ID_alpha_selection_rule.json"] = {
        "rule_name": "KEEP_0.1_IF_MATERIAL_ELSE_ACTIVE_MEDIAN_SCALE_MATCH",
        "CASE_A": "RATIO_0 < 10 -> alpha_final = 0.1",
        "CASE_B": "RATIO_0 >= 10 -> alpha_scale = G/(M+1e-12); require 0.1<=alpha_scale<=1.0 AND "
                  "all 3 scenario ratios < 10 AND Q95_RATIO<=5 AND Q99_RATIO<=10 -> alpha_final=alpha_scale; "
                  "else NO alpha frozen and R1_REWARD_READY=NO",
        "single_deterministic_formula": "alpha_scale = G/(M+1e-12)",
        "no_performance_metric_in_calculation": True,
        "forbidden_sweep": ["0.2", "0.3", "0.5", "1.0 chosen by MIGD/reward/action/validation"],
        "no_rounding_for_scientific_value": True, "store_full_double_precision": True}

    C["E4ID_alpha_bounds.json"] = {
        "ALPHA_LOCAL_MIN": ALPHA_LOCAL_MIN, "ALPHA_LOCAL_MAX": ALPHA_LOCAL_MAX,
        "frozen_before_diagnostic_values_observed": True,
        "rationale": ["0.1 is the original coefficient; branch selected because local credit too "
                      "weak, so correction must not reduce below original",
                      "g_r bounded in [-1,1]", "alpha<=1 bounds |weighted local| by 1 per transition",
                      "engineering/numerical bound, NOT performance-tuned"],
        "no_silent_clipping": True,
        "ALPHA_BOUND_GATE": "FAIL (and R1_REWARD_READY=NO) if alpha_scale<0.1 or alpha_scale>1.0",
        "note": "operative bounds for the reward-scale decision; narrow E4I reward_r1.py preliminary "
                "[0.01,10.0]; E4I is not rewritten."}

    C["E4ID_scenario_scale_gate.json"] = {
        "rule": "for each scenario s: RATIO_CANDIDATE[s] = median|weighted_global| / "
                "(median|alpha_candidate*g_r| + 1e-12) < 10 on LOCAL_ACTIVE of that scenario",
        "SCENARIO_RATIO_MAX": SCENARIO_RATIO_MAX, "scenarios": SCENARIOS,
        "purpose": "local term not functionally negligible in any single scenario"}

    C["E4ID_tail_safety_gate.json"] = {
        "Q95_RATIO": "q95|alpha*g_r| / (q95|weighted_global| + 1e-12)",
        "Q99_RATIO": "q99|alpha*g_r| / (q99|weighted_global| + 1e-12)",
        "Q95_RATIO_MAX": Q95_RATIO_MAX, "Q99_RATIO_MAX": Q99_RATIO_MAX,
        "gate": "Q95_RATIO<=5 AND Q99_RATIO<=10",
        "purpose": "prevent median scale-matching from creating extreme local-reward tails",
        "reward_scale_safety_not_performance": True}

    C["E4ID_total_reward_safety_contract.json"] = {
        "R1_total": "weighted_global + alpha*g_r - weighted_FE_penalty",
        "report": ["min", "max", "mean", "std", "q01", "q05", "median", "q95", "q99"],
        "require": "all finite", "no_performance_preference": True}

    C["E4ID_sign_semantics_contract.json"] = {
        "L_post<L_pre": "g_r>0", "L_post>L_pre": "g_r<0", "exact_tie": "g_r==0 (within tolerance)",
        "require_zero_sign_violations": True, "tolerance": EPS_ACTIVE}

    C["E4ID_fe_contract.json"] = {
        "ADDITIONAL_OBJECTIVE_EVALUATIONS_FOR_R1": 0,
        "L_pre_L_post_g_r_from_already_evaluated_populations": True,
        "per_episode_extra_FE_required": 0}

    C["E4ID_matrix_completeness_contract.json"] = {
        "expected_diagnostic_episodes": DIAG_EPISODES,
        "DIAGNOSTIC_MATRIX_COMPLETE_required": True,
        "on_missing": "do not analyze reduced matrix; do not replace seed; do not replace policy; "
                      "scale decision NOT computed"}

    C["E4ID_execution_order.json"] = {
        "order": ["scenario in [PERSISTENT, STATIONARY, MIXED]",
                  "diagnostic_environment_seed 1701..1720 ascending",
                  "policy_seed 3001..3005 ascending"],
        "no_outcome_dependent_early_stopping": True}

    C["E4ID_future_seed_firewall.json"] = {
        "future_reward_train": [1501, 1560], "future_reward_validation": [1561, 1580],
        "future_reward_final": [1581, 1640], "future_reward_policy": REWARD_POLICY_SEEDS,
        "do_not_instantiate_in_E4ID_or_E4IDD": True,
        "REWARD_TRAIN_ACCESS_COUNT": _access["REWARD_TRAIN"],
        "REWARD_VALIDATION_ACCESS_COUNT": _access["REWARD_VALIDATION"],
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _access["REWARD_FINAL"],
        "diagnostic_findings_do_not_change_future_seeds": True}

    C["E4ID_e4idd_output_schema.json"] = {
        "results_dir": "results/E4IDD_reward_scale_diagnostic_execution/",
        "artifacts": ["E4IDD_pre_execution_manifest.json", "E4IDD_checkpoint_integrity.json",
                      "E4IDD_diagnostic_seed_usage.json", "E4IDD_episode_results.csv",
                      "E4IDD_regional_reward_records.csv", "E4IDD_r1_activity.json",
                      "E4IDD_r1_active_magnitude.csv", "E4IDD_r1_scale_by_scenario.csv",
                      "E4IDD_initial_alpha_scale.json", "E4IDD_alpha_candidate.json",
                      "E4IDD_tail_safety.json", "E4IDD_reward_distribution.json",
                      "E4IDD_sign_semantics.json", "E4IDD_fe_accounting.json",
                      "E4IDD_alpha_freeze.json", "E4IDD_reward_r1_config.json",
                      "E4IDD_checkpoint_post_integrity.json", "E4IDD_future_seed_firewall.json",
                      "E4IDD_verdict.json", "E4IDD_report.md"],
        "reward_r1_config_must_contain": ["reward_version", "R1_formula_hash", "alpha_local",
                                          "alpha_global", "beta", "diagnostic_protocol_hash",
                                          "diagnostic_data_hash", "REWARD_R1_CONFIG_SHA256"],
        "future_R1_checkpoints_carry": "REWARD_R1_CONFIG_SHA256"}

    C["E4ID_readiness_logic.json"] = {
        "R1_REWARD_READY_requires_all": [
            "diagnostic matrix complete (300/300)", "R1 formula integrity PASS",
            "local metric integrity PASS", "extra FE = 0", "activity gate PASS", "M > 0",
            "alpha decision completed", "alpha bound gate PASS", "all scenario scale ratios < 10",
            "tail safety gate PASS", "total rewards finite", "sign semantics PASS", "no training",
            "future reward scientific seeds untouched", "full regression PASS"],
        "e4id_does_not_make_r1_ready": True}

    C["E4ID_future_next_stage_logic.json"] = {
        "if_R1_REWARD_READY_YES": "E4II_REWARD_IMPLEMENTATION_VALIDATION (do NOT immediately train)",
        "if_NO_due_to_alpha_scale_tail": "E4I_REWARD_SCALE_RULE_REVIEW",
        "if_metric_sign_FE_fail": "E4I_REWARD_METRIC_REVIEW"}

    C["E4ID_single_factor_reaffirmation.json"] = {
        "CONTROL": {"normalization": NORM_V3, "reward": REWARD_R0_NAME, "alpha_local_R0": 0.1},
        "TREATMENT": {"normalization": NORM_V3, "reward": REWARD_R1_NAME,
                      "alpha_local_R1": "alpha_final (frozen in E4IDD)"},
        "R0_alpha_local_unchanged": True,
        "single_experimental_factor": "reward specification (including its frozen coefficient)",
        "network": "9-128-128-4 Shared Double DQN", "joint_representation": "IMPLICIT_STATE_ONLY",
        "no_CTDE": True, "no_QMIX": True, "no_VDN": True, "no_autoregressive": True,
        "future_reward_seeds_unchanged": {"train": [1501, 1560], "val": [1561, 1580],
                                          "final": [1581, 1640], "policy": REWARD_POLICY_SEEDS}}

    for name, obj in C.items():
        _jdump(obj, RESULTS_DIR / name)

    protocol_hash_material = {k: C[k] for k in sorted(C)}
    diagnostic_protocol_hash = hashlib.sha256(
        json.dumps(protocol_hash_material, sort_keys=True).encode()).hexdigest()

    e4id_complete = integrity_ok
    _jdump({"stage": "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN", "E4ID_COMPLETE": "YES" if e4id_complete else "NO",
            "selected_branch": "B_REWARD_CREDIT", "fixed_normalization": NORM_V3,
            "behavior_policies": [f"V3_{p}" for p in BEHAVIOR_POLICY_SEEDS],
            "diagnostic_seeds": [1701, 1720], "diagnostic_episodes_planned": DIAG_EPISODES,
            "alpha_selection_rule": "KEEP_0.1_IF_MATERIAL_ELSE_ACTIVE_MEDIAN_SCALE_MATCH",
            "alpha_bounds": [ALPHA_LOCAL_MIN, ALPHA_LOCAL_MAX],
            "diagnostic_protocol_hash": diagnostic_protocol_hash,
            "components": list(C.keys()),
            "next_stage": "E4IDD_REWARD_SCALE_DIAGNOSTIC_EXECUTION"},
           RESULTS_DIR / "E4ID_protocol_frozen.json")

    verdict = build_verdict(e4id_complete, hist, diag_disjoint, diag_vs_future)
    _jdump(verdict, RESULTS_DIR / "E4ID_verdict.json")
    write_report(verdict, hist)
    emit_terminal(verdict)
    print(f"[E4ID] wall {time.time()-t0:.2f}s")
    return verdict


def build_verdict(complete, hist, diag_disjoint, diag_vs_future):
    return {
        "stage": "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN",
        "E4ID_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "SELECTED_BRANCH": "B_REWARD_CREDIT",
        "R1_REWARD": REWARD_R1_NAME,
        "FIXED_NORMALIZATION": NORM_V3,
        "BEHAVIOR_POLICIES": "V3_3001..V3_3005",
        "DIAGNOSTIC_SEEDS": "1701..1720",
        "DIAGNOSTIC_SEED_LABEL": DIAG_SEED_LABEL,
        "DIAGNOSTIC_EPISODES_PLANNED": DIAG_EPISODES,
        "LOCAL_ACTIVE_THRESHOLD": EPS_ACTIVE,
        "ACTIVITY_GATE": "NONZERO_FRACTION_GTE_0.20",
        "MAGNITUDE_ESTIMATOR": "MEDIAN_ABS_CONDITIONAL_ON_LOCAL_ACTIVE",
        "GLOBAL_LOCAL_SUBSET_MATCHED": "YES",
        "INITIAL_ALPHA_LOCAL": INITIAL_ALPHA_LOCAL,
        "INITIAL_GLOBAL_OVERWHELMS_THRESHOLD": GLOBAL_OVERWHELMS_THRESHOLD,
        "ALPHA_SELECTION_RULE": "KEEP_0.1_IF_MATERIAL_ELSE_ACTIVE_MEDIAN_SCALE_MATCH",
        "ALPHA_LOCAL_MIN": ALPHA_LOCAL_MIN,
        "ALPHA_LOCAL_MAX": ALPHA_LOCAL_MAX,
        "SCENARIO_SCALE_GATE": "EACH_GLOBAL_LOCAL_RATIO_LT_10",
        "TAIL_Q95_GATE": "LOCAL_TO_GLOBAL_LTE_5",
        "TAIL_Q99_GATE": "LOCAL_TO_GLOBAL_LTE_10",
        "ALPHA_FINAL": "NOT_YET_COMPUTED",
        "R1_REWARD_READY": "NO",
        "SEEDS_DISJOINT": diag_disjoint and diag_vs_future,
        "CHECKPOINT_INTEGRITY": "PASS" if hist["checkpoint_all_match"] else "FAIL",
        "REWARD_TRAIN_ACCESS_COUNT": _access["REWARD_TRAIN"],
        "REWARD_VALIDATION_ACCESS_COUNT": _access["REWARD_VALIDATION"],
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _access["REWARD_FINAL"],
        "NEW_TRAINING": "NO",
        "DIAGNOSTIC_EXECUTION": "NOT_RUN",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4IDD_REWARD_SCALE_DIAGNOSTIC_EXECUTION" if complete else "E4ID_PROTOCOL_REVIEW"),
    }


def write_report(v, hist):
    L = ["# E4ID — Reward Scale Diagnostic Design (Branch B)", "",
         f"**E4ID_COMPLETE = {v['E4ID_COMPLETE']} → NEXT_STAGE = {v['NEXT_STAGE']}**", "",
         "Design/freeze only. No diagnostic environment execution, no training, no network change, "
         "no alpha selection. Future reward seeds 1501..1640 untouched; diagnostic seeds 1701..1720 "
         "reserved POST_HOC. E4I is authoritative; historical verdicts immutable.", "",
         "## A. Historical result preservation",
         f"- E4F = FAILED; E4HF normalization causal = FAILED; E4I COMPLETE with R1_REWARD_READY = NO; "
         f"5 V3 behavior checkpoints hash-match (integrity {v['CHECKPOINT_INTEGRITY']}); V3 constants unchanged.", "",
         "## B. Why E4ID is necessary",
         "- E4I froze R1 but its SCALE gate was NOT_AUDITABLE offline (only the local-reward difference "
         "was ever persisted, not L_pre and L_post separately). E4ID freezes a small post-hoc diagnostic "
         "that measures g_r's magnitude directly, so alpha_local can be chosen from reward-scale only.", "",
         "## C. Structural-zero issue in local reward scale",
         "- g_r has a point mass at 0 (~49.5% nonzero). Unconditional median|g_r| can be 0 even when the "
         "nonzero signal is meaningful, so it must NOT be the scale estimator.", "",
         "## D. Active-local transition definition",
         f"- LOCAL_ACTIVE iff |g_r| > {EPS_ACTIVE}. LOCAL_ACTIVE_FRACTION over valid transitions; "
         "MEDIAN_ABS_G_ACTIVE over LOCAL_ACTIVE only.", "",
         "## E. Frozen behavior policies",
         "- The five E4HTR/E4HF selected V3 policies (seeds 3001..3005); not retrained/fine-tuned/reselected; "
         "used only as fixed behavior for reward logging.", "",
         "## F. Fresh diagnostic seeds",
         f"- 1701..1720 (n=20), label {DIAG_SEED_LABEL}; disjoint from all history and from future reward "
         f"seeds 1501..1640 ({v['SEEDS_DISJOINT']}).", "",
         "## G. Diagnostic matrix",
         f"- 5 policies × 20 seeds × 3 scenarios = {v['DIAGNOSTIC_EPISODES_PLANNED']} episodes, 10 changes each; "
         "one fixed behavior policy; R0 and R1 computed on the SAME trajectory.", "",
         "## H. R0/R1 side-by-side logging",
         "- Global reward, weighted global, FE cost, FE penalty are identical for R0 and R1; only the local "
         "term differs. R1 logged as g_r, local_active, 0.1*g_r, and total at alpha=0.1.", "",
         "## I. Activity metric",
         "- ACTIVITY_GATE PASS iff LOCAL_ACTIVE_FRACTION ≥ 0.20 (pooled), scenario values also reported.", "",
         "## J. Conditional active magnitude metric",
         "- Global and local magnitudes are compared on the SAME LOCAL_ACTIVE subset; comparing all-transition "
         "global vs nonzero-only local is forbidden.", "",
         "## K. Initial alpha=0.1 test",
         "- G = median|weighted_global| on active, M = median|g_r| on active, RATIO_0 = G/(0.1*M+1e-12); "
         "RATIO_0 ≥ 10 ⇒ GLOBAL_OVERWHELMS_LOCAL, else LOCAL_MATERIAL (keep 0.1).", "",
         "## L. Deterministic scale-matching rule",
         "- Only if RATIO_0 ≥ 10: alpha_scale = G/(M+1e-12) (single deterministic formula, no sweep, no "
         "performance metric).", "",
         "## M. Alpha engineering bounds",
         f"- ALPHA_LOCAL_MIN = {ALPHA_LOCAL_MIN}, ALPHA_LOCAL_MAX = {ALPHA_LOCAL_MAX}, frozen before "
         "diagnostic values; no silent clipping (out-of-bounds ⇒ ALPHA_BOUND_GATE FAIL ⇒ R1 not ready).", "",
         "## N. Scenario robustness gate",
         "- With the chosen alpha, each scenario's global/local active ratio must be < 10.", "",
         "## O. Tail safety gate",
         "- Q95_RATIO ≤ 5 and Q99_RATIO ≤ 10 (local/global tails) so median matching does not create extreme "
         "local-reward tails.", "",
         "## P. FE/sign semantics",
         "- EXTRA FE for R1 = 0 (already-evaluated populations). Sign audit: L_post<L_pre⇒g_r>0, "
         "L_post>L_pre⇒g_r<0, tie⇒0; zero sign violations required.", "",
         "## Q. R1 readiness logic",
         "- R1_REWARD_READY becomes YES in E4IDD only if all 15 conditions pass; E4ID keeps it NO.", "",
         "## R. Future scientific-seed firewall",
         "- REWARD_TRAIN/VALIDATION/FINAL_ACCESS_COUNT = 0; future reward seeds untouched by E4ID/E4IDD; "
         "diagnostic findings only set reward scale/alpha, never architecture/checkpoint/verdict.", "",
         f"## S. Next execution stage: {v['NEXT_STAGE']}",
         "- E4IDD runs the 300-episode diagnostic, measures g_r scale, applies the frozen deterministic rule "
         "and gates, and freezes one admissible alpha_local (or returns R1_REWARD_READY = NO)."]
    (RESULTS_DIR / "E4ID_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    for k in ["E4ID_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "SELECTED_BRANCH", "R1_REWARD", "FIXED_NORMALIZATION", "BEHAVIOR_POLICIES",
              "DIAGNOSTIC_SEEDS", "DIAGNOSTIC_SEED_LABEL", "DIAGNOSTIC_EPISODES_PLANNED",
              "LOCAL_ACTIVE_THRESHOLD", "ACTIVITY_GATE", "MAGNITUDE_ESTIMATOR",
              "GLOBAL_LOCAL_SUBSET_MATCHED", "INITIAL_ALPHA_LOCAL", "INITIAL_GLOBAL_OVERWHELMS_THRESHOLD",
              "ALPHA_SELECTION_RULE", "ALPHA_LOCAL_MIN", "ALPHA_LOCAL_MAX", "SCENARIO_SCALE_GATE",
              "TAIL_Q95_GATE", "TAIL_Q99_GATE", "ALPHA_FINAL", "R1_REWARD_READY",
              "REWARD_TRAIN_ACCESS_COUNT", "REWARD_VALIDATION_ACCESS_COUNT", "REWARD_FINAL_EVAL_ACCESS_COUNT",
              "NEW_TRAINING", "DIAGNOSTIC_EXECUTION", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<38} = {v[k]}")


if __name__ == "__main__":
    main()
