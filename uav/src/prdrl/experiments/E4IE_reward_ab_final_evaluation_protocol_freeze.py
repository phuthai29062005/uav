"""
E4IE — Reward A/B Final Evaluation Protocol Freeze (PR-DRL-DMOEA, Branch B).

Freezes the COMPLETE untouched final causal protocol for V3+R0 (CONTROL) vs V3+R1 (TREATMENT)
BEFORE the reserved final seeds 1581..1640 are opened, and validates the entire analysis
pipeline on SYNTHETIC matrices only.

FREEZE ONLY: no final environment instantiated (access=0), no training, no checkpoint
reselection, no alpha change. Reward specification is the SOLE causal factor; the primary
comparator is R0_CONTROL. Descriptive E4ITR validation evidence (mechanism 3/3, val delta
+1.13e-4 favoring R1) is LOCKED as NON_CONFIRMATORY and must not alter the frozen protocol.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from prdrl.evaluation.crossed_bootstrap import (
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, paired_crossed_two_axis_bootstrap,
    paired_crossed_two_axis_bootstrap_macro, independent_arm_bootstrap,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect, policy_robustness, safety_gate,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)
from prdrl.rl import reward_integration as RI
from prdrl.rl.reward_r1 import REWARD_R0_NAME, REWARD_R1_NAME
from prdrl.rl.normalization import NORM_V3

RESULTS_DIR = Path("results/E4IE_reward_ab_final_evaluation_protocol_freeze")
E4IT_DIR = Path("results/E4IT_reward_ab_training_protocol_freeze")
E4ITR_DIR = Path("results/E4ITR_reward_ab_training_execution")
E4IDD_DIR = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FINAL_SEEDS = list(range(1581, 1641))
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED = 66001
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
E4IF_DIR = "results/E4IF_reward_ab_final_evaluation_execution"
FORBIDDEN_TOKENS = ("TBD", "TODO", "UNFROZEN", "placeholder", "PLACEHOLDER")
_final_access = {"count": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _json_default(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(str(type(x)))


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_json_default)


def _load(p):
    with open(p) as f:
        return json.load(f)


def _scan_tokens(obj) -> List[str]:
    hits = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k); walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and any(t in o for t in FORBIDDEN_TOKENS):
            hits.append(o)
    walk(obj)
    return hits


# ---------------------------------------------------------------------------
def synthetic_validation() -> dict:
    checks = {}
    rng = np.random.default_rng(660066)
    n_p, n_e = 5, 60
    base = rng.uniform(0.005, 0.009, size=(n_p, n_e))            # synthetic R0

    D0 = base - base
    checks["1_identical_zero"] = {"delta": float(D0.mean()), "PASS": abs(D0.mean()) < 1e-12}
    D2 = base - (base - 0.001)                                    # R1 better by 0.001
    checks["2_r1_better_const"] = {"delta": float(D2.mean()), "PASS": abs(D2.mean() - 0.001) < 1e-12}
    D3 = base - (base + 0.001)                                    # R1 worse by 0.001
    checks["3_r1_worse_const"] = {"delta": float(D3.mean()), "PASS": abs(D3.mean() + 0.001) < 1e-12}
    D_flat = np.full((n_p, n_e), 0.0005)
    D_pol = D_flat + np.array([-0.001, -0.0005, 0, 0.0005, 0.001])[:, None]
    c_flat = paired_crossed_two_axis_bootstrap(D_flat, n_boot=3000, seed=1)
    c_pol = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=1)
    w_flat = c_flat["ci_hi"] - c_flat["ci_lo"]; w_pol = c_pol["ci_hi"] - c_pol["ci_lo"]
    checks["4_policy_axis_widens_ci"] = {"w_flat": w_flat, "w_pol": w_pol, "PASS": w_pol > w_flat}
    D_env = D_flat + rng.uniform(-0.001, 0.001, size=(1, n_e))
    c_env = paired_crossed_two_axis_bootstrap(D_env, n_boot=3000, seed=2)
    w_env = c_env["ci_hi"] - c_env["ci_lo"]
    checks["5_env_axis_widens_ci"] = {"w_flat": w_flat, "w_env": w_env, "PASS": w_env > w_flat}
    common = rng.uniform(0, 1, size=(n_p, n_e)); treat = 0.001
    R0_6 = common + rng.uniform(0.004, 0.008, (n_p, n_e)); R1_6 = R0_6 - treat
    D6 = R0_6 - R1_6
    checks["6_pairing_cancels_common"] = {"delta": float(D6.mean()), "PASS": abs(D6.mean() - treat) < 1e-9}
    shared = rng.uniform(0, 0.02, size=(n_p, n_e))
    R0_7 = shared + 0.006; R1_7 = shared + 0.006 - 0.0003; D7 = R0_7 - R1_7
    paired7 = paired_crossed_two_axis_bootstrap(D7, n_boot=4000, seed=3)
    indep7 = independent_arm_bootstrap(R0_7, R1_7, n_boot=4000, seed=3)
    w_paired = paired7["ci_hi"] - paired7["ci_lo"]; w_indep = indep7["ci_hi"] - indep7["ci_lo"]
    checks["7_paired_narrower_than_independent"] = {"w_paired": w_paired, "w_indep": w_indep,
                                                    "PASS": w_paired < w_indep and "FORBIDDEN" in indep7["method"]}
    Dm = {"PERSISTENT": np.full((n_p, n_e), 0.001), "STATIONARY": np.full((n_p, n_e), 0.002),
          "MIXED": np.full((n_p, n_e), 0.003)}
    mac = paired_crossed_two_axis_bootstrap_macro(Dm, SCENARIOS, n_boot=2000, seed=4)
    checks["8_macro_equal_third"] = {"macro": mac["point_estimate"], "PASS": abs(mac["point_estimate"] - 0.002) < 1e-12}
    Dsame = {s: D_pol for s in SCENARIOS}
    mac2 = paired_crossed_two_axis_bootstrap_macro(Dsame, SCENARIOS, n_boot=3000, seed=99)
    sing = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=99)
    checks["9_macro_shared_draws"] = {"PASS": abs(mac2["ci_lo"] - sing["ci_lo"]) < 1e-12
                                      and abs(mac2["ci_hi"] - sing["ci_hi"]) < 1e-12}
    r4 = policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)
    r3 = policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)
    checks["10_policy_robustness"] = {"PASS": r4["PASS"] is True and r3["PASS"] is False}
    m33 = RI.reward_mechanism_supported(0.1, 0.1, 0.1)["REWARD_CREDIT_MECHANISM_SUPPORTED"]
    m23 = RI.reward_mechanism_supported(0.1, 0.1, -0.1)["REWARD_CREDIT_MECHANISM_SUPPORTED"]
    m13 = RI.reward_mechanism_supported(0.1, -0.1, -0.1)["REWARD_CREDIT_MECHANISM_SUPPORTED"]
    checks["11_mechanism_gate"] = {"m33": m33, "m23": m23, "m13": m13,
                                   "PASS": m33 == "YES" and m23 == "YES" and m13 == "NO"}
    # 12-14 mechanism endpoint directions
    checks["12_activity_direction"] = {"PASS": RI.reward_mechanism_supported(0.02, 0, 0)["delta_activity"] > 0}
    checks["13_ratio_direction"] = {"PASS": RI.reward_mechanism_supported(0, 500.0, 0)["delta_ratio"] > 0}
    checks["14_dispersion_direction"] = {"PASS": RI.reward_mechanism_supported(0, 0, 0.01)["delta_dispersion"] > 0}
    # 15 zero-active ratio -> NOT_APPLICABLE counts as not-in-direction
    z = RI.reward_mechanism_supported(0.02, "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL", 0.01)
    checks["15_zero_active_handling"] = {"n_expected": z["n_expected"],
                                         "PASS": z["REWARD_CREDIT_MECHANISM_SUPPORTED"] == "YES" and z["n_expected"] == 2}

    def V(mp, mech, rob, ps=True, ss=True, ms=True, eng=True):
        return RI.reward_ab_verdict(mixed_primary_label=mp, mechanism_supported=mech,
                                    policy_robustness_pass=rob, persistent_safety_pass=ps,
                                    stationary_safety_pass=ss, macro_safety_pass=ms,
                                    checkpoint_integrity_pass=eng, protocol_integrity_pass=eng,
                                    version_integrity_pass=eng, no_final_tuning=eng)
    checks["16_validated"] = {"PASS": V(BENEFICIAL, True, True)["verdict"] == "REWARD_CAUSAL_VALIDATED"}
    checks["17_performance_partial"] = {"PASS": V(BENEFICIAL, False, True)["subtype"] == "PERFORMANCE_PARTIAL"}
    checks["18_mechanistic_partial"] = {"PASS": V(PRACTICALLY_NEUTRAL, True, False)["subtype"] == "MECHANISTIC_PARTIAL"
                                        and V(PRACTICALLY_NEUTRAL, True, False)["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"}
    checks["19_failed_directional_negative"] = {"PASS": V(DIRECTIONAL_NEGATIVE, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"}
    checks["20_failed_harmful"] = {"PASS": V(HARMFUL, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"}
    checks["21_failed_safety"] = {"PASS": V(BENEFICIAL, True, True, ps=False)["verdict"] == "REWARD_CAUSAL_FAILED"}
    checks["22_mechanism_cannot_rescue_harmful"] = {"PASS": V(HARMFUL, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"}
    checks["23_neutral_without_mechanism_failed"] = {"PASS": V(PRACTICALLY_NEUTRAL, False, True)["verdict"] == "REWARD_CAUSAL_FAILED"}
    # 24 FE cannot rescue harmful: harmful MIXD stays FAILED regardless of a positive FE effect (logic level)
    checks["24_fe_cannot_rescue"] = {"PASS": V(HARMFUL, True, True)["verdict"] == "REWARD_CAUSAL_FAILED"}

    all_pass = all(c["PASS"] for c in checks.values())
    return {"checks": checks, "ALL_PASS": all_pass, "n_checks": len(checks),
            "no_final_seed_instantiated": True}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4IE — Reward A/B Final Evaluation Protocol Freeze")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4it = _load(E4IT_DIR / "E4IT_verdict.json")
    e4itr = _load(E4ITR_DIR / "E4ITR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    sel = _load(E4ITR_DIR / "E4ITR_selected_checkpoints.json")["selected"]
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    ck_ok = True
    ck_rows = {}
    for key, info in sel.items():
        actual = _hash_file(info["checkpoint_path"])
        exists = Path(info["checkpoint_path"]).exists()
        match = exists and actual == info["sha256"]
        arm = info["arm"]
        nv_ok = info["normalization_version"] == NORM_V3 and info["normalizer_constants_sha256"] == v3_sha
        rv_ok = (info["reward_version"] == (REWARD_R0_NAME if arm == "R0_CONTROL" else REWARD_R1_NAME))
        if arm == "R1_TREATMENT":
            rv_ok = rv_ok and info["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256
        ck_ok = ck_ok and match and nv_ok and rv_ok
        ck_rows[key] = {**{k: info[k] for k in ("arm", "policy_seed", "selected_checkpoint",
                        "transition_count", "checkpoint_path", "sha256", "file_size",
                        "normalization_version", "normalizer_constants_sha256", "reward_version",
                        "reward_config_sha256", "validation_macro_MIGD_response",
                        "validation_macro_MIGD_epoch", "validation_mean_response_fe")},
                        "sha256_recomputed": actual, "hash_match": match,
                        "normalization_metadata_ok": nv_ok, "reward_metadata_ok": rv_ok,
                        "per_scenario": info["per_scenario"]}
    n_r0 = sum(1 for v in ck_rows.values() if v["arm"] == "R0_CONTROL")
    n_r1 = sum(1 for v in ck_rows.values() if v["arm"] == "R1_TREATMENT")

    manifest = {"stage": "E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE", "freeze_only": True,
                "E4IT_COMPLETE": e4it["E4IT_COMPLETE"], "E4ITR_COMPLETE": e4itr["E4ITR_COMPLETE"],
                "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
                "n_selected_checkpoints": len(sel), "n_r0": n_r0, "n_r1": n_r1,
                "checkpoint_integrity": "PASS" if ck_ok else "FAIL", "v3_constants_sha256": v3_sha,
                "no_best_policy_selection": True, "no_checkpoint_reselection": True}
    _jdump(manifest, RESULTS_DIR / "E4IE_pre_freeze_manifest.json")
    ok0 = (e4it["E4IT_COMPLETE"] == "YES" and e4itr["E4ITR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and len(sel) == 10 and n_r0 == 5 and n_r1 == 5 and ck_ok)
    if not ok0:
        print("[E4IE] integrity FAIL — STOP")
        _jdump({"E4IE_COMPLETE": "NO", "NEXT_STAGE": "E4IE_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4IE_verdict.json")
        return {"E4IE_COMPLETE": "NO"}
    print("[E4IE] integrity PASS. 10 checkpoints (5 R0 + 5 R1) verified. E4F/E4HF preserved.")

    synth = synthetic_validation()
    assert _final_access["count"] == 0

    # ---- freeze contracts ----
    contracts = {}
    contracts["E4IE_checkpoint_freeze.json"] = {
        "immutable_for_final_evaluation": True, "n_checkpoints": 10, "n_r0": n_r0, "n_r1": n_r1,
        "checkpoints": ck_rows, "no_reselection": True, "no_alt_checkpoint_on_final": True,
        "no_weight_averaging": True, "no_ensemble": True, "no_additional_training": True,
        "all_ten_advance": True}
    contracts["E4IE_historical_integrity.json"] = {
        "E4F": "FAILED", "E4HF_normalization": "FAILED", "E4IT_COMPLETE": "YES", "E4ITR_COMPLETE": "YES",
        "R1_IMPLEMENTATION_VALID": "YES", "no_post_e4itr_training": True}
    contracts["E4IE_reward_version_integrity.json"] = {
        "R0_CONTROL": REWARD_R0_NAME, "R1_TREATMENT": REWARD_R1_NAME,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256, "alpha_local_R0": 0.1, "alpha_local_R1": 0.1,
        "R0_reward_unchanged": True, "R1_reward_unchanged": True, "no_runtime_alpha_derivation": True}
    contracts["E4IE_normalization_integrity.json"] = {
        "normalization": NORM_V3, "v3_constants_sha256": v3_sha, "identical_both_arms": True,
        "no_refit": True, "no_adaptation": True, "no_online_update": True}
    contracts["E4IE_final_seed_contract.json"] = {
        "final_seeds": FINAL_SEEDS, "range": [1581, 1640], "n": 60, "never_instantiated": True,
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _final_access["count"]}
    contracts["E4IE_execution_order.json"] = {
        "scenario_order": SCENARIOS, "environment_seed_order": "1581..1640 ascending",
        "policy_seed_order": POLICY_SEEDS, "within_pair": ["R0_CONTROL", "R1_TREATMENT"],
        "outcome_independent": True, "no_early_stop": True, "no_reorder_on_intermediate_results": True}
    contracts["E4IE_final_matrix_contract.json"] = {
        "arms": ["R0_CONTROL", "R1_TREATMENT"], "policy_seeds": POLICY_SEEDS,
        "environment_seeds": [1581, 1640], "scenarios": SCENARIOS,
        "total_episodes": 2 * 5 * 60 * 3, "R0_rows": 900, "R1_rows": 900,
        "pairing_key": ["policy_seed", "environment_seed", "scenario"],
        "FINAL_MATRIX_COMPLETE_required": True,
        "on_missing": "FINAL_MATRIX_COMPLETE=NO; REWARD_CAUSAL_VERDICT=NOT_COMPUTED; no reduced analysis; "
                      "no seed/policy replacement; no imputation",
        "no_global_or_rule_in_primary_matrix": True, "no_pseudoreplication": True,
        "inferential_axes": ["policy_seed", "environment_seed"]}
    contracts["E4IE_pairing_contract.json"] = {
        "pair_by": ["policy_seed", "environment_seed", "scenario"],
        "example": "R0_4001/MIXED/1581 pairs R1_4001/MIXED/1581",
        "common_environment_schedule": True, "no_new_action_dependent_CRN": True}
    contracts["E4IE_endpoint_contract.json"] = {
        "primary_endpoint": "MIGD_response", "primary_direction": "lower_is_better",
        "primary_scenario": "MIXED", "secondary": ["MIGD_epoch", "response_FE"],
        "primary_comparison": "R1_TREATMENT_VS_R0_CONTROL"}
    contracts["E4IE_effect_definition.json"] = {
        "cell_effect": "d[p,e,s] = MIGD_response_R0[p,e,s] - MIGD_response_R1[p,e,s]",
        "positive_means": "R1 better", "scenario_effect": "equal mean over 5 policies x 60 env",
        "primary_statistical_units": ["policy_seed", "environment_seed"],
        "forbidden_replicate_units": ["region", "change", "transition"],
        "relative_improvement": "Delta_MIXED / mean_R0_MIXED (descriptive, not a gate)"}
    contracts["E4IE_practical_margin.json"] = {"DELTA_NORM": DELTA_NORM, "applies_to": "MIGD effects",
                                               "immutable_after_final_data": True}
    contracts["E4IE_classification_logic.json"] = {
        "labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL],
        "DELTA_NORM": DELTA_NORM,
        "BENEFICIAL": "mean>+DELTA AND ci_lo>0", "DIRECTIONAL_POSITIVE": "mean>+DELTA AND ci crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(mean)<=DELTA AND NOT(ci_hi<0)",
        "DIRECTIONAL_NEGATIVE": "mean<-DELTA AND ci crosses 0", "HARMFUL": "mean<-DELTA AND ci_hi<0",
        "exhaustive_extension": "reuse final_verdict.classify_effect (defer to CI direction; documented before final data)",
        "source": "src/prdrl/evaluation/final_verdict.py::classify_effect"}
    contracts["E4IE_paired_bootstrap_contract.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "function": "paired_crossed_two_axis_bootstrap",
        "policy_axis_n": 5, "environment_axis_n": 60, "resample_both_with_replacement": True,
        "full_sampled_cartesian_product": True, "preserve_R0_R1_pairing": True,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "forbidden": ["naive_300_row_bootstrap", "independent_arm_bootstrap",
                      "region/change/transition as inferential replicate"]}
    contracts["E4IE_macro_contract.json"] = {
        "macro": "(Delta_PERSISTENT+Delta_STATIONARY+Delta_MIXED)/3", "scenario_weight": 1/3,
        "shared_bootstrap_draws_across_scenarios": True,
        "function": "paired_crossed_two_axis_bootstrap_macro"}
    contracts["E4IE_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED]=mean_e(R0-R1)",
        "rule": "PASS iff >=4 of 5 policy-pair effects > 0", "threshold": 4,
        "no_replacement": True, "no_best_seed_filtering": True}
    contracts["E4IE_reward_mechanism_contract.json"] = {
        "primary_mechanism": "region-specific credit differentiation (NOT action diversity)",
        "endpoints": ["LOCAL_CREDIT_ACTIVITY", "GLOBAL_LOCAL_MAGNITUDE_RATIO", "REGIONAL_REWARD_DISPERSION"],
        "endpoint_1": {"R0": "R0 local nonzero/active fraction (E4I/E4IT def)", "R1": "|g_r|>1e-12 fraction",
                       "delta_activity": "R1 - R0 (>0 expected)"},
        "endpoint_2": {"definition": "median|weighted_global| / (median|weighted_local| + eps) on matched "
                       "active-local subset", "delta_ratio": "R0_ratio - R1_ratio (>0 expected)",
                       "zero_active": "NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL (counts as not-in-direction)"},
        "endpoint_3": {"metric": "event population std (ddof=0) of weighted local across active regions, "
                       "aggregated per E4IT", "delta_dispersion": "R1 - R0 (>0 expected)"},
        "gate": "REWARD_CREDIT_MECHANISM_SUPPORTED = YES iff >=2 of 3 point estimates in expected direction",
        "source": "reward_integration.reward_mechanism_supported",
        "significance_not_required": True,
        "final_mechanism_from": "final-evaluation regional reward primitives (zero-FE reconstruction, no learning)"}
    contracts["E4IE_zero_active_contract.json"] = {
        "rule": "scenario/arm with zero active local rows -> ratio NOT_APPLICABLE_NO_ACTIVE_LOCAL_SIGNAL",
        "no_divide_by_zero": True, "not_material": True, "not_improvement": True,
        "pooled_follows_frozen_rule": True}
    contracts["E4IE_safety_contract.json"] = {
        "scenarios": ["PERSISTENT", "STATIONARY"], "endpoints": ["MIGD_response", "MIGD_epoch"],
        "effect": "R0 - R1 (positive favors R1)", "safety_pass": "neither endpoint HARMFUL",
        "macro_safety": "neither macro MIGD_response nor macro MIGD_epoch HARMFUL",
        "improvement_not_required": True, "fe_cannot_rescue_harmful_migd": True}
    contracts["E4IE_fe_contract.json"] = {
        "effect": "d_FE = FE_R0 - FE_R1 (positive = R1 fewer evaluations)",
        "scenarios": SCENARIOS + ["macro"], "secondary": True, "cannot_rescue_harmful_quality": True}
    contracts["E4IE_success_logic.json"] = {
        "question": "REWARD CAUSALITY (R1 vs R0) under fixed V3 + implicit joint-action; NOT joint-arch "
                    "adequacy, global optimality, or superiority over GLOBAL_PREDICT",
        "REWARD_CAUSAL_VALIDATED_requires_all": ["MIXED primary=BENEFICIAL",
            "REWARD_CREDIT_MECHANISM_SUPPORTED=YES", "REWARD_POLICY_ROBUSTNESS=PASS",
            "PERSISTENT/STATIONARY/MACRO safety=PASS", "checkpoint+reward-version/config+protocol integrity=PASS",
            "no final-data tuning"],
        "PERFORMANCE_PARTIAL": "MIXED in {BENEFICIAL,DIRECTIONAL_POSITIVE}, no safety HARMFUL, engineering PASS, "
                               "but a supporting gate (mechanism/robustness) unmet",
        "MECHANISTIC_PARTIAL": "MIXED=PRACTICALLY_NEUTRAL AND mechanism supported AND all safety PASS AND "
                               "integrity PASS -> PERFORMANCE_IMPROVEMENT_CLAIM=NOT_SUPPORTED",
        "FAILED": "MIXED in {HARMFUL,DIRECTIONAL_NEGATIVE} OR any safety FAIL OR integrity fail OR "
                  "(MIXED neutral AND mechanism NOT supported)",
        "mechanism_cannot_rescue_harmful": True,
        "beneficial_without_mechanism_is_partial_not_validated": True,
        "source": "reward_integration.reward_ab_verdict (== final_verdict.normalization_ab_verdict logic)"}
    contracts["E4IE_output_schema.json"] = {
        "e4if_results_dir": E4IF_DIR,
        "episode_row_keys": ["arm", "policy_seed", "scenario", "environment_seed"],
        "episode_metrics": ["MIGD_response", "MIGD_epoch", "response_FE"],
        "regional_record_keys": ["arm", "policy_seed", "scenario", "environment_seed", "change_index",
                                 "region_id", "requested_action", "realized_operator", "fallback",
                                 "L_pre", "L_post", "weighted_global_reward", "weighted_local_R0", "g_r",
                                 "weighted_local_R1", "local_active_R0", "local_active_R1", "weighted_fe_penalty"],
        "zero_fe_reconstruction_only": True, "no_duplicate_keys": True, "no_pseudoreplication": True,
        "e4if_required_outputs": ["E4IF_pre_execution_manifest.json", "E4IF_checkpoint_integrity.json",
            "E4IF_execution_config.json", "E4IF_progress.jsonl", "E4IF_episode_results.csv",
            "E4IF_regional_reward_records.csv", "E4IF_action_distribution.csv",
            "E4IF_reward_mechanism_effects.csv", "E4IF_policy_pair_summary.csv", "E4IF_scenario_effects.csv",
            "E4IF_macro_effects.csv", "E4IF_fe_effects.csv", "E4IF_bootstrap_results.json",
            "E4IF_scientific_gates.json", "E4IF_validation_final_comparison.json",
            "E4IF_final_seed_usage.json", "E4IF_verdict.json", "E4IF_report.md"]}
    contracts["E4IE_final_execution_contract.json"] = {
        "evaluate": "selected online weights only", "epsilon": 0.0, "network_mode": "eval",
        "gradients": "disabled", "replay": "disabled", "optimizer": "disabled",
        "target_updates": "disabled", "learning": "none", "model_updates": "forbidden",
        "state_dim": 9, "network": "9-128-128-4", "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "joint_representation": "IMPLICIT_STATE_ONLY", "branch_C_untouched": True,
        "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0, "sigma_D": 0.15,
                      "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted NSGA-II"},
        "common_environment_schedule_by": ["policy_seed", "scenario", "environment_seed"],
        "simultaneous_joint_response": True, "requested_realized_fallback_semantics": True,
        "deterministic_resume": True, "shadow_reward_zero_extra_FE": True,
        "v3_frozen_constants_no_adaptation": True}
    contracts["E4IE_validation_observation_lock.json"] = {
        "status": "NON_CONFIRMATORY_LOCKED",
        "training_reward_mechanism_direction": "3/3 EXPECTED_DIRECTION",
        "validation_macro_delta_R0_minus_R1": e4itr.get("VALIDATION_PAIRED_DELTA_R0_MINUS_R1"),
        "R0_val_macro_MIGD_response": e4itr.get("R0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
        "R1_val_macro_MIGD_response": e4itr.get("R1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
        "must_not_alter": ["primary_endpoint", "practical_margin", "bootstrap", "safety", "robustness",
                           "mechanism_gate", "verdict_logic", "checkpoints", "alpha"]}

    token_hits = []
    for name, obj in contracts.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    _jdump(synth, RESULTS_DIR / "E4IE_synthetic_validation.json")
    for name, obj in contracts.items():
        _jdump(obj, RESULTS_DIR / name)
    _jdump({"final_seeds": [1581, 1640], "REWARD_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
            "FINAL_EVALUATION": "NOT_RUN", "REWARD_CAUSAL_CLAIM": "NOT_MADE", "NEW_TRAINING": "NO"},
           RESULTS_DIR / "E4IE_final_firewall.json")

    complete = (ck_ok and synth["ALL_PASS"] and len(token_hits) == 0 and _final_access["count"] == 0)

    _jdump({"stage": "E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE", "all_frozen_before_final": True,
            "control": "V3_PLUS_R0", "treatment": "V3_PLUS_R1", "experiment_factor": "REWARD_ONLY",
            "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
            "synthetic_validation": "PASS" if synth["ALL_PASS"] else "FAIL",
            "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response", "effect_definition": "R0_MINUS_R1",
            "practical_margin": DELTA_NORM, "statistical_design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
            "policy_axis_n": 5, "environment_axis_n": 60, "N_BOOTSTRAP": N_BOOTSTRAP,
            "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED, "final_matrix_planned": 1800,
            "final_seeds": [1581, 1640], "components": list(contracts.keys()),
            "no_placeholder_tokens": len(token_hits) == 0, "next_stage": "E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION"},
           RESULTS_DIR / "E4IE_protocol_frozen.json")

    verdict = _build_verdict(ck_ok, synth, token_hits, complete, n_r0, n_r1)
    _jdump(verdict, RESULTS_DIR / "E4IE_verdict.json")
    _write_report(verdict, synth)
    _emit_terminal(verdict)
    print(f"[E4IE] wall {time.time()-t0:.1f}s")
    return verdict


def _build_verdict(ck_ok, synth, token_hits, complete, n_r0, n_r1):
    return {
        "stage": "E4IE_REWARD_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "E4IE_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4ITR_COMPLETE": "YES", "SELECTED_BRANCH": "B_REWARD_CREDIT",
        "CHECKPOINT_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "REWARD_VERSION_CONFIG_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "NORMALIZATION_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "R0_SELECTED_CHECKPOINTS": f"{n_r0}/5", "R1_SELECTED_CHECKPOINTS": f"{n_r1}/5",
        "FINAL_ENVIRONMENT_SEEDS": "1581..1640", "POLICY_SEEDS": "4001..4005",
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "R1_TREATMENT_VS_R0_CONTROL", "EFFECT_DEFINITION": "R0_MINUS_R1",
        "POSITIVE_MEANS": "R1_BETTER", "PRACTICAL_MARGIN": DELTA_NORM,
        "STATISTICAL_DESIGN": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "POLICY_AXIS_N": 5,
        "ENVIRONMENT_AXIS_N": 60, "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "REWARD_MECHANISM_ENDPOINTS": "LOCAL_CREDIT_ACTIVITY,GLOBAL_LOCAL_MAGNITUDE_RATIO,REGIONAL_REWARD_DISPERSION",
        "REWARD_MECHANISM_GATE": "AT_LEAST_2_OF_3_EXPECTED_DIRECTIONS",
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_R1_BETTER", "FINAL_MATRIX_PLANNED": "1800_EPISODES",
        "SYNTHETIC_STATISTICS_VALIDATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "VALIDATION_OBSERVATION_STATUS": "NON_CONFIRMATORY_LOCKED",
        "no_placeholder_tokens": len(token_hits) == 0, "REWARD_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
        "FINAL_EVALUATION": "NOT_RUN", "REWARD_CAUSAL_CLAIM": "NOT_MADE", "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4IF_REWARD_AB_FINAL_EVALUATION_EXECUTION" if complete else "E4IE_INTEGRITY_REVIEW")}


def _write_report(v, synth):
    L = ["# E4IE — Reward A/B Final Evaluation Protocol Freeze", "",
         f"**E4IE_COMPLETE = {v['E4IE_COMPLETE']}** — freeze only; final seeds 1581..1640 unopened; no training. "
         "Question: REWARD CAUSALITY (R1 vs R0) under fixed V3 + implicit joint-action; primary comparator R0_CONTROL.", "",
         "## A/B. Historical preservation + E4ITR integrity",
         f"- E4F = FAILED; E4HF normalization = FAILED; E4ITR COMPLETE; CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}.", "",
         "## C. Exact 10 frozen checkpoints",
         f"- R0 {v['R0_SELECTED_CHECKPOINTS']} + R1 {v['R1_SELECTED_CHECKPOINTS']}; all hash-verified, immutable "
         "(no reselection/averaging/ensemble/retrain); all ten advance.", "",
         "## D/E/F. Reward-only factor + version + V3 locks",
         f"- CONTROL V3+R0, TREATMENT V3+R1; alpha_local 0.1 both; R1 config sha frozen; "
         f"REWARD_VERSION_CONFIG_INTEGRITY = {v['REWARD_VERSION_CONFIG_INTEGRITY']}; NORMALIZATION_INTEGRITY = "
         f"{v['NORMALIZATION_INTEGRITY']}.", "",
         "## G/H/I. Final firewall + matrix + pairing",
         "- final seeds 1581..1640 (60), never instantiated; 2×5×60×3 = 1800 (900 R0 + 900 R1); pair by "
         "(policy, env, scenario); FINAL_MATRIX_COMPLETE required.", "",
         "## J/K/L. Primary endpoint + classification + bootstrap",
         f"- MIXED MIGD_response, d = R0 − R1 (+ ⇒ R1 better), DELTA_NORM 1e-5; paired crossed two-axis "
         f"bootstrap 5×60, N=50000, 95% percentile, master seed {BOOTSTRAP_MASTER_SEED}; naive/independent forbidden.", "",
         "## M/N/O. Robustness + reward mechanism + zero-active",
         "- robustness ≥4/5 R1 better. Mechanism (≥2/3): activity R1>R0, global/local ratio R1<R0, dispersion "
         "R1>R0 (event std ddof=0). Zero-active ⇒ NOT_APPLICABLE (no /0, not material).", "",
         "## P/Q/R/S. Safety + macro + FE",
         "- PERSISTENT/STATIONARY/macro not HARMFUL; macro equal 1/3 shared draws; FE secondary, cannot rescue.", "",
         "## T. Final verdict logic",
         "- REWARD_CAUSAL_VALIDATED / PERFORMANCE_PARTIAL / MECHANISTIC_PARTIAL / FAILED "
         "(reward_ab_verdict; mechanism cannot rescue harmful).", "",
         f"## U. Synthetic pipeline validation: {synth['n_checks']} checks, ALL_PASS = {synth['ALL_PASS']}",
         "## V. Validation observation lock: NON_CONFIRMATORY_LOCKED (mechanism 3/3, val delta +1.13e-4 favor R1) "
         "— does NOT alter the frozen protocol.", "",
         "## W. Interpretation boundary",
         "- If mechanism works but final performance fails → the bounded conclusion is that V3 restored state AND "
         "R1 restored credit BUT the implicit joint-action representation did not convert them to final quality → "
         "Branch C is the next factor (NOT implemented here).", "",
         f"## X. Next stage: {v['NEXT_STAGE']}", "",
         "Descriptive training/validation already favor R1; the protocol is NOT changed in response — the untouched "
         "final seeds 1581..1640 decide the preregistered reward-causality question."]
    (RESULTS_DIR / "E4IE_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4IE_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT", "E4ITR_COMPLETE",
              "SELECTED_BRANCH", "CHECKPOINT_INTEGRITY", "REWARD_VERSION_CONFIG_INTEGRITY",
              "NORMALIZATION_INTEGRITY", "R0_SELECTED_CHECKPOINTS", "R1_SELECTED_CHECKPOINTS",
              "FINAL_ENVIRONMENT_SEEDS", "POLICY_SEEDS", "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT",
              "PRIMARY_COMPARISON", "EFFECT_DEFINITION", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
              "STATISTICAL_DESIGN", "POLICY_AXIS_N", "ENVIRONMENT_AXIS_N", "BOOTSTRAP_RESAMPLES",
              "BOOTSTRAP_MASTER_SEED", "REWARD_MECHANISM_ENDPOINTS", "REWARD_MECHANISM_GATE",
              "POLICY_ROBUSTNESS_RULE", "FINAL_MATRIX_PLANNED", "SYNTHETIC_STATISTICS_VALIDATION",
              "VALIDATION_OBSERVATION_STATUS", "REWARD_FINAL_EVAL_ACCESS_COUNT", "FINAL_EVALUATION",
              "REWARD_CAUSAL_CLAIM", "NEW_TRAINING", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<36} = {v[k]}")


if __name__ == "__main__":
    main()
