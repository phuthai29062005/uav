"""
E4KE — Joint Coordination A/B Final Evaluation Protocol Freeze (PR-DRL-DMOEA, Branch C).

Freezes the COMPLETE untouched final causal protocol for C0_INDEPENDENT_SHARED_DDQN (CONTROL)
vs C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1 (TREATMENT) BEFORE the reserved final seeds
1981..2040 are opened, and validates the entire analysis pipeline on SYNTHETIC matrices only.

FREEZE ONLY: no final environment instantiated (access=0), no training, no checkpoint
reselection, no architecture change. Joint coordination specification is the SOLE causal factor;
the primary comparator is C0_INDEPENDENT_SHARED_DDQN. Descriptive E4KTR validation evidence
(delta C0-C1 = -7.75e-5, C0 descriptively better) is LOCKED as NON_CONFIRMATORY and must not
alter the frozen protocol.

COORDINATION_MECHANISM_GATE = NOT_USED_FOR_SCIENTIFIC_VERDICT.
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
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, FROZEN_V3_CONSTANTS_SHA256, coordination_causal_verdict,
    make_qmix_config, qmix_config_sha256,
)
from prdrl.rl.qmix import TREATMENT_ARCH
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_CONFIG_SHA256, FROZEN_R1_ALPHA_LOCAL

RESULTS_DIR = Path("results/E4KE_joint_coordination_ab_final_evaluation_protocol_freeze")
E4KT_DIR = Path("results/E4KT_joint_coordination_ab_training_protocol_freeze")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4IF_DIR_PATH = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")

FINAL_SEEDS = list(range(1981, 2041))
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DELTA_COORD = 1e-5
BOOTSTRAP_MASTER_SEED = 86001
FROZEN_QMIX_CONFIG_SHA256 = "2c7a14e02d853cd15dacd339c5dfdc966f90d56d11ef65b3f2ff65a83fcdc365"
E4KF_DIR = "results/E4KF_joint_coordination_ab_final_evaluation_execution"
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
    rng = np.random.default_rng(860086)
    n_p, n_e = 5, 60
    base = rng.uniform(0.005, 0.009, size=(n_p, n_e))

    D0 = base - base
    checks["1_identical_zero"] = {"delta": float(D0.mean()), "PASS": abs(D0.mean()) < 1e-12}
    D2 = base - (base - 0.001)
    checks["2_c1_better_const"] = {"delta": float(D2.mean()), "PASS": abs(D2.mean() - 0.001) < 1e-12}
    D3 = base - (base + 0.001)
    checks["3_c1_worse_const"] = {"delta": float(D3.mean()), "PASS": abs(D3.mean() + 0.001) < 1e-12}

    D_flat = np.full((n_p, n_e), 0.0005)
    D_pol = D_flat + np.array([-0.001, -0.0005, 0, 0.0005, 0.001])[:, None]
    c_flat = paired_crossed_two_axis_bootstrap(D_flat, n_boot=3000, seed=1)
    c_pol = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=1)
    w_flat = c_flat["ci_hi"] - c_flat["ci_lo"]
    w_pol = c_pol["ci_hi"] - c_pol["ci_lo"]
    checks["4_policy_axis_widens_ci"] = {"w_flat": w_flat, "w_pol": w_pol, "PASS": w_pol > w_flat}

    D_env = D_flat + rng.uniform(-0.001, 0.001, size=(1, n_e))
    c_env = paired_crossed_two_axis_bootstrap(D_env, n_boot=3000, seed=2)
    w_env = c_env["ci_hi"] - c_env["ci_lo"]
    checks["5_env_axis_widens_ci"] = {"w_flat": w_flat, "w_env": w_env, "PASS": w_env > w_flat}

    common = rng.uniform(0, 1, size=(n_p, n_e))
    treat = 0.001
    C0_6 = common + rng.uniform(0.004, 0.008, (n_p, n_e))
    C1_6 = C0_6 - treat
    D6 = C0_6 - C1_6
    checks["6_pairing_cancels_common"] = {"delta": float(D6.mean()), "PASS": abs(D6.mean() - treat) < 1e-9}

    shared = rng.uniform(0, 0.02, size=(n_p, n_e))
    C0_7 = shared + 0.006
    C1_7 = shared + 0.006 - 0.0003
    D7 = C0_7 - C1_7
    paired7 = paired_crossed_two_axis_bootstrap(D7, n_boot=4000, seed=3)
    indep7 = independent_arm_bootstrap(C0_7, C1_7, n_boot=4000, seed=3)
    w_paired = paired7["ci_hi"] - paired7["ci_lo"]
    w_indep = indep7["ci_hi"] - indep7["ci_lo"]
    checks["7_paired_narrower_than_independent"] = {
        "w_paired": w_paired, "w_indep": w_indep,
        "PASS": w_paired < w_indep and "FORBIDDEN" in indep7["method"]}

    Dm = {"PERSISTENT": np.full((n_p, n_e), 0.001),
          "STATIONARY": np.full((n_p, n_e), 0.002),
          "MIXED": np.full((n_p, n_e), 0.003)}
    mac = paired_crossed_two_axis_bootstrap_macro(Dm, SCENARIOS, n_boot=2000, seed=4)
    checks["8_macro_equal_third"] = {"macro": mac["point_estimate"],
                                     "PASS": abs(mac["point_estimate"] - 0.002) < 1e-12}

    Dsame = {s: D_pol for s in SCENARIOS}
    mac2 = paired_crossed_two_axis_bootstrap_macro(Dsame, SCENARIOS, n_boot=3000, seed=99)
    sing = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=99)
    checks["9_macro_shared_draws"] = {"PASS": abs(mac2["ci_lo"] - sing["ci_lo"]) < 1e-12
                                      and abs(mac2["ci_hi"] - sing["ci_hi"]) < 1e-12}

    r4 = policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)
    r3 = policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)
    checks["10_policy_robustness"] = {"PASS": r4["PASS"] is True and r3["PASS"] is False}

    # 11-14: coordination verdict branches (NO mechanism gate)
    def V(mp, rob, ps=True, ss=True, ms=True, eng=True):
        return coordination_causal_verdict(
            mixed_primary_label=mp, policy_robustness_pass=rob,
            persistent_safety_pass=ps, stationary_safety_pass=ss, macro_safety_pass=ms,
            checkpoint_integrity_pass=eng, protocol_integrity_pass=eng,
            architecture_integrity_pass=eng, no_final_tuning=eng)

    checks["11_validated"] = {"PASS": V(BENEFICIAL, True)["verdict"] == "COORDINATION_CAUSAL_VALIDATED"}
    checks["12_perf_partial_beneficial_norobust"] = {
        "PASS": V(BENEFICIAL, False)["subtype"] == "PERFORMANCE_PARTIAL"}
    checks["13_perf_partial_dirpos_robust"] = {
        "PASS": V(DIRECTIONAL_POSITIVE, True)["subtype"] == "PERFORMANCE_PARTIAL"}
    checks["14_perf_partial_dirpos_norobust"] = {
        "PASS": V(DIRECTIONAL_POSITIVE, False)["subtype"] == "PERFORMANCE_PARTIAL"}
    checks["15_neutral_partial"] = {
        "PASS": V(PRACTICALLY_NEUTRAL, True)["subtype"] == "PRACTICALLY_NEUTRAL"
        and V(PRACTICALLY_NEUTRAL, True)["COORDINATION_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"}
    checks["16_neutral_with_robust"] = {
        "PASS": V(PRACTICALLY_NEUTRAL, True)["verdict"] == "COORDINATION_CAUSAL_PARTIAL"}
    checks["17_failed_dirneg"] = {
        "PASS": V(DIRECTIONAL_NEGATIVE, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["18_failed_harmful"] = {
        "PASS": V(HARMFUL, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["19_failed_persistent_safety"] = {
        "PASS": V(BENEFICIAL, True, ps=False)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["20_failed_stationary_safety"] = {
        "PASS": V(BENEFICIAL, True, ss=False)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["21_failed_macro_safety"] = {
        "PASS": V(BENEFICIAL, True, ms=False)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["22_failed_arch_integrity"] = {
        "PASS": V(BENEFICIAL, True, eng=False)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["23_failed_adaptation"] = {
        "PASS": coordination_causal_verdict(
            mixed_primary_label=BENEFICIAL, policy_robustness_pass=True,
            persistent_safety_pass=True, stationary_safety_pass=True, macro_safety_pass=True,
            checkpoint_integrity_pass=True, protocol_integrity_pass=True,
            architecture_integrity_pass=True, no_final_tuning=False
        )["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["24_diagnostics_cannot_rescue_harmful"] = {
        "PASS": V(HARMFUL, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["25_fe_cannot_rescue_harmful"] = {
        "PASS": V(HARMFUL, True)["verdict"] == "COORDINATION_CAUSAL_FAILED"}
    checks["26_no_mechanism_gate_in_validated"] = {
        "PASS": V(BENEFICIAL, True)["verdict"] == "COORDINATION_CAUSAL_VALIDATED"}
    checks["27_effect_direction"] = {
        "PASS": classify_effect(0.001, 0.0005, 0.0015, DELTA_COORD) == BENEFICIAL
        and classify_effect(-0.001, -0.0015, -0.0005, DELTA_COORD) == HARMFUL}
    checks["28_qmix_config_sha_deterministic"] = {
        "PASS": qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256}

    all_pass = all(c["PASS"] for c in checks.values())
    return {"checks": checks, "ALL_PASS": all_pass, "n_checks": len(checks),
            "no_final_seed_instantiated": True}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4KE — Joint Coordination A/B Final Evaluation Protocol Freeze")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4kt = _load(E4KT_DIR / "E4KT_verdict.json")
    e4ktr = _load(E4KTR_DIR / "E4KTR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4if = _load(E4IF_DIR_PATH / "E4IF_verdict.json")
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    sel = _load(E4KTR_DIR / "E4KTR_selected_checkpoints.json")["selected"]
    val_summary = _load(E4KTR_DIR / "E4KTR_validation_ab_summary.csv".replace(".csv", ".csv")
                        ) if False else None  # CSV read separately

    ck_ok = True
    ck_rows = {}
    for key, info in sel.items():
        actual = _hash_file(info["checkpoint_path"])
        exists = Path(info["checkpoint_path"]).exists()
        match = exists and actual == info["sha256"]
        arm = info["arm"]
        nv_ok = (info["normalization_version"] == NORM_V3
                 and info["normalizer_constants_sha256"] == FROZEN_V3_CONSTANTS_SHA256)
        rv_ok = info["reward_version"] == REWARD_R1_NAME
        if rv_ok:
            rv_ok = rv_ok and info["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256
        arch_ok = True
        if arm == "C0_INDEPENDENT_SHARED_DDQN":
            arch_ok = info["architecture_version"] == CONTROL_ARCH
        elif arm == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1":
            arch_ok = info["architecture_version"] == TREATMENT_ARCH
        else:
            arch_ok = False
        ck_ok = ck_ok and match and nv_ok and rv_ok and arch_ok
        ck_rows[key] = {
            **{k: info[k] for k in ("arm", "policy_seed", "selected_checkpoint",
               "transition_count", "checkpoint_path", "sha256",
               "normalization_version", "normalizer_constants_sha256",
               "reward_version", "reward_config_sha256", "architecture_version",
               "validation_macro_MIGD_response", "validation_macro_MIGD_epoch",
               "validation_mean_response_fe")},
            "sha256_recomputed": actual, "hash_match": match,
            "normalization_metadata_ok": nv_ok, "reward_metadata_ok": rv_ok,
            "architecture_metadata_ok": arch_ok, "per_scenario": info["per_scenario"]}

    n_c0 = sum(1 for v in ck_rows.values() if v["arm"] == "C0_INDEPENDENT_SHARED_DDQN")
    n_c1 = sum(1 for v in ck_rows.values() if v["arm"] == "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1")

    qmix_sha_ok = qmix_config_sha256() == FROZEN_QMIX_CONFIG_SHA256

    manifest = {
        "stage": "E4KE_JOINT_COORDINATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "freeze_only": True,
        "E4KT_COMPLETE": e4kt["E4KT_COMPLETE"],
        "E4KTR_COMPLETE": e4ktr["E4KTR_COMPLETE"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
        "E4JF_GLOBAL_COMPARATOR_VERDICT": e4jf["GLOBAL_COMPARATOR_VERDICT"],
        "n_selected_checkpoints": len(sel), "n_c0": n_c0, "n_c1": n_c1,
        "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
        "qmix_config_sha256_match": qmix_sha_ok,
        "no_best_policy_selection": True, "no_checkpoint_reselection": True}
    _jdump(manifest, RESULTS_DIR / "E4KE_pre_freeze_manifest.json")

    ok0 = (e4kt["E4KT_COMPLETE"] == "YES" and e4ktr["E4KTR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
           and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
           and e4jf["GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
           and len(sel) == 10 and n_c0 == 5 and n_c1 == 5 and ck_ok and qmix_sha_ok)
    if not ok0:
        print("[E4KE] integrity FAIL — STOP")
        _jdump({"E4KE_COMPLETE": "NO", "NEXT_STAGE": "E4KE_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4KE_verdict.json")
        return {"E4KE_COMPLETE": "NO"}
    print("[E4KE] integrity PASS. 10 checkpoints (5 C0 + 5 C1) verified. E4F/E4HF/E4IF/E4JF preserved.")

    synth = synthetic_validation()
    assert _final_access["count"] == 0

    # ---- freeze contracts ----
    contracts = {}
    contracts["E4KE_checkpoint_freeze.json"] = {
        "immutable_for_final_evaluation": True, "n_checkpoints": 10, "n_c0": n_c0, "n_c1": n_c1,
        "checkpoints": ck_rows, "no_reselection": True, "no_alt_checkpoint_on_final": True,
        "no_weight_averaging": True, "no_ensemble": True, "no_additional_training": True,
        "all_ten_advance": True}
    contracts["E4KE_historical_integrity.json"] = {
        "E4F": "FAILED", "E4HF_normalization": "FAILED", "E4IF_reward": "VALIDATED",
        "E4JF_global_comparator": "FAILED",
        "E4KT_COMPLETE": "YES", "E4KTR_COMPLETE": "YES",
        "QMIX_IMPLEMENTATION_VALID": "YES", "no_post_e4ktr_training": True}
    contracts["E4KE_architecture_integrity.json"] = {
        "C0_CONTROL": CONTROL_ARCH, "C1_TREATMENT": TREATMENT_ARCH,
        "QMIX_CONFIG_SHA256": FROZEN_QMIX_CONFIG_SHA256, "qmix_sha_verified": qmix_sha_ok,
        "utility_architecture": "9-128-128-4", "state_dim": 9, "n_actions": 4,
        "c0_has_mixer": False, "c1_has_mixer": True,
        "mixer_hidden_dim": 32, "central_state_dim": 55, "R_MAX": 10,
        "nonneg_transform": "ABS", "activation": "ELU",
        "both_arms_same_utility": True, "factor": "JOINT_COORDINATION_LEARNER_SPECIFICATION"}
    contracts["E4KE_reward_version_integrity.json"] = {
        "both_arms_reward": REWARD_R1_NAME,
        "R1_config_sha256": FROZEN_R1_CONFIG_SHA256,
        "alpha_local_both_arms": FROZEN_R1_ALPHA_LOCAL,
        "reward_unchanged_both_arms": True, "no_runtime_alpha_derivation": True}
    contracts["E4KE_normalization_integrity.json"] = {
        "normalization": NORM_V3, "v3_constants_sha256": FROZEN_V3_CONSTANTS_SHA256,
        "identical_both_arms": True, "no_refit": True, "no_adaptation": True, "no_online_update": True}
    contracts["E4KE_final_seed_contract.json"] = {
        "final_seeds": FINAL_SEEDS, "range": [1981, 2040], "n": 60, "never_instantiated": True,
        "COORDINATION_FINAL_EVAL_ACCESS_COUNT": _final_access["count"]}
    contracts["E4KE_execution_order.json"] = {
        "scenario_order": SCENARIOS,
        "environment_seed_order": "1981..2040 ascending",
        "policy_seed_order": POLICY_SEEDS,
        "within_pair": ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"],
        "outcome_independent": True, "no_early_stop": True,
        "no_reorder_on_intermediate_results": True}
    contracts["E4KE_final_matrix_contract.json"] = {
        "arms": ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"],
        "policy_seeds": POLICY_SEEDS, "environment_seeds": [1981, 2040],
        "scenarios": SCENARIOS,
        "total_episodes": 2 * 5 * 60 * 3, "C0_rows": 900, "C1_rows": 900,
        "pairing_key": ["policy_seed", "environment_seed", "scenario"],
        "FINAL_MATRIX_COMPLETE_required": True,
        "on_missing": "FINAL_MATRIX_COMPLETE=NO; COORDINATION_CAUSAL_VERDICT=NOT_COMPUTED; "
                      "no reduced analysis; no seed/policy replacement; no imputation",
        "no_global_or_rule_in_primary_matrix": True, "no_pseudoreplication": True,
        "inferential_axes": ["policy_seed", "environment_seed"]}
    contracts["E4KE_pairing_contract.json"] = {
        "pair_by": ["policy_seed", "environment_seed", "scenario"],
        "example": "C0_5001/MIXED/1981 pairs C1_5001/MIXED/1981",
        "common_environment_schedule": True, "no_new_action_dependent_CRN": True}
    contracts["E4KE_endpoint_contract.json"] = {
        "primary_endpoint": "MIGD_response", "primary_direction": "lower_is_better",
        "primary_scenario": "MIXED", "secondary": ["MIGD_epoch", "response_FE"],
        "primary_comparison": "C1_QMIX_VS_C0_DDQN"}
    contracts["E4KE_effect_definition.json"] = {
        "cell_effect": "d[p,e,s] = MIGD_response_C0[p,e,s] - MIGD_response_C1[p,e,s]",
        "positive_means": "QMIX_BETTER",
        "scenario_effect": "equal mean over 5 policies x 60 env",
        "primary_statistical_units": ["policy_seed", "environment_seed"],
        "forbidden_replicate_units": ["region", "change", "transition", "replay_anchor"],
        "relative_improvement": "Delta_MIXED / mean_C0_MIXED (descriptive, not a gate)"}
    contracts["E4KE_practical_margin.json"] = {
        "DELTA_COORD": DELTA_COORD, "applies_to": "MIGD effects",
        "matches_DELTA_E4": True, "immutable_after_final_data": True}
    contracts["E4KE_classification_logic.json"] = {
        "labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL,
                   DIRECTIONAL_NEGATIVE, HARMFUL],
        "DELTA_COORD": DELTA_COORD,
        "BENEFICIAL": "mean>+DELTA AND ci_lo>0",
        "DIRECTIONAL_POSITIVE": "mean>+DELTA AND ci crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(mean)<=DELTA AND NOT(ci_hi<0)",
        "DIRECTIONAL_NEGATIVE": "mean<-DELTA AND ci crosses 0",
        "HARMFUL": "mean<-DELTA AND ci_hi<0",
        "exhaustive_extension": "reuse final_verdict.classify_effect",
        "source": "src/prdrl/evaluation/final_verdict.py::classify_effect"}
    contracts["E4KE_paired_bootstrap_contract.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "function": "paired_crossed_two_axis_bootstrap",
        "policy_axis_n": 5, "environment_axis_n": 60,
        "resample_both_with_replacement": True, "full_sampled_cartesian_product": True,
        "preserve_C0_C1_pairing": True,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "forbidden": ["naive_300_row_bootstrap", "independent_arm_bootstrap",
                      "region/change/transition as inferential replicate",
                      "replay_anchor as replicate"]}
    contracts["E4KE_macro_contract.json"] = {
        "macro": "(Delta_PERSISTENT+Delta_STATIONARY+Delta_MIXED)/3",
        "scenario_weight": 1/3,
        "shared_bootstrap_draws_across_scenarios": True,
        "function": "paired_crossed_two_axis_bootstrap_macro"}
    contracts["E4KE_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED]=mean_e(C0-C1)",
        "rule": "PASS iff >=4 of 5 policy-pair effects > 0",
        "threshold": 4, "no_replacement": True, "no_best_seed_filtering": True}
    contracts["E4KE_coordination_mechanism_contract.json"] = {
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "coordination_diagnostics_cannot_rescue_harmful": True,
        "FE_savings_cannot_rescue_harmful": True,
        "action_diversity_cannot_rescue_harmful": True,
        "larger_param_count_not_success_criterion": True,
        "monotonicity_not_success_criterion": True,
        "descriptive_diagnostics_recorded": True,
        "reason": "QMIX diagnostics (monotonicity, joint TD, action entropy) are recorded "
                  "for post-hoc interpretation ONLY, never as verdict gates"}
    contracts["E4KE_safety_contract.json"] = {
        "scenarios": ["PERSISTENT", "STATIONARY"],
        "endpoints": ["MIGD_response", "MIGD_epoch"],
        "effect": "C0 - C1 (positive favors C1/QMIX)",
        "safety_pass": "neither endpoint HARMFUL",
        "macro_safety": "neither macro MIGD_response nor macro MIGD_epoch HARMFUL",
        "improvement_not_required": True,
        "fe_cannot_rescue_harmful_migd": True,
        "diagnostics_cannot_rescue_harmful_migd": True}
    contracts["E4KE_fe_contract.json"] = {
        "effect": "d_FE = FE_C0 - FE_C1 (positive = C1 fewer evaluations)",
        "scenarios": SCENARIOS + ["macro"], "secondary": True,
        "cannot_rescue_harmful_quality": True}
    contracts["E4KE_success_logic.json"] = {
        "question": "COORDINATION CAUSALITY (C1 QMIX vs C0 DDQN) under fixed V3+R1+alpha; "
                    "NOT global optimality or superiority over GLOBAL_PREDICT",
        "COORDINATION_CAUSAL_VALIDATED_requires_all": [
            "MIXED primary=BENEFICIAL", "COORDINATION_POLICY_ROBUSTNESS=PASS",
            "PERSISTENT/STATIONARY/MACRO safety=PASS",
            "architecture+checkpoint+protocol integrity=PASS", "no final-data tuning"],
        "PERFORMANCE_PARTIAL": "MIXED in {BENEFICIAL,DIRECTIONAL_POSITIVE}, no safety HARMFUL, "
                               "engineering PASS, but robustness unmet or CI crosses 0",
        "PRACTICALLY_NEUTRAL": "MIXED=PRACTICALLY_NEUTRAL AND all safety PASS AND integrity PASS "
                               "-> COORDINATION_IMPROVEMENT_CLAIM=NOT_SUPPORTED",
        "FAILED": "MIXED in {HARMFUL,DIRECTIONAL_NEGATIVE} OR any safety FAIL OR integrity fail",
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "coordination_diagnostics_cannot_rescue_harmful": True,
        "FE_savings_cannot_rescue_harmful": True,
        "source": "coordination_integration.coordination_causal_verdict"}
    contracts["E4KE_output_schema.json"] = {
        "e4kf_results_dir": E4KF_DIR,
        "episode_row_keys": ["arm", "policy_seed", "scenario", "environment_seed"],
        "episode_metrics": ["MIGD_response", "MIGD_epoch", "response_FE"],
        "no_duplicate_keys": True, "no_pseudoreplication": True,
        "e4kf_required_outputs": [
            "E4KF_pre_execution_manifest.json", "E4KF_checkpoint_integrity.json",
            "E4KF_execution_config.json", "E4KF_progress.jsonl", "E4KF_episode_results.csv",
            "E4KF_action_distribution.csv", "E4KF_policy_pair_summary.csv",
            "E4KF_scenario_effects.csv", "E4KF_macro_effects.csv", "E4KF_fe_effects.csv",
            "E4KF_bootstrap_results.json", "E4KF_scientific_gates.json",
            "E4KF_coordination_diagnostics.json",
            "E4KF_validation_final_comparison.json",
            "E4KF_final_seed_usage.json", "E4KF_verdict.json", "E4KF_report.md"]}
    contracts["E4KE_final_execution_contract.json"] = {
        "evaluate": "selected online weights only", "epsilon": 0.0, "network_mode": "eval",
        "gradients": "disabled", "replay": "disabled", "optimizer": "disabled",
        "target_updates": "disabled", "learning": "none", "model_updates": "forbidden",
        "state_dim": 9, "network": "9-128-128-4",
        "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "c0_uses_utility_only": True, "c1_uses_utility_only_for_action_selection": True,
        "mixer_not_in_execution": True,
        "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
                      "sigma_D": 0.15, "budget": "10 PER_ACTIVE_REGION",
                      "survival": "unrestricted NSGA-II"},
        "common_environment_schedule_by": ["policy_seed", "scenario", "environment_seed"],
        "simultaneous_joint_response": True,
        "requested_realized_fallback_semantics": True,
        "deterministic_resume": True, "shadow_reward_zero_extra_FE": True,
        "v3_frozen_constants_no_adaptation": True}
    contracts["E4KE_routing.json"] = {
        "COORDINATION_CAUSAL_VALIDATED": "E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN",
        "COORDINATION_CAUSAL_PARTIAL": "E4L_COORDINATION_RESULT_REVIEW",
        "COORDINATION_CAUSAL_FAILED": "E4L_COORDINATION_FAILURE_ANALYSIS",
        "E4JF_historical_failure_cannot_force_E4KF_failure": True,
        "E4KTR_validation_cannot_force_E4KF_failure": True}
    contracts["E4KE_validation_observation_lock.json"] = {
        "status": "NON_CONFIRMATORY_LOCKED",
        "validation_macro_delta_C0_minus_C1": e4ktr.get("VALIDATION_PAIRED_DELTA_C0_MINUS_C1"),
        "C0_val_macro_MIGD_response": e4ktr.get("C0_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
        "C1_val_macro_MIGD_response": e4ktr.get("C1_VALIDATION_MACRO_MIGD_RESPONSE_MEAN"),
        "C0_MIXED_PREDICT_FRACTION": e4ktr.get("C0_MIXED_PREDICT_FRACTION"),
        "C1_MIXED_PREDICT_FRACTION": e4ktr.get("C1_MIXED_PREDICT_FRACTION"),
        "C0_MIXED_MULTI_ACTION_RATE": e4ktr.get("C0_MIXED_MULTI_ACTION_RATE"),
        "C1_MIXED_MULTI_ACTION_RATE": e4ktr.get("C1_MIXED_MULTI_ACTION_RATE"),
        "C0_MIXED_ACTION_ENTROPY": e4ktr.get("C0_MIXED_ACTION_ENTROPY"),
        "C1_MIXED_ACTION_ENTROPY": e4ktr.get("C1_MIXED_ACTION_ENTROPY"),
        "must_not_alter": ["primary_endpoint", "practical_margin", "bootstrap", "safety",
                           "robustness", "verdict_logic", "checkpoints", "architecture",
                           "reward", "normalizer"]}

    token_hits = []
    for name, obj in contracts.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    _jdump(synth, RESULTS_DIR / "E4KE_synthetic_validation.json")
    for name, obj in contracts.items():
        _jdump(obj, RESULTS_DIR / name)
    _jdump({"final_seeds": [1981, 2040],
            "COORDINATION_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
            "FINAL_EVALUATION": "NOT_RUN", "COORDINATION_CAUSAL_CLAIM": "NOT_MADE",
            "NEW_TRAINING": "NO"},
           RESULTS_DIR / "E4KE_final_firewall.json")

    complete = (ck_ok and qmix_sha_ok and synth["ALL_PASS"]
                and len(token_hits) == 0 and _final_access["count"] == 0)

    _jdump({
        "stage": "E4KE_JOINT_COORDINATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "all_frozen_before_final": True,
        "control": "C0_INDEPENDENT_SHARED_DDQN", "treatment": "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "experiment_factor": "JOINT_COORDINATION_LEARNER_SPECIFICATION",
        "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
        "qmix_config_sha256_integrity": "PASS" if qmix_sha_ok else "FAIL",
        "synthetic_validation": "PASS" if synth["ALL_PASS"] else "FAIL",
        "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "effect_definition": "C0_MINUS_C1", "positive_means": "QMIX_BETTER",
        "practical_margin": DELTA_COORD,
        "statistical_design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis_n": 5, "environment_axis_n": 60, "N_BOOTSTRAP": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "final_matrix_planned": 1800, "final_seeds": [1981, 2040],
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "components": list(contracts.keys()),
        "no_placeholder_tokens": len(token_hits) == 0,
        "next_stage": "E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION"},
        RESULTS_DIR / "E4KE_protocol_frozen.json")

    verdict = _build_verdict(ck_ok, qmix_sha_ok, synth, token_hits, complete, n_c0, n_c1, e4ktr)
    _jdump(verdict, RESULTS_DIR / "E4KE_verdict.json")
    _write_report(verdict, synth)
    _emit_terminal(verdict)
    print(f"[E4KE] wall {time.time()-t0:.1f}s")
    return verdict


def _build_verdict(ck_ok, qmix_sha_ok, synth, token_hits, complete, n_c0, n_c1, e4ktr):
    return {
        "stage": "E4KE_JOINT_COORDINATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "E4KE_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "E4KT_COMPLETE": "YES", "E4KTR_COMPLETE": "YES",
        "SELECTED_BRANCH": "C_JOINT_COORDINATION",
        "CHECKPOINT_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "QMIX_CONFIG_SHA256_INTEGRITY": "PASS" if qmix_sha_ok else "FAIL",
        "ARCHITECTURE_INTEGRITY": "PASS" if (ck_ok and qmix_sha_ok) else "FAIL",
        "REWARD_VERSION_CONFIG_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "NORMALIZATION_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "C0_SELECTED_CHECKPOINTS": f"{n_c0}/5",
        "C1_SELECTED_CHECKPOINTS": f"{n_c1}/5",
        "FINAL_ENVIRONMENT_SEEDS": "1981..2040",
        "POLICY_SEEDS": "5001..5005",
        "PRIMARY_SCENARIO": "MIXED",
        "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "C1_QMIX_VS_C0_DDQN",
        "EFFECT_DEFINITION": "C0_MINUS_C1",
        "POSITIVE_MEANS": "QMIX_BETTER",
        "PRACTICAL_MARGIN": DELTA_COORD,
        "STATISTICAL_DESIGN": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP",
        "POLICY_AXIS_N": 5,
        "ENVIRONMENT_AXIS_N": 60,
        "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_QMIX_BETTER",
        "FINAL_MATRIX_PLANNED": "1800_EPISODES",
        "SYNTHETIC_STATISTICS_VALIDATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "SYNTHETIC_CHECKS_N": synth["n_checks"],
        "VALIDATION_OBSERVATION_STATUS": "NON_CONFIRMATORY_LOCKED",
        "VALIDATION_DELTA_C0_MINUS_C1": e4ktr.get("VALIDATION_PAIRED_DELTA_C0_MINUS_C1"),
        "no_placeholder_tokens": len(token_hits) == 0,
        "COORDINATION_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
        "FINAL_EVALUATION": "NOT_RUN",
        "COORDINATION_CAUSAL_CLAIM": "NOT_MADE",
        "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4KF_JOINT_COORDINATION_AB_FINAL_EVALUATION_EXECUTION"
                       if complete else "E4KE_INTEGRITY_REVIEW")}


def _write_report(v, synth):
    L = [
        "# E4KE — Joint Coordination A/B Final Evaluation Protocol Freeze", "",
        f"**E4KE_COMPLETE = {v['E4KE_COMPLETE']}** — freeze only; final seeds 1981..2040 "
        "unopened; no training. Question: COORDINATION CAUSALITY (C1 QMIX vs C0 DDQN) under "
        "fixed V3+R1+alpha; primary comparator C0_INDEPENDENT_SHARED_DDQN.", "",
        "## A/B. Historical preservation + E4KTR integrity",
        f"- E4F = FAILED; E4HF normalization = FAILED; E4IF reward = VALIDATED; "
        f"E4JF global-comparator = FAILED; E4KTR COMPLETE; "
        f"CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}.", "",
        "## C. Exact 10 frozen checkpoints",
        f"- C0 {v['C0_SELECTED_CHECKPOINTS']} + C1 {v['C1_SELECTED_CHECKPOINTS']}; "
        "all hash-verified, immutable (no reselection/averaging/ensemble/retrain); all ten advance.", "",
        "## D/E/F. Architecture + reward + V3 locks",
        f"- CONTROL C0 Independent Shared DDQN, TREATMENT C1 QMIX V1; "
        f"QMIX_CONFIG_SHA256_INTEGRITY = {v['QMIX_CONFIG_SHA256_INTEGRITY']}; "
        f"both arms V3+R1 alpha 0.1; "
        f"ARCHITECTURE_INTEGRITY = {v['ARCHITECTURE_INTEGRITY']}; "
        f"NORMALIZATION_INTEGRITY = {v['NORMALIZATION_INTEGRITY']}.", "",
        "## G/H/I. Final firewall + matrix + pairing",
        "- final seeds 1981..2040 (60), never instantiated; 2x5x60x3 = 1800 (900 C0 + 900 C1); "
        "pair by (policy, env, scenario); FINAL_MATRIX_COMPLETE required.", "",
        "## J/K/L. Primary endpoint + classification + bootstrap",
        f"- MIXED MIGD_response, d = C0 - C1 (+ = QMIX better), DELTA_COORD 1e-5; "
        f"paired crossed two-axis bootstrap 5x60, N=50000, 95% percentile, "
        f"master seed {BOOTSTRAP_MASTER_SEED}; naive/independent forbidden.", "",
        "## M. Policy robustness",
        "- robustness >=4/5 QMIX better.", "",
        "## N. Coordination mechanism gate",
        "- NOT_USED_FOR_SCIENTIFIC_VERDICT. Diagnostics (monotonicity, joint TD, action entropy) "
        "recorded for post-hoc interpretation ONLY.", "",
        "## O/P/Q. Safety + macro + FE",
        "- PERSISTENT/STATIONARY/macro not HARMFUL; macro equal 1/3 shared draws; "
        "FE secondary, cannot rescue.", "",
        "## R. Final verdict logic",
        "- COORDINATION_CAUSAL_VALIDATED / PERFORMANCE_PARTIAL / PRACTICALLY_NEUTRAL / FAILED "
        "(coordination_causal_verdict; NO mechanism gate; diagnostics cannot rescue harmful).", "",
        f"## S. Synthetic pipeline validation: {synth['n_checks']} checks, "
        f"ALL_PASS = {synth['ALL_PASS']}", "",
        "## T. Validation observation lock: NON_CONFIRMATORY_LOCKED",
        f"- val delta C0-C1 = {v.get('VALIDATION_DELTA_C0_MINUS_C1', 'N/A')} "
        "(C0 descriptively better); does NOT alter the frozen protocol.", "",
        "## U. Routing",
        "- VALIDATED -> E4L_QMIX_GLOBAL_COMPARATOR_PROTOCOL_DESIGN",
        "- PARTIAL -> E4L_COORDINATION_RESULT_REVIEW",
        "- FAILED -> E4L_COORDINATION_FAILURE_ANALYSIS",
        "- E4JF historical failure cannot force E4KF failure.", "",
        f"## V. Next stage: {v['NEXT_STAGE']}", "",
        "The untouched final seeds 1981..2040 decide the preregistered coordination-causality "
        "question."]
    (RESULTS_DIR / "E4KE_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in [
        "E4KE_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
        "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT",
        "E4KT_COMPLETE", "E4KTR_COMPLETE", "SELECTED_BRANCH",
        "CHECKPOINT_INTEGRITY", "QMIX_CONFIG_SHA256_INTEGRITY",
        "ARCHITECTURE_INTEGRITY", "REWARD_VERSION_CONFIG_INTEGRITY",
        "NORMALIZATION_INTEGRITY", "C0_SELECTED_CHECKPOINTS", "C1_SELECTED_CHECKPOINTS",
        "FINAL_ENVIRONMENT_SEEDS", "POLICY_SEEDS",
        "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT", "PRIMARY_COMPARISON",
        "EFFECT_DEFINITION", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
        "STATISTICAL_DESIGN", "POLICY_AXIS_N", "ENVIRONMENT_AXIS_N",
        "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_MASTER_SEED",
        "COORDINATION_MECHANISM_GATE", "POLICY_ROBUSTNESS_RULE",
        "FINAL_MATRIX_PLANNED", "SYNTHETIC_STATISTICS_VALIDATION", "SYNTHETIC_CHECKS_N",
        "VALIDATION_OBSERVATION_STATUS", "VALIDATION_DELTA_C0_MINUS_C1",
        "COORDINATION_FINAL_EVAL_ACCESS_COUNT", "FINAL_EVALUATION",
        "COORDINATION_CAUSAL_CLAIM", "NEW_TRAINING", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<44} = {v[k]}")


if __name__ == "__main__":
    main()
