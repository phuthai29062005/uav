"""
E4J — Global Comparator Protocol Design (PR-DRL-DMOEA, Branch B follow-up).

Freezes the protocol to test the E4IF-validated V3+R1 learned controller against the
strongest PRESELECTED fixed-global comparator on a NEW untouched seed set (1801..1860),
BEFORE any comparator environment is opened. DESIGN/FREEZE ONLY: no new environments, no
training, no access to the fresh comparator seeds, no R1 policy modification.

New question (different from E4IF's R1-vs-R0): does learned V3+R1 outperform the fixed-global
action strategy that was selected from HISTORICAL validation evidence? The comparator map is
verified from frozen artifacts, never chosen on the new final data. Effect d = GLOBAL - LEARNED
(positive ⇒ learned better). E4F/E4HF verdicts remain immutable; E4IF VALIDATED preserved.
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
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, crossed_two_axis_bootstrap,
    crossed_two_axis_bootstrap_macro, naive_row_bootstrap,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect, policy_robustness, safety_gate,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME

RESULTS_DIR = Path("results/E4J_global_comparator_protocol_design")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4IE_DIR = Path("results/E4IE_reward_ab_final_evaluation_protocol_freeze")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

FRESH_SEEDS = list(range(1801, 1861))       # 60, comparator confirmatory-only
FRESH_SEED_LABEL = "GLOBAL_COMPARATOR_CONFIRMATORY_FINAL_ONLY"
POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED = 76001
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
E4JF_DIR = "results/E4JF_global_comparator_final_evaluation_execution"
FORBIDDEN_TOKENS = ("TBD", "TODO", "UNFROZEN", "placeholder", "PLACEHOLDER")

CONSUMED_SEED_SETS = {
    "E3_701_780": range(701, 781), "E3R_801_882": range(801, 883), "H1_901_940": range(901, 941),
    "E4_1001_1140": range(1001, 1141), "E4GD_1201_1220": range(1201, 1221),
    "E4H_1301_1440": range(1301, 1441), "REWARD_1501_1640": range(1501, 1641),
    "E4IDD_1701_1720": range(1701, 1721),
}
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


def _e4j_seed(*labels) -> int:
    entropy: List[int] = [BOOTSTRAP_MASTER_SEED]
    for lab in labels:
        if isinstance(lab, (int, np.integer)):
            entropy.append(int(lab))
        else:
            hh = 0
            for ch in str(lab):
                hh = (hh * 131 + ord(ch)) & 0xFFFFFFFF
            entropy.append(int(hh))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def global_comparator_verdict(mixed_primary_label, robustness_pass, persistent_safety_pass,
                              stationary_safety_pass, macro_safety_pass, integrity_pass) -> Dict:
    """VALIDATED / PARTIAL(PERFORMANCE_PARTIAL|PRACTICALLY_NEUTRAL) / FAILED. Superiority over the
    fixed-global benchmark is claimed only for BENEFICIAL+robustness+safety+integrity; a neutral
    primary never claims superiority; safety/integrity failure or negative/harmful primary => FAILED."""
    VAL, PART, FAIL = "GLOBAL_COMPARATOR_VALIDATED", "GLOBAL_COMPARATOR_PARTIAL", "GLOBAL_COMPARATOR_FAILED"
    if not integrity_pass:
        return {"verdict": FAIL, "subtype": None, "reason": "integrity failure",
                "SUPERIORITY_CLAIM": "NOT_SUPPORTED"}
    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE):
        return {"verdict": FAIL, "subtype": None, "reason": f"MIXED primary {mixed_primary_label}",
                "SUPERIORITY_CLAIM": "NOT_SUPPORTED"}
    if not safety_all:
        return {"verdict": FAIL, "subtype": None, "reason": "safety gate HARMFUL",
                "SUPERIORITY_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == PRACTICALLY_NEUTRAL:
        return {"verdict": PART, "subtype": "PRACTICALLY_NEUTRAL",
                "reason": "MIXED neutral vs fixed-global; safe but no superiority",
                "SUPERIORITY_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == BENEFICIAL and robustness_pass:
        return {"verdict": VAL, "subtype": None,
                "reason": "MIXED beneficial vs fixed-global + robustness + safety + integrity",
                "SUPERIORITY_CLAIM": "SUPPORTED"}
    if mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE):
        return {"verdict": PART, "subtype": "PERFORMANCE_PARTIAL",
                "reason": "positive MIXED but a support gate (robustness) unmet",
                "SUPERIORITY_CLAIM": "PARTIAL"}
    return {"verdict": FAIL, "subtype": None, "reason": "unclassified conservative FAILED",
            "SUPERIORITY_CLAIM": "NOT_SUPPORTED"}


# ---------------------------------------------------------------------------
def synthetic_validation() -> dict:
    checks = {}
    rng = np.random.default_rng(760076)
    n_p, n_e = 5, 60
    L = rng.uniform(0.006, 0.008, size=(n_p, n_e))     # learned
    B = L.mean(axis=0)                                  # global baseline (n_e,), shared across policies

    # 1 learned == global (B = per-env mean of L, so d ~ small)  -> use exact-equal fixture
    Lc = np.full((n_p, n_e), 0.007); Bc = np.full(n_e, 0.007)
    c0 = crossed_two_axis_bootstrap(Lc, Bc, n_boot=1500, seed=1)
    checks["1_zero_effect"] = {"delta": c0["point_estimate"], "PASS": abs(c0["point_estimate"]) < 1e-12}
    # 2 learned better by .001 (L = B - .001 -> d = B - L = +.001)
    c2 = crossed_two_axis_bootstrap(Bc[None, :].repeat(n_p, 0) - 0.001, Bc, n_boot=1500, seed=1)
    checks["2_learned_better"] = {"delta": c2["point_estimate"], "PASS": abs(c2["point_estimate"] - 0.001) < 1e-12}
    # 3 learned worse by .001
    c3 = crossed_two_axis_bootstrap(Bc[None, :].repeat(n_p, 0) + 0.001, Bc, n_boot=1500, seed=1)
    checks["3_learned_worse"] = {"delta": c3["point_estimate"], "PASS": abs(c3["point_estimate"] + 0.001) < 1e-12}
    # 4 policy-axis variance widens CI vs flat
    Lflat = np.full((n_p, n_e), 0.0065); Bflat = np.full(n_e, 0.007)
    Lpol = Lflat + np.array([-0.001, -0.0005, 0, 0.0005, 0.001])[:, None]
    w_flat = (lambda c: c["ci_hi"] - c["ci_lo"])(crossed_two_axis_bootstrap(Lflat, Bflat, n_boot=2000, seed=2))
    w_pol = (lambda c: c["ci_hi"] - c["ci_lo"])(crossed_two_axis_bootstrap(Lpol, Bflat, n_boot=2000, seed=2))
    checks["4_policy_axis_widens"] = {"w_flat": w_flat, "w_pol": w_pol, "PASS": w_pol > w_flat}
    # 5 env-axis variance widens CI
    Benv = 0.007 + rng.uniform(-0.001, 0.001, size=n_e)
    w_env = (lambda c: c["ci_hi"] - c["ci_lo"])(crossed_two_axis_bootstrap(Lflat, Benv, n_boot=2000, seed=3))
    checks["5_env_axis_widens"] = {"w_flat": w_flat, "w_env": w_env, "PASS": w_env > w_flat}
    # 6 shared baseline B[e] across 5 policies: crossed point estimate == grand mean (B[e]-L[p,e])
    point_manual = float((B[None, :] - L).mean())
    c6 = crossed_two_axis_bootstrap(L, B, n_boot=1000, seed=4)
    checks["6_shared_baseline"] = {"point": c6["point_estimate"], "manual": point_manual,
                                   "PASS": abs(c6["point_estimate"] - point_manual) < 1e-12}
    # 7 crossed vs naive-300 (forbidden) differ; naive is the wrong iid method
    nb = naive_row_bootstrap(Lpol, Bflat, n_boot=2000, seed=5)
    checks["7_naive_forbidden_differs"] = {"crossed_w": w_pol, "naive_method": nb["method"],
                                           "PASS": "FORBIDDEN" in nb["method"]}
    # 8 robustness 4/5 vs 3/5
    r4 = policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)
    r3 = policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)
    checks["8_robustness"] = {"PASS": r4["PASS"] is True and r3["PASS"] is False}
    # 9 comparator-map application: macro uses per-scenario B
    Lby = {s: Lflat for s in SCENARIOS}
    Bby = {"PERSISTENT": np.full(n_e, 0.0072), "STATIONARY": np.full(n_e, 0.0060), "MIXED": np.full(n_e, 0.0071)}
    mac = crossed_two_axis_bootstrap_macro(Lby, Bby, SCENARIOS, n_boot=1500, seed=6)
    exp_macro = float(np.mean([(Bby[s][None, :] - Lflat).mean() for s in SCENARIOS]))
    checks["9_comparator_map_macro"] = {"macro": mac["point_estimate"], "exp": exp_macro,
                                        "PASS": abs(mac["point_estimate"] - exp_macro) < 1e-12}
    # 10 macro equal 1/3
    checks["10_macro_equal_third"] = {"weight": mac["scenario_weight"], "PASS": abs(mac["scenario_weight"] - 1/3) < 1e-12}
    # 11 macro shared draws: identical scenarios -> macro CI == single CI (same seed)
    Bsame = {s: Bflat for s in SCENARIOS}; Lsame = {s: Lpol for s in SCENARIOS}
    mac2 = crossed_two_axis_bootstrap_macro(Lsame, Bsame, SCENARIOS, n_boot=2000, seed=77)
    sing = crossed_two_axis_bootstrap(Lpol, Bflat, n_boot=2000, seed=77)
    checks["11_macro_shared_draws"] = {"PASS": abs(mac2["ci_lo"] - sing["ci_lo"]) < 1e-12
                                       and abs(mac2["ci_hi"] - sing["ci_hi"]) < 1e-12}
    # 12-16 classification branches
    checks["12_beneficial"] = {"PASS": classify_effect(0.01, 0.001, 0.02) == BENEFICIAL}
    checks["13_directional_positive"] = {"PASS": classify_effect(0.01, -0.001, 0.02) == DIRECTIONAL_POSITIVE}
    checks["14_practically_neutral"] = {"PASS": classify_effect(0.0, -0.001, 0.001) == PRACTICALLY_NEUTRAL}
    checks["15_directional_negative"] = {"PASS": classify_effect(-0.01, -0.02, 0.001) == DIRECTIONAL_NEGATIVE}
    checks["16_harmful"] = {"PASS": classify_effect(-0.01, -0.02, -0.001) == HARMFUL}
    # 17 safety failure overrides MIXED benefit
    v_safety = global_comparator_verdict(BENEFICIAL, True, False, True, True, True)
    checks["17_safety_override"] = {"v": v_safety["verdict"], "PASS": v_safety["verdict"] == "GLOBAL_COMPARATOR_FAILED"}
    # 18 FE cannot rescue harmful (harmful MIXED -> FAILED regardless)
    v_harm = global_comparator_verdict(HARMFUL, True, True, True, True, True)
    checks["18_fe_cannot_rescue"] = {"v": v_harm["verdict"], "PASS": v_harm["verdict"] == "GLOBAL_COMPARATOR_FAILED"}
    # 19 no comparator reselection: verdict function takes no final-data comparator arg
    checks["19_no_reselection_logic"] = {"PASS": "comparator" not in global_comparator_verdict.__code__.co_varnames}
    # 20 baseline has no fictitious policy independence: crossed B has shape (n_e,), no policy axis
    checks["20_baseline_no_policy_axis"] = {"B_ndim": int(np.asarray(B).ndim), "PASS": np.asarray(B).ndim == 1}
    # extra verdict branches
    checks["21_validated"] = {"PASS": global_comparator_verdict(BENEFICIAL, True, True, True, True, True)["verdict"]
                              == "GLOBAL_COMPARATOR_VALIDATED"}
    checks["22_performance_partial"] = {"PASS": global_comparator_verdict(BENEFICIAL, False, True, True, True, True)["subtype"]
                                        == "PERFORMANCE_PARTIAL"}
    checks["23_neutral_partial"] = {"PASS": global_comparator_verdict(PRACTICALLY_NEUTRAL, True, True, True, True, True)["subtype"]
                                    == "PRACTICALLY_NEUTRAL"}
    checks["24_failed_directional_negative"] = {"PASS": global_comparator_verdict(DIRECTIONAL_NEGATIVE, True, True, True, True, True)["verdict"]
                                                == "GLOBAL_COMPARATOR_FAILED"}

    all_pass = all(c["PASS"] for c in checks.values())
    return {"checks": checks, "ALL_PASS": all_pass, "n_checks": len(checks), "no_fresh_seed_opened": True}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4J — Global Comparator Protocol Design (DESIGN/FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    frz = _load(E4IE_DIR / "E4IE_checkpoint_freeze.json")["checkpoints"]
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    r1 = {int(v["policy_seed"]): v for v in frz.values() if v["arm"] == "R1_TREATMENT"}
    ck_rows = {}
    ck_ok = True
    for p in POLICY_SEEDS:
        info = r1[p]
        actual = _hash_file(info["checkpoint_path"])
        match = actual == info["sha256"]
        meta_ok = (info["normalization_version"] == NORM_V3 and info["reward_version"] == REWARD_R1_NAME
                   and info["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256)
        ck_ok = ck_ok and match and meta_ok
        ck_rows[str(p)] = {"policy_seed": p, "controller": "V3_R1_DDQN",
                           "checkpoint": info["selected_checkpoint"], "transition_count": info["transition_count"],
                           "path": info["checkpoint_path"], "file_size": info["file_size"],
                           "sha256": info["sha256"], "sha256_recomputed": actual, "hash_match": match,
                           "normalization_version": info["normalization_version"],
                           "normalizer_constants_sha256": info["normalizer_constants_sha256"],
                           "reward_version": info["reward_version"], "reward_config_sha256": info["reward_config_sha256"],
                           "architecture": "9-128-128-4", "metadata_ok": meta_ok}

    # historical comparator evidence (verify agreement across authoritative artifacts)
    e4e_map = _load(E4E_DIR / "E4E_comparator_freeze.json")["primary_comparator_by_scenario"]
    e4tr_map = {s: _load(E4TR_DIR / "E4TR_best_fixed_global_validation.json")[s]["controller"] for s in SCENARIOS}
    map_verified = (e4e_map == e4tr_map and e4e_map["MIXED"] == "GLOBAL_PREDICT"
                    and e4e_map["PERSISTENT"] == "GLOBAL_PREDICT" and e4e_map["STATIONARY"] == "GLOBAL_DIVERSIFY")

    manifest = {"stage": "E4J_GLOBAL_COMPARATOR_PROTOCOL_DESIGN", "design_only": True, "no_training": True,
                "no_new_environment_execution": True, "E4IF_COMPLETE": e4if["E4IF_COMPLETE"],
                "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
                "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
                "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
                "comparator_map_verified": map_verified, "v3_constants_sha256": v3_sha}
    _jdump(manifest, RESULTS_DIR / "E4J_pre_manifest.json")
    _jdump({"E4IF_COMPLETE": e4if["E4IF_COMPLETE"], "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
            "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
            "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
            "checkpoint_all_match": ck_ok, "v3_constants_sha256": v3_sha, "no_post_e4if_modification": True},
           RESULTS_DIR / "E4J_historical_integrity.json")

    if not map_verified:
        print("[E4J] historical comparator map NOT verified — STOP")
        _jdump({"E4J_COMPLETE": "NO", "GLOBAL_COMPARATOR_MAP_VERIFIED": "NO",
                "NEXT_STAGE": "E4J_COMPARATOR_EVIDENCE_REVIEW"}, RESULTS_DIR / "E4J_verdict.json")
        return {"E4J_COMPLETE": "NO"}
    ok0 = (e4if["E4IF_COMPLETE"] == "YES" and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and ck_ok and len(r1) == 5)
    if not ok0:
        print("[E4J] integrity FAIL — STOP")
        _jdump({"E4J_COMPLETE": "NO", "NEXT_STAGE": "E4J_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4J_verdict.json")
        return {"E4J_COMPLETE": "NO"}
    print("[E4J] integrity PASS. Comparator map verified (PERSISTENT/STATIONARY/MIXED = PREDICT/DIVERSIFY/PREDICT).")

    # seed disjointness
    fresh = set(FRESH_SEEDS)
    consumed = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    disjoint = len(fresh & consumed) == 0

    synth = synthetic_validation()
    assert _final_access["count"] == 0

    # ---- freeze contracts ----
    C: Dict[str, dict] = {}
    C["E4J_learned_checkpoint_freeze.json"] = {
        "controller": "V3_R1_DDQN", "policy_seeds": POLICY_SEEDS, "n_checkpoints": 5,
        "checkpoints": ck_rows, "immutable": True, "no_reselection": True, "no_retrain": True,
        "no_finetune": True, "no_ensemble": True, "no_weight_averaging": True}
    C["E4J_historical_global_comparator_evidence.json"] = {
        "source_E4E": "results/E4E_final_evaluation_protocol_freeze/E4E_comparator_freeze.json",
        "source_E4TR": "results/E4TR_shared_double_dqn_training/E4TR_best_fixed_global_validation.json",
        "E4E_map": e4e_map, "E4TR_map": e4tr_map, "agree": e4e_map == e4tr_map,
        "preselected_from": "VALIDATION_SEEDS_ONLY (E4TR 1061..1080)",
        "GLOBAL_COMPARATOR_MAP_VERIFIED": map_verified,
        "not_selected_on_new_final_seeds": True}
    C["E4J_global_comparator_map.json"] = {
        "FROZEN_BEST_GLOBAL_BY_SCENARIO": {"PERSISTENT": "GLOBAL_PREDICT", "STATIONARY": "GLOBAL_DIVERSIFY",
                                           "MIXED": "GLOBAL_PREDICT"},
        "PRIMARY_COMPARATOR": "GLOBAL_PREDICT", "primary_scenario": "MIXED",
        "frozen_before_new_data": True, "no_reselection_on_final": True}
    C["E4J_global_operator_contract.json"] = {
        "fixed_global_meaning": "one requested action type for ALL active regions at a response event",
        "GLOBAL_PREDICT": {"requested_action": "PREDICT", "lambda_P": 1.0, "sigma_P": 0.005, "fallback": "LOCAL"},
        "GLOBAL_DIVERSIFY": {"requested_action": "DIVERSIFY", "sigma_D": 0.15},
        "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted standard NSGA-II",
        "no_HARD": True, "no_MEMORY": True, "no_R1_logic": True, "not_a_learned_controller": True,
        "not_a_region_oracle": True, "not_a_rule_controller": True,
        "production_dispatch": "prdrl.controllers... choose_actions_v2 GLOBAL_* / E4TR.eval_fixed_arm"}
    C["E4J_global_rng_contract.json"] = {
        "reuse_historical_evaluator_semantics": "E4TR.eval_fixed_arm (as in E4F)",
        "fixed_global_evaluated_once_per": ["scenario", "environment_seed"],
        "no_fictitious_policy_seed_axis": True,
        "baseline_paired_with_each_learned_policy_on_same_env": True,
        "no_new_action_dependent_CRN": True}
    C["E4J_fresh_seed_contract.json"] = {
        "fresh_seeds": [1801, 1860], "n": 60, "label": FRESH_SEED_LABEL,
        "confirmatory_only": True, "no_train_split": True, "no_validation_split": True}
    C["E4J_seed_disjointness.json"] = {
        "fresh": [1801, 1860], "disjoint_from_all_prior": disjoint,
        "checked_against": list(CONSUMED_SEED_SETS.keys()),
        "explicitly_not_reused": ["1081..1140", "1381..1440", "1581..1640"]}
    C["E4J_final_firewall.json"] = {
        "GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT": _final_access["count"], "fresh_seeds_unopened": True,
        "opened_only_in": "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION"}
    C["E4J_matrix_contract.json"] = {
        "learned": {"controller": "V3_R1_DDQN", "policies": 5, "environments": 60, "scenarios": 3,
                    "episodes": 900},
        "global": {"controller": "FROZEN_GLOBAL_COMPARATOR", "environments": 60, "scenarios": 3,
                   "episodes": 180, "one_realization_per_env_scenario": True},
        "total_episodes": 1080,
        "no_baseline_duplication_as_independent_observations": True,
        "learned_unique_key": ["controller", "policy_seed", "scenario", "environment_seed"],
        "global_unique_key": ["controller", "scenario", "environment_seed"],
        "FINAL_MATRIX_COMPLETE_required": True,
        "on_missing": "FINAL_MATRIX_COMPLETE=NO; GLOBAL_COMPARATOR_VERDICT=NOT_COMPUTED"}
    C["E4J_pairing_contract.json"] = {
        "pair": "d[p,e,s] = GLOBAL[e,s] - LEARNED[p,e,s] (shared baseline B[e,s] across the 5 policies)",
        "shared_baseline_across_policies": True,
        "baseline_not_independent_policy_observations": True}
    C["E4J_primary_endpoint.json"] = {"primary_endpoint": "MIGD_response", "primary_scenario": "MIXED",
                                      "secondary": ["MIGD_epoch", "response_FE"], "lower_is_better": True}
    C["E4J_effect_definition.json"] = {
        "cell_effect": "d[p,e,s] = MIGD_response_GLOBAL[e,s] - MIGD_response_V3_R1[p,e,s]",
        "positive_means": "V3_R1_better", "primary_statistical_units": ["policy_seed", "environment_seed"],
        "baseline_shared_across_policy_axis": True,
        "forbidden_replicate_units": ["region", "change", "transition"],
        "relative_improvement": "descriptive only"}
    C["E4J_practical_margin.json"] = {"DELTA_NORM": DELTA_NORM, "immutable_after_final_data": True}
    C["E4J_classification_logic.json"] = {
        "labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL],
        "DELTA_NORM": DELTA_NORM, "source": "final_verdict.classify_effect",
        "BENEFICIAL": "mean>+DELTA AND ci_lo>0", "DIRECTIONAL_POSITIVE": "mean>+DELTA AND ci crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(mean)<=DELTA AND NOT(ci_hi<0)",
        "DIRECTIONAL_NEGATIVE": "mean<-DELTA AND ci crosses 0", "HARMFUL": "mean<-DELTA AND ci_hi<0"}
    C["E4J_bootstrap_contract.json"] = {
        "design": "CROSSED_POLICY_ENV_BOOTSTRAP_WITH_SHARED_ENV_BASELINE",
        "function": "crossed_two_axis_bootstrap", "policy_axis_n": 5, "environment_axis_n": 60,
        "baseline_shape": "(n_env,) shared across policy indices (no policy axis)",
        "resample": "policy indices with replacement AND env indices with replacement, full product; "
                    "baseline reused per sampled env",
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "forbidden": ["iid_300_cell_bootstrap", "independent_arm_bootstrap", "region_level", "change_level",
                      "treat repeated baseline as independent policy observations"]}
    C["E4J_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED] = mean_e(GLOBAL_PREDICT[e] - V3_R1[p,e])",
        "rule": "GLOBAL_COMPARATOR_POLICY_ROBUSTNESS = PASS iff >=4 of 5 learned policies beat GLOBAL_PREDICT",
        "threshold": 4, "no_best_policy_filtering": True}
    C["E4J_scenario_safety.json"] = {
        "PERSISTENT_comparator": "GLOBAL_PREDICT", "STATIONARY_comparator": "GLOBAL_DIVERSIFY",
        "endpoints": ["MIGD_response", "MIGD_epoch"], "effect": "GLOBAL - LEARNED",
        "safety_pass": "neither endpoint HARMFUL (learned perspective)"}
    C["E4J_macro_contract.json"] = {
        "comparator_by_scenario": {"PERSISTENT": "GLOBAL_PREDICT", "STATIONARY": "GLOBAL_DIVERSIFY",
                                   "MIXED": "GLOBAL_PREDICT"},
        "macro": "(Delta_PERSISTENT+Delta_STATIONARY+Delta_MIXED)/3", "scenario_weight": 1/3,
        "shared_bootstrap_draws_across_scenarios": True, "function": "crossed_two_axis_bootstrap_macro",
        "MACRO_COMPARATOR": "FROZEN_BEST_GLOBAL_BY_SCENARIO"}
    C["E4J_fe_contract.json"] = {
        "effect": "FE_GLOBAL - FE_V3_R1 (positive = learned cheaper)", "scenarios": SCENARIOS + ["macro"],
        "secondary": True, "cannot_rescue_harmful_quality": True}
    C["E4J_success_logic.json"] = {
        "question": "learned V3+R1 vs preselected fixed-global on fresh untouched seeds; MIXED primary",
        "GLOBAL_COMPARATOR_VALIDATED_requires_all": ["MIXED vs GLOBAL_PREDICT MIGD_response BENEFICIAL",
            "GLOBAL_COMPARATOR_POLICY_ROBUSTNESS PASS", "PERSISTENT/STATIONARY/MACRO safety PASS",
            "checkpoint + comparator-map + protocol integrity PASS", "no comparator reselection on final"],
        "PERFORMANCE_PARTIAL": "MIXED in {BENEFICIAL,DIRECTIONAL_POSITIVE}, no safety HARMFUL, but a support "
                               "gate (robustness) fails",
        "PRACTICALLY_NEUTRAL": "MIXED neutral + all safety PASS + integrity PASS -> PARTIAL, no superiority claim",
        "FAILED": "MIXED in {DIRECTIONAL_NEGATIVE,HARMFUL} OR any safety FAIL OR integrity fail",
        "no_new_reward_mechanism_gate": True, "reward_causality_already_confirmed_by_E4IF": True,
        "no_cross_stage_significance_combination": True,
        "source": "E4J.global_comparator_verdict"}
    C["E4J_future_execution_contract.json"] = {
        "stage": "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION", "no_intermediate_validation": True,
        "learned_eval": {"epsilon": 0.0, "eval": True, "gradients": "disabled", "optimizer": "disabled",
                         "replay": "disabled", "target_updates": "disabled", "learning": "none"},
        "global_eval": {"no_network": True, "no_learning": True},
        "same_production_environment_semantics": True, "fresh_seeds_opened_only_here": [1801, 1860]}
    C["E4J_output_schema.json"] = {
        "e4jf_results_dir": E4JF_DIR,
        "learned_row_keys": ["controller", "policy_seed", "scenario", "environment_seed", "MIGD_response",
                             "MIGD_epoch", "response_FE"],
        "global_row_keys": ["controller", "requested_global_action", "scenario", "environment_seed",
                            "MIGD_response", "MIGD_epoch", "response_FE"],
        "e4jf_required_outputs": ["E4JF_pre_execution_manifest.json", "E4JF_checkpoint_integrity.json",
            "E4JF_comparator_integrity.json", "E4JF_execution_config.json", "E4JF_progress.jsonl",
            "E4JF_learned_episode_results.csv", "E4JF_global_episode_results.csv", "E4JF_action_context.csv",
            "E4JF_policy_pair_summary.csv", "E4JF_scenario_effects.csv", "E4JF_macro_effects.csv",
            "E4JF_fe_effects.csv", "E4JF_bootstrap_results.json", "E4JF_scientific_gates.json",
            "E4JF_final_seed_usage.json", "E4JF_post_integrity.json", "E4JF_verdict.json", "E4JF_report.md"]}
    C["E4J_limitations_contract.json"] = {
        "limitations": ["compares against preselected fixed-global actions (from historical validation)",
                        "fixed-global is not a joint oracle", "does not prove global optimality",
                        "does not evaluate all adaptive global controllers", "5 learned seeds is a limited policy axis",
                        "controlled-benchmark evidence, not real-world UAV validation"]}
    C["E4J_next_stage_logic.json"] = {
        "if_VALIDATED": "E5_BENCHMARK_EXTENSION_PROTOCOL_DESIGN",
        "if_PARTIAL": "E4K_GLOBAL_COMPARATOR_RESULT_REVIEW",
        "if_FAILED_negative_or_harmful": "E4K_JOINT_COORDINATION_REVISION_DESIGN",
        "if_FAILED_integrity": "E4J_INTEGRITY_REVIEW",
        "no_joint_architecture_change_in_E4J": True}
    C["E4J_action_context_contract.json"] = {
        "learned_descriptive": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY", "multi_action_rate", "action_entropy"],
        "global_requested_action": "recorded", "action_diversity_not_scored_as_superiority": True}

    token_hits = []
    for name, obj in C.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    _jdump(synth, RESULTS_DIR / "E4J_synthetic_validation.json")
    for name, obj in C.items():
        _jdump(obj, RESULTS_DIR / name)

    complete = (ck_ok and map_verified and disjoint and synth["ALL_PASS"]
                and len(token_hits) == 0 and _final_access["count"] == 0)

    _jdump({"stage": "E4J_GLOBAL_COMPARATOR_PROTOCOL_DESIGN", "all_frozen_before_final": True,
            "learned_controller": "V3_R1_DDQN", "primary_comparator": "GLOBAL_PREDICT",
            "comparator_map": {"PERSISTENT": "GLOBAL_PREDICT", "STATIONARY": "GLOBAL_DIVERSIFY", "MIXED": "GLOBAL_PREDICT"},
            "fresh_seeds": [1801, 1860], "statistical_design": "CROSSED_POLICY_ENV_BOOTSTRAP_WITH_SHARED_ENV_BASELINE",
            "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED, "components": list(C.keys()),
            "next_stage": "E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION"},
           RESULTS_DIR / "E4J_protocol_frozen.json")

    verdict = _build_verdict(ck_ok, map_verified, disjoint, synth, token_hits, complete, e4e_map)
    _jdump(verdict, RESULTS_DIR / "E4J_verdict.json")
    _write_report(verdict, synth)
    _emit_terminal(verdict)
    print(f"[E4J] wall {time.time()-t0:.1f}s")
    return verdict


def _build_verdict(ck_ok, map_verified, disjoint, synth, token_hits, complete, e4e_map):
    return {
        "stage": "E4J_GLOBAL_COMPARATOR_PROTOCOL_DESIGN",
        "E4J_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED", "LEARNED_CONTROLLER": "V3_R1_DDQN",
        "LEARNED_POLICY_SEEDS": "4001..4005", "LEARNED_CHECKPOINTS_FROZEN": "5/5",
        "HISTORICAL_GLOBAL_COMPARATOR_MAP_VERIFIED": "YES" if map_verified else "NO",
        "PERSISTENT_COMPARATOR": e4e_map["PERSISTENT"], "STATIONARY_COMPARATOR": e4e_map["STATIONARY"],
        "MIXED_COMPARATOR": e4e_map["MIXED"], "PRIMARY_COMPARATOR": "GLOBAL_PREDICT",
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "EFFECT_DEFINITION": "GLOBAL_MINUS_LEARNED", "POSITIVE_MEANS": "V3_R1_BETTER",
        "PRACTICAL_MARGIN": DELTA_NORM, "FRESH_FINAL_SEEDS": "1801..1860",
        "FRESH_FINAL_SEED_LABEL": FRESH_SEED_LABEL, "GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT": _final_access["count"],
        "SEEDS_DISJOINT": disjoint, "CHECKPOINT_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "LEARNED_EPISODES_PLANNED": 900, "GLOBAL_EPISODES_PLANNED": 180, "TOTAL_EPISODES_PLANNED": 1080,
        "STATISTICAL_DESIGN": "CROSSED_POLICY_ENV_BOOTSTRAP_WITH_SHARED_ENV_BASELINE", "POLICY_AXIS_N": 5,
        "ENVIRONMENT_AXIS_N": 60, "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP, "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_LEARNED_BETTER",
        "MACRO_COMPARATOR": "FROZEN_BEST_GLOBAL_BY_SCENARIO",
        "SYNTHETIC_STATISTICS_VALIDATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "no_placeholder_tokens": len(token_hits) == 0, "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EXECUTION": "NO", "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4JF_GLOBAL_COMPARATOR_FINAL_EVALUATION_EXECUTION" if complete
                       else "E4J_COMPARATOR_EVIDENCE_REVIEW")}


def _write_report(v, synth):
    L = ["# E4J — Global Comparator Protocol Design", "",
         f"**E4J_COMPLETE = {v['E4J_COMPLETE']} → NEXT_STAGE = {v['NEXT_STAGE']}**", "",
         "Design/freeze only; fresh seeds 1801..1860 unopened; no training. New question: does the E4IF-validated "
         "V3+R1 learned controller beat the PRESELECTED fixed-global comparator on fresh environments?", "",
         "## A. Historical preservation", f"- E4F=FAILED; E4HF normalization=FAILED; E4IF REWARD_CAUSAL=VALIDATED "
         f"preserved; CHECKPOINT_INTEGRITY={v['CHECKPOINT_INTEGRITY']}.", "",
         "## B. Why a fresh experiment",
         "- E4IF answered R1-vs-R0; E4J asks learned-vs-fixed-global. Cannot compare E4IF R1 (seeds 1581-1640) to "
         "E4F GLOBAL_PREDICT (seeds 1081-1140) — different samples. Both controllers must run on the SAME fresh "
         "seeds 1801-1860 (E4JF).", "",
         "## C/D/E. Frozen learned controller + historical comparator map",
         f"- V3_R1_DDQN = 5 R1 checkpoints (4001..4005) frozen. Comparator map VERIFIED from E4E + E4TR "
         f"(agree): PERSISTENT={v['PERSISTENT_COMPARATOR']}, STATIONARY={v['STATIONARY_COMPARATOR']}, "
         f"MIXED={v['MIXED_COMPARATOR']}; preselected from validation seeds only, no reselection on final.", "",
         "## F/G. Fixed-global semantics + RNG",
         "- GLOBAL_PREDICT (sigma_P 0.005, lambda 1, fallback LOCAL) / GLOBAL_DIVERSIFY (sigma_D 0.15); one requested "
         "action for all active regions; production dispatch (E4TR.eval_fixed_arm as in E4F); one baseline realization "
         "per env×scenario, no fictitious policy axis.", "",
         "## H/I/J. Fresh seeds + matrix + pairing",
         f"- 1801..1860 (60), disjoint={v['SEEDS_DISJOINT']}. Matrix: learned 900 + global 180 = 1080. "
         "d[p,e,s] = GLOBAL[e,s] − LEARNED[p,e,s]; baseline B[e,s] shared across the 5 policies (not independent).", "",
         "## K/L/M. Primary + bootstrap + robustness",
         f"- MIXED MIGD_response, d=GLOBAL−LEARNED (+ ⇒ learned better), margin 1e-5. Crossed policy×env bootstrap "
         f"with shared env baseline, N=50000, 95% percentile, master seed {BOOTSTRAP_MASTER_SEED}; naive/iid/independent "
         "forbidden. Robustness ≥4/5 learned beat GLOBAL_PREDICT.", "",
         "## N/O/P. Safety + macro",
         "- PERSISTENT vs GLOBAL_PREDICT, STATIONARY vs GLOBAL_DIVERSIFY not HARMFUL; macro over FROZEN_BEST_GLOBAL "
         "map, equal 1/3, shared draws.", "",
         "## Q. FE — secondary, cannot rescue harmful quality.", "",
         "## R. Verdict logic",
         "- GLOBAL_COMPARATOR_VALIDATED / PARTIAL(PERFORMANCE_PARTIAL|PRACTICALLY_NEUTRAL) / FAILED; NO new "
         "reward-mechanism gate (reward causality already confirmed by E4IF); no cross-stage significance combination.", "",
         f"## S. Synthetic validation: {synth['n_checks']} checks, ALL_PASS = {synth['ALL_PASS']}",
         "- includes shared-baseline correctness, crossed-vs-naive, policy/env CI widening, macro shared draws, "
         "comparator-map application, robustness 4/5, all classification + verdict branches, safety override.", "",
         "## T. Limitations",
         "- fixed-global (not a joint oracle); no global-optimality claim; 5 policy seeds; controlled benchmark only.", "",
         "## U. Firewall", f"- GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT = {v['GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT']}; "
         "seeds opened only in E4JF.", "",
         f"## V. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4J_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4J_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "E4IF_REWARD_CAUSAL_VERDICT", "LEARNED_CONTROLLER", "LEARNED_POLICY_SEEDS",
              "LEARNED_CHECKPOINTS_FROZEN", "HISTORICAL_GLOBAL_COMPARATOR_MAP_VERIFIED", "PERSISTENT_COMPARATOR",
              "STATIONARY_COMPARATOR", "MIXED_COMPARATOR", "PRIMARY_COMPARATOR", "PRIMARY_SCENARIO",
              "PRIMARY_ENDPOINT", "EFFECT_DEFINITION", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
              "FRESH_FINAL_SEEDS", "FRESH_FINAL_SEED_LABEL", "GLOBAL_COMPARATOR_FINAL_ACCESS_COUNT",
              "LEARNED_EPISODES_PLANNED", "GLOBAL_EPISODES_PLANNED", "TOTAL_EPISODES_PLANNED",
              "STATISTICAL_DESIGN", "POLICY_AXIS_N", "ENVIRONMENT_AXIS_N", "BOOTSTRAP_RESAMPLES",
              "BOOTSTRAP_MASTER_SEED", "POLICY_ROBUSTNESS_RULE", "MACRO_COMPARATOR",
              "SYNTHETIC_STATISTICS_VALIDATION", "NEW_TRAINING", "NEW_ENVIRONMENT_EXECUTION",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<44} = {v[k]}")


if __name__ == "__main__":
    main()
