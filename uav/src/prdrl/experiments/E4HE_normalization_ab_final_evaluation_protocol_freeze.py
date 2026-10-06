"""
E4HE — Normalization A/B Final Evaluation Protocol Freeze (PR-DRL-DMOEA, Branch A).

Freezes the COMPLETE final causal protocol for the V1-vs-V3 experiment BEFORE the
reserved final seeds 1381..1440 are opened, and validates the entire analysis
pipeline on SYNTHETIC matrices only.

FREEZE ONLY: does not instantiate final environments, does not execute final
evaluation, does not train, does not reselect checkpoints, and does not change the
immutable E4F verdict (FAILED). The causal comparator is V1_CONTROL (not GLOBAL_PREDICT).
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
    DELTA_E4, classify_effect, mechanism_decollapse_supported, normalization_ab_verdict,
    policy_robustness, safety_gate, BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL,
    DIRECTIONAL_NEGATIVE, HARMFUL,
)
from prdrl.rl.normalization import NORM_V1, NORM_V3

RESULTS_DIR = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HT_DIR = Path("results/E4HT_normalization_ab_training_protocol_freeze")
E4HTR_DIR = Path("results/E4HTR_normalization_ab_training_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")

FINAL_SEEDS = list(range(1381, 1441))
POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED = 56001
E4HF_DIR = "results/E4HF_normalization_ab_final_evaluation_execution"
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
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(f"not serializable: {type(x)}")


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
# SYNTHETIC validation of the final analysis pipeline (no final data)
# ---------------------------------------------------------------------------

def synthetic_validation() -> dict:
    checks = {}
    rng = np.random.default_rng(202609)
    n_p, n_e = 5, 60
    base = rng.uniform(0.005, 0.009, size=(n_p, n_e))   # synthetic V1 (paired baseline)

    # 1 identical
    D0 = base - base
    checks["1_identical_zero"] = {"delta": float(D0.mean()), "PASS": abs(D0.mean()) < 1e-12}
    # 2 V3 better by 0.001 (V3 = V1 - 0.001 -> D = V1 - V3 = +0.001)
    D2 = base - (base - 0.001)
    checks["2_v3_better_const"] = {"delta": float(D2.mean()), "PASS": abs(D2.mean() - 0.001) < 1e-12}
    # 3 V3 worse by 0.001
    D3 = base - (base + 0.001)
    checks["3_v3_worse_const"] = {"delta": float(D3.mean()), "PASS": abs(D3.mean() + 0.001) < 1e-12}
    # 4 policy-axis variation -> CI widens vs no variation
    D_flat = np.full((n_p, n_e), 0.0005)
    D_pol = D_flat + np.array([-0.001, -0.0005, 0, 0.0005, 0.001])[:, None]
    c_flat = paired_crossed_two_axis_bootstrap(D_flat, n_boot=3000, seed=1)
    c_pol = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=1)
    w_flat = c_flat["ci_hi"] - c_flat["ci_lo"]; w_pol = c_pol["ci_hi"] - c_pol["ci_lo"]
    checks["4_policy_axis_widens_ci"] = {"w_flat": w_flat, "w_pol": w_pol, "PASS": w_pol > w_flat}
    # 5 env-axis variation -> CI widens
    D_env = D_flat + rng.uniform(-0.001, 0.001, size=(1, n_e))
    c_env = paired_crossed_two_axis_bootstrap(D_env, n_boot=3000, seed=2)
    w_env = c_env["ci_hi"] - c_env["ci_lo"]
    checks["5_env_axis_widens_ci"] = {"w_flat": w_flat, "w_env": w_env, "PASS": w_env > w_flat}
    # 6 paired common-component cancellation: D depends only on treatment diff, not shared offset
    common = rng.uniform(0, 1, size=(n_p, n_e))
    treat = 0.001
    V1_6 = common + rng.uniform(0.004, 0.008, (n_p, n_e))
    V3_6 = V1_6 - treat
    D6 = V1_6 - V3_6
    checks["6_pairing_cancels_common"] = {"delta": float(D6.mean()),
                                          "PASS": abs(D6.mean() - treat) < 1e-9}
    # 7 paired narrower than independent on shared-variance fixture
    shared = rng.uniform(0, 0.02, size=(n_p, n_e))   # large shared variance
    V1_7 = shared + 0.006
    V3_7 = shared + 0.006 - 0.0003   # small consistent treatment
    D7 = V1_7 - V3_7
    paired7 = paired_crossed_two_axis_bootstrap(D7, n_boot=4000, seed=3)
    indep7 = independent_arm_bootstrap(V1_7, V3_7, n_boot=4000, seed=3)
    w_paired = paired7["ci_hi"] - paired7["ci_lo"]; w_indep = indep7["ci_hi"] - indep7["ci_lo"]
    checks["7_paired_narrower_than_independent"] = {"w_paired": w_paired, "w_indep": w_indep,
                                                    "PASS": w_paired < w_indep}
    # 8 macro equal 1/3
    Dm = {"PERSISTENT": np.full((n_p, n_e), 0.001), "STATIONARY": np.full((n_p, n_e), 0.002),
          "MIXED": np.full((n_p, n_e), 0.003)}
    mac = paired_crossed_two_axis_bootstrap_macro(Dm, SCENARIOS, n_boot=2000, seed=4)
    checks["8_macro_equal_third"] = {"macro": mac["point_estimate"],
                                     "PASS": abs(mac["point_estimate"] - 0.002) < 1e-12}
    # 9 shared draws across scenarios: identical scenarios -> macro CI == single CI (same seed)
    Dsame = {s: D_pol for s in SCENARIOS}
    mac2 = paired_crossed_two_axis_bootstrap_macro(Dsame, SCENARIOS, n_boot=3000, seed=99)
    sing = paired_crossed_two_axis_bootstrap(D_pol, n_boot=3000, seed=99)
    checks["9_macro_shared_draws"] = {
        "macro_ci": [mac2["ci_lo"], mac2["ci_hi"]], "single_ci": [sing["ci_lo"], sing["ci_hi"]],
        "PASS": abs(mac2["ci_lo"] - sing["ci_lo"]) < 1e-12 and abs(mac2["ci_hi"] - sing["ci_hi"]) < 1e-12}
    # 10 policy robustness 4/5 PASS, 3/5 FAIL
    r4 = policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)
    r3 = policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)
    checks["10_policy_robustness"] = {"pass4": r4["PASS"], "fail3": r3["PASS"],
                                      "PASS": r4["PASS"] is True and r3["PASS"] is False}
    # 11 mechanism gate 3/3 YES, 2/3 YES, 1/3 NO
    m33 = mechanism_decollapse_supported(0.1, 0.1, 0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"]
    m23 = mechanism_decollapse_supported(0.1, 0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"]
    m13 = mechanism_decollapse_supported(0.1, -0.1, -0.1)["MECHANISM_DECOLLAPSE_SUPPORTED"]
    checks["11_mechanism_gate"] = {"m33": m33, "m23": m23, "m13": m13,
                                   "PASS": m33 == "YES" and m23 == "YES" and m13 == "NO"}
    # 12-18 verdict logic
    def V(mp, mech, rob, ps=True, ss=True, ms=True, eng=True):
        return normalization_ab_verdict(
            mixed_primary_label=mp, mechanism_supported=mech, policy_robustness_pass=rob,
            persistent_safety_pass=ps, stationary_safety_pass=ss, macro_safety_pass=ms,
            checkpoint_integrity_pass=eng, protocol_integrity_pass=eng,
            version_integrity_pass=eng, no_final_tuning=eng)
    v_val = V(BENEFICIAL, True, True)
    v_perf = V(BENEFICIAL, False, True)                 # beneficial but no mechanism -> PERFORMANCE_PARTIAL
    v_mech = V(PRACTICALLY_NEUTRAL, True, False)        # neutral + mechanism -> MECHANISTIC_PARTIAL
    v_dirneg = V(DIRECTIONAL_NEGATIVE, True, True)
    v_safety = V(BENEFICIAL, True, True, ps=False)
    v_neutral_nomech = V(PRACTICALLY_NEUTRAL, False, True)
    v_harmful = V(HARMFUL, True, True)                  # mechanism cannot rescue harmful
    checks["12_validated"] = {"v": v_val["verdict"], "PASS": v_val["verdict"] == "NORMALIZATION_CAUSAL_VALIDATED"}
    checks["13_performance_partial"] = {"v": v_perf["subtype"], "PASS": v_perf["subtype"] == "PERFORMANCE_PARTIAL"}
    checks["14_mechanistic_partial"] = {"v": v_mech["subtype"], "PASS": v_mech["subtype"] == "MECHANISTIC_PARTIAL"
                                        and v_mech["PERFORMANCE_IMPROVEMENT_CLAIM"] == "NOT_SUPPORTED"}
    checks["15_failed_directional_negative"] = {"v": v_dirneg["verdict"],
                                                "PASS": v_dirneg["verdict"] == "NORMALIZATION_CAUSAL_FAILED"}
    checks["16_failed_safety"] = {"v": v_safety["verdict"], "PASS": v_safety["verdict"] == "NORMALIZATION_CAUSAL_FAILED"}
    checks["17_neutral_no_mechanism_failed"] = {"v": v_neutral_nomech["verdict"],
                                                "PASS": v_neutral_nomech["verdict"] == "NORMALIZATION_CAUSAL_FAILED"}
    checks["18_harmful_not_rescued"] = {"v": v_harmful["verdict"],
                                        "PASS": v_harmful["verdict"] == "NORMALIZATION_CAUSAL_FAILED"}

    all_pass = all(c["PASS"] for c in checks.values())
    return {"checks": checks, "ALL_PASS": all_pass, "n_checks": len(checks),
            "no_final_seed_instantiated": True}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4HE — Normalization A/B Final Evaluation Protocol Freeze")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4ht = _load(E4HT_DIR / "E4HT_verdict.json")
    e4htr = _load(E4HTR_DIR / "E4HTR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    sel = _load(E4HTR_DIR / "E4HTR_selected_checkpoints.json")["selected"]
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    ck_ok = True
    ck_rows = {}
    import torch
    for key, info in sel.items():
        actual = _hash_file(info["checkpoint_path"])
        exists = Path(info["checkpoint_path"]).exists()
        match = exists and actual == info["sha256"]
        # normalization metadata guard
        payload = torch.load(info["checkpoint_path"], weights_only=False) if exists else {}
        nv = payload.get("normalization_version")
        nv_ok = (nv == NORM_V1 if info["arm"] == "V1_CONTROL" else nv == NORM_V3)
        if info["arm"] == "V3_TREATMENT":
            nv_ok = nv_ok and payload.get("normalizer_constants_sha256") == v3_sha
        ck_ok = ck_ok and match and nv_ok
        ck_rows[key] = {**{k: info[k] for k in ("arm", "policy_seed", "selected_checkpoint",
                        "transition_count", "checkpoint_path", "sha256", "file_size",
                        "normalization_version", "normalizer_constants_sha256")},
                        "sha256_recomputed": actual, "hash_match": match,
                        "normalization_metadata_ok": nv_ok,
                        "validation_macro_MIGD_response": info["validation_macro_MIGD_response"],
                        "validation_macro_MIGD_epoch": info["validation_macro_MIGD_epoch"],
                        "validation_mean_response_fe": info["validation_mean_response_fe"]}

    manifest = {
        "stage": "E4HE_NORMALIZATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE", "freeze_only": True,
        "E4HT_COMPLETE": e4ht["E4HT_COMPLETE"], "E4HTR_COMPLETE": e4htr["E4HTR_COMPLETE"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "n_selected_checkpoints": len(sel), "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
        "no_best_policy_selection": True, "v3_constants_sha256": v3_sha,
    }
    _jdump(manifest, RESULTS_DIR / "E4HE_pre_freeze_manifest.json")
    ok0 = (e4ht["E4HT_COMPLETE"] == "YES" and e4htr["E4HTR_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and len(sel) == 10 and ck_ok)
    if not ok0:
        print("[E4HE] integrity FAIL — STOP"); _emit_stop(); return {"E4HE_COMPLETE": "NO"}
    print(f"[E4HE] integrity PASS. 10 checkpoints verified. E4F FAILED preserved.")

    # ---- synthetic validation ----
    synth = synthetic_validation()
    assert _final_access["count"] == 0

    # ---- freeze contracts ----
    contracts = {}
    contracts["E4HE_checkpoint_freeze.json"] = {
        "immutable_for_final_evaluation": True, "n_checkpoints": 10, "checkpoints": ck_rows,
        "no_reselection": True, "no_alt_checkpoint_on_final": True, "no_weight_averaging": True,
        "no_ensemble": True, "no_additional_training": True}
    contracts["E4HE_normalization_integrity.json"] = {
        "V1_CONTROL": NORM_V1, "V3_TREATMENT": NORM_V3, "v3_constants_sha256": v3_sha,
        "v3_immutable": True, "v3_no_refit_no_adapt": True}
    contracts["E4HE_final_seed_contract.json"] = {
        "final_seeds": FINAL_SEEDS, "range": [1381, 1440], "n": 60, "never_instantiated": True,
        "NEW_FINAL_EVAL_ACCESS_COUNT": _final_access["count"]}
    contracts["E4HE_execution_order.json"] = {
        "scenario_order": SCENARIOS, "environment_seed_order": "1381..1440 ascending",
        "policy_seed_order": POLICY_SEEDS, "within_pair": ["V1_CONTROL", "V3_TREATMENT"],
        "outcome_independent": True, "no_early_stop": True}
    contracts["E4HE_final_matrix_contract.json"] = {
        "arms": ["V1_CONTROL", "V3_TREATMENT"], "policy_seeds": POLICY_SEEDS,
        "environment_seeds": [1381, 1440], "scenarios": SCENARIOS,
        "total_episodes": 2 * 5 * 60 * 3, "V1_rows": 900, "V3_rows": 900,
        "pairing_key": ["policy_seed", "environment_seed", "scenario"],
        "FINAL_MATRIX_COMPLETE_required": True,
        "on_missing": "FINAL_MATRIX_COMPLETE=NO; FINAL_CAUSAL_VERDICT=NOT_COMPUTED; no reduced analysis; "
                      "no seed/policy replacement; no imputation",
        "no_global_or_rule_in_primary_matrix": True}
    contracts["E4HE_endpoint_contract.json"] = {
        "primary_endpoint": "MIGD_response", "primary_direction": "lower_is_better",
        "primary_scenario": "MIXED", "secondary": ["MIGD_epoch", "response_FE"]}
    contracts["E4HE_effect_definition.json"] = {
        "cell_effect": "d[p,e,s] = MIGD_response_V1[p,e,s] - MIGD_response_V3[p,e,s]",
        "positive_means": "V3 better", "scenario_effect": "equal mean over 5 policies x 60 env",
        "primary_statistical_units": ["policy_seed", "environment_seed"],
        "forbidden_replicate_units": ["region", "change", "transition"]}
    contracts["E4HE_paired_bootstrap_contract.json"] = {
        "design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "function": "paired_crossed_two_axis_bootstrap",
        "policy_axis_n": 5, "environment_axis_n": 60, "resample_both_with_replacement": True,
        "full_sampled_cartesian_product": True, "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL,
        "CI_METHOD": CI_METHOD, "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "forbidden": ["naive_300_row_bootstrap", "independent_arm_bootstrap",
                      "treat 300 cells as independent"]}
    contracts["E4HE_macro_contract.json"] = {
        "macro": "(Delta_PERSISTENT+Delta_STATIONARY+Delta_MIXED)/3", "scenario_weight": 1/3,
        "shared_bootstrap_draws_across_scenarios": True,
        "function": "paired_crossed_two_axis_bootstrap_macro"}
    contracts["E4HE_practical_margin.json"] = {"DELTA_NORM": DELTA_NORM, "applies_to": "MIGD effects",
                                               "immutable_after_final_data": True}
    contracts["E4HE_policy_robustness.json"] = {
        "definition": "Delta_policy[p,MIXED]=mean_e(V1-V3)",
        "rule": "PASS iff >=4 of 5 policy-pair effects > 0", "threshold": 4,
        "no_replacement": True, "no_subgroup_selection": True}
    contracts["E4HE_mechanism_contract.json"] = {
        "endpoints": ["PREDICT_fraction", "within_event_multi_action_rate", "regional_action_entropy"],
        "scenario": "MIXED",
        "delta_predict": "PREDICT_frac_V1 - PREDICT_frac_V3 (>0 expected)",
        "delta_multi": "multi_action_V3 - multi_action_V1 (>0 expected)",
        "delta_entropy": "entropy_V3 - entropy_V1 (>0 expected)",
        "gate": "MECHANISM_DECOLLAPSE_SUPPORTED = YES iff >=2 of 3 point estimates in expected direction",
        "uses_requested_action": True, "predict_fallback_semantics": "requested=PREDICT, realized=LOCAL",
        "entropy_definition": "Shannon natural-log over requested-action fractions of active regions "
                              "(exact E4HT/E4HTR definition; base/normalization unchanged)"}
    contracts["E4HE_safety_contract.json"] = {
        "scenarios": ["PERSISTENT", "STATIONARY"], "endpoints": ["MIGD_response", "MIGD_epoch"],
        "effect": "V1 - V3 (positive favors V3)", "safety_pass": "neither endpoint HARMFUL",
        "macro_safety": "neither macro MIGD_response nor macro MIGD_epoch HARMFUL",
        "improvement_not_required": True, "fe_cannot_rescue_harmful_migd": True}
    contracts["E4HE_fe_contract.json"] = {
        "effect": "d_FE = FE_V1 - FE_V3 (positive = V3 fewer evaluations)",
        "scenarios": SCENARIOS + ["macro"], "secondary": True, "cannot_rescue_harmful_quality": True}
    contracts["E4HE_classification_logic.json"] = {
        "labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL],
        "DELTA_NORM": DELTA_NORM,
        "BENEFICIAL": "mean>+DELTA AND ci_lo>0", "DIRECTIONAL_POSITIVE": "mean>+DELTA AND ci crosses 0",
        "PRACTICALLY_NEUTRAL": "abs(mean)<=DELTA AND NOT(ci_hi<0)",
        "DIRECTIONAL_NEGATIVE": "mean<-DELTA AND ci crosses 0", "HARMFUL": "mean<-DELTA AND ci_hi<0",
        "exhaustive_extension": "reuse E4E/final_verdict.classify_effect (defer to CI direction for "
                                "uncovered combos; documented before final data)",
        "source": "src/prdrl/evaluation/final_verdict.py::classify_effect"}
    contracts["E4HE_success_logic.json"] = {
        "question": "NORMALIZATION CAUSALITY (V3 vs V1); NOT overall PR-DRL superiority; NOT vs GLOBAL_PREDICT",
        "VALIDATED_requires_all": ["MIXED primary=BENEFICIAL", "MECHANISM_DECOLLAPSE_SUPPORTED=YES",
                                   "NORMALIZATION_POLICY_ROBUSTNESS=PASS", "PERSISTENT/STATIONARY/MACRO safety=PASS",
                                   "checkpoint+protocol+version integrity=PASS", "no final-data tuning"],
        "PERFORMANCE_PARTIAL": "MIXED in {BENEFICIAL,DIRECTIONAL_POSITIVE}, no safety HARMFUL, engineering PASS, "
                               "but a supporting gate (mechanism/robustness) unmet",
        "MECHANISTIC_PARTIAL": "MIXED=PRACTICALLY_NEUTRAL AND mechanism supported AND all safety PASS AND "
                               "engineering PASS -> records mechanism change, PERFORMANCE_IMPROVEMENT_CLAIM=NOT_SUPPORTED",
        "FAILED": "MIXED in {HARMFUL,DIRECTIONAL_NEGATIVE} OR any safety FAIL OR integrity fail OR "
                  "(MIXED neutral AND mechanism NOT supported)",
        "mechanism_cannot_rescue_harmful": True,
        "beneficial_without_mechanism_is_partial_not_validated": True,
        "source": "src/prdrl/evaluation/final_verdict.py::normalization_ab_verdict"}
    contracts["E4HE_output_schema.json"] = {
        "e4hf_results_dir": E4HF_DIR,
        "episode_row_keys": ["arm", "policy_seed", "scenario", "environment_seed"],
        "episode_metrics": ["MIGD_response", "MIGD_epoch", "response_FE"],
        "change_level_diagnostics": ["arm", "policy_seed", "scenario", "environment_seed", "change_index",
                                     "active_region_count", "requested_actions", "realized_operators",
                                     "fallback_flags", "event_multi_action_indicator",
                                     "regional_action_entropy", "MIGD_response_change",
                                     "MIGD_epoch_change", "response_FE_change"],
        "no_duplicate_keys": True, "no_pseudoreplication": True}
    contracts["E4HE_final_execution_contract.json"] = {
        "evaluate": "selected online weights only", "epsilon": 0.0, "network_mode": "eval",
        "gradients": "disabled", "replay": "disabled", "optimizer": "disabled",
        "target_updates": "disabled", "learning": "none", "checkpoint_modification": "forbidden",
        "state_dim": 9, "network": "9-128-128-4", "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0, "sigma_D": 0.15,
                      "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted NSGA-II"},
        "common_environment_schedule_by": ["policy_seed", "scenario", "environment_seed"],
        "production_rng_unchanged": True, "v1_legacy_normalization": True,
        "v3_frozen_constants_no_adaptation": True}

    # placeholder scan
    token_hits = []
    for name, obj in contracts.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    _jdump(synth, RESULTS_DIR / "E4HE_synthetic_validation.json")
    for name, obj in contracts.items():
        _jdump(obj, RESULTS_DIR / name)
    _jdump({"final_seeds": [1381, 1440], "NEW_FINAL_EVAL_ACCESS_COUNT": _final_access["count"],
            "FINAL_EVALUATION": "NOT_RUN", "NORMALIZATION_CAUSAL_CLAIM": "NOT_MADE"},
           RESULTS_DIR / "E4HE_final_firewall.json")

    e4he_complete = (ck_ok and synth["ALL_PASS"] and len(token_hits) == 0
                     and _final_access["count"] == 0)

    master = {
        "stage": "E4HE_NORMALIZATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "all_frozen_before_final": True, "control": NORM_V1, "treatment": NORM_V3,
        "checkpoint_integrity": "PASS" if ck_ok else "FAIL",
        "synthetic_validation": "PASS" if synth["ALL_PASS"] else "FAIL",
        "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "effect_definition": "V1_MINUS_V3", "practical_margin": DELTA_NORM,
        "statistical_design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "policy_axis_n": 5,
        "environment_axis_n": 60, "N_BOOTSTRAP": N_BOOTSTRAP, "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "final_matrix_planned": 1800, "final_seeds": [1381, 1440], "policy_seeds": POLICY_SEEDS,
        "no_placeholder_tokens": len(token_hits) == 0, "components": list(contracts.keys()),
        "next_stage": "E4HF_NORMALIZATION_AB_FINAL_EVALUATION_EXECUTION"}
    _jdump(master, RESULTS_DIR / "E4HE_protocol_frozen.json")

    verdict = _build_verdict(ck_ok, synth, token_hits, e4he_complete)
    _jdump(verdict, RESULTS_DIR / "E4HE_verdict.json")
    _write_report(verdict, synth)
    _emit_terminal(verdict, synth)
    print(f"[E4HE] wall {time.time()-t0:.0f}s")
    return verdict


def _build_verdict(ck_ok, synth, token_hits, complete):
    return {
        "stage": "E4HE_NORMALIZATION_AB_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "E4HE_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HTR_COMPLETE": "YES",
        "CHECKPOINT_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "NORMALIZATION_VERSION_INTEGRITY": "PASS" if ck_ok else "FAIL",
        "V1_SELECTED_CHECKPOINTS": "5/5", "V3_SELECTED_CHECKPOINTS": "5/5",
        "FINAL_ENVIRONMENT_SEEDS": "1381..1440", "POLICY_SEEDS": "3001..3005",
        "PRIMARY_SCENARIO": "MIXED", "PRIMARY_ENDPOINT": "MIGD_RESPONSE",
        "PRIMARY_COMPARISON": "V3_TREATMENT_VS_V1_CONTROL", "EFFECT_DEFINITION": "V1_MINUS_V3",
        "POSITIVE_MEANS": "V3_BETTER", "PRACTICAL_MARGIN": DELTA_NORM,
        "STATISTICAL_DESIGN": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "POLICY_AXIS_N": 5,
        "ENVIRONMENT_AXIS_N": 60, "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "MECHANISM_ENDPOINTS": "PREDICT_FRACTION,MULTI_ACTION_RATE,ACTION_ENTROPY",
        "MECHANISM_GATE": "AT_LEAST_2_OF_3_EXPECTED_DIRECTIONS",
        "POLICY_ROBUSTNESS_RULE": "AT_LEAST_4_OF_5_V3_BETTER",
        "FINAL_MATRIX_PLANNED": "1800_EPISODES",
        "SYNTHETIC_STATISTICS_VALIDATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "no_placeholder_tokens": len(token_hits) == 0,
        "NEW_FINAL_EVAL_ACCESS_COUNT": _final_access["count"], "FINAL_EVALUATION": "NOT_RUN",
        "NORMALIZATION_CAUSAL_CLAIM": "NOT_MADE", "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4HF_NORMALIZATION_AB_FINAL_EVALUATION_EXECUTION" if complete
                       else "E4HE_PROTOCOL_REVIEW")}


def _write_report(v, synth):
    L = ["# E4HE — Normalization A/B Final Evaluation Protocol Freeze", "",
         f"**E4HE_COMPLETE = {v['E4HE_COMPLETE']}** — freeze only; final seeds 1381..1440 unopened; "
         "no training. Question: NORMALIZATION CAUSALITY (V3 vs V1), NOT superiority over GLOBAL_PREDICT.", "",
         "## Frozen decisions",
         f"- 10 checkpoints frozen & verified (CHECKPOINT_INTEGRITY = {v['CHECKPOINT_INTEGRITY']}, "
         f"NORMALIZATION_VERSION_INTEGRITY = {v['NORMALIZATION_VERSION_INTEGRITY']}).",
         "- Primary: MIXED MIGD_response, effect d = V1 − V3 (positive ⇒ V3 better), DELTA_NORM = 1e-5.",
         "- PAIRED crossed two-axis bootstrap (policy 5 × env 60), N=50000, 95% percentile, master seed 56001.",
         "- Mechanism gate: ≥2/3 of {PREDICT↓, multi-action↑, entropy↑}. Policy robustness ≥4/5 pairs V3 better.",
         "- Safety: PERSISTENT/STATIONARY/macro not HARMFUL; FE cannot rescue harmful quality.",
         "- Success: VALIDATED / PERFORMANCE_PARTIAL / MECHANISTIC_PARTIAL / FAILED (mechanism cannot rescue "
         "harmful; neutral+mechanism = MECHANISTIC_PARTIAL, not VALIDATED).", "",
         f"## Synthetic pipeline validation: {synth['n_checks']} checks, ALL_PASS = {synth['ALL_PASS']}",
         "- zero/±0.001 effects, policy/env-axis CI widening, paired common-component cancellation, "
         "paired narrower than independent-arm, macro 1/3 with shared draws, robustness 4/5 vs 3/5, "
         "mechanism 3/2/1-of-3, and VALIDATED / PERFORMANCE_PARTIAL / MECHANISTIC_PARTIAL / FAILED "
         "(directional-negative, safety, neutral-without-mechanism, harmful-not-rescued).", "",
         f"## Status: FINAL_EVALUATION = NOT_RUN; NORMALIZATION_CAUSAL_CLAIM = NOT_MADE; "
         f"NEW_FINAL_EVAL_ACCESS_COUNT = {v['NEW_FINAL_EVAL_ACCESS_COUNT']}.",
         f"## Next stage: {v['NEXT_STAGE']}", "",
         "Validation showed V3 strongly de-collapses but is not better on macro MIGD; the protocol is NOT "
         "changed in response — the untouched final seeds 1381..1440 decide the preregistered causal question."]
    (RESULTS_DIR / "E4HE_report.md").write_text("\n".join(L))


def _emit_stop():
    print("\nE4HE_COMPLETE = NO\nNEXT_STAGE = E4HE_INTEGRITY_REVIEW")


def _emit_terminal(v, synth):
    print()
    for k in ["E4HE_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HTR_COMPLETE", "CHECKPOINT_INTEGRITY",
              "NORMALIZATION_VERSION_INTEGRITY", "V1_SELECTED_CHECKPOINTS", "V3_SELECTED_CHECKPOINTS",
              "FINAL_ENVIRONMENT_SEEDS", "POLICY_SEEDS", "PRIMARY_SCENARIO", "PRIMARY_ENDPOINT",
              "PRIMARY_COMPARISON", "EFFECT_DEFINITION", "POSITIVE_MEANS", "PRACTICAL_MARGIN",
              "STATISTICAL_DESIGN", "POLICY_AXIS_N", "ENVIRONMENT_AXIS_N", "BOOTSTRAP_RESAMPLES",
              "BOOTSTRAP_MASTER_SEED", "MECHANISM_ENDPOINTS", "MECHANISM_GATE", "POLICY_ROBUSTNESS_RULE",
              "FINAL_MATRIX_PLANNED", "SYNTHETIC_STATISTICS_VALIDATION", "NEW_FINAL_EVAL_ACCESS_COUNT",
              "FINAL_EVALUATION", "NORMALIZATION_CAUSAL_CLAIM", "NEW_TRAINING", "FULL_REGRESSION",
              "NEXT_STAGE"]:
        print(f"{k:<34} = {v[k]}")


if __name__ == "__main__":
    main()
