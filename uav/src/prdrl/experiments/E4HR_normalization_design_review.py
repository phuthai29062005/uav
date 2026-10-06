"""
E4HR — Zero-Aware Normalization Design Review (PR-DRL-DMOEA, Branch A continued).

E4H's ROBUST_QUANTILE_NORMALIZATION_V2 FAILED the saturation gate because
zero-inflated nonnegative features (~33-50% structural zeros) piled the zero mass
onto the lower bound. E4H remains FAIL and V2 remains REJECTED (never trained).

This DESIGN/REVIEW/FREEZE stage:
  - preserves E4F=FAILED, E4H=NO, V2=REJECTED (no retroactive pass),
  - audits zero inflation on TRAIN-only states (seeds 1001..1060),
  - freezes a NEW candidate ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3 (structural zero -> 0;
    positive values -> robust log spread) and its constants,
  - freezes ZERO-AWARE saturation gates (structural zero != clipping) BEFORE validation,
  - freezes the V1-vs-V3 A/B supersession + fresh seeds.

It does NOT train, does NOT run the V3 gate validation (that is E4HI), does NOT change
state semantics/dim/network/reward/optimizer, and does NOT open new final seeds.
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

from prdrl.experiments.E4H_state_normalization_revision_design import (
    make_env_context, load_nets, collect_raw_states,
    LOCAL_FEATURES, GLOBAL_FEATURES, FEATURE_NAMES,
    NEW_TRAIN_SEEDS, NEW_VAL_SEEDS, NEW_FINAL_SEEDS, NEW_POLICY_SEEDS,
    NORM_FIT_SEEDS, NEAR_CONST_RANGE, NEAR_CONST_STD, COMPRESSED_LOCAL,
)
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, NORM_V3, ZeroAwareRobustLogNormalizerV3, normalize_v1,
    RAW_ORDER, ZERO_INFLATED_DIMS, DELTA_DIM, MIN_POSITIVE_SAMPLES, EPS,
)

RESULTS_DIR = Path("results/E4HR_normalization_design_review")
E4F_DIR = Path("results/E4F_final_evaluation")
E4H_DIR = Path("results/E4H_state_normalization_revision_design")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]


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
    print("E4HR — Zero-Aware Normalization Design Review")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- preservation / integrity ----
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4h = _load(E4H_DIR / "E4H_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    old_ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                    == ck["policies"][str(ps)]["sha256"] for ps in [2001, 2002, 2003, 2004, 2005])
    preservation = {
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4H_ORIGINAL_RESULT": e4h["E4H_COMPLETE"],
        "E4H_NORMALIZATION_V2_READY": e4h["NORMALIZATION_V2_READY"],
        "E4H_SATURATION_GATE": e4h["SATURATION_GATE"],
        "V2_STATUS": "REJECTED_NOT_TRAINED",
        "v2_kept_for_traceability": "ROBUST_QUANTILE_NORMALIZATION_V2 (DESIGN_REJECTED_BY_E4H_SATURATION_GATE)",
        "no_retroactive_v2_pass": True,
        "correct_statement": "E4H exposed a mismatch between the V2 transform and the saturation "
                             "diagnostic for zero-inflated features; E4H remains FAIL.",
        "old_checkpoints_unchanged": old_ck_ok,
    }
    _jdump(preservation, RESULTS_DIR / "E4HR_e4h_failure_preservation.json")

    manifest = {
        "stage": "E4HR_NORMALIZATION_DESIGN_REVIEW", "design_review_freeze_only": True,
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"], "E4H_COMPLETE": e4h["E4H_COMPLETE"],
        "branch": "A_STATE_NORMALIZATION", "single_factor": "normalization_version (V1 vs V3)",
        "no_training": True, "no_new_final_seed": True, "state_dim": 9,
        "e4f_verdict_hash": _hash_file(E4F_DIR / "E4F_verdict.json"),
        "e4h_verdict_hash": _hash_file(E4H_DIR / "E4H_verdict.json"),
    }
    _jdump(manifest, RESULTS_DIR / "E4HR_pre_manifest.json")
    if not (e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4h["E4H_COMPLETE"] == "NO" and old_ck_ok):
        print("[E4HR] integrity FAIL — STOP"); return {"E4HR_COMPLETE": "NO"}
    print("[E4HR] integrity PASS. E4F=FAILED, E4H=NO, V2 rejected, old artifacts immutable.")

    # ---- reconstruct TRAIN-only fit states (1001..1060) ----
    env = make_env_context()
    nets = load_nets()
    print("[E4HR] reconstructing TRAIN-only fit states (seeds 1001..1060)...")
    X_fit, _ = collect_raw_states(nets, env, NORM_FIT_SEEDS, SCENARIOS)
    print(f"[E4HR] fit states: {X_fit.shape[0]}")
    np.save(RESULTS_DIR / "E4HR_fit_states.npy", X_fit)   # for deterministic reuse in E4HI

    # ---- HR1: zero inflation audit ----
    audit = {"fit_seed_range": [NORM_FIT_SEEDS[0], NORM_FIT_SEEDS[-1]],
             "n_total": int(X_fit.shape[0]), "per_feature": {}}
    for d in ZERO_INFLATED_DIMS:
        col = X_fit[:, d]
        pos = col[col > 0.0]
        rec = {"n_total": int(col.shape[0]), "n_zero": int(np.sum(col <= 0.0)),
               "zero_fraction": float(np.mean(col <= 0.0)), "n_positive": int(pos.shape[0])}
        if pos.shape[0] > 0:
            rec.update({"min_positive": float(pos.min()), "q01_positive": float(np.percentile(pos, 1)),
                        "q05_positive": float(np.percentile(pos, 5)),
                        "median_positive": float(np.median(pos)),
                        "q95_positive": float(np.percentile(pos, 95)),
                        "q99_positive": float(np.percentile(pos, 99)), "max_positive": float(pos.max())})
        audit["per_feature"][RAW_ORDER[d]] = rec
    dcol = np.abs(X_fit[:, DELTA_DIM])
    dpos = dcol[dcol > 0.0]
    audit["delta_severity_abs_positive"] = {
        "n_positive": int(dpos.shape[0]),
        "q99_positive": float(np.percentile(dpos, 99)) if dpos.shape[0] else 0.0,
        "structural_zero_fraction": float(np.mean(dcol <= 0.0))}
    _jdump(audit, RESULTS_DIR / "E4HR_zero_inflation_audit.json")

    # ---- HR2: fit + freeze V3 constants ----
    v3 = ZeroAwareRobustLogNormalizerV3.fit(X_fit, min_positive=MIN_POSITIVE_SAMPLES)
    v3c = v3.to_dict()
    v3_valid = v3c["V3_SCALER_VALID"] == "YES"
    consts_out = {
        "version": NORM_V3, "eps": EPS, "min_positive_samples": MIN_POSITIVE_SAMPLES,
        "zero_inflated_dims": ZERO_INFLATED_DIMS,
        "constants": {RAW_ORDER[int(d)]: v3c["zero_aware"][str(d)] for d in ZERO_INFLATED_DIMS},
        "delta_q99_abs_positive": v3c["delta_q99_abs_positive"],
        "n_samples": v3c["n_samples"], "invalid_features": v3c["invalid_features"],
        "V3_SCALER_VALID": v3c["V3_SCALER_VALID"],
        "constants_sha256": v3.constants_sha256(),
        "fit_seed_range": [NORM_FIT_SEEDS[0], NORM_FIT_SEEDS[-1]],
        "source_artifact": "reconstructed E4TR training states (seeds 1001..1060); train-only",
        "fit_states_sha256": _hash_file(RESULTS_DIR / "E4HR_fit_states.npy"),
    }
    _jdump(consts_out, RESULTS_DIR / "E4HR_v3_constants.json")

    # ---- V3 design contract ----
    design = {
        "version": NORM_V3,
        "zero_inflated_features": [RAW_ORDER[d] for d in ZERO_INFLATED_DIMS],
        "zero_inflated_dims": ZERO_INFLATED_DIMS,
        "nonneg_transform": "x<=0 -> 0 ; x>0 -> clip(log1p(x/m)/log1p(c/m), 0, 1)",
        "m": "median(positive training subset) per dim", "c": "q99(positive training subset) per dim",
        "occupancy(2)": "identity clip[0,1]", "active_region_fraction(4)": "identity clip[0,1]",
        "delta_severity(3)": "signed clip(x/(a3+eps), -1, 1); a3=q99(|x| : |x|>0); zero/sign preserved; no median-centering",
        "eps": EPS, "monotonic_nonneg": True, "structural_zero_preserved": True,
        "degeneracy_rule": "invalid if n_positive<MIN_POSITIVE_SAMPLES or m<=0 or c<=0 or "
                           "log1p(c/m)<=1e-12 (or a3<=0 for delta); never silently substitute",
        "MIN_POSITIVE_SAMPLES": MIN_POSITIVE_SAMPLES,
        "min_positive_rationale": "principled fixed floor; ample given ~tens-of-thousands positive "
                                  "samples per feature in the train-only design set",
    }
    _jdump(design, RESULTS_DIR / "E4HR_v3_design_contract.json")

    # ---- HR3: freeze gates (definitions only; applied in E4HI) ----
    zero_aware_sat = {
        "STRUCTURAL_ZERO_FRACTION": "P(raw x = 0) per nonneg feature; REPORTED, not gated (not clipping)",
        "POSITIVE_LOWER_CLIP_FRACTION": "among raw x>0: fraction with z <= tolerance_zero(1e-12); GATE <= 0.05",
        "POSITIVE_UPPER_CLIP_FRACTION": "among raw x>0: fraction with z >= 1-1e-12; GATE <= 0.05",
        "SIGNED_DELTA_SATURATION": "among |raw delta|>0: fraction with |z| >= 1-1e-12; GATE <= 0.05",
        "tolerance_zero": 1e-12, "gate": 0.05,
        "distinguishes_structural_zero_from_clipping": True,
        "frozen_before_validation": True,
    }
    _jdump(zero_aware_sat, RESULTS_DIR / "E4HR_zero_aware_saturation_contract.json")

    signal_gate = {
        "rule": "range >= 0.05 AND std >= 0.02 (existing E4H/E4G near-constant rule; reused unchanged)",
        "near_const_range": NEAR_CONST_RANGE, "near_const_std": NEAR_CONST_STD,
        "previously_compressed_features": COMPRESSED_LOCAL,
        "requirement": "each previously-compressed local feature must NO longer be near-constant under V3",
        "frozen_before_validation": True,
    }
    _jdump(signal_gate, RESULTS_DIR / "E4HR_signal_gate_contract.json")

    sep_gate = {
        "methodology": "exact E4H/E4GD D2 (MIXED regime label; logistic/tree/kNN; 5-fold; "
                       "balanced_accuracy; identical folds)",
        "compare": ["RAW", "V1", "V2_rejected", "V3"],
        "requirement": "V3 aggregate separability >= V1 AND V3 not classified worse than V1",
        "progression_also_requires": "V3 aggregate > V1 OR V3 D2-loss classification improves over V1",
        "frozen_before_validation": True,
    }
    _jdump(sep_gate, RESULTS_DIR / "E4HR_separability_gate_contract.json")

    # ---- semantic invariance (verify V3 formula properties; cheap) ----
    semantic = _semantic_invariance(v3) if v3_valid else {"all_invariants_pass": False,
                                                          "note": "V3 invalid; not verified"}
    _jdump({**semantic, "state_dim": 9, "state_ordering": RAW_ORDER,
            "no_feature_added_or_removed": True, "no_region_id": True},
           RESULTS_DIR / "E4HR_semantic_invariance_contract.json")

    # ---- HR5: versioning ----
    versioning = {
        "versions": {
            "V1": {"name": NORM_V1, "status": "LEGACY (old E4TR/E4F checkpoints require this)"},
            "V2": {"name": NORM_V2, "status": "REJECTED (DESIGN_REJECTED_BY_E4H_SATURATION_GATE; never trained)"},
            "V3": {"name": NORM_V3, "status": "CANDIDATE (future A/B treatment)"},
        },
        "old_checkpoints_use": NORM_V1, "no_old_checkpoint_uses_V3": True,
        "future_v3_checkpoints_store": ["normalization_version", "normalizer_constants_sha256"],
        "v3_constants_sha256": v3.constants_sha256(),
        "mismatch_raises": True, "guard_function": "check_normalization_compatibility",
    }
    _jdump(versioning, RESULTS_DIR / "E4HR_versioning_contract.json")

    # ---- HR6: A/B supersession (V1 vs V3) ----
    ab = {
        "supersedes_not_erases": "E4H planned V1-vs-V2 (V2 never trained); superseded by V1-vs-V3",
        "design": "MATCHED_V1_VS_V3 A/B (single factor = normalization_version)",
        "control_arm": {"name": "ORIGINAL_NORMALIZATION_V1", "normalization_version": NORM_V1},
        "treatment_arm": {"name": NORM_V3, "normalization_version": NORM_V3,
                          "constants_sha256": v3.constants_sha256()},
        "v2_not_a_training_candidate": True,
        "identical_across_arms": {
            "state_semantics": "V2 local + GlobalContextV1", "state_dim": 9,
            "network": "9-128-128-4 Shared Double DQN",
            "reward": {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01},
            "gamma": 0.99, "learning_rate": 1e-4, "tau": 0.005, "batch": 128,
            "replay": 50000, "warmup": 1000, "update_frequency": 4,
            "epsilon": "1.0->0.05 over 50000 regional transitions", "epochs": 6,
            "sigma_L": 0.02, "sigma_P": 0.005, "sigma_D": 0.15, "lambda_P": 1.0,
            "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted NSGA-II",
            "no_HARD": True, "no_MEMORY": True, "response_operators": "unchanged",
            "joint_decision_semantics": "unchanged",
        },
        "only_experimental_factor": "normalization_version",
        "policy_seeds": NEW_POLICY_SEEDS, "policy_seeds_paired_across_arms": True,
        "primary_performance_endpoint": "MIGD_response (V3 DDQN vs V1 DDQN)",
        "secondary_performance": ["MIGD_epoch", "response_FE"],
        "primary_mechanism_endpoints": ["MIXED_PREDICT_fraction", "within_event_multi_action_rate",
                                        "regional_action_entropy"],
        "expected_decollapse_direction": "PREDICT fraction decreases, multi-action rate increases, "
                                         "entropy increases (uniform NOT required)",
    }
    _jdump(ab, RESULTS_DIR / "E4HR_ab_contract_supersession.json")

    # ---- future seeds (unchanged; still unconsumed) ----
    all_hist = (set(range(701, 781)) | set(range(801, 883)) | set(range(901, 941))
                | set(range(1001, 1141)) | set(range(1201, 1221)))
    new_all = set(NEW_TRAIN_SEEDS) | set(NEW_VAL_SEEDS) | set(NEW_FINAL_SEEDS)
    seed_contract = {
        "train": {"seeds": NEW_TRAIN_SEEDS, "range": [1301, 1360]},
        "validation": {"seeds": NEW_VAL_SEEDS, "range": [1361, 1380]},
        "final": {"seeds": NEW_FINAL_SEEDS, "range": [1381, 1440], "FIREWALL": "UNOPENED"},
        "policy_seeds": NEW_POLICY_SEEDS,
        "disjoint_from_historical": len(new_all & all_hist) == 0,
        "no_seed_consumed_yet": True, "NEW_FINAL_EVAL_ACCESS_COUNT": 0,
    }
    _jdump(seed_contract, RESULTS_DIR / "E4HR_future_seed_contract.json")

    # ---- HR7: scientific hypothesis / future experiment ----
    future = {
        "H0_control_interpretation": "changing normalization alone does not materially alter "
                                     "regional specialization/performance",
        "treatment_hypothesis": "zero-aware normalization (V3) restores usable local-state variation "
                                "and reduces policy collapse relative to otherwise-identical V1",
        "no_claim_v3_beats_global_predict": True,
        "primary_causal_comparison": "V3 DDQN vs V1 DDQN on MIGD_response",
        "secondary": ["MIGD_epoch", "response_FE"],
        "mechanism_endpoints": ["MIXED_PREDICT_fraction", "within_event_multi_action_rate",
                                "regional_action_entropy"],
        "future_training": {"arms": ["V1", "V3"], "policies_per_arm": 5, "episodes_per_policy": 1080,
                            "total_future_episodes": 10800},
        "no_training_in_E4HR": True,
    }
    _jdump(future, RESULTS_DIR / "E4HR_future_experiment_contract.json")

    # ---- success ----
    ready = (v3_valid and semantic.get("all_invariants_pass", False)
             and seed_contract["disjoint_from_historical"]
             and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4h["E4H_COMPLETE"] == "NO")

    master = {
        "stage": "E4HR_NORMALIZATION_DESIGN_REVIEW",
        "V1": NORM_V1, "V2_status": "REJECTED_NOT_TRAINED", "V3": NORM_V3,
        "v3_constants_sha256": v3.constants_sha256(), "V3_SCALER_VALID": v3c["V3_SCALER_VALID"],
        "zero_inflated_features": [RAW_ORDER[d] for d in ZERO_INFLATED_DIMS],
        "structural_zero_handling": "PRESERVED_AS_ZERO",
        "gates_frozen_before_validation": True,
        "future_control": NORM_V1, "future_treatment": NORM_V3,
        "new_train_seeds": [1301, 1360], "new_validation_seeds": [1361, 1380],
        "new_final_seeds": [1381, 1440], "new_policy_seeds": NEW_POLICY_SEEDS,
        "state_dim": 9, "state_schema_changed": False,
        "NORMALIZATION_V3_READY_FOR_VALIDATION": "YES" if ready else "NO",
        "next_stage": ("E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready
                       else "E4HR_DESIGN_FIX_REQUIRED"),
    }
    _jdump(master, RESULTS_DIR / "E4HR_protocol_frozen.json")

    verdict = _build_verdict(v3c, consts_out, semantic, seed_contract, ready)
    _jdump(verdict, RESULTS_DIR / "E4HR_verdict.json")
    _write_report(verdict, audit, consts_out, semantic)
    _emit_terminal(verdict, consts_out)
    print(f"[E4HR] wall {time.time()-t0:.0f}s")
    return verdict


def _semantic_invariance(v3):
    checks = {}
    mono = {}
    for d in ZERO_INFLATED_DIMS:
        xs = np.linspace(0, 12, 60)
        base = np.zeros((60, 9)); base[:, d] = xs
        z = v3.transform_batch(base)[:, d]
        mono[RAW_ORDER[d]] = bool(np.all(np.diff(z) >= -1e-12))
    checks["monotonic_nonneg"] = mono
    # zero -> zero, positive -> positive for a zero-inflated dim
    z0 = v3.transform(np.zeros(9))
    checks["structural_zero_maps_to_zero"] = bool(all(abs(z0[d]) < 1e-12 for d in ZERO_INFLATED_DIMS))
    xp = np.zeros(9); xp[0] = 1e-6; xp[1] = 1e-6
    zp = v3.transform(xp)
    checks["small_positive_maps_positive"] = bool(zp[0] > 0 and zp[1] > 0)
    # delta sign/zero
    checks["delta_zero_preserved"] = bool(abs(v3.transform(np.zeros(9))[DELTA_DIM]) < 1e-12)
    dp = np.zeros(9); dp[DELTA_DIM] = 0.3
    dn = np.zeros(9); dn[DELTA_DIM] = -0.3
    checks["delta_sign_preserved"] = bool(v3.transform(dp)[DELTA_DIM] > 0 and v3.transform(dn)[DELTA_DIM] < 0)
    # occupancy / active identity
    oc = np.zeros(9); oc[2] = 0.37
    ac = np.zeros(9); ac[4] = 0.42
    checks["occupancy_identity"] = bool(abs(v3.transform(oc)[2] - 0.37) < 1e-9)
    checks["active_fraction_identity"] = bool(abs(v3.transform(ac)[4] - 0.42) < 1e-9)
    checks["all_invariants_pass"] = bool(
        all(mono.values()) and checks["structural_zero_maps_to_zero"]
        and checks["small_positive_maps_positive"] and checks["delta_zero_preserved"]
        and checks["delta_sign_preserved"] and checks["occupancy_identity"]
        and checks["active_fraction_identity"])
    return checks


def _build_verdict(v3c, consts, semantic, seed_contract, ready):
    return {
        "stage": "E4HR_NORMALIZATION_DESIGN_REVIEW",
        "E4HR_COMPLETE": "YES" if ready else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4H_ORIGINAL_RESULT": "NO",
        "V2_STATUS": "REJECTED_NOT_TRAINED", "V3_NAME": NORM_V3,
        "ZERO_INFLATED_FEATURES": [RAW_ORDER[d] for d in ZERO_INFLATED_DIMS],
        "STRUCTURAL_ZERO_HANDLING": "PRESERVED_AS_ZERO",
        "POSITIVE_LOWER_SATURATION_GATE": "<=0.05",
        "POSITIVE_UPPER_SATURATION_GATE": "<=0.05",
        "DELTA_NONZERO_SATURATION_GATE": "<=0.05",
        "V3_CONSTANTS_FIT_SEEDS": "1001..1060", "FIT_LEAKAGE": "NONE",
        "V3_SCALER_VALID": v3c["V3_SCALER_VALID"],
        "V3_INVALID_FEATURES": v3c["invalid_features"],
        "SEMANTIC_INVARIANCE": "PASS" if semantic.get("all_invariants_pass") else "FAIL",
        "STATE_SCHEMA_CHANGED": "NO", "STATE_DIM": 9,
        "NETWORK_CHANGED": "NO", "REWARD_CHANGED": "NO", "JOINT_ARCHITECTURE_CHANGED": "NO",
        "FUTURE_CONTROL": NORM_V1, "FUTURE_TREATMENT": NORM_V3,
        "NEW_TRAIN_SEEDS": "1301..1360", "NEW_VALIDATION_SEEDS": "1361..1380",
        "NEW_FINAL_SEEDS": "1381..1440", "NEW_POLICY_SEEDS": "3001..3005",
        "NEW_FINAL_EVAL_ACCESS_COUNT": 0, "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready
                       else "E4HR_DESIGN_FIX_REQUIRED"),
    }


def _write_report(v, audit, consts, semantic):
    L = ["# E4HR — Zero-Aware Normalization Design Review", "",
         f"**E4HR_COMPLETE = {v['E4HR_COMPLETE']}** — design/review/freeze only; no training. "
         "E4F remains FAILED, E4H remains NO, V2 remains REJECTED.", "",
         "## Why a new candidate (not a gate relaxation)",
         "E4H V2 failed saturation because it conflated **structural zeros** (raw 0 → 0, correct) "
         "with **clipping saturation** (raw >0 → 0, information loss). V3 keeps structural zeros at 0 "
         "and spreads positive values via a robust log; the saturation gate is redefined to measure "
         "clipping among positive samples only. E4H stays FAIL.", "",
         "## Zero inflation (train-only design data)", "",
         "| feature | zero_fraction | n_positive | median+ | q99+ |", "|---|---|---|---|---|"]
    for f, r in audit["per_feature"].items():
        L.append(f"| {f} | {r['zero_fraction']:.3f} | {r['n_positive']} | "
                 f"{r.get('median_positive', float('nan')):.4g} | {r.get('q99_positive', float('nan')):.4g} |")
    L += ["", "## V3 (frozen)",
          f"- {NORM_V3}; constants sha256 {consts['constants_sha256'][:16]}…; "
          f"V3_SCALER_VALID = {consts['V3_SCALER_VALID']}; invalid = {consts['invalid_features']}",
          "- nonneg: x≤0→0; x>0→clip(log1p(x/m)/log1p(c/m),0,1). delta: signed x/(a3), zero/sign kept. "
          "occupancy/active identity. dim 9.",
          f"- semantic invariance: {'PASS' if semantic.get('all_invariants_pass') else 'FAIL'}", "",
          "## Frozen gates (applied later in E4HI, not here)",
          "- Signal: previously-compressed local features must be non-near-constant (range≥0.05 & std≥0.02).",
          "- Zero-aware saturation: positive-lower-clip ≤5%, positive-upper-clip ≤5%, "
          "signed-delta nonzero-saturation ≤5% (structural zeros REPORTED, not gated).",
          "- Separability: V3 ≥ V1 and not worse-classified than V1.", "",
          "## Superseded A/B (frozen, not trained)",
          f"- V1 control vs V3 treatment; only factor = normalization_version; seeds train 1301-1360 / "
          f"val 1361-1380 / final 1381-1440 (unopened); policy seeds 3001-3005.", "",
          f"## Status: NEXT_STAGE = {v['NEXT_STAGE']}",
          "E4HI must independently validate V3 offline (RAW/V1/V2/V3) before any V1-vs-V3 training."]
    (RESULTS_DIR / "E4HR_report.md").write_text("\n".join(L))


def _emit_terminal(v, consts):
    print()
    print(f"E4HR_COMPLETE                    = {v['E4HR_COMPLETE']}")
    print(f"ORIGINAL_E4F_VERDICT             = {v['ORIGINAL_E4F_VERDICT']}")
    print(f"E4H_ORIGINAL_RESULT              = {v['E4H_ORIGINAL_RESULT']}")
    print(f"V2_STATUS                        = {v['V2_STATUS']}")
    print(f"V3_NAME                          = {v['V3_NAME']}")
    print(f"ZERO_INFLATED_FEATURES           = {','.join(v['ZERO_INFLATED_FEATURES'])}")
    print(f"STRUCTURAL_ZERO_HANDLING         = {v['STRUCTURAL_ZERO_HANDLING']}")
    print(f"POSITIVE_LOWER_SATURATION_GATE   = {v['POSITIVE_LOWER_SATURATION_GATE']}")
    print(f"POSITIVE_UPPER_SATURATION_GATE   = {v['POSITIVE_UPPER_SATURATION_GATE']}")
    print(f"DELTA_NONZERO_SATURATION_GATE    = {v['DELTA_NONZERO_SATURATION_GATE']}")
    print(f"V3_CONSTANTS_FIT_SEEDS           = {v['V3_CONSTANTS_FIT_SEEDS']}")
    print(f"FIT_LEAKAGE                      = {v['FIT_LEAKAGE']}")
    print(f"V3_SCALER_VALID                  = {v['V3_SCALER_VALID']}")
    print(f"STATE_SCHEMA_CHANGED             = {v['STATE_SCHEMA_CHANGED']}")
    print(f"STATE_DIM                        = {v['STATE_DIM']}")
    print(f"NETWORK_CHANGED                  = {v['NETWORK_CHANGED']}")
    print(f"REWARD_CHANGED                   = {v['REWARD_CHANGED']}")
    print(f"JOINT_ARCHITECTURE_CHANGED       = {v['JOINT_ARCHITECTURE_CHANGED']}")
    print(f"FUTURE_CONTROL                   = {v['FUTURE_CONTROL']}")
    print(f"FUTURE_TREATMENT                 = {v['FUTURE_TREATMENT']}")
    print(f"NEW_TRAIN_SEEDS                  = {v['NEW_TRAIN_SEEDS']}")
    print(f"NEW_VALIDATION_SEEDS             = {v['NEW_VALIDATION_SEEDS']}")
    print(f"NEW_FINAL_SEEDS                  = {v['NEW_FINAL_SEEDS']}")
    print(f"NEW_POLICY_SEEDS                 = {v['NEW_POLICY_SEEDS']}")
    print(f"NEW_FINAL_EVAL_ACCESS_COUNT      = {v['NEW_FINAL_EVAL_ACCESS_COUNT']}")
    print(f"NEW_TRAINING                     = {v['NEW_TRAINING']}")
    print(f"FULL_REGRESSION                  = {v['FULL_REGRESSION']}")
    print(f"NEXT_STAGE                       = {v['NEXT_STAGE']}")


if __name__ == "__main__":
    main()
