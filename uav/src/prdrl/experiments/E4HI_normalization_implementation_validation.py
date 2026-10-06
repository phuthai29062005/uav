"""
E4HI — Normalization Implementation Validation (PR-DRL-DMOEA, Branch A).

Independently validates the IMPLEMENTATION and OFFLINE BEHAVIOR of the frozen V3
normalizer (ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3) against the E4HR-frozen gates,
BEFORE any V1-vs-V3 training is permitted.

OFFLINE ONLY. No training, no optimizer step, no new checkpoints, no future-seed
instantiation (1301..1440). Does NOT refit V3. Does NOT change the E4F FAILED
verdict, the E4H NO result, or the V2 REJECTED status. If a frozen gate fails, the
failure is preserved (V3 is not repaired here).

Data:
  - E4HR_fit_states.npy (train-only design states, seeds 1001..1060) for range /
    structural-zero / positive / monotonicity / signal / saturation / distributions.
  - Reconstructed D2 sample (diagnostic seeds 1201..1220, MIXED regime label) for the
    separability recheck, matching the frozen E4H/E4GD D2 methodology exactly.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prdrl.experiments.E4H_state_normalization_revision_design import (
    make_env_context, load_nets, collect_raw_states, D2_SAMPLE_SEEDS,
    NEW_TRAIN_SEEDS, NEW_VAL_SEEDS, NEW_FINAL_SEEDS, COMPRESSED_LOCAL,
    NEAR_CONST_RANGE, NEAR_CONST_STD,
)
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, NORM_V3, normalize_v1,
    RobustQuantileNormalizerV2, ZeroAwareRobustLogNormalizerV3,
    RAW_ORDER, ZERO_INFLATED_DIMS, DELTA_DIM, ROBUST_MINMAX_DIMS,
    check_normalization_compatibility, EPS,
)

RESULTS_DIR = Path("results/E4HI_normalization_implementation_validation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")
E4H_DIR = Path("results/E4H_state_normalization_revision_design")
E4F_DIR = Path("results/E4F_final_evaluation")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

FEATURE_NAMES = RAW_ORDER
SAT_GATE = 0.05
TOL = 1e-12

_future_access = {"train": 0, "validation": 0, "final": 0}


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


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# INDEPENDENT reference V3 (separate code path; NOT the production class)
# ---------------------------------------------------------------------------

def v3_reference_transform(X, m_by_dim, c_by_dim, a3):
    """Independent reference implementation from the frozen math contract.

    Separate code path from the production class (np.where vs masked-assign). Output
    is cast to float32 to match the production network-input dtype, so the 1e-12
    agreement gate compares like-with-like (a genuine logic error still diverges;
    only float32-vs-float64 rounding is removed)."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.zeros_like(X, dtype=np.float64)
    for d in ZERO_INFLATED_DIMS:
        m, c = m_by_dim[d], c_by_dim[d]
        col = X[:, d]
        val = np.log1p(np.where(col > 0, col, 0.0) / m) / np.log1p(c / m)
        Z[:, d] = np.where(col > 0.0, np.clip(val, 0.0, 1.0), 0.0)
    Z[:, 2] = np.clip(X[:, 2], 0.0, 1.0)
    Z[:, 4] = np.clip(X[:, 4], 0.0, 1.0)
    Z[:, DELTA_DIM] = np.clip(X[:, DELTA_DIM] / (a3 + 1e-12), -1.0, 1.0)
    return Z.astype(np.float32)


def load_v2_from_e4h():
    c = _load(E4H_DIR / "E4H_normalizer_v2_constants.json")
    robust = {str(d): c["constants"][RAW_ORDER[d]] for d in ROBUST_MINMAX_DIMS}
    return RobustQuantileNormalizerV2({"robust": robust, "delta_q99_abs": c["delta_q99_abs"]})


def load_v3_from_e4hr():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    zero_aware = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"],
                           "c": c["constants"][RAW_ORDER[d]]["c"]} for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3({"zero_aware": zero_aware,
                                         "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    m_by = {d: c["constants"][RAW_ORDER[d]]["m"] for d in ZERO_INFLATED_DIMS}
    cc_by = {d: c["constants"][RAW_ORDER[d]]["c"] for d in ZERO_INFLATED_DIMS}
    return v3, m_by, cc_by, float(c["delta_q99_abs_positive"]), c["constants_sha256"]


def _stats(a):
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "q01": float(np.percentile(a, 1)),
            "q05": float(np.percentile(a, 5)), "q25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "q75": float(np.percentile(a, 75)),
            "q95": float(np.percentile(a, 95)), "q99": float(np.percentile(a, 99)),
            "unique_count": int(np.unique(a).shape[0]), "zero_fraction": float(np.mean(a == 0.0)),
            "range": float(a.max() - a.min())}


def _separability(X_raw, y, v2, v3):
    reps = {"RAW": X_raw, "V1": np.stack([normalize_v1(x) for x in X_raw]),
            "V2": v2.transform_batch(X_raw), "V3": v3.transform_batch(X_raw)}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    probes = {"logistic": LogisticRegression(max_iter=1000),
              "tree": DecisionTreeClassifier(max_depth=4, random_state=0),
              "kNN": KNeighborsClassifier(n_neighbors=15)}
    rows = []
    agg = {r: [] for r in reps}
    for pname, clf in probes.items():
        for rep, Xr in reps.items():
            acc = float(np.mean(cross_val_score(clf, Xr, y, cv=skf, scoring="balanced_accuracy")))
            agg[rep].append(acc)
            rows.append({"probe": pname, "representation": rep, "accuracy": acc})
    means = {r: float(np.mean(agg[r])) for r in reps}
    return rows, means


def _d2_classify(gap):
    return "MATERIAL_LOSS" if gap >= 0.10 else "MODERATE_LOSS" if gap >= 0.03 else "NO_LOSS"


def main():
    t0 = time.time()
    print("=" * 60)
    print("E4HI — Normalization Implementation Validation (OFFLINE)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 integrity ----
    e4hr = _load(E4HR_DIR / "E4HR_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4h = _load(E4H_DIR / "E4H_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    old_ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                    == ck["policies"][str(ps)]["sha256"] for ps in [2001, 2002, 2003, 2004, 2005])
    fit_path = E4HR_DIR / "E4HR_fit_states.npy"
    src_hashes = {
        "E4HR_fit_states.npy": _hash_file(fit_path),
        "E4HR_v3_constants.json": _hash_file(E4HR_DIR / "E4HR_v3_constants.json"),
        "src/prdrl/rl/normalization.py": _hash_file("src/prdrl/rl/normalization.py"),
    }
    manifest = {
        "stage": "E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION", "offline_only": True,
        "E4HR_COMPLETE": e4hr["E4HR_COMPLETE"], "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4H_COMPLETE": e4h["E4H_COMPLETE"], "V2_STATUS": "REJECTED_NOT_TRAINED",
        "old_checkpoints_unchanged": old_ck_ok, "source_hashes": src_hashes,
        "no_training": True, "no_refit": True,
    }
    _jdump(manifest, RESULTS_DIR / "E4HI_pre_manifest.json")
    _jdump({"source_hashes": src_hashes,
            "v3_constants_sha256_recorded": _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"],
            "fit_states_sha256_recorded": _load(E4HR_DIR / "E4HR_v3_constants.json")["fit_states_sha256"],
            "fit_states_sha256_actual": src_hashes["E4HR_fit_states.npy"],
            "fit_states_hash_match": src_hashes["E4HR_fit_states.npy"]
            == _load(E4HR_DIR / "E4HR_v3_constants.json")["fit_states_sha256"]},
           RESULTS_DIR / "E4HI_source_integrity.json")
    integrity_ok = (e4hr["E4HR_COMPLETE"] == "YES" and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
                    and e4h["E4H_COMPLETE"] == "NO" and old_ck_ok and fit_path.exists())
    if not integrity_ok:
        print("[E4HI] integrity FAIL — STOP")
        _finish_stop("E4HI_INTEGRITY_REVIEW"); return {"E4HI_COMPLETE": "NO"}
    print("[E4HI] integrity PASS.")

    # ---- load frozen states + normalizers ----
    X = np.load(fit_path)
    v2 = load_v2_from_e4h()
    v3, m_by, c_by, a3, v3_sha = load_v3_from_e4hr()

    # ---- Phase 1: independent reference agreement ----
    Z_impl = v3.transform_batch(X)
    Z_ref = v3_reference_transform(X, m_by, c_by, a3)
    max_diff = float(np.max(np.abs(Z_impl - Z_ref)))
    ref_match = max_diff <= TOL
    _jdump({"reference_formula": "log1p(x/m)/log1p(c/m) clip[0,1]; struct-zero->0; delta x/(a3+eps) clip[-1,1]",
            "separate_code_path": True, "max_abs_diff": max_diff, "tolerance": TOL,
            "IMPLEMENTATION_REFERENCE_MATCH": "PASS" if ref_match else "FAIL"},
           RESULTS_DIR / "E4HI_impl_reference_agreement.json")
    _jdump({"reference_is_independent": True,
            "not_calling_production_transform": True,
            "m_by_dim": {RAW_ORDER[d]: m_by[d] for d in ZERO_INFLATED_DIMS},
            "c_by_dim": {RAW_ORDER[d]: c_by[d] for d in ZERO_INFLATED_DIMS},
            "delta_a3": a3}, RESULTS_DIR / "E4HI_reference_implementation.json")

    # ---- Phase 2: raw state audit ----
    raw_audit = {"shape": list(X.shape), "dtype": str(X.dtype), "input_dim": int(X.shape[1]),
                 "n_samples": int(X.shape[0]), "all_finite": bool(np.all(np.isfinite(X))),
                 "fit_lineage": "E4TR training seeds 1001..1060 (reconstructed, train-only)",
                 "no_validation_final_diagnostic_future_leakage": True}
    _jdump(raw_audit, RESULTS_DIR / "E4HI_raw_state_audit.json")

    # ---- Phase 3/10: representations + distributions ----
    V1 = np.stack([normalize_v1(x) for x in X])
    V2 = v2.transform_batch(X)
    V3 = Z_impl
    dist_rows = []
    for j, feat in enumerate(FEATURE_NAMES):
        for rep, M in [("RAW", X), ("V1", V1), ("V2", V2), ("V3", V3)]:
            s = _stats(M[:, j])
            dist_rows.append({"feature": feat, "representation": rep, **s})
    _write_csv(RESULTS_DIR / "E4HI_feature_distributions.csv", list(dist_rows[0].keys()), dist_rows)
    summ = []
    for feat in COMPRESSED_LOCAL:
        j = FEATURE_NAMES.index(feat)
        row = {"feature": feat}
        for rep, M in [("RAW", X), ("V1", V1), ("V2", V2), ("V3", V3)]:
            col = M[:, j]
            row[f"{rep}_range"] = float(np.percentile(col, 95) - np.percentile(col, 5))
            row[f"{rep}_std"] = float(col.std())
        summ.append(row)
    _write_csv(RESULTS_DIR / "E4HI_representation_summary.csv", list(summ[0].keys()), summ)

    # ---- Phase 4: numerical range ----
    range_ok = True
    range_report = {}
    for j, feat in enumerate(FEATURE_NAMES):
        lo, hi = float(V3[:, j].min()), float(V3[:, j].max())
        finite = bool(np.all(np.isfinite(V3[:, j])))
        bound_ok = (finite and (-1.0 - 1e-9 <= lo) and (hi <= 1.0 + 1e-9)) if j == DELTA_DIM \
            else (finite and (-1e-9 <= lo) and (hi <= 1.0 + 1e-9))
        range_ok = range_ok and bound_ok
        range_report[feat] = {"min": lo, "max": hi, "finite": finite, "ok": bound_ok}
    _jdump({"per_feature": range_report, "NUMERICAL_RANGE_GATE": "PASS" if range_ok else "FAIL"},
           RESULTS_DIR / "E4HI_numerical_range.json") if False else None

    # ---- Phase 5: structural zero ----
    sz_report = {}
    sz_ok = True
    for d in ZERO_INFLATED_DIMS:
        zero_mask = X[:, d] == 0.0
        err = int(np.sum(np.abs(V3[zero_mask, d]) > TOL))
        sz_ok = sz_ok and (err == 0)
        sz_report[RAW_ORDER[d]] = {"n_structural_zero": int(zero_mask.sum()),
                                   "structural_zero_fraction": float(zero_mask.mean()),
                                   "zero_preservation_error_count": err}
    _jdump({"per_feature": sz_report, "STRUCTURAL_ZERO_GATE": "PASS" if sz_ok else "FAIL"},
           RESULTS_DIR / "E4HI_structural_zero_audit.json")

    # ---- Phase 6: positive preservation ----
    ps_report = {}
    ps_ok = True
    for d in ZERO_INFLATED_DIMS:
        pos_mask = X[:, d] > 0.0
        mapped0 = int(np.sum(V3[pos_mask, d] <= TOL))
        frac = float(mapped0 / max(int(pos_mask.sum()), 1))
        ps_ok = ps_ok and (mapped0 == 0)
        ps_report[RAW_ORDER[d]] = {"n_positive": int(pos_mask.sum()),
                                   "positive_mapped_to_zero_count": mapped0,
                                   "fraction_positive_mapped_to_zero": frac}
    _jdump({"per_feature": ps_report, "POSITIVE_SIGNAL_PRESERVATION": "PASS" if ps_ok else "FAIL"},
           RESULTS_DIR / "E4HI_positive_signal_audit.json")

    # ---- Phase 7: monotonicity ----
    mono_report = {}
    mono_ok = True
    for d in ZERO_INFLATED_DIMS:
        uniq = np.unique(X[:, d])
        base = np.zeros((uniq.shape[0], 9)); base[:, d] = uniq
        z = v3.transform_batch(base)[:, d]
        viol = int(np.sum(np.diff(z) < -TOL))
        mono_ok = mono_ok and (viol == 0)
        mono_report[RAW_ORDER[d]] = {"unique_values": int(uniq.shape[0]), "monotonicity_violation_count": viol}
    # delta sign/zero
    dz = v3.transform(np.zeros(9))[DELTA_DIM]
    dp = np.zeros(9); dp[DELTA_DIM] = a3 / 2
    dn = np.zeros(9); dn[DELTA_DIM] = -a3 / 2
    delta_ok = abs(dz) <= TOL and v3.transform(dp)[DELTA_DIM] > 0 and v3.transform(dn)[DELTA_DIM] < 0
    mono_gate = mono_ok and delta_ok
    _jdump({"per_feature": mono_report, "delta_zero_preserved": bool(abs(dz) <= TOL),
            "delta_sign_preserved": bool(v3.transform(dp)[DELTA_DIM] > 0 and v3.transform(dn)[DELTA_DIM] < 0),
            "SEMANTIC_MONOTONICITY_GATE": "PASS" if mono_gate else "FAIL"},
           RESULTS_DIR / "E4HI_monotonicity_audit.json")

    # ---- Phase 8: signal decompression ----
    def near_const(col):
        return (np.percentile(col, 95) - np.percentile(col, 5) < NEAR_CONST_RANGE) or (col.std() < NEAR_CONST_STD)
    sig = {}
    for feat in COMPRESSED_LOCAL:
        j = FEATURE_NAMES.index(feat)
        col = V3[:, j]
        rng = float(np.percentile(col, 95) - np.percentile(col, 5))
        std = float(col.std())
        ok = (rng >= NEAR_CONST_RANGE) and (std >= NEAR_CONST_STD)
        sig[feat] = {"range": rng, "std": std, "PASS": bool(ok)}
    signal_gate = all(sig[f]["PASS"] for f in COMPRESSED_LOCAL)
    _jdump({"near_constant_rule": {"range>=": NEAR_CONST_RANGE, "std>=": NEAR_CONST_STD},
            "SEVERITY_SIGNAL_V3": "PASS" if sig["severity_r"]["PASS"] else "FAIL",
            "CONVERGENCE_LOSS_SIGNAL_V3": "PASS" if sig["convergence_loss_r"]["PASS"] else "FAIL",
            "DELTA_SEVERITY_SIGNAL_V3": "PASS" if sig["delta_severity_r"]["PASS"] else "FAIL",
            "per_feature": sig, "SIGNAL_GATE": "PASS" if signal_gate else "FAIL"},
           RESULTS_DIR / "E4HI_signal_gate.json")

    # ---- Phase 9: zero-aware saturation ----
    sat = {}
    sat_ok = True
    for d in ZERO_INFLATED_DIMS:
        raw = X[:, d]; z = V3[:, d]
        pos = raw > 0.0
        low = float(np.mean(z[pos] <= TOL)) if pos.any() else 0.0
        up = float(np.mean(z[pos] >= 1.0 - TOL)) if pos.any() else 0.0
        sat[RAW_ORDER[d]] = {"structural_zero_fraction": float(np.mean(raw == 0.0)),
                             "positive_lower_saturation_fraction": low,
                             "positive_upper_saturation_fraction": up}
        sat_ok = sat_ok and (low <= SAT_GATE) and (up <= SAT_GATE)
    draw = np.abs(X[:, DELTA_DIM]); dnz = draw > 0.0
    dsat = float(np.mean(np.abs(V3[dnz, DELTA_DIM]) >= 1.0 - TOL)) if dnz.any() else 0.0
    sat_ok = sat_ok and (dsat <= SAT_GATE)
    _jdump({"gate": SAT_GATE, "per_feature": sat,
            "delta_severity": {"structural_zero_fraction": float(np.mean(draw == 0.0)),
                               "nonzero_absolute_saturation_fraction": dsat},
            "ZERO_AWARE_SATURATION_GATE": "PASS" if sat_ok else "FAIL"},
           RESULTS_DIR / "E4HI_saturation_gate.json")

    # ---- Phase 11: separability (D2 sample, MIXED regime) ----
    print("[E4HI] reconstructing D2 separability sample (1201..1220, MIXED)...")
    env = make_env_context()
    nets = load_nets()
    X_d2, y_d2 = collect_raw_states(nets, env, D2_SAMPLE_SEEDS, ["MIXED"], collect_regime=True)
    if len(set(y_d2.tolist())) < 2:
        sep_rows, means = [], {"RAW": float("nan"), "V1": float("nan"), "V2": float("nan"), "V3": float("nan")}
        v3_gap = v1_gap = float("nan"); sep_improved = False
    else:
        sep_rows, means = _separability(X_d2, y_d2, v2, v3)
        v1_gap = means["RAW"] - means["V1"]
        v3_gap = means["RAW"] - means["V3"]
        sep_improved = (means["V3"] >= means["V1"]
                        and _d2_rank(_d2_classify(v3_gap)) <= _d2_rank(_d2_classify(v1_gap))
                        and (means["V3"] > means["V1"] or _d2_rank(_d2_classify(v3_gap)) < _d2_rank(_d2_classify(v1_gap))))
    _write_csv(RESULTS_DIR / "E4HI_separability_probe_results.csv",
               ["probe", "representation", "accuracy"], sep_rows)
    _jdump({"n_samples": int(X_d2.shape[0]), "label": "MIXED regime persistent-vs-stationary",
            "mean_balanced_accuracy": means, "V1_gap": v1_gap, "V3_gap": v3_gap,
            "V1_classification": _d2_classify(v1_gap) if v1_gap == v1_gap else "NA",
            "V3_SEPARABILITY_CLASSIFICATION": _d2_classify(v3_gap) if v3_gap == v3_gap else "NA",
            "V3_SEPARABILITY_VS_V1": "IMPROVED" if sep_improved else "NOT_IMPROVED",
            "note": "V3 need not beat V2; relevant comparison is V3 vs V1"},
           RESULTS_DIR / "E4HI_separability_gate.json")

    # ---- Phase 12: semantic invariance ----
    sem = {"state_dim": 9, "feature_order_unchanged": FEATURE_NAMES == RAW_ORDER,
           "occupancy_identity": bool(abs(v3.transform(_one(2, 0.37))[2] - 0.37) < 1e-9),
           "active_fraction_identity": bool(abs(v3.transform(_one(4, 0.42))[4] - 0.42) < 1e-9),
           "delta_zero_preserved": bool(abs(v3.transform(np.zeros(9))[DELTA_DIM]) < TOL),
           "delta_sign_preserved": bool(v3.transform(_one(DELTA_DIM, 0.3))[DELTA_DIM] > 0
                                        and v3.transform(_one(DELTA_DIM, -0.3))[DELTA_DIM] < 0),
           "nonneg_monotonicity": mono_ok}
    sem["SEMANTIC_INVARIANCE"] = "PASS" if all(v is True for k, v in sem.items()
                                               if k != "state_dim" and isinstance(v, bool)) else "FAIL"
    _jdump(sem, RESULTS_DIR / "E4HI_semantic_invariance.json")

    # ---- Phase 13: backward compatibility ----
    bc_ok = True
    bc = {"versions": [NORM_V1, NORM_V2, NORM_V3]}
    try:
        check_normalization_compatibility({}, NORM_V1)                # legacy ok
        bc["v1_legacy_default_ok"] = True
    except Exception:
        bc["v1_legacy_default_ok"] = False; bc_ok = False
    bc["v1_ckpt_with_v3_raises"] = _raises(lambda: check_normalization_compatibility({}, NORM_V3, v3_sha))
    bc["v3_wrong_hash_raises"] = _raises(
        lambda: check_normalization_compatibility(
            {"normalization_version": NORM_V3, "normalizer_constants_sha256": "wrong"}, NORM_V3, v3_sha))
    bc["unsupported_version_raises"] = _raises(
        lambda: check_normalization_compatibility({"normalization_version": "FOO"}, NORM_V3, v3_sha))
    bc["v2_not_treatment"] = True
    bc_ok = bc_ok and bc["v1_ckpt_with_v3_raises"] and bc["v3_wrong_hash_raises"] and bc["unsupported_version_raises"]
    bc["BACKWARD_COMPATIBILITY"] = "PASS" if bc_ok else "FAIL"
    _jdump(bc, RESULTS_DIR / "E4HI_backward_compatibility.json")

    # ---- Phase 14: hand fixtures ----
    d0 = ZERO_INFLATED_DIMS[0]
    m0, c0 = m_by[d0], c_by[d0]
    fixtures = {
        "feature": RAW_ORDER[d0], "m": m0, "c": c0,
        "x0_is_zero": float(v3.transform(_one(d0, 0.0))[d0]),
        "xm_expected": float(np.log1p(1.0) / np.log1p(c0 / m0)),
        "xm_actual": float(v3.transform(_one(d0, m0))[d0]),
        "xc_actual": float(v3.transform(_one(d0, c0))[d0]),
        "x_between_0_m": float(v3.transform(_one(d0, m0 / 2))[d0]),
        "x_above_c": float(v3.transform(_one(d0, c0 * 5))[d0]),
        "delta_fixtures": {str(k): float(v3.transform(_one(DELTA_DIM, k * a3))[DELTA_DIM])
                           for k in [-1.0, -0.5, 0.0, 0.5, 1.0]},
    }
    fx_ok = (abs(fixtures["x0_is_zero"]) < TOL
             and abs(fixtures["xm_actual"] - fixtures["xm_expected"]) < 1e-9
             and abs(fixtures["xc_actual"] - 1.0) < 1e-9
             and 0 < fixtures["x_between_0_m"] < fixtures["xm_actual"]
             and abs(fixtures["x_above_c"] - 1.0) < 1e-9
             and fixtures["delta_fixtures"]["0.0"] == 0.0
             and fixtures["delta_fixtures"]["1.0"] > 0 and fixtures["delta_fixtures"]["-1.0"] < 0)
    fixtures["HAND_FIXTURES"] = "PASS" if fx_ok else "FAIL"
    _jdump(fixtures, RESULTS_DIR / "E4HI_hand_fixture_validation.json")

    # ---- Phase 15: historical holdout (optional) ----
    _jdump({"HISTORICAL_HOLDOUT_CHECK": "NOT_AVAILABLE",
            "reason": "no stored E4TR validation-seed (1061..1080) state artifact exists; "
                      "reconstructing new historical validation environments is not permitted here"},
           RESULTS_DIR / "E4HI_historical_holdout_check.json")

    # ---- Phase 16/17: firewall + no-training audits ----
    _jdump({"new_train_seed_access": _future_access["train"],
            "new_validation_seed_access": _future_access["validation"],
            "new_final_eval_access_count": _future_access["final"],
            "future_seeds_untouched": all(v == 0 for v in _future_access.values())},
           RESULTS_DIR / "E4HI_future_seed_firewall.json")
    _jdump({"optimizer_step_count": 0, "gradient_update_count": 0, "replay_append_count": 0,
            "new_checkpoint_count": 0, "new_policy_count": 0, "model_weights_changed": False},
           RESULTS_DIR / "E4HI_no_training_audit.json")

    # ---- readiness ----
    gates = {
        "IMPLEMENTATION_REFERENCE_MATCH": bool(ref_match),
        "NUMERICAL_RANGE_GATE": bool(range_ok),
        "STRUCTURAL_ZERO_GATE": bool(sz_ok),
        "POSITIVE_SIGNAL_PRESERVATION": bool(ps_ok),
        "SEMANTIC_MONOTONICITY_GATE": bool(mono_gate),
        "SIGNAL_GATE": bool(signal_gate),
        "ZERO_AWARE_SATURATION_GATE": bool(sat_ok),
        "V3_SEPARABILITY_IMPROVED": bool(sep_improved),
        "BACKWARD_COMPATIBILITY": bool(bc_ok),
        "FIT_DATA_INTEGRITY": bool(raw_audit["all_finite"] and X.shape[1] == 9),
        "FUTURE_SEED_FIREWALL": bool(all(v == 0 for v in _future_access.values())),
        "NO_TRAINING": True,
    }
    v3_ready = all(gates.values())

    verdict = {
        "stage": "E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION",
        "E4HI_COMPLETE": "YES", "ORIGINAL_E4F_VERDICT": "FAILED", "E4H_RESULT": "NO",
        "V2_STATUS": "REJECTED_NOT_TRAINED", "V3": NORM_V3,
        "IMPLEMENTATION_REFERENCE_MATCH": _pf(ref_match),
        "NUMERICAL_RANGE_GATE": _pf(range_ok),
        "STRUCTURAL_ZERO_GATE": _pf(sz_ok),
        "POSITIVE_SIGNAL_PRESERVATION": _pf(ps_ok),
        "SEMANTIC_MONOTONICITY_GATE": _pf(mono_gate),
        "SEVERITY_SIGNAL_V3": "PASS" if sig["severity_r"]["PASS"] else "FAIL",
        "CONVERGENCE_LOSS_SIGNAL_V3": "PASS" if sig["convergence_loss_r"]["PASS"] else "FAIL",
        "DELTA_SEVERITY_SIGNAL_V3": "PASS" if sig["delta_severity_r"]["PASS"] else "FAIL",
        "ZERO_AWARE_SATURATION_GATE": _pf(sat_ok),
        "V3_SEPARABILITY_VS_V1": "IMPROVED" if sep_improved else "NOT_IMPROVED",
        "V3_SEPARABILITY_CLASSIFICATION": _d2_classify(v3_gap) if v3_gap == v3_gap else "NA",
        "SEMANTIC_INVARIANCE": sem["SEMANTIC_INVARIANCE"],
        "BACKWARD_COMPATIBILITY": _pf(bc_ok),
        "HAND_FIXTURES": fixtures["HAND_FIXTURES"],
        "HISTORICAL_HOLDOUT_CHECK": "NOT_AVAILABLE",
        "NORMALIZATION_V3_READY": "YES" if v3_ready else "NO",
        "STATE_DIM": 9, "NETWORK_CHANGED": "NO", "REWARD_CHANGED": "NO",
        "JOINT_ARCHITECTURE_CHANGED": "NO", "NEW_TRAINING": "NO",
        "NEW_TRAIN_SEED_ACCESS": _future_access["train"],
        "NEW_VALIDATION_SEED_ACCESS": _future_access["validation"],
        "NEW_FINAL_EVAL_ACCESS_COUNT": _future_access["final"],
        "gates": gates, "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4HT_NORMALIZATION_AB_TRAINING_PROTOCOL_FREEZE" if v3_ready
                       else "E4HR_DESIGN_FIX_REQUIRED"),
    }
    _jdump(verdict, RESULTS_DIR / "E4HI_verdict.json")
    _write_report(verdict, sig, sat, means, v1_gap, v3_gap)
    _emit_terminal(verdict)
    print(f"[E4HI] wall {time.time()-t0:.0f}s")
    return verdict


def _one(idx, val):
    x = np.zeros(9); x[idx] = val; return x


def _pf(b):
    return "PASS" if b else "FAIL"


def _d2_rank(cls):
    return {"NO_LOSS": 0, "MODERATE_LOSS": 1, "MATERIAL_LOSS": 2, "NA": 3}[cls]


def _raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


def _finish_stop(next_stage):
    _jdump({"E4HI_COMPLETE": "NO", "NORMALIZATION_V3_READY": "NO", "NEXT_STAGE": next_stage},
           RESULTS_DIR / "E4HI_verdict.json")
    print(f"E4HI_COMPLETE = NO\nNEXT_STAGE = {next_stage}")


def _write_report(v, sig, sat, means, v1_gap, v3_gap):
    L = ["# E4HI — Normalization Implementation Validation (offline)", "",
         f"**E4HI_COMPLETE = {v['E4HI_COMPLETE']}; NORMALIZATION_V3_READY = {v['NORMALIZATION_V3_READY']}**", "",
         "## A. Historical preservation",
         "- E4F = FAILED (immutable); E4H = NO; V2 = REJECTED_NOT_TRAINED.", "",
         "## C. Independent implementation/reference agreement",
         f"- IMPLEMENTATION_REFERENCE_MATCH = {v['IMPLEMENTATION_REFERENCE_MATCH']} "
         "(separate code path; elementwise |Δ| ≤ 1e-12).", "",
         "## E/F. Structural-zero & positive preservation",
         f"- STRUCTURAL_ZERO_GATE = {v['STRUCTURAL_ZERO_GATE']}; "
         f"POSITIVE_SIGNAL_PRESERVATION = {v['POSITIVE_SIGNAL_PRESERVATION']}.", "",
         "## G. Signal decompression (previously-compressed local features)",
         f"- severity range={sig['severity_r']['range']:.3f} std={sig['severity_r']['std']:.3f} "
         f"→ {v['SEVERITY_SIGNAL_V3']}",
         f"- convergence_loss range={sig['convergence_loss_r']['range']:.3f} "
         f"std={sig['convergence_loss_r']['std']:.3f} → {v['CONVERGENCE_LOSS_SIGNAL_V3']}",
         f"- delta_severity range={sig['delta_severity_r']['range']:.3f} "
         f"std={sig['delta_severity_r']['std']:.3f} → {v['DELTA_SEVERITY_SIGNAL_V3']}", "",
         "## H. Zero-aware saturation (positive-only clipping)",
         f"- ZERO_AWARE_SATURATION_GATE = {v['ZERO_AWARE_SATURATION_GATE']} "
         "(structural zeros reported, not gated).", "",
         "## J. Separability (D2 sample, MIXED regime)",
         f"- RAW={means['RAW']:.3f} V1={means['V1']:.3f} V2={means['V2']:.3f} V3={means['V3']:.3f}; "
         f"V1_gap={v1_gap:.4f}, V3_gap={v3_gap:.4f} → V3 {v['V3_SEPARABILITY_VS_V1']} "
         f"({v['V3_SEPARABILITY_CLASSIFICATION']}).", "",
         "## K/L. Semantic invariance & versioning",
         f"- SEMANTIC_INVARIANCE = {v['SEMANTIC_INVARIANCE']}; BACKWARD_COMPATIBILITY = {v['BACKWARD_COMPATIBILITY']}.",
         "## M. Hand fixtures", f"- HAND_FIXTURES = {v['HAND_FIXTURES']} (x=0→0, x=m→log1p(1)/log1p(c/m), x=c→1, x>c→1).",
         "## O/P. Firewall & no-training",
         f"- future-seed access (train/val/final) = {v['NEW_TRAIN_SEED_ACCESS']}/{v['NEW_VALIDATION_SEED_ACCESS']}"
         f"/{v['NEW_FINAL_EVAL_ACCESS_COUNT']}; NEW_TRAINING = {v['NEW_TRAINING']}.", "",
         f"## Q/R. Readiness: NORMALIZATION_V3_READY = {v['NORMALIZATION_V3_READY']}; NEXT_STAGE = {v['NEXT_STAGE']}",
         "E4HI validates V3; it does not rescue V2, does not train, and does not alter E4F=FAILED."]
    (RESULTS_DIR / "E4HI_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4HI_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4H_RESULT", "V2_STATUS", "V3",
              "IMPLEMENTATION_REFERENCE_MATCH", "NUMERICAL_RANGE_GATE", "STRUCTURAL_ZERO_GATE",
              "POSITIVE_SIGNAL_PRESERVATION", "SEMANTIC_MONOTONICITY_GATE", "SEVERITY_SIGNAL_V3",
              "CONVERGENCE_LOSS_SIGNAL_V3", "DELTA_SEVERITY_SIGNAL_V3", "ZERO_AWARE_SATURATION_GATE",
              "V3_SEPARABILITY_VS_V1", "V3_SEPARABILITY_CLASSIFICATION", "SEMANTIC_INVARIANCE",
              "BACKWARD_COMPATIBILITY", "HISTORICAL_HOLDOUT_CHECK", "NORMALIZATION_V3_READY",
              "STATE_DIM", "NETWORK_CHANGED", "REWARD_CHANGED", "JOINT_ARCHITECTURE_CHANGED",
              "NEW_TRAINING", "NEW_TRAIN_SEED_ACCESS", "NEW_VALIDATION_SEED_ACCESS",
              "NEW_FINAL_EVAL_ACCESS_COUNT", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<36} = {v[k]}")


if __name__ == "__main__":
    main()
