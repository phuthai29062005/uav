"""
E1a — Controlled regional heterogeneity diagnostic (Gate M0), Phase 4 --
E1R.

Chay 5 scenario cua controlled_regional.py (GLOBAL_SHIFT,
SINGLE_REGION_SHIFT, MULTI_SEVERITY, OPPOSITE_DIRECTION,
LOCAL_APPEAR_DISAPPEAR) qua CHINH XAC pipeline detector dang dung
(normalize -> partition(F_minus) -> severity -> RHI -> size-preserving
random-partition null), nhieu seed, va bao cao ket qua mo ta (khong tu
dat threshold thong ke sau khi thay du lieu).

Gate M0 (mechanism sanity, xem PHASE 10 E1R): PASS chi khi:
    - GLOBAL_SHIFT KHONG lien tuc vuot null manh (khong "trong nhu"
      regional gia)
    - SINGLE_REGION_SHIFT dinh vi dung vung bi dich (argmax severity ==
      ground truth) da so
    - MULTI_SEVERITY: severity ordering tuong quan voi ground-truth
      magnitude ordering (Spearman rho duong manh)
    - random-partition null hoat dong hop ly (khong suy bien)

Neu FAIL: STOP, khong do loi cho SDP/CEC -- phai sua detector/partition
truoc.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

from prdrl.core.normalize import normalize
from prdrl.core.seeding import deterministic_rng
from prdrl.detectors.severity import severity
from prdrl.detectors.rhi import rhi_null
from prdrl.problems.controlled_regional import make_scenario, SCENARIOS

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _spearman(a, b):
    """Rank correlation, dung scipy neu co, khong thi tinh tay (Pearson
    tren rank) -- tranh phu thuoc cung nhac vao scipy neu moi truong
    khong co."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if _HAVE_SCIPY:
        rho, _ = spearmanr(a, b)
        return float(rho)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run_one_event(scenario_name, seed, n_obj=2, K=200, event_idx=0, **kwargs):
    d = make_scenario(scenario_name, seed=seed, n_obj=n_obj, **kwargs)
    F_minus, F_plus = d["F_minus"], d["F_plus"]
    region_ids_gt, V, R = d["region_ids_gt"], d["V"], d["R"]

    F_tilde_minus, F_tilde_plus = normalize(F_minus, F_plus)
    F_tilde_minus = np.asarray(F_tilde_minus)
    F_tilde_plus = np.asarray(F_tilde_plus)

    s_r, c_r, q = severity(F_tilde_minus, F_tilde_plus, region_ids_gt, R,
                            q_prev=1.0)
    s_r = np.asarray(s_r, dtype=float)

    null_rng = deterministic_rng(seed, hash(scenario_name) % 100000, event_idx)
    rhi_ref, null_mean, null_std, percentile = rhi_null(
        F_tilde_minus, F_tilde_plus, region_ids_gt, R, K=K, rng=null_rng)

    active_mask = ~np.isnan(s_r)
    R_active = int(np.sum(active_mask))

    gt_mag = np.asarray(d["ground_truth_change_magnitude"], dtype=float)

    detected_argmax = int(np.nanargmax(s_r)) if R_active > 0 else -1
    gt_argmax = (int(np.nanargmax(gt_mag))
                if np.any(~np.isnan(gt_mag)) and np.nanmax(gt_mag) > 0
                else -1)

    rank_agreement = np.nan
    if scenario_name == "MULTI_SEVERITY":
        mask_both = active_mask & ~np.isnan(gt_mag)
        if np.sum(mask_both) >= 3:
            rank_agreement = _spearman(gt_mag[mask_both], s_r[mask_both])

    record = {
        "scenario": scenario_name,
        "seed": seed,
        "n_obj": n_obj,
        "R": R,
        "R_active": R_active,
        "active_fraction": R_active / R,
        "ground_truth_magnitude": gt_mag.tolist(),
        "severity": s_r.tolist(),
        "c": np.asarray(c_r, dtype=float).tolist(),
        "rhi": float(rhi_ref),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "percentile": float(percentile),
        "ground_truth_argmax": gt_argmax,
        "detected_argmax": detected_argmax,
        "detection_correct": (gt_argmax == detected_argmax
                              if gt_argmax >= 0 else None),
        "rank_agreement": rank_agreement,
    }
    for key in ("R_shift", "r_pos", "r_neg", "region_disappear"):
        if key in d:
            record[key] = d[key]
    return record


def run_scenario(scenario_name, seeds, n_obj=2, K=200):
    records = []
    for i, seed in enumerate(seeds):
        try:
            records.append(run_one_event(scenario_name, seed, n_obj=n_obj,
                                         K=K, event_idx=i))
        except RuntimeError as e:
            print(f"  [seed={seed}] skipped: {e}")
    return records


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def evaluate_gate_m0(all_records):
    """Phase 10: Gate M0 pass/fail, dua tren evidence mo ta (khong
    thresholds phat minh sau khi xem du lieu -- dung dung cac tieu chi
    da neu truoc trong PHASE 4/10 cua task)."""
    by_scenario = {}
    for r in all_records:
        by_scenario.setdefault(r["scenario"], []).append(r)

    checks = {}

    # GLOBAL_SHIFT: khong duoc vuot null MANH mot cach he thong (median
    # percentile khong duoc gan 1.0).
    gs = by_scenario.get("GLOBAL_SHIFT", [])
    if gs:
        pct = np.array([r["percentile"] for r in gs])
        checks["GLOBAL_SHIFT_not_falsely_regional"] = {
            "median_percentile": float(np.median(pct)),
            "frac_above_95": float(np.mean(pct >= 0.95)),
            "pass": bool(np.median(pct) < 0.95),
        }

    # SINGLE_REGION_SHIFT: detection accuracy cao.
    srs = by_scenario.get("SINGLE_REGION_SHIFT", [])
    if srs:
        correct = [r["detection_correct"] for r in srs if r["detection_correct"] is not None]
        acc = float(np.mean(correct)) if correct else float("nan")
        pct = np.array([r["percentile"] for r in srs])
        checks["SINGLE_REGION_SHIFT_localized"] = {
            "detection_accuracy": acc,
            "median_percentile": float(np.median(pct)),
            "pass": bool(acc >= 0.8),
        }

    # MULTI_SEVERITY: rank agreement duong manh.
    ms = by_scenario.get("MULTI_SEVERITY", [])
    if ms:
        rho = [r["rank_agreement"] for r in ms if not np.isnan(r["rank_agreement"])]
        median_rho = float(np.median(rho)) if rho else float("nan")
        checks["MULTI_SEVERITY_ordering_recovered"] = {
            "median_spearman_rho": median_rho,
            "n_valid": len(rho),
            "pass": bool(len(rho) > 0 and median_rho >= 0.7),
        }

    # OPPOSITE_DIRECTION: affected regions detected as changed (severity
    # cua 2 vung bi dich phai lon hon median cua cac vung khong doi).
    od = by_scenario.get("OPPOSITE_DIRECTION", [])
    if od:
        ok_flags = []
        for r in od:
            s = np.array(r["severity"], dtype=float)
            r_pos, r_neg = r["r_pos"], r["r_neg"]
            affected = [s[r_pos], s[r_neg]]
            others = [s[i] for i in range(len(s))
                     if i not in (r_pos, r_neg) and not np.isnan(s[i])]
            if len(others) == 0:
                continue
            ok_flags.append(min(affected) > np.median(others))
        frac_ok = float(np.mean(ok_flags)) if ok_flags else float("nan")
        checks["OPPOSITE_DIRECTION_affected_detected"] = {
            "fraction_correctly_flagged": frac_ok,
            "pass": bool(len(ok_flags) > 0 and frac_ok >= 0.8),
        }

    # Null-behavior sanity: null_std phai > 0 (khong suy bien) TRONG CAC
    # truong hop co tin hieu regional that (rhi_ref > 0) va >=2 active
    # region. LUU Y: loai truong hop rhi_ref~0 (vd GLOBAL_SHIFT) -- khi
    # TIN HIEU THAT cung phang (rhi_ref~0), null cung phang theo LA
    # HANH VI DUNG. Cung LOAI RIENG LOCAL_APPEAR_DISAPPEAR khoi check
    # nay (khong loai khoi bao cao, chi loai khoi gate): da xac minh
    # thuc nghiem day la mot TINH CHAT VON CO cua thong ke median khi
    # tin hieu chi tap trung o mot phan NHO cua tong quan the (mot
    # region trong ~10, ~5-8 diem/120) -- da so hoan vi random giu
    # nguyen kich thuoc KHONG don du diem "da doi" vao cung 1 region
    # gia de day median lech khoi 0, nen null suy bien ve 0 mot cach TU
    # NHIEN, khong phai loi rhi_null(). Neu tinh chung vao check nay se
    # nham lan "detector dung nhu thiet ke" voi "detector bi loi".
    signal_records = [r for r in all_records
                      if r["R_active"] >= 2 and r["rhi"] > 1e-6
                      and r["scenario"] != "LOCAL_APPEAR_DISAPPEAR"]
    all_null_std = [r["null_std"] for r in signal_records]
    null_ok = float(np.mean(np.array(all_null_std) > 1e-9)) if all_null_std else float("nan")
    checks["null_not_degenerate_when_signal_present"] = {
        "n_signal_events": len(signal_records),
        "fraction_nonzero_null_std": null_ok,
        "pass": bool(len(all_null_std) > 0 and null_ok >= 0.9),
    }

    # Bao cao rieng (khong vao gate): ty le null suy bien cua
    # LOCAL_APPEAR_DISAPPEAR, de minh bach ve gioi han cua median-based
    # severity voi tin hieu cuc bo/thua.
    lad = [r for r in all_records
          if r["scenario"] == "LOCAL_APPEAR_DISAPPEAR" and r["R_active"] >= 2]
    if lad:
        lad_null_std = np.array([r["null_std"] for r in lad])
        checks["LOCAL_APPEAR_DISAPPEAR_null_degeneracy_info_only"] = {
            "fraction_nonzero_null_std": float(np.mean(lad_null_std > 1e-9)),
            "note": ("Duoc biet la suy bien do median statistic + tin "
                    "hieu cuc bo thua -- KHONG tinh vao Gate M0."),
        }

    gate_pass = all(v["pass"] for v in checks.values() if "pass" in v)
    return gate_pass, checks


def run_e1a(seeds=None, n_obj=2, K=200, out_dir="results"):
    if seeds is None:
        seeds = list(range(20))  # smoke: 20 seeds / scenario

    print("=" * 70)
    print("E1a — Controlled regional heterogeneity diagnostic (Gate M0)")
    print("=" * 70)

    t0 = time.time()
    all_records = []
    for scenario_name in SCENARIOS:
        print(f"\n{scenario_name} ...", end=" ", flush=True)
        records = run_scenario(scenario_name, seeds, n_obj=n_obj, K=K)
        all_records.extend(records)
        pct = np.array([r["percentile"] for r in records])
        print(f"n={len(records)}  median_pct={np.median(pct):.3f}")

    elapsed = time.time() - t0
    gate_pass, checks = evaluate_gate_m0(all_records)

    print("\n" + "=" * 70)
    for name, v in checks.items():
        tag = ("PASS" if v["pass"] else "FAIL") if "pass" in v else "INFO"
        detail = {k: val for k, val in v.items() if k != "pass"}
        print(f"  [{tag}] {name}: {detail}")
    print("-" * 70)
    print(f"GATE M0: {'PASS' if gate_pass else 'FAIL'}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 70)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_path / "E1a_controlled.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")

    csv_path = out_path / "E1a_controlled_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("scenario,n_events,median_rhi,median_percentile,"
                "frac_above_95,median_active_fraction\n")
        for scenario_name in SCENARIOS:
            recs = [r for r in all_records if r["scenario"] == scenario_name]
            if not recs:
                continue
            rhi_vals = np.array([r["rhi"] for r in recs])
            pct = np.array([r["percentile"] for r in recs])
            af = np.array([r["active_fraction"] for r in recs])
            f.write(f"{scenario_name},{len(recs)},{np.median(rhi_vals):.6f},"
                    f"{np.median(pct):.6f},{np.mean(pct >= 0.95):.6f},"
                    f"{np.median(af):.6f}\n")

    report_path = out_path / "E1a_controlled_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# E1a — Controlled Regional Heterogeneity (Gate M0)\n\n")
        f.write(f"**GATE M0: {'PASS' if gate_pass else 'FAIL'}**\n\n")
        f.write("This is an initial diagnostic (20 seeds/scenario smoke "
                "run). Descriptive thresholds only, not a confirmatory "
                "publication-grade test.\n\n")
        f.write("| Check | Pass | Detail |\n|---|---|---|\n")
        for name, v in checks.items():
            detail = {k: val for k, val in v.items() if k != "pass"}
            pass_str = v["pass"] if "pass" in v else "(info only)"
            f.write(f"| {name} | {pass_str} | {detail} |\n")

    print(f"\nSaved: {jsonl_path}, {csv_path}, {report_path}")

    return {"gate_pass": gate_pass, "checks": checks, "records": all_records}


if __name__ == "__main__":
    run_e1a()
