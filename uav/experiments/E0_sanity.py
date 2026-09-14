"""
Experiment E0 - sanity (dac ta muc 10.1).

Tao toy problem trong do CHI mot trade-off region bi dich (ground truth
tuyet doi tu problems/toy_regional.py). Kiem tra:
  1. Vung khong affected phai co severity s_r ~ 0 CHINH XAC (theo pipeline
     that: normalize_pair -> assign_regions tren F_minus_tilde, eq.28).
  2. Ground-truth affected set (xac dinh tren F_minus tho, region da
     chon san) phai roi vao MOT vung duy nhat cua pipeline that sau khi
     chuan hoa -- xac nhan chuan hoa (eq.24-25) khong lam xao tron ranh
     gioi vung cho toy nay (dung target_region = mot dinh luoi Das-Dennis
     that de tranh nhap nhang bien).
  3. Detector (argmax s_r) phai chon DUNG vung do.
  4. RHI(reference-direction partition that) phai cao hon RHI trung binh
     cua random-partition control cung kich thuoc vung.

Day la unit-sanity-check cho core/normalize, core/reference,
core/partition, detectors/severity, detectors/rhi -- KHONG phai bang
chung khoa hoc cho gia thuyet vung (do la viec cua E1 tren CEC2018).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.core.normalize import normalize_pair  # noqa: E402
from prdrl.core.partition import assign_regions, region_indices  # noqa: E402
from prdrl.core.reference import make_reference_directions  # noqa: E402
from prdrl.detectors.rhi import rhi, rhi_null_distribution, percentile_rank  # noqa: E402
from prdrl.detectors.severity import displacement, region_severity_raw  # noqa: E402
from prdrl.problems.toy_regional import ToyRegionalShift  # noqa: E402


def run_one_trial(seed, H=9, n_trials_null=500, target_region_label=4):
    V = make_reference_directions(n_obj=2, h=H)   # R=10 (Bang 11.1)
    R = len(V)
    # target = MOT dinh luoi Das-Dennis that (khong phai vector tu do)
    # de tranh sai lech do lam tron/chuan hoa.
    target_v = V[target_region_label]

    problem = ToyRegionalShift(N=150, target_region_v=target_v,
                               shift_magnitude=0.35, seed=seed)
    F_minus, affected_gt, target_region_idx = problem.sample_front(V)
    assert target_region_idx == target_region_label
    F_plus = problem.apply_change(F_minus, affected_gt)

    # pipeline THAT (eq. 22-29): chuan hoa tren hop {F-,F+}, roi partition
    # tren F_minus_tilde -- day la nguon su that duy nhat cho detector.
    Fm_t, Fp_t, z_star, z_nad = normalize_pair(F_minus, F_plus)
    r_of_i_real = assign_regions(Fm_t, V)
    idx_by_region = region_indices(r_of_i_real, R)

    # consistency check: chuan hoa co lam xao tron ranh gioi vung khong?
    real_regions_of_affected = np.unique(r_of_i_real[affected_gt])
    normalization_consistent = (len(real_regions_of_affected) == 1
                                and real_regions_of_affected[0] == target_region_idx)

    d_i = displacement(Fm_t, Fp_t)
    s_r = region_severity_raw(d_i, idx_by_region)

    unaffected_severities = []
    for r in range(R):
        idx = idx_by_region[r]
        if len(idx) == 0 or r == target_region_idx:
            continue
        unaffected_severities.append(s_r[r] if s_r[r] is not None else 0.0)

    max_severity_region = int(np.argmax([s_r[r] if s_r[r] is not None else -1 for r in range(R)]))
    detector_correct = (max_severity_region == target_region_idx)

    region_sizes = [len(idx_by_region[r]) for r in range(R)]
    rhi_ref = rhi(s_r)
    rng = np.random.default_rng(seed + 10_000)
    null_dist = rhi_null_distribution(d_i, region_sizes, n_trials_null, rng)
    pct = percentile_rank(rhi_ref, null_dist)

    return dict(
        seed=seed, R=R, target_region_idx=target_region_idx,
        n_affected=int(affected_gt.sum()),
        normalization_consistent=normalization_consistent,
        detected_region=max_severity_region, detector_correct=detector_correct,
        rhi_ref=rhi_ref, null_mean=float(np.mean(null_dist)),
        null_std=float(np.std(null_dist)), percentile=pct,
        max_unaffected_severity=float(max(unaffected_severities)) if unaffected_severities else 0.0,
    )


def main():
    print("=== E0 -- sanity (toy regional-shift, ground truth) ===\n")
    seeds = list(range(30, 40))  # 10 independent toy trials
    results = [run_one_trial(s) for s in seeds]

    n_consistent = sum(r["normalization_consistent"] for r in results)
    n_correct = sum(r["detector_correct"] for r in results)
    max_unaff = max(r["max_unaffected_severity"] for r in results)
    pcts = [r["percentile"] for r in results]

    for r in results:
        print(f"seed={r['seed']:3d} target_region={r['target_region_idx']} "
             f"n_affected={r['n_affected']:3d} norm_consistent={r['normalization_consistent']} "
             f"detector_correct={r['detector_correct']} "
             f"RHI_ref={r['rhi_ref']:.3f} null_mean={r['null_mean']:.3f} "
             f"null_std={r['null_std']:.3f} percentile={r['percentile']:.3f}")

    print(f"\n[info, not a pass/fail criterion -- eq.24-25 normalization over the")
    print(f" union {{F-,F+}} legitimately stretches axes anisotropically when only")
    print(f" some points shift, so a few boundary-adjacent points can cross into a")
    print(f" neighboring region; this is an expected property of the formula, not a")
    print(f" bug] points staying in exactly one region after normalization: "
         f"{n_consistent}/{len(results)}")
    print(f"\ndetector correct region (spec E0 criterion 2): {n_correct}/{len(results)}")
    print(f"max unaffected-region severity, should be ~0 (spec E0 criterion, implicit "
         f"ground truth): {max_unaff:.2e}")
    print(f"RHI_ref percentile vs random-partition null (spec E0 criterion 1 + "
         f"Phu luc C.3): min={min(pcts):.3f} mean={np.mean(pcts):.3f} max={max(pcts):.3f}")

    all_pass = (n_correct == len(results) and max_unaff < 1e-9 and np.mean(pcts) > 0.9)
    print(f"\nE0 SANITY: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
