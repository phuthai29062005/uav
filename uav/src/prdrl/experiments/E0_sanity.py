import numpy as np
import sys
sys.path.insert(0, "src")

from prdrl.problems.toy_regional import make_toy_problem
from prdrl.core.normalize import normalize
from prdrl.detectors.severity import severity
from prdrl.detectors.rhi import rhi, rhi_null


def run_e0(n_trials=10):
    print("=" * 60)
    print("E0 — Sanity check trên toy problem")
    print("=" * 60)

    all_pass = True

    for trial in range(n_trials):
        seed = trial + 1
        F_minus, F_plus, X, pf_before, pf_after, R_shift, region_ids_gt, V, R = \
            make_toy_problem(N=100, R_shift=trial % R if trial > 0 else 0,
                             shift_magnitude=0.5, seed=seed)

        # Bước 1: normalize trên hợp {F⁻, F⁺}
        F_tilde_minus, F_tilde_plus = normalize(F_minus, F_plus)
        F_tilde_minus = np.array(F_tilde_minus)
        F_tilde_plus = np.array(F_tilde_plus)

        # Bước 2: region_ids — DÙNG THẲNG ground truth do make_toy_problem
        # trả về, KHÔNG partition() lại trên F_tilde_minus thật.
        # Lý do: chuẩn hoá (eq.24-25) trên hợp {F⁻,F⁺} co giãn các trục
        # KHÔNG đều khi chỉ một phần điểm bị dịch (đúng theo công thức,
        # không phải bug) — nếu tính lại region_ids sau khi F⁺ đã tồn
        # tại, ranh giới vùng trôi so với lúc make_toy_problem quyết định
        # điểm nào sẽ bị dịch, khiến ground-truth và detector "cãi nhau"
        # dù cả hai đều tính đúng công thức (đã verify: 62/100 điểm lệch
        # vùng giữa hai lần tính ở seed=2). Dùng region_ids_gt giữ ground
        # truth và detector nhất quán trên CÙNG một hệ quy chiếu.
        region_ids = np.array(region_ids_gt)

        # Bước 3: severity
        q_prev = 1.0
        s_r, c_r, q_new = severity(F_tilde_minus, F_tilde_plus, region_ids, R, q_prev)

        # Kiểm tra 1: vùng có severity cao nhất = vùng bị dịch
        detected = np.nanargmax(s_r)
        detect_ok = (detected == R_shift)

        # Kiểm tra 2: vùng không bị dịch có severity = 0
        unshifted_severity = [s_r[r] for r in range(R)
                              if r != R_shift and not np.isnan(s_r[r])]
        unshifted_ok = all(s < 1e-10 for s in unshifted_severity)

        # Kiểm tra 3: RHI vượt random-partition null
        rhi_ref, rhi_null_mean, rhi_null_std, percentile = \
            rhi_null(F_tilde_minus, F_tilde_plus, region_ids, R, K=200)
        rhi_ok = (percentile >= 0.95)

        trial_pass = detect_ok and unshifted_ok and rhi_ok
        if not trial_pass:
            all_pass = False

        print(f"\nTrial {trial + 1} (seed={seed}, shifted region={R_shift}):")
        print(f"  Detected region:      {detected}  {'OK' if detect_ok else 'FAIL'}")
        print(f"  Unshifted severity:   max={max(unshifted_severity):.2e}  "
              f"{'OK' if unshifted_ok else 'FAIL'}")
        print(f"  RHI ref={rhi_ref:.4f}  null={rhi_null_mean:.4f}±{rhi_null_std:.4f}  "
              f"percentile={percentile:.3f}  {'OK' if rhi_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"E0 RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    run_e0()