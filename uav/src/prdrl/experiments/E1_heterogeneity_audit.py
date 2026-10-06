"""
E1c — CEC2018 control (Phase 9, E1R).

REFRAME (E1R): file nay tung la "E1 heterogeneity audit" voi mot gate
cung nhi phan "10/14 DF phai co median percentile > 0.95". Gate do
KHONG con phu hop ve mat khoa hoc (xem PHASE 9/10 trong task E1R): CEC2018
gio la CONTROL/generalization result, khong phai gate quyet dinh
song-con cua toan bo y tuong regional. Verdict moi la mo ta
(CEC-LOW/MIXED/HIGH), khong phai pass/fail cung. Xem thêm
E1a_controlled_heterogeneity.py (Gate M0, co ground truth) va
E1b_sdp_heterogeneity.py (Gate SDP, benchmark chinh) -- ca hai moi la
can cu quyet dinh co proceed sang E2 hay khong.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, "src")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.core.seeding import deterministic_rng
from prdrl.detectors.severity import severity
from prdrl.detectors.rhi import rhi, rhi_null


# =====================================================================
# CEC2018 DF adapter — bạn cần implement evaluate_df(df_id, X, t)
# trả về ma trận objectives (N, n_obj).
# Nếu đã có sẵn thư viện CEC2018 thì thay hàm này.
# =====================================================================

def get_df_config(df_id):
    """Cấu hình từng DF: số mục tiêu, số biến, bounds.

    FIX (Phase 1, E1R): trước đây hardcode lower=zeros(D), upper=ones(D)
    cho MỌI DF -- SAI vì nhiều DF không có bounds trong [0,1] (vd DF3
    x1∈[-1,2], DF7 x0∈[1,4], DF10 x2..∈[-1,1]). Giờ đọc bounds THẬT từ
    adapter get_df_bounds() (prdrl.problems.cec2018), không duplicate
    bảng bounds trong file experiment này.
    """
    n_obj = get_n_obj(df_id)
    D = 10
    lower, upper = get_df_bounds(df_id, D=D)
    return n_obj, D, lower, upper


def compute_time(generation, n_t=10, tau_t=10):
    """Tham số thời gian CEC2018: t = (1/n_t) * floor(gen / tau_t)."""
    return (1.0 / n_t) * np.floor(generation / tau_t)


# evaluate_df / get_true_pf: implement thật trong
# src/prdrl/problems/cec2018.py (adapter quanh pymoo.problems.dynamic.df
# -- xem docstring file do). Import trực tiếp, không viết lại ở đây.
from prdrl.problems.cec2018 import evaluate_df, get_true_pf, get_df_bounds, get_n_obj  # noqa: E402


# =====================================================================
# Chạy NSGA-II trên một DF, thu population tại mỗi change event
# =====================================================================

class DFProblem(Problem):
    """Wrapper pymoo cho DF problem tại thời điểm t cố định."""
    def __init__(self, df_id, t, n_obj, D, lower, upper):
        super().__init__(n_var=D, n_obj=n_obj, xl=lower, xu=upper)
        self.df_id = df_id
        self.t = t

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = evaluate_df(self.df_id, X, self.t)


def run_nsga2_with_changes(df_id, n_changes=15, pop_size=100,
                           gen_per_change=10, n_t=10, tau_t=10, seed=42):
    """
    Chạy NSGA-II, thu thập (F_minus, F_plus) tại mỗi change event.

    FIX (Phase 8, E1R): truoc day moi change event tao NSGA2(pop_size=..)
    MOI (khong sampling=), tuc la RESET ve quan the ngau nhien hoan toan
    o MOI change event trong khi tu nhan la "tracking trajectory" --
    khong phai continuous trajectory that. Va dung res.X (chi tap
    nondominated, so luong co the < pop_size) lam X_pop ke tiep. Sua:
    (1) truyen sampling=X_pop de quan the hien tai la initial population
    cua lan tien hoa tiep theo (continuous trajectory that); (2) dung
    result.pop.get("X") (toan bo quan the, dung pop_size) chu khong
    phai result.X.

    Returns
    -------
    events : list of dict, mỗi dict chứa:
        F_minus  : objectives trước change (N, n_obj)
        F_plus   : objectives sau change (N, n_obj)
        X        : decision vectors (N, D)
    """
    rng = np.random.default_rng(seed)
    n_obj, D, lower, upper = get_df_config(df_id)

    # Khởi tạo population
    X_pop = rng.uniform(lower, upper, size=(pop_size, D))
    t_current = compute_time(0, n_t, tau_t)

    events = []
    gen = 0

    for change_idx in range(n_changes):
        # Chạy NSGA-II trong gen_per_change thế hệ tại t hiện tại,
        # TIẾP TỤC từ quan the hien co (khong reset ngau nhien).
        problem = DFProblem(df_id, t_current, n_obj, D, lower, upper)
        algo = NSGA2(pop_size=pop_size, sampling=X_pop)
        res = minimize(problem, algo, ('n_gen', gen_per_change),
                       seed=seed + change_idx, verbose=False)
        X_pop = np.asarray(res.pop.get("X"), dtype=float)
        gen += gen_per_change

        # Freeze: F_minus = objectives tại t hiện tại
        F_minus = evaluate_df(df_id, X_pop, t_current)

        # Đổi môi trường
        t_new = compute_time(gen, n_t, tau_t)
        if abs(t_new - t_current) < 1e-15:
            # Chưa đổi thực sự, bỏ qua
            continue

        # Re-evaluate cùng X tại t mới → F_plus
        F_plus = evaluate_df(df_id, X_pop, t_new)

        events.append({
            "F_minus": F_minus,
            "F_plus": F_plus,
            "X": X_pop.copy(),
            "t_old": t_current,
            "t_new": t_new,
            "change_idx": change_idx,
        })

        t_current = t_new

    return events


# =====================================================================
# E1c — CEC2018 control (descriptive, not a pass/fail gate)
# =====================================================================

def audit_one_df(df_id, seeds, n_changes=15, pop_size=100, K=200):
    """Chạy audit RHI trên một DF, nhiều seed. Trả về list of per-event
    dict (record) thay vì chỉ percentile, để phục vụ output JSONL/CSV
    (Phase 14) và occupancy diagnostic (Phase 12)."""
    n_obj, D, lower, upper = get_df_config(df_id)
    # FIX (Phase 2, E1R): doi ten bien H -> n_partitions cho ro rang --
    # day la tham so n_partitions cua Das-Dennis, KHONG phai so regions
    # R. R = len(V) duoc suy ra SAU khi goi reference(), khong duoc
    # gia dinh R == n_partitions.
    n_partitions = 9 if n_obj == 2 else 4
    V = reference(n_obj, n_partitions)
    R = len(V)

    records = []

    for seed in seeds:
        events = run_nsga2_with_changes(
            df_id, n_changes=n_changes, pop_size=pop_size, seed=seed
        )

        q_prev = 1.0
        for ev in events:
            F_tilde_minus, F_tilde_plus = normalize(ev["F_minus"], ev["F_plus"])
            F_tilde_minus = np.array(F_tilde_minus)
            F_tilde_plus = np.array(F_tilde_plus)

            region_ids = partition(F_tilde_minus, V)
            region_ids = np.array(region_ids)

            s_r, c_r, q_prev = severity(
                F_tilde_minus, F_tilde_plus, region_ids, R, q_prev
            )

            # FIX (Phase 11, E1R): rng tat dinh tu (seed, df_id, change_idx)
            # -- reproducible, khong con dung np.random khong seed ngam.
            null_rng = deterministic_rng(seed, df_id, ev["change_idx"])
            rhi_ref, rhi_null_mean, rhi_null_std, pct = rhi_null(
                F_tilde_minus, F_tilde_plus, region_ids, R, K=K, rng=null_rng
            )

            # Phase 12: occupancy / active-region diagnostic.
            active_mask = ~np.isnan(s_r)
            R_active = int(np.sum(active_mask))

            records.append({
                "df_id": df_id,
                "seed": seed,
                "change_idx": ev["change_idx"],
                "t_old": float(ev["t_old"]),
                "t_new": float(ev["t_new"]),
                "R_total": R,
                "R_active": R_active,
                "active_fraction": R_active / R,
                "rhi": float(rhi_ref),
                "null_mean": float(rhi_null_mean),
                "null_std": float(rhi_null_std),
                "percentile": float(pct),
            })

    return records


def summarize_df(df_id, records):
    pct = np.array([r["percentile"] for r in records])
    rhi_vals = np.array([r["rhi"] for r in records])
    active_frac = np.array([r["active_fraction"] for r in records])
    return {
        "df_id": df_id,
        "n_events": len(records),
        "median_rhi": float(np.median(rhi_vals)),
        "median_percentile": float(np.median(pct)),
        "mean_percentile": float(np.mean(pct)),
        "pct_above_95": float(np.mean(pct >= 0.95)),
        "median_active_fraction": float(np.median(active_frac)),
    }


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
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def cec_verdict(df_summaries):
    """Phase 10/17: CEC la CONTROL, verdict mo ta (LOW/MIXED/HIGH),
    KHONG phai gate quyet dinh, va KHONG duoc gop vao verdict SDP."""
    if len(df_summaries) == 0:
        return "CEC-LOW"
    frac_strong = np.mean([s["median_percentile"] >= 0.95 for s in df_summaries])
    if frac_strong >= 0.7:
        return "CEC-HIGH"
    if frac_strong <= 0.2:
        return "CEC-LOW"
    return "CEC-MIXED"


def run_e1(seeds=(1, 2, 3), n_changes=15, pop_size=100, out_dir="results"):
    print("=" * 70)
    print("E1c — CEC2018 control (descriptive, not a pass/fail gate)")
    print("    RHI of reference-direction partition vs random-partition null")
    print("=" * 70)

    all_records = []
    df_summaries = []

    t0 = time.time()

    for df_id in range(1, 15):
        n_obj = get_n_obj(df_id)
        print(f"\nDF{df_id} ({n_obj} obj) ...", end=" ", flush=True)
        try:
            records = audit_one_df(df_id, list(seeds), n_changes=n_changes,
                                   pop_size=pop_size)
        except NotImplementedError as e:
            print(f"SKIPPED — {e}")
            continue
        all_records.extend(records)
        summ = summarize_df(df_id, records)
        df_summaries.append(summ)
        print(f"median pct={summ['median_percentile']:.3f}  "
              f"events={summ['n_events']}  "
              f"active_frac={summ['median_active_fraction']:.2f}")

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"{'DF':<6} {'median RHI':>11} {'median pct':>12} {'%>=0.95':>8} "
          f"{'active_frac':>12} {'events':>8}")
    print("-" * 65)
    for s in df_summaries:
        print(f"DF{s['df_id']:<4} {s['median_rhi']:>11.4f} "
              f"{s['median_percentile']:>12.3f} "
              f"{s['pct_above_95']:>7.1%} "
              f"{s['median_active_fraction']:>12.2f} "
              f"{s['n_events']:>8}")

    n_strong = sum(1 for s in df_summaries if s["median_percentile"] >= 0.95)
    total = len(df_summaries)
    verdict = cec_verdict(df_summaries)

    print("-" * 65)
    print(f"DFs with median percentile >= 0.95: {n_strong}/{total}  "
          f"(descriptive only -- NOT a pass/fail gate, see Phase 9/10)")
    print(f"Time: {elapsed:.1f}s")
    print()
    print(f"CEC VERDICT: {verdict}  (reported separately, does NOT "
          f"override Gate M0 / Gate SDP -- see E1a/E1b)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 14: output files
    # ------------------------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_path / "E1c_cec2018_control.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")

    csv_path = out_path / "E1c_cec2018_control_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = ["df_id", "n_events", "median_rhi", "median_percentile",
                "mean_percentile", "pct_above_95", "median_active_fraction"]
        f.write(",".join(cols) + "\n")
        for s in df_summaries:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    report_path = out_path / "E1c_cec2018_control_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# E1c — CEC2018 Control\n\n")
        f.write("CEC2018 is a CONTROL/generalization result, not the "
                "sole gate for the regional-heterogeneity hypothesis. "
                "See E1a (Gate M0) and E1b (Gate SDP) for the decision "
                "evidence.\n\n")
        f.write(f"**CEC VERDICT: {verdict}**\n\n")
        f.write(f"DFs with median percentile >= 0.95: {n_strong}/{total}\n\n")
        f.write("| DF | median RHI | median pct | %>=0.95 | active_frac | events |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in df_summaries:
            f.write(f"| DF{s['df_id']} | {s['median_rhi']:.4f} | "
                    f"{s['median_percentile']:.3f} | {s['pct_above_95']:.1%} | "
                    f"{s['median_active_fraction']:.2f} | {s['n_events']} |\n")

    print(f"\nSaved: {jsonl_path}, {csv_path}, {report_path}")

    return {
        "df_summaries": df_summaries,
        "verdict": verdict,
        "n_strong": n_strong,
        "total": total,
    }


if __name__ == "__main__":
    run_e1()
