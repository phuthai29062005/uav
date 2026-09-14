"""
Experiment E1 - heterogeneity audit (dac ta muc 10.2, Gate A1).

Tren CEC2018 DF1-DF14:
  1. partition theo reference directions (dong bang tai F_minus, eq.28-29);
  2. tinh s_r o tung change (eq.30-33, khong dung EMA lam gate quyet dinh
     -- Gate A1 dung s_r/RHI THO cua moi change de so voi null, EMA c_r
     la tin hieu cho state cua RL sau nay, khong phai cho audit nay);
  3. so RHI voi random partition (Phu luc C.3);
  4. bao early/mid/late.

Base population evolution: pymoo NSGA2 (ask/tell) -- Giai doan A CHUA
can base MOEA "chinh thuc" (moea/nsga3.py la cong viec cua "Giai doan A
tiep", sau Gate A1). Khong regional response nao duoc ap dung o day --
E1 la DETECT-ONLY audit, dung population tu nhien tien hoa lien tuc qua
cac moi truong.

Quyet dinh Gate A1: "RHI_ref co vuot phan phoi null (random partition)
khong?" -- do bang median percentile_rank(RHI_ref, null_dist) tren tat
ca change events cua tung DF, va tong hop qua 14 DF.
"""
import json
import os
import sys
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.dynamic.df import (DF1, DF2, DF3, DF4, DF6, DF7, DF8,
                                       DF9, DF10, DF11, DF14)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.core.normalize import normalize_pair  # noqa: E402
from prdrl.core.partition import assign_regions, region_indices  # noqa: E402
from prdrl.core.reference import make_reference_directions  # noqa: E402
from prdrl.detectors.rhi import rhi, rhi_null_distribution, percentile_rank  # noqa: E402
from prdrl.detectors.severity import displacement, region_severity_raw  # noqa: E402

# DF5/DF12/DF13 -- numerically-hardened time formulas were part of the
# archived SA-DRL-DMOEA codebase (src/df_stable.py, sua loi 4.10C.1: snap
# sin(...) gan-0 truoc floor() de tranh discontinuity gia do sai so dau
# phay dong tai moc chu ky). Module do da bi xoa cung src/ o Buoc 2. E1
# dung truc tiep DF5/DF12/DF13 CHUAN cua pymoo (chua hardening) va CHI
# dung cac diem thoi gian t=k/n_t voi k nho -- rui ro discontinuity gia
# thap o day nhung PHAI ghi lai; se can dung lai ban hardened truoc khi
# dua DF5/12/13 vao bat ky benchmark chinh thuc nao sau nay.
from pymoo.problems.dynamic.df import DF5, DF12, DF13

DFS = {
    "DF1": (DF1, 2), "DF2": (DF2, 2), "DF3": (DF3, 2), "DF4": (DF4, 2),
    "DF5": (DF5, 2), "DF6": (DF6, 2), "DF7": (DF7, 2), "DF8": (DF8, 2),
    "DF9": (DF9, 2),
    "DF10": (DF10, 3), "DF11": (DF11, 3), "DF12": (DF12, 3),
    "DF13": (DF13, 3), "DF14": (DF14, 3),
}

D = 10
N_BI, N_TRI = 100, 150       # Bang 15.1
H_BI, H_TRI = 9, 4           # Bang 11.1 -> R=10 / R=15
N_T, TAU_T, WARM_UP = 10, 10, 50
N_CHANGES = 15
SEEDS = [30, 31, 32]
N_NULL_TRIALS = 200

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_one_df_seed(df_name, cls, n_obj, seed):
    N = N_TRI if n_obj == 3 else N_BI
    H = H_TRI if n_obj == 3 else H_BI
    V = make_reference_directions(n_obj, H)
    R = len(V)

    p0 = cls(time=0.0, n_var=D)
    algo = NSGA2(pop_size=N)
    algo.setup(p0, termination=("n_gen", 10 ** 9), seed=seed, verbose=False)
    for _ in range(WARM_UP):
        algo.next()

    rows = []
    rng = np.random.default_rng(seed + 999_000)
    for k in range(1, N_CHANGES + 1):
        t_new = k / N_T
        X = algo.pop.get("X").copy()
        F_minus = algo.pop.get("F").copy()
        p_new = cls(time=t_new, n_var=D)
        F_plus = p_new.evaluate(X)

        Fm_t, Fp_t, z_star, z_nad = normalize_pair(F_minus, F_plus)
        r_of_i = assign_regions(Fm_t, V)
        idx_by_region = region_indices(r_of_i, R)
        d_i = displacement(Fm_t, Fp_t)
        s_r = region_severity_raw(d_i, idx_by_region)
        region_sizes = [len(idx_by_region[r]) for r in range(R)]

        rhi_ref = rhi(s_r)
        null_dist = rhi_null_distribution(d_i, region_sizes, N_NULL_TRIALS, rng)
        pct = percentile_rank(rhi_ref, null_dist)
        n_active = sum(1 for v in s_r.values() if v is not None)

        rows.append(dict(DF=df_name, seed=seed, change=k, t=t_new,
                         rhi_ref=rhi_ref, null_mean=float(np.mean(null_dist)),
                         null_std=float(np.std(null_dist)), percentile=pct,
                         n_active_regions=n_active, R=R))

        algo.problem = p_new
        algo.pop.set("F", F_plus)
        for _ in range(TAU_T):
            algo.next()

    return rows


def main():
    t0 = time.time()
    all_rows = []
    for df_name, (cls, n_obj) in DFS.items():
        for seed in SEEDS:
            rows = run_one_df_seed(df_name, cls, n_obj, seed)
            all_rows.extend(rows)
        print(f"  {df_name}: {len(SEEDS)} seeds x {N_CHANGES} changes done "
             f"({time.time()-t0:.0f}s elapsed)")

    out_path = os.path.join(RESULTS_DIR, "prdrl_E1_heterogeneity_audit.jsonl")
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path} ({len(all_rows)} rows)")

    # ---- per-DF summary + early/mid/late ----
    import collections
    by_df = collections.defaultdict(list)
    for r in all_rows:
        by_df[r["DF"]].append(r)

    third = N_CHANGES // 3
    summary = {}
    print(f"\n{'DF':6s} {'median_pct':>11s} {'frac>0.95':>10s} "
         f"{'early_pct':>10s} {'mid_pct':>9s} {'late_pct':>9s}")
    for df_name in DFS:
        rows = by_df[df_name]
        pcts = np.array([r["percentile"] for r in rows])
        early = np.array([r["percentile"] for r in rows if r["change"] <= third])
        mid = np.array([r["percentile"] for r in rows if third < r["change"] <= 2 * third])
        late = np.array([r["percentile"] for r in rows if r["change"] > 2 * third])
        summary[df_name] = dict(
            median_percentile=float(np.median(pcts)),
            frac_above_95=float(np.mean(pcts > 0.95)),
            early_median=float(np.median(early)) if len(early) else None,
            mid_median=float(np.median(mid)) if len(mid) else None,
            late_median=float(np.median(late)) if len(late) else None,
            n_rows=len(rows),
        )
        s = summary[df_name]
        print(f"{df_name:6s} {s['median_percentile']:11.3f} {s['frac_above_95']:10.3f} "
             f"{s['early_median']:10.3f} {s['mid_median']:9.3f} {s['late_median']:9.3f}")

    all_pcts = np.array([r["percentile"] for r in all_rows], dtype=float)
    overall_median_pct = float(np.median(all_pcts))
    overall_frac_95 = float(np.mean(all_pcts > 0.95))
    n_dfs_pass = sum(1 for s in summary.values() if s["median_percentile"] > 0.95)

    gate_a1 = (overall_median_pct > 0.95 and n_dfs_pass >= 10)  # >=10/14 DF, pre-registered threshold

    summary_path = os.path.join(RESULTS_DIR, "prdrl_E1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(per_df=summary, overall_median_percentile=overall_median_pct,
                      overall_frac_above_95=overall_frac_95,
                      n_dfs_median_above_95=n_dfs_pass, n_dfs_total=len(DFS),
                      gate_a1_pass=gate_a1,
                      config=dict(D=D, N_BI=N_BI, N_TRI=N_TRI, H_BI=H_BI, H_TRI=H_TRI,
                                 n_t=N_T, tau_t=TAU_T, warm_up=WARM_UP,
                                 n_changes=N_CHANGES, seeds=SEEDS,
                                 n_null_trials=N_NULL_TRIALS)),
                 f, indent=2)
    print(f"\nwrote {summary_path}")
    print(f"\noverall median percentile RHI_ref vs random-partition null: {overall_median_pct:.3f}")
    print(f"DFs with median percentile > 0.95: {n_dfs_pass}/{len(DFS)}")
    print(f"\nGATE A1: {'PASS' if gate_a1 else 'FAIL'} "
         f"(RHI_ref {'DOES' if gate_a1 else 'does NOT'} clearly exceed random-partition null)")
    print(f"Total time: {time.time()-t0:.0f}s")
    return gate_a1


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
