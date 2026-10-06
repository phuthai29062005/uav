"""
E1b — SDP heterogeneity audit (Gate SDP, PRIMARY benchmark), Phase 7 --
E1R.

Dung LAI CHINH XAC pipeline detector nhu E1a/E1c (normalize -> partition
tren F_minus -> severity -> RHI -> size-preserving random-partition
null). KHONG viet detector rieng cho SDP, KHONG threshold rieng cho
tung benchmark.

Tap SDP pre-registered (Phase 6, chon TRUOC khi chay, KHONG chon sau
khi xem ket qua):
    SDP2, SDP4, SDP5, SDP8, SDP9, SDP10, SDP11
Ly do: day la 7/13 SDP da implement DUOC XAC NHAN cong thuc day du,
khong phu thuoc vao phan mu_linear (eq.3) bi trich xuat PDF mo ho (xem
docstring dau file prdrl/problems/sdp.py) -- chon truoc khi chay, dung
vi "khong con eq(3)-ambiguous" tinh co, ma vi day CHINH LA tap 7 bai da
duoc pre-register trong prompt task nay TRUOC KHI bat dau code SDP
adapter.

Thoi gian: SAME convention voi CEC2018 (paper SDP tu xac nhan: "t =
(1/nt) floor(tau/taut)") -- KHONG can dinh nghia rieng.
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
from prdrl.detectors.rhi import rhi_null
from prdrl.problems.sdp import evaluate_sdp, get_sdp_bounds, get_sdp_true_pf


PRESELECTED_SDP = [2, 4, 5, 8, 9, 10, 11]


def compute_time(generation, n_t=10, tau_t=10):
    return (1.0 / n_t) * np.floor(generation / tau_t)


class SDPProblem(Problem):
    def __init__(self, problem_id, t, n_obj, D, lower, upper, seed):
        super().__init__(n_var=D, n_obj=n_obj, xl=lower, xu=upper)
        self.problem_id = problem_id
        self.t = t
        self.seed = seed

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = evaluate_sdp(self.problem_id, X, self.t, self.n_obj,
                                seed=self.seed)


def run_nsga2_with_changes(problem_id, n_obj, n_changes=15, pop_size=100,
                           gen_per_change=10, n_t=10, tau_t=10, seed=42,
                           D=10):
    """Giong het E1c: continuous trajectory (sampling=X_pop,
    result.pop.get("X")), SAME X cho F_minus/F_plus."""
    rng = np.random.default_rng(seed)
    lower, upper = get_sdp_bounds(problem_id, n_obj, n_var=D)

    X_pop = rng.uniform(lower, upper, size=(pop_size, D))
    t_current = compute_time(0, n_t, tau_t)

    events = []
    gen = 0

    for change_idx in range(n_changes):
        problem = SDPProblem(problem_id, t_current, n_obj, D, lower, upper, seed)
        algo = NSGA2(pop_size=pop_size, sampling=X_pop)
        res = minimize(problem, algo, ("n_gen", gen_per_change),
                       seed=seed + change_idx, verbose=False)
        X_pop = np.asarray(res.pop.get("X"), dtype=float)
        gen += gen_per_change

        F_minus = evaluate_sdp(problem_id, X_pop, t_current, n_obj, seed=seed)

        t_new = compute_time(gen, n_t, tau_t)
        if abs(t_new - t_current) < 1e-15:
            continue

        F_plus = evaluate_sdp(problem_id, X_pop, t_new, n_obj, seed=seed)

        events.append({
            "F_minus": F_minus, "F_plus": F_plus, "X": X_pop.copy(),
            "t_old": t_current, "t_new": t_new, "change_idx": change_idx,
        })
        t_current = t_new

    return events


def audit_one_sdp(problem_id, n_obj, seeds, n_changes=15, pop_size=100, K=200,
                  D=10):
    n_partitions = 9 if n_obj == 2 else 4
    V = reference(n_obj, n_partitions)
    R = len(V)

    records = []
    for seed in seeds:
        events = run_nsga2_with_changes(
            problem_id, n_obj, n_changes=n_changes, pop_size=pop_size,
            seed=seed, D=D)

        q_prev = 1.0
        for ev in events:
            F_tilde_minus, F_tilde_plus = normalize(ev["F_minus"], ev["F_plus"])
            F_tilde_minus = np.asarray(F_tilde_minus)
            F_tilde_plus = np.asarray(F_tilde_plus)

            region_ids = np.asarray(partition(F_tilde_minus, V))

            s_r, c_r, q_prev = severity(F_tilde_minus, F_tilde_plus,
                                        region_ids, R, q_prev)

            null_rng = deterministic_rng(seed, 1000 + problem_id, ev["change_idx"])
            rhi_ref, null_mean, null_std, pct = rhi_null(
                F_tilde_minus, F_tilde_plus, region_ids, R, K=K, rng=null_rng)

            active_mask = ~np.isnan(np.asarray(s_r, dtype=float))
            R_active = int(np.sum(active_mask))

            records.append({
                "problem_id": problem_id, "n_obj": n_obj, "seed": seed,
                "change_idx": ev["change_idx"],
                "t_old": float(ev["t_old"]), "t_new": float(ev["t_new"]),
                "same_X_verified": bool(ev["X"].shape[0] == pop_size),
                "R_total": R, "R_active": R_active,
                "active_fraction": R_active / R,
                "rhi": float(rhi_ref), "null_mean": float(null_mean),
                "null_std": float(null_std), "percentile": float(pct),
            })

    return records


def summarize(problem_id, n_obj, records):
    pct = np.array([r["percentile"] for r in records])
    rhi_vals = np.array([r["rhi"] for r in records])
    af = np.array([r["active_fraction"] for r in records])
    return {
        "problem_id": problem_id, "n_obj": n_obj, "n_events": len(records),
        "median_rhi": float(np.median(rhi_vals)),
        "median_percentile": float(np.median(pct)),
        "mean_percentile": float(np.mean(pct)),
        "pct_above_95": float(np.mean(pct >= 0.95)),
        "median_active_fraction": float(np.median(af)),
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


def sdp_verdict(summaries):
    """Phase 10: Gate SDP -- KHONG can toan bo 7 bai pass, dua tren
    evidence lap lai tren NHIEU bai (khong chi 1 bai don le)."""
    strong = [s for s in summaries if s["median_percentile"] >= 0.95]
    n_strong = len(strong)
    n_total = len(summaries)
    if n_total == 0:
        return "SDP-WEAK"
    if n_strong >= 2 and n_strong / n_total >= 0.4:
        return "SDP-REGIONAL-SIGNAL"
    if n_strong >= 1:
        return "SDP-MIXED"
    return "SDP-WEAK"


def run_e1b(problem_ids=None, n_obj=2, seeds=(1, 2, 3), n_changes=15,
           pop_size=100, out_dir="results"):
    if problem_ids is None:
        problem_ids = PRESELECTED_SDP

    print("=" * 70)
    print(f"E1b — SDP heterogeneity audit (Gate SDP), M={n_obj}")
    print(f"    Pre-registered subset: {problem_ids}")
    print("=" * 70)

    all_records = []
    summaries = []
    t0 = time.time()

    for pid in problem_ids:
        print(f"\nSDP{pid} ...", end=" ", flush=True)
        try:
            records = audit_one_sdp(pid, n_obj, list(seeds),
                                    n_changes=n_changes, pop_size=pop_size)
        except NotImplementedError as e:
            print(f"SKIPPED (not implemented) — {e}")
            print("STOP: pre-registered SDP problem does not support the "
                  "assumed configuration. Per Phase 6, do NOT silently "
                  "replace it.")
            continue
        all_records.extend(records)
        summ = summarize(pid, n_obj, records)
        summaries.append(summ)
        print(f"median pct={summ['median_percentile']:.3f}  "
              f"events={summ['n_events']}  "
              f"active_frac={summ['median_active_fraction']:.2f}")

    elapsed = time.time() - t0
    verdict = sdp_verdict(summaries)

    print("\n" + "=" * 70)
    print(f"{'SDP':<8} {'median RHI':>11} {'median pct':>12} {'%>=0.95':>8} "
          f"{'active_frac':>12} {'events':>8}")
    print("-" * 65)
    for s in summaries:
        print(f"SDP{s['problem_id']:<5} {s['median_rhi']:>11.4f} "
              f"{s['median_percentile']:>12.3f} "
              f"{s['pct_above_95']:>7.1%} "
              f"{s['median_active_fraction']:>12.2f} "
              f"{s['n_events']:>8}")
    print("-" * 65)
    n_strong = sum(1 for s in summaries if s["median_percentile"] >= 0.95)
    print(f"SDP problems with median percentile >= 0.95: "
          f"{n_strong}/{len(summaries)}")
    print(f"Time: {elapsed:.1f}s")
    print()
    print(f"GATE SDP VERDICT: {verdict}")
    print("=" * 70)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_path / "E1b_sdp.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")

    csv_path = out_path / "E1b_sdp_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = ["problem_id", "n_obj", "n_events", "median_rhi",
                "median_percentile", "mean_percentile", "pct_above_95",
                "median_active_fraction"]
        f.write(",".join(cols) + "\n")
        for s in summaries:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    report_path = out_path / "E1b_sdp_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# E1b — SDP Heterogeneity Audit (Gate SDP)\n\n")
        f.write(f"Pre-registered subset (chosen BEFORE running, per Phase "
                f"6): SDP{problem_ids}\n\n")
        f.write(f"**GATE SDP VERDICT: {verdict}**\n\n")
        f.write("| SDP | median RHI | median pct | %>=0.95 | active_frac | events |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summaries:
            f.write(f"| SDP{s['problem_id']} | {s['median_rhi']:.4f} | "
                    f"{s['median_percentile']:.3f} | {s['pct_above_95']:.1%} | "
                    f"{s['median_active_fraction']:.2f} | {s['n_events']} |\n")

    print(f"\nSaved: {jsonl_path}, {csv_path}, {report_path}")

    return {"summaries": summaries, "verdict": verdict, "records": all_records}


if __name__ == "__main__":
    run_e1b()
