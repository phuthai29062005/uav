"""
4.10D — Frozen-probe representativeness diagnostic (READ-ONLY).

Chay SA path voi KEEP-only policy (NSGA2Baseline) de tach probe drift
khoi hanh vi controller. Do khoang cach probe<->population trong khong
gian quyet dinh chuan hoa z=(x-xl)/(xu-xl), ||.||/sqrt(D). Khong sua
production; diagnostic FE (offline pop eval cho ND) KHONG vao fes_used.
"""
import csv
import os
import sys

import numpy as np
from pymoo.problems.dynamic.df import DF1, DF2, DF4, DF6, DF10
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import dynamic_runner as dr  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from df_stable import DF12Stable, DF13Stable  # noqa: E402
from train import problem_setup  # noqa: E402

DFS = {"DF1": DF1, "DF2": DF2, "DF4": DF4, "DF6": DF6,
       "DF10": DF10, "DF12": DF12Stable, "DF13": DF13Stable}
D, K, WARM, TAU, NT = 10, 100, 50, 10, 10


def zdist_min(A, B, xl, xu):
    """Min normalized-z distance /sqrt(D) tu moi hang A den tap B."""
    w = xu - xl
    Za, Zb = (A - xl) / w, (B - xl) / w
    out = []
    for za in Za:
        out.append(np.min(np.linalg.norm(Zb - za, axis=1)) / np.sqrt(len(xl)))
    return np.array(out)


def run_one(name, cls, seed=0):
    N, ref = problem_setup(name)
    captured = {"probe": None, "pops": [], "problems": []}

    orig_prime = dr.ChangeDetector.prime

    def prime_spy(self, X, p):
        captured["probe"] = np.array(X, copy=True)
        return orig_prime(self, X, p)

    orig_apply = dr.apply_hierarchical_response

    def apply_spy(pop, pop_prev, gate, seg, segments, pop_mem, xl, xu):
        captured["pops"].append(np.array(pop, copy=True))
        return orig_apply(pop, pop_prev, gate, seg, segments, pop_mem, xl, xu)

    qs = []
    orig_compute = dr.ChangeDetector.compute

    def compute_spy(self, *a, **k):
        r = orig_compute(self, *a, **k)
        qs.append(float(self._q))
        return r

    dr.ChangeDetector.prime = prime_spy
    dr.apply_hierarchical_response = apply_spy
    dr.ChangeDetector.compute = compute_spy
    try:
        res = dr.run_sa_drl(cls, n_t=NT, tau_t=TAU, N=N, D=D, n_changes=K,
                            warm_up=WARM, agent=NSGA2Baseline(n_segments=2),
                            segments=[[0], list(range(1, D))], seed=seed,
                            training=False, ref_point=ref, n_elite=10)
    finally:
        dr.ChangeDetector.prime = orig_prime
        dr.apply_hierarchical_response = orig_apply
        dr.ChangeDetector.compute = orig_compute

    probe = captured["probe"]
    p0 = cls(time=0.0, n_var=D)
    xl, xu = np.asarray(p0.xl, float), np.asarray(p0.xu, float)
    rows = []
    for k in range(K):
        pop = captured["pops"][k]
        t = (k + 1) / NT
        prob = cls(time=t, n_var=D)
        pp = zdist_min(probe, pop, xl, xu)          # probe->pop
        ppop = zdist_min(pop, probe, xl, xu)        # pop->probe
        F = prob.evaluate(pop)                       # DIAGNOSTIC FE (offline)
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pnd = zdist_min(probe, pop[nd], xl, xu)     # probe->ND
        rc = res["raw_change"][k]
        c = res["normalized_change"][k]
        rows.append(dict(
            df=name, seed=seed, change=k + 1, time=round(t, 3),
            probe_pop_mean=pp.mean(), probe_pop_max=pp.max(),
            pop_probe_mean=ppop.mean(), probe_nd_mean=pnd.mean(),
            raw_c1=rc[0], raw_c2=rc[1], c1=c[0], c2=c[1],
            q_scale=qs[k], d_mem=res["d_mem"][k],
            sig_sat=res["sig_saturation"][k]))
    return probe, rows, res


def band(rows, key, lo, hi):
    return np.mean([r[key] for r in rows if lo <= r["change"] <= hi])


def main():
    allrows = []
    print(f"{'DF':5s} {'pp_early':>9} {'pp_mid':>8} {'pp_late':>8} "
          f"{'late/early':>10} {'rawc_late':>10} {'c>0.95%':>8} "
          f"{'sat_late':>9} {'verdict':>8}")
    for name, cls in DFS.items():
        probe, rows, res = run_one(name, cls)
        allrows += rows
        e = band(rows, "probe_pop_mean", 1, 10)
        m = band(rows, "probe_pop_mean", 46, 55)
        la = band(rows, "probe_pop_mean", 91, 100)
        ratio = la / e if e > 1e-9 else float("nan")
        rawc_late = np.mean([max(r["raw_c1"], r["raw_c2"])
                             for r in rows if r["change"] >= 91])
        csat = 100 * np.mean([1 for r in rows for cc in (r["c1"], r["c2"])
                              if cc > 0.95]) / (2 * len(rows))
        satlate = np.mean([r["sig_sat"] for r in rows if r["change"] >= 91])
        healthy = (np.isfinite(rawc_late) and rawc_late > 1e-6
                   and csat < 50 and satlate < 0.5)
        v = "PASS" if healthy else "CHECK"
        print(f"{name:5s} {e:9.4f} {m:8.4f} {la:8.4f} {ratio:10.2f} "
              f"{rawc_late:10.4f} {csat:8.1f} {satlate:9.3f} {v:>8}")
    with open("results/probe_repr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print("\nCSV: results/probe_repr.csv  (", len(allrows), "rows )")


if __name__ == "__main__":
    main()
