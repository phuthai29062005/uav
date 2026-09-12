"""
5.2 — Final statistical analysis of the frozen 14x30 paired benchmark.
Reads results/final_benchmark_runs.jsonl (data-collection artifact, 5.1),
never modifies it. Data-analysis only: no rerun/retrain/tune.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from stats_analysis import (classify_primary, effect_magnitude,  # noqa: E402
                            holm_correction, rank_biserial, wilcoxon_paired)

RESULTS = os.path.join(_HERE, "..", "results")
RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
DFS = [f"DF{i}" for i in range(1, 15)]
EXPECTED_SEEDS = set(range(100000, 100030))
ALPHA = 0.05


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                          capture_output=True, text=True).stdout.strip()


# ============================================================ PHAN 0: FREEZE
def load_and_verify():
    raw_sha = sha256_of(RAW_PATH)
    rows = [json.loads(l) for l in open(RAW_PATH) if l.strip()]

    errs = []
    if len(rows) != 840:
        errs.append(f"row count {len(rows)} != 840")
    keys = [(r["method"], r["problem"], r["seed"]) for r in rows]
    if len(set(keys)) != 840:
        errs.append("duplicate (method,problem,seed) keys")
    paired = {}
    for r in rows:
        paired.setdefault((r["problem"], r["seed"]), set()).add(r["method"])
    if len(paired) != 420:
        errs.append(f"paired keys {len(paired)} != 420")
    if not all(len(v) == 2 for v in paired.values()):
        errs.append("some paired keys do not have exactly 2 methods")
    for df in DFS:
        n_sa = sum(1 for r in rows if r["problem"] == df and r["method"] == "SA-DRL-frozen")
        n_bl = sum(1 for r in rows if r["problem"] == df and r["method"] == "NSGA2-clean")
        if n_sa != 30 or n_bl != 30:
            errs.append(f"{df}: n_sa={n_sa} n_bl={n_bl} (expect 30/30)")
    seeds_seen = set(r["seed"] for r in rows)
    if seeds_seen != EXPECTED_SEEDS:
        errs.append(f"seed set mismatch: {seeds_seen ^ EXPECTED_SEEDS}")
    for r in rows:
        for k in ("migd_end", "migd_response", "fes_used"):
            v = r.get(k)
            if v is None or not np.isfinite(v):
                errs.append(f"non-finite {k} in row {r.get('experiment_id')}")

    if errs:
        print("!!! DATA INTEGRITY FAILURE:")
        for e in errs:
            print("  -", e)
        sys.exit(1)

    print(f"Integrity OK: 840 rows, 420 pairs, 14 DFs x 30/30, seeds "
         f"{min(seeds_seen)}..{max(seeds_seen)}, no NaN/Inf.")
    return rows, raw_sha


# ============================================================ PHAN 1: PAIR
def build_paired(rows):
    by_key = {}
    for r in rows:
        by_key.setdefault((r["problem"], r["seed"]), {})[r["method"]] = r
    out = []
    for (df, seed), methods in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        sa, bl = methods["SA-DRL-frozen"], methods["NSGA2-clean"]
        out.append(dict(
            DF=df, seed=seed,
            SA_migd_end=sa["migd_end"], BL_migd_end=bl["migd_end"],
            SA_migd_response=sa["migd_response"], BL_migd_response=bl["migd_response"],
            SA_FE=sa["fes_used"], BL_FE=bl["fes_used"],
            SA_FE_detector=sa["FE_detector"], SA_FE_signature=sa["FE_signature"],
            SA_FE_response_extra=sa["FE_response_extra"],
            SA_FE_environment_reeval=sa["FE_environment_reeval"],
            SA_FE_nsga2=sa["fe_breakdown"]["nsga2"],
            BL_FE_environment_reeval=bl["FE_environment_reeval"],
            BL_FE_nsga2=bl["fe_breakdown"]["nsga2"],
        ))
    df = pd.DataFrame(out)
    assert len(df) == 420
    return df


# ============================================================ PHAN 3-8: PRIMARY/SECONDARY
def analyze_family(paired, sa_col, bl_col, label):
    """One Wilcoxon+Holm+rank-biserial family across 14 DFs for a metric."""
    rows = []
    for df_name in DFS:
        sub = paired[paired.DF == df_name]
        sa, bl = sub[sa_col].values, sub[bl_col].values
        d = bl - sa

        wtest = wilcoxon_paired(bl, sa)
        rb = rank_biserial(bl, sa)

        bl_pos = np.all(bl > 0)
        rel_gain = np.median((bl - sa) / bl * 100) if bl_pos else None

        rows.append(dict(
            DF=df_name, n=len(sub),
            SA_mean=sa.mean(), SA_std=sa.std(ddof=1), SA_median=np.median(sa),
            SA_Q1=np.percentile(sa, 25), SA_Q3=np.percentile(sa, 75),
            BL_mean=bl.mean(), BL_std=bl.std(ddof=1), BL_median=np.median(bl),
            BL_Q1=np.percentile(bl, 25), BL_Q3=np.percentile(bl, 75),
            median_diff_BL_minus_SA=np.median(d), mean_diff_BL_minus_SA=d.mean(),
            Q1_d=np.percentile(d, 25), Q3_d=np.percentile(d, 75),
            W=wtest["W"], n_nonzero=wtest["n_nonzero"], p_raw=wtest["p_raw"],
            W_plus=rb["W_plus"], W_minus=rb["W_minus"], rank_biserial=rb["r_rb"],
            median_relative_gain_pct=rel_gain,
        ))
    fam = pd.DataFrame(rows)
    p_holm, reject = holm_correction(fam["p_raw"].values, alpha=ALPHA)
    fam["p_holm"] = p_holm
    fam["reject_holm"] = reject
    fam["effect_magnitude"] = fam["rank_biserial"].apply(effect_magnitude)
    fam["result_symbol"] = fam.apply(
        lambda r: classify_primary(r["p_holm"], r["median_diff_BL_minus_SA"], ALPHA), axis=1)
    print(f"\n[{label}] Holm family: {len(fam)} p-values corrected.")
    return fam


# ============================================================ PHAN 10/13: FE
def fe_summary(paired):
    rows = []
    for df_name in DFS:
        sub = paired[paired.DF == df_name]
        sa_fe, bl_fe = sub.SA_FE.values, sub.BL_FE.values
        med_sa, med_bl = np.median(sa_fe), np.median(bl_fe)
        rows.append(dict(
            DF=df_name, n=len(sub),
            SA_FE_mean=sa_fe.mean(), SA_FE_std=sa_fe.std(ddof=1),
            SA_FE_median=med_sa, SA_FE_min=sa_fe.min(), SA_FE_max=sa_fe.max(),
            BL_FE_mean=bl_fe.mean(), BL_FE_std=bl_fe.std(ddof=1),
            BL_FE_median=med_bl, BL_FE_min=bl_fe.min(), BL_FE_max=bl_fe.max(),
            median_overhead_abs=med_sa - med_bl,
            median_overhead_pct=(med_sa - med_bl) / med_bl * 100,
            detector_FE_median=np.median(sub.SA_FE_detector.values),
            signature_FE_median=np.median(sub.SA_FE_signature.values),
            response_extra_FE_median=np.median(sub.SA_FE_response_extra.values),
            environment_reeval_FE_median=np.median(sub.SA_FE_environment_reeval.values),
            nsga2_FE_median=np.median(sub.SA_FE_nsga2.values),
        ))
    return pd.DataFrame(rows)


# ============================================================ PLOTS
def make_plots(primary, secondary, fe_tab, paired):
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)

    # A: paired-difference plot MIGD_end, per-seed points per DF
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, df_name in enumerate(DFS):
        sub = paired[paired.DF == df_name]
        d = sub.BL_migd_end - sub.SA_migd_end
        jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=10, alpha=0.5, color="#4477AA")
        ax.scatter([i], [np.median(d)], color="black", marker="_", s=200, zorder=5)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(range(14)); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("BL - SA  (MIGD_end)  [positive favors SA]")
    ax.set_title("Paired difference per seed, MIGD_end (primary)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "paired_diff_migd_end.png"), dpi=150)
    plt.close(fig)

    # B: per-DF median paired point plot (SA vs BL medians)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(14)
    ax.plot(x, primary.SA_median, "o-", label="SA median", color="#EE6677")
    ax.plot(x, primary.BL_median, "o-", label="BL median", color="#4477AA")
    ax.set_xticks(x); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("MIGD_end (median over 30 seeds)")
    ax.legend(); ax.set_title("Per-DF median MIGD_end (primary)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "median_migd_end_per_df.png"), dpi=150)
    plt.close(fig)

    # C: FE overhead by DF
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, fe_tab.median_overhead_pct, color="#CCBB44")
    ax.set_xticks(x); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("Median FE overhead (%) SA vs baseline")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("FE overhead by DF (same physical schedule; not equal-FE)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fe_overhead_by_df.png"), dpi=150)
    plt.close(fig)

    # D: effect-size forest plot, primary
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(14)
    colors = ["#228833" if s == "SA+" else "#CC3311" if s == "SA-" else "#888888"
             for s in primary.result_symbol]
    ax.scatter(primary.rank_biserial, y, color=colors, s=60, zorder=3)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(DFS)
    ax.set_xlabel("rank-biserial r  (positive favors SA)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title("Effect size (primary, MIGD_end) -- no invented CI")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "effect_size_forest_primary.png"), dpi=150)
    plt.close(fig)


# ============================================================ REPORT
def write_report(raw_sha, code_commit, primary, secondary, fe_tab, models):
    def fmt_row(r):
        return (f"| {r.DF} | {r.SA_median:.4f} [{r.SA_Q1:.4f}, {r.SA_Q3:.4f}] | "
               f"{r.BL_median:.4f} [{r.BL_Q1:.4f}, {r.BL_Q3:.4f}] | "
               f"{r.median_diff_BL_minus_SA:+.4f} | {r.p_raw:.4g} | {r.p_holm:.4g} | "
               f"{r.rank_biserial:+.3f} ({r.effect_magnitude}) | {r.result_symbol} |")

    primary_counts = primary.result_symbol.value_counts().to_dict()
    secondary_counts = secondary.result_symbol.value_counts().to_dict()

    lines = []
    lines.append("# Statistical Report — Frozen 14x30 Paired Benchmark (5.2)\n")
    lines.append("## 1. Dataset integrity\n")
    lines.append(f"- Raw file: `results/final_benchmark_runs.jsonl`")
    lines.append(f"- SHA256: `{raw_sha}`")
    lines.append(f"- Code commit: `{code_commit}`")
    lines.append(f"- 840 rows, 420 paired (DF,seed) cases, 14 DFs x 30 seeds x 2 methods\n")

    lines.append("## 2. Locked statistical protocol\n")
    lines.append("- Primary endpoint: MIGD_end (lower better). Paired difference d = BL - SA "
                 "(d>0 favors SA).")
    lines.append("- Test: paired two-sided Wilcoxon signed-rank, `zero_method=\"wilcox\"`.")
    lines.append("- Correction: Holm step-down, alpha=0.05, separately for primary (14 p) "
                 "and secondary (14 p) families.")
    lines.append("- Effect size: paired rank-biserial correlation (ties -> average ranks, "
                 "zeros excluded, consistent with the Wilcoxon zero policy).")
    lines.append("- Classification: p_holm<0.05 & median(d)>0 -> SA+; <0 -> SA-; else \"=\".\n")

    lines.append("## 3. Primary MIGD_end results\n")
    lines.append("| DF | SA median [Q1,Q3] | BL median [Q1,Q3] | median(BL-SA) | p_raw | "
                 "p_holm | rank-biserial | symbol |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in primary.iterrows():
        lines.append(fmt_row(r))
    lines.append(f"\nCounts across 14 DFs: {primary_counts}\n")

    lines.append("## 4. Secondary MIGD_response results\n")
    lines.append("| DF | SA median [Q1,Q3] | BL median [Q1,Q3] | median(BL-SA) | p_raw | "
                 "p_holm | rank-biserial | symbol |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in secondary.iterrows():
        lines.append(fmt_row(r))
    lines.append(f"\nCounts across 14 DFs: {secondary_counts}\n")

    lines.append("## 5. FE overhead\n")
    lines.append("Locked protocol: same physical schedule, NOT equal FE (4.10B). "
                 "SA incurs additional evaluations for change detection, signature "
                 "computation, and response reevaluation.\n")
    lines.append("| DF | SA FE median | BL FE median | overhead (abs) | overhead (%) |")
    lines.append("|---|---|---|---|---|")
    for _, r in fe_tab.iterrows():
        lines.append(f"| {r.DF} | {r.SA_FE_median:.0f} | {r.BL_FE_median:.0f} | "
                     f"{r.median_overhead_abs:+.0f} | {r.median_overhead_pct:+.2f}% |")

    lines.append("\n## 6. Effect-size interpretation\n")
    lines.append("Magnitude bands (descriptive, not pass/fail): |r|<0.1 negligible, "
                 "0.1-0.3 small, 0.3-0.5 moderate, >=0.5 large.\n")

    lines.append("## 7. Horizon interpretation\n")
    reversals = []
    pm = primary.set_index("DF").result_symbol
    sm = secondary.set_index("DF").result_symbol
    for d in DFS:
        if pm[d] != "=" and sm[d] != "=" and pm[d] != sm[d]:
            reversals.append(d)
    if reversals:
        lines.append(f"DFs where primary (end) and secondary (response) Holm-significant "
                     f"directions differ: {reversals}. Interpreted as stronger/weaker "
                     f"recovery-vs-immediate-response, not a contradiction (Phan 22).")
    else:
        lines.append("No DF shows opposite Holm-significant directions between "
                     "end-of-environment and immediate-response horizons.")

    lines.append("\n## 8. Limitations (carried forward, not inferred from p-values)\n")
    lines.append("1. Segmentation is a fixed index-based two-block partition "
                 "(B1=[x0], B2=[x1..x_{D-1}]), not a universal position/distance "
                 "partition (4.10A).")
    lines.append("2. Frozen change-detection probes: bounded drift, detector signal "
                 "healthy; partial signature soft-saturation observed on some "
                 "tri-objective DFs (DF10 late-horizon) (4.10D, D2).")
    lines.append("3. One model seed per DF (MODEL_SEED=30); training-seed variance "
                 "across independent retrainings is not captured by this dataset.")
    lines.append("4. FE protocol is same-physical-schedule, not equal-FE (4.10B); SA "
                 "incurs real additional evaluations.")
    lines.append("5. The FE cost term in the reward is action-invariant in the current "
                 "CEC implementation (full-population reevaluation every response).")

    lines.append("\n## 9. Reproducibility metadata\n")
    lines.append(f"- Models manifest: {len(models)} rows (`final_benchmark_models.jsonl`)")
    lines.append(f"- Test seeds: 100000..100029 (30)")
    lines.append(f"- Training seeds: 30..229 (200 episodes), model_seed=30")
    lines.append(f"- Generated: `results/statistics_primary_migd_end.csv`, "
                 f"`statistics_secondary_migd_response.csv`, `statistics_fe_summary.csv`, "
                 f"`statistics_paired_420.csv`, `statistics_summary.json`, "
                 f"`table_primary_migd_end.tex`, `figures/*.png`")

    with open(os.path.join(RESULTS, "statistical_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex(primary):
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"DF & SA median [IQR] & BL median [IQR] & $p_{holm}$ & $r_{rb}$ & Result \\",
        r"\hline",
    ]
    for _, r in primary.iterrows():
        sa_bold = r.SA_median <= r.BL_median
        sa_cell = f"\\textbf{{{r.SA_median:.4f}}}" if sa_bold else f"{r.SA_median:.4f}"
        bl_cell = f"\\textbf{{{r.BL_median:.4f}}}" if not sa_bold else f"{r.BL_median:.4f}"
        sa_cell += f" [{r.SA_Q1:.3f}, {r.SA_Q3:.3f}]"
        bl_cell += f" [{r.BL_Q1:.3f}, {r.BL_Q3:.3f}]"
        lines.append(f"{r.DF} & {sa_cell} & {bl_cell} & {r.p_holm:.3g} & "
                    f"{r.rank_biserial:+.3f} & {r.result_symbol} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    with open(os.path.join(RESULTS, "table_primary_migd_end.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    code_commit = git_head()
    rows, raw_sha = load_and_verify()
    paired = build_paired(rows)
    paired.to_csv(os.path.join(RESULTS, "statistics_paired_420.csv"), index=False)
    assert len(paired) == 420

    primary = analyze_family(paired, "SA_migd_end", "BL_migd_end", "PRIMARY migd_end")
    secondary = analyze_family(paired, "SA_migd_response", "BL_migd_response",
                               "SECONDARY migd_response")
    fe_tab = fe_summary(paired)

    primary.to_csv(os.path.join(RESULTS, "statistics_primary_migd_end.csv"), index=False)
    secondary.to_csv(os.path.join(RESULTS, "statistics_secondary_migd_response.csv"), index=False)
    fe_tab.to_csv(os.path.join(RESULTS, "statistics_fe_summary.csv"), index=False)

    make_plots(primary, secondary, fe_tab, paired)

    models = [json.loads(l) for l in open(os.path.join(RESULTS, "final_benchmark_models.jsonl"))]

    write_report(raw_sha, code_commit, primary, secondary, fe_tab, models)
    write_latex(primary)

    summary = dict(
        raw_data_sha256=raw_sha, code_commit=code_commit,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        primary_metric="migd_end", secondary_metric="migd_response",
        test="paired Wilcoxon signed-rank, two-sided",
        zero_method="wilcox", correction="Holm", alpha=ALPHA,
        effect_size="paired rank-biserial correlation",
        effect_sign="positive favors SA-DRL",
        primary_counts=primary.result_symbol.value_counts().to_dict(),
        secondary_counts=secondary.result_symbol.value_counts().to_dict(),
        n_pairs=420, n_dfs=14, n_seeds_per_df=30,
        files=dict(
            paired_raw="results/statistics_paired_420.csv",
            primary_table="results/statistics_primary_migd_end.csv",
            secondary_table="results/statistics_secondary_migd_response.csv",
            fe_table="results/statistics_fe_summary.csv",
            report="results/statistical_report.md",
            latex_table="results/table_primary_migd_end.tex",
            figures_dir="results/figures/",
        ),
    )
    with open(os.path.join(RESULTS, "statistics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    sha_after = sha256_of(RAW_PATH)
    assert sha_after == raw_sha, "RAW FILE WAS MODIFIED DURING ANALYSIS"

    print("\n=== PRIMARY (migd_end) result symbols ===")
    print(primary[["DF", "p_raw", "p_holm", "rank_biserial", "result_symbol"]]
         .to_string(index=False))
    print("\nPrimary counts:", primary.result_symbol.value_counts().to_dict())
    print("Secondary counts:", secondary.result_symbol.value_counts().to_dict())
    print("\nraw sha256 unchanged after analysis:", sha_after == raw_sha)
    print("\nSTAT-PASS: statistical pipeline executed cleanly, all integrity checks held.")


if __name__ == "__main__":
    main()
