"""
5.3C -- Confirmatory ablation statistics on the frozen 5.3B holdout dataset.

Reads results/ablation_5_3B_runs.jsonl (AB-DATA-PASS, 2100 rows, 5 variants x
14 DF x 30 holdout seeds 200000-200029) and produces the pre-registered
confirmatory analysis for three independent hypothesis families:

    H1: A4 Full SA-DRL vs A1 Global RL (S=1)
    H2: A4 Full SA-DRL vs A2 Segmented RL without c_s observation
    H3: A4 Full SA-DRL vs A3 Segmented RL without Memory subsystem

Primary metric: migd_end. Sign convention: d = ablated - A4 (d>0 favors A4).
A0 (clean NSGA-II) is a descriptive contextual anchor only -- no inferential
test is computed for it in this phase.

Reuses (does not modify) src/stats_analysis.py: holm_correction,
rank_biserial, wilcoxon_paired, effect_magnitude, classify_primary.

Read-only w.r.t. results/ablation_5_3B_runs.jsonl and
results/final_benchmark_runs.jsonl -- this script never writes to either.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from stats_analysis import (classify_primary, effect_magnitude,  # noqa: E402
                            holm_correction, rank_biserial, wilcoxon_paired)

RESULTS = os.path.join(_HERE, "..", "results")
RUNS_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")
MODELS_PATH = os.path.join(RESULTS, "ablation_5_3B_models.jsonl")
MANIFEST_PATH = os.path.join(RESULTS, "ablation_5_3B_manifest.json")
MAIN_RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
MAIN_STATS_PATH = os.path.join(RESULTS, "statistics_primary_migd_end.csv")

MAIN_SHA_EXPECTED = ("f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6"
                    "fe090f09")

DFS = [f"DF{i}" for i in range(1, 15)]
VARIANTS = ["A0", "A1", "A2", "A3", "A4"]
HOLDOUT_SEEDS = list(range(200000, 200030))
ALPHA = 0.05

HYPOTHESES = {"H1": "A1", "H2": "A2", "H3": "A3"}

MECHANISM_NOTES = {
    "DF4": "DF4 high intervention (5.3A)",
    "DF7": "DF7 horizon reversal (5.3A)",
    "DF10": "DF10 memory ignored (5.3A)",
    "DF11": "DF11 high memory use (5.3A)",
    "DF12": "DF12 weak d_mem + mid-memory spike (5.3A)",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def q1(x):
    return float(np.percentile(x, 25))


def q3(x):
    return float(np.percentile(x, 75))


# =====================================================================
# PHAN 1 -- DATA INTEGRITY
# =====================================================================
def load_and_verify():
    sha_before = sha256_of(RUNS_PATH)
    main_sha = sha256_of(MAIN_RAW_PATH)
    assert main_sha == MAIN_SHA_EXPECTED, \
        f"STOP: main benchmark SHA changed: {main_sha}"

    rows = [json.loads(l) for l in open(RUNS_PATH)]
    assert len(rows) == 2100, f"STOP: expected 2100 rows, got {len(rows)}"

    keys = [(r["variant"], r["problem"], r["seed"]) for r in rows]
    assert len(set(keys)) == 2100, "STOP: duplicate (variant,DF,seed) keys"

    groups = {}
    for r in rows:
        groups.setdefault((r["problem"], r["seed"]), set()).add(r["variant"])
    assert len(groups) == 420, f"STOP: expected 420 groups, got {len(groups)}"
    for k, v in groups.items():
        assert v == set(VARIANTS), f"STOP: group {k} missing variants: {v}"

    for variant in VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant]
        assert len(vrows) == 420, \
            f"STOP: variant {variant} has {len(vrows)} rows, expected 420"
        for df in DFS:
            dfrows = [r for r in vrows if r["problem"] == df]
            assert len(dfrows) == 30, \
                f"STOP: {variant}/{df} has {len(dfrows)} rows, expected 30"
            seeds = sorted(r["seed"] for r in dfrows)
            assert seeds == HOLDOUT_SEEDS, \
                f"STOP: {variant}/{df} seeds mismatch"

    for r in rows:
        for k in ("migd_end", "migd_response", "fes_used"):
            v = r[k]
            assert np.isfinite(v), f"STOP: non-finite {k} in row {r.get('experiment_id')}"

    print(f"[PHAN 1] Integrity OK: 2100 rows, 420 groups x 5 variants, "
         f"0 NaN/Inf, 0 duplicates, 0 missing pairs.")
    print(f"[PHAN 1] Main benchmark SHA verified: {main_sha}")
    return rows, sha_before


# =====================================================================
# PHAN 2 -- PAIRED ANALYSIS TABLE (420 rows)
# =====================================================================
def build_paired_table(rows):
    idx = {(r["variant"], r["problem"], r["seed"]): r for r in rows}
    records = []
    for df in DFS:
        for seed in HOLDOUT_SEEDS:
            rec = {"DF": df, "seed": seed}
            for v in VARIANTS:
                r = idx[(v, df, seed)]
                rec[f"{v}_migd_end"] = r["migd_end"]
                rec[f"{v}_migd_response"] = r["migd_response"]
                rec[f"{v}_FE"] = r["fes_used"]
            records.append(rec)
    df_table = pd.DataFrame.from_records(records)
    assert len(df_table) == 420
    out = os.path.join(RESULTS, "ablation_statistics_paired_420.csv")
    df_table.to_csv(out, index=False)
    print(f"[PHAN 2] wrote {out} ({len(df_table)} rows)")
    return df_table


# =====================================================================
# PHAN 3-8 -- per-hypothesis DF-level Wilcoxon + Holm + rank-biserial
# =====================================================================
def analyze_hypothesis(paired, abl_variant):
    """One row per DF for A4 vs abl_variant. Sign convention: d = ABL - A4."""
    per_df = []
    for df in DFS:
        sub = paired[paired["DF"] == df].sort_values("seed")
        a4 = sub["A4_migd_end"].to_numpy(dtype=float)
        abl = sub[f"{abl_variant}_migd_end"].to_numpy(dtype=float)
        assert len(a4) == 30 and len(abl) == 30

        d_vec = abl - a4
        wres = wilcoxon_paired(abl, a4)          # d = abl - a4 internally
        rres = rank_biserial(abl, a4)             # r_rb>0 favors a4 (2nd arg)

        per_df.append(dict(
            DF=df, n=30,
            A4_mean=float(np.mean(a4)), A4_std=float(np.std(a4, ddof=1)),
            A4_median=float(np.median(a4)), A4_Q1=q1(a4), A4_Q3=q3(a4),
            ABL_mean=float(np.mean(abl)), ABL_std=float(np.std(abl, ddof=1)),
            ABL_median=float(np.median(abl)), ABL_Q1=q1(abl), ABL_Q3=q3(abl),
            median_diff_ABL_minus_A4=float(np.median(d_vec)),
            mean_diff_ABL_minus_A4=float(np.mean(d_vec)),
            Q1_d=q1(d_vec), Q3_d=q3(d_vec),          # extra (Phan 4), kept in report only
            W=wres["W"], n_nonzero=wres["n_nonzero"], p_raw=wres["p_raw"],
            W_plus=rres["W_plus"], W_minus=rres["W_minus"],
            rank_biserial=rres["r_rb"],
        ))

    p_raw_list = [r["p_raw"] for r in per_df]
    assert len(p_raw_list) == 14
    p_holm, reject = holm_correction(p_raw_list, alpha=ALPHA)
    for i, r in enumerate(per_df):
        r["p_holm"] = p_holm[i]
        r["reject_holm"] = reject[i]
        r["effect_magnitude"] = effect_magnitude(r["rank_biserial"])
        raw_symbol = classify_primary(r["p_holm"], r["median_diff_ABL_minus_A4"],
                                      alpha=ALPHA)
        symbol_map = {"SA+": "FULL+", "SA-": "ABL+", "=": "=",
                     "significant/mixed-zero-median": "significant_mixed_zero_median"}
        r["result_symbol"] = symbol_map[raw_symbol]

    return pd.DataFrame(per_df)


CSV_COLS = ["DF", "n", "A4_mean", "A4_std", "A4_median", "A4_Q1", "A4_Q3",
           "ABL_mean", "ABL_std", "ABL_median", "ABL_Q1", "ABL_Q3",
           "median_diff_ABL_minus_A4", "mean_diff_ABL_minus_A4",
           "W", "n_nonzero", "p_raw", "p_holm", "reject_holm",
           "W_plus", "W_minus", "rank_biserial", "effect_magnitude",
           "result_symbol"]


def run_hypothesis_families(paired):
    tables = {}
    for h, abl_variant in HYPOTHESES.items():
        tbl = analyze_hypothesis(paired, abl_variant)
        out = os.path.join(RESULTS, f"ablation_stats_{h}_A4_vs_{abl_variant}.csv")
        tbl[CSV_COLS].to_csv(out, index=False)
        print(f"[PHAN 15] wrote {out} ({len(tbl)} rows)")
        tables[h] = tbl
    return tables


# =====================================================================
# PHAN 16 -- five-variant descriptive table
# =====================================================================
def five_variant_descriptive(rows):
    idx = {}
    for r in rows:
        idx.setdefault((r["variant"], r["problem"]), []).append(r)

    records = []
    for df in DFS:
        rec = {"DF": df}
        for v in VARIANTS:
            vals = np.array([r["migd_end"] for r in idx[(v, df)]], dtype=float)
            fes = np.array([r["fes_used"] for r in idx[(v, df)]], dtype=float)
            rec[f"{v}_median_migd_end"] = float(np.median(vals))
            rec[f"{v}_IQR_migd_end"] = float(q3(vals) - q1(vals))
            rec[f"{v}_mean_migd_end"] = float(np.mean(vals))
            rec[f"{v}_std_migd_end"] = float(np.std(vals, ddof=1))
            rec[f"{v}_median_FE"] = float(np.median(fes))
        records.append(rec)
    tbl = pd.DataFrame.from_records(records)
    out = os.path.join(RESULTS, "ablation_descriptive_all_variants.csv")
    tbl.to_csv(out, index=False)
    print(f"[PHAN 16] wrote {out} ({len(tbl)} rows)")
    return tbl


# =====================================================================
# PHAN 14 -- FE descriptive summary
# =====================================================================
def fe_summary(rows):
    idx = {}
    for r in rows:
        idx.setdefault((r["variant"], r["problem"]), []).append(r)

    records = []
    for df in DFS:
        a4_fe = np.array([r["fes_used"] for r in idx[("A4", df)]], dtype=float)
        a4_med = float(np.median(a4_fe))
        for v in VARIANTS:
            fes = np.array([r["fes_used"] for r in idx[(v, df)]], dtype=float)
            med = float(np.median(fes))
            records.append(dict(
                DF=df, variant=v, median_FE=med, min_FE=float(np.min(fes)),
                max_FE=float(np.max(fes)),
                FE_diff_vs_A4=med - a4_med,
                FE_diff_pct_vs_A4=(100.0 * (med - a4_med) / a4_med
                                   if a4_med else float("nan")),
            ))
    tbl = pd.DataFrame.from_records(records)
    out = os.path.join(RESULTS, "ablation_fe_summary.csv")
    tbl.to_csv(out, index=False)
    print(f"[PHAN 14] wrote {out} ({len(tbl)} rows)")
    return tbl


# =====================================================================
# PHAN 17 -- hypothesis summary JSON
# =====================================================================
def verdict_rule(full_count, abl_count, total=14):
    if full_count == 0 and abl_count == 0:
        return ("NOT_SUPPORTED",
                "No DF showed a Holm-significant difference in either "
                "direction; no evidence the ablated component matters on "
                "this holdout set.")
    if abl_count > full_count:
        return ("NOT_SUPPORTED",
                f"Ablated variant was significantly better than Full A4 on "
                f"more DFs ({abl_count}/{total}) than Full was better than "
                f"ablated ({full_count}/{total}).")
    if full_count > 0 and abl_count > 0:
        return ("PARTIALLY_SUPPORTED",
                f"Evidence is problem-dependent: Full A4 significantly "
                f"better on {full_count}/{total} DFs, ablated variant "
                f"significantly better on {abl_count}/{total} DFs.")
    if full_count > 0 and abl_count == 0:
        if full_count >= total / 2:
            return ("SUPPORTED",
                    f"Full A4 significantly outperformed the ablated "
                    f"variant on {full_count}/{total} DFs (majority) with "
                    f"no DF favoring the ablated variant.")
        return ("PARTIALLY_SUPPORTED",
                f"Full A4 significantly outperformed the ablated variant "
                f"on {full_count}/{total} DFs (a minority) with no DF "
                f"favoring the ablated variant; effect is problem-"
                f"dependent, not general.")
    return ("NOT_SUPPORTED", "No material evidence found.")


def hypothesis_summary(tables):
    summary = {}
    for h, tbl in tables.items():
        full_c = int((tbl["result_symbol"] == "FULL+").sum())
        abl_c = int((tbl["result_symbol"] == "ABL+").sum())
        eq_c = int((tbl["result_symbol"] == "=").sum())
        mixed_c = int((tbl["result_symbol"] == "significant_mixed_zero_median").sum())
        large_full = int(((tbl["effect_magnitude"] == "large")
                          & (tbl["rank_biserial"] > 0)).sum())
        large_abl = int(((tbl["effect_magnitude"] == "large")
                         & (tbl["rank_biserial"] < 0)).sum())
        v, reason = verdict_rule(full_c, abl_c)
        summary[h] = dict(
            ablated_variant=HYPOTHESES[h],
            FULL_plus_count=full_c,
            ABL_plus_count=abl_c,
            no_significant_difference_count=eq_c,
            significant_mixed_zero_median_count=mixed_c,
            large_effect_FULL_count=large_full,
            large_effect_ABL_count=large_abl,
            verdict=v,
            reasoning=reason,
        )
    out = os.path.join(RESULTS, "ablation_hypothesis_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[PHAN 17] wrote {out}")
    return summary


# =====================================================================
# PHAN 19 -- cross-hypothesis pattern table
# =====================================================================
def cross_hypothesis_table(tables):
    main_sym = {}
    if os.path.exists(MAIN_STATS_PATH):
        mdf = pd.read_csv(MAIN_STATS_PATH)
        main_sym = dict(zip(mdf["DF"], mdf["result_symbol"]))

    records = []
    for df in DFS:
        rec = {"DF": df}
        for h in ("H1", "H2", "H3"):
            row = tables[h][tables[h]["DF"] == df].iloc[0]
            rec[f"{h}_A4_vs_{HYPOTHESES[h]}"] = row["result_symbol"]
        rec["main_5_2_result"] = main_sym.get(df, "n/a")
        rec["mechanism_note"] = MECHANISM_NOTES.get(df, "")
        records.append(rec)
    tbl = pd.DataFrame.from_records(records)
    out = os.path.join(RESULTS, "ablation_cross_hypothesis_patterns.csv")
    tbl.to_csv(out, index=False)
    print(f"[PHAN 19] wrote {out} ({len(tbl)} rows)")
    return tbl


# =====================================================================
# PHAN 22 -- figures
# =====================================================================
def make_figures(tables, five_var, fe_tbl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A. Effect-size heatmap: rows=DF, cols=H1,H2,H3, value=r_rb, zero-centered
    mat = np.zeros((14, 3))
    for j, h in enumerate(("H1", "H2", "H3")):
        for i, df in enumerate(DFS):
            mat[i, j] = tables[h][tables[h]["DF"] == df]["rank_biserial"].iloc[0]

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["H1\n(A4 vs A1)", "H2\n(A4 vs A2)", "H3\n(A4 vs A3)"])
    ax.set_yticks(range(14))
    ax.set_yticklabels(DFS)
    for i in range(14):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Rank-biserial effect size (>0 favors Full A4)")
    fig.colorbar(im, ax=ax, label="r_rb")
    fig.tight_layout()
    p1 = os.path.join(RESULTS, "ablation_fig_effect_heatmap.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # B. Per-hypothesis forest/scatter of r_rb per DF
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for ax, h in zip(axes, ("H1", "H2", "H3")):
        tbl = tables[h].sort_values("DF", key=lambda s: s.str.replace("DF", "").astype(int))
        y = range(len(tbl))
        colors = ["#c0392b" if s == "FULL+" else "#2980b9" if s == "ABL+" else "#888"
                 for s in tbl["result_symbol"]]
        ax.scatter(tbl["rank_biserial"], y, c=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(tbl["DF"])
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{h}: A4 vs {HYPOTHESES[h]}")
        ax.set_xlabel("rank-biserial r (favors A4 >0)")
    fig.tight_layout()
    p2 = os.path.join(RESULTS, "ablation_fig_forest_scatter.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # C. Five-variant median MIGD_end plot, normalized per-DF by A0 median
    #    (VISUALIZATION ONLY -- never used for hypothesis tests)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(14)
    width = 0.15
    for k, v in enumerate(VARIANTS):
        norm = five_var[f"{v}_median_migd_end"] / five_var["A0_median_migd_end"]
        ax.bar(x + (k - 2) * width, norm, width, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(DFS, rotation=45)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="A0 baseline")
    ax.set_ylabel("median MIGD_end / A0 median MIGD_end (visualization only)")
    ax.set_title("Five-variant median MIGD_end, normalized per-DF by A0 (NOT used for hypothesis tests)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(RESULTS, "ablation_fig_five_variant_median.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # D. FE median overhead per variant
    piv = fe_tbl.pivot(index="DF", columns="variant", values="median_FE")
    piv = piv.reindex(DFS)
    fig, ax = plt.subplots(figsize=(12, 6))
    for v in VARIANTS:
        ax.plot(DFS, piv[v], marker="o", label=v)
    ax.set_ylabel("median FE used")
    ax.set_title("Median function-evaluation budget by variant and DF")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(RESULTS, "ablation_fig_fe_overhead.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    print(f"[PHAN 22] wrote figures: {p1}, {p2}, {p3}, {p4}")
    return [p1, p2, p3, p4]


# =====================================================================
# main
# =====================================================================
def main():
    rows, sha_before = load_and_verify()
    paired = build_paired_table(rows)
    tables = run_hypothesis_families(paired)
    five_var = five_variant_descriptive(rows)
    fe_tbl = fe_summary(rows)
    summary = hypothesis_summary(tables)
    cross = cross_hypothesis_table(tables)
    figs = make_figures(tables, five_var, fe_tbl)

    sha_after = sha256_of(RUNS_PATH)
    assert sha_after == sha_before, "STOP: ablation raw file was modified"
    main_sha_after = sha256_of(MAIN_RAW_PATH)
    assert main_sha_after == MAIN_SHA_EXPECTED, \
        "STOP: main benchmark file was modified"

    print("\n=== SHA VERIFICATION ===")
    print(f"ablation raw SHA before: {sha_before}")
    print(f"ablation raw SHA after:  {sha_after}")
    print(f"main benchmark SHA:      {main_sha_after}")
    print(f"unchanged: {sha_after == sha_before and main_sha_after == MAIN_SHA_EXPECTED}")

    print("\n=== HYPOTHESIS SUMMARY ===")
    for h, s in summary.items():
        print(f"{h} (A4 vs {s['ablated_variant']}): FULL+={s['FULL_plus_count']} "
             f"ABL+={s['ABL_plus_count']} ==={s['no_significant_difference_count']} "
             f"-> {s['verdict']}")

    return dict(rows=rows, paired=paired, tables=tables, five_var=five_var,
               fe_tbl=fe_tbl, summary=summary, cross=cross,
               sha_before=sha_before, sha_after=sha_after)


if __name__ == "__main__":
    main()
