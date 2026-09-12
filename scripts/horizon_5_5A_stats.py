"""
5.5A statistics -- pre-specified contrasts C1-C4 on the frozen
horizon_5_5A_runs.jsonl dataset (750 rows, 5 variants x 5 DF x 30 seeds).

TARGETED EXPLORATORY DIAGNOSTIC (Phase 15 of the 5.5A spec), NOT new
general CEC confirmatory evidence -- DFs were selected after prior
experiments for contrasting mechanism patterns.

Sign convention (Phase 13): d = first - second, positive favors the
SECOND/target condition. Reuses (unmodified) src/stats_analysis.py.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from stats_analysis import (classify_primary, effect_magnitude,  # noqa: E402
                            holm_correction, rank_biserial, wilcoxon_paired)

RESULTS = os.path.join(_HERE, "..", "results")
RUNS_PATH = os.path.join(RESULTS, "horizon_5_5A_runs.jsonl")
MODELS_PATH = os.path.join(RESULTS, "horizon_5_5A_models.jsonl")
WINDOWS_PATH = os.path.join(RESULTS, "horizon_5_5A_training_windows.csv")
MAIN_RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
ABLATION_RAW_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")
LOG_PATH = os.path.join(RESULTS, "horizon_5_5A_run.log")

MAIN_SHA_EXPECTED = ("f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6"
                    "fe090f09")
ABLATION_SHA_EXPECTED = ("7191ee747155528a3d31ab7ec8ad8b7a87d499abb80bc5778c2"
                        "9c22eab138eff")

DFS = ["DF1", "DF7", "DF10", "DF11", "DF12"]
VARIANTS = ["A0", "H30_B6K", "H100_B6K", "H30_B18K", "H100_B18K"]
ALPHA = 0.05

# (name, first, second) -- Phase 13
CONTRASTS = {
    "C1_horizon_6K":  ("H30_B6K", "H100_B6K"),
    "C2_budget_H30":  ("H30_B6K", "H30_B18K"),
    "C3_horizon_18K": ("H30_B18K", "H100_B18K"),
    "C4_budget_H100": ("H100_B6K", "H100_B18K"),
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


def load_and_verify():
    sha_before = sha256_of(RUNS_PATH)
    main_sha = sha256_of(MAIN_RAW_PATH)
    abl_sha = sha256_of(ABLATION_RAW_PATH)
    assert main_sha == MAIN_SHA_EXPECTED
    assert abl_sha == ABLATION_SHA_EXPECTED

    rows = [json.loads(l) for l in open(RUNS_PATH)]
    assert len(rows) == 750
    keys = [(r["variant"], r["problem"], r["seed"]) for r in rows]
    assert len(set(keys)) == 750
    groups = {}
    for r in rows:
        groups.setdefault((r["problem"], r["seed"]), set()).add(r["variant"])
    assert len(groups) == 150
    for k, v in groups.items():
        assert v == set(VARIANTS), (k, v)
    print(f"[integrity] 750 rows, 150 groups x 5 variants OK; "
         f"main/ablation SHA verified.")
    return rows, sha_before


# ---------------------------------------------------------------------
# Phase 2/22 -- paired table
# ---------------------------------------------------------------------
def build_paired_table(rows):
    idx = {(r["variant"], r["problem"], r["seed"]): r for r in rows}
    records = []
    for df in DFS:
        for seed in range(300000, 300030):
            rec = {"DF": df, "seed": seed}
            for v in VARIANTS:
                r = idx[(v, df, seed)]
                rec[f"{v}_migd_end"] = r["migd_end"]
                rec[f"{v}_migd_response"] = r["migd_response"]
                rec[f"{v}_FE"] = r["fes_used"]
            rec["d_C1"] = rec["H30_B6K_migd_end"] - rec["H100_B6K_migd_end"]
            rec["d_C2"] = rec["H30_B6K_migd_end"] - rec["H30_B18K_migd_end"]
            rec["d_C3"] = rec["H30_B18K_migd_end"] - rec["H100_B18K_migd_end"]
            rec["d_C4"] = rec["H100_B6K_migd_end"] - rec["H100_B18K_migd_end"]
            horizon_gain_6K = rec["H30_B6K_migd_end"] - rec["H100_B6K_migd_end"]
            horizon_gain_18K = rec["H30_B18K_migd_end"] - rec["H100_B18K_migd_end"]
            rec["interaction"] = horizon_gain_18K - horizon_gain_6K
            records.append(rec)
    df_table = pd.DataFrame.from_records(records)
    assert len(df_table) == 150
    out = os.path.join(RESULTS, "horizon_5_5A_paired.csv")
    df_table.to_csv(out, index=False)
    print(f"[paired] wrote {out} ({len(df_table)} rows)")
    return df_table


# ---------------------------------------------------------------------
# Phase 13-15, 23 -- contrasts C1-C4
# ---------------------------------------------------------------------
def analyze_contrast(paired, first_v, second_v):
    per_df = []
    for df in DFS:
        sub = paired[paired["DF"] == df].sort_values("seed")
        first = sub[f"{first_v}_migd_end"].to_numpy(dtype=float)
        second = sub[f"{second_v}_migd_end"].to_numpy(dtype=float)
        assert len(first) == 30 and len(second) == 30

        d_vec = first - second
        wres = wilcoxon_paired(first, second)     # d = first-second internally
        rres = rank_biserial(first, second)        # r_rb>0 favors second

        per_df.append(dict(
            DF=df, n=30,
            median_first=float(np.median(first)),
            median_second=float(np.median(second)),
            mean_first=float(np.mean(first)), mean_second=float(np.mean(second)),
            median_diff=float(np.median(d_vec)),
            W=wres["W"], n_nonzero=wres["n_nonzero"], p_raw=wres["p_raw"],
            W_plus=rres["W_plus"], W_minus=rres["W_minus"],
            rank_biserial=rres["r_rb"],
        ))

    p_raw_list = [r["p_raw"] for r in per_df]
    assert len(p_raw_list) == 5
    p_holm, reject = holm_correction(p_raw_list, alpha=ALPHA)
    for i, r in enumerate(per_df):
        r["p_holm"] = p_holm[i]
        r["reject_holm"] = reject[i]
        r["effect_magnitude"] = effect_magnitude(r["rank_biserial"])
        raw_symbol = classify_primary(r["p_holm"], r["median_diff"], alpha=ALPHA)
        symbol_map = {"SA+": "SECOND+", "SA-": "FIRST+", "=": "=",
                     "significant/mixed-zero-median": "significant_mixed_zero_median"}
        r["result"] = symbol_map[raw_symbol]
    return pd.DataFrame(per_df)


CSV_COLS = ["DF", "n", "median_first", "median_second", "median_diff",
           "W", "p_raw", "p_holm", "rank_biserial", "result"]


def run_contrasts(paired):
    tables = {}
    for name, (first_v, second_v) in CONTRASTS.items():
        tbl = analyze_contrast(paired, first_v, second_v)
        out = os.path.join(RESULTS, f"horizon_5_5A_{name}.csv")
        tbl[CSV_COLS].to_csv(out, index=False)
        print(f"[{name}] wrote {out} ({len(tbl)} rows)  "
             f"first={first_v} second={second_v}")
        tables[name] = tbl
    return tables


# ---------------------------------------------------------------------
# Phase 18 -- A0 contextual descriptive medians (no inferential test)
# ---------------------------------------------------------------------
def a0_context(paired):
    records = []
    for df in DFS:
        sub = paired[paired["DF"] == df]
        rec = {"DF": df}
        for v in VARIANTS:
            vals = sub[f"{v}_migd_end"].to_numpy(dtype=float)
            rec[f"{v}_median"] = float(np.median(vals))
            rec[f"{v}_IQR"] = float(q3(vals) - q1(vals))
        records.append(rec)
    tbl = pd.DataFrame.from_records(records)
    out = os.path.join(RESULTS, "horizon_5_5A_a0_context.csv")
    tbl.to_csv(out, index=False)
    print(f"[a0_context] wrote {out} ({len(tbl)} rows)")
    return tbl


# ---------------------------------------------------------------------
# Phase 24 -- training-behavior table (already aggregated by the main
# script into horizon_5_5A_training_windows.csv; just re-load/validate)
# ---------------------------------------------------------------------
def load_training_windows():
    if not os.path.exists(WINDOWS_PATH):
        return None
    return pd.read_csv(WINDOWS_PATH)


# ---------------------------------------------------------------------
# Phase 17 hypothesis summary + Phase 16 decision logic
# ---------------------------------------------------------------------
def contrast_summary(tables):
    summary = {}
    for name, tbl in tables.items():
        second_c = int((tbl["result"] == "SECOND+").sum())
        first_c = int((tbl["result"] == "FIRST+").sum())
        eq_c = int((tbl["result"] == "=").sum())
        summary[name] = dict(
            first=CONTRASTS[name][0], second=CONTRASTS[name][1],
            SECOND_plus_count=second_c, FIRST_plus_count=first_c,
            no_significant_difference_count=eq_c,
            per_df_results=dict(zip(tbl["DF"], tbl["result"])),
        )
    return summary


def decide_verdict(summary):
    """Phase 16 mechanistic decision logic (not prettiest-average)."""
    c1 = summary["C1_horizon_6K"]      # H30_B6K vs H100_B6K, SECOND+ = H100 better
    c2 = summary["C2_budget_H30"]      # H30_B6K vs H30_B18K, SECOND+ = more budget helps
    c3 = summary["C3_horizon_18K"]     # H30_B18K vs H100_B18K, SECOND+ = H100 better
    c4 = summary["C4_budget_H100"]     # H100_B6K vs H100_B18K, SECOND+ = more budget helps

    horizon_wins = c1["SECOND_plus_count"] + c3["SECOND_plus_count"]
    horizon_losses = c1["FIRST_plus_count"] + c3["FIRST_plus_count"]
    budget_wins = c2["SECOND_plus_count"] + c4["SECOND_plus_count"]
    budget_losses = c2["FIRST_plus_count"] + c4["FIRST_plus_count"]

    horizon_reliable = horizon_wins > 0 and horizon_losses == 0
    budget_reliable = budget_wins > 0 and budget_losses == 0
    horizon_any = horizon_wins > 0
    budget_any = budget_wins > 0
    horizon_mixed = horizon_wins > 0 and horizon_losses > 0
    budget_mixed = budget_wins > 0 and budget_losses > 0

    if horizon_reliable and not budget_any:
        verdict = "HORIZON-SUPPORTED"
        reason = (f"Horizon contrasts (C1,C3) show {horizon_wins} DF-level "
                 f"SECOND+ (H100 better) results and 0 FIRST+ (H30 better); "
                 f"budget contrasts (C2,C4) show 0 SECOND+ (more transitions "
                 f"at fixed horizon did not help).")
    elif budget_reliable and not horizon_any:
        verdict = "BUDGET-SUPPORTED"
        reason = (f"Budget contrasts (C2,C4) show {budget_wins} DF-level "
                 f"SECOND+ (more transitions helped) results and 0 FIRST+; "
                 f"horizon contrasts (C1,C3) show 0 SECOND+.")
    elif horizon_reliable and budget_reliable:
        verdict = "BOTH-SUPPORTED"
        reason = (f"Both horizon ({horizon_wins} SECOND+, 0 FIRST+) and "
                 f"budget ({budget_wins} SECOND+, 0 FIRST+) contrasts show "
                 f"reliable one-directional support.")
    elif not horizon_any and not budget_any:
        verdict = "NEITHER-SUPPORTED"
        reason = ("Neither horizon nor budget contrasts show any "
                 "Holm-significant DF favoring the larger-horizon or "
                 "larger-budget condition.")
    else:
        verdict = "PROBLEM-DEPENDENT"
        reason = (f"Mixed/reversing evidence: horizon wins={horizon_wins} "
                 f"losses={horizon_losses} (mixed={horizon_mixed}); "
                 f"budget wins={budget_wins} losses={budget_losses} "
                 f"(mixed={budget_mixed}). Effects do not point consistently "
                 f"in one direction across DFs or across the two contrasts "
                 f"testing the same factor -- no single global mechanism is "
                 f"forced.")
    return verdict, reason, dict(
        horizon_wins=horizon_wins, horizon_losses=horizon_losses,
        budget_wins=budget_wins, budget_losses=budget_losses)


def main():
    t0 = time.time()
    rows, sha_before = load_and_verify()
    paired = build_paired_table(rows)
    tables = run_contrasts(paired)
    a0_tbl = a0_context(paired)
    windows = load_training_windows()

    summary = contrast_summary(tables)
    verdict, reason, counts = decide_verdict(summary)

    out_summary = dict(
        contrasts=summary, decision_counts=counts,
        scientific_verdict=verdict, verdict_reason=reason,
        label="targeted_exploratory_diagnostic",
        primary_metric="migd_end", n_dfs=5, n_seeds_per_df=30,
    )
    with open(os.path.join(RESULTS, "horizon_5_5A_summary.json"), "w") as f:
        json.dump(out_summary, f, indent=2, default=str)
    print(f"\n[summary] scientific verdict: {verdict}")
    print(f"  reason: {reason}")

    sha_after = sha256_of(RUNS_PATH)
    assert sha_after == sha_before, "STOP: horizon raw file mutated"
    main_sha_after = sha256_of(MAIN_RAW_PATH)
    abl_sha_after = sha256_of(ABLATION_RAW_PATH)
    assert main_sha_after == MAIN_SHA_EXPECTED
    assert abl_sha_after == ABLATION_SHA_EXPECTED
    print(f"\nSHA unchanged: horizon={sha_after==sha_before} "
         f"main={main_sha_after==MAIN_SHA_EXPECTED} "
         f"ablation={abl_sha_after==ABLATION_SHA_EXPECTED}")
    print(f"Stats pipeline time: {time.time()-t0:.1f}s")

    return dict(paired=paired, tables=tables, a0_tbl=a0_tbl,
               windows=windows, summary=out_summary)


if __name__ == "__main__":
    main()
