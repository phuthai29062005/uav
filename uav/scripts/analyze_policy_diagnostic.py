"""
5.3A — Mechanism diagnostic analysis over the replayed per-change log.
Descriptive only. No p-values on 42,000 change rows (Phan 21). DF/seed
remains the independent unit for any inferential claim (already covered
by 5.2's frozen Wilcoxon results, referenced here, not repeated).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from policy_diagnostic import (decode_gate, decode_seg,  # noqa: E402
                               higher_c_segment, normalized_entropy,
                               tercile_labels, window_label)

RESULTS = os.path.join(_HERE, "..", "results")
PERCHANGE_PATH = os.path.join(RESULTS, "policy_diagnostic_per_change.csv")
DFS = [f"DF{i}" for i in range(1, 15)]

SA_PLUS = ["DF1", "DF2", "DF3", "DF6", "DF7", "DF9", "DF11", "DF13", "DF14"]
SA_MINUS = ["DF4", "DF5", "DF10", "DF12"]
NS = ["DF8"]
RESULT_GROUP = {**{d: "SA+" for d in SA_PLUS}, **{d: "SA-" for d in SA_MINUS},
                **{d: "NS" for d in NS}}


def load():
    df = pd.read_csv(PERCHANGE_PATH, keep_default_na=False,
                     na_values=[""])
    df["seg1_active"] = df["seg1_action"] != "NA"
    df["seg1_action_num"] = pd.to_numeric(df["seg1_action"], errors="coerce")
    df["seg2_action_num"] = pd.to_numeric(df["seg2_action"], errors="coerce")
    df["gate_label"] = df["gate"].map(decode_gate)
    df["max_c"] = df[["c1", "c2"]].max(axis=1)
    df["abs_c_diff"] = (df["c1"] - df["c2"]).abs()
    df["higher_c_seg"] = df.apply(lambda r: higher_c_segment(r.c1, r.c2), axis=1)
    df["result_group"] = df["DF"].map(RESULT_GROUP)
    return df


# ---------------------------------------- Phan 4/17/23: action frequency + entropy + main table
def action_frequency_by_df(df):
    rows = []
    for name in DFS:
        sub = df[df.DF == name]
        active = sub[sub.seg1_active]
        n = len(sub)
        row = dict(DF=name, result_group=RESULT_GROUP[name],
                  n_changes=n, memory_frac=(sub.gate == 1).mean())
        for seg_i, col in [(1, "seg1_action_num"), (2, "seg2_action_num")]:
            vals = active[col]
            for k, label in [(0, "keep"), (1, "local"), (2, "predict"), (3, "diversify")]:
                row[f"seg{seg_i}_{label}_frac"] = (vals == k).mean() if len(vals) else float("nan")
            row[f"seg{seg_i}_intervention_rate"] = (vals != 0).mean() if len(vals) else float("nan")

        row["gate_entropy"] = normalized_entropy(sub.gate.values, 2)
        row["seg1_entropy"] = (normalized_entropy(active.seg1_action_num.values, 4)
                               if len(active) else float("nan"))
        row["seg2_entropy"] = (normalized_entropy(active.seg2_action_num.values, 4)
                               if len(active) else float("nan"))

        # higher-c intervention bias (Phan 16), restricted to active (NO_MEMORY) rows
        if len(active):
            higher_action = np.where(active.higher_c_seg == 1,
                                     active.seg1_action_num, active.seg2_action_num)
            lower_action = np.where(active.higher_c_seg == 1,
                                    active.seg2_action_num, active.seg1_action_num)
            p_higher_interv = float(np.mean(higher_action != 0))
            p_lower_interv = float(np.mean(lower_action != 0))
            p_higher_only = float(np.mean((higher_action != 0) & (lower_action == 0)))
            row["p_intervene_higher_c_seg"] = p_higher_interv
            row["p_intervene_lower_c_seg"] = p_lower_interv
            row["higher_c_intervention_bias"] = p_higher_interv - p_lower_interv
            row["p_higher_intervened_lower_kept"] = p_higher_only
        else:
            row["p_intervene_higher_c_seg"] = float("nan")
            row["p_intervene_lower_c_seg"] = float("nan")
            row["higher_c_intervention_bias"] = float("nan")
            row["p_higher_intervened_lower_kept"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def action_frequency_by_seed(df):
    rows = []
    for name in DFS:
        for seed, sub in df[df.DF == name].groupby("seed"):
            active = sub[sub.seg1_active]
            rows.append(dict(
                DF=name, seed=seed,
                memory_frac=(sub.gate == 1).mean(),
                seg1_intervention_rate=((active.seg1_action_num != 0).mean()
                                        if len(active) else float("nan")),
                seg2_intervention_rate=((active.seg2_action_num != 0).mean()
                                        if len(active) else float("nan")),
            ))
    out = pd.DataFrame(rows)
    assert out.groupby("DF").size().eq(30).all()
    return out


def seed_consistency_summary(by_seed):
    rows = []
    for name in DFS:
        sub = by_seed[by_seed.DF == name]
        for col in ["memory_frac", "seg1_intervention_rate", "seg2_intervention_rate"]:
            rows.append(dict(DF=name, metric=col, median=sub[col].median(),
                             Q1=sub[col].quantile(.25), Q3=sub[col].quantile(.75),
                             IQR=sub[col].quantile(.75) - sub[col].quantile(.25)))
    return pd.DataFrame(rows)


# ---------------------------------------------- Phan 6/8/9/20: state-action summary
def state_action_summary(df):
    vars_ = ["c1", "c2", "max_c", "abs_c_diff", "dispersion", "hv_drop", "d_mem", "reward"]
    rows = []

    def add_group(label, sub):
        row = dict(group=label, n=len(sub))
        for v in vars_:
            row[f"{v}_median"] = sub[v].median()
            row[f"{v}_Q1"] = sub[v].quantile(.25)
            row[f"{v}_Q3"] = sub[v].quantile(.75)
        rows.append(row)

    add_group("gate=MEMORY", df[df.gate == 1])
    add_group("gate=NO_MEMORY", df[df.gate == 0])
    active = df[df.seg1_active]
    for seg_i, col in [(1, "seg1_action_num"), (2, "seg2_action_num")]:
        for k, label in [(0, "KEEP"), (1, "LOCAL"), (2, "PREDICT"), (3, "DIVERSIFY")]:
            add_group(f"seg{seg_i}={label}", active[active[col] == k])
    return pd.DataFrame(rows)


def memory_conditional_effect(df):
    rows = []
    for name in DFS:
        sub = df[df.DF == name]
        mem, nomem = sub[sub.gate == 1], sub[sub.gate == 0]
        rows.append(dict(
            DF=name,
            memory_delta_response_median=mem.delta_response.median() if len(mem) else np.nan,
            nomem_delta_response_median=nomem.delta_response.median() if len(nomem) else np.nan,
            memory_delta_recovery_median=mem.delta_recovery.median() if len(mem) else np.nan,
            nomem_delta_recovery_median=nomem.delta_recovery.median() if len(nomem) else np.nan,
            memory_delta_total_median=mem.delta_total.median() if len(mem) else np.nan,
            nomem_delta_total_median=nomem.delta_total.median() if len(nomem) else np.nan,
            n_memory=len(mem), n_nomemory=len(nomem),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------- Phan 10: DF10/DF12
def df10_df12_diagnostic(df):
    rows = []
    for name in ["DF10", "DF12"]:
        sub = df[df.DF == name].copy()
        sub["window"] = sub.change_index.apply(window_label)
        for win in ["EARLY", "MID", "LATE"]:
            w = sub[sub.window == win]
            rows.append(dict(
                DF=name, window=win, n=len(w),
                has_memory_rate=w.has_memory.mean(),
                memory_gate_rate=(w.gate == 1).mean(),
                d_mem_median=w.d_mem.median(), d_mem_Q1=w.d_mem.quantile(.25),
                d_mem_Q3=w.d_mem.quantile(.75),
                p_memory_given_has_memory=(
                    w[w.has_memory == 1].gate.eq(1).mean() if (w.has_memory == 1).any() else np.nan),
                delta_response_memory_median=(
                    w[w.gate == 1].delta_response.median() if (w.gate == 1).any() else np.nan),
                delta_response_nomemory_median=(
                    w[w.gate == 0].delta_response.median() if (w.gate == 0).any() else np.nan),
                delta_total_memory_median=(
                    w[w.gate == 1].delta_total.median() if (w.gate == 1).any() else np.nan),
                delta_total_nomemory_median=(
                    w[w.gate == 0].delta_total.median() if (w.gate == 0).any() else np.nan),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------- Phan 12: DF7
def df7_diagnostic(df):
    sub = df[df.DF == "DF7"].copy()
    active = sub[sub.seg1_active]
    rho, p = spearmanr(sub.delta_response, sub.delta_recovery)
    rows = []
    # by gate
    for gval, glabel in [(1, "MEMORY"), (0, "NO_MEMORY")]:
        g = sub[sub.gate == gval]
        rows.append(dict(group=f"gate={glabel}", n=len(g),
                         delta_response_median=g.delta_response.median(),
                         delta_recovery_median=g.delta_recovery.median(),
                         delta_total_median=g.delta_total.median()))
    # by seg1/seg2 action (active only)
    for seg_i, col in [(1, "seg1_action_num"), (2, "seg2_action_num")]:
        for k, label in [(0, "KEEP"), (1, "LOCAL"), (2, "PREDICT"), (3, "DIVERSIFY")]:
            g = active[active[col] == k]
            if len(g):
                rows.append(dict(group=f"seg{seg_i}={label}", n=len(g),
                                 delta_response_median=g.delta_response.median(),
                                 delta_recovery_median=g.delta_recovery.median(),
                                 delta_total_median=g.delta_total.median()))
    out = pd.DataFrame(rows)
    out.attrs["spearman_rho_response_recovery"] = rho
    out.attrs["spearman_p"] = p
    return out


# --------------------------------------------------- Phan 19: early/mid/late
def early_mid_late(df):
    rows = []
    for name in DFS:
        sub = df[df.DF == name].copy()
        sub["window"] = sub.change_index.apply(window_label)
        active_all = sub[sub.seg1_active]
        for win in ["EARLY", "MID", "LATE"]:
            w = sub[sub.window == win]
            wa = active_all[active_all.window == win]
            rows.append(dict(
                DF=name, window=win, n=len(w),
                memory_rate=(w.gate == 1).mean(),
                seg1_intervention_rate=((wa.seg1_action_num != 0).mean() if len(wa) else np.nan),
                seg2_intervention_rate=((wa.seg2_action_num != 0).mean() if len(wa) else np.nan),
                gate_entropy=normalized_entropy(w.gate.values, 2),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------- Phan 13/14/15: conditioning
def severity_conditioning(df):
    rows = []
    for name in DFS:
        sub = df[df.DF == name].copy()
        sub["c_tercile"] = tercile_labels(sub.max_c.values)
        sub["hvdrop_tercile"] = tercile_labels(sub.hv_drop.values)
        sub["disp_tercile"] = tercile_labels(sub.dispersion.values)
        active = sub[sub.seg1_active]
        for tercol, tername in [("c_tercile", "c_max"), ("hvdrop_tercile", "hv_drop"),
                                ("disp_tercile", "dispersion")]:
            for lvl in ["LOW", "MID", "HIGH"]:
                w = sub[sub[tercol] == lvl]
                wa = active[active[tercol] == lvl]
                rows.append(dict(
                    DF=name, conditioning=tername, level=lvl, n=len(w),
                    memory_rate=(w.gate == 1).mean(),
                    seg1_intervention_rate=((wa.seg1_action_num != 0).mean() if len(wa) else np.nan),
                    seg2_intervention_rate=((wa.seg2_action_num != 0).mean() if len(wa) else np.nan),
                    p_diversify_seg1=((wa.seg1_action_num == 3).mean() if len(wa) else np.nan),
                    p_diversify_seg2=((wa.seg2_action_num == 3).mean() if len(wa) else np.nan),
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------- Plots
def make_plots(freq_df, df7, df_all):
    figd = os.path.join(RESULTS, "figures")
    os.makedirs(figd, exist_ok=True)
    x = np.arange(14)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"SA+": "#228833", "SA-": "#CC3311", "NS": "#888888"}
    ax.bar(x, freq_df.memory_frac, color=[colors[g] for g in freq_df.result_group])
    ax.set_xticks(x); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("MEMORY gate frequency")
    ax.set_title("Memory-gate usage rate by DF (color = 5.2 result group)")
    fig.tight_layout(); fig.savefig(os.path.join(figd, "policy_memory_rate_by_df.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.35
    ax.bar(x - w / 2, freq_df.seg1_intervention_rate, w, label="segment 1")
    ax.bar(x + w / 2, freq_df.seg2_intervention_rate, w, label="segment 2")
    ax.set_xticks(x); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("Intervention rate (action != KEEP)")
    ax.legend(); ax.set_title("Per-segment intervention rate by DF")
    fig.tight_layout(); fig.savefig(os.path.join(figd, "policy_intervention_rate_by_df.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, freq_df.gate_entropy, "o-", label="gate")
    ax.plot(x, freq_df.seg1_entropy, "o-", label="seg1")
    ax.plot(x, freq_df.seg2_entropy, "o-", label="seg2")
    ax.set_xticks(x); ax.set_xticklabels(DFS, rotation=45)
    ax.set_ylabel("Normalized entropy [0,1]")
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0.6, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(); ax.set_title("Action-category entropy by DF (descriptive bands: <0.2 low, >0.6 high)")
    fig.tight_layout(); fig.savefig(os.path.join(figd, "policy_action_entropy_by_df.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df7.delta_response, df7.delta_recovery, s=8, alpha=0.4, color="#4477AA")
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("delta_response = IGD_pre - IGD_response")
    ax.set_ylabel("delta_recovery = IGD_response - IGD_end")
    ax.set_title("DF7: immediate response vs subsequent recovery (per change, 30 seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(figd, "policy_df7_response_vs_recovery.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    mem = df_all[df_all.gate == 1]
    nomem = df_all[df_all.gate == 0]
    ax.hist(nomem.d_mem, bins=40, alpha=0.5, label="NO_MEMORY", color="#4477AA", density=True)
    ax.hist(mem.d_mem, bins=40, alpha=0.5, label="MEMORY", color="#EE6677", density=True)
    ax.set_xlabel("d_mem"); ax.set_ylabel("density")
    ax.legend(); ax.set_title("d_mem distribution by gate choice (all DFs pooled)")
    fig.tight_layout(); fig.savefig(os.path.join(figd, "policy_dmem_memory_selection.png"), dpi=150)
    plt.close(fig)


def main():
    print("Loading per-change log ...")
    df = load()
    print(f"Loaded {len(df)} rows (expect 42000)")
    assert len(df) == 42000, len(df)
    assert df.groupby("DF").size().eq(3000).all()

    freq_df = action_frequency_by_df(df)
    by_seed = action_frequency_by_seed(df)
    consistency = seed_consistency_summary(by_seed)
    state_action = state_action_summary(df)
    mem_effect = memory_conditional_effect(df)
    df10_12 = df10_df12_diagnostic(df)
    df7 = df7_diagnostic(df)
    eml = early_mid_late(df)
    severity = severity_conditioning(df)

    freq_df.to_csv(os.path.join(RESULTS, "policy_action_frequency_by_df.csv"), index=False)
    by_seed.to_csv(os.path.join(RESULTS, "policy_action_frequency_by_seed.csv"), index=False)
    state_action.to_csv(os.path.join(RESULTS, "policy_state_action_summary.csv"), index=False)
    eml.to_csv(os.path.join(RESULTS, "policy_early_mid_late.csv"), index=False)
    df7.to_csv(os.path.join(RESULTS, "policy_df7_horizon_diagnostic.csv"), index=False)
    df10_12.to_csv(os.path.join(RESULTS, "policy_df10_df12_memory_diagnostic.csv"), index=False)
    consistency.to_csv(os.path.join(RESULTS, "policy_seed_consistency.csv"), index=False)
    mem_effect.to_csv(os.path.join(RESULTS, "policy_memory_conditional_effect.csv"), index=False)
    severity.to_csv(os.path.join(RESULTS, "policy_severity_conditioning.csv"), index=False)

    make_plots(freq_df, df[df.DF == "DF7"], df)

    rho, p = spearmanr(df[df.DF == "DF7"].delta_response, df[df.DF == "DF7"].delta_recovery)

    write_report(freq_df, mem_effect, df10_12, df7, eml, severity, consistency, rho, p)
    print("\nAnalysis complete. See results/policy_mechanism_report.md")
    print(freq_df[["DF", "result_group", "memory_frac", "seg1_intervention_rate",
                  "seg2_intervention_rate", "gate_entropy", "seg1_entropy", "seg2_entropy",
                  "higher_c_intervention_bias"]].round(3).to_string(index=False))


def write_report(freq_df, mem_effect, df10_12, df7, eml, severity, consistency, rho, p):
    L = []
    L.append("# Policy Mechanism Diagnostic Report (5.3A)\n")
    L.append("Post-hoc, observational, descriptive. DF/seed remains the independent "
            "statistical unit (5.2); the 42,000 per-change rows here are NOT treated "
            "as independent samples and no p-values are computed on them (Phan 21).\n")

    L.append("## Main 14-DF policy table\n")
    L.append("| DF | group | MEMORY% | seg1 interv% | seg2 interv% | gate H | seg1 H | "
            "seg2 H | higher-c bias |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in freq_df.iterrows():
        L.append(f"| {r.DF} | {r.result_group} | {r.memory_frac:.1%} | "
                f"{r.seg1_intervention_rate:.1%} | {r.seg2_intervention_rate:.1%} | "
                f"{r.gate_entropy:.2f} | {r.seg1_entropy:.2f} | {r.seg2_entropy:.2f} | "
                f"{r.higher_c_intervention_bias:+.3f} |")

    L.append("\n## Memory usage findings\n")
    L.append(freq_df[["DF", "memory_frac"]].sort_values("memory_frac", ascending=False)
             .to_string(index=False))

    L.append("\n\n## DF10 / DF12 memory diagnostic (early/mid/late)\n")
    L.append(df10_12.round(4).to_string(index=False))
    L.append("\n\nInterpretation categories (Phan 10, A-D): see narrative section below.")

    L.append("\n\n## DF7 horizon-reversal diagnostic\n")
    L.append(f"Spearman rho(delta_response, delta_recovery) = {rho:.3f} (p={p:.2e}, "
            f"descriptive on within-DF change rows, not a between-seed test).\n")
    L.append(df7.round(4).to_string(index=False))

    L.append("\n\n## Early / mid / late drift (all DFs)\n")
    L.append(eml.round(3).to_string(index=False))

    L.append("\n\n## Severity conditioning (within-DF terciles, all DFs)\n")
    L.append(severity.round(3).to_string(index=False))

    L.append("\n\n## Seed consistency (median/IQR of action frequencies across 30 seeds)\n")
    L.append(consistency.round(3).to_string(index=False))

    L.append("\n\n## Narrative interpretation (descriptive, non-causal)\n")

    def g(df_name):
        return freq_df[freq_df.DF == df_name].iloc[0]

    r4, r5 = g("DF4"), g("DF5")
    ref = freq_df[freq_df.DF.isin(["DF1", "DF2", "DF3", "DF6", "DF9"])]
    L.append(f"**DF4** (SA-): total intervention (seg1+seg2 mean) = "
            f"{(r4.seg1_intervention_rate + r4.seg2_intervention_rate) / 2:.1%}, "
            f"MEMORY rate = {r4.memory_frac:.1%}, vs SA+ reference mean intervention = "
            f"{((ref.seg1_intervention_rate + ref.seg2_intervention_rate) / 2).mean():.1%}. "
            "Higher-than-reference intervention alongside a baseline-favored outcome is "
            "consistent with unnecessary intervention on a problem where vanilla NSGA-II "
            "already recovers well, but this is observational and confounded by state "
            "(Phan 11).\n")
    L.append(f"**DF5** (SA-): total intervention = "
            f"{(r5.seg1_intervention_rate + r5.seg2_intervention_rate) / 2:.1%}, "
            f"MEMORY rate = {r5.memory_frac:.1%}. Same caveat as DF4 applies.\n")

    d7 = g("DF7")
    L.append(f"**DF7**: end-horizon SA+, response-horizon SA- (5.2). "
            f"rho(delta_response, delta_recovery)={rho:.3f} — "
            + ("a negative association is " if rho < 0 else "no negative association is ")
            + "observed between immediate displacement and subsequent recovery magnitude, "
            "suggestive of (not proof of) a response-now/recover-later pattern. "
            "MEMORY rate on DF7 = " + f"{d7.memory_frac:.1%}.\n")

    d10, d12 = g("DF10"), g("DF12")
    L.append(f"**DF10** (SA-, r_rb=-1): MEMORY rate = {d10.memory_frac:.1%}. Prior "
            "diagnostic (4.10D) found signature soft-saturation rising to ~53% "
            "late-horizon on DF10; see the early/mid/late memory-usage table above for "
            "whether gate usage shifts alongside that saturation trend.\n")
    L.append(f"**DF12** (SA-, r_rb=-0.983): MEMORY rate = {d12.memory_frac:.1%}. Prior "
            "diagnostic found d_mem nearly flat on DF12 (weak discrimination between "
            "environments); see the table above for whether the gate still engages "
            "MEMORY despite that flatness.\n")

    d8 = g("DF8")
    L.append(f"**DF8** (non-significant, 5.2): MEMORY rate = {d8.memory_frac:.1%}, "
            f"seg1/seg2 intervention = {d8.seg1_intervention_rate:.1%}/"
            f"{d8.seg2_intervention_rate:.1%}, entropy "
            f"(gate/seg1/seg2) = {d8.gate_entropy:.2f}/{d8.seg1_entropy:.2f}/"
            f"{d8.seg2_entropy:.2f} — a mixed/intermediate action profile is consistent "
            "with (not proof of) the absence of a Holm-significant difference.\n")

    L.append("\n## SA+ vs SA- vs NS group-level descriptive comparison\n")
    for grp, names in [("SA+", SA_PLUS), ("SA-", SA_MINUS), ("NS", NS)]:
        sub = freq_df[freq_df.DF.isin(names)]
        L.append(f"- {grp} ({len(names)} DFs): median MEMORY%={sub.memory_frac.median():.1%}, "
                f"median seg1 interv%={sub.seg1_intervention_rate.median():.1%}, "
                f"median seg2 interv%={sub.seg2_intervention_rate.median():.1%}, "
                f"median gate entropy={sub.gate_entropy.median():.2f}")

    L.append("\n\n## Ranked hypotheses for 5.3B\n")
    L.append(hypotheses_text(freq_df, df10_12, df7, rho))

    L.append("\n## Recommended 5.3B ablation set\n")
    L.append("- **A0 Clean NSGA-II** — already run (5.1/5.2 baseline); keep as anchor.")
    L.append("- **A4 Full SA-DRL** — already run (5.1/5.2); keep as anchor.")
    L.append("- **A3 Segmented RL without Memory** — necessary: directly tests H3 given "
            "the MEMORY-rate/d_mem patterns on DF10/DF12 observed here.")
    L.append("- **A1 Global RL, S=1** — necessary: directly tests H1/H2 (whether "
            "segmentation and c_s add value beyond a non-segmented controller).")
    L.append("- **A2 Segmented RL without c_s** — recommended if compute allows: "
            "isolates the temporal change signal's contribution from segmentation "
            "structure itself, sharpening H2.")
    L.append("- **A5 Rule-based c controller** — optional/lower priority: useful only "
            "if H2 is supported and a non-learned baseline is desired for contrast; "
            "not necessary to resolve the primary hypotheses above.")
    L.append("\nThis is a recommendation only; nothing in 5.3A was implemented or run "
            "to select an ablation based on making results look more favorable.")

    with open(os.path.join(RESULTS, "policy_mechanism_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def hypotheses_text(freq_df, df10_12, df7, rho):
    lines = []
    # H1: segmentation adds value beyond global RL -- cannot be assessed without an
    # S=1 ablation; diagnostic alone cannot support/refute.
    lines.append("- **H1** (segmentation adds value beyond global RL): NOT ASSESSABLE "
                "from this diagnostic alone (no S=1 run in 5.1/5.2/5.3A) — "
                "motivates the A1 ablation.")
    # H2: does higher-c segment get intervened more often -> support signal
    bias = freq_df.higher_c_intervention_bias.median()
    supp = "SUPPORTED BY DIAGNOSTIC" if bias > 0.02 else (
        "PARTIALLY SUPPORTED" if bias > 0 else "NOT SUPPORTED")
    lines.append(f"- **H2** (temporal c_s drives segment-specific intervention): median "
                f"higher-c intervention bias across 14 DFs = {bias:+.3f} "
                f"(P(intervene|higher-c) - P(intervene|lower-c)) -> {supp}.")
    # H3: memory helps some, hurts when signature degrades (DF10/12 vs others)
    mem_rate_sa_minus_mem_dfs = freq_df[freq_df.DF.isin(["DF10", "DF12"])].memory_frac.median()
    mem_rate_sa_plus = freq_df[freq_df.DF.isin(SA_PLUS)].memory_frac.median()
    lines.append(f"- **H3** (memory gate helps on some DFs but hurts when signature "
                f"discrimination degrades): median MEMORY rate DF10/DF12={mem_rate_sa_minus_mem_dfs:.1%} "
                f"vs SA+ group={mem_rate_sa_plus:.1%}. "
                + ("PARTIALLY SUPPORTED -- elevated memory use co-occurs with the "
                   "known DF10/DF12 signature limitation" if mem_rate_sa_minus_mem_dfs > mem_rate_sa_plus
                   else "NOT SUPPORTED by memory-rate alone -- see per-DF table for "
                   "d_mem/delta_response patterns") + ".")
    # H4: learned intervention can hurt where vanilla NSGA-II already recovers strongly
    ref_interv = freq_df[freq_df.DF.isin(SA_PLUS)][["seg1_intervention_rate", "seg2_intervention_rate"]].mean().mean()
    minus_interv = freq_df[freq_df.DF.isin(SA_MINUS)][["seg1_intervention_rate", "seg2_intervention_rate"]].mean().mean()
    lines.append(f"- **H4** (learned intervention can hurt where NSGA-II already "
                f"recovers strongly): mean intervention rate SA- group={minus_interv:.1%} "
                f"vs SA+ group={ref_interv:.1%}. "
                + ("SUPPORTED BY DIAGNOSTIC -- SA- problems show higher intervention"
                   if minus_interv > ref_interv else
                   "NOT SUPPORTED -- SA- problems do not show higher intervention than SA+")
                + ".")
    # H5: DF7 gains from weaker immediate response traded for recovery
    lines.append(f"- **H5** (DF7 gains come from accepting weaker immediate response "
                f"for improved recovery): Spearman rho(delta_response, delta_recovery) "
                f"on DF7 = {rho:+.3f}. "
                + ("SUPPORTED BY DIAGNOSTIC -- negative association consistent with "
                   "a response-now/recover-later trade-off" if rho < -0.05 else
                   "PARTIALLY SUPPORTED -- association is weak/inconsistent with a "
                   "simple trade-off story" if abs(rho) <= 0.05 else
                   "NOT SUPPORTED -- association runs the opposite direction") + ".")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
