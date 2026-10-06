"""E4V Statistical Analysis: Friedman + Wilcoxon signed-rank pairwise.

Reads e4v_full_results.json, produces:
  - Table A: MIGD results per problem × algorithm (mean ± std over 20 seeds)
  - Table B: Friedman test + Wilcoxon pairwise comparisons
  - Core hash verification
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_e4v_v1.src.problems.registry import BI_OBJECTIVE, TRI_OBJECTIVE

RESULTS_DIR = ROOT / "benchmark_e4v_v1" / "results"
PROBLEMS = BI_OBJECTIVE + TRI_OBJECTIVE
ALGORITHMS = ["NSGA-II", "D-NSGA-II", "B-C1", "B-C3", "B-C4"]
TAUTS = [10, 30]


def load_results():
    path = RESULTS_DIR / "e4v_full_results.json"
    with open(path) as f:
        return json.load(f)


def organize_migd(results):
    """Organize MIGD values into {taut: {problem: {algorithm: [values]}}}."""
    data = {}
    for taut in TAUTS:
        data[taut] = {}
        for prob in PROBLEMS:
            data[taut][prob] = {}
            for alg in ALGORITHMS:
                vals = [r["migd"] for r in results
                        if r["problem"] == prob and r["algorithm"] == alg
                        and r["taut"] == taut and "error" not in r]
                data[taut][prob][alg] = vals
    return data


def table_a(data, taut):
    """Mean ± std MIGD table, with rank per problem."""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"TABLE A: MIGD Results (τ_t = {taut})")
    lines.append(f"{'='*100}")

    header = f"{'Problem':<8}"
    for alg in ALGORITHMS:
        header += f"  {alg:>16}"
    lines.append(header)
    lines.append("-" * 100)

    rank_sums = {alg: 0 for alg in ALGORITHMS}
    win_counts = {alg: 0 for alg in ALGORITHMS}

    for prob in PROBLEMS:
        row = f"{prob:<8}"
        means = {}
        for alg in ALGORITHMS:
            vals = data[taut][prob][alg]
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                means[alg] = m
                row += f"  {m:.4f}±{s:.4f}"
            else:
                means[alg] = float("inf")
                row += f"  {'N/A':>16}"

        sorted_algs = sorted(ALGORITHMS, key=lambda a: means[a])
        for rank, alg in enumerate(sorted_algs, 1):
            rank_sums[alg] += rank
        win_counts[sorted_algs[0]] += 1
        lines.append(row)

    lines.append("-" * 100)
    row = f"{'Avg Rank':<8}"
    for alg in ALGORITHMS:
        avg_rank = rank_sums[alg] / len(PROBLEMS)
        row += f"  {avg_rank:>16.2f}"
    lines.append(row)

    row = f"{'Wins':<8}"
    for alg in ALGORITHMS:
        row += f"  {win_counts[alg]:>16d}"
    lines.append(row)

    return "\n".join(lines)


def friedman_test(data, taut):
    """Friedman test across all problems for given taut."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"FRIEDMAN TEST (τ_t = {taut})")
    lines.append(f"{'='*80}")

    all_migd = defaultdict(list)
    for prob in PROBLEMS:
        for alg in ALGORITHMS:
            vals = data[taut][prob][alg]
            all_migd[alg].append(np.mean(vals) if vals else float("inf"))

    groups = [all_migd[alg] for alg in ALGORITHMS]
    stat, p_value = stats.friedmanchisquare(*groups)
    lines.append(f"Friedman χ² = {stat:.4f}, p = {p_value:.6f}")
    lines.append(f"Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")

    return "\n".join(lines), p_value


def wilcoxon_pairwise(data, taut):
    """Wilcoxon signed-rank pairwise comparisons."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"WILCOXON SIGNED-RANK PAIRWISE (τ_t = {taut})")
    lines.append(f"{'='*80}")

    all_migd = {}
    for alg in ALGORITHMS:
        vals = []
        for prob in PROBLEMS:
            v = data[taut][prob][alg]
            vals.append(np.mean(v) if v else float("inf"))
        all_migd[alg] = np.array(vals)

    n_comparisons = len(ALGORITHMS) * (len(ALGORITHMS) - 1) // 2
    alpha_bonferroni = 0.05 / n_comparisons

    lines.append(f"{'Comparison':<25} {'Statistic':>10} {'p-value':>12} {'Significant':>12} {'Winner':>12}")
    lines.append("-" * 80)

    for i, alg_a in enumerate(ALGORITHMS):
        for alg_b in ALGORITHMS[i + 1:]:
            diff = all_migd[alg_a] - all_migd[alg_b]
            if np.all(diff == 0):
                lines.append(f"{alg_a} vs {alg_b:<12} {'N/A':>10} {'N/A':>12} {'TIED':>12} {'—':>12}")
                continue
            try:
                stat_val, p_val = stats.wilcoxon(all_migd[alg_a], all_migd[alg_b])
                sig = "YES" if p_val < alpha_bonferroni else "NO"
                winner = alg_a if np.mean(diff) < 0 else alg_b
                lines.append(f"{alg_a} vs {alg_b:<12} {stat_val:>10.1f} {p_val:>12.6f} {sig:>12} {winner:>12}")
            except Exception as e:
                lines.append(f"{alg_a} vs {alg_b:<12} {'ERR':>10} {str(e)[:20]:>12}")

    lines.append(f"\nBonferroni-corrected α = {alpha_bonferroni:.6f} ({n_comparisons} comparisons)")

    return "\n".join(lines)


def verify_core_hashes():
    """Verify frozen core files haven't been modified."""
    manifest_path = ROOT / "benchmark_e4v_v1" / "hashes" / "ORIGINAL_CORE_HASH_MANIFEST_BEFORE.json"
    if not manifest_path.exists():
        return "HASH MANIFEST NOT FOUND — CANNOT VERIFY"

    with open(manifest_path) as f:
        manifest = json.load(f)

    files = manifest.get("FILES", manifest)
    mismatches = []
    checked = 0
    for fpath, expected_hash in files.items():
        full_path = ROOT / fpath
        if not full_path.exists():
            mismatches.append(f"MISSING: {fpath}")
            continue
        actual = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual != expected_hash:
            mismatches.append(f"CHANGED: {fpath}")
        checked += 1

    if mismatches:
        return "CORE HASH VERIFICATION FAILED:\n  " + "\n  ".join(mismatches)
    return f"CORE HASH VERIFICATION: PASS ({checked} files unchanged)"


def main():
    results = load_results()
    data = organize_migd(results)

    errors = [r for r in results if "error" in r]
    n_ok = len(results) - len(errors)

    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("E4V BENCHMARK AUDIT — STATISTICAL ANALYSIS")
    output_lines.append("=" * 100)
    output_lines.append(f"Total runs: {len(results)} ({n_ok} OK, {len(errors)} errors)")
    output_lines.append("")

    hash_status = verify_core_hashes()
    output_lines.append(hash_status)

    for taut in TAUTS:
        output_lines.append(table_a(data, taut))
        friedman_text, p_val = friedman_test(data, taut)
        output_lines.append(friedman_text)
        if p_val < 0.05:
            output_lines.append(wilcoxon_pairwise(data, taut))

    output_lines.append("\n" + "=" * 100)
    output_lines.append("OVERALL VERDICT")
    output_lines.append("=" * 100)

    for taut in TAUTS:
        rank_sums = {alg: 0 for alg in ALGORITHMS}
        for prob in PROBLEMS:
            means = {alg: np.mean(data[taut][prob][alg]) if data[taut][prob][alg] else float("inf")
                     for alg in ALGORITHMS}
            sorted_algs = sorted(ALGORITHMS, key=lambda a: means[a])
            for rank, alg in enumerate(sorted_algs, 1):
                rank_sums[alg] += rank
        avg_ranks = {alg: rank_sums[alg] / len(PROBLEMS) for alg in ALGORITHMS}
        best = min(avg_ranks, key=avg_ranks.get)
        output_lines.append(f"\nτ_t={taut}: Best avg rank = {best} ({avg_ranks[best]:.2f})")
        drl_ranks = {a: avg_ranks[a] for a in ["B-C1", "B-C3", "B-C4"]}
        baseline_ranks = {a: avg_ranks[a] for a in ["NSGA-II", "D-NSGA-II"]}
        output_lines.append(f"  DRL ranks: {drl_ranks}")
        output_lines.append(f"  Baseline ranks: {baseline_ranks}")
        best_drl = min(drl_ranks, key=drl_ranks.get)
        best_baseline = min(baseline_ranks, key=baseline_ranks.get)
        if drl_ranks[best_drl] < baseline_ranks[best_baseline]:
            output_lines.append(f"  DRL ({best_drl}) BEATS best baseline ({best_baseline})")
        else:
            output_lines.append(f"  Best baseline ({best_baseline}) BEATS best DRL ({best_drl})")

    report = "\n".join(output_lines)
    print(report)

    outpath = RESULTS_DIR / "e4v_analysis_report.txt"
    with open(outpath, "w") as f:
        f.write(report)
    print(f"\nReport saved to {outpath}")

    summary_path = RESULTS_DIR / "e4v_summary.json"
    summary = {
        "total_runs": len(results),
        "errors": len(errors),
        "hash_verification": "PASS" if "PASS" in hash_status else "FAIL",
    }
    for taut in TAUTS:
        rank_sums = {alg: 0 for alg in ALGORITHMS}
        for prob in PROBLEMS:
            means = {alg: np.mean(data[taut][prob][alg]) if data[taut][prob][alg] else float("inf")
                     for alg in ALGORITHMS}
            sorted_algs = sorted(ALGORITHMS, key=lambda a: means[a])
            for rank, alg in enumerate(sorted_algs, 1):
                rank_sums[alg] += rank
        summary[f"avg_ranks_taut{taut}"] = {alg: rank_sums[alg] / len(PROBLEMS) for alg in ALGORITHMS}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
