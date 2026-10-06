"""E4V Full Experiment: 14 problems × 2 τ_t × 20 seeds × 5 arms = 2800 runs.

Protocol: CEC2018 DMOP benchmark, n_var=10, pop=100, nt=10, changes=30.
Arms: NSGA-II, D-NSGA-II, B-C1, B-C3, B-C4.
Seeds: 1-20 (experiment seeds, distinct from policy seeds 8001-8005).
DRL uses policy seed 8001 (first selected checkpoint per arm).

Incremental saving: results written after each problem×taut block to survive timeouts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import glob

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_e4v_v1.src.runner import run_single_experiment
from benchmark_e4v_v1.src.problems.registry import BI_OBJECTIVE, TRI_OBJECTIVE

PROBLEMS = BI_OBJECTIVE + TRI_OBJECTIVE
TAUTS = [10, 30]
SEEDS = list(range(1, 21))
ALGORITHMS = ["NSGA-II", "D-NSGA-II", "B-C1", "B-C3", "B-C4"]

CHECKPOINT_DIR = ROOT / "results" / "E4TTR_c4_gated_coordination_training_execution"
OUTPUT_DIR = ROOT / "benchmark_e4v_v1" / "results"


def find_checkpoint(arm_prefix: str, policy_seed: int = 8001) -> str:
    pattern = str(CHECKPOINT_DIR / f"{arm_prefix}*" / f"policy_{policy_seed}" / "checkpoints" / "ckpt_final.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return paths[0]


def save_results(all_results, outpath):
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = OUTPUT_DIR / "e4v_full_results.json"

    if outpath.exists():
        with open(outpath) as f:
            all_results = json.load(f)
        done_keys = {(r["problem"], r["algorithm"], r["seed"], r["taut"]) for r in all_results}
        print(f"Resuming: {len(all_results)} results already saved", flush=True)
    else:
        all_results = []
        done_keys = set()

    checkpoints = {
        "C1": find_checkpoint("C1_QMIX"),
        "C3": find_checkpoint("C3_PAIR"),
        "C4": find_checkpoint("C4_CAUSAL"),
    }

    total = len(PROBLEMS) * len(TAUTS) * len(SEEDS) * len(ALGORITHMS)
    remaining = total - len(done_keys)
    print(f"E4V FULL EXPERIMENT: {total} total, {remaining} remaining", flush=True)

    done = len(done_keys)
    t0 = time.time()
    block_count = 0

    for taut in TAUTS:
        for prob_name in PROBLEMS:
            block_new = 0
            for alg in ALGORITHMS:
                cp = checkpoints.get(alg[2:]) if alg.startswith("B-") else None
                for seed in SEEDS:
                    key = (prob_name, alg, seed, taut)
                    if key in done_keys:
                        continue
                    try:
                        r = run_single_experiment(
                            prob_name, alg, seed=seed, taut=taut, checkpoint_path=cp)
                        all_results.append(r)
                    except Exception as e:
                        print(f"ERROR: {prob_name}/{alg}/tau{taut}/s{seed}: {e}", flush=True)
                        all_results.append({
                            "problem": prob_name, "algorithm": alg,
                            "seed": seed, "taut": taut,
                            "migd": float("inf"), "error": str(e),
                        })
                    done += 1
                    block_new += 1
                    if done % 100 == 0 or done == total:
                        elapsed = time.time() - t0
                        rate = (done - len(done_keys) + len(done_keys)) / elapsed if elapsed > 0 else 0
                        actual_rate = block_new / elapsed if elapsed > 0 else 0
                        eta = (total - done) / actual_rate if actual_rate > 0 else 0
                        print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA", flush=True)

            if block_new > 0:
                save_results(all_results, outpath)
                block_count += 1
                print(f"  Saved ({len(all_results)} results, block {block_count})", flush=True)

    save_results(all_results, outpath)
    print(f"\nFinal results saved to {outpath}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)

    errors = [r for r in all_results if "error" in r]
    if errors:
        print(f"\n{len(errors)} ERRORS:", flush=True)
        for e in errors[:10]:
            print(f"  {e['problem']}/{e['algorithm']}/tau{e['taut']}/s{e['seed']}: {e['error']}", flush=True)


if __name__ == "__main__":
    main()
