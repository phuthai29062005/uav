"""
Huan luyen va danh gia SA-DRL-DMOEA tren CEC 2018 DF.

Bon che do:
  debug    : train va test cung bai — CO overfit, chi de bat bug
  online   : agent moi moi run, hoc trong luc chay (Bang 1)
  loo      : train 13 bai, dong bang, test bai con lai (Bang 2)
  ablation : S=1 vs S=2 — phan doan co dong gop khong (Bang 3)

Vi du:
  python train.py --mode debug    --problems DF1 --episodes 100
  python train.py --mode online   --problems all --runs 20
  python train.py --mode loo      --problems all --episodes 280
  python train.py --mode ablation --problems DF1,DF2,DF3 --episodes 150
"""
import argparse
import json
import os
import sys
import time

# Cho phep import cac module trong src/ khi chay tu scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
import torch
from pymoo.problems.dynamic.df import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14,
)

from baselines import NSGA2Baseline
from dqn_agent import DQNAgent
from dynamic_runner import run_sa_drl

ALL_PROBLEMS = {
    "DF1": DF1, "DF2": DF2, "DF3": DF3, "DF4": DF4, "DF5": DF5,
    "DF6": DF6, "DF7": DF7, "DF8": DF8, "DF9": DF9, "DF10": DF10,
    "DF11": DF11, "DF12": DF12, "DF13": DF13, "DF14": DF14,
}
TRI_OBJECTIVE = {"DF10", "DF11", "DF12", "DF13", "DF14"}

CFG = dict(D=10, n_t=10, tau_t=10, warm_up=50, n_elite=10)
TRAIN_CHANGES = 30        # rut ngan khi train cho nhanh
EVAL_CHANGES = 100        # dung chuan CEC khi danh gia
N_EVAL_RUNS = 30          # so seed doc lap khi danh gia (MSO: toi thieu 30)
EVAL_SEED_BASE = 100_000  # tach hoan toan khoi dai seed training


def problem_setup(name):
    """N va ref_point phu hop so muc tieu."""
    if name in TRI_OBJECTIVE:
        return 150, (2.0, 2.0, 2.0)
    return 100, (2.0, 2.0)


def make_segments(D, n_seg=2):
    """n_seg=2: [vi tri, khoang cach].  n_seg=1: khong phan doan."""
    if n_seg == 1:
        return [list(range(D))]
    return [[0], list(range(1, D))]


def new_agent(segments, seed, eps_fixed=None):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return DQNAgent(state_dim=len(segments) + 4,
                    n_segments=len(segments),
                    eps_fixed=eps_fixed)


def one_run(name, agent, segments, seed, training, n_changes):
    """Goi run_sa_drl voi cau hinh chuan cho mot bai."""
    N, ref_point = problem_setup(name)
    return run_sa_drl(
        ALL_PROBLEMS[name],
        n_t=CFG["n_t"], tau_t=CFG["tau_t"], N=N, D=CFG["D"],
        n_changes=n_changes, warm_up=CFG["warm_up"],
        agent=agent, segments=segments, seed=seed,
        training=training, ref_point=ref_point, n_elite=CFG["n_elite"],
    )


def train_on(problem_names, n_episodes, segments, seed=30, log_every=20):
    """Huan luyen mot agent tren tap bai cho truoc."""
    agent = new_agent(segments, seed)
    log = []
    for ep in range(n_episodes):
        name = problem_names[ep % len(problem_names)]
        migd, _ = one_run(name, agent, segments, seed + ep,
                          training=True, n_changes=TRAIN_CHANGES)
        log.append({"ep": ep, "problem": name, "migd": migd, "eps": agent.eps})
        if log_every and ep % log_every == 0:
            recent = np.mean([r["migd"] for r in log[-log_every:]])
            print(f"  Ep {ep:4d} [{name:5s}] MIGD={migd:.5f} "
                  f"avg={recent:.5f} eps={agent.eps:.3f} "
                  f"buf={len(agent.replay_buffer)}")
    return agent, log


def eval_frozen(agent, problem_names, segments, n_runs, n_changes,
                verbose=True):
    """Danh gia agent da dong bang tren cac bai cho truoc."""
    results = {}
    for name in problem_names:
        migds, t0 = [], time.time()
        for run in range(n_runs):
            migd, _ = one_run(name, agent, segments, EVAL_SEED_BASE + run,
                              training=False, n_changes=n_changes)
            migds.append(migd)
        results[name] = (float(np.mean(migds)), float(np.std(migds)), migds)
        if verbose:
            print(f"  {name:5s} MIGD = {np.mean(migds):.6f} "
                  f"+/- {np.std(migds):.6f}   ({time.time()-t0:.0f}s)")
    return results


# ------------------------------------------------------------------ modes
def mode_debug(names, n_episodes, n_runs, n_changes):
    print("=== DEBUG — train va test cung bai, CO overfit ===")
    print("    Chi de kiem tra agent co hoc khong. Khong dua vao luan van.\n")
    segments = make_segments(CFG["D"])
    agent, log = train_on(names, n_episodes, segments)

    k = min(20, n_episodes // 2)
    first = np.mean([r["migd"] for r in log[:k]])
    last = np.mean([r["migd"] for r in log[-k:]])
    print(f"\n  MIGD {k} ep dau  = {first:.6f}")
    print(f"  MIGD {k} ep cuoi = {last:.6f}")
    print("  -> Agent CO hoc" if last < first
          else "  -> Agent KHONG hoc, kiem tra reward va learn()")

    agent.eps = 0.0
    print()
    return eval_frozen(agent, names, segments, n_runs, n_changes)


def mode_online(names, n_runs, n_changes):
    print("=== ONLINE — khong tien huan luyen, so cong bang ===\n")
    segments = make_segments(CFG["D"])
    results = {}
    for name in names:
        migds, t0 = [], time.time()
        for run in range(n_runs):
            seed = EVAL_SEED_BASE + run
            # eps co dinh 0.3: online chi co 1 episode, khong co "qua
            # thoi gian" de decay -> giu 30% tham do trong run.
            agent = new_agent(segments, seed, eps_fixed=0.3)
            migd, _ = one_run(name, agent, segments, seed,
                              training=True, n_changes=n_changes)
            migds.append(migd)
        results[name] = (float(np.mean(migds)), float(np.std(migds)), migds)
        print(f"  {name:5s} MIGD = {np.mean(migds):.6f} "
              f"+/- {np.std(migds):.6f}   ({time.time()-t0:.0f}s)")
    return results


def mode_loo(names, n_episodes, n_runs, n_changes):
    print("=== LEAVE-ONE-OUT — kha nang tong quat hoa ===\n")
    segments = make_segments(CFG["D"])
    results = {}
    for held_out in names:
        train_set = [n for n in ALL_PROBLEMS if n != held_out]
        print(f"[{held_out}] train tren {len(train_set)} bai con lai")
        agent, _ = train_on(train_set, n_episodes, segments,
                            log_every=max(1, n_episodes // 4))
        agent.eps = 0.0                            # dong bang
        res = eval_frozen(agent, [held_out], segments, n_runs, n_changes,
                          verbose=False)
        results[held_out] = res[held_out]
        m, s, _ = res[held_out]
        print(f"  -> test {held_out}: MIGD = {m:.6f} +/- {s:.6f}\n")
    return results


def mode_ablation(names, n_episodes, n_runs, n_changes):
    print("=== ABLATION — phan doan co dong gop khong ===\n")
    results = {}
    for label, n_seg in [("S=1 khong phan doan", 1), ("S=2 co phan doan", 2)]:
        print(label)
        segments = make_segments(CFG["D"], n_seg)
        agent, _ = train_on(names, n_episodes, segments,
                            log_every=max(1, n_episodes // 3))
        agent.eps = 0.0
        results[label] = eval_frozen(agent, names, segments,
                                     n_runs, n_changes)
        print()
    return results


def mode_baseline(names, n_runs, n_changes):
    print("=== BASELINE — NSGA-II thuan, khong phan ung ===\n")
    segments = make_segments(CFG["D"])
    results = {}
    for name in names:
        migds, t0 = [], time.time()
        for run in range(n_runs):
            seed = EVAL_SEED_BASE + run
            # baseline khong can torch/random seed cho mang,
            # nhung run_sa_drl van seed noi bo theo `seed`
            agent = NSGA2Baseline(n_segments=len(segments))
            migd, _ = one_run(name, agent, segments, seed,
                              training=False, n_changes=n_changes)
            migds.append(migd)
        results[name] = (float(np.mean(migds)),
                         float(np.std(migds)), migds)
        print(f"  {name:5s} MIGD = {np.mean(migds):.6f} "
              f"+/- {np.std(migds):.6f}   ({time.time()-t0:.0f}s)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="debug",
                    choices=["debug", "online", "loo", "ablation",
                             "baseline"])
    ap.add_argument("--problems", default="DF1",
                    help="vi du: DF1  |  DF1,DF2,DF3  |  all")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--runs", type=int, default=N_EVAL_RUNS)
    ap.add_argument("--changes", type=int, default=EVAL_CHANGES)
    ap.add_argument("--out", default=None, help="file json luu ket qua")
    args = ap.parse_args()

    names = (list(ALL_PROBLEMS) if args.problems == "all"
             else [s.strip() for s in args.problems.split(",")])
    for n in names:
        if n not in ALL_PROBLEMS:
            raise SystemExit(f"Khong biet bai: {n}")

    t0 = time.time()
    if args.mode == "debug":
        res = mode_debug(names, args.episodes, args.runs, args.changes)
    elif args.mode == "online":
        res = mode_online(names, args.runs, args.changes)
    elif args.mode == "loo":
        res = mode_loo(names, args.episodes, args.runs, args.changes)
    elif args.mode == "baseline":
        res = mode_baseline(names, args.runs, args.changes)
    else:
        res = mode_ablation(names, args.episodes, args.runs, args.changes)

    print(f"\nTong thoi gian: {time.time()-t0:.0f}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Da luu: {args.out}")


if __name__ == "__main__":
    main()
