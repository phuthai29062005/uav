"""Main experiment runner for SA-DRL-DMOEA C4 benchmark validation.

Runs all algorithms on dynamic benchmark problems and collects MIGD/MHV results.
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from benchmark_c4_v1.problems import get_problem, ALL_PROBLEMS, SMOKE_PROBLEMS
from benchmark_c4_v1.problems.registry import PROBLEM_REGISTRY
from benchmark_c4_v1.metrics.igd import igd, migd
from benchmark_c4_v1.metrics.hv import hypervolume, mhv
from benchmark_c4_v1.controllers.ledger import EvaluationLedger
from benchmark_c4_v1.controllers.segmentation import segment_population, BLIND_FIXED_2SEG
from benchmark_c4_v1.controllers.state import BenchmarkStateBuilder
from benchmark_c4_v1.controllers.arms import (
    BenchmarkC1Arm, BenchmarkC3Arm, BenchmarkC4Arm,
    N_ACTIONS, R_MAX, STATE_DIM,
)
from benchmark_c4_v1.baselines.nsga2 import NSGA2Baseline
from benchmark_c4_v1.baselines.dnsga2 import DNSGA2Baseline

from prdrl.rl.network import ACTION_ORDER

POP_SIZE = 100
N_CHANGES = 30
NT = 10
N_PF_POINTS = 1000
SCHEME = BLIND_FIXED_2SEG
B_OFFSPRING = 10
SIGMA_LOCAL = 0.02
SIGMA_PREDICT = 0.005
SIGMA_DIVERSIFY = 0.15
LAMBDA_PRED = 1.0


def compute_max_fe(n_changes: int, taut: int) -> int:
    return POP_SIZE * (n_changes * taut + 50)


def run_drl_algorithm(problem, arm, arm_name: str, taut: int, run_seed: int,
                      training: bool = False) -> Dict:
    """Run a DRL-based algorithm (B-C1, B-C3, or B-C4) on one problem instance."""
    rng = np.random.default_rng(run_seed)
    max_fe = compute_max_fe(N_CHANGES, taut)
    ledger = EvaluationLedger(max_fe)
    state_builder = BenchmarkStateBuilder(n_regions=2)

    X = rng.uniform(problem.xl, problem.xu, size=(POP_SIZE, problem.n_var))
    if not ledger.charge(POP_SIZE):
        return {"migd": float("inf"), "fe_used": ledger.count}
    F = problem.evaluate(X, problem.time(0))

    igd_per_change = []
    hv_per_change = []
    prev_centroids = [None, None]

    total_tau = N_CHANGES * taut + 50
    for tau in range(total_tau):
        problem.set_tau(tau)
        t = problem.time(tau)

        if tau > 0 and (tau % taut == 0):
            if not ledger.charge(POP_SIZE):
                break
            F = problem.evaluate(X, t)

            true_pf = problem.true_pf(t, N_PF_POINTS)
            regions = segment_population(X, F, problem.n_var, SCHEME)
            states = state_builder.build_states(regions, X, F, true_pf, POP_SIZE)

            active_mask = np.zeros(R_MAX, dtype=np.float32)
            for r_id in states:
                active_mask[r_id] = 1.0

            if arm_name == "B-C1":
                if training:
                    actions_dict = arm.select_actions(states, rng)
                else:
                    actions_dict = arm.select_actions_greedy(states)
                action_vector = np.zeros(R_MAX, dtype=int)
                for r_id, a in actions_dict.items():
                    action_vector[r_id] = a
            else:
                if training:
                    action_vector = arm.select_actions(states, active_mask, rng)
                else:
                    action_vector = arm.select_actions_greedy(states, active_mask)

            for r_id, idx_arr in regions.items():
                if len(idx_arr) == 0:
                    continue
                a = int(action_vector[r_id]) if r_id < R_MAX else 0
                action_name = ACTION_ORDER[a]

                if action_name == "KEEP":
                    pass
                elif action_name in ("LOCAL", "PREDICT", "DIVERSIFY"):
                    n_offspring = B_OFFSPRING
                    if action_name == "LOCAL":
                        sigma = SIGMA_LOCAL
                    elif action_name == "PREDICT":
                        sigma = SIGMA_PREDICT
                    else:
                        sigma = SIGMA_DIVERSIFY

                    if not ledger.charge(n_offspring):
                        break

                    offspring = _generate_offspring(X[idx_arr], problem.xl, problem.xu,
                                                    n_offspring, sigma, rng, action_name,
                                                    prev_centroids, r_id)
                    F_off = problem.evaluate(offspring, t)
                    X, F = _survival_selection(X, F, offspring, F_off, POP_SIZE)

            igd_val = igd(F, true_pf)
            igd_per_change.append(igd_val)

            ref_point = np.max(true_pf, axis=0) * 1.1 + 0.1
            hv_val = hypervolume(F, ref_point)
            hv_per_change.append(hv_val)

            centroids = []
            for r_id in range(2):
                r_idx = regions.get(r_id, np.array([], dtype=int))
                if len(r_idx) > 0:
                    centroids.append(np.mean(F[r_idx], axis=0))
                else:
                    centroids.append(None)
            prev_centroids = centroids

            if training:
                next_F = F.copy()
                next_regions = segment_population(X, next_F, problem.n_var, SCHEME)
                next_true_pf = true_pf
                next_states = state_builder.build_states(next_regions, X, next_F,
                                                         next_true_pf, POP_SIZE)
                next_active_mask = np.zeros(R_MAX, dtype=np.float32)
                for r_id in next_states:
                    next_active_mask[r_id] = 1.0

                r_joint = -igd_val

                if arm_name == "B-C1":
                    joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                    for r_id, s in states.items():
                        joint_states_arr[r_id] = s
                    next_joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                    for r_id, s in next_states.items():
                        next_joint_states_arr[r_id] = s

                    n_active = max(1, int(active_mask.sum()))
                    central_state = np.zeros(55, dtype=np.float32)
                    for r_id in range(min(R_MAX, 2)):
                        if r_id in states:
                            central_state[r_id * 5:(r_id + 1) * 5] = states[r_id][:5]
                    central_state[50:55] = [float(active_mask.sum()) / R_MAX,
                                            float(np.mean(sev_values := [states[r][0] for r in states])),
                                            float(np.max(sev_values)),
                                            float(np.mean(cl_values := [states[r][1] for r in states])),
                                            float(np.max(cl_values))]

                    transition = {
                        "states": joint_states_arr,
                        "actions": action_vector,
                        "rewards": np.zeros(R_MAX, dtype=np.float32),
                        "next_states": next_joint_states_arr,
                        "active_mask": active_mask,
                        "done": float(tau >= total_tau - 1),
                        "central_state": central_state,
                    }
                    for r_id in states:
                        transition["rewards"][r_id] = r_joint / n_active
                    arm.store_transition(transition)
                    arm.update()
                else:
                    joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                    for r_id, s in states.items():
                        joint_states_arr[r_id] = s
                    next_joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                    for r_id, s in next_states.items():
                        next_joint_states_arr[r_id] = s

                    transition = {
                        "joint_states": joint_states_arr,
                        "action_vector": action_vector,
                        "active_mask": active_mask,
                        "r_joint": r_joint,
                        "next_joint_states": next_joint_states_arr,
                        "next_active_mask": next_active_mask,
                        "done": float(tau >= total_tau - 1),
                    }
                    arm.store_transition(transition)
                    arm.update()

        if ledger.exhausted:
            break

    return {
        "migd": migd(igd_per_change) if igd_per_change else float("inf"),
        "mhv": mhv(hv_per_change) if hv_per_change else 0.0,
        "igd_per_change": igd_per_change,
        "hv_per_change": hv_per_change,
        "fe_used": ledger.count,
        "max_fe": max_fe,
        "n_changes_observed": len(igd_per_change),
    }


def _generate_offspring(X_region, xl, xu, n_offspring, sigma, rng,
                         action_name, prev_centroids, r_id):
    if action_name == "PREDICT" and prev_centroids[r_id] is not None:
        centroid_now = np.mean(X_region, axis=0) if len(X_region) > 0 else None
        if centroid_now is not None and prev_centroids[r_id] is not None:
            try:
                if len(prev_centroids[r_id]) == len(centroid_now):
                    delta_hat = LAMBDA_PRED * (centroid_now[:len(xl)] - np.mean(X_region, axis=0))
                    delta_hat = np.zeros(len(xl))
                else:
                    delta_hat = np.zeros(len(xl))
            except (TypeError, ValueError):
                delta_hat = np.zeros(len(xl))
        else:
            delta_hat = np.zeros(len(xl))
    else:
        delta_hat = None

    offspring = []
    for _ in range(n_offspring):
        idx = rng.integers(len(X_region))
        parent = X_region[idx]
        z = (parent - xl) / (xu - xl + 1e-30)
        xi = rng.normal(0, 1, size=len(parent))
        if delta_hat is not None:
            z_child = z + delta_hat + sigma * xi
        else:
            z_child = z + sigma * xi
        z_child = np.clip(z_child, 0, 1)
        child = xl + z_child * (xu - xl)
        offspring.append(child)
    return np.array(offspring)


def _survival_selection(X, F, offspring_X, offspring_F, pop_size):
    from benchmark_c4_v1.baselines.nsga2 import _fast_non_dominated_sort, _crowding_distance
    combined_X = np.vstack([X, offspring_X])
    combined_F = np.vstack([F, offspring_F])
    fronts, rank = _fast_non_dominated_sort(combined_F)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            remaining = pop_size - len(selected)
            cd = _crowding_distance(combined_F, front)
            sorted_by_cd = np.argsort(-cd)
            for idx in sorted_by_cd[:remaining]:
                selected.append(front[idx])
            break
    return combined_X[selected], combined_F[selected]


def run_nsga2(problem, taut: int, run_seed: int) -> Dict:
    """Run standard NSGA-II on one problem instance."""
    max_fe = compute_max_fe(N_CHANGES, taut)
    ledger = EvaluationLedger(max_fe)
    algo = NSGA2Baseline(pop_size=POP_SIZE, seed=run_seed)
    X = algo.initialize(problem.xl, problem.xu, problem.n_var)
    if not ledger.charge(POP_SIZE):
        return {"migd": float("inf"), "fe_used": ledger.count}
    F = problem.evaluate(X, problem.time(0))
    algo.set_fitness(F)

    igd_per_change = []
    hv_per_change = []
    total_tau = N_CHANGES * taut + 50

    for tau in range(total_tau):
        problem.set_tau(tau)
        t = problem.time(tau)

        if tau > 0 and (tau % taut == 0):
            if not ledger.charge(POP_SIZE):
                break
            F = problem.evaluate(algo.X, t)
            algo.set_fitness(F)

        offspring_X = algo.step()
        if not ledger.charge(len(offspring_X)):
            break
        offspring_F = problem.evaluate(offspring_X, t)
        algo.select(offspring_X, offspring_F)

        if tau > 0 and (tau % taut == 0):
            true_pf = problem.true_pf(t, N_PF_POINTS)
            igd_val = igd(algo.F, true_pf)
            igd_per_change.append(igd_val)
            ref_point = np.max(true_pf, axis=0) * 1.1 + 0.1
            hv_val = hypervolume(algo.F, ref_point)
            hv_per_change.append(hv_val)

        if ledger.exhausted:
            break

    return {
        "migd": migd(igd_per_change) if igd_per_change else float("inf"),
        "mhv": mhv(hv_per_change) if hv_per_change else 0.0,
        "igd_per_change": igd_per_change,
        "hv_per_change": hv_per_change,
        "fe_used": ledger.count,
        "max_fe": max_fe,
        "n_changes_observed": len(igd_per_change),
    }


def run_dnsga2(problem, taut: int, run_seed: int) -> Dict:
    """Run D-NSGA-II on one problem instance."""
    max_fe = compute_max_fe(N_CHANGES, taut)
    ledger = EvaluationLedger(max_fe)
    algo = DNSGA2Baseline(pop_size=POP_SIZE, seed=run_seed)
    X = algo.initialize(problem.xl, problem.xu, problem.n_var)
    if not ledger.charge(POP_SIZE):
        return {"migd": float("inf"), "fe_used": ledger.count}
    F = problem.evaluate(X, problem.time(0))
    algo.set_fitness(F)

    igd_per_change = []
    hv_per_change = []
    total_tau = N_CHANGES * taut + 50

    for tau in range(total_tau):
        problem.set_tau(tau)
        t = problem.time(tau)

        if tau > 0 and (tau % taut == 0):
            if not ledger.charge(POP_SIZE + algo.n_detectors):
                break
            changed = algo.detect_change(problem)
            if changed:
                algo.respond_to_change()
            F = problem.evaluate(algo.X, t)
            algo.set_fitness(F)

        offspring_X = algo.step()
        if not ledger.charge(len(offspring_X)):
            break
        offspring_F = problem.evaluate(offspring_X, t)
        algo.select(offspring_X, offspring_F)

        if tau > 0 and (tau % taut == 0):
            true_pf = problem.true_pf(t, N_PF_POINTS)
            igd_val = igd(algo.F, true_pf)
            igd_per_change.append(igd_val)
            ref_point = np.max(true_pf, axis=0) * 1.1 + 0.1
            hv_val = hypervolume(algo.F, ref_point)
            hv_per_change.append(hv_val)

        if ledger.exhausted:
            break

    return {
        "migd": migd(igd_per_change) if igd_per_change else float("inf"),
        "mhv": mhv(hv_per_change) if hv_per_change else 0.0,
        "igd_per_change": igd_per_change,
        "hv_per_change": hv_per_change,
        "fe_used": ledger.count,
        "max_fe": max_fe,
        "n_changes_observed": len(igd_per_change),
    }


def train_drl_arm(arm, arm_name: str, problems: List[str], taut: int,
                   n_episodes: int, policy_seed: int) -> Dict:
    """Train a DRL arm across multiple problem instances.

    Uses a simplified fast training loop: only 5 change events per episode
    to keep training time feasible while still exposing the controller to
    dynamic optimization decisions.
    """
    rng = np.random.default_rng(policy_seed * 100)
    losses = []
    n_train_changes = 5
    for ep in range(n_episodes):
        prob_name = problems[ep % len(problems)]
        problem = get_problem(prob_name, n_var=10, nt=NT, taut=taut)
        ep_seed = int(rng.integers(0, 2**31))
        ep_rng = np.random.default_rng(ep_seed)
        X = ep_rng.uniform(problem.xl, problem.xu, size=(POP_SIZE, problem.n_var))
        state_builder = BenchmarkStateBuilder(n_regions=2)
        ep_igd = []
        for ch in range(n_train_changes):
            tau = (ch + 1) * taut
            problem.set_tau(tau)
            t = problem.time(tau)
            F = problem.evaluate(X, t)
            true_pf = problem.true_pf(t, 500)
            regions = segment_population(X, F, problem.n_var, SCHEME)
            states = state_builder.build_states(regions, X, F, true_pf, POP_SIZE)
            active_mask = np.zeros(R_MAX, dtype=np.float32)
            for r_id in states:
                active_mask[r_id] = 1.0
            if arm_name == "B-C1":
                actions_dict = arm.select_actions(states, ep_rng)
                action_vector = np.zeros(R_MAX, dtype=int)
                for r_id, a in actions_dict.items():
                    action_vector[r_id] = a
            else:
                action_vector = arm.select_actions(states, active_mask, ep_rng)
            for r_id, idx_arr in regions.items():
                if len(idx_arr) == 0:
                    continue
                a = int(action_vector[r_id]) if r_id < R_MAX else 0
                action_name = ACTION_ORDER[a]
                if action_name in ("LOCAL", "PREDICT", "DIVERSIFY"):
                    sigma = {"LOCAL": SIGMA_LOCAL, "PREDICT": SIGMA_PREDICT,
                             "DIVERSIFY": SIGMA_DIVERSIFY}[action_name]
                    offspring = _generate_offspring(X[idx_arr], problem.xl, problem.xu,
                                                    B_OFFSPRING, sigma, ep_rng,
                                                    action_name, [None, None], r_id)
                    F_off = problem.evaluate(offspring, t)
                    X, F = _survival_selection(X, F, offspring, F_off, POP_SIZE)
            igd_val = igd(F, true_pf)
            ep_igd.append(igd_val)
            next_F = F.copy()
            next_regions = segment_population(X, next_F, problem.n_var, SCHEME)
            next_states = state_builder.build_states(next_regions, X, next_F, true_pf, POP_SIZE)
            next_active_mask = np.zeros(R_MAX, dtype=np.float32)
            for r_id in next_states:
                next_active_mask[r_id] = 1.0
            r_joint = -igd_val
            if arm_name == "B-C1":
                joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                for r_id, s in states.items():
                    joint_states_arr[r_id] = s
                next_joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                for r_id, s in next_states.items():
                    next_joint_states_arr[r_id] = s
                n_active = max(1, int(active_mask.sum()))
                central_state = np.zeros(55, dtype=np.float32)
                for r_id in range(min(R_MAX, 2)):
                    if r_id in states:
                        central_state[r_id * 5:(r_id + 1) * 5] = states[r_id][:5]
                sev_vals = [states[r][0] for r in states]
                cl_vals = [states[r][1] for r in states]
                central_state[50:55] = [float(active_mask.sum()) / R_MAX,
                                        float(np.mean(sev_vals)), float(np.max(sev_vals)),
                                        float(np.mean(cl_vals)), float(np.max(cl_vals))]
                transition = {
                    "states": joint_states_arr, "actions": action_vector,
                    "rewards": np.zeros(R_MAX, dtype=np.float32),
                    "next_states": next_joint_states_arr, "active_mask": active_mask,
                    "done": float(ch >= n_train_changes - 1), "central_state": central_state,
                }
                for r_id in states:
                    transition["rewards"][r_id] = r_joint / n_active
                arm.store_transition(transition)
                arm.update()
            else:
                joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                for r_id, s in states.items():
                    joint_states_arr[r_id] = s
                next_joint_states_arr = np.zeros((R_MAX, STATE_DIM), dtype=np.float32)
                for r_id, s in next_states.items():
                    next_joint_states_arr[r_id] = s
                transition = {
                    "joint_states": joint_states_arr, "action_vector": action_vector,
                    "active_mask": active_mask, "r_joint": r_joint,
                    "next_joint_states": next_joint_states_arr,
                    "next_active_mask": next_active_mask,
                    "done": float(ch >= n_train_changes - 1),
                }
                arm.store_transition(transition)
                arm.update()
        losses.append(float(np.mean(ep_igd)) if ep_igd else float("inf"))
    return {"n_episodes": n_episodes, "mean_migd": float(np.mean(losses)),
            "final_migd": float(losses[-1]) if losses else float("inf")}


def run_full_experiment(profile: str = "SMOKE", taut_values: List[int] = None,
                        n_runs: int = 20, training_episodes: int = 200,
                        policy_seeds: List[int] = None,
                        output_dir: str = None) -> Dict:
    """Run the complete benchmark experiment."""
    if taut_values is None:
        taut_values = [10, 30]
    if policy_seeds is None:
        policy_seeds = [101, 102, 103, 104, 105]
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    if profile == "SMOKE":
        problems = SMOKE_PROBLEMS
    else:
        problems = ALL_PROBLEMS

    all_results = {}
    algorithms = ["NSGA-II", "D-NSGA-II", "B-C1", "B-C3", "B-C4"]

    print(f"{'='*60}")
    print(f"SA-DRL-DMOEA C4 BENCHMARK VALIDATION")
    print(f"Profile: {profile}")
    print(f"Problems: {len(problems)}")
    print(f"taut values: {taut_values}")
    print(f"Runs per config: {n_runs}")
    print(f"Training episodes: {training_episodes}")
    print(f"Policy seeds: {policy_seeds}")
    print(f"{'='*60}")

    for taut in taut_values:
        taut_key = f"taut_{taut}"
        all_results[taut_key] = {}

        print(f"\n--- Training DRL arms (taut={taut}) ---")
        trained_arms = {}
        for ps in policy_seeds:
            for arm_name, ArmClass in [("B-C1", BenchmarkC1Arm),
                                         ("B-C3", BenchmarkC3Arm),
                                         ("B-C4", BenchmarkC4Arm)]:
                print(f"  Training {arm_name} (seed={ps})...", end=" ", flush=True)
                t0 = time.time()
                arm = ArmClass(ps)
                train_result = train_drl_arm(arm, arm_name, problems, taut,
                                              training_episodes, ps)
                dt = time.time() - t0
                print(f"done ({dt:.1f}s, mean_migd={train_result['mean_migd']:.6f})", flush=True)
                trained_arms[(arm_name, ps)] = arm

        for prob_name in problems:
            print(f"\n  Problem: {prob_name} (taut={taut})")
            all_results[taut_key][prob_name] = {}

            for alg_name in algorithms:
                migd_runs = []
                mhv_runs = []
                t0 = time.time()

                for run_idx in range(n_runs):
                    run_seed = taut * 10000 + run_idx * 100

                    if alg_name == "NSGA-II":
                        problem = get_problem(prob_name, n_var=10, nt=NT, taut=taut)
                        result = run_nsga2(problem, taut, run_seed)
                    elif alg_name == "D-NSGA-II":
                        problem = get_problem(prob_name, n_var=10, nt=NT, taut=taut)
                        result = run_dnsga2(problem, taut, run_seed)
                    else:
                        ps = policy_seeds[run_idx % len(policy_seeds)]
                        arm = trained_arms[(alg_name, ps)]
                        problem = get_problem(prob_name, n_var=10, nt=NT, taut=taut)
                        result = run_drl_algorithm(problem, arm, alg_name, taut,
                                                    run_seed, training=False)

                    migd_runs.append(result["migd"])
                    mhv_runs.append(result.get("mhv", 0.0))

                dt = time.time() - t0
                mean_migd = float(np.mean(migd_runs))
                std_migd = float(np.std(migd_runs))
                mean_mhv = float(np.mean(mhv_runs))
                print(f"    {alg_name:12s}: MIGD={mean_migd:.6f} ± {std_migd:.6f} "
                      f"MHV={mean_mhv:.6f} ({dt:.1f}s)", flush=True)

                all_results[taut_key][prob_name][alg_name] = {
                    "migd_runs": migd_runs,
                    "mhv_runs": mhv_runs,
                    "mean_migd": mean_migd,
                    "std_migd": std_migd,
                    "mean_mhv": mean_mhv,
                    "n_runs": n_runs,
                }

    result_path = os.path.join(output_dir, f"benchmark_{profile.lower()}_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")

    return all_results


def statistical_analysis(results: Dict, output_dir: str) -> Dict:
    """Run statistical analysis on experiment results."""
    from benchmark_c4_v1.metrics.statistics import (
        friedman_ranking, friedman_test, pairwise_comparisons
    )

    analysis = {}
    algorithms = ["NSGA-II", "D-NSGA-II", "B-C1", "B-C3", "B-C4"]

    for taut_key, problems_data in results.items():
        analysis[taut_key] = {}
        migd_by_alg = {a: [] for a in algorithms}

        for prob_name, alg_data in problems_data.items():
            for alg_name in algorithms:
                if alg_name in alg_data:
                    migd_by_alg[alg_name].append(alg_data[alg_name]["mean_migd"])

        present_algs = {a: v for a, v in migd_by_alg.items() if v}
        if len(present_algs) >= 2:
            rankings = friedman_ranking(present_algs)
            chi2, p_val = friedman_test(present_algs)
            pairs = pairwise_comparisons(present_algs)

            analysis[taut_key] = {
                "friedman_rankings": rankings,
                "friedman_chi2": chi2,
                "friedman_p": p_val,
                "pairwise": pairs,
            }

    analysis_path = os.path.join(output_dir, "benchmark_statistical_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Statistical analysis saved to {analysis_path}")
    return analysis


def verify_core_unchanged() -> bool:
    """Verify that no frozen core files were modified."""
    manifest_path = Path(__file__).parent.parent / "ORIGINAL_CORE_HASH_MANIFEST.json"
    if not manifest_path.exists():
        print("WARNING: Hash manifest not found")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    file_map = manifest["ORIGINAL_CORE_HASH_MANIFEST"]
    base = Path(__file__).parent.parent.parent
    changed = []
    for rel_path, expected_sha in file_map.items():
        fpath = base / rel_path
        if not fpath.exists():
            changed.append(f"MISSING: {rel_path}")
            continue
        with open(fpath, "rb") as f:
            current_sha = hashlib.sha256(f.read()).hexdigest()
        if current_sha != expected_sha:
            changed.append(f"CHANGED: {rel_path}")

    if changed:
        print("CORE INTEGRITY FAILURE:")
        for c in changed:
            print(f"  {c}")
        return False
    else:
        print(f"CORE INTEGRITY: PASS ({len(file_map)} files verified)")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SA-DRL-DMOEA C4 Benchmark Validation")
    parser.add_argument("--profile", choices=["SMOKE", "FULL"], default="SMOKE")
    parser.add_argument("--taut", nargs="+", type=int, default=[10, 30])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--training-episodes", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    core_ok = verify_core_unchanged()
    if not core_ok:
        print("ABORTING: Core files have been modified")
        sys.exit(1)

    results = run_full_experiment(
        profile=args.profile,
        taut_values=args.taut,
        n_runs=args.runs,
        training_episodes=args.training_episodes,
        output_dir=args.output_dir,
    )

    output_dir = args.output_dir or str(Path(__file__).parent.parent / "results")
    statistical_analysis(results, output_dir)
    verify_core_unchanged()
