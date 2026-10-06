"""Experiment runner for CEC2018 dynamic benchmark.

CEC2018 protocol:
  - n_var=10, pop_size=100, nt=10, taut in {10,30}
  - n_changes=30, T0=50
  - Total generations = T0 + n_changes * taut
  - FE_reference = pop_size * total_gen (baseline; D-NSGA-II uses more for detection)
  - IGD measured at end of each dynamic period (30 measurements)
  - MIGD = mean of all IGD values

IGD measurement timing:
  Change k (k=1..30) starts at tau = T0 + 1 + (k-1)*taut.
  We measure at the LAST gen before the next change, i.e. at
  tau = T0 + k*taut (which has the same t as the period's interior).
  For the last period (k=30), we measure at total_gen - 1 = T0 + 30*taut - 1.
  In practice we record when t_now != t_prev (catching the transition) and
  once after the loop ends.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from .problems.base import DynamicBenchmarkProblem, T0
from .metrics.igd import igd, migd
from .metrics.ledger import EvaluationLedger
from .baselines.nsga2 import NSGA2Baseline
from .baselines.dnsga2 import DNSGA2Baseline


def run_baseline(
    problem: DynamicBenchmarkProblem,
    algorithm: NSGA2Baseline,
    n_changes: int = 30,
    n_pf_points: int = 1000,
) -> Dict:
    """Run a baseline algorithm on a dynamic problem.

    Returns dict with 'igd_values', 'migd', 'fe_count', 'n_measurements'.
    """
    pop_size = algorithm.pop_size
    taut = problem.taut
    total_gen = T0 + n_changes * taut

    ledger = EvaluationLedger(problem, budget=None)
    problem.reset()

    X = algorithm.initialize(problem.xl, problem.xu, problem.n_var)
    F = ledger.evaluate(X, t=problem.time())
    algorithm.set_fitness(F)

    igd_values: List[float] = []
    is_dnsga2 = isinstance(algorithm, DNSGA2Baseline)
    t_prev = problem.time()

    for gen in range(1, total_gen):
        problem.set_tau(gen)
        t_now = problem.time()

        if t_now != t_prev and t_prev > 0.0:
            pf = problem.true_pf(t=t_prev, n_points=n_pf_points)
            igd_values.append(igd(algorithm.F, pf))

        if is_dnsga2:
            changed = algorithm.detect_change(
                lambda x: ledger.evaluate(x, t=t_now)
            )
            if changed:
                algorithm.respond_to_change()
                F_new = ledger.evaluate(algorithm.X, t=t_now)
                algorithm.set_fitness(F_new)

        offspring_X = algorithm.step()
        offspring_F = ledger.evaluate(offspring_X, t=t_now)
        algorithm.select(offspring_X, offspring_F)

        t_prev = t_now

    if t_prev > 0.0:
        pf = problem.true_pf(t=t_prev, n_points=n_pf_points)
        igd_values.append(igd(algorithm.F, pf))

    return {
        "igd_values": igd_values,
        "migd": migd(igd_values),
        "fe_count": ledger.count,
        "n_measurements": len(igd_values),
    }


def run_drl(
    problem: DynamicBenchmarkProblem,
    base_algorithm: NSGA2Baseline,
    controller,
    n_changes: int = 30,
    n_pf_points: int = 1000,
    seed: int = 42,
) -> Dict:
    """Run a DRL-controlled algorithm on a dynamic problem.

    Uses NSGA-II as the base optimizer. On environment change, the DRL
    controller selects per-region response operators instead of D-NSGA-II's
    fixed partial reinitialization.

    Returns dict with 'igd_values', 'migd', 'fe_count', 'n_measurements'.
    """
    pop_size = base_algorithm.pop_size
    taut = problem.taut
    total_gen = T0 + n_changes * taut
    rng = np.random.default_rng(seed)

    ledger = EvaluationLedger(problem, budget=None)
    problem.reset()

    X = base_algorithm.initialize(problem.xl, problem.xu, problem.n_var)
    F = ledger.evaluate(X, t=problem.time())
    base_algorithm.set_fitness(F)
    F_prev = F.copy()

    igd_values: List[float] = []
    t_prev = problem.time()

    for gen in range(1, total_gen):
        problem.set_tau(gen)
        t_now = problem.time()

        if t_now != t_prev and t_prev > 0.0:
            pf = problem.true_pf(t=t_prev, n_points=n_pf_points)
            igd_values.append(igd(base_algorithm.F, pf))

        if t_now != t_prev and t_prev >= 0.0:
            F_plus = ledger.evaluate(X, t=t_now)
            ledger_eval_fn = lambda x: ledger.evaluate(x, t=t_now)
            X_new, F_new, fe_used = controller.respond(
                X, F_prev, F_plus, problem.xl, problem.xu,
                ledger_eval_fn, rng)[:3]
            X = X_new
            F = F_new
            base_algorithm.X = X.copy()
            base_algorithm.set_fitness(F)
            F_prev = F.copy()
        else:
            offspring_X = base_algorithm.step()
            offspring_F = ledger.evaluate(offspring_X, t=t_now)
            base_algorithm.select(offspring_X, offspring_F)
            X = base_algorithm.X.copy()
            F = base_algorithm.F.copy()
            F_prev = F.copy()

        t_prev = t_now

    if t_prev > 0.0:
        pf = problem.true_pf(t=t_prev, n_points=n_pf_points)
        igd_values.append(igd(base_algorithm.F, pf))

    return {
        "igd_values": igd_values,
        "migd": migd(igd_values),
        "fe_count": ledger.count,
        "n_measurements": len(igd_values),
    }


def run_single_experiment(
    problem_name: str,
    algorithm_name: str,
    seed: int,
    n_var: int = 10,
    pop_size: int = 100,
    nt: int = 10,
    taut: int = 10,
    n_changes: int = 30,
    n_pf_points: int = 1000,
    checkpoint_path: str = None,
) -> Dict:
    """Run one experiment: one problem, one algorithm, one seed."""
    from .problems.registry import get_problem

    np.random.seed(seed)
    problem = get_problem(problem_name, n_var=n_var, nt=nt, taut=taut)

    if algorithm_name == "NSGA-II":
        alg = NSGA2Baseline(pop_size=pop_size, seed=seed)
        result = run_baseline(problem, alg, n_changes=n_changes, n_pf_points=n_pf_points)
    elif algorithm_name == "D-NSGA-II":
        alg = DNSGA2Baseline(pop_size=pop_size, seed=seed)
        result = run_baseline(problem, alg, n_changes=n_changes, n_pf_points=n_pf_points)
    elif algorithm_name.startswith("B-C"):
        from .controllers.drl_adapter import DRLController
        controller_type = algorithm_name[2:]
        n_obj = problem.n_obj
        ctrl = DRLController(controller_type, checkpoint_path, n_obj)
        base_alg = NSGA2Baseline(pop_size=pop_size, seed=seed)
        result = run_drl(problem, base_alg, ctrl, n_changes=n_changes, n_pf_points=n_pf_points, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    result["problem"] = problem_name
    result["algorithm"] = algorithm_name
    result["seed"] = seed
    result["taut"] = taut
    return result
