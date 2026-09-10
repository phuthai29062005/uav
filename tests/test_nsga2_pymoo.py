import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from pymoo.problems.dynamic.df import DF1
from pymoo.indicators.igd import IGD

from nsga2_pymoo import nsga2_one_generation


def test_one_generation_returns_correct_shape():
    np.random.seed(0)
    problem = DF1(time=0.5, n_var=10)
    pop = np.random.rand(100, 10)
    F = problem.evaluate(pop)

    new_pop, new_F = nsga2_one_generation(pop, F, problem, 100)

    assert new_pop.shape == (100, 10)
    assert new_F.shape == (100, 2)
    assert np.all(new_pop >= 0) and np.all(new_pop <= 1)


def test_igd_improves_over_generations():
    np.random.seed(42)
    problem = DF1(time=0.5, n_var=10)
    PF = problem.pareto_front()

    pop = np.random.rand(100, 10)
    F = problem.evaluate(pop)
    igd_start = IGD(PF)(F)

    for _ in range(200):
        pop, F = nsga2_one_generation(pop, F, problem, 100)

    igd_end = IGD(PF)(F)
    assert igd_end < igd_start, f"IGD did not improve: {igd_start:.6f} -> {igd_end:.6f}"
    assert igd_end < 0.01, f"IGD too high after 200 gens: {igd_end:.6f}"
