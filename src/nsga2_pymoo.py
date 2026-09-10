import numpy as np
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

_sbx = SBX(prob=0.9, eta=20)
_pm = PM(eta=20)
_survival = RankAndCrowding()


def nsga2_one_generation(pop_X, F_pop, problem, N):
    pop = Population.new("X", pop_X)
    pop.set("F", F_pop)

    parents_idx = np.column_stack([
        np.random.permutation(N)[:N // 2],
        np.random.permutation(N)[:N // 2],
    ])

    off = _sbx.do(problem, pop, parents_idx)
    off = _pm.do(problem, off)

    off_X = np.clip(off.get("X"), 0.0, 1.0)
    off_F = problem.evaluate(off_X)

    off_eval = Population.new("X", off_X)
    off_eval.set("F", off_F)

    merged = Population.merge(pop, off_eval)
    survivors = _survival.do(problem, merged, n_survive=N)

    return survivors.get("X"), survivors.get("F")
