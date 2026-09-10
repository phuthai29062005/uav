import numpy as np
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

_sbx = SBX(prob=0.9, eta=20)
_pm = PM(eta=20)
_survival = RankAndCrowding()

_rng = np.random.RandomState(0)


def seed_nsga2(seed):
    global _rng
    _rng = np.random.RandomState(seed)


def nsga2_one_generation(pop_X, F_pop, problem, N):
    pop = Population.new("X", pop_X)
    pop.set("F", F_pop)

    parents_idx = np.column_stack([
        _rng.permutation(N)[:N // 2],
        _rng.permutation(N)[:N // 2],
    ])

    rs = np.random.default_rng(_rng.randint(0, 2**31))
    off = _sbx.do(problem, pop, parents_idx, random_state=rs)
    off = _pm.do(problem, off, random_state=rs)

    off_X = np.clip(off.get("X"), 0.0, 1.0)
    off_F = problem.evaluate(off_X)

    off_eval = Population.new("X", off_X)
    off_eval.set("F", off_F)

    merged = Population.merge(pop, off_eval)
    rs_surv = np.random.default_rng(_rng.randint(0, 2**31))
    survivors = _survival.do(problem, merged, n_survive=N,
                             random_state=rs_surv)

    return survivors.get("X"), survivors.get("F")
