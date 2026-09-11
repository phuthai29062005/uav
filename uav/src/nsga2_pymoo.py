import numpy as np
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

_sbx = SBX(prob=0.9, eta=20)
_pm = PM(eta=20)
_survival = RankAndCrowding()

_rng = np.random.RandomState(0)


def seed_nsga2(seed):
    global _rng
    _rng = np.random.RandomState(seed)


def compute_rank_and_crowding(F):
    """
    F: (N, M)
    return: rank (N,) int (front 0 = nondominated), crowding (N,) float.

    Crowding canonical cuc bo (khong obj_scale ngoai, khong ref_point):
    moi front, bien = inf; interior = tong tren cac objective cua
    (F[i+1,m]-F[i-1,m])/(Fmax_m-Fmin_m), objective co Fmax==Fmin dong 0.
    """
    F = np.asarray(F, dtype=float)
    N = len(F)
    rank = np.zeros(N, dtype=int)
    crowding = np.zeros(N, dtype=float)
    fronts = NonDominatedSorting().do(F)
    for r, front in enumerate(fronts):
        front = np.asarray(front)
        rank[front] = r
        if len(front) <= 2:
            crowding[front] = np.inf
            continue
        Ff = F[front]
        cd = np.zeros(len(front))
        for m in range(F.shape[1]):
            order = np.argsort(Ff[:, m], kind="stable")
            fmin, fmax = Ff[order[0], m], Ff[order[-1], m]
            cd[order[0]] = cd[order[-1]] = np.inf
            if fmax == fmin:
                continue
            span = fmax - fmin
            for k in range(1, len(front) - 1):
                cd[order[k]] += (Ff[order[k + 1], m]
                                 - Ff[order[k - 1], m]) / span
        crowding[front] = cd
    return rank, crowding


def binary_tournament(rank, crowding, rng):
    """
    Mot winner tu hai candidate random KHAC nhau. Rule canonical NSGA-II:
    rank thap hon -> thang; cung rank -> crowding lon hon -> thang;
    tie chinh xac -> rng tie-break (khong default index -> khong bias).
    """
    i, j = rng.choice(len(rank), size=2, replace=False)
    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j
    if crowding[i] > crowding[j]:
        return i
    if crowding[j] > crowding[i]:
        return j
    return i if rng.random() < 0.5 else j


def nsga2_one_generation(pop_X, F_pop, problem, N):
    pop = Population.new("X", pop_X)
    pop.set("F", F_pop)

    # Binary tournament theo rank+crowding: MOI parent slot mot tournament
    # rieng -> N winner cho N offspring (N/2 cap, moi cap 2 con).
    assert N % 2 == 0, f"N phai chan cho ghep cap, nhan N={N}"
    rank, crowding = compute_rank_and_crowding(F_pop)
    mating_idx = np.array([binary_tournament(rank, crowding, _rng)
                           for _ in range(N)])
    parents_idx = mating_idx.reshape(-1, 2)

    rs = np.random.default_rng(_rng.randint(0, 2**31))
    off = _sbx.do(problem, pop, parents_idx, random_state=rs)
    off = _pm.do(problem, off, random_state=rs)

    off_X = np.clip(off.get("X"), problem.xl, problem.xu)
    off_F = problem.evaluate(off_X)

    off_eval = Population.new("X", off_X)
    off_eval.set("F", off_F)

    merged = Population.merge(pop, off_eval)
    rs_surv = np.random.default_rng(_rng.randint(0, 2**31))
    survivors = _survival.do(problem, merged, n_survive=N,
                             random_state=rs_surv)

    return survivors.get("X"), survivors.get("F")
