"""T1-T10: canonical NSGA-II binary tournament by rank+crowding."""
import os
import sys

import numpy as np
import pytest
from pymoo.indicators.igd import IGD
from pymoo.problems.dynamic.df import DF1, DF4

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import nsga2_pymoo as ng  # noqa: E402
from nsga2_pymoo import (binary_tournament,  # noqa: E402
                         compute_rank_and_crowding, nsga2_one_generation,
                         seed_nsga2)


def init_pop(p, N, seed):
    return p.xl + np.random.RandomState(seed).rand(N, len(p.xl)) * (p.xu - p.xl)


# ---------------------------------------- T1: RANK DOMINATES CROWDING
def test_t1_rank_dominates_crowding():
    rank = np.array([0, 1])
    crowding = np.array([0.0, 1e9])
    seed_nsga2(0)
    assert binary_tournament(rank, crowding, ng._rng) == 0


# ------------------------------------ T2: CROWDING BREAKS SAME-RANK TIE
def test_t2_crowding_breaks_tie():
    rank = np.array([0, 0])
    crowding = np.array([0.1, 10.0])
    seed_nsga2(0)
    assert binary_tournament(rank, crowding, ng._rng) == 1


# ------------------------------- T3: EXACT TIE USES RNG, REPRODUCIBLE
def test_t3_exact_tie_uses_rng_reproducible():
    rank = np.array([0, 0])
    crowding = np.array([1.0, 1.0])
    seed_nsga2(42)
    seq1 = [binary_tournament(rank, crowding, ng._rng) for _ in range(100)]
    seed_nsga2(42)
    seq2 = [binary_tournament(rank, crowding, ng._rng) for _ in range(100)]
    assert seq1 == seq2
    assert 0 in seq1 and 1 in seq1


# --------------------------- T4: RANK/CROWDING COMPUTATION CORRECT
def test_t4_rank_crowding_correct():
    F = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.],
                  [4., 4.], [5., 5.]])
    rank, crowding = compute_rank_and_crowding(F)
    assert list(rank[:4]) == [0, 0, 0, 0]
    assert rank[4] > 0
    assert rank[5] >= rank[4]
    # boundary cua front 0: F[:,0] min=idx0, max=idx3 -> inf
    assert np.isinf(crowding[0]) and np.isinf(crowding[3])
    # interior finite, >= 0
    assert np.isfinite(crowding[1]) and crowding[1] >= 0
    assert np.isfinite(crowding[2]) and crowding[2] >= 0


# -------------------- T5: MATING NOT RANDOM W.R.T. FITNESS
def test_t5_selection_pressure():
    rank = np.array([0] * 10 + [1] * 10)
    crowding = np.zeros(20)
    seed_nsga2(1)
    wins = [binary_tournament(rank, crowding, ng._rng) for _ in range(5000)]
    p_front0 = np.mean([w < 10 for w in wins])
    print(f"\n  p(front0 winner) = {p_front0:.3f}  (random se ~0.5)")
    assert p_front0 > 0.68


# ------------------------------ T6: REPRODUCIBILITY END-TO-END
def test_t6_reproducible_end_to_end():
    p = DF1(time=0.5, n_var=10)
    P0 = init_pop(p, 100, seed=7)
    F0 = p.evaluate(P0)
    seed_nsga2(42)
    P1, F1 = nsga2_one_generation(P0, F0, p, 100)
    seed_nsga2(42)
    P2, F2 = nsga2_one_generation(P0, F0, p, 100)
    assert np.allclose(P1, P2) and np.allclose(F1, F2)


# ----------------------- T7: BOUNDS REGRESSION SURVIVES TOURNAMENT
def test_t7_bounds_survive_tournament():
    D, N = 10, 100
    p = DF4(n_var=D, time=0.0)
    xl, xu = p.xl, p.xu
    pop = np.array([xl + 0.1 * (xu - xl), xu - 0.1 * (xu - xl)] * (N // 2))
    F = p.evaluate(pop)
    seed_nsga2(42)
    new_pop, _ = nsga2_one_generation(pop, F, p, N)
    assert np.any(new_pop < 0.0) or np.any(new_pop > 1.0)
    assert np.all(new_pop >= p.xl - 1e-9) and np.all(new_pop <= p.xu + 1e-9)


# --------------------------------------- T8: DF1 STATIC CONVERGENCE
def test_t8_df1_static_converges():
    p = DF1(time=0.5, n_var=10)
    seed_nsga2(42)
    pop = init_pop(p, 100, seed=42)
    F = p.evaluate(pop)
    for _ in range(200):
        pop, F = nsga2_one_generation(pop, F, p, 100)
    igd = IGD(p.pareto_front())(F)
    print(f"\n  DF1 static IGD (200 gen) = {igd:.5f}")
    assert igd < 0.01, igd


# ------------------------------------ T9: DF4 STATIC CANONICAL GATE
def test_t9_df4_static_gate():
    p = DF4(time=0.0, n_var=10)
    seed_nsga2(42)
    pop = init_pop(p, 100, seed=42)
    F = p.evaluate(pop)
    for _ in range(200):
        pop, F = nsga2_one_generation(pop, F, p, 100)
    igd = IGD(p.pareto_front())(F)
    print(f"\n  DF4 static IGD (200 gen) = {igd:.5f}  "
          f"(audit ~0.024, 4.5B random-mating 0.0138)")
    assert igd < 0.05, igd


# ------------------ T10: OPTIONAL PAIRED DIAGNOSTIC (KHONG GATE)
def _random_mating(pop_X, F_pop, problem, N):
    """Test-only control: random mating, KHONG dung o live path."""
    from pymoo.core.population import Population
    pop = Population.new("X", pop_X); pop.set("F", F_pop)
    parents_idx = np.column_stack([ng._rng.permutation(N)[:N // 2],
                                   ng._rng.permutation(N)[:N // 2]])
    rs = np.random.default_rng(ng._rng.randint(0, 2**31))
    off = ng._sbx.do(problem, pop, parents_idx, random_state=rs)
    off = ng._pm.do(problem, off, random_state=rs)
    off_X = np.clip(off.get("X"), problem.xl, problem.xu)
    off_eval = Population.new("X", off_X); off_eval.set("F", problem.evaluate(off_X))
    merged = Population.merge(pop, off_eval)
    surv = ng._survival.do(problem, merged, n_survive=N,
                           random_state=np.random.default_rng(ng._rng.randint(0, 2**31)))
    return surv.get("X"), surv.get("F")


def test_t10_paired_diagnostic():
    p = DF4(time=0.0, n_var=10)
    PF = p.pareto_front()
    print("\n  seed | IGD_random | IGD_canonical")
    for seed in [0, 1, 2, 3, 4]:
        P0 = init_pop(p, 100, seed=seed)
        F0 = p.evaluate(P0)
        seed_nsga2(seed)
        pr, fr = P0.copy(), F0.copy()
        for _ in range(200):
            pr, fr = _random_mating(pr, fr, p, 100)
        seed_nsga2(seed)
        pc, fc = P0.copy(), F0.copy()
        for _ in range(200):
            pc, fc = nsga2_one_generation(pc, fc, p, 100)
        print(f"   {seed}   | {IGD(PF)(fr):.5f}    | {IGD(PF)(fc):.5f}")
