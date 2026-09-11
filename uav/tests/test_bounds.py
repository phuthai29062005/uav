"""B0-B7: propagate problem bounds, remove hard-coded [0,1]."""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from change_detector import ChangeDetector  # noqa: E402
from nsga2_pymoo import nsga2_one_generation, seed_nsga2  # noqa: E402
from sa_drl_dmoea import (action_diversify, action_local,  # noqa: E402
                          action_memory_global, action_predict,
                          apply_hierarchical_response, compute_entropy, repair)

ALL = [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10, DF11, DF12, DF13, DF14]
DIFF_BOUNDS = {3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14}   # bounds != [0,1]


def init_pop(problem, N=40, seed=0):
    xl, xu = problem.xl, problem.xu
    return xl + np.random.RandomState(seed).rand(N, len(xl)) * (xu - xl)


# --------------------- B0: OFFSPRING KHONG BI CLIP VE [0,1] (loi Phan 0)
def test_b0_offspring_not_clipped_to_unit():
    """Tat dinh: parent co dinh + seed co dinh, khong may rui."""
    D, N = 10, 100
    p = DF4(n_var=D, time=0.0)
    xl, xu = p.xl, p.xu
    p_low = xl + 0.1 * (xu - xl)          # ~ -1.6 cho DF4
    p_high = xu - 0.1 * (xu - xl)         # ~ +1.6 cho DF4
    pop = np.array([p_low, p_high] * (N // 2))
    F = p.evaluate(pop)

    seed_nsga2(42)
    new_pop, _ = nsga2_one_generation(pop, F, p, N)

    assert np.any(new_pop < 0.0) or np.any(new_pop > 1.0), \
        "offspring van bi ep ve [0,1] — bug Phan 0 chua sua"
    assert np.all(new_pop >= p.xl - 1e-9)
    assert np.all(new_pop <= p.xu + 1e-9)


# ------------------------------------------------- B1: INIT DUNG MIEN
def test_b1_init_respects_domain():
    p = DF4(time=0.0, n_var=10)
    pop = init_pop(p, N=200)
    assert pop.min() < 0                       # DF4 mien [-2,2]
    assert np.all(pop >= p.xl) and np.all(pop <= p.xu)


# --------------------------------------------- B2: ACTIONS GIU MIEN
def test_b2_actions_stay_in_domain():
    p = DF4(time=0.0, n_var=10)
    xl, xu = p.xl, p.xu
    segs = [[0], list(range(1, 10))]
    pop = init_pop(p, seed=1)
    prev = init_pop(p, seed=2)
    pop_mem = init_pop(p, seed=3)
    np.random.seed(0)
    outs = {
        "local": action_local(pop, segs[1], xl, xu),
        "predict": action_predict(pop, prev, segs[1], xl, xu),
        "diversify": action_diversify(pop, segs[1], xl, xu),
        "memory": action_memory_global(pop, pop_mem, xl, xu),
        "repair": repair(pop * 3.0, xl, xu),
    }
    for name, out in outs.items():
        assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9), name
    assert outs["diversify"].min() < 0         # co the < 0 trong [-2,2]


# ---------------------------------------------------- B3: ENTROPY CHUAN
def test_b3_entropy_normalized():
    p = DF4(time=0.0, n_var=10)
    pop = init_pop(p, N=500, seed=7)
    e = compute_entropy(pop, p.xl, p.xu)
    assert 0.8 < e <= 1.0, e                    # uniform -> ~1, KHONG ~3.4


# ------------------------------------------- B4: DETECTOR TAI BIEN THAT
def test_b4_detector_at_real_boundary():
    p = DF4(time=0.0, n_var=10)
    X = init_pop(p, N=8, seed=5)
    X[:, 0] = p.xl[0]                           # x_0 = -2, bien duoi that
    d = ChangeDetector(segments=[[0], list(range(1, 10))],
                       obj_scale=np.array([5.0, 5.0]),
                       lb=p.xl, ub=p.xu, eps=0.01, verbose=False)
    d.prime(X, DF4(time=0.0, n_var=10))
    res = d.compute(X, DF4(time=0.1, n_var=10))
    assert np.all(np.isfinite(res["c_tilde"]))
    assert d.last_prime_stats["onesided2"] > 0 or \
        d.last_prime_stats["onesided1"] > 0


# ---------------------------------------- B5: DF1 KHONG DOI (regression)
def test_b5_df1_unchanged():
    """DF1 bounds [0,1], width=1 -> hanh vi phai giong het truoc 4.5B."""
    import dynamic_runner
    from baselines import NSGA2Baseline

    def run():
        return dynamic_runner.run_sa_drl(
            DF1, n_t=10, tau_t=10, N=100, D=10, n_changes=10, warm_up=50,
            agent=NSGA2Baseline(n_segments=2),
            segments=[[0], list(range(1, 10))],
            seed=42, training=False, ref_point=(2.0, 2.0), n_elite=10)
    r1, r2 = run()["igd_history"], run()["igd_history"]
    assert len(r1) == len(r2) == 10
    assert max(abs(a - b) for a, b in zip(r1, r2)) < 1e-9


# --------------------------------- B6: DF4 STATIC CAI THIEN MANH
def test_b6_df4_static_converges():
    p = DF4(time=0.0, n_var=10)
    seed_nsga2(0)
    pop = init_pop(p, N=100)
    F = p.evaluate(pop)
    from pymoo.indicators.igd import IGD
    PF = p.pareto_front()
    for _ in range(200):
        pop, F = nsga2_one_generation(pop, F, p, 100)
    igd = IGD(PF)(F)
    print(f"\n  DF4 static IGD sau 200 gen = {igd:.4f}")
    assert igd < 0.1, igd


# ----------------------------- B7: TAT CA 14 BAI KHOI TAO DUNG
@pytest.mark.parametrize("i,cls", list(enumerate(ALL, start=1)))
def test_b7_all_problems_init(i, cls):
    p = cls(time=0.0, n_var=10)
    pop = init_pop(p, N=100, seed=i)
    assert np.all(pop >= p.xl) and np.all(pop <= p.xu)
    if i in DIFF_BOUNDS:
        assert np.any(pop < 0) or np.any(pop > 1), f"DF{i} bounds != [0,1]"
