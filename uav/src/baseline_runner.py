"""
Clean Dynamic NSGA-II baseline (4.9).

Chay NSGA-II dong THUAN: timeline dung (4.6A), reevaluate population khi
environment doi, canonical NSGA-II (4.5A), bounds dung (4.5B), FE chinh
xac (4.6B), pre/end metrics (4.2). KHONG di qua SA-DRL controller:
khong ChangeDetector, MemoryArchive, signature, d_mem, build_state,
DQNAgent, hierarchical gate/actions, response operators.
"""
import random

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from nsga2_pymoo import nsga2_one_generation, seed_nsga2


def _validate_ref_point(ref_point, n_obj):
    """ref_point phai duoc truyen tuong minh va khop so objective."""
    if ref_point is None:
        raise ValueError(
            "ref_point must be explicitly provided and match problem.n_obj")
    ref = np.asarray(ref_point, dtype=float)
    if ref.ndim != 1 or len(ref) != n_obj:
        raise ValueError(
            f"ref_point dim {ref.shape} != problem.n_obj {n_obj}")
    if not (np.all(np.isfinite(ref)) and np.all(ref > 0)):
        raise ValueError(f"ref_point must be finite and positive: {ref}")


def run_nsga2_baseline(problem_class, n_t, tau_t, N, D, n_changes, warm_up,
                       seed=0, ref_point=None, verbose=False):
    np.random.seed(seed)
    random.seed(seed)
    seed_nsga2(seed)
    _p0 = problem_class(time=0.0, n_var=D)
    _validate_ref_point(ref_point, _p0.n_obj)
    # Timeline y het SA runner (4.6A): t=0 chay tau_t dynamic gens truoc
    # change 1; change k (1-based) -> t_k = k/n_t; K actual transitions.
    total_gens = warm_up + (n_changes + 1) * tau_t
    hv_calc = HV(ref_point=np.array(ref_point))

    xl, xu = np.asarray(_p0.xl, dtype=float), np.asarray(_p0.xu, dtype=float)
    pop = xl + np.random.rand(N, D) * (xu - xl)
    problem = problem_class(time=0.0, n_var=D)
    F = problem.evaluate(pop)

    fes_counter = 0
    fe_breakdown = {"initial": 0, "nsga2": 0, "environment_reeval": 0}

    def add_fe(key, k):
        nonlocal fes_counter
        fes_counter += int(k)
        fe_breakdown[key] += int(k)

    add_fe("initial", len(pop))

    def _igd(prob, Fvals):
        pf = prob.pareto_front()
        return float(IGD(pf)(Fvals)) if pf is not None and len(pf) else None

    igd_pre_history, igd_response_history, igd_end_history = [], [], []
    hv_pre_history, hv_response_history, hv_end_history = [], [], []
    timeline = []
    current_time = 0.0

    for gen in range(total_gens):
        is_change = (gen >= warm_up + tau_t) and ((gen - warm_up) % tau_t == 0)

        if is_change:
            change_count = (gen - warm_up) // tau_t
            t_old, t_new = current_time, change_count / n_t
            timeline.append({"change_index": change_count, "t_old": t_old,
                             "t_new": t_new, "gens_in_old_env": tau_t})

            # END metric cua env vua ket thuc (t_{k-1}), duoi problem hien tai.
            if igd_pre_history:
                igd_end_history.append(_igd(problem, F))
                hv_end_history.append(float(hv_calc(F)))

            # Reevaluate inherited population MOT LAN duoi env moi — chi phi
            # can thiet cua dynamic NSGA-II (fitness cu khong con dung).
            problem_new = problem_class(time=t_new, n_var=D)
            F_pre = problem_new.evaluate(pop)
            add_fe("environment_reeval", len(pop))
            hv_pre = float(hv_calc(F_pre))
            igd_pre = _igd(problem_new, F_pre)
            igd_pre_history.append(igd_pre)
            hv_pre_history.append(hv_pre)
            # Baseline KHONG co explicit response -> alias metric, KHONG
            # evaluate lan 2 (response FE = 0).
            igd_response_history.append(igd_pre)
            hv_response_history.append(hv_pre)

            problem = problem_new
            F = F_pre
            current_time = t_new

        pop, F, nsga_fe = nsga2_one_generation(pop, F, problem, N,
                                               return_fe=True)
        add_fe("nsga2", nsga_fe)

    # END metric cua env cuoi t_K.
    if igd_pre_history:
        igd_end_history.append(_igd(problem, F))
        hv_end_history.append(float(hv_calc(F)))

    assert fes_counter == sum(fe_breakdown.values()), \
        (fes_counter, fe_breakdown)

    def _mean(h):
        return float(np.mean(h)) if h else float("inf")

    return {
        "migd": _mean(igd_end_history),          # alias -> end
        "migd_end": _mean(igd_end_history),
        "migd_response": _mean(igd_response_history),
        "igd_pre_history": igd_pre_history,
        "igd_response_history": igd_response_history,
        "igd_end_history": igd_end_history,
        "hv_pre_history": hv_pre_history,
        "hv_response_history": hv_response_history,
        "hv_end_history": hv_end_history,
        "reward_history": [],                    # baseline khong co RL reward
        "timeline": timeline,
        "fes_used": int(fes_counter),
        "fe_breakdown": dict(fe_breakdown),
        "feasible_ratio": 1.0,
        "hv_final": float(hv_calc(F)),
    }
