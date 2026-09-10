import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from change_detector import ChangeDetector
from memory_archive import MemoryArchive, compute_signature
from nsga2_pymoo import nsga2_one_generation, seed_nsga2
from sa_drl_dmoea import (apply_actions, build_state, compute_entropy,
                          compute_hv_drop, compute_phase, compute_reward,
                          count_fe)


def run_sa_drl(problem_class, n_t, tau_t,
               N, D, n_changes, warm_up,
               agent, segments, seed=0, training=True,
               ref_point=(2.0, 2.0), n_elite=10, verbose=False):

    np.random.seed(seed)
    import random
    random.seed(seed)
    seed_nsga2(seed)
    total_gens = warm_up + n_changes * tau_t
    hv_calc = HV(ref_point=np.array(ref_point))
    hv_ref = float(np.prod(ref_point))
    fe_budget = N * tau_t

    archive = MemoryArchive()
    archive.clear()
    obj_scale = np.asarray(ref_point, dtype=float)
    detector = ChangeDetector(
        segments=segments,
        obj_scale=np.asarray(ref_point, dtype=float),
        eps=0.01, kappa=2.0, lam=0.05, verbose=False,
    )
    # Probe set dong bang ca episode: c_tilde phai do environmental
    # change, khong tron voi population movement (xem QUY TAC 3).
    X_probe = None
    pop = np.random.rand(N, D)
    pop_prev = pop.copy()
    problem = problem_class(time=0.0, n_var=D)
    F = problem.evaluate(pop)

    pending_state = pending_action = pending_hv_base = None
    pending_fe = 0
    fes_counter = N  # initial evaluate
    igd_history = []
    raw_change_history = []
    normalized_change_history = []
    d_mem_history = []
    has_memory_history = []
    pending_signature = None

    for gen in range(total_gens):
        is_change = (gen >= warm_up) and ((gen - warm_up) % tau_t == 0)

        if is_change:
            change_idx = (gen - warm_up) // tau_t
            t_new = change_idx / n_t

            # 1. Chot reward cho quyet dinh truoc
            if pending_state is not None:
                hv_after = hv_calc(F)
                reward = compute_reward(hv_after, pending_hv_base, hv_ref,
                                        pending_fe, fe_budget)
            else:
                reward = None

            # 2. STORE population cuoi cua environment VUA ROI, voi
            #    signature cua environment VUA ROI. Entry cua t chi
            #    duoc store sau khi da roi khoi t -> khong tu retrieve.
            if pending_signature is not None:
                archive.store(pending_signature, pop)

            # 3. Sang moi truong moi
            problem_new = problem_class(time=t_new, n_var=D)
            F_before = F.copy()
            F_after_change = problem_new.evaluate(pop)
            fes_counter += N
            hv_base = hv_calc(F_after_change)

            # RETRIEVE — truoc khi store bat cu thu gi cua t
            sig_t = compute_signature(X_probe, problem_new, obj_scale)
            fes_counter += len(X_probe)
            mem = archive.query(sig_t)
            d_mem_history.append(mem["d_mem"])
            has_memory_history.append(mem["has_memory"])

            # 4. State moi
            res_c   = detector.compute(X_probe, problem_new)
            c       = res_c["c"]
            fes_counter += res_c["fe_used"]
            raw_change_history.append(res_c["c_tilde"].tolist())
            normalized_change_history.append(c.tolist())
            entropy = compute_entropy(pop)
            hv_drop = compute_hv_drop(F_before, F_after_change, ref_point)
            phase   = compute_phase(t_new)
            state   = build_state(c, entropy, hv_drop, 0, tau_t, phase)

            # 5. Push transition
            if reward is not None:
                agent.replay_buffer.push(pending_state, pending_action,
                                         reward, state, False)
                if training:
                    agent.learn()

            # 6. Chon + thi hanh
            actions = agent.select_action(state, training=training)
            pop_new = apply_actions(pop, pop_prev, actions, segments,
                                    mem["pop"])

            pop_prev = pop.copy()
            pop = pop_new
            problem = problem_new
            F = problem.evaluate(pop)

            # 7. Cat pending
            pending_state, pending_action = state, actions
            pending_hv_base = hv_base
            pending_fe = count_fe(actions, N, n_elite, len(segments))
            fes_counter += pending_fe
            pending_signature = sig_t

            # 8. Ghi IGD
            PF = problem.pareto_front()
            if PF is not None and len(PF) > 0:
                igd_history.append(IGD(PF)(F))

            if verbose and change_idx % 20 == 0:
                print(f"  Change {change_idx:3d}, t={t_new:.2f}, "
                      f"c={np.round(c,2)}, actions={actions}, "
                      f"IGD={igd_history[-1]:.6f}")

        pop, F = nsga2_one_generation(pop, F, problem, N)
        fes_counter += N

        if gen == warm_up - 1:
            X_probe = pop[:n_elite].copy()
            fes_counter += detector.prime(X_probe, problem)

    if training:
        agent.decay_epsilon()

    return {
        "migd": float(np.mean(igd_history)) if igd_history else float("inf"),
        "igd_history": igd_history,
        "fes_used": int(fes_counter),
        "raw_change": raw_change_history,
        "normalized_change": normalized_change_history,
        "d_mem": d_mem_history,
        "has_memory": has_memory_history,
        "feasible_ratio": 1.0,
        "hv_final": float(hv_calc(F)),
    }