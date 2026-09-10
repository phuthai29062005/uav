import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from nsga2_pymoo import nsga2_one_generation
from sa_drl_dmoea import (Memory, apply_actions, build_state,
                          compute_change_vector, compute_entropy,
                          compute_env_key, compute_hv_drop, compute_phase,
                          compute_reward, count_fe)


def run_sa_drl(problem_class, n_t, tau_t,
               N, D, n_changes, warm_up,
               agent, segments, seed=0, training=True,
               ref_point=(2.0, 2.0), n_elite=10, verbose=False):

    np.random.seed(seed)
    import random
    random.seed(seed)
    total_gens = warm_up + n_changes * tau_t
    hv_calc = HV(ref_point=np.array(ref_point))
    hv_ref = float(np.prod(ref_point))
    fe_budget = N * tau_t

    memory = Memory()
    pop = np.random.rand(N, D)
    pop_prev = pop.copy()
    problem = problem_class(time=0.0, n_var=D)
    F = problem.evaluate(pop)

    pending_state = pending_action = pending_hv_base = None
    pending_fe = 0
    fes_counter = N  # initial evaluate
    igd_history = []

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

            # 2. Store population da hoi tu
            if gen > warm_up:
                memory.store(compute_env_key(pop), pop)

            # 3. Sang moi truong moi
            problem_new = problem_class(time=t_new, n_var=D)
            F_before = F.copy()
            F_after_change = problem_new.evaluate(pop)
            fes_counter += N
            hv_base = hv_calc(F_after_change)

            # 4. State moi
            elite = pop[:n_elite]
            c       = compute_change_vector(elite, problem_new, segments, D)
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
            key = compute_env_key(pop)
            pop_new = apply_actions(pop, pop_prev, actions, segments, memory, key)

            pop_prev = pop.copy()
            pop = pop_new
            problem = problem_new
            F = problem.evaluate(pop)

            # 7. Cat pending
            pending_state, pending_action = state, actions
            pending_hv_base = hv_base
            pending_fe = count_fe(actions, N, n_elite, len(segments))
            fes_counter += pending_fe

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

    if training:
        agent.decay_epsilon()

    return {
        "migd": float(np.mean(igd_history)) if igd_history else float("inf"),
        "igd_history": igd_history,
        "fes_used": int(fes_counter),
        "feasible_ratio": 1.0,
        "hv_final": float(hv_calc(F)),
    }