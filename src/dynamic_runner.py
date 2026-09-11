import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from change_detector import ChangeDetector
from memory_archive import (MemoryArchive, calibrate_scale,
                            compute_signature)
from nsga2_pymoo import nsga2_one_generation, seed_nsga2
from sa_drl_dmoea import (IDX_HAS_MEMORY, apply_hierarchical_response,
                          build_state, compute_entropy, compute_hv_drop,
                          compute_phase, compute_reward, count_fe)


def validate_transition(state, gate):
    """MEMORY chi hop le khi archive khong rong tai luc chon."""
    if gate == 1 and state[IDX_HAS_MEMORY] < 0.5:
        raise RuntimeError(
            "MEMORY action recorded while archive was empty "
            f"(state has_memory={state[IDX_HAS_MEMORY]})")


def run_sa_drl(problem_class, n_t, tau_t,
               N, D, n_changes, warm_up,
               agent, segments, seed=0, training=True,
               ref_point=(2.0, 2.0), n_elite=10, verbose=False):

    np.random.seed(seed)
    import random
    random.seed(seed)
    seed_nsga2(seed)
    # Moi env (t=0 va K env sau change) chay tau_t dynamic gens.
    total_gens = warm_up + (n_changes + 1) * tau_t
    hv_calc = HV(ref_point=np.array(ref_point))
    hv_ref = float(np.prod(ref_point))
    fe_budget = N * tau_t

    archive = MemoryArchive()
    archive.clear()
    obj_scale = np.asarray(ref_point, dtype=float)   # cho ChangeDetector
    _p0 = problem_class(time=0.0, n_var=D)
    xl, xu = np.asarray(_p0.xl, dtype=float), np.asarray(_p0.xu, dtype=float)
    detector = ChangeDetector(
        segments=segments,
        obj_scale=np.asarray(ref_point, dtype=float),
        lb=xl, ub=xu,
        eps=0.01, kappa=2.0, lam=0.05, verbose=False,
    )
    # Probe set dong bang ca episode: c_tilde phai do environmental
    # change, khong tron voi population movement (xem QUY TAC 3).
    X_probe = None
    pop = xl + np.random.rand(N, D) * (xu - xl)
    pop_prev = pop.copy()
    problem = problem_class(time=0.0, n_var=D)
    F = problem.evaluate(pop)
    # Ledger FE: moi cong FE cong ca vao breakdown. Invariant cuoi run:
    # fes_counter == sum(fe_breakdown.values()). 1 FE = 1 candidate -> 1 F.
    fes_counter = 0
    fe_breakdown = {"initial": 0, "nsga2": 0, "detector": 0,
                    "signature": 0, "response": 0}

    def add_fe(key, k):
        nonlocal fes_counter
        fes_counter += int(k)
        fe_breakdown[key] += int(k)

    add_fe("initial", len(pop))

    pending_state = pending_gate = pending_seg = pending_hv_base = None
    pending_fe = 0
    igd_history = []
    raw_change_history = []
    normalized_change_history = []
    d_mem_history = []
    has_memory_history = []
    sig_saturation_history = []
    gate_history = []
    seg_history = []
    pending_signature = None
    current_time = 0.0
    timeline = []

    for gen in range(total_gens):
        # Change thu k (1-based) bắn SAU khi env hien tai da chay du tau_t
        # dynamic gens. Env t=0 vi the co dung tau_t gens truoc change 1.
        # KHONG con fake 0.0 -> 0.0.
        is_change = (gen >= warm_up + tau_t) and ((gen - warm_up) % tau_t == 0)

        if is_change:
            change_count = (gen - warm_up) // tau_t     # 1, 2, 3, ...
            t_old = current_time
            t_new = change_count / n_t
            timeline.append({"change_index": change_count,
                             "t_old": t_old, "t_new": t_new,
                             "gens_in_old_env": tau_t})

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
            add_fe("response", len(pop))
            hv_base = hv_calc(F_after_change)

            # RETRIEVE — truoc khi store bat cu thu gi cua t
            sig_t, sat = compute_signature(X_probe, problem_new,
                                           archive.obj_scale,
                                           return_saturation=True)
            add_fe("signature", len(X_probe))
            sig_saturation_history.append(sat)
            mem = archive.query(sig_t)
            d_mem_history.append(mem["d_mem"])
            has_memory_history.append(mem["has_memory"])

            # 4. State moi
            res_c   = detector.compute(X_probe, problem_new)
            c       = res_c["c"]
            add_fe("detector", res_c["fe_used"])
            raw_change_history.append(res_c["c_tilde"].tolist())
            normalized_change_history.append(c.tolist())
            entropy = compute_entropy(pop, xl, xu)
            hv_drop = compute_hv_drop(F_before, F_after_change, ref_point)
            phase   = compute_phase(t_new)
            state   = build_state(c, entropy, hv_drop, 0, tau_t, phase,
                                  d_mem=mem["d_mem"],
                                  has_memory=float(mem["has_memory"]))

            # 5. Push transition
            if reward is not None:
                validate_transition(pending_state, pending_gate)
                agent.replay_buffer.push(pending_state, pending_gate,
                                         pending_seg, reward, state, False)
                if training:
                    agent.learn()

            # 6. Chon + thi hanh
            gate, seg = agent.select_action(state, training=training)
            gate_history.append(int(gate))
            seg_history.append([int(a) for a in seg])
            pop_new = apply_hierarchical_response(pop, pop_prev, gate, seg,
                                                  segments, mem["pop"],
                                                  xl, xu)

            pop_prev = pop.copy()
            pop = pop_new
            problem = problem_new
            F = problem.evaluate(pop)
            add_fe("response", len(pop))   # eval THUC TE, luon N (xem audit)

            # 7. Cat pending. count_fe (DEPRECATED) khong con vao FE path;
            #    pending_fe cho reward van dung uoc luong cu (4.2 se sua).
            pending_state, pending_gate, pending_seg = state, gate, seg
            pending_hv_base = hv_base
            pending_fe = count_fe(gate, seg, N)
            pending_signature = sig_t

            # 8. Ghi IGD
            PF = problem.pareto_front()
            if PF is not None and len(PF) > 0:
                igd_history.append(IGD(PF)(F))

            current_time = t_new

            if verbose and change_count % 20 == 0:
                print(f"  Change {change_count:3d}, t={t_new:.2f}, "
                      f"c={np.round(c,2)}, gate={gate}, seg={seg}, "
                      f"IGD={igd_history[-1]:.6f}")

        pop, F, nsga_fe = nsga2_one_generation(pop, F, problem, N,
                                               return_fe=True)
        add_fe("nsga2", nsga_fe)

        if gen == warm_up - 1:
            X_probe = pop[:n_elite].copy()
            add_fe("detector", detector.prime(X_probe, problem))
            # Scale signature do MOT LAN tai t0, co dinh trong run.
            archive.obj_scale = calibrate_scale(problem.evaluate(X_probe))
            add_fe("signature", len(X_probe))

    if training:
        agent.decay_epsilon()

    assert fes_counter == sum(fe_breakdown.values()), \
        (fes_counter, fe_breakdown)
    return {
        "migd": float(np.mean(igd_history)) if igd_history else float("inf"),
        "igd_history": igd_history,
        "fes_used": int(fes_counter),
        "fe_breakdown": dict(fe_breakdown),
        "raw_change": raw_change_history,
        "timeline": timeline,
        "normalized_change": normalized_change_history,
        "d_mem": d_mem_history,
        "has_memory": has_memory_history,
        "sig_saturation": sig_saturation_history,
        "gate_action": gate_history,
        "seg_actions": seg_history,
        "feasible_ratio": 1.0,
        "hv_final": float(hv_calc(F)),
    }