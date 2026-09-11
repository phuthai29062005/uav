import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from change_detector import ChangeDetector
from memory_archive import (MemoryArchive, calibrate_scale,
                            compute_signature)
from nsga2_pymoo import nsga2_one_generation, seed_nsga2
from sa_drl_dmoea import (IDX_HAS_MEMORY, apply_hierarchical_response,
                          build_state, compute_dispersion, compute_hv_drop,
                          compute_reward)


def validate_transition(state, gate):
    """MEMORY chi hop le khi archive khong rong tai luc chon."""
    if gate == 1 and state[IDX_HAS_MEMORY] < 0.5:
        raise RuntimeError(
            "MEMORY action recorded while archive was empty "
            f"(state has_memory={state[IDX_HAS_MEMORY]})")


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


def run_sa_drl(problem_class, n_t, tau_t,
               N, D, n_changes, warm_up,
               agent, segments, seed=0, training=True,
               ref_point=None, n_elite=10, verbose=False,
               mask_change_state=False, disable_memory=False):
    """
    mask_change_state, disable_memory (5.3B ablation hooks): both default
    False, in which case behavior is EXACTLY the production A4 path
    (verified by the Phan 7 regression guard). mask_change_state=True
    (A2): ChangeDetector still runs and real c is still logged, but the
    policy is fed zeros in place of c. disable_memory=True (A3): signature
    calibration/computation and archive query/store are skipped entirely
    (zero signature FE, zero query/store calls); state always carries
    d_mem=1.0, has_memory=0.0, which makes the existing hierarchical mask
    in DQNAgent structurally forbid the MEMORY gate — no separate gate
    override is needed.
    """

    np.random.seed(seed)
    import random
    random.seed(seed)
    seed_nsga2(seed)
    _p0 = problem_class(time=0.0, n_var=D)
    _validate_ref_point(ref_point, _p0.n_obj)
    # Moi env (t=0 va K env sau change) chay tau_t dynamic gens.
    total_gens = warm_up + (n_changes + 1) * tau_t
    hv_calc = HV(ref_point=np.array(ref_point))
    hv_ref = float(np.prod(ref_point))
    fe_budget = N * tau_t

    archive = MemoryArchive()
    archive.clear()
    obj_scale = np.asarray(ref_point, dtype=float)   # cho ChangeDetector
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

    def _igd(prob, Fvals):
        pf = prob.pareto_front()
        return float(IGD(pf)(Fvals)) if pf is not None and len(pf) else None

    pending_state = pending_gate = pending_seg = pending_hv_pre = None
    pending_action_fe = 0
    igd_pre_history = []
    igd_response_history = []
    igd_end_history = []
    hv_pre_history = []
    hv_response_history = []
    hv_end_history = []
    reward_history = []
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

            # 1. END metric cua environment VUA KET THUC (t_{k-1}), do
            #    duoi problem HIEN TAI (van la t_{k-1}); dong thoi chot
            #    reward cho action t_{k-1}: quality = (HV_end - HV_pre).
            if pending_state is not None:
                hv_end = hv_calc(F)
                igd_end_history.append(_igd(problem, F))
                hv_end_history.append(float(hv_end))
                reward = compute_reward(hv_end, pending_hv_pre, hv_ref,
                                        pending_action_fe, fe_budget)
                reward_history.append(float(reward))
            else:
                reward = None

            # 2. STORE population cuoi cua environment VUA ROI.
            if not disable_memory and pending_signature is not None:
                archive.store(pending_signature, pop)

            # 3. Sang moi truong moi + PRE metric (pop ke thua duoi t_k)
            problem_new = problem_class(time=t_new, n_var=D)
            F_before = F.copy()
            F_pre = problem_new.evaluate(pop)
            add_fe("response", len(pop))
            hv_pre = hv_calc(F_pre)
            igd_pre_history.append(_igd(problem_new, F_pre))
            hv_pre_history.append(float(hv_pre))

            # RETRIEVE — truoc khi store bat cu thu gi cua t
            if disable_memory:
                sig_t = None
                sig_saturation_history.append(None)
                mem = {"has_memory": False, "d_mem": 1.0, "pop": None, "index": None}
            else:
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
            normalized_change_history.append(c.tolist())          # log real c
            c_for_state = np.zeros_like(c) if mask_change_state else c
            dispersion = compute_dispersion(pop, xl, xu)
            hv_drop = compute_hv_drop(F_before, F_pre, ref_point)
            state   = build_state(c_for_state, dispersion, hv_drop,
                                  d_mem=mem["d_mem"],
                                  has_memory=float(mem["has_memory"]))

            # 5. Push transition cua action truoc (next_state = state moi)
            if reward is not None:
                validate_transition(pending_state, pending_gate)
                agent.replay_buffer.push(pending_state, pending_gate,
                                         pending_seg, reward, state, False)
                if training:
                    agent.learn()

            # 6. Chon + thi hanh + RESPONSE metric
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
            add_fe("response", len(pop))   # F_response, luon N (xem audit 4.6B)
            igd_response_history.append(_igd(problem, F))
            hv_response_history.append(float(hv_calc(F)))

            # 7. Cat pending. Cost = ACTUAL FE sau response (4.6B), khong
            #    count_fe. Implementation CEC eval ca population sau moi
            #    response nen FE cost bang nhau moi action; giu lai cho
            #    tinh lien tuc protocol va setting tuong lai co action-
            #    dependent evaluation cost.
            pending_state, pending_gate, pending_seg = state, gate, seg
            pending_hv_pre = hv_pre
            pending_action_fe = len(pop)
            pending_signature = sig_t

            current_time = t_new

            if verbose and change_count % 20 == 0:
                print(f"  Change {change_count:3d}, t={t_new:.2f}, "
                      f"c={np.round(c,2)}, gate={gate}, seg={seg}, "
                      f"IGD_resp={igd_response_history[-1]:.6f}")

        pop, F, nsga_fe = nsga2_one_generation(pop, F, problem, N,
                                               return_fe=True)
        add_fe("nsga2", nsga_fe)

        if gen == warm_up - 1:
            X_probe = pop[:n_elite].copy()
            add_fe("detector", detector.prime(X_probe, problem))
            if not disable_memory:
                # Scale signature do MOT LAN tai t0, co dinh trong run.
                archive.obj_scale = calibrate_scale(problem.evaluate(X_probe))
                add_fe("signature", len(X_probe))

    # END metric + TERMINAL transition cua environment cuoi t_K (4.7).
    # F la pop cuoi t_K sau du tau_t generations; problem van la t_K.
    # KHONG tao env K+1, KHONG detector/query/signature moi, KHONG eval
    # objective moi -> FE khong tang.
    if pending_state is not None:
        hv_end = hv_calc(F)
        igd_end_history.append(_igd(problem, F))
        hv_end_history.append(float(hv_end))
        reward = compute_reward(hv_end, pending_hv_pre, hv_ref,
                                pending_action_fe, fe_budget)
        reward_history.append(float(reward))
        # Terminal next_state chi la cau truc; done=True mask bootstrap.
        terminal_next_state = np.zeros_like(pending_state)
        validate_transition(pending_state, pending_gate)
        agent.replay_buffer.push(pending_state, pending_gate, pending_seg,
                                 reward, terminal_next_state, True)
        if training:
            agent.learn()

    if training:
        agent.decay_epsilon()

    assert fes_counter == sum(fe_breakdown.values()), \
        (fes_counter, fe_breakdown)
    return {
        # "migd" giu cho tuong thich; tu 4.2 no LA end-of-environment MIGD.
        "migd": (float(np.mean(igd_end_history)) if igd_end_history
                 else float("inf")),
        "migd_end": (float(np.mean(igd_end_history)) if igd_end_history
                     else float("inf")),
        "migd_response": (float(np.mean(igd_response_history))
                          if igd_response_history else float("inf")),
        "igd_history": igd_end_history,      # alias -> end
        "igd_pre_history": igd_pre_history,
        "igd_response_history": igd_response_history,
        "igd_end_history": igd_end_history,
        "hv_pre_history": hv_pre_history,
        "hv_response_history": hv_response_history,
        "hv_end_history": hv_end_history,
        "reward_history": reward_history,
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