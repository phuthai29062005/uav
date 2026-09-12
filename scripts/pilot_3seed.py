"""
PRE-BENCHMARK PILOT: 3 eval seeds x 5 structural-diverse DFs.

Protocol/infrastructure smoke for mode_frozen (PRIMARY). NOT a performance
benchmark. Real training budget (200 episodes), no shortcuts.
"""
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import dynamic_runner as dr  # noqa: E402
import train  # noqa: E402
from baseline_runner import run_nsga2_baseline  # noqa: E402
from change_detector import ChangeDetector  # noqa: E402
from memory_archive import MemoryArchive  # noqa: E402
from experiment_logger import ExperimentLogger  # noqa: E402

DFS = ["DF1", "DF2", "DF4", "DF10", "DF12"]
EVAL_SEEDS = [100000, 100001, 100002]
MODEL_SEED = 30
N_EP = 200


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


class Instrument:
    """Patch counters/capture for ONE run; restore on exit."""

    def __enter__(self):
        self.detector_compute = 0
        self.archive_query = 0
        self.archive_init = 0
        self.detector_init = 0
        self.pop_min = np.inf
        self.pop_max = -np.inf
        self.init_pop = None

        self._oc = ChangeDetector.compute
        self._oq = MemoryArchive.query
        self._oai = MemoryArchive.__init__
        self._odi = ChangeDetector.__init__
        self._orand = np.random.rand
        self._onsga = dr.nsga2_one_generation
        self._oresp = dr.apply_hierarchical_response
        self._orandom = train.run_sa_drl if False else None

        inst = self

        def compute_spy(self_d, *a, **k):
            inst.detector_compute += 1
            return inst._oc(self_d, *a, **k)

        def query_spy(self_a, *a, **k):
            inst.archive_query += 1
            return inst._oq(self_a, *a, **k)

        def ainit_spy(self_a, *a, **k):
            inst.archive_init += 1
            return inst._oai(self_a, *a, **k)

        def dinit_spy(self_d, *a, **k):
            inst.detector_init += 1
            return inst._odi(self_d, *a, **k)

        def rand_spy(*args, **kwargs):
            r = inst._orand(*args, **kwargs)
            if inst.init_pop is None and len(args) == 2:
                inst.init_pop = r.copy()
            return r

        def nsga_spy(pop_X, F_pop, problem, N, **kw):
            out = inst._onsga(pop_X, F_pop, problem, N, **kw)
            newpop = out[0]
            inst.pop_min = min(inst.pop_min, float(np.min(newpop)))
            inst.pop_max = max(inst.pop_max, float(np.max(newpop)))
            return out

        def resp_spy(pop, pop_prev, gate, seg, segments, pop_mem, xl, xu):
            out = inst._oresp(pop, pop_prev, gate, seg, segments, pop_mem,
                              xl, xu)
            inst.pop_min = min(inst.pop_min, float(np.min(out)))
            inst.pop_max = max(inst.pop_max, float(np.max(out)))
            return out

        ChangeDetector.compute = compute_spy
        MemoryArchive.query = query_spy
        MemoryArchive.__init__ = ainit_spy
        ChangeDetector.__init__ = dinit_spy
        np.random.rand = rand_spy
        dr.nsga2_one_generation = nsga_spy
        dr.apply_hierarchical_response = resp_spy
        return self

    def __exit__(self, *exc):
        ChangeDetector.compute = self._oc
        MemoryArchive.query = self._oq
        MemoryArchive.__init__ = self._oai
        ChangeDetector.__init__ = self._odi
        np.random.rand = self._orand
        dr.nsga2_one_generation = self._onsga
        dr.apply_hierarchical_response = self._oresp


def instrument_baseline():
    """Lighter instrument for baseline (init pop capture only)."""
    orig = np.random.rand
    captured = {}

    def spy(*a, **k):
        r = orig(*a, **k)
        if "pop" not in captured and len(a) == 2:
            captured["pop"] = r.copy()
        return r
    np.random.rand = spy
    return orig, captured


def finite_ok(res):
    keys = ["igd_pre_history", "igd_response_history", "igd_end_history",
            "hv_pre_history", "hv_response_history", "hv_end_history"]
    for k in keys:
        vals = [v for v in res.get(k, []) if v is not None]
        if not np.all(np.isfinite(vals)):
            return False
    if "reward_history" in res and res["reward_history"]:
        if not np.all(np.isfinite(res["reward_history"])):
            return False
    if not np.isfinite(res["fes_used"]):
        return False
    if not np.isfinite(res["migd"]) or not np.isfinite(res["migd_end"]):
        return False
    return True


def main():
    t_start = time.time()
    seg = train.make_segments(train.CFG["D"])
    training_sanity = []
    trained = {}

    # ================================================== TRAIN (per DF)
    print("=" * 70)
    print("PHASE 1: PER-DF TRAINING (real budget, 200 episodes)")
    print("=" * 70)
    for name in DFS:
        t0 = time.time()
        agent, log = train.train_on([name], N_EP, seg, seed=MODEL_SEED,
                                    log_every=100)
        dt = time.time() - t0
        online_hash = hsd(agent.online.state_dict())
        target_hash = hsd(agent.target.state_dict())
        dist = float(np.sqrt(sum(
            float(((a.detach() - b.detach()) ** 2).sum())
            for a, b in zip(agent.online.parameters(),
                            agent.target.parameters()))))
        nan_w = any(not torch.all(torch.isfinite(p)).item()
                    for p in agent.online.parameters())
        trained[name] = agent
        row = dict(df=name, model_seed=MODEL_SEED, n_episodes=N_EP,
                  train_time_s=round(dt, 1), online_hash=online_hash[:12],
                  target_hash=target_hash[:12], replay_size=len(agent.replay_buffer),
                  eps_final=round(agent.eps, 4), online_target_dist=round(dist, 4),
                  nan_weights=nan_w)
        training_sanity.append(row)
        print(f"  {name:5s} trained in {dt:6.1f}s  eps_final={agent.eps:.3f}  "
              f"replay={len(agent.replay_buffer)}  dist={dist:.3f}  "
              f"nan={nan_w}")

    # ================================================== EVAL (paired)
    print("\n" + "=" * 70)
    print("PHASE 2: PAIRED EVALUATION (3 unseen seeds x 5 DFs)")
    print("=" * 70)
    pilot_rows = []
    struct_rows = []
    logger = ExperimentLogger("results/pilot_3seed.jsonl")
    seen_keys = set()

    for name in DFS:
        N, ref = train.problem_setup(name)
        agent = trained[name]
        frozen_online = copy.deepcopy(agent.online.state_dict())
        frozen_target = copy.deepcopy(agent.target.state_dict())
        pf = train.ALL_PROBLEMS[name]
        n_obj_expected = 3 if name in train.TRI_OBJECTIVE else 2

        for seed in EVAL_SEEDS:
            fail_reasons = []

            eval_agent = train.new_agent(seg, seed)
            eval_agent.online.load_state_dict(frozen_online)
            eval_agent.target.load_state_dict(frozen_target)
            eval_agent.eps = 0.0
            step_before = eval_agent.step_count

            with Instrument() as inst:
                sa_res = dr.run_sa_drl(
                    pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                    N=N, D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
                    warm_up=train.CFG["warm_up"], agent=eval_agent,
                    segments=seg, seed=seed, training=False,
                    ref_point=ref, n_elite=train.CFG["n_elite"])
                sa_detector_calls = inst.detector_compute
                sa_query_calls = inst.archive_query
                sa_archive_inits = inst.archive_init
                sa_detector_inits = inst.detector_init
                sa_pop_min, sa_pop_max = inst.pop_min, inst.pop_max
                sa_init_pop = inst.init_pop

            # frozen invariants
            on_after = hsd(eval_agent.online.state_dict())
            tg_after = hsd(eval_agent.target.state_dict())
            frozen_ok = (on_after == hsd(frozen_online)
                        and tg_after == hsd(frozen_target)
                        and eval_agent.step_count == step_before
                        and eval_agent.eps == 0.0)
            if not frozen_ok:
                fail_reasons.append("FROZEN_WEIGHTS_CHANGED")

            # timeline
            tl = sa_res["timeline"]
            timeline_ok = (len(tl) == train.EVAL_CHANGES
                           and abs(tl[0]["t_old"] - 0.0) < 1e-9
                           and abs(tl[0]["t_new"] - 0.1) < 1e-9
                           and abs(tl[-1]["t_new"] - 10.0) < 1e-9)
            if not timeline_ok:
                fail_reasons.append("TIMELINE")
            if sa_detector_calls != train.EVAL_CHANGES:
                fail_reasons.append(f"DETECTOR_CALLS={sa_detector_calls}")
            if sa_query_calls != train.EVAL_CHANGES:
                fail_reasons.append(f"QUERY_CALLS={sa_query_calls}")
            if sa_archive_inits != 1 or sa_detector_inits != 1:
                fail_reasons.append("NOT_FRESH_EPISODE_STATE")

            # histories
            K = train.EVAL_CHANGES
            # Post-4.7: terminal transition IS pushed -> reward_history == K
            # (not K-1). Verified directly: dr.run_sa_drl(...)["reward_history"]
            # has length 100 for K=100 regardless of agent.
            hist_ok = (len(sa_res["igd_response_history"]) == K
                      and len(sa_res["igd_end_history"]) == K
                      and len(sa_res["reward_history"]) == K)
            if not hist_ok:
                fail_reasons.append("HISTORY_LENGTH")

            # FE ledger
            fe_ok = sa_res["fes_used"] == sum(sa_res["fe_breakdown"].values())
            if not fe_ok:
                fail_reasons.append("FE_LEDGER")

            # metric semantics
            migd_ok = (sa_res["migd"] == sa_res["migd_end"]
                      and abs(sa_res["migd_end"]
                              - np.mean(sa_res["igd_end_history"])) < 1e-9
                      and abs(sa_res["migd_response"]
                              - np.mean(sa_res["igd_response_history"])) < 1e-9)
            if not migd_ok:
                fail_reasons.append("MIGD_SEMANTICS")

            # bounds
            p0 = pf(time=0.0, n_var=train.CFG["D"])
            xl, xu = float(np.min(p0.xl)), float(np.max(p0.xu))
            bounds_ok = (sa_pop_min >= xl - 1e-9 and sa_pop_max <= xu + 1e-9)
            if not bounds_ok:
                fail_reasons.append(f"BOUNDS[{sa_pop_min},{sa_pop_max}]"
                                    f" vs [{xl},{xu}]")

            # objective dim
            obj_dim_ok = p0.n_obj == n_obj_expected
            if not obj_dim_ok:
                fail_reasons.append("OBJ_DIM")

            finite_ok_ = finite_ok(sa_res)
            if not finite_ok_:
                fail_reasons.append("NON_FINITE")

            # -------- baseline (paired, same seed) --------
            orig_rand, bl_captured = instrument_baseline()
            try:
                bl_res = run_nsga2_baseline(
                    pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                    N=N, D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
                    warm_up=train.CFG["warm_up"], seed=seed, ref_point=ref)
            finally:
                np.random.rand = orig_rand
            bl_init_pop = bl_captured.get("pop")

            paired_ok = (sa_init_pop is not None and bl_init_pop is not None
                        and np.allclose(sa_init_pop, bl_init_pop))
            if not paired_ok:
                fail_reasons.append("PAIRED_INIT_MISMATCH")

            bl_resp_eq_pre = np.allclose(
                [v for v in bl_res["igd_response_history"] if v is not None],
                [v for v in bl_res["igd_pre_history"] if v is not None])
            if not bl_resp_eq_pre:
                fail_reasons.append("BASELINE_RESPONSE_NE_PRE")

            bl_hist_ok = (len(bl_res["igd_response_history"]) == K
                         and len(bl_res["igd_end_history"]) == K)
            if not bl_hist_ok:
                fail_reasons.append("BASELINE_HISTORY_LENGTH")

            bl_finite_ok = finite_ok(bl_res)
            if not bl_finite_ok:
                fail_reasons.append("BASELINE_NON_FINITE")

            # -------- file uniqueness / logging --------
            key = ("SA-DRL-frozen", name, MODEL_SEED, seed)
            key_bl = ("NSGA2-clean", name, seed)
            dup = key in seen_keys or key_bl in seen_keys
            seen_keys.add(key)
            seen_keys.add(key_bl)
            if dup:
                fail_reasons.append("DUPLICATE_KEY")

            with logger.run(algo="SA-DRL-frozen-pilot", problem=name,
                            seed=seed, params={"model_seed": MODEL_SEED}):
                logger.record(migd=sa_res["migd"], hv=sa_res["hv_final"],
                              fes=sa_res["fes_used"],
                              feasible_ratio=sa_res["feasible_ratio"])
            with logger.run(algo="NSGA2-clean-pilot", problem=name,
                            seed=seed, params={}):
                logger.record(migd=bl_res["migd"], hv=bl_res["hv_final"],
                              fes=bl_res["fes_used"],
                              feasible_ratio=bl_res["feasible_ratio"])

            pilot_rows.append(dict(
                df=name, seed=seed,
                sa_migd_end=round(sa_res["migd_end"], 5),
                bl_migd_end=round(bl_res["migd_end"], 5),
                sa_migd_resp=round(sa_res["migd_response"], 5),
                bl_migd_resp=round(bl_res["migd_response"], 5),
                sa_fe=sa_res["fes_used"], bl_fe=bl_res["fes_used"],
                finite=finite_ok_ and bl_finite_ok,
                frozen=frozen_ok, paired_init=paired_ok,
                fail_reasons=fail_reasons))
            status = "PASS" if not fail_reasons else "FAIL:" + ",".join(fail_reasons)
            print(f"  {name:5s} seed={seed}  SA_end={sa_res['migd_end']:.4f} "
                  f"BL_end={bl_res['migd_end']:.4f}  SA_FE={sa_res['fes_used']} "
                  f"BL_FE={bl_res['fes_used']}  {status}")

        # per-DF structural summary
        df_rows = [r for r in pilot_rows if r["df"] == name]
        struct_rows.append(dict(
            df=name,
            finite_3of3=sum(r["finite"] for r in df_rows),
            frozen_ok=all(r["frozen"] for r in df_rows),
            fe_ledger_ok=all(not any("FE_LEDGER" in f for f in r["fail_reasons"])
                             for r in df_rows),
            hist_ok=all(not any("HISTORY" in f for f in r["fail_reasons"])
                       for r in df_rows),
            paired_init_ok=all(r["paired_init"] for r in df_rows),
            verdict="PASS" if all(not r["fail_reasons"] for r in df_rows)
            else "FAIL"))

    # ================================================== REPORT
    print("\n" + "=" * 70)
    print("15-ROW PAIRED PILOT TABLE")
    print("=" * 70)
    hdr = (f"{'DF':5s}{'Seed':>8}{'SA_end':>9}{'BL_end':>9}{'SA_resp':>9}"
          f"{'BL_resp':>9}{'SA_FE':>8}{'BL_FE':>8}{'fin':>5}{'froz':>6}{'pair':>6}")
    print(hdr)
    for r in pilot_rows:
        print(f"{r['df']:5s}{r['seed']:>8}{r['sa_migd_end']:>9.4f}"
              f"{r['bl_migd_end']:>9.4f}{r['sa_migd_resp']:>9.4f}"
              f"{r['bl_migd_resp']:>9.4f}{r['sa_fe']:>8}{r['bl_fe']:>8}"
              f"{str(r['finite']):>5}{str(r['frozen']):>6}{str(r['paired_init']):>6}")

    print("\n" + "=" * 70)
    print("STRUCTURAL SUMMARY TABLE")
    print("=" * 70)
    for r in struct_rows:
        print(f"  {r['df']:5s} finite={r['finite_3of3']}/3  frozen={r['frozen_ok']}  "
              f"FE={r['fe_ledger_ok']}  hist100={r['hist_ok']}  "
              f"paired={r['paired_init_ok']}  VERDICT={r['verdict']}")

    with open("results/pilot_summary.json", "w") as f:
        json.dump({"pilot_rows": pilot_rows, "struct_rows": struct_rows,
                   "training_sanity": training_sanity}, f, indent=2, default=str)

    overall_fail = any(r["verdict"] == "FAIL" for r in struct_rows)
    print(f"\nTotal pilot time: {time.time()-t_start:.0f}s")
    print("OVERALL:", "FAIL" if overall_fail else "STRUCTURAL PASS (see report for WARN items)")


if __name__ == "__main__":
    main()
