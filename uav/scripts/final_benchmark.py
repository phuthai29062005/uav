"""
5.1 — FINAL MAIN BENCHMARK DATA COLLECTION
14 DFs x 30 paired test seeds. Data collection ONLY, no statistics,
no performance interpretation. Resume-safe (checkpoint + append-only JSONL).
"""
import copy
import hashlib
import json
import os
import subprocess
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

DFS = [f"DF{i}" for i in range(1, 15)]
TEST_SEEDS = list(range(100000, 100030))
MODEL_SEED = 30
N_EP = 200
CKPT_DIR = os.path.join(_HERE, "..", "results", "checkpoints")
RUNS_PATH = os.path.join(_HERE, "..", "results", "final_benchmark_runs.jsonl")
MODELS_PATH = os.path.join(_HERE, "..", "results", "final_benchmark_models.jsonl")
MANIFEST_PATH = os.path.join(_HERE, "..", "results", "final_benchmark_manifest.json")
FAILURES_PATH = os.path.join(_HERE, "..", "results", "final_benchmark_failures.jsonl")

os.makedirs(CKPT_DIR, exist_ok=True)


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                          capture_output=True, text=True).stdout.strip()


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def log_failure(row):
    append_jsonl(FAILURES_PATH, row)
    print(f"  !!! FAILURE: {row}")


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
            inst.pop_min = min(inst.pop_min, float(np.min(out[0])))
            inst.pop_max = max(inst.pop_max, float(np.max(out[0])))
            return out

        def resp_spy(pop, pop_prev, gate, seg, segments, pop_mem, xl, xu):
            out = inst._oresp(pop, pop_prev, gate, seg, segments, pop_mem, xl, xu)
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


def finite_ok(res):
    keys = ["igd_pre_history", "igd_response_history", "igd_end_history",
            "hv_pre_history", "hv_response_history", "hv_end_history"]
    for k in keys:
        vals = [v for v in res.get(k, []) if v is not None]
        if vals and not np.all(np.isfinite(vals)):
            return False
    if res.get("reward_history"):
        if not np.all(np.isfinite(res["reward_history"])):
            return False
    return (np.isfinite(res["fes_used"]) and np.isfinite(res["migd"])
           and np.isfinite(res["migd_end"]))


def train_or_load(name, seg, existing_model_rows):
    """Resume-safe: reuse checkpoint if manifest row + file both present and
    hash matches; else train from scratch (deterministic, verified 4.10E.1)."""
    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
    existing = next((r for r in existing_model_rows if r["problem"] == name), None)

    if existing is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        agent = train.new_agent(seg, MODEL_SEED)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        h = hsd(agent.online.state_dict())
        if h != existing["final_online_hash"]:
            raise RuntimeError(
                f"RESUME HASH MISMATCH for {name}: ckpt={h} "
                f"manifest={existing['final_online_hash']} -- STOP")
        print(f"  {name}: resumed from checkpoint (hash {h[:12]} verified)")
        return agent, existing, True

    t0 = time.time()
    agent, _ = train.train_on([name], N_EP, seg, seed=MODEL_SEED,
                              log_every=100)
    dt = time.time() - t0
    online_sd = agent.online.state_dict()
    target_sd = agent.target.state_dict()
    finite_weights = all(torch.all(torch.isfinite(p)).item()
                         for p in agent.online.parameters())
    dist = float(np.sqrt(sum(
        float(((a.detach() - b.detach()) ** 2).sum())
        for a, b in zip(agent.online.parameters(), agent.target.parameters()))))
    torch.save({"online": online_sd, "target": target_sd}, ckpt_path)
    row = dict(problem=name, model_seed=MODEL_SEED,
              train_seed_start=train.TRAIN_SEED_BASE, n_train_episodes=N_EP,
              final_online_hash=hsd(online_sd), final_target_hash=hsd(target_sd),
              learn_steps=agent.step_count, replay_size=len(agent.replay_buffer),
              final_epsilon=round(agent.eps, 6),
              target_online_distance=round(dist, 6),
              finite_weights=finite_weights,
              selection_rule="fixed_final_checkpoint",
              train_time_s=round(dt, 1))
    append_jsonl(MODELS_PATH, row)
    print(f"  {name}: trained {dt:.1f}s  hash={row['final_online_hash'][:12]}  "
          f"finite={finite_weights}")
    return agent, row, False


def main():
    t_start = time.time()
    head_start = git_head()
    seg = train.make_segments(train.CFG["D"])

    # ---- manifest ----
    import pymoo, platform, datetime
    manifest = dict(
        code_commit=head_start, protocol_version="5.1",
        timestamp=datetime.datetime.now().isoformat(),
        python_version=sys.version.split()[0], pymoo_version=pymoo.__version__,
        torch_version=torch.__version__, numpy_version=np.__version__,
        platform=platform.platform(),
        problems=DFS, test_seeds=[TEST_SEEDS[0], TEST_SEEDS[-1]],
        n_test_seeds=len(TEST_SEEDS),
        n_train_episodes=N_EP, train_seed_start=train.TRAIN_SEED_BASE,
        model_seed=MODEL_SEED, selection_rule="fixed_final_checkpoint",
        segmentation="fixed_index_two_block: B1=[x0], B2=[x1..x_{D-1}]",
        target_tau=0.01, primary_metric="migd_end",
        secondary_metric="migd_response", fe_protocol="same_physical_schedule",
        warm_up=train.CFG["warm_up"], tau_t=train.CFG["tau_t"],
        n_t=train.CFG["n_t"], changes=train.EVAL_CHANGES, D=train.CFG["D"],
        numerical_time_hardening=True,
        probe_limitation_note=("4.10D D2: frozen probe drift bounded/stable; "
                               "signature soft-saturation observed on some "
                               "tri-objective DFs (DF10 late-horizon); accepted "
                               "known limitation, not gated"),
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    existing_models = load_jsonl(MODELS_PATH)
    existing_runs = load_jsonl(RUNS_PATH)
    done_keys = {(r["method"], r["problem"], r["seed"]) for r in existing_runs
                if r.get("finite_pass") and r.get("frozen_weights_pass") is not False}

    print(f"Resume state: {len(existing_models)} models, {len(existing_runs)} runs already present.")

    n_stopped_dfs = []

    for name in DFS:
        if git_head() != head_start:
            print("!!! SOURCE COMMIT CHANGED DURING BENCHMARK -- STOPPING ALL")
            sys.exit(1)

        print(f"\n=== {name} ===")
        N, ref = train.problem_setup(name)
        pf = train.ALL_PROBLEMS[name]
        n_obj_expected = 3 if name in train.TRI_OBJECTIVE else 2

        agent, model_row, resumed = train_or_load(name, seg, existing_models)
        if not resumed:
            existing_models.append(model_row)
        frozen_online = {k: v.clone() for k, v in agent.online.state_dict().items()}
        frozen_target = {k: v.clone() for k, v in agent.target.state_dict().items()}
        frozen_hash = hsd(frozen_online)

        df_failed = False
        for seed in TEST_SEEDS:
            if ("SA-DRL-frozen", name, seed) in done_keys:
                continue

            eval_agent = train.new_agent(seg, seed)
            eval_agent.online.load_state_dict(frozen_online)
            eval_agent.target.load_state_dict(frozen_target)
            eval_agent.eps = 0.0
            step_before = eval_agent.step_count
            eps_before = eval_agent.eps
            on_before_hash = hsd(eval_agent.online.state_dict())

            if on_before_hash != frozen_hash:
                log_failure(dict(df=name, seed=seed, reason="WRONG_MODEL_HASH_LOADED"))
                df_failed = True
                break

            try:
                with Instrument() as inst:
                    sa_res = dr.run_sa_drl(
                        pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                        N=N, D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
                        warm_up=train.CFG["warm_up"], agent=eval_agent,
                        segments=seg, seed=seed, training=False,
                        ref_point=ref, n_elite=train.CFG["n_elite"])
            except Exception as e:
                log_failure(dict(df=name, seed=seed, method="SA", error=str(e)))
                df_failed = True
                break

            on_after = hsd(eval_agent.online.state_dict())
            tg_after = hsd(eval_agent.target.state_dict())
            frozen_pass = (on_after == frozen_hash
                          and tg_after == hsd(frozen_target)
                          and eval_agent.step_count == step_before
                          and eval_agent.eps == 0.0)

            tl = sa_res["timeline"]
            timeline_pass = (len(tl) == train.EVAL_CHANGES
                             and abs(tl[0]["t_old"]) < 1e-9
                             and abs(tl[0]["t_new"] - 0.1) < 1e-9
                             and abs(tl[-1]["t_new"] - 10.0) < 1e-9
                             and inst.detector_compute == train.EVAL_CHANGES
                             and inst.archive_query == train.EVAL_CHANGES
                             and inst.archive_init == 1 and inst.detector_init == 1)

            K = train.EVAL_CHANGES
            history_pass = (len(sa_res["igd_pre_history"]) == K
                            and len(sa_res["igd_response_history"]) == K
                            and len(sa_res["igd_end_history"]) == K
                            and len(sa_res["reward_history"]) == K)

            fe_ledger_pass = sa_res["fes_used"] == sum(sa_res["fe_breakdown"].values())
            migd_pass = (sa_res["migd"] == sa_res["migd_end"]
                        and abs(sa_res["migd_end"] - np.mean(sa_res["igd_end_history"])) < 1e-9
                        and abs(sa_res["migd_response"] - np.mean(sa_res["igd_response_history"])) < 1e-9)

            p0 = pf(time=0.0, n_var=train.CFG["D"])
            xl_lo, xu_hi = float(np.min(p0.xl)), float(np.max(p0.xu))
            bounds_pass = (inst.pop_min >= xl_lo - 1e-9 and inst.pop_max <= xu_hi + 1e-9)
            obj_dim_pass = p0.n_obj == n_obj_expected
            finite_pass = finite_ok(sa_res)
            sa_init_pop = inst.init_pop

            # -------- baseline --------
            orig_rand = np.random.rand
            bl_captured = {}

            def rand_spy(*a, **k):
                r = orig_rand(*a, **k)
                if "pop" not in bl_captured and len(a) == 2:
                    bl_captured["pop"] = r.copy()
                return r
            np.random.rand = rand_spy
            try:
                bl_res = run_nsga2_baseline(
                    pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                    N=N, D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
                    warm_up=train.CFG["warm_up"], seed=seed, ref_point=ref)
            except Exception as e:
                np.random.rand = orig_rand
                log_failure(dict(df=name, seed=seed, method="BL", error=str(e)))
                df_failed = True
                break
            np.random.rand = orig_rand

            bl_init_pop = bl_captured.get("pop")
            paired_init_pass = (sa_init_pop is not None and bl_init_pop is not None
                                and np.allclose(sa_init_pop, bl_init_pop))
            bl_resp_eq_pre = np.allclose(
                [v for v in bl_res["igd_response_history"] if v is not None],
                [v for v in bl_res["igd_pre_history"] if v is not None])
            bl_history_pass = (len(bl_res["igd_pre_history"]) == K
                               and len(bl_res["igd_response_history"]) == K
                               and len(bl_res["igd_end_history"]) == K
                               and bl_resp_eq_pre)
            bl_fe_ledger_pass = bl_res["fes_used"] == sum(bl_res["fe_breakdown"].values())
            bl_finite_pass = finite_ok(bl_res)

            all_pass = (frozen_pass and timeline_pass and history_pass
                       and fe_ledger_pass and migd_pass and bounds_pass
                       and obj_dim_pass and finite_pass and paired_init_pass
                       and bl_history_pass and bl_fe_ledger_pass and bl_finite_pass)
            if not all_pass:
                log_failure(dict(
                    df=name, seed=seed, frozen_pass=frozen_pass,
                    timeline_pass=timeline_pass, history_pass=history_pass,
                    fe_ledger_pass=fe_ledger_pass, migd_pass=migd_pass,
                    bounds_pass=bounds_pass, obj_dim_pass=obj_dim_pass,
                    finite_pass=finite_pass, paired_init_pass=paired_init_pass,
                    bl_history_pass=bl_history_pass,
                    bl_fe_ledger_pass=bl_fe_ledger_pass,
                    bl_finite_pass=bl_finite_pass))
                df_failed = True
                break

            fe_bd = sa_res["fe_breakdown"]
            sa_row = dict(
                experiment_id=f"frozen_{name}_model{MODEL_SEED}_seed{seed}",
                protocol="frozen_per_df", method="SA-DRL-frozen", problem=name,
                seed=seed, code_commit=head_start,
                model_seed=MODEL_SEED, train_seed_start=train.TRAIN_SEED_BASE,
                n_train_episodes=N_EP, selection_rule="fixed_final_checkpoint",
                N=N, D=train.CFG["D"], warm_up=train.CFG["warm_up"],
                tau_t=train.CFG["tau_t"], n_t=train.CFG["n_t"],
                changes=train.EVAL_CHANGES,
                segmentation="fixed_index_two_block",
                migd_end=sa_res["migd_end"], migd_response=sa_res["migd_response"],
                fes_used=sa_res["fes_used"], fe_breakdown=fe_bd,
                FE_environment_reeval=K * N, FE_response_extra=K * N,
                FE_detector=fe_bd.get("detector", 0),
                FE_signature=fe_bd.get("signature", 0),
                frozen_weights_pass=frozen_pass, paired_init_pass=paired_init_pass,
                finite_pass=finite_pass, timeline_pass=timeline_pass,
                history_pass=history_pass, fe_ledger_pass=fe_ledger_pass)
            bl_fe_bd = bl_res["fe_breakdown"]
            bl_row = dict(
                experiment_id=f"clean_nsga2_{name}_seed{seed}",
                protocol="clean_nsga2", method="NSGA2-clean", problem=name,
                seed=seed, code_commit=head_start,
                model_seed=None, train_seed_start=None, n_train_episodes=None,
                selection_rule="not_applicable",
                N=N, D=train.CFG["D"], warm_up=train.CFG["warm_up"],
                tau_t=train.CFG["tau_t"], n_t=train.CFG["n_t"],
                changes=train.EVAL_CHANGES,
                segmentation="fixed_index_two_block",
                migd_end=bl_res["migd_end"], migd_response=bl_res["migd_response"],
                fes_used=bl_res["fes_used"], fe_breakdown=bl_fe_bd,
                FE_environment_reeval=bl_fe_bd.get("environment_reeval", 0),
                FE_response_extra=0, FE_detector=0, FE_signature=0,
                frozen_weights_pass=None, paired_init_pass=paired_init_pass,
                finite_pass=bl_finite_pass, timeline_pass=True,
                history_pass=bl_history_pass, fe_ledger_pass=bl_fe_ledger_pass)
            append_jsonl(RUNS_PATH, sa_row)
            append_jsonl(RUNS_PATH, bl_row)
            done_keys.add(("SA-DRL-frozen", name, seed))
            done_keys.add(("NSGA2-clean", name, seed))

        if df_failed:
            n_stopped_dfs.append(name)
            print(f"  {name}: STOPPED due to failure (see failures log)")
        else:
            n_done = sum(1 for k in done_keys if k[1] == name)
            print(f"  {name}: {n_done} run-rows complete")

    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    print(f"Stopped DFs: {n_stopped_dfs if n_stopped_dfs else 'none'}")


if __name__ == "__main__":
    main()
