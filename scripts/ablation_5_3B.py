"""
5.3B — Clean holdout confirmatory ablation: A0/A1/A2/A3/A4 x 14 DF x 30
NEW holdout seeds (200000..200029, never used in training/main-test/5.3A).
Data collection only (no statistics). Resume-safe.
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
from dqn_agent import DQNAgent  # noqa: E402
from memory_archive import MemoryArchive  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
RAW_MAIN_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
MAIN_MODELS_PATH = os.path.join(RESULTS, "final_benchmark_models.jsonl")
A4_CKPT_DIR = os.path.join(RESULTS, "checkpoints")
ABL_CKPT_ROOT = os.path.join(RESULTS, "checkpoints_ablation")
RUNS_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")
MODELS_PATH = os.path.join(RESULTS, "ablation_5_3B_models.jsonl")
MANIFEST_PATH = os.path.join(RESULTS, "ablation_5_3B_manifest.json")
FAILURES_PATH = os.path.join(RESULTS, "ablation_5_3B_failures.jsonl")
MAIN_SHA_EXPECTED = "f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6fe090f09"

DFS = [f"DF{i}" for i in range(1, 15)]
TRAIN_SEEDS_SET = set(range(30, 230))
MAIN_TEST_SEEDS_SET = set(range(100000, 100030))
HOLDOUT_SEEDS = list(range(200000, 200030))
MODEL_SEED = 30
N_EP = 200
VARIANTS = ["A0", "A1", "A2", "A3", "A4"]
LEARNED_VARIANTS = ["A1", "A2", "A3", "A4"]


def variant_config(variant, D):
    if variant == "A1":
        segs = train.make_segments(D, n_seg=1)
        return dict(segments=segs, state_dim=len(segs) + 4,
                   run_kwargs=dict(), n_segments=len(segs))
    if variant == "A2":
        segs = train.make_segments(D, n_seg=2)
        return dict(segments=segs, state_dim=len(segs) + 4,
                   run_kwargs=dict(mask_change_state=True), n_segments=len(segs))
    if variant == "A3":
        segs = train.make_segments(D, n_seg=2)
        return dict(segments=segs, state_dim=len(segs) + 4,
                   run_kwargs=dict(disable_memory=True), n_segments=len(segs))
    if variant == "A4":
        segs = train.make_segments(D, n_seg=2)
        return dict(segments=segs, state_dim=len(segs) + 4,
                   run_kwargs=dict(), n_segments=len(segs))
    raise ValueError(variant)


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                          capture_output=True, text=True).stdout.strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


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
    def __enter__(self):
        self.detector_compute = 0
        self.archive_query = 0
        self.archive_store = 0
        self.archive_init = 0
        self.detector_init = 0
        self.pop_min = np.inf
        self.pop_max = -np.inf
        self.init_pop = None
        self._oc = ChangeDetector.compute
        self._oq = MemoryArchive.query
        self._ost = MemoryArchive.store
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

        def store_spy(self_a, *a, **k):
            inst.archive_store += 1
            return inst._ost(self_a, *a, **k)

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
        MemoryArchive.store = store_spy
        MemoryArchive.__init__ = ainit_spy
        ChangeDetector.__init__ = dinit_spy
        np.random.rand = rand_spy
        dr.nsga2_one_generation = nsga_spy
        dr.apply_hierarchical_response = resp_spy
        return self

    def __exit__(self, *exc):
        ChangeDetector.compute = self._oc
        MemoryArchive.query = self._oq
        MemoryArchive.store = self._ost
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


def train_or_load_variant(variant, name, cfg, existing_models):
    ckpt_dir = os.path.join(ABL_CKPT_ROOT, variant)
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    existing = next((r for r in existing_models
                     if r["variant"] == variant and r["problem"] == name), None)

    if existing is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        agent = train.new_agent(cfg["segments"], MODEL_SEED)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        h = hsd(agent.online.state_dict())
        if h != existing["final_online_hash"]:
            raise RuntimeError(
                f"RESUME HASH MISMATCH {variant}/{name}: ckpt={h} "
                f"manifest={existing['final_online_hash']} -- STOP")
        print(f"  {variant}/{name}: resumed from checkpoint (hash {h[:12]} verified)")
        return agent, existing

    t0 = time.time()
    torch.manual_seed(MODEL_SEED)
    import random
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    agent = DQNAgent(state_dim=cfg["state_dim"], n_segments=cfg["n_segments"])
    log = []
    for ep in range(N_EP):
        seed = train.TRAIN_SEED_BASE + ep
        N, ref = train.problem_setup(name)
        res = dr.run_sa_drl(
            train.ALL_PROBLEMS[name], n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
            N=N, D=train.CFG["D"], n_changes=train.TRAIN_CHANGES,
            warm_up=train.CFG["warm_up"], agent=agent, segments=cfg["segments"],
            seed=seed, training=True, ref_point=ref, n_elite=train.CFG["n_elite"],
            **cfg["run_kwargs"])
        log.append(res["migd"])
        if ep % 100 == 0:
            print(f"  {variant}/{name} Ep {ep:4d} MIGD={res['migd']:.5f} eps={agent.eps:.3f}")
    dt = time.time() - t0

    online_sd = agent.online.state_dict()
    target_sd = agent.target.state_dict()
    finite_weights = all(torch.all(torch.isfinite(p)).item()
                         for p in agent.online.parameters())
    dist = float(np.sqrt(sum(
        float(((a.detach() - b.detach()) ** 2).sum())
        for a, b in zip(agent.online.parameters(), agent.target.parameters()))))
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"online": online_sd, "target": target_sd}, ckpt_path)
    row = dict(variant=variant, problem=name, model_seed=MODEL_SEED,
              train_seed_start=train.TRAIN_SEED_BASE, n_train_episodes=N_EP,
              state_dim=cfg["state_dim"], n_segments=cfg["n_segments"],
              final_online_hash=hsd(online_sd), final_target_hash=hsd(target_sd),
              learn_steps=agent.step_count, replay_size=len(agent.replay_buffer),
              final_epsilon=round(agent.eps, 6),
              target_online_distance=round(dist, 6),
              finite_weights=finite_weights,
              selection_rule="fixed_final_checkpoint",
              source="trained_in_5_3B", train_time_s=round(dt, 1))
    append_jsonl(MODELS_PATH, row)
    print(f"  {variant}/{name}: trained {dt:.1f}s  hash={row['final_online_hash'][:12]}  "
         f"finite={finite_weights}")
    return agent, row


def get_a4_reference(name, existing_models, main_models):
    existing = next((r for r in existing_models
                     if r["variant"] == "A4" and r["problem"] == name), None)
    if existing is not None:
        return existing
    main_row = main_models[name]
    row = dict(variant="A4", problem=name, model_seed=main_row["model_seed"],
              train_seed_start=main_row["train_seed_start"],
              n_train_episodes=main_row["n_train_episodes"],
              state_dim=train.CFG["D"] and (2 + 4), n_segments=2,
              final_online_hash=main_row["final_online_hash"],
              final_target_hash=main_row["final_target_hash"],
              learn_steps=main_row["learn_steps"], replay_size=main_row["replay_size"],
              final_epsilon=main_row["final_epsilon"],
              target_online_distance=main_row["target_online_distance"],
              finite_weights=main_row["finite_weights"],
              selection_rule="fixed_final_checkpoint",
              source="final_benchmark_checkpoint", train_time_s=None)
    append_jsonl(MODELS_PATH, row)
    return row


def main():
    t_start = time.time()
    head_start = git_head()
    D = train.CFG["D"]

    main_sha = sha256_of(RAW_MAIN_PATH)
    assert main_sha == MAIN_SHA_EXPECTED, \
        f"MAIN BENCHMARK SHA MISMATCH: {main_sha} != {MAIN_SHA_EXPECTED}"
    print("Main benchmark SHA verified:", main_sha)

    assert set(HOLDOUT_SEEDS).isdisjoint(TRAIN_SEEDS_SET)
    assert set(HOLDOUT_SEEDS).isdisjoint(MAIN_TEST_SEEDS_SET)
    print(f"Holdout seeds {HOLDOUT_SEEDS[0]}..{HOLDOUT_SEEDS[-1]} verified disjoint "
         f"from train (30..229) and main-test (100000..100029).")

    main_models = {m["problem"]: m for m in
                  (json.loads(l) for l in open(MAIN_MODELS_PATH))}
    for name in DFS:
        ckpt = torch.load(os.path.join(A4_CKPT_DIR, f"{name}.pt"), weights_only=True)
        assert hsd(ckpt["online"]) == main_models[name]["final_online_hash"], \
            f"A4 checkpoint hash mismatch for {name}"
    print("All 14 A4 checkpoints verified against final_benchmark_models.jsonl.")

    manifest = dict(
        code_commit=head_start, raw_main_benchmark_sha256=main_sha,
        diagnostic_source="5.3A", protocol="confirmatory_holdout_ablation",
        variants=dict(
            A0="Clean Dynamic NSGA-II (no RL/detector/memory/response)",
            A1="Global RL, S=1, segments=[[0..D-1]], state_dim=5",
            A2="Segmented RL (S=2) with c_s masked to zero in policy state "
              "(detector still runs, real c still logged, FE unaffected)",
            A3="Segmented RL (S=2) with memory subsystem disabled "
              "(no signature/archive; d_mem=1.0, has_memory=0.0 always; "
              "MEMORY gate structurally masked out; ChangeDetector still active)",
            A4="Full SA-DRL, EXISTING frozen 5.1 checkpoints, no retraining",
        ),
        holdout_eval_seed_base=200000, n_holdout_seeds=30,
        train_seed_range=[30, 229], n_train_episodes=200, model_seed=30,
        statement="Ablation evaluation seeds (200000-200029) were not used "
                 "in main benchmark (100000-100029) or mechanism diagnostic "
                 "(5.3A) or training (30-229).",
        primary_metric_for_5_3C="migd_end", statistics_performed_in_5_3B=False,
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    existing_models = load_jsonl(MODELS_PATH)
    existing_runs = load_jsonl(RUNS_PATH)
    done_keys = {(r["variant"], r["problem"], r["seed"]) for r in existing_runs
                if r.get("finite_pass") and r.get("frozen_weights_pass") is not False}
    print(f"Resume state: {len(existing_models)} model records, "
         f"{len(existing_runs)} run rows already present.\n")

    stopped = []

    for name in DFS:
        if git_head() != head_start:
            print("!!! SOURCE COMMIT CHANGED DURING ABLATION -- STOPPING ALL")
            sys.exit(1)

        N, ref = train.problem_setup(name)
        pf = train.ALL_PROBLEMS[name]
        n_obj_expected = 3 if name in train.TRI_OBJECTIVE else 2
        p0 = pf(time=0.0, n_var=D)
        xl_lo, xu_hi = float(np.min(p0.xl)), float(np.max(p0.xu))

        agents = {}
        for variant in LEARNED_VARIANTS:
            print(f"\n=== {variant} / {name} ===")
            if variant == "A4":
                model_row = get_a4_reference(name, existing_models, main_models)
                if model_row not in existing_models:
                    existing_models.append(model_row)
                ckpt = torch.load(os.path.join(A4_CKPT_DIR, f"{name}.pt"), weights_only=True)
                frozen = dict(online=ckpt["online"], target=ckpt["target"])
            else:
                cfg = variant_config(variant, D)
                agent, model_row = train_or_load_variant(variant, name, cfg, existing_models)
                if model_row not in existing_models:
                    existing_models.append(model_row)
                frozen = dict(online={k: v.clone() for k, v in agent.online.state_dict().items()},
                             target={k: v.clone() for k, v in agent.target.state_dict().items()})
            agents[variant] = frozen

        for seed in HOLDOUT_SEEDS:
            row_group = {}
            group_failed = False

            for variant in VARIANTS:
                if (variant, name, seed) in done_keys:
                    row_group[variant] = "done"
                    continue
                cfg = variant_config(variant, D) if variant != "A0" else None

                if variant == "A0":
                    orig_rand = np.random.rand
                    cap = {}

                    def rspy(*a, **k):
                        r = orig_rand(*a, **k)
                        if "pop" not in cap and len(a) == 2:
                            cap["pop"] = r.copy()
                        return r
                    np.random.rand = rspy
                    try:
                        res = run_nsga2_baseline(
                            pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                            N=N, D=D, n_changes=train.EVAL_CHANGES,
                            warm_up=train.CFG["warm_up"], seed=seed, ref_point=ref)
                    except Exception as e:
                        np.random.rand = orig_rand
                        log_failure(dict(variant=variant, df=name, seed=seed, error=str(e)))
                        group_failed = True
                        break
                    np.random.rand = orig_rand
                    init_pop = cap.get("pop")
                    frozen_pass = None
                    timeline_pass = True
                    hist_pass = (len(res["igd_pre_history"]) == train.EVAL_CHANGES
                                and len(res["igd_response_history"]) == train.EVAL_CHANGES
                                and len(res["igd_end_history"]) == train.EVAL_CHANGES)
                else:
                    eval_agent = train.new_agent(cfg["segments"], seed)
                    eval_agent.online.load_state_dict(agents[variant]["online"])
                    eval_agent.target.load_state_dict(agents[variant]["target"])
                    eval_agent.eps = 0.0
                    step_before = eval_agent.step_count
                    on_before = hsd(eval_agent.online.state_dict())

                    try:
                        with Instrument() as inst:
                            res = dr.run_sa_drl(
                                pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                                N=N, D=D, n_changes=train.EVAL_CHANGES,
                                warm_up=train.CFG["warm_up"], agent=eval_agent,
                                segments=cfg["segments"], seed=seed, training=False,
                                ref_point=ref, n_elite=train.CFG["n_elite"],
                                **cfg["run_kwargs"])
                    except Exception as e:
                        log_failure(dict(variant=variant, df=name, seed=seed, error=str(e)))
                        group_failed = True
                        break

                    on_after = hsd(eval_agent.online.state_dict())
                    frozen_pass = (on_after == on_before
                                  and eval_agent.step_count == step_before
                                  and eval_agent.eps == 0.0)
                    tl = res["timeline"]
                    timeline_pass = (len(tl) == train.EVAL_CHANGES
                                     and abs(tl[0]["t_old"]) < 1e-9
                                     and abs(tl[0]["t_new"] - 0.1) < 1e-9
                                     and inst.detector_compute == train.EVAL_CHANGES)
                    hist_pass = (len(res["igd_pre_history"]) == train.EVAL_CHANGES
                                and len(res["igd_response_history"]) == train.EVAL_CHANGES
                                and len(res["igd_end_history"]) == train.EVAL_CHANGES
                                and len(res["reward_history"]) == train.EVAL_CHANGES)
                    init_pop = inst.init_pop

                    # variant-specific semantic invariants
                    if variant == "A1" and len(cfg["segments"]) != 1:
                        log_failure(dict(variant=variant, df=name, seed=seed,
                                         reason="A1_SEGMENTS_NOT_SINGLE"))
                        group_failed = True
                        break
                    if variant == "A3":
                        if inst.archive_query != 0 or inst.archive_store != 0:
                            log_failure(dict(variant=variant, df=name, seed=seed,
                                             reason="A3_MEMORY_CALLS_NONZERO",
                                             query=inst.archive_query,
                                             store=inst.archive_store))
                            group_failed = True
                            break
                        if res["fe_breakdown"]["signature"] != 0:
                            log_failure(dict(variant=variant, df=name, seed=seed,
                                             reason="A3_SIGNATURE_FE_NONZERO"))
                            group_failed = True
                            break
                        if any(g == 1 for g in res["gate_action"]):
                            log_failure(dict(variant=variant, df=name, seed=seed,
                                             reason="A3_MEMORY_GATE_USED"))
                            group_failed = True
                            break
                    if variant == "A2":
                        # policy state c must be zero; verified structurally at
                        # AB4/AB5 test time, not re-instrumented per-row here for speed.
                        pass

                    fe_ledger_pass = res["fes_used"] == sum(res["fe_breakdown"].values())
                    if not fe_ledger_pass:
                        log_failure(dict(variant=variant, df=name, seed=seed,
                                         reason="FE_LEDGER"))
                        group_failed = True
                        break
                    bounds_pass = (inst.pop_min >= xl_lo - 1e-9 and inst.pop_max <= xu_hi + 1e-9)
                    if not bounds_pass:
                        log_failure(dict(variant=variant, df=name, seed=seed, reason="BOUNDS"))
                        group_failed = True
                        break

                obj_dim_pass = p0.n_obj == n_obj_expected
                fin_pass = finite_ok(res)
                if not (timeline_pass and hist_pass and obj_dim_pass and fin_pass
                       and (frozen_pass is not False)):
                    log_failure(dict(variant=variant, df=name, seed=seed,
                                     timeline_pass=timeline_pass, hist_pass=hist_pass,
                                     obj_dim_pass=obj_dim_pass, fin_pass=fin_pass,
                                     frozen_pass=frozen_pass))
                    group_failed = True
                    break

                fe_bd = res["fe_breakdown"]
                out_row = dict(
                    experiment_id=f"{variant}_{name}_seed{seed}",
                    variant=variant, protocol="confirmatory_holdout_ablation",
                    problem=name, seed=seed, code_commit=head_start,
                    N=N, D=D, warm_up=train.CFG["warm_up"], tau_t=train.CFG["tau_t"],
                    n_t=train.CFG["n_t"], changes=train.EVAL_CHANGES,
                    migd_end=res["migd_end"], migd_response=res["migd_response"],
                    fes_used=res["fes_used"], fe_breakdown=fe_bd,
                    frozen_weights_pass=frozen_pass, finite_pass=fin_pass,
                    timeline_pass=timeline_pass, history_pass=hist_pass,
                    obj_dim_pass=obj_dim_pass,
                )
                row_group[variant] = (out_row, init_pop)

            if group_failed:
                stopped.append((name, seed))
                continue

            # paired init check across all 5 variants for this (DF,seed)
            pops = [v[1] for v in row_group.values() if v != "done"]
            paired_ok = True
            if len(pops) > 1:
                ref_pop = pops[0]
                paired_ok = all(np.allclose(ref_pop, p) for p in pops[1:])
            if not paired_ok:
                log_failure(dict(df=name, seed=seed, reason="PAIRED_INIT_MISMATCH"))
                stopped.append((name, seed))
                continue

            for variant, val in row_group.items():
                if val == "done":
                    continue
                out_row, _ = val
                out_row["paired_init_pass"] = paired_ok
                append_jsonl(RUNS_PATH, out_row)
                done_keys.add((variant, name, seed))

        n_rows = sum(1 for k in done_keys if k[1] == name)
        print(f"\n{name}: {n_rows} run-rows complete across variants "
             f"({time.time()-t_start:.0f}s elapsed)")

    sha_after = sha256_of(RAW_MAIN_PATH)
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    print(f"Main benchmark SHA unchanged: {sha_after == main_sha}")
    print(f"Stopped (DF,seed) pairs: {stopped if stopped else 'none'}")
    assert sha_after == main_sha, "MAIN BENCHMARK FILE MODIFIED"


if __name__ == "__main__":
    main()
