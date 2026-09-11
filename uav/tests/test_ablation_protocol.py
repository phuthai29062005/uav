"""AB1-AB18: 5.3B ablation variant protocol invariants (pre-run gate)."""
import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import dynamic_runner as dr  # noqa: E402
import train  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402
from change_detector import ChangeDetector  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from memory_archive import MemoryArchive, compute_signature  # noqa: E402
import memory_archive as ma  # noqa: E402
from pymoo.problems.dynamic.df import DF1  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
D = 10


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def short_run(segments, seed=0, K=6, W=5, T=3, N=20, training=True,
             mask_change_state=False, disable_memory=False, eps_fixed=0.5):
    torch.manual_seed(seed)
    ag = DQNAgent(state_dim=len(segments) + 4, n_segments=len(segments),
                 eps_fixed=eps_fixed)
    res = dr.run_sa_drl(DF1, n_t=10, tau_t=T, N=N, D=D, n_changes=K,
                        warm_up=W, agent=ag, segments=segments, seed=seed,
                        training=training, ref_point=(2.0, 2.0), n_elite=5,
                        mask_change_state=mask_change_state,
                        disable_memory=disable_memory)
    return ag, res


# ------------------------------------------------------- AB1: seed disjoint
def test_ab1_holdout_seeds_disjoint():
    train_seeds = set(range(30, 230))
    main_test_seeds = set(range(100000, 100030))
    holdout_seeds = set(range(200000, 200030))
    assert holdout_seeds.isdisjoint(train_seeds)
    assert holdout_seeds.isdisjoint(main_test_seeds)
    assert len(holdout_seeds) == 30


# --------------------------------------------------- AB2/AB3: A1 semantics
def test_ab2_a1_single_full_d_block():
    segs = train.make_segments(D, n_seg=1)
    assert segs == [list(range(D))]
    assert len(segs) == 1


def test_ab3_a1_state_dim_five():
    segs = train.make_segments(D, n_seg=1)
    ag = DQNAgent(state_dim=len(segs) + 4, n_segments=len(segs))
    assert ag.online.trunk[0].in_features == 5
    x = torch.rand(3, 5)
    q_gate, adv = ag.online(x)
    assert q_gate.shape == (3, 2) and adv.shape == (3, 1, 4)


# ------------------------------------------------ AB4/AB5/AB6: A2 semantics
def test_ab4_a2_state_c_masked_but_detector_c_real():
    seen_states = []
    orig_bs = dr.build_state

    def spy(*a, **k):
        st = orig_bs(*a, **k)
        seen_states.append(st.copy())
        return st
    dr.build_state = spy
    try:
        segs = [[0], list(range(1, D))]
        ag, res = short_run(segs, seed=0, K=6, mask_change_state=True)
    finally:
        dr.build_state = orig_bs

    assert len(seen_states) == 6
    for st in seen_states:
        assert st[0] == 0.0 and st[1] == 0.0
    real_c = [c for row in res["normalized_change"] for c in row]
    assert any(abs(v) > 1e-9 for v in real_c), "detector c should still vary/be logged"


def test_ab5_a2_replay_states_have_zero_c():
    segs = [[0], list(range(1, D))]
    ag, res = short_run(segs, seed=1, K=6, training=True, mask_change_state=True)
    assert len(ag.replay_buffer) > 0
    for tr in ag.replay_buffer.buffer:
        state, next_state = tr[0], tr[4]
        assert state[0] == 0.0 and state[1] == 0.0
        # terminal next_state is all-zero structurally; non-terminal next_state
        # is a real state and must also have masked c.
        assert next_state[0] == 0.0 and next_state[1] == 0.0


def test_ab6_a2_detector_fe_still_counted():
    segs = [[0], list(range(1, D))]
    ag, res = short_run(segs, seed=2, K=6, mask_change_state=True)
    assert res["fe_breakdown"]["detector"] > 0


# ------------------------------------------ AB7-AB11: A3 semantics
def test_ab7_a3_dmem_one_hasmemory_zero():
    segs = [[0], list(range(1, D))]
    ag, res = short_run(segs, seed=3, K=6, disable_memory=True)
    assert all(v == 1.0 for v in res["d_mem"])
    assert all(v == 0.0 or v is False for v in res["has_memory"])


def test_ab8_a3_memory_gate_impossible():
    segs = [[0], list(range(1, D))]
    for training in (True, False):
        ag, res = short_run(segs, seed=4, K=6, training=training,
                            disable_memory=True, eps_fixed=1.0)
        assert all(g == 0 for g in res["gate_action"])


def test_ab9_a3_zero_query_store_signature_calls():
    calls = {"query": 0, "store": 0, "sig": 0}
    oq, ost, osig = MemoryArchive.query, MemoryArchive.store, ma.compute_signature

    def qspy(self, *a, **k):
        calls["query"] += 1
        return oq(self, *a, **k)

    def sspy(self, *a, **k):
        calls["store"] += 1
        return ost(self, *a, **k)

    def sigspy(*a, **k):
        calls["sig"] += 1
        return osig(*a, **k)

    MemoryArchive.query = qspy
    MemoryArchive.store = sspy
    dr.compute_signature = sigspy
    try:
        segs = [[0], list(range(1, D))]
        short_run(segs, seed=5, K=6, disable_memory=True)
    finally:
        MemoryArchive.query = oq
        MemoryArchive.store = ost
        dr.compute_signature = osig
    assert calls == {"query": 0, "store": 0, "sig": 0}


def test_ab10_a3_signature_fe_zero():
    segs = [[0], list(range(1, D))]
    ag, res = short_run(segs, seed=6, K=6, disable_memory=True)
    assert res["fe_breakdown"]["signature"] == 0


def test_ab11_a3_detector_still_active():
    calls = {"n": 0}
    oc = ChangeDetector.compute

    def spy(self, *a, **k):
        calls["n"] += 1
        return oc(self, *a, **k)
    ChangeDetector.compute = spy
    try:
        segs = [[0], list(range(1, D))]
        ag, res = short_run(segs, seed=7, K=6, disable_memory=True)
    finally:
        ChangeDetector.compute = oc
    assert calls["n"] == 6


# --------------------------------------------- AB12/AB13: A4 regression guard
@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen benchmark not present")
def test_ab12_default_full_behavior_unchanged():
    orig_rows = {(r["problem"], r["seed"]): r for r in
                (json.loads(l) for l in open(RAW_PATH)) if r["method"] == "SA-DRL-frozen"}
    models = {m["problem"]: m for m in
             (json.loads(l) for l in open(os.path.join(RESULTS, "final_benchmark_models.jsonl")))}
    seg = train.make_segments(train.CFG["D"])
    for name, seed in [("DF1", 100000), ("DF4", 100001)]:
        ckpt = torch.load(os.path.join(RESULTS, "checkpoints", f"{name}.pt"),
                          weights_only=True)
        ag = train.new_agent(seg, seed)
        ag.online.load_state_dict(ckpt["online"])
        ag.target.load_state_dict(ckpt["target"])
        ag.eps = 0.0
        N, ref = train.problem_setup(name)
        res = dr.run_sa_drl(train.ALL_PROBLEMS[name], n_t=10, tau_t=10, N=N, D=10,
                            n_changes=100, warm_up=50, agent=ag, segments=seg,
                            seed=seed, training=False, ref_point=ref, n_elite=10)
        orig = orig_rows[(name, seed)]
        assert abs(res["migd_end"] - orig["migd_end"]) < 1e-9
        assert abs(res["migd_response"] - orig["migd_response"]) < 1e-9
        assert res["fes_used"] == orig["fes_used"]


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen benchmark not present")
def test_ab13_a4_checkpoint_hash_matches_manifest():
    models = [json.loads(l) for l in
             open(os.path.join(RESULTS, "final_benchmark_models.jsonl"))]
    for m in models:
        ckpt = torch.load(os.path.join(RESULTS, "checkpoints", f"{m['problem']}.pt"),
                          weights_only=True)
        assert hsd(ckpt["online"]) == m["final_online_hash"]


# ---------------------------------------------- AB14: paired init across variants
def test_ab14_paired_init_across_variants():
    def init_pop(seed):
        np.random.seed(seed)
        import random
        random.seed(seed)
        from nsga2_pymoo import seed_nsga2
        seed_nsga2(seed)
        p = DF1(time=0.0, n_var=D)
        return p.xl + np.random.rand(20, D) * (p.xu - p.xl)
    a = init_pop(200000)
    b = init_pop(200000)
    assert np.allclose(a, b)


# --------------------------------------------- AB15: frozen eval params
def test_ab15_frozen_eval_params_unchanged_across_variants():
    for segs, kw in [([[0], list(range(1, D))], {}),
                     ([list(range(D))], {}),
                     ([[0], list(range(1, D))], dict(mask_change_state=True)),
                     ([[0], list(range(1, D))], dict(disable_memory=True))]:
        torch.manual_seed(0)
        ag = DQNAgent(state_dim=len(segs) + 4, n_segments=len(segs))
        ag.eps = 0.0
        on_before = copy.deepcopy(ag.online.state_dict())
        tg_before = copy.deepcopy(ag.target.state_dict())
        dr.run_sa_drl(DF1, n_t=10, tau_t=3, N=20, D=D, n_changes=6, warm_up=5,
                      agent=ag, segments=segs, seed=0, training=False,
                      ref_point=(2.0, 2.0), n_elite=5, **kw)
        assert all(torch.equal(on_before[k], ag.online.state_dict()[k]) for k in on_before)
        assert all(torch.equal(tg_before[k], ag.target.state_dict()[k]) for k in tg_before)
        assert ag.eps == 0.0


# ---------------------------------------------------- AB16: unique keys
@pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")),
                    reason="ablation data not yet collected")
def test_ab16_ablation_output_unique_keys():
    rows = [json.loads(l) for l in
           open(os.path.join(RESULTS, "ablation_5_3B_runs.jsonl"))]
    keys = [(r["variant"], r["problem"], r["seed"]) for r in rows]
    assert len(keys) == len(set(keys))


# ------------------------------------------------ AB17: raw SHA unchanged
@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen benchmark not present")
def test_ab17_raw_main_benchmark_sha_unchanged():
    assert sha256_of(RAW_PATH) == \
        "f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6fe090f09"


# --------------------------------------------- AB18: stabilized DF mapping
def test_ab18_stable_df_mapping_retained():
    for n in ["DF5", "DF12", "DF13"]:
        assert train.ALL_PROBLEMS[n].__name__ == f"{n}Stable"
