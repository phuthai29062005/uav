"""
5.5A -- Training-horizon bottleneck diagnostic (2x2 horizon x transition-
budget). EXPLORATORY MECHANISM DIAGNOSIS on 5 pre-selected DFs
(DF1, DF7, DF10, DF11, DF12) -- NOT a replacement confirmatory benchmark.

=====================================================================
PHASE 1 AUDIT (answered here; grounds the implementation below)
=====================================================================

Q1. Is TRAIN_CHANGES configurable?
    NO at the train.train_on()/mode_frozen() level: train_on() calls
    one_run(..., n_changes=TRAIN_CHANGES) with the module-level constant
    hardcoded, not passed through as a parameter. HOWEVER the underlying
    primitive train.one_run() (and run_sa_drl() beneath it) already
    accepts n_changes as an explicit argument. Implementing the 2x2
    design therefore requires a thin wrapper that calls train.one_run()
    directly in a loop (see train_horizon_cell() below) -- NOT a
    modification of train.py, dynamic_runner.py, or any production
    algorithm file.

Q2. Fresh problem/population/archive/detector per episode?
    YES. run_sa_drl() creates `archive = MemoryArchive()`, a fresh
    `detector = ChangeDetector(...)`, and `pop = xl + rand(...)*(...)`
    seeded via `np.random.seed(seed)` at the TOP of every call. Every
    episode is one independent call to run_sa_drl() with its own seed
    (seed=seed+ep in train_on()/our wrapper), so all of these are fresh
    every episode.

Q3. Does the DQN/replay buffer persist across training episodes?
    YES. train_on() (and our wrapper) creates ONE `agent` object before
    the episode loop and reuses it for every episode -- only the
    environment-side objects (archive/detector/pop/problem) reset each
    episode; agent.online/target/replay_buffer/step_count/eps persist
    and accumulate across the whole training run.

Q4. For TRAIN_CHANGES=30, episodes=200: exact replay transitions?
    Confirmed exactly 6000 (30*200), matching
    results/final_benchmark_models.jsonl (replay_size=6000,
    learn_steps=5937 for every DF). See Q5/Q6 for why 6000 transitions
    yields exactly 5937 learn steps.

Q5/Q6. Terminal transition once per episode; K changes -> K transitions?
    YES, confirmed by direct trace of dynamic_runner.run_sa_drl(): the
    `is_change` block pushes one non-terminal transition per completed
    prior action for change_count=2..K (K-1 pushes), plus exactly one
    terminal push after the main loop (guarded by
    `if pending_state is not None`, always true for K>=1). Total = K.
    ReplayBuffer.push()+learn() gate: DQNAgent.learn() is a no-op while
    len(replay_buffer) < batch_size(64) (src/dqn_agent.py:146). Since the
    buffer persists across the WHOLE training run (Q3) and is a
    deque(maxlen=10000) that never resets, this 64-sample warmup happens
    ONCE at the very start of training regardless of episode boundaries:
    push #1..#63 are skipped (len<64), push #64 onward all trigger
    learn(). Predicted learn_steps = total_transitions - 63 for ANY
    episode/n_changes split that sums to the same total, as long as
    total_transitions <= buffer capacity keeps len>=64 monotonically
    (true once buffer has filled past 64, regardless of later eviction
    -- see Q7b). This predicts:
        B6K  (6000 transitions)  -> learn_steps = 5937  (matches
             the existing frozen A4 manifest exactly)
        B18K (18000 transitions) -> learn_steps = 17937
    for BOTH cells at a given budget, independent of how that budget is
    split into episodes/changes-per-episode. Verified empirically after
    training (see report).

Q7. Archive lifecycle: within-episode only, no cross-episode leak?
    YES. `archive = MemoryArchive()` is a local variable created fresh
    inside every run_sa_drl() call (one call = one episode in training);
    it is never module-level/global state, so it cannot leak into the
    next episode's call. Within one episode it correctly accumulates
    query/store calls across that episode's own changes.

    Q7b (buffer-capacity confound, flagged not hidden): ReplayBuffer's
    capacity is 10000 (src/dqn_agent.py, DQNAgent default buffer_size=
    10000), UNCHANGED per instruction ("keep EXACTLY current ... replay
    capacity"). For the B6K cells (6000 transitions) the buffer never
    fills, so every training transition remains available for sampling
    throughout. For the B18K cells (18000 transitions > 10000 capacity),
    the deque(maxlen=10000) evicts the oldest transitions once full: by
    the end of training only the most recent ~10000 of the 18000 pushed
    transitions are still in the buffer. This does not change the
    predicted learn_steps formula (Q5/Q6 -- len stays >=64 throughout
    once past warmup, eviction or not), but it IS a genuine difference
    in what data is available to later learn() calls between B6K and
    B18K cells, on top of the pre-registered warmup/reset confound in
    Phase 6. Reported, not concealed.

Q8. Does a TRAIN_CHANGES=100 episode expose genuinely different
    late-horizon states (bigger/older archive, late detector/signature
    state, environment recurrence past change 30)?
    Archive/detector state: YES structurally -- both persist WITHIN one
    episode (Q7), so by change 61-100 of a single H100 episode the
    archive holds far more accumulated (signature, pop) entries than any
    30-change episode ever reaches (max ~29 stores). This exposure never
    occurs in any H30 cell, regardless of how many H30 episodes are run,
    because each H30 episode's archive is discarded and rebuilt from
    empty at every reset.
    Environment recurrence: NOT assumed -- audited directly from the
    actual pymoo DF source (t = change_index/n_t, n_t=10):
      DF1 : v=sin(0.5*pi*t)          -> period_t=4  -> period_k=40 changes
      DF7 : a=5*cos(0.5*pi*t)        -> period_t=4  -> period_k=40 changes
      DF10: G=sin(0.5*pi*t), H=cos.. -> period_t=4  -> period_k=40 changes
      DF11: G=|sin(0.5*pi*t)|        -> period_t=2  -> period_k=20 changes
              (abs() halves the period vs DF1/7/10)
      DF12(Stable): k(t)=10*snap(sin(pi*t)) -> period_t=2 -> period_k=20
              changes for the discontinuous g-component; but g also has
              a sin(time*x0) term whose effective frequency depends on
              the decision variable x0 itself, so that component has NO
              fixed period independent of the population -- "recurrence"
              for DF12 is partial/component-specific, not a clean whole-
              problem period.
    Consequence: for DF1/DF7/DF10 (period 40), an H30 episode (k<=30)
    NEVER completes one full period (t only reaches 3.0 of a 4.0-period
    cycle) -- no recurrence is possible within H30 for these three. An
    H100 episode (k<=100) covers 2.5 periods, so genuine recurrence
    occurs. For DF11 and DF12 (period 20), recurrence already begins
    WITHIN a single H30 episode (k=1..30 already covers 1.5 periods) --
    H100 for these two DFs adds MORE repeated cycles (5 periods by
    k=100) rather than introducing recurrence for the first time.
    This is DF-specific and reported per-DF, not generalized.

Additional confound found during this audit, not explicitly named by the
original prompt but material to interpretation (Phase 4/16):
    epsilon decays ONCE PER EPISODE (agent.decay_epsilon() is called once
    at the end of every training run_sa_drl() call when training=True),
    not once per transition. Because the four cells use different
    n_episodes (200/60/600/180) to hit their target transition budgets,
    their FINAL training-time epsilon differs sharply even though the
    epsilon schedule FORMULA (eps_end=0.01, eps_decay=0.995 per episode)
    is kept perfectly unchanged, as required:
        H30_B6K  (200 ep) -> eps_final ~= 0.367 (matches frozen A4 exactly)
        H100_B6K ( 60 ep) -> eps_final ~= 0.740
        H30_B18K (600 ep) -> eps_final ~= 0.049
        H100_B18K(180 ep) -> eps_final ~= 0.406
    This means the transition-budget-matched pairs (T0 vs T1, T2 vs T3)
    are NOT matched on how much of their training was still highly
    exploratory -- H100_B6K's final policy comes from a training run
    that was still ~74% exploration-annealed rather than ~37%. This is
    an unavoidable, honestly-reported consequence of the pre-registered
    episodic protocol (identical to the pre-registered warmup/reset
    confound in Phase 6), NOT a hyperparameter change -- eps_decay,
    eps_start, eps_end are untouched. It must be weighed when
    interpreting C1/C4 (both involve H100_B6K).

CONCLUSION: no material deviation requiring a STOP was found. Two
additional confounds (buffer eviction at 18K; epsilon-schedule
divergence by episode count) are flagged above and carried into the
final report's limitations, on top of the pre-registered
warmup/reset-FE confound.
"""
import hashlib
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import dynamic_runner as dr  # noqa: E402
import train  # noqa: E402
from baseline_runner import run_nsga2_baseline  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from policy_diagnostic import (decode_gate, decode_seg,  # noqa: E402
                               intervention_rate, normalized_entropy)

RESULTS = os.path.join(_HERE, "..", "results")
MAIN_RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
MAIN_MODELS_PATH = os.path.join(RESULTS, "final_benchmark_models.jsonl")
ABLATION_RAW_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")
A4_CKPT_DIR = os.path.join(RESULTS, "checkpoints")
HORIZON_CKPT_ROOT = os.path.join(RESULTS, "checkpoints_horizon_5_5A")

RUNS_PATH = os.path.join(RESULTS, "horizon_5_5A_runs.jsonl")
MODELS_PATH = os.path.join(RESULTS, "horizon_5_5A_models.jsonl")
MANIFEST_PATH = os.path.join(RESULTS, "horizon_5_5A_manifest.json")
FAILURES_PATH = os.path.join(RESULTS, "horizon_5_5A_failures.jsonl")
TRAINING_WINDOWS_PATH = os.path.join(RESULTS, "horizon_5_5A_training_windows.csv")
TRAINING_WINDOWS_JSONL = os.path.join(RESULTS, "horizon_5_5A_training_windows.jsonl")

MAIN_SHA_EXPECTED = ("f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6"
                    "fe090f09")
ABLATION_SHA_EXPECTED = ("7191ee747155528a3d31ab7ec8ad8b7a87d499abb80bc5778c2"
                        "9c22eab138eff")

DFS = ["DF1", "DF7", "DF10", "DF11", "DF12"]
MODEL_SEED = 30

HORIZON_EVAL_SEED_BASE = 300000
N_EVAL = 30
EVAL_SEEDS = list(range(HORIZON_EVAL_SEED_BASE, HORIZON_EVAL_SEED_BASE + N_EVAL))

# ---- the 2x2 design (Phase 3) ----
# name -> (n_changes, n_episodes, source)
CELLS = {
    "H30_B6K":   dict(n_changes=30,  n_episodes=200, source="frozen_A4_reference"),
    "H100_B6K":  dict(n_changes=100, n_episodes=60,  source="newly_trained"),
    "H30_B18K":  dict(n_changes=30,  n_episodes=600, source="newly_trained"),
    "H100_B18K": dict(n_changes=100, n_episodes=180, source="newly_trained"),
}
VARIANTS = ["A0"] + list(CELLS.keys())
LEARNED_CELLS = ["H100_B6K", "H30_B18K", "H100_B18K"]  # newly trained

for _name, _cfg in CELLS.items():
    assert _cfg["n_changes"] * _cfg["n_episodes"] == (
        6000 if "B6K" in _name else 18000), _name


def training_seed_range(n_episodes):
    return list(range(MODEL_SEED, MODEL_SEED + n_episodes))


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    """Convert numpy scalar types to native Python types (proper fix --
    NOT the default=str fallback used by earlier scripts, which would
    silently turn e.g. numpy.bool_(True) into the STRING "True" rather
    than JSON boolean true)."""
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def log_failure(d):
    append_jsonl(FAILURES_PATH, d)
    print(f"  !! FAILURE logged: {d}")


def git_head():
    import subprocess
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.join(_HERE, "..")
    ).decode().strip()


def window_55a(change_index, n_changes):
    """EARLY=1..30, MID=31..60, LATE=61..100 (Phase 9). For H30 cells
    (n_changes=30) change_index never exceeds 30 -> only EARLY appears,
    naturally, with no special-casing (TH18)."""
    if 1 <= change_index <= 30:
        return "EARLY"
    if 31 <= change_index <= 60:
        return "MID"
    if 61 <= change_index <= 100:
        return "LATE"
    return "OTHER"


# =====================================================================
# training with window-behavior logging (Phase 9), via monkeypatch --
# NO modification to dynamic_runner.py / sa_drl_dmoea.py.
# =====================================================================
class _WindowLogger:
    """Captures dispersion/hv_drop per change by wrapping the two
    functions dynamic_runner.py calls by name (compute_dispersion,
    compute_hv_drop), without touching that frozen file. Restores the
    originals in __exit__ unconditionally."""

    def __enter__(self):
        self.dispersions = []
        self.hv_drops = []
        self._orig_disp = dr.compute_dispersion
        self._orig_hvdrop = dr.compute_hv_drop
        outer = self

        def disp_spy(*a, **k):
            v = outer._orig_disp(*a, **k)
            outer.dispersions.append(float(v))
            return v

        def hvdrop_spy(*a, **k):
            v = outer._orig_hvdrop(*a, **k)
            outer.hv_drops.append(float(v))
            return v

        dr.compute_dispersion = disp_spy
        dr.compute_hv_drop = hvdrop_spy
        return self

    def __exit__(self, *exc):
        dr.compute_dispersion = self._orig_disp
        dr.compute_hv_drop = self._orig_hvdrop


def train_horizon_cell(name_df, segments, n_changes, n_episodes, window_rows):
    """Mirrors train.train_on() exactly (one agent, episode loop, seed=
    MODEL_SEED+ep, training=True) but takes n_changes as an explicit
    parameter instead of the module constant TRAIN_CHANGES (Q1), and
    additionally logs per-change window behavior for Phase 9/24 plus the
    full training FE ledger for Phase 6 (returned as fe_totals).
    window_rows: list to append raw per-change dicts into (aggregated
    by the caller into results/horizon_5_5A_training_windows.csv).
    """
    agent = train.new_agent(segments, MODEL_SEED)
    fe_totals = {"initial": 0, "nsga2": 0, "detector": 0,
                "signature": 0, "response": 0}
    for ep in range(n_episodes):
        seed = MODEL_SEED + ep
        with _WindowLogger() as wl:
            res = train.one_run(name_df, agent, segments, seed,
                                training=True, n_changes=n_changes)
        for k in fe_totals:
            fe_totals[k] += res["fe_breakdown"][k]
        gate_hist = res["gate_action"]
        seg_hist = res["seg_actions"]
        c_hist = res["normalized_change"]
        dmem_hist = res["d_mem"]
        hasmem_hist = res["has_memory"]
        n = len(gate_hist)
        assert n == n_changes, (n, n_changes)
        assert len(wl.dispersions) == n_changes and len(wl.hv_drops) == n_changes
        for i in range(n):
            window_rows.append(dict(
                window=window_55a(i + 1, n_changes),
                gate=int(gate_hist[i]),
                seg1=int(seg_hist[i][0]), seg2=int(seg_hist[i][1]),
                c1=float(c_hist[i][0]), c2=float(c_hist[i][1]),
                d_mem=float(dmem_hist[i]), has_memory=float(hasmem_hist[i]),
                dispersion=wl.dispersions[i], hv_drop=wl.hv_drops[i],
            ))
    fe_totals["total"] = sum(fe_totals.values())
    return agent, fe_totals


def aggregate_window_rows(df_name, variant, window_rows):
    """One aggregated row per (DF, variant, window) present."""
    out = []
    windows_present = sorted(set(r["window"] for r in window_rows),
                             key=lambda w: {"EARLY": 0, "MID": 1, "LATE": 2}[w])
    for w in windows_present:
        rows = [r for r in window_rows if r["window"] == w]
        n = len(rows)
        gates = [r["gate"] for r in rows]
        seg1s = [r["seg1"] for r in rows]
        seg2s = [r["seg2"] for r in rows]
        has_mem = [r["has_memory"] for r in rows]
        avail_idx = [i for i, h in enumerate(has_mem) if h > 0.5]
        p_mem_given_avail = (float(np.mean([gates[i] for i in avail_idx]))
                             if avail_idx else None)   # TH19: never /0
        out.append(dict(
            DF=df_name, variant=variant, window=w, n=n,
            memory_pct=float(np.mean(gates)),
            p_memory_given_available=p_mem_given_avail,
            n_available=len(avail_idx),
            seg1_intervention_pct=intervention_rate(seg1s),
            seg2_intervention_pct=intervention_rate(seg2s),
            gate_entropy=normalized_entropy(gates, 2),
            seg1_entropy=normalized_entropy(seg1s, 4),
            seg2_entropy=normalized_entropy(seg2s, 4),
            d_mem_median=float(np.median([r["d_mem"] for r in rows])),
            c1_median=float(np.median([r["c1"] for r in rows])),
            c2_median=float(np.median([r["c2"] for r in rows])),
            hv_drop_median=float(np.median([r["hv_drop"] for r in rows])),
            dispersion_median=float(np.median([r["dispersion"] for r in rows])),
        ))
    return out


# =====================================================================
# resume-safe model training/loading (Phase 12, 21)
# =====================================================================
def train_or_load_cell(cell_name, df_name, segments, existing_models):
    cfg = CELLS[cell_name]
    n_changes, n_episodes = cfg["n_changes"], cfg["n_episodes"]
    ckpt_dir = os.path.join(HORIZON_CKPT_ROOT, cell_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{df_name}.pt")

    existing = next((r for r in existing_models
                     if r["variant"] == cell_name and r["DF"] == df_name), None)
    if existing is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        agent = train.new_agent(segments, MODEL_SEED)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        h = hsd(agent.online.state_dict())
        if (h != existing["model_hash"]
                or existing["horizon"] != n_changes
                or existing["transition_budget"] != n_changes * n_episodes
                or existing["episodes"] != n_episodes
                or existing["model_seed"] != MODEL_SEED
                or existing["DF"] != df_name):
            raise RuntimeError(
                f"RESUME MISMATCH {cell_name}/{df_name}: config or hash "
                f"does not match manifest -- STOP (existing={existing})")
        print(f"  {cell_name}/{df_name}: resumed (hash {h[:12]} verified)")
        return agent, existing

    t0 = time.time()
    window_rows = []
    agent, fe_totals = train_horizon_cell(df_name, segments, n_changes,
                                          n_episodes, window_rows)
    dt = time.time() - t0

    agg = aggregate_window_rows(df_name, cell_name, window_rows)
    for r in agg:
        append_jsonl(TRAINING_WINDOWS_JSONL, r)

    online_sd = agent.online.state_dict()
    target_sd = agent.target.state_dict()
    finite_weights = all(torch.all(torch.isfinite(p)).item()
                         for p in agent.online.parameters())
    torch.save({"online": online_sd, "target": target_sd}, ckpt_path)

    row = dict(
        DF=df_name, variant=cell_name, horizon=n_changes,
        transition_budget=n_changes * n_episodes, episodes=n_episodes,
        model_seed=MODEL_SEED,
        training_seed_range=[MODEL_SEED, MODEL_SEED + n_episodes - 1],
        replay_size=len(agent.replay_buffer),
        number_of_rl_transitions=n_changes * n_episodes,
        number_of_episode_resets=n_episodes,
        learn_steps=agent.step_count,
        final_epsilon=round(agent.eps, 6),
        training_total_FE=fe_totals["total"],
        training_initial_FE=fe_totals["initial"],
        training_nsga2_FE=fe_totals["nsga2"],
        training_detector_FE=fe_totals["detector"],
        training_signature_FE=fe_totals["signature"],
        training_response_FE=fe_totals["response"],
        model_hash=hsd(online_sd), target_hash=hsd(target_sd),
        finite_weights=finite_weights,
        source="newly_trained",
        checkpoint_path=os.path.relpath(ckpt_path, os.path.join(_HERE, "..")),
        train_time_s=round(dt, 1),
    )
    append_jsonl(MODELS_PATH, row)
    print(f"  {cell_name}/{df_name}: trained {dt:.1f}s  "
         f"transitions={row['number_of_rl_transitions']} "
         f"learn_steps={row['learn_steps']} eps_final={row['final_epsilon']} "
         f"hash={row['model_hash'][:12]} finite={finite_weights}")
    return agent, row


def get_frozen_a4_reference(df_name, existing_models, main_models):
    existing = next((r for r in existing_models
                     if r["variant"] == "H30_B6K" and r["DF"] == df_name), None)
    if existing is not None:
        return existing
    main_row = next(r for r in main_models if r["problem"] == df_name)
    row = dict(
        DF=df_name, variant="H30_B6K", horizon=30, transition_budget=6000,
        episodes=200, model_seed=MODEL_SEED,
        training_seed_range=[MODEL_SEED, MODEL_SEED + 199],
        replay_size=main_row["replay_size"],
        number_of_rl_transitions=6000,
        number_of_episode_resets=200,
        learn_steps=main_row["learn_steps"],
        final_epsilon=main_row["final_epsilon"],
        training_total_FE=None, training_initial_FE=None,
        training_nsga2_FE=None, training_detector_FE=None,
        training_signature_FE=None, training_response_FE=None,
        training_fe_note=("not recomputed -- reference to the existing "
                          "frozen 5.1 checkpoint per protocol (T0 is not "
                          "retrained); per-category training FE was not "
                          "persisted by the original 5.1 pipeline"),
        model_hash=main_row["final_online_hash"],
        target_hash=main_row["final_target_hash"],
        finite_weights=main_row["finite_weights"],
        source="frozen_A4_reference",
        checkpoint_path=os.path.relpath(
            os.path.join(A4_CKPT_DIR, f"{df_name}.pt"),
            os.path.join(_HERE, "..")),
        train_time_s=main_row.get("train_time_s"),
    )
    append_jsonl(MODELS_PATH, row)
    return row


# =====================================================================
# frozen evaluation (Phase 7-8)
# =====================================================================
class EvalInstrument:
    def __enter__(self):
        self.detector_compute = 0
        self.init_pop = None
        from change_detector import ChangeDetector
        self._oc = ChangeDetector.compute
        self._orand = np.random.rand
        inst = self

        def compute_spy(self_d, *a, **k):
            inst.detector_compute += 1
            return inst._oc(self_d, *a, **k)

        def rand_spy(*args, **kwargs):
            r = inst._orand(*args, **kwargs)
            if inst.init_pop is None and len(args) == 2:
                inst.init_pop = r.copy()
            return r

        ChangeDetector.compute = compute_spy
        np.random.rand = rand_spy
        return self

    def __exit__(self, *exc):
        from change_detector import ChangeDetector
        ChangeDetector.compute = self._oc
        np.random.rand = self._orand


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


def evaluate_cell(cell_name, df_name, seg, seed, frozen_sd, N, ref,
                  n_obj_expected):
    pf = train.ALL_PROBLEMS[df_name]
    eval_agent = train.new_agent(seg, seed)
    eval_agent.online.load_state_dict(frozen_sd["online"])
    eval_agent.target.load_state_dict(frozen_sd["target"])
    eval_agent.eps = 0.0
    step_before = eval_agent.step_count
    on_before = hsd(eval_agent.online.state_dict())
    tg_before = hsd(eval_agent.target.state_dict())

    with EvalInstrument() as inst:
        res = dr.run_sa_drl(
            pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"], N=N,
            D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
            warm_up=train.CFG["warm_up"], agent=eval_agent, segments=seg,
            seed=seed, training=False, ref_point=ref,
            n_elite=train.CFG["n_elite"])

    on_after = hsd(eval_agent.online.state_dict())
    tg_after = hsd(eval_agent.target.state_dict())
    frozen_pass = (on_after == on_before and tg_after == tg_before
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
    obj_dim_pass = pf(time=0.0, n_var=train.CFG["D"]).n_obj == n_obj_expected
    fin_pass = finite_ok(res)
    fe_pass = res["fes_used"] == sum(res["fe_breakdown"].values())

    row = dict(
        experiment_id=f"5.5A/{cell_name}/{df_name}/{seed}",
        variant=cell_name, protocol="horizon_5_5A", problem=df_name,
        seed=seed, N=N, D=train.CFG["D"], warm_up=train.CFG["warm_up"],
        tau_t=train.CFG["tau_t"], n_t=train.CFG["n_t"],
        changes=train.EVAL_CHANGES,
        migd_end=res["migd_end"], migd_response=res["migd_response"],
        fes_used=res["fes_used"], fe_breakdown=res["fe_breakdown"],
        frozen_weights_pass=frozen_pass, finite_pass=fin_pass,
        timeline_pass=timeline_pass, history_pass=hist_pass,
        obj_dim_pass=obj_dim_pass, fe_ledger_pass=fe_pass,
    )
    ok = (frozen_pass and fin_pass and timeline_pass and hist_pass
         and obj_dim_pass and fe_pass)
    return row, ok, inst.init_pop


def evaluate_a0(df_name, seed, N, ref):
    pf = train.ALL_PROBLEMS[df_name]
    n_obj_expected = 3 if df_name in train.TRI_OBJECTIVE else 2
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
            pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"], N=N,
            D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
            warm_up=train.CFG["warm_up"], seed=seed, ref_point=ref)
    finally:
        np.random.rand = orig_rand

    hist_pass = (len(res["igd_pre_history"]) == train.EVAL_CHANGES
                and len(res["igd_response_history"]) == train.EVAL_CHANGES
                and len(res["igd_end_history"]) == train.EVAL_CHANGES)
    obj_dim_pass = pf(time=0.0, n_var=train.CFG["D"]).n_obj == n_obj_expected
    fin_pass = finite_ok(res)
    fe_pass = res["fes_used"] == sum(res["fe_breakdown"].values())
    row = dict(
        experiment_id=f"5.5A/A0/{df_name}/{seed}",
        variant="A0", protocol="horizon_5_5A", problem=df_name, seed=seed,
        N=N, D=train.CFG["D"], warm_up=train.CFG["warm_up"],
        tau_t=train.CFG["tau_t"], n_t=train.CFG["n_t"],
        changes=train.EVAL_CHANGES,
        migd_end=res["migd_end"], migd_response=res["migd_response"],
        fes_used=res["fes_used"], fe_breakdown=res["fe_breakdown"],
        frozen_weights_pass=None, finite_pass=fin_pass,
        timeline_pass=True, history_pass=hist_pass,
        obj_dim_pass=obj_dim_pass, fe_ledger_pass=fe_pass,
    )
    ok = fin_pass and hist_pass and obj_dim_pass and fe_pass
    return row, ok, cap.get("pop")


def regenerate_training_windows_csv():
    rows = load_jsonl(TRAINING_WINDOWS_JSONL)
    if not rows:
        return
    cols = ["DF", "variant", "window", "n", "memory_pct",
           "p_memory_given_available", "n_available",
           "seg1_intervention_pct", "seg2_intervention_pct",
           "gate_entropy", "seg1_entropy", "seg2_entropy",
           "d_mem_median", "c1_median", "c2_median",
           "hv_drop_median", "dispersion_median"]
    import csv
    with open(TRAINING_WINDOWS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"wrote {TRAINING_WINDOWS_PATH} ({len(rows)} rows)")


# =====================================================================
# main
# =====================================================================
def main():
    t_start = time.time()
    head = git_head()

    main_sha = sha256_of(MAIN_RAW_PATH)
    assert main_sha == MAIN_SHA_EXPECTED, f"STOP: main SHA changed: {main_sha}"
    abl_sha = sha256_of(ABLATION_RAW_PATH)
    assert abl_sha == ABLATION_SHA_EXPECTED, f"STOP: ablation SHA changed: {abl_sha}"
    print(f"Main benchmark SHA verified: {main_sha}")
    print(f"Ablation 5.3B SHA verified:  {abl_sha}")

    # eval seeds disjoint from every previously-used seed family
    prior_seed_sets = [
        set(range(100000, 100030)), set(range(200000, 200030)),
    ]
    for cell_name, cfg in CELLS.items():
        prior_seed_sets.append(set(training_seed_range(cfg["n_episodes"])))
    for s in prior_seed_sets:
        assert set(EVAL_SEEDS).isdisjoint(s), \
            f"STOP: horizon eval seeds overlap prior set {sorted(s)[:3]}..."
    assert EVAL_SEEDS == list(range(300000, 300030))
    print(f"Horizon eval seeds verified disjoint: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}")

    main_models = load_jsonl(MAIN_MODELS_PATH)
    for df_name in DFS:
        row = next((r for r in main_models if r["problem"] == df_name), None)
        assert row is not None, f"STOP: no main A4 model for {df_name}"
        ckpt = torch.load(os.path.join(A4_CKPT_DIR, f"{df_name}.pt"),
                          weights_only=True)
        h = hsd(ckpt["online"])
        assert h == row["final_online_hash"], \
            f"STOP: A4 checkpoint hash mismatch for {df_name}"
    print(f"All {len(DFS)} frozen A4 checkpoints hash-verified against "
         f"final_benchmark_models.jsonl")

    os.makedirs(HORIZON_CKPT_ROOT, exist_ok=True)
    for cell_name in LEARNED_CELLS:
        os.makedirs(os.path.join(HORIZON_CKPT_ROOT, cell_name), exist_ok=True)

    manifest = dict(
        code_commit=head, protocol_version="5.5A",
        problems=DFS, cells={k: v for k, v in CELLS.items()},
        model_seed=MODEL_SEED, eval_seed_base=HORIZON_EVAL_SEED_BASE,
        n_eval=N_EVAL, eval_seeds=[EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        main_benchmark_sha256=main_sha, ablation_5_3B_sha256=abl_sha,
        primary_metric="migd_end", secondary_metric="migd_response",
        statement=("EXPLORATORY MECHANISM DIAGNOSIS on 5 pre-selected DFs; "
                  "not a replacement confirmatory benchmark."),
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    existing_models = load_jsonl(MODELS_PATH)
    existing_runs = load_jsonl(RUNS_PATH)
    done_keys = {(r["variant"], r["problem"], r["seed"]) for r in existing_runs}
    print(f"Resume state: {len(existing_models)} model records, "
         f"{len(existing_runs)} run rows already present.")

    seg = train.make_segments(train.CFG["D"])  # S=2, unchanged (no ablation here)

    for df_name in DFS:
        if git_head() != head:
            print("!!! SOURCE COMMIT CHANGED DURING RUN -- STOPPING")
            sys.exit(1)
        print(f"\n=== {df_name} ===")
        N, ref = train.problem_setup(df_name)
        n_obj_expected = 3 if df_name in train.TRI_OBJECTIVE else 2

        frozen_sds = {}
        get_frozen_a4_reference(df_name, existing_models, main_models)
        existing_models = load_jsonl(MODELS_PATH)
        a4_ckpt = torch.load(os.path.join(A4_CKPT_DIR, f"{df_name}.pt"),
                             weights_only=True)
        frozen_sds["H30_B6K"] = a4_ckpt

        for cell_name in LEARNED_CELLS:
            agent, _ = train_or_load_cell(cell_name, df_name, seg, existing_models)
            existing_models = load_jsonl(MODELS_PATH)
            frozen_sds[cell_name] = dict(
                online={k: v.clone() for k, v in agent.online.state_dict().items()},
                target={k: v.clone() for k, v in agent.target.state_dict().items()})

        for seed in EVAL_SEEDS:
            group_rows = {}
            group_init = {}
            group_failed = False

            if ("A0", df_name, seed) not in done_keys:
                row, ok, init_pop = evaluate_a0(df_name, seed, N, ref)
                if not ok:
                    log_failure(dict(variant="A0", df=df_name, seed=seed,
                                     reason="A0_INVARIANT_FAIL"))
                    group_failed = True
                else:
                    group_rows["A0"] = row
                    group_init["A0"] = init_pop
            else:
                group_rows["A0"] = "done"

            if not group_failed:
                for cell_name in CELLS:
                    if (cell_name, df_name, seed) in done_keys:
                        group_rows[cell_name] = "done"
                        continue
                    try:
                        row, ok, init_pop = evaluate_cell(
                            cell_name, df_name, seg, seed,
                            frozen_sds[cell_name], N, ref, n_obj_expected)
                    except Exception as e:
                        log_failure(dict(variant=cell_name, df=df_name,
                                         seed=seed, error=str(e)))
                        group_failed = True
                        break
                    if not ok:
                        log_failure(dict(variant=cell_name, df=df_name,
                                         seed=seed, reason="INVARIANT_FAIL",
                                         row=row))
                        group_failed = True
                        break
                    group_rows[cell_name] = row
                    group_init[cell_name] = init_pop

            if group_failed:
                continue
            if all(v == "done" for v in group_rows.values()):
                continue

            real_inits = [v for v in group_init.values() if v is not None]
            paired_init_pass = all(np.allclose(real_inits[0], p)
                                   for p in real_inits[1:]) if real_inits else True
            if not paired_init_pass:
                log_failure(dict(df=df_name, seed=seed,
                                 reason="PAIRED_INIT_MISMATCH"))
                continue

            for variant, row in group_rows.items():
                if row == "done":
                    continue
                row["paired_init_pass"] = paired_init_pass
                append_jsonl(RUNS_PATH, row)
                done_keys.add((variant, df_name, seed))

        print(f"{df_name}: evaluation complete for {len(EVAL_SEEDS)} seeds "
             f"({time.time()-t_start:.0f}s elapsed total)")

    regenerate_training_windows_csv()

    sha_after_main = sha256_of(MAIN_RAW_PATH)
    sha_after_abl = sha256_of(ABLATION_RAW_PATH)
    assert sha_after_main == MAIN_SHA_EXPECTED, "STOP: main benchmark file mutated"
    assert sha_after_abl == ABLATION_SHA_EXPECTED, "STOP: ablation file mutated"

    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    print(f"Main benchmark SHA unchanged: {sha_after_main == MAIN_SHA_EXPECTED}")
    print(f"Ablation SHA unchanged: {sha_after_abl == ABLATION_SHA_EXPECTED}")


if __name__ == "__main__":
    main()
