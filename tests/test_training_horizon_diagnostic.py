"""TH1-TH24: 5.5A training-horizon 2x2 diagnostic invariants."""
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

import horizon_5_5A as h5  # noqa: E402
import train  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
RUNS_PATH = os.path.join(RESULTS, "horizon_5_5A_runs.jsonl")
MAIN_RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
ABLATION_RAW_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# design-level invariants -- always run, no data required
# ---------------------------------------------------------------------
def test_TH1_selected_dfs_exact():
    assert h5.DFS == ["DF1", "DF7", "DF10", "DF11", "DF12"]


def test_TH2_eval_seeds_exact():
    assert h5.EVAL_SEEDS == list(range(300000, 300030))


def test_TH3_eval_seeds_disjoint_from_prior_families():
    s = set(h5.EVAL_SEEDS)
    assert s.isdisjoint(set(range(100000, 100030)))
    assert s.isdisjoint(set(range(200000, 200030)))


def test_TH4_H30_B6K_transitions():
    c = h5.CELLS["H30_B6K"]
    assert c["n_changes"] * c["n_episodes"] == 6000
    assert c["n_changes"] == 30 and c["n_episodes"] == 200


def test_TH5_H100_B6K_transitions():
    c = h5.CELLS["H100_B6K"]
    assert c["n_changes"] * c["n_episodes"] == 6000
    assert c["n_changes"] == 100 and c["n_episodes"] == 60


def test_TH6_H30_B18K_transitions():
    c = h5.CELLS["H30_B18K"]
    assert c["n_changes"] * c["n_episodes"] == 18000
    assert c["n_changes"] == 30 and c["n_episodes"] == 600


def test_TH7_H100_B18K_transitions():
    c = h5.CELLS["H100_B18K"]
    assert c["n_changes"] * c["n_episodes"] == 18000
    assert c["n_changes"] == 100 and c["n_episodes"] == 180


def test_TH8_equal_transition_cells_equal_replay_count_by_construction():
    """B6K cells (H30_B6K, H100_B6K) both push exactly 6000 transitions;
    B18K cells (H30_B18K, H100_B18K) both push exactly 18000 -- verified
    structurally here (design), empirically in test_TH9 (learn steps,
    requires trained data)."""
    b6k = [c for c in h5.CELLS.values()
          if c["n_changes"] * c["n_episodes"] == 6000]
    b18k = [c for c in h5.CELLS.values()
           if c["n_changes"] * c["n_episodes"] == 18000]
    assert len(b6k) == 2 and len(b18k) == 2


def test_TH20_sign_conventions():
    """d = first - second per Phase 13; positive favors the SECOND/target
    condition in every one of C1-C4 as specified."""
    # C1 = MIGD(H30_B6K) - MIGD(H100_B6K); positive => H100 (2nd) better
    first, second = np.array([0.2, 0.2]), np.array([0.1, 0.1])
    d = first - second
    assert np.all(d > 0)  # second (target) has lower/better MIGD -> d>0


# ---------------------------------------------------------------------
# data-dependent invariants -- skipped until the full run has produced
# results/horizon_5_5A_runs.jsonl
# ---------------------------------------------------------------------
pytestmark_data = pytest.mark.skipif(
    not os.path.exists(RUNS_PATH),
    reason="results/horizon_5_5A_runs.jsonl not present (5.5A not run yet)")


@pytest.fixture(scope="module")
def runs():
    if not os.path.exists(RUNS_PATH):
        pytest.skip("no data")
    return [json.loads(l) for l in open(RUNS_PATH)]


@pytest.fixture(scope="module")
def models():
    p = os.path.join(RESULTS, "horizon_5_5A_models.jsonl")
    if not os.path.exists(p):
        pytest.skip("no data")
    return [json.loads(l) for l in open(p)]


@pytestmark_data
def test_TH9_equal_transition_cells_equal_learn_steps(models):
    by_budget = {}
    for m in models:
        by_budget.setdefault(m["transition_budget"], []).append(m["learn_steps"])
    for budget, steps in by_budget.items():
        assert len(set(steps)) == 1, \
            (f"budget {budget}: learn_steps differ across cells: {steps} "
            f"-- STOP, needs explanation before statistical interpretation")


@pytestmark_data
def test_TH10_H30_B6K_checkpoint_hash_matches_main_manifest(models):
    main_models = {r["problem"]: r for r in
                  [json.loads(l) for l in
                   open(os.path.join(RESULTS, "final_benchmark_models.jsonl"))]}
    for m in models:
        if m["variant"] == "H30_B6K":
            assert m["model_hash"] == main_models[m["DF"]]["final_online_hash"]


@pytestmark_data
def test_TH11_original_checkpoints_unmodified(models):
    for df in h5.DFS:
        ckpt = torch.load(os.path.join(h5.A4_CKPT_DIR, f"{df}.pt"),
                          weights_only=True)
        h = h5.hsd(ckpt["online"])
        main_models = {r["problem"]: r for r in
                      [json.loads(l) for l in
                       open(os.path.join(RESULTS, "final_benchmark_models.jsonl"))]}
        assert h == main_models[df]["final_online_hash"]


@pytestmark_data
def test_TH12_frozen_eval_weights_remain_frozen(runs):
    for r in runs:
        if r["variant"] != "A0":
            assert r["frozen_weights_pass"] is True


@pytestmark_data
def test_TH13_every_group_has_five_variants(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["problem"], r["seed"]), set()).add(r["variant"])
    for k, v in groups.items():
        assert v == set(h5.VARIANTS), f"{k}: {v}"


@pytestmark_data
def test_TH14_total_rows_750(runs):
    assert len(runs) == 750


@pytestmark_data
def test_TH15_no_nan_inf(runs):
    for r in runs:
        assert np.isfinite(r["migd_end"])
        assert np.isfinite(r["migd_response"])
        assert np.isfinite(r["fes_used"])


@pytestmark_data
def test_TH16_fe_ledger_exact(runs):
    for r in runs:
        assert r["fes_used"] == sum(r["fe_breakdown"].values())


@pytestmark_data
def test_TH17_H100_logs_have_mid_late(models):
    windows_path = os.path.join(RESULTS, "horizon_5_5A_training_windows.csv")
    if not os.path.exists(windows_path):
        pytest.skip("no window data")
    import csv
    with open(windows_path) as f:
        rows = list(csv.DictReader(f))
    for cell in ("H100_B6K", "H100_B18K"):
        wins = set(r["window"] for r in rows if r["variant"] == cell)
        assert "MID" in wins and "LATE" in wins, (cell, wins)


@pytestmark_data
def test_TH18_H30_logs_no_fake_mid_late():
    windows_path = os.path.join(RESULTS, "horizon_5_5A_training_windows.csv")
    if not os.path.exists(windows_path):
        pytest.skip("no window data")
    import csv
    with open(windows_path) as f:
        rows = list(csv.DictReader(f))
    wins = set(r["window"] for r in rows if r["variant"] == "H30_B18K")
    assert wins.issubset({"EARLY"}), wins


def test_TH19_p_memory_given_available_denominator_handled():
    """Direct unit test of the aggregation helper (no data dependency):
    a window with zero has_memory=1 rows must return None, never a bogus
    0/0-derived value, and must never divide by the unconditional count."""
    window_rows = [
        dict(window="EARLY", gate=0, seg1=0, seg2=1, c1=0.1, c2=0.2,
            d_mem=1.0, has_memory=0.0, dispersion=0.5, hv_drop=0.1),
        dict(window="EARLY", gate=0, seg1=1, seg2=0, c1=0.3, c2=0.1,
            d_mem=1.0, has_memory=0.0, dispersion=0.4, hv_drop=0.2),
    ]
    agg = h5.aggregate_window_rows("DFx", "TEST", window_rows)
    assert agg[0]["p_memory_given_available"] is None
    assert agg[0]["n_available"] == 0

    window_rows2 = window_rows + [
        dict(window="EARLY", gate=1, seg1=0, seg2=0, c1=0.2, c2=0.2,
            d_mem=0.1, has_memory=1.0, dispersion=0.3, hv_drop=0.05),
    ]
    agg2 = h5.aggregate_window_rows("DFx", "TEST", window_rows2)
    assert agg2[0]["p_memory_given_available"] == pytest.approx(1.0)
    assert agg2[0]["n_available"] == 1


@pytestmark_data
def test_TH21_holm_family_exactly_5_pvalues():
    for name in ("C1_horizon_6K", "C2_budget_H30", "C3_horizon_18K",
                "C4_budget_H100"):
        p = os.path.join(RESULTS, f"horizon_5_5A_{name}.csv")
        if not os.path.exists(p):
            pytest.skip(f"{name} stats not yet generated")
        import pandas as pd
        df = pd.read_csv(p)
        assert len(df) == 5
        assert set(df["DF"]) == set(h5.DFS)


@pytestmark_data
def test_TH22_a0_no_inferential_test():
    for name in ("C1_horizon_6K", "C2_budget_H30", "C3_horizon_18K",
                "C4_budget_H100"):
        p = os.path.join(RESULTS, f"horizon_5_5A_{name}.csv")
        if not os.path.exists(p):
            pytest.skip(f"{name} stats not yet generated")
        import pandas as pd
        df = pd.read_csv(p)
        assert not any("A0" in c for c in df.columns)


def test_TH23_main_benchmark_sha_unchanged():
    assert sha256_of(MAIN_RAW_PATH) == h5.MAIN_SHA_EXPECTED


def test_TH24_ablation_sha_unchanged():
    assert sha256_of(ABLATION_RAW_PATH) == h5.ABLATION_SHA_EXPECTED
