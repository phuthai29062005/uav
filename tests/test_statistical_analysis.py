"""ST1-ST15: statistical analysis implementation + frozen-dataset integrity (5.2)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from stats_analysis import (classify_primary, holm_correction,  # noqa: E402
                            rank_biserial, wilcoxon_paired)

RESULTS = os.path.join(_HERE, "..", "results")
RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")


# --------------------------------------------------- ST1-ST5: rank-biserial
def test_st1_all_positive():
    sa = np.array([1., 2, 3, 4, 5])
    bl = sa + 1
    assert rank_biserial(bl, sa)["r_rb"] == 1.0


def test_st2_all_negative():
    sa = np.array([1., 2, 3, 4, 5])
    bl = sa - 1
    assert rank_biserial(bl, sa)["r_rb"] == -1.0


def test_st3_balanced_near_zero():
    sa = np.zeros(4)
    bl = np.array([1., 2, -1, -2])
    assert abs(rank_biserial(bl, sa)["r_rb"]) < 1e-9


def test_st4_zeros_ignored():
    sa = np.array([1., 2, 3])
    bl = np.array([1., 3, 2])          # d = [0, 1, -1]
    r = rank_biserial(bl, sa)
    assert r["n_nonzero"] == 2


def test_st5_shuffle_invariant():
    sa = np.array([1., 2, 3, 5, 8])
    bl = np.array([2., 1, 5, 3, 9])
    idx = [4, 0, 3, 1, 2]
    a = rank_biserial(bl, sa)["r_rb"]
    b = rank_biserial(bl[idx], sa[idx])["r_rb"]
    assert a == b


def test_wilcoxon_all_zero_diff_policy():
    sa = np.array([1., 2, 3])
    bl = np.array([1., 2, 3])
    w = wilcoxon_paired(bl, sa)
    assert w["W"] == 0.0 and w["p_raw"] == 1.0 and w["n_nonzero"] == 0


# --------------------------------------------------------- ST6-ST9: Holm
def test_st6_holm_known_values():
    p_holm, reject = holm_correction([0.001, 0.01, 0.04, 0.2])
    assert np.allclose(p_holm, [0.004, 0.03, 0.08, 0.2])
    assert reject == [True, True, False, False]


def test_st7_holm_bounds():
    raw = [0.001, 0.01, 0.04, 0.2, 0.5]
    p_holm, _ = holm_correction(raw)
    for r, h in zip(raw, p_holm):
        assert h >= r - 1e-12
        assert h <= 1.0


def test_st8_primary_p_count():
    primary = load_csv("statistics_primary_migd_end.csv")
    assert len(primary) == 14


def test_st9_secondary_p_count():
    secondary = load_csv("statistics_secondary_migd_response.csv")
    assert len(secondary) == 14


# ----------------------------------------------- helpers for ST10+
def load_csv(name):
    import csv
    path = os.path.join(RESULTS, name)
    with open(path) as f:
        return list(csv.DictReader(f))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# -------------------------------------------- ST10-ST15: data integrity
@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st10_paired_table_420_rows():
    paired = load_csv("statistics_paired_420.csv")
    assert len(paired) == 420


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st11_each_df_30_rows():
    paired = load_csv("statistics_paired_420.csv")
    from collections import Counter
    c = Counter(r["DF"] for r in paired)
    assert all(v == 30 for v in c.values())
    assert len(c) == 14


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st12_seeds_identical_between_methods():
    rows = [json.loads(l) for l in open(RAW_PATH)]
    from collections import defaultdict
    seeds_by_method = defaultdict(set)
    for r in rows:
        seeds_by_method[(r["problem"], r["method"])].add(r["seed"])
    for df in [f"DF{i}" for i in range(1, 15)]:
        sa_seeds = seeds_by_method[(df, "SA-DRL-frozen")]
        bl_seeds = seeds_by_method[(df, "NSGA2-clean")]
        assert sa_seeds == bl_seeds
        assert len(sa_seeds) == 30


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st13_no_duplicate_df_seed():
    rows = [json.loads(l) for l in open(RAW_PATH)]
    keys = [(r["method"], r["problem"], r["seed"]) for r in rows]
    assert len(keys) == len(set(keys))


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st14_analysis_metrics_finite():
    primary = load_csv("statistics_primary_migd_end.csv")
    for r in primary:
        for k in ("SA_median", "BL_median", "p_raw", "p_holm", "rank_biserial"):
            assert np.isfinite(float(r[k])), (r["DF"], k)


@pytest.mark.skipif(not os.path.exists(RAW_PATH), reason="frozen dataset not present")
def test_st15_raw_sha256_unchanged():
    summary = json.load(open(os.path.join(RESULTS, "statistics_summary.json")))
    assert summary["raw_data_sha256"] == sha256_of(RAW_PATH)


# ---------------------------------------------- classification rule check
def test_classify_primary_rule():
    assert classify_primary(0.01, 0.5) == "SA+"
    assert classify_primary(0.01, -0.5) == "SA-"
    assert classify_primary(0.5, 0.5) == "="
    assert classify_primary(0.01, 0.0) == "significant/mixed-zero-median"
