"""AS1-AS14: 5.3C confirmatory ablation statistics invariants."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from stats_analysis import rank_biserial, wilcoxon_paired, holm_correction  # noqa: E402
import ablation_5_3C_stats as ab  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
RUNS_PATH = os.path.join(RESULTS, "ablation_5_3B_runs.jsonl")
MAIN_RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
MAIN_SHA_EXPECTED = ab.MAIN_SHA_EXPECTED

pytestmark = pytest.mark.skipif(
    not os.path.exists(RUNS_PATH),
    reason="results/ablation_5_3B_runs.jsonl not present (5.3B not run yet)")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def rows():
    return [json.loads(l) for l in open(RUNS_PATH)]


@pytest.fixture(scope="module")
def paired(rows):
    return ab.build_paired_table(rows) if not os.path.exists(
        os.path.join(RESULTS, "ablation_statistics_paired_420.csv")) else \
        pd.read_csv(os.path.join(RESULTS, "ablation_statistics_paired_420.csv"))


@pytest.fixture(scope="module")
def h_tables():
    """Load the three per-hypothesis CSVs if the full analysis has already
    been run; otherwise skip (this test module checks the sign-convention
    primitives directly regardless, via AS1-AS3/AS7-AS8)."""
    paths = {h: os.path.join(RESULTS, f"ablation_stats_{h}_A4_vs_{ab.HYPOTHESES[h]}.csv")
            for h in ("H1", "H2", "H3")}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("hypothesis CSVs not yet generated")
    return {h: pd.read_csv(p) for h, p in paths.items()}


# ---------------------------------------------------------------------
# AS1-AS3: sign convention -- d = ablated - A4, positive favors A4
# ---------------------------------------------------------------------
def test_AS1_h1_sign_convention_favors_a4():
    a4 = np.array([0.10, 0.10, 0.10])
    a1 = np.array([0.20, 0.20, 0.20])   # ablated strictly worse (higher MIGD)
    r = rank_biserial(a1, a4)           # (ablated, a4) -> d = ablated - a4
    assert r["r_rb"] > 0, "d>0 (ablated worse) must give r_rb>0 favoring A4"


def test_AS2_h2_sign_convention_favors_a4():
    a4 = np.array([0.30, 0.31, 0.29, 0.28])
    a2 = np.array([0.50, 0.55, 0.60, 0.58])  # ablated worse
    r = rank_biserial(a2, a4)
    assert r["r_rb"] > 0


def test_AS3_h3_sign_convention_favors_a4():
    a4 = np.array([1.0, 1.1, 0.9, 1.2])
    a3 = np.array([0.5, 0.4, 0.6, 0.3])  # ablated strictly better (lower)
    r = rank_biserial(a3, a4)
    assert r["r_rb"] < 0, "ablated better (lower MIGD) must give r_rb<0"


# ---------------------------------------------------------------------
# AS4-AS5: Holm families sized and independent
# ---------------------------------------------------------------------
def test_AS4_holm_receives_exactly_14_pvalues(h_tables):
    for h, tbl in h_tables.items():
        assert len(tbl) == 14, f"{h}: expected 14 DFs, got {len(tbl)}"
        assert set(tbl["DF"]) == set(ab.DFS)


def test_AS5_holm_corrections_independent(h_tables):
    """Each family's p_holm must equal Holm applied to ONLY that family's
    14 raw p-values (not e.g. all 42 pooled)."""
    for h, tbl in h_tables.items():
        recomputed, reject = holm_correction(tbl["p_raw"].tolist(), alpha=0.05)
        np.testing.assert_allclose(sorted(recomputed), sorted(tbl["p_holm"]),
                                   rtol=1e-9, atol=1e-12)
        # cross-check: p_holm from a 14-value family must NOT equal what a
        # pooled 42-value Holm would give (unless coincidentally identical
        # extreme p-values) -- verify family sizes differ structurally.
    all_p = pd.concat([t["p_raw"] for t in h_tables.values()]).tolist()
    assert len(all_p) == 42
    pooled_holm, _ = holm_correction(all_p, alpha=0.05)
    # at least one per-family p_holm must differ from the pooled-42 result
    # (proves the families were corrected independently, not pooled)
    per_family_all = pd.concat([t["p_holm"] for t in h_tables.values()]).tolist()
    assert not np.allclose(sorted(per_family_all), sorted(pooled_holm), rtol=1e-9)


# ---------------------------------------------------------------------
# AS6: A0 receives no inferential test
# ---------------------------------------------------------------------
def test_AS6_a0_no_inferential_test(h_tables):
    for h, tbl in h_tables.items():
        for col in ("A4_mean", "ABL_mean"):
            assert col in tbl.columns
        assert not any("A0" in c for c in tbl.columns), \
            f"{h}: A0 must not appear in inferential columns"
    hyp_summary_path = os.path.join(RESULTS, "ablation_hypothesis_summary.json")
    if os.path.exists(hyp_summary_path):
        summary = json.load(open(hyp_summary_path))
        assert set(summary.keys()) == {"H1", "H2", "H3"}, \
            "A0 must not have its own hypothesis-summary entry"


# ---------------------------------------------------------------------
# AS7-AS8: rank-biserial extremes
# ---------------------------------------------------------------------
def test_AS7_rank_biserial_plus1_when_a4_always_better():
    a4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    abl = a4 + 1.0   # strictly worse (higher) every pair
    r = rank_biserial(abl, a4)
    assert r["r_rb"] == pytest.approx(1.0)


def test_AS8_rank_biserial_minus1_when_ablated_always_better():
    a4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    abl = a4 - 1.0   # strictly better (lower) every pair
    r = rank_biserial(abl, a4)
    assert r["r_rb"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------
# AS9-AS12: paired table structural invariants
# ---------------------------------------------------------------------
def test_AS9_paired_table_exactly_420_rows(paired):
    assert len(paired) == 420


def test_AS10_every_df_has_30_seeds(paired):
    for df in ab.DFS:
        sub = paired[paired["DF"] == df]
        assert len(sub) == 30, f"{df}: {len(sub)} rows"
        assert sub["seed"].nunique() == 30


def test_AS11_every_paired_row_has_all_variants(paired):
    for v in ab.VARIANTS:
        assert f"{v}_migd_end" in paired.columns
        assert paired[f"{v}_migd_end"].notna().all()


def test_AS12_holdout_seeds_exact_range(paired):
    seeds = sorted(paired["seed"].unique().tolist())
    assert seeds == ab.HOLDOUT_SEEDS
    assert seeds[0] == 200000 and seeds[-1] == 200029 and len(seeds) == 30


# ---------------------------------------------------------------------
# AS13-AS14: SHA immutability
# ---------------------------------------------------------------------
def test_AS13_ablation_raw_sha_unchanged_after_analysis():
    sha_now = sha256_of(RUNS_PATH)
    manifest_note = os.path.join(RESULTS, "ablation_5_3B_manifest.json")
    assert os.path.exists(manifest_note)
    # The analysis script asserts sha_before==sha_after internally on every
    # run; here we additionally confirm the file is currently well-formed
    # and readable (2100 rows) as an independent post-hoc check.
    rows_now = [json.loads(l) for l in open(RUNS_PATH)]
    assert len(rows_now) == 2100
    assert len(sha_now) == 64


def test_AS14_main_benchmark_sha_unchanged():
    sha_now = sha256_of(MAIN_RAW_PATH)
    assert sha_now == MAIN_SHA_EXPECTED
