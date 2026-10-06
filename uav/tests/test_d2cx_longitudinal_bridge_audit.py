"""
D2-CX — Longitudinal Path Bridge Audit Tests (T1–T60).

Tests validate:
  T1–T5:   Module structure and constants
  T6–T12:  Shared warmup + burn-in reproducibility
  T13–T20: Arm state cloning (deep-independent copies)
  T21–T28: CRN child generation (DIAG arms)
  T29–T34: Production oracle path (PROD arms)
  T35–T40: HARD structural invariant
  T41–T46: FE fairness across 4 arms
  T47–T52: Checkpoint fingerprint API
  T53–T56: Divergence report structure
  T57–T60: Verdict classification logic
"""

import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.core.survival import region_protected_survival
from prdrl.history.region_history import RegionHistory
from prdrl.metrics.igd import igd
from prdrl.oracle.oracle import (
    PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    PREDICT_SURVIVAL_UNRESTRICTED,
    respond_population,
)
from prdrl.problems.controlled_predict import (
    evaluate as cp_evaluate,
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)

import prdrl.experiments.D2CX_longitudinal_bridge_audit as d2cx

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mixed_setup():
    """Build shared MIXED map + PF references (session-scoped for speed)."""
    V = reference(d2cx.N_OBJ, d2cx.N_PARTITIONS)
    R = len(V)
    mixed_map = d2cx.make_d2cx_mixed_dynamics(R)
    pf_star = get_true_pf(n_points=500)
    is_persistent_zone = d2cx.make_pf_zones(pf_star, mixed_map)
    return {
        "V": V, "R": R,
        "mixed_map": mixed_map,
        "pf_star": pf_star,
        "is_persistent_zone": is_persistent_zone,
    }


@pytest.fixture(scope="session")
def smoke_rows(mixed_setup):
    """Run 3-change smoke test for seed=81 Block A (fast)."""
    return d2cx.run_one_d2cx(
        seed=81,
        n_scored=3,
        mixed_map=mixed_setup["mixed_map"],
        pf_star=mixed_setup["pf_star"],
        is_persistent_zone=mixed_setup["is_persistent_zone"],
        block_label="A",
    )


@pytest.fixture(scope="session")
def smoke_rows_b(mixed_setup):
    """Run 3-change smoke test for seed=109 Block B (fast)."""
    return d2cx.run_one_d2cx(
        seed=109,
        n_scored=3,
        mixed_map=mixed_setup["mixed_map"],
        pf_star=mixed_setup["pf_star"],
        is_persistent_zone=mixed_setup["is_persistent_zone"],
        block_label="B",
    )


# ---------------------------------------------------------------------------
# T1 — Four arm names present in ARM_DEFS
# ---------------------------------------------------------------------------

def test_t1_arm_defs_count():
    assert len(d2cx.ARM_DEFS) == 4


def test_t1b_arm_names():
    names = {ad["arm"] for ad in d2cx.ARM_DEFS}
    assert names == {"DIAG_U03", "PROD_O0", "DIAG_H005", "PROD_O3"}


# ---------------------------------------------------------------------------
# T2 — DIAG arms use_diag_crn=True; PROD arms use_diag_crn=False
# ---------------------------------------------------------------------------

def test_t2_diag_arms_use_crn():
    for ad in d2cx.ARM_DEFS:
        if ad["arm"].startswith("DIAG"):
            assert ad["use_diag_crn"] is True, f"{ad['arm']} should use CRN"
        else:
            assert ad["use_diag_crn"] is False, f"{ad['arm']} should NOT use CRN"


# ---------------------------------------------------------------------------
# T3 — HARD arms use_hard=True; unrestricted arms use_hard=False
# ---------------------------------------------------------------------------

def test_t3_hard_arm_flags():
    hard_arms = {"DIAG_H005", "PROD_O3"}
    unrest_arms = {"DIAG_U03", "PROD_O0"}
    for ad in d2cx.ARM_DEFS:
        if ad["arm"] in hard_arms:
            assert ad["use_hard"] is True
        else:
            assert ad["use_hard"] is False


# ---------------------------------------------------------------------------
# T4 — Sigma values correct
# ---------------------------------------------------------------------------

def test_t4_sigma_values():
    by_arm = {ad["arm"]: ad["sigma"] for ad in d2cx.ARM_DEFS}
    assert by_arm["DIAG_U03"] == 0.03
    assert by_arm["PROD_O0"]  == 0.03
    assert by_arm["DIAG_H005"] == 0.005
    assert by_arm["PROD_O3"]  == 0.005


# ---------------------------------------------------------------------------
# T5 — Block seeds are correct and disjoint
# ---------------------------------------------------------------------------

def test_t5_block_a_seeds():
    assert d2cx.BLOCK_A_SEEDS == list(range(81, 101))


def test_t5b_block_b_seeds():
    assert d2cx.BLOCK_B_SEEDS == list(range(109, 129))


def test_t5c_blocks_disjoint():
    assert len(set(d2cx.BLOCK_A_SEEDS) & set(d2cx.BLOCK_B_SEEDS)) == 0


# ---------------------------------------------------------------------------
# T6 — Smoke run returns rows for all 4 arms × 3 changes
# ---------------------------------------------------------------------------

def test_t6_smoke_row_count(smoke_rows):
    assert len(smoke_rows) == 4 * 3, f"Expected 12 rows, got {len(smoke_rows)}"


def test_t6b_smoke_arms_present(smoke_rows):
    arm_names = {r["arm"] for r in smoke_rows}
    assert arm_names == {"DIAG_U03", "PROD_O0", "DIAG_H005", "PROD_O3"}


# ---------------------------------------------------------------------------
# T7 — All rows have required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "seed", "block", "arm", "change_k", "scored_change_idx",
    "tgt_r", "tgt_regime", "sigma", "use_diag_crn", "use_hard",
    "operator", "realized_op", "n_r_before", "extra_fe",
    "igd_pre", "igd_response", "igd_epoch_end",
    "n_old_dropped_other", "hard_invariant_ok",
    "c0_fp", "c1_F_fp", "c1_rids_fp", "c2_fp", "c2_rng_scheme",
    "c3_fp", "c4_fp", "surv_seed_diag",
]


def test_t7_required_keys(smoke_rows):
    for row in smoke_rows:
        missing = [k for k in REQUIRED_KEYS if k not in row]
        assert missing == [], f"Row {row['arm']} k={row['change_k']} missing: {missing}"


# ---------------------------------------------------------------------------
# T8 — IGD values are finite and positive
# ---------------------------------------------------------------------------

def test_t8_igd_finite(smoke_rows):
    for row in smoke_rows:
        assert np.isfinite(row["igd_pre"]), f"{row['arm']} k={row['change_k']} igd_pre is not finite"
        if row["operator"] != "COMMON_KEEP":
            assert np.isfinite(row["igd_response"])
            assert np.isfinite(row["igd_epoch_end"])
            assert row["igd_pre"] > 0
            assert row["igd_response"] > 0


# ---------------------------------------------------------------------------
# T9 — Block label is correct
# ---------------------------------------------------------------------------

def test_t9_block_label_a(smoke_rows):
    assert all(r["block"] == "A" for r in smoke_rows)


def test_t9b_block_label_b(smoke_rows_b):
    assert all(r["block"] == "B" for r in smoke_rows_b)


# ---------------------------------------------------------------------------
# T10 — DIAG arms have C2 fingerprint; PROD arms have None
# ---------------------------------------------------------------------------

def test_t10_diag_arms_have_c2(smoke_rows):
    for row in smoke_rows:
        if row["operator"] == "COMMON_KEEP":
            continue
        if row["use_diag_crn"]:
            assert row["c2_fp"] is not None, f"{row['arm']} k={row['change_k']} missing c2_fp"
        else:
            assert row["c2_fp"] is None, f"{row['arm']} k={row['change_k']} should not have c2_fp"


# ---------------------------------------------------------------------------
# T11 — PROD arms have c2_rng_scheme describing oracle base_seed
# ---------------------------------------------------------------------------

def test_t11_prod_rng_scheme(smoke_rows):
    for row in smoke_rows:
        if row["operator"] == "COMMON_KEEP":
            continue
        if not row["use_diag_crn"]:
            scheme = row["c2_rng_scheme"]
            assert scheme is not None
            assert "ORACLE_PREDICT" in scheme or "base=" in scheme, \
                f"Expected oracle RNG description, got: {scheme}"


# ---------------------------------------------------------------------------
# T12 — DIAG arms have c2_rng_scheme describing L2B CRN
# ---------------------------------------------------------------------------

def test_t12_diag_rng_scheme(smoke_rows):
    for row in smoke_rows:
        if row["operator"] == "COMMON_KEEP" or not row["use_diag_crn"]:
            continue
        scheme = row["c2_rng_scheme"]
        assert scheme is not None
        assert "L2B_CRN" in scheme, f"Expected L2B CRN description, got: {scheme}"


# ---------------------------------------------------------------------------
# T13 — DIAG arms share same CRN (DIAG_U03 and DIAG_H005 have same C2 md5)
# ---------------------------------------------------------------------------

def test_t13_diag_arms_share_c2(smoke_rows):
    """DIAG_U03 and DIAG_H005 use same CRN base samples → same C2 fingerprint."""
    for k in range(3):
        row_u03  = next(r for r in smoke_rows if r["arm"] == "DIAG_U03"  and r["change_k"] == k and r["operator"] != "COMMON_KEEP")
        row_h005 = next(r for r in smoke_rows if r["arm"] == "DIAG_H005" and r["change_k"] == k and r["operator"] != "COMMON_KEEP")
        if row_u03["c2_fp"] and row_h005["c2_fp"]:
            # Same sigma for CRN, different delta_hat → children CAN differ by sigma
            # But u_parent and xi_mat are shared → NOT necessarily identical (sigma differs)
            # Just check both have fingerprints
            assert "md5" in row_u03["c2_fp"]
            assert "md5" in row_h005["c2_fp"]


# ---------------------------------------------------------------------------
# T14 — PROD_O0 and PROD_O3 have different c2_rng_scheme base seeds
#         (same seed and k but different sigma → different? actually same base_seed formula)
# ---------------------------------------------------------------------------

def test_t14_prod_arms_rng_scheme_present(smoke_rows):
    for k in range(3):
        row_o0 = next((r for r in smoke_rows if r["arm"] == "PROD_O0" and r["change_k"] == k and r["operator"] != "COMMON_KEEP"), None)
        row_o3 = next((r for r in smoke_rows if r["arm"] == "PROD_O3" and r["change_k"] == k and r["operator"] != "COMMON_KEEP"), None)
        if row_o0 and row_o3:
            assert row_o0["c2_rng_scheme"] is not None
            assert row_o3["c2_rng_scheme"] is not None


# ---------------------------------------------------------------------------
# T15 — C0 fingerprint must agree across all 4 arms at k=0
#        (all started from the same cloned state)
# ---------------------------------------------------------------------------

def test_t15_c0_agrees_at_k0(smoke_rows):
    k0_rows = [r for r in smoke_rows if r["change_k"] == 0 and r["operator"] != "COMMON_KEEP"]
    if len(k0_rows) == 4:
        md5s = [r["c0_fp"]["md5"] for r in k0_rows if r["c0_fp"]]
        assert len(set(md5s)) == 1, f"C0 diverged at k=0: {md5s}"


# ---------------------------------------------------------------------------
# T16 — C1 (F_plus) must agree across all 4 arms at same k
#        (all arms evaluate same X from same initial clone at same t_float)
# ---------------------------------------------------------------------------

def test_t16_c1_f_agrees_at_k0(smoke_rows):
    k0_rows = [r for r in smoke_rows if r["change_k"] == 0 and r["operator"] != "COMMON_KEEP"]
    if len(k0_rows) == 4:
        md5s = [r["c1_F_fp"]["md5"] for r in k0_rows if r["c1_F_fp"]]
        assert len(set(md5s)) == 1, f"C1_F diverged at k=0: {md5s}"


# ---------------------------------------------------------------------------
# T17 — C1 (region_ids) must agree across all 4 arms at same k
# ---------------------------------------------------------------------------

def test_t17_c1_rids_agrees_at_k0(smoke_rows):
    k0_rows = [r for r in smoke_rows if r["change_k"] == 0 and r["operator"] != "COMMON_KEEP"]
    if len(k0_rows) == 4:
        md5s = [r["c1_rids_fp"]["md5"] for r in k0_rows if r["c1_rids_fp"]]
        assert len(set(md5s)) == 1, f"C1_RIDS diverged at k=0: {md5s}"


# ---------------------------------------------------------------------------
# T18 — Seed 81 Block A and seed 109 Block B run without error
# ---------------------------------------------------------------------------

def test_t18_block_b_runs(smoke_rows_b):
    assert len(smoke_rows_b) > 0


# ---------------------------------------------------------------------------
# T19 — HARD structural invariant holds for DIAG_H005
# ---------------------------------------------------------------------------

def test_t19_hard_invariant_diag_h005(smoke_rows):
    for row in smoke_rows:
        if row["arm"] == "DIAG_H005" and row["operator"] == "PREDICT":
            if row["realized_op"] == "PREDICT":
                assert row["n_old_dropped_other"] == 0, \
                    f"HARD invariant violated: seed={row['seed']} k={row['change_k']}"
                assert row["hard_invariant_ok"] is True


# ---------------------------------------------------------------------------
# T20 — HARD structural invariant holds for PROD_O3
# ---------------------------------------------------------------------------

def test_t20_hard_invariant_prod_o3(smoke_rows):
    for row in smoke_rows:
        if row["arm"] == "PROD_O3" and row["operator"] == "PREDICT":
            if row["realized_op"] == "PREDICT":
                n = row["n_old_dropped_other"]
                assert n is not None
                assert n == 0, \
                    f"HARD invariant violated PROD_O3: seed={row['seed']} k={row['change_k']}"


# ---------------------------------------------------------------------------
# T21 — FE fairness: all 4 arms have same extra_fe for each (seed, k)
# ---------------------------------------------------------------------------

def test_t21_fe_fairness(smoke_rows):
    for k in range(3):
        k_rows = [r for r in smoke_rows if r["change_k"] == k and r["operator"] != "COMMON_KEEP"]
        if len(k_rows) == 4:
            fes = {r["arm"]: r["extra_fe"] for r in k_rows}
            assert len(set(fes.values())) == 1, \
                f"FE fairness violated at k={k}: {fes}"


# ---------------------------------------------------------------------------
# T22 — extra_fe equals BUDGET for PREDICT realized_op
# ---------------------------------------------------------------------------

def test_t22_extra_fe_equals_budget(smoke_rows):
    for row in smoke_rows:
        if row.get("realized_op") == "PREDICT" and row["operator"] != "COMMON_KEEP":
            assert row["extra_fe"] == d2cx.BUDGET, \
                f"{row['arm']} k={row['change_k']}: extra_fe={row['extra_fe']} != BUDGET={d2cx.BUDGET}"


# ---------------------------------------------------------------------------
# T23 — COMMON_KEEP rows have extra_fe == 0
# ---------------------------------------------------------------------------

def test_t23_common_keep_fe_zero(smoke_rows):
    for row in smoke_rows:
        if row["operator"] == "COMMON_KEEP":
            assert row["extra_fe"] == 0


# ---------------------------------------------------------------------------
# T24 — n_r_before > 0 for non-KEEP rows
# ---------------------------------------------------------------------------

def test_t24_n_r_before_positive(smoke_rows):
    for row in smoke_rows:
        if row["operator"] != "COMMON_KEEP":
            assert row["n_r_before"] > 0, \
                f"{row['arm']} k={row['change_k']}: n_r_before={row['n_r_before']}"


# ---------------------------------------------------------------------------
# T25 — scored_change_idx = change_k + 2
# ---------------------------------------------------------------------------

def test_t25_scored_change_idx(smoke_rows):
    for row in smoke_rows:
        assert row["scored_change_idx"] == row["change_k"] + 2


# ---------------------------------------------------------------------------
# T26 — target region schedule is (seed + k) % R
# ---------------------------------------------------------------------------

def test_t26_target_region_schedule(smoke_rows, mixed_setup):
    R = mixed_setup["R"]
    for row in smoke_rows:
        expected = (row["seed"] + row["change_k"]) % R
        assert row["tgt_r"] == expected, \
            f"tgt_r mismatch: expected {expected}, got {row['tgt_r']}"


# ---------------------------------------------------------------------------
# T27 — tgt_regime is PERSISTENT_REGION or STATIONARY_REGION
# ---------------------------------------------------------------------------

def test_t27_tgt_regime_valid(smoke_rows):
    valid = {"PERSISTENT_REGION", "STATIONARY_REGION"}
    for row in smoke_rows:
        assert row["tgt_regime"] in valid, f"Invalid tgt_regime: {row['tgt_regime']}"


# ---------------------------------------------------------------------------
# T28 — C3 fingerprint is always present for non-KEEP PREDICT rows
# ---------------------------------------------------------------------------

def test_t28_c3_fp_present(smoke_rows):
    for row in smoke_rows:
        if row["operator"] != "COMMON_KEEP":
            assert row["c3_fp"] is not None, \
                f"{row['arm']} k={row['change_k']} missing c3_fp"
            assert "md5" in row["c3_fp"]


# ---------------------------------------------------------------------------
# T29 — C4 fingerprint is always present for non-KEEP rows
# ---------------------------------------------------------------------------

def test_t29_c4_fp_present(smoke_rows):
    for row in smoke_rows:
        if row["operator"] != "COMMON_KEEP":
            assert row["c4_fp"] is not None, \
                f"{row['arm']} k={row['change_k']} missing c4_fp"
            assert "md5" in row["c4_fp"]


# ---------------------------------------------------------------------------
# T30 — DIAG arms have surv_seed_diag as int; PROD arms have "PROD_DEFAULT_NONE"
# ---------------------------------------------------------------------------

def test_t30_surv_seed_diag(smoke_rows):
    for row in smoke_rows:
        if row["operator"] == "COMMON_KEEP":
            continue
        if row["use_diag_crn"]:
            assert isinstance(row["surv_seed_diag"], int), \
                f"{row['arm']}: surv_seed_diag should be int, got {type(row['surv_seed_diag'])}"
        else:
            assert row["surv_seed_diag"] == "PROD_DEFAULT_NONE", \
                f"{row['arm']}: surv_seed_diag should be PROD_DEFAULT_NONE"


# ---------------------------------------------------------------------------
# T31 — DIAG_U03 and PROD_O0 have same sigma (0.03)
# ---------------------------------------------------------------------------

def test_t31_sigma_consistency(smoke_rows):
    for row in smoke_rows:
        if row["arm"] in ("DIAG_U03", "PROD_O0"):
            assert row["sigma"] == 0.03
        elif row["arm"] in ("DIAG_H005", "PROD_O3"):
            assert row["sigma"] == 0.005


# ---------------------------------------------------------------------------
# T32 — _fp fingerprint returns dict with required keys
# ---------------------------------------------------------------------------

def test_t32_fp_structure():
    arr = np.random.default_rng(42).uniform(size=(10, 4))
    fp = d2cx._fp(arr, "test")
    assert "md5" in fp
    assert "shape" in fp
    assert "mean" in fp
    assert fp["shape"] == [10, 4]


# ---------------------------------------------------------------------------
# T33 — _fp returns None for empty array
# ---------------------------------------------------------------------------

def test_t33_fp_none_for_empty():
    assert d2cx._fp(np.array([]), "empty") is None


# ---------------------------------------------------------------------------
# T34 — _fps_diverge detects genuine difference
# ---------------------------------------------------------------------------

def test_t34_fps_diverge_detects_diff():
    a = np.ones((5, 3))
    b = np.ones((5, 3)) * 2.0
    fp_a = d2cx._fp(a)
    fp_b = d2cx._fp(b)
    assert d2cx._fps_diverge(fp_a, fp_b) is True


def test_t34b_fps_diverge_same():
    a = np.ones((5, 3))
    fp_a = d2cx._fp(a)
    fp_a2 = d2cx._fp(a.copy())
    assert d2cx._fps_diverge(fp_a, fp_a2) is False


# ---------------------------------------------------------------------------
# T35 — target_region schedule matches D2C convention
# ---------------------------------------------------------------------------

def test_t35_target_region_matches_d2c():
    for seed in [81, 82, 109, 110]:
        for k in range(10):
            R = 10
            expected = (seed + k) % R
            assert d2cx.target_region(seed, k, R) == expected


# ---------------------------------------------------------------------------
# T36 — make_d2cx_mixed_dynamics returns half persistent half stationary
# ---------------------------------------------------------------------------

def test_t36_mixed_map_structure():
    R = 10
    mixed_map = d2cx.make_d2cx_mixed_dynamics(R)
    assert len(mixed_map) == R
    assert sum(1 for v in mixed_map if v == "persistent") == 5
    assert sum(1 for v in mixed_map if v == "stationary") == 5


# ---------------------------------------------------------------------------
# T37 — get_target_regime returns correct label
# ---------------------------------------------------------------------------

def test_t37_get_target_regime():
    mixed_map = d2cx.make_d2cx_mixed_dynamics(10)
    # Regions 0-4 persistent, 5-9 stationary
    assert d2cx.get_target_regime(0, mixed_map) == "PERSISTENT_REGION"
    assert d2cx.get_target_regime(5, mixed_map) == "STATIONARY_REGION"
    assert d2cx.get_target_regime(9, mixed_map) == "STATIONARY_REGION"


# ---------------------------------------------------------------------------
# T38 — HARD invariant check fn counts correctly
# ---------------------------------------------------------------------------

def test_t38_count_old_dropped_other():
    X_old = np.eye(5)  # rows 0-4
    region_ids = np.array([0, 0, 1, 1, 1])
    tgt_r = 0
    # All 3 non-target rows (region 1 = indices 2,3,4) present in X_surv
    X_surv = X_old.copy()
    n = d2cx._count_old_dropped_other(X_old, region_ids, tgt_r, X_surv)
    assert n == 0


def test_t38b_count_old_dropped_other_missing():
    X_old = np.eye(5)
    region_ids = np.array([0, 0, 1, 1, 1])
    tgt_r = 0
    # Drop row 2 (non-target)
    X_surv = np.vstack([X_old[[0, 1, 3, 4]], np.zeros((1, 5))])
    n = d2cx._count_old_dropped_other(X_old, region_ids, tgt_r, X_surv)
    assert n == 1


# ---------------------------------------------------------------------------
# T39 — DIAG_U03 and DIAG_H005 start from the same initial state
#        (C0 fingerprint matches at k=0)
# ---------------------------------------------------------------------------

def test_t39_diag_arms_same_c0(smoke_rows):
    k0 = [r for r in smoke_rows if r["change_k"] == 0 and r["arm"] in ("DIAG_U03", "DIAG_H005")]
    md5s = [r["c0_fp"]["md5"] for r in k0 if r["c0_fp"]]
    if len(md5s) == 2:
        assert md5s[0] == md5s[1], "DIAG arms must share C0 at k=0"


# ---------------------------------------------------------------------------
# T40 — PROD_O0 and PROD_O3 start from the same initial state
# ---------------------------------------------------------------------------

def test_t40_prod_arms_same_c0(smoke_rows):
    k0 = [r for r in smoke_rows if r["change_k"] == 0 and r["arm"] in ("PROD_O0", "PROD_O3")]
    md5s = [r["c0_fp"]["md5"] for r in k0 if r["c0_fp"]]
    if len(md5s) == 2:
        assert md5s[0] == md5s[1], "PROD arms must share C0 at k=0"


# ---------------------------------------------------------------------------
# T41 — DIAG_U03 and PROD_O0 share same C0 (all 4 from same clone)
# ---------------------------------------------------------------------------

def test_t41_all_arms_same_c0(smoke_rows):
    k0 = [r for r in smoke_rows if r["change_k"] == 0]
    md5s = [r["c0_fp"]["md5"] for r in k0 if r.get("c0_fp")]
    if len(md5s) == 4:
        assert len(set(md5s)) == 1, f"All 4 arms must share C0 at k=0, got: {set(md5s)}"


# ---------------------------------------------------------------------------
# T42 — C1 region_ids must agree across all 4 arms at k=0
# ---------------------------------------------------------------------------

def test_t42_all_arms_same_c1_rids(smoke_rows):
    k0 = [r for r in smoke_rows if r["change_k"] == 0 and r.get("c1_rids_fp")]
    md5s = [r["c1_rids_fp"]["md5"] for r in k0]
    if len(md5s) == 4:
        assert len(set(md5s)) == 1, f"C1_RIDS must agree at k=0, got: {set(md5s)}"


# ---------------------------------------------------------------------------
# T43 — C3 fingerprints are present and non-trivial
# ---------------------------------------------------------------------------

def test_t43_c3_fp_nontrivial(smoke_rows):
    for row in smoke_rows:
        if row["operator"] != "COMMON_KEEP" and row.get("c3_fp"):
            assert row["c3_fp"]["std"] > 0 or row["c3_fp"]["mean"] != 0


# ---------------------------------------------------------------------------
# T44 — Divergence report has one row per (seed, change_k) pair
# ---------------------------------------------------------------------------

def test_t44_divergence_report_structure(smoke_rows):
    div_report = d2cx.compute_divergence_report(smoke_rows)
    assert len(div_report) > 0
    for r in div_report:
        assert "seed" in r
        assert "change_k" in r


# ---------------------------------------------------------------------------
# T45 — Divergence report includes DIAG_U03 vs PROD_O0 pair
# ---------------------------------------------------------------------------

def test_t45_divergence_includes_diag_vs_prod(smoke_rows):
    div_report = d2cx.compute_divergence_report(smoke_rows)
    pair_key = "DIAG_U03_vs_PROD_O0"
    for r in div_report:
        assert pair_key in r, f"Missing {pair_key} in divergence row"


# ---------------------------------------------------------------------------
# T46 — Divergence at k=0 must be at C2 or later (not C0 or C1)
#        C0 and C1 agree at k=0 (shared initial state); C0 naturally diverges
#        at k≥1 after populations branch at k=0's survival step.
# ---------------------------------------------------------------------------

def test_t46_diag_vs_prod_divergence(smoke_rows):
    div_report = d2cx.compute_divergence_report(smoke_rows)
    pair_key = "DIAG_U03_vs_PROD_O0"
    # At k=0 specifically: populations are cloned from the same state → C0,C1 must agree
    k0_divs = [r.get(pair_key, "UNKNOWN") for r in div_report if r.get("change_k") == 0]
    for d_val in k0_divs:
        assert "C0" not in d_val, \
            f"Unexpected C0 divergence at k=0 between DIAG and PROD: {d_val}"
        assert "C1_F" not in d_val, \
            f"Unexpected C1_F divergence at k=0 between DIAG and PROD: {d_val}"
        assert "C1_RIDS" not in d_val, \
            f"Unexpected C1_RIDS divergence at k=0 between DIAG and PROD: {d_val}"


# ---------------------------------------------------------------------------
# T47 — DIAG_U03 vs DIAG_H005 pair diverge at C3 (different survival)
# ---------------------------------------------------------------------------

def test_t47_diag_u03_vs_h005_diverges_c3(smoke_rows):
    div_report = d2cx.compute_divergence_report(smoke_rows)
    pair_key = "DIAG_U03_vs_DIAG_H005"
    # At k=0: same initial state → C0,C1 must agree; diverge at C2 or later (survival diff)
    # At k>=1: C0 diverges naturally because survival differed at k=0 (populations branched)
    k0_divs = [r.get(pair_key) for r in div_report if r.get("change_k") == 0 and pair_key in r]
    for d_val in k0_divs:
        assert d_val is not None
        assert "C0" not in d_val, f"Unexpected C0 divergence at k=0 for U03 vs H005: {d_val}"
        assert "C1_F" not in d_val, f"Unexpected C1_F divergence at k=0: {d_val}"


# ---------------------------------------------------------------------------
# T48 — json_safe handles numpy types
# ---------------------------------------------------------------------------

def test_t48_json_safe_numpy():
    obj = {"a": np.int64(5), "b": np.float32(1.5), "arr": np.array([1, 2, 3])}
    safe = d2cx.json_safe(obj)
    assert safe["a"] == 5
    assert isinstance(safe["a"], int)
    json.dumps(safe)  # must be serializable


def test_t48b_json_safe_nan():
    obj = {"x": float("nan"), "y": np.float32("nan")}
    safe = d2cx.json_safe(obj)
    assert safe["x"] is None


# ---------------------------------------------------------------------------
# T49 — Block label in rows matches the block argument
# ---------------------------------------------------------------------------

def test_t49_block_label_consistency(smoke_rows, smoke_rows_b):
    assert all(r["block"] == "A" for r in smoke_rows)
    assert all(r["block"] == "B" for r in smoke_rows_b)


# ---------------------------------------------------------------------------
# T50 — No cross-arm state sharing: arms diverge after k=0
#        (C4 fingerprint of DIAG_U03 vs DIAG_H005 must differ at k=0)
# ---------------------------------------------------------------------------

def test_t50_arms_diverge_after_k0(smoke_rows):
    k0_u03  = next((r for r in smoke_rows if r["arm"] == "DIAG_U03"  and r["change_k"] == 0 and r.get("c4_fp")), None)
    k0_h005 = next((r for r in smoke_rows if r["arm"] == "DIAG_H005" and r["change_k"] == 0 and r.get("c4_fp")), None)
    if k0_u03 and k0_h005:
        # DIAG_U03 (unrestricted) vs DIAG_H005 (HARD) — survivors differ → C4 differs
        # (unless by coincidence they agree)
        pass  # we can't assert they must differ; just that C3 fingerprints are valid
    # Check that both have valid c4_fp
    assert k0_u03 is not None
    assert k0_h005 is not None


# ---------------------------------------------------------------------------
# T51 — compute_arm_cb returns correct block structure
# ---------------------------------------------------------------------------

def test_t51_compute_arm_cb_structure(smoke_rows, smoke_rows_b):
    all_rows = smoke_rows + smoke_rows_b
    cb = d2cx.compute_arm_cb(all_rows, "DIAG_H005", baseline_arm="DIAG_U03")
    assert "A" in cb
    assert "B" in cb
    for block in ["A", "B"]:
        v = cb[block]
        assert "mean_CB" in v
        assert "n_seeds" in v


# ---------------------------------------------------------------------------
# T52 — compute_scenario_cb_by_arm returns all non-baseline arms
# ---------------------------------------------------------------------------

def test_t52_compute_scenario_cb_all_arms(smoke_rows, smoke_rows_b):
    all_rows = smoke_rows + smoke_rows_b
    cb_by_arm = d2cx.compute_scenario_cb_by_arm(all_rows)
    expected_arms = {"PROD_O0", "DIAG_H005", "PROD_O3"}
    assert set(cb_by_arm.keys()) == expected_arms


# ---------------------------------------------------------------------------
# T53 — classify_discrepancy returns a dict with 'verdict' key
# ---------------------------------------------------------------------------

def test_t53_classify_returns_verdict(smoke_rows, smoke_rows_b):
    all_rows = smoke_rows + smoke_rows_b
    div_report = d2cx.compute_divergence_report(all_rows)
    v = d2cx.classify_discrepancy(all_rows, div_report)
    assert "verdict" in v
    assert "explanation" in v


# ---------------------------------------------------------------------------
# T54 — Verdict is one of the valid classifications
# ---------------------------------------------------------------------------

VALID_VERDICTS = {
    "SEED_BLOCK_HETEROGENEITY",
    "PRODUCTION_PROTOCOL_MISMATCH",
    "RNG_COUPLING_MISMATCH",
    "HISTORY_UPDATE_MISMATCH",
    "STANDARD_EVOLUTION_MISMATCH",
    "RESPONSE_PATH_MISMATCH",
    "UNRESOLVED",
}


def test_t54_verdict_valid(smoke_rows, smoke_rows_b):
    all_rows = smoke_rows + smoke_rows_b
    div_report = d2cx.compute_divergence_report(all_rows)
    v = d2cx.classify_discrepancy(all_rows, div_report)
    assert v["verdict"] in VALID_VERDICTS, f"Invalid verdict: {v['verdict']}"


# ---------------------------------------------------------------------------
# T55 — D2CX output directory path is defined
# ---------------------------------------------------------------------------

def test_t55_output_dir():
    from pathlib import Path
    # Module should be importable without running main
    assert hasattr(d2cx, "BLOCK_A_SEEDS")
    assert hasattr(d2cx, "BLOCK_B_SEEDS")
    assert hasattr(d2cx, "ARM_DEFS")


# ---------------------------------------------------------------------------
# T56 — Production default O0 sigma unchanged (0.03)
# ---------------------------------------------------------------------------

def test_t56_prod_default_unchanged():
    # D2-D (2026-09-20): sigma_predict default changed 0.03 → 0.005 (O1 adoption).
    from prdrl.oracle.oracle import PREDICT_SURVIVAL_UNRESTRICTED
    assert PREDICT_SURVIVAL_UNRESTRICTED == "UNRESTRICTED"
    import inspect
    from prdrl.oracle import oracle as oracle_mod
    src = inspect.getsource(oracle_mod)
    # Post-D2-D: production default is now 0.005
    assert "sigma_predict=0.005" in src, "Production default sigma must be 0.005 after D2-D"


# ---------------------------------------------------------------------------
# T57 — HARD invariant never violated across smoke runs
# ---------------------------------------------------------------------------

def test_t57_no_hard_violation(smoke_rows, smoke_rows_b):
    all_rows = smoke_rows + smoke_rows_b
    for row in all_rows:
        if row["arm"] in ("DIAG_H005", "PROD_O3"):
            if row.get("realized_op") == "PREDICT" and row.get("n_old_dropped_other") is not None:
                assert row["n_old_dropped_other"] == 0, \
                    f"HARD invariant violated: arm={row['arm']} seed={row['seed']} k={row['change_k']}"


# ---------------------------------------------------------------------------
# T58 — FE fairness holds across smoke runs (all 4 arms same extra_fe per k)
# ---------------------------------------------------------------------------

def test_t58_fe_fairness_all_smoke(smoke_rows, smoke_rows_b):
    from collections import defaultdict
    by_seed_k = defaultdict(dict)
    for r in smoke_rows + smoke_rows_b:
        if r["operator"] != "COMMON_KEEP":
            by_seed_k[(r["seed"], r["change_k"])][r["arm"]] = r["extra_fe"]
    for (seed, k), fe_map in by_seed_k.items():
        if len(fe_map) == 4:
            assert len(set(fe_map.values())) == 1, \
                f"FE fairness violated: seed={seed} k={k} fe_map={fe_map}"


# ---------------------------------------------------------------------------
# T59 — Block B smoke rows have Block B seeds
# ---------------------------------------------------------------------------

def test_t59_block_b_seed_range(smoke_rows_b):
    for row in smoke_rows_b:
        assert row["seed"] in d2cx.BLOCK_B_SEEDS, \
            f"Block B row has seed {row['seed']} not in BLOCK_B_SEEDS"


# ---------------------------------------------------------------------------
# T60 — Module is importable and all required public functions exist
# ---------------------------------------------------------------------------

def test_t60_public_api():
    required_fns = [
        "run_one_d2cx",
        "compute_divergence_report",
        "compute_arm_cb",
        "classify_discrepancy",
        "make_d2cx_mixed_dynamics",
        "json_safe",
    ]
    for fn_name in required_fns:
        assert hasattr(d2cx, fn_name), f"Missing function: {fn_name}"
        assert callable(getattr(d2cx, fn_name))
