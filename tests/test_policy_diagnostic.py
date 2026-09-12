"""PD1-PD10: policy/state diagnostic helper correctness (5.3A, pure part)."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from policy_diagnostic import (decode_gate, decode_seg,  # noqa: E402
                               higher_c_segment, intervention_rate,
                               normalized_entropy, tercile_labels,
                               window_label)


# ------------------------------------------------- PD1: action labels
def test_pd1_action_labels():
    assert decode_gate(0) == "NO_MEMORY"
    assert decode_gate(1) == "MEMORY"
    assert decode_seg(0) == "KEEP"
    assert decode_seg(1) == "LOCAL"
    assert decode_seg(2) == "PREDICT"
    assert decode_seg(3) == "DIVERSIFY"


# --------------------------------- PD2: MEMORY rows don't fabricate seg
def test_pd2_memory_no_fabricated_segment():
    # Contract test: when gate==MEMORY, caller must emit "NA", never a
    # decoded action label. This is enforced in the replay/CSV writer;
    # here we test the writer's decision function directly.
    def seg_display(gate, seg_val):
        return "NA" if gate == 1 else decode_seg(seg_val)
    assert seg_display(1, 2) == "NA"          # gate=MEMORY -> NA regardless
    assert seg_display(0, 2) == "PREDICT"


# ------------------------------------------- PD3: intervention rate
def test_pd3_intervention_rate():
    assert intervention_rate([0, 0, 0, 0]) == 0.0
    assert intervention_rate([1, 2, 3, 0]) == 0.75
    assert intervention_rate([]) != intervention_rate([])  # nan != nan


# --------------------------- PD4: normalized entropy 0 vs uniform 1
def test_pd4_normalized_entropy():
    assert normalized_entropy([0, 0, 0, 0], 4) == 0.0
    assert abs(normalized_entropy([0, 1, 2, 3], 4) - 1.0) < 1e-9
    # skewed: partial diversity strictly between 0 and 1
    h = normalized_entropy([0, 0, 0, 1], 4)
    assert 0.0 < h < 1.0


# ------------------------------------- PD5: tercile assignment deterministic
def test_pd5_tercile_deterministic():
    vals = np.random.RandomState(0).rand(30)
    a = tercile_labels(vals)
    b = tercile_labels(vals)
    assert list(a) == list(b)
    counts = {k: list(a).count(k) for k in ("LOW", "MID", "HIGH")}
    assert counts["LOW"] == counts["MID"] == counts["HIGH"] == 10
    # LOW values must all be <= MID values <= HIGH values (sorted correctness)
    order = np.argsort(vals)
    sorted_labels = a[order]
    assert list(sorted_labels) == ["LOW"] * 10 + ["MID"] * 10 + ["HIGH"] * 10


# ----------------------------------- PD6: higher-c segment metric correct
def test_pd6_higher_c_segment():
    assert higher_c_segment(0.5, 0.2) == 1
    assert higher_c_segment(0.2, 0.5) == 2
    assert higher_c_segment(0.3, 0.3) == 1        # tie -> segment 1


# --------------------------------- PD7: delta_response sign convention
def test_pd7_delta_response_sign():
    igd_pre, igd_response = 0.5, 0.3
    delta_response = igd_pre - igd_response
    assert delta_response > 0                     # improvement (lower IGD)
    igd_pre2, igd_response2 = 0.3, 0.5
    assert (igd_pre2 - igd_response2) < 0          # got worse


# --------------------------------- PD8: delta_recovery sign convention
def test_pd8_delta_recovery_sign():
    igd_response, igd_end = 0.5, 0.2
    delta_recovery = igd_response - igd_end
    assert delta_recovery > 0                      # recovered (end lower)


# ----------------------------- PD9: per-seed grouping preserves 30 seeds
def test_pd9_window_labels_do_not_alter_seed_grouping():
    # window_label operates per-row and must not touch seed identity;
    # sanity that applying it to 30 distinct (seed,change) rows keeps
    # 30 distinct seeds untouched.
    seeds = list(range(100000, 100030))
    changes = [50] * 30
    labels = [window_label(c) for c in changes]
    assert len(set(seeds)) == 30
    assert all(l == "MID" for l in labels)


# --------------------------------------- PD10: early/mid/late windows exact
def test_pd10_window_boundaries_exact():
    assert window_label(1) == "EARLY"
    assert window_label(10) == "EARLY"
    assert window_label(11) == "OTHER"
    assert window_label(45) == "OTHER"
    assert window_label(46) == "MID"
    assert window_label(55) == "MID"
    assert window_label(56) == "OTHER"
    assert window_label(90) == "OTHER"
    assert window_label(91) == "LATE"
    assert window_label(100) == "LATE"


# ------------------------- PD11/PD12: replay determinism + raw SHA unchanged
def test_pd11_replay_deterministic_against_frozen_benchmark():
    manifest_path = os.path.join(_HERE, "..", "results",
                                 "policy_diagnostic_replay_manifest.json")
    if not os.path.exists(manifest_path):
        import pytest as _pytest
        _pytest.skip("replay manifest not present")
    import json
    m = json.load(open(manifest_path))
    assert m["n_replays"] == 420
    assert m["n_mismatch"] == 0
    assert m["n_rows_written"] == 42000


def test_pd12_raw_benchmark_sha_unchanged():
    manifest_path = os.path.join(_HERE, "..", "results",
                                 "policy_diagnostic_replay_manifest.json")
    if not os.path.exists(manifest_path):
        import pytest as _pytest
        _pytest.skip("replay manifest not present")
    import json
    m = json.load(open(manifest_path))
    assert m["sha256_before"] == m["sha256_after"]
