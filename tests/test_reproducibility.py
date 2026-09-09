"""
Test tinh tai lap (A3) — sau khi seed module `random`.

Chi co MOT bo sinh ngau nhien dung module `random` truc tiep:
ReplayBuffer.sample (src/dqn_agent.py). Con lai dung np.random.
Sau khi seed ca `random`, `np.random` va `torch` tu mot nguon seed
duy nhat, hai lan chay cung seed phai cho MIGD trung khop.

Chay:  python -m pytest tests/test_reproducibility.py -v
"""
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import train  # noqa: E402  (import sau khi vao sys.path)

TOL = 1e-6  # khop den 6 chu so thap phan


def _fmt(xs):
    return "[" + ", ".join(f"{x:.6f}" for x in xs) + "]"


def test_online_reproducible(tmp_path):
    """mode online, 2 lan cung seed -> MIGD khop den 6 chu so."""
    r1 = train.mode_online(["DF1"], n_runs=2, n_changes=5,
                           log_path=str(tmp_path / "r1.jsonl"))
    r2 = train.mode_online(["DF1"], n_runs=2, n_changes=5,
                           log_path=str(tmp_path / "r2.jsonl"))
    m1, m2 = r1["DF1"][2], r2["DF1"][2]   # danh sach MIGD moi run

    assert len(m1) == len(m2) == 2
    diffs = [abs(a - b) for a, b in zip(m1, m2)]
    assert max(diffs) < TOL, (
        "MIGD hai lan chay KHAC nhau du cung seed:\n"
        f"  lan 1 = {_fmt(m1)}\n"
        f"  lan 2 = {_fmt(m2)}\n"
        f"  |chenh| = {_fmt(diffs)}"
    )


@pytest.mark.slow  # ~36s (> 30s)
def test_debug_reproducible(monkeypatch):
    """
    Nhanh huan luyen cua mode debug (train_on) chay 2 lan cung seed
    -> log MIGD cua TAT CA episode phai khop element-wise den 6 chu so.

    Dung train_on truc tiep vi mode_debug goi dung ham nay de huan
    luyen, va chi train_on moi tra ve log MIGD tung episode.
    Rut TRAIN_CHANGES xuong cho test chay nhanh.
    """
    monkeypatch.setattr(train, "TRAIN_CHANGES", 3)
    segs = train.make_segments(train.CFG["D"])

    _, log1 = train.train_on(["DF1"], 4, segs)
    _, log2 = train.train_on(["DF1"], 4, segs)
    mig1 = [r["migd"] for r in log1]
    mig2 = [r["migd"] for r in log2]

    assert len(mig1) == len(mig2) == 4
    diffs = [abs(a - b) for a, b in zip(mig1, mig2)]
    assert max(diffs) < TOL, (
        "Log MIGD hai lan train KHAC nhau du cung seed:\n"
        f"  lan 1 = {_fmt(mig1)}\n"
        f"  lan 2 = {_fmt(mig2)}\n"
        f"  |chenh| = {_fmt(diffs)}"
    )


def test_different_seed_gives_different_result(monkeypatch, tmp_path):
    """
    Sanity check: mode online voi seed KHAC nhau -> MIGD KHAC nhau.
    Neu test nay pass ma test 1&2 cung pass, ta chac reproducibility
    la that chu khong phai trivial pass (vi du moi thu bi hardcode 0).
    """
    r1 = train.mode_online(["DF1"], n_runs=2, n_changes=5,
                           log_path=str(tmp_path / "r1.jsonl"))
    monkeypatch.setattr(train, "EVAL_SEED_BASE", train.EVAL_SEED_BASE + 777)
    r2 = train.mode_online(["DF1"], n_runs=2, n_changes=5,
                           log_path=str(tmp_path / "r2.jsonl"))
    m1, m2 = r1["DF1"][2], r2["DF1"][2]

    diffs = [abs(a - b) for a, b in zip(m1, m2)]
    assert max(diffs) > TOL, (
        "Seed khac nhau nhung MIGD lai trung (nghi ngo seed khong co tac dung):\n"
        f"  seed base goc  = {_fmt(m1)}\n"
        f"  seed base +777 = {_fmt(m2)}"
    )
