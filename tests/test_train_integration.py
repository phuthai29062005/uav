import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import train


def test_online_writes_log(tmp_path):
    log_path = str(tmp_path / "online.jsonl")
    train.mode_online(["DF1"], n_runs=2, n_changes=3,
                      log_path=log_path)

    assert os.path.exists(log_path)
    lines = open(log_path).read().strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        row = json.loads(line)
        assert row["algo"] == "SA-DRL-DMOEA-online"
        assert row["status"] == "success"
        assert np.isfinite(row["migd"])
        assert np.isfinite(row["fes"])
        assert np.isfinite(row["hv"])
        assert row["params"]["eps_fixed"] == 0.3


def test_baseline_writes_log(tmp_path):
    log_path = str(tmp_path / "baseline.jsonl")
    train.mode_baseline(["DF1"], n_runs=2, n_changes=3,
                        log_path=log_path)

    lines = open(log_path).read().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["algo"] == "NSGA2-baseline"


def test_log_path_override(tmp_path):
    custom_path = str(tmp_path / "custom.jsonl")
    train.mode_baseline(["DF1"], n_runs=1, n_changes=3,
                        log_path=custom_path)

    assert os.path.exists(custom_path)
    default_path = str(tmp_path / "baseline_nsga2.jsonl")
    assert not os.path.exists(default_path)


def test_fes_counter_reasonable(tmp_path):
    log_path = str(tmp_path / "fes.jsonl")
    train.mode_online(["DF1"], n_runs=1, n_changes=3,
                      log_path=log_path)

    row = json.loads(open(log_path).readline())
    fes = row["fes"]
    assert 3000 <= fes <= 10000, f"fes={fes} ngoai khoang hop ly"
