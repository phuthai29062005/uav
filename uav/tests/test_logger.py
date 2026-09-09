import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from experiment_logger import ExperimentLogger


def test_success_run_writes_one_line(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    with logger.run(algo="TestAlgo", problem="DF1",
                    seed=42, params={"N": 100}):
        logger.record(migd=0.05, hv=3.1, fes=500, feasible_ratio=1.0)

    lines = open(path).read().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    for field in ExperimentLogger.FIELDS:
        assert field in row, f"missing field: {field}"
    assert row["status"] == "success"
    assert row["runtime_sec"] > 0
    assert row["migd"] == 0.05
    assert row["error_msg"] is None


def test_error_run_ghi_traceback_va_reraise(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    with pytest.raises(ValueError):
        with logger.run(algo="A", problem="DF2",
                        seed=1, params={}):
            raise ValueError("test")

    lines = open(path).read().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] == "error"
    assert "ValueError" in row["error_msg"]
    assert "test" in row["error_msg"]
    assert row["runtime_sec"] > 0


def test_multiple_runs_append(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    for i in range(3):
        with logger.run(algo="A", problem="DF1",
                        seed=i, params={}):
            logger.record(migd=float(i))

    lines = open(path).read().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert row["seed"] == i


def test_record_only_allowed_fields(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    with logger.run(algo="A", problem="DF1", seed=0, params={}):
        with pytest.raises(ValueError):
            logger.record(foo=1)
        logger.record(migd=0.1)


def test_record_converts_numpy(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    with logger.run(algo="A", problem="DF1", seed=0, params={}):
        logger.record(migd=np.float64(0.5), fes=np.int64(1000))

    row = json.loads(open(path).readline())
    assert type(row["migd"]) is float
    assert type(row["fes"]) is int


def test_record_outside_context_raises(tmp_path):
    path = str(tmp_path / "out.jsonl")
    logger = ExperimentLogger(path)

    with pytest.raises(RuntimeError):
        logger.record(migd=0.1)
