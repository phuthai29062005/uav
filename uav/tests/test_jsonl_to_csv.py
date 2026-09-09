import csv
import io
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from jsonl_to_csv import COLUMNS, convert
from experiment_logger import ExperimentLogger


def _make_entry(seed=0, **overrides):
    entry = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "algo": "TestAlgo",
        "problem": "DF1",
        "seed": seed,
        "params": {"N": 100},
        "runtime_sec": 1.5,
        "fes": 500,
        "migd": 0.05,
        "hv": 3.1,
        "feasible_ratio": 1.0,
        "status": "success",
        "error_msg": None,
        "git_commit": "abc123",
        "hostname": "testhost",
    }
    entry.update(overrides)
    return entry


def test_convert_basic(tmp_path):
    jsonl = tmp_path / "data.jsonl"
    entries = [_make_entry(seed=i) for i in range(3)]
    jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    out = io.StringIO()
    n_ok, n_bad = convert(str(jsonl), out)
    assert n_ok == 3
    assert n_bad == 0

    out.seek(0)
    reader = csv.DictReader(out)
    assert list(reader.fieldnames) == COLUMNS

    rows = list(reader)
    assert len(rows) == 3
    parsed_params = json.loads(rows[0]["params"])
    assert isinstance(parsed_params, dict)
    assert parsed_params["N"] == 100


def test_convert_empty_file(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")

    out = io.StringIO()
    n_ok, n_bad = convert(str(jsonl), out)
    assert n_ok == 0
    assert n_bad == 0

    out.seek(0)
    reader = csv.DictReader(out)
    assert list(reader.fieldnames) == COLUMNS
    assert list(reader) == []


def test_convert_malformed_line_skipped(tmp_path, capsys):
    jsonl = tmp_path / "bad.jsonl"
    lines = [
        json.dumps(_make_entry(seed=1)),
        '{"broken',
        json.dumps(_make_entry(seed=3)),
    ]
    jsonl.write_text("\n".join(lines) + "\n")

    out = io.StringIO()
    n_ok, n_bad = convert(str(jsonl), out)
    assert n_ok == 2
    assert n_bad == 1

    out.seek(0)
    rows = list(csv.DictReader(out))
    assert len(rows) == 2

    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "line 2" in captured.err


def test_convert_missing_fields_empty_cells(tmp_path):
    jsonl = tmp_path / "sparse.jsonl"
    sparse = {"algo": "A", "problem": "DF1", "seed": 0}
    jsonl.write_text(json.dumps(sparse) + "\n")

    out = io.StringIO()
    n_ok, _ = convert(str(jsonl), out)
    assert n_ok == 1

    out.seek(0)
    rows = list(csv.DictReader(out))
    assert rows[0]["algo"] == "A"
    assert rows[0]["migd"] == ""
    assert rows[0]["runtime_sec"] == ""


def test_convert_end_to_end_from_logger(tmp_path):
    jsonl_path = str(tmp_path / "e2e.jsonl")
    logger = ExperimentLogger(jsonl_path)

    with logger.run(algo="SA-DRL", problem="DF1",
                    seed=100, params={"N": 100}):
        logger.record(migd=0.05, hv=3.1, fes=500, feasible_ratio=1.0)

    with pytest.raises(ValueError):
        with logger.run(algo="SA-DRL", problem="DF2",
                        seed=200, params={}):
            raise ValueError("boom")

    out = io.StringIO()
    n_ok, n_bad = convert(jsonl_path, out)
    assert n_ok == 2
    assert n_bad == 0

    out.seek(0)
    rows = list(csv.DictReader(out))
    assert len(rows) == 2
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == "error"
    assert "boom" in rows[1]["error_msg"]
