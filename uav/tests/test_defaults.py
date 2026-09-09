"""
Test D2: so seed danh gia mac dinh >= 30 (yeu cau MSO).

argparse `--runs` gio tham chieu default=N_EVAL_RUNS, nen kiem tra
hang so nay la du (no la nguon su that duy nhat cho gia tri mac dinh).

Chay:  python -m pytest tests/test_defaults.py -v
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import train  # noqa: E402


def test_default_runs_is_30():
    assert train.N_EVAL_RUNS == 30, (
        f"N_EVAL_RUNS phai = 30 (MSO toi thieu 30 seed), dang = "
        f"{train.N_EVAL_RUNS}"
    )
