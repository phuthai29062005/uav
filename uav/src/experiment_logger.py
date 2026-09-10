import json
import os
import time
import traceback
import subprocess
import socket
import threading
from contextlib import contextmanager
from datetime import datetime, timezone


def _get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=True
        )
        return result.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


class ExperimentLogger:
    FIELDS = [
        "timestamp", "algo", "problem", "seed", "params",
        "runtime_sec", "fes", "migd", "hv", "feasible_ratio",
        "status", "error_msg", "git_commit", "hostname",
        "raw_change", "normalized_change", "d_mem", "has_memory",
        "sig_saturation", "gate_action", "seg_actions",
    ]

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._hostname = socket.gethostname()
        self._git_commit = _get_git_commit()
        self._pending = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @contextmanager
    def run(self, algo, problem, seed, params):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algo": algo,
            "problem": problem,
            "seed": int(seed),
            "params": params,
            "runtime_sec": None,
            "fes": None,
            "migd": None,
            "hv": None,
            "feasible_ratio": None,
            "status": "success",
            "error_msg": None,
            "git_commit": self._git_commit,
            "hostname": self._hostname,
            "raw_change": None,
            "normalized_change": None,
            "d_mem": None,
            "has_memory": None,
            "sig_saturation": None,
            "gate_action": None,
            "seg_actions": None,
        }
        self._pending = entry
        t0 = time.perf_counter()
        try:
            yield self
            entry["runtime_sec"] = time.perf_counter() - t0
        except Exception:
            entry["runtime_sec"] = time.perf_counter() - t0
            entry["status"] = "error"
            entry["error_msg"] = traceback.format_exc()
            self._write(entry)
            self._pending = None
            raise
        self._write(entry)
        self._pending = None

    def record(self, **kwargs):
        if self._pending is None:
            raise RuntimeError("record() must be called inside run() context")
        allowed = {"migd", "hv", "fes", "feasible_ratio",
                   "raw_change", "normalized_change",
                   "d_mem", "has_memory", "sig_saturation",
                   "gate_action", "seg_actions"}
        for k, v in kwargs.items():
            if k not in allowed:
                raise ValueError(f"Unknown record field: {k}. "
                                 f"Allowed: {allowed}")
            if hasattr(v, "item"):
                v = v.item()
            self._pending[k] = v

    def _write(self, entry):
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
