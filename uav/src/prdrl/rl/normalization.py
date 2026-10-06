"""
Versioned state normalizers for PR-DRL-DMOEA.

Two versions, byte-for-byte separable, so old checkpoints keep V1 and any future
V2 experiment keeps V2 (never cross them):

  NORMALIZATION_V1_LEGACY          — the exact E4-P/E4TR/E4F production mapping.
  ROBUST_QUANTILE_NORMALIZATION_V2 — per-dimension robust (q01/q99) minmax, fit from
                                     E4TR TRAINING states only; occupancy/active-fraction
                                     identity; delta_severity sign/zero-preserving.

State semantics are UNCHANGED between versions — only the numeric mapping differs.
Raw 9-vector order (must match compute_global_context_v1 + V2 local state):
  0 severity_r            (local)
  1 convergence_loss_r    (local)
  2 occupancy_r           (local)
  3 delta_severity_r      (local)
  4 active_region_fraction(global)
  5 mean_severity         (global)
  6 max_severity          (global)
  7 mean_convergence_loss (global)
  8 max_convergence_loss  (global)
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

import numpy as np

NORM_V1 = "NORMALIZATION_V1_LEGACY"
NORM_V2 = "ROBUST_QUANTILE_NORMALIZATION_V2"          # DESIGN_REJECTED_BY_E4H_SATURATION_GATE
NORM_V3 = "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"

EPS = 1e-12
RAW_ORDER = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r",
             "active_region_fraction", "mean_severity", "max_severity",
             "mean_convergence_loss", "max_convergence_loss"]

# V2 dimension roles (frozen in E4H design)
ROBUST_MINMAX_DIMS = [0, 1, 5, 6, 7, 8]   # own q01/q99 robust minmax -> [0,1]
IDENTITY_DIMS = [2, 4]                     # occupancy, active_region_fraction: clip[0,1]
DELTA_DIM = 3                             # delta_severity: x/(q99_abs) clip[-1,1]


# ---------------------------------------------------------------------------
# V1 LEGACY — exact replica of E4-P normalize_local_state + normalize_global_context
# ---------------------------------------------------------------------------

def normalize_v1(raw9: np.ndarray) -> np.ndarray:
    """Exact E4-P/E4TR/E4F production normalization (must never change)."""
    x = np.asarray(raw9, dtype=np.float64)
    z = np.zeros(9, dtype=np.float32)
    z[0] = float(np.clip(x[0], 0.0, 1.0))            # severity_r          identity[0,1]
    z[1] = float(np.clip(x[1], 0.0, 10.0)) / 10.0    # convergence_loss_r  clip[0,10]/10
    z[2] = float(np.clip(x[2], 0.0, 1.0))            # occupancy_r         identity
    z[3] = float(np.clip(x[3], -1.0, 1.0))           # delta_severity_r    clip[-1,1]
    z[4] = float(np.clip(x[4], 0.0, 1.0))            # active_region_fraction
    z[5] = float(np.clip(x[5], 0.0, 1.0))            # mean_severity
    z[6] = float(np.clip(x[6], 0.0, 1.0))            # max_severity
    z[7] = float(np.clip(x[7], 0.0, 10.0)) / 10.0    # mean_convergence_loss clip[0,10]/10
    z[8] = float(np.clip(x[8], 0.0, 10.0)) / 10.0    # max_convergence_loss  clip[0,10]/10
    return z


# ---------------------------------------------------------------------------
# V2 ROBUST QUANTILE
# ---------------------------------------------------------------------------

class RobustQuantileNormalizerV2:
    """Per-dimension robust normalizer. Constants fit from TRAINING states only."""

    def __init__(self, constants: Dict):
        self.constants = constants
        # constants["robust"][str(dim)] = {"q01":.., "q99":..}; constants["delta_q99_abs"]
        self._robust = {int(k): v for k, v in constants["robust"].items()}
        self._delta_q99_abs = float(constants["delta_q99_abs"])

    def transform(self, raw9: np.ndarray) -> np.ndarray:
        x = np.asarray(raw9, dtype=np.float64)
        z = np.zeros(9, dtype=np.float32)
        for d in ROBUST_MINMAX_DIMS:
            q01 = self._robust[d]["q01"]; q99 = self._robust[d]["q99"]
            z[d] = float(np.clip((x[d] - q01) / (q99 - q01 + EPS), 0.0, 1.0))
        z[2] = float(np.clip(x[2], 0.0, 1.0))                       # occupancy identity
        z[4] = float(np.clip(x[4], 0.0, 1.0))                       # active fraction identity
        z[DELTA_DIM] = float(np.clip(x[DELTA_DIM] / (self._delta_q99_abs + EPS), -1.0, 1.0))
        return z

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = np.zeros_like(X, dtype=np.float32)
        for d in ROBUST_MINMAX_DIMS:
            q01 = self._robust[d]["q01"]; q99 = self._robust[d]["q99"]
            Z[:, d] = np.clip((X[:, d] - q01) / (q99 - q01 + EPS), 0.0, 1.0)
        Z[:, 2] = np.clip(X[:, 2], 0.0, 1.0)
        Z[:, 4] = np.clip(X[:, 4], 0.0, 1.0)
        Z[:, DELTA_DIM] = np.clip(X[:, DELTA_DIM] / (self._delta_q99_abs + EPS), -1.0, 1.0)
        return Z

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "RobustQuantileNormalizerV2":
        """Fit robust constants from training raw states (Nx9). Raises on degeneracy."""
        X = np.asarray(X_train, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 9
        robust = {}
        degenerate = []
        for d in ROBUST_MINMAX_DIMS:
            q01 = float(np.percentile(X[:, d], 1))
            q99 = float(np.percentile(X[:, d], 99))
            if (q99 - q01) <= 1e-12:
                degenerate.append(RAW_ORDER[d])
            robust[str(d)] = {"q01": q01, "q99": q99}
        delta_q99_abs = float(np.percentile(np.abs(X[:, DELTA_DIM]), 99))
        if delta_q99_abs <= 1e-12:
            degenerate.append(RAW_ORDER[DELTA_DIM])
        constants = {"robust": robust, "delta_q99_abs": delta_q99_abs,
                     "eps": EPS, "n_samples": int(X.shape[0]),
                     "degenerate_features": degenerate}
        return cls(constants)

    def to_dict(self) -> Dict:
        return self.constants

    def constants_sha256(self) -> str:
        payload = json.dumps({"robust": {str(k): self._robust[k] for k in sorted(self._robust)},
                              "delta_q99_abs": self._delta_q99_abs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# V3 ZERO-AWARE ROBUST LOG  (E4HR candidate; supersedes rejected V2 for training)
# ---------------------------------------------------------------------------

ZERO_INFLATED_DIMS = [0, 1, 5, 6, 7, 8]   # nonneg zero-inflated features
MIN_POSITIVE_SAMPLES = 100                 # frozen minimum positive support per dim


class ZeroAwareRobustLogNormalizerV3:
    """Zero-aware robust log normalizer.

    Nonnegative zero-inflated dims (0,1,5,6,7,8): x<=0 -> 0 (structural zero kept);
      x>0 -> clip(log1p(x/m)/log1p(c/m), 0, 1), m=median(positive), c=q99(positive).
    occupancy(2), active_region_fraction(4): identity clip[0,1].
    delta_severity(3): signed clip(x/(a3+eps), -1, 1), a3=q99(|x| : |x|>0), zero/sign preserved.
    """

    def __init__(self, constants: Dict):
        self.constants = constants
        self._m = {int(k): float(v["m"]) for k, v in constants["zero_aware"].items()}
        self._c = {int(k): float(v["c"]) for k, v in constants["zero_aware"].items()}
        self._delta_a3 = float(constants["delta_q99_abs_positive"])

    def _z_nonneg(self, x, d):
        if x <= 0.0:
            return 0.0
        m, c = self._m[d], self._c[d]
        return float(np.clip(np.log1p(x / m) / np.log1p(c / m), 0.0, 1.0))

    def transform(self, raw9: np.ndarray) -> np.ndarray:
        x = np.asarray(raw9, dtype=np.float64)
        z = np.zeros(9, dtype=np.float32)
        for d in ZERO_INFLATED_DIMS:
            z[d] = self._z_nonneg(x[d], d)
        z[2] = float(np.clip(x[2], 0.0, 1.0))
        z[4] = float(np.clip(x[4], 0.0, 1.0))
        z[DELTA_DIM] = float(np.clip(x[DELTA_DIM] / (self._delta_a3 + EPS), -1.0, 1.0))
        return z

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = np.zeros_like(X, dtype=np.float32)
        for d in ZERO_INFLATED_DIMS:
            m, c = self._m[d], self._c[d]
            pos = X[:, d] > 0.0
            Z[pos, d] = np.clip(np.log1p(X[pos, d] / m) / np.log1p(c / m), 0.0, 1.0)
            # x<=0 stays 0 (structural zero)
        Z[:, 2] = np.clip(X[:, 2], 0.0, 1.0)
        Z[:, 4] = np.clip(X[:, 4], 0.0, 1.0)
        Z[:, DELTA_DIM] = np.clip(X[:, DELTA_DIM] / (self._delta_a3 + EPS), -1.0, 1.0)
        return Z

    @classmethod
    def fit(cls, X_train: np.ndarray, min_positive: int = MIN_POSITIVE_SAMPLES):
        """Fit from training raw states (Nx9). Records validity; never silently substitutes."""
        X = np.asarray(X_train, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 9
        zero_aware = {}
        invalid = []
        for d in ZERO_INFLATED_DIMS:
            col = X[:, d]
            pos = col[col > 0.0]
            n_pos = int(pos.shape[0])
            m = float(np.median(pos)) if n_pos > 0 else 0.0
            c = float(np.percentile(pos, 99)) if n_pos > 0 else 0.0
            valid = (n_pos >= min_positive and m > 0.0 and c > 0.0
                     and c >= m and np.log1p(c / m + 1e-300) > 1e-12)
            if not valid:
                invalid.append(RAW_ORDER[d])
            zero_aware[str(d)] = {"m": m, "c": c, "n_positive": n_pos,
                                  "structural_zero_fraction": float(np.mean(col <= 0.0)),
                                  "valid": bool(valid)}
        dcol = np.abs(X[:, DELTA_DIM])
        dpos = dcol[dcol > 0.0]
        a3 = float(np.percentile(dpos, 99)) if dpos.shape[0] > 0 else 0.0
        if a3 <= 0.0:
            invalid.append(RAW_ORDER[DELTA_DIM])
        constants = {"zero_aware": zero_aware, "delta_q99_abs_positive": a3,
                     "min_positive_samples": min_positive, "eps": EPS,
                     "n_samples": int(X.shape[0]), "invalid_features": invalid,
                     "V3_SCALER_VALID": "YES" if not invalid else "NO"}
        return cls(constants)

    def to_dict(self) -> Dict:
        return self.constants

    def constants_sha256(self) -> str:
        payload = json.dumps({"zero_aware": {str(k): {"m": self._m[k], "c": self._c[k]}
                                             for k in sorted(self._m)},
                              "delta_q99_abs_positive": self._delta_a3}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def normalize_state(raw9: np.ndarray, version: str, normalizer=None):
    if version == NORM_V1:
        return normalize_v1(raw9)
    if version in (NORM_V2, NORM_V3):
        if normalizer is None:
            raise ValueError(f"{version} requires a fitted normalizer")
        return normalizer.transform(raw9)
    raise ValueError(f"unknown normalization version {version}")


# ---------------------------------------------------------------------------
# Backward-compatibility guard
# ---------------------------------------------------------------------------

def check_normalization_compatibility(checkpoint_meta: Dict, expected_version: str,
                                      expected_constants_sha256: str = None) -> None:
    """Raise if a checkpoint's normalization version/constants mismatch expectations.
    Old E4TR/E4F checkpoints (no field) are treated as V1 legacy."""
    ckpt_version = checkpoint_meta.get("normalization_version", NORM_V1)
    if ckpt_version != expected_version:
        raise ValueError(
            f"normalization version mismatch: checkpoint={ckpt_version} expected={expected_version}")
    if expected_version != NORM_V1:   # V2/V3 carry fitted constants
        ckpt_sha = checkpoint_meta.get("normalizer_constants_sha256")
        if expected_constants_sha256 is not None and ckpt_sha != expected_constants_sha256:
            raise ValueError(
                f"normalizer constants mismatch: checkpoint={ckpt_sha} expected={expected_constants_sha256}")
