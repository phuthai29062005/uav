"""Hypervolume (HV) and Mean HV (MHV).

Uses a simple exact 2D/3D implementation. For higher dimensions,
a WFG-based algorithm would be needed but our benchmarks are 2-3 obj.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def hypervolume(approx: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact hypervolume for 2D or 3D objective space.
    approx: (N, M), ref_point: (M,) -> scalar."""
    if len(approx) == 0:
        return 0.0
    approx = np.atleast_2d(approx)
    ref_point = np.asarray(ref_point)
    n_obj = approx.shape[1]
    dominated = np.all(approx <= ref_point, axis=1)
    approx = approx[dominated]
    if len(approx) == 0:
        return 0.0
    if n_obj == 2:
        return _hv_2d(approx, ref_point)
    elif n_obj == 3:
        return _hv_3d(approx, ref_point)
    raise ValueError(f"Hypervolume not implemented for {n_obj} objectives")


def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    sorted_idx = np.argsort(points[:, 0])
    points = points[sorted_idx]
    hv = 0.0
    prev_y = ref[1]
    for i in range(len(points)):
        if i < len(points) - 1:
            width = points[i + 1, 0] - points[i, 0]
        else:
            width = ref[0] - points[i, 0]
        height = prev_y - points[i, 1]
        if height > 0 and width > 0:
            hv += width * height
        prev_y = min(prev_y, points[i, 1])
    last_width = ref[0] - points[-1, 0]
    last_height = ref[1] - points[-1, 1]
    sorted_by_f1 = points[np.argsort(points[:, 0])]
    hv = 0.0
    sorted_by_f2 = points[np.argsort(points[:, 1])]
    sorted_idx = np.argsort(points[:, 0])
    pts = points[sorted_idx]
    hv = 0.0
    last_f2 = ref[1]
    for i in range(len(pts)):
        if pts[i, 1] < last_f2:
            if i < len(pts) - 1:
                dx = pts[i + 1, 0] - pts[i, 0]
            else:
                dx = ref[0] - pts[i, 0]
            hv += dx * (last_f2 - pts[i, 1])
            last_f2 = pts[i, 1]
    return hv


def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
    from itertools import product
    n = len(points)
    if n == 0:
        return 0.0
    total = np.prod(ref - points.min(axis=0))
    hv = 0.0
    sorted_z = np.argsort(points[:, 2])
    pts = points[sorted_z]
    prev_z = ref[2]
    for i in range(len(pts)):
        z_slice = pts[:i + 1]
        z_height = prev_z - pts[i, 2]
        if z_height <= 0:
            continue
        hv_2d = _hv_2d_proper(z_slice[:, :2], ref[:2])
        hv += hv_2d * z_height
        prev_z = pts[i, 2]
    return hv


def _hv_2d_proper(points: np.ndarray, ref: np.ndarray) -> float:
    """Proper non-dominated 2D HV sweep."""
    dominated = np.all(points <= ref, axis=1)
    points = points[dominated]
    if len(points) == 0:
        return 0.0
    nd = _non_dominated_2d(points)
    sorted_idx = np.argsort(nd[:, 0])
    nd = nd[sorted_idx]
    hv = 0.0
    for i in range(len(nd)):
        if i < len(nd) - 1:
            dx = nd[i + 1, 0] - nd[i, 0]
        else:
            dx = ref[0] - nd[i, 0]
        dy = ref[1] - nd[i, 1]
        hv += dx * dy
    return hv


def _non_dominated_2d(points: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(points[:, 0])
    pts = points[sorted_idx]
    nd = [pts[0]]
    min_f2 = pts[0, 1]
    for i in range(1, len(pts)):
        if pts[i, 1] < min_f2:
            nd.append(pts[i])
            min_f2 = pts[i, 1]
    return np.array(nd)


def mhv(hv_values: List[float]) -> float:
    """Mean hypervolume across time steps."""
    if not hv_values:
        return 0.0
    return float(np.mean(hv_values))
