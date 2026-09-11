"""
4.10A — CEC2018 DF1-DF14 segmentation audit (READ-ONLY).

Khoa lai cac su kien ve vai tro bien (positional vs distance) rut ra tu
IMPLEMENTATION THUC TE cua pymoo DF, de:
  - tai lap ket qua audit,
  - bat neu ai do gia dinh current [x0]/[rest] la position/distance
    tren DF ma structure khong support.

KHONG test production behavior; chi kiem tra tinh chat cua benchmark.
"""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import (
    DF1, DF2, DF3, DF4, DF5, DF6, DF7,
    DF8, DF9, DF10, DF11, DF12, DF13, DF14)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

D = 10

# Bang audit: so positional var (parameterize PF) tu formula thuc te.
# 1 positional (x0)  -> [x0]/[rest] la position/distance HOP LE.
# 2 positional (x0,x1) tri-obj -> [x0]/[rest] dat x1 (positional) vao
#   distance block -> KHONG hop le.
# DF2 -> positional index DONG theo t -> [x0]/[rest] KHONG hop le.
EXPECTED = {
    "DF1": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF2": dict(M=2, n_pos=1, dynamic_pos=True),
    "DF3": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF4": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF5": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF6": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF7": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF8": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF9": dict(M=2, n_pos=1, dynamic_pos=False),
    "DF10": dict(M=3, n_pos=2, dynamic_pos=False),
    "DF11": dict(M=3, n_pos=2, dynamic_pos=False),
    "DF12": dict(M=3, n_pos=2, dynamic_pos=False),
    "DF13": dict(M=3, n_pos=2, dynamic_pos=False),
    "DF14": dict(M=3, n_pos=2, dynamic_pos=False),
}
CLASSES = {cls.__name__: cls for cls in [
    DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9,
    DF10, DF11, DF12, DF13, DF14]}


# ------------------------------ SEG1: objective count khop bang audit
@pytest.mark.parametrize("name", list(EXPECTED))
def test_seg1_objective_count(name):
    p = CLASSES[name](time=0.0, n_var=D)
    assert p.n_obj == EXPECTED[name]["M"]


# ------------- SEG2: DF2 positional index r(t) THAY DOI theo t
def test_seg2_df2_positional_index_is_dynamic():
    rs = []
    for k in range(11):
        t = k / 10
        G = abs(np.sin(0.5 * np.pi * t))
        rs.append(int((D - 1) * G))          # dung cong thuc trong DF2._evaluate
    assert rs[0] == 0                          # t=0 -> x0
    assert any(r != 0 for r in rs)             # nhung KHONG luon x0
    assert max(rs) == D - 1                     # quet toi x_{D-1}
    # ngay t=0.1 da lech khoi x0:
    assert rs[1] != 0


# ------- SEG3: tri-objective co 2 positional var (distance bat dau x2)
@pytest.mark.parametrize("name", ["DF10", "DF12", "DF13", "DF14"])
def test_seg3_triobj_two_positional_via_bounds(name):
    # xl[2:] bi shift (-1) danh dau distance vars -> positional = {x0,x1}.
    p = CLASSES[name](time=0.0, n_var=D)
    assert p.n_obj == 3
    assert p.xl[0] == 0.0 and p.xl[1] == 0.0     # positional trong [0,1]
    assert np.all(p.xl[2:] == -1.0)              # distance vars shifted


def test_seg3_df11_two_positional_no_bound_shift():
    # DF11 khong shift bounds nhung g dung x[2:] va f dung x0,x1 -> 2 positional.
    p = DF11(time=0.0, n_var=D)
    assert p.n_obj == 3
    assert np.all(p.xl == 0.0) and np.all(p.xu == 1.0)


# ---- SEG4: bi-obj clean co 1 positional (distance markers = x1..)
@pytest.mark.parametrize("name", ["DF3", "DF5", "DF6", "DF8", "DF9"])
def test_seg4_biobj_distance_from_x1(name):
    p = CLASSES[name](time=0.0, n_var=D)
    assert p.n_obj == 2
    assert np.all(p.xl[1:] == -1.0)              # x1.. distance (shifted)
    assert p.xl[0] == 0.0                         # x0 positional [0,1]


def test_seg4_df7_positional_range_shifted():
    # DF7: positional x0 in [1,4], distance x1.. in [0,1].
    p = DF7(time=0.0, n_var=D)
    assert p.xl[0] == 1.0 and p.xu[0] == 4.0
    assert np.all(p.xl[1:] == 0.0)


# ---- SEG5: current fixed [x0]/[rest] khop position/distance CHI cho subset
def test_seg5_fixed_partition_validity_subset():
    from train import make_segments
    seg = make_segments(D, n_seg=2)
    assert seg == [[0], list(range(1, D))]       # current partition

    valid = {n for n, e in EXPECTED.items()
             if e["n_pos"] == 1 and not e["dynamic_pos"]}
    invalid = set(EXPECTED) - valid
    # 8 bi-obj clean hop le; DF2 (dynamic) + 5 tri-obj (2 positional) KHONG.
    assert valid == {"DF1", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9"}
    assert invalid == {"DF2", "DF10", "DF11", "DF12", "DF13", "DF14"}
    assert len(valid) == 8 and len(invalid) == 6
