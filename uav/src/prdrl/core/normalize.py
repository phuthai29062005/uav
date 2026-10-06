import numpy as np


def normalize(F_minus, F_plus, alpha=1e-12):
    """
    Chuan hoa objective space tren hop {F-} union {F+} (eq. 24-25).

    FIX (ISSUE-6/7 + performance): them assert N khop giua F_minus/F_plus
    (eq.22 doi hoi CUNG decision vectors o ca hai phia -- N khac nhau la
    loi ngam, khong duoc de crash am tham/sai chi so xuoi dong); assert
    N>=1 voi thong bao ro rang thay vi IndexError kho hieu; thay vong for
    long nhau bang numpy vectorized (ket qua giong het ban goc, nhanh hon
    nhieu tren E1 630 change-events).

    Edge case da xac nhan DUNG san (khong doi): mau so = 0 khi Z_nad==Z_m
    (moi nghiem giong nhau o 1 muc tieu) -- tu so cung luon = 0 trong
    truong hop do nen ket qua la 0/alpha = 0.0, khong NaN.

    Tra ve numpy.ndarray (khong con list-of-list) -- tuong thich nguoc
    100% voi moi noi goi da co (ho deu tu wrap np.array(...) sau khi goi,
    np.array(mot_array_co_san) chi la copy vo hai), dong thoi sua duoc
    ISSUE-5: partition()/severity() doi hoi ndarray, list thuan se crash.
    """
    F_minus = np.asarray(F_minus, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)

    assert len(F_minus) == len(F_plus), (
        f"F_minus va F_plus phai cung so nghiem (cung decision vectors, "
        f"eq.22): got {len(F_minus)} vs {len(F_plus)}")
    assert len(F_minus) >= 1, "normalize() can it nhat 1 nghiem, nhan duoc 0"

    stacked = np.vstack([F_minus, F_plus])
    Z_m = stacked.min(axis=0)
    Z_nad = stacked.max(axis=0)

    F_tidle_minus = (F_minus - Z_m) / (Z_nad - Z_m + alpha)
    F_tidle_plus = (F_plus - Z_m) / (Z_nad - Z_m + alpha)
    return F_tidle_minus, F_tidle_plus


def ideal_nadir(F_minus, F_plus):
    """
    FIX (checklist muc 7): normalize() giu nguyen chu ky tra ve 2 gia tri
    (khong doi de khong pha vo cac noi da goi 2-value unpack: toy_regional.py,
    E0_sanity.py, E1_heterogeneity_audit.py). Ham rieng nay expose them
    z_star/z_nad cho debug/test ma KHONG thay doi normalize().
    """
    F_minus = np.asarray(F_minus, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    stacked = np.vstack([F_minus, F_plus])
    return stacked.min(axis=0), stacked.max(axis=0)
