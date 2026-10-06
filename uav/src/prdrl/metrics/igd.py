import numpy as np

def igd(A, P_star):
    """
    IGD_t (eq. 64): (1/|P*|) * sum_{y in P*} min_{x in A} ||y - x||.

    FIX (edge case, checklist muc 3): A rong -> np.min([]) raise
    ValueError kho hieu ("zero-size array to reduction operation
    minimum"); P_star rong -> ZeroDivisionError o phep chia cuoi. Ca
    hai gio bao loi ro rang. Ngoai ra vong for ngoai (`for y in P_star`)
    duoc vector hoa bang broadcasting -- cung mot cong thuc, nhanh hon
    khi P_star lon.
    """
    A = np.asarray(A, dtype=float)
    P_star = np.asarray(P_star, dtype=float)
    if len(A) == 0:
        raise ValueError("igd(): approximation set A rong (0 nghiem)")
    if len(P_star) == 0:
        raise ValueError("igd(): reference front P_star rong (0 diem)")

    # (|P*|, |A|) khoang cach cap doi, roi min theo A cho tung diem P*.
    dists = np.linalg.norm(P_star[:, None, :] - A[None, :, :], axis=2)
    return float(np.mean(dists.min(axis=1)))


def migd(igd_list):
    """
    Mean IGD qua T environments (eq. 65).

    FIX (edge case): danh sach rong -> np.mean([]) tra ve NaN kem
    RuntimeWarning am tham; gio bao loi ro rang thay vi de NaN lan
    truyen im lang xuoi dong.
    """
    if len(igd_list) == 0:
        raise ValueError("migd(): igd_list rong (0 environment)")
    return float(np.mean(igd_list))
