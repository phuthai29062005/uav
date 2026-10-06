import numpy as np


def partition(F_minus, V, alpha=1e-12):
    """
    Phan vung theo huong tham chieu, CHI dung F_minus (eq. 28-29).

    FIX (ISSUE-5, robustness): np.asarray() o dau ham -- ban goc doi hoi
    F_minus da la ndarray (`i / (x+alpha)` crash TypeError voi list
    thuan); gio chap nhan ca list-of-list lan ndarray, khong con phu
    thuoc caller phai nho wrap truoc.

    FIX (performance, checklist muc 4): thay 2 vong for long nhau (N x R)
    bang numpy vectorized (norm + matmul + argmax theo hang). np.argmax
    chon PHAN TU DAU TIEN dat max khi co tie -- giong het hanh vi vong
    lap goc (chi cap nhat khi ">" chat, nen max dau tien gap la nguoi
    thang) -- da verify ket qua giong het ban vong lap bang test.

    Edge case da xac nhan (khong sua, khong sai spec): nghiem trung diem
    ideal (F_minus_i = [0,...,0] sau chuan hoa) -> vector chuan hoa u_i
    cung la [0,...,0] -> cos-sim voi MOI huong deu = 0 -> region duoc
    chon la V[0] (huong dau tien) mot cach xac dinh (deterministic), vi
    o vong lap goc "0 > -inf" luon dung cho j=0 va khong j nao sau do
    lam "0 > 0" dung. Khoa lai bang test, khong tu y dinh nghia lai vi
    spec khong noi ro hanh vi cho diem suy bien nay.
    """
    F_minus = np.asarray(F_minus, dtype=float)
    V = np.asarray(V, dtype=float)

    norms_u = np.linalg.norm(F_minus, axis=1, keepdims=True) + alpha
    U = F_minus / norms_u                              # (N, M)
    norms_v = np.linalg.norm(V, axis=1) + alpha         # (R,)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # tren mot so build BLAS (vd Apple Accelerate) matmul kich thuoc
        # M nho co the phat RuntimeWarning gia du du lieu vao/ra hoan
        # toan huu han -- khong anh huong ket qua, da verify rieng.
        cos_sim = (U @ V.T) / norms_v[None, :]          # (N, R)
    region_ids = np.argmax(cos_sim, axis=1)
    return region_ids
