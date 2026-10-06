import numpy as np


def severity(F_minus, F_plus, region_ids, R, q_prev, kappa=1.0, rho=0.05):
    """
    Muc thay doi theo vung + causal EMA scale (eq. 30-33).

    FIX (BUG-2, CRITICAL): vung rong (khong nghiem nao) truoc day duoc
    gan S[i]=0 -- SAI theo spec ("vung rong danh dau inactive") va KHAC
    voi chinh rhi.py::_severity_from_dis (cung tinh 1 dai luong, dung
    NaN cho vung rong). Gan S[i]=0 lam mot vung "gia" lot qua bo loc
    `not np.isnan(s)` trong rhi.rhi(), lam meo std/mean cua ca RHI that
    lan RHI null. Sua: vung rong -> NaN, dong bo voi rhi.py.

    FIX (BUG-3, CRITICAL, he qua truc tiep cua BUG-2): Q dung
    np.median(S) (khong bo qua NaN) -- mot khi S co NaN (dung sau khi
    sua BUG-2), np.median tra ve NaN, va vi Q chinh la q_prev cua lan
    goi SAU, NaN se LAN TRUYEN VINH VIEN qua toan bo phan con lai cua
    EMA scale ngay khi gap 1 vung rong dau tien. Sua: dung np.nanmedian.

    FIX (BUG-4): kappa (eq.32) truoc day an ngam =1.0, khong dat ten
    bien, khong expose duoc -- gio la tham so tuy chon, mac dinh giu
    nguyen kappa=1.0 nen KHONG lam doi hanh vi cac loi goi cu (positional
    5 tham so van hoat dong y het).

    C[i] = NaN tu dong cho vung rong (khong can special-case: NaN lan
    truyen qua phep tru/chia/exp mot cach tu nhien, dung y nghia "c
    khong xac dinh cho vung inactive").

    FIX (ISSUE-5, robustness): np.asarray() cho F_minus/F_plus -- ban
    goc doi hoi ca hai da la ndarray ("F_minus - F_plus" crash TypeError
    voi list thuan).
    """
    F_minus = np.asarray(F_minus, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    region_ids = np.asarray(region_ids)

    Dis = np.linalg.norm(F_minus - F_plus, axis=1)
    S = []
    for i in range(R):
        mask = region_ids == i
        S.append(float(np.median(Dis[mask])) if np.any(mask) else float("nan"))

    C = [1 - np.exp(-S[i] / (kappa * q_prev + 1e-12)) for i in range(R)]

    Q = (1 - rho) * q_prev + rho * np.nanmedian(S)
    return S, C, Q
