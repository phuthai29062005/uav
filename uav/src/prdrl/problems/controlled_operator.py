"""
Controlled OPERATOR benchmark (E2a), Phase 3 -- E2a task.

KHAC voi controlled_regional.py (E1): E1's benchmark tao F_minus/F_plus
TRUC TIEP (X chi la vector gia, khong lien quan gi den F) -- du cho
detector (partition/severity/RHI) nhung KHONG du cho operator oracle,
vi LOCAL/PREDICT/DIVERSIFY sinh offspring trong DECISION SPACE va can
duoc EVALUATE THAT qua mot ham X -> F(X,t).

Thiet ke: mot bien the DTLZ2-nhu (quarter-circle PF) 2 muc tieu:

    g(x,t)  = 1 + sum_{i=2}^{D} (x_i - target(bin(x1), t))^2
    f1(x,t) = g(x,t) * cos(0.5*pi*x1)
    f2(x,t) = g(x,t) * sin(0.5*pi*x1)

x1 (bien dau tien) dong vai tro "goc" (angle position) tren PF --
tuong tu cach cac chuong trinh khac trong repo (DF/SDP) dung x1 lam
vi tri front va cac bien con lai lam "distance". Khi g=0 (dat duoc khi
MOI x_i (i>=2) bang dung target hien tai), diem nam DUNG tren duong
tron don vi (PF that, KHONG doi theo t hay theo dynamic type -- xem
get_true_pf()).

MOI diem thuoc ve 1 trong n_bins "logical region" xac dinh THUAN TUY
tu x1 (chia [0,1] thanh n_bins doan bang nhau) -- KHONG dung
partition()/reference() cua E1 ben trong benchmark nay (benchmark
khong duoc biet gi ve region_ids "that" ma detector se tinh sau, tranh
circular dependency). Trong thuc te, voi PF dang 1/4 duong tron va
Das-Dennis (M=2) la cac huong deu nhau, ranh gioi region tu partition()
thuc te se GAN nhu trung voi cac bin goc deu nay -- nhung day chi la
xap xi hop ly, KHONG duoc gia dinh trung khop tuyet doi; experiment
script se dung region_ids THAT (tu partition() tren F_minus) de tong
hop ket qua, khong dua vao bin index nay.

Moi bin duoc gan 1 "dynamic type" (xem DYNAMIC_TYPES) -- quyet dinh
target(t) cua CAC DIEM CO x1 ROI VAO BIN DO. Day la co che TAO RA
heterogeneous dynamics giua cac vung objective-space, KHONG phai
license de operator "biet truoc" scenario nao dang chay (operator
oracle van phai tu suy ra tu IGD, khong doc dynamic type).
"""
import numpy as np


D_DEFAULT = 10


def get_bounds(D=D_DEFAULT):
    return np.zeros(D), np.ones(D)


# =====================================================================
# Dynamic-type target trajectories (KHOA truoc khi chay -- Section 5
# tinh than: khong tune sau khi xem ket qua)
# =====================================================================

def _target_stationary(t):
    return 0.0


def _target_small_shift(t, magnitude=0.05):
    """Dich nho, CO DINH tu khi moi truong bat dau doi (t>0) -- khong
    tang dan, chi la mot muc lech nho, hang so."""
    return magnitude if t > 1e-12 else 0.0


def _target_persistent(t, rate=0.08):
    """Dich TUYEN TINH theo t -- co the ngoai suy dung tu lich su
    (mu_{t-1}-mu_{t-2}), day chinh la dieu kien PREDICT co loi."""
    return rate * t


def _target_abrupt(t, jump_time=0.3, magnitude=0.35):
    """Nhay dot ngot tai jump_time, KHONG phai ngoai suy tuyen tinh tu
    lich su truoc do (truoc jump_time la stationary)."""
    return magnitude if t >= jump_time else 0.0


DYNAMIC_TYPES = {
    "stationary": _target_stationary,
    "small_shift": _target_small_shift,
    "persistent": _target_persistent,
    "abrupt": _target_abrupt,
}


# =====================================================================
# Scenario -> per-bin dynamic-type map (deterministic, KHONG ngau
# nhien -- cung mot scenario luon cho cung 1 mapping)
# =====================================================================

def make_region_dynamics_map(scenario, n_bins):
    if scenario == "GLOBAL_STABLE_OR_SMALL_SHIFT":
        return ["small_shift"] * n_bins

    if scenario == "MIXED_STATIONARY_AND_LOCAL":
        half = n_bins // 2
        return (["stationary"] * half
               + ["small_shift"] * (n_bins - half))

    if scenario == "PERSISTENT_DIRECTIONAL_SHIFT":
        half = n_bins // 2
        return (["persistent"] * half
               + ["stationary"] * (n_bins - half))

    if scenario == "ABRUPT_LARGE_SHIFT":
        half = n_bins // 2
        return (["abrupt"] * half
               + ["stationary"] * (n_bins - half))

    if scenario == "MIXED_RESPONSE":
        cycle = ["stationary", "small_shift", "persistent", "abrupt"]
        return [cycle[i % len(cycle)] for i in range(n_bins)]

    raise ValueError(f"Unknown scenario '{scenario}'. Valid: "
                     f"{list(SCENARIOS)}")


SCENARIOS = (
    "GLOBAL_STABLE_OR_SMALL_SHIFT",
    "MIXED_STATIONARY_AND_LOCAL",
    "PERSISTENT_DIRECTIONAL_SHIFT",
    "ABRUPT_LARGE_SHIFT",
    "MIXED_RESPONSE",
)


def bin_of(x1, n_bins):
    x1 = np.asarray(x1, dtype=float)
    idx = np.floor(x1 * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def evaluate(X, t, region_dynamics):
    """
    X               : (N, D) decision vectors, D>=2, tat ca trong [0,1].
    t               : float, thoi diem hien tai.
    region_dynamics : list[str] do dai n_bins, moi phan tu la mot key
                      cua DYNAMIC_TYPES -- do make_region_dynamics_map()
                      tra ve.

    Returns
    -------
    F : (N, 2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"X phai shape (N,D>=2), nhan duoc {X.shape}")

    n_bins = len(region_dynamics)
    x1 = X[:, 0]
    xd = X[:, 1:]

    bins = bin_of(x1, n_bins)
    targets = np.empty(len(X))
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        fn = DYNAMIC_TYPES[region_dynamics[b]]
        targets[mask] = fn(t)

    g = 1.0 + np.sum((xd - targets[:, None]) ** 2, axis=1)
    f1 = g * np.cos(0.5 * np.pi * x1)
    f2 = g * np.sin(0.5 * np.pi * x1)
    return np.column_stack([f1, f2])


def get_true_pf(n_points=500):
    """PF that: g=0 dat duoc o MOI goc bat ky luc nao (dat x_i=target
    hien tai cho moi bien distance) -> duong tron don vi 1/4, KHONG doi
    theo t hay theo dynamic type. Vi vay ham nay KHONG nhan tham so t
    (co chu y, khong phai thieu sot)."""
    theta = np.linspace(0.0, 0.5 * np.pi, n_points)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def ground_truth_dynamic_label(X, region_dynamics):
    """
    (E2a-R, Phase 9-10) Nhan dynamic-type GOC THAT (ground truth) cho
    tung nghiem trong X, dua TRUC TIEP tren bin(x1) -- KHONG lien quan
    gi den region_ids do partition() tinh sau nay tren F. Ham nay CHI
    them moi (khong sua evaluate()/get_true_pf() hien co), dung de
    audit xem "actual reference-region id" (tu partition, dua tren goc
    do cosine-similarity voi Das-Dennis directions) co THAT SU trung
    voi "ground-truth bin" (chia deu theo x1) hay khong -- KHONG duoc
    gia dinh trung khop tuyet doi (xem docstring dau file).

    Returns
    -------
    labels : (N,) array of str, moi phan tu la mot key cua
             DYNAMIC_TYPES ung voi bin ma nghiem do roi vao.
    """
    X = np.asarray(X, dtype=float)
    n_bins = len(region_dynamics)
    bins = bin_of(X[:, 0], n_bins)
    return np.array([region_dynamics[b] for b in bins])
