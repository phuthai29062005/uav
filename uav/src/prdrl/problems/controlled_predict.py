"""
Controlled PREDICT-mechanism benchmark (E2-P), Phase 4-7.

Cung cap ground-truth Pareto-Set (PS) center motion RO RANG cho tung
region, de tach biet "movement estimation" (population centroid) khoi
"PS movement that" (analytic) -- thu duoc tu chinh cau truc benchmark,
KHONG can du doan/xap xi.

Thiet ke: giong het controlled_operator.py (quarter-circle DTLZ2-nhu,
x1 la "goc" vi tri, cac bien con lai la "distance"), NHUNG:

    - Chi 2 loai dynamic: "stationary" (target=0 moi t) va
      "persistent" (target=rate*t, TUYEN TINH theo t) -- dung theo
      Section 6 ("Không cần abrupt/diversify trong E2-P").
    - Them ham `region_center(region_dynamics, bin_index, t)` tra ve
      VECTOR muc tieu (D-1 chieu, chinh la "mu_star" trong khong gian
      quyet dinh CHUAN HOA [0,1]) -- ground truth PS-center tai bin do
      o thoi diem t, dung de tinh Delta_GT/Delta_NEXT ma KHONG can uoc
      luong gi ca (analytic, dung tuyet doi theo cau truc benchmark).

g=0 (dat duoc khi MOI bien distance = target hien tai) van cho PF that
= duong tron don vi 1/4, KHONG doi theo t (giong controlled_operator.py).
"""
import numpy as np


def get_bounds(D):
    return np.zeros(D), np.ones(D)


def _target_stationary(t):
    return 0.0


def _target_persistent(t, rate=0.10):
    """Tuyen tinh theo t (KHOA truoc, khong tune) -- vi du minh hoa
    Section 5 dung buoc 0.10/moc thoi gian (0.2->0.3->0.4,...)."""
    return rate * t


DYNAMIC_TYPES = {
    "stationary": _target_stationary,
    "persistent": _target_persistent,
}


def make_region_dynamics_map(n_bins):
    """LINEAR_PERSISTENT_SHIFT (Section 5-6): nua stationary, nua
    persistent, KHOI LIEN TUC (khong xen ke) -- giong pattern
    PERSISTENT_DIRECTIONAL_SHIFT cua controlled_operator.py, da xac
    nhan thuc nghiem co ~100% ground-truth purity o quy mo dan so
    thuc (xem E2a-R report muc 9) vi khong xen ke nhieu loai."""
    half = n_bins // 2
    return (["persistent"] * half) + (["stationary"] * (n_bins - half))


def bin_of(x1, n_bins):
    x1 = np.asarray(x1, dtype=float)
    idx = np.floor(x1 * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def evaluate(X, t, region_dynamics):
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
    theta = np.linspace(0.0, 0.5 * np.pi, n_points)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def region_center(region_dynamics, bin_index, t, n_distance_vars):
    """Ground-truth PS-center (mu_star) trong khong gian quyet dinh
    CHUAN HOA [0,1] cho bin `bin_index` tai thoi diem t -- vector
    (n_distance_vars,), moi phan tu = target(t) (broadcast, dung cau
    truc benchmark: TOAN BO bien distance deu chia se CUNG 1 target
    vo huong). Analytic, khong uoc luong."""
    fn = DYNAMIC_TYPES[region_dynamics[bin_index]]
    return np.full(n_distance_vars, fn(t))


def ground_truth_dynamic_label(X, region_dynamics):
    n_bins = len(region_dynamics)
    bins = bin_of(X[:, 0], n_bins)
    return np.array([region_dynamics[b] for b in bins])


def distance_to_region_ps(X_children, region_dynamics, bin_index, t):
    """Khoang cach (chuan hoa) tu moi child den PS that cua CHINH bin
    do tai t -- vi bien vi tri (x1) tu do tren PS trong pham vi bin,
    khoang cach ngan nhat la GIU NGUYEN x1 cua child va chi so sanh
    phan distance-vars voi target (dung dang phan tich, khong can
    sample PS)."""
    X_children = np.asarray(X_children, dtype=float)
    n_distance = X_children.shape[1] - 1
    target = region_center(region_dynamics, bin_index, t, n_distance)
    xd = X_children[:, 1:]
    return np.linalg.norm(xd - target[None, :], axis=1)
