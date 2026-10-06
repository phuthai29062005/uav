"""
Controlled regional dynamic benchmark (M0), Phase 3 -- E1R.

Muc dich: benchmark TOY co ground-truth chinh xac (biet truoc vung nao
bi dich, bao nhieu) de xac nhan detector/partition pipeline (partition
-> severity -> RHI) hoat dong DUNG truoc khi tin tuong ket qua tren
benchmark ngoai (SDP/CEC). Day la Gate M0 (mechanism sanity).

Nguyen tac BAT BUOC (giu nguyen tu du an):
    - Region ground truth duoc xac dinh TRUOC khi tao F_plus (frozen
      membership) -- KHONG de joint normalization vo tinh doi ground
      truth.
    - F_minus/F_plus dung CUNG X (khong lien quan operator/RL).

Moi scenario tra ve mot dict voi it nhat:
    F_minus, F_plus, X, region_ids_gt, V, R, scenario, seed,
    ground_truth_change_magnitude (mang (R,), NaN cho vung rong).
"""
import numpy as np

from ..core.reference import reference
from ..core.partition import partition
from ..core.normalize import normalize


def _make_base_population(N, n_obj, seed):
    """Sinh mot Pareto front don gian (M-simplex) lam F_minus, cung X
    gia (khong dung cho detection, chi de dong bo interface voi cac
    problem khac trong prdrl.problems).

    Dung reference-direction sampling (Das-Dennis) de sinh diem PHU
    tren simplex chuan hoa sum(f)=1, roi scale nhe theo nhieu ngau
    nhien nho de tranh cac diem trung nhau tuyet doi (giu cac diem van
    nam sat simplex, khong lam sai "PF").
    """
    rng = np.random.default_rng(seed)
    D = 10
    X = rng.random((N, D))

    if n_obj == 2:
        t = np.linspace(0.02, 0.98, N)
        F = np.column_stack([t, 1 - t])
    else:
        # simplex sampling: Dirichlet(1,...,1) cho M chieu, N diem.
        F = rng.dirichlet(np.ones(n_obj), size=N)

    return F, X


def _ground_truth_regions(F_minus, n_obj, n_partitions, seed):
    """Xac dinh region ground truth TRUOC khi tao F_plus (frozen
    membership) -- dung normalize(F_minus, F_minus) (khong co F_plus
    nao ca o buoc nay) + partition() giong y het pipeline that, chi
    khac la F_plus chua ton tai nen dung F_minus cho ca hai ve cua
    normalize (xem cung logic da xac nhan trong toy_regional.py)."""
    V = reference(n_obj, n_partitions)
    R = len(V)
    F_tilde_minus, _ = normalize(F_minus, F_minus)
    region_ids_gt = partition(np.asarray(F_tilde_minus), V)
    return region_ids_gt, V, R


def _active_regions(region_ids_gt, R):
    return sorted(set(int(r) for r in np.unique(region_ids_gt)) & set(range(R)))


def _ground_truth_magnitude_from_shift(region_ids_gt, R, region_shift_map):
    """region_shift_map: dict {region_id: shift_vector or None}. Tra ve
    mang (R,) magnitude = ||shift_vector|| cho vung active co shift,
    NaN cho vung khong active (khong co nghiem nao trong ground truth
    -- KHAC voi "active nhung shift=0", van la mot so thuc 0.0)."""
    mag = np.full(R, np.nan)
    active = _active_regions(region_ids_gt, R)
    for r in active:
        shift = region_shift_map.get(r)
        mag[r] = 0.0 if shift is None else float(np.linalg.norm(shift))
    return mag


# =====================================================================
# M0-A -- GLOBAL_SHIFT (negative control)
# =====================================================================

def make_global_shift(N=120, n_obj=2, n_partitions=9, shift_magnitude=0.15,
                       jitter_std=0.02, seed=0):
    """Moi region deu bi dich GIONG NHAU (cung phan phoi, khong lien
    quan region) -- negative control: severity phai gan dong deu, RHI
    thap, khong vuot null manh.

    LUU Y: them jitter doc lap tung diem (jitter_std, KHONG lien quan
    region_ids_gt) quanh shift chung. Neu KHONG co jitter, moi diem
    dich mot vector giong het nhau tuyet doi -> Dis giong het nhau toi
    ~1e-17 (sai so dau phay dong) cho MOI diem -> RHI that va RHI null
    deu ~1e-16 (nhieu so hoc, khong phai tin hieu that) -> percentile
    tro nen VO NGHIA (so sanh nhieu voi nhieu, co the ra bat ky gia tri
    nao tu 0 den 1 thuan tuy do ngau nhien lam tron). Jitter nho, doc
    lap voi region, giu dung ban chat "global, khong phu thuoc region"
    nhung lam severity co phuong sai thuc (khac 0 nhung KHONG lien
    quan region) de RHI-vs-null la mot so sanh co dieu kien tot."""
    F_minus, X = _make_base_population(N, n_obj, seed)
    region_ids_gt, V, R = _ground_truth_regions(F_minus, n_obj, n_partitions, seed)

    rng = np.random.default_rng(seed + 1)
    direction = rng.normal(size=n_obj)
    direction /= np.linalg.norm(direction) + 1e-12
    shift = direction * shift_magnitude

    per_point_jitter = rng.normal(scale=jitter_std, size=(N, n_obj))
    F_plus = F_minus + shift[None, :] + per_point_jitter

    active = _active_regions(region_ids_gt, R)
    ground_truth_change_magnitude = _ground_truth_magnitude_from_shift(
        region_ids_gt, R, {r: shift for r in active})

    return {
        "scenario": "GLOBAL_SHIFT",
        "seed": seed,
        "F_minus": F_minus,
        "F_plus": F_plus,
        "X": X,
        "region_ids_gt": np.asarray(region_ids_gt),
        "V": V,
        "R": R,
        "ground_truth_change_magnitude": ground_truth_change_magnitude,
        "metadata": {"shift_magnitude": shift_magnitude},
    }


# =====================================================================
# M0-B -- SINGLE_REGION_SHIFT
# =====================================================================

def make_single_region_shift(N=120, n_obj=2, n_partitions=9,
                             shift_magnitude=0.3, seed=0):
    """Chi MOT region bi dich -- expected: argmax severity = ground
    truth shifted region, RHI cao, percentile cao so voi null."""
    F_minus, X = _make_base_population(N, n_obj, seed)
    region_ids_gt, V, R = _ground_truth_regions(F_minus, n_obj, n_partitions, seed)

    active = _active_regions(region_ids_gt, R)
    if len(active) == 0:
        raise RuntimeError("SINGLE_REGION_SHIFT: khong co active region nao "
                           "(seed/n_partitions khong hop le)")
    rng = np.random.default_rng(seed + 1)
    R_shift = int(rng.choice(active))

    direction = rng.normal(size=n_obj)
    direction /= np.linalg.norm(direction) + 1e-12
    shift = direction * shift_magnitude

    F_plus = F_minus.copy()
    mask = region_ids_gt == R_shift
    F_plus[mask] = F_minus[mask] + shift[None, :]

    ground_truth_change_magnitude = _ground_truth_magnitude_from_shift(
        region_ids_gt, R, {R_shift: shift})

    return {
        "scenario": "SINGLE_REGION_SHIFT",
        "seed": seed,
        "F_minus": F_minus,
        "F_plus": F_plus,
        "X": X,
        "region_ids_gt": np.asarray(region_ids_gt),
        "V": V,
        "R": R,
        "R_shift": R_shift,
        "ground_truth_change_magnitude": ground_truth_change_magnitude,
        "metadata": {"shift_magnitude": shift_magnitude},
    }


# =====================================================================
# M0-C -- MULTI_SEVERITY
# =====================================================================

def make_multi_severity(N=120, n_obj=2, n_partitions=9, seed=0,
                        magnitude_levels=(0.0, 0.08, 0.18, 0.35)):
    """Cac active region bi dich voi magnitude KHAC nhau (tang dan).
    Khong hardcode dung 4 regions -- xay tong quat theo so active
    regions thuc te, gan magnitude_levels theo chu ky neu active
    regions nhieu hon len(magnitude_levels)."""
    F_minus, X = _make_base_population(N, n_obj, seed)
    region_ids_gt, V, R = _ground_truth_regions(F_minus, n_obj, n_partitions, seed)
    active = _active_regions(region_ids_gt, R)

    rng = np.random.default_rng(seed + 1)
    directions = {}
    magnitudes = {}
    for k, r in enumerate(active):
        d = rng.normal(size=n_obj)
        d /= np.linalg.norm(d) + 1e-12
        directions[r] = d
        magnitudes[r] = magnitude_levels[k % len(magnitude_levels)]

    F_plus = F_minus.copy()
    shift_map = {}
    for r in active:
        mask = region_ids_gt == r
        shift = directions[r] * magnitudes[r]
        shift_map[r] = shift
        F_plus[mask] = F_minus[mask] + shift[None, :]

    ground_truth_change_magnitude = _ground_truth_magnitude_from_shift(
        region_ids_gt, R, shift_map)

    return {
        "scenario": "MULTI_SEVERITY",
        "seed": seed,
        "F_minus": F_minus,
        "F_plus": F_plus,
        "X": X,
        "region_ids_gt": np.asarray(region_ids_gt),
        "V": V,
        "R": R,
        "ground_truth_change_magnitude": ground_truth_change_magnitude,
        "metadata": {"magnitude_levels": list(magnitude_levels),
                     "active_regions": active},
    }


# =====================================================================
# M0-D -- OPPOSITE_DIRECTION
# =====================================================================

def make_opposite_direction(N=120, n_obj=2, n_partitions=9,
                            shift_magnitude=0.25, seed=0):
    """Hai (hoac nhieu) region dich theo huong NGUOC nhau. E1 chi can
    xac nhan cac region bi thay doi khac voi region khong doi -- KHONG
    claim gi ve operator behavior (do la viec cua sau nay, PREDICT)."""
    F_minus, X = _make_base_population(N, n_obj, seed)
    region_ids_gt, V, R = _ground_truth_regions(F_minus, n_obj, n_partitions, seed)
    active = _active_regions(region_ids_gt, R)
    if len(active) < 2:
        raise RuntimeError("OPPOSITE_DIRECTION: can >=2 active region")

    rng = np.random.default_rng(seed + 1)
    r_pos, r_neg = rng.choice(active, size=2, replace=False)
    direction = rng.normal(size=n_obj)
    direction /= np.linalg.norm(direction) + 1e-12

    F_plus = F_minus.copy()
    shift_map = {}
    mask_pos = region_ids_gt == r_pos
    shift_pos = direction * shift_magnitude
    F_plus[mask_pos] = F_minus[mask_pos] + shift_pos[None, :]
    shift_map[int(r_pos)] = shift_pos

    mask_neg = region_ids_gt == r_neg
    shift_neg = -direction * shift_magnitude
    F_plus[mask_neg] = F_minus[mask_neg] + shift_neg[None, :]
    shift_map[int(r_neg)] = shift_neg

    ground_truth_change_magnitude = _ground_truth_magnitude_from_shift(
        region_ids_gt, R, shift_map)

    return {
        "scenario": "OPPOSITE_DIRECTION",
        "seed": seed,
        "F_minus": F_minus,
        "F_plus": F_plus,
        "X": X,
        "region_ids_gt": np.asarray(region_ids_gt),
        "V": V,
        "R": R,
        "r_pos": int(r_pos),
        "r_neg": int(r_neg),
        "ground_truth_change_magnitude": ground_truth_change_magnitude,
        "metadata": {"shift_magnitude": shift_magnitude},
    }


# =====================================================================
# M0-E -- LOCAL_APPEAR_DISAPPEAR (simplified deterministic version)
# =====================================================================

def make_local_appear_disappear(N=120, n_obj=2, n_partitions=9,
                                disappear_fraction=0.7, reappear_shift=0.4,
                                seed=0):
    """Mot phan front (mot region) 'bien mat' (cac diem trong vung do
    bi dich MANH ra khoi vung cu, sang mot vung khac o F_plus) --
    simplified deterministic version cua "appear/disappear": khong
    thuc su xoa diem (giu N co dinh cho detector pipeline dang co, vi
    partition/severity chua ho tro N thay doi giua F_minus/F_plus theo
    eq.22 "cung decision vectors"), ma dich TOAN BO 1 region ra xa den
    muc no khong con thuoc region cu nua khi nhin theo F_plus (te bao
    'bien mat' khoi vung ground-truth cu). Ground truth ro rang: vung
    bi anh huong = region_disappear."""
    F_minus, X = _make_base_population(N, n_obj, seed)
    region_ids_gt, V, R = _ground_truth_regions(F_minus, n_obj, n_partitions, seed)
    active = _active_regions(region_ids_gt, R)
    if len(active) == 0:
        raise RuntimeError("LOCAL_APPEAR_DISAPPEAR: khong co active region")

    rng = np.random.default_rng(seed + 1)
    region_disappear = int(rng.choice(active))

    mask = region_ids_gt == region_disappear
    n_in_region = int(np.sum(mask))
    n_move = max(1, int(round(n_in_region * disappear_fraction)))
    idx_in_region = np.flatnonzero(mask)
    idx_move = rng.choice(idx_in_region, size=n_move, replace=False)

    # Day manh cac diem duoc chon ra xa theo huong doi dien tam simplex,
    # de "bien mat" khoi vung cu roi vao vung khac o F_plus.
    direction = rng.normal(size=n_obj)
    direction /= np.linalg.norm(direction) + 1e-12

    F_plus = F_minus.copy()
    F_plus[idx_move] = F_minus[idx_move] + direction[None, :] * reappear_shift

    shift_map = {region_disappear: direction * reappear_shift}
    ground_truth_change_magnitude = _ground_truth_magnitude_from_shift(
        region_ids_gt, R, shift_map)

    return {
        "scenario": "LOCAL_APPEAR_DISAPPEAR",
        "seed": seed,
        "F_minus": F_minus,
        "F_plus": F_plus,
        "X": X,
        "region_ids_gt": np.asarray(region_ids_gt),
        "V": V,
        "R": R,
        "region_disappear": region_disappear,
        "idx_moved": idx_move.tolist(),
        "ground_truth_change_magnitude": ground_truth_change_magnitude,
        "metadata": {"disappear_fraction": disappear_fraction,
                     "reappear_shift": reappear_shift},
    }


SCENARIOS = {
    "GLOBAL_SHIFT": make_global_shift,
    "SINGLE_REGION_SHIFT": make_single_region_shift,
    "MULTI_SEVERITY": make_multi_severity,
    "OPPOSITE_DIRECTION": make_opposite_direction,
    "LOCAL_APPEAR_DISAPPEAR": make_local_appear_disappear,
}


def make_scenario(name, **kwargs):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Valid: {list(SCENARIOS)}")
    return SCENARIOS[name](**kwargs)
