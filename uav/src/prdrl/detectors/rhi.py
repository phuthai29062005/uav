import numpy as np

# Ghi chu review: khong tim thay bug logic trong file nay. _severity_from_dis
# von da dung NaN dung cho vung rong (khop spec "vung rong danh dau
# inactive") -- day chinh la ban tham chieu ma severity.py duoc sua lai
# (BUG-2/3) de dong bo theo, khong phai nguoc lai.


def rhi(s_r):
    active = [s for s in s_r if not np.isnan(s)]
    if len(active) < 2:
        return 0.0
    return np.std(active) / (np.mean(active) + 1e-12)

def rhi_null(F_tilde_minus, F_tilde_plus, region_ids, R, K=200, rng=None):
    """
    Size-preserving random-partition null cho RHI.

    FIX (Phase 11, E1R): ban dau dung np.random.permutation() khong seed
    -- moi lan goi cho ket qua khac nhau, khong the reproduce lai duoc
    mot ket qua E1 cu tu cung 1 (experiment seed, df/sdp id, change idx).
    Gio nhan `rng` tuong minh (np.random.Generator); neu khong truyen,
    tao mot Generator KHONG seed (giu backward-compat cho code cu chua
    cap nhat), nhung moi noi goi MOI trong E1a/E1b/E1c deu PHAI truyen
    rng duoc seed tat dinh tu (experiment seed, problem id, change idx).
    """
    if rng is None:
        rng = np.random.default_rng()

    Dis = np.linalg.norm(np.array(F_tilde_plus) - np.array(F_tilde_minus), axis=1)
    s_r_ref = _severity_from_dis(Dis, region_ids, R)
    rhi_ref = rhi(s_r_ref)
    sizes = []
    for r in range(R):
        sizes.append(np.sum(np.array(region_ids) == r))
    rhi_nulls = []
    N = len(region_ids)
    for _ in range(K):
        perm = rng.permutation(N)
        fake_ids = np.empty(N, dtype=int)
        start = 0
        for r in range(R):
            fake_ids[perm[start:start + sizes[r]]] = r
            start += sizes[r]
        s_r_fake = _severity_from_dis(Dis, fake_ids, R)
        rhi_nulls.append(rhi(s_r_fake))
    rhi_nulls = np.array(rhi_nulls)
    percentile = np.mean(rhi_nulls < rhi_ref)
    return rhi_ref, np.mean(rhi_nulls), np.std(rhi_nulls), percentile

def _severity_from_dis(Dis, region_ids, R):
    region_ids = np.array(region_ids)
    s_r = []
    for r in range(R):
        mask = region_ids == r
        if np.sum(mask) == 0:
            s_r.append(np.nan)
        else:
            s_r.append(np.median(Dis[mask]))
    return s_r
