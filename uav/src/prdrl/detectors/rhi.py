"""
Regional Heterogeneity Index (dac ta muc 5.7, eq. 38) + random-partition
control (muc 5.7, bat buoc theo checklist Phu luc C muc 3):

    RHI(t) = std_{r in R_active}(s_r) / (mean_{r in R_active}(s_r) + eps)

"RHI phai duoc so voi random partition control cung kich thuoc vung.
Neu reference-direction partition khong tao heterogeneity cao hon
random partition, khong co bang chung rang cau truc objective-space
dang cung cap thong tin huu ich."
"""
import numpy as np

EPS = 1e-12


def rhi(s_r_dict, eps=EPS):
    """eq. 38. Chi tinh tren vung active (s_r != None). >=2 vung active
    can co de std co y nghia; 0 hoac 1 vung active -> RHI=0.0 (khong co
    do phan tan de do)."""
    active = np.array([s for s in s_r_dict.values() if s is not None], dtype=float)
    if len(active) < 2:
        return 0.0
    return float(np.std(active) / (np.mean(active) + eps))


def random_partition_same_sizes(N, region_sizes, rng):
    """
    Sinh MOT random partition co CUNG so vung va CUNG kich thuoc tung
    vung nhu partition tham chieu that (region_sizes: list int, tong
    = N), nhung gan ngau nhien nghiem nao vao vung nao.

    return: dict {region_label -> np.array(indices)}, region_label la
            0..len(region_sizes)-1 (khop thu tu voi region_sizes).
    """
    assert sum(region_sizes) == N, (sum(region_sizes), N)
    perm = rng.permutation(N)
    out = {}
    pos = 0
    for r, size in enumerate(region_sizes):
        out[r] = perm[pos:pos + size]
        pos += size
    return out


def rhi_null_distribution(d_i, region_sizes, n_trials, rng):
    """
    Phan phoi null cua RHI duoi random partition CUNG kich thuoc vung,
    tren CUNG vector displacement d_i cua change event nay.

    return: np.array (n_trials,) cac gia tri RHI ngau nhien.
    """
    N = len(d_i)
    out = np.empty(n_trials)
    for k in range(n_trials):
        idx_by_region = random_partition_same_sizes(N, region_sizes, rng)
        s_r = {r: (float(np.median(d_i[idx])) if len(idx) > 0 else None)
              for r, idx in idx_by_region.items()}
        out[k] = rhi(s_r)
    return out


def percentile_rank(value, null_dist):
    """Ty le mau null <= value -- dung de bao cao RHI_ref nam o
    percentile nao cua phan phoi null (Gate A1)."""
    null_dist = np.asarray(null_dist, dtype=float)
    return float(np.mean(null_dist <= value))
