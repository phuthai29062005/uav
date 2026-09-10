import warnings

import numpy as np


def compute_signature(X_probe, problem_t, obj_scale):
    """
    Chu ky cua MOI TRUONG, khong phai cua population hien tai.

    Dung CUNG X_probe voi ChangeDetector: signature va change vector
    phai noi ve cung mot tap diem, neu khong hai tin hieu trong state
    se khong nhat quan.

    X_probe  : (n, D) probe set CO DINH cua episode
    problem_t: problem tai t
    obj_scale: (M,) thang co dinh, dung ref_point

    return: np.ndarray (L,) signature, L = 5*M, moi component in [0,1]
    """
    F = problem_t.evaluate(np.asarray(X_probe, dtype=float))
    F_hat = F / np.asarray(obj_scale, dtype=float)

    stats = []
    for m in range(F_hat.shape[1]):
        col = F_hat[:, m]
        stats.extend([
            col.mean(),
            col.std(),
            np.percentile(col, 25),
            np.percentile(col, 50),
            np.percentile(col, 75),
        ])
    e = np.asarray(stats, dtype=float)

    # Clip de d_mem chuan hoa theo sqrt(L) co y nghia. Neu F vuot
    # ref_point thi clip lam mat thong tin -> canh bao.
    # Message co dinh (khong nhung so) de warnings dedup — canh bao nay
    # kich hoat o phan lon lan doi: X_probe dong bang tu warm-up nen o
    # cac environment sau no nam xa PF va F vuot ref_point.
    if ((e < 0.0) | (e > 1.0)).mean() > 0.01:
        warnings.warn(
            "compute_signature: >1% component bi clip ve [0,1] — "
            "ref_point co the qua hep so voi probe set dong bang",
            RuntimeWarning, stacklevel=2)
    return np.clip(e, 0.0, 1.0)


class MemoryArchive:
    """
    Luu TOAN BO population cuoi moi environment, khoa theo
    environment signature.

    Lifecycle BAT BUOC:
        RETRIEVE -> RESPOND -> OPTIMIZE -> STORE
    Entry cua environment t chi duoc store SAU KHI da roi khoi t,
    nen no khong the tu retrieve chinh no.
    """

    def __init__(self, max_size=50):
        self.signatures = []
        self.pops = []
        self.max_size = max_size

    def __len__(self):
        return len(self.signatures)

    def query(self, signature):
        """
        Tim entry gan nhat. KHONG co threshold — luon tra ve nearest
        neu archive khong rong.

        return: dict {has_memory, d_mem, pop (ban copy), index}
        """
        if not self.signatures:
            # "khong co gi giong" — state luon co gia tri hop le
            return {"has_memory": False, "d_mem": 1.0,
                    "pop": None, "index": None}

        L = len(signature)
        dists = [np.linalg.norm(signature - e) for e in self.signatures]
        i_star = int(np.argmin(dists))
        d_mem = float(np.clip(dists[i_star] / np.sqrt(L), 0.0, 1.0))
        return {"has_memory": True, "d_mem": d_mem,
                "pop": self.pops[i_star].copy(), "index": i_star}

    def store(self, signature, pop):
        """FIFO khi day."""
        if len(self.signatures) >= self.max_size:
            self.signatures.pop(0)
            self.pops.pop(0)
        self.signatures.append(np.array(signature, copy=True))
        self.pops.append(np.array(pop, copy=True))

    def clear(self):
        self.signatures.clear()
        self.pops.clear()
