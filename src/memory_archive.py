import numpy as np


def calibrate_scale(F0):
    """
    F0: (n, M) = F(X_probe, t0), evaluate tai cuoi warm-up.
    return: (M,) scale co dinh cho toan run.

    Median thay vi max: bat bien voi outlier. Do tai t0 de z ~ 1 o
    dau run (giua thang squash) roi tang dan khi moi truong troi,
    thay vi bat dau tu vung da bao hoa. Khong phai tune: duoc do,
    tat dinh theo seed, tu thich nghi sang UAV.
    """
    F0 = np.asarray(F0, dtype=float)
    assert np.all(F0 >= 0), "objectives phai non-negative"
    return np.median(F0, axis=0) + 1e-8


def compute_signature(X_probe, problem_t, obj_scale, return_saturation=False):
    """
    Chu ky cua MOI TRUONG, khong phai cua population hien tai.

    Dung CUNG X_probe voi ChangeDetector: signature va change vector
    phai noi ve cung mot tap diem, neu khong hai tin hieu trong state
    se khong nhat quan.

    X_probe  : (n, D) probe set CO DINH cua episode
    problem_t: problem tai t
    obj_scale: (M,) scale do tai t0 bang calibrate_scale, co dinh trong run

    return: np.ndarray (L,) signature, L = 5*M, moi component in [0,1)
            (kem soft_sat_rate neu return_saturation)
    """
    F = problem_t.evaluate(np.asarray(X_probe, dtype=float))
    assert np.all(F >= 0), f"negative objective: min={F.min()}"

    Z = F / np.asarray(obj_scale, dtype=float)
    U = Z / (1.0 + Z)                 # soft squash, don anh, in [0, 1)

    stats = []
    for m in range(U.shape[1]):
        col = U[:, m]
        stats.extend([
            col.mean(),
            2.0 * col.std(),          # std cua [0,1] toi da 0.5
            np.percentile(col, 25),
            np.percentile(col, 50),
            np.percentile(col, 75),
        ])
    e = np.asarray(stats, dtype=float)
    if return_saturation:
        return e, float(np.mean(U > 0.99))
    return e


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
        self.obj_scale = None   # do MOT LAN tai prime, co dinh trong run

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
