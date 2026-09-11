import numpy as np
from pymoo.problems.dynamic.df import DF1
from pymoo.indicators.hv import HV

class Memory:
    """
    DEPRECATED: khoa theo centroid khong gian quyet dinh, mu voi
    moi truong. Thay bang MemoryArchive (src/memory_archive.py) tu 4.1.
    Giu lai de tai lap ket qua truoc 4.1.
    """

    def __init__(self, max_size=50):
        self.keys = []      # list các khoá (centroid)
        self.pops = []      # list các population tương ứng
        self.max_size = max_size
    
    def store(self, key, pop):
        """
        Lưu 1 population đã hội tụ vào bộ nhớ
        key: np.array — chữ ký môi trường (centroid)
        pop: np.array (N, D)
        Nếu đầy → xoá cái cũ nhất
        """
        if len(self.keys) >= self.max_size:
            self.keys.pop(0)
            self.pops.pop(0)
        self.keys.append(key.copy())
        self.pops.append(pop.copy())
        
    
    def retrieve(self, key):
        """
        Tìm population có khoá GẦN NHẤT với key
        return: pop (bản copy), hoặc None nếu bộ nhớ rỗng
        """
        if not self.keys:
            return None
        # Tính khoảng cách giữa key và các khoá trong memory
        distances = [np.linalg.norm(key - k) for k in self.keys]
        idx_min = np.argmin(distances)
        return self.pops[idx_min].copy()

def repair(pop, xl, xu):
    """
    Sua nghiem vi pham rang buoc: clip ve [xl, xu] cua bai toan.
    Tren UAV: them kiem tra va cham, do cao, toc do, nang luong.
    KHONG con default [0,1] — goi thieu xl/xu se TypeError, fail som.
    """
    return np.clip(pop, xl, xu)

def compute_env_key(pop):
    """
    DEPRECATED: khong nhan problem nen hai moi truong khac nhau voi
    cung population cho ra key GIONG HET. Thay bang
    memory_archive.compute_signature tu 4.1.

    Tạo chữ ký môi trường từ population đã hội tụ.
    Ý tưởng: môi trường giống nhau → POS ở cùng vị trí → centroid giống nhau
    
    return: np.array — khoá dùng cho Memory.store() và Memory.retrieve()
    """
    centroid = np.mean(pop, axis=0)   # gợi ý: np.mean(pop, axis=0)
    return centroid

def compute_change_vector(elite, problem_new, segments, D):
    """
    DEPRECATED: do local sensitivity, khong phai temporal change.
    Giu lai de tai lap ket qua Bang 1/3 truoc 4.3.
    """
    S = len(segments)                    # số segments (=2 trên CEC)
    n_elite = len(elite)
    raw_c = np.zeros(S)                  # tích luỹ chênh lệch

    for i in range(n_elite):
        x = elite[i]
        F_goc = problem_new.evaluate(x.reshape(1, -1))   # shape (1, M)

        for s in range(S):
            x_copy = x.copy()                 # copy x
            for var_idx in segments[s]:      # loop qua các biến của segment s
                x_copy[var_idx] += 0.01      # nhiễu
            x_copy = np.clip(x_copy, 0.0, 1.0)   # giu nhieu trong mien [0,1]
            F_nhieu = problem_new.evaluate(x_copy.reshape(1, -1))   # evaluate x_copy
            delta = np.linalg.norm(F_nhieu - F_goc)                   # norm(F_nhieu - F_goc)
            raw_c[s] += delta

    # Trung bình qua elite
    raw_c = raw_c / n_elite

    # Chuẩn hoá về [0, 1]
    max_c = np.max(raw_c)
    if max_c > 1e-10:
        c = raw_c / max_c
    else:
        c = np.zeros_like(raw_c)

    return c

def compute_entropy(pop, xl, xu):
    """
    Normalized dispersion cua population, chuan hoa theo do rong moi
    chieu. Uniform[xl,xu] -> ~1. (Van khong phai entropy that, rename 4.8.)
    """
    width = np.asarray(xu, dtype=float) - np.asarray(xl, dtype=float)
    sigma = np.std(pop, axis=0)
    dispersion = sigma / (width / np.sqrt(12.0))
    return float(np.clip(np.mean(dispersion), 0.0, 1.0))

def compute_hv_drop(F_before, F_after, ref_point):
    """
    F_before: fitness TRƯỚC khi môi trường đổi
    F_after:  fitness cùng population, đánh giá SAU khi đổi
    ref_point: điểm tham chiếu, ví dụ [2.0, 2.0]
    
    return: tỉ lệ suy giảm ∈ [0, 1]
    """
    hv = HV(ref_point=np.array(ref_point))
    hv_before = hv(F_before)      # không phải hv.do(F_before)
    hv_after = hv(F_after)
    
    if hv_before < 1e-10:
        return 0.0
    else:
        drop = (hv_before - hv_after) / hv_before
        return np.clip(drop, 0, 1)

def compute_phase(t, period=4.0):
    """
    Pha trong chu kỳ, ∈ [0, 1)
    Trên CEC: G(t) = sin(0.5πt) → chu kỳ = 4
    """
    phase = (t % period) / period
    return phase

# Segment action IDs (4.1B): MEMORY khong con la segment action.
SEG_KEEP, SEG_LOCAL, SEG_PREDICT, SEG_DIVERSIFY = 0, 1, 2, 3
GATE_NO_MEMORY, GATE_MEMORY = 0, 1
# Vi tri trong state — moi noi doc d_mem/has_memory DUNG hang so nay.
IDX_D_MEM = -2
IDX_HAS_MEMORY = -1


def build_state(c, entropy, hv_drop, g_last, tau_t, phase,
                d_mem, has_memory):
    """
    [c_1..c_S, entropy, hv_drop, g_norm, phase, d_mem, has_memory]
    return: np.array shape (S + 6,). has_memory LUON la phan tu cuoi.
    """
    g_norm = g_last / tau_t
    return np.concatenate([c, [entropy, hv_drop, g_norm, phase,
                               float(d_mem), float(has_memory)]])
    
def action_keep(pop, segment_vars):
    return pop.copy()

def action_local(pop, segment_vars, xl, xu, sigma_frac=0.05):
    """
    Nhieu Gaussian nho vao cac bien cua segment. sigma scale theo do
    rong moi chieu (sigma_frac * width), KHONG hang so tuyet doi.
    """
    pop_new = pop.copy()
    width = np.asarray(xu, dtype=float) - np.asarray(xl, dtype=float)
    seg = list(segment_vars)
    noise = np.random.normal(0.0, sigma_frac, size=(pop.shape[0], len(seg)))
    pop_new[:, seg] += noise * width[seg]
    return np.clip(pop_new, xl, xu)

def action_predict(pop, pop_prev, segment_vars, xl, xu, noise_frac=0.02):
    """
    Du doan tuyen tinh theo huong dich chuyen cua CENTROID. Nhieu scale
    theo do rong moi chieu; clip theo [xl, xu].
    """
    pop_new = pop.copy()
    seg = list(segment_vars)
    width = np.asarray(xu, dtype=float) - np.asarray(xl, dtype=float)
    direction = (np.mean(pop[:, seg], axis=0)
                 - np.mean(pop_prev[:, seg], axis=0))
    pop_new[:, seg] += direction
    pop_new[:, seg] += (np.random.normal(0.0, noise_frac,
                                         size=(pop.shape[0], len(seg)))
                        * width[seg])
    return np.clip(pop_new, xl, xu)

def action_memory_global(pop, pop_mem, xl, xu):
    """
    Thay TOAN BO population bang population lich su.
    KHONG splice segment — day la thay doi thiet ke, xem 4.1.

    Neu N_mem != N: lay min(N, N_mem) ca the dau, phan con lai giu
    tu pop hien tai.
    """
    if pop_mem is None:
        return pop.copy()
    n = min(len(pop), len(pop_mem))
    pop_new = pop.copy()
    pop_new[:n] = pop_mem[:n]
    return repair(pop_new, xl, xu)


def action_memory(pop, memory, key, segment_vars, xl, xu):
    """
    DEPRECATED: splice segment mau thuan voi canh bao coupling muc 5.8;
    thay bang action_memory_global tu 4.1.
    """
    pop_mem = memory.retrieve(key)
    if pop_mem is None:
        return action_local(pop, segment_vars, xl, xu)
    pop_new = pop.copy()
    n = min(len(pop), len(pop_mem))
    pop_new[:n, segment_vars] = pop_mem[:n, segment_vars]
    return repair(pop_new, xl, xu)
    
def action_diversify(pop, segment_vars, xl, xu, ratio=0.5):
    """
    Thay ratio% ca the bang gia tri random trong [xl, xu] cua segment.
    """
    pop_new = pop.copy()
    xl = np.asarray(xl, dtype=float); xu = np.asarray(xu, dtype=float)
    n_replace = int(pop.shape[0] * ratio)
    indices = np.random.choice(pop.shape[0], n_replace, replace=False)
    for idx in indices:
        for j in segment_vars:
            pop_new[idx, j] = xl[j] + np.random.rand() * (xu[j] - xl[j])
    return pop_new


def compute_reward(hv_after, hv_base, hv_ref, fe_used, fe_budget,
                   alpha=1.0, beta=0.1):
    """
    hv_after:  HV cuối epoch (gen 70) — sau hành động + NSGA-II chạy xong
    hv_base:   HV ngay khi môi trường đổi (gen 60) — trước khi phản ứng
    hv_ref:    hằng số chuẩn hoá, ví dụ tích các thành phần ref_point
    fe_used:   số lần evaluate đã dùng trong epoch
    fe_budget: ngân sách FE cho 1 epoch
    """
    quality = (hv_after - hv_base) / hv_ref   # (hv_after - hv_base) / hv_ref
    cost = fe_used / fe_budget
    return alpha * quality - beta * cost

ACTION_KEEP = 0

def count_fe(gate, seg_actions, N):
    """
    FE cua buoc phan ung: evaluate lai ca population (N) neu co bat ky
    thay doi — gate=MEMORY hoac segment nao khac KEEP.
    FE cua ChangeDetector va signature duoc dem tai nguon (fe_used,
    len(X_probe)), KHONG cong o day de tranh dem hai lan.
    """
    if gate == GATE_MEMORY or np.any(np.asarray(seg_actions) != SEG_KEEP):
        return N
    return 0


def apply_hierarchical_response(pop, pop_prev, gate, seg_actions,
                                segments, pop_mem, xl, xu):
    """
    gate=1 (MEMORY): thay TOAN BO pop bang pop_mem. seg_actions BO QUA.
    gate=0: chay seg_actions tren tung segment, KHONG dung pop_mem.
    """
    if gate == GATE_MEMORY:
        assert pop_mem is not None, "MEMORY chosen but archive empty"
        return action_memory_global(pop, pop_mem, xl, xu)

    pop_new = pop.copy()
    for s, a in enumerate(seg_actions):
        v = segments[s]
        if a == SEG_KEEP:
            pass
        elif a == SEG_LOCAL:
            pop_new = action_local(pop_new, v, xl, xu)
        elif a == SEG_PREDICT:
            pop_new = action_predict(pop_new, pop_prev, v, xl, xu)
        elif a == SEG_DIVERSIFY:
            pop_new = action_diversify(pop_new, v, xl, xu)
        else:
            raise ValueError(f"bad seg action {a}")
    return repair(pop_new, xl, xu)

def apply_actions(pop, pop_prev, actions, segments, xl, xu, pop_mem=None):
    """
    DEPRECATED (4.1B): map cu 3=memory, 4=diversify. MEMORY gio la
    gate toan cuc — dung apply_hierarchical_response.

    Áp dụng hành động cho từng segment lên CÙNG một population

    pop:      population hiện tại (N, D)
    pop_prev: population 2 epoch trước — cho action_predict
    actions:  np.array (S,) — hành động mỗi segment
    segments: [[0], [1,...,9]]
    pop_mem:  population lich su tu MemoryArchive.query(), hoac None

    return: pop mới đã repair
    """
    pop_new = pop.copy()

    for s, a in enumerate(actions):
        seg_vars = segments[s]

        if a == 0:
            pass                                    # keep
        elif a == 1:
            pop_new = action_local(pop_new, seg_vars, xl, xu)
        elif a == 2:
            pop_new = action_predict(pop_new, pop_prev, seg_vars, xl, xu)
        elif a == 3:
            pop_new = (action_memory_global(pop_new, pop_mem, xl, xu)
                       if pop_mem is not None
                       else action_local(pop_new, seg_vars, xl, xu))
        elif a == 4:
            pop_new = action_diversify(pop_new, seg_vars, xl, xu)

    return repair(pop_new, xl, xu)