import numpy as np
from pymoo.problems.dynamic.df import DF1
from pymoo.indicators.hv import HV

class Memory:
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

def repair(pop):
    """
    Sửa nghiệm vi phạm ràng buộc.
    Trên CEC: chỉ cần clip về [0,1]
    Trên UAV: thêm kiểm tra va chạm, độ cao, tốc độ, năng lượng
    """
    return np.clip(pop, 0, 1)

def compute_env_key(pop):
    """
    Tạo chữ ký môi trường từ population đã hội tụ.
    Ý tưởng: môi trường giống nhau → POS ở cùng vị trí → centroid giống nhau
    
    return: np.array — khoá dùng cho Memory.store() và Memory.retrieve()
    """
    centroid = np.mean(pop, axis=0)   # gợi ý: np.mean(pop, axis=0)
    return centroid

def compute_change_vector(elite, problem_new, segments, D):
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

def compute_entropy(pop):
    """
    Đo độ đa dạng của population, chuẩn hoá về [0, 1]
    """
    # std của mỗi biến, rồi lấy trung bình
    # std tối đa của phân bố đều trên [0,1] là 1/sqrt(12) ≈ 0.289
    x = np.std(pop, axis=0)
    return np.clip(np.mean(x) / 0.289, 0, 1)

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

def build_state(c, entropy, hv_drop, g_last, tau_t, phase):
    """
    Ghép tất cả thành 1 vector duy nhất cho RL agent
    
    return: np.array shape (S + 4,)
            với S=2 → shape (6,)
    """
    g_norm = g_last / tau_t     # → ∈ [0, 1)
    state = np.concatenate([c, [entropy, hv_drop, g_norm, phase]])
    return state
    
def action_keep(pop, segment_vars):
    return pop.copy()

def action_local(pop, segment_vars, sigma=0.05):
    """
    Thêm nhiễu Gaussian nhỏ vào các biến của segment
    sigma: độ lớn nhiễu (nhỏ = tinh chỉnh nhẹ)
    """
    pop_new = pop.copy()
    for i in range(pop.shape[0]):
        for var_idx in segment_vars:
            noise = np.random.normal(0, sigma)
            pop_new[i, var_idx] += noise
            pop_new[i, var_idx] = np.clip(pop_new[i, var_idx], 0, 1)
    return pop_new

def action_predict(pop, pop_prev, segment_vars, noise=0.02):
    """
    Dự đoán tuyến tính dựa trên hướng dịch chuyển của CENTROID
    """
    pop_new = pop.copy()
    
    # 1. Tính centroid của pop tại segment_vars
    centroid = np.mean(pop[:, segment_vars], axis=0)        # gợi ý: np.mean(pop[:, segment_vars], axis=0)
    
    # 2. Tính centroid của pop_prev tại segment_vars
    centroid_prev = np.mean(pop_prev[:, segment_vars], axis=0)
    
    # 3. Hướng dịch chuyển
    direction = centroid - centroid_prev
    
    # 4. Dịch toàn bộ pop theo direction
    pop_new[:, segment_vars] += direction
    
    # 5. Thêm nhiễu nhỏ để tránh dồn cục
    pop_new[:, segment_vars] += np.random.normal(0, noise, size=pop_new[:, segment_vars].shape)

    # 6. Clip
    pop_new = np.clip(pop_new, 0, 1)
    
    return pop_new

def action_memory(pop, memory, key, segment_vars):
    pop_mem = memory.retrieve(key)
    
    if pop_mem is None:
        return action_local(pop, segment_vars)
    
    pop_new = pop.copy()
    n = min(len(pop), len(pop_mem))
    pop_new[:n, segment_vars] = pop_mem[:n, segment_vars]
    
    return repair(pop_new)    # ← thêm bước repair
    
def action_diversify(pop, segment_vars, ratio=0.5):
    """
    Thay ratio% cá thể bằng giá trị random ở các biến của segment
    """
    pop_new = pop.copy()
    n_replace = int(pop.shape[0] * ratio)
    indices = np.random.choice(pop.shape[0], n_replace, replace=False)

    for idx in indices:
        for var_idx in segment_vars:
            pop_new[idx, var_idx] = np.random.rand()  # random [0,1]

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

def count_fe(actions, N, n_elite, n_segments):
    """
    Đếm số lần evaluate cá thể trong 1 epoch phản ứng

    actions:    np.array (S,) — hành động mỗi segment
    N:          population size
    n_elite:    số elite dùng tính change vector
    n_segments: S
    """
    # Chi phí cố định: compute_change_vector
    #   mỗi elite cần 1 lần evaluate gốc + S lần evaluate nhiễu
    fe = n_elite * (1 + n_segments)  # 1 lần evaluate gốc + S lần evaluate nhiễu

    # Chi phí biến đổi: nếu CÓ segment nào khác keep
    #   → phải evaluate lại cả population, nhưng CHỈ 1 lần
    if np.any(actions != ACTION_KEEP):
        fe += N

    return fe

def apply_actions(pop, pop_prev, actions, segments, memory, key):
    """
    Áp dụng hành động cho từng segment lên CÙNG một population

    pop:      population hiện tại (N, D)
    pop_prev: population 2 epoch trước — cho action_predict
    actions:  np.array (S,) — hành động mỗi segment
    segments: [[0], [1,...,9]]
    memory:   đối tượng Memory
    key:      khoá môi trường để retrieve

    return: pop mới đã repair
    """
    pop_new = pop.copy()

    for s, a in enumerate(actions):
        seg_vars = segments[s]

        if a == 0:
            pass                                    # keep
        elif a == 1:
            pop_new = action_local(pop_new, seg_vars)
        elif a == 2:
            pop_new = action_predict(pop_new, pop_prev, seg_vars)
        elif a == 3:
            pop_new = action_memory(pop_new, memory, key, seg_vars)
        elif a == 4:
            pop_new = action_diversify(pop_new, seg_vars)

    return repair(pop_new)