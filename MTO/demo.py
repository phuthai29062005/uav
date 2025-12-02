import numpy as np

# Cấu hình in ấn để số liệu thẳng hàng, dễ đọc
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ==============================================================================
# PHẦN 1: CLASS RBM (Mạng Nơ-ron Tái tạo)
# Lý thuyết: Mục 2.3 trong bài báo
# ==============================================================================

class RBM:
    def __init__(self, visible_dim, hidden_dim, learning_rate=0.1, name="RBM"):
        self.name = name
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Khởi tạo trọng số (W) và bias (a, b) theo Eq. [cite_start]11 [cite: 236]
        self.W = np.random.normal(0, 0.1, (visible_dim, hidden_dim))
        self.a = np.zeros(hidden_dim) # Bias lớp ẩn
        self.b = np.zeros(visible_dim) # Bias lớp hiển thị
        
        print(f"[{self.name}] Khởi tạo: Input={visible_dim} chiều -> Hidden={hidden_dim} chiều")

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # Eq. [cite_start]14: Encode (Visible -> Hidden) [cite: 248]
    # Chuyển đổi từ không gian bài toán -> không gian đặc trưng
    def encode(self, v):
        activation = np.dot(v, self.W) + self.a
        prob_h = self.sigmoid(activation)
        return prob_h

    # Eq. [cite_start]15: Decode (Hidden -> Visible) [cite: 249]
    # Chuyển đổi từ không gian đặc trưng -> không gian bài toán
    def decode(self, h):
        activation = np.dot(h, self.W.T) + self.b
        prob_v = self.sigmoid(activation)
        return prob_v

    # Huấn luyện RBM (Dùng Contrastive Divergence)
    # [cite_start]Được nhắc đến trong Algorithm 3 dòng 3 và 5 [cite: 304, 305]
    def train(self, data, epochs=20):
        print(f"[{self.name}] Đang training ({len(data)} mẫu, {epochs} epochs)...")
        for epoch in range(epochs):
            for v0 in data:
                # 1. Positive Phase (Xuôi)
                prob_h0 = self.encode(v0)
                h0 = (prob_h0 > np.random.rand(self.hidden_dim)).astype(float)
                
                # 2. Negative Phase (Ngược)
                prob_v1 = self.decode(h0)
                prob_h1 = self.encode(prob_v1)
                
                # 3. Cập nhật trọng số
                positive_grad = np.outer(v0, prob_h0)
                negative_grad = np.outer(prob_v1, prob_h1)
                
                self.W += self.lr * (positive_grad - negative_grad)
                self.b += self.lr * (v0 - prob_v1)
                self.a += self.lr * (prob_h0 - prob_h1)

# ==============================================================================
# PHẦN 2: TÍNH TOÁN KÍCH THƯỚC LỚP ẨN (D)
# Lý thuyết: Eq. 16 và Eq. [cite_start]17 [cite: 264-268]
# ==============================================================================

def calculate_hidden_dimension_D(population, N_top=10):
    print("\n" + "-"*30)
    print("BƯỚC 1: TÍNH TOÁN D (Hidden Dimension)")
    print("-"*30)

    # Lấy N_top cá thể tốt nhất
    elite_pop = population[:N_top]
    
    # Eq. 17: Biến đổi thành 0 hoặc 1 dựa trên ngưỡng 0.05
    binary_indicator = (elite_pop >= 0.05).astype(int)
    print("Ma trận chỉ thị nhị phân (Ngưỡng 0.05):")
    print(binary_indicator)
    
    # Eq. 17: Tính xác suất xuất hiện trên từng cột
    prob_Y = np.sum(binary_indicator, axis=0) / N_top
    print("\nXác suất quan trọng của từng chiều (prob_Y):")
    print(prob_Y)
    
    # Eq. 16: D = Tổng số chiều được giữ lại (nơi xác suất > 0.5)
    Y_d = (prob_Y > 0.5).astype(int)
    D = int(np.sum(Y_d))
    if D < 1: D = 1 # Fallback an toàn
    
    print(f"\n=> Kết luận: Số chiều ẩn D = {D} (Giữ lại các cột có giá trị 1 ở trên)")
    return D

# ==============================================================================
# [cite_start]PHẦN 3: THUẬT TOÁN RDA (ALGORITHM 3) [cite: 310]
# ==============================================================================

def RDA_Algorithm(target_pop, source_pop, target_task_name, source_task_name):
    print("\n" + "="*80)
    print(f"BẮT ĐẦU QUÁ TRÌNH RDA: {source_task_name} -> {target_task_name}")
    print("="*80)
    
    # [cite_start]1. Xác định kích thước D dựa trên Target Task [cite: 304]
    D = calculate_hidden_dimension_D(target_pop, N_top=len(target_pop))
    
    # [cite_start]2. Huấn luyện 2 mạng RBM [cite: 304, 305]
    print("\n" + "-"*30)
    print("BƯỚC 2: HUẤN LUYỆN RBM")
    print("-"*30)
    
    # RBM Target: Học cách giải nén từ D -> Target Space
    rbm_target = RBM(visible_dim=target_pop.shape[1], hidden_dim=D, name="RBM_Target")
    rbm_target.train(target_pop, epochs=50)
    
    # RBM Source: Học cách nén từ Source Space -> D
    rbm_source = RBM(visible_dim=source_pop.shape[1], hidden_dim=D, name="RBM_Source")
    rbm_source.train(source_pop, epochs=50)
    
    # [cite_start]3. Chuyển giao tri thức [cite: 306-308]
    print("\n" + "-"*30)
    print("BƯỚC 3: MAPPING & TRANSFER")
    print("-"*30)
    
    transferred_solutions = []
    
    # Lấy ví dụ 1 cá thể đầu tiên để in chi tiết
    sample_idx = 0 
    sol_source = source_pop[sample_idx]
    
    print(f"Ví dụ chuyển đổi cá thể số {sample_idx+1}:")
    print(f"   [1] GỐC (Source Space - {len(sol_source)} chiều):\n       {np.round(sol_source, 3)}")
    
    # [cite_start]Encode: Source -> Feature [cite: 307]
    feature = rbm_source.encode(sol_source)
    print(f"   [2] NÉN (Feature Space - {D} chiều):\n       {np.round(feature, 3)}")
    
    # [cite_start]Decode: Feature -> Target [cite: 308]
    sol_target = rbm_target.decode(feature)
    print(f"   [3] GIẢI NÉN (Target Space - {target_pop.shape[1]} chiều):\n       {np.round(sol_target, 3)}")
    
    return sol_target

# ==============================================================================
# MAIN (CHẠY CHƯƠNG TRÌNH)
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42) # Giữ cố định random để kết quả giống nhau mỗi lần chạy

    print("\n" + "#"*80)
    print("KHỞI TẠO DỮ LIỆU BAN ĐẦU")
    print("#"*80)

    # --- TẠO DỮ LIỆU TARGET (6 chiều) ---
    # Đặc điểm: Cột 0, 2, 4 có giá trị lớn. Cột 1, 3, 5 rất nhỏ (gần 0).
    # Điều này mô phỏng bài toán tối ưu thưa (sparse).
    target_data = np.array([
        [0.85, 0.01, 0.92, 0.02, 0.75, 0.00],
        [0.78, 0.03, 0.88, 0.01, 0.65, 0.04],
        [0.91, 0.00, 0.79, 0.04, 0.81, 0.01],
        [0.65, 0.02, 0.95, 0.01, 0.55, 0.02],
        [0.82, 0.01, 0.84, 0.03, 0.72, 0.00],
        [0.75, 0.04, 0.77, 0.01, 0.68, 0.03]
    ])
    
    print("\n>>> MA TRẬN TARGET TASK (Dữ liệu đích - 6 chiều):")
    print("Chú ý: Các cột lẻ (1, 3, 5) có giá trị rất nhỏ -> Thuật toán sẽ loại bỏ chúng.")
    print(target_data)

    # --- TẠO DỮ LIỆU SOURCE (8 chiều) ---
    # [cite_start]Source có số chiều KHÁC Target (8 vs 6) để chứng minh khả năng Heterogeneous Transfer [cite: 255]
    source_data = np.random.rand(6, 8)
    
    print("\n>>> MA TRẬN SOURCE TASK (Dữ liệu nguồn - 8 chiều):")
    print("Dữ liệu ngẫu nhiên, số chiều lớn hơn Target.")
    print(source_data)

    # --- CHẠY THUẬT TOÁN ---
    RDA_Algorithm(
        target_pop=target_data,
        source_pop=source_data,
        target_task_name="Target Task (6D)",
        source_task_name="Source Task (8D)"
    )