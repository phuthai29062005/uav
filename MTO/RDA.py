import numpy as np

# Hàm kích hoạt Sigmoid (Eq. 14, 15)
def sigmoid(x):
    # Clip x để tránh overflow khi tính exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

class RBM:
    """
    Restricted Boltzmann Machine (Mục 2.3) [cite: 232-243]
    """
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # Khởi tạo trọng số nhỏ theo phân phối chuẩn
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_hidden) # Bias ẩn
        self.b = np.zeros(n_visible) # Bias hiển thị
        
        # Thêm momentum để học nhanh hơn
        self.mW = np.zeros_like(self.W)
        self.ma = np.zeros_like(self.a)
        self.mb = np.zeros_like(self.b)

    def train(self, data, epochs=20, lr=0.1, momentum=0.5):
        """
        Huấn luyện RBM bằng Contrastive Divergence (CD-1)
        Được gọi tại dòng 3 và 5 của Algorithm 3 [cite: 304, 305]
        """
        # Đảm bảo data là numpy array
        data = np.array(data)
        num_samples = data.shape[0]
        
        for _ in range(epochs):
            # --- Positive phase (Visible -> Hidden) ---
            v0 = data
            # P(h=1|v) - Eq. 15 [cite: 249]
            h0_prob = sigmoid(np.dot(v0, self.W) + self.a)
            # Lấy mẫu h0 (Stochastic)
            h0_state = (np.random.rand(num_samples, self.n_hidden) < h0_prob).astype(float)

            # --- Negative phase (Hidden -> Visible -> Hidden) ---
            # Reconstruction (v1) -> P(v=1|h) - Eq. 14 [cite: 248]
            v1_prob = sigmoid(np.dot(h0_state, self.W.T) + self.b)
            # Với dữ liệu liên tục [0,1], ta dùng trực tiếp xác suất làm trạng thái (Mean-field approximation)
            v1_state = v1_prob  
            
            h1_prob = sigmoid(np.dot(v1_state, self.W) + self.a)

            # --- Update Weights & Biases (Contrastive Divergence) ---
            pos_grad = np.dot(v0.T, h0_prob)
            neg_grad = np.dot(v1_state.T, h1_prob)

            # Cập nhật có Momentum
            self.mW = momentum * self.mW + lr * (pos_grad - neg_grad) / num_samples
            self.ma = momentum * self.ma + lr * np.mean(h0_prob - h1_prob, axis=0)
            self.mb = momentum * self.mb + lr * np.mean(v0 - v1_state, axis=0)

            self.W += self.mW
            self.a += self.ma
            self.b += self.mb

    def transform(self, v):
        """Encode: Chuyển từ Search Space sang Feature Space (Eq. 15)"""
        v = np.array(v)
        return sigmoid(np.dot(v, self.W) + self.a)

    def recover(self, h):
        """Decode: Chuyển từ Feature Space về Search Space (Eq. 14)"""
        h = np.array(h)
        return sigmoid(np.dot(h, self.W.T) + self.b)

def get_hidden_dimension(population, n_top=None):
    """
    Tính chiều không gian ẩn D dựa trên độ thưa (Sparsity) của top cá thể.
    Eq. 16: D = Sum(Y_d) [cite: 264]
    Eq. 17: P(Y_d=1) dựa trên ngưỡng 0.05 [cite: 268]
    """
    pop_arr = np.array(population)
    
    # Nếu không chỉ định n_top, lấy toàn bộ (hoặc một phần để tính)
    if n_top is None:
        n_top = len(pop_arr)
    
    # Lấy top cá thể đầu tiên (giả sử pop đã được sort hoặc lấy mẫu đại diện)
    # Trong bài báo dùng top N_top, ở đây ta dùng toàn bộ mẫu truyền vào để ước lượng tốt hơn
    samples = pop_arr[:n_top]
    
    # Eq. 17: Tính xác suất kích hoạt (Gene > 0.05 được coi là active/1)
    # f(x) = 1 nếu x > 0.05 else 0
    f_x = (samples > 0.05).astype(float)
    prob_Y = np.mean(f_x, axis=0)
    
    # Eq. 16: D = Tổng các biến nhị phân Y_d (Ước lượng bằng tổng xác suất)
    # Ta làm tròn để ra số nguyên
    D = int(np.round(np.sum(prob_Y)))
    
    # Đảm bảo D tối thiểu là 1 để RBM hoạt động
    return max(1, D)

def RDA(P, fitness_list, target_idx, L_sources, transferred_tasks=5, Ntop=10):
    """
    RBM-based Domain Adaptation (Algorithm 3) 
    Input:
        P: List chứa quần thể của tất cả các task
        fitness_list: List chứa fitness (để sort lấy elite)
        target_idx: ID task đích
        L_sources: List ID các task nguồn
    """
    # Chuyển đổi an toàn sang numpy array
    pop_target = np.array(P[target_idx])
    fit_target = np.array(fitness_list[target_idx])
    
    #  Line 2: Tính D dựa trên Target Task
    # Sort để lấy top cá thể tốt nhất phục vụ tính D chính xác hơn
    sorted_idx = np.argsort(fit_target)
    pop_target_sorted = pop_target[sorted_idx]
    
    # Tính D
    D = get_hidden_dimension(pop_target_sorted, Ntop)
    
    #  Line 3: Train RBM cho Target (Học cách Decode)
    rbm_target = RBM(n_visible=pop_target.shape[1], n_hidden=D)
    rbm_target.train(pop_target, epochs=20) # Epochs thấp để chạy nhanh

    transferred_solutions = []

    #  Line 4: Duyệt qua các nguồn
    for src_id in L_sources:
        pop_source = np.array(P[src_id])
        fit_source = np.array(fitness_list[src_id])
        
        #  Line 5: Train RBM cho Source (Học cách Encode)
        # Lưu ý: Train trên toàn bộ quần thể nguồn để nắm bắt phân phối
        rbm_source = RBM(n_visible=pop_source.shape[1], n_hidden=D)
        rbm_source.train(pop_source, epochs=20)
        
        # [cite: 306] Line 6: Chọn Top N_top cá thể tốt nhất từ nguồn (S1)
        src_sorted_idx = np.argsort(fit_source)
        S1 = pop_source[src_sorted_idx[:Ntop]]
        
        # [cite: 307] Line 7: Encode Source -> Feature Space (S2)
        S2 = rbm_source.transform(S1)
        
        # [cite: 308] Line 8: Decode Feature Space -> Target Space (S3)
        S3 = rbm_target.recover(S2)
        
        # [cite: 309] Line 9: Gộp vào tập giải pháp chuyển giao
        if len(transferred_solutions) == 0:
            transferred_solutions = S3
        else:
            transferred_solutions = np.vstack([transferred_solutions, S3])
            
    return np.array(transferred_solutions)