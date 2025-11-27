import numpy as np

class RBM:
    """
    Cài đặt mạng Restricted Boltzmann Machine đơn giản để chuyển giao tri thức.
    """
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Khởi tạo trọng số và bias ngẫu nhiên
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, data, epochs=20):
        """Huấn luyện RBM bằng giải thuật Contrastive Divergence (CD-1)"""
        num_samples = data.shape[0]
        
        for epoch in range(epochs):
            # 1. Positive phase (Visible -> Hidden)
            # P(h=1|v)
            pos_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
            pos_hidden_states = (pos_hidden_probs > np.random.rand(*pos_hidden_probs.shape)).astype(float)
            
            # Tính tương quan Positive (v * h)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # 2. Negative phase (Hidden -> Visible -> Hidden)
            # P(v=1|h) - Tái tạo lại dữ liệu (Reconstruction)
            neg_visible_probs = self.sigmoid(np.dot(pos_hidden_states, self.weights.T) + self.visible_bias)
            
            # P(h=1|v_recon)
            neg_hidden_probs = self.sigmoid(np.dot(neg_visible_probs, self.weights) + self.hidden_bias)
            
            # Tính tương quan Negative
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # 3. Cập nhật trọng số và bias
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_samples)
            self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
            self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def transform(self, data):
        """Mã hóa: Visible Space -> Hidden Space (Feature Space)"""
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        # Trả về trạng thái (0/1) hoặc xác suất đều được, ở đây dùng xác suất cho mượt
        return hidden_probs

    def inverse_transform(self, hidden_data):
        """Giải mã: Hidden Space -> Visible Space (Target Space)"""
        visible_probs = self.sigmoid(np.dot(hidden_data, self.weights.T) + self.visible_bias)
        return visible_probs

def calculate_hidden_dimension(P, n_top):
    """
    Tính kích thước D của lớp ẩn dựa trên độ thưa (Eq. 16, 17)
    """
    # Chọn top các cá thể tốt nhất (Giả sử P đã được sắp xếp hoặc lấy mẫu đại diện)
    # Trong bài báo dùng top N_top cá thể để tính
    # Ở đây ta dùng logic đơn giản hóa: Dựa trên toàn bộ P nếu chưa có fitness
    # Hoặc lấy mẫu ngẫu nhiên nếu P quá lớn
    
    samples = P[:n_top] if len(P) > n_top else P
    dim_visible = P.shape[1]
    
    # Hàm f(x): 0 nếu x < 0.05, 1 nếu ngược lại (Eq. 17)
    # Lưu ý: P đang ở Unified Space [0, 1]
    f_x = (samples >= 0.05).astype(float)
    
    # Tính xác suất p(Y_d = 1)
    probs = np.mean(f_x, axis=0)
    
    # Tính D (Eq. 16)
    D = np.sum(probs)
    
    # Đảm bảo D ít nhất là 1 để không lỗi
    return int(max(D, 1))

def RDA(target_idx, source_indices, P, fitness, n_top=10):
    """
    Algorithm 3: RBM-based Domain Adaptation
    Input:
        target_idx: Index tác vụ đích
        source_indices: List index các tác vụ nguồn (L_c từ MTS)
        P: List quần thể các task (Unified Space)
        fitness: Mảng fitness tương ứng để chọn lọc
        n_top: Số lượng cá thể ưu tú cần chuyển giao
    Output:
        T_c: Tập hợp các giải pháp đã được chuyển giao (Transferred Solutions)
    """
    # Lấy dữ liệu đích
    P_target = P[target_idx]
    
    # 1. Xác định kích thước hidden layer D
    # (Tính toán dựa trên đặc điểm quần thể đích)
    D = calculate_hidden_dimension(P_target, n_top)
    
    # 2. Huấn luyện RBM cho tác vụ Đích (RBM_c)
    # Input dimension là số chiều của Unified Space (dim_max)
    input_dim = P_target.shape[1]
    
    rbm_target = RBM(n_visible=input_dim, n_hidden=D)
    rbm_target.train(P_target, epochs=10) # Train nhanh 10 epoch
    
    T_c = []
    
    # 3. Vòng lặp qua các tác vụ nguồn
    for src_idx in source_indices:
        P_source = P[src_idx]
        fit_source = fitness[src_idx]
        
        # --- S1: Chọn lọc (Selection) ---
        # Lấy index của top N_top cá thể tốt nhất (fitness nhỏ nhất)
        # Bỏ qua index 0 (do logic 1-based của bạn trong main)
        valid_indices = np.argsort(fit_source[1:])[:n_top] 
        # +1 vì argsort trả về index bắt đầu từ 0, nhưng mảng ta cắt từ 1
        S1 = P_source[valid_indices + 1] 
        
        # --- Huấn luyện RBM Nguồn ---
        rbm_source = RBM(n_visible=input_dim, n_hidden=D)
        rbm_source.train(P_source, epochs=10)
        
        # --- S2: Mã hóa (Mapping) ---
        # Source Space -> Hidden Space
        S2 = rbm_source.transform(S1)
        
        # --- S3: Giải mã (Recovery) ---
        # Hidden Space -> Target Space (dùng RBM Target)
        S3 = rbm_target.inverse_transform(S2)
        
        # Thêm vào tập kết quả
        if len(T_c) == 0:
            T_c = S3
        else:
            T_c = np.vstack((T_c, S3))
            
    return T_c