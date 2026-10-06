# PR-DRL-DMOEA: Tối ưu đa mục tiêu động nhận biết vùng Pareto cho tái định vị UAV cảnh báo sớm đa thiên tai

Đặc tả nghiên cứu mới từ đầu — phân vùng tập nghiệm trong không gian mục tiêu

**Bản thiết kế nghiên cứu**

*Nguồn biên tập chính (canonical) là file Markdown này. Bản PDF đi kèm chỉ để đọc/in. Phiên bản O1-aligned: 2026-09-20.*

*Lưu ý về phiên bản: tài liệu này là phiên bản O1-adopted (D2-D 2026-09-20). Bản gốc 2026-09-14 (SHA256: 66e667e63a7e9570680b2115bf3604836156415197c222e3ad7080b1746b9eae) được bảo tồn không thay đổi tại paper/PR_DRL_DMOEA_UAV_New_Design_2026-09-14.pdf.*

---

## Tóm tắt

Tài liệu này xây dựng một phương án nghiên cứu mới từ đầu, không phụ thuộc vào bất kỳ cài đặt thuật toán cũ nào. Ý tưởng trung tâm là thay vì chia biến quyết định thành các khối, ta chia tập nghiệm xấp xỉ Pareto trong không gian mục tiêu thành các vùng trade-off ổn định theo các hướng tham chiếu. Khi môi trường thay đổi, cùng các nghiệm cũ được đánh giá trong môi trường mới để đo mức suy giảm theo từng vùng. Nếu các vùng phản ứng không đồng đều, một bộ điều khiển học tăng cường sâu dùng chung cho mọi vùng sẽ chọn toán tử phản ứng riêng cho từng vùng: KEEP, LOCAL, PREDICT, DIVERSIFY.

Phương pháp tạm gọi là PR-DRL-DMOEA (Pareto-Region-aware Deep Reinforcement Learning-assisted Dynamic Multi-Objective Evolutionary Algorithm). Đóng góp cần kiểm chứng là: (i) đo suy giảm cục bộ theo vùng bằng cùng tập nghiệm trước/sau thay đổi; (ii) kiểm tra trước bằng regional-oracle xem điều khiển theo vùng có thật sự đáng học hay không; (iii) dùng một policy chia sẻ tham số để chọn toán tử và ngân sách FE riêng cho từng vùng; và (iv) áp dụng cơ chế này vào bài tái định vị đội UAV.

**Trạng thái**: thiết kế nghiên cứu mới; chưa được xem là kết quả thực nghiệm hay tuyên bố vượt trội.

**Trạng thái triển khai (2026-09-20)**:

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| KEEP (B_extra=0) | VALIDATED — đã triển khai | Response FE = 0 |
| LOCAL (sigma_L=0.02) | VALIDATED — đã triển khai | B=10 fixed budget |
| PREDICT (sigma_P=0.005, lambda_P=1.0) | VALIDATED — đã triển khai | UNRESTRICTED survival; fallback=LOCAL |
| DIVERSIFY (sigma_D=0.15) | VALIDATED — đã triển khai | B=10 fixed budget |
| Response budget B=10 | VALIDATED — đã triển khai | Ngân sách cố định per change event |
| Base MOEA | CURRENT = NSGA-II | NSGA-III là khuyến nghị cho tương lai |
| Regional-oracle (E2) | CHƯA CHẠY — Giai đoạn A | Bước bắt buộc trước DRL |
| Shared Double DQN | CHƯA XÂY DỰNG — DRL_STATUS = NOT_STARTED | Chỉ sau khi oracle xác nhận |
| Action-dependent B_r(a) | CHƯA TRIỂN KHAI — thiết kế tương lai | Công thức (48) là kế hoạch, chưa là sản xuất |
| NSGA-III base | CHƯA TRIỂN KHAI — khuyến nghị tương lai | §7.2 |

**Từ khóa**: dynamic multi-objective optimization, DMOEA, Pareto region, objective-space decomposition, deep reinforcement learning, UAV, Age of Information, multi-hazard early warning.

---

## 1. Bối cảnh và phạm vi

### 1.1. Bài toán ứng dụng

Xét một vùng thiên tai có tập cảm biến mặt đất cố định Q = {(q_m, theta_m)}_{m=1}^{M}, trong đó q_m là vị trí cảm biến và theta_m là loại hiểm họa mà cảm biến theo dõi (lũ, sạt lở, cháy). Hạ tầng truyền thông mặt đất có thể bị suy giảm hoặc mất hoàn toàn. Một đội gồm K UAV được dùng như các trạm chuyển tiếp di động.

Môi trường thay đổi theo thời gian vì trường nguy cơ, mức ưu tiên của các khu vực, điều kiện kênh, pin UAV và tuổi thông tin đều biến thiên. Do đó ta không giải một MOP tĩnh duy nhất mà giải chuỗi bài toán đa mục tiêu phụ thuộc thời gian.

### 1.2. Câu hỏi nghiên cứu mới

Câu hỏi cốt lõi của nghiên cứu này là: Khi môi trường thay đổi, các vùng trade-off khác nhau của tập nghiệm Pareto có suy giảm khác nhau hay không, và nếu có thì việc dùng toán tử phản ứng khác nhau theo vùng có cải thiện khả năng bám Pareto front so với một phản ứng toàn cục hay không?

### 1.3. Phạm vi

Tài liệu tập trung vào bài tái định vị UAV và phục vụ cảm biến, không tối ưu vị trí đặt cảm biến. Phần cốt lõi của thuật toán được kiểm chứng trước trên benchmark DMOP, sau đó mới đánh giá trên mô phỏng UAV.

---

## 2. Tham số hệ thống UAV

Bảng dưới là cấu hình khởi đầu phục vụ mô phỏng. Đây là giá trị điển hình, không phải hằng số bắt buộc.

| Ký hiệu | Ý nghĩa | Giá trị khởi đầu đề xuất |
|---|---|---|
| A | Vùng quan trắc | 1000 x 1000 m |
| M | Số cảm biến | 50-200 |
| H | Tập loại hiểm họa | 3 loại |
| q_0 | Vị trí trạm mặt đất | 1 điểm cố định |
| K | Số UAV | 3-10 |
| tau | Độ dài một epoch | 30s |
| z_min, z_max | Giới hạn độ cao | 50-300m |
| v_max | Tốc độ tối đa | 20m/s |
| d_min | Khoảng cách an toàn UAV-UAV | 20m |
| gamma_th | Ngưỡng SNR sensor-UAV | theo kịch bản |
| gamma_0_th | Ngưỡng SNR UAV-base | theo kịch bản |
| r_m | Tốc độ dữ liệu cảm biến m | theo loại cảm biến |
| C_k | Dung lượng backhaul UAV k | theo mô hình kênh |
| E_0_k | Pin ban đầu | theo loại UAV |
| E_res_k | Pin dự trữ tối thiểu | theo chính sách an toàn |
| c_mv, c_hv | Hệ số năng lượng bay/hover | hiệu chỉnh từ mô hình UAV |

---

## 3. Đặc tả Input / Output

### 3.1. Input tại epoch t

1. Vị trí UAV trước đó: P(t-1) = {p_k(t-1)}_{k=1}^{K}, p_k = (x_k, y_k, z_k).
2. Tuổi thông tin của cảm biến: Delta(t-1) = [Delta_1(t-1), ..., Delta_M(t-1)].
3. Pin UAV: E(t-1) = [E_1(t-1), ..., E_K(t-1)].
4. Quan sát mới o(t) từ các cảm biến được phục vụ.
5. Belief môi trường b(t) về trường nguy cơ, được cập nhật từ lịch sử và quan sát.
6. Quần thể nghiệm gần nhất P(t-1) của bộ tối ưu đa mục tiêu.

### 3.2. Output tại epoch t

1. Tập các phương án UAV không trội: P(t) = {X_i(t)}_{i=1}^{N}.
2. Mỗi nghiệm X_i chứa vị trí đội UAV và, tùy cách cài đặt, ma trận phân công cảm biến-UAV hoặc tham số để bộ giải phân công nội suy ra A_i(t).
3. Một phương án vận hành cuối cùng có thể được chọn từ Pareto set theo ưu tiên thời điểm thực tế; bước quyết định cuối này nằm ngoài lõi DMOEA.

---

## 4. Mô hình toán học bài UAV

### 4.1. Biến quyết định

Mô hình đầy đủ dùng biến hỗn hợp: X(t) = (P(t), A(t)), với P(t) = {p_k(t)}_{k=1}^{K}, A(t) = [a_{mk}(t)] in {0,1}^{MxK}.

Trong đó a_{mk} = 1 nếu cảm biến m được UAV k phục vụ ở epoch hiện tại. Để giảm chiều, một triển khai thực dụng có thể tối ưu chỉ P(t) ở outer loop và dùng một decoder cố định hoặc một bài toán MILP bên trong để sinh A(t).

### 4.2. Mô hình kênh Air-to-Ground

Với cảm biến m tại q_m = (u_m, v_m, 0) và UAV k tại p_k = (x_k, y_k, z_k):

$$
r_{mk} = sqrt((x_k - u_m)^2 + (y_k - v_m)^2)
d_{mk} = sqrt(r_{mk}^2 + z_k^2)
vartheta_{mk} = arctan(z_k / (r_{mk} + epsilon))
$$

Một dạng mô hình xác suất LoS thường dùng là:

$$
P_LoS_{mk} = 1 / (1 + a*exp[-b*(vartheta_mk - a)])    (1)
$$

Suy hao trung bình:

$$
L_{mk} = 20*log10(4*pi*f_c*d_{mk}/c) + P_LoS_{mk}*eta_LoS + (1 - P_LoS_{mk})*eta_NLoS    (2)
$$

### 4.3. Tuổi thông tin (AoI)

Định nghĩa biến phục vụ thành công s_m(t) = sum_{k=1}^{K} a_{mk}(t) * 1{gamma_{mk}(t) >= gamma_th}.

AoI cập nhật theo:

$$
Delta_m(t+1) = delta_m(t)    nếu s_m(t) >= 1
Delta_m(t+1) = Delta_m(t) + tau    nếu s_m(t) = 0    (3)
$$

### 4.4. Trọng số nguy cơ động

$$
w_tilde_m(t) = w_0 + lambda_h * rho_hat_{theta_m}(q_m, t)
w_m(t) = w_tilde_m(t) / (sum_{j=1}^{M} w_tilde_j(t) + epsilon)    (4)
$$

### 4.5. Ba hàm mục tiêu

**Mục tiêu 1 - AoI ưu tiên trung bình:**

$$
f_1(X,t) = sum_{m=1}^{M} w_m(t) * Delta_m(t+1)    (5)
$$

**Mục tiêu 2 - AoI tệ nhất theo nhóm hiểm họa:**

$$
f_2(X,t) = max_{h in H} [sum_{m in M_h} w_m(t)*Delta_m(t+1)] / [sum_{m in M_h} w_m(t) + epsilon]    (6)
$$

**Mục tiêu 3 - Năng lượng đội UAV:**

$$
f_3(X,t) = sum_{k=1}^{K} [E_move_k(t) + E_hover_k(t) + E_comm_k(t)]    (7)
E_move_k(t) = c_mv * ||p_k(t) - p_k(t-1)||_2    (8)
E_hover_k(t) = c_hv * tau    (9)
$$

### 4.6. Ràng buộc

$$
(x_k, y_k) in A, z_min <= z_k <= z_max    (10)
||p_k(t) - p_k(t-1)||_2 <= v_max * tau    (11)
||p_k(t) - p_j(t)||_2 >= d_min, k != j    (12)
sum_{k=1}^{K} a_{mk}(t) <= 1, forall m    (13)
a_{mk}(t) * [gamma_th - gamma_{mk}(t)] <= 0    (14)
gamma_{k0}(t) >= gamma_0_th khi UAV k hoạt động    (15)
sum_{m=1}^{M} a_{mk}(t)*r_m <= C_k(t)    (16)
E_k(t) = E_k(t-1) - E_move_k - E_hover_k - E_comm_k >= E_res_k    (17)
$$

### 4.7. Phát biểu DMOP

$$
min_{X(t)} F(X,t) = (f_1(X,t), f_2(X,t), f_3(X,t))    (18)
sao cho X(t) in Omega(t, X(t-1), E(t-1), b(t))    (19)
$$

Miền khả thi phụ thuộc lịch sử, do đó bài UAV chứa time-linkage tự nhiên.

### 4.8. Quan sát bộ phận và belief

$$
o(t) = {y_m(t) : s_m(t) >= 1}
y_m(t) = rho_{theta_m}(q_m, t) + epsilon_m(t)    (20)
b(t) = B(b(t-1), o(t), u(t-1))    (21)
$$

---

## 5. Mô hình đề xuất: PR-DRL-DMOEA

### 5.1. Ý tưởng cốt lõi

Gọi quần thể ngay trước thay đổi là P^- = {x_i}_{i=1}^{N}. Mỗi x_i là một nghiệm hoàn chỉnh. Trên UAV, một x_i mô tả toàn bộ đội UAV.

$$
F^-_i = F(x_i, t^-),  F^+_i = F(x_i, t^+)    (22)
$$

Sau đó chia các nghiệm theo vùng trade-off trong objective space. Mỗi vùng nhận một tín hiệu suy giảm và một action riêng.

**Giả thuyết nghiên cứu:** Tồn tại r_1, r_2 sao cho c_{r1} != c_{r2} và a*_{r1} != a*_{r2}. (23)

Nếu cả hai điều kiện này không xuất hiện đủ thường xuyên, điều khiển theo vùng không có lý do tồn tại.

### 5.2. Chuẩn hóa objective space

$$
z*_m = min_i {f^-_{im}, f^+_{im}}
z_nad_m = max_i {f^-_{im}, f^+_{im}}    (24)
f_tilde_{im} = (f_{im} - z*_m) / (z_nad_m - z*_m + epsilon)    (25)
$$

### 5.3. Phân vùng bằng hướng tham chiếu

$$
V = {v_r}_{r=1}^{R}, v_r >= 0, sum_{m=1}^{Mo} v_{rm} = 1    (26)
R = C(H + Mo - 1, Mo - 1)    (27)
u^-_i = F_tilde^-_i / (||F_tilde^-_i||_2 + epsilon)
r(i) = argmax_r (u^-_i)^T v_r / (||v_r||_2 + epsilon)    (28)
I_r = {i : r(i) = r}    (29)
$$

**Nguyên tắc quan trọng**: membership I_r được đóng băng trong phép đo trước/sau. Không repartition ở t+ rồi đem "region 1 mới" so với "region 1 cũ".

### 5.4. Mức thay đổi theo vùng

$$
d_i = ||F_tilde^+_i - F_tilde^-_i||_2    (30)
s_r = median_{i in I_r} d_i    (31)
c_r(t) = 1 - exp[-s_r(t) / (kappa*q_{t-1} + epsilon)]    (32)
q_t = (1 - rho)*q_{t-1} + rho*median_r s_r(t)    (33)
$$

### 5.5. Tín hiệu suy giảm convergence theo vùng

Với PBI scalarization:

$$
d^(r)_{1,i} = (F_tilde_i)^T v_r / ||v_r||_2
d^(r)_{2,i} = ||F_tilde_i - d^(r)_{1,i} * v_r/||v_r||_2||_2
g_r(F_tilde_i) = d^(r)_{1,i} + theta_PBI * d^(r)_{2,i}    (34)
l_r = median_{i in I_r} [g_r(F_tilde^+_i) - g_r(F_tilde^-_i)] / (|g_r(F_tilde^-_i)| + epsilon)    (35)
$$

### 5.6. Độ đa dạng và occupancy của vùng

$$
o_r = |I_r| / N    (36)
D_r = median_{i in I_r} min_{j in I_r, j!=i} ||F_tilde_i - F_tilde_j||_2    (37)
$$

### 5.7. Chỉ số heterogeneity — kiểm tra trước khi học RL

$$
RHI(t) = std_{r in R_act}(s_r) / (mean_{r in R_act}(s_r) + epsilon)    (38)
$$

RHI cao nghĩa là các vùng bị tác động khác nhau. RHI thấp nghĩa là môi trường gây một biến đổi gần đồng đều trên toàn approximation set.

RHI phải được so với random partition control có cùng kích thước vùng. Nếu reference-direction partition không tạo heterogeneity cao hơn random partition, không có bằng chứng rằng cấu trúc objective-space đang cung cấp thông tin hữu ích.

### 5.8. Không gian trạng thái theo vùng

$$
z_r(t) = [c_r, l_r, o_r, D_r, Delta_c_r]    (39)
Delta_c_r = c_r(t) - c_r(t-1)
g(t) = [Delta_HV_global, D_global, B_remain_FE]    (40)
s_r(t) = [z_r(t), g(t)]    (41)
$$

Không đưa "pha chu kỳ thật" hoặc thông tin tương lai vào state.

### 5.9. Không gian hành động

Core version dùng bốn action:

$$
A = {KEEP, LOCAL, PREDICT, DIVERSIFY}    (42)
$$

**KEEP**

Giữ nguyên decision vectors của vùng. Các nghiệm vẫn phải được đánh giá ở môi trường mới nếu objective đã thay đổi, nhưng không sinh thêm response offspring:

$$
B_extra_r = 0    (43)
$$

**LOCAL**

Với nghiệm x:

$$
x' = Pi_Omega_t [x + sigma_L*(u-l) o xi],  xi ~ N(0,I)    (44)
$$

với sigma_L nhỏ. Giá trị sản xuất: **sigma_L = 0.02**.

**PREDICT**

Dùng centroid decision-space lịch sử của cùng objective region:

$$
mu^{t-1}_r = (1/|I^{t-1}_r|) * sum_{i in I^{t-1}_r} x_i
Delta_hat_r = lambda_P * (mu^{t-1}_r - mu^{t-2}_r)    (45)
x' = Pi_Omega_t [x + Delta_hat_r + sigma_P*(u-l) o xi]    (46)
$$

Giá trị sản xuất cuối cùng (O1, áp dụng D2-D 2026-09-20): **sigma_P = 0.005**, **lambda_P = 1.0**.

Nếu không đủ lịch sử, PREDICT phải fallback về LOCAL hoặc một predictor được định nghĩa trước; không được dùng thông tin tương lai.

**Response survival (PREDICT_RESPONSE_SURVIVAL = UNRESTRICTED)**: Offspring sinh bởi PREDICT không bị giữ lại theo region sau khi tạo; chúng cạnh tranh tự do trong NSGA-II standard survival cùng với phần còn lại của quần thể. Không có cơ chế bảo vệ vùng (region-protected survival) hay HARD trong sản xuất.

**DIVERSIFY**

Tạo offspring rộng hơn:

$$
x' = Pi_Omega_t [x + sigma_D*(u-l) o xi]    (47)
$$

Giá trị sản xuất: **sigma_D = 0.15**.

*Ghi chú về tỷ lệ nhiễu (eq. 47)*: Trong phiên bản thiết kế ban đầu, tài liệu ghi sigma_D > sigma_P > sigma_L như một giả định thứ tự thiết kế heuristic. Với các tham số sản xuất cuối cùng đã kiểm chứng (sigma_D=0.15, sigma_L=0.02, sigma_P=0.005), thứ tự thực tế là sigma_D > sigma_L > sigma_P. Điều này là có chủ ý: toán tử PREDICT đã có thành phần dịch chuyển Delta_hat_r cung cấp hướng di chuyển, nên nhiễu sigma_P chỉ cần nhỏ để bổ sung khám phá cục bộ quanh hướng dự đoán. Yêu cầu thiết kế thực sự là sigma_D đủ lớn hơn các tỷ lệ nhiễu cục bộ; không có ràng buộc toán học nào yêu cầu sigma_P > sigma_L.

### 5.10. Ngân sách FE phụ thuộc action (thiết kế tương lai cho giai đoạn regional policy)

**Phân biệt triển khai hiện tại và kế hoạch tương lai**:

- **Triển khai sản xuất hiện tại (Giai đoạn A)**: ngân sách phản ứng cố định **B = 10 FE per change event**. Toàn bộ ngân sách này được dùng cho toán tử response được chọn; không có phân bổ theo vùng.
- **Thiết kế tương lai (khi shared DRL được triển khai)**: Công thức (48) dưới đây mô tả cơ chế phân bổ FE phụ thuộc hành động theo vùng, dự kiến cho giai đoạn regional policy. Chưa được triển khai trong sản xuất.

$$
B_r(a) = 0                   nếu a = KEEP
B_r(a) = ceil(eta_L * n_r)  nếu a = LOCAL
B_r(a) = ceil(eta_P * n_r)  nếu a = PREDICT
B_r(a) = ceil(eta_D * n_r)  nếu a = DIVERSIFY    (48)
với 0 < eta_L <= eta_P < eta_D
$$

Khi triển khai trong tương lai: nếu sum_r B_r > B_max_resp thì scale quota theo tỷ lệ nhưng không thay đổi action label.

### 5.11. Shared regional policy

$$
Q_theta(s_r, a),  forall r    (49)
a*_r = argmax_{a in A} Q_theta(s_r, a)    (50)
$$

### 5.12. Reward cục bộ + toàn cục

$$
J_r(P,t) = -median_{x_i in P_r} g_r(F_tilde_i(t))    (51)
r_{r,t} = alpha_r*(J_r_end - J_r_pre)/(|J_r_pre| + epsilon) + alpha_g*(HV_end - HV_pre)/(HV_scale + epsilon) - beta*B_r(a_r)/B_max_resp - chi*V_r    (52)
$$

### 5.13. Double DQN

$$
a* = argmax_a Q_theta(s'_r, a)
y = r_r + gamma*(1-d)*Q_theta_bar(s'_r, a*)    (53)
L(theta) = E[Huber(Q_theta(s_r,a_r) - y)]    (54)
theta_bar <- (1 - tau_tgt)*theta_bar + tau_tgt*theta    (55)
$$

### 5.14. Bộ nhớ — extension, không phải lõi phiên bản 1

Để giữ thiết kế sạch, core PR-DRL-DMOEA không cần memory. Sau khi regional control đã được chứng minh có ích, có thể thêm action thứ năm RECALL.

Archive lưu nghiệm hoàn chỉnh gắn với region và environment signature: M = {(kappa_j, v_r, P_{r,j})}. RECALL chỉ đưa các nghiệm hoàn chỉnh từ archive vào candidate pool của region tương ứng. Đây là module phải được ablation riêng.

### 5.15. Hypervolume theo vùng — chỉ dùng khi phù hợp

$$
Delta_HV_i = (x_{i+1} - x_i)*(r_2 - y_i)    (57)
HV = sum_{i=1}^{n} Delta_HV_i    (58)
$$

Detector chính dùng displacement + PBI loss, không phụ thuộc local HV.

---

## 6. Regional-oracle: bước bắt buộc trước khi train DRL

### 6.1. Vì sao cần oracle audit

Việc các region có severity khác nhau chưa chứng minh rằng chúng cần action khác nhau. Do đó trước khi huấn luyện RL phải kiểm tra operator heterogeneity.

### 6.2. Utility của action theo region

$$
U_{r,a} = Q(P^{r,a}) - Q(P_KEEP)    (59)
a*_r = argmax_{a in A} U_{r,a}    (60)
$$

### 6.3. Operator Heterogeneity Index

$$
OHI = [-sum_{a in A} p_a*log(p_a + epsilon)] / log|A|    (61)
$$

### 6.4. Regional Oracle Gain

$$
a*_G = argmax_{a in A} Q(P_global(a))    (62)
P_RO = Respond(P, {a*_r}_{r=1}^{R})
ROG = [Q(P_RO) - Q(P_global(a*_G))] / (|Q(P_global(a*_G))| + epsilon)    (63)
$$

Điều kiện tiếp tục sang DRL phải được pre-register. Ví dụ: ROG dương ổn định trên nhiều problem/change classes, không chỉ một vài event đẹp.

---

## 7. Quy trình thuật toán đầy đủ

### 7.1. Một lần thay đổi môi trường

- **Bước 1 - Freeze population.** Lưu P^- và F^-.
- **Bước 2 - Re-evaluate.** Đánh giá cùng decision vectors trong môi trường mới để có F^+.
- **Bước 3 - Normalize.** Tính (24)-(25).
- **Bước 4 - Partition.** Gán region theo F^- bằng (28)-(29).
- **Bước 5 - Diagnose.** Tính c_r, l_r, o_r, D_r.
- **Bước 6 - Policy.** Shared DQN chọn a_r cho từng active region.
- **Bước 7 - Regional response.** Sinh số offspring theo (48); KEEP không sinh offspring.
- **Bước 8 - Merge.** Hợp inherited population và response offspring.
- **Bước 9 - Base MOEA recovery.** Chạy số generation/budget cố định.
- **Bước 10 - Reward.** Tính local/global reward ở cuối recovery horizon.
- **Bước 11 - Learn.** Push regional transitions vào replay và cập nhật Double DQN khi training.
- **Bước 12 - Advance environment.** Lặp lại.

### 7.2. Base MOEA đề xuất

**Triển khai sản xuất hiện tại (Giai đoạn A — validated)**: NSGA-II standard. Tất cả kết quả kiểm chứng response operator (KEEP, LOCAL, PREDICT, DIVERSIFY) sử dụng NSGA-II làm base MOEA.

**Khuyến nghị cho thiết kế đầy đủ PR-DRL (Giai đoạn C trở đi — chưa triển khai)**: Vì chính thuật toán sử dụng reference directions, base optimizer khuyến nghị là NSGA-III hoặc một reference-vector MOEA. Phiên bản đầu nên dùng một base duy nhất để tránh confound.

Không được diễn giải "khuyến nghị NSGA-III" là "sản xuất hiện tại đang dùng NSGA-III". Chúng là hai điều khác nhau.

### 7.3. Pseudocode

```
Input: dynamic problem, reference directions V, base MOEA, shared DQN
Initialize population P
Optimize in initial environment
Initialize replay buffer
for each environmental change t:
  P_minus <- copy(P)
  F_minus <- previous objectives of P_minus
  F_plus <- reevaluate same decision vectors in new environment
  normalize F_minus and F_plus
  regions <- assign each solution using F_minus and reference directions
  for each active region r:
    compute c_r, quality loss l_r, occupancy o_r, diversity D_r
    build state s_r
    choose a_r using shared policy
    generate response offspring O_r with action-dependent quota
  P_resp <- merge inherited solutions and all O_r
  P_end <- run base MOEA for fixed recovery budget
  for each active region r:
    compute local + global + FE reward r_r
    build next state s'_r
    store (s_r, a_r, r_r, s'_r, done)
  if training:
    update Double DQN
  P <- P_end
```

---

## 8. Benchmark và kế hoạch thực nghiệm

### 8.1. CEC2018 DF1-DF14 — benchmark nền bắt buộc

CEC2018 đưa ra 14 dynamic multi-objective functions; DF1-DF9 là bi-objective và DF10-DF14 là tri-objective. Bộ này bao phủ nhiều đặc tính như PF/PS biến đổi theo thời gian, PF bất quy tắc, disconnected front và knee regions [1].

### 8.2. SDP1-SDP15 — kiểm tra scalability và many-objective

Scalable Dynamic Problems (SDP) gồm 15 bài toán, thiết kế để khắc phục hạn chế về khả năng mở rộng của nhiều benchmark DMOP cũ [2]. Đây là benchmark rất phù hợp với PR-DRL vì reference-region representation phải chứng minh nó không chỉ hoạt động ở 2D/3D.

### 8.3. CEC2025 Dynamic Multiobjective Optimisation

CEC2025 giới thiệu 20 benchmark functions với các đặc tính đa dạng hơn: irregular PS/PF changes, disconnectivity, degeneration, detectability difficulty và thay đổi số biến/số mục tiêu [3].

### 8.4. Benchmark scalable 2024

Sun và cộng sự đề xuất các DMOP cải tiến và một scalable dynamic many-objective benchmark framework [4].

### 8.5. Generalized continuous DMOO suite 2026

Benchmark framework 2026 của Shao và cộng sự đưa vào hypersurface-based PS changes, controlled variable-contribution imbalance, dynamic rotation, irregular temporal perturbations và generalized time-linkage [5].

### 8.6. Thứ tự benchmark khuyến nghị

| Tầng | Benchmark | Mục đích |
|---|---|---|
| B0 | Synthetic toy regional-shift cases | xác minh công thức và regional oracle |
| B1 | CEC2018 DF1-DF14 | continuity với literature + 2D/3D |
| B2 | SDP1-SDP15 | scalability và many-objective |
| B3 | CEC2025 Track 1 | irregular/difficult modern DMOP |
| B4 | 2026 generalized suite | heterogeneity, nonseparability, time-linkage |
| B5 | CEC2025 constrained | kiểm tra regional control dưới constraints |
| B6 | UAV simulator | kiểm chứng ứng dụng |

---

## 9. Metric và thống kê

### 9.1. IGD và MIGD

$$
IGD_t = (1/|P*_t|) * sum_{y in P*_t} min_{x in A_t} ||y - x||_2    (64)
MIGD = (1/T) * sum_{t=1}^{T} IGD_t    (65)
$$

### 9.2. Mean Hypervolume

$$
MHV = (1/T) * sum_{t=1}^{T} HV(A_t; r_ref)    (66)
$$

Reference point phải cố định/pre-register trong cùng benchmark protocol.

### 9.3. Recovery Time

$$
RT_eta(t) = min{g : Q_t(g) >= Q_target_t(eta)}    (67)
$$

### 9.4. FE

Phải tách: mandatory environment reevaluation; regional diagnosis nếu cần extra evaluation; response offspring FE; base-MOEA FE.

### 9.5. Metric riêng để xác minh ý tưởng

Bắt buộc báo: RHI — Regional Heterogeneity Index; OHI — Operator Heterogeneity Index; ROG — Regional Oracle Gain; action distribution theo severity/quality loss; FE by action; fraction of KEEP regions.

### 9.6. Thống kê

Đề xuất tối thiểu 30 independent evaluation seeds. Với so sánh từng cặp trên mỗi benchmark problem: paired Wilcoxon signed-rank; Holm correction trong family đã pre-register; paired rank-biserial effect size.

Không pool hàng nghìn environment events thành iid samples.

---

## 10. Thực nghiệm quyết định trước khi xây DRL lớn

### 10.1. Experiment E0 — sanity

Tạo toy problems trong đó chỉ một trade-off region bị dịch. Kỳ vọng RHI tăng và detector chỉ ra đúng region.

### 10.2. Experiment E1 — heterogeneity audit

Trên mọi benchmark: (1) partition theo reference directions; (2) tính s_r ở từng change; (3) so RHI với random partition; (4) báo early/mid/late nếu environment có horizon dài.

Nếu objective-space partition không hơn random partition, dừng hoặc sửa cách định nghĩa region.

### 10.3. Experiment E2 — operator oracle

Không RL. Với từng region thử KEEP/LOCAL/PREDICT/DIVERSIFY bằng budget cố định. Tính a*_r, OHI và ROG.

Điều kiện để sang RL: nhiều change events có OHI>0; ROG dương ổn định; lợi ích không chỉ do một benchmark duy nhất.

### 10.4. Experiment E3 — rule-based regional controller

Tạo rule đơn giản: c_r thấp và l_r thấp -> KEEP; trung bình -> LOCAL; cao + predictable history -> PREDICT; cao + diversity thấp -> DIVERSIFY.

### 10.5. Experiment E4 — shared DRL

Train shared Double DQN. Training seeds, model seeds và evaluation seeds phải tách. Evaluation dùng policy frozen.

### 10.6. Experiment E5 — ablation

| Variant | Ý nghĩa |
|---|---|
| A0 | Base MOEA, không dynamic response |
| A1 | Best global response action |
| A2 | Rule-based regional |
| A3 | PR-DRL không PREDICT |
| A4 | Full PR-DRL |
| A5 | Full + optional memory |
| A6 | Random regions + shared DRL |

A6 rất quan trọng: nó kiểm tra lợi ích đến từ objective-space structure hay chỉ từ việc chia population thành nhóm.

---

## 11. Không gian tham số đề xuất

### 11.1. Reference regions

| Số mục tiêu M_o | Simplex divisions H | Số region R | Ghi chú |
|---|---|---|---|
| 2 | 9 | 10 | trực quan, đủ cá thể/vùng với N >= 100 |
| 3 | 4 | 15 | phù hợp bài UAV 3 mục tiêu |
| 5 | 2 | 15 | many-objective ban đầu |
| 7+ | chọn để R ≈ 10-30 | tùy | tránh quá nhiều empty regions |

### 11.2. Detector/region features — giá trị khởi tạo, chưa tối ưu

| Tham số | Ký hiệu | Giá trị/range khởi đầu |
|---|---|---|
| Numerical epsilon | epsilon | 1e-12 |
| Severity scale | kappa | 1.0 |
| EMA rate | rho | 0.05-0.2 |
| PBI penalty | theta_PBI | 5 |
| Sparse region threshold | n_min | 3 |

### 11.3. Response operators — tham số sản xuất cuối cùng (O1, D2-D 2026-09-20)

Bảng dưới phân biệt giá trị khởi tạo ban đầu (sensitivity study starting points) và giá trị sản xuất cuối cùng đã kiểm chứng và áp dụng.

| Tham số | Giá trị khởi tạo (ban đầu) | Giá trị sản xuất cuối cùng | Ghi chú |
|---|---|---|---|
| eta_L | 0.25 | 0.25 | chưa thay đổi |
| eta_P | 0.50 | 0.50 | chưa thay đổi |
| eta_D | 1.00 | 1.00 | chưa thay đổi |
| sigma_L | 0.02 | **0.02** | không thay đổi |
| sigma_P | 0.05 | **0.005** | áp dụng O1 (D2-D 2026-09-20); xem D2-R, D1-R |
| sigma_D | 0.20 | **0.15** | giảm sau calibration (N1, N2-C) |
| lambda_P | — | **1.0** | hệ số dự đoán PREDICT (eq. 45) |
| Response FE budget B | 0.5N đến N (phạm vi thiết kế) | **B = 10** | ngân sách FE cố định trong triển khai sản xuất |

**Bằng chứng cho sigma_P = 0.005 (O1)**:
- PERSISTENT: lợi ích dương được kiểm chứng (D2-R pooled +1.581e-4, CI dương hoàn toàn)
- STATIONARY: xấp xỉ trung tính, an toàn (PRACTICALLY_NEUTRAL, thỏa tiêu chí S1-S4)
- MIXED: ổn định và không gây hại (DIRECTIONAL_POSITIVE, cả hai block dương)

**Không tuyên bố**: sigma_P = 0.005 là tối ưu toàn cục; chỉ 0.03 và 0.005 được kiểm tra.

### 11.4. Shared Double DQN — cấu hình khởi đầu

| Thành phần | Cấu hình đề xuất |
|---|---|
| MLP | 2 hidden layers, 128-128 |
| Activation | ReLU hoặc SiLU |
| Output | 4 Q-values |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Discount gamma | 0.99 |
| Replay capacity | 1e5 regional transitions |
| Batch size | 128 |
| Target Polyak tau_tgt | 0.005 |
| Exploration | epsilon: 1.0 -> 0.05 theo transition count |
| Loss | Huber |

Epsilon decay nên được định nghĩa theo transition count để tránh việc thay đổi episode length làm thay đổi exploration regime.

---

## 12. So sánh với các hướng nghiên cứu hiện có

Việc chia vùng objective space **không phải hoàn toàn mới**. Literature đã có self-adaptive response dựa trên objective-space decomposition [6], multiregional co-evolution [7], regional prediction/fusion [8], region-aware multi-scale prediction [9], và population partition/prediction [10]. Vì vậy không được tuyên bố "lần đầu chia Pareto front thành region".

| Nhóm/phương pháp | Cách dùng vùng/RL | Khác với PR-DRL đề xuất |
|---|---|---|
| RL-DMOEA | Q-learning chọn response theo severity toàn cục | không điều khiển từng objective-region [11] |
| Objective-space decomposition response | decomposition + self-adaptive response | là prior art trực tiếp; phải dùng làm comparator [6] |
| MRCDMO | multiregional prediction/co-evolution | tập trung prediction/diversity, không phải shared DRL operator policy [7] |
| RMDIF | regional multi-directional prediction | dự đoán theo vùng, không phải action selection theo regional degradation [8] |
| RADMOEA | region-aware shared-point/multi-scale prediction | prior art rất gần về region awareness; phải so trực tiếp [9] |
| DMOPSO-ARS | severity + LSTM + adaptive response | severity chủ yếu global, PSO-based [12] |
| MFRLBS | multi-modal state + RL + bidirectional search | RL có environmental features nhưng không phải shared objective-region policy [13] |
| PPDMOEA | partition elite/regular + set-LSTM prediction | partition theo vai trò population, không phải trade-off sectors [10] |
| PR-DRL-DMOEA | frozen-membership objective regions + local degradation + shared DRL + action-dependent FE | contribution giả thuyết cần kiểm chứng |

Novelty an toàn để kiểm tra: không phải "regional DMOP" nói chung, mà là sự kết hợp của regional degradation measurement + pre-RL oracle validation + shared per-region operator selection + explicit action-dependent FE cost, đặc biệt trong bài UAV đa mục tiêu động.

---

## 13. Ánh xạ PR-DRL-DMOEA sang UAV

### 13.1. Ý nghĩa của objective regions trên UAV

Với (f1, f2, f3) = (AoI trung bình, AoI worst-hazard, năng lượng), các reference regions đại diện các ưu tiên trade-off, ví dụ: vùng ưu tiên AoI thấp; vùng ưu tiên fairness giữa hiểm họa; vùng tiết kiệm năng lượng; các vùng compromise.

Một trận lũ cục bộ **không mặc định** đồng nghĩa "một region Pareto bị ảnh hưởng". Điều cần đo là liệu những trade-off regions nào thực sự suy giảm khác nhau dưới cập nhật belief mới.

### 13.2. Tại sao dùng solution-regions thay vì UAV-regions

Mỗi cá thể là một deployment đầy đủ của cả đội UAV. Ràng buộc va chạm, backhaul, assignment và pin tạo coupling mạnh giữa UAV. Việc tác động lên các nghiệm hoàn chỉnh giảm nguy cơ phá cấu trúc bằng cách ghép các mảnh quyết định không tương thích.

### 13.3. Change signal trên UAV

$$
F^-_i = F(x_i; b(t-1)),  F^+_i = F(x_i; b(t))    (68)
$$

Sau đó dùng chính công thức region severity (30)-(35). Trong hệ thật, không cần UAV "bay thử" mọi nghiệm; evaluation phải dựa trên mô hình/digital twin cập nhật từ sensor observations.

### 13.4. Xử lý assignment

- U1 - Mixed encoding: tối ưu trực tiếp P và A; chính xác mô hình nhưng chiều lớn.
- U2 - Outer positions + inner MILP: tối ưu P; với mỗi P giải assignment bằng MILP.
- U3 - Outer positions + deterministic repair/heuristic: rẻ hơn, phù hợp mô phỏng lớn.

Nên bắt đầu U3 để xây simulator và chuyển sang U2 khi cần đánh giá chính xác hơn.

---

## 14. Quy trình nghiên cứu từ đầu đến cuối

**Giai đoạn A — chứng minh giả thuyết vùng trước**

1. Cài objective normalization và reference-direction partition.
2. Xây toy regional-shift problems có ground truth.
3. Chạy CEC2018 và tính RHI.
4. Chạy random-partition control.
5. Chạy operator oracle, tính OHI/ROG.
6. Decision gate A: nếu ROG không dương ổn định, dừng PR-DRL và không train RL.

**Giai đoạn B — controller không học**

7. Xây rule-based regional controller.
8. So với best-global response.
9. Decision gate B: nếu regional rule không hơn global, xem lại state/operator trước khi dùng DRL.

**Giai đoạn C — shared DRL**

10. Xây shared Double DQN.
11. Train trên training environments/seeds.
12. Freeze checkpoint.
13. Test trên unseen seeds/environments.
14. Chạy ablation A0-A6.
15. Báo MIGD/MHV/RT/FE + RHI/OHI/ROG.

**Giai đoạn D — benchmark hiện đại**: CEC2018 đầy đủ; SDP many-objective; CEC2025 Track 1; 2026 generalized suite; nếu core ổn: constrained CEC2025.

**Giai đoạn E — UAV**: Xây simulator AoI-kênh-pin-belief; validate static snapshot trước; tạo dynamic hazard scenarios có cả global change và local/heterogeneous change; chạy E1/E2 trên UAV để xác nhận objective-region heterogeneity thật sự tồn tại; train UAV-specific shared policy; so với heuristic UAV, NSGA-II/NSGA-III dynamic baseline và DMOEA competitors khả thi.

---

## 15. Kế hoạch benchmark tham số

### 15.1. CEC2018 — cấu hình khởi đầu đề xuất

| Tham số | Bi-objective | Tri-objective |
|---|---|---|
| Decision dimension D | 10 | 10 |
| Population N | 100 | 150 |
| Number of regions R | 10 | 15 |
| Independent evaluation runs | 30 | 30 |
| Primary metric | MIGD | MIGD |
| Secondary | MHV, RT, FE, RHI, OHI, ROG | như trái |

### 15.2. SDP/CEC2025/2026

Không áp một cấu hình N, D, R duy nhất cho mọi benchmark. Dùng settings chính thức nếu competition/report quy định; nếu không, pre-register theo objective count và đảm bảo mỗi region có occupancy đủ lớn.

---

## 16. Baseline và competitor cần có

Tối thiểu:
1. Static/dynamic NSGA-II.
2. NSGA-III không regional response.
3. Global-response version dùng cùng action set nhưng một action cho toàn quần thể.
4. Objective-space decomposition/self-adaptive response [6].
5. Một multiregional/region-aware predictor như MRCDMO/RMDIF/RADMOEA tùy code khả dụng.
6. RL-DMOEA [11].
7. DMOPSO-ARS [12].
8. MFRLBS [13] nếu tái lập được.

Nếu competitor không chạy trong cùng harness, số liệu literature chỉ được ghi là tham khảo, không được coi là head-to-head.

---

## 17. Rủi ro khoa học và tiêu chí dừng

| Rủi ro | Dấu hiệu | Quyết định |
|---|---|---|
| Regions thay đổi gần đồng đều | RHI thấp, giống random partition | không cần regional control |
| Regions khác nhau nhưng cùng action tốt nhất | OHI gần 0 | global controller đủ |
| Regional oracle không hơn global | ROG <= 0 | dừng trước DRL |
| Sparse regions | nhiều n_r < 3 | giảm R/merge regions |
| Local reward xung đột global | regional gain nhưng HV tổng giảm | tăng global term/sửa selection |
| Prediction không ổn định | PREDICT thường thua LOCAL | bỏ PREDICT khỏi core |
| DRL không hơn rule | A4 ~ A2 | giữ rule-based, không over-engineer |
| FE tăng quá nhiều | quality gain nhỏ, FE lớn | tăng KEEP/cost penalty |
| UAV không có regional heterogeneity | RHI/ROG thấp trên scenario thật | không claim ứng dụng region-aware |

---

## 18. Định vị đóng góp và cách viết claim

### 18.1. Có thể tuyên bố sau khi có bằng chứng

Nếu experiments xác nhận, claim hợp lý là: "Phương pháp khai thác heterogeneity của sự suy giảm theo các vùng trade-off trong objective space và học một chính sách response chia sẻ để phân bổ toán tử và ngân sách đánh giá riêng cho từng vùng."

### 18.2. Không được tuyên bố trước khi kiểm chứng

- "Regional segmentation luôn tốt hơn global response."
- "Ý tưởng chia Pareto front thành vùng là mới."
- "DRL cần thiết."
- "Policy CEC chuyển thẳng sang UAV."
- "UAV local hazard chắc chắn tạo local Pareto-region change."
- "Thuật toán SOTA."

---

## 19. Tổng kết

Thiết kế mới chuyển đơn vị thích nghi từ nhóm biến sang nhóm nghiệm theo trade-off region trong objective space. Một region không phải một UAV hay một khu vực địa lý; nó là một sector của approximation front. Khi môi trường thay đổi, cùng các decision vectors được theo dõi qua hai môi trường, từ đó đo displacement và quality loss cục bộ. Các region chỉ đáng có controller riêng nếu hai điều được chứng minh: heterogeneous change và heterogeneous best response.

Do literature đã có nhiều regional/decomposition DMOEA, hướng nghiên cứu không nên bán bằng "phân vùng" đơn thuần. Trọng tâm nên là quy trình kiểm chứng regional oracle, shared DRL operator policy và ngân sách FE phụ thuộc action.

Nếu E1/E2 thất bại, dự án nên dừng ở đó. Nếu RHI, OHI và ROG đều cho tín hiệu ổn định, khi đó việc đầu tư vào shared DRL mới có cơ sở khoa học rõ ràng.

---

## Tài liệu tham khảo

[1] S. Jiang, S. Yang, X. Yao, K. C. Tan, M. Kaiser, and N. Krasnogor, "Benchmark Problems for CEC'2018 Competition on Dynamic Multiobjective Optimisation," Technical Report, 2017/2018.

[2] S. Jiang, M. Kaiser, S. Yang, S. Kollias, and N. Krasnogor, "A Scalable Test Suite for Continuous Dynamic Multiobjective Optimization," IEEE Transactions on Cybernetics, vol. 50, no. 6, pp. 2814-2826, 2020. DOI: 10.1109/TCYB.2019.2896021.

[3] J. Zou, S. Jiang, Z. Hou, X. Yu, Y. Hu, and S. Yang, "Benchmark Problems for CEC'2025 Competition on Dynamic Multiobjective Optimisation," Technical Report, 2025.

[4] B. Sun, C. Zhang, H. Zhao, and Z. Yu, "Scalable benchmarks and performance measures for dynamic multi-objective optimization," Applied Soft Computing, vol. 159, 111600, 2024. DOI: 10.1016/j.asoc.2024.111600.

[5] C. Shao, Q. Zhao, N. Pu, S. Cheng, J. Jiang, and Y. Shi, "Benchmarking continuous dynamic multi-objective optimization: Survey and generalized test suite," Swarm and Evolutionary Computation, vol. 104, 102361, 2026. DOI: 10.1016/j.swevo.2026.102361.

[6] R. Liu, J. Li, Y. Jin, and L. Jiao, "A Self-Adaptive Response Strategy for Dynamic Multiobjective Evolutionary Optimization Based on Objective Space Decomposition," Evolutionary Computation, vol. 29, no. 4, pp. 491-519, 2021. DOI: 10.1162/evco_a_00289.

[7] "Multiregional co-evolutionary algorithm for dynamic multiobjective optimization," Information Sciences, vol. 545, pp. 1-24, 2021. DOI: 10.1016/j.ins.2020.07.009.

[8] "Guided prediction strategy based on regional multi-directional information fusion for dynamic multi-objective optimization," Information Sciences, vol. 669, 120565, 2024. DOI: 10.1016/j.ins.2024.120565.

[9] Y. Hu et al., "Region-aware prediction strategy based on shared points and multiple scales for dynamic multi-objective optimization," Expert Systems with Applications, 2026.

[10] "A population partition and prediction strategy-based evolutionary algorithm for dynamic multi-objective optimization," Swarm and Evolutionary Computation, vol. 107, 102476, 2026. DOI: 10.1016/j.swevo.2026.102476.

[11] "A reinforcement learning approach for dynamic multi-objective optimization," Information Sciences, vol. 546, pp. 815-834, 2021. DOI: 10.1016/j.ins.2020.08.101.

[12] N. Kouka, R. Fourati, R. Fdhila, A. Hussain, and A. M. Alimi, "A change severity degree-based dynamic multi-objective optimization algorithm with adaptive response strategy," Information Sciences, vol. 677, 120794, 2024. DOI: 10.1016/j.ins.2024.120794.

[13] Q. Li, L. Wang, and Y. Ma, "A dynamic multi-objective optimization evolutionary algorithm based on multi-modal feature fusion and entropy-driven reinforcement learning," Swarm and Evolutionary Computation, 2025.

---

## Phụ lục A. Bảng tóm tắt toàn bộ biến chính

| Ký hiệu | Ý nghĩa |
|---|---|
| N | population size |
| D | số biến quyết định |
| M_o | số mục tiêu |
| R | số objective regions |
| v_r | reference direction của region r |
| I_r | tập cá thể thuộc region r theo membership trước change |
| s_r | raw regional displacement severity |
| c_r | normalized regional change severity |
| l_r | regional convergence loss |
| o_r | occupancy ratio |
| D_r | regional diversity |
| RHI | Regional Heterogeneity Index |
| OHI | Operator Heterogeneity Index |
| ROG | Regional Oracle Gain |
| B_r | extra response FE của region |
| J_r | regional quality score |
| r_{r,t} | RL reward của region |
| sigma_P | PREDICT perturbation scale = **0.005** (O1, D2-D 2026-09-20) |
| sigma_L | LOCAL perturbation scale = 0.02 |
| sigma_D | DIVERSIFY perturbation scale = **0.15** |
| lambda_P | PREDICT displacement scaling = 1.0 |
| B | response FE budget = **10** |

---

## Phụ lục B. Phân biệt ba loại "vùng"

| Khái niệm | Định nghĩa | Có dùng làm region của PR-DRL? |
|---|---|---|
| Geographic region | khu vực vật lý trên bản đồ thiên tai | Không trực tiếp |
| Decision-variable block | nhóm tọa độ/biến của decision vector | Không |
| Objective/Pareto region | nhóm nghiệm có trade-off direction gần nhau | Có |

---

## Phụ lục C. Checklist trước khi gọi một kết quả là bằng chứng cho ý tưởng

1. Reference-region partition phải được khóa trước khi nhìn kết quả chính.
2. Same solutions phải được dùng khi đo trước/sau change.
3. RHI phải được so với random partition.
4. OHI/ROG phải được đo trước DRL.
5. Training/test seeds phải tách.
6. Policy phải frozen ở evaluation.
7. FE phải được đếm thật, riêng mandatory reevaluation và extra response FE.
8. Baseline global response phải dùng cùng action/operator và cùng base MOEA.
9. Ablation random regions phải có.
10. Không dùng số liệu từ paper khác làm head-to-head nếu protocol khác.
