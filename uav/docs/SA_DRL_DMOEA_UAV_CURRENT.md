# SA-DRL-DMOEA: Bài toán tái định vị UAV và trạng thái hiện tại (6.0)

*Tài liệu thay thế `paper/baitoan_uav.pdf`. Nguồn biên tập chính (canonical) là file Markdown này; bản PDF/DOCX đi kèm chỉ để đọc/in.*

---

## 1. Mục tiêu tài liệu

Bản `baitoan_uav.pdf` cũ mô tả **thiết kế dự định ban đầu** của
SA-DRL-DMOEA (khoảng đầu dự án). Từ đó đến nay, quá trình audit và sửa
lỗi (giai đoạn 4.1–4.10) đã thay đổi khá nhiều chi tiết thuật toán so
với thiết kế ban đầu, và quá trình benchmark/ablation (5.1–5.5A) đã cho
ra kết quả thực nghiệm cụ thể. Tài liệu cũ **không còn phản ánh đúng
code hiện tại** ở nhiều chỗ quan trọng (state, action, memory key, tuyên
bố về phân đoạn...). Tài liệu này viết lại toàn bộ, trả lời ba câu hỏi:

1. **Thuật toán hiện tại là gì** — đặc tả đúng theo code đang chạy, không
   theo thiết kế cũ đã lỗi thời.
2. **Đã kiểm chứng tới đâu** — kết quả benchmark CEC 2018 (5.1–5.5A) đã
   xác nhận và chưa xác nhận điều gì, có bằng chứng thống kê cụ thể.
3. **Phần UAV còn ở đâu** — vẫn là bài toán/mô hình toán học, **chưa cài
   đặt, chưa chạy thực nghiệm nào**; tài liệu chỉ ra rõ ràng phần nào
   là giả thuyết mới cần kiểm chứng riêng.

Quy ước đánh dấu trong tài liệu này:

- 🟢 **ĐÃ CÀI ĐẶT & KIỂM CHỨNG** — có code, có test, có kết quả thống kê.
- 🟡 **ĐÃ CÀI ĐẶT, CHƯA/KHÔNG ĐƯỢC ỦNG HỘ CHUNG** — có chạy thực nghiệm
  nhưng kết quả không xác nhận lợi ích tổng quát (ví dụ: phân đoạn).
- 🔵 **CHỈ LÀ ĐẶC TẢ TOÁN HỌC** — chưa có một dòng code nào (toàn bộ phần
  UAV).

---

## 2. Bài toán UAV

*(Giữ nguyên phần lớn nội dung từ `baitoan_uav.pdf` mục 1–4 — mô tả bài
toán chưa đổi; phần thuật toán ở mục 5 trở đi được viết lại hoàn toàn ở
§4 của tài liệu này.)*

### 2.1 Bối cảnh

Sau thiên tai, hạ tầng viễn thông vùng bị ảnh hưởng có thể bị phá huỷ.
Cảm biến mặt đất (đã bố trí sẵn, cố định — bài toán bố trí cảm biến
không nằm trong phạm vi tài liệu này) đo được dữ liệu môi trường nhưng
không tự truyền về trung tâm được. UAV đóng vai trò **trạm chuyển tiếp
di động**: bay tới lơ lửng trên cụm cảm biến, thu dữ liệu, chuyển tiếp
về trạm mặt đất. Vì số UAV nhỏ hơn số cụm cần phục vụ, mỗi epoch phải
quyết định lại vị trí UAV và phân công cảm biến. 🔵

### 2.2 Tham số hệ thống (điển hình, chưa cố định — xem §6 để so sánh với
tham số CEC đã dùng thật)

| Ký hiệu | Ý nghĩa | Giá trị điển hình |
|---|---|---|
| 𝒜 | Vùng quan trắc | 1000 × 1000 m |
| M | Số cảm biến | 50–200 |
| **q**ₘ, θₘ | Vị trí, loại hiểm hoạ của cảm biến m | cố định |
| ℋ | Tập loại hiểm hoạ (lũ, sạt lở, cháy) | \|ℋ\|=3 |
| **q**₀ | Vị trí trạm mặt đất | 1 điểm |
| K | Số UAV | 3–10 |
| τ | Độ dài một epoch | 30 s |
| z_min, z_max | Giới hạn độ cao bay | 50–300 m |
| v_max | Tốc độ tối đa UAV | 20 m/s |
| d_min | Khoảng cách an toàn giữa hai UAV | 20 m |
| γ_th, γ⁰_th | Ngưỡng SNR đường lên / backhaul | dB |
| rₘ | Tốc độ dữ liệu cảm biến m | Mbps |
| C_k | Dung lượng backhaul của UAV k | Mbps |
| E⁰_k, E_k^res | Pin ban đầu / pin dự trữ tối thiểu | Wh |
| c_mv, c_hv | Hệ số tiêu thụ khi bay / hover | — |

Kênh Air-to-Ground dùng mô hình LoS/NLoS xác suất theo góc ngẩng: tăng
độ cao làm tăng xác suất LoS nhưng cũng tăng suy hao đường truyền — do
đó z_k **bắt buộc** là biến quyết định, không phải hằng số. 🔵

### 2.3 Input / Output mỗi epoch t 🔵

**Input:** P(t−1) (vị trí UAV epoch trước), Δ(t−1) (tuổi thông tin từng
cảm biến), E(t−1) (pin còn lại), o(t) (quan sát mới), b(t) (belief về
trường nguy cơ, cập nhật từ o(t)).

**Output:** P(t) = [p₁,...,p_K], pₖ=(xₖ,yₖ,zₖ) (vị trí UAV mới); A(t) =
[a_mk] ∈ {0,1}^(M×K) (ma trận phân công cảm biến–UAV). Vì ba mục tiêu
xung đột, output thực chất là **tập nghiệm không trội** (xấp xỉ Pareto).

---

## 3. Mô hình toán học (UAV) — 🔵 đặc tả, chưa cài đặt

### 3.1 Biến quyết định

**x**(t) = (P(t), A(t)) — hỗn hợp liên tục (P) và nhị phân (A).

### 3.2 Tuổi thông tin

Δₘ(t) = 0 nếu Σₖ a_mk(t)=1, ngược lại Δₘ(t)=Δₘ(t−1)+τ.

Đây là đại lượng chuyển bài toán từ "phủ sóng tĩnh" thành "cảnh báo
động": không chỉ cần phủ, mà phải quay lại đủ thường xuyên.

### 3.3 Trọng số ưu tiên

wₘ(t) = ρ̂_θm(**q**ₘ, t) — ước lượng từ belief. Vị trí cảm biến cố định
nhưng wₘ(t) trôi theo t — đây là cơ chế bơm tính động vào bài toán.

### 3.4 Hàm mục tiêu

$$f_1(\mathbf{x},t)=\frac{\sum_m w_m(t)\Delta_m(t)}{\sum_m w_m(t)}\quad\text{(trễ cảnh báo trung bình, min)}$$

$$f_2(\mathbf{x},t)=\max_{h\in\mathcal H}\frac{\sum_{m:\theta_m=h} w_m(t)\Delta_m(t)}{\sum_{m:\theta_m=h} w_m(t)}\quad\text{(trễ hiểm hoạ tệ nhất, min)}$$

$$f_3(\mathbf{x},t)=\sum_k\Big[c_{mv}\lVert\mathbf{p}_k(t)-\mathbf{p}_k(t-1)\rVert+c_{hv}\tau+\sum_m a_{mk}(t)e_{mk}\Big]\quad\text{(năng lượng, min)}$$

f₁ và f₂ xung đột (dồn UAV vào hiểm hoạ nóng vs. phân tán); f₃ xung đột
với cả hai. Số hạng c_mv‖pₖ(t)−pₖ(t−1)‖ khiến bài toán **không tách rời
theo epoch**.

### 3.5 Ràng buộc

| # | Ràng buộc | Ý nghĩa |
|---|---|---|
| (4) | z_min ≤ z_k(t) ≤ z_max, (x_k,y_k)∈𝒜 | không phận |
| (5) | ‖p_k(t)−p_k(t−1)‖ ≤ v_max·τ | động học |
| (6) | ‖p_k(t)−p_j(t)‖ ≥ d_min, k≠j | tránh va chạm |
| (7) | Σ_k a_mk(t) ≤ 1 ∀m | mỗi cảm biến tối đa 1 UAV |
| (8) | a_mk(t)(γ_th − γ_mk(t)) ≤ 0 | SNR đường lên (có điều kiện) |
| (9) | γ_k0(t) ≥ γ⁰_th ∀k | backhaul về trạm |
| (10) | Σ_m a_mk(t) r_m ≤ C_k(γ_k0(t)) | dung lượng backhaul |
| (11) | E_k(t)=E_k(t−1)−e_k(t) ≥ E_k^res | ngân sách năng lượng |

**Căng thẳng hình học cốt lõi:** ràng buộc (9) kéo UAV *về phía trạm*;
ràng buộc (8) kéo UAV *ra phía cảm biến*. Mâu thuẫn này chỉ xuất hiện
khi mô hình hoá đầy đủ đường backhaul, và làm bài toán không tầm
thường.

### 3.6 Phát biểu gọn

$$\min_{\mathbf x(t)} (f_1,f_2,f_3)\quad\text{s.t.}\quad \mathbf x(t)\in\Omega(t,\mathbf x(t-1),\mathbf E(t-1))$$

Miền khả thi phụ thuộc **nghiệm epoch trước** — khác biệt cốt lõi so
với benchmark CEC (nơi Ω cố định theo t, không phụ thuộc lịch sử).

### 3.7 Quan sát bộ phận

o(t) = {yₘ(t) : Σ_k a_mk(t)=1}, yₘ = ρ_θm(**q**ₘ,t) + εₘ. Chỉ quan sát
được nơi đã phục vụ ⇒ đánh đổi thăm dò–khai thác ⇒ **POMDP nội sinh** từ
chính cấu trúc bài toán, không phải do ta chọn mô hình hoá theo POMDP.

---

## 4. SA-DRL-DMOEA hiện tại (🟢 phần đã cài đặt, đúng theo code)

Đây là phần **viết lại hoàn toàn** so với `baitoan_uav.pdf` §5. Mọi công
thức dưới đây lấy trực tiếp từ code hiện tại (không phải từ ý tưởng ban
đầu).

### 4.1 Phân đoạn (segmentation)

**Protocol CEC chính thức: S=2, phân hoạch theo CHỈ SỐ CỐ ĐỊNH**
(fixed index-based two-block partition):

- B1 = {x₀}
- B2 = {x₁, ..., x_{D−1}}

**KHÔNG được gọi là "position/distance segmentation"** như một tuyên bố
tổng quát — đây chỉ là một cách chia chỉ số cố định, không dựa trên vai
trò cấu trúc thực sự của biến trong từng bài toán DF. Audit 4.10A đã
xác nhận cách chia này chỉ khớp đúng vai trò cấu trúc (biến vị trí vs.
biến khoảng cách) trên **8/14 DF** — 6 DF còn lại phân hoạch này không
có ý nghĩa cấu trúc rõ ràng. `src/train.py::make_segments(D, n_seg=2)`.

### 4.2 ChangeDetector — temporal directional sensitivity

`src/change_detector.py`. Đo **sự thay đổi theo thời gian** của độ nhạy
hướng (directional derivative) tại các điểm probe cố định, KHÔNG phải
độ nhạy tức thời:

- g_{s,t}(x) = đạo hàm hướng quanh điểm probe cố định, theo hướng riêng
  của từng segment s.
- c_raw = thay đổi trong độ nhạy hướng đã chuẩn hoá, giữa t−1 và t, với
  **cùng một bước nhiễu chuẩn hoá** cho mọi segment bất kể số chiều:
  a = eps/√(n_s).
- Sai phân nhận biết biên (boundary-aware finite differences): `central`
  (bậc 2, dùng khi cả hai phía đều khả thi), `onesided2` (bậc 2 một
  phía), `onesided1` (bậc 1, fallback khi không đủ chỗ — có cảnh báo
  runtime nếu tỷ lệ dùng fallback này > 5%).
- Chuẩn hoá nhân quả (causal, EMA):
  $$c_s = 1-\exp\left(-\frac{c_{tilde,s}}{\kappa\, q_{prev}+\epsilon}\right)$$
  với q được cập nhật **SAU KHI** đã dùng để tính c (không nhìn trước).

Probe cố định: `n_probe = n_elite = 10` — chọn từ elite population cuối
warm-up, **dùng lại suốt cả episode** (không đổi theo t).

**Giới hạn cần ghi rõ:** probe có trôi theo thời gian ở một mức độ nhất
định (audit 4.10D), nhưng detector không sụp đổ (chưa quan sát thấy suy
biến nghiêm trọng) trong giới hạn thực nghiệm đã chạy.

### 4.3 State hiện tại

`src/sa_drl_dmoea.py::build_state()`. `state_dim = S + 4`:

```
[c_1, ..., c_S,  dispersion,  hv_drop,  d_mem,  has_memory]
```

Với S=2 (protocol CEC): **state_dim = 6**.

| Thành phần | Ý nghĩa | Ghi chú |
|---|---|---|
| c₁..c_S | mức độ thay đổi theo phân đoạn (§4.2) | ∈[0,1) |
| dispersion | độ phân tán quần thể trong không gian quyết định, chuẩn hoá theo σ của phân bố Uniform | **KHÔNG PHẢI Shannon entropy** |
| hv_drop | tỉ lệ suy giảm hypervolume do môi trường đổi gây ra | ∈[0,1] |
| d_mem | khoảng cách chuẩn hoá đến chữ ký môi trường lịch sử gần nhất | ∈[0,1] |
| has_memory | archive có ít nhất 1 mục hay không | {0,1} |

**Đã bỏ hoàn toàn** so với thiết kế cũ: entropy quần thể (H(t)), g_last
(số thế hệ từ lần đổi cuối), phase/oracle ϕ(t) (chỉ dùng nội bộ cho
audit, không nằm trong state sản xuất — `compute_phase()` được giữ lại
trong code với ghi chú rõ "ORACLE-ONLY, live runner KHÔNG gọi").

### 4.4 Không gian hành động — hierarchical, KHÔNG phải 5 action/segment

Đây là điểm **sai khác lớn nhất** so với thiết kế cũ. Thiết kế cũ mô tả
"5 hành động mỗi segment" trong đó MEMORY là một action cấp segment.
**Code hiện tại KHÔNG như vậy**:

- **Gate** (cấp toàn cục, chọn 1 lần mỗi lượt đổi môi trường):
  `NO_MEMORY` hoặc `MEMORY`.
- Nếu `MEMORY`: **toàn bộ population** bị thay bằng population lịch sử
  gần nhất (không cắt ghép từng đoạn). Không còn action nào khác được
  chọn ở lượt này.
- Nếu `NO_MEMORY`: chọn 1 trong 4 action cho **mỗi segment**:
  `KEEP` (0), `LOCAL` (1, nhiễu Gaussian nhỏ), `PREDICT` (2, ngoại suy
  tuyến tính theo hướng dịch chuyển centroid), `DIVERSIFY` (3, thay ngẫu
  nhiên một phần quần thể trong segment).

MEMORY là **gate toàn cục**, không phải action cấp segment.

### 4.5 DQN

`src/dqn_agent.py`. MLP dùng chung (shared trunk, 2 lớp ẩn 64 nơ-ron,
ReLU), tách thành hai đầu ra:

- `gate_head`: 2 giá trị Q (NO_MEMORY, MEMORY)
- `segment_head`: S×4 giá trị, reshape thành advantage đã trừ mean theo
  action (mean-centered advantage, giống dueling-style cho phần
  segment)

Q tổng hợp theo cấu trúc phân cấp:
- Nếu MEMORY: Q = q_gate(MEMORY)
- Nếu NO_MEMORY: Q = q_gate(NO_MEMORY) + mean(segment advantages đã
  chọn)

**Double DQN**: mạng online chọn action tốt nhất ở next_state (có mask
MEMORY nếu `has_memory=0`), mạng target đánh giá action đó. Soft target
update (Polyak) sau **MỖI** bước học — không phải hard-copy định kỳ:

$$\theta_{target} \leftarrow (1-\tau)\,\theta_{target} + \tau\,\theta_{online},\quad \tau = 0.01$$

Replay transition lưu: `(state, gate, seg_actions, reward, next_state,
done)`.

### 4.6 Memory

`src/memory_archive.py`. Lưu **toàn bộ population** cuối mỗi
environment (không cắt ghép mảnh), khoá theo **chữ ký môi trường đã
hiệu chỉnh** (calibrated environment signature) — KHÔNG phải "pha chu kỳ
+ chữ ký hình dạng PF" như thiết kế cũ mô tả.

Vòng đời bắt buộc: **RETRIEVE → RESPOND → OPTIMIZE → STORE**. Entry của
environment t chỉ được lưu **SAU KHI** đã rời khỏi t (không thể tự truy
xuất chính nó).

**Chữ ký (signature):** với mỗi objective, 5 thống kê trên population đã
squash mềm: mean, 2·std, Q25, Q50, Q75 → tổng chiều = 5×M. Chuẩn hoá
bằng `obj_scale` đo **một lần** ở cuối warm-up (median của F tại điểm
probe, bất biến outlier).

Archive: FIFO, `max_size=50`. `query()` luôn trả entry gần nhất (không
có ngưỡng) — nếu archive rỗng, trả `has_memory=False, d_mem=1.0`.

**Cảnh báo thiết kế (giữ nguyên từ bản cũ, vẫn đúng):** không cắt ghép
mảnh nghiệm từ nhiều mục bộ nhớ — chỉ hợp lệ khi biến tách rời được;
trong nhiều DF (và trong bài UAV với ràng buộc va chạm/phủ chồng lấn)
các biến phụ thuộc chặt.

**Giới hạn quan sát được:** chẩn đoán 4.10D và 5.3A ghi nhận hiện tượng
suy biến/bão hoà tín hiệu signature trên một số DF tam mục tiêu muộn
trong horizon (đặc biệt DF10, DF12 — xem §9).

### 4.7 Reward

`src/sa_drl_dmoea.py::compute_reward()`:

$$R(t)=\underbrace{\alpha\cdot\frac{HV_{end}-HV_{pre}}{HV_{ref}}}_{\text{chất lượng bám vết}} - \underbrace{\beta\cdot\frac{FE_{action}}{N\cdot\tau_t}}_{\text{chi phí}},\quad \alpha=1,\ \beta=0.1$$

Chất lượng đo từ **HV_pre → HV_end** (sau khi hồi phục đủ τ_t thế hệ),
KHÔNG dùng HV ngay sau response — response tệ có thể "hồi phục giả tạo"
mà không phải công của action đó.

**Điểm quan trọng phải nói rõ:** trong cài đặt CEC hiện tại,
`FE_action = N` (đúng bằng kích thước quần thể) cho **MỌI** response
action, không phân biệt action nào. Nghĩa là số hạng chi phí trong
reward là **hằng số theo action** (action-invariant) — reward **KHÔNG
học được** đánh đổi chi phí-chất lượng ở cấp độ từng action trong cài
đặt hiện tại. Số hạng này được giữ lại cho các thiết lập tương lai nơi
chi phí đánh giá phụ thuộc action, không phải lỗi.

### 4.8 Chỉ số đánh giá (Metrics)

- **MIGD_end** (chính): IGD trung bình đo ở **cuối** mỗi environment
  (sau τ_t thế hệ NSGA-II hồi phục), ngay trước khi môi trường đổi tiếp.
- **MIGD_response** (phụ, mô tả): IGD trung bình đo **ngay sau** response,
  trước khi hồi phục.
- Alias `migd` = `migd_end` (giữ tương thích code cũ).

Lý do dùng END làm chỉ số chính: đây là chất lượng thực tế mà hệ thống
duy trì trong phần lớn thời gian của mỗi environment (τ_t thế hệ), còn
RESPONSE chỉ là trạng thái tức thời ngay sau một hành động.

### 4.9 Timeline

Environment đầu tiên chạy tại t=0, đủ τ_t thế hệ trước lần đổi đầu tiên
— **không có "đổi giả" từ 0.0→0.0**. Lần đổi thứ k (k=1..K):
t_k = k/n_t. Environment cuối cùng cũng được hồi phục đủ τ_t thế hệ.
Transition RL cuối cùng (pending) được đẩy vào replay buffer với
`done=True` — episode có đúng K transition (không phải K−1).

Protocol CEC chính: n_t=10, τ_t=10, K=100 (đánh giá) hoặc K=30 (huấn
luyện — xem §6), warm_up=50.

### 4.10 FE (Function Evaluations)

Định nghĩa: **1 FE = một cá thể được đánh giá objective một lần.**
Sổ cái FE (`fe_breakdown`) tách theo nguồn: `initial`, `nsga2`,
`detector`, `signature`, `response` (baseline sạch chỉ có `initial`,
`nsga2`, `environment_reeval` — không chịu chi phí riêng của SA-DRL).
Assertion cuối mỗi lần chạy: `fes_used == sum(fe_breakdown.values())`.

### 4.11 NSGA-II

`src/nsga2_pymoo.py`. **Canonical**: binary tournament theo (rank trước,
crowding sau, tie ngẫu nhiên chính xác — không thiên vị theo chỉ số
mặc định); SBX (prob=0.9, eta=20); PM (eta=20);
`RankAndCrowding` survival của pymoo. Dùng bounds THẬT của problem
(`problem.xl`/`problem.xu`), không hardcode clip [0,1]. Dùng chung cho
cả SA-DRL controller và baseline sạch.

---

## 5. Quy trình một lần thay đổi môi trường

```
1. Môi trường đổi từ t_{k-1} sang t_k.
2. Đo MIGD_end của môi trường VỪA RỜI (dùng population kế thừa,
   vẫn dưới problem cũ) + chốt reward cho action của môi trường đó.
3. STORE population cuối của môi trường vừa rời vào MemoryArchive
   (chữ ký = signature đo tại t_{k-1}).
4. Sang môi trường mới t_k: đánh giá lại population kế thừa (PRE metric).
5. RETRIEVE: tính signature tại t_k, query MemoryArchive -> d_mem,
   has_memory, pop lịch sử (nếu có).
6. Tính c(t_k) qua ChangeDetector (probe cố định, so t_{k-1} vs t_k).
7. build_state(c, dispersion, hv_drop, d_mem, has_memory).
8. Đẩy transition của action TRƯỚC (reward vừa chốt ở bước 2,
   next_state = state vừa tính) vào replay buffer; agent.learn()
   nếu đang train.
9. Agent chọn gate + segment actions cho state hiện tại.
10. Thực thi response (apply_hierarchical_response) -> RESPONSE metric.
11. Chạy tau_t thế hệ NSGA-II để hồi phục -> cuối giai đoạn này mới
    đo lại MIGD_end ở lần đổi TIẾP THEO (quay lại bước 2).
```

Đây chính xác là những gì `src/dynamic_runner.py::run_sa_drl()` thực
hiện — không có bước "ẩn" nào khác.

---

## 6. CEC 2018 — protocol thực nghiệm đã dùng thật

### Bảng A — Cấu hình bài toán chính

| Tham số | Bi-objective | Tri-objective |
|---|---|---|
| D | 10 | 10 |
| N (population) | 100 | 150 |
| warm_up | 50 | 50 |
| τ_t | 10 | 10 |
| n_t | 10 | 10 |
| K (số lần đổi, đánh giá) | 100 | 100 |
| S (số phân đoạn) | 2 | 2 |
| Số điểm probe | 10 | 10 |
| Số seed đánh giá/DF | 30 | 30 |

Seed đánh giá chính (5.1/5.2): `100000..100029`. Seed holdout ablation
(5.3B/5.3C): `200000..200029`. Seed holdout chẩn đoán horizon (5.5A):
`300000..300029`. **Ba dải này hoàn toàn tách biệt** (đã assert bằng
code, không giao nhau).

### Bảng B — Huấn luyện (protocol A4 chính, đóng băng)

| Tham số | Giá trị |
|---|---|
| N_TRAIN_EPISODES | 200 |
| TRAIN_CHANGES (số lần đổi/episode khi train) | 30 |
| Seed episode huấn luyện | 30..229 |
| model_seed | 30 |
| Số transition RL/model | 6000 (=30×200) |
| learn_steps thực tế | 5937 |
| epsilon cuối huấn luyện | ≈0.367 |
| Quy tắc chọn checkpoint | fixed-final-checkpoint (không validation/model-selection) |

Chú ý: TRAIN_CHANGES(30) ≠ EVAL_CHANGES(100) — huấn luyện thấy horizon
ngắn hơn đánh giá 3.3 lần. Đây chính là câu hỏi mà chẩn đoán 5.5A đặt ra
(xem §11).

### Bảng C — Siêu tham số DQN (đọc trực tiếp từ `src/dqn_agent.py`)

| Tham số | Giá trị |
|---|---|
| state_dim | S+4 (=6 với S=2) |
| hidden size | 64 (2 lớp ẩn, ReLU) |
| n_seg_actions | 4 (KEEP/LOCAL/PREDICT/DIVERSIFY) |
| gamma (γ) | 0.95 |
| learning rate | 1e-3 |
| optimizer | Adam |
| batch_size | 64 |
| replay capacity | 10000 (deque, FIFO khi đầy) |
| epsilon start | 1.0 |
| epsilon end | 0.01 |
| epsilon decay | 0.995 **mỗi episode** (không phải mỗi transition) |
| target_tau (Polyak) | 0.01, cập nhật sau MỖI bước học |
| Số gate | 2 (NO_MEMORY/MEMORY) |

### Bảng D — Reward

| Tham số | Giá trị |
|---|---|
| alpha (α) | 1.0 |
| beta (β) | 0.1 |
| HV_ref | prod(ref_point) |
| FE budget/environment | N × τ_t |
| Hành vi FE-action hiện tại | **hằng số = N cho mọi action** (xem §4.7) |

### Bảng E — ChangeDetector

| Tham số | Giá trị |
|---|---|
| eps (bước nhiễu chuẩn hoá h) | 0.01 |
| kappa (κ, chống bão hoà) | 2.0 |
| lam (λ, tốc độ EMA) | 0.05 |
| Số điểm probe | n_elite = 10 |
| Quy tắc sai phân | central → onesided2 → onesided1 (theo khả thi biên) |

### Bảng F — Bounds / ref point / lớp ổn định theo từng DF

| DF | n_obj | Bounds đặc biệt | ref_point | N | Lớp dùng |
|---|---|---|---|---|---|
| DF1 | 2 | [0,1]^10 | (2,2) | 100 | DF1 |
| DF2 | 2 | [0,1]^10 | (2,2) | 100 | DF2 |
| DF3 | 2 | x₁∈[−1,2], còn lại [0,1] | (2,2) | 100 | DF3 |
| DF4 | 2 | [−2,2]^10 | (5,5) | 100 | DF4 |
| DF5 | 2 | x₁∈[−1,1], còn lại [0,1] | (2,2) | 100 | **DF5Stable** |
| DF6 | 2 | x₁∈[−1,1], còn lại [0,1] | (2,2) | 100 | DF6 |
| DF7 | 2 | x₀∈[1,4], còn lại [0,1] | (15,6) | 100 | DF7 |
| DF8 | 2 | x₁∈[−1,1], còn lại [0,1] | (2,2) | 100 | DF8 |
| DF9 | 2 | x₁∈[−1,1], còn lại [0,1] | (2,2) | 100 | DF9 |
| DF10 | 3 | x₂..∈[−1,1], còn lại [0,1] | (2,2,2) | 150 | DF10 |
| DF11 | 3 | [0,1]^10 | (2,2,2) | 150 | DF11 |
| DF12 | 3 | [0,1]^10 | (3,2,3) | 150 | **DF12Stable** |
| DF13 | 3 | [0,1]^10 | (2,2,6) | 150 | **DF13Stable** |
| DF14 | 3 | [0,1]^10 | (2,2,2) | 150 | DF14 |

`*Stable`: chặn hiện tượng snap gần-0 của `sin(...)` trước `floor()`,
tránh discontinuity giả do sai số dấu phẩy động tại mốc chu kỳ
(`src/df_stable.py`, sửa lỗi 4.10C.1).

---

## 7. Lịch sử sửa lỗi đúng đắn (tóm tắt, không kể từng commit)

Kết quả sơ bộ trước các đợt sửa 4.1–4.10 **không còn giá trị tham khảo**
— các sửa lỗi sau đây thay đổi hành vi số học/thống kê đủ lớn để làm
kết quả cũ không so sánh được với kết quả hiện tại:

- **Bounds**: bỏ hardcode `[0,1]`, dùng đúng `problem.xl/xu` của từng DF.
- **NSGA-II canonical**: thay implementation viết tay bằng pymoo chuẩn
  (binary tournament rank+crowding không thiên vị).
- **Timeline**: bỏ "đổi giả" 0.0→0.0; environment cuối được hồi phục đủ
  τ_t.
- **FE ledger**: đếm chính xác theo nguồn, không đếm hai lần.
- **Reward horizon**: chuyển từ đo tại RESPONSE sang đo tại END.
- **Terminal transition**: K transition (không phải K−1); soft target
  update.
- **Memory**: chuyển từ centroid-key (mù với môi trường) sang
  environment-signature calibrated; từ splice-segment sang thay toàn bộ
  population.
- **State cleanup**: bỏ entropy/g_last/phase oracle khỏi state sản
  xuất.
- **Baseline separation**: tách hoàn toàn baseline sạch khỏi
  controller.
- **Stable time formulas**: DF5/12/13.
- **Training/eval freeze**: protocol `mode_frozen` — train xong đóng
  băng trọng số, đánh giá bằng agent mới nạp trọng số, không học thêm.

---

## 8. Kết quả chính CEC (5.1 → 5.2, xác nhận)

Protocol: 14 DF × 30 seed đánh giá cặp đôi (paired) `100000..100029`,
Wilcoxon signed-rank hai phía + hiệu chỉnh Holm trên 14 DF, α=0.05.

**Kết luận đúng:** "**lợi thế đa số phụ thuộc bài toán**"
(problem-dependent majority advantage) — **KHÔNG** phải "luôn luôn tốt
hơn" (consistently superior).

| DF | SA median [IQR] | BL median [IQR] | p_holm | r_rb | Kết quả |
|---|---|---|---|---|---|
| DF1 | **0.1082** [0.104, 0.114] | 0.1206 [0.117, 0.122] | 1.86e-07 | +0.966 | SA+ |
| DF2 | **0.1316** [0.124, 0.140] | 0.1896 [0.185, 0.193] | 2.61e-08 | +1.000 | SA+ |
| DF3 | **0.2101** [0.197, 0.222] | 0.2976 [0.289, 0.301] | 2.61e-08 | +1.000 | SA+ |
| DF4 | 0.0730 [0.073, 0.073] | **0.0667** [0.066, 0.067] | 2.61e-08 | −1.000 | SA− |
| DF5 | 0.0748 [0.063, 0.085] | **0.0669** [0.066, 0.069] | 0.0291 | −0.505 | SA− |
| DF6 | **0.9566** [0.796, 1.100] | 4.9893 [4.689, 5.142] | 2.61e-08 | +1.000 | SA+ |
| DF7 | **0.6936** [0.687, 0.705] | 0.7406 [0.722, 0.759] | 5.97e-06 | +0.888 | SA+ |
| DF8 | **0.0709** [0.069, 0.073] | 0.0731 [0.072, 0.074] | 0.0803 | +0.368 | = |
| DF9 | **0.2811** [0.249, 0.326] | 1.0745 [1.021, 1.094] | 2.61e-08 | +1.000 | SA+ |
| DF10 | 0.1049 [0.100, 0.119] | **0.0542** [0.054, 0.055] | 2.61e-08 | −1.000 | SA− |
| DF11 | **0.0716** [0.071, 0.072] | 0.0796 [0.079, 0.080] | 2.61e-08 | +1.000 | SA+ |
| DF12 | 0.1702 [0.163, 0.176] | **0.1372** [0.118, 0.143] | 6.52e-08 | −0.983 | SA− |
| DF13 | **0.1467** [0.144, 0.148] | 0.2186 [0.215, 0.225] | 2.61e-08 | +1.000 | SA+ |
| DF14 | **0.2215** [0.211, 0.244] | 0.3883 [0.357, 0.438] | 2.61e-08 | +1.000 | SA+ |

**Tổng: 9 SA+, 4 SA−, 1 "="** (14/14 DF). (SA = Full A4; BL = clean
NSGA-II; **in đậm** = giá trị median tốt hơn.)

---

## 9. Chẩn đoán cơ chế (5.3A, chỉ mô tả — KHÔNG kiểm định thống kê)

Dữ liệu per-change (42.000 dòng) là **quan sát**, không độc lập giữa các
dòng cùng seed — không tính p-value trên các dòng này.

- **DF4** (SA−): can thiệp segment cao (~100% seg1) nhưng vẫn thua
  baseline — gợi ý vấn đề không nằm ở tần suất can thiệp.
- **DF5** (SA−): gate entropy trung bình, không có mẫu hình nổi bật.
- **DF7** (SA+): quan sát "đảo chiều horizon" — Spearman
  ρ(Δresponse, Δrecovery) = **−0.160** (p=1.4e-18, mô tả trên dòng
  không độc lập, KHÔNG phải kiểm định giữa các seed).
- **DF10** (SA−): memory gần như không được dùng (memory_frac≈0.8%);
  seg1/seg2 can thiệp gần như luôn luôn.
- **DF11** (SA+): dùng MEMORY nhiều nhất (memory_frac≈53%).
- **DF12** (SA−): d_mem yếu, có đợt tăng đột biến dùng memory giữa
  horizon dù tín hiệu d_mem không mạnh.

---

## 10. Ablation (5.3C, xác nhận — holdout mới `200000..200029`)

Ba giả thuyết độc lập, mỗi giả thuyết tự hiệu chỉnh Holm trên 14 DF
riêng (không gộp 42 p-value). Quy ước dấu: FULL+ = A4 tốt hơn có ý
nghĩa; ABL+ = biến thể bị cắt bớt tốt hơn có ý nghĩa; "=" = không có
khác biệt Holm-significant (KHÔNG phải "tương đương").

### H1 — Phân đoạn (A4 vs A1, S=1 toàn cục)

| DF | A4 | Ablated (A1) | p_holm | r_rb | Kết quả |
|---|---|---|---|---|---|
| DF1 | 0.1056 | 0.0955 | 1.99e-05 | −0.888 | ABL+ |
| DF2 | 0.1245 | 0.0372 | 2.61e-08 | −1.000 | ABL+ |
| DF3 | 0.2159 | 0.2446 | 5.59e-03 | +0.673 | FULL+ |
| DF4 | 0.0732 | 0.0741 | 3.06e-06 | +0.935 | FULL+ |
| DF5 | 0.0703 | 0.0632 | 8.09e-01 | −0.178 | = |
| DF6 | 0.8828 | 1.1878 | 2.07e-01 | +0.441 | = |
| DF7 | 0.6966 | 0.6920 | 2.52e-01 | −0.363 | = |
| DF8 | 0.0697 | 0.0689 | 2.07e-01 | −0.415 | = |
| DF9 | 0.3162 | 0.3126 | 8.09e-01 | −0.174 | = |
| DF10 | 0.1057 | 0.0615 | 2.61e-08 | −1.000 | ABL+ |
| DF11 | 0.0715 | 0.0735 | 7.60e-06 | +0.914 | FULL+ |
| DF12 | 0.1748 | 0.1544 | 3.41e-05 | −0.871 | ABL+ |
| DF13 | 0.1465 | 0.1188 | 3.41e-05 | −0.871 | ABL+ |
| DF14 | 0.2224 | 0.2416 | 2.07e-01 | +0.432 | = |

**Đếm: 3 FULL+, 5 ABL+, 6 "="** → **KHÔNG được ủng hộ chung** 🟡. Nhiều
DF hơn cho thấy bộ điều khiển toàn cục (không phân đoạn) tốt hơn.
**Giới hạn:** A1 khác A4 không chỉ ở phân đoạn — state_dim, số nhánh
action, độ chi tiết detector cũng đổi theo — đây là so sánh kiến trúc
toàn cục-vs-phân đoạn, không phải "bật/tắt phân đoạn" thuần tuý.

### H2 — Tín hiệu thay đổi c_s (A4 vs A2, c bị che = 0 trong state)

| DF | A4 | Ablated (A2) | p_holm | r_rb | Kết quả |
|---|---|---|---|---|---|
| DF1 | 0.1056 | 0.1165 | 9.13e-08 | +0.983 | FULL+ |
| DF2 | 0.1245 | 0.0402 | 2.61e-08 | −1.000 | ABL+ |
| DF3 | 0.2159 | 0.2663 | 2.98e-08 | +0.996 | FULL+ |
| DF4 | 0.0732 | 0.0728 | 1.03e-03 | −0.725 | ABL+ |
| DF5 | 0.0703 | 0.0630 | 7.64e-04 | −0.746 | ABL+ |
| DF6 | 0.8828 | 1.2516 | 1.33e-01 | +0.381 | = |
| DF7 | 0.6966 | 0.5587 | 2.61e-08 | −1.000 | ABL+ |
| DF8 | 0.0697 | 0.0647 | 3.69e-07 | −0.961 | ABL+ |
| DF9 | 0.3162 | 0.7380 | 2.61e-08 | +1.000 | FULL+ |
| DF10 | 0.1057 | 0.0544 | 2.61e-08 | −1.000 | ABL+ |
| DF11 | 0.0715 | 0.0722 | 8.30e-03 | +0.609 | FULL+ |
| DF12 | 0.1748 | 0.1693 | 1.33e-01 | −0.385 | = |
| DF13 | 0.1465 | 0.1727 | 2.61e-08 | +1.000 | FULL+ |
| DF14 | 0.2224 | 0.1277 | 2.61e-08 | −1.000 | ABL+ |

**Đếm: 5 FULL+, 7 ABL+, 2 "="** → c_s làm policy input trực tiếp
**KHÔNG được ủng hộ chung** 🟡 — đây là ablation sạch nhất (FE khớp
hoàn toàn, cả hai đều chạy detector đầy đủ, chỉ khác việc policy có
"nhìn thấy" c hay không).

### H3 — Memory (A4 vs A3, tắt hoàn toàn bộ nhớ)

| DF | A4 | Ablated (A3) | p_holm | r_rb | Kết quả |
|---|---|---|---|---|---|
| DF1 | 0.1056 | 0.0834 | 2.61e-08 | −1.000 | ABL+ |
| DF2 | 0.1245 | 0.1446 | 2.42e-04 | +0.802 | FULL+ |
| DF3 | 0.2159 | 0.2371 | 1.16e-03 | +0.729 | FULL+ |
| DF4 | 0.0732 | 0.0751 | 2.61e-08 | +1.000 | FULL+ |
| DF5 | 0.0703 | 0.0636 | 3.70e-03 | −0.652 | ABL+ |
| DF6 | 0.8828 | 1.0183 | 3.10e-02 | +0.449 | FULL+ |
| DF7 | 0.6966 | 0.6484 | 1.86e-07 | −0.978 | ABL+ |
| DF8 | 0.0697 | 0.0737 | 1.59e-05 | +0.888 | FULL+ |
| DF9 | 0.3162 | 0.2380 | 6.15e-08 | −0.991 | ABL+ |
| DF10 | 0.1057 | 0.0769 | 9.22e-07 | −0.953 | ABL+ |
| DF11 | 0.0715 | 0.0729 | 1.16e-03 | +0.720 | FULL+ |
| DF12 | 0.1748 | 0.1680 | 1.63e-02 | −0.544 | ABL+ |
| DF13 | 0.1465 | 0.1629 | 2.61e-08 | +1.000 | FULL+ |
| DF14 | 0.2224 | 0.2556 | 9.17e-04 | +0.746 | FULL+ |

**Đếm: 8 FULL+, 6 ABL+, 0 "="** (mọi DF đều có ý nghĩa thống kê theo
một hướng nào đó). Cách nói đúng — **KHÔNG dùng từ "mạnh nhất"
(strongest) một cách không định nghĩa**:

> *Memory là thành phần DUY NHẤT trong ba thành phần được kiểm định mà
> đa số DF (8/14) ủng hộ Full; tuy nhiên 6/14 DF vẫn ủng hộ việc bỏ
> memory có ý nghĩa thống kê.* → **phụ thuộc bài toán mạnh**
> (strongly problem-dependent), không phải "memory luôn giúp ích".

---

## 11. Chẩn đoán horizon huấn luyện (5.5A, khám phá — holdout mới
`300000..300029`)

**Câu hỏi:** huấn luyện chỉ thấy 30 lần đổi/episode trong khi đánh giá
dùng 100 — liệu đây có phải nút thắt?

Thiết kế 2×2 trên 5 DF chọn trước (DF1, DF7, DF10, DF11, DF12):

|  | 6000 transition | 18000 transition |
|---|---|---|
| horizon=30 | H30_B6K (=A4 gốc) | H30_B18K |
| horizon=100 | H100_B6K | H100_B18K |

**Kết luận: phụ thuộc bài toán (PROBLEM-DEPENDENT)** — không ép về
"horizon giúp ích". Horizon (C1+C3): 5 SECOND+/4 FIRST+. Budget
(C2+C4): 5 SECOND+/2 FIRST+. Không yếu tố nào thắng một chiều.

**Hai confound quan trọng phát hiện thêm** (không nằm trong thiết kế
gốc, cần biết khi đọc kết quả):
1. Epsilon giảm dần **theo episode**, không theo transition → 4 cell có
   epsilon cuối huấn luyện rất khác nhau dù cùng số transition
   (0.367 / 0.740 / 0.049 / 0.406).
2. Replay buffer (capacity 10000) bị tràn ở ngân sách 18000 transition
   → các cell 18K bị eviction dữ liệu cũ, các cell 6K thì không.

**DF10**: kể cả horizon lẫn budget lớn nhất cũng KHÔNG kéo A4 gần lại
baseline sạch → gợi ý vấn đề DF10 mang tính **kiến trúc**, không phải
do huấn luyện ngắn.

*5.5A KHÔNG thay thế/diễn giải lại 5.2 hay 5.3C — chỉ là chẩn đoán khám
phá bổ sung.*

---

## 12. Điều gì hiện được ủng hộ (claim lock, 5.4)

| # | Tuyên bố | Bằng chứng | Verdict |
|---|---|---|---|
| C1 | Full A4 vs NSGA-II sạch | 5.2: 9 SA+/4 SA−/1 "=" | **SUPPORTED AS PROBLEM-DEPENDENT MAJORITY ADVANTAGE** (không phải "luôn tốt hơn") |
| C2 | Phân đoạn S=2 giúp ích | 5.3C H1: 3 FULL+/5 ABL+/6 "=" | **NOT SUPPORTED GENERALLY** |
| C3 | c_s cải thiện policy | 5.3C H2: 5 FULL+/7 ABL+/2 "=" | **NOT SUPPORTED GENERALLY** |
| C4 | Memory giúp ích | 5.3C H3: 8 FULL+/6 ABL+/0 "=" | **PARTIALLY SUPPORTED / STRONGLY PROBLEM-DEPENDENT** — memory là thành phần duy nhất mà đa số DF ủng hộ Full, nhưng không phổ quát |
| C5 | Policy phản ứng theo c riêng từng segment | 5.3A: "higher-c bias" dao động −0.56..+0.23, không nhất quán chiều | **weak/partial observational support only** — 30 seed đánh giá/DF nhưng chỉ 1 seed huấn luyện, và quan sát per-change không độc lập |
| C6 | DF7 lợi do đánh đổi response lấy recovery | 5.3A: ρ=−0.160 (mô tả, per-change) | **NOT causally established** — chỉ là liên kết mô tả, không kiểm định nhân quả |
| C7 | SA tiết kiệm FE hơn | Sổ cái FE mọi phase | **NOT SUPPORTED** — cùng lịch chạy nhưng A4 luôn tốn FE nhiều hơn baseline |
| C8 | Reward học được đánh đổi FE theo action | Code: `FE_action=N` cố định | **NOT SUPPORTED** — số hạng phạt FE bất biến theo action trong cài đặt hiện tại |

---

## 13. So sánh với tài liệu tham khảo (literature)

*(Giữ bảng khái niệm từ bản cũ — đây là so sánh Ý TƯỞNG, KHÔNG phải
head-to-head bằng số.)*

| Thuật toán | Đo thay đổi | Cơ chế phản ứng | Bộ điều khiển | Phân đoạn | Ứng dụng |
|---|---|---|---|---|---|
| NSGA-II/MOEA-D | Không | Khởi tạo lại | Không | Không | — |
| RL-DMOEA (2020) | Vô hướng, 3 mức | KBP,CBP,ILS | Bảng Q | Không | CEC2015 |
| MFRLBS (2025) | Đa mô thức, 5 mức | Đột biến thích nghi | Bảng Q+entropy | Không | CEC2018 |
| ACRM (2024) | Vô hướng | Ngẫu nhiên+dự đoán+bộ nhớ | RL (tỉ lệ trộn) | Không | CEC |
| DMOPSO-ARS (2024) | Mức độ thay đổi | Khởi tạo+học ưu tú | Luật cứng | Không | CEC2018 |
| CRDV/DVC | Theo loại biến | Riêng từng loại | Luật cứng | Có (tĩnh) | CEC |
| DRL-AOS (2024) | Trạng thái quần thể | Chọn toán tử | DQN | Không | CMOP tĩnh |
| **SA-DRL-DMOEA** | Vector theo đoạn | Gate memory + 4 action/đoạn | Double DQN | Có (động) | UAV-IoT (dự kiến) |

**Cảnh báo bắt buộc:** đây KHÔNG phải so sánh head-to-head — protocol,
benchmark, hyperparameter khác nhau giữa các paper gốc. **KHÔNG được**
suy ra "phương pháp này tốt hơn" từ việc đặt số liệu các paper cạnh
nhau. Muốn so sánh thật cần tự cài đặt lại các baseline này trên đúng
protocol hiện tại (chưa làm).

**Đóng góp nên phát biểu thế nào (giữ nguyên từ bản cũ, vẫn đúng):**
không tuyên bố "phát hiện thay đổi theo mức độ + bộ nhớ + RL chọn cơ
chế" là mới (trùng RL-DMOEA/ACRM). Nên nói: nâng cơ chế phản ứng từ
vô hướng/bảng Q lên vector-phân-đoạn/deep-RL, đặt trong khung
MOEA×DRL — **nhưng lưu ý bằng chứng ablation (§10) hiện KHÔNG xác nhận
bản thân phân đoạn là có lợi tổng quát**, nên tuyên bố đóng góp cần nói
"kiến trúc có cơ chế phân đoạn", không nói "phân đoạn đã được chứng
minh có lợi".

---

## 14. Quan hệ với bài toán UAV — viết lại có phê phán

### 14.1 Phân đoạn CEC vs. phân đoạn UAV: HAI GIẢ THUYẾT KHÁC NHAU

Phân đoạn CEC (§4.1, §10) là chia theo **chỉ số cố định** trên một
benchmark tổng hợp — audit 4.10A đã chỉ ra cách chia này không có ý
nghĩa cấu trúc trên 6/14 DF, và ablation H1 (§10) không xác nhận lợi
ích tổng quát của nó.

Phân đoạn UAV được đề xuất (mỗi UAV/cụm UAV là một segment) là một tiêu
chí **hoàn toàn khác** — theo **thực thể vật lý**, không theo vị trí
chỉ số. Bảng cũ liệt kê c(t) dạng [0.9, 0.8, 0.1, 0.05, 0.1] khi vùng
nguy cơ dịch chuyển cục bộ — đây là **một giả thuyết mới**, KHÔNG được
CEC chứng minh hay bác bỏ, vì tiêu chí phân đoạn khác nhau. Kết quả H1
không nói gì về việc phân đoạn theo thực thể vật lý có ích hay không.

### 14.2 Vấn đề kích thước mạng: S=2 (CEC) vs S=K (UAV)

`state_dim = S+4` và `segment_head` có kích thước `S × 4` — phụ thuộc
trực tiếp vào S. CEC dùng S=2 cố định; UAV dự kiến S=K (K=3–10, có thể
đổi giữa các kịch bản). **Việc chuyển checkpoint CEC sang UAV KHÔNG tự
động** — mạng huấn luyện với S=2 không nạp được cho S≠2. Batalinh cũ
"policy tiền huấn luyện chuyển giao được" cần loại bỏ/viết lại.

**Ba lựa chọn** (chưa quyết định, cần chọn khi bắt tay UAV):

- **T0** — huấn luyện agent MỚI riêng cho UAV (không transfer). Đơn
  giản nhất, khuyến nghị làm trước.
- **T1** — cố định K cho một mô hình UAV (S=K cố định theo kịch bản cụ
  thể), vẫn phải huấn luyện lại riêng nhưng kiến trúc mạng không đổi
  giữa các lần chạy cùng K.
- **T2** — redesign kiến trúc hỗ trợ S biến đổi (segment encoder chia
  sẻ tham số/permutation-equivariant) — phức tạp hơn nhiều, chỉ làm nếu
  cần K thay đổi linh hoạt giữa các kịch bản.

**Khuyến nghị cho luận án/prototype hiện tại: T0.**

---

## 15. Vấn đề phân công UAV–cảm biến (A(t))

Bản cũ tuyên bố "A(t) được giải tối ưu bằng bài toán phân công có trọng
số trong thời gian đa thức". **Không được giữ tuyên bố này nếu chưa
chứng minh** — bài toán phân công tổng quát có ràng buộc dung lượng
(constraint (10)) có thể **NP-hard** (dạng generalized assignment
problem), không tự động là bài toán phân công hai phía (bipartite
matching) giải được trong thời gian đa thức.

**Ba lựa chọn thay thế** (chưa quyết định):

- **U1** — tối ưu đồng thời P và A trong cùng MOEA (không gian tìm kiếm
  lớn, 3K+MK chiều).
- **U2** — MOEA ngoài tối ưu vị trí UAV P; bài toán con giải bằng
  MILP/thuật toán phân công chính xác khi cấu trúc ràng buộc cho phép
  (cần kiểm tra case-by-case, không giả định luôn giải được đa thức).
- **U3** — heuristic/luật sửa lỗi (repair) xác định cho phân công, đánh
  đổi tối ưu lấy tốc độ.

---

## 16. Giới hạn (bắt buộc đọc trước khi viết kết luận cho luận án)

- **Chỉ MỘT seed huấn luyện** (`model_seed=30`) cho mọi model đã huấn
  luyện xuyên suốt 5.1–5.5A. Mọi kiểm định thống kê chỉ đo biến thiên
  theo seed **đánh giá** (30 seed), KHÔNG đo biến thiên theo seed khởi
  tạo huấn luyện.
- A4 dùng **cùng lịch chạy vật lý** với baseline nhưng tốn FE nhiều hơn
  (§4.10, C7) — không có tuyên bố hiệu quả tính toán.
- Phân đoạn chỉ-số-cố-định (§4.1) không được ablation ủng hộ tổng quát
  (C2).
- c_s làm input trực tiếp cho policy không được ablation ủng hộ tổng
  quát (C3).
- Memory phụ thuộc bài toán mạnh (C4), không phải lợi ích phổ quát.
- Probe cố định (§4.2) có trôi nhẹ theo thời gian, chưa quan sát sụp đổ
  nhưng chưa loại trừ hoàn toàn cho horizon rất dài.
- Suy biến/bão hoà tín hiệu signature quan sát được trên DF10/DF12 muộn
  trong horizon (5.3A/4.10D).
- Số hạng phạt FE trong reward hiện bất biến theo action (C8) — không
  học được đánh đổi chi phí-chất lượng ở cấp action.
- **Chuyển giao CEC → UAV chưa được chứng minh** — khác tiêu chí phân
  đoạn (§14.1), khác kích thước mạng (§14.2).
- **Chưa có so sánh head-to-head với bất kỳ đối thủ nào trong bảng
  §13** — chỉ so sánh khái niệm.

---

## 17. Cây quyết định bước tiếp theo

| Nếu mục tiêu là... | Nên làm |
|---|---|
| Luận án/prototype (hoàn thành sớm) | DỪNG vòng CEC ở trạng thái hiện tại; bắt đầu cài đặt bài toán UAV (mô hình toán học §3, chọn T0 ở §14.2, chọn U1/U2/U3 ở §15); huấn luyện agent riêng cho UAV. |
| Bài báo hội nghị (DMOEA) | Robustness có mục tiêu: huấn luyện thêm 2–3 seed cho các DF/thành phần trọng tâm (đặc biệt memory, §10 H3) + cài đặt head-to-head với ít nhất 1–2 đối thủ trong bảng §13. |
| Bài báo tạp chí (tuyên bố kiến trúc tổng quát) | Nghiên cứu multi-training-seed đầy đủ + thiết kế state/phân đoạn mạnh hơn + nhiều đối thủ DMOEA head-to-head. |
| Cải thiện Full A4 trên CEC | Điều tra kiến trúc state/action/reward, đặc biệt DF10 (§10, §11) — **không đơn giản là huấn luyện lâu hơn** (5.5A đã cho thấy horizon/budget không giải quyết được DF10). |

---

# Phụ lục A — Toàn bộ tham số thực nghiệm

Xem Bảng A–F ở §6. Không lặp lại ở đây để tránh sai lệch khi cập nhật
(giữ một nguồn duy nhất).

# Phụ lục B — Bounds và ref point theo từng DF

Xem Bảng F ở §6.

# Phụ lục C — Cấu hình model/huấn luyện

Xem Bảng B, C ở §6.

# Phụ lục D — Hash dataset/checkpoint (đóng băng)

| Dataset | SHA256 |
|---|---|
| `results/final_benchmark_runs.jsonl` (5.1, 840 dòng) | `f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6fe090f09` |
| `results/ablation_5_3B_runs.jsonl` (5.3B, 2100 dòng) | `7191ee747155528a3d31ab7ec8ad8b7a87d499abb80bc5778c29c22eab138eff` |

**KHÔNG được ghi đè** hai file này. Checkpoint gốc: `results/checkpoints/{DF}.pt`
(14 file, hash trong `results/final_benchmark_models.jsonl`).

# Phụ lục E — Dải seed theo từng thực nghiệm

| Thực nghiệm | Seed huấn luyện | Seed đánh giá |
|---|---|---|
| 5.1/5.2 (A4 chính) | 30..229 (200 episode) | 100000..100029 |
| 5.3B/5.3C (ablation) | 30..229 (A1/A2/A3 huấn luyện lại cùng dải) | 200000..200029 |
| 5.5A (horizon 2×2) | 30..89 / 30..629 / 30..209 (khác nhau theo cell — xem §11) | 300000..300029 |

**Lưu ý quan trọng:** các dải seed huấn luyện ở 5.5A CHỒNG LẤN nhau có
chủ đích (đều là tiền tố của cùng dãy `30+episode_index`) — đây là thiết
kế để khớp ngân sách transition, KHÔNG phải để tạo tập huấn luyện độc
lập. Không được diễn giải các cell 5.5A là "huấn luyện độc lập".

---

# Glossary (thuật ngữ)

- **DMOP** (Dynamic Multi-objective Optimization Problem): bài toán tối
  ưu đa mục tiêu mà hàm mục tiêu/ràng buộc thay đổi theo thời gian.
- **DMOEA**: thuật toán tiến hoá đa mục tiêu cho DMOP.
- **PF** (Pareto Front): tập các điểm mục tiêu không trội lẫn nhau, tối
  ưu.
- **PS** (Pareto Set): tập nghiệm (biến quyết định) tương ứng với PF.
- **IGD** (Inverted Generational Distance): khoảng cách trung bình từ
  mỗi điểm PF thật đến điểm gần nhất trong quần thể — càng nhỏ càng
  tốt.
- **MIGD**: IGD trung bình qua nhiều lần môi trường đổi.
- **HV** (Hypervolume): thể tích không gian mục tiêu bị "bao phủ" bởi
  quần thể, tính đến một điểm tham chiếu — càng lớn càng tốt.
- **FE** (Function Evaluation): một lần đánh giá objective cho một cá
  thể.
- **AoI** (Age of Information): tuổi thông tin — thời gian từ lần cập
  nhật gần nhất.
- **DQN**: Deep Q-Network, xấp xỉ hàm giá trị Q bằng mạng nơ-ron.
- **Double DQN**: biến thể DQN dùng mạng online để CHỌN action tốt
  nhất, mạng target để ĐÁNH GIÁ action đó — giảm ước lượng quá lạc
  quan.
- **Replay buffer**: bộ nhớ lưu các transition (state, action, reward,
  next_state, done) để lấy mẫu ngẫu nhiên khi học (phá tương quan thời
  gian).
- **Target network**: bản sao (chậm cập nhật) của mạng online, dùng để
  tính giá trị mục tiêu (target) ổn định hơn khi học.
- **Segment**: một nhóm biến quyết định được xử lý như một đơn vị phân
  đoạn.
- **Environment signature**: chữ ký thống kê của một môi trường (dùng
  để so khớp trong bộ nhớ).
- **Frozen evaluation**: đánh giá với trọng số mạng đã "đóng băng"
  (không cập nhật), epsilon=0.
- **Paired seed**: cùng một seed được dùng cho nhiều phương pháp để so
  sánh trên cùng điều kiện khởi tạo.
- **Holm correction**: phương pháp hiệu chỉnh nhiều kiểm định giả
  thuyết (step-down), kiểm soát tỷ lệ sai số loại I trên cả họ kiểm
  định.
- **Rank-biserial effect size**: cỡ hiệu ứng cho kiểm định Wilcoxon bắt
  cặp, dấu cho biết hướng ưu thế, độ lớn [−1,1].
