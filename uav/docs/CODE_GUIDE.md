# Hướng dẫn đọc code (CODE_GUIDE)

Tài liệu này viết cho **chính tác giả dự án**, để khi quay lại sau vài
tháng vẫn hiểu ngay mình đã làm gì, code chạy theo trình tự nào. Không
phải API doc kiểu chuyên nghiệp — giải thích bằng lời, không copy
docstring.

## 1. Nên bắt đầu đọc file nào?

Theo thứ tự này:

1. `src/dynamic_runner.py` — trái tim của hệ thống. Đọc hàm `run_sa_drl()`
   từ đầu đến cuối, đây là toàn bộ một lần chạy (episode huấn luyện HOẶC
   một lần đánh giá) của SA-DRL-DMOEA.
2. `src/sa_drl_dmoea.py` — các hàm nguyên thuỷ mà `dynamic_runner.py`
   gọi: tính state, tính reward, thực thi action.
3. `src/dqn_agent.py` — bộ não RL (mạng nơ-ron + Double DQN).
4. `scripts/train.py` — cách gọi `run_sa_drl()` để huấn luyện/đánh giá
   theo đúng protocol CEC (hàm `mode_frozen`).
5. Sau đó mới đọc các script phân tích (`scripts/ablation_*`,
   `scripts/horizon_*`, `scripts/analyze_*`) — đây là lớp NGOÀI, chỉ gọi
   lại `run_sa_drl()`/`train.py` với tham số khác nhau, không chứa thuật
   toán mới.

## 2. Main entry point nằm ở đâu?

Không có một "main.py" duy nhất. Điểm vào chính là:

```bash
python scripts/train.py --mode frozen --problems DF1 --episodes 200 --runs 30
```

Hàm `main()` cuối `scripts/train.py` chỉ parse argument rồi gọi
`mode_frozen()` (protocol chính) hoặc các `mode_*` khác (chỉ để debug/so
sánh cũ, không dùng cho kết quả chính thức).

## 3. Training flow nằm ở đâu?

`scripts/train.py::mode_frozen()` →
`scripts/train.py::train_on()` (lặp qua các episode, mỗi episode gọi
`one_run(..., training=True)`) →
`scripts/train.py::one_run()` →
`src/dynamic_runner.py::run_sa_drl(..., training=True)`.

Agent (`DQNAgent`) được tạo **một lần** trước vòng lặp episode và dùng
lại cho mọi episode — replay buffer và trọng số mạng tích luỹ xuyên suốt
quá trình train (xem mục 3 trong `docs/repository_audit.md`... không,
xem giải thích Q3 trong `scripts/horizon_5_5A.py` docstring — cùng một
cơ chế).

## 4. Evaluation flow nằm ở đâu?

`scripts/train.py::mode_frozen()` (phần sau khi train xong) →
với MỖI seed đánh giá, tạo **agent mới** (`new_agent`), nạp trọng số đã
đóng băng (`load_state_dict`), đặt `eps=0.0`, gọi
`run_sa_drl(..., training=False)`. Agent đánh giá không bao giờ gọi
`learn()` — trọng số không đổi trong suốt quá trình eval (được test xác
nhận bằng cách so hash trước/sau).

## 5. `dynamic_runner.py` làm gì?

Vòng lặp chính của một episode (train hoặc eval):

- Tạo population ban đầu, problem tại `t=0`, archive/detector mới toanh.
- Chạy `warm_up` thế hệ NSGA-II trên môi trường `t=0`.
- Mỗi khi môi trường đổi (`is_change=True`): tính state, chọn
  gate+segment action, thực thi response, chạy tiếp `tau_t` thế hệ
  NSGA-II để "hồi phục", rồi tính reward khi đến lần đổi TIẾP THEO (hoặc
  ở cuối episode).
- Đếm FE (function evaluations) chính xác theo từng loại: `initial`,
  `nsga2`, `detector`, `signature`, `response`.

## 6. DQN làm gì?

`src/dqn_agent.py`. `HierarchicalQNetwork` nhận state, xuất ra:
- `q_gate` (2 giá trị: NO_MEMORY / MEMORY)
- `adv` (S segment × 4 action, đã trừ mean để thành "advantage")

`DQNAgent.select_action()` chọn gate trước (nếu archive rỗng thì
KHÔNG BAO GIỜ được chọn MEMORY — bị mask cứng), rồi nếu NO_MEMORY thì
chọn 1 action cho mỗi segment. `learn()` là một bước Double DQN: mạng
online chọn action tốt nhất ở next_state, mạng target đánh giá action
đó — giảm hiện tượng ước lượng quá lạc quan.

## 7. NSGA-II làm gì?

`src/nsga2_pymoo.py::nsga2_one_generation()`. Đây là bản CANONICAL
(binary tournament theo rank+crowding, SBX, PM, `RankAndCrowding`
survival của pymoo) — dùng cho CẢ hai path: SA-DRL controller
(`dynamic_runner.py`) và baseline sạch (`baseline_runner.py`). Bản
NSGA-II viết tay cũ (`nsga2.py`/`operators.py`/`sorting.py`) đã archive,
không còn dùng.

## 8. ChangeDetector làm gì?

`src/change_detector.py`. KHÔNG đo độ nhạy mục tiêu tại một thời điểm —
đo **thay đổi theo thời gian** của độ nhạy đó (đạo hàm hướng tại các
điểm probe cố định, so sánh giữa `t-1` và `t`). Có 3 kiểu sai phân tuỳ
điểm probe có nằm sát biên hay không (`central`/`onesided2`/`onesided1`).
Kết quả chuẩn hoá bằng EMA để ra `c_s ∈ [0,1)`.

## 9. Memory làm gì?

`src/memory_archive.py`. Lưu TOÀN BỘ population cuối mỗi environment,
khoá theo "chữ ký môi trường" (`compute_signature` — 5 thống kê/objective
từ population tại các điểm probe). Khi gate=MEMORY được chọn, population
hiện tại bị THAY THẾ HOÀN TOÀN bằng population lịch sử gần nhất (không
cắt ghép từng đoạn).

## 10. Reward tính ở đâu?

`src/sa_drl_dmoea.py::compute_reward()`. `alpha * (HV_end - HV_pre)/HV_ref
- beta * FE_dùng/FE_ngân_sách`. Đánh giá ở **cuối** environment (sau
`tau_t` thế hệ hồi phục), không phải ngay sau response — tránh việc
response tệ nhưng "phục hồi giả tạo" được tính công.

## 11. State tạo ở đâu?

`src/sa_drl_dmoea.py::build_state()`. `[c_1..c_S, dispersion, hv_drop,
d_mem, has_memory]`, `state_dim = S+4`. Toàn bộ đã chuẩn hoá, không đơn
vị vật lý.

## 12. Action được decode ở đâu?

`src/sa_drl_dmoea.py::apply_hierarchical_response()`. Nếu gate=MEMORY:
gọi `action_memory_global`. Nếu NO_MEMORY: chạy action riêng cho từng
segment (`action_local`/`action_predict`/`action_diversify`, hoặc giữ
nguyên nếu KEEP).

## 13. FE được đếm ở đâu?

Trong `run_sa_drl()`, hàm nội bộ `add_fe(key, k)` cộng dồn vào
`fe_breakdown` (dict theo category). Assertion cuối cùng
`fes_counter == sum(fe_breakdown.values())` đảm bảo không đếm thiếu/thừa.

## 14. Timeline t thay đổi ở đâu?

Trong vòng lặp `run_sa_drl()`: `change_count = (gen - warm_up) // tau_t`,
`t_new = change_count / n_t`. Environment đầu (`t=0`) chạy đủ `tau_t`
thế hệ trước lần đổi đầu tiên — không có "đổi giả" từ 0.0 sang 0.0.

## 15. Metrics IGD/HV tính ở đâu?

`run_sa_drl()` gọi trực tiếp `pymoo.indicators.igd.IGD` và
`pymoo.indicators.hv.HV`. `migd_end` = trung bình IGD đo ở cuối mỗi
environment (chỉ số chính). `migd_response` = trung bình IGD đo ngay sau
response (chỉ số phụ, mô tả).

## 16. Bounds lấy ở đâu?

Từ chính đối tượng problem của pymoo: `problem.xl`, `problem.xu` — KHÔNG
hardcode `[0,1]`.

## 17. CEC DF stable classes ở đâu?

`src/df_stable.py`. `DF5Stable`, `DF12Stable`, `DF13Stable` — chặn hiện
tượng "snap gần-0" của `sin(...)` trước khi `floor()`, tránh
discontinuity giả do sai số dấu phẩy động tại các mốc chu kỳ.

## 18. Baseline sạch nằm ở đâu?

`src/baseline_runner.py::run_nsga2_baseline()`. Chạy NSGA-II thuần,
KHÔNG đi qua detector/memory/DQN — cấu trúc code hoàn toàn tách biệt với
`dynamic_runner.py` để đảm bảo baseline không "vô tình" thừa hưởng chi
phí/hành vi của SA-DRL.

## 19. Ablation hooks nằm ở đâu?

Hai flag `mask_change_state`, `disable_memory` trong
`run_sa_drl()` (`src/dynamic_runner.py`) — mặc định `False` cả hai, khi
đó hành vi giống hệt path production (đã verify bit-identical). Chỉ
dùng bởi `scripts/ablation_5_3B.py`.

## 20. Các script 5.1–5.5 dùng để làm gì?

| Script | Việc |
|---|---|
| `scripts/final_benchmark.py` | 5.1 — sinh `results/final_benchmark_runs.jsonl` (ĐÃ ĐÓNG BĂNG, không chạy lại) |
| `scripts/analyze_benchmark.py` | 5.2 — thống kê xác nhận (Wilcoxon+Holm) trên dataset 5.1 |
| `scripts/replay_policy_diagnostic.py` + `scripts/analyze_policy_diagnostic.py` | 5.3A — chẩn đoán cơ chế hành động/state (chỉ mô tả, không kiểm định) |
| `scripts/ablation_5_3B.py` | 5.3B — sinh `results/ablation_5_3B_runs.jsonl` (ĐÃ ĐÓNG BĂNG) |
| `scripts/ablation_5_3C_stats.py` | 5.3C — thống kê xác nhận ablation (H1/H2/H3) |
| `scripts/horizon_5_5A.py` + `scripts/horizon_5_5A_stats.py` | 5.5A — chẩn đoán horizon/budget huấn luyện (thăm dò, không thay thế 5.1-5.3C) |

## Sơ đồ luồng (text)

```
TRAIN:
    scripts/train.py (mode_frozen)
      -> train_on(): tạo 1 agent, lặp N_EPISODES
        -> one_run(training=True)
          -> dynamic_runner.run_sa_drl(training=True)
             -> mỗi lần đổi môi trường:
                ChangeDetector.compute() -> c
                MemoryArchive.query()    -> d_mem, has_memory
                build_state()            -> state
                agent.select_action()    -> gate, seg_actions
                apply_hierarchical_response() -> population mới
                tau_t thế hệ NSGA-II (nsga2_pymoo)
             -> đẩy transition vào agent.replay_buffer
             -> agent.learn()  (Double DQN, 1 bước, nếu buffer đủ 64)

EVAL:
    frozen checkpoint (.pt: online + target state_dict)
      -> new_agent() rồi load_state_dict() (mỗi seed một agent mới)
      -> eps = 0.0
      -> dynamic_runner.run_sa_drl(training=False)
      -> đọc kết quả: migd_end (chính), migd_response (phụ), fes_used
```
