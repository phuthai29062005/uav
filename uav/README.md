# SA-DRL-DMOEA — UAV tái định vị (CEC 2018 dynamic MOP benchmark)

## 1. Dự án này là gì

Segment-Aware Deep-RL-assisted Dynamic Multi-Objective Evolutionary
Algorithm (SA-DRL-DMOEA): một Double-DQN điều khiển phản ứng của
NSGA-II khi môi trường đổi, với state/action theo từng phân đoạn của
vector nghiệm. Đã huấn luyện/kiểm chứng đầy đủ trên benchmark động
CEC 2018 (DF1–DF14). Phần ứng dụng UAV (bài toán tái định vị UAV đa
thiên tai) hiện **chỉ là đặc tả toán học, chưa cài đặt**.

**Đọc trước tiên:** [`docs/SA_DRL_DMOEA_UAV_CURRENT.md`](docs/SA_DRL_DMOEA_UAV_CURRENT.md)
— tài liệu duy nhất mô tả đúng thuật toán hiện tại + toàn bộ kết quả đã
kiểm chứng + giới hạn + bước tiếp theo. Bản PDF/DOCX cùng thư mục chỉ để
đọc/in (nguồn chỉnh sửa chính là file `.md`).

## 2. Trạng thái khoa học hiện tại (tóm tắt — chi tiết xem tài liệu trên)

- **5.2 (xác nhận):** Full SA-DRL-DMOEA (A4) tốt hơn NSGA-II sạch có ý
  nghĩa thống kê trên **9/14 DF**, thua trên 4/14, không khác biệt trên
  1/14 — "lợi thế đa số phụ thuộc bài toán", KHÔNG phải "luôn tốt hơn".
- **5.3C (ablation, xác nhận):** phân đoạn cố định và tín hiệu thay đổi
  c_s làm input trực tiếp **không được ủng hộ chung**; memory là thành
  phần duy nhất có đa số DF ủng hộ Full nhưng vẫn phụ thuộc bài toán
  mạnh.
- **5.5A (chẩn đoán, khám phá):** kéo dài horizon huấn luyện hoặc tăng
  ngân sách transition đều **phụ thuộc bài toán**, không giải quyết
  được điểm yếu ở DF10.
- **Phần UAV:** 🔵 chưa cài đặt.

## 3. Cấu trúc repo

```
src/        module thuật toán lõi (dynamic_runner, dqn_agent, nsga2_pymoo, ...)
scripts/    entry point train/eval + các script phân tích 5.1-5.5A
tests/      pytest, 448+ test — chạy trước/sau MỌI thay đổi
results/    dataset/checkpoint ĐÃ ĐÓNG BĂNG + output phân tích (gitignored)
data/       bản vendor tham khảo CEC15/18 (không phải code sản xuất — production dùng package pymoo đã cài)
docs/       tài liệu chính (paper mới, code guide, audit)
paper/      PDF gốc + tài liệu tham khảo
archive/    file cũ đã xác nhận không dùng, giữ lại để tham khảo (xem archive/README.md)
```

Đọc thêm: [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) (nên đọc file nào
trước, luồng train/eval, mỗi module làm gì) và
[`docs/repository_audit.md`](docs/repository_audit.md) (vì sao mỗi file
được giữ/archive).

## 4. Chạy test

```bash
/opt/miniconda3/envs/dl/bin/python -m pytest tests/ -q
```

Kỳ vọng: toàn bộ pass (448+ tùy thời điểm), 0 fail/skip bất ngờ. **Luôn
chạy lệnh này trước và sau bất kỳ thay đổi nào** trong `src/`/`scripts/`.

## 5. Chạy smoke nhỏ (không tốn thời gian)

```bash
cd scripts
python train.py --mode frozen --problems DF1 --episodes 5 --runs 2 \
  --log-path /tmp/smoke.jsonl
```

Đây chỉ để kiểm tra wiring (chạy được, không lỗi) — **không dùng số liệu
từ lệnh này cho bất kỳ kết luận khoa học nào** (episode/seed quá ít).

## 6. Cách KHÔNG vô tình chạy lại benchmark chính

`scripts/final_benchmark.py` và `scripts/ablation_5_3B.py` là các script
**resume-safe** (tự bỏ qua các dòng đã có trong file kết quả), nhưng vẫn
**không nên chạy lại** trừ khi thật sự cần tái tạo dữ liệu — chúng ghi
trực tiếp vào hai file đã đóng băng bên dưới. Nếu chỉ muốn thử nghiệm,
luôn trỏ `--log-path`/sửa biến `RUNS_PATH` sang một file tạm khác trước.

## 7. Dataset/checkpoint đã đóng băng — KHÔNG được ghi đè

| File | SHA256 |
|---|---|
| `results/final_benchmark_runs.jsonl` | `f9d6bf87ea86556f6b258aa044f5e33a9ac7785488ddf376ce3f7ad6fe090f09` |
| `results/ablation_5_3B_runs.jsonl` | `7191ee747155528a3d31ab7ec8ad8b7a87d499abb80bc5778c29c22eab138eff` |

Mọi script phân tích đều verify hai hash này trước khi chạy và assert
lại sau khi chạy — nếu thấy lỗi assertion SHA, **dừng lại, không tự sửa
bằng cách chạy lại benchmark**; đó là dấu hiệu file đã bị đổi.

## 8. Tài liệu/paper ở đâu

- `docs/SA_DRL_DMOEA_UAV_CURRENT.md` (+ `.pdf`/`.docx`) — tài liệu chính,
  đọc đầu tiên.
- `docs/CODE_GUIDE.md` — hướng dẫn đọc code.
- `docs/repository_audit.md` — audit toàn bộ file trong repo.
- `paper/baitoan_uav.pdf` — bản thiết kế gốc (đã lỗi thời một phần, giữ
  lại để đối chiếu lịch sử — xem §7/§14 trong tài liệu chính để biết
  phần nào đã đổi).
- `paper/*.pdf` khác — tài liệu tham khảo (papers liên quan, quy chế
  MSO).

## 9. Nên đọc file nào trước

1. `docs/SA_DRL_DMOEA_UAV_CURRENT.md` — toàn cảnh.
2. `docs/CODE_GUIDE.md` — nếu cần đọc/sửa code.
3. `src/dynamic_runner.py` — nếu cần hiểu chi tiết thuật toán.
4. `results/*.md` (các report từng phase 5.1–5.5A) — nếu cần chi tiết
   thống kê đầy đủ hơn bản tóm tắt trong tài liệu chính.
