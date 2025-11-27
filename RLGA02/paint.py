import matplotlib.pyplot as plt
import numpy as np
import sys

# --- Định nghĩa tên file ---
filename_rgha = 'rhga51.txt'
filename_eax = 'eax_rhga51.txt' # <-- Đảm bảo tên file EAX-GA của bạn là đúng


# --- Đọc dữ liệu ---

# Tải file RHGA:
# Bỏ qua 1 dòng đầu (dòng "Best cost...")
# Tách cột bằng dấu ": "
data_rgha = np.loadtxt(filename_rgha, skiprows=1, delimiter=':')

# Tải file EAX-GA:
data_eax = np.loadtxt(filename_eax, skiprows=1, delimiter=':')

# Lấy dữ liệu từ các cột
# Cột 0 là số thế hệ (1, 2, 3...)
# Cột 1 là cost trung bình

# Dữ liệu RHGA
runs_rgha = data_rgha[:, 0]
avg_cost_rgha = data_rgha[:, 1]

# Dữ liệu EAX_GA
runs_eax = data_eax[:, 0]
avg_cost_eax = data_eax[:, 1]

# --- Bắt đầu vẽ biểu đồ ---
plt.figure(figsize=(12, 7))

# Vẽ đường của RHGA
plt.plot(
    runs_rgha,
    avg_cost_rgha,
    '-',          # Chỉ vẽ đường, 'o-' (chấm tròn) sẽ quá dày
    label='Cost RHGA (Trung bình 30 seed)',
    color='blue',
    linewidth=2   # Cho đường đậm hơn
    )

# Vẽ đường của EAX_GA
plt.plot(
        runs_eax, 
        avg_cost_eax, 
        '-',          # Chỉ vẽ đường
        label='Cost EAX_GA (Trung bình 30 seed)',
        color='red',
        linewidth=2
    )

# --- Tùy chỉnh biểu đồ ---
plt.title('So sánh RHGA và EAX-GA cho eil51 (Trung bình 30 lần chạy)')
plt.xlabel('Thế hệ (Gen)')
plt.ylabel('Tổng quãng đường trung bình (Average Cost)')

plt.legend()  # Hiển thị chú thích (label)
plt.grid(True)  # Hiển thị lưới

# Đặt giới hạn trục X (từ 0 đến thế hệ cuối cùng)
plt.xlim(0, max(len(runs_rgha), len(runs_eax)) + 1)

# --- Lưu và hiển thị ---
output_image_file = 'compare_RGHA_EAX-GA_TSP_chart.png'
plt.savefig(output_image_file)

print(f"Biểu đồ đã được lưu vào file: {output_image_file}")
plt.show()

