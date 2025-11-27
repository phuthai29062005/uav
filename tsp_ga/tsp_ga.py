#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
filename = 'best_avg_GAcx.txt'

data = np.loadtxt(filename)
best_fitness = data

runs = np.arange(1, len(best_fitness) + 1)

plt.figure(figsize=(12, 7))

plt.plot(
        runs, 
        best_fitness, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Quãng đường Tốt nhất (Best)', 
        color='blue'
    )
plt.title('Kết quả avg của 500 gen cho eil51 (30 lần chạy)')
plt.xlabel('Gen')
plt.ylabel('Tổng quãng đường (Cost)')

plt.legend()  # Hiển thị chú thích (label)
plt.grid(True)  # Hiển thị lưới

# Đặt giới hạn trục x để gọn gàng
plt.xlim(0, len(runs) + 1)

# --- 4. Hiển thị hoặc lưu file ---

# Lưu biểu đồ ra file ảnh
plt.savefig('ga_results_chart.png')

# Hiển thị biểu đồ (sẽ mở 1 cửa sổ mới)
print("Đang hiển thị biểu đồ...")
plt.show()


#%%
filename = 'best_avg_mlea.txt'

data = np.loadtxt(filename)
best_fitness51 = data[:, 0]
best_fitness101 = data[:, 1]

runs = np.arange(1, len(best_fitness51) + 1)

plt.figure(figsize=(12, 7))

plt.plot(
        runs, 
        best_fitness51, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Cost eil51', 
        color='blue',
        markersize=10
    )

plt.plot(
        runs, 
        best_fitness101, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Cost eil101', 
        color='red',
        markersize=10
    )
plt.title('Kết quả avg của 500 gen cho eil51 và eil101(30 lần chạy)')
plt.xlabel('Gen')
plt.ylabel('Tổng quãng đường (Cost)')

plt.legend()  # Hiển thị chú thích (label)
plt.grid(True)  # Hiển thị lưới

# Đặt giới hạn trục x để gọn gàng
plt.xlim(0, len(runs) + 1)

# --- 4. Hiển thị hoặc lưu file ---

# Lưu biểu đồ ra file ảnh
plt.savefig('ga_results_mlea_chart.png')

# Hiển thị biểu đồ (sẽ mở 1 cửa sổ mới)
print("Đang hiển thị biểu đồ...")
plt.show()
# %%
filename1 = 'best_avg_mfea.txt'
filename2 = 'best_avg_GAox.txt'
filename3 = 'best_avg_reinforcement.txt'
data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)
data3 = np.loadtxt(filename3)
best_fitness1 = data1[:, 0]
best_fitness2 = data2
best_fitness3 = data3

runs = np.arange(1, len(best_fitness1) + 1)

plt.figure(figsize=(12, 7))

plt.plot(
        runs, 
        best_fitness1, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Cost MFEA ga', 
        color='blue',
        markersize=10
    )

plt.plot(
        runs, 
        best_fitness2, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Cost GA ox',
        color='red',
        markersize=10
    )

plt.plot(
        runs, 
        best_fitness3, 
        'o-',          # 'o' là chấm tròn, '-' là đường nối
        label='Cost reinforcement ',
        color='green',
        markersize=10
    )

plt.title('Kết quả avg của 500 gen cho eil51')
plt.xlabel('Gen')
plt.ylabel('Tổng quãng đường (Cost)')

plt.legend()  # Hiển thị chú thích (label)
plt.grid(True)  # Hiển thị lưới

# Đặt giới hạn trục x để gọn gàng
plt.xlim(0, len(runs) + 1)

# --- 4. Hiển thị hoặc lưu file ---

# Lưu biểu đồ ra file ảnh
plt.savefig('ga_results_gamfearein51_chart.png')

# Hiển thị biểu đồ (sẽ mở 1 cửa sổ mới)
print("Đang hiển thị biểu đồ...")
plt.show()
# %%
