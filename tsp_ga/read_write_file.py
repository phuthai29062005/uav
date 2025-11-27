#%%
import math

# --- PHẦN 1: XỬ LÝ FILE EIL51.TSP ---
N = 1000
adj51 = [(0, 0) for i in range(N + 1)]
cost51 = [[0 for i in range(N + 1)] for i in range(N + 1)]
output_filename = 'cost1000.txt' 

#%%
print("Bắt đầu đọc file pr299.tsp")

with open('dsj1000.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # Dữ liệu tọa độ từ dòng 7 đến 57
        if 7 <= i + 1 <= 1006:
            parts = line.split()
            if len(parts) >= 3:
                id, x, y = map(int, parts[:3])
                adj51[id] = (x, y)
print("Đã đọc xong file pr299.txt")

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(f"{N}\n")
    
    for i in range (1, N + 1):
        # Tính toán và điền vào ma trận cost
        for j in range (1, N + 1):
            x, y = adj51[i]
            u, v = adj51[j]
            if i == j:
                cost51[i][j] = 0
            else:
                distance = math.sqrt((x - u) ** 2 + (y - v) ** 2)
                cost51[i][j] = int(round(distance))
        
        row_as_strings = [str(cost51[i][j]) for j in range(1, N + 1)]
        line_to_write = ' '.join(row_as_strings)
        f.write(line_to_write + '\n')
        
print(f"Đã lưu ma trận cost của eil51 vào file: {output_filename}\n")

#%%
# --- PHẦN 2: XỬ LÝ FILE EIL101.TSP ---
M = 101
adj101 = [(0, 0) for i in range(M + 1)]
cost101 = [[0 for i in range(M + 1)] for i in range(M + 1)]

print("Bắt đầu đọc file eil101.tsp")

with open('eil101.tsp', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # Dữ liệu tọa độ từ dòng 7 đến 107
        if 7 <= i + 1 <= 107:
            parts = line.split()
            if len(parts) >= 3:
                id, x, y = map(int, parts[:3])
                adj101[id] = (x, y)
print("Đã đọc xong file eil101.tsp.")

# 1. Mở file ở chế độ 'a' (append) để ghi tiếp vào file đã có
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(f"{M}\n") # Thêm một dòng trống để tách biệt 2 ma trận
    
    for i in range (1, M + 1):
        # Tính toán và điền vào ma trận cost
        for j in range (1, M + 1):
            x, y = adj101[i]
            u, v = adj101[j]
            if i == j:
                cost101[i][j] = 0
            else:
                distance = math.sqrt((x - u) ** 2 + (y - v) ** 2)
                cost101[i][j] = int(round(distance))
        
        # Ghi một hàng của ma trận cost ra file
        row_as_strings = [str(cost101[i][j]) for j in range(1, M + 1)]
        line_to_write = ' '.join(row_as_strings)
        f.write(line_to_write + '\n')
        
print(f"Đã thêm ma trận cost của eil101 vào file: {output_filename}")
print("\nHoàn tất!")



# %%
