import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CẤU HÌNH ---
# Đường dẫn thư mục chứa file
folder_path = './results/seed0/check_250' 
# Tên file ảnh muốn lưu ra
output_image_name = 'ket_qua_10_tasks.png'

# Tạo khung hình lớn chứa 10 biểu đồ (2 hàng, 5 cột)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
axes = axes.flatten() # Duỗi mảng để dễ dùng vòng lặp

# Duyệt từ task 1 đến task 10
for i in range(1, 11):
    # Giả sử tên file là task_1.txt hoặc task1.txt, bạn chỉnh lại dòng dưới cho đúng tên file thực tế
    # Ví dụ dưới đây đang tìm file tên: 1.txt, 2.txt... hoặc task1.txt tùy bạn đặt
    filename = f"task{i}.txt"  # <--- SỬA TÊN FILE NẾU CẦN (ví dụ: f"{i}.txt")
    full_path = os.path.join(folder_path, filename)
    
    ax = axes[i-1] # Lấy khung hình tương ứng
    
    if os.path.exists(full_path):
        try:
            # Đọc file: Bỏ dòng đầu (skiprows=1), phân tách bằng khoảng trắng
            df = pd.read_csv(full_path, header=None, skiprows=1, sep=r'\s+')
            
            # Lấy dữ liệu: Cột 0 là Gen (X), Cột 1 là Kết quả (Y)
            x = df.iloc[:, 0] 
            y = df.iloc[:, 1]

            # VẼ BIỂU ĐỒ
            # marker='o': Chấm tròn
            # linestyle='-': Đường gạch nối
            ax.plot(x, y, 
                    linestyle='-', linewidth=1.5, color='tab:blue', # Đường đi
                    marker='o', markersize=5, markerfacecolor='red', # Chấm tròn
                    markevery=100) # <--- CHỈNH CÁI NÀY: Cách 100 điểm mới chấm 1 cái
            
            # Trang trí
            ax.set_title(f'Task {i}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Gen', fontsize=10)
            ax.set_ylabel('Kết quả', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

        except Exception as e:
            ax.text(0.5, 0.5, 'Lỗi đọc file', ha='center', color='red')
            print(f"Lỗi tại file {filename}: {e}")
    else:
        ax.text(0.5, 0.5, f'Không tìm thấy\n{filename}', ha='center', color='gray')

# Căn chỉnh và Lưu ảnh
plt.tight_layout()
plt.suptitle(f"Biểu đồ kết quả 10 Task từ {folder_path}", fontsize=18, y=1.02)

# LƯU ẢNH (Quan trọng: Phải đặt trước plt.show())
plt.savefig(output_image_name, dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ thành công vào file: {output_image_name}")

# Hiển thị lên màn hình (nếu chạy trên Jupyter/IDE có hỗ trợ GUI)
plt.show()