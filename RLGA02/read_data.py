#%%
import numpy as np

filename = 'cost299.txt'
cost_matrix = np.array([])
city = 0

try:
    with open(filename, 'r') as f:
        
        city = int(f.readline().strip())
        
        cost_matrix = np.loadtxt(f, dtype = int)
        
    print(f"City", city)
    print(f"Cost matrix:")
    print(cost_matrix)
    
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{filename}'")
except ValueError:
    print("Lỗi: Dòng đầu tiên không phải là một số (n) hoặc dữ liệu ma trận không hợp lệ.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
# %%
