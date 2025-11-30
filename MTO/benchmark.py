import numpy as np

def get_task_info(task_id):
    
    dim = 50
    bounds = [-100, 100]
    
    if task_id == "T1":   # Sphere, Shift 0
        shift = np.zeros(dim)
    elif task_id == "T2": # Sphere, Shift 80
        shift = np.full(dim, 80.0)
    elif task_id == "T3": # Sphere, Shift -80
        shift = np.full(dim, -80.0)
    elif task_id == "T4": # Weierstrass, D=25
        dim = 25
        shift = np.full(dim, -0.4)
        bounds = [-0.5, 0.5]
    elif task_id == "T5": # Rosenbrock
        shift = np.zeros(dim)
        bounds = [-50, 50]
    elif task_id == "T6": # Ackley
        shift = np.full(dim, 40.0)
        bounds = [-50, 50]
    elif task_id == "T7": # Weierstrass, D=50
        dim = 50
        shift = np.full(dim, -0.4)
        bounds = [-0.5, 0.5]
    elif task_id == "T8": # Schwefel
        # Schwefel chuẩn có đáy tại 420.9687
        shift = np.full(dim, 420.9687) 
        bounds = [-500, 500]
    elif task_id == "T9": # Griewank (Hỗn hợp)
        shift = np.concatenate([np.full(25, -80.0), np.full(25, 80.0)])
        bounds = [-100, 100]
    elif task_id == "T10": # Rastrigin (Hỗn hợp)
        shift = np.concatenate([np.full(25, 40.0), np.full(25, -40.0)])
        bounds = [-50, 50]
    else:
        return None, None, None

    return shift, dim, bounds

def sphere(x, shift):
    """T1, T2, T3: Hàm hình cầu đơn giản"""
    z = x - shift
    return np.sum(z**2)

def rosenbrock(x, shift):
    """T5: Hàm thung lũng (Banana function)"""
    # Để cực trị nằm tại 'shift', ta biến đổi z sao cho khi x=shift thì z=1
    # Vì Rosenbrock gốc có cực trị tại (1, 1, ...)
    z = x - shift + 1 
    # Công thức: sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)

def ackley(x, shift):
    """T6: Hàm Ackley (Cái phễu đục lỗ)"""
    z = x - shift
    dim = len(x)
    sum_sq = np.sum(z**2)
    sum_cos = np.sum(np.cos(2 * np.pi * z))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    return term1 + term2 + 20 + np.e

def griewank(x, shift):
    """T9: Hàm Griewank (Bát gợn sóng)"""
    z = x - shift
    sum_sq = np.sum(z**2) / 4000
    
    # Tạo mảng chỉ số 1, 2, ..., D để tính căn bậc 2
    indices = np.arange(1, len(x) + 1)
    prod_cos = np.prod(np.cos(z / np.sqrt(indices)))
    
    return sum_sq - prod_cos + 1

def rastrigin(x, shift):
    """T10: Hàm Rastrigin (Vỉ đựng trứng)"""
    z = x - shift
    dim = len(x)
    return 10 * dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

def schwefel(x, shift):
    """T8: Hàm Schwefel"""
    # Lưu ý: Schwefel rất nhạy cảm với biên. 
    # Với T8, shift vector chính là 420.9687 (đích đến).
    # Ta dùng biến đổi z để giữ nguyên tính chất địa hình.
    # Ở đây dùng phiên bản CEC benchmark: z = x - shift + 420.9687
    
    z = x - shift + 420.9687
    
    # Ràng buộc biên cứng để tránh lỗi sqrt số âm nếu thuật toán văng quá xa
    # (Trong thực tế MFEA, nên xử lý biên ở thuật toán, nhưng đây là phòng hờ)
    z = np.clip(z, -500, 500) 
    
    dim = len(x)
    return 418.9829 * dim - np.sum(z * np.sin(np.sqrt(np.abs(z))))

def weierstrass(x, shift):
    """T4, T7: Hàm Weierstrass (Bề mặt nhám/fractal)"""
    z = x - shift
    dim = len(x)
    
    # Các hằng số chuẩn của Weierstrass
    a = 0.5
    b = 3
    k_max = 20
    
    # Tính tổng thành phần 1
    sum1 = 0
    for i in range(dim):
        for k in range(k_max + 1):
            sum1 += (a**k) * np.cos(2 * np.pi * (b**k) * (z[i] + 0.5))
            
    # Tính tổng thành phần 2 (hằng số phạt)
    sum2 = 0
    for k in range(k_max + 1):
        sum2 += (a**k) * np.cos(2 * np.pi * (b**k) * 0.5)
    
    return sum1 - dim * sum2


def calculate_objective_function(task_id, x, shift):
    if task_id in ["T1", "T2", "T3"]:
        return sphere(x, shift)
    elif task_id == "T4":
        return weierstrass(x, shift)
    elif task_id == "T5":
        return rosenbrock(x, shift)
    elif task_id == "T6":
        return ackley(x, shift)
    elif task_id == "T7":
        return weierstrass(x, shift)
    elif task_id == "T8":
        return schwefel(x, shift)
    elif task_id == "T9":
        return griewank(x, shift)
    elif task_id == "T10":
        return rastrigin(x, shift)
    else:
        raise ValueError(f"Unknown task ID: {task_id}")