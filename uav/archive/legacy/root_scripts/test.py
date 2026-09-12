import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems.dynamic.df import DF1
from pymoo.indicators.igd import IGD

t = 0.5
problem = DF1(time=t, n_var=10)
PF = problem.pareto_front()  # dap an dung

# === TẠO 3 LOẠI POPULATION ===

np.random.seed(30)

# 1. Population XẤU: random hoàn toàn
bad_pop = np.random.rand(100, 10)
F_bad = problem.evaluate(bad_pop)

# 2. Population TRUNG BÌNH: biết đặt xII gần G(t) nhưng chưa hoàn hảo
import math
G = math.sin(0.5 * math.pi * t)
medium_pop = np.random.rand(100, 10)
for i in range(100):
    for j in range(1, 10):
        medium_pop[i][j] = G + np.random.normal(0, 0.1)  # gần G nhưng có noise
        medium_pop[i][j] = np.clip(medium_pop[i][j], 0, 1)
F_medium = problem.evaluate(medium_pop)

# 3. Population TỐT: xII = G(t) chính xác, x1 trải đều trên [0,1]
good_pop = np.zeros((100, 10))
for i in range(100):
    good_pop[i][0] = i / 99  # x1 trải đều
    for j in range(1, 10):
        good_pop[i][j] = G    # xII = G chính xác → g = 0
F_good = problem.evaluate(good_pop)

# === TÍNH IGD (khoảng cách đến đáp án đúng) ===
igd = IGD(PF)
igd_bad = igd(F_bad)
igd_medium = igd(F_medium)
igd_good = igd(F_good)

# === VẼ ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Xấu
axes[0].plot(PF[:, 0], PF[:, 1], 'r-', linewidth=2, label='Dap an dung')
axes[0].scatter(F_bad[:, 0], F_bad[:, 1], c='blue', s=15, alpha=0.5)
axes[0].set_title(f'XAU - IGD = {igd_bad:.4f}\n(cang LON cang xau)')
axes[0].set_xlabel('f1')
axes[0].set_ylabel('f2')
axes[0].set_xlim(-0.1, 4)
axes[0].set_ylim(-0.1, 4)
axes[0].legend()

# Trung bình
axes[1].plot(PF[:, 0], PF[:, 1], 'r-', linewidth=2, label='Dap an dung')
axes[1].scatter(F_medium[:, 0], F_medium[:, 1], c='green', s=15, alpha=0.5)
axes[1].set_title(f'TRUNG BINH - IGD = {igd_medium:.4f}')
axes[1].set_xlabel('f1')
axes[1].set_ylabel('f2')
axes[1].set_xlim(-0.1, 4)
axes[1].set_ylim(-0.1, 4)
axes[1].legend()

# Tốt
axes[2].plot(PF[:, 0], PF[:, 1], 'r-', linewidth=2, label='Dap an dung')
axes[2].scatter(F_good[:, 0], F_good[:, 1], c='lime', s=15, alpha=0.5)
axes[2].set_title(f'TOT - IGD = {igd_good:.4f}\n(cang NHO cang tot)')
axes[2].set_xlabel('f1')
axes[2].set_ylabel('f2')
axes[2].set_xlim(-0.1, 4)
axes[2].set_ylim(-0.1, 4)
axes[2].legend()

plt.tight_layout()
plt.savefig("fitness_tot_xau.png", dpi=150)
plt.show()

print(f"\n=== KET QUA ===")
print(f"Population XAU:        IGD = {igd_bad:.6f}   (xa dap an)")
print(f"Population TRUNG BINH: IGD = {igd_medium:.6f}   (gan hon)")
print(f"Population TOT:        IGD = {igd_good:.6f}   (sat dap an)")
print(f"\nIGD cang nho = fitness cang tot")