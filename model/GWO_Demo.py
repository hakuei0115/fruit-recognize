import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化參數
num_wolves = 10  # 狼的數量
num_iterations = 50  # 總迭代次數
search_space = [-100, 100]  # 搜索空間範圍
prey_position = np.array([0, 0])  # 獵物（目標解）的位置

# 隨機初始化狼群位置（分散在搜索空間內）
wolves = np.random.uniform(search_space[0], search_space[1], (num_wolves, 2))
alpha = wolves[0]  # 初始化 α 狼位置
beta = wolves[1]   # 初始化 β 狼位置
delta = wolves[2]  # 初始化 δ 狼位置

# 記錄歷史數據
wolves_history = []
alpha_history = []
beta_history = []
delta_history = []

# 模擬數據生成：逐步收縮過程
for i in range(num_iterations):
    # 普通狼逐漸接近獵物
    wolves += (prey_position - wolves) * np.random.uniform(0.02, 0.1, size=(num_wolves, 2))
    # α、β、δ 狼帶頭逼近
    alpha += (prey_position - alpha) * 0.15
    beta += (prey_position - beta) * 0.1
    delta += (prey_position - delta) * 0.08
    # 保存歷史數據
    wolves_history.append(wolves.copy())
    alpha_history.append(alpha.copy())
    beta_history.append(beta.copy())
    delta_history.append(delta.copy())

# 繪製初始化
fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], color='blue', label='Wolves', s=50)
alpha_scat = ax.scatter([], [], color='gold', label='Alpha', s=100)
beta_scat = ax.scatter([], [], color='silver', label='Beta', s=80)
delta_scat = ax.scatter([], [], color='brown', label='Delta', s=70)
prey_scat = ax.scatter(prey_position[0], prey_position[1], color='red', label='Prey', s=100)
ax.set_xlim(search_space[0], search_space[1])
ax.set_ylim(search_space[0], search_space[1])
ax.legend()
ax.set_title("GWO Hunting Process")
ax.grid(True)

# 更新動畫
def animate(frame):
    wolves = wolves_history[frame]
    alpha = alpha_history[frame]
    beta = beta_history[frame]
    delta = delta_history[frame]
    scat.set_offsets(wolves)
    alpha_scat.set_offsets(alpha)
    beta_scat.set_offsets(beta)
    delta_scat.set_offsets(delta)
    return scat, alpha_scat, beta_scat, delta_scat, prey_scat

# 動態生成動畫
ani = FuncAnimation(fig, animate, frames=len(wolves_history), interval=100, blit=True)  # 每幀100毫秒

# 顯示動畫
plt.show()