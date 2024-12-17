import numpy as np

class APGWO:
    def __init__(self, obj_func, dim, pop_size, max_iter, bounds, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
        self.obj_func = obj_func  # 目標函數
        self.dim = dim            # 維度
        self.pop_size = pop_size  # 狼群大小
        self.max_iter = max_iter  # 最大迭代次數
        self.bounds = bounds      # 範圍（上下界）
        self.w_max = w_max        # 最大慣性權重
        self.w_min = w_min        # 最小慣性權重
        self.c1 = c1              # 個體學習因子
        self.c2 = c2              # 全局學習因子

    def optimize(self):
        # 初始化位置和速度
        wolves = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        velocity = np.zeros_like(wolves)
        
        # 個體最佳位置（pbest）和全局最佳位置（alpha, beta, delta）
        pbest = wolves.copy()
        pbest_scores = np.array([self.obj_func(wolf) for wolf in pbest])
        alpha, beta, delta = None, None, None
        alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf

        for iter_num in range(self.max_iter):
            # 評估當前所有狼的適應度
            scores = np.array([self.obj_func(wolf) for wolf in wolves])
            sorted_indices = np.argsort(scores)
            alpha, beta, delta = wolves[sorted_indices[:3]]
            alpha_score, beta_score, delta_score = scores[sorted_indices[:3]]

            # 更新個體最佳位置
            for i in range(self.pop_size):
                if scores[i] < pbest_scores[i]:
                    pbest[i] = wolves[i]
                    pbest_scores[i] = scores[i]

            # 計算動態慣性權重
            w = self.w_max - iter_num * (self.w_max - self.w_min) / self.max_iter

            # 更新速度和位置
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    # PSO 的速度更新規則
                    velocity[i, j] = (
                        w * velocity[i, j]
                        + self.c1 * r1 * (pbest[i, j] - wolves[i, j])  # 個體最佳影響
                        + self.c2 * r2 * (alpha[j] - wolves[i, j])     # 全局最佳影響
                    )
                    # 灰狼位置更新規則
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * (1 - iter_num / self.max_iter) * r1 - 1
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * (1 - iter_num / self.max_iter) * r1 - 1
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i, j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * (1 - iter_num / self.max_iter) * r1 - 1
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i, j])
                    X3 = delta[j] - A3 * D_delta

                    # 更新位置
                    wolves[i, j] = np.clip((X1 + X2 + X3) / 3 + velocity[i, j], self.bounds[j, 0], self.bounds[j, 1])

            print(f"Iteration {iter_num + 1}/{self.max_iter}, Best Score: {alpha_score:.4f}")

        return alpha, alpha_score