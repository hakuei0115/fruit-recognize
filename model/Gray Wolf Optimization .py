import numpy as np

class GWO:
    def __init__(self, obj_func, dim, pop_size, max_iter, bounds):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds

    def optimize(self):
        alpha, beta, delta = None, None, None
        alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf
        wolves = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))

        for iter_num in range(self.max_iter):
            scores = np.array([self.obj_func(wolf) for wolf in wolves])
            sorted_indices = np.argsort(scores)
            alpha, beta, delta = wolves[sorted_indices[:3]]
            alpha_score, beta_score, delta_score = scores[sorted_indices[:3]]

            a = 2 - iter_num * (2 / self.max_iter)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i, j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i, j])
                    X3 = delta[j] - A3 * D_delta

                    wolves[i, j] = np.clip((X1 + X2 + X3) / 3, self.bounds[j, 0], self.bounds[j, 1])

        return alpha, alpha_score