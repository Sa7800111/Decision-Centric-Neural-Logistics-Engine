import numpy as np

class DecisionFocusedScaling:
    def __init__(self, feature_dim):
        self.scales = np.ones(feature_dim)
        self.bias = np.zeros(feature_dim)

    def fit_by_regret(self, x, y_true, solver, n_iters=100):
        for _ in range(n_iters):
            idx = np.random.randint(len(x))
            x_i, y_i = x[idx], y_true[idx]
            
            y_p = (x_i - self.bias) * self.scales
            z_p = solver(y_p)
            z_star = solver(y_i)
            
            grad = (z_p - z_star) * (x_i - self.bias)
            self.scales -= 0.01 * grad
            
        return self.scales

    def transform(self, x):
        return (np.asarray(x) - self.bias) * self.scales