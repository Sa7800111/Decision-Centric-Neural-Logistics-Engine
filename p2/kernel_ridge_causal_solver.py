import numpy as np

class KernelRidgeCausalSolver:
    def __init__(self, kernel_fn, alpha=1.0):
        self.kernel_fn = kernel_fn
        self.alpha = float(alpha)
        self.dual_weights = None
        self.train_x = None

    def fit(self, x, y):
        self.train_x = np.asarray(x)
        y_val = np.asarray(y)
        k = self.kernel_fn(self.train_x, self.train_x)
        n = k.shape[0]
        self.dual_weights = np.linalg.solve(k + self.alpha * np.eye(n), y_val)

    def predict(self, x_new):
        k_cross = self.kernel_fn(x_new, self.train_x)
        return k_cross @ self.dual_weights