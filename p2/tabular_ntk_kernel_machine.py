import numpy as np

class TabularNTKKernelMachine:
    def __init__(self, x_train, sigma=1.0):
        self.x_train = np.asarray(x_train)
        self.sigma = float(sigma)
        self.k_inv = None

    def _compute_ntk(self, x1, x2):
        dot = x1 @ x2.T
        norm1 = np.sum(x1**2, axis=1).reshape(-1, 1)
        norm2 = np.sum(x2**2, axis=1).reshape(1, -1)
        cos_theta = dot / (np.sqrt(norm1 @ norm2) + 1e-9)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return (dot * (np.pi - theta) / (2 * np.pi))

    def fit(self, y_train, reg=1e-4):
        k = self._compute_ntk(self.x_train, self.x_train)
        n = k.shape[0]
        self.k_inv_y = np.linalg.solve(k + reg * np.eye(n), y_train)

    def predict(self, x_test):
        k_cross = self._compute_ntk(x_test, self.x_train)
        return k_cross @ self.k_inv_y