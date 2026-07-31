import numpy as np

class TabularKernelAlignment:
    def __init__(self, x_data):
        self.x = np.asarray(x_data, dtype=float)
        self.n = self.x.shape[0]

    def compute_ntk_approximation(self, sigma=1.0):
        dot_prod = self.x @ self.x.T
        norm = np.diag(dot_prod).reshape(-1, 1)
        cos_theta = dot_prod / (np.sqrt(norm @ norm.T) + 1e-9)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return (dot_prod * (np.pi - theta) / (2 * np.pi))

    def evaluate_alignment(self, target_k):
        k_ntk = self.compute_ntk_approximation()
        k_target = np.asarray(target_k, dtype=float)
        num = np.trace(k_ntk @ k_target)
        den = np.sqrt(np.trace(k_ntk @ k_ntk) * np.trace(k_target @ k_target))
        return float(num / (den + 1e-9))