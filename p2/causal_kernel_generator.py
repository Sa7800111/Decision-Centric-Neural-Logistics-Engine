import numpy as np

class CausalKernelGenerator:
    def __init__(self, bandwidth=1.0):
        self.h = float(bandwidth)

    def compute_rbf_kernel(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-sq_dist / (2 * self.h**2))

    def compute_causal_alignment(self, k_matrix, treatment_vec):
        t = np.asarray(treatment_vec, dtype=float).reshape(-1, 1)
        t_kernel = t @ t.T
        alignment = np.trace(k_matrix @ t_kernel)
        norm = np.linalg.norm(k_matrix) * np.linalg.norm(t_kernel)
        return float(alignment / (norm + 1e-9))