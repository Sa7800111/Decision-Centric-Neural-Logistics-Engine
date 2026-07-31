import numpy as np

class NTKKernelAlignment:
    def __init__(self, x_data):
        self.X = np.asarray(x_data, dtype=float)
        self.n = self.X.shape[0]

    def compute_linear_ntk(self):
        return self.X @ self.X.T

    def compute_alignment_score(self, target_k):
        k_ntk = self.compute_linear_ntk()
        k_target = np.asarray(target_k, dtype=float)
        
        upper = np.trace(k_ntk @ k_target)
        lower = np.sqrt(np.trace(k_ntk @ k_ntk) * np.trace(k_target @ k_target))
        
        return float(upper / (lower + 1e-9))