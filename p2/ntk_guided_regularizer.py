import numpy as np

class NTKRegularizer:
    def __init__(self, ntk_matrix, alpha=0.1):
        self.k = np.asarray(ntk_matrix)
        self.alpha = alpha

    def compute_penalty(self, function_values):
        f = np.asarray(function_values).reshape(-1, 1)
        try:
            inv_k = np.linalg.inv(self.k + 1e-6 * np.eye(len(self.k)))
            penalty = f.T @ inv_k @ f
            return float(self.alpha * penalty)
        except np.linalg.LinAlgError:
            return 0.0