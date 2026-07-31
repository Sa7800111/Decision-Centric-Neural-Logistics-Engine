
import numpy as np

class NTKRegularizer:
    def __init__(self, ntk_matrix):
        self.K = np.asarray(ntk_matrix, dtype=float)
        try:
            self.K_inv = np.linalg.inv(self.K + 1e-6 * np.eye(self.K.shape[0]))
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(self.K)

    def compute_complexity_penalty(self, function_values):
        f = np.asarray(function_values, dtype=float).reshape(-1, 1)
        penalty = f.T @ self.K_inv @ f
        return float(np.squeeze(penalty))