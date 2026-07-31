import numpy as np

class KernelTargetAlignment:
    def __init__(self, kernel_matrix):
        self.K = np.asarray(kernel_matrix, dtype=float)
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError()

    def compute_alignment(self, target_vector):
        y = np.asarray(target_vector, dtype=float).reshape(-1, 1)
        y_kernel = y @ y.T
        
        numerator = np.trace(self.K @ y_kernel)
        denominator = np.sqrt(np.trace(self.K @ self.K) * np.trace(y_kernel @ y_kernel))
        
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def center_kernel(self):
        n = self.K.shape[0]
        unit = np.ones((n, n)) / n
        self.K = self.K - unit @ self.K - self.K @ unit + unit @ self.K @ unit
        return self.K