import numpy as np

class RandomFourierAlignment:
    def __init__(self, n_components):
        self.m = n_components

    def compute_spectral_alignment(self, x, kernel_target):
        n = x.shape[0]
        freqs = np.random.standard_normal((self.m, x.shape[1]))
        
        phi = np.sqrt(2.0 / self.m) * np.cos(x @ freqs.T)
        k_approx = phi @ phi.T
        
        alignment = np.trace(k_approx @ kernel_target)
        norm = np.linalg.norm(k_approx) * np.linalg.norm(kernel_target)
        return float(alignment / (norm + 1e-9))