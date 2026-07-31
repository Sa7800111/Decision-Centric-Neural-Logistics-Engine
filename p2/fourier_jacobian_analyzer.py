import numpy as np

class FourierJacobianAnalyzer:
    def __init__(self, mapping_weights, freq_scale):
        self.w = mapping_weights
        self.scale = freq_scale

    def compute_jacobian_norm(self, x):
        x = np.asarray(x).flatten()
        proj = self.w @ x
        derivatives = -np.sin(proj)
        jacobian = derivatives.reshape(-1, 1) * self.w
        return float(np.linalg.norm(jacobian, ord=2))

    def check_lipschitz_bound(self):
        return float(np.max(np.abs(self.w)) * self.scale)