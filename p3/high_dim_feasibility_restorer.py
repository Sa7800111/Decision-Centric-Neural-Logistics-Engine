import numpy as np

class FeasibilityRestorer:
    def __init__(self, projection_func):
        self.project = projection_func

    def restore(self, z_raw, max_iters=10):
        z = np.asarray(z_raw)
        for _ in range(max_iters):
            z_new = self.project(z)
            if np.allclose(z, z_new, atol=1e-7):
                break
            z = z_new
        return z

    def get_violation_norm(self, z_raw):
        return float(np.linalg.norm(z_raw - self.project(z_raw)))