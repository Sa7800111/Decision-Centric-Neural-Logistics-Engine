import numpy as np

class PolicyBoundaryFinder:
    def __init__(self, predictor_func, feature_bounds):
        self.predict = predictor_func
        self.bounds = feature_bounds

    def find_threshold_crossing(self, x_start, x_end, threshold=0.5, steps=50):
        t_vals = np.linspace(0, 1, steps)
        for t in t_vals:
            x_interp = x_start + t * (x_end - x_start)
            prob = self.predict(x_interp.reshape(1, -1))
            if prob >= threshold:
                return x_interp, float(t)
        return None, 1.0

    def compute_boundary_gradient(self, x_boundary, eps=1e-4):
        grad = np.zeros_like(x_boundary)
        for i in range(len(x_boundary)):
            x_plus = x_boundary.copy()
            x_plus[i] += eps
            x_minus = x_boundary.copy()
            x_minus[i] -= eps
            grad[i] = (self.predict(x_plus.reshape(1, -1)) - self.predict(x_minus.reshape(1, -1))) / (2 * eps)
        return grad / (np.linalg.norm(grad) + 1e-9)