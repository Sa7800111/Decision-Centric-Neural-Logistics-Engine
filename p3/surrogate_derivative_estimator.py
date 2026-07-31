import numpy as np

class SurrogateDerivativeEstimator:
    def __init__(self, eps=1e-3):
        self.eps = float(eps)

    def estimate_jacobian(self, y_pred, solver_func, context):
        y_p = np.asarray(y_pred).flatten()
        n = len(y_p)
        z_base = solver_func(y_p, context)
        m = len(z_base)
        
        jacobian = np.zeros((m, n))
        for i in range(n):
            y_forward = y_p.copy()
            y_forward[i] += self.eps
            z_forward = solver_func(y_forward, context)
            jacobian[:, i] = (z_forward - z_base) / self.eps
            
        return jacobian

    def get_directional_derivative(self, y_pred, direction, solver_func, context):
        jac = self.estimate_jacobian(y_pred, solver_func, context)
        return jac @ np.asarray(direction)