import numpy as np

class SolverSensitivityAnalyzer:
    def __init__(self, solver_func, delta=1e-3):
        self.solver = solver_func
        self.delta = float(delta)

    def compute_local_jacobian(self, y_pred, context):
        y_p = np.asarray(y_pred).flatten()
        n = len(y_p)
        z_base = self.solver(y_p, context)
        m = len(z_base)
        
        jacobian = np.zeros((m, n))
        for i in range(n):
            y_plus = y_p.copy()
            y_plus[i] += self.delta
            z_plus = self.solver(y_plus, context)
            jacobian[:, i] = (z_plus - z_base) / self.delta
            
        return jacobian

    def estimate_stability_score(self, y_pred, context):
        jac = self.compute_local_jacobian(y_pred, context)
        return float(np.linalg.norm(jac, ord=2))