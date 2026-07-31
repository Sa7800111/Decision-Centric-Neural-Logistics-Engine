import numpy as np
from scipy.optimize import minimize

class NonConvexFTOSolver:
    def __init__(self, objective_fn, constraints):
        self.obj = objective_fn
        self.cons = constraints

    def solve_global(self, y_pred, n_restarts=5):
        best_res = None
        dim = len(y_pred)
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(-1, 1, dim)
            res = minimize(self.obj, x0, args=(y_pred,), constraints=self.cons)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        return best_res.x if best_res else np.zeros(dim)