import numpy as np

class CombinatorialFTOSolver:
    def __init__(self, cost_func, constraint_evaluator):
        self.cost = cost_func
        self.check = constraint_evaluator

    def greedy_solve(self, weights, context):
        w = np.asarray(weights).flatten()
        n = len(w)
        indices = np.argsort(w)
        
        best_z = np.zeros(n)
        for i in indices:
            test_z = best_z.copy()
            test_z[i] = 1
            if self.check(test_z, context):
                best_z[i] = 1
        return best_z

    def compute_regret(self, y_true, y_pred, context):
        z_true = self.greedy_solve(y_true, context)
        z_pred = self.greedy_solve(y_pred, context)
        return float(np.dot(y_true, z_pred - z_true))