import numpy as np

class SPOLinearProxy:
    def __init__(self, constraints_A, bounds_b):
        self.A = np.asarray(constraints_A)
        self.b = np.asarray(bounds_b)

    def solve_relaxed(self, cost_vector):
        c = np.asarray(cost_vector)
        from scipy.optimize import linprog
        res = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
        if res.success:
            return res.x
        return np.zeros_like(c)

    def get_spo_direction(self, c_true, c_pred):
        w_true = self.solve_relaxed(c_true)
        w_pred = self.solve_relaxed(c_pred)
        return w_true - w_pred