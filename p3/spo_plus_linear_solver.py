import numpy as np
from scipy.optimize import linprog

class SPOPlusLinearSolver:
    def __init__(self, A_ub, b_ub):
        self.A = A_ub
        self.b = b_ub

    def solve(self, cost_vec):
        c = np.asarray(cost_vec)
        res = linprog(c, A_ub=self.A, b_ub=self.b, method='highs')
        if res.success:
            return res.x
        return np.zeros_like(c)

    def get_spo_plus_grad(self, c_true, c_pred):
        w_true = self.solve(c_true)
        w_spo = self.solve(2 * np.asarray(c_pred) - np.asarray(c_true))
        return w_true - w_spo