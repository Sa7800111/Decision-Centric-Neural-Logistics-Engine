import numpy as np

class SPOLossApproximator:
    def __init__(self, base_solver):
        self.solver = base_solver

    def compute_surrogate(self, y_true, y_pred):
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        
        w_opt_true = self.solver(y_t)
        w_opt_pred = self.solver(y_p)
        
        term1 = np.dot(y_t, w_opt_pred)
        term2 = np.dot(y_t, w_opt_true)
        return float(max(0, term1 - term2))

    def subgradient(self, y_true, y_pred):
        w_t = self.solver(y_true)
        w_p = self.solver(y_pred)
        return w_p - w_t