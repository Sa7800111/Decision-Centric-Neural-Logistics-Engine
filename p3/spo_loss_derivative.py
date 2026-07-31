import numpy as np

class SPOLossDerivative:
    def __init__(self, linear_solver):
        self.solver = linear_solver

    def compute_subgradient(self, c_true, c_pred):
        c_t = np.asarray(c_true)
        c_p = np.asarray(c_pred)
        
        w_star_true = self.solver(c_t)
        w_star_spo = self.solver(2 * c_p - c_t)
        
        return (w_star_true - w_star_spo).flatten()