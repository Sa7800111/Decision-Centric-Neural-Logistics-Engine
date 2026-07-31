import numpy as np

class SPOPlusLoss:
    def __init__(self, linear_solver):
        self.solver = linear_solver

    def compute_loss(self, c_true, c_pred):
        c_t = np.asarray(c_true)
        c_p = np.asarray(c_pred)
        
        w_opt_true = self.solver(c_t)
        
        c_spo = 2 * c_p - c_t
        w_opt_spo = self.solver(c_spo)
        
        term1 = np.dot(c_spo, w_opt_spo)
        term2 = np.dot(c_spo, w_opt_true)
        
        return float(term1 - term2)