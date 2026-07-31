import numpy as np

class NonLinearSPOLoss:
    def __init__(self, solver):
        self.solver = solver

    def compute_loss(self, y_true, y_pred):
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        
        w_true = self.solver(y_t)
        w_spo = self.solver(2 * y_p - y_t)
        
        loss = np.dot(2 * y_p - y_t, w_spo - w_true)
        return float(max(0, loss))