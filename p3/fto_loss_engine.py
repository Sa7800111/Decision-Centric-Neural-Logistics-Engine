import numpy as np

class FTOLossEngine:
    def __init__(self, solver_function):
        if not callable(solver_function):
            raise TypeError()
        self.solver = solver_function

    def compute_decision_loss(self, y_true, y_pred, context):
        z_true = self.solver(y_true, context)
        z_pred = self.solver(y_pred, context)
        
        diff = np.asarray(z_true) - np.asarray(z_pred)
        return float(np.sum(diff**2))

    def compute_regret(self, y_true, y_pred, context):
        z_opt_true = self.solver(y_true, context)
        
        cost_true_under_pred = np.dot(y_true, self.solver(y_pred, context))
        cost_true_under_true = np.dot(y_true, z_opt_true)
        
        return float(cost_true_under_pred - cost_true_under_true)