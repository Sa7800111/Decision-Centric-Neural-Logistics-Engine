import numpy as np

class DecisionMetricSuite:
    def __init__(self, solver):
        self.solver = solver

    def compute_all(self, y_true, y_pred, context):
        z_true = self.solver(y_true, context)
        z_pred = self.solver(y_pred, context)
        
        regret = np.dot(y_true, z_pred) - np.dot(y_true, z_true)
        mse = np.mean((y_true - y_pred)**2)
        
        return {
            "regret": float(max(0, regret)),
            "mse": float(mse),
            "decision_error_ratio": float(regret / (mse + 1e-9)),
            "suboptimality": float(regret / (np.abs(np.dot(y_true, z_true)) + 1e-9))
        }