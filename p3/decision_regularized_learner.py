import numpy as np

class DecisionRegularizer:
    def __init__(self, solver, alpha=0.1):
        self.solver = solver
        self.alpha = float(alpha)

    def compute_penalty(self, y_pred, context):
        z_pred = self.solver(y_pred, context)
        return self.alpha * np.linalg.norm(z_pred, ord=1)

    def total_objective(self, y_true, y_pred, context):
        mse = np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)
        return mse + self.compute_penalty(y_pred, context)