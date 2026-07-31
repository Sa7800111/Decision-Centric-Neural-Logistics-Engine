import numpy as np

class DecisionRegularizer:
    def __init__(self, solver, alpha=0.1):
        self.solver = solver
        self.alpha = float(alpha)

    def penalty(self, y_pred, context):
        z = self.solver(y_pred, context)
        return self.alpha * np.linalg.norm(z, ord=2)**2

    def gradient(self, y_pred, context, eps=1e-5):
        y = np.asarray(y_pred)
        grad = np.zeros_like(y)
        for i in range(len(y)):
            y_p = y.copy()
            y_p[i] += eps
            p_plus = self.penalty(y_p, context)
            p_minus = self.penalty(y - eps, context)
            grad[i] = (p_plus - p_minus) / (2 * eps)
        return grad