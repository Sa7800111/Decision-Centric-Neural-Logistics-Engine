import numpy as np

class FTOTaskEvaluator:
    def __init__(self, optimizer_oracle):
        self.oracle = optimizer_oracle

    def evaluate_performance(self, y_true, y_pred, cost_matrix):
        z_star = self.oracle(y_true, cost_matrix)
        z_hat = self.oracle(y_pred, cost_matrix)
        
        true_cost = np.dot(y_true.flatten(), z_star.flatten())
        perceived_cost = np.dot(y_true.flatten(), z_hat.flatten())
        
        return {
            "optimality_gap": float(perceived_cost - true_cost),
            "relative_error": float((perceived_cost - true_cost) / (abs(true_cost) + 1e-9)),
            "decision_match": bool(np.allclose(z_star, z_hat, atol=1e-5))
        }