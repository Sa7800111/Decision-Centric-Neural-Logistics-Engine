import numpy as np

class TabularDLObjective:
    def __init__(self, lambda_reg=0.01):
        self.lam = float(lambda_reg)

    def causal_mse_loss(self, y_true, y_pred, propensity_scores):
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        ps = np.asarray(propensity_scores)
        
        weights = 1.0 / (ps + 1e-6)
        weighted_error = weights * (y_t - y_p)**2
        return float(np.mean(weighted_error))

    def ntk_complexity_penalty(self, params, ntk_matrix):
        p = np.asarray(params).flatten()
        penalty = p.T @ ntk_matrix @ p
        return self.lam * float(penalty)