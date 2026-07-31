import numpy as np

class DecisionNoiseRobustifier:
    def __init__(self, solver, n_samples=20):
        self.solver = solver
        self.n = n_samples

    def get_robust_decision(self, y_pred, context, sigma=0.05):
        y_p = np.asarray(y_pred)
        decisions = []
        for _ in range(self.n):
            y_noisy = y_p + np.random.normal(0, sigma, y_p.shape)
            decisions.append(self.solver(y_noisy, context))
        
        return np.mean(decisions, axis=0)