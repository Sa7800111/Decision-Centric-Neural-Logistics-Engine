import numpy as np

class FTOEvaluator:
    def __init__(self, solver):
        self.solver = solver

    def compute_regret_statistics(self, y_true_list, y_pred_list, contexts):
        regrets = []
        for yt, yp, ctx in zip(y_true_list, y_pred_list, contexts):
            z_star = self.solver(yt, ctx)
            z_hat = self.solver(yp, ctx)
            
            cost_opt = np.dot(yt, z_star)
            cost_pred = np.dot(yt, z_hat)
            regrets.append(max(0, cost_pred - cost_opt))
            
        return {
            "mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets)),
            "p95_regret": float(np.percentile(regrets, 95)),
            "zero_regret_rate": float(np.mean(np.array(regrets) < 1e-7))
        }