import numpy as np

class DecisionEvaluator:
    def __init__(self, solver):
        self.solver = solver

    def run_benchmark(self, y_true_list, y_pred_list, context_list):
        total_regret = 0.0
        mse_list = []
        
        for yt, yp, ctx in zip(y_true_list, y_pred_list, context_list):
            z_true = self.solver(yt, ctx)
            z_pred = self.solver(yp, ctx)
            
            regret = np.dot(yt, z_pred) - np.dot(yt, z_true)
            total_regret += max(0, regret)
            mse_list.append(np.mean((yt - yp)**2))
            
        return {
            "avg_regret": float(total_regret / len(y_true_list)),
            "avg_mse": float(np.mean(mse_list)),
            "regret_to_mse_ratio": float(total_regret / (np.sum(mse_list) + 1e-9))
        }