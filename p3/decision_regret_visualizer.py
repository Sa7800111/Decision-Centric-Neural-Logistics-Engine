import numpy as np

class RegretVisualizer:
    def __init__(self, log_data):
        self.data = log_data

    def generate_report(self):
        mse = [d['mse'] for d in self.data]
        regret = [d['regret'] for d in self.data]
        
        correlation = np.corrcoef(mse, regret)[0, 1]
        return {
            "pearson_r": float(correlation),
            "mse_trend": "decreasing" if mse[-1] < mse[0] else "increasing",
            "regret_trend": "decreasing" if regret[-1] < regret[0] else "increasing"
        }