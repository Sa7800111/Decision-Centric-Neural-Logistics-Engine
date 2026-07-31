import numpy as np

class DecisionRiskManager:
    def __init__(self, risk_aversion=0.5):
        self.lam = float(risk_aversion)

    def compute_cvar(self, costs, alpha=0.95):
        sorted_costs = np.sort(costs)
        index = int(alpha * len(costs))
        var = sorted_costs[index]
        cvar = np.mean(sorted_costs[index:])
        return float(cvar)

    def risk_adjusted_loss(self, y_true, y_pred, decisions):
        errors = np.dot(y_true, decisions.T)
        mean_loss = np.mean(errors)
        volatility = np.std(errors)
        return float(mean_loss + self.lam * volatility)