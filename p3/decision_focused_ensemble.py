import numpy as np

class DecisionEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_and_aggregate(self, x, context, solver):
        predictions = [m.predict(x) for m in self.models]
        decisions = [solver(p, context) for p in predictions]
        
        avg_decision = np.mean(decisions, axis=0)
        return avg_decision

    def compute_diversity_regret(self, y_true, x, context, solver):
        z_true = solver(y_true, context)
        model_regrets = []
        for m in self.models:
            z_p = solver(m.predict(x), context)
            model_regrets.append(np.dot(y_true, z_p - z_true))
        return np.var(model_regrets)