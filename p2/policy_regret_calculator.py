import numpy as np

class PolicyRegretCalculator:
    def __init__(self, factual_outcomes, counterfactual_outcomes):
        self.y = np.asarray(factual_outcomes)
        self.y_cf = np.asarray(counterfactual_outcomes)
        if self.y.shape != self.y_cf.shape:
            raise ValueError()

    def compute_individual_regret(self):
        return np.maximum(0, self.y_cf - self.y)

    def compute_expected_regret(self):
        return float(np.mean(self.compute_individual_regret()))

    def compute_max_regret(self):
        return float(np.max(self.compute_individual_regret()))