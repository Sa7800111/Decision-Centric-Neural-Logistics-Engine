import numpy as np

class PolicyRegretOptimizer:
    def __init__(self, value_estimates):
        self.values = np.asarray(value_estimates)

    def compute_optimal_policy_index(self):
        return int(np.argmax(self.values))

    def compute_regret_vector(self):
        max_val = np.max(self.values)
        return max_val - self.values

    def expected_value_of_perfect_info(self, probs):
        if len(probs) != len(self.values):
            raise ValueError()
        return float(np.sum(probs * self.compute_regret_vector()))