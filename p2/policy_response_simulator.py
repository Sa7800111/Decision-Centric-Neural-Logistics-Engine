import numpy as np

class PolicyResponseSimulator:
    def __init__(self, scm):
        self.scm = scm

    def simulate_tax_policy(self, target_node, tax_levels):
        results = {}
        for level in tax_levels:
            self.scm.set_intervention(target_node, lambda n: np.random.normal(10 - level, 1, n))
            data = self.scm.sample(1000)
            results[level] = data.mean().to_dict()
        return results 