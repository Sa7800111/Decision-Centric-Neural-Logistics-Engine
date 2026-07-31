import numpy as np

class StochasticFTOSimulator:
    def __init__(self, distribution_sampler, n_scenarios=100):
        self.sampler = distribution_sampler
        self.n = n_scenarios

    def simulate_decision_scenarios(self, policy_func):
        scenarios = self.sampler(self.n)
        costs = []
        for s in scenarios:
            decision = policy_func(s)
            costs.append(np.sum(s * decision))
        return np.array(costs)

    def estimate_expected_regret(self, policy_func, optimal_func):
        scenarios = self.sampler(self.n)
        regrets = []
        for s in scenarios:
            cost_p = np.sum(s * policy_func(s))
            cost_o = np.sum(s * optimal_func(s))
            regrets.append(cost_p - cost_o)
        return float(np.mean(regrets))