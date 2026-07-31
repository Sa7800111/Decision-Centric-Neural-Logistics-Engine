import numpy as np

class PolicyRobustnessEval:
    def __init__(self, policy_func):
        self.policy = policy_func

    def compute_adversarial_vulnerability(self, x, epsilon=0.01):
        x = np.asarray(x)
        original_action = self.policy(x)
        
        noise = np.random.uniform(-epsilon, epsilon, x.shape)
        perturbed_x = x + noise
        new_action = self.policy(perturbed_x)
        
        return bool(original_action != new_action)

    def monte_carlo_reliability(self, x, n_trials=100, sigma=0.05):
        actions = []
        for _ in range(n_trials):
            noise = np.random.normal(0, sigma, x.shape)
            actions.append(self.policy(x + noise))
        return float(np.mean(actions))