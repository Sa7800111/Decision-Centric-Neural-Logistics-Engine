import numpy as np

class PolicyGradientEstimator:
    def __init__(self, policy_model):
        self.policy = policy_model

    def compute_grad_log_prob(self, x, action):
        probs = self.policy.forward(x)
        if action == 1:
            return (1 - probs) * x
        else:
            return -probs * x

    def update_step(self, x, action, reward, lr=0.001):
        grad = self.compute_grad_log_prob(x, action)
        return lr * grad * reward