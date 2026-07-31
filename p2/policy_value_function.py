import numpy as np

class PolicyValueFunction:
    def __init__(self, reward_model):
        self.model = reward_model

    def evaluate_policy(self, data, policy_func):
        expected_values = []
        for _, row in data.iterrows():
            action = policy_func(row)
            pred_reward = self.model.predict(row, action)
            expected_values.append(pred_reward)
        return float(np.mean(expected_values))