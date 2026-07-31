import numpy as np

class CausalVOI:
    def __init__(self, reward_function, distribution_p):
        self.reward = reward_function
        self.p = distribution_p

    def expected_value_do(self, x_val):
        return np.sum(self.reward(x_val) * self.p)

    def compute_voi(self, observation_node, possible_obs_values):
        base_val = np.max([self.expected_value_do(x) for x in [0, 1]])
        
        conditional_vals = []
        for val in possible_obs_values:
            p_obs = self._get_p_obs(val)
            best_action_val = np.max([self._expected_val_given_obs(x, val) for x in [0, 1]])
            conditional_vals.append(p_obs * best_action_val)
            
        return float(np.sum(conditional_vals) - base_val)

    def _get_p_obs(self, val):
        return 0.5 

    def _expected_val_given_obs(self, x, obs):
        return 1.0