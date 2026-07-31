import numpy as np

class PolicyGradientOptimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = float(learning_rate)

    def update_parameters(self, params, grads, rewards):
        p = np.asarray(params, dtype=float)
        g = np.asarray(grads, dtype=float)
        r = np.asarray(rewards, dtype=float)
        
        if r.ndim == 1:
            r = r.reshape(-1, 1)
            
        update = self.lr * np.mean(g * r, axis=0)
        return p + update

    def compute_advantage(self, rewards, baseline):
        return np.asarray(rewards) - float(baseline)