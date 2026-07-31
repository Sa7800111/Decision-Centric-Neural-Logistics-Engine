import numpy as np

class SubgradientFTOOptimizer:
    def __init__(self, model_params, learning_rate=0.001):
        self.theta = np.asarray(model_params)
        self.lr = float(learning_rate)

    def step(self, x, y_true, y_pred, z_opt_true, z_opt_pred):
        grad_decision = np.asarray(z_opt_true) - np.asarray(z_opt_pred)
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        update = x @ grad_decision.reshape(1, -1)
        self.theta -= self.lr * update
        return self.theta

    def get_params(self):
        return self.theta.copy()