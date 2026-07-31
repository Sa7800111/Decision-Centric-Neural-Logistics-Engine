import numpy as np

class GradientFreeFTOProxy:
    def __init__(self, solver, n_particles=30):
        self.solver = solver
        self.n = n_particles

    def estimate_direction(self, y_pred, context, loss_func, sigma=0.1):
        y_p = np.asarray(y_pred)
        noise = np.random.normal(0, sigma, (self.n, *y_p.shape))
        losses = np.array([loss_func(y_p + n, context) for n in noise])
        
        weights = np.exp(- (losses - np.min(losses)) / (np.std(losses) + 1e-9))
        weights /= np.sum(weights)
        
        return np.sum(weights.reshape(-1, 1) * noise, axis=0)