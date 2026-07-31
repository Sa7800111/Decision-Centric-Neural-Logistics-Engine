import numpy as np

class TaskLossGradEstimator:
    def __init__(self, smoothing_sigma=0.1, n_samples=50):
        self.sigma = float(smoothing_sigma)
        self.n = int(n_samples)

    def estimate_grad(self, y_pred, context, task_loss_func):
        y_p = np.asarray(y_pred)
        total_grad = np.zeros_like(y_p)
        
        for _ in range(self.n):
            noise = np.random.normal(0, self.sigma, y_p.shape)
            y_noisy = y_p + noise
            loss = task_loss_func(y_noisy, context)
            total_grad += loss * noise
            
        return total_grad / (self.n * self.sigma**2)