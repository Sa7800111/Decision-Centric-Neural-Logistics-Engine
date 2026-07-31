import numpy as np

class PerturbedOptimizer:
    def __init__(self, solver, sigma=0.1, n_samples=50):
        self.solver = solver
        self.sigma = float(sigma)
        self.n = int(n_samples)

    def forward(self, y_pred, context):
        y_p = np.asarray(y_pred)
        noise = np.random.normal(0, self.sigma, (self.n, *y_p.shape))
        
        decisions = []
        for i in range(self.n):
            decisions.append(self.solver(y_p + noise[i], context))
            
        return np.mean(decisions, axis=0)

    def compute_gradient(self, y_pred, context, loss_grad):
        y_p = np.asarray(y_pred)
        noise = np.random.normal(0, self.sigma, (self.n, *y_p.shape))
        
        grad_sum = np.zeros_like(y_p)
        for i in range(self.n):
            z_perturbed = self.solver(y_p + noise[i], context)
            grad_sum += np.outer(z_perturbed, noise[i]) @ np.asarray(loss_grad)
            
        return grad_sum / (self.n * self.sigma**2)