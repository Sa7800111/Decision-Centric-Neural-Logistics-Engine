import numpy as np

class TaskVarianceReducer:
    def __init__(self, exponential_decay=0.9):
        self.decay = exponential_decay
        self.running_var = 1.0

    def scale_gradient(self, gradient):
        g = np.asarray(gradient)
        current_var = np.var(g)
        self.running_var = self.decay * self.running_var + (1 - self.decay) * current_var
        return g / np.sqrt(self.running_var + 1e-8)

    def get_coefficient_of_variation(self, batch_grads):
        mean_norm = np.mean([np.linalg.norm(g) for g in batch_grads])
        std_norm = np.std([np.linalg.norm(g) for g in batch_grads])
        return float(std_norm / (mean_norm + 1e-9))