import numpy as np

class SurrogateDecisionLoss:
    def __init__(self, quadratic_param=0.5):
        self.mu = float(quadratic_param)

    def compute_loss(self, y_true, y_pred, z_opt):
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        z = np.asarray(z_opt)
        
        linear_term = np.dot(y_t - y_p, z)
        quad_term = self.mu * np.sum((y_t - y_p)**2)
        return float(linear_term + quad_term)

    def compute_gradient(self, y_true, y_pred, z_opt):
        return -(np.asarray(z_opt) + 2 * self.mu * (np.asarray(y_true) - np.asarray(y_pred)))