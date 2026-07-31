import numpy as np

class IdentityPlusGradient:
    def __init__(self, solver, smoothing=1.0):
        self.solver = solver
        self.mu = float(smoothing)

    def get_gradient_proxy(self, y_pred, context, loss_grad):
        y_p = np.asarray(y_pred)
        z_opt = self.solver(y_p, context)
        
        y_perturbed = y_p + self.mu * np.asarray(loss_grad)
        z_perturbed = self.solver(y_perturbed, context)
        
        return (z_perturbed - z_opt) / self.mu

    def compute_update_direction(self, y_true, y_pred, context):
        z_true = self.solver(y_true, context)
        z_pred = self.solver(y_pred, context)
        return z_pred - z_true