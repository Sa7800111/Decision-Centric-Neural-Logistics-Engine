import numpy as np

class DecisionGradientBackprop:
    def __init__(self, solver_jacobian):
        self.jac = np.asarray(solver_jacobian, dtype=float)

    def compute_upstream_grad(self, downstream_loss_grad):
        dl_dz = np.asarray(downstream_loss_grad, dtype=float)
        if dl_dz.ndim == 1:
            dl_dz = dl_dz.reshape(-1, 1)
            
        dz_dy = self.jac
        return (dz_dy.T @ dl_dz).flatten()

    def update_jacobian(self, new_jacobian):
        self.jac = np.asarray(new_jacobian, dtype=float)