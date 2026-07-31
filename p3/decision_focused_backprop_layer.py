import numpy as np

class DecisionBackpropLayer:
    def __init__(self, solver_jacobian_fn):
        self.get_jac = solver_jacobian_fn

    def backward(self, d_loss_d_z, y_pred, context):
        jac = self.get_jac(y_pred, context)
        d_z = np.asarray(d_loss_d_z).reshape(-1, 1)
        
        if jac.shape[0] != d_z.shape[0]:
            return np.zeros_like(y_pred)
            
        return (jac.T @ d_z).flatten()