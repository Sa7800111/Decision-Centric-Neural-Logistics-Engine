import numpy as np

class PolicyNTKUpdate:
    def __init__(self, ntk_matrix, lr=1e-2):
        self.k = np.asarray(ntk_matrix)
        self.lr = float(lr)

    def compute_natural_gradient(self, vanilla_grads):
        g = np.asarray(vanilla_grads).reshape(-1, 1)
        try:
            return np.linalg.solve(self.k + 1e-5 * np.eye(self.k.shape[0]), g)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.k) @ g

    def update_parameters(self, params, grads):
        nat_grad = self.compute_natural_gradient(grads)
        return params - self.lr * nat_grad.flatten()