import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class NTKFourierOptimizer:
    def __init__(self, fourier_features, targets):
        self.phi = np.asarray(fourier_features, dtype=float)
        self.y = np.asarray(targets, dtype=float).reshape(-1, 1)

    def _objective(self, w, lam):
        w = w.reshape(-1, 1)
        preds = self.phi @ w
        mse = np.mean((self.y - preds)**2)
        reg = 0.5 * lam * (w.T @ w)
        grad = (2.0 / len(self.y)) * self.phi.T @ (preds - self.y) + lam * w
        return float(mse + reg), grad.flatten()

    def fit(self, lam=0.01):
        init_w = np.zeros(self.phi.shape[1])
        res = fmin_l_bfgs_b(self._objective, init_w, args=(lam,))
        return res[0]