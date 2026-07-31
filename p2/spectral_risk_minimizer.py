import numpy as np
from scipy.optimize import minimize

class SpectralRiskMinimizer:
    def __init__(self, fourier_features, targets):
        self.phi = np.asarray(fourier_features, dtype=float)
        self.y = np.asarray(targets, dtype=float).flatten()

    def _loss(self, weights, lam):
        preds = self.phi @ weights
        mse = np.mean((self.y - preds)**2)
        reg = lam * np.sum(weights**2)
        return mse + reg

    def optimize(self, lam=0.01):
        init_w = np.zeros(self.phi.shape[1])
        res = minimize(self._loss, init_w, args=(lam,), method='BFGS')
        return res.x