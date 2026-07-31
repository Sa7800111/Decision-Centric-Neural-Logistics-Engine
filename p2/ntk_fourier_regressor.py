import numpy as np

class NTKFourierRegressor:
    def __init__(self, fourier_dim, lam=1e-3):
        self.d = fourier_dim
        self.lam = float(lam)
        self.weights = None

    def fit(self, phi, y):
        phi = np.asarray(phi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, p = phi.shape
        
        if p > n:
            self.weights = phi.T @ np.linalg.solve(phi @ phi.T + self.lam * np.eye(n), y)
        else:
            self.weights = np.linalg.solve(phi.T @ phi + self.lam * np.eye(p), phi.T @ y)

    def predict(self, phi_new):
        if self.weights is None:
            raise RuntimeError()
        return np.asarray(phi_new) @ self.weights

    def get_ntk_variance(self, phi_new):
        phi_new = np.asarray(phi_new)
        return np.sum(phi_new**2, axis=1) * self.lam