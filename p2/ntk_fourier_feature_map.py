import numpy as np

class NTKFeatureMap:
    def __init__(self, input_dim, n_freqs, sigma=1.0):
        self.w = np.random.normal(0, 1.0 / sigma, (n_freqs, input_dim))
        self.b = np.random.uniform(0, 2 * np.pi, n_freqs)

    def map(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        proj = x @ self.w.T + self.b
        return np.sqrt(2.0 / self.w.shape[0]) * np.cos(proj)

    def get_jacobian(self, x):
        x = np.asarray(x).flatten()
        proj = self.w @ x + self.b
        outer = -np.sqrt(2.0 / self.w.shape[0]) * np.sin(proj)
        return outer.reshape(-1, 1) * self.w