import numpy as np

class FourierBasisConstructor:
    def __init__(self, d_in, d_out, bandwidth=1.0):
        self.d_in = d_in
        self.d_out = d_out
        self.sigma = float(bandwidth)
        self.W = np.random.normal(0, 1.0 / self.sigma, (d_out, d_in))
        self.b = np.random.uniform(0, 2 * np.pi, d_out)

    def get_basis(self, x):
        x_val = np.asarray(x, dtype=float)
        if x_val.ndim == 1:
            x_val = x_val.reshape(1, -1)
        
        projection = x_val @ self.W.T + self.b
        return np.sqrt(2.0 / self.d_out) * np.cos(projection)

    def update_bandwidth(self, new_sigma):
        self.sigma = float(new_sigma)
        self.W = np.random.normal(0, 1.0 / self.sigma, (self.d_out, self.d_in))