import numpy as np

class FourierFeatureGenerator:
    def __init__(self, n_components, gamma=1.0, seed=None):
        self.n_components = n_components
        self.gamma = float(gamma)
        if seed is not None:
            np.random.seed(seed)
        self.W = None
        self.b = None

    def fit(self, x_dim):
        self.W = np.random.normal(0, np.sqrt(2 * self.gamma), (x_dim, self.n_components))
        self.b = np.random.uniform(0, 2 * np.pi, self.n_components)

    def generate(self, x_data):
        x = np.asarray(x_data, dtype=float)
        if self.W is None:
            self.fit(x.shape[1])
            
        projection = x @ self.W + self.b
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)