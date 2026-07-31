import numpy as np

class NTKFourierPreprocessor:
    def __init__(self, input_dim, num_features, sigma=1.0):
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = float(sigma)
        self.B = np.random.normal(0, self.sigma, (input_dim, num_features))

    def _compute_ntk_weights(self, x_data):
        x = np.asarray(x_data, dtype=float)
        norm_x = np.linalg.norm(x, axis=1, keepdims=True)
        return np.exp(-(norm_x**2) / (2 * self.sigma**2))

    def transform(self, x_data):
        x = np.asarray(x_data, dtype=float)
        if x.shape[1] != self.input_dim:
            raise ValueError()
            
        projection = x @ self.B
        fourier_features = np.concatenate([np.cos(projection), np.sin(projection)], axis=-1)
        
        weights = self._compute_ntk_weights(x)
        return fourier_features * weights

    def update_projection_matrix(self, new_sigma):
        self.sigma = float(new_sigma)
        self.B = np.random.normal(0, self.sigma, (self.input_dim, self.num_features))