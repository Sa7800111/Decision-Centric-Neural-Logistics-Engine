import numpy as np

class FourierPreprocessorV2:
    def __init__(self, d_in, d_hidden, sigma_ntk=1.0):
        self.w = np.random.normal(0, 1.0 / sigma_ntk, (d_hidden, d_in))
        self.b = np.random.uniform(0, 2 * np.pi, d_hidden)

    def fit_transform(self, x):
        x_val = np.asarray(x, dtype=float)
        proj = x_val @ self.w.T + self.b
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)

    def get_feature_importance(self, weights):
        w_abs = np.abs(self.w)
        importance = np.dot(np.abs(weights[:self.w.shape[0]]), w_abs)
        return importance / (np.sum(importance) + 1e-9)