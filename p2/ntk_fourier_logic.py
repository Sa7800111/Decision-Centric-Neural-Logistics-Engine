import numpy as np

class NTKFourierLogic:
    def __init__(self, input_dim, sigma=1.0):
        self.d = input_dim
        self.sigma = float(sigma)

    def compute_spectral_density(self, w_vector):
        w = np.asarray(w_vector, dtype=float)
        norm_sq = np.sum(w**2)
        return np.exp(-self.sigma**2 * norm_sq / 2)

    def get_ntk_conditioned_frequencies(self, n_features):
        freqs = np.random.normal(0, 1.0 / self.sigma, (n_features, self.d))
        return freqs

    def evaluate_kernel_approximation(self, x1, x2, freqs):
        proj1 = x1 @ freqs.T
        proj2 = x2 @ freqs.T
        phi1 = np.concatenate([np.cos(proj1), np.sin(proj1)])
        phi2 = np.concatenate([np.cos(proj2), np.sin(proj2)])
        return np.dot(phi1, phi2) / freqs.shape[0]