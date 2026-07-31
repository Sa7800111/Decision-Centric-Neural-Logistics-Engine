import numpy as np

class NoiseImpactSimulator:
    def __init__(self, variance_levels, seed=None):
        if not isinstance(variance_levels, dict):
            raise TypeError()
        self.variances = variance_levels
        if seed is not None:
            np.random.seed(seed)

    def generate_noise(self, n_samples):
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError()
            
        noise_data = {}
        for node, var in self.variances.items():
            if var < 0:
                raise ValueError()
            noise_data[node] = np.random.normal(0, np.sqrt(var), n_samples)
        return noise_data

    def scale_noise(self, scale_factor):
        if scale_factor < 0:
            raise ValueError()
        for node in self.variances:
            self.variances[node] *= scale_factor