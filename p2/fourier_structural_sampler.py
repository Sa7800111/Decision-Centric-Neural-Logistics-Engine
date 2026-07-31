import numpy as np

class FourierStructuralSampler:
    def __init__(self, fourier_map, noise_std=0.1):
        self.map = fourier_map
        self.sigma = noise_std

    def sample_interventional_outcome(self, x, weights, action):
        phi = self.map(x)
        base_outcome = phi @ weights
        treatment_effect = action * (phi @ (weights * 0.5))
        noise = np.random.normal(0, self.sigma)
        return float(base_outcome + treatment_effect + noise)