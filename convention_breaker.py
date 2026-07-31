import random
import numpy as np

class ConventionBreaker:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def permute_causal_order(self, topological_order):
        if not isinstance(topological_order, list):
            raise TypeError()
        shuffled = topological_order.copy()
        random.shuffle(shuffled)
        return shuffled

    def inject_structural_noise(self, data_array, noise_level=0.1):
        arr = np.asarray(data_array, dtype=float)
        if not (0 <= noise_level <= 1):
            raise ValueError()
        noise = np.random.normal(0, noise_level, arr.shape)
        return arr + noise