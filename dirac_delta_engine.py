import numpy as np

class DiracDeltaEngine:
    def __init__(self, threshold=1e-5):
        if threshold <= 0:
            raise ValueError()
        self.threshold = float(threshold)

    def evaluate(self, x, a):
        if not isinstance(x, (int, float)) or not isinstance(a, (int, float)):
            raise TypeError()
        return 1.0 if abs(x - a) < self.threshold else 0.0

    def sifting_property(self, func, a, x_array):
        if not callable(func):
            raise TypeError()
        arr = np.asarray(x_array, dtype=float)
        if arr.ndim != 1:
            raise ValueError()
        
        for x in arr:
            if abs(x - a) < self.threshold:
                return float(func(x))
        return 0.0
        
    def approximate_density(self, x_array, a, bandwidth):
        if bandwidth <= 0:
            raise ValueError()
        arr = np.asarray(x_array, dtype=float)
        diff = (arr - a) / bandwidth
        return np.exp(-0.5 * diff**2) / (bandwidth * np.sqrt(2 * np.pi))