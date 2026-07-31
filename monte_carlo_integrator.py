import numpy as np

class MonteCarloIntegrator:
    def __init__(self, n_samples=10000, seed=None):
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError()
        self.n_samples = n_samples
        if seed is not None:
            np.random.seed(seed)

    def integrate_1d(self, func, bounds):
        if not callable(func):
            raise TypeError()
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError()
            
        a, b = bounds
        samples = np.random.uniform(a, b, self.n_samples)
        evals = np.array([func(x) for x in samples])
        
        if not np.all(np.isfinite(evals)):
            raise ValueError()
            
        volume = b - a
        integral = volume * np.mean(evals)
        variance = (volume**2 / self.n_samples) * np.var(evals)
        
        return float(integral), float(variance)

    def integrate_mc(self, func, sampler, volume=1.0):
        if not callable(func) or not callable(sampler):
            raise TypeError()
        if volume <= 0:
            raise ValueError()
            
        samples = sampler(self.n_samples)
        evals = np.apply_along_axis(func, 1, np.atleast_2d(samples))
        
        integral = volume * np.mean(evals)
        return float(integral)