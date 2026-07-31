import numpy as np

class LogNormalExpectationUtility:
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma <= 0:
            raise ValueError()
        self.mu = float(mu)
        self.sigma = float(sigma)

    def get_expectation(self):
        return float(np.exp(self.mu + (self.sigma**2) / 2))

    def get_variance(self):
        return float((np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2))

    def get_median(self):
        return float(np.exp(self.mu))

    def get_mode(self):
        return float(np.exp(self.mu - self.sigma**2))

    def sample(self, n):
        if n <= 0:
            raise ValueError()
        return np.random.lognormal(self.mu, self.sigma, n)