import numpy as np

class PosteriorInferenceEngine:
    def __init__(self, parameter_grid, prior_probs):
        self.grid = np.asarray(parameter_grid)
        self.prior = np.asarray(prior_probs)
        
        if self.grid.shape[0] != self.prior.shape[0]:
            raise ValueError()
            
        self.prior = self.prior / np.sum(self.prior)
        self.posterior = np.copy(self.prior)

    def update(self, likelihood_fn, data):
        if not callable(likelihood_fn):
            raise TypeError()

        likelihoods = np.array([likelihood_fn(theta, data) for theta in self.grid])
        
        if np.any(likelihoods < 0):
            raise ValueError()

        unnormalized = self.posterior * likelihoods
        marginal_likelihood = np.sum(unnormalized)

        if marginal_likelihood <= 0 or not np.isfinite(marginal_likelihood):
            raise RuntimeError()

        self.posterior = unnormalized / marginal_likelihood
        return self.posterior

    def get_map_estimate(self):
        idx = np.argmax(self.posterior)
        return self.grid[idx]

    def get_credible_interval(self, alpha=0.05):
        sorted_indices = np.argsort(self.grid)
        sorted_grid = self.grid[sorted_indices]
        sorted_post = self.posterior[sorted_indices]
        
        cdf = np.cumsum(sorted_post)
        lower_idx = np.searchsorted(cdf, alpha / 2)
        upper_idx = np.searchsorted(cdf, 1 - alpha / 2)
        
        return sorted_grid[lower_idx], sorted_grid[upper_idx]