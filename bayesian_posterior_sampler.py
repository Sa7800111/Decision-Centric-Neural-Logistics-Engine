import numpy as np

class BayesianPosteriorSampler:
    def __init__(self, log_prior_fn, log_likelihood_fn):
        if not callable(log_prior_fn) or not callable(log_likelihood_fn):
            raise ValueError()
        self.log_prior = log_prior_fn
        self.log_likelihood = log_likelihood_fn

    def log_posterior(self, theta):
        prior_val = self.log_prior(theta)
        if not np.isfinite(prior_val):
            return -np.inf
            
        ll_val = self.log_likelihood(theta)
        if not np.isfinite(ll_val):
            return -np.inf
            
        return prior_val + ll_val

    def metropolis_hastings(self, init_theta, n_samples, proposal_cov, burn_in=0):
        init_theta = np.asarray(init_theta, dtype=float)
        proposal_cov = np.asarray(proposal_cov, dtype=float)
        
        if proposal_cov.ndim != 2 or proposal_cov.shape[0] != proposal_cov.shape[1]:
            raise ValueError()

        dim = init_theta.shape[0]
        samples = np.empty((n_samples + burn_in, dim))
        samples[0] = init_theta

        current_log_post = self.log_posterior(init_theta)
        accepted = 0

        for i in range(1, n_samples + burn_in):
            proposal = np.random.multivariate_normal(samples[i-1], proposal_cov)
            proposal_log_post = self.log_posterior(proposal)
            
            log_alpha = proposal_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_alpha:
                samples[i] = proposal
                current_log_post = proposal_log_post
                if i >= burn_in:
                    accepted += 1
            else:
                samples[i] = samples[i-1]

        acceptance_rate = accepted / n_samples if n_samples > 0 else 0
        return samples[burn_in:], acceptance_rate