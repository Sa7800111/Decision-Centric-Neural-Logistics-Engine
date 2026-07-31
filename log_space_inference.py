import numpy as np

class LogSpaceInference:
    def __init__(self, clip_min=-1e4, clip_max=1e4):
        if clip_min >= clip_max:
            raise ValueError()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def safe_log(self, probs):
        p = np.asarray(probs, dtype=float)
        if np.any(p < 0):
            raise ValueError()
        p = np.clip(p, 1e-300, 1.0)
        return np.clip(np.log(p), self.clip_min, self.clip_max)

    def log_sum_exp(self, log_probs, axis=None):
        lp = np.asarray(log_probs, dtype=float)
        if lp.size == 0:
            raise ValueError()
            
        max_log = np.max(lp, axis=axis, keepdims=True)
        valid_max = np.where(np.isneginf(max_log), 0, max_log)
        
        sum_exp = np.sum(np.exp(lp - valid_max), axis=axis, keepdims=True)
        result = valid_max + np.log(np.clip(sum_exp, 1e-300, np.inf))
        
        if axis is None:
            return np.squeeze(result).item()
        return np.squeeze(result, axis=axis)

    def compute_posterior(self, log_prior, log_likelihood):
        lp = np.asarray(log_prior, dtype=float)
        ll = np.asarray(log_likelihood, dtype=float)
        
        if lp.shape != ll.shape:
            raise ValueError()
            
        log_unnormalized = lp + ll
        log_marginal = self.log_sum_exp(log_unnormalized)
        
        return np.clip(log_unnormalized - log_marginal, self.clip_min, self.clip_max)