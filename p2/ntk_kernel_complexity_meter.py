import numpy as np

class KernelComplexityMeter:
    def __init__(self, ntk_matrix):
        self.k = np.asarray(ntk_matrix)

    def effective_rank(self):
        s = np.linalg.svd(self.k, compute_uv=False)
        s = s[s > 1e-10]
        p = s / np.sum(s)
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(np.exp(entropy))

    def compute_rademacher_complexity(self):
        n = self.k.shape[0]
        return float(np.sqrt(np.trace(self.k)) / n)