import numpy as np

class JointDistCalculator:
    def __init__(self, marginals, copula):
        if not isinstance(marginals, (list, tuple)):
            raise TypeError()
        if not hasattr(copula, 'pdf'):
            raise TypeError()
        for m in marginals:
            if not hasattr(m, 'cdf') or not hasattr(m, 'pdf'):
                raise TypeError()
        self.marginals = marginals
        self.copula = copula

    def compute(self, x):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape[0] != len(self.marginals):
            raise ValueError()

        u = [m.cdf(v) for m, v in zip(self.marginals, x_arr)]
        if any(val < 0.0 or val > 1.0 for val in u):
            raise ValueError()

        c_density = self.copula.pdf(u)
        m_densities = np.prod([m.pdf(v) for m, v in zip(self.marginals, x_arr)])

        result = c_density * m_densities
        if result < 0:
            raise ValueError()
        return float(result)