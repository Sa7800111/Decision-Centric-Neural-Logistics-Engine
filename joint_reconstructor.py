import numpy as np

class JointReconstructor:
    def __init__(self, marginals, copula_matrix):
        if not isinstance(marginals, (list, tuple)):
            raise TypeError()
        self.marginals = [np.asarray(m, dtype=float) for m in marginals]
        self.copula = np.asarray(copula_matrix, dtype=float)
        
        if len(self.marginals) != 2:
            raise ValueError()
        
        m1, m2 = self.marginals
        if self.copula.shape != (m1.shape[0], m2.shape[0]):
            raise ValueError()

    def reconstruct(self):
        m1, m2 = self.marginals
        if np.any(m1 < 0) or np.any(m2 < 0):
            raise ValueError()
        if not np.allclose(np.sum(m1), 1.0) or not np.allclose(np.sum(m2), 1.0):
            raise ValueError()

        outer_product = np.outer(m1, m2)
        joint = outer_product * self.copula
        
        total = np.sum(joint)
        if total == 0:
            raise RuntimeError()
            
        return joint / total