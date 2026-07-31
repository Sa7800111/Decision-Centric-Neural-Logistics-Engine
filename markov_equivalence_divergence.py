import numpy as np

class EquivalenceDivergence:
    def __init__(self, tester_engine):
        if not hasattr(tester_engine, 'test_equivalence'):
            raise TypeError()
        self.tester = tester_engine

    def kl_divergence(self, p_dist, q_dist):
        p = np.asarray(p_dist, dtype=float)
        q = np.asarray(q_dist, dtype=float)

        if p.shape != q.shape:
            raise ValueError()
        if not np.allclose(np.sum(p), 1.0) or not np.allclose(np.sum(q), 1.0):
            raise ValueError()

        mask = (p > 0) & (q > 0)
        if not np.any(mask):
            return float('inf')

        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    def evaluate_graph_divergence(self, p_dist, q_dist, g1, g2):
        self.tester.g1 = g1
        self.tester.g2 = g2
        
        if not self.tester.test_equivalence():
            return float('inf')

        return self.kl_divergence(p_dist, q_dist)