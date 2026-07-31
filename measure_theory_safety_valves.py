import numpy as np

class MeasureTheorySafetyValves:
    def __init__(self, tolerance=1e-7):
        if tolerance <= 0:
            raise ValueError()
        self.tol = float(tolerance)

    def enforce_probability_axiom(self, distribution):
        dist = np.asarray(distribution, dtype=float)
        
        if np.any(dist < -self.tol):
            raise ValueError()
            
        dist = np.clip(dist, 0.0, 1.0)
        total = np.sum(dist)
        
        if abs(total - 1.0) > self.tol:
            if total == 0:
                raise ValueError()
            dist = dist / total
            
        return dist

    def check_sigma_additivity(self, disjoint_sets_probs, union_prob):
        sum_probs = sum(disjoint_sets_probs)
        if abs(sum_probs - union_prob) > self.tol:
            raise ValueError()
        return True