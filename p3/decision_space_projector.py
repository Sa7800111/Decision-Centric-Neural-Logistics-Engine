import numpy as np

class DecisionSpaceProjector:
    def __init__(self, constraints_matrix, bounds_vector):
        self.A = np.asarray(constraints_matrix)
        self.b = np.asarray(bounds_vector)

    def is_feasible(self, decision_vector):
        z = np.asarray(decision_vector)
        constraints_check = np.all(self.A @ z <= self.b + 1e-7)
        return bool(constraints_check)

    def project_to_simplex(self, v):
        v_sorted = np.sort(v)[::-1]
        cssv = np.cumsum(v_sorted)
        indices = np.arange(len(v)) + 1
        rho = np.where(v_sorted + (1 - cssv) / indices > 0)[0][-1]
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)