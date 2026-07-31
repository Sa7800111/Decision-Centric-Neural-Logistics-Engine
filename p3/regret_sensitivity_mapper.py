import numpy as np

class RegretSensitivityMapper:
    def __init__(self, solver):
        self.solver = solver

    def map_sensitivity(self, y_true, epsilon=0.01):
        y_t = np.asarray(y_true)
        n = len(y_t)
        sensitivity = np.zeros(n)
        z_star = self.solver(y_t)
        
        for i in range(n):
            y_pert = y_t.copy()
            y_pert[i] += epsilon
            z_pert = self.solver(y_pert)
            sensitivity[i] = np.dot(y_t, z_pert - z_star) / epsilon
        return sensitivity