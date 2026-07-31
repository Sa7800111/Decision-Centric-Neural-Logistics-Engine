import numpy as np
from scipy.optimize import minimize

class SPOQuadraticSolver:
    def __init__(self, Q_matrix, A_matrix, b_vector):
        self.Q = np.asarray(Q_matrix)
        self.A = np.asarray(A_matrix)
        self.b = np.asarray(b_vector)

    def solve(self, p_vector):
        p = np.asarray(p_vector)
        cons = {'type': 'ineq', 'fun': lambda z: self.b - self.A @ z}
        res = minimize(lambda z: 0.5 * z.T @ self.Q @ z + p.T @ z, 
                       x0=np.zeros(len(p)), constraints=cons)
        return res.x if res.success else np.zeros(len(p))

    def compute_spo_loss(self, p_true, p_pred):
        z_true = self.solve(p_true)
        z_spo = self.solve(2 * np.asarray(p_pred) - np.asarray(p_true))
        return float(np.dot(p_true, z_spo - z_true))