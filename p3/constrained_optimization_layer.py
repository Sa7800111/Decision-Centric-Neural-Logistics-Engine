import numpy as np

class ConstrainedOptLayer:
    def __init__(self, A_mat, b_vec):
        self.A = np.asarray(A_mat)
        self.b = np.asarray(b_vec)

    def forward(self, logits):
        from scipy.optimize import minimize
        l = np.asarray(logits)
        
        def obj(z):
            return np.linalg.norm(z - l)**2
            
        cons = {'type': 'ineq', 'fun': lambda z: self.b - self.A @ z}
        res = minimize(obj, x0=l, constraints=cons)
        return res.x

    def compute_penalty(self, z):
        violations = self.A @ z - self.b
        return float(np.sum(np.maximum(0, violations)**2))