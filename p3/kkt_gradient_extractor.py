import numpy as np

class KKTGradientExtractor:
    def __init__(self, objective_hessian, constraint_jacobian):
        self.H = np.asarray(objective_hessian)
        self.A = np.asarray(constraint_jacobian)

    def compute_implicit_grad(self, dl_dz):
        dl_dz = np.asarray(dl_dz).reshape(-1, 1)
        n = self.H.shape[0]
        m = self.A.shape[0]
        
        kkt_matrix = np.block([
            [self.H, self.A.T],
            [self.A, np.zeros((m, m))]
        ])
        
        rhs = np.vstack([dl_dz, np.zeros((m, 1))])
        try:
            solution = np.linalg.solve(kkt_matrix, rhs)
            return solution[:n].flatten()
        except np.linalg.LinAlgError:
            return np.linalg.pinv(kkt_matrix) @ rhs[:n].flatten()