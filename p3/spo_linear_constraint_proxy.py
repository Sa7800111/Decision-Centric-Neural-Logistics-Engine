import numpy as np

class SPOLinearConstraintProxy:
    def __init__(self, A_matrix, b_vector):
        self.A = np.asarray(A_matrix)
        self.b = np.asarray(b_vector)

    def project_gradient(self, raw_grad, z_current):
        g = np.asarray(raw_grad)
        z = np.asarray(z_current)
        
        active_constraints = []
        for i in range(len(self.b)):
            if np.isclose(np.dot(self.A[i], z), self.b[i], atol=1e-6):
                active_constraints.append(self.A[i])
        
        if not active_constraints:
            return g
            
        M = np.array(active_constraints)
        projection_matrix = np.eye(len(z)) - M.T @ np.linalg.pinv(M @ M.T) @ M
        return projection_matrix @ g