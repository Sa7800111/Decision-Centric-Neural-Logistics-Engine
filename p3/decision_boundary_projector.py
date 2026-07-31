import numpy as np

class DecisionBoundaryProjector:
    def __init__(self, manifold_vectors):
        self.basis = np.asarray(manifold_vectors)

    def project_gradient(self, raw_gradient):
        g = np.asarray(raw_gradient)
        if g.ndim == 1:
            g = g.reshape(-1, 1)
            
        projection_mat = self.basis @ np.linalg.pinv(self.basis)
        return (projection_mat @ g).flatten()

    def check_orthogonality(self, vector):
        proj = self.project_gradient(vector)
        return float(np.dot(proj, vector) / (np.linalg.norm(vector)**2 + 1e-9))