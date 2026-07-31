import numpy as np

class NTKFeatureSelector:
    def __init__(self, ntk_matrix):
        self.k = np.asarray(ntk_matrix)

    def select_active_dimensions(self, threshold=1e-3):
        eigenvalues, eigenvectors = np.linalg.eigh(self.k)
        active_idx = np.where(eigenvalues > threshold)[0]
        return active_idx, eigenvalues[active_idx]

    def compute_effective_dimension(self):
        eigenvalues = np.linalg.eigvalsh(self.k)
        eigenvalues = eigenvalues[eigenvalues > 1e-9]
        return float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))