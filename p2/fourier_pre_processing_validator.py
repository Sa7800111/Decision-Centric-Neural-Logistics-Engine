import numpy as np

class FourierValidator:
    def __init__(self, original_x, transformed_phi):
        self.x = np.asarray(original_x)
        self.phi = np.asarray(transformed_phi)

    def check_rank_preservation(self):
        rank_x = np.linalg.matrix_rank(self.x)
        rank_phi = np.linalg.matrix_rank(self.phi)
        return rank_phi >= rank_x

    def compute_coherence(self):
        gram = self.phi.T @ self.phi
        diag = np.diag(gram)
        norm_gram = gram / (np.sqrt(np.outer(diag, diag)) + 1e-9)
        np.fill_diagonal(norm_gram, 0)
        return float(np.max(np.abs(norm_gram)))