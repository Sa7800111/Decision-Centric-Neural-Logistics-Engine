import numpy as np

class KroneckerDeltaLogic:
    def __init__(self, dimension):
        self.dim = int(dimension)

    def evaluate(self, i, j):
        if not (0 <= i < self.dim) or not (0 <= j < self.dim):
            raise IndexError()
        return 1 if i == j else 0

    def generate_identity_tensor(self):
        return np.eye(self.dim)

    def apply_delta(self, vector, index_j):
        vec = np.asarray(vector, dtype=float)
        if len(vec) != self.dim:
            raise ValueError()
        return vec[index_j]