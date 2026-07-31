import numpy as np

class SchurComplementUtility:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be 2D and square")
        self.size = self.matrix.shape[0]

    def compute(self, block_size):
        if not isinstance(block_size, int) or block_size <= 0 or block_size >= self.size:
            raise ValueError("Invalid block size")

        A = self.matrix[:block_size, :block_size]
        B = self.matrix[:block_size, block_size:]
        C = self.matrix[block_size:, :block_size]
        D = self.matrix[block_size:, block_size:]

        try:
            D_inv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            try:
                D_inv = np.linalg.pinv(D)
            except np.linalg.LinAlgError:
                raise RuntimeError("Block D is strictly singular and pseudo-inverse failed")

        return A - B @ D_inv @ C

    def compute_inverse_block(self, block_size):
        schur_comp = self.compute(block_size)
        try:
            return np.linalg.inv(schur_comp)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(schur_comp)

    def check_positive_definite(self):
        try:
            np.linalg.cholesky(self.matrix)
            return True
        except np.linalg.LinAlgError:
            return False