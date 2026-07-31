import numpy as np

class ENTKCalculator:
    def __init__(self, model_grads):
        self.grads = [np.asarray(g).flatten() for g in model_grads]
        self.n = len(self.grads)

    def compute_matrix(self):
        ntk = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                val = np.dot(self.grads[i], self.grads[j])
                ntk[i, j] = val
                ntk[j, i] = val
        return ntk

    def compute_trace_norm(self):
        ntk = self.compute_matrix()
        return float(np.trace(ntk))

    def get_eigen_spectrum(self):
        ntk = self.compute_matrix()
        return np.sort(np.linalg.eigvalsh(ntk))[::-1]