import numpy as np

class NTKOODStabilityTester:
    def __init__(self, train_ntk, test_ntk):
        self.k_train = np.asarray(train_ntk)
        self.k_test = np.asarray(test_ntk)

    def compute_spectral_overlap(self):
        u_train, s_train, _ = np.linalg.svd(self.k_train)
        u_test, s_test, _ = np.linalg.svd(self.k_test)
        
        projection = u_train.T @ u_test
        return float(np.linalg.norm(projection, ord='fro') / np.sqrt(len(s_train)))

    def estimate_generalization_gap(self):
        tr_train = np.trace(self.k_train)
        tr_test = np.trace(self.k_test)
        return abs(float(tr_train - tr_test)) / (tr_train + 1e-9)