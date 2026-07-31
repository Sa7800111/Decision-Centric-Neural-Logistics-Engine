import numpy as np

class CausalFeatureSelector:
    def __init__(self, feature_matrix, treatment, outcome):
        self.X = np.asarray(feature_matrix, dtype=float)
        self.t = np.asarray(treatment, dtype=float)
        self.y = np.asarray(outcome, dtype=float)

    def compute_mutual_info_approx(self, col_idx):
        feat = self.X[:, col_idx]
        corr_t = np.corrcoef(feat, self.t)[0, 1]
        corr_y = np.corrcoef(feat, self.y)[0, 1]
        return abs(corr_y) * (1 - abs(corr_t))

    def select_top_k(self, k):
        scores = [self.compute_mutual_info_approx(i) for i in range(self.X.shape[1])]
        return np.argsort(scores)[-k:][::-1]