import numpy as np

class NTKBandwidthTuner:
    def __init__(self, x_data, y_data):
        self.x = np.asarray(x_data, dtype=float)
        self.y = np.asarray(y_data, dtype=float).flatten()

    def estimate_median_heuristic(self):
        from scipy.spatial.distance import pdist
        distances = pdist(self.x)
        return float(np.median(distances))

    def compute_ntk_loss(self, sigma):
        n = self.x.shape[0]
        dist_sq = np.sum(self.x**2, axis=1).reshape(-1, 1) + np.sum(self.x**2, axis=1) - 2 * (self.x @ self.x.T)
        k = np.exp(-dist_sq / (2 * sigma**2))
        
        try:
            alpha = np.linalg.solve(k + 1e-6 * np.eye(n), self.y)
            return float(np.mean((self.y - k @ alpha)**2))
        except np.linalg.LinAlgError:
            return float('inf')