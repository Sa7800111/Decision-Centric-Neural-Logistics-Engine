import numpy as np

class NTKSolver:
    def __init__(self, depth, width):
        self.L = depth
        self.m = width

    def compute_ntk(self, x_data):
        x = np.asarray(x_data, dtype=float)
        n = x.shape[0]
        
        sigma = x @ x.T
        norm = np.diag(sigma).reshape(-1, 1)
        cos_theta = sigma / np.sqrt(norm @ norm.T)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        theta = np.arccos(cos_theta)
        dot_sigma = (np.pi - theta) / (2 * np.pi)
        
        ntk_matrix = sigma * dot_sigma
        for _ in range(self.L - 1):
            ntk_matrix = ntk_matrix * dot_sigma + sigma
            
        return ntk_matrix