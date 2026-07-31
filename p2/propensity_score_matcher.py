import numpy as np
from sklearn.neighbors import NearestNeighbors

class PropensityScoreMatcher:
    def __init__(self, treatment, propensity_scores):
        self.t = np.asarray(treatment)
        self.ps = np.asarray(propensity_scores).reshape(-1, 1)

    def match(self, caliper=0.05):
        treated_idx = np.where(self.t == 1)[0]
        control_idx = np.where(self.t == 0)[0]
        
        nn = NearestNeighbors(n_neighbors=1, radius=caliper)
        nn.fit(self.ps[control_idx])
        
        distances, indices = nn.kneighbors(self.ps[treated_idx])
        
        matches = []
        for i, idx in enumerate(indices):
            if distances[i] <= caliper:
                matches.append((treated_idx[i], control_idx[idx[0]]))
        return matches