import numpy as np
from sklearn.linear_model import Ridge

class PolicyInvariantPredictor:
    def __init__(self, environments):
        self.envs = environments

    def find_invariant_features(self, target_idx):
        n_features = self.envs[0].shape[1]
        invariant_set = []
        
        for i in range(n_features):
            if i == target_idx: continue
            
            coefs = []
            for env in self.envs:
                X = env[:, [i]]
                y = env[:, target_idx]
                model = Ridge().fit(X, y)
                coefs.append(model.coef_[0])
            
            if np.var(coefs) < 1e-4:
                invariant_set.append(i)
                
        return invariant_set