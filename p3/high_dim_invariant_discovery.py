import numpy as np
import networkx as nx

class InvariantDiscovery:
    def __init__(self, environments_data):
        self.envs = environments_data
        self.n_envs = len(environments_data)

    def discover_invariant_structure(self, target_idx, threshold=1e-3):
        n_features = self.envs[0].shape[1]
        stable_parents = []
        
        for i in range(n_features):
            if i == target_idx:
                continue
                
            env_coefs = []
            for data in self.envs:
                x = data[:, [i]]
                y = data[:, target_idx]
                coef = np.linalg.lstsq(x, y, rcond=None)[0]
                env_coefs.append(coef)
                
            if np.var(env_coefs) < threshold:
                stable_parents.append(i)
                
        return stable_parents