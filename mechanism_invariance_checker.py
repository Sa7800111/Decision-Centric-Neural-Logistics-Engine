import numpy as np
import pandas as pd

class MechanismInvarianceChecker:
    def __init__(self, environments_data):
        if not isinstance(environments_data, list):
            raise TypeError()
        if len(environments_data) < 2:
            raise ValueError()
        self.envs = [pd.DataFrame(e) for e in environments_data]

    def test_linear_invariance(self, target, predictors, threshold=1e-4):
        if not isinstance(target, str) or not isinstance(predictors, list):
            raise TypeError()
        
        weights = []
        for env in self.envs:
            if target not in env.columns:
                raise KeyError()
            for p in predictors:
                if p not in env.columns:
                    raise KeyError()
                    
            X = env[predictors].values
            y = env[target].values
            X = np.c_[np.ones(X.shape[0]), X]
            
            try:
                w = np.linalg.inv(X.T @ X) @ X.T @ y
                weights.append(w)
            except np.linalg.LinAlgError:
                raise RuntimeError()
                
        weights = np.array(weights)
        variances = np.var(weights, axis=0)
        
        return bool(np.all(variances < threshold)), variances.tolist()