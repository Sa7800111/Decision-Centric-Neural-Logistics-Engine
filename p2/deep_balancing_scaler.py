import numpy as np

class DeepBalancingScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features, treatment):
        x = np.asarray(features, dtype=float)
        t = np.asarray(treatment, dtype=float)
        
        x1 = x[t == 1]
        x0 = x[t == 0]
        
        self.mean = (np.mean(x1, axis=0) + np.mean(x0, axis=0)) / 2
        self.std = np.sqrt((np.var(x1, axis=0) + np.var(x0, axis=0)) / 2) + 1e-9

    def transform(self, features):
        if self.mean is None:
            raise RuntimeError()
        return (np.asarray(features) - self.mean) / self.std