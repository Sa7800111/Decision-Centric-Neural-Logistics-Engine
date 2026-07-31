import numpy as np

class CovariateBalanceTester:
    def __init__(self, features, treatment):
        self.x = np.asarray(features)
        self.t = np.asarray(treatment)

    def compute_smd(self):
        x1 = self.x[self.t == 1]
        x0 = self.x[self.t == 0]
        
        mean_diff = np.mean(x1, axis=0) - np.mean(x0, axis=0)
        var_pooled = (np.var(x1, axis=0) + np.var(x0, axis=0)) / 2
        
        return mean_diff / np.sqrt(var_pooled + 1e-9)

    def is_balanced(self, threshold=0.1):
        smds = self.compute_smd()
        return bool(np.all(np.abs(smds) < threshold))