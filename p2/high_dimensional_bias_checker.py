import numpy as np

class HighDimBiasChecker:
    def __init__(self, features, treatment, outcome):
        self.X = np.asarray(features, dtype=float)
        self.t = np.asarray(treatment, dtype=float).flatten()
        self.y = np.asarray(outcome, dtype=float).flatten()

    def compute_selection_bias_index(self):
        t1_idx = np.where(self.t == 1)[0]
        t0_idx = np.where(self.t == 0)[0]
        
        if len(t1_idx) == 0 or len(t0_idx) == 0:
            raise ValueError()
            
        mean_diff = np.abs(np.mean(self.X[t1_idx], axis=0) - np.mean(self.X[t0_idx], axis=0))
        pooled_std = np.sqrt((np.var(self.X[t1_idx], axis=0) + np.var(self.X[t0_idx], axis=0)) / 2)
        
        smd = mean_diff / (pooled_std + 1e-9)
        return float(np.mean(smd))

    def check_overlap_violation(self, threshold=0.01):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(self.X, self.t)
        probs = model.predict_proba(self.X)[:, 1]
        
        violations = np.sum((probs < threshold) | (probs > 1 - threshold))
        return bool(violations > 0), float(violations / len(probs))