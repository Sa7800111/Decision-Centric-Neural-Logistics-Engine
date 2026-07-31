import numpy as np

class InterventionalMatcher:
    def __init__(self, source_phi, target_phi):
        self.s = np.asarray(source_phi)
        self.t = np.asarray(target_phi)

    def compute_mmd(self, sigma=1.0):
        def rbf(x, y):
            dist = np.sum((x[:, None] - y[None, :])**2, axis=-1)
            return np.exp(-dist / (2 * sigma**2))

        k_ss = rbf(self.s, self.s)
        k_tt = rbf(self.t, self.t)
        k_st = rbf(self.s, self.t)
        
        return float(np.mean(k_ss) + np.mean(k_tt) - 2 * np.mean(k_st))

    def get_importance_weights(self):
        from sklearn.linear_model import LogisticRegression
        combined = np.vstack([self.s, self.t])
        labels = np.hstack([np.zeros(len(self.s)), np.ones(len(self.t))])
        
        clf = LogisticRegression().fit(combined, labels)
        probs = clf.predict_proba(self.s)[:, 1]
        return probs / (1 - probs + 1e-9)