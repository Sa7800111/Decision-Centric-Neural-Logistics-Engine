import numpy as np
import pandas as pd

class SelectionBiasMitigator:
    def __init__(self, data, selection_indicator):
        self.df = pd.DataFrame(data)
        self.s = selection_indicator

    def compute_inverse_probability_weights(self, covariates):
        from sklearn.linear_model import LogisticRegression
        X = self.df[covariates]
        y = self.df[self.s]
        
        model = LogisticRegression().fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        
        weights = 1.0 / np.clip(probs, 1e-6, 1.0)
        return weights

    def apply_weighted_mean(self, target_col, weights):
        return np.sum(self.df[target_col] * weights) / np.sum(weights)