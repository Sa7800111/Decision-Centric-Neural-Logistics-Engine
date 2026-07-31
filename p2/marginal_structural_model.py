import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class MarginalStructuralModel:
    def __init__(self, data, treatment_timeline, outcome):
        self.df = data
        self.t_cols = treatment_timeline
        self.y = outcome

    def compute_stabilized_weights(self, baseline_covs):
        weights = np.ones(len(self.df))
        
        for i, t in enumerate(self.t_cols):
            prev_t = self.t_cols[:i]
            
            num_model = LogisticRegression().fit(self.df[prev_t] if prev_t else np.ones((len(self.df), 1)), self.df[t])
            den_model = LogisticRegression().fit(self.df[prev_t + baseline_covs], self.df[t])
            
            num_prob = num_model.predict_proba(self.df[prev_t] if prev_t else np.ones((len(self.df), 1)))[:, 1]
            den_prob = den_model.predict_proba(self.df[prev_t + baseline_covs])[:, 1]
            
            weights *= np.where(self.df[t] == 1, num_prob / den_prob, (1 - num_prob) / (1 - den_prob))
            
        return weights

    def estimate_causal_effect(self, weights):
        total_treatment = self.df[self.t_cols].sum(axis=1)
        model = LinearRegression().fit(total_treatment.values.reshape(-1, 1), self.df[self.y], sample_weight=weights)
        return float(model.coef_[0])