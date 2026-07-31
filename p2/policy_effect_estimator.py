import numpy as np
import pandas as pd

class PolicyEffectEstimator:
    def __init__(self, data, treatment_col, outcome_col):
        self.df = pd.DataFrame(data)
        self.t = treatment_col
        self.y = outcome_col
        if self.t not in self.df.columns or self.y not in self.df.columns:
            raise KeyError()

    def estimate_conditional_ate(self, condition_col, condition_val):
        subset = self.df[self.df[condition_col] == condition_val]
        if subset.empty:
            raise ValueError()
            
        t1 = subset[subset[self.t] == 1][self.y]
        t0 = subset[subset[self.t] == 0][self.y]
        
        if len(t1) == 0 or len(t0) == 0:
            return 0.0
        return float(np.mean(t1) - np.mean(t0))

    def weighted_policy_impact(self, weights_dict):
        total_impact = 0.0
        for val, weight in weights_dict.items():
            total_impact += self.estimate_conditional_ate(list(weights_dict.keys())[0], val) * weight
        return total_impact