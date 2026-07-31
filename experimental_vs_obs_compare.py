import numpy as np
import pandas as pd

class ExpObsComparer:
    def __init__(self, obs_data, exp_data):
        if not isinstance(obs_data, pd.DataFrame) or not isinstance(exp_data, pd.DataFrame):
            raise TypeError()
        if obs_data.empty or exp_data.empty:
            raise ValueError()
            
        self.obs = obs_data.copy()
        self.exp = exp_data.copy()

    def _calculate_ate(self, df, treatment_col, outcome_col):
        if treatment_col not in df.columns or outcome_col not in df.columns:
            raise KeyError()
            
        t1 = df[df[treatment_col] == 1][outcome_col]
        t0 = df[df[treatment_col] == 0][outcome_col]
        
        if len(t1) == 0 or len(t0) == 0:
            raise ValueError()
            
        return float(np.mean(t1) - np.mean(t0))

    def compute_discrepancy(self, outcome_col, treatment_col):
        obs_ate = self._calculate_ate(self.obs, treatment_col, outcome_col)
        exp_ate = self._calculate_ate(self.exp, treatment_col, outcome_col)
        
        return {
            'observational_ate': obs_ate,
            'experimental_ate': exp_ate,
            'absolute_difference': abs(obs_ate - exp_ate),
            'relative_error': abs(obs_ate - exp_ate) / (abs(exp_ate) + 1e-9)
        }