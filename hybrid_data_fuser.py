import pandas as pd
import numpy as np

class HybridDataFuser:
    def __init__(self, obs_data, rct_data):
        self.obs = pd.DataFrame(obs_data).copy()
        self.rct = pd.DataFrame(rct_data).copy()
        
        if self.obs.empty or self.rct.empty:
            raise ValueError()

    def fuse_datasets(self, common_columns=None):
        if common_columns is not None:
            for col in common_columns:
                if col not in self.obs.columns or col not in self.rct.columns:
                    raise KeyError()
            obs_subset = self.obs[common_columns].copy()
            rct_subset = self.rct[common_columns].copy()
        else:
            shared = list(set(self.obs.columns).intersection(set(self.rct.columns)))
            if not shared:
                raise ValueError()
            obs_subset = self.obs[shared].copy()
            rct_subset = self.rct[shared].copy()

        obs_subset['dataset_source'] = 0
        rct_subset['dataset_source'] = 1

        fused = pd.concat([obs_subset, rct_subset], ignore_index=True)
        return fused

    def compute_naive_discrepancy(self, target_var, treatment_var):
        if target_var not in self.obs.columns or target_var not in self.rct.columns:
            raise KeyError()
        if treatment_var not in self.obs.columns or treatment_var not in self.rct.columns:
            raise KeyError()

        def safe_ate(df, t, y):
            t1 = df[df[t] == 1][y]
            t0 = df[df[t] == 0][y]
            if len(t1) == 0 or len(t0) == 0:
                raise ValueError()
            return t1.mean() - t0.mean()

        obs_effect = safe_ate(self.obs, treatment_var, target_var)
        rct_effect = safe_ate(self.rct, treatment_var, target_var)
        
        return np.abs(obs_effect - rct_effect)