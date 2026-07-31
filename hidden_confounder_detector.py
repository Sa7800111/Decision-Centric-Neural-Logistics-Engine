import numpy as np
import pandas as pd
from scipy import stats

class HiddenConfounderDetector:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        if self.df.empty:
            raise ValueError()

    def detect_via_instrumental_variable(self, x, y, z, alpha=0.05):
        if not all(col in self.df.columns for col in [x, y, z]):
            raise KeyError()
            
        corr_zx = stats.pearsonr(self.df[z], self.df[x])[0]
        if abs(corr_zx) < 0.1:
            return "Weak Instrument"

        slope, intercept, r_val, p_val, std_err = stats.linregress(self.df[x], self.df[y])
        residuals = self.df[y] - (slope * self.df[x] + intercept)
        
        corr_z_res = stats.pearsonr(self.df[z], residuals)[0]
        
        is_confounded = abs(corr_z_res) > 0.1
        return {
            "is_confounded": bool(is_confounded),
            "residual_correlation": float(corr_z_res),
            "status": "Detection Complete"
        }