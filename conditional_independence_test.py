import numpy as np
import pandas as pd
from scipy import stats

class ConditionalIndependenceTester:
    def __init__(self, data):
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError()
        self.df = pd.DataFrame(data)
        if self.df.empty:
            raise ValueError()

    def partial_correlation(self, x, y, z):
        if x not in self.df.columns or y not in self.df.columns:
            raise KeyError()
            
        cols = [x, y]
        if isinstance(z, str):
            z = [z]
        elif not isinstance(z, (list, tuple, set)):
            raise TypeError()
            
        for col in z:
            if col not in self.df.columns:
                raise KeyError()
            cols.append(col)

        cov_matrix = np.cov(self.df[cols].values, rowvar=False)
        
        try:
            prec_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            raise RuntimeError()
            
        rho = -prec_matrix[0, 1] / np.sqrt(prec_matrix[0, 0] * prec_matrix[1, 1])
        return float(np.clip(rho, -1.0, 1.0))

    def test_independence(self, x, y, z, alpha=0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError()
            
        n = len(self.df)
        k = len(z) if isinstance(z, (list, tuple, set)) else 1
        
        if n <= k + 3:
            raise ValueError()
            
        r = self.partial_correlation(x, y, z)
        
        fisher_z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n - k - 3)
        z_stat = fisher_z / se
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return bool(p_value > alpha), float(p_value)