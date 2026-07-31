import numpy as np
from scipy.optimize import minimize

class GEstimationEngine:
    def __init__(self, data, treatment, outcome, confounders):
        self.data = data
        self.t = treatment
        self.y = outcome
        self.c = confounders

    def _blip_function(self, psi, l_data):
        return psi * l_data

    def _estimating_equation(self, psi):
        h_psi = self.data[self.y] - self._blip_function(psi, self.data[self.t])
        
        from sklearn.linear_model import LogisticRegression
        X = self.data[self.c]
        model = LogisticRegression().fit(X, self.data[self.t])
        propensity = model.predict_proba(X)[:, 1]
        
        return np.mean((self.data[self.t] - propensity) * h_psi)

    def solve_psi(self):
        res = minimize(lambda p: self._estimating_equation(p)**2, x0=0.0)
        return float(res.x[0])