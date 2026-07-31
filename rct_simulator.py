import numpy as np
import pandas as pd

class RCTSimulator:
    def __init__(self, n_samples=1000, seed=None):
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError()
        self.n = n_samples
        if seed is not None:
            np.random.seed(seed)

    def _sigmoid(self, x):
        return np.clip(1 / (1 + np.exp(-x)), 1e-9, 1 - 1e-9)

    def run_trial(self, treatment_prob=0.5, compliance_rate=1.0):
        if not (0 < treatment_prob < 1):
            raise ValueError()
        if not (0 <= compliance_rate <= 1):
            raise ValueError()

        age = np.random.normal(50, 10, self.n)
        age = np.clip(age, 18, 90)
        
        assigned_treatment = np.random.binomial(1, treatment_prob, self.n)
        
        complies = np.random.binomial(1, compliance_rate, self.n)
        actual_treatment = np.where(complies == 1, assigned_treatment, 1 - assigned_treatment)
        
        logit_recovery = -0.05 * (age - 50) + 1.5 * actual_treatment
        prob_recovery = self._sigmoid(logit_recovery)
        recovery = np.random.binomial(1, prob_recovery)

        return pd.DataFrame({
            'Age': age, 
            'Assigned_Aspirin': assigned_treatment,
            'Actual_Aspirin': actual_treatment,
            'Recovery': recovery,
            'Complied': complies
        })