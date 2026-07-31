import numpy as np
import pandas as pd

class AspirinDataGenerator:
    def __init__(self, n_samples=1000, seed=None):
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError()
        self.n = n_samples
        if seed is not None:
            np.random.seed(seed)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_observational(self):
        age = np.random.normal(50, 10, self.n)
        age = np.clip(age, 18, 90)
        
        logit_aspirin = 0.1 * (age - 50)
        prob_aspirin = self._sigmoid(logit_aspirin)
        aspirin = np.random.binomial(1, prob_aspirin)
        
        logit_recovery = -0.05 * (age - 50) + 1.5 * aspirin
        prob_recovery = self._sigmoid(logit_recovery)
        recovery = np.random.binomial(1, prob_recovery)

        df = pd.DataFrame({
            'Age': age, 
            'Aspirin': aspirin, 
            'Recovery': recovery
        })
        
        if df.isnull().values.any():
            raise RuntimeError()
            
        return df