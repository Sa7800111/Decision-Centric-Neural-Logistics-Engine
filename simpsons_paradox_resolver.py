import pandas as pd
import numpy as np

class SimpsonsParadoxResolver:
    def __init__(self, data):
        if not isinstance(data, (pd.DataFrame, dict, np.ndarray)):
            raise TypeError()
        self.df = pd.DataFrame(data)
        if self.df.empty:
            raise ValueError()

    def calculate_marginal_effect(self, treatment, outcome):
        if treatment not in self.df.columns or outcome not in self.df.columns:
            raise KeyError()
        grouped = self.df.groupby(treatment)[outcome].mean()
        if len(grouped) < 2:
            raise ValueError()
        return float(grouped.diff().iloc[-1])

    def calculate_conditional_effect(self, treatment, outcome, confounder):
        if confounder not in self.df.columns:
            raise KeyError()
        grouped = self.df.groupby([confounder, treatment])[outcome].mean().unstack()
        if grouped.shape[1] < 2:
            raise ValueError()
        effects = grouped.diff(axis=1).iloc[:, -1]
        weights = self.df[confounder].value_counts(normalize=True)
        return float((effects * weights).sum())

    def detect_paradox(self, treatment, outcome, confounder):
        marginal = self.calculate_marginal_effect(treatment, outcome)
        conditional = self.calculate_conditional_effect(treatment, outcome, confounder)
        
        if np.isnan(marginal) or np.isnan(conditional):
            raise ValueError()
            
        is_paradox = (np.sign(marginal) != np.sign(conditional)) and (marginal != 0) and (conditional != 0)
        return bool(is_paradox), marginal, conditional