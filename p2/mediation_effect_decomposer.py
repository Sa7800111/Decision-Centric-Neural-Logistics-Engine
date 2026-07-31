import numpy as np
import pandas as pd

class MediationDecomposer:
    def __init__(self, data, treatment, mediator, outcome):
        self.df = pd.DataFrame(data)
        self.t = treatment
        self.m = mediator
        self.y = outcome

    def compute_nie(self):
        t1_data = self.df[self.df[self.t] == 1]
        t0_data = self.df[self.df[self.t] == 0]
        
        prob_m_t1 = t1_data[self.m].value_counts(normalize=True).sort_index()
        prob_m_t0 = t0_data[self.m].value_counts(normalize=True).sort_index()
        
        expected_y_m_t0 = self.df[self.df[self.t] == 0].groupby(self.m)[self.y].mean().sort_index()
        
        common_indices = prob_m_t1.index.intersection(prob_m_t0.index).intersection(expected_y_m_t0.index)
        
        nie = np.sum((prob_m_t1.loc[common_indices] - prob_m_t0.loc[common_indices]) * expected_y_m_t0.loc[common_indices])
        return float(nie)

    def compute_nde(self):
        t1_data = self.df[self.df[self.t] == 1]
        t0_data = self.df[self.df[self.t] == 0]
        
        expected_y_m_t1 = t1_data.groupby(self.m)[self.y].mean().sort_index()
        expected_y_m_t0 = t0_data.groupby(self.m)[self.y].mean().sort_index()
        
        prob_m_t0 = t0_data[self.m].value_counts(normalize=True).sort_index()
        
        common_indices = expected_y_m_t1.index.intersection(expected_y_m_t0.index).intersection(prob_m_t0.index)
        
        nde = np.sum((expected_y_m_t1.loc[common_indices] - expected_y_m_t0.loc[common_indices]) * prob_m_t0.loc[common_indices])
        return float(nde)