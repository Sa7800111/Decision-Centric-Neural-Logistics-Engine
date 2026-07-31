import pandas as pd
import numpy as np

class IndividualInferenceQuery:
    def __init__(self, counterfactual_engine):
        if not hasattr(counterfactual_engine, 'compute_counterfactuals'):
            raise TypeError()
        self.cf_engine = counterfactual_engine

    def query_ite(self, individual_obs, treatment_node, outcome_node):
        if not isinstance(individual_obs, dict):
            raise TypeError()
            
        obs_df = pd.DataFrame([individual_obs])
        
        cf_0 = self.cf_engine.compute_counterfactuals(
            obs_df, 
            list(individual_obs.keys()), 
            {treatment_node: 0}
        )
        
        cf_1 = self.cf_engine.compute_counterfactuals(
            obs_df, 
            list(individual_obs.keys()), 
            {treatment_node: 1}
        )
        
        y0 = cf_0[outcome_node].iloc[0]
        y1 = cf_1[outcome_node].iloc[0]
        
        return float(y1 - y0)