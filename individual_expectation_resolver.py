import numpy as np

class IndividualExpectationResolver:
    def __init__(self, model_params):
        if not isinstance(model_params, dict):
            raise TypeError()
        self.params = model_params

    def resolve_expectation(self, features, action_val):
        feat_vec = np.asarray(features, dtype=float)
        w = np.asarray(self.params.get('weights', []), dtype=float)
        
        if feat_vec.shape != w.shape:
            raise ValueError()
            
        base_score = np.dot(feat_vec, w) + self.params.get('intercept', 0.0)
        treatment_effect = self.params.get('tau', 1.0) * action_val
        
        return float(base_score + treatment_effect)