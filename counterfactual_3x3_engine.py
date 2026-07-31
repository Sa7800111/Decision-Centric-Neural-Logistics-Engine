import numpy as np

class Counterfactual3x3Engine:
    def __init__(self, transition_tensor):
        self.T = np.asarray(transition_tensor, dtype=float)
        if self.T.shape != (3, 3, 3):
            raise ValueError()

    def compute_counterfactual_state(self, current_state_vec, intervention_vec):
        c = np.asarray(current_state_vec, dtype=float)
        i = np.asarray(intervention_vec, dtype=float)
        
        if c.shape != (3,) or i.shape != (3,):
            raise ValueError()
            
        if not np.isclose(np.sum(c), 1.0) or not np.isclose(np.sum(i), 1.0):
            raise ValueError()

        res = np.einsum('ijk,i,j->k', self.T, c, i)
        return res / np.sum(res)