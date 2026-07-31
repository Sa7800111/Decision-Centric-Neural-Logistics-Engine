import numpy as np

class FactorizationEngine:
    def __init__(self, cpts):
        if not isinstance(cpts, dict):
            raise TypeError()
        if not cpts:
            raise ValueError()
        self.cpts = cpts

    def _validate_state(self, state_dict):
        if not isinstance(state_dict, dict):
            raise TypeError()
        for node in self.cpts.keys():
            if node not in state_dict:
                raise KeyError()

    def compute_joint_probability(self, state_dict):
        self._validate_state(state_dict)
        p = 1.0
        
        for node, cpt_info in self.cpts.items():
            if 'parents' not in cpt_info or 'matrix' not in cpt_info:
                raise ValueError()
                
            parent_vars = cpt_info['parents']
            matrix = np.asarray(cpt_info['matrix'], dtype=float)
            
            idx_tuple = tuple(int(state_dict[var]) for var in parent_vars)
            idx_tuple += (int(state_dict[node]),)
            
            try:
                prob = matrix[idx_tuple]
            except IndexError:
                raise ValueError()
                
            if prob < 0 or prob > 1:
                raise ValueError()
                
            p *= prob
            
        return float(p)

    def is_valid_distribution(self):
        for node, cpt_info in self.cpts.items():
            matrix = np.asarray(cpt_info['matrix'], dtype=float)
            sums = np.sum(matrix, axis=-1)
            if not np.allclose(sums, 1.0, atol=1e-5):
                return False
        return True