import copy

class SoftInterventionEngine:
    def __init__(self, scm):
        if not hasattr(scm, 'mechanisms'):
            raise TypeError()
        self.base_scm = scm

    def apply_stochastic_intervention(self, node, distribution_func):
        if not callable(distribution_func):
            raise TypeError()
        if node not in self.base_scm.graph.nodes:
            raise KeyError()
            
        modified_scm = copy.deepcopy(self.base_scm)
        modified_scm.mechanisms[node] = lambda parents, n_samples=1: distribution_func(n_samples)
        return modified_scm

    def apply_additive_shift(self, node, shift_magnitude):
        if node not in self.base_scm.mechanisms:
            raise KeyError()
            
        original_mech = self.base_scm.mechanisms[node]
        modified_scm = copy.deepcopy(self.base_scm)
        
        def shifted_mech(parents):
            return original_mech(parents) + shift_magnitude
            
        modified_scm.mechanisms[node] = shifted_mech
        return modified_scm