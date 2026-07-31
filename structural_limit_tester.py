import numpy as np

class StructuralLimitTester:
    def __init__(self, scm, bounds_dict):
        if scm is None or not bounds_dict:
            raise ValueError()
        self.scm = scm
        self.bounds = bounds_dict

        for node, limits in self.bounds.items():
            if len(limits) != 2 or limits[0] >= limits[1]:
                raise ValueError()

    def test_extreme_interventions(self, target_node, outcome_node, n_samples=1000):
        if target_node not in self.bounds:
            raise KeyError()
        if target_node not in self.scm.graph.nodes or outcome_node not in self.scm.graph.nodes:
            raise ValueError()

        lower, upper = self.bounds[target_node]
        
        self.scm.reset_interventions()
        self.scm.set_intervention(target_node, lower)
        res_lower = self.scm.sample(n_samples)
        
        self.scm.reset_interventions()
        self.scm.set_intervention(target_node, upper)
        res_upper = self.scm.sample(n_samples)
        
        self.scm.reset_interventions()

        if outcome_node not in res_lower.columns or outcome_node not in res_upper.columns:
            raise RuntimeError()

        return {
            'lower_limit_effect': float(np.mean(res_lower[outcome_node])),
            'upper_limit_effect': float(np.mean(res_upper[outcome_node]))
        }