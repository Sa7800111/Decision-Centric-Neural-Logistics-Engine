import numpy as np

class ShorthandVsTwinAnalysis:
    def __init__(self, shorthand_engine, twin_engine):
        if not hasattr(shorthand_engine, 'estimate_ate_linear'):
            raise TypeError()
        if not hasattr(twin_engine, 'simulate_twin_world'):
            raise TypeError()
        self.shorthand = shorthand_engine
        self.twin = twin_engine

    def compare_effects(self, data, treatment, outcome, intervention_val, n_samples):
        if not isinstance(intervention_val, (int, float)) or not isinstance(n_samples, int):
            raise TypeError()
        if n_samples <= 0:
            raise ValueError()

        adj_sets = self.shorthand.find_valid_adjustment_sets(treatment, outcome)
        if not adj_sets:
            raise ValueError()
            
        optimal_set = min(adj_sets, key=len)
        shorthand_ate = self.shorthand.estimate_ate_linear(data, treatment, outcome, optimal_set)

        self.twin.build_twin_network({treatment: intervention_val})
        
        noise_gen = {
            node: lambda n: np.random.normal(0, 1, n) 
            for node in self.twin.base_scm.graph.nodes
        }
        
        twin_data = self.twin.simulate_twin_world(n_samples, noise_gen)
        twin_cf_effect = float(np.mean(twin_data[f"{outcome}_star"]) - np.mean(twin_data[outcome]))

        return {
            'shorthand_estimate': float(shorthand_ate),
            'twin_network_estimate': twin_cf_effect,
            'absolute_difference': abs(float(shorthand_ate) - twin_cf_effect)
        }