import numpy as np

class InterventionalExpectation:
    def __init__(self, causal_model):
        if not hasattr(causal_model, 'sample') or not hasattr(causal_model, 'set_intervention'):
            raise TypeError()
        self.model = causal_model

    def calculate_expectation(self, target, intervention_dict, n_samples=5000):
        if not isinstance(target, str) or not isinstance(intervention_dict, dict):
            raise TypeError()
        
        self.model.reset_interventions()
        for node, val in intervention_dict.items():
            self.model.set_intervention(node, val)
            
        data = self.model.sample(n_samples)
        if target not in data.columns:
            raise KeyError()
            
        mean_val = np.mean(data[target])
        std_err = np.std(data[target]) / np.sqrt(n_samples)
        
        self.model.reset_interventions()
        return float(mean_val), float(std_err)

    def compare_interventions(self, target, int_a, int_b, n_samples=5000):
        mean_a, _ = self.calculate_expectation(target, int_a, n_samples)
        mean_b, _ = self.calculate_expectation(target, int_b, n_samples)
        return float(mean_a - mean_b)