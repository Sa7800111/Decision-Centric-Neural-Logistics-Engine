import numpy as np

class NonIdentifiabilityDemo:
    def __init__(self, model_a_generator, model_b_generator):
        if not callable(model_a_generator) or not callable(model_b_generator):
            raise TypeError()
        self.model_a = model_a_generator
        self.model_b = model_b_generator

    def generate_observational_equivalence(self, n_samples):
        if n_samples <= 0:
            raise ValueError()
            
        data_a = np.asarray(self.model_a(n_samples), dtype=float)
        data_b = np.asarray(self.model_b(n_samples), dtype=float)
        
        if data_a.shape != data_b.shape:
            raise ValueError()
            
        mean_diff = np.abs(np.mean(data_a) - np.mean(data_b))
        var_diff = np.abs(np.var(data_a) - np.var(data_b))
        
        return mean_diff < 1e-2 and var_diff < 1e-2

    def demonstrate_interventional_divergence(self, intervention_val):
        res_a = self.model_a(1000, do_x=intervention_val)
        res_b = self.model_b(1000, do_x=intervention_val)
        
        diff = np.abs(np.mean(res_a) - np.mean(res_b))
        return diff > 0.5, float(diff)