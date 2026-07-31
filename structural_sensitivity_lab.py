import numpy as np

class StructuralSensitivityLab:
    def __init__(self, model_evaluator):
        if not callable(model_evaluator):
            raise TypeError()
        self.evaluator = model_evaluator

    def compute_one_at_a_time_sensitivity(self, base_params, delta=0.01):
        if not isinstance(base_params, dict):
            raise TypeError()
            
        base_score = self.evaluator(base_params)
        sensitivities = {}
        
        for key, val in base_params.items():
            if not isinstance(val, (int, float)):
                continue
                
            perturbed_params = base_params.copy()
            perturbed_params[key] = val * (1 + delta)
            
            new_score = self.evaluator(perturbed_params)
            sensitivity = (new_score - base_score) / (val * delta + 1e-9)
            sensitivities[key] = float(sensitivity)
            
        return sensitivities