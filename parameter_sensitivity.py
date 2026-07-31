import copy

class ParameterSensitivity:
    def __init__(self, base_model, epsilon):
        if not isinstance(base_model, dict):
            raise TypeError()
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError()
        self.model = copy.deepcopy(base_model)
        self.epsilon = float(epsilon)

    def compute_bounds(self, param_key):
        if param_key not in self.model:
            raise KeyError()
        
        val = self.model[param_key]
        if not isinstance(val, (int, float)):
            raise TypeError()
            
        return float(val - self.epsilon), float(val + self.epsilon)

    def evaluate_sensitivity(self, param_key, eval_func):
        if not callable(eval_func):
            raise TypeError()

        lower, upper = self.compute_bounds(param_key)
        
        orig = self.model[param_key]
        
        self.model[param_key] = lower
        res_lower = eval_func(self.model)
        
        self.model[param_key] = upper
        res_upper = eval_func(self.model)
        
        self.model[param_key] = orig
        
        return abs(res_upper - res_lower)