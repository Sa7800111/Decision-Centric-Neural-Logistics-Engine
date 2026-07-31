import numpy as np

class CausalSensitivityBounds:
    def __init__(self, observed_ate, selection_bias_param):
        self.ate = float(observed_ate)
        self.gamma = float(selection_bias_param)
        if self.gamma < 1.0:
            raise ValueError()

    def compute_manski_bounds(self, y_min, y_max):
        lower = self.ate / self.gamma - (1 - 1/self.gamma) * (y_max - y_min)
        upper = self.ate * self.gamma + (self.gamma - 1) * (y_max - y_min)
        return float(lower), float(upper)

    def find_break_even_gamma(self, null_hypothesis=0.0):
        if self.ate == null_hypothesis:
            return 1.0
        return abs(self.ate / (self.ate - null_hypothesis))