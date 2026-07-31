import numpy as np

class RegretBoundEstimator:
    def __init__(self, diameter, strong_convexity_constant):
        self.D = float(diameter)
        self.sigma = float(strong_convexity_constant)

    def compute_upper_bound(self, prediction_error):
        err = float(prediction_error)
        return (self.D / self.sigma) * err

    def compute_lipschitz_regret(self, l_constant, prediction_error):
        return l_constant * float(prediction_error)