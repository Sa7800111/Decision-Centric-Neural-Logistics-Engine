import numpy as np

class StructuralNestedModel:
    def __init__(self, phi_param):
        self.phi = phi_param

    def predict_counterfactual_y(self, obs_y, obs_t, obs_l):
        return obs_y - self.phi * obs_t * obs_l

    def average_treatment_effect(self, data_l):
        return self.phi * np.mean(data_l)