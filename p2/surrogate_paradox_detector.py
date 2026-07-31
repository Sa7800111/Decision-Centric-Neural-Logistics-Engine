import numpy as np

class SurrogateParadoxDetector:
    def __init__(self, ate_s, ate_y_given_s):
        self.ate_s = float(ate_s)
        self.ate_y_s = float(ate_y_given_s)

    def is_paradoxical(self, direct_effect_y):
        total_effect = self.ate_s * self.ate_y_s + direct_effect_y
        
        if np.sign(self.ate_s) == np.sign(total_effect):
            return False
        return True

    def calculate_min_direct_effect_to_flip(self):
        return -(self.ate_s * self.ate_y_s)