import numpy as np

class DiscreteArgmaxSmoother:
    def __init__(self, temperature=1.0):
        self.temp = float(temperature)

    def soft_argmax(self, scores):
        s = np.asarray(scores)
        e_s = np.exp(s / self.temp)
        return e_s / np.sum(e_s)

    def log_gradient_estimate(self, scores, selected_idx):
        probs = self.soft_argmax(scores)
        grad = -probs
        grad[selected_idx] += 1.0
        return grad / self.temp