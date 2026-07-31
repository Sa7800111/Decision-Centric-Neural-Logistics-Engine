import numpy as np

class LogLinearTransform:
    def __init__(self, data, epsilon=1e-9):
        self.data = np.asarray(data, dtype=float)
        if self.data.size == 0:
            raise ValueError()
        if epsilon <= 0:
            raise ValueError()
        self.epsilon = epsilon

    def to_log_space(self):
        if np.any(self.data < 0):
            raise ValueError()
        return np.log(self.data + self.epsilon)

    def to_linear_space(self, log_data):
        ld = np.asarray(log_data, dtype=float)
        return np.maximum(np.exp(ld) - self.epsilon, 0.0)