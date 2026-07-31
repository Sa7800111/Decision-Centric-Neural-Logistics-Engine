import numpy as np

class DecisionSmoother:
    def __init__(self, window_size=5):
        self.window = int(window_size)
        self.history = []

    def smooth_decision(self, new_z):
        self.history.append(np.asarray(new_z))
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def compute_volatility(self):
        if len(self.history) < 2:
            return 0.0
        diffs = np.diff(self.history, axis=0)
        return float(np.mean(np.linalg.norm(diffs, axis=1)))