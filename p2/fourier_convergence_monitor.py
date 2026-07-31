import numpy as np

class FourierConvergenceMonitor:
    def __init__(self, tolerance=1e-5):
        self.tol = float(tolerance)
        self.history = []

    def add_step(self, weights):
        self.history.append(np.asarray(weights).copy())

    def check_stability(self, window=5):
        if len(self.history) < window:
            return False
        
        recent = self.history[-window:]
        diffs = [np.linalg.norm(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        return bool(np.mean(diffs) < self.tol)

    def compute_learning_speed(self):
        if len(self.history) < 2:
            return 0.0
        return float(np.linalg.norm(self.history[-1] - self.history[0]) / len(self.history))