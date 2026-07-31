import numpy as np

class TaskGradClipper:
    def __init__(self, threshold=5.0):
        self.limit = float(threshold)

    def apply(self, gradient):
        g = np.asarray(gradient)
        norm = np.linalg.norm(g)
        if norm > self.limit:
            return (g / norm) * self.limit
        return g

    def dynamic_clip(self, gradient, history):
        avg_norm = np.mean([np.linalg.norm(h) for h in history[-10:]]) if history else self.limit
        return self.apply(gradient * (self.limit / (avg_norm + 1e-9)))