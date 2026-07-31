import numpy as np

class FTOGradientStabilizer:
    def __init__(self, clip_value=1.0, decay=0.99):
        self.clip = float(clip_value)
        self.decay = float(decay)
        self.grad_history = None

    def stabilize(self, raw_grad):
        g = np.asarray(raw_grad, dtype=float)
        if self.grad_history is None:
            self.grad_history = np.zeros_like(g)
        
        self.grad_history = self.decay * self.grad_history + (1 - self.decay) * g
        norm = np.linalg.norm(self.grad_history)
        
        if norm > self.clip:
            return (self.grad_history / norm) * self.clip
        return self.grad_history

    def reset(self):
        self.grad_history = None