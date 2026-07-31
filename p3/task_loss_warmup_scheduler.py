import numpy as np

class TaskLossWarmup:
    def __init__(self, total_steps, start_alpha=0.0, end_alpha=1.0):
        self.total = total_steps
        self.start = float(start_alpha)
        self.end = float(end_alpha)
        self.current_step = 0

    def get_alpha(self):
        progress = min(1.0, self.current_step / self.total)
        alpha = self.start + progress * (self.end - self.start)
        self.current_step += 1
        return float(alpha)

    def hybrid_loss(self, mse, decision_loss):
        a = self.get_alpha()
        return (1 - a) * mse + a * decision_loss