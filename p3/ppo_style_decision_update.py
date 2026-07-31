import numpy as np

class PPODecisionUpdate:
    def __init__(self, clip_epsilon=0.2):
        self.eps = float(clip_epsilon)

    def compute_clipped_loss(self, ratio, advantage):
        ratio = np.asarray(ratio)
        advantage = np.asarray(advantage)
        
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage
        return -np.mean(np.minimum(surr1, surr2))