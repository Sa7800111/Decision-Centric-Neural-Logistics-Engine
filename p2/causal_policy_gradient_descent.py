import numpy as np

class CausalPolicyGD:
    def __init__(self, initial_w, lr=0.01):
        self.w = np.asarray(initial_w, dtype=float)
        self.lr = lr

    def update(self, x_batch, rewards, phi_map):
        grads = []
        for x, r in zip(x_batch, rewards):
            phi = phi_map(x)
            logits = phi @ self.w
            prob = 1 / (1 + np.exp(-logits))
            grad = (r * (1 - prob)) * phi
            grads.append(grad)
        
        self.w += self.lr * np.mean(grads, axis=0)
        return self.w