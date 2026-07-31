import numpy as np

class NNPolicyLearner:
    def __init__(self, input_dim, hidden_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.w2 = np.random.randn(hidden_dim, 1) * 0.01

    def forward(self, x):
        self.z1 = x @ self.w1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.w2
        return 1 / (1 + np.exp(-self.z2))

    def get_action(self, x, threshold=0.5):
        prob = self.forward(x)
        return (prob >= threshold).astype(int)