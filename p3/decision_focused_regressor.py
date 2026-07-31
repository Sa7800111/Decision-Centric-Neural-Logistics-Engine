import numpy as np

class DecisionFocusedRegressor:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.w = np.random.randn(input_dim, output_dim) * 0.01
        self.lr = lr

    def forward(self, x):
        return np.asarray(x) @ self.w

    def update_with_decision_grad(self, x, decision_grad):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        weight_grad = x.T @ decision_grad.reshape(1, -1)
        self.w -= self.lr * weight_grad
        return self.w