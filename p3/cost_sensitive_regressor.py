import numpy as np

class CostSensitiveRegressor:
    def __init__(self, cost_matrix):
        self.costs = np.asarray(cost_matrix, dtype=float)
        self.w = None

    def fit(self, x, y, lr=0.01, epochs=100):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.w = np.random.randn(x.shape[1], y.shape[1]) * 0.1
        
        for _ in range(epochs):
            preds = x @ self.w
            errors = preds - y
            weighted_grad = x.T @ (errors @ self.costs)
            self.w -= lr * (weighted_grad / len(x))
        return self.w

    def predict(self, x):
        return np.asarray(x) @ self.w