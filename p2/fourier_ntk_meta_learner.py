import numpy as np

class FourierNTKMetaLearner:
    def __init__(self, task_dims, n_components=128):
        self.d = task_dims
        self.m = n_components
        self.meta_w = np.random.normal(0, 1.0, (self.m, self.d))
        self.meta_b = np.random.uniform(0, 2 * np.pi, self.m)

    def get_task_features(self, x):
        proj = np.asarray(x) @ self.meta_w.T + self.meta_b
        return np.sqrt(2.0 / self.m) * np.cos(proj)

    def adapt_to_task(self, x_train, y_train, lam=0.1):
        phi = self.get_task_features(x_train)
        n, p = phi.shape
        w_task = np.linalg.solve(phi.T @ phi + lam * np.eye(p), phi.T @ y_train)
        return w_task

    def meta_update_weights(self, grads, lr=1e-3):
        self.meta_w -= lr * grads