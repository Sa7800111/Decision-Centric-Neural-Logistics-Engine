import numpy as np

class FTOMetaOptimizer:
    def __init__(self, outer_lr=0.001):
        self.lr = outer_lr

    def meta_step(self, weights, grads_list):
        avg_grad = np.mean(grads_list, axis=0)
        return weights - self.lr * avg_grad

    def compute_inner_loss(self, model, batch, solver):
        x, y, ctx = batch
        yp = model.predict(x)
        return solver.compute_regret(y, yp, ctx)