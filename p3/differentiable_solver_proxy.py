import numpy as np

class DifferentiableSolverProxy:
    def __init__(self, eps=1e-4):
        self.eps = float(eps)

    def surrogate_gradient(self, y_pred, context, loss_func):
        y_p = np.asarray(y_pred, dtype=float)
        grad = np.zeros_like(y_p)
        
        for i in range(len(y_p)):
            y_plus = y_p.copy()
            y_plus[i] += self.eps
            y_minus = y_p.copy()
            y_minus[i] -= self.eps
            
            l_plus = loss_func(y_plus, context)
            l_minus = loss_func(y_minus, context)
            grad[i] = (l_plus - l_minus) / (2 * self.eps)
            
        return grad