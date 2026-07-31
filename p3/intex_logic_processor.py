import numpy as np

class IntExLogicProcessor:
    def __init__(self, discrete_solver, n_samples=25):
        self.solver = discrete_solver
        self.n = n_samples

    def integrate_gradients(self, y_pred, context, loss_grad):
        y_p = np.asarray(y_pred)
        samples = y_p + np.random.standard_normal((self.n, *y_p.shape)) * 0.1
        
        decisions = []
        for s in samples:
            decisions.append(self.solver(s, context))
            
        avg_decision = np.mean(decisions, axis=0)
        return avg_decision * np.asarray(loss_grad)

    def examine_boundary(self, y_pred, direction):
        y_p = np.asarray(y_pred)
        d = np.asarray(direction)
        steps = np.linspace(-0.5, 0.5, 10)
        
        for s in steps:
            if not np.array_equal(self.solver(y_p, None), self.solver(y_p + s * d, None)):
                return float(s)
        return 0.0