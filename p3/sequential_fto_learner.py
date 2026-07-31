
import numpy as np

class SequentialFTOLearner:
    def __init__(self, model, optimizer):
        self.model = model
        self.opt = optimizer
        self.decision_history = []

    def update(self, x, y_true):
        y_pred = self.model.forward(x)
        z_pred = self.opt.solve(y_pred)
        self.decision_history.append(z_pred)
        
        z_true = self.opt.solve(y_true)
        grad = z_pred - z_true
        self.model.backward(grad)