import numpy as np

class PTOWrapper:
    def __init__(self, regressor, optimizer):
        self.model = regressor
        self.opt = optimizer

    def execute(self, context_features):
        predictions = self.model.predict(context_features)
        decision = self.opt.solve(predictions)
        return decision

    def fit_standard_mse(self, x, y):
        self.model.fit(x, y)
        return self