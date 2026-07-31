import numpy as np

class FTOEnsembleV2:
    def __init__(self, predictors, weights=None):
        self.models = predictors
        self.w = weights if weights else [1.0/len(predictors)] * len(predictors)

    def predict_weighted(self, x):
        preds = [m.predict(x) for m in self.models]
        return np.average(preds, axis=0, weights=self.w)

    def update_weights_by_regret(self, x, y_true, solver, lr=0.01):
        regrets = []
        for m in self.models:
            yp = m.predict(x)
            regrets.append(solver.compute_regret(y_true, yp))
        
        self.w = self.w * np.exp(-lr * np.array(regrets))
        self.w /= np.sum(self.w)
        return self.w