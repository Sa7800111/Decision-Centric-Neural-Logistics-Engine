import numpy as np

class StructuralStabilityMetrics:
    def __init__(self, model_list):
        self.models = model_list

    def compute_weight_drift(self):
        drifts = []
        base_w = self.models[0].weights.flatten()
        for m in self.models[1:]:
            drifts.append(np.linalg.norm(m.weights.flatten() - base_w))
        return np.array(drifts)

    def compute_interventional_variance(self, x_test):
        preds = [m.predict(x_test).flatten() for m in self.models]
        return np.var(preds, axis=0)

    def is_structurally_invariant(self, x_test, tol=0.05):
        v = self.compute_interventional_variance(x_test)
        return bool(np.mean(v) < tol)