import numpy as np

class StructuralPolicyOptimizer:
    def __init__(self, causal_model, fourier_engine):
        self.model = causal_model
        self.fe = fourier_engine

    def evaluate_policy_candidate(self, weight_vector, n_samples=500):
        w = np.asarray(weight_vector)
        def policy(x):
            phi = self.fe.map(x)
            return 1 if (phi @ w) > 0 else 0
            
        self.model.reset_interventions()
        data = self.model.sample(n_samples)
        outcomes = []
        for i in range(len(data)):
            action = policy(data.iloc[i].values)
            self.model.set_intervention('T', action)
            res = self.model.sample(1)
            outcomes.append(res['Y'].iloc[0])
            
        return float(np.mean(outcomes))

    def optimize_step(self, current_w, step_size=0.01):
        grad = np.random.normal(0, 1, len(current_w))
        new_w = current_w + step_size * grad
        return new_w / np.linalg.norm(new_w)