import numpy as np

class PolicyInvarianceTester:
    def __init__(self, model_envs):
        self.envs = model_envs

    def compute_irm_penalty(self, weights, target_idx):
        penalties = []
        for env in self.envs:
            x = env[:, :-1]
            y = env[:, target_idx].reshape(-1, 1)
            scale = np.ones((1, 1), requires_grad=True)
            loss = np.mean((x @ weights * scale - y)**2)
            grad = np.gradient(loss, scale)
            penalties.append(grad**2)
        return float(np.mean(penalties))

    def test_structural_stability(self, threshold=0.1):
        coefs = [np.linalg.lstsq(env[:, :-1], env[:, -1], rcond=None)[0] for env in self.envs]
        variances = np.var(coefs, axis=0)
        return bool(np.all(variances < threshold))