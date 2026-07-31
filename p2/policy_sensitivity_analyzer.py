import numpy as np

class PolicySensitivityAnalyzer:
    def __init__(self, estimator_func):
        self.estimator = estimator_func

    def perturb_params(self, base_params, variance=0.1, n_iterations=100):
        results = []
        for _ in range(n_iterations):
            noisy_params = {k: v + np.random.normal(0, variance) for k, v in base_params.items()}
            results.append(self.estimator(noisy_params))
        
        return {
            "mean_effect": float(np.mean(results)),
            "std_dev": float(np.std(results)),
            "ci_lower": float(np.percentile(results, 2.5)),
            "ci_upper": float(np.percentile(results, 97.5))
        }