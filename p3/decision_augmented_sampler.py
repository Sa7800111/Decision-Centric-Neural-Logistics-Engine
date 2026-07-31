import numpy as np

class DecisionAugmentedSampler:
    def __init__(self, solver, noise_level=0.05):
        self.solver = solver
        self.sigma = float(noise_level)

    def generate_critical_scenarios(self, y_base, context, n_samples=10):
        y_b = np.asarray(y_base)
        z_base = self.solver(y_b, context)
        
        augmented_data = []
        for _ in range(n_samples):
            y_noisy = y_b + np.random.normal(0, self.sigma, y_b.shape)
            z_noisy = self.solver(y_noisy, context)
            
            if not np.allclose(z_base, z_noisy, atol=1e-4):
                augmented_data.append((y_noisy, z_noisy))
                
        return augmented_data