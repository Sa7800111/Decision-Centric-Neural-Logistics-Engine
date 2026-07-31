import numpy as np

class FTODataAugmentor:
    def __init__(self, solver, noise_std=0.05):
        self.solver = solver
        self.sigma = float(noise_std)

    def generate_adversarial_scenarios(self, y_true, context, n_samples=10):
        y_t = np.asarray(y_true)
        z_opt = self.solver(y_t, context)
        
        augmented_set = []
        for _ in range(n_samples):
            y_noise = y_t + np.random.normal(0, self.sigma, y_t.shape)
            z_noise = self.solver(y_noise, context)
            
            if not np.allclose(z_opt, z_noise, atol=1e-5):
                augmented_set.append((y_noise, z_noise))
                
        return augmented_set

    def compute_mixing_ratio(self, original_batch, augmented_batch):
        return len(augmented_batch) / (len(original_batch) + 1e-9)