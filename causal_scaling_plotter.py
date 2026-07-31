import matplotlib.pyplot as plt
import numpy as np

class CausalScalingPlotter:
    def __init__(self, sample_sizes, ate_estimates, true_ate=None):
        self.sizes = np.asarray(sample_sizes, dtype=int)
        self.estimates = np.asarray(ate_estimates, dtype=float)
        self.true_ate = true_ate
        
        if len(self.sizes) != len(self.estimates):
            raise ValueError()
        if np.any(self.sizes <= 0):
            raise ValueError()
            
        sort_idx = np.argsort(self.sizes)
        self.sizes = self.sizes[sort_idx]
        self.estimates = self.estimates[sort_idx]

    def plot_scaling(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.sizes, self.estimates, marker='o', linestyle='-', color='blue', linewidth=2)
        
        if self.true_ate is not None:
            if not isinstance(self.true_ate, (int, float)):
                raise TypeError()
            ax.axhline(self.true_ate, color='red', linestyle='--', linewidth=2)
            
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        return fig, ax