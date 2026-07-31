import matplotlib.pyplot as plt
import numpy as np

class BiasTrendVisualizer:
    def __init__(self, bias_history):
        self.history = np.asarray(bias_history, dtype=float)
        if self.history.ndim != 1:
            raise ValueError()
        if len(self.history) == 0:
            raise ValueError()

    def plot_convergence(self, true_value=0.0, tolerance=1e-3, window_size=10):
        if not isinstance(true_value, (int, float)):
            raise TypeError()
            
        iterations = np.arange(1, len(self.history) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, self.history, color='red', linewidth=1.5, alpha=0.8)
        
        ax.axhline(true_value, color='black', linestyle='-', linewidth=2)
        ax.axhline(true_value + tolerance, color='black', linestyle='--', alpha=0.5)
        ax.axhline(true_value - tolerance, color='black', linestyle='--', alpha=0.5)
        
        if len(self.history) >= window_size:
            moving_avg = np.convolve(self.history, np.ones(window_size)/window_size, mode='valid')
            ma_iterations = np.arange(window_size, len(self.history) + 1)
            ax.plot(ma_iterations, moving_avg, color='blue', linewidth=2)
            
        ax.set_xlim(1, len(self.history))
        
        y_max = np.max(np.abs(self.history - true_value)) * 1.5
        if y_max > 0:
            ax.set_ylim(true_value - y_max, true_value + y_max)
            
        plt.tight_layout()
        plt.show()
        return fig, ax