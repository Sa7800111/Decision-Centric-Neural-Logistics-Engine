import numpy as np

class DistributionShiftResolver:
    def __init__(self, source_data, target_data):
        self.source = np.asarray(source_data, dtype=float)
        self.target = np.asarray(target_data, dtype=float)
        if self.source.ndim != self.target.ndim:
            raise ValueError()

    def compute_importance_weights(self, bins=20):
        s_hist, edges = np.histogram(self.source, bins=bins, density=True)
        t_hist, _ = np.histogram(self.target, bins=edges, density=True)
        
        weights = np.divide(t_hist, s_hist, out=np.zeros_like(t_hist), where=s_hist!=0)
        return weights, edges

    def resample_source(self, weights, edges):
        indices = np.digitize(self.source, edges) - 1
        indices = np.clip(indices, 0, len(weights) - 1)
        
        sample_probs = weights[indices]
        sample_probs /= np.sum(sample_probs)
        
        resampled_indices = np.random.choice(
            len(self.source), 
            size=len(self.source), 
            p=sample_probs
        )
        return self.source[resampled_indices]