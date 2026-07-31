import numpy as np

class NTKGuidedSampling:
    def __init__(self, ntk_matrix):
        self.k = ntk_matrix

    def get_representative_indices(self, n_select):
        n = self.k.shape[0]
        selected = []
        remaining = list(range(n))
        
        first = np.argmax(np.diag(self.k))
        selected.append(first)
        remaining.remove(first)
        
        for _ in range(n_select - 1):
            sub_k = self.k[remaining][:, selected]
            scores = np.min(sub_k, axis=1)
            next_idx = remaining[np.argmin(scores)]
            selected.append(next_idx)
            remaining.remove(next_idx)
            
        return selected