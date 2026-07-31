import numpy as np

class FourierCausalEmbeddings:
    def __init__(self, input_dim, embed_dim, freq_scale=1.0):
        self.w = np.random.normal(0, freq_scale, (embed_dim, input_dim))
        self.b = np.random.uniform(0, 2 * np.pi, embed_dim)

    def embed(self, x, treatment=None):
        x = np.asarray(x)
        proj = x @ self.w.T + self.b
        phi = np.sqrt(2.0 / self.w.shape[0]) * np.cos(proj)
        
        if treatment is not None:
            t = np.asarray(treatment).reshape(-1, 1)
            return np.hstack([phi, t * phi])
        return phi

    def compute_causal_distance(self, x1, x2):
        phi1 = self.embed(x1)
        phi2 = self.embed(x2)
        return np.linalg.norm(phi1 - phi2, axis=1)