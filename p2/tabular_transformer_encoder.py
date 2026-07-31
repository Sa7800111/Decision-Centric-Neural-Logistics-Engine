import numpy as np

class TabularTransformerEncoder:
    def __init__(self, embed_dim, n_heads):
        self.d = embed_dim
        self.h = n_heads
        self.wq = np.random.randn(embed_dim, embed_dim) * 0.01
        self.wk = np.random.randn(embed_dim, embed_dim) * 0.01
        self.wv = np.random.randn(embed_dim, embed_dim) * 0.01

    def scaled_dot_product_attention(self, q, k, v):
        dk = q.shape[-1]
        scores = (q @ k.T) / np.sqrt(dk)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return weights @ v

    def forward(self, x):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        return self.scaled_dot_product_attention(q, k, v)