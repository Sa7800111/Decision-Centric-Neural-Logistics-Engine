import numpy as np

class TabularEmbeddingLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.normal(0, 0.1, (input_dim, output_dim))

    def forward(self, x_indices):
        indices = np.asarray(x_indices, dtype=int)
        return self.weights[indices]

    def update(self, indices, gradients, lr=0.01):
        np.add.at(self.weights, indices, -lr * gradients)