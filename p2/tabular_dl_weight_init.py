import numpy as np

class TabularWeightInit:
    def __init__(self, input_dim, output_dim):
        self.d_in = input_dim
        self.d_out = output_dim

    def ntk_scaling_init(self):
        return np.random.normal(0, 1.0, (self.d_in, self.d_out)) / np.sqrt(self.d_in)

    def fourier_aware_init(self, freq_matrix):
        w = np.random.normal(0, 1.0, (self.d_in, self.d_out))
        projection = w @ freq_matrix.T
        return projection / (np.linalg.norm(projection, axis=0) + 1e-9)