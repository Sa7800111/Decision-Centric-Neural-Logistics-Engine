import numpy as np

class VAEAncestralSampler:
    def __init__(self, decoder_model, latent_dim):
        self.decoder = decoder_model
        self.latent_dim = int(latent_dim)

    def sample_latent_space(self, n_samples, mean=0.0, std=1.0):
        if n_samples <= 0:
            raise ValueError()
        return np.random.normal(mean, std, (n_samples, self.latent_dim))

    def generate_ancestral_data(self, n_samples):
        z = self.sample_latent_space(n_samples)
        try:
            generated_data = self.decoder.predict(z)
        except AttributeError:
            generated_data = np.dot(z, np.random.randn(self.latent_dim, 10))
            
        return generated_data