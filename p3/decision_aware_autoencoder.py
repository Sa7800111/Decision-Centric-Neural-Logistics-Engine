import numpy as np

class DecisionAwareAutoencoder:
    def __init__(self, encoder, decoder, solver):
        self.enc = encoder
        self.dec = decoder
        self.solver = solver

    def reconstruct_and_solve(self, x, context):
        latent = self.enc(x)
        y_recon = self.dec(latent)
        decision = self.solver(y_recon, context)
        return y_recon, decision

    def total_loss(self, x, y_true, context, alpha=0.5):
        y_p, z_p = self.reconstruct_and_solve(x, context)
        recon_loss = np.mean((y_true - y_p)**2)
        decision_loss = np.dot(y_true, self.solver(y_true, context)) - np.dot(y_true, z_p)
        return (1 - alpha) * recon_loss + alpha * max(0, decision_loss)