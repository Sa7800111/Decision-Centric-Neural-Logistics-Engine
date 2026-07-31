import numpy as np

class FTOLandscapeMapper:
    def __init__(self, loss_engine, grid_size=20):
        self.engine = loss_engine
        self.grid = int(grid_size)

    def map_2d_surface(self, y_true, y_base, context, directions):
        u, v = directions
        alpha = np.linspace(-1, 1, self.grid)
        beta = np.linspace(-1, 1, self.grid)
        surface = np.zeros((self.grid, self.grid))
        
        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                y_perturbed = y_base + a * u + b * v
                surface[i, j] = self.engine.compute_decision_loss(y_true, y_perturbed, context)
        return surface, alpha, beta