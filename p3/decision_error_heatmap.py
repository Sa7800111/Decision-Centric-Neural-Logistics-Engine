import numpy as np

class DecisionErrorHeatmap:
    def __init__(self, solver, resolution=10):
        self.solver = solver
        self.res = resolution

    def generate_map(self, y_true, dim1_range, dim2_range):
        d1 = np.linspace(*dim1_range, self.res)
        d2 = np.linspace(*dim2_range, self.res)
        grid = np.zeros((self.res, self.res))
        
        z_opt = self.solver(y_true)
        for i, v1 in enumerate(d1):
            for j, v2 in enumerate(d2):
                y_curr = np.array([v1, v2])
                z_curr = self.solver(y_curr)
                grid[i, j] = np.dot(y_true, z_curr) - np.dot(y_true, z_opt)
        return grid