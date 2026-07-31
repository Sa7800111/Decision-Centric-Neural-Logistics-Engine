import numpy as np

class FTOGradientInterpolator:
    def __init__(self, sample_points):
        self.points = np.asarray(sample_points)

    def interpolate_grad(self, query_y, gradients):
        from scipy.interpolate import Rbf
        y = np.asarray(query_y).flatten()
        
        interpolators = []
        for i in range(gradients.shape[1]):
            rbf = Rbf(*self.points.T, gradients[:, i], function='multiquadric')
            interpolators.append(rbf)
            
        return np.array([r(*y) for r in interpolators])

    def get_extrapolation_risk(self, query_y):
        dist = np.min(np.linalg.norm(self.points - query_y, axis=1))
        return float(dist)