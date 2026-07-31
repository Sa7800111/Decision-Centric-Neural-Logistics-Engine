import numpy as np
from scipy.spatial import ConvexHull

class ConvexHullDifferentiator:
    def __init__(self, points):
        self.points = np.asarray(points)
        self.hull = ConvexHull(self.points)

    def project_to_hull(self, query_point):
        from scipy.optimize import minimize
        q = np.asarray(query_point)
        
        def objective(weights):
            p = weights @ self.points[self.hull.vertices]
            return np.linalg.norm(p - q)**2
            
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bnds = [(0, 1) for _ in range(len(self.hull.vertices))]
        
        res = minimize(objective, x0=np.ones(len(self.hull.vertices))/len(self.hull.vertices),
                       bounds=bnds, constraints=cons)
        return res.x @ self.points[self.hull.vertices]