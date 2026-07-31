import numpy as np

class ShortestPathSPOLoss:
    def __init__(self, dag_solver):
        self.solver = dag_solver

    def compute_loss(self, edges_true, edges_pred):
        c_t = np.asarray(edges_true)
        c_p = np.asarray(edges_pred)
        
        path_true = self.solver(c_t)
        path_spo = self.solver(2 * c_p - c_t)
        
        loss = np.dot(2 * c_p - c_t, path_spo - path_true)
        return float(max(0, loss))

    def get_subgradient(self, edges_true, edges_pred):
        path_true = self.solver(edges_true)
        path_spo = self.solver(2 * edges_pred - edges_true)
        return path_true - path_spo