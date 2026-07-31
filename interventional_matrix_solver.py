import numpy as np

class InterventionalMatrixSolver:
    def __init__(self, weight_matrix, node_names):
        self.W = np.asarray(weight_matrix, dtype=float)
        self.nodes = list(node_names)
        if self.W.shape[0] != self.W.shape[1] or self.W.shape[0] != len(self.nodes):
            raise ValueError()
        self.I = np.eye(self.W.shape[0])

    def solve_interventional_system(self, intervention_nodes, values):
        if len(intervention_nodes) != len(values):
            raise ValueError()
            
        W_do = self.W.copy()
        for node in intervention_nodes:
            idx = self.nodes.index(node)
            W_do[idx, :] = 0
            
        try:
            total_impact = np.linalg.inv(self.I - W_do)
        except np.linalg.LinAlgError:
            total_impact = np.linalg.pinv(self.I - W_do)
            
        v_ext = np.zeros(len(self.nodes))
        for node, val in zip(intervention_nodes, values):
            v_ext[self.nodes.index(node)] = val
            
        return total_impact @ v_ext