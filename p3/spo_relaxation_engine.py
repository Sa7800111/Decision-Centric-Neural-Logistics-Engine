import numpy as np

class SPORelaxationEngine:
    def __init__(self, poly_vertices):
        self.vertices = np.asarray(poly_vertices)

    def find_best_vertex(self, cost_vec):
        c = np.asarray(cost_vec)
        scores = self.vertices @ c
        return self.vertices[np.argmin(scores)]

    def compute_spo_loss(self, c_true, c_pred):
        v_true = self.find_best_vertex(c_true)
        v_spo = self.find_best_vertex(2 * np.asarray(c_pred) - np.asarray(c_true))
        return float(np.dot(c_true, v_spo - v_true))