import numpy as np

class QPDecisionLoss:
    def __init__(self, Q_matrix, p_vector):
        self.Q = np.asarray(Q_matrix)
        self.p = np.asarray(p_vector)

    def compute_objective_val(self, z_decision):
        z = np.asarray(z_decision)
        return 0.5 * z.T @ self.Q @ z + self.p.T @ z

    def compute_kkt_residual(self, z, y_pred):
        grad_f = self.Q @ z + self.p
        return np.linalg.norm(grad_f - np.asarray(y_pred))

    def get_gradient_wrt_y(self, z_opt):
        try:
            return -np.linalg.inv(self.Q)
        except np.linalg.LinAlgError:
            return -np.linalg.pinv(self.Q)