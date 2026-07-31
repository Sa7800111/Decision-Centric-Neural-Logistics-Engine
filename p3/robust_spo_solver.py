import numpy as np

class RobustSPOSolver:
    def __init__(self, solver, uncertainty_set):
        self.solver = solver
        self.u_set = uncertainty_set

    def solve_robust(self, y_pred):
        y_p = np.asarray(y_pred)
        worst_case_cost = -float('inf')
        best_decision = None
        
        for delta in self.u_set:
            y_perturbed = y_p + delta
            decision = self.solver(y_perturbed)
            cost = np.dot(y_perturbed, decision)
            if cost > worst_case_cost:
                worst_case_cost = cost
                best_decision = decision
        return best_decision