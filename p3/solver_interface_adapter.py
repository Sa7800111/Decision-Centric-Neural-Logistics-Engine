import numpy as np

class SolverInterfaceAdapter:
    def __init__(self, external_solver_fn):
        self.solver = external_solver_fn
        self.last_input = None
        self.last_output = None

    def solve_and_track(self, cost_vector, constraints):
        c = np.asarray(cost_vector).flatten()
        self.last_input = c
        try:
            res = self.solver(c, constraints)
            self.last_output = np.asarray(res)
            return self.last_output
        except Exception:
            return np.zeros_like(c)

    def get_cache(self):
        return self.last_input, self.last_output