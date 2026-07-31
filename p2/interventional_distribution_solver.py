import numpy as np

class InterventionalSolver:
    def __init__(self, cpts, graph):
        self.cpts = cpts
        self.g = graph

    def solve_do_x(self, x_node, x_val, target_node):
        mod_cpts = self.cpts.copy()
        mod_cpts[x_node] = np.zeros_like(self.cpts[x_node])
        
        idx = [slice(None)] * mod_cpts[x_node].ndim
        idx[-1] = x_val
        mod_cpts[x_node][tuple(idx)] = 1.0
        
        return self._marginalize(mod_cpts, target_node)

    def _marginalize(self, cpts, target):
        return "Numerical marginalization result"