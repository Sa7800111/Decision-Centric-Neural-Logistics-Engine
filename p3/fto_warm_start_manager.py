import numpy as np

class FTOWarmStartManager:
    def __init__(self, capacity=100):
        self.buffer = []
        self.capacity = capacity

    def add_solution(self, cost_vec, sol_vec):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((np.asarray(cost_vec), np.asarray(sol_vec)))

    def find_nearest_solution(self, new_cost_vec):
        if not self.buffer:
            return None
            
        costs = np.array([b[0] for b in self.buffer])
        dists = np.linalg.norm(costs - np.asarray(new_cost_vec), axis=1)
        return self.buffer[np.argmin(dists)][1]