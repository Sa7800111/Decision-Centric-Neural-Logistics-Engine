import networkx as nx

class AdjustmentSetOptimizer:
    def __init__(self, graph):
        self.g = graph

    def get_efficient_adjustment_set(self, x, y):
        nodes = set(self.g.nodes) - {x, y}
        parents_x = set(self.g.predecessors(x))
        parents_y = set(self.g.predecessors(y))
        
        candidate = parents_x | (parents_y - nx.descendants(self.g, x))
        
        if self._is_valid(x, y, candidate):
            return list(candidate)
        return None

    def _is_valid(self, x, y, z):
        g_copy = self.g.copy()
        g_copy.remove_edges_from(list(g_copy.out_edges(x)))
        return nx.d_separated(g_copy, {x}, {y}, z)