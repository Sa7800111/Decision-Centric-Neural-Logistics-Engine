import networkx as nx

class TransportabilityIDEngine:
    def __init__(self, selection_diagram):
        if not isinstance(selection_diagram, nx.DiGraph):
            raise TypeError()
        self.g = selection_diagram
        self.s_nodes = [n for n in self.g.nodes if str(n).startswith('S_')]

    def is_locally_transportable(self, y, x):
        g_x_bar = self.g.copy()
        g_x_bar.remove_edges_from(list(g_x_bar.in_edges(x)))
        
        for s in self.s_nodes:
            if not nx.d_separated(g_x_bar, {s}, {y}, {x}):
                return False
        return True

    def get_transport_formula(self, y, x):
        if self.is_locally_transportable(y, x):
            return f"P*({y}|do({x})) = P({y}|do({x}))"
        return "Complex transport required"