import networkx as nx

class ZIdentifiabilityChecker:
    def __init__(self, graph):
        self.graph = graph

    def check_z_identifiability(self, y, x, z):
        g_x_bar = self.graph.copy()
        g_x_bar.remove_edges_from(list(g_x_bar.in_edges(x)))
        
        g_z_under = self.graph.copy()
        g_z_under.remove_edges_from(list(g_z_under.out_edges(z)))
        
        rule2_check = nx.d_separated(g_x_bar, {y}, {z}, {x})
        return bool(rule2_check)