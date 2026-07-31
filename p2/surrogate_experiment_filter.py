import networkx as nx

class SurrogateExperimentFilter:
    def __init__(self, graph):
        self.g = graph

    def can_use_surrogate(self, x, y, z):
        g_z_bar = self.g.copy()
        g_z_bar.remove_edges_from(list(g_z_bar.in_edges(z)))
        
        condition1 = nx.d_separated(g_z_bar, {x}, {y}, {z})
        
        g_x_bar_z_under = self.g.copy()
        g_x_bar_z_under.remove_edges_from(list(g_x_bar_z_under.in_edges(x)))
        g_x_bar_z_under.remove_edges_from(list(g_x_bar_z_under.out_edges(z)))
        
        condition2 = nx.d_separated(g_x_bar_z_under, {z}, {y}, {x})
        
        return bool(condition1 and condition2)