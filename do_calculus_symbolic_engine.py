import networkx as nx

class DoCalculusSymbolicEngine:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.graph = graph

    def _d_separated(self, y, z, given, graph):
        moral_graph = nx.moral_graph(graph)
        for node in given:
            if node in moral_graph:
                moral_graph.remove_node(node)
                
        return not any(
            nx.has_path(moral_graph, z_node, y_node) 
            for z_node in z for y_node in y 
            if z_node in moral_graph and y_node in moral_graph
        )

    def rule_1_insertion_deletion(self, y, x, z, w):
        if not all(isinstance(i, list) for i in [y, x, z, w]):
            raise TypeError()
            
        g_x_bar = self.graph.copy()
        for node in x:
            if node in g_x_bar:
                g_x_bar.remove_edges_from(list(g_x_bar.in_edges(node)))
                
        return self._d_separated(y, z, x + w, g_x_bar)

    def rule_2_action_observation(self, y, x, z, w):
        if not all(isinstance(i, list) for i in [y, x, z, w]):
            raise TypeError()
            
        g_x_bar_z_underbar = self.graph.copy()
        for node in x:
            if node in g_x_bar_z_underbar:
                g_x_bar_z_underbar.remove_edges_from(list(g_x_bar_z_underbar.in_edges(node)))
        for node in z:
            if node in g_x_bar_z_underbar:
                g_x_bar_z_underbar.remove_edges_from(list(g_x_bar_z_underbar.out_edges(node)))
                
        return self._d_separated(y, z, x + w, g_x_bar_z_underbar)