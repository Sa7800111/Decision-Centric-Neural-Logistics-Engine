class TransportabilityMapper:
    def __init__(self, source_graph, target_graph):
        self.s_g = source_graph
        self.t_g = target_graph

    def identify_selection_nodes(self):
        s_nodes = []
        for node in self.s_g.nodes:
            if self.s_g.nodes[node] != self.t_g.nodes.get(node):
                s_nodes.append(f"S_{node}")
        return s_nodes

    def can_transport(self, x, y, s_nodes):
        import networkx as nx
        aug_graph = self.t_g.copy()
        for s in s_nodes:
            target = s.split('_')[1]
            aug_graph.add_edge(s, target)
            
        g_x_bar = aug_graph.copy()
        g_x_bar.remove_edges_from(list(g_x_bar.in_edges(x)))
        
        return nx.d_separated(g_x_bar, set(s_nodes), {y}, {x})