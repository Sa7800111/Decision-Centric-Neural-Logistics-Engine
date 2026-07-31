import networkx as nx

class CComponentDecomposer:
    def __init__(self, graph, bidirected_edges):
        self.g = graph
        self.confounded_graph = nx.Graph()
        self.confounded_graph.add_nodes_from(graph.nodes)
        self.confounded_graph.add_edges_from(bidirected_edges)

    def get_c_components(self):
        return [list(c) for c in nx.connected_components(self.confounded_graph)]

    def factorize_joint(self, p_joint):
        components = self.get_c_components()
        factors = []
        for c in components:
            factors.append(f"Q[{','.join(c)}]")
        return " * ".join(factors)