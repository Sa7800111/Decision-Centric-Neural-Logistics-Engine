import networkx as nx

class IdentifiabilityChecker:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.graph = graph.copy()

    def check_markovian(self):
        for u, v, data in self.graph.edges(data=True):
            if data.get('bidirected', False):
                return False
        return True

    def get_c_components(self):
        bidirected_graph = nx.Graph()
        bidirected_graph.add_nodes_from(self.graph.nodes())
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('bidirected', False):
                bidirected_graph.add_edge(u, v)
                
        return list(nx.connected_components(bidirected_graph))