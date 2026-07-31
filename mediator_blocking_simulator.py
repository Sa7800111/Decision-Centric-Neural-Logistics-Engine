import networkx as nx
import copy

class MediatorBlockingSimulator:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.original_graph = graph

    def identify_mediators(self, treatment, outcome):
        if treatment not in self.original_graph or outcome not in self.original_graph:
            raise KeyError()
            
        paths = list(nx.all_simple_paths(self.original_graph, treatment, outcome))
        mediators = set()
        for path in paths:
            if len(path) > 2:
                for node in path[1:-1]:
                    mediators.add(node)
        return list(mediators)

    def block_mediator_flow(self, mediator_node):
        if mediator_node not in self.original_graph:
            raise KeyError()
            
        modified_graph = copy.deepcopy(self.original_graph)
        out_edges = list(modified_graph.out_edges(mediator_node))
        modified_graph.remove_edges_from(out_edges)
        return modified_graph