import networkx as nx
from itertools import combinations

class BackdoorCriterionResolver:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError()
        self.graph = graph

    def is_valid_adjustment_set(self, x, y, z):
        if not isinstance(z, (list, set)):
            raise TypeError()
        if x in z or y in z:
            return False
        
        descendants_x = nx.descendants(self.graph, x)
        if any(node in descendants_x for node in z):
            return False

        manipulated_graph = self.graph.copy()
        manipulated_graph.remove_edges_from(list(manipulated_graph.out_edges(x)))

        moral_graph = nx.moral_graph(manipulated_graph)
        for node in z:
            if node in moral_graph:
                moral_graph.remove_node(node)

        if x in moral_graph and y in moral_graph:
            return not nx.has_path(moral_graph, x, y)
        return True

    def find_all_sets(self, x, y):
        if x not in self.graph or y not in self.graph:
            raise KeyError()
            
        nodes = set(self.graph.nodes) - {x, y}
        descendants_x = nx.descendants(self.graph, x)
        valid_nodes = list(nodes - descendants_x)
        
        valid_sets = []
        for i in range(len(valid_nodes) + 1):
            for subset in combinations(valid_nodes, i):
                if self.is_valid_adjustment_set(x, y, set(subset)):
                    valid_sets.append(set(subset))
        return valid_sets