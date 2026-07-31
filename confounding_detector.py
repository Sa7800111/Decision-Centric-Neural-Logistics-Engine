import networkx as nx

class ConfoundingDetector:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.graph = graph

    def get_unobserved_confounders(self):
        unobserved = [n for n, attr in self.graph.nodes(data=True) if not attr.get('observed', True)]
        confounders = []
        for u in unobserved:
            children = list(self.graph.successors(u))
            if len(children) > 1:
                confounders.append(u)
        return confounders

    def is_confounded(self, x, y):
        if x not in self.graph or y not in self.graph:
            raise KeyError()
        
        undirected = self.graph.to_undirected()
        paths = list(nx.all_simple_paths(undirected, x, y))
        
        for path in paths:
            if len(path) > 2:
                if self.graph.has_edge(path[1], x):
                    return True
        return False

    def filter_minimal_sets(self, sets):
        if not isinstance(sets, list):
            raise TypeError()
        minimal = []
        for s in sorted(sets, key=len):
            if not any(m.issubset(s) for m in minimal):
                minimal.append(s)
        return minimal