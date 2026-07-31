import networkx as nx

class BNIndependenceChecker:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError()
        self.graph = graph

    def is_d_separated(self, x, y, given=None):
        if given is None:
            given = set()
        elif not isinstance(given, (list, set, tuple)):
            raise TypeError()
        else:
            given = set(given)

        if x not in self.graph or y not in self.graph:
            raise KeyError()
        for node in given:
            if node not in self.graph:
                raise KeyError()

        return nx.d_separated(self.graph, {x}, {y}, given)