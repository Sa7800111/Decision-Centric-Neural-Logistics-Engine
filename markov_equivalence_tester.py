import networkx as nx
from itertools import combinations

class MarkovEquivalenceTester:
    def __init__(self, graph1, graph2):
        if not isinstance(graph1, nx.DiGraph) or not isinstance(graph2, nx.DiGraph):
            raise TypeError()
        self.g1 = graph1
        self.g2 = graph2

    def _get_skeleton(self, graph):
        return graph.to_undirected()

    def _get_v_structures(self, graph):
        v_structures = set()
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            if len(parents) >= 2:
                for p1, p2 in combinations(parents, 2):
                    if not graph.has_edge(p1, p2) and not graph.has_edge(p2, p1):
                        v_structures.add((frozenset([p1, p2]), node))
        return v_structures

    def test_equivalence(self):
        skel1 = self._get_skeleton(self.g1)
        skel2 = self._get_skeleton(self.g2)
        if not nx.is_isomorphic(skel1, skel2):
            return False

        v1 = self._get_v_structures(self.g1)
        v2 = self._get_v_structures(self.g2)
        return v1 == v2