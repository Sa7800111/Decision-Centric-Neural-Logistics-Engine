import networkx as nx

class CausalIdentifiabilityLogic:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.graph = graph

    def is_identifiable_via_backdoor(self, treatment, outcome):
        if treatment not in self.graph or outcome not in self.graph:
            raise KeyError()
            
        nodes = set(self.graph.nodes) - {treatment, outcome}
        descendants_x = nx.descendants(self.graph, treatment)
        candidate_nodes = list(nodes - descendants_x)
        
        from itertools import combinations
        for i in range(len(candidate_nodes) + 1):
            for subset in combinations(candidate_nodes, i):
                if self._verify_set(treatment, outcome, set(subset)):
                    return True, list(subset)
        return False, []

    def _verify_set(self, x, y, z):
        g_copy = self.graph.copy()
        g_copy.remove_edges_from(list(g_copy.out_edges(x)))
        moral = nx.moral_graph(g_copy)
        for node in z:
            if node in moral:
                moral.remove_node(node)
        return not nx.has_path(moral, x, y)