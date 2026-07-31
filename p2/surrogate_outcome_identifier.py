import networkx as nx

class SurrogateIdentifier:
    def __init__(self, graph):
        self.graph = graph

    def is_valid_surrogate(self, treatment, surrogate, outcome):
        if not nx.has_path(self.graph, treatment, surrogate) or not nx.has_path(self.graph, surrogate, outcome):
            return False
            
        g_prime = self.graph.copy()
        g_prime.remove_edges_from(list(g_prime.out_edges(treatment)))
        
        if nx.has_path(g_prime, treatment, outcome):
            paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
            for path in paths:
                if surrogate not in path:
                    return False
        return True

    def find_all_surrogates(self, treatment, outcome):
        potential = set(self.graph.nodes) - {treatment, outcome}
        valid = []
        for node in potential:
            if self.is_valid_surrogate(treatment, node, outcome):
                valid.append(node)
        return valid