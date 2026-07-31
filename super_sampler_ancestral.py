import networkx as nx
import numpy as np

class SuperSamplerAncestral:
    def __init__(self, graph):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError()
        self.graph = graph

    def sample(self, mechanisms, n_samples=1000):
        data = {}
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            if not parents:
                data[node] = mechanisms[node](n_samples)
            else:
                parent_data = {p: data[p] for p in parents}
                data[node] = mechanisms[node](parent_data, n_samples)
        return data