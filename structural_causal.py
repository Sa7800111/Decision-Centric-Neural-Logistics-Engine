import networkx as nx
import numpy as np

class StructuralCausalEmulator:
    def __init__(self, nodes, edges):
        self.g = nx.DiGraph()
        self.g.add_nodes_from(nodes)
        self.g.add_edges_from(edges)

    def emulate_system_response(self, input_vector, mechanism_weights):
        ordered_nodes = list(nx.topological_sort(self.g))
        values = {}
        
        for node in ordered_nodes:
            parents = list(self.g.predecessors(node))
            if not parents:
                values[node] = input_vector.get(node, 0.0)
            else:
                weighted_sum = sum(values[p] * mechanism_weights.get((p, node), 1.0) for p in parents)
                values[node] = weighted_sum + np.random.normal(0, 0.1)
                
        return values