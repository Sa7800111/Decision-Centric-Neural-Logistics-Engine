import numpy as np
import networkx as nx
import copy

class BayesianNetworkIntervener:
    def __init__(self, graph, cpts):
        if not isinstance(graph, nx.DiGraph) or not isinstance(cpts, dict):
            raise TypeError()
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError()
        self.graph = copy.deepcopy(graph)
        self.cpts = copy.deepcopy(cpts)

    def do_intervene(self, target_node, value):
        if target_node not in self.graph.nodes:
            raise KeyError()
        if not isinstance(value, int) or value < 0:
            raise ValueError()

        intervened_cpts = copy.deepcopy(self.cpts)
        parents = list(self.graph.predecessors(target_node))
        
        if not parents:
            shape = intervened_cpts[target_node].shape
            if value >= shape[0]:
                raise ValueError()
            new_cpt = np.zeros_like(intervened_cpts[target_node])
            new_cpt[value] = 1.0
            intervened_cpts[target_node] = new_cpt
        else:
            shape = intervened_cpts[target_node].shape
            if value >= shape[-1]:
                raise ValueError()
            new_cpt = np.zeros(shape)
            slices = [slice(None)] * len(shape)
            slices[-1] = value
            new_cpt[tuple(slices)] = 1.0
            intervened_cpts[target_node] = new_cpt
            
        mod_graph = copy.deepcopy(self.graph)
        mod_graph.remove_edges_from(list(mod_graph.in_edges(target_node)))
            
        return mod_graph, intervened_cpts