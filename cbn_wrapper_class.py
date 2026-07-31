import networkx as nx
import copy

class CausalBayesianNetwork:
    def __init__(self, dag, cpts):
        if not isinstance(dag, nx.DiGraph):
            raise TypeError()
        if not isinstance(cpts, dict):
            raise TypeError()
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError()
            
        self.dag = copy.deepcopy(dag)
        self.cpts = copy.deepcopy(cpts)
        
        for node in self.dag.nodes:
            if node not in self.cpts:
                raise KeyError()

    def get_cpt(self, node):
        if node not in self.cpts:
            raise KeyError()
        return self.cpts[node]

    def set_cpt(self, node, new_cpt):
        if node not in self.dag.nodes:
            raise KeyError()
        if not isinstance(new_cpt, dict):
            raise TypeError()
        self.cpts[node] = copy.deepcopy(new_cpt)

    def do_operator(self, target, value):
        if target not in self.dag.nodes:
            raise KeyError()
        if not isinstance(value, (int, float)):
            raise TypeError()
            
        in_edges = list(self.dag.in_edges(target))
        self.dag.remove_edges_from(in_edges)
        
        self.cpts[target] = {'do_val': value}
        return self