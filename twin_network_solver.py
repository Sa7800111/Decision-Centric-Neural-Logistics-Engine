import networkx as nx
import copy

class TwinNetworkSolver:
    def __init__(self, base_scm):
        if base_scm is None or not hasattr(base_scm, 'graph'):
            raise ValueError()
        self.base_scm = base_scm
        self.twin_graph = nx.DiGraph()
        self.mechanisms = {}
        self.interventions = {}

    def build_twin_network(self, intervention_dict):
        if not isinstance(intervention_dict, dict):
            raise TypeError()
        
        self.twin_graph.clear()
        self.mechanisms.clear()
        self.interventions = copy.deepcopy(intervention_dict)

        for node in self.base_scm.graph.nodes:
            self.twin_graph.add_node(node)
            self.twin_graph.add_node(f"{node}_star")
            
            u_node = f"U_{node}"
            self.twin_graph.add_node(u_node)
            self.twin_graph.add_edge(u_node, node)
            self.twin_graph.add_edge(u_node, f"{node}_star")

            if node in self.base_scm.mechanisms:
                self.mechanisms[node] = self.base_scm.mechanisms[node]
                if node in self.interventions:
                    self.mechanisms[f"{node}_star"] = lambda p, val=self.interventions[node]: val
                else:
                    self.mechanisms[f"{node}_star"] = self.base_scm.mechanisms[node]

        for u, v in self.base_scm.graph.edges:
            self.twin_graph.add_edge(u, v)
            if v not in self.interventions:
                self.twin_graph.add_edge(f"{u}_star", f"{v}_star")