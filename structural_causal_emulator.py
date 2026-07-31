import networkx as nx
import numpy as np
import pandas as pd

class StructuralCausalModel:
    def __init__(self, graph=None):
        self.graph = graph if graph is not None else nx.DiGraph()
        self.mechanisms = {}
        self.noise_distributions = {}
        self.interventions = {}
        self.is_fit = False

    def add_endogenous_var(self, name, parents, mechanism):
        for parent in parents:
            self.graph.add_edge(parent, name)
        self.mechanisms[name] = mechanism
        if name not in self.graph:
            self.graph.add_node(name)

    def add_exogenous_var(self, name, distribution):
        self.graph.add_node(name)
        self.noise_distributions[name] = distribution

    def _get_topological_order(self):
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains cycles")

    def set_intervention(self, node, value):
        if node not in self.graph.nodes:
            raise KeyError(node)
        self.interventions[node] = value

    def reset_interventions(self):
        self.interventions.clear()

    def sample(self, n_samples=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        data = pd.DataFrame(index=range(n_samples))
        order = self._get_topological_order()

        for node in order:
            if node in self.interventions:
                if callable(self.interventions[node]):
                    data[node] = self.interventions[node](n_samples)
                else:
                    data[node] = self.interventions[node]
                continue

            if node in self.noise_distributions:
                data[node] = self.noise_distributions[node](n_samples)
            else:
                parents = list(self.graph.predecessors(node))
                if not parents:
                    data[node] = np.zeros(n_samples)
                    continue
                    
                parent_data = data[parents].to_dict(orient='list')
                parent_data = {k: np.array(v) for k, v in parent_data.items()}
                
                if node not in self.mechanisms:
                    raise RuntimeError(node)
                    
                data[node] = self.mechanisms[node](parent_data)

        return data