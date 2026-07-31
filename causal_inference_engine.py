import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

class CausalInferenceEngine:
    def __init__(self, scm):
        self.scm = scm
        self.graph = scm.graph

    def _get_descendants(self, node):
        return set(nx.descendants(self.graph, node))

    def check_backdoor_criterion(self, treatment, outcome, adjustment_set):
        if treatment in adjustment_set or outcome in adjustment_set:
            return False
            
        descendants_x = self._get_descendants(treatment)
        if any(z in descendants_x for z in adjustment_set):
            return False

        manipulated_graph = self.graph.copy()
        manipulated_graph.remove_edges_from(list(manipulated_graph.out_edges(treatment)))

        moral_graph = nx.moral_graph(manipulated_graph)
        for node in adjustment_set:
            if node in moral_graph:
                moral_graph.remove_node(node)

        return not nx.has_path(moral_graph, treatment, outcome)

    def find_valid_adjustment_sets(self, treatment, outcome):
        valid_sets = []
        nodes = set(self.graph.nodes) - {treatment, outcome}
        descendants_x = self._get_descendants(treatment)
        candidate_nodes = list(nodes - descendants_x)

        for i in range(len(candidate_nodes) + 1):
            for subset in combinations(candidate_nodes, i):
                if self.check_backdoor_criterion(treatment, outcome, set(subset)):
                    valid_sets.append(set(subset))
                    
        return valid_sets

    def estimate_ate_linear(self, data, treatment, outcome, adjustment_set):
        df = pd.DataFrame(data)
        
        if treatment not in df.columns or outcome not in df.columns:
            raise ValueError("Missing columns")

        if not adjustment_set:
            cov = np.cov(df[treatment], df[outcome])
            var = np.var(df[treatment])
            if var == 0:
                return 0.0
            return cov[0, 1] / var

        features = list(adjustment_set) + [treatment]
        X = df[features].values
        X = np.c_[np.ones(X.shape[0]), X] 
        y = df[outcome].values

        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            return beta[-1] 
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular matrix")

    def evaluate_policy(self, data, treatment, outcome):
        sets = self.find_valid_adjustment_sets(treatment, outcome)
        if not sets:
            raise ValueError("Unidentifiable")
            
        minimal_set = min(sets, key=len)
        return self.estimate_ate_linear(data, treatment, outcome, minimal_set)