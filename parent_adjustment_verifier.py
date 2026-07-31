import networkx as nx

class ParentAdjustmentVerifier:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.graph = graph

    def verify_parent_set(self, treatment, outcome, proposed_parents):
        if treatment not in self.graph or outcome not in self.graph:
            raise KeyError()
            
        actual_parents = set(self.graph.predecessors(treatment))
        proposed_set = set(proposed_parents)
        
        if not proposed_set.issubset(set(self.graph.nodes)):
            raise ValueError()

        is_complete = actual_parents.issubset(proposed_set)
        
        descendants_x = nx.descendants(self.graph, treatment)
        contains_descendant = any(node in descendants_x for node in proposed_set)
        
        return {
            "is_complete_parent_set": bool(is_complete),
            "contains_descendants": bool(contains_descendant),
            "is_valid_adjustment": bool(is_complete and not contains_descendant)
        }