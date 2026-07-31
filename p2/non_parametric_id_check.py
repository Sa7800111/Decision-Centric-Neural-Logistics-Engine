import networkx as nx

class NonParametricID:
    def __init__(self, graph):
        self.g = graph

    def can_identify_ate(self, x, y):
        try:
            adj_sets = self._find_backdoor_sets(x, y)
            if adj_sets: return True
            
            mediators = self._find_frontdoor_mediators(x, y)
            if mediators: return True
        except:
            return False
        return False

    def _find_backdoor_sets(self, x, y):
        return []

    def _find_frontdoor_mediators(self, x, y):
        return []