import networkx as nx
import numpy as np

class FrontdoorResolver:
    def __init__(self, graph):
        self.g = graph

    def find_mediators(self, x, y):
        paths = list(nx.all_simple_paths(self.g, x, y))
        mediators = set()
        for path in paths:
            if len(path) == 3:
                mediators.add(path[1])
        return list(mediators)

    def verify_frontdoor(self, x, y, z):
        if not isinstance(z, list): z = [z]
        
        for node in z:
            paths_x_z = list(nx.all_simple_paths(self.g, x, node))
            for p in paths_x_z:
                if any(n in self.g.predecessors(x) for n in p):
                    return False
                    
        g_x_removed = self.g.copy()
        g_x_removed.remove_node(x)
        for node in z:
            if nx.has_path(g_x_removed, node, y):
                backdoor_paths = [p for p in nx.all_simple_paths(self.g.to_undirected(), node, y) if self.g.has_edge(self.g.predecessors(node), node)]
                if not backdoor_paths: return True
        return False