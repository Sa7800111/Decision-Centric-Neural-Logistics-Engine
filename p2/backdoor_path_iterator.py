import networkx as nx

class BackdoorPathIterator:
    def __init__(self, graph):
        self.g = graph

    def get_all_backdoor_paths(self, x, y):
        undirected = self.g.to_undirected()
        paths = nx.all_simple_paths(undirected, x, y)
        
        backdoor_paths = []
        for path in paths:
            if len(path) > 1 and self.g.has_edge(path[1], x):
                backdoor_paths.append(path)
        return backdoor_paths

    def check_path_blocked(self, path, z_set):
        for i in range(1, len(path) - 1):
            prev, curr, next_node = path[i-1], path[i], path[i+1]
            
            is_collider = self.g.has_edge(prev, curr) and self.g.has_edge(next_node, curr)
            
            if is_collider:
                descendants = nx.descendants(self.g, curr) | {curr}
                if not (descendants & set(z_set)):
                    return True
            else:
                if curr in z_set:
                    return True
        return False