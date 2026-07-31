import networkx as nx

class IDAlgorithmCore:
    def __init__(self, graph):
        self.g = graph

    def identify(self, y_set, x_set):
        if not x_set:
            return "P(" + ",".join(y_set) + ")"
            
        anc_y = nx.ancestors(self.g, tuple(y_set)) | set(y_set)
        g_anc = self.g.subgraph(anc_y).copy()
        
        if not (set(g_anc.nodes) - set(y_set) - set(x_set)):
            return "Summation over V \ (Y U X)"
            
        return "Recursive Step Required"