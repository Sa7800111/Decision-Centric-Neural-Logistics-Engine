import networkx as nx

class DomainAdaptationCausality:
    def __init__(self, selection_diagram):
        if not isinstance(selection_diagram, nx.DiGraph):
            raise TypeError()
        self.diagram = selection_diagram
        self.selection_nodes = [n for n in self.diagram.nodes if str(n).startswith('S_')]

    def is_transportable(self, target, source, adjustment_set):
        if target not in self.diagram or source not in self.diagram:
            raise KeyError()
        if not isinstance(adjustment_set, set):
            raise TypeError()

        modified_graph = self.diagram.copy()
        modified_graph.remove_edges_from(list(modified_graph.in_edges(source)))
        
        moral = nx.moral_graph(modified_graph)
        for node in adjustment_set:
            if node in moral:
                moral.remove_node(node)
                
        for s_node in self.selection_nodes:
            if s_node in moral and target in moral:
                if nx.has_path(moral, s_node, target):
                    return False
        return True