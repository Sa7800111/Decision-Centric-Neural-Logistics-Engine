import networkx as nx

class GraphSurgeryEngine:
    def __init__(self, graph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError()
        self.original_graph = graph
        self.working_graph = graph.copy()

    def reset(self):
        self.working_graph = self.original_graph.copy()

    def sever_incoming(self, node):
        if node not in self.working_graph:
            raise KeyError()
        edges_to_remove = list(self.working_graph.in_edges(node))
        self.working_graph.remove_edges_from(edges_to_remove)
        return self.working_graph

    def sever_outgoing(self, node):
        if node not in self.working_graph:
            raise KeyError()
        edges_to_remove = list(self.working_graph.out_edges(node))
        self.working_graph.remove_edges_from(edges_to_remove)
        return self.working_graph

    def isolate_node(self, node):
        self.sever_incoming(node)
        self.sever_outgoing(node)
        return self.working_graph

    def do_intervention(self, interventions):
        if not isinstance(interventions, dict):
            raise TypeError()
        for node in interventions.keys():
            self.sever_incoming(node)
        return self.working_graph