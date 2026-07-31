import numpy as np
import networkx as nx

class ProbabilisticProgramSampler:
    def __init__(self, trace_graph):
        if not isinstance(trace_graph, nx.DiGraph):
            raise TypeError()
        if not nx.is_directed_acyclic_graph(trace_graph):
            raise ValueError()
        self.graph = trace_graph
        self.samplers = {}

    def register_sampler(self, variable_name, distribution_fn):
        if variable_name not in self.graph.nodes:
            raise KeyError()
        if not callable(distribution_fn):
            raise TypeError()
        self.samplers[variable_name] = distribution_fn

    def forward_sample(self, n_traces=1):
        if not self.samplers:
            raise RuntimeError()

        missing = set(self.graph.nodes) - set(self.samplers.keys())
        if missing:
            raise ValueError()

        order = list(nx.topological_sort(self.graph))
        traces = []

        for _ in range(n_traces):
            current_trace = {}
            for node in order:
                parents = list(self.graph.predecessors(node))
                parent_values = {p: current_trace[p] for p in parents}
                
                try:
                    current_trace[node] = self.samplers[node](parent_values)
                except Exception as e:
                    raise RuntimeError() from e
                    
            traces.append(current_trace)

        return traces

    def rejection_sample(self, condition_fn, n_accepted):
        accepted_traces = []
        attempts = 0
        max_attempts = n_accepted * 1000

        while len(accepted_traces) < n_accepted and attempts < max_attempts:
            trace = self.forward_sample(n_traces=1)[0]
            attempts += 1
            
            if condition_fn(trace):
                accepted_traces.append(trace)

        if len(accepted_traces) < n_accepted:
            raise RuntimeError()

        return accepted_traces