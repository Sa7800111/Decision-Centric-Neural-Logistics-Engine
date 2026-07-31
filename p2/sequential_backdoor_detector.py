import networkx as nx

class SequentialBackdoorDetector:
    def __init__(self, time_series_graph):
        self.g = time_series_graph

    def is_sequentially_identifiable(self, treatments, outcomes, covariates):
        if len(treatments) != len(covariates):
            raise ValueError()

        for t in range(len(treatments)):
            past_treatments = treatments[:t]
            past_covariates = covariates[:t+1]
            
            g_t_bar = self.g.copy()
            g_t_bar.remove_edges_from(list(g_t_bar.out_edges(treatments[t])))
            
            if not nx.d_separated(g_t_bar, {treatments[t]}, set(outcomes), set(past_treatments) | set(past_covariates)):
                return False
        return True