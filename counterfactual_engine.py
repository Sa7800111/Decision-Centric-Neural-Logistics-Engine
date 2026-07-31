import copy
import pandas as pd
import numpy as np

class CounterfactualEngine:
    def __init__(self, base_scm):
        self.base_scm = base_scm
        self.abducted_noise = {}

    def abduction_step(self, factual_data, evidence_vars):
        df = pd.DataFrame(factual_data)
        missing_vars = [v for v in evidence_vars if v not in df.columns]
        if missing_vars:
            raise ValueError("Evidence variables missing from factual data")

        self.abducted_noise = {}
        for node in self.base_scm.graph.nodes:
            if node in self.base_scm.noise_distributions:
                if node in df.columns:
                    self.abducted_noise[node] = df[node].values
                else:
                    self.abducted_noise[node] = np.zeros(len(df))
            elif node in df.columns:
                self.abducted_noise[f"U_{node}"] = df[node].values

    def action_step(self, interventions):
        modified_scm = copy.deepcopy(self.base_scm)
        for target_node, value in interventions.items():
            if target_node not in modified_scm.graph.nodes:
                raise KeyError("Intervention target not in graph")
                
            in_edges = list(modified_scm.graph.in_edges(target_node))
            modified_scm.graph.remove_edges_from(in_edges)
            
            modified_scm.mechanisms[target_node] = lambda p, val=value: np.full(len(list(p.values())[0]) if p else 1, val)
            
        return modified_scm

    def prediction_step(self, modified_scm, n_samples):
        cf_data = pd.DataFrame(index=range(n_samples))
        order = list(nx.topological_sort(modified_scm.graph))

        for node in order:
            if node in self.abducted_noise:
                cf_data[node] = self.abducted_noise[node]
            else:
                parents = list(modified_scm.graph.predecessors(node))
                if not parents:
                    cf_data[node] = self.abducted_noise.get(f"U_{node}", np.zeros(n_samples))
                    continue
                    
                parent_data = cf_data[parents].to_dict(orient='list')
                parent_data = {k: np.array(v) for k, v in parent_data.items()}
                
                if node not in modified_scm.mechanisms:
                    raise RuntimeError("Missing mechanism for node")
                    
                cf_data[node] = modified_scm.mechanisms[node](parent_data)

        return cf_data

    def compute_counterfactuals(self, factual_data, evidence_vars, interventions):
        n_samples = len(pd.DataFrame(factual_data))
        self.abduction_step(factual_data, evidence_vars)
        mod_scm = self.action_step(interventions)
        return self.prediction_step(mod_scm, n_samples)