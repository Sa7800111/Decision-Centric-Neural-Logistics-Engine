import numpy as np

class AbductionInferenceEngine:
    def __init__(self, prior_u_distributions, mechanisms):
        if not isinstance(prior_u_distributions, dict) or not isinstance(mechanisms, dict):
            raise TypeError()
        self.priors = prior_u_distributions
        self.mechanisms = mechanisms

    def infer_u(self, evidence_dict):
        if not isinstance(evidence_dict, dict):
            raise TypeError()
            
        inferred_u = {}
        for node, mechanism in self.mechanisms.items():
            if node in evidence_dict:
                try:
                    u_val = mechanism.inverse(evidence_dict)
                    inferred_u[node] = u_val
                except AttributeError:
                    inferred_u[node] = self.priors.get(node, lambda: 0.0)()
            else:
                inferred_u[node] = self.priors.get(node, lambda: 0.0)()
        return inferred_u