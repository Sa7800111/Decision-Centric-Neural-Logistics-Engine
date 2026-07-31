import copy

class InterventionalMapping:
    def __init__(self, base_scm):
        if base_scm is None:
            raise ValueError()
        self.scm = base_scm
        self.registry = {}

    def register_policy(self, name, interventions):
        if not isinstance(name, str) or not isinstance(interventions, dict):
            raise TypeError()
        if not interventions:
            raise ValueError()
        self.registry[name] = interventions

    def apply_policy(self, name):
        if name not in self.registry:
            raise KeyError()
            
        mod_scm = copy.deepcopy(self.scm)
        for node, val in self.registry[name].items():
            if node not in mod_scm.graph.nodes:
                raise ValueError()
            mod_scm.set_intervention(node, val)
            
        return mod_scm