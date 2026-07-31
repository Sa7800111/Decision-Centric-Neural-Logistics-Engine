import copy

class CounterfactualGenerator:
    def __init__(self, scm, abduction_engine):
        if scm is None or abduction_engine is None:
            raise ValueError()
        self.scm = scm
        self.abduction = abduction_engine

    def generate(self, factual_evidence, action_dict):
        if not isinstance(factual_evidence, dict) or not isinstance(action_dict, dict):
            raise TypeError()
        if not factual_evidence or not action_dict:
            raise ValueError()

        inferred_u = self.abduction.infer_u(factual_evidence)
        if not inferred_u:
            raise RuntimeError()

        modified_scm = copy.deepcopy(self.scm)
        for target, val in action_dict.items():
            modified_scm.set_intervention(target, val)

        try:
            cf_state = modified_scm.evaluate_state(inferred_u)
        except AttributeError:
            raise RuntimeError()
            
        return cf_state