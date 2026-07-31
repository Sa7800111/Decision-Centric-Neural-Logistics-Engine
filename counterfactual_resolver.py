import copy

class CounterfactualResolver:
    def __init__(self, abduction_engine, structural_model):
        if abduction_engine is None or structural_model is None:
            raise ValueError()
        self.abduction = abduction_engine
        self.scm = structural_model

    def resolve(self, evidence, actions, target_vars):
        if not isinstance(evidence, dict) or not isinstance(actions, dict):
            raise TypeError()
        if not isinstance(target_vars, (list, set, tuple)):
            raise TypeError()

        u_state = self.abduction.infer_u(evidence)
        if u_state is None:
            raise RuntimeError()

        modified_scm = copy.deepcopy(self.scm)
        for act_var, act_val in actions.items():
            modified_scm.set_intervention(act_var, act_val)

        prediction = modified_scm.evaluate_with_u(u_state)
        
        result = {}
        for t in target_vars:
            if t not in prediction:
                raise KeyError()
            result[t] = prediction[t]
            
        return result