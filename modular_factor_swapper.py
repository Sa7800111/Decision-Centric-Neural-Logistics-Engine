import copy

class ModularFactorSwapper:
    def __init__(self, model_a, model_b):
        if not hasattr(model_a, 'mechanisms') or not hasattr(model_b, 'mechanisms'):
            raise TypeError()
        self.model_a = model_a
        self.model_b = model_b

    def swap_local_mechanism(self, node_name):
        if node_name not in self.model_a.mechanisms or node_name not in self.model_b.mechanisms:
            raise KeyError()
            
        temp_mech = copy.deepcopy(self.model_a.mechanisms[node_name])
        self.model_a.mechanisms[node_name] = copy.deepcopy(self.model_b.mechanisms[node_name])
        self.model_b.mechanisms[node_name] = temp_mech
        
        return True

    def hybridize_models(self, nodes_from_b):
        new_model = copy.deepcopy(self.model_a)
        for node in nodes_from_b:
            if node in self.model_b.mechanisms:
                new_model.mechanisms[node] = copy.deepcopy(self.model_b.mechanisms[node])
        return new_model