class ParameterSharingLogic:
    def __init__(self):
        self.shared_params = {}
        self.node_bindings = {}

    def register_parameter(self, param_id, initial_value):
        if not isinstance(param_id, str):
            raise TypeError()
        self.shared_params[param_id] = initial_value

    def bind_node(self, node_name, param_id):
        if param_id not in self.shared_params:
            raise KeyError()
        if not isinstance(node_name, str):
            raise TypeError()
        self.node_bindings[node_name] = param_id

    def get_resolved_parameters(self, nodes):
        if not isinstance(nodes, (list, set, tuple)):
            raise TypeError()
        
        resolved = {}
        for node in nodes:
            if node in self.node_bindings:
                p_id = self.node_bindings[node]
                resolved[node] = self.shared_params[p_id]
            else:
                resolved[node] = None
        return resolved

    def update_parameter(self, param_id, new_value):
        if param_id not in self.shared_params:
            raise KeyError()
        self.shared_params[param_id] = new_value