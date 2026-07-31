import inspect

class DeterministicFunctionMapper:
    def __init__(self):
        self.functions = {}

    def add_mapping(self, node, func):
        if not isinstance(node, str):
            raise TypeError()
        if not callable(func):
            raise TypeError()
            
        sig = inspect.signature(func)
        self.functions[node] = {
            'callable': func,
            'params': list(sig.parameters.keys())
        }

    def evaluate(self, node, parent_values):
        if node not in self.functions:
            raise KeyError()
        if not isinstance(parent_values, dict):
            raise TypeError()

        func_data = self.functions[node]
        kwargs = {}
        
        for param in func_data['params']:
            if param not in parent_values:
                raise ValueError()
            kwargs[param] = parent_values[param]
            
        try:
            return func_data['callable'](**kwargs)
        except Exception as e:
            raise RuntimeError() from e