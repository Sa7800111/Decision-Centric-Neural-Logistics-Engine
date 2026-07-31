class ComplexityEquivalenceTester:
    def __init__(self, model_a, model_b):
        if not isinstance(model_a, dict) or not isinstance(model_b, dict):
            raise TypeError()
        self.ma = model_a
        self.mb = model_b

    def get_parameter_count(self, model):
        count = 0
        for node, parents in model.items():
            if not isinstance(parents, (list, set, tuple)):
                raise TypeError()
            count += 2 ** len(parents)
        return count

    def is_equivalent_complexity(self):
        ca = self.get_parameter_count(self.ma)
        cb = self.get_parameter_count(self.mb)
        return ca == cb