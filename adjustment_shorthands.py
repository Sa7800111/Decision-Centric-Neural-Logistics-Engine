class AdjustmentShorthands:
    def __init__(self, valid_sets):
        if not isinstance(valid_sets, (list, set, tuple)):
            raise TypeError()
        self.sets = [set(s) for s in valid_sets]

    def get_minimal(self):
        if not self.sets:
            raise ValueError()
        return min(self.sets, key=len)

    def get_maximal(self):
        if not self.sets:
            raise ValueError()
        return max(self.sets, key=len)

    def has_empty_set(self):
        return any(len(s) == 0 for s in self.sets)

    def filter_by_inclusion(self, required_nodes):
        if not isinstance(required_nodes, (list, set)):
            raise TypeError()
        req = set(required_nodes)
        return [s for s in self.sets if req.issubset(s)]