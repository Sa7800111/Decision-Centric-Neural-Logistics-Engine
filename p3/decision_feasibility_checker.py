import numpy as np

class DecisionFeasibilityChecker:
    def __init__(self, constraints, tolerances=1e-6):
        self.constraints = constraints
        self.tol = float(tolerances)

    def check_all(self, z_decision):
        results = {}
        for name, func in self.constraints.items():
            val = func(z_decision)
            results[name] = {
                "value": float(val),
                "is_satisfied": bool(val <= self.tol)
            }
        return results

    def get_violation_magnitude(self, z_decision):
        res = self.check_all(z_decision)
        return sum(max(0, d["value"]) for d in res.values())