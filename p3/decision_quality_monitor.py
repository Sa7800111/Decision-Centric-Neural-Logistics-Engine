import numpy as np

class DecisionQualityMonitor:
    def __init__(self):
        self.history = []

    def update(self, regret, feasibility_gap):
        self.history.append({
            "regret": float(regret),
            "gap": float(feasibility_gap)
        })

    def get_rolling_stats(self, window=50):
        if not self.history:
            return {}
        recent = self.history[-window:]
        return {
            "avg_regret": np.mean([h["regret"] for h in recent]),
            "max_gap": np.max([h["gap"] for h in recent]),
            "reliability_score": np.mean([h["gap"] < 1e-6 for h in recent])
        }