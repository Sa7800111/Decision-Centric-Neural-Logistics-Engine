import numpy as np

class InventoryPolicyOptimizer:
    def __init__(self, holding_cost, stockout_cost):
        self.h = float(holding_cost)
        self.p = float(stockout_cost)

    def solve_newsvendor(self, demand_forecast):
        demand = np.asarray(demand_forecast)
        if np.any(demand < 0):
            raise ValueError()
            
        critical_ratio = self.p / (self.p + self.h)
        return np.percentile(demand, critical_ratio * 100)

    def calculate_cost(self, order_qty, actual_demand):
        q = float(order_qty)
        d = float(actual_demand)
        return self.h * max(0, q - d) + self.p * max(0, d - q)