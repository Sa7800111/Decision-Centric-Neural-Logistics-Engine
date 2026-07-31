class PolicySimulator:
    def __init__(self, base_price, base_demand, elasticity):
        if base_price <= 0 or base_demand < 0:
            raise ValueError()
        if elasticity > 0:
            raise ValueError()
        self.p0 = float(base_price)
        self.q0 = float(base_demand)
        self.e = float(elasticity)

    def apply_tax(self, tax_rate):
        if tax_rate < 0:
            raise ValueError()
            
        new_price = self.p0 * (1 + tax_rate)
        price_ratio = new_price / self.p0
        
        new_demand = self.q0 * (price_ratio ** self.e)
        revenue = (new_price - self.p0) * new_demand
        
        return {
            'new_price': float(new_price),
            'new_demand': float(max(0.0, new_demand)),
            'tax_revenue': float(max(0.0, revenue))
        }