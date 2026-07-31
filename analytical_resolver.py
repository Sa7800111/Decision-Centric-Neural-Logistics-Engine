import sympy as sp

class AnalyticalResolver:
    def __init__(self):
        self.symbols = {}
        self.equations = []

    def define_symbol(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError()
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name)
        return self.symbols[name]

    def add_equation(self, lhs_name, rhs_expr):
        if not isinstance(lhs_name, str):
            raise TypeError()
        lhs_sym = self.symbols.get(lhs_name, self.define_symbol(lhs_name))
        
        if not isinstance(rhs_expr, sp.Expr):
            try:
                rhs_expr = sp.sympify(rhs_expr)
            except sp.SympifyError:
                raise ValueError()
                
        eq = sp.Eq(lhs_sym, rhs_expr)
        self.equations.append(eq)

    def solve_system(self, target_vars):
        if not target_vars:
            raise ValueError()
            
        targets = []
        for v in target_vars:
            if v not in self.symbols:
                raise KeyError()
            targets.append(self.symbols[v])
            
        solutions = sp.solve(self.equations, targets, dict=True)
        if not solutions:
            raise RuntimeError()
        return solutions