import sympy as sp

class AnalyticalExpectationSolver:
    def __init__(self, pdf_expr, domain_bounds, variable_name):
        self.var = sp.Symbol(variable_name)
        
        try:
            self.pdf = sp.sympify(pdf_expr)
        except sp.SympifyError:
            raise ValueError()
            
        if len(domain_bounds) != 2:
            raise ValueError()
        self.lower, self.upper = domain_bounds

        integral_check = sp.integrate(self.pdf, (self.var, self.lower, self.upper))
        if integral_check == 0:
            raise ValueError()

    def compute_expectation(self, function_expr):
        try:
            func = sp.sympify(function_expr)
        except sp.SympifyError:
            raise ValueError()
            
        integrand = func * self.pdf
        result = sp.integrate(integrand, (self.var, self.lower, self.upper))
        
        if result.has(sp.Integral):
            raise RuntimeError()
            
        return sp.simplify(result)

    def compute_variance(self):
        ex = self.compute_expectation(self.var)
        ex2 = self.compute_expectation(self.var**2)
        return sp.simplify(ex2 - ex**2)