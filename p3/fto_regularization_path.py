import numpy as np

class FTORegularizationPath:
    def __init__(self, model, lambdas):
        self.model = model
        self.lambdas = sorted(lambdas)
        self.coefficients = []

    def compute_path(self, x, y_true, solver):
        for lam in self.lambdas:
            weights = self.model.fit_with_reg(x, y_true, solver, lam)
            self.coefficients.append(weights.copy())
        return self.coefficients

    def get_best_lambda(self, x_val, y_val, solver):
        errors = []
        for coef in self.coefficients:
            self.model.set_weights(coef)
            y_p = self.model.predict(x_val)
            z_p = solver(y_p)
            errors.append(np.dot(y_val, z_p))
        return self.lambdas[np.argmin(errors)]