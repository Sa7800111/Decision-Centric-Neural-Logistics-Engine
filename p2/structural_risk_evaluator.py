import numpy as np

class StructuralRiskEvaluator:
    def __init__(self, lambda_reg=0.1):
        self.l = float(lambda_reg)

    def compute_risk(self, y_true, y_pred, model_complexity):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mse = np.mean((y_true - y_pred)**2)
        return float(mse + self.l * model_complexity)

    def compare_models(self, risk_list):
        return int(np.argmin(risk_list))