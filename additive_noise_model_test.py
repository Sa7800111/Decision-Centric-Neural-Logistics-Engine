import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

class AdditiveNoiseModelTest:
    def __init__(self, x_data, y_data):
        self.x = np.asarray(x_data, dtype=float).reshape(-1, 1)
        self.y = np.asarray(y_data, dtype=float).reshape(-1, 1)

    def _get_residual_score(self, indep, dep):
        model = LinearRegression()
        model.fit(indep, dep)
        preds = model.predict(indep)
        residuals = dep - preds
        return abs(pearsonr(indep.flatten(), residuals.flatten())[0])

    def determine_direction(self):
        score_xy = self._get_residual_score(self.x, self.y)
        score_yx = self._get_residual_score(self.y, self.x)
        
        return {
            "direction": "X->Y" if score_xy < score_yx else "Y->X",
            "scores": {"X->Y": score_xy, "Y->X": score_yx}
        }