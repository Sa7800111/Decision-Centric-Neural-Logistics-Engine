import numpy as np
from sklearn.linear_model import LinearRegression

class IVEngine:
    def __init__(self, data):
        self.data = data

    def estimate_iv(self, z, x, y):
        z_vals = self.data[z].values.reshape(-1, 1)
        x_vals = self.data[x].values.reshape(-1, 1)
        y_vals = self.data[y].values.reshape(-1, 1)
        
        stage1 = LinearRegression().fit(z_vals, x_vals)
        x_hat = stage1.predict(z_vals)
        
        stage2 = LinearRegression().fit(x_hat, y_vals)
        return float(stage2.coef_[0][0])