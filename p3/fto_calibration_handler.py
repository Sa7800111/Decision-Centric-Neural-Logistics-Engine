import numpy as np

class FTOCalibrationHandler:
    def __init__(self, target_variance):
        self.target = float(target_variance)
        self.scale_factor = 1.0

    def calibrate_forecasts(self, y_pred):
        y = np.asarray(y_pred)
        current_var = np.var(y)
        self.scale_factor = np.sqrt(self.target / (current_var + 1e-9))
        return y * self.scale_factor

    def get_calibration_bias(self, y_true, y_pred):
        return float(np.mean(y_true) - np.mean(y_pred))