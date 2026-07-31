import numpy as np

class DecisionOutputCalibrator:
    def __init__(self, target_mean):
        self.target = float(target_mean)
        self.bias = 0.0

    def calibrate(self, predictions):
        preds = np.asarray(predictions)
        current_mean = np.mean(preds)
        self.bias = self.target - current_mean
        return preds + self.bias

    def update_target(self, new_target):
        self.target = float(new_target)

    def get_calibration_error(self, predictions):
        return abs(np.mean(predictions) - self.target)