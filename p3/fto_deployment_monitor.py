import time

class FTODeploymentMonitor:
    def __init__(self, alert_threshold=0.2):
        self.threshold = alert_threshold
        self.drift_log = []

    def check_drift(self, y_true, y_pred, solver):
        regret = solver.compute_regret(y_true, y_pred)
        self.drift_log.append((time.time(), regret))
        
        if regret > self.threshold:
            return True, regret
        return False, regret

    def get_summary(self):
        return {
            "mean_regret": np.mean([x[1] for x in self.drift_log]),
            "uptime": self.drift_log[-1][0] - self.drift_log[0][0] if self.drift_log else 0
        }