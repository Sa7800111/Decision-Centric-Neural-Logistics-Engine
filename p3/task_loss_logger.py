import json
import time

class TaskLossLogger:
    def __init__(self, log_path):
        self.path = log_path
        self.logs = []

    def log_step(self, epoch, mse, decision_loss, regret):
        entry = {
            "timestamp": time.time(),
            "epoch": epoch,
            "mse": float(mse),
            "decision_loss": float(decision_loss),
            "regret": float(regret)
        }
        self.logs.append(entry)
        
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.logs, f, indent=4)