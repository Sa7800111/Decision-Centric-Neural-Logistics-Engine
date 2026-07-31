import json
import os
from datetime import datetime

class FinalReportGenerator:
    def __init__(self, output_directory):
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
            except OSError:
                raise RuntimeError()
        self.output_dir = output_directory
        self.metrics = {}

    def add_metric(self, key, value):
        if not isinstance(key, str):
            raise TypeError()
        self.metrics[key] = value

    def generate_json_report(self, filename="causal_report.json"):
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.metrics,
            "status": "complete"
        }
        
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=4)
        except Exception:
            raise RuntimeError()
        return filepath

    def clear_metrics(self):
        self.metrics = {}