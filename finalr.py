import json

class FinalReportGenerator:
    def __init__(self, metrics_dict):
        self.data = metrics_dict

    def summarize_to_text(self):
        summary = "--- Causal Analysis Report ---\n"
        for key, val in self.data.items():
            summary += f"{key}: {val}\n"
        return summary

    def export_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4)