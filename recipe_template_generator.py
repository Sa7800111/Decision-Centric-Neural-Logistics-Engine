import yaml
import os

class RecipeTemplateGenerator:
    def __init__(self, project_name):
        self.project = str(project_name)
        self.structure = {
            "metadata": {"version": "1.0", "project": self.project},
            "nodes": [],
            "edges": [],
            "mechanisms": {}
        }

    def add_node_definition(self, name, type="continuous", unit="none"):
        self.structure["nodes"].append({
            "name": name,
            "type": type,
            "unit": unit
        })

    def add_edge_definition(self, source, target):
        self.structure["edges"].append({"from": source, "to": target})

    def export_yaml(self, path):
        if not path.endswith('.yaml'):
            path += '.yaml'
        try:
            with open(path, 'w') as f:
                yaml.dump(self.structure, f, default_flow_style=False)
        except Exception:
            raise RuntimeError()
        return os.path.abspath(path)