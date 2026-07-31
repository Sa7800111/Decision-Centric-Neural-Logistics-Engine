import re

class SyntaxTranslator:
    def __init__(self):
        self.do_pattern = re.compile(r"do\s*\(\s*([A-Za-z0-9_]+)\s*=\s*([0-9.-]+)\s*\)")
        self.cond_pattern = re.compile(r"\|\s*(.*)\)")
        self.target_pattern = re.compile(r"P\s*\(\s*([A-Za-z0-9_]+)\s*(?:=|\||\))")

    def parse_query(self, query_string):
        if not isinstance(query_string, str):
            raise TypeError()
            
        target_match = self.target_pattern.search(query_string)
        if not target_match:
            raise ValueError()
        target_var = target_match.group(1).strip()

        interventions = {}
        for match in self.do_pattern.finditer(query_string):
            var, val = match.groups()
            interventions[var.strip()] = float(val.strip())

        conditions = {}
        cond_match = self.cond_pattern.search(query_string)
        if cond_match:
            cond_str = cond_match.group(1)
            cond_str = self.do_pattern.sub("", cond_str)
            
            parts = [p.strip() for p in cond_str.split(',') if p.strip()]
            for p in parts:
                if '=' in p:
                    k, v = p.split('=')
                    conditions[k.strip()] = float(v.strip())

        return {
            'target': target_var,
            'interventions': interventions,
            'conditions': conditions
        }