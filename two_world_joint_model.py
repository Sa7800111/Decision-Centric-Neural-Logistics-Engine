import pandas as pd
import numpy as np

class TwoWorldJointModel:
    def __init__(self, real_data, cf_data):
        if not isinstance(real_data, pd.DataFrame) or not isinstance(cf_data, pd.DataFrame):
            raise TypeError()
        if len(real_data) != len(cf_data):
            raise ValueError()
        
        self.real = real_data.copy()
        self.cf = cf_data.copy()
        self.joint = None

    def merge_worlds(self, suffixes=('_real', '_cf')):
        if len(suffixes) != 2:
            raise ValueError()
        self.joint = self.real.join(self.cf, lsuffix=suffixes[0], rsuffix=suffixes[1])
        return self.joint

    def probability_of_necessity(self, outcome_var, treatment_var):
        if self.joint is None:
            self.merge_worlds()
            
        real_y = f"{outcome_var}_real"
        cf_y = f"{outcome_var}_cf"
        real_t = f"{treatment_var}_real"

        if real_y not in self.joint.columns or cf_y not in self.joint.columns or real_t not in self.joint.columns:
            raise KeyError()

        subset = self.joint[(self.joint[real_t] == 1) & (self.joint[real_y] == 1)]
        if len(subset) == 0:
            return 0.0

        pn_count = len(subset[subset[cf_y] == 0])
        return float(pn_count / len(subset))

    def probability_of_sufficiency(self, outcome_var, treatment_var):
        if self.joint is None:
            self.merge_worlds()

        real_y = f"{outcome_var}_real"
        cf_y = f"{outcome_var}_cf"
        real_t = f"{treatment_var}_real"

        if real_y not in self.joint.columns or cf_y not in self.joint.columns or real_t not in self.joint.columns:
            raise KeyError()

        subset = self.joint[(self.joint[real_t] == 0) & (self.joint[real_y] == 0)]
        if len(subset) == 0:
            return 0.0

        ps_count = len(subset[subset[cf_y] == 1])
        return float(ps_count / len(subset))