import pandas as pd
import numpy as np

class TabularDLPreprocessor:
    def __init__(self, categorical_cols, numerical_cols):
        self.cat = categorical_cols
        self.num = numerical_cols
        self.stats = {}

    def fit(self, df):
        for col in self.num:
            self.stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}
        for col in self.cat:
            self.stats[col] = df[col].unique().tolist()

    def transform(self, df):
        out = df.copy()
        for col in self.num:
            out[col] = (df[col] - self.stats[col]['mean']) / (self.stats[col]['std'] + 1e-9)
        for col in self.cat:
            out[col] = pd.Categorical(df[col], categories=self.stats[col]).codes
        return out