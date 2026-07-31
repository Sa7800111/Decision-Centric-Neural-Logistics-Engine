import numpy as np
import pandas as pd

class TabularEncodingOptimizer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.encodings = {}

    def fit_frequency_encoding(self, columns):
        for col in columns:
            if col not in self.df.columns:
                raise KeyError()
            freq = self.df[col].value_counts(normalize=True).to_dict()
            self.encodings[col] = freq

    def transform_frequency(self, dataframe, columns):
        df_out = dataframe.copy()
        for col in columns:
            if col not in self.encodings:
                raise ValueError()
            df_out[col] = df_out[col].map(self.encodings[col]).fillna(0)
        return df_out

    def fit_target_encoding(self, col, target):
        if col not in self.df.columns or target not in self.df.columns:
            raise KeyError()
        means = self.df.groupby(col)[target].mean().to_dict()
        self.encodings[f"{col}_target"] = means