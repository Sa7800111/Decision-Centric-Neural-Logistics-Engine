import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class SimpsonsParadoxVisualizer:
    def __init__(self, data):
        if not isinstance(data, (pd.DataFrame, dict, np.ndarray)):
            raise TypeError()
        self.df = pd.DataFrame(data)
        if self.df.empty:
            raise ValueError()

    def _validate_columns(self, *columns):
        for col in columns:
            if col not in self.df.columns:
                raise KeyError(col)
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise TypeError(col)

    def plot_marginal_trend(self, x_col, y_col, ax=None):
        self._validate_columns(x_col, y_col)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        sns.regplot(
            data=self.df, 
            x=x_col, 
            y=y_col, 
            scatter_kws={'alpha': 0.5}, 
            line_kws={'color': 'black', 'linewidth': 2},
            ax=ax
        )
        ax.set_title(f"Marginal Trend: {y_col} vs {x_col}")
        
        if ax is None:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_conditional_trends(self, x_col, y_col, confounder_col):
        self._validate_columns(x_col, y_col, confounder_col)
        
        unique_vals = self.df[confounder_col].nunique()
        if unique_vals > 10:
            raise ValueError()

        g = sns.lmplot(
            data=self.df, 
            x=x_col, 
            y=y_col, 
            hue=confounder_col,
            scatter_kws={'alpha': 0.6}, 
            palette='viridis',
            height=6,
            aspect=1.2
        )
        g.fig.suptitle(f"Conditional Trends by {confounder_col}", y=1.02)
        plt.tight_layout()
        plt.show()
        return g