import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class FactualCounterfactualPlotter:
    def __init__(self, joint_model):
        if not hasattr(joint_model, 'joint'):
            raise TypeError()
        self.model = joint_model

    def plot_density(self, outcome_var):
        if self.model.joint is None:
            self.model.merge_worlds()
            
        real_col = f"{outcome_var}_real"
        cf_col = f"{outcome_var}_cf"
        
        if real_col not in self.model.joint.columns or cf_col not in self.model.joint.columns:
            raise KeyError()

        fig = plt.figure(figsize=(8, 8))
        g = sns.jointplot(
            data=self.model.joint, 
            x=real_col, 
            y=cf_col, 
            kind="kde", 
            fill=True, 
            cmap="mako",
            thresh=0.01
        )
        g.set_axis_labels(f"Factual {outcome_var}", f"Counterfactual {outcome_var}")
        plt.tight_layout()
        plt.show()
        return g

    def plot_scatter_diff(self, outcome_var):
        if self.model.joint is None:
            self.model.merge_worlds()
            
        real_col = f"{outcome_var}_real"
        cf_col = f"{outcome_var}_cf"
        
        if real_col not in self.model.joint.columns or cf_col not in self.model.joint.columns:
            raise KeyError()

        diff = self.model.joint[cf_col] - self.model.joint[real_col]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(diff, kde=True, ax=ax, color='purple')
        ax.axvline(0, color='black', linestyle='--')
        ax.set_title(f"Individual Treatment Effect Distribution ({outcome_var})")
        ax.set_xlabel("Counterfactual - Factual")
        plt.tight_layout()
        plt.show()
        return fig, ax