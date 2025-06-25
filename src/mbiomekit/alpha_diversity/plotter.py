import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6      # affects whiskers, medians, caps (all Line2D)

class AlphaDiversityPlot:
    def __init__(self):
        self.alpha_group = None

    def load_alpha_diversity(self, df):
        self.df_alpha = df
        self.alpha_name = df.name

    def load_group(self, df, order=None):
        self.df_group = df
        self.group_name = df.name

        if order:
            self.group_order = order
        else:
            self.group_order = list(dict.fromkeys(self.df_group.values))

    def merge_data(self):
        self.alpha_group = pd.concat(
            [self.df_alpha, self.df_group], axis=1, join='inner')

    def plot_by_group(self, group_order=None, palette=None, alpha=0.8, fout=None):
        """
        Plot alpha diversity as a box plot. 

        For statistics analysis, Kruskal-Wallis test followed by Dunn's test were employed. 

        """
        # merge the alpha diversity and group data frame 
        self.merge_data()

        # generated grouped alpha diversity vectors for Kruskal-Wallis test
        grouped_data = []
        for g in self.alpha_group.groupby(self.group_name):
            grouped_data.append(g[1][self.alpha_name].to_list())

        # Kruskal-Wallis test
        stat, p_value = kruskal(*grouped_data)
        print(
            f"Kruskal-Wallis H-statistic: {stat:.5f}, p-value: {p_value:.5f}")

        # Dunn’s test (Bonferroni correction)
        posthoc_results = sp.posthoc_dunn(
            self.alpha_group, val_col=self.alpha_name, group_col=self.group_name, p_adjust='bonferroni')
        print(posthoc_results)
        
        # preparations for boxplot 
        flierprops = {
            "marker": "o",
            "markersize": 2,
            "alpha": 0.6,
        }
        
        if not group_order: 
            group_order = list(set(self.df_group.values))

        # boxplot 
        fig = plt.figure(figsize=(1.5, 3))
        ax = fig.add_subplot(111)
        sns.boxplot(
            x=self.group_name, 
            y=self.alpha_name, 
            data=self.alpha_group, 
            order=group_order, 
            palette=palette,
            width=0.6, 
            linewidth=0.6,
            flierprops = flierprops,
            boxprops={'linewidth': 0, 'edgecolor': '#ffffff', 'alpha': alpha}, 
            ax=ax,
        )
        
        # title and axes labels 
        ax.set_title('Alpha Diversity')
        ax.set_label(self.group_name)
        ax.set_ylabel(self.alpha_name)
        
        # statistical significance
        comparisons = list(itertools.combinations(group_order, 2))
        y_max = self.alpha_group[self.alpha_name].max()
        y_offset = y_max * 0.05
        height = y_max + y_offset

        for i, (group1, group2) in enumerate(comparisons):
            p = posthoc_results.loc[group1, group2]
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            if sig != 'ns':  # ns는 생략
                x1, x2 = group_order.index(group1), group_order.index(group2)
                x_center = (x1 + x2) / 2
                sig_height_multiplier = 0.5 
                line_height = height + y_offset * sig_height_multiplier
                ax.plot([x1, x1, x2, x2], [height, line_height, line_height, height], lw=0.8, c='k')
                ax.text(x_center, height + y_offset + (y_offset * -0.6), sig, ha='center', va='bottom', fontsize=7)
                height += y_offset * 1.7

        if fout:
            fig.savefig(fout)

        