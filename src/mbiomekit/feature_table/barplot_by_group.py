import pandas as pd 
import seaborn as sns 
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6      # affects whiskers, medians, caps (all Line2D)

class PlotterGroup:
    def __init__(self):
        self.feature_table = None
        self.group = None

    def load_feature_table(self, fin):
        flag_biom = False
        with open(fin) as f:
            first_line = f.readline()
            if first_line.startswith("# Constructed from biom file"):
                flag_biom = True
                
        if flag_biom:
            df = pd.read_csv(fin, sep='\t', header=0, index_col=0, skiprows=1)
        else:
            df = pd.read_csv(fin, sep='\t', header=0, index_col=0)

        self.feature_table = df 

    def load_group(self, df):
        self.group = df 

    def barplot_by_group(self, features, group_order=None, palette=None, fout=None, read_per_sample=10000):
        ft_sel = self.feature_table.loc[features, :]
        ft_sel = ft_sel / 10000 * 100

        ft_sel_grp = ft_sel.T.groupby(self.group).mean().T

        rs = []
        for id, row in ft_sel_grp.iterrows():
            for g, v in row.items():
                rs.append([id, g, v])
        rs = pd.DataFrame(rs, columns=['feature', 'group', 'proportion'])
        fig = plt.figure(figsize=(3, 1.5))
        ax = fig.add_subplot(111)

        plot = sns.barplot(
            data=rs,
            y="proportion", x="group", hue='feature',
            hue_order=features,
            order=group_order,
            palette=palette, 
            edgecolor='white',
            linewidth=0.4, ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if fout:
            fig.savefig(fout)
