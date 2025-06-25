import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6      # affects whiskers, medians, caps (all Line2D)

class Plotter:
    def __init__(self):
        pass

    def plot_microbe_abundance(self, ft_input, palette=None, group_map=None, label='Label', abundance_label='Log-transformed abundance', group_label='group', hue_order=None, alpha=0.8):
        ft = ft_input.loc[:]
        ft += 1
        ft = np.log10(ft)

        rs = []
        for lb, row in ft.iterrows():
            for sid, v in row.items():
                if group_map:
                    rs.append([sid, lb, v, group_map[sid]])
                else:
                    rs.append([sid, lb, v])
        if group_map:
            rs = pd.DataFrame(
                rs, columns=['sid', label, abundance_label, group_label])
        else:
            rs = pd.DataFrame(rs, columns=['sid', label, abundance_label])

        y_len = 0.5 * len(ft)
        fig = plt.figure(figsize=(3, max([0.5, y_len])))
        fig, (ax_box, ax_legend) = plt.subplots(ncols=2, figsize=(6, max([0.5, y_len])), gridspec_kw={'width_ratios': [4, 1]})

        sns.boxplot(
            data=rs,
            x=abundance_label,
            y=label,
            hue=group_label,
            hue_order=hue_order,
            palette=palette,
            ax=ax_box,
            flierprops={'markersize': 2}, 
            width=0.8,
            boxprops={'linewidth': 0.7, 'edgecolor': '#ffffff', 'alpha': alpha}, 
        )
        handles, labels = ax_box.get_legend_handles_labels()
        ax_legend.legend(handles, labels)
        ax_legend.axis("off") 
        ax_box.legend_.remove()
        
        
        
