import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
# affects whiskers, medians, caps (all Line2D)
plt.rcParams['lines.linewidth'] = 0.6


class Plotter:
    def __init__(self):
        pass

    def plot_microbe_abundance(self, ft_input, fout=None, palette=None, group_map=None, label='Label', abundance_label='Log-transformed abundance', group_label='group', hue_order=None, alpha=0.8):
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

        y_len = 0.375 * len(ft)
        # fig = plt.figure(figsize=(1.5, max([0.375, y_len])))
        fig, (ax_box, ax_legend) = plt.subplots(ncols=2, figsize=(
            3, max([0.375, y_len])), gridspec_kw={'width_ratios': [5, 2]})

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
            boxprops={'linewidth': 0.7,
                      'edgecolor': '#ffffff', 'alpha': alpha},
        )
        handles, labels = ax_box.get_legend_handles_labels()
        ax_legend.legend(handles, labels)
        ax_legend.axis("off")
        ax_box.legend_.remove()

        if fout:
            fig.savefig(fout)

    def plot_dunns_test_three_groups(self, fin_kruskal, fin_dunn, x_group, y_group, reference_group, n_samples, output_figure=None, fout=None, kruskal_wallis_cutoff=0.05):
        """
        
        Plot Group analysis results: Kruskal-Wallis followed by Dunn's test. 
        

        group 0, 1, 2 (mean rank)
        between group p-value 
        n groups -> n*(n-1)/2 pairs -> n*(n-1)/2 dimensional vector 
        dimension reduction?

        n = 3, -> 3 dimension (1 standard) -> 2 dimension 

        1 - 0 (rank)
        2 - 0 (rank)

        """

        rs_kw = pd.read_csv(fin_kruskal, sep='\t', header=0, index_col=0)

        ids = rs_kw.index[rs_kw['adjusted-p'] < kruskal_wallis_cutoff]

        rs_kw_sig = rs_kw.loc[ids, :]

        with open(fin_dunn, 'r') as file:
            data = json.load(file)

        diff = []
        for ft in rs_kw_sig.index:
            mr = data[ft]['mean_ranks']

            v1 = mr[x_group] - mr[reference_group]
            v2 = mr[y_group] - mr[reference_group]

            for e in data[ft]['dunn_bf_pv']:
                # pvalue for x axis group vs. reference group
                if e[0] == x_group and e[1] == reference_group:
                    pv_x = e[2]
                elif e[0] == reference_group and e[1] == x_group:
                    pv_x = e[2]

                # pvalue for y axis group vs. reference group
                if e[0] == y_group and e[1] == reference_group:
                    pv_y = e[2]
                elif e[0] == reference_group and e[1] == y_group:
                    pv_y = e[2]

                # pvalue for x axis group vs. y axis group
                if e[0] == x_group and e[1] == y_group:
                    pv_xy = e[2]
                elif e[0] == y_group and e[1] == x_group:
                    pv_xy = e[2]

            # coding the differences
            if pv_x < 0.05 and pv_y < 0.05 and pv_xy < 0.05:
                dg = 'all_diff'
            elif pv_x < 0.05 and pv_y < 0.05:
                dg = 'ref_specific'
            elif pv_x < 0.05 and pv_xy < 0.05:
                dg = 'x_specific'
            elif pv_y < 0.05 and pv_xy < 0.05:
                dg = 'y_specific'
            elif pv_x < 0.05:
                dg = 'x_ref_only'
            elif pv_y < 0.05:
                dg = 'y_ref_only'
            elif pv_xy < 0.05:
                dg = 'x_y_only'
            else:
                dg = 'none'
            kw_pv = -np.log10(rs_kw_sig.loc[ft, 'adjusted-p'])
            v1 = v1 / n_samples
            v2 = v2 / n_samples
            diff.append([ft, v1, v2, kw_pv, dg])

        palette = {'all_diff': '#7876b1c0',
                   'ref_specific': '#20854ea0',
                   'x_specific': '#9fbccaa0',
                   'y_specific': '#326a81a0',
                   'x_ref_only': '#ffdc91a0',
                   'y_ref_only': '#e18727a0',
                   'x_y_only': '#aaadb2c0',
                   'none': '#ffffff', }

        hue_order = ['all_diff', 
                     'y_specific', 'x_specific', 'ref_specific',
                     'y_ref_only', 'x_ref_only', 'x_y_only', 'none']

        diff = pd.DataFrame(diff, columns=['feature', 'x', 'y', 'kw_pv', 'difference_group'])

        fig, (ax_box, ax_legend) = plt.subplots(
            ncols=2, figsize=(3.75, 2.25), gridspec_kw={'width_ratios': [3, 2]})
        sns.scatterplot(diff, x='x', y='y', size='kw_pv',
                        hue='difference_group', palette=palette, ax=ax_box, zorder=10, hue_order=hue_order)
        ax_box.grid(zorder=1)
        mx = np.max([diff['x'].abs().max(), diff['y'].abs().max()])
        ax_box.plot([-mx*1.05, mx*1.05], [-mx*1.05, mx*1.05],
                    c='k', linestyle='--', zorder=3)
        ax_box.plot([0, 0], [-mx*1.05, mx*1.05], c='k', linewidth=1, zorder=2)
        ax_box.plot([-mx*1.05, mx*1.05], [0, 0], c='k', linewidth=1, zorder=2)

        ax_box.set_xlabel(f'Δ Mean Rank ({x_group}-{reference_group})')
        ax_box.set_ylabel(f'Δ Mean Rank ({y_group}-{reference_group})')

        ax_box.set_xlim([np.min([diff['x'].min()*1.05, diff['y'].min()*1.05]),
                        np.max([diff['x'].max()*1.05, diff['y'].max()*1.05])])
        ax_box.set_ylim([np.min([diff['x'].min()*1.05, diff['y'].min()*1.05]),
                        np.max([diff['x'].max()*1.05, diff['y'].max()*1.05])])

        handles, labels = ax_box.get_legend_handles_labels()
        # ax_legend.legend(handles, labels)
        ax_legend.axis("off")
        new_labels = ['All pairs diff.', 
                      f'{y_group} spec.', f'{x_group} spec.', f'{reference_group} spec.', 
                      f'{y_group}-{reference_group} diff.', f'{x_group}-{reference_group} diff.', f'{x_group}-{y_group} diff.',
                      'None']
    
        label_map = {}
        for i, dg in enumerate(hue_order):
            label_map[dg] = new_labels[i]
    
        # ax_legend.legend(handles, labels=new_labels, title='Difference group')
        labels[0] = 'Difference group'
        labels = ['Difference group'] + new_labels + \
            ['-log(p-value)'] + labels[10:]
        ax_legend.legend(handles, labels)
        ax_box.legend_.remove()

        if output_figure:
            fig.savefig(output_figure)
        
        if fout:
            diff['difference_group'] = diff['difference_group'].map(label_map)
            diff.sort_values(['difference_group', 'kw_pv'], inplace=True, ascending=False)
            diff.to_csv(fout, sep='\t')
        
        
