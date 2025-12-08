import pandas as pd
from collections import defaultdict
import json
import numpy as np

from scipy.stats import kruskal, rankdata
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests
from venn import venn
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6

class Analyzer:
    def __init__(self):
        self.ft = None
        self.ft_group_T = None
        self.group = None
        self.group_index = None

        self.result_kruskal = {}
        self.result_dunn = {}

    def set_feature_table(self, ft):
        self.ft = ft

    def set_group(self, group):
        self.ft_group_T = self.ft.loc[:]
        self.ft_group_T.loc['_group'] = group.loc[self.ft.columns]
        self.ft_group_T = self.ft_group_T.T

        self.group_index = defaultdict(list)
        for id, g in group.items():
            self.group_index[g].append(id)

    def group_analysis(self):
        for id in self.ft.index:
            df = self.ft_group_T.loc[:, [id, '_group']]

            # Kruskall-Wallis test
            grouped = df.groupby('_group')[id].apply(list)
            try:
                stat, pv = kruskal(*grouped)
                self.result_kruskal[id] = [stat, pv]

                # Dunn's test
                self.dunn_test(id, df)

            except ValueError as e:
                if "All numbers are identical" in str(e):
                    stat = 0.0
                    pv = 1.0
                    self.result_kruskal[id] = [stat, pv]

                    # Dunn's test (fail)
                    self.dunn_test_fail(id, df)

                else:
                    raise

        # multi-test correction of kruskal-wallis test
        self.result_kruskal = pd.DataFrame(self.result_kruskal, index=[
                                           'statistic', 'p-value']).T
        self.result_kruskal['adjusted-p'] = multipletests(
            self.result_kruskal['p-value'], alpha=0.05, method='fdr_bh')[1]
        self.result_kruskal.sort_values('adjusted-p', inplace=True)

    def dunn_test(self, id, df):
        # Dunn's test
        df_dunn = df[[id, '_group']]
        n = len(df_dunn)
        dunn_result = posthoc_dunn(
            df_dunn, val_col=id, group_col='_group', p_adjust='bonferroni')

        # Evaluate ranks of the group
        df['_rank'] = rankdata(df[id])
        mean_ranks = df.groupby('_group')['_rank'].mean()

        self.result_dunn[id] = {'mean_ranks': {},
                                'dunn_bf_pv': [],
                                'n_samples': n,
                                'status': 'Success'}

        for g, val in mean_ranks.items():
            self.result_dunn[id]['mean_ranks'][g] = val

        gs = list(mean_ranks.index)

        for i1, g1 in enumerate(gs):
            for i2, g2 in enumerate(gs):
                if i1 < i2:
                    self.result_dunn[id]['dunn_bf_pv'].append(
                        (g1, g2, dunn_result.loc[g1, g2]))

    def dunn_test_fail(self, id, df):
        self.result_dunn[id] = {'mean_ranks': {},
                                'dunn_bf_pv': [],
                                'status': 'Fail'}

    def save_results(self, output_kruskal='kruskal.tsv', output_dunn='dunn.json'):
        self.result_kruskal.to_csv(output_kruskal, sep='\t')
        with open(output_dunn, 'w') as f:
            json.dump(self.result_dunn, f)


class ComparativeAnalyzer:
    def __init__(self):
        self.kw = {}
        self.dunn = {}
        self.kw_sig = {}

    def load_kruskal_wallis_dunn_result(self, label, fin_kw, fin_dunn):
        self.kw[label] = pd.read_csv(fin_kw, sep='\t', header=0, index_col=0)

        with open(fin_dunn, 'r') as file:
            self.dunn[label] = json.load(file)

    def filter_by_significance(self, adjusted_p=0.05):
        for lb in self.kw:
            id = self.kw[lb].index[self.kw[lb]['adjusted-p'] < adjusted_p]
            self.kw_sig[lb] = self.kw[lb].loc[id, :]

    def evaluate_common_and_specific_group(self, grps=None, grp_labels=None, fout=None, fout_venn=None, alpha=0.05, venn_cmap=None):
        results = {}

        # Set analysis group
        if grps:
            pass
        else:
            grps = list(self.kw_sig.keys())

        print('Analysis group:', grps)

        grp_sets = {}
        for g in grps:
            grp_sets[g] = set(self.kw_sig[g].index)

        # Set group label
        if grp_labels:
            pass
        else:
            grp_labels = map(str, grps)

        # members shared by all groups
        shared_members = set.intersection(*grp_sets.values())
        results['shared_members'] = list(shared_members)
        results['shared_members_statistics'] = {}
        results['shared_members_observed_agreement'] = {}

        obs_agr = {}

        for ft in shared_members: 
            # for each microbial feature 
            rs = []
            for g in grps:
                # for each comparison group (eg. enterotypes, datasets)
                
                ms = self.dunn[g][ft]['mean_ranks']
                n = self.dunn[g][ft]['n_samples']
                pattern = []
                for e in self.dunn[g][ft]['dunn_bf_pv']: 
                    k = f'{e[0]}-{e[1]}'
                    if e[2] < alpha: 
                        if ms[e[0]] > ms[e[1]]: 
                            d = '+'
                        else:
                            d = '-'
                    else:
                        d = '0'
                    rs.append([ft, g, e[0], e[1], ms[e[0]]/n, ms[e[1]]/n, k, e[2], d])
            rs = pd.DataFrame(rs, columns=['feature', 'group', 'status1', 'status2', 'rel_mean_rank1', 'rel_mean_rank2', 'pair', 'adjusted-p', 'pattern'])
            rs_p = rs.pivot(index='group', columns='pair', values='pattern')

            results['shared_members_statistics'][ft] = rs.to_dict()
            results['shared_members_observed_agreement'][ft] = {}

            for c in rs_p.columns:
                results['shared_members_observed_agreement'][ft][c] = self.observed_agreement(rs_p[c])

        # members specifically associated to a group
        specific_members = {}
        for group, members in grp_sets.items():
            others = set.union(*(grp_sets[g] for g in grp_sets if g != group))
            specific_members[group] = list(members - others)

        results['specific_members'] = specific_members

        # Results output
        print(f'Number of all shared members: {len(shared_members)}')
        for g in specific_members:
            print(
                f'Number of {g} specific members: {len(specific_members[g])}')
        
        if fout:
            with open(fout, 'w') as file: 
                json.dump(results, file)


        if venn_cmap:
            vc = []
            for g in grp_sets:
                vc.append(venn_cmap[g])
            custom_cmap = ListedColormap(vc)
        else:
            custom_cmap = 'tab10'

        fig, ax = plt.subplots()
        v = venn(grp_sets, cmap=custom_cmap, ax=ax)
        plt.show()

        if fout_venn:
            fig.savefig(fout_venn)



    def observed_agreement(self, data):
        """
        Observed agreement 
        (1/(k(k-1)) SUM_{j=1}^{C}{n_j*(n_j-1)}

        * k: number of raters
        * c: number of classes 
        * nj: number of rate j

        data: Series with classes (categorical)
        """
        k = len(data)
        s = sum(nj * (nj - 1) for nj in data.value_counts())
        return s / (k * (k - 1))
