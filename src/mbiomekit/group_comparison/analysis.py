import pandas as pd
from collections import defaultdict
from scipy.stats import kruskal, rankdata
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests
import json


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
        dunn_result = posthoc_dunn(
            df_dunn, val_col=id, group_col='_group', p_adjust='bonferroni')

        # Evaluate ranks of the group
        df['_rank'] = rankdata(df[id])
        mean_ranks = df.groupby('_group')['_rank'].mean()

        self.result_dunn[id] = {'mean_ranks': {},
                                'dunn_bf_pv': [],
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
