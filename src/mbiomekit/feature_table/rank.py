import pandas as pd 
import numpy as np 

class RankByAbundance:
    def __init__(self):
        self.feature_table = None
        self.feature_table_taxon = None
        self.taxonomy_annotation = None
    
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

        print(df.shape)
        print(np.sum(df.iloc[:, 0]))
        self.feature_table = df 

    def rank_by_abundance(self, rank_cutoff=10):
        tot_abund = []
        for id, row in self.feature_table.iterrows():
            tot_abund.append([np.sum(row), id])
        tot_abund.sort()
        tot_abund.reverse()

        tops = []
        for i in range(10):
            print(tot_abund[i])
            tops.append(tot_abund[i][1])
        print(tops)

