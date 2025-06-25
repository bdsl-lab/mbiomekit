import pandas as pd 
import numpy as np 

class Filter:
    def __init__(self):
        self.feature_table = None
        self.feature_table_filtered = None

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

    def filter_by_prevalence(self, prevalence_cutoff=0.1, fout=None):
        filter = []
        for id, row in self.feature_table.iterrows():
            prv = float(np.count_nonzero(row)) / len(row)
            if prv > prevalence_cutoff: 
                filter.append(id)
        
        self.feature_table_filtered = self.feature_table.loc[filter, :]
        print(self.feature_table_filtered.shape)

        if fout:
            self.feature_table_filtered.to_csv(fout, sep='\t')

