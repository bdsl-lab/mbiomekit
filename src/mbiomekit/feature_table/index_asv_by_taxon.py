import numpy as np 
import pandas as pd

class ASVTaxon:
    def __init__(self):
        self.feature_table = None

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

    def load_taxnomy_annotation(self, fin):
        tx = pd.read_csv(fin, sep='\t', header=0, index_col=0)
        self.taxonomy_annotation = tx

    def index_asv_by_taxon(self, fout=None):
        # Order feature by abundance 
        tc = []
        for id in self.feature_table.index:
            t = np.sum(self.feature_table.loc[id, :])
            tc.append([t, id])
        tc.sort()
        tc.reverse() 

        # taxonomy label 
        lb_cnt = {}
        tx_label = {}
        for e in tc: 
            fid = e[1]
            t = self.taxonomy_annotation.loc[fid, 'Taxon']
            lb = t.split(';')[-1]
            if lb in lb_cnt: 
                lb_cnt[lb] += 1
            else:
                lb_cnt[lb] = 1
            
            tx_label[fid] = lb + f' ({lb_cnt[lb]})'

        # 
        self.feature_table_tax = self.feature_table.loc[:]
        self.feature_table_tax.rename(tx_label, axis=0, inplace=True)

        if fout:
            self.feature_table_tax.to_csv(fout, sep='\t')

        

        