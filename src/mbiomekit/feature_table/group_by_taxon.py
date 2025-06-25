import pandas as pd 
import numpy as np 

class GroupByTaxon:
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
    
    def load_taxnomy_annotation(self, fin):
        tx = pd.read_csv(fin, sep='\t', header=0, index_col=0)
        self.taxonomy_annotation = tx
   
    def group_by_taxon(self, level=7, fout=None):
        tax_group = {}
        for id, row in self.taxonomy_annotation.iterrows():
            if id not in self.feature_table.index:
                continue 
            t = row['Taxon']
            s = t.split(';')
            
            if len(s) < level:
                continue 
            else:
                tx = tuple(s[:level])[-1]
                if tx in tax_group: 
                    tax_group[tx].append(id)
                else:
                    tax_group[tx] = [id]
        id_to_taxon = {
            member: group for group, members in tax_group.items() for member in members
        }

        df = self.feature_table.loc[:]
        nm = df.index.name
        df['taxon'] = df.index.map(id_to_taxon)

        # group by taxon 
        collapsed_df = df.groupby('taxon').agg('sum').reset_index()
        collapsed_df.set_index('taxon', inplace=True)

        self.feature_table_taxon = collapsed_df
        self.feature_table_taxon.to_csv(fout, sep='\t')





        
                