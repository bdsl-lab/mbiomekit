import pandas as pd
from importlib.resources import files

class PicrustPathway:
    def __init__(self):
        self.pathway = None
        self.pathway_info = None

    def load_pathway(self, filename):
        self.pathway = pd.read_csv('data/pathway.tsv', header=0, index_col=0)

    def rename_index_by_pathway_name(self):
        fn = files("mbiomekit.data").joinpath("metacyc/metacyc_pathways_info.txt")
        df = pd.read_csv(fn, sep='\t', names=['id', 'name'])
        self.pathway_info = {}
        for id, row in df.iterrows():
            self.pathway_info[row['id']] = row['name']

        self.pathway.rename(self.pathway_info, axis=0, inplace=True)
        print(self.pathway)





