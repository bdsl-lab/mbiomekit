import pytest
import pandas as pd
from pathlib import Path
from mbiomekit.machine_learning import random_forest

DATA_DIR = Path(__file__).parent / "data"

def test_random_forest():
    
    ft = pd.read_csv(DATA_DIR / 'ft_genus_prv01.tsv', sep='\t', header=0, index_col=0)
    lbs = pd.read_csv(DATA_DIR / 'group.tsv', sep='\t', header=0, index_col=0)['group1']
        
    prd = random_forest.RandomForestPredictor()
    prd.load_data(ft.T, lbs)
    prd.build_random_forest_model()
    
    

